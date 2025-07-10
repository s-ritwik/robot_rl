from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
from pxr import Gf, UsdGeom
import omni.usd  



def sin_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    phase = torch.tensor(2*torch.pi * (env.sim.current_time / period))
    sphase = torch.sin(phase)

    sphase = torch.ones((env.num_envs, 1), device=env.device) * sphase

    return sphase

def cos_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    phase = torch.tensor(2*torch.pi * (env.sim.current_time / period))
    cphase = torch.cos(phase)

    cphase = torch.ones((env.num_envs, 1), device=env.device) * cphase

    return cphase

def is_ground_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    sp = sin_phase(env, period)
    cp = cos_phase(env, period)

    return torch.tensor([(sp < 0.0), (cp < 0.0)])

# gravity constant
_G = 9.81


def _phi_contact_scalar(phi: float) -> float:
    """Eq.(18): sin(2πφ) / sqrt(sin²(2πφ)+0.04)."""
    s = math.sin(2 * math.pi * phi)
    return s / math.sqrt(s * s + 0.04)


@torch.no_grad()
def future_foot_targets_lip(
    env: ManagerBasedRLEnv,
    Ts: float                    = 0.4,          # single-step duration
    nom_height: float            = 0.45,         # COM height used in LIP
    wdes: float                  = 0.10,         # desired lateral width (m)
    command_name: str            = "base_velocity",
    asset_cfg: SceneEntityCfg    = SceneEntityCfg("robot"),
    force_thresh: float          = 1.0,          # just for stance detection
) -> torch.Tensor:
    """
    Observation term that returns **future left & right foot positions** computed
    with a simple Linear-Inverted-Pendulum ICP algorithm.  Values update only
    when the corresponding foot enters its swing half-cycle, so the vector is
    piece-wise constant.
    """
    amber  = env.scene[asset_cfg.name]
    device = env.device
    N      = env.num_envs

    # ------------------------------------------------------------------ #
    # 0) persistent buffers                                              #
    # ------------------------------------------------------------------ #
    fn = future_foot_targets_lip
    if not hasattr(fn, "_init"):
        # target positions we will output/hold     shape [N,2,3]  (L,R)
        pos0       = amber.data.body_pos_w
        B          = pos0.shape[1]
        init_left  = pos0[:, B-2, :].clone()
        init_right = pos0[:, B-1, :].clone()
        targets    = torch.stack((init_left, init_right), dim=1)  # [N,2,3]

        fn._targets     = targets.to(device)
        fn._prev_sign   = torch.zeros(N, dtype=torch.long, device=device)
        fn._orig_y_off  = targets[:, :, 1].clone()                # store native Y
        fn._init        = True

    targets   = fn._targets
    prev_sign = fn._prev_sign
    orig_y    = fn._orig_y_off

    # ------------------------------------------------------------------ #
    # 1) current phase sign                                              #
    # ------------------------------------------------------------------ #
    t          = env.sim.current_time
    phi_cycle  = (t % (2 * Ts)) / (2 * Ts)          # φ ∈ [0,1)
    phi_cont   = _phi_contact_scalar(phi_cycle)     # scalar
    sign_now   = 1 if phi_cont > 0 else -1          # +1 right-swing, -1 left-swing
    sign_now_tensor = torch.full((N,), 1 if phi_cont > 0 else -1,
                             device=device, dtype=torch.long)
    # ------------------------------------------------------------------ #
    # 2) if half-cycle just changed sign → compute a new swing target    #
    # ------------------------------------------------------------------ #
    changed = sign_now != prev_sign
    changed_mask = sign_now_tensor != prev_sign        # bool[N]

    if changed_mask.any():
        swing_leg = 1 if sign_now > 0 else 0        # 0=L, 1=R
        stance_leg= 1 - swing_leg

        # ---------------- LIP-ICP ------------------------------------- #
        cmd      = env.command_manager.get_command(command_name)[:, :2]  # [N,2]
        root_pos = amber.data.body_pos_w[:, 3, :]                        # [N,3]
        omega    = math.sqrt(_G / nom_height)

        icp_0 = torch.zeros((N,3), device=device)
        icp_0[:, :2] = cmd / omega

        # feet positions & stance-foot
        pos_w   = amber.data.body_pos_w
        foot_L  = pos_w[:, -2, :]
        foot_R  = pos_w[:, -1, :]
        feet    = torch.stack((foot_L, foot_R), dim=1)        # [N,2,3]
        stance  = feet[:, stance_leg, :].clone()
        stance[:, 2] = 0.0

        to_local  = lambda v, q: quat_rotate(yaw_quat(quat_inv(q)), v)
        to_global = lambda v, q: quat_rotate(yaw_quat(q), v)

        root_q   = amber.data.body_quat_w[:, 3, :]

        exp_wT   = math.exp(omega * Ts)
        icp_f    = exp_wT * icp_0 + (1 - exp_wT) * to_local(root_pos - stance, root_q)
        icp_f[:, 2] = 0.0

        sd   = cmd[:, 0].abs() * Ts
        wd   = wdes * torch.ones(N, device=device)
        bx   = sd / (exp_wT - 1.0)
        by   = sign_now * wd / (exp_wT + 1.0)
        b    = torch.stack((bx, by, torch.zeros_like(bx)), dim=1)

        p_loc        = icp_f.clone()
        p_loc[:, 0]  = torch.clamp(p_loc[:, 0] - b[:, 0], -0.5, 0.5)
        p_loc[:, 1]  = torch.clamp(p_loc[:, 1] - b[:, 1], -0.3, 0.3)
        p_world      = to_global(p_loc, root_q) + root_pos
        p_world[:, 2]= 0.0

        # keep original lateral offset
        p_world[:, 1] = orig_y[:, swing_leg]

        # update target for swing leg
        targets[:, swing_leg, :] = p_world

    # store sign for next step
    prev_sign.fill_(sign_now)

    # ------------------------------------------------------------------ #
    # 3) flatten to [N,6] observation                                    #
    # ------------------------------------------------------------------ #
    obs = targets.reshape(N, 6)
    return obs


@torch.no_grad()
def current_foot_positions(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Observation term returning the **current** left‐ and right‐foot positions
    in world‐frame.

    Outputs a tensor of shape [num_envs, 6]:
      [x_L, y_L, z_L,  x_R, y_R, z_R]
    where “left” is body index B-2 and “right” is B-1.
    """
    asset = env.scene[asset_cfg.name]
    # all body positions: [N, num_bodies, 3]
    pos_w = asset.data.body_pos_w
    B     = pos_w.shape[1]

    # left foot = index B-2, right foot = index B-1
    left  = pos_w[:, B-2, :]   # [N,3]
    right = pos_w[:, B-1, :]   # [N,3]

    # concatenate → [N,6]
    return torch.cat((left, right), dim=1)