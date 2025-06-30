# add_amber.py (modified to load a policy instead of random actions)
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import csv
from pathlib import Path
import os
from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser(
    description="Script to add Amber with a learned policy in Isaac Lab."
)
parser.add_argument(
    "--csv_out",
    type=Path,
    default=Path("amber_joint_log.csv"),
    help="Path to the CSV file that will store joint positions."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn (must be 1 for policy wrapper)."
)
parser.add_argument(
    "--policy_config",
    type=Path,
    required=True,
    help="Path to RL policy YAML config file."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ────────── Launch Isaac Sim ──────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ────────── Now import RLPolicy and its dependencies ──────────
import yaml
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath('/home/s-ritwik/src/robot_rl'))

from transfer.sim.rl_policy_wrapper import RLPolicy  # your wrapper:contentReference[oaicite:0]{index=0}

# Isaac Lab & Amber imports
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
import omni.physics.tensors as physx
import omni.usd
from pxr import Gf, UsdPhysics, UsdGeom

# (Your existing AMBER_CONFIG, NewRobotsSceneCfg definitions go here unchanged…)
# …

def load_policy(config_path: Path) -> RLPolicy:
    """Load RLPolicy from a small YAML config (same fields as g1_runner)."""
    cfg = yaml.safe_load(open(config_path, "r"))
    return RLPolicy(
        dt=cfg["dt"],
        checkpoint_path=cfg["checkpoint_path"],
        num_obs=cfg["num_obs"],
        num_action=cfg["num_action"],
        cmd_scale=cfg["command_scale"],
        period=cfg["period"],
        action_scale=cfg["action_scale"],
        default_angles=np.array(cfg["default_angles"], dtype=np.float32),
        qvel_scale=cfg["qvel_scale"],
        ang_vel_scale=cfg["ang_vel_scale"],
        height_map_scale=cfg.get("height_map_scale", None),
    )

def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Match the Mujoco-style projected gravity for RLPolicy."""
    qw, qx, qy, qz = quat
    pg = np.zeros(3, dtype=np.float32)
    pg[0] = 2 * (-qz * qx + qw * qy)
    pg[1] = -2 * (qz * qy + qw * qx)
    pg[2] = 1 - 2 * (qw * qw + qz * qz)
    return pg

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, policy: RLPolicy):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step = 0

    amber = scene["Amber"]
    device = amber.data.default_root_state.device
    assert args_cli.num_envs == 1, "Policy wrapper currently supports a single env."

    # CSV setup
    joint_names = amber.data.joint_names
    csv_path = args_cli.csv_out.expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["step", "sim_time", "env_id", *joint_names])

    try:
        while simulation_app.is_running():
            # 1) Write data → sim
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # 2) Read current state
            # Amber root_state_w: (1, 13) → [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
            root_state = amber.data.root_state_w.cpu().numpy()[0]
            qjoints = amber.data.joint_pos.cpu().numpy()[0]
            qvel = amber.data.joint_vel.cpu().numpy()[0]

            # 3) Build observation
            body_ang_vel = root_state[10:13]                    # wx,wy,wz
            projected_grav = get_projected_gravity(root_state[3:7])
            des_vel = np.zeros(3, dtype=np.float32)             # e.g. zero target velocity

            obs = policy.create_obs(
                qjoints,
                body_ang_vel,
                qvel,
                sim_time,
                projected_grav,
                des_vel,
                height_map=None
            )

            # 4) Get action from policy
            action = policy.get_action(obs)                     # numpy array of length num_action

            # 5) Send to sim
            target = torch.from_numpy(action).to(device).unsqueeze(0)
            amber.set_joint_position_target(target)

            # 6) Log
            cur_pos = qjoints
            writer.writerow([step, sim_time, 0, *cur_pos])
            if step % 100 == 0:
                csv_fh.flush()

            # Advance time
            sim_time += sim_dt
            step += 1

    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted – CSV saved at: {csv_path}")
    finally:
        csv_fh.close()

def main():
    # Initialize sim & scene
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=5,
        device=args_cli.device
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # Load policy
    policy = load_policy(args_cli.policy_config)

    # Reset & run
    sim.reset()
    scene.reset()
    print("[INFO] Setup complete – running policy…")
    run_simulator(sim, scene, policy)

if __name__ == "__main__":
    main()
    simulation_app.close()
