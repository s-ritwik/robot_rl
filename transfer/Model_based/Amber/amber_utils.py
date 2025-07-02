import csv
import numpy as np
import torch
from pxr import Gf, UsdGeom
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
import omni.usd
import math
from source.robot_rl.robot_rl.tasks.manager_based.robot_rl.amber.amber_env_cfg import PERIOD,WDES

def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    """
    quat: [qw, qx, qy, qz]
    returns projected gravity in the body frame.
    """
    qw, qx, qy, qz = quat
    pg = np.zeros(3, dtype=np.float32)
    pg[0] = 2 * (-qz * qx + qw * qy)
    pg[1] = -2 * (qz * qy + qw * qx)
    pg[2] = 1 - 2 * (qw * qw + qz * qz)
    return pg



def compute_step_location_direct(
    sim_time: float,
    scene,
    args_cli,
    des_vel: np.ndarray,      # shape (3,) → [vx, vy, vyaw]
    nom_height: float,
    Tswing: float,
    wdes: float,
    visualize: bool = True
) -> torch.Tensor:
    """
    Compute Amber’s next footstep in world‐frame, given raw sim_time, scene, and a direct
    base-velocity command (no Hydra env needed).
    Returns a tensor [1,3] of the desired foothold.
    """
    amber = scene["Amber"]
    device = amber.data.default_root_state.device
    stage = omni.usd.get_context().get_stage()
    
    def _update_arrow_x(prim_base: str,
                        origin: np.ndarray,
                        vec: np.ndarray,
                        color: tuple):
        """
        Draw / update a single arrow (shaft + cone) whose length, orientation and
        colour are set from `vec`, but **only the x-component** of vec is used.
        Uses a fixed prim path so we overwrite the same geometry each frame.
        """
        stage = omni.usd.get_context().get_stage()

        # keep only vx, sign preserves direction along world-X
        vx = float(vec[0])
        vec = np.array([vx, 0.0, 0.0], dtype=np.float64)

        length = abs(vx)
        if length < 1e-4:
            length = 1e-4                                            # non-zero

        # unit +X or -X
        dir_n = np.array([np.sign(vx) if vx != 0 else 1.0, 0.0, 0.0])

        shaft_path = prim_base + "/shaft"
        head_path  = prim_base + "/head"

        # create the prims once
        if not stage.GetPrimAtPath(shaft_path):
            UsdGeom.Cylinder.Define(stage, shaft_path).GetRadiusAttr().Set(0.01)
            cone = UsdGeom.Cone.Define(stage, head_path)
            cone.GetRadiusAttr().Set(0.02)
            cone.GetHeightAttr().Set(0.04)

        # fetch prims as Xformables
        shaft_xf = UsdGeom.Xformable(stage.GetPrimAtPath(shaft_path))
        head_xf  = UsdGeom.Xformable(stage.GetPrimAtPath(head_path))

        # clear previous ops (avoids stacking new ops each frame)
        shaft_xf.ClearXformOpOrder()
        head_xf.ClearXformOpOrder()

        # shaft: scale Z to 0.8*length, translate to mid-point
        shaft_xf.AddScaleOp().Set(Gf.Vec3f(1, 1, length * 0.4))
        mid = origin + dir_n * (length * 0.2)
        shaft_xf.AddTranslateOp().Set(Gf.Vec3d(*mid.tolist()))
        UsdGeom.Cylinder(shaft_xf.GetPrim()).GetDisplayColorAttr().Set([color])

        # head: translate to (origin + 0.8*vec + 0.02 along dir), no need to orient
        # tip = origin + dir_n * (length * 0.8 + 0.02)
        # head_xf.AddTranslateOp().Set(Gf.Vec3d(*tip.tolist()))
        # UsdGeom.Cone(head_xf.GetPrim()).GetDisplayColorAttr().Set([color])

    # device = torch.device(scene._sim.device)  # grab the torch device from your sim
    # 1) current COM velocity (green)
    com_pos   = amber.data.body_pos_w.cpu().numpy()[0,3,:]
    com_vel   = amber.data.body_link_lin_vel_w.cpu().numpy()[0,3,:]
    # _update_arrow_x("/World/debug/vel_arrow", com_pos, com_vel, (0.0, 1.0, 0.0))

    # 2) commanded base velocity (blue)
    # rotate local [vx,vy,0] into world
    root_quat = amber.data.root_quat_w[0]                    # => Tensor on device
    cmd_local = torch.tensor([*args_cli.desired_vel[:2], 0.0],
                         dtype=torch.float32, device=device)
    # use your quat_rotate / yaw_quat helpers:
    cmd_world = quat_rotate(yaw_quat(root_quat), cmd_local)
    cmd_world_np = cmd_world.cpu().numpy()
    # _update_arrow_x("/World/debug/cmd_arrow", com_pos, cmd_world_np, (0.0, 0.0, 1.0))

    # 1) “command” is just your desired_vel XY
    vx = torch.tensor([[des_vel[0]]], device=device)  # shape (1,1)    # pad to 3D so we can reuse the same code
    # cmd3 = torch.cat([cmd, torch.zeros((1,1),device=device)], dim=1)  # [1,3]

    # 2) COM position from Amber’s body_pos_w slot #3
    r = amber.data.body_pos_w[:, 3, :]    # [1,3]
    print(r)
    # 3) ICP‐initial
    g     = 9.81
    omega = math.sqrt(g / nom_height)
    icp0 = torch.zeros((1,3), device=device)
    icp0[0,0] = vx / omega    # x-axis only

    # 4) stance foot pos (zero Z)
    pos = amber.data.body_pos_w           # [1, bodies, 3]
    B   = pos.shape[1]
    feet = pos[:, [B-1, B-2], :]          # [1,2,3]
    # print(feet)
    # phase clock
    tp    = (sim_time % (2*Tswing)) / (2*Tswing)
    phi_c = math.sin(2*math.pi*tp) / math.sqrt(math.sin(2*math.pi*tp)**2 + Tswing)
    idx   = int((0.5 - 0.5 * np.sign(phi_c)))
    stance = feet[:, idx, :].clone()
    stance[0,2] = 0.0
    # print("stance:",stance)
    # print("phase",phi_c)
    # 5) local/global transforms
    def to_local(v):
        return quat_rotate(
            yaw_quat(quat_inv(amber.data.root_quat_w)), v
        )
    def to_global(v):
        return quat_rotate(
            yaw_quat(amber.data.root_quat_w), v
        )

    # 6) capture‐point final in local
    expT = math.exp(omega * Tswing)
    icpf = expT*icp0 + (1-expT)*to_local(r - stance)
    icpf[0,2] = 0.0

    # 7) offset b
    sd = abs(des_vel[0])*Tswing
    bx = sd / (expT - 1.0)
    # y-bias removed
    b  = torch.tensor([[bx, 0.0, 0.0]], device=device)

    # 8) clip
    p_local = icpf.clone()
    p_local[0,0] = torch.clamp(icpf[0,0]-b[0,0], -0.5, 0.5)
    p_local[0,1] = 0.0   # Y always zero for planar x-motion
    # 9) back to world
    p = to_global(p_local) + r
    p[0,1] = 0.0  # force Y in world to 0
    p[0,2] = 0.0

    # 10) optional USD viz
    if visualize:
        path = "/World/debug/future_step_0"
        if not stage.GetPrimAtPath(path):
            sph = UsdGeom.Sphere.Define(stage, path)
            sph.GetRadiusAttr().Set(0.02)
        else:
            sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
        coords = p[0].cpu().numpy().tolist()    # -> [float, float, float]
        UsdGeom.XformCommonAPI(sph).SetTranslate(
            Gf.Vec3d(*coords)
        )
        com_path = "/World/debug/com_sphere"
        if not stage.GetPrimAtPath(com_path):
            com_sph = UsdGeom.Sphere.Define(stage, com_path)
            com_sph.GetRadiusAttr().Set(0.02)
        else:
            com_sph = UsdGeom.Sphere(stage.GetPrimAtPath(com_path))
        # grab COM world‐position from body_pos_w index 3
        com_xyz = amber.data.body_pos_w.cpu().numpy()[0, 3, :].tolist()  # [x,y,z]
        UsdGeom.XformCommonAPI(com_sph).SetTranslate(
            Gf.Vec3d(*com_xyz)
        )
        # tint it red
        com_sph.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
    return p  # a [1,3] tensor


def compute_step_location_local(
    sim_time: float,
    scene,
    args_cli,
    nom_height: float,
    Tswing: float,
    wdes: float,
    visualize: bool = True
) -> torch.Tensor:
    """
    Compute next foothold in world‐frame using a local‐frame LIP ICP method,
    for all envs in `scene["Amber"]`. Returns [n_envs×3] torch.Tensor.
    """
    amber  = scene["Amber"]
    device = amber.data.default_root_state.device
    n_envs = args_cli.num_envs

    # 1) commanded velocity in local frame [N,2]
    cmd_np  = np.array(args_cli.desired_vel, dtype=np.float32)
    command = torch.from_numpy(cmd_np[:2]).to(device) \
                    .unsqueeze(0).repeat(n_envs, 1)

    # 2) COM position in world [N,3]
    r = amber.data.body_pos_w[:, 3, :]                      # [N, 3]
    # print(r)
    # 3) build ICP base
    g     = 9.81
    omega = math.sqrt(g / nom_height)
    icp_0 = torch.zeros((n_envs, 3), device=device)
    icp_0[:, :2] = command[:, :2] / omega

    # 4) get last two foot positions in world [N,2,3]
    pos      = amber.data.body_pos_w          # [N, bodies, 3]
    B        = pos.shape[1]
    foot_pos = pos[:, [B-1, B-2], :]           # [N,2,3]

    # 5) compute phase clock and choose stance index
    tp    = (sim_time % (2*Tswing)) / (2*Tswing)
    phi_c = torch.tensor(
        math.sin(2*math.pi*tp) / math.sqrt(math.sin(2*math.pi*tp)**2 + Tswing),
        device=device
    )
    stance_idx = int(0.5 - 0.5 * torch.sign(phi_c).item())
    stance_foot = foot_pos[:, stance_idx, :].clone()
    stance_foot[:, 2] = 0.0

    # 6) helper transforms
    def to_local(v, quat):
        return quat_rotate(yaw_quat(quat_inv(quat)), v)
    def to_global(v, quat):
        return quat_rotate(yaw_quat(quat), v)

    # 7) final ICP in local frame
    exp_omT = math.exp(omega * Tswing)
    icp_f = (
        exp_omT * icp_0
        + (1 - exp_omT) * to_local(r - stance_foot, amber.data.root_quat_w)
    )
    icp_f[:, 2] = 0.0

    # 8) compute offset b = [bx, by, 0]
    sd = torch.abs(command[:, 0]) * Tswing
    wd = wdes * torch.ones(n_envs, device=device)
    bx = sd / (exp_omT - 1.0)
    by = torch.sign(phi_c) * wd / (exp_omT + 1.0)
    b  = torch.stack((bx, by, torch.zeros_like(bx)), dim=1)

    # 9) clip in local frame
    p_local = icp_f.clone()
    p_local[:, 0] = torch.clamp(icp_f[:, 0] - b[:, 0], -0.5, 0.5)
    p_local[:, 1] = torch.clamp(icp_f[:, 1] - b[:, 1], -0.3, 0.3)

    # 10) back to world frame and zero z
    p = to_global(p_local, amber.data.root_quat_w) + r
    p[:, 2] = 0.0

    # 11) optional USD visualization
    if visualize:
        stage = omni.usd.get_context().get_stage()
        com = amber.data.body_pos_w[:, 3, :]     # [N,3] tensor
        for i in range(n_envs):
            prim_path = f"/World/debug/future_step_{i}"
            if not stage.GetPrimAtPath(prim_path):
                sph = UsdGeom.Sphere.Define(stage, prim_path)
                sph.GetRadiusAttr().Set(0.02)
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(prim_path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(
                Gf.Vec3d(p[i,0].item(), p[i,1].item(), p[i,2].item())
            )
        for i in range(com.shape[0]):
            prim_path = f"/World/debug/com_sphere_{i}"
            # define or fetch the sphere:
            if not stage.GetPrimAtPath(prim_path):
                sph = UsdGeom.Sphere.Define(stage, prim_path)
                sph.GetRadiusAttr().Set(0.1)  # make this “big” relative to your 0.02 step‐spheres
                # red color:
                sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(prim_path))
            # move it to the CoM:
            x, y, z = com[i].cpu().numpy().tolist()
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(x, y, z))
    # 12) stash into scene for later use
    if not hasattr(scene, "current_des_step"):
        scene.current_des_step = torch.zeros((n_envs, 3), device=device)
    scene.current_des_step[:] = p

    return p




def run_simulator(sim, scene, policy, simulation_app, args_cli):
    """
    sim            : isaaclab.sim.SimulationContext
    scene          : isaaclab.scene.InteractiveScene
    policy         : RLPolicy
    simulation_app : the AppLauncher.app instance
    args_cli       : parsed CLI args (num_envs, desired_vel, csv_out, etc.)
    """
    sim_dt     = sim.get_physics_dt()
    sim_time   = 0.0
    count      = 0
    just_reset = False

    amber    = scene["Amber"]
    device   = amber.data.default_root_state.device
    n_envs   = args_cli.num_envs
    # assert n_envs == 1, "Policy loop only supports a single env (0)."

    # ─── CSV SET-UP ───
    csv_path = args_cli.csv_out.expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["step", "sim_time", "env_id", *amber.data.joint_names])
    # Reset setup
    # track last reset step per env
    last_reset_step = torch.full(
        (n_envs,), -1_000_000, dtype=torch.int32, device=device
    )
    COOLDOWN = 10  # frames to wait before allowing another reset


    try:
        while simulation_app.is_running():
            # pump Kit events
            simulation_app.update()

            # ─── Gather sensor data from env 0 ───
            qpos = amber.data.joint_pos.cpu().numpy()[0]      # (7,)
            qvel = amber.data.joint_vel.cpu().numpy()[0]      # (7,)
            root = amber.data.root_state_w.cpu().numpy()[0]   # (13,)
            ori  = root[3:7]
            quat = np.array([ori[3], ori[0], ori[1], ori[2]], dtype=np.float32)
            body_ang_vel = root[10:13].astype(np.float32)     # (3,)

            des_vel = np.array(args_cli.desired_vel, dtype=np.float32)

            # ─── Build observation & run policy ───
            obs = policy.create_obs(
                qjoints=     qpos,
                body_ang_vel=body_ang_vel,
                qvel=        qvel,
                time=        sim_time,
                projected_gravity=get_projected_gravity(quat),
                des_vel=     des_vel,
            )
            _ = policy.get_action(obs.to(device))   # updates policy.action_isaac
            action_isaac = policy.action_isaac       # (4,)

            # ─── Convert to torch targets ───
            default_all   = amber.data.default_joint_pos.clone()  # (1,7)
            target_tensor = torch.from_numpy(action_isaac).to(device).unsqueeze(0)  # (1,4)
            joint_targets = default_all.clone()

            # scatter into exactly those 4 actuated joints
            actuated_names = ["q1_left","q2_left","q1_right","q2_right"]
            all_names      = list(amber.data.joint_names)
            for i, name in enumerate(actuated_names):
                idx = all_names.index(name)
                joint_targets[:, idx] = target_tensor[0, i]

            amber.set_joint_position_target(joint_targets)

            # ─── Log CSV ───
            cur_pos = amber.data.joint_pos.cpu().numpy()
            for env_id in range(n_envs):
                writer.writerow([count, sim_time, env_id, *cur_pos[env_id]])
            if count % 100 == 0:
                csv_fh.flush()

            # ─── Step physics ───
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            sim_time += sim_dt
            count   += 1            
        
            # ─────────────────────────── LIP model ────────────────────────────────────
            # next_foot = compute_step_location_direct(
            #     sim_time=sim_time,
            #     scene=scene,
            #     args_cli=args_cli ,
            #     des_vel=args_cli.desired_vel,   # your [vx,vy,vyaw]
            #     nom_height=0.3,
            #     Tswing=PERIOD/2.0,
            #     wdes=WDES,                      # import WDES or hardcode it
            #     visualize=True
            # )
            next_foot = compute_step_location_local(
                sim_time = sim_time,
                scene    = scene,
                args_cli = args_cli,
                nom_height=0.1,
                Tswing    = PERIOD/2.0,
                wdes      = WDES,
                visualize = True
            )
            # print(f"[INFO] Next desired step: {next_foot.cpu().numpy()}")

            # ─────────────────────────── contact-based reset of torso ──────────────────
            forces      = scene["contact_forces"].data.net_forces_w  # (n_envs, n_sensors, 3)
            contact_sum = forces.abs().sum(dim=(1, 2))                # (n_envs,)
            fallen= (forces.abs().sum(dim=(1,2))>0.05)
            to_reset = fallen & ((count - last_reset_step) > COOLDOWN)

            if to_reset.any() :
                # --- masked reset for only the fallen envs ---
                # 1) compute default root states in world
                default_root = amber.data.default_root_state.clone()
                default_root[:, :3] += scene.env_origins

                # 2) overwrite fallen envs' pose + zero velocities
                root_state = amber.data.root_state_w.clone()
                root_state[to_reset] = default_root[to_reset]
                amber.write_root_pose_to_sim(root_state[:, :7])
                amber.write_root_velocity_to_sim(root_state[:, 7:])

                # 3) restore joint positions & velocities for fallen envs
                cur_jpos = amber.data.joint_pos.clone()
                cur_jvel = amber.data.joint_vel.clone()
                cur_jpos[to_reset] = amber.data.default_joint_pos[to_reset]
                cur_jvel[to_reset] = amber.data.default_joint_vel[to_reset]
                amber.write_joint_state_to_sim(cur_jpos, cur_jvel)

                # 4) push writes, step once to flush sensors
                scene.write_data_to_sim()
                sim.step(); scene.update(sim_dt)
                # 5) record reset step
                last_reset_step[to_reset] = count
                # skip remainder of loop so new random kicks don't apply this frame

                # if not to_reset.any():
                #     just_reset = False
                
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C – CSV saved at:", csv_path)
    finally:
        csv_fh.close()