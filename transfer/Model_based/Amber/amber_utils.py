import csv
import numpy as np
import torch
from pxr import Gf, UsdGeom
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
import omni.usd
import math
from source.robot_rl.robot_rl.tasks.manager_based.robot_rl.amber.amber_env_cfg import PERIOD,WDES
import casadi as ca
reference_step = ca.Function.load("transfer/Model_based/Amber/amber_reference_step.casadi")
from pathlib import Path
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


def debug_print_joints(amber, env_id: int = 0):
    """
    Prints the current q1_left, q2_left, q1_right, q2_right angles for the given env.
    amber   : scene["Amber"]
    env_id  : which env index to print (0…n_envs-1)
    """
    # pull names & positions
    names = list(amber.data.joint_names)
    pos   = amber.data.joint_pos.cpu().numpy()  # shape (n_envs, n_joints)
    
    # lookup indices
    idx = { 
        "q1_left":  names.index("q1_left"),
        "q2_left":  names.index("q2_left"),
        "q1_right": names.index("q1_right"),
        "q2_right": names.index("q2_right"),
    }
    q = pos[env_id]  # 1D array of length n_joints

    print(f"[DEBUG][env {env_id}] "
          f"q1_left={180/math.pi*q[idx['q1_left']]:.4f}, "
          f"q2_left={180/math.pi*q[idx['q2_left']]:.4f}, "
          f"q1_right={180/math.pi*q[idx['q1_right']]:.4f}, "
          f"q2_right={180/math.pi*q[idx['q2_right']]:.4f}")


def feed_reference_trajectory(sim_time, scene, args_cli, ref_dir="/home/s-ritwik/src/cusadi/Amber/references"):
    """
    Load (once) and apply precomputed gait references to Amber.
    Call this each step instead of the policy to drive pure replay.
    """
    amber = scene["Amber"]
    device = amber.data.default_root_state.device
    # print(amber.data.joint_names)
    # —— one‐time load & cache ——
    if not hasattr(feed_reference_trajectory, "loaded"):
        ref_path = Path(ref_dir)
        vxs      = np.load(ref_path / "amber_vxs.npy")
        ts       = np.load(ref_path / "amber_reference_ts.npy")
        q_refs   = np.load(ref_path / "amber_reference_qs.npy")           # (n_vx, N, 4)
        T        = float(ts.max())
        # pick the closest precomputed speed
        vx       = float(args_cli.desired_vel[0])
        ix       = int(np.argmin(np.abs(vxs - vx)))

        feed_reference_trajectory.vxs            = vxs
        feed_reference_trajectory.ts             = ts
        feed_reference_trajectory.q_refs         = q_refs
        feed_reference_trajectory.T              = T
        feed_reference_trajectory.ix             = ix
        feed_reference_trajectory.actuated_names = actuated_names = ["q1_left","q2_left","q1_right","q2_right"]

        feed_reference_trajectory.loaded         = True

    ts       = feed_reference_trajectory.ts
    q_refs   = feed_reference_trajectory.q_refs
    T        = feed_reference_trajectory.T
    ix       = feed_reference_trajectory.ix
    names    = feed_reference_trajectory.actuated_names

    # —— compute current phase & indices ——
    phase   = sim_time % T
    idx_l   = int(np.argmin(np.abs(ts - phase)))
    # half-cycle shift for right leg:
    phase_r = (phase + T/2.0) % T
    idx_r   = int(np.argmin(np.abs(ts - phase_r)))

    # extract joint arrays
    # q_l = q_refs[ix, idx_l, :2]   # left-leg joints
    # q_r = q_refs[ix, idx_r, 2:4]  # right-leg joints
    q1_l = float(q_refs[ix, idx_l, 0])
    q2_l = float(q_refs[ix, idx_l, 1])
    q1_r = float(q_refs[ix, idx_r, 2])
    q2_r = float(q_refs[ix, idx_r, 3])
    # —— scatter into a full (1×7) target and send to sim ——
    default_all   = amber.data.default_joint_pos.clone()  # (1,7) tensor
    print(default_all)
    joint_targets = default_all.clone()
    # target_np     = np.hstack([q_l, q_r])                # shape (4,)
    target_np = np.array([q1_l, q2_l,q1_r, q2_r], dtype=np.float32)

    target_tensor = torch.from_numpy(target_np).to(device).unsqueeze(0)  # (1,4)
    all_names     = list(amber.data.joint_names)
    # print("Available joint_names:")
    # for j_idx, j_name in enumerate(all_names):
    #     print(f"  [{j_idx}] {j_name}")

    # DEBUG: print mapping from actuated_names to indices
    print("Mapping actuated_names -> joint_names indices:")
    for i, name in enumerate(feed_reference_trajectory.actuated_names):
        if name in all_names:
            idx = all_names.index(name)
            val = target_tensor[0, i].item()
            print(f"  actuated[{i}] '{name}' -> joint_names[{idx}] (setting value {val*180/math.pi:.4f})")
            joint_targets[:, idx] = target_tensor[0, i]
        else:
            print(f"  actuated[{i}] '{name}' NOT FOUND in joint_names!")
    # for i, name in enumerate(names):
    #     idx = all_names.index(name)
    #     joint_targets[:, idx] = target_tensor[0, i]

    amber.set_joint_position_target(joint_targets)


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
    but only update once every Tswing (half‐cycle). Returns [n_envs×3].
    """
    amber  = scene["Amber"]
    device = amber.data.default_root_state.device
    n_envs = args_cli.num_envs

    # --- static storage for last update ---
    if not hasattr(compute_step_location_local, "_last_time"):
        # force an immediate first‐update at t=0
        compute_step_location_local._last_time = -Tswing
        compute_step_location_local._last_p    = torch.zeros((n_envs, 3), device=device)
        # capture each foot’s original lateral world‐Y offsets:
        pos0 = amber.data.body_pos_w               # (n_envs, n_bodies, 3)
        B    = pos0.shape[1]
        feet0 = pos0[:, [B-1, B-2], :]             # (n_envs, 2, 3)
        # store as (n_envs,2) so we can pick left/right later
        compute_step_location_local._orig_foot_y = feet0[:, :, 1].clone()

    # check if we crossed a half‐cycle boundary
    if (sim_time - compute_step_location_local._last_time) >= Tswing:
        # ---- do a fresh LIP‐ICP compute ----
        # 1) commanded velocity in local frame [N,2]
        cmd_np  = np.array(args_cli.desired_vel, dtype=np.float32)
        command = torch.from_numpy(cmd_np[:2]).to(device) \
                        .unsqueeze(0).repeat(n_envs, 1)
        # print(command)
        # 2) COM position in world from body index 3 [N,3]
        r = amber.data.body_pos_w[:, 3, :]                 
        # print("com:",r)
        # 3) build ICP base
        omega = math.sqrt(9.81 / nom_height)
        icp_0 = torch.zeros((n_envs, 3), device=device)
        icp_0[:, :2] = command[:, :2] / omega

        # 4) last two foot positions [N,2,3]
        pos      = amber.data.body_pos_w               
        B        = pos.shape[1]
        foot_pos = pos[:, [B-1, B-2], :]
        # print(foot_pos)
        # 5) phase clock → stance foot
        tp    = (sim_time % (2*Tswing)) / (2*Tswing)
        phi_c = torch.tensor(
            math.sin(2*math.pi*tp) / math.sqrt(math.sin(2*math.pi*tp)**2 + Tswing),
            device=device
        )
        stance_idx  = int(0.5 - 0.5 * torch.sign(phi_c).item())
        stance_foot = foot_pos[:, stance_idx, :].clone()
        stance_foot[:, 2] = 0.0

        # 6) transforms
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

        # 8) compute bias b
        sd = torch.abs(command[:, 0]) * Tswing
        wd = wdes * torch.ones(n_envs, device=device)
        bx = sd / (exp_omT - 1.0)
        by = torch.sign(phi_c) * wd / (exp_omT + 1.0)
        b  = torch.stack((bx, by, torch.zeros_like(bx)), dim=1)

        # 9) clip in local
        p_local = icp_f.clone()
        p_local[:, 0] = torch.clamp(icp_f[:, 0] - b[:, 0], -0.5, 0.5)
        p_local[:, 1] = torch.clamp(icp_f[:, 1] - b[:, 1], -0.3, 0.3)

        # 10) back to world, zero Z
        p = to_global(p_local, amber.data.root_quat_w) + r
        # print("quat root:",amber.data.root_quat_w[3])
        p[:, 2] = 0.0
        # --- OVERRIDE lateral step to your robot’s original Y ---
        # swing foot index = the opposite of stance_idx
        swing_idx = 1 - stance_idx
        # orig_foot_y: (n_envs,2)  → pick the correct column
        orig_y = compute_step_location_local._orig_foot_y[:, swing_idx]
        p[:, 1] = orig_y
        # store for reuse
        compute_step_location_local._last_time = sim_time
        compute_step_location_local._last_p    = p.clone()

    else:
        # reuse the last computed target
        p = compute_step_location_local._last_p

    # --- USD visualization (always show the stored target + COM) ---
    if visualize:
        stage = omni.usd.get_context().get_stage()

        # future‐step spheres
        for i in range(n_envs):
            path = f"/World/debug/future_step_{i}"
            if not stage.GetPrimAtPath(path):
                sph = UsdGeom.Sphere.Define(stage, path)
                sph.GetRadiusAttr().Set(0.02)
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*p[i].cpu().tolist()))

        # COM as big red sphere
        com = amber.data.body_pos_w[:, 3, :]
        for i in range(n_envs):
            path = f"/World/debug/com_sphere_{i}"
            if not stage.GetPrimAtPath(path):
                com_sph = UsdGeom.Sphere.Define(stage, path)
                com_sph.GetRadiusAttr().Set(0.1)
                com_sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            else:
                com_sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
            UsdGeom.XformCommonAPI(com_sph).SetTranslate(Gf.Vec3d(*com[i].cpu().tolist()))

    # stash into scene for downstream use
    if not hasattr(scene, "current_des_step"):
        scene.current_des_step = torch.zeros((n_envs, 3), device=device)
    scene.current_des_step[:] = p

    return p


def sinusoid_test(amber, sim_time, amplitude=0.5, frequency=0.5):
    """
    Drive q2_left and q2_right joints on a sine wave.
    
    amber     : scene["Amber"] handle
    sim_time  : current simulation time (seconds)
    amplitude : peak deflection in radians
    frequency : oscillation rate in Hz
    """
    # 1) grab the default joint‐pose tensor (n_envs × n_joints)
    target = amber.data.default_joint_pos.clone()
    
    # 2) compute sine value
    phase  = 2 * math.pi * frequency * sim_time
    s      = amplitude * math.sin(phase)
    
    # 3) find indices
    names  = amber.data.joint_names  # list of strings
    idx_l  = names.index("q2_left")
    idx_r  = names.index("q2_right")
    print("angle given:",s)
    # 4) assign to all envs
    target[:, idx_l] = s
    target[:, idx_r] = s
    
    # 5) send to sim
    amber.set_joint_position_target(target)


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
            actuated_names = actuated_names = ["q1_left","q1_right","q2_left","q2_right"]

            all_names      = list(amber.data.joint_names)
            for i, name in enumerate(actuated_names):
                idx = all_names.index(name)
                joint_targets[:, idx] = target_tensor[0, i]

            # amber.set_joint_position_target(joint_targets)
            # sinusoid_test(amber, sim_time, amplitude=0.5, frequency=2)

            feed_reference_trajectory(sim_time, scene, args_cli)
            # ─── Log CSV ───
            cur_pos = amber.data.joint_pos.cpu().numpy()
            for env_id in range(n_envs):
                writer.writerow([count, sim_time, env_id, *cur_pos[env_id]])
            if count % 100 == 0:
                csv_fh.flush()
            
            # ─── Step physics ───
            scene.write_data_to_sim()
            sim.step()
            qpos = amber.data.joint_pos.cpu().numpy()[0]
            qvel = amber.data.joint_vel.cpu().numpy()[0]
            print(f" q2_left pos={np.degrees(qpos[5]):.4f}°, vel={np.degrees(qvel[5]):.4f}°/s")
            print(f" q2_right pos={np.degrees(qpos[6]):.4f}°, vel={np.degrees(qvel[6]):.4f}°/s")    
            scene.update(sim_dt)
            for eid in range(amber.data.joint_pos.shape[0]):
                debug_print_joints(amber, env_id=eid)
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
                nom_height=1.38,
                Tswing    = PERIOD/2.0,
                wdes      = WDES,
                visualize = True
            )
            # print(f"[INFO] Next desired step: {next_foot.cpu().numpy()}, time:{sim_time}")

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