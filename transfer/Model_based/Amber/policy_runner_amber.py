# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import time
from pathlib import Path

# 1) Base CLI + AppLauncher (must come before any Isaac Lab imports)
from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(
    description="Run Amber with a learned policy checkpoint instead of random actions."
)
# RSL-RL args: --experiment_name, --run_name, --resume, --load_run, --checkpoint, etc.
cli_args.add_rsl_rl_args(parser)

# playback args (same as play.py)
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=200, help="Number of frames to record.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Load published checkpoint.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time, if possible.")

# env/task selection
parser.add_argument(
    "--task",
    type=str,
    default="Amber-flat-vel",
    help="Gym ID of the Amber task (e.g. Amber-flat-vel or Amber-rough-vel).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and fall back to USD I/O operations."
)
# number of envs (for vectorized eval)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
# append AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
if args_cli.video:
    # ensure cameras are on if we're recording
    args_cli.enable_cameras = True

# 2) Launch Omniverse / Isaac App
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3) Now import everything else
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.utils.dict import print_dict
import robot_rl.tasks  # noqa: F401

def main():
    # — load env config from registry (uses Amber_env_cfg) —
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # — load agent config from CLI —
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # find checkpoint to load
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] No published checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # — make the Gym environment —
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # if multi-agent under the hood, collapse to single agent
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # optional video wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "rollout"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video rollout with:", video_kwargs)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # vectorize + clip actions
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # — build & load runner —
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # get the TorchScript policy (for inference)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # also extract the raw network if you ever want to re-export
    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    # optional: export a fresh JIT/ONNX copy
    export_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_dir, filename="policy.onnx")

    # rollout loop
    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()
    step = 0
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        step += 1

        # stop after recording length, if requested
        if args_cli.video and step >= args_cli.video_length:
            break

        # enforce real-time pacing if desired
        dt_remain = dt - (time.time() - t0)
        if args_cli.real_time and dt_remain > 0:
            time.sleep(dt_remain)

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
