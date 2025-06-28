#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from isaaclab.app import AppLauncher
from datetime import datetime
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Amber robot in Isaac Sim (plain or rough environment)."
    )
    parser.add_argument(
        "--env",
        choices=["plain", "rough"],
        default="plain",
        help="Which Amber environment to load"
    )
    # bring in all the usual IsaacLab CLI args (e.g. --renderer, --device)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Launch the Omniverse app
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app  # keeps running until closed :contentReference[oaicite:0]{index=0}

    # 2) Import the right Amber Env config class
    if args.env == "plain":
        from amber_env_cfg import AmberEnvCfg as EnvCfg
    else:
        from amber_rough_env_cfg import AmberRoughEnvCfg as EnvCfg

    # 3) Instantiate and apply the config
    env_cfg = EnvCfg()
    # (Later you can override env_cfg parameters here or via CLI if you wire up Hydra)

    # 4) Build the environment
    # This assumes your configclass has a method to register/build a ManagerBasedRLEnv.
    # Adjust import/path as needed.
    from isaaclab.managers import build_env
    env = build_env(env_cfg)

    # 5) Reset and step loop
    obs = env.reset()
    print(f"[INFO] Started Amber ({args.env}) env at {datetime.now():%Y-%m-%d %H:%M:%S}")

    while simulation_app.is_running():
        # here you’d plug in your controller/LIP model later
        action = env.action_space.sample()  # placeholder
        obs, reward, done, info = env.step(action)

        # advance the simulator and sync
        env.render()             # if you need to render observations
        simulation_app.update()  # pump Omniverse

        if done:
            obs = env.reset()

    # 6) Cleanup
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
