#!/usr/bin/env python3
"""
Sequentially play trained policies. Can take an explicit list of policy files,
sweep a directory for a specific file, or sweep checkpoints from one or more
runs at a given interval.
"""
import argparse
import os
import subprocess
import sys
import pickle
import glob
import re
import itertools
from typing import List, Optional

import torch
import numpy as np
import pandas as pd


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion to a yaw angle."""
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return yaw

# ---

def find_checkpoints_in_dir(directory: str, start: int, step: int, end: Optional[int]) -> List[str]:
    """
    Finds and filters checkpoints in a single directory based on a given range and step.
    """
    if not os.path.isdir(directory):
        print(f"[WARNING] Directory not found: {directory}, skipping.")
        return []

    all_ckpts = glob.glob(os.path.join(directory, "model_*.pt"))
    if not all_ckpts:
        return []

    ckpt_map = {}
    for ckpt_path in all_ckpts:
        match = re.search(r'model_(\d+)\.pt', os.path.basename(ckpt_path))
        if match:
            ckpt_map[int(match.group(1))] = ckpt_path

    if not ckpt_map:
        return []

    # Determine the end number if not explicitly provided
    last_available_ckpt = max(ckpt_map.keys())
    end_num = end if end is not None else last_available_ckpt

    # Find all checkpoints that fall within the specified range
    selected_paths = []
    for i in range(start, end_num + 1, step):
        if i in ckpt_map:
            selected_paths.append(ckpt_map[i])

    return sorted(selected_paths)

# ---

def save_summary(play_dirs: list[str]) -> None:
    """Calculates and saves summary metrics from all playback runs into a single CSV file."""
    DT = 0.02
    ROBOT_MASS = 35  # Mass of the robot in kg. Adjust if necessary.
    GRAVITY = 9.81     # Acceleration due to gravity in m/s^2.
    
    run_data = []
    print("\n--- Calculating Summary Metrics for all runs ---")

    for play_dir in play_dirs:
        try:
            playback_folder_name = os.path.basename(play_dir)
            
            # Load all required data files
            with open(os.path.join(play_dir, "base_velocity.pkl"), "rb") as f: cmd_vel_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "root_pos.pkl"), "rb") as f: root_pos_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "root_velocity.pkl"), "rb") as f: root_vel_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "terminations.pkl"), "rb") as f: terminations_raw = torch.tensor(pickle.load(f), dtype=torch.bool)
            with open(os.path.join(play_dir, "root_ang_vel_w.pkl"), "rb") as f: root_ang_vel_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "root_quat_w.pkl"), "rb") as f: root_quat_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "joint_torques.pkl"), "rb") as f: joint_torques_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "joint_velocities.pkl"), "rb") as f: joint_vel_raw = torch.tensor(pickle.load(f))
            
            # --- Tracking & RMSE Calculations ---
            did_fall = torch.any(terminations_raw, dim=0)
            fall_rate = torch.mean(did_fall.float()).item() * 100.0

            linear_cmd_vel = torch.stack([cmd_vel_raw[..., 0], cmd_vel_raw[..., 1], torch.zeros_like(cmd_vel_raw[..., 0])], dim=-1)
            cmd_pos_integrated = torch.cumsum(linear_cmd_vel * DT, dim=0)
            root_pos_relative = root_pos_raw - root_pos_raw[0, :, :]
            vel_rmse_x = torch.sqrt(torch.mean((cmd_vel_raw[..., 0] - root_vel_raw[..., 0]) ** 2)).item()
            vel_rmse_y = torch.sqrt(torch.mean((cmd_vel_raw[..., 1] - root_vel_raw[..., 1]) ** 2)).item()
            pos_rmse_x = torch.sqrt(torch.mean((cmd_pos_integrated[..., 0] - root_pos_relative[..., 0]) ** 2)).item()
            pos_rmse_y = torch.sqrt(torch.mean((cmd_pos_integrated[..., 1] - root_pos_relative[..., 1]) ** 2)).item()

            cmd_yaw_vel = cmd_vel_raw[..., 2]
            actual_yaw_vel = root_ang_vel_raw[..., 2]
            vel_rmse_yaw = torch.sqrt(torch.mean((cmd_yaw_vel - actual_yaw_vel) ** 2)).item()
            cmd_yaw_pos = torch.cumsum(cmd_yaw_vel * DT, dim=0)
            actual_yaw_pos = quat_to_yaw(root_quat_raw)
            actual_yaw_pos_unwrapped = torch.from_numpy(np.unwrap(actual_yaw_pos.cpu().numpy(), axis=0))
            actual_yaw_pos_relative = actual_yaw_pos_unwrapped - actual_yaw_pos_unwrapped[0, :]
            pos_rmse_yaw = torch.sqrt(torch.mean((cmd_yaw_pos - actual_yaw_pos_relative) ** 2)).item()

            avg_pos_rmse = np.mean([pos_rmse_x, pos_rmse_y, pos_rmse_yaw])
            avg_vel_rmse = np.mean([vel_rmse_x, vel_rmse_y, vel_rmse_yaw])
            
            # --- Efficiency (Cost of Transport) Calculation ---
            final_pos_per_env = root_pos_relative[-1, ~did_fall, :]
            total_distance = torch.sqrt(final_pos_per_env[:, 0]**2 + final_pos_per_env[:, 1]**2).mean().item() if final_pos_per_env.shape[0] > 0 else 0.0
            power_per_step = torch.sum(torch.abs(joint_torques_raw * joint_vel_raw), dim=-1)
            total_energy = torch.sum(power_per_step, dim=0).mean().item() * DT
            weight = ROBOT_MASS * GRAVITY
            cost_of_transport = total_energy / (weight * total_distance) if total_distance > 0.01 else 0.0
            
            # --- Smoothness Metrics Calculation ---
            joint_accel = torch.diff(joint_vel_raw, dim=0) / DT
            avg_joint_accel = torch.mean(torch.norm(joint_accel, dim=-1)).item()
            torque_rate = torch.diff(joint_torques_raw, dim=0) / DT
            avg_torque_rate = torch.mean(torch.norm(torque_rate, dim=-1)).item()
            
            run_data.append({
                "Policy": playback_folder_name.replace("playback_", ""),
                "Cost of Transport": cost_of_transport,
                "Avg Joint Acceleration": avg_joint_accel,
                "Avg Torque Rate": avg_torque_rate,
                "Fall Rate": fall_rate,
                "Average Position RMSE": avg_pos_rmse,
                "pos_rmse_x": pos_rmse_x, "pos_rmse_y": pos_rmse_y, "pos_rmse_yaw": pos_rmse_yaw,
                "Average Velocity RMSE": avg_vel_rmse,
                "vel_rmse_x": vel_rmse_x, "vel_rmse_y": vel_rmse_y, "vel_rmse_yaw": vel_rmse_yaw,
            })
            print(f"   - Processed: {playback_folder_name}")

        except FileNotFoundError as e:
            print(f"[WARNING] Could not load data for '{os.path.basename(play_dir)}': {e}. Skipping.")
            continue
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while processing '{os.path.basename(play_dir)}': {e}. Skipping.")
            continue

    if not run_data:
        print("[ERROR] No data found to calculate summary.")
        return

    df = pd.DataFrame(run_data)
    df = df.sort_values(by="Policy")

    final_column_order = [
        "Policy", "Cost of Transport", "Avg Joint Acceleration", "Avg Torque Rate", 
        "Average Position RMSE", "Average Velocity RMSE", "Fall Rate",
        "pos_rmse_x", "pos_rmse_y", "pos_rmse_yaw",
        "vel_rmse_x", "vel_rmse_y", "vel_rmse_yaw",
    ]
    df = df[[col for col in final_column_order if col in df.columns]]
    
    output_dir = os.getcwd()
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"\n[SUCCESS] Aggregated summary saved to: {summary_path}")

# ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run policy playback.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--policy_paths", nargs="+", metavar="PATH", help="Explicit list of policy files to play.")
    group.add_argument("--sweep_dir", type=str, metavar="DIR", help="Directory to search recursively for a specific file (e.g., model_999.pt).")
    group.add_argument("--checkpoint_dir", type=str, metavar="DIR", help="Directory of a single run to play checkpoints from at intervals.")
    group.add_argument("--sweep_checkpoints_dir", type=str, metavar="DIR", help="Parent directory containing multiple runs to sweep for checkpoints.")

    parser.add_argument("--ckpt_start", type=int, default=0, help="Start number for checkpoint sweep.")
    parser.add_argument("--ckpt_step", type=int, default=500, help="Interval for checkpoint sweep.")
    parser.add_argument("--ckpt_end", type=int, help="End number for checkpoint sweep (inclusive). Defaults to last available.")
    parser.add_argument("--save_summary", action="store_true", default=False, help="Generate a CSV summary of all runs.")
    
    
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument("--sweep", nargs='+', help="Perform a grid sweep (all combinations) of playback parameters. Format: 'key:v1,v2,...'")
    sweep_group.add_argument("--paired_sweep", nargs='+', help="Perform a paired sweep (element-wise combinations). All lists must be same length. Format: 'key:v1,v2,...'")

    args, passthrough = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    play_script = os.path.join(script_dir, "play_policy.py")
    if not os.path.exists(play_script):
        sys.exit(f"[ERROR] play_policy.py not found at {play_script}")

    # --- Find Policies to Run ---
    policy_paths_to_run = []
    if args.checkpoint_dir:
        policy_paths_to_run = find_checkpoints_in_dir(args.checkpoint_dir, args.ckpt_start, args.ckpt_step, args.ckpt_end)
    elif args.sweep_checkpoints_dir:
        print(f"[INFO] Starting multi-directory checkpoint sweep in: {args.sweep_checkpoints_dir}")
        try:
            run_dirs = [entry.path for entry in os.scandir(args.sweep_checkpoints_dir) if entry.is_dir()]
        except FileNotFoundError:
            sys.exit(f"[ERROR] Parent sweep directory not found: {args.sweep_checkpoints_dir}")
        
        print(f"[INFO] Found {len(run_dirs)} potential run directories. Sweeping each...")
        for run_dir in sorted(run_dirs):
            paths_from_run = find_checkpoints_in_dir(run_dir, args.ckpt_start, args.ckpt_step, args.ckpt_end)
            if paths_from_run:
                print(f"   - Found {len(paths_from_run)} checkpoints in {os.path.basename(run_dir)}")
                policy_paths_to_run.extend(paths_from_run)
    elif args.sweep_dir:
        print(f"[INFO] Sweeping for the latest checkpoint in each subdirectory of: {args.sweep_dir}")
        try:
            # Find all immediate subdirectories
            run_dirs = [entry.path for entry in os.scandir(args.sweep_dir) if entry.is_dir()]
        except FileNotFoundError:
            sys.exit(f"[ERROR] Sweep directory not found: {args.sweep_dir}")

        for run_dir in run_dirs:
            ckpt_map = {}
            # Find all model files in the subdirectory
            for ckpt_path in glob.glob(os.path.join(run_dir, "model_*.pt")):
                match = re.search(r'model_(\d+)\.pt', os.path.basename(ckpt_path))
                if match:
                    ckpt_map[int(match.group(1))] = ckpt_path
            
            # If checkpoints were found, add the one with the highest number
            if ckpt_map:
                latest_ckpt_num = max(ckpt_map.keys())
                latest_ckpt_path = ckpt_map[latest_ckpt_num]
                policy_paths_to_run.append(latest_ckpt_path)
                print(f"   - Found latest checkpoint: {os.path.basename(latest_ckpt_path)} in {os.path.basename(run_dir)}")
        
        policy_paths_to_run.sort()
    else:
        policy_paths_to_run = args.policy_paths

    if not policy_paths_to_run:
        sys.exit("[ERROR] No matching policies found to run.")

    # --- Parse Sweep Parameters ---
    param_combinations = [{}] # Default for a single run with no sweep
    
    if args.sweep:
        print("[INFO] Parsing GRID sweep parameters...")
        sweep_params = {}
        for arg in args.sweep:
            if ":" not in arg or "," not in arg:
                sys.exit(f"[ERROR] Invalid sweep format: '{arg}'. Expected 'key:val1,val2,...'")
            key, values_str = arg.split(':', 1)
            values = values_str.split(',')
            sweep_params[key] = values
            print(f"  - Sweeping '{key}' with {len(values)} values.")
        
        keys, values = zip(*sweep_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    elif args.paired_sweep:
        print("[INFO] Parsing PAIRED sweep parameters...")
        sweep_params = {}
        for arg in args.paired_sweep:
            if ":" not in arg or "," not in arg:
                sys.exit(f"[ERROR] Invalid sweep format: '{arg}'. Expected 'key:val1,val2,...'")
            key, values_str = arg.split(':', 1)
            values = values_str.split(',')
            sweep_params[key] = values
            print(f"  - Sweeping '{key}' with {len(values)} values.")
        
        # Validate that all value lists have the same length
        it = iter(sweep_params.values())
        first_len = len(next(it))
        if not all(len(lst) == first_len for lst in it):
            print("\n❌ ERROR: For a paired sweep, all parameter value lists must have the same length.")
            for name, values in sweep_params.items():
                print(f"  - {name}: {len(values)} values")
            sys.exit(1)

        # Use zip to create paired combinations
        keys, values = zip(*sweep_params.items())
        param_combinations = [dict(zip(keys, v)) for v in zip(*values)]

    # --- Execute Playback Runs ---
    all_playback_dirs = []
    total_runs = len(policy_paths_to_run) * len(param_combinations)
    print(f"\n[INFO] A total of {len(policy_paths_to_run)} policies will be played across {len(param_combinations)} sweep configurations ({total_runs} total runs).")

    for policy_path in policy_paths_to_run:
        abs_ckpt = os.path.abspath(policy_path)
        if not os.path.exists(abs_ckpt):
            print(f"[WARNING] Skipping '{abs_ckpt}' – not found.")
            continue
        
        base_run_dir = os.path.dirname(abs_ckpt)
        base_run_name = os.path.basename(base_run_dir)
        ckpt_name = os.path.splitext(os.path.basename(abs_ckpt))[0]
        base_play_dir_name = f"playback_{base_run_name}_{ckpt_name}"

        for params in param_combinations:
            run_env, override_tags = os.environ.copy(), []
            for key, value in params.items():
                run_env[key] = str(value)
                simple_key = key.split('.')[-1]
                override_tags.append(f"{simple_key}_{value}")
            
            final_play_dir_name = base_play_dir_name + (f"_{'_'.join(override_tags)}" if override_tags else "")
            play_dir = os.path.join(base_run_dir, final_play_dir_name)
            all_playback_dirs.append(play_dir)
            
            cmd: List[str] = [sys.executable, play_script, "--policy_paths", abs_ckpt, "--play_log_dir", play_dir] + passthrough
            
            try:
                print(f"--- Running playback for: {os.path.basename(play_dir)} ---")
                subprocess.run(cmd, env=run_env, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"\n[FATAL ERROR] Subprocess for '{os.path.basename(play_dir)}' failed with code {exc.returncode}. Output is above.")
                sys.exit("Stopping script due to subprocess failure.")
            except KeyboardInterrupt:
                sys.exit("\nPlayback sweep interrupted by user.")

    # --- Final Summary ---
    if args.save_summary:
        save_summary(all_playback_dirs)

    print("\nSweep complete.")


if __name__ == "__main__":
    main()