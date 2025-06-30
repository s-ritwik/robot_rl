#!/usr/bin/env python3
"""
Sequentially play trained policies, optionally sweeping multiple parameters and generating a single summary CSV.
"""

import argparse
import os
import subprocess
import sys
import csv
import pickle
from typing import List

import torch

## --- THIS FUNCTION IS MOVED HERE FROM play_policy.py --- ##
def calculate_and_save_summary_metrics(play_dirs: list[str]) -> None:
    """
    Calculates overall tracking error (RMSE) and fall rate for each run
    and saves the aggregated results to a single CSV file.
    """
    DT = 0.02
    summary_data = []
    print("\n--- Calculating Summary Metrics for all runs ---")
    for play_dir in play_dirs:
        original_run_name = os.path.basename(os.path.dirname(play_dir))
        playback_folder_name = os.path.basename(play_dir)
        if "playback_" in playback_folder_name:
            override_suffix = playback_folder_name.replace("playback_", "", 1)
            run_name = f"{original_run_name}_{override_suffix}"
        else:
            run_name = original_run_name
        try:
            with open(os.path.join(play_dir, "base_velocity.pkl"), "rb") as f: cmd_vel_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "root_pos.pkl"), "rb") as f: root_pos_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "root_velocity.pkl"), "rb") as f: root_vel_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "terminations.pkl"), "rb") as f: terminations_raw = torch.tensor(pickle.load(f), dtype=torch.bool)
            did_fall = torch.any(terminations_raw, dim=0)
            fall_rate = torch.mean(did_fall.float()).item() * 100.0
            linear_cmd_vel = torch.stack([cmd_vel_raw[..., 0], cmd_vel_raw[..., 1], torch.zeros_like(cmd_vel_raw[..., 0])], dim=-1)
            cmd_pos_integrated = torch.cumsum(linear_cmd_vel * DT, dim=0)
            root_pos_relative = root_pos_raw - root_pos_raw[0, :, :]
            vel_error_x_sq = (cmd_vel_raw[..., 0] - root_vel_raw[..., 0]) ** 2
            vel_error_y_sq = (cmd_vel_raw[..., 1] - root_vel_raw[..., 1]) ** 2
            vel_rmse_x = torch.sqrt(torch.mean(vel_error_x_sq)).item()
            vel_rmse_y = torch.sqrt(torch.mean(vel_error_y_sq)).item()
            pos_error_x_sq = (cmd_pos_integrated[..., 0] - root_pos_relative[..., 0]) ** 2
            pos_error_y_sq = (cmd_pos_integrated[..., 1] - root_pos_relative[..., 1]) ** 2
            pos_rmse_x = torch.sqrt(torch.mean(pos_error_x_sq)).item()
            pos_rmse_y = torch.sqrt(torch.mean(pos_error_y_sq)).item()
            summary_data.append({
                "run_name": run_name, "fall_rate_percent": fall_rate,
                "vel_rmse_x": vel_rmse_x, "vel_rmse_y": vel_rmse_y,
                "pos_rmse_x": pos_rmse_x, "pos_rmse_y": pos_rmse_y,
            })
            print(f"  - Processed: {run_name}")
        except FileNotFoundError as e:
            print(f"[WARNING] Could not load data for '{run_name}': {e}. Skipping metric calculation for this run.")
            continue
    if not summary_data:
        print("[ERROR] No data found to calculate summary.")
        return
    output_dir = os.getcwd()
    base_filename = "comparison_summary"
    counter = 0
    output_path = os.path.join(output_dir, f"{base_filename}.csv")
    while os.path.exists(output_path):
        counter += 1
        output_path = os.path.join(output_dir, f"{base_filename}_{counter}.csv")
    try:
        with open(output_path, "w", newline="") as csvfile:
            fieldnames = summary_data[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\n[SUCCESS] Summary metrics for all runs saved to: {output_path}")
    except (IOError, IndexError) as e:
        print(f"[ERROR] Could not write to CSV file: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play policies, optionally sweeping multiple hyperparameters.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--policy_paths", nargs="+", required=True, metavar="PATH", help="One or more checkpoint files to play.")
    parser.add_argument("--sweep", nargs='+', help="Define a sweep over one or more parameters. Format: 'param.path:val1,val2,...'.")
    parser.add_argument("--save_summary", action="store_true", default=False, help="Calculate summary metrics for the entire sweep and save to one CSV.")

    args, passthrough = parser.parse_known_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    play_script = os.path.join(script_dir, "play_policy.py")
    if not os.path.exists(play_script):
        sys.exit(f"[ERROR] play_policy.py not found at {play_script}")

    sweep_params, num_values = {}, 0
    if args.sweep:
        for arg in args.sweep:
            try:
                param, values_str = arg.split(':', 1)
                values = values_str.split(',')
                if num_values == 0: num_values = len(values)
                elif len(values) != num_values: sys.exit(f"[ERROR] All param lists in --sweep must have same length. Expected {num_values}, got {len(values)} for '{param}'.")
                sweep_params[param] = values
            except ValueError:
                sys.exit(f"[ERROR] Invalid format for --sweep argument: '{arg}'.")
    
    num_runs = num_values if args.sweep else 1
    all_playback_dirs = [] # To collect output directories for the final summary

    print("\n━━━━━━━━━━ Policy Playback Sweep ━━━━━━━━━━\n")

    for policy_path in args.policy_paths:
        abs_ckpt = os.path.abspath(policy_path)
        if not os.path.exists(abs_ckpt):
            print(f"[WARNING] Skipping '{abs_ckpt}' – not found.")
            continue
        run_dir = os.path.dirname(abs_ckpt)

        for i in range(num_runs):
            run_env, override_tags = os.environ.copy(), []
            run_description = f"Playing: {os.path.basename(abs_ckpt)}"
            override_descriptions = []

            if args.sweep:
                for j, (param_name, values) in enumerate(sweep_params.items()):
                    value = values[i]
                    run_env[f"PARAM_OVERRIDE_{j}"] = f"{param_name}={value}"
                    override_descriptions.append(f"{param_name.split('.')[-1]}={value}")
                    # Recreate the tag logic to predict the directory name
                    tag_path = "_".join(param_name.split("."))
                    override_tags.append(f"{tag_path}_{str(value).replace('.', 'p')}")
                run_description += f" with {', '.join(override_descriptions)}"

            # Predict the directory that play_policy.py will create
            play_dir_name = "playback" + (f"_{'_'.join(override_tags)}" if override_tags else "")
            play_dir = os.path.join(run_dir, play_dir_name)
            all_playback_dirs.append(play_dir)

            print(f"🚀 {run_description}")
            print("───────────────────────────────────────────")
            cmd: List[str] = ([sys.executable, play_script, "--policy_paths", abs_ckpt, "--play_log_dir", play_dir] + passthrough)
            try:
                subprocess.run(cmd, env=run_env, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[ERROR] play_policy.py exited with code {exc.returncode}")
            except KeyboardInterrupt:
                print("\nPlayback sweep interrupted by user.")
                sys.exit(0)
            print("───────────────────────────────────────────\n")

    # After all runs are complete, calculate the summary if requested
    if args.save_summary:
        calculate_and_save_summary_metrics(all_playback_dirs)

    print("Sweep complete.")

if __name__ == "__main__":
    main()