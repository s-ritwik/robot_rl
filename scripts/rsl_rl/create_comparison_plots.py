#!/usr/bin/env python3
# =============================================================================
# create_comparison_plots.py
# ---------------------------------------------------------------------------
# Generates comparison plots from multiple policy playback data logs.
#
# This script reads the .pkl files (base_velocity, root_pos, root_velocity)
# from several playback directories, calculates statistics for each, and
# plots them on the same axes for easy comparison.
#
# Usage
# -----
# python create_comparison_plots.py \
#     path/to/run_A/playback \
#     path/to/run_B/playback \
#     path/to/run_C/playback
#
# =============================================================================

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

def generate_comparison_plots(playback_dirs: list[str], output_dir: str) -> None:
    """
    Calculates, saves, and plots comparative statistics for multiple policy playbacks.

    Args:
        playback_dirs: A list of paths to the 'playback' directories to compare.
        output_dir: The directory where the final plot images will be saved.
    """

    # --- IMPORTANT ---
    # This DT must match the simulation dt in the environment configuration
    # for the position integration to be accurate. Common values are 0.02 (50Hz)
    # or 0.01 (100Hz).
    DT = 0.02

    # --- Data Loading & Statistics Calculation ---
    all_stats_data = {}
    valid_run_labels = []

    print("[INFO] Loading and processing data for each run...")
    for play_dir in playback_dirs:
        # Assumes parent directory is the unique run name (e.g., "run_A")
        run_label = os.path.basename(os.path.dirname(os.path.abspath(play_dir)))
        try:
            # Load raw data from pickle files
            with open(os.path.join(play_dir, "base_velocity.pkl"), "rb") as fh:
                vel_raw = torch.tensor(pickle.load(fh))
            with open(os.path.join(play_dir, "root_pos.pkl"), "rb") as fh:
                pos_raw = torch.tensor(pickle.load(fh))
            with open(os.path.join(play_dir, "root_velocity.pkl"), "rb") as fh:
                root_vel_raw = torch.tensor(pickle.load(fh))

            # Calculate mean and std deviation across the environment dimension (dim=1)
            stats = {
                "cmd_vel_mean": vel_raw.mean(dim=1),
                "cmd_vel_std": vel_raw.std(dim=1),
                "root_pos_mean": pos_raw.mean(dim=1),
                "root_pos_std": pos_raw.std(dim=1),
                "root_vel_mean": root_vel_raw.mean(dim=1),
                "root_vel_std": root_vel_raw.std(dim=1),
            }

            # Integrate commanded velocity to get commanded position
            linear_cmd_vel = torch.stack([stats["cmd_vel_mean"][:, 0], stats["cmd_vel_mean"][:, 1], torch.zeros_like(stats["cmd_vel_mean"][:, 0])], dim=1)
            stats["cmd_pos_mean"] = torch.cumsum(linear_cmd_vel * DT, dim=0)
            stats["cmd_pos_std"] = torch.zeros_like(stats["root_pos_std"])  # Placeholder

            all_stats_data[run_label] = stats
            valid_run_labels.append(run_label)
            print(f"  ✔ Successfully processed '{run_label}'")

        except FileNotFoundError as e:
            print(f"  ✖ [WARNING] Could not load data for '{run_label}': {e}. Skipping this run.")
            continue
        except Exception as e:
            print(f"  ✖ [ERROR] Failed to process data for '{run_label}': {e}. Skipping this run.")
            continue

    if not all_stats_data:
        print("\n[ERROR] No valid data found to plot. Exiting.")
        return

    # --- Create short, readable labels for filenames and legends ---
    short_labels = {long_label: f"run{i+1:02d}" for i, long_label in enumerate(valid_run_labels)}
    print("\n[INFO] Mapping long run names to short labels for filenames and plots:")
    for long, short in short_labels.items():
        print(f"  - {short}: {long}")

    # --- Plotting ---
    print("\n[INFO] Generating comparison plots...")
    os.makedirs(output_dir, exist_ok=True)
    # Use the new short labels for the filename suffix
    plot_suffix = "_vs_".join(sorted(short_labels.values()))


    # Plot definitions: (filename_key, title, list_of_series_to_plot)
    # Each series is a tuple of (mean_key, std_key, axis_index, legend_label_suffix)
    plot_definitions = [
        ("x_vel", "X Velocity Comparison", [("cmd_vel", 0, "Cmd"), ("root_vel", 0, "Actual")]),
        ("y_vel", "Y Velocity Comparison", [("cmd_vel", 1, "Cmd"), ("root_vel", 1, "Actual")]),
        ("z_vel", "Z Velocity Comparison", [("root_vel", 2, "Actual")]),
        ("x_pos", "X Position Comparison", [("cmd_pos", 0, "Cmd"), ("root_pos", 0, "Actual")]),
        ("y_pos", "Y Position Comparison", [("cmd_pos", 1, "Cmd"), ("root_pos", 1, "Actual")]),
        ("z_pos", "Z Position Comparison", [("root_pos", 2, "Actual")]),
    ]

    for f_key, title, series_list in plot_definitions:
        plt.figure(figsize=(12, 7))
        plt.title(title)

        for run_label, stats in all_stats_data.items():
            for data_key, index, legend_suffix in series_list:
                mean_key = f"{data_key}_mean"
                std_key = f"{data_key}_std"

                mean_data = np.array(stats[mean_key])[:, index]
                std_data = np.array(stats[std_key])[:, index]
                
                # Get the short label for the legend
                short_run_label = short_labels[run_label]
                line_label = f"{short_run_label} {legend_suffix}".strip()

                # Plot the mean line
                line, = plt.plot(mean_data, label=line_label)
                # Plot the shaded standard deviation region
                plt.fill_between(range(len(mean_data)), mean_data - std_data, mean_data + std_data, color=line.get_color(), alpha=0.2)

        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        outfile = os.path.join(output_dir, f"{f_key}_{plot_suffix}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✔ Plot saved → {outfile}")

    print("\n[INFO] All plots generated successfully.")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Create comparison plots from multiple policy playback logs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "playback_dirs",
        nargs="+",
        metavar="DIR",
        help="Path(s) to the 'playback' directories containing the .pkl log files."
    )
    parser.add_argument(
        "--output_dir",
        default="comparison_plots",
        help="Directory to save the generated plot images (default: 'comparison_plots')."
    )
    args = parser.parse_args()

    # Basic validation
    valid_dirs = []
    for directory in args.playback_dirs:
        if os.path.isdir(directory):
            valid_dirs.append(directory)
        else:
            print(f"[WARNING] Provided path is not a directory, skipping: {directory}")

    if len(valid_dirs) < 1:
        print("[ERROR] No valid playback directories provided. Nothing to do.")
        return

    generate_comparison_plots(valid_dirs, args.output_dir)

if __name__ == "__main__":
    main()