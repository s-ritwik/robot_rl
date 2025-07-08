#!/usr/bin/env python3
"""
Creates a final, clean, and readable interactive parallel coordinates plot
using HiPlot, with four correctly scaled and ordered hyperparameter axes.
"""

import argparse
import os
import re
import pandas as pd
import hiplot as hip

def create_hiplot_visualization(input_filepath: str):
    """
    Reads a top policies CSV, prepares a clean DataFrame, and generates an
    interactive HiPlot visualization.

    Args:
        input_filepath (str): Path to the _top10.csv file.
    """
    try:
        print(f"📄 Reading data from: {input_filepath}")
        df = pd.read_csv(input_filepath)
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at '{input_filepath}'")
        return
    except Exception as e:
        print(f"❌ ERROR: Failed to read CSV file. Reason: {e}")
        return

    # This DataFrame will ONLY contain the four columns we want to plot as axes.
    plot_df = pd.DataFrame()
    print("⚙️  Parsing hyperparameter values...")

    params_to_extract = {
        'clf_decreasing_condition_weight': r"clf_decreasing_condition_weight_(-?\d+p?\d*)",
        'clf_reward_weight': r"clf_reward_weight_(-?\d+p?\d*)",
        'holonomic_constraint_vel_weight': r"holonomic_constraint_vel_weight_(-?\d+p?\d*)",
        'holonomic_constraint_weight': r"holonomic_constraint_(?!vel_weight_)(-?\d+p?\d*)",
    }

    for param_name, pattern in params_to_extract.items():
        extracted_series = df['Policy'].str.extract(pattern, expand=False)
        numeric_series = pd.to_numeric(
            extracted_series.str.replace('p', '.'),
            errors='coerce'
        )
        clean_name = param_name.replace('_', ' ').title()

        unique_vals = sorted(numeric_series.dropna().unique())
        
        if len(unique_vals) >= 3:
            # Create special string labels that force the correct alphabetical sort order
            categories = ['Low', 'Mid', 'High'] + [f"{i+1}_Value" for i in range(3, len(unique_vals))]
            val_map = {val: f"{i+1}_{categories[i]} ({val})" for i, val in enumerate(unique_vals)}
            plot_df[clean_name] = numeric_series.map(val_map)
        else:
            plot_df[clean_name] = numeric_series

    plot_df.dropna(inplace=True)

    if plot_df.empty:
        print("❌ ERROR: Data frame is empty after parsing. Check that hyperparameter names in the script match your CSV.")
        return

    print(f"✅ Successfully prepared {len(plot_df)} policies for plotting.")
    print("🎨 Generating HiPlot visualization...")

    # Create the experiment from the clean 4-column DataFrame.
    exp = hip.Experiment.from_dataframe(plot_df)

    # The line causing the error has been removed.

    # Create an HTML filename
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    output_path = os.path.join(os.path.dirname(input_filepath), f"{base_filename}_hiplot.html")

    exp.to_html(output_path)
    
    print(f"✔️ HiPlot visualization saved to: {output_path}")
    print("➡️  Open the HTML file and right-click the 'uid' axis title to hide it.")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create an interactive HiPlot visualization from a top policies summary file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the top10.csv summary file."
    )
    args = parser.parse_args()
    
    create_hiplot_visualization(args.input_file)


if __name__ == "__main__":
    main()