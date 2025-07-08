#!/usr/bin/env python3
"""
Analyzes an existing CSV of simulation run data to produce aggregated summaries,
combination counts, and trend analysis, saving all results into a
dedicated output folder.
"""
import argparse
import os
import re
import pandas as pd
from collections import defaultdict

def perform_trend_analysis(top_10_df: pd.DataFrame, params_to_track: dict, output_filepath: str):
    """
    Analyzes and saves parameter trends from the top 10% of runs.

    Args:
        top_10_df: DataFrame containing the top 10% of policies.
        params_to_track: Dictionary mapping parameter names to their regex patterns.
        output_filepath: Path to save the trend analysis report.
    """
    print("📈 Starting trend analysis...")

    param_df = pd.DataFrame()
    for param_name, pattern in params_to_track.items():
        extracted_values = top_10_df['Policy'].str.extract(pattern, expand=False)
        # Clean values by replacing 'p' with '.' for numeric conversion
        param_df[param_name] = pd.to_numeric(
            extracted_values.str.replace('p', '.'),
            errors='coerce'
        )

    param_df.dropna(how='all', inplace=True)

    if param_df.shape[1] < 2:
        print("⚠️  Not enough parameter data to perform trend analysis.")
        return

    with open(output_filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Top 10% Parameter Trend Analysis\n")
        f.write("="*60 + "\n\n")

        # --- Analysis 1: Correlation Matrix ---
        f.write("## 1. Parameter Correlation Matrix\n\n")
        f.write(
            "This matrix shows the linear relationship between parameter values.\n"
            "  - A value near +1.0 means when one parameter is high, the other is also high.\n"
            "  - A value near -1.0 means when one is high, the other is low.\n"
            "  - A value near 0.0 means there is no linear correlation.\n\n"
        )
        correlation_matrix = param_df.corr()
        f.write(correlation_matrix.to_string(float_format='%.3f'))
        f.write("\n\n" + "-"*60 + "\n\n")

        # --- Analysis 2: Conditional Value Analysis ---
        f.write("## 2. Conditional Value Analysis\n\n")
        f.write(
            "This section shows how the average of other parameters behaves when a\n"
            "primary parameter's value is in its LOW, MEDIUM, or HIGH range.\n"
            "(Binning is done by data frequency (quantile) if possible, otherwise by value).\n\n"
        )
        
        parameter_columns = list(param_df.columns)

        for primary_param in parameter_columns:
            if param_df[primary_param].nunique() < 3:
                continue

            binned_series = None
            try:
                # Try quantile-based binning first
                binned_series = pd.qcut(
                    param_df[primary_param],
                    q=3,
                    labels=['LOW (quantile)', 'MEDIUM (quantile)', 'HIGH (quantile)'],
                    duplicates='raise'
                )
            except ValueError:
                # Fall back to value-based binning if quantile fails
                try:
                    binned_series = pd.cut(
                        param_df[primary_param],
                        bins=3,
                        labels=['LOW (value)', 'MEDIUM (value)', 'HIGH (value)'],
                        include_lowest=True
                    )
                except ValueError:
                    continue # Skip if both binning methods fail

            if binned_series is None:
                continue

            binned_series.name = f"{primary_param}_bin"
            f.write(f"### Trends based on '{primary_param}'\n")

            other_params = [p for p in parameter_columns if p != primary_param]
            if not other_params:
                continue

            conditional_means = param_df.groupby(binned_series)[other_params].mean()
            f.write(conditional_means.to_string(float_format='%.3f'))
            f.write("\n\n")

    print(f"✔️ Trend analysis saved to: {output_filepath}")


def analyze_existing_csv(input_filepath: str):
    """
    Loads a CSV, processes it, and saves new summary files into a new folder.
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

    # Rename 'Policy' column to 'run_name' to force aggregation logic
    if 'Policy' in df.columns and 'run_name' not in df.columns:
        df.rename(columns={'Policy': 'run_name'}, inplace=True)

    def get_base_policy_signature(run_name):
        """
        Extracts the base policy signature by finding the part of the string
        up to and including "_model_xxxx".
        """
        match = re.search(r'(.*_model_\d+)', str(run_name))
        if match:
            return match.group(1)
        # Fallback if the pattern doesn't match for some reason
        return str(run_name)

    # Define required columns for calculations
    base_required_cols = {'pos_rmse_x', 'pos_rmse_y', 'pos_rmse_yaw', 'vel_rmse_x', 'vel_rmse_y', 'vel_rmse_yaw'}
    new_required_cols = {'Cost of Transport', 'Avg Joint Acceleration', 'Avg Torque Rate'}
    
    fall_rate_col = 'Fall Rate' if 'Fall Rate' in df.columns else 'fall_rate_percent'
    required_cols = base_required_cols | new_required_cols
    required_cols.add(fall_rate_col)


    if not required_cols.issubset(df.columns):
        print(f"❌ ERROR: CSV is missing required columns: {required_cols - set(df.columns)}")
        return

    # Always calculate fresh averages before grouping
    df['Average Position RMSE'] = df[['pos_rmse_x', 'pos_rmse_y', 'pos_rmse_yaw']].mean(axis=1)
    df['Average Velocity RMSE'] = df[['vel_rmse_x', 'vel_rmse_y', 'vel_rmse_yaw']].mean(axis=1)

    # Main logic for processing and aggregation
    if 'run_name' in df.columns:
        # Apply the corrected function here
        df['Policy'] = df['run_name'].apply(get_base_policy_signature)
        print(f"✅ Successfully processed {len(df)} individual runs.")
        
        columns_to_average = [
            'Average Position RMSE', 'Average Velocity RMSE', fall_rate_col,
            'Cost of Transport', 'Avg Joint Acceleration', 'Avg Torque Rate'
        ]
        summary_df = df.groupby('Policy')[columns_to_average].mean().reset_index()
        print(f"✅ Aggregated into {len(summary_df)} unique policies.")
    else:
        print("❌ ERROR: The CSV must contain either a 'run_name' or a 'Policy' column.")
        return

    # Create output directory
    input_dir = os.path.dirname(os.path.abspath(input_filepath))
    summary_dir = os.path.join(input_dir, "summary")
    base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
    output_folder_path = os.path.join(summary_dir, f"{base_filename}_analysis")
    os.makedirs(output_folder_path, exist_ok=True)
    print(f"📂 Created output directory: {output_folder_path}")

    # Standardize fall rate column name
    if fall_rate_col == 'fall_rate_percent' and 'Fall Rate' not in summary_df.columns:
        summary_df.rename(columns={'fall_rate_percent': 'Fall Rate'}, inplace=True)
    
    if 'Fall Rate' not in summary_df.columns:
        print("❌ ERROR: 'Fall Rate' column is missing after processing.")
        return

    # Save aggregated summary
    sort_columns = [
        "Cost of Transport", "Avg Joint Acceleration", "Avg Torque Rate",
        "Average Position RMSE", "Average Velocity RMSE", "Fall Rate"
    ]
    summary_df = summary_df.sort_values(by=sort_columns)
    summary_output_path = os.path.join(output_folder_path, f"{base_filename}_aggregated.csv")
    summary_df.to_csv(summary_output_path, index=False, float_format='%.4f')
    print(f"✔️ Aggregated summary saved to: {summary_output_path}")

    # --- Top 10% Analysis ---
    # Note: For these metrics, lower is better.
    pos_rmse_threshold = summary_df["Average Position RMSE"].quantile(0.10)
    vel_rmse_threshold = summary_df["Average Velocity RMSE"].quantile(0.10)
    fall_rate_threshold = summary_df["Fall Rate"].quantile(0.10)
    cot_threshold = summary_df["Cost of Transport"].quantile(0.10)
    joint_accel_threshold = summary_df["Avg Joint Acceleration"].quantile(0.10)
    torque_rate_threshold = summary_df["Avg Torque Rate"].quantile(0.10)

    top_10_df = summary_df[
        (summary_df["Average Position RMSE"] <= pos_rmse_threshold) |
        (summary_df["Average Velocity RMSE"] <= vel_rmse_threshold) |
        (summary_df["Fall Rate"] <= fall_rate_threshold) |
        (summary_df["Cost of Transport"] <= cot_threshold) |
        (summary_df["Avg Joint Acceleration"] <= joint_accel_threshold) |
        (summary_df["Avg Torque Rate"] <= torque_rate_threshold)
    ]

    if not top_10_df.empty:
        top_10_df_concise = top_10_df[[
            "Policy", "Cost of Transport", "Avg Joint Acceleration", "Avg Torque Rate",
            "Average Position RMSE", "Average Velocity RMSE", "Fall Rate"
        ]]
        top_10_output_path = os.path.join(output_folder_path, f"{base_filename}_top10.csv")
        top_10_df_concise.to_csv(top_10_output_path, index=False, float_format='%.4f')
        print(f"✔️ Top 10% policies saved to: {top_10_output_path}")

        # Define parameter patterns for trend analysis
        params_to_track = {
            'clf_decreasing_condition_weight': re.compile(r"clf_decreasing_condition_weight_(-?\d+p?\d*)"),
            'clf_reward_weight': re.compile(r"clf_reward_weight_(-?\d+p?\d*)"),
            'holonomic_constraint_vel_weight': re.compile(r"holonomic_constraint_vel_weight_(-?\d+p?\d*)"),
            'holonomic_constraint': re.compile(r"holonomic_constraint_weight_(-?\d+p?\d*)"),
        }

        analysis_output_path = os.path.join(output_folder_path, f"{base_filename}_top10_analysis.txt")
        print(f"✔️ Combination analysis can be found in: {analysis_output_path}")

        trend_output_path = os.path.join(output_folder_path, f"{base_filename}_trend_analysis.txt")
        perform_trend_analysis(top_10_df.copy(), params_to_track, trend_output_path)
    else:
        print("ℹ️ No policies met the top 10% criteria.")


def main():
    parser = argparse.ArgumentParser(
        description="Process a run summary CSV to create aggregated, combination, and trend reports.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()
    analyze_existing_csv(args.input_file)


if __name__ == "__main__":
    main()