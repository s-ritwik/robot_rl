import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import re

def parse_run_name(run_name: str) -> dict:
    """
    Parses the complex run_name string to extract hyperparameter key-value pairs.

    Args:
        run_name: The string from the 'run_name' column.

    Returns:
        A dictionary of extracted hyperparameters.
    """
    params = {}
    # Remove the timestamp (e.g., '2025-06-30_18-09-50_')
    param_string = re.sub(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_', '', run_name)
    
    parts = param_string.split('_')
    current_key_parts = []
    
    for part in parts:
        # A part is considered a value if it's a number, potentially with 'p' for decimal or negative sign.
        # This regex checks for optional '-', digits, optional 'p', and more digits.
        if re.fullmatch(r'-?(\d+p\d+|\d+)', part):
            value_str = part.replace('p', '.')
            value = float(value_str)
            
            if current_key_parts:
                key = '_'.join(current_key_parts)
                params[key] = value
                current_key_parts = []
        else:
            current_key_parts.append(part)
            
    return params

def main(args):
    """
    Main function to load data, parse it, and generate the plot.
    """
    # --- 1. Load Data ---
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found at '{args.csv_path}'")
        return
        
    print(f"Loading data from '{args.csv_path}'...")
    df = pd.read_csv(args.csv_path)

    # --- 2. Parse Hyperparameters ---
    print("Parsing 'run_name' to extract hyperparameters...")
    # Apply the parsing function to each row and create a new DataFrame from the results
    params_df = df['run_name'].apply(parse_run_name).apply(pd.Series)
    
    # Combine the original data with the new hyperparameter columns
    df = pd.concat([df, params_df], axis=1)
    
    # Identify the newly created parameter columns
    param_cols = list(params_df.columns)
    print(f"Successfully parsed parameters: {param_cols}")

    # --- 3. Validate Arguments ---
    all_cols = list(df.columns)
    if args.y_axis not in all_cols:
        print(f"Error: Y-axis metric '{args.y_axis}' not found in data columns.")
        print(f"Available columns: {all_cols}")
        return
    if args.x_axis not in all_cols:
        print(f"Error: X-axis hyperparameter '{args.x_axis}' not found in parsed parameters.")
        print(f"Available parameters: {param_cols}")
        return
    if args.group_by and args.group_by not in all_cols:
        print(f"Error: Grouping parameter '{args.group_by}' not found in parsed parameters.")
        print(f"Available parameters: {param_cols}")
        return

    # --- 4. Create Visualization ---
    print(f"Generating plot: '{args.y_axis}' vs '{args.x_axis}'...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    sns.boxplot(
        data=df,
        x=args.x_axis,
        y=args.y_axis,
        hue=args.group_by,
        ax=ax,
        palette='viridis' # A nice color palette
    )

    # --- 5. Customize and Save Plot ---
    title = f"Effect of '{args.x_axis}' on '{args.y_axis}'"
    if args.group_by:
        title += f", Grouped by '{args.group_by}'"
    
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel(args.x_axis, fontsize=14)
    ax.set_ylabel(args.y_axis, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    
    if args.group_by:
        plt.legend(title=args.group_by, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Define output path
    if args.save_path:
        output_filename = args.save_path
    else:
        # Create a descriptive default filename
        group_suffix = f"_grouped_by_{args.group_by}" if args.group_by else ""
        output_filename = f"{args.y_axis}_vs_{args.x_axis}{group_suffix}.png"

    print(f"Saving plot to '{output_filename}'...")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print("Done!")
    # To display the plot interactively, uncomment the line below
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize hyperparameter sweep results from a CSV file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("csv_path", type=str, help="Path to the input CSV file.")
    parser.add_argument("-y", "--y_axis", type=str, required=True, help="The metric to plot on the Y-axis (e.g., 'fall_rate_percent').")
    parser.add_argument("-x", "--x_axis", type=str, required=True, help="The hyperparameter to plot on the X-axis (e.g., 'events_push_robot_params_velocity_range_x').")
    parser.add_argument("-g", "--group_by", type=str, default=None, help="Optional: A second hyperparameter to group the data by (using color).")
    parser.add_argument("-s", "--save_path", type=str, default=None, help="Optional: Full path to save the output plot. If not provided, a default name is used.")

    cli_args = parser.parse_args()
    main(cli_args)
