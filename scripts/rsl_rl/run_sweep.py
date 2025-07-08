import argparse
import os
import subprocess
import sys
from typing import List
from itertools import product

def main():
    """
    Main function to parse arguments and run the hyperparameter sweep.
    """
    parser = argparse.ArgumentParser(
        description="Run multiple training sessions with varying hyperparameters for IsaacLab.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
----------------
# Run a sweep over the entropy coefficient for the 'custom' environment
python run_sweep.py \\
    --param_name "algorithm.entropy_coef" \\
    --values 0.006 0.008 0.01 \\
    -- \\
    --env_type custom --num_envs 4096

# Run a grid sweep over learning rate and entropy coefficient
python run_sweep.py \\
    --grid_sweep "algorithm.learning_rate:1e-3,5e-4" "algorithm.entropy_coef:0.005,0.01" \\
    -- \\
    --env_type custom --max_iterations 10000

NOTE: Use '--' to separate the sweep script's arguments from the arguments
      that need to be passed to the underlying 'train_policy.py' script.
"""
    )
    parser.add_argument(
        "--param_name",
        type=str,
        help="The full name of the hyperparameter to vary in the agent config (e.g., 'algorithm.entropy_coef')."
    )
    parser.add_argument(
        "--values",
        nargs='+',
        help="A list of values to assign to the hyperparameter for each run."
    )
    parser.add_argument(
        "--grid_sweep",
        nargs='+',
        help="Define a grid sweep. Format: 'param_name:val1,val2,...'"
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to a checkpoint (.pt) file to resume training from."
    )

    args, pass_through_args = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_policy.py")

    if not os.path.isfile(train_script):
        print(f"Error: cannot find train_policy.py at {train_script}")
        sys.exit(1)

    if args.grid_sweep:
        if "--grid_sweep" not in pass_through_args:
            pass_through_args.append("--grid_sweep")
            
        param_grid = {}
        for arg in args.grid_sweep:
            param, values_str = arg.split(':', 1)
            param_grid[param] = values_str.split(',')

        param_names = list(param_grid.keys())
        value_combinations = list(product(*param_grid.values()))

        print("Starting grid sweep with the following parameters:")
        for name, values in param_grid.items():
            print(f"  {name}: {values}")
        print(f"Total number of runs: {len(value_combinations)}")
        print("-" * 80)

        for combo in value_combinations:
            run_env = os.environ.copy()
            combo_str_parts = []
            param_overrides = []
            for i, value in enumerate(combo):
                param_name = param_names[i]
                param_overrides.append(f"{param_name}={value}")
                combo_str_parts.append(f"{param_name}={value}")

            # If resuming, add the resume parameters to the overrides
            if args.resume_path:
                print(f"   -> Resuming from checkpoint: {args.resume_path}")
                param_overrides.append("agent.resume=True")
                # Use an absolute path to be safe
                abs_resume_path = os.path.abspath(args.resume_path)
                param_overrides.append(f"agent.resume_path={abs_resume_path}")

            # Set the environment variables from the collected overrides
            for i, override_str in enumerate(param_overrides):
                run_env[f"PARAM_OVERRIDE_{i}"] = override_str
            
            print(f"🚀 Starting run with {', '.join(combo_str_parts)}")
            print("-" * 80)

            command: List[str] = [sys.executable, train_script] + pass_through_args

            try:
                subprocess.run(command, env=run_env, check=True)
                print(f"✅ Finished run with {', '.join(combo_str_parts)}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error during run with {', '.join(combo_str_parts)}. Exit code: {e.returncode}")
                print("Stopping sweep due to error.")
                break
            except KeyboardInterrupt:
                print("\nSweep interrupted by user. Exiting.")
                sys.exit(0)
            print("-" * 80)

    elif args.param_name and args.values:
        print(f"Starting hyperparameter sweep for: '{args.param_name}'")
        print(f"Values to test: {args.values}")
        print(f"Arguments passed to train_policy.py: {pass_through_args}")
        print("-" * 80)

        for value in args.values:
            print(f"🚀 Starting run with {args.param_name} = {value}")
            print("-" * 80)

            run_env = os.environ.copy()
            run_env["PARAM_OVERRIDE"] = f"{args.param_name}={value}"

            command: List[str] = [sys.executable, train_script] + pass_through_args

            try:
                subprocess.run(command, env=run_env, check=True)
                print(f"✅ Finished run with {args.param_name} = {value}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error during run with {args.param_name} = {value}. Exit code: {e.returncode}")
                print("Stopping sweep due to error.")
                break
            except KeyboardInterrupt:
                print("\nSweep interrupted by user. Exiting.")
                sys.exit(0)
            print("-" * 80)
    else:
        parser.print_help()
        sys.exit(1)

    print("\nSweep complete.")

if __name__ == "__main__":
    main()