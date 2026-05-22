"""
Execution script for running LOAD in MPDAG experiments using all nodes strategy.

Usage:
    python run_experiments.py --targets 2 --alpha 0.01 --ci_test fisherz --save_path results.csv
    
    Or with custom parameters:
    python run_experiments.py --targets 3 --samples_num 2000 --observed 10 15 20 --save_path /path/to/results.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.running import run_algorithm_mpdag

# Import or define your run_experiment_mpdag function
# If it doesn't exist, we'll create a wrapper below
try:
    from src.running import run_experiment_mpdag_v2_light
except ImportError:
    def run_experiment_mpdag(args, base_seed, n_exp):
        """
        Placeholder wrapper. Replace with actual implementation or import.
        """
        raise NotImplementedError("Please ensure run_experiment_mpdag is available in src.running")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LOAD in MPDAG experiments with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save results CSV file"
    )
    
    # Base experimental parameters
    parser.add_argument(
        "--exp_degree",
        type=float,
        default=2.0,
        help="Experimental degree"
    )
    parser.add_argument(
        "--max_degree",
        type=int,
        default=5,
        help="Maximum degree"
    )
    parser.add_argument(
        "--targets",
        type=int,
        default=2,
        help="Number of target nodes"
    )
    parser.add_argument(
        "--identifiable",
        type=bool,
        default=True,
        help="Whether targets are identifiable"
    )
    parser.add_argument(
        "--min_adj_size",
        type=int,
        default=0,
        help="Minimum adjacency size"
    )
    parser.add_argument(
        "--samples_num",
        type=int,
        default=1000,
        help="Number of samples"
    )
    parser.add_argument(
        "--expl_anc",
        type=bool,
        default=False,
        help="Explicitly include ancestors"
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Save experimental data to a file identified by the seed and the kwargs (deprecated - use save_path instead)"
    )
    parser.add_argument(
        "--discrete",
        type=bool,
        default=False,
        help="Whether data is discrete"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Significance level for CI tests"
    )
    parser.add_argument(
        "--ci_test",
        type=str,
        default="fisherz",
        help="Conditional independence test to use"
    )
    parser.add_argument(
        "--logging",
        type=bool,
        default=False,
        help="Enable logging"
    )
    
    # Loop parameters
    parser.add_argument(
        "--observed",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 50, 100],
        help="List of observed variables to iterate over"
    )
    parser.add_argument(
        "--frac_req",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5],
        help="List of fraction required values to iterate over"
    )
    parser.add_argument(
        "--frac_forb",
        type=float,
        default=0.0,
        help="Fraction forbidden (constant across experiments)"
    )
    
    # Experiment control
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--n_exp",
        type=int,
        default=1000,
        help="Number of experiments per configuration"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Ensure save directory exists
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build base_args_skeleton from parsed arguments
    base_args_skeleton = {
        "exp_degree": args.exp_degree,
        "max_degree": args.max_degree,
        "targets": args.targets,
        "identifiable": args.identifiable,
        "min_adj_size": args.min_adj_size,
        "samples_num": args.samples_num,
        "expl_anc": args.expl_anc,
        "discrete": args.discrete,
        "save": args.save,
        "alpha": args.alpha,
        "ci_test": args.ci_test,
        "logging": args.logging,
    }
    
    print("=" * 80)
    print("LOAD in MPDAG Experiment Runner")
    print("=" * 80)
    print(f"\nExperiment Configuration:")
    print(f"  Base seed: {args.base_seed}")
    print(f"  Number of experiments per config: {args.n_exp}")
    print(f"  Fraction forbidden: {args.frac_forb}")
    print(f"\nParameter grid:")
    print(f"  Observed: {args.observed}")
    print(f"  Fraction Required: {args.frac_req}")
    print(f"\nBase arguments:")
    for key, value in sorted(base_args_skeleton.items()):
        print(f"  {key}: {value}")
    print(f"\nSaving results to: {args.save_path}")
    print("=" * 80 + "\n")
    
    results_list = []
    
    # Main experiment loops
    for observed in tqdm(args.observed, desc="Observed Loop", position=0):
        for frac_req in tqdm(args.frac_req, desc="Fraction Required", leave=False, position=1):
            
            # Update args
            base_args_skeleton["observed"] = observed
            base_args_skeleton["fraction_forbidden"] = args.frac_forb
            base_args_skeleton["fraction_required"] = frac_req
            
            # Run experiment
            print(f"\n  Running: observed={observed}, frac_req={frac_req}")
            f1_dict = run_experiment_mpdag_v2_light(base_args_skeleton, args.base_seed, args.n_exp)
            
            # Flatten the dictionary into a row
            row = {
                "observed": observed,
                "exp_degree": args.exp_degree,
                "n_samples": args.samples_num,
                "fraction_forbidden": args.frac_forb,
                "fraction_required": frac_req,
            }
            
            for metric_key, stats in f1_dict.items():
                # Convert np.float64 to standard float for CSV compatibility
                row[f"{metric_key}_mean"] = float(stats['mean'])
                row[f"{metric_key}_std"] = float(stats['std'])
            
            # Add to our local list
            results_list.append(row)
            
            # Save at every iteration
            df_temp = pd.DataFrame(results_list)
            df_temp.to_csv(args.save_path, index=False)
            print(f"  ✓ Results saved to {args.save_path}")
    
    print("\n" + "=" * 80)
    print(f"Experiment complete. Final results saved to: {args.save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
