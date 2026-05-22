"""
PC with background knowledge — local sampling variant.

Required edges are sampled from true edges touching the target pair
(same strategy as LOAD-in-MPDAG local experiments). Sweeps over
graph size (--observed) and fraction of required edges (--frac_req).

Usage:
    python run_experiments_b-pc.py --save_path results_bpc.csv
    python run_experiments_b-pc.py --observed 10 15 20 25 50 --frac_req 0.2 0.4 --n_exp 500 --save_path results_bpc.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.running import run_experiment_bpc_light


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run PC+BK baseline experiments (local BK sampling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save results CSV file"
    )

    # Graph / data parameters (match LOAD experiment defaults)
    parser.add_argument("--exp_degree", type=float, default=2.0)
    parser.add_argument("--max_degree", type=int, default=5)
    parser.add_argument("--targets", type=int, default=2, help="Number of target nodes")
    parser.add_argument("--identifiable", type=bool, default=True)
    parser.add_argument("--min_adj_size", type=int, default=0)
    parser.add_argument("--samples_num", type=int, default=1000)
    parser.add_argument("--expl_anc", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=False)
    parser.add_argument("--discrete", type=bool, default=False)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--ci_test", type=str, default="fisherz")

    # Sweep parameters
    parser.add_argument(
        "--observed",
        type=int,
        nargs="+",
        default=[10, 15, 20, 25, 50, 100],
        help="Graph sizes to sweep over",
    )
    parser.add_argument(
        "--frac_req",
        type=float,
        nargs="+",
        default=[0.2, 0.3, 0.4, 0.5],
        help="Fraction of required edges to sweep over",
    )
    parser.add_argument(
        "--frac_forb",
        type=float,
        default=0.0,
        help="Fraction of forbidden edges (ignored, kept for interface compatibility)",
    )

    # Experiment control
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--n_exp", type=int, default=1000, help="Experiments per config")

    return parser.parse_args()


def main():
    args = parse_arguments()

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    base_args = {
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
        "fraction_forbidden": 0.0,
    }

    print("=" * 80)
    print("PC + Background Knowledge Experiment Runner  [local sampling]")
    print("=" * 80)
    print(f"  Base seed      : {args.base_seed}")
    print(f"  N exp          : {args.n_exp}")
    print(f"  Observed       : {args.observed}")
    print(f"  Fraction req   : {args.frac_req}")
    print(f"  Save path      : {args.save_path}")
    print("=" * 80 + "\n")

    results_list = []

    for observed in tqdm(args.observed, desc="Observed", position=0):
        for frac_req in tqdm(args.frac_req, desc="Frac req", leave=False, position=1):
            base_args["observed"] = observed
            base_args["fraction_required"] = frac_req

            print(f"\n  Running PC+BK (local): observed={observed}, frac_req={frac_req}")
            metrics = run_experiment_bpc_light(base_args, args.base_seed, args.n_exp)

            row = {
                "algorithm": "bpc_local",
                "observed": observed,
                "exp_degree": args.exp_degree,
                "n_samples": args.samples_num,
                "fraction_required": frac_req,
                "fraction_forbidden": 0.0,
            }
            for metric_key, stats in metrics.items():
                row[f"{metric_key}_mean"] = stats["mean"]
                row[f"{metric_key}_std"] = stats["std"]

            results_list.append(row)
            pd.DataFrame(results_list).to_csv(args.save_path, index=False)
            print(f"  Saved to {args.save_path}")

    print(f"\nDone. Results at: {args.save_path}")


if __name__ == "__main__":
    main()
