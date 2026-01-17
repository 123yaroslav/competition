from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/00_baseline_score.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from scripts.lib_stepwise import score_stepwise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="datasets/valid.parquet")
    ap.add_argument(
        "--which", type=str, default="example_onnx", choices=["example_onnx"]
    )
    ap.add_argument(
        "--max-seq",
        type=int,
        default=0,
        help="If >0, score only first N sequences (debug).",
    )
    args = ap.parse_args()

    data_path = Path(args.data)

    allowed = None
    if args.max_seq and args.max_seq > 0:
        import pandas as pd

        # Cheaply read only seq_ix column to choose a subset
        seqs = pd.read_parquet(data_path, columns=["seq_ix"])[:1_000_000][
            "seq_ix"
        ].unique()
        allowed = set(int(s) for s in seqs[: args.max_seq])

    if args.which == "example_onnx":
        # Uses example_solution/solution.py PredictionModel
        from example_solution.solution import PredictionModel

        model = PredictionModel()
        res = score_stepwise(data_path, model, allowed_seq_ix=allowed)
        print(f"Scored rows: {res.n_scored}")
        print(f"Mean Weighted Pearson correlation: {res.weighted_pearson:.6f}")
        print(f"  t0: {res.t0:.6f}")
        print(f"  t1: {res.t1:.6f}")


if __name__ == "__main__":
    main()
