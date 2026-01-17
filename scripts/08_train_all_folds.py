from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Allow running as `python scripts/08_train_all_folds.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--folds", type=str, default="artifacts/folds/folds.parquet")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default="artifacts/models")

    # Training hyperparams passed through
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-diff", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(args.n_folds):
        out_path = out_dir / f"gru_fold{fold}.pt"

        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "scripts", "03_train_gru.py"),
            "--cache",
            args.cache,
            "--folds",
            args.folds,
            "--fold",
            str(fold),
            "--out",
            str(out_path),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--hidden",
            str(args.hidden),
            "--layers",
            str(args.layers),
            "--dropout",
            str(args.dropout),
            "--seed",
            str(args.seed),
        ]
        if args.use_diff:
            cmd.append("--use-diff")

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
