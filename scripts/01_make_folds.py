from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/01_make_folds.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from scripts.lib_io import unique_seq_ix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="datasets/train.parquet")
    ap.add_argument("--valid", type=str, default="datasets/valid.parquet")
    ap.add_argument(
        "--include-valid",
        action="store_true",
        help="Include valid.parquet sequences into folds by offsetting their ids.\n"
        "Note: seq_ix overlaps between train/valid in this package.",
    )
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="artifacts/folds/folds.parquet")
    args = ap.parse_args()

    train_seq = unique_seq_ix(args.train)

    # IMPORTANT: valid.parquet uses seq_ix in [0, 1443], which overlaps with train.parquet.
    # When merging, use an offset so ids remain unique.
    if args.include_valid:
        valid_seq = unique_seq_ix(args.valid)
        offset = int(train_seq.max()) + 1
        all_seq_id = np.concatenate([train_seq, valid_seq + offset], axis=0)
        sources = np.array(["train"] * len(train_seq) + ["valid"] * len(valid_seq))
        seq_ix = np.concatenate([train_seq, valid_seq], axis=0)
    else:
        valid_seq = np.array([], dtype=np.int64)
        all_seq_id = train_seq
        sources = np.array(["train"] * len(train_seq))
        seq_ix = train_seq

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold = np.full(all_seq_id.shape[0], -1, dtype=np.int64)
    for i, (_, va) in enumerate(kf.split(all_seq_id)):
        fold[va] = i

    out = pd.DataFrame(
        {
            "seq_id": all_seq_id.astype(np.int64),
            "source": sources,
            "seq_ix": seq_ix.astype(np.int64),
            "fold": fold.astype(np.int64),
        }
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print(f"Wrote folds: {out_path}")
    print(
        f"Sequences: {len(all_seq_id)} (train={len(train_seq)} valid={len(valid_seq)}) include_valid={args.include_valid}"
    )


if __name__ == "__main__":
    main()
