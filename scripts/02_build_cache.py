from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/02_build_cache.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from scripts.lib_io import iter_parquet_batches


FEATURE_NAMES = [
    *[f"p{i}" for i in range(12)],
    *[f"v{i}" for i in range(12)],
    *[f"dp{i}" for i in range(4)],
    *[f"dv{i}" for i in range(4)],
]
TARGET_NAMES = ["t0", "t1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=str, default="datasets/train.parquet")
    ap.add_argument("--out", type=str, default="artifacts/cache/train_raw32")
    ap.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    ap.add_argument("--batch-size", type=int, default=256_000)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.parquet)
    n_rows = pf.metadata.num_rows
    if n_rows % 1000 != 0:
        raise ValueError(f"Expected num_rows multiple of 1000, got {n_rows}")

    n_seq = n_rows // 1000
    seqs = np.empty((n_seq,), dtype=np.int64)

    dtype = np.float16 if args.dtype == "float16" else np.float32

    X_path = out_dir / "X.dat"
    Y_path = out_dir / "Y.dat"
    M_path = out_dir / "need_pred.dat"

    X = np.memmap(X_path, mode="w+", dtype=dtype, shape=(n_seq, 1000, 32))
    Y = np.memmap(Y_path, mode="w+", dtype=dtype, shape=(n_seq, 1000, 2))
    M = np.memmap(M_path, mode="w+", dtype=np.bool_, shape=(n_seq, 1000))

    cols = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_NAMES, *TARGET_NAMES]

    seq_cursor = 0

    pbar = tqdm(total=int(n_seq), desc=f"cache:{Path(args.parquet).name}", unit="seq")
    for batch in iter_parquet_batches(
        args.parquet, columns=cols, batch_size=args.batch_size
    ):
        batch_rows = len(batch["seq_ix"])
        if batch_rows % 1000 != 0:
            raise ValueError(
                f"Batch size must be multiple of 1000 (got {batch_rows}). "
                "Use --batch-size multiple of 1000."
            )

        n_seq_batch = batch_rows // 1000

        seq_ix = batch["seq_ix"].astype(np.int64, copy=False)
        step = batch["step_in_seq"].astype(np.int64, copy=False)

        # Sanity checks: contiguous sequences ordered by step
        if not np.all(step.reshape(n_seq_batch, 1000) == np.arange(1000)):
            raise ValueError("Unexpected step_in_seq ordering; cannot build cache fast")

        seqs[seq_cursor : seq_cursor + n_seq_batch] = seq_ix.reshape(n_seq_batch, 1000)[
            :, 0
        ]

        feats = np.stack([batch[n] for n in FEATURE_NAMES], axis=1).astype(
            dtype, copy=False
        )
        targs = np.stack([batch[n] for n in TARGET_NAMES], axis=1).astype(
            dtype, copy=False
        )
        need = batch["need_prediction"].astype(bool, copy=False)

        X[seq_cursor : seq_cursor + n_seq_batch] = feats.reshape(n_seq_batch, 1000, 32)
        Y[seq_cursor : seq_cursor + n_seq_batch] = targs.reshape(n_seq_batch, 1000, 2)
        M[seq_cursor : seq_cursor + n_seq_batch] = need.reshape(n_seq_batch, 1000)

        seq_cursor += n_seq_batch
        pbar.update(n_seq_batch)

    pbar.close()

    if seq_cursor != n_seq:
        raise RuntimeError(f"Wrote {seq_cursor} sequences, expected {n_seq}")

    # Metadata
    meta = {
        "parquet": str(args.parquet),
        "n_seq": int(n_seq),
        "seq_ix": seqs.astype(np.int64),
        "dtype": args.dtype,
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
    }
    np.save(out_dir / "meta.npy", meta, allow_pickle=True)

    print(f"Wrote cache to: {out_dir}")
    print(f"X: {X_path} shape={X.shape} dtype={X.dtype}")


if __name__ == "__main__":
    main()
