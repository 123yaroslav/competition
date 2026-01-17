from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from scripts.lib_io import iter_parquet_batches
from utils import DataPoint


class StepwiseModel(Protocol):
    def predict(self, data_point: DataPoint) -> np.ndarray | None: ...


@dataclass(frozen=True)
class StepwiseScoreResult:
    t0: float
    t1: float
    weighted_pearson: float
    n_scored: int


def score_stepwise(
    parquet_path: str | Path,
    model: StepwiseModel,
    batch_size: int = 256_000,
    allowed_seq_ix: set[int] | None = None,
) -> StepwiseScoreResult:
    """Exact step-by-step scorer mirroring the competition interface.

    This streams parquet in batches, calls `model.predict(DataPoint)` for each row,
    and scores only rows where `need_prediction` is True.
    """

    # Streaming metric accumulators (avoid storing ~1.4M predictions in RAM)
    # For each target we keep weighted sums:
    #   Sw, Swy, Swp, Swyy, Swpp, Swyp
    Sw = np.zeros(2, dtype=np.float64)
    Swy = np.zeros(2, dtype=np.float64)
    Swp = np.zeros(2, dtype=np.float64)
    Swyy = np.zeros(2, dtype=np.float64)
    Swpp = np.zeros(2, dtype=np.float64)
    Swyp = np.zeros(2, dtype=np.float64)

    n_scored = 0

    # Column layout expected by the package: first 3 meta, next 32 features, last 2 targets
    cols = None  # read all; relying on physical column order is brittle

    pf = pq.ParquetFile(str(parquet_path))
    total_rows = int(pf.metadata.num_rows)

    pbar = tqdm(total=total_rows, desc=f"score:{Path(parquet_path).name}", unit="rows")
    for batch in iter_parquet_batches(
        parquet_path, columns=cols, batch_size=batch_size
    ):
        seq_ix = batch["seq_ix"].astype(np.int64, copy=False)
        step_in_seq = batch["step_in_seq"].astype(np.int64, copy=False)
        need_prediction = batch["need_prediction"].astype(bool, copy=False)

        # Features/targets by explicit column names to avoid schema drift
        feature_names = [
            *[f"p{i}" for i in range(12)],
            *[f"v{i}" for i in range(12)],
            *[f"dp{i}" for i in range(4)],
            *[f"dv{i}" for i in range(4)],
        ]
        target_names = ["t0", "t1"]

        X = np.stack([batch[n] for n in feature_names], axis=1).astype(
            np.float32, copy=False
        )
        Y = np.stack([batch[n] for n in target_names], axis=1).astype(
            np.float32, copy=False
        )

        batch_len = len(seq_ix)

        # Update progress as we actually process rows (not upfront).
        processed_in_batch = 0

        for i in range(batch_len):
            s_ix = int(seq_ix[i])
            if allowed_seq_ix is not None and s_ix not in allowed_seq_ix:
                processed_in_batch += 1
                if processed_in_batch % 4096 == 0:
                    pbar.update(4096)
                continue

            dp = DataPoint(
                s_ix,
                int(step_in_seq[i]),
                bool(need_prediction[i]),
                X[i],
            )
            pred = model.predict(dp)

            if not dp.need_prediction:
                if pred is not None:
                    raise ValueError(f"Prediction not needed but returned for {dp}")

                processed_in_batch += 1
                if processed_in_batch % 4096 == 0:
                    pbar.update(4096)
                continue

            if pred is None:
                raise ValueError(f"Prediction required but returned None for {dp}")

            pred = np.asarray(pred, dtype=np.float32)
            if pred.shape != (2,):
                raise ValueError(
                    f"Prediction must be shape (2,), got {pred.shape} for {dp}"
                )

            # Streaming update for both targets
            y = Y[i].astype(np.float64, copy=False)
            p = np.clip(pred.astype(np.float64, copy=False), -6.0, 6.0)

            w = np.abs(y)
            w = np.maximum(w, 1e-8)

            Sw += w
            Swy += w * y
            Swp += w * p
            Swyy += w * y * y
            Swpp += w * p * p
            Swyp += w * y * p

            n_scored += 1

            processed_in_batch += 1
            if processed_in_batch % 4096 == 0:
                pbar.update(4096)

        # flush remainder progress
        rem = processed_in_batch % 4096
        if rem:
            pbar.update(rem)

    pbar.close()

    def _corr(ix: int) -> float:
        if Sw[ix] <= 0:
            return 0.0
        mean_y = Swy[ix] / Sw[ix]
        mean_p = Swp[ix] / Sw[ix]

        cov = (Swyp[ix] / Sw[ix]) - mean_y * mean_p
        var_y = (Swyy[ix] / Sw[ix]) - mean_y * mean_y
        var_p = (Swpp[ix] / Sw[ix]) - mean_p * mean_p

        if var_y <= 0 or var_p <= 0:
            return 0.0
        return float(cov / (np.sqrt(var_y) * np.sqrt(var_p)))

    t0 = _corr(0)
    t1 = _corr(1)

    return StepwiseScoreResult(
        t0=t0,
        t1=t1,
        weighted_pearson=float((t0 + t1) / 2),
        n_scored=int(n_scored),
    )
