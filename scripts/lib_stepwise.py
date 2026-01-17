from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from scripts.lib_io import iter_parquet_batches
from scripts.lib_metric import score_predictions
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

    preds: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    # Column layout expected by the package: first 3 meta, next 32 features, last 2 targets
    cols = None  # read all; relying on physical column order is brittle

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

        for i in range(len(seq_ix)):
            s_ix = int(seq_ix[i])
            if allowed_seq_ix is not None and s_ix not in allowed_seq_ix:
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
                continue

            if pred is None:
                raise ValueError(f"Prediction required but returned None for {dp}")

            pred = np.asarray(pred, dtype=np.float32)
            if pred.shape != (2,):
                raise ValueError(
                    f"Prediction must be shape (2,), got {pred.shape} for {dp}"
                )

            preds.append(pred)
            ys.append(Y[i])

    pred_arr = np.asarray(preds, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)

    scores = score_predictions(pred_arr, y_arr)
    return StepwiseScoreResult(
        t0=scores["t0"],
        t1=scores["t1"],
        weighted_pearson=scores["weighted_pearson"],
        n_scored=int(pred_arr.shape[0]),
    )
