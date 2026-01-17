from __future__ import annotations

import numpy as np

from utils import weighted_pearson_correlation


def score_predictions(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Compute per-target + mean score.

    Args:
        pred: shape (n, 2)
        y: shape (n, 2)
    """

    s0 = weighted_pearson_correlation(y[:, 0], pred[:, 0])
    s1 = weighted_pearson_correlation(y[:, 1], pred[:, 1])
    return {"t0": float(s0), "t1": float(s1), "weighted_pearson": float((s0 + s1) / 2)}
