from __future__ import annotations

import numpy as np


def build_features_np(
    x: np.ndarray,
    *,
    use_diff: bool,
    use_ewma: bool,
    ewma_alpha: float = 0.05,
    eps: float = 1e-4,
) -> np.ndarray:
    """Build per-step features for sequence models.

    Args:
        x: shape (B, T, 32) float32

    Returns:
        features: shape (B, T, D)
    """

    feats = [x]

    if use_diff:
        d = np.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        feats.append(d)

    if use_ewma:
        # EWMA mean/var computed per sequence, per feature.
        # Vectorized across batch.
        b, t, f = x.shape
        mean = np.zeros((b, f), dtype=np.float32)
        var = np.zeros((b, f), dtype=np.float32)

        norm = np.empty_like(x)

        a = float(ewma_alpha)
        one_minus_a = 1.0 - a

        for i in range(t):
            xi = x[:, i]
            if i == 0:
                mean[:] = xi
                var[:] = 0.0
            else:
                delta = xi - mean
                mean[:] = one_minus_a * mean + a * xi
                var[:] = one_minus_a * var + a * (delta * delta)

            norm[:, i] = (xi - mean) / np.sqrt(var + eps)

        feats.append(norm)

    return np.concatenate(feats, axis=-1).astype(np.float32, copy=False)
