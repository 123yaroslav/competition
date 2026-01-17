from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/04_calibrate.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import numpy as np

from utils import weighted_pearson_correlation


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Fit (a,b) for clip(a*y_pred + b, -6, 6) maximizing Weighted Pearson."""

    best = (-1e9, 1.0, 0.0)

    # Coarse grid then refine
    for a in np.linspace(0.5, 2.5, 41):
        for b in np.linspace(-0.5, 0.5, 41):
            yp = np.clip(a * y_pred + b, -6.0, 6.0)
            s = weighted_pearson_correlation(y_true, yp)
            if s > best[0]:
                best = (float(s), float(a), float(b))

    return best[1], best[2], best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred", type=str, required=True, help=".npy predictions shape (n,2)"
    )
    ap.add_argument("--true", type=str, required=True, help=".npy targets shape (n,2)")
    ap.add_argument("--out", type=str, default="artifacts/models/calibration.npy")
    args = ap.parse_args()

    pred = np.load(args.pred)
    y = np.load(args.true)

    out = []
    for t in range(2):
        a, b, s = fit_affine(y[:, t], pred[:, t])
        out.append({"target": f"t{t}", "a": a, "b": b, "score": s})
        print(out[-1])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out, allow_pickle=True)
    print(f"Wrote calibration: {args.out}")


if __name__ == "__main__":
    main()
