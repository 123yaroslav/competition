from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as `python scripts/06_oof_predict.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _load_model(ckpt_path: str):
    import torch
    import torch.nn as nn

    payload = torch.load(ckpt_path, map_location="cpu")
    meta = payload.get("meta", {})
    in_dim = int(meta.get("in_dim", 32))
    hidden = int(meta.get("hidden", 256))
    layers = int(meta.get("layers", 2))
    dropout = float(meta.get("dropout", 0.0))
    use_diff = bool(meta.get("use_diff", False))

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):
            h, _ = self.gru(x)
            z = self.head(h)
            return 6.0 * torch.tanh(z)

    model = GRUModel()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    return model, use_diff


def _make_features(x: np.ndarray, use_diff: bool) -> np.ndarray:
    if not use_diff:
        return x
    d = np.zeros_like(x)
    d[:, 1:] = x[:, 1:] - x[:, :-1]
    return np.concatenate([x, d], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--folds", type=str, default="artifacts/folds/folds.parquet")
    ap.add_argument("--models-dir", type=str, default="artifacts/models")
    ap.add_argument("--pattern", type=str, default="gru_fold{fold}.pt")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-dir", type=str, default="artifacts/oof")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float16"]
    )
    args = ap.parse_args()

    import torch

    torch.set_num_threads(1)

    cache_dir = Path(args.cache)
    meta = np.load(cache_dir / "meta.npy", allow_pickle=True).item()
    seqs = meta["seq_ix"].astype(np.int64)

    dtype_np = np.float16 if meta["dtype"] == "float16" else np.float32

    X = np.memmap(
        cache_dir / "X.dat", mode="r", dtype=dtype_np, shape=(len(seqs), 1000, 32)
    )
    Y = np.memmap(
        cache_dir / "Y.dat", mode="r", dtype=dtype_np, shape=(len(seqs), 1000, 2)
    )
    M = np.memmap(
        cache_dir / "need_pred.dat", mode="r", dtype=np.bool_, shape=(len(seqs), 1000)
    )

    folds = pd.read_parquet(args.folds)
    if "source" in folds.columns:
        folds = folds[folds["source"] == "train"].copy()

    seq_to_fold = dict(
        zip(folds["seq_ix"].astype(int).tolist(), folds["fold"].astype(int).tolist())
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof_pred_parts: list[np.ndarray] = []
    oof_true_parts: list[np.ndarray] = []

    for fold in range(args.n_folds):
        ckpt = Path(args.models_dir) / args.pattern.format(fold=fold)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {ckpt}")

        model, use_diff = _load_model(str(ckpt))

        idx = [i for i, s in enumerate(seqs) if seq_to_fold.get(int(s), -1) == fold]
        if not idx:
            print(f"Fold {fold}: no sequences found")
            continue

        preds: list[np.ndarray] = []
        trues: list[np.ndarray] = []

        with torch.no_grad():
            for start in tqdm(
                range(0, len(idx), args.batch_size),
                desc=f"oof:fold{fold}",
                unit="batch",
            ):
                bidx = idx[start : start + args.batch_size]
                xb = _make_features(X[bidx].astype(np.float32), use_diff)
                yb = Y[bidx].astype(np.float32)
                mb = M[bidx]

                x = torch.from_numpy(xb)
                p = model(x).cpu().numpy()

                mask = mb.reshape(-1)
                preds.append(p.reshape(-1, 2)[mask])
                trues.append(yb.reshape(-1, 2)[mask])

        fold_pred = np.concatenate(preds, axis=0)
        fold_true = np.concatenate(trues, axis=0)

        np.save(out_dir / f"oof_pred_fold{fold}.npy", fold_pred)
        np.save(out_dir / f"oof_true_fold{fold}.npy", fold_true)

        oof_pred_parts.append(fold_pred)
        oof_true_parts.append(fold_true)

        from scripts.lib_metric import score_predictions

        s = score_predictions(fold_pred, fold_true)
        print(
            f"Fold {fold}: mean={s['weighted_pearson']:.6f} t0={s['t0']:.6f} t1={s['t1']:.6f} n={len(fold_pred)}"
        )

    oof_pred = np.concatenate(oof_pred_parts, axis=0)
    oof_true = np.concatenate(oof_true_parts, axis=0)

    np.save(out_dir / "oof_pred.npy", oof_pred)
    np.save(out_dir / "oof_true.npy", oof_true)

    from scripts.lib_metric import score_predictions

    s = score_predictions(oof_pred, oof_true)
    print(
        f"OOF total: mean={s['weighted_pearson']:.6f} t0={s['t0']:.6f} t1={s['t1']:.6f} n={len(oof_pred)}"
    )


if __name__ == "__main__":
    main()
