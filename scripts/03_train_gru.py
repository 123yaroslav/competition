from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as `python scripts/03_train_gru.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache", type=str, required=True, help="e.g. artifacts/cache/train_raw32"
    )
    ap.add_argument("--folds", type=str, default="artifacts/folds/folds.parquet")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", type=str, default="artifacts/models/gru_fold0.pt")

    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)

    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-diff", action="store_true", help="concat x and x_t-x_{t-1}")
    ap.add_argument("--max-train-seq", type=int, default=0)
    ap.add_argument("--max-valid-seq", type=int, default=0)

    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device (auto uses cuda if available).",
    )
    ap.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp autocast+GradScaler (CUDA only).",
    )

    ap.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float16"]
    )
    args = ap.parse_args()

    log_path = Path(f"artifacts/logs/gru_fold{args.fold}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    import torch.nn as nn
    from tqdm.auto import tqdm

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Keep CPU threads low; GPU kernels do the work.
    torch.set_num_threads(1)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

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

    idx_train = [
        i for i, s in enumerate(seqs) if seq_to_fold.get(int(s), -1) != args.fold
    ]
    idx_valid = [
        i for i, s in enumerate(seqs) if seq_to_fold.get(int(s), -1) == args.fold
    ]

    if args.max_train_seq and args.max_train_seq > 0:
        idx_train = idx_train[: args.max_train_seq]
    if args.max_valid_seq and args.max_valid_seq > 0:
        idx_valid = idx_valid[: args.max_valid_seq]

    if len(idx_valid) == 0:
        raise RuntimeError(
            f"No validation sequences for fold={args.fold}. Did you run scripts/01_make_folds.py?"
        )

    in_dim = 32 * (2 if args.use_diff else 1)

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_size=in_dim,
                hidden_size=args.hidden,
                num_layers=args.layers,
                dropout=args.dropout if args.layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(args.hidden, 2)

        def forward(self, x):
            h, _ = self.gru(x)
            z = self.head(h)
            return 6.0 * torch.tanh(z)

    model = GRUModel().to(device)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def make_features(x_np: np.ndarray) -> np.ndarray:
        if not args.use_diff:
            return x_np
        d = np.zeros_like(x_np)
        d[:, 1:] = x_np[:, 1:] - x_np[:, :-1]
        return np.concatenate([x_np, d], axis=-1)

    def weighted_huber(pred, y, w, delta=1.0):
        # pred,y,w: (B,T,2)
        err = pred - y
        abs_err = torch.abs(err)
        quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device))
        lin = abs_err - quad
        huber = 0.5 * quad * quad + delta * lin
        return (w * huber).mean()

    def eval_split(indices: list[int]) -> dict[str, float]:
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for start in range(0, len(indices), args.batch_size):
                batch_idx = indices[start : start + args.batch_size]
                x_np = make_features(X[batch_idx].astype(np.float32))
                y_np = Y[batch_idx].astype(np.float32)
                m_np = M[batch_idx]

                x = torch.from_numpy(x_np).to(device, non_blocking=True)
                y = torch.from_numpy(y_np).to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    p = model(x)

                # only scored steps
                mask = torch.from_numpy(m_np)
                mask2 = mask.unsqueeze(-1).expand_as(p)

                preds.append(p[mask2].view(-1, 2).detach().cpu().numpy())
                trues.append(y[mask2].view(-1, 2).detach().cpu().numpy())

        pred_arr = np.concatenate(preds, axis=0)
        y_arr = np.concatenate(trues, axis=0)
        from scripts.lib_metric import score_predictions

        return score_predictions(pred_arr, y_arr)

    best = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + epoch)
        rng.shuffle(idx_train)

        losses = []
        for start in tqdm(
            range(0, len(idx_train), args.batch_size),
            desc=f"train:fold{args.fold}:ep{epoch}",
            unit="batch",
        ):
            batch_idx = idx_train[start : start + args.batch_size]
            x_np = make_features(X[batch_idx].astype(np.float32))
            y_np = Y[batch_idx].astype(np.float32)
            m_np = M[batch_idx]

            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)

            mask = torch.from_numpy(m_np)
            mask2 = mask.unsqueeze(-1).expand_as(pred)

            pred_s = pred[mask2].view(-1, 2)
            y_s = y[mask2].view(-1, 2)

            w = torch.abs(y_s) + 1e-3
            loss = weighted_huber(pred_s, y_s, w)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(float(loss.detach().float().cpu().item()))

        val = eval_split(idx_valid)
        train_loss = float(np.mean(losses)) if losses else float("nan")

        print(
            f"epoch {epoch:02d} loss={train_loss:.5f} val_mean={val['weighted_pearson']:.6f} t0={val['t0']:.6f} t1={val['t1']:.6f}"
        )

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val": val,
                        "config": {
                            "fold": int(args.fold),
                            "hidden": int(args.hidden),
                            "layers": int(args.layers),
                            "dropout": float(args.dropout),
                            "lr": float(args.lr),
                            "weight_decay": float(args.weight_decay),
                            "batch_size": int(args.batch_size),
                            "use_diff": bool(args.use_diff),
                            "seed": int(args.seed),
                        },
                    }
                )
                + "\n"
            )

        if best is None or val["weighted_pearson"] > best[0]:
            best = (val["weighted_pearson"], epoch, val)
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "meta": {
                        "in_dim": in_dim,
                        "hidden": args.hidden,
                        "layers": args.layers,
                        "dropout": args.dropout,
                        "use_diff": bool(args.use_diff),
                        "fold": int(args.fold),
                    },
                    "best_val": val,
                },
                out_path,
            )

    if best is not None:
        print(f"Best: epoch={best[1]} mean={best[0]:.6f} details={best[2]}")


if __name__ == "__main__":
    main()
