from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as `python scripts/11_train_tcn.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, required=True)
    ap.add_argument("--folds", type=str, default="artifacts/folds/folds.parquet")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", type=str, default="artifacts/models/tcn_fold0.pt")

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)

    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--levels", type=int, default=7, help="Number of dilated blocks")
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--use-diff", action="store_true")
    ap.add_argument("--use-ewma", action="store_true")
    ap.add_argument("--ewma-alpha", type=float, default=0.05)

    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Training device (auto uses cuda if available).",
    )
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--max-train-seq", type=int, default=0)
    ap.add_argument("--max-valid-seq", type=int, default=0)
    args = ap.parse_args()

    log_path = Path(f"artifacts/logs/tcn_fold{args.fold}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    if not idx_valid:
        raise RuntimeError(
            f"No validation sequences for fold={args.fold}. Run scripts/01_make_folds.py"
        )

    from scripts.lib_features import build_features_np

    in_dim = 32
    if args.use_diff:
        in_dim += 32
    if args.use_ewma:
        in_dim += 32

    def make_features(x_np: np.ndarray) -> np.ndarray:
        return build_features_np(
            x_np.astype(np.float32, copy=False),
            use_diff=bool(args.use_diff),
            use_ewma=bool(args.use_ewma),
            ewma_alpha=float(args.ewma_alpha),
        )

    class CausalConv1d(nn.Module):
        def __init__(self, cin: int, cout: int, kernel: int, dilation: int):
            super().__init__()
            self.kernel = kernel
            self.dilation = dilation
            self.conv = nn.Conv1d(cin, cout, kernel_size=kernel, dilation=dilation)

        def forward(self, x):
            # x: (B, C, T)
            pad = (self.kernel - 1) * self.dilation
            x = F.pad(x, (pad, 0))
            return self.conv(x)

    class TCNBlock(nn.Module):
        def __init__(
            self, cin: int, cout: int, kernel: int, dilation: int, dropout: float
        ):
            super().__init__()
            self.c1 = CausalConv1d(cin, cout, kernel, dilation)
            self.c2 = CausalConv1d(cout, cout, kernel, dilation)
            self.dropout = nn.Dropout(dropout)
            self.down = nn.Conv1d(cin, cout, kernel_size=1) if cin != cout else None

        def forward(self, x):
            y = self.c1(x)
            y = torch.relu(y)
            y = self.dropout(y)
            y = self.c2(y)
            y = torch.relu(y)
            y = self.dropout(y)
            res = x if self.down is None else self.down(x)
            return y + res

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            cin = in_dim
            for i in range(args.levels):
                dilation = 2**i
                layers.append(
                    TCNBlock(cin, args.channels, args.kernel, dilation, args.dropout)
                )
                cin = args.channels
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Conv1d(args.channels, 2, kernel_size=1)

        def forward(self, x):
            # x: (B, T, D) -> (B, D, T)
            x = x.transpose(1, 2)
            h = self.tcn(x)
            y = self.head(h)  # (B, 2, T)
            y = y.transpose(1, 2)  # (B, T, 2)
            return 6.0 * torch.tanh(y)

    model = TCN().to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def weighted_huber(pred, y, w, delta=1.0):
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
                x_np = make_features(X[batch_idx])
                y_np = Y[batch_idx].astype(np.float32)
                m_np = M[batch_idx]

                x = torch.from_numpy(x_np).to(device)
                y = torch.from_numpy(y_np).to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    p = model(x)

                mask = torch.from_numpy(m_np).to(device)
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
            desc=f"tcn:fold{args.fold}:ep{epoch}",
            unit="batch",
        ):
            batch_idx = idx_train[start : start + args.batch_size]
            x_np = make_features(X[batch_idx])
            y_np = Y[batch_idx].astype(np.float32)
            m_np = M[batch_idx]

            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x)

                mask = torch.from_numpy(m_np).to(device)
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

        msg = (
            f"epoch {epoch:02d} loss={train_loss:.5f} "
            f"fold_mean={val['weighted_pearson']:.6f} t0={val['t0']:.6f} t1={val['t1']:.6f}"
        )
        print(msg)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val": val,
                        "config": {
                            "fold": int(args.fold),
                            "channels": int(args.channels),
                            "levels": int(args.levels),
                            "kernel": int(args.kernel),
                            "dropout": float(args.dropout),
                            "lr": float(args.lr),
                            "weight_decay": float(args.weight_decay),
                            "batch_size": int(args.batch_size),
                            "use_diff": bool(args.use_diff),
                            "use_ewma": bool(args.use_ewma),
                            "ewma_alpha": float(args.ewma_alpha),
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
                        "arch": "tcn",
                        "in_dim": int(in_dim),
                        "channels": int(args.channels),
                        "levels": int(args.levels),
                        "kernel": int(args.kernel),
                        "dropout": float(args.dropout),
                        "use_diff": bool(args.use_diff),
                        "use_ewma": bool(args.use_ewma),
                        "ewma_alpha": float(args.ewma_alpha),
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
