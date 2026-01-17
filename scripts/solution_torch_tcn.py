from __future__ import annotations

import glob

import numpy as np


def _load_checkpoints():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_num_threads(1)

    ckpts = sorted(glob.glob("tcn_*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found. Include files like tcn_fold0.pt")

    class CausalConv1d(nn.Module):
        def __init__(self, cin: int, cout: int, kernel: int, dilation: int):
            super().__init__()
            self.kernel = kernel
            self.dilation = dilation
            self.conv = nn.Conv1d(cin, cout, kernel_size=kernel, dilation=dilation)

        def forward(self, x):
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
        def __init__(
            self, in_dim: int, channels: int, levels: int, kernel: int, dropout: float
        ):
            super().__init__()
            layers = []
            cin = in_dim
            for i in range(levels):
                dilation = 2**i
                layers.append(TCNBlock(cin, channels, kernel, dilation, dropout))
                cin = channels
            self.tcn = nn.Sequential(*layers)
            self.head = nn.Conv1d(channels, 2, kernel_size=1)

        def forward(self, x):
            # x: (B, D, T)
            h = self.tcn(x)
            y = self.head(h)
            return 6.0 * torch.tanh(y)

    loaded = []
    for path in ckpts:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})

        in_dim = int(meta.get("in_dim", 32))
        channels = int(meta.get("channels", 128))
        levels = int(meta.get("levels", 7))
        kernel = int(meta.get("kernel", 3))
        dropout = float(meta.get("dropout", 0.1))

        use_diff = bool(meta.get("use_diff", False))
        use_ewma = bool(meta.get("use_ewma", False))
        ewma_alpha = float(meta.get("ewma_alpha", 0.05))

        model = TCN(
            in_dim=in_dim,
            channels=channels,
            levels=levels,
            kernel=kernel,
            dropout=dropout,
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()

        # receptive field window
        rf = 1 + (kernel - 1) * sum(2**i for i in range(levels))
        loaded.append((model, use_diff, use_ewma, ewma_alpha, int(in_dim), int(rf)))

    return loaded


class PredictionModel:
    """TCN step-by-step inference.

    Maintains a rolling feature window of length = receptive field.
    """

    def __init__(self):
        self.models = (
            _load_checkpoints()
        )  # list[(model, use_diff, use_ewma, alpha, in_dim, window)]
        self.current_seq_ix: int | None = None
        self.prev_state: np.ndarray | None = None

        self.buffers: list[np.ndarray] = []
        self.buf_ix: list[int] = []
        self.filled: list[int] = []

        self.ewma_mean: list[np.ndarray | None] = []
        self.ewma_var: list[np.ndarray | None] = []

        self._reset_state()

    def _reset_state(self):
        self.prev_state = None
        self.buffers = []
        self.buf_ix = []
        self.filled = []
        self.ewma_mean = []
        self.ewma_var = []

        for model, use_diff, use_ewma, alpha, in_dim, window in self.models:
            self.buffers.append(np.zeros((window, in_dim), dtype=np.float32))
            self.buf_ix.append(0)
            self.filled.append(0)
            self.ewma_mean.append(None)
            self.ewma_var.append(None)

    def predict(self, data_point) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self._reset_state()

        x0 = np.asarray(data_point.state, dtype=np.float32)
        if self.prev_state is None:
            diff = np.zeros_like(x0)
        else:
            diff = x0 - self.prev_state
        self.prev_state = x0

        import torch

        preds = []

        for mi, (model, use_diff, use_ewma, alpha, in_dim, window) in enumerate(
            self.models
        ):
            parts = [x0]
            if use_diff:
                parts.append(diff)

            if use_ewma:
                a = float(alpha)
                one_minus_a = 1.0 - a

                mean = self.ewma_mean[mi]
                var = self.ewma_var[mi]
                if mean is None or var is None:
                    mean = x0.copy()
                    var = np.zeros_like(x0)

                delta = x0 - mean
                mean = one_minus_a * mean + a * x0
                var = one_minus_a * var + a * (delta * delta)

                self.ewma_mean[mi] = mean
                self.ewma_var[mi] = var

                norm = (x0 - mean) / np.sqrt(var + 1e-4)
                parts.append(norm.astype(np.float32, copy=False))

            feat = np.concatenate(parts, axis=0)
            if feat.shape[0] != in_dim:
                # Safety fallback
                feat = feat[:in_dim]

            # update ring buffer
            b = self.buffers[mi]
            ix = self.buf_ix[mi]
            b[ix] = feat
            self.buf_ix[mi] = (ix + 1) % window
            self.filled[mi] = min(window, self.filled[mi] + 1)

            if not data_point.need_prediction:
                continue

            # chronological window
            if self.filled[mi] < window:
                w = np.zeros_like(b)
                w[-self.filled[mi] :] = b[: self.filled[mi]]
            else:
                start = self.buf_ix[mi]
                w = np.concatenate([b[start:], b[:start]], axis=0)

            # model expects (B, D, T)
            x = torch.from_numpy(w.T).unsqueeze(0)
            y = model(x)  # (1, 2, T)
            pred = y[0, :, -1].detach().cpu().numpy()
            preds.append(pred)

        if not data_point.need_prediction:
            return None

        pred = np.mean(np.stack(preds, axis=0), axis=0)
        pred = np.clip(pred, -6.0, 6.0)
        return pred.astype(np.float32)
