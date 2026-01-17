from __future__ import annotations

import glob
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _LoadedModel:
    model: "GRUNet"
    use_diff: bool


class GRUNet:
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.nn = nn

        self.net = nn.Module()
        self.net.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.net.head = nn.Linear(hidden, 2)

        self.hidden = hidden
        self.layers = layers

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict, strict=True)

    def eval(self):
        self.net.eval()

    def init_hidden(self):
        return self.torch.zeros((self.layers, 1, self.hidden), dtype=self.torch.float32)

    def step(self, x, h):
        # x: (1, 1, in_dim), h: (layers, 1, hidden)
        out, h2 = self.net.gru(x, h)
        z = self.net.head(out)
        y = 6.0 * self.torch.tanh(z)
        return y, h2


class PredictionModel:
    """CPU-only PyTorch GRU inference (submission `solution.py`).

    Expected artifacts in the submission zip:
    - one or more `gru_*.pt` checkpoint files produced by `scripts/03_train_gru.py`

    The code auto-detects `gru_*.pt` in the current directory and ensembles them.
    """

    def __init__(self):
        import torch

        torch.set_num_threads(1)

        ckpts = sorted(glob.glob("gru_*.pt"))
        if not ckpts:
            raise FileNotFoundError(
                "No checkpoints found. Include files like gru_fold0.pt in the submission zip."
            )

        self.models: list[_LoadedModel] = []
        self.hidden: list[torch.Tensor] = []

        for path in ckpts:
            payload = torch.load(path, map_location="cpu")
            meta = payload.get("meta", {})

            in_dim = int(meta.get("in_dim", 32))
            hidden = int(meta.get("hidden", 256))
            layers = int(meta.get("layers", 2))
            dropout = float(meta.get("dropout", 0.0))
            use_diff = bool(meta.get("use_diff", False))

            model = GRUNet(in_dim=in_dim, hidden=hidden, layers=layers, dropout=dropout)
            model.load_state_dict(payload["state_dict"])
            model.eval()

            self.models.append(_LoadedModel(model=model, use_diff=use_diff))

        self.current_seq_ix: int | None = None
        self.prev_state: np.ndarray | None = None
        self._reset_state()

    def _reset_state(self):
        self.prev_state = None
        self.hidden = [lm.model.init_hidden() for lm in self.models]

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
        for mi, lm in enumerate(self.models):
            feat = np.concatenate([x0, diff], axis=0) if lm.use_diff else x0
            x = torch.from_numpy(feat).view(1, 1, -1)
            y, h2 = lm.model.step(x, self.hidden[mi])
            self.hidden[mi] = h2
            preds.append(y.view(-1).detach().cpu().numpy())

        pred = np.mean(np.stack(preds, axis=0), axis=0)
        pred = np.clip(pred, -6.0, 6.0)

        if not data_point.need_prediction:
            return None
        return pred.astype(np.float32)
