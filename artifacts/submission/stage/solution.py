from __future__ import annotations

import glob

import numpy as np


def _load_checkpoints():
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)

    ckpts = sorted(glob.glob("gru_*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            "No checkpoints found. Include files like gru_fold0.pt in the submission zip."
        )

    class GRUModel(nn.Module):
        def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.in_dim = in_dim
            self.hidden = hidden
            self.layers = layers

            self.gru = nn.GRU(
                input_size=in_dim,
                hidden_size=hidden,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden, 2)

        def step(self, x: torch.Tensor, h: torch.Tensor):
            # x: (1, 1, in_dim), h: (layers, 1, hidden)
            out, h2 = self.gru(x, h)
            z = self.head(out)
            y = 6.0 * torch.tanh(z)
            return y, h2

    loaded = []
    for path in ckpts:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("meta", {})

        in_dim = int(meta.get("in_dim", 32))
        hidden = int(meta.get("hidden", 256))
        layers = int(meta.get("layers", 2))
        dropout = float(meta.get("dropout", 0.0))
        use_diff = bool(meta.get("use_diff", False))

        model = GRUModel(in_dim=in_dim, hidden=hidden, layers=layers, dropout=dropout)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.eval()

        h0 = torch.zeros((layers, 1, hidden), dtype=torch.float32)
        loaded.append((model, use_diff, h0))

    return loaded


class PredictionModel:
    """CPU-only PyTorch GRU inference (submission `solution.py`).

    - Loads one or more `gru_*.pt` checkpoints from the zip root.
    - Maintains GRU hidden state per sequence.
    - Optional feature augmentation: concat first differences (if `use_diff`).
    - Ensembles by averaging predictions across checkpoints.
    """

    def __init__(self):
        self.models = _load_checkpoints()  # list[(model, use_diff, hidden)]

        self.current_seq_ix: int | None = None
        self.prev_state: np.ndarray | None = None

    def _reset_state(self):
        # Reset per-sequence state
        self.prev_state = None
        # Reset hidden state for each model
        import torch

        reset = []
        for model, use_diff, _ in self.models:
            h0 = torch.zeros((model.layers, 1, model.hidden), dtype=torch.float32)
            reset.append((model, use_diff, h0))
        self.models = reset

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
        updated = []
        for model, use_diff, h in self.models:
            feat = np.concatenate([x0, diff], axis=0) if use_diff else x0
            x = torch.from_numpy(feat).view(1, 1, -1)
            y, h2 = model.step(x, h)
            updated.append((model, use_diff, h2))
            preds.append(y.view(-1).detach().cpu().numpy())

        self.models = updated

        # For warm-up steps we still update state, but return None.
        if not data_point.need_prediction:
            return None

        pred = np.mean(np.stack(preds, axis=0), axis=0)
        pred = np.clip(pred, -6.0, 6.0)
        return pred.astype(np.float32)
