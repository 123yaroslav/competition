import os
import sys

import numpy as np
import onnxruntime as ort

# Allow importing utils.py from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))

from utils import DataPoint


class PredictionModel:
    """GRU baseline inference (ONNX) with light calibration.

    - Maintains a 100-step rolling window per `seq_ix`.
    - Runs CPU-only ONNX Runtime inference.
    - Applies a per-target scale (tuned on a validation subset) and clips to [-6, 6].
    """

    _WINDOW = 100
    _DIM = 2

    def __init__(self):
        self.current_seq_ix: int | None = None

        # Rolling window buffer: shape (100, 32)
        self.window = np.zeros((self._WINDOW, 32), dtype=np.float32)
        self.window_ix = 0  # next write position
        self.filled = 0

        # Simple per-target calibration (scale only).
        # Tuned on a subset of `datasets/valid.parquet` locally.
        self.scale = np.array([1.90, 1.05], dtype=np.float32)

        onnx_path = os.path.join(CURRENT_DIR, "baseline.onnx")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            self.ort_session = ort.InferenceSession(
                onnx_path, sess_options, providers=["CPUExecutionProvider"]
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
        except Exception:
            # If model load fails, fall back to zeros.
            self.ort_session = None
            self.input_name = None
            self.output_name = None

    def _reset_sequence(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.window.fill(0.0)
        self.window_ix = 0
        self.filled = 0

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        # Update rolling window
        self.window[self.window_ix] = data_point.state.astype(np.float32, copy=False)
        self.window_ix = (self.window_ix + 1) % self._WINDOW
        self.filled = min(self._WINDOW, self.filled + 1)

        if not data_point.need_prediction:
            return None

        if self.ort_session is None:
            return np.zeros(self._DIM, dtype=np.float32)

        # Reconstruct chronological window
        if self.filled < self._WINDOW:
            seq_window = np.zeros_like(self.window)
            seq_window[-self.filled :] = self.window[: self.filled]
        else:
            # window_ix points to the oldest element (next write position)
            seq_window = np.concatenate(
                [self.window[self.window_ix :], self.window[: self.window_ix]], axis=0
            )

        output = self.ort_session.run(
            [self.output_name], {self.input_name: seq_window[None, :, :]}
        )[0]

        # Expected output: (1, 2) or (1, 100, 2)
        pred = output[0] if output.ndim == 2 else output[0, -1, :]
        pred = pred.astype(np.float32, copy=False)

        pred = pred * self.scale
        pred = np.clip(pred, -6.0, 6.0)
        return pred


if __name__ == "__main__":
    # Optional local scoring (requires `pyarrow` + `onnxruntime`).
    from utils import ScorerStepByStep

    test_file = os.path.join(CURRENT_DIR, "..", "datasets", "valid.parquet")
    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        results = scorer.score(model)
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for target in scorer.targets:
            print(f"  {target}: {results[target]:.6f}")
    else:
        print("valid.parquet not found")
