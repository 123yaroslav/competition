# Submission guide

This is a **code competition**. You submit a single `.zip` containing your inference code (and any artifacts like weights).

## What to submit
- A `.zip` file
- `solution.py` **at the root of the zip**
- `solution.py` must define a `PredictionModel` class with a `predict(...)` method

## Required interface
```python
import numpy as np
from utils import DataPoint

class PredictionModel:
    def __init__(self):
        # Load weights, init state, etc.
        pass

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # Reset any recurrent state when seq_ix changes.
        # Return None when need_prediction is False.
        if not data_point.need_prediction:
            return None

        # When need_prediction is True, return shape (2,) for (t0, t1).
        return np.zeros(2)
```

`DataPoint` fields:
- `seq_ix: int`
- `step_in_seq: int`
- `need_prediction: bool`
- `state: np.ndarray` (32 features: `p0..p11`, `v0..v11`, `dp0..dp3`, `dv0..dv3`)

## Packaging
From your solution directory:
```bash
zip -r ../submission.zip .
```

Common pitfalls:
- `solution.py` must be at the top level inside the zip (not nested in a folder).
- Include any weights/configs you load at runtime.

## Runtime constraints
Your submission runs in an isolated Linux container:
- CPU: 1 vCPU core (no GPU)
- RAM: 16 GB
- Time limit: 60 minutes (entire test set)
- No internet access

## Determinism
Your code should be deterministic: set seeds and avoid any nondeterministic ops.

## Notes on the scoring container
The scorer image is based on `python:3.11-slim-bookworm` and uses environment variables to keep common ML libraries offline at runtime.
If you need a specific package added to the scorer image, contact the organizers.
