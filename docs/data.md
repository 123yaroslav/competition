# Data

This competition uses a sequence dataset built from Limit Order Book (LOB) snapshots and recent trade activity.

## Files
- `datasets/train.parquet`: training split (10,721 sequences)
- `datasets/valid.parquet`: validation split (1,444 sequences)
- Public/Private test sets: hidden; similar scale to `valid.parquet`

## Row schema
Each row is one time step within one independent sequence.

**Metadata**
- `seq_ix` (int): sequence id; when it changes, you must reset any model state
- `step_in_seq` (int): step number in `[0, 999]`
- `need_prediction` (bool): if `True`, your `PredictionModel.predict(...)` must return a `(2,)` prediction

**Features (32 floats)**
Your model receives them as `data_point.state: np.ndarray`.

- Price features: `p0`…`p11` (12)
- Volume features: `v0`…`v11` (12)
- Trade price features: `dp0`…`dp3` (4)
- Trade volume features: `dv0`…`dv3` (4)

**Targets (2 floats)**
- `t0`, `t1`: future price-movement indicators (anonymized)

## Sequences and scoring window
- Each sequence is exactly **1000** steps long.
- Steps **0–98** are a **warm-up** period: use them to build context/state; they are not scored.
- Steps **99–999** are scored: these are the rows where `need_prediction == True`.

Because sequences are independent and shuffled, do not carry state between different `seq_ix` values.

## Validation strategy
`valid.parquet` is provided as a convenient holdout, but you can get a more robust estimate by splitting by `seq_ix`:
- Random split by `seq_ix`
- K-fold cross validation by `seq_ix`

## Evaluation metric (reminder)
We evaluate with **Weighted Pearson Correlation**, averaged across `t0` and `t1`.
- Predictions are clipped to `[-6, 6]` before scoring.
- Samples are weighted by `abs(target)` (large moves matter more).

See `utils.weighted_pearson_correlation` for the exact implementation.
