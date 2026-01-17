# FAQ

## Competition mechanics

### What is the goal of this competition?
Predict future market indicators (`t0`, `t1`) from a sequence of previous market states. It’s a sequence modeling task.

### What’s the evaluation metric?
We use **Weighted Pearson Correlation**, averaged across the two targets.
- Predictions are clipped to `[-6, 6]`.
- Samples are weighted by `abs(target)` (large moves matter more).

See `utils.weighted_pearson_correlation`.

### Can I work in a team?
Yes. You can participate solo or as a team.

### How many submissions can I make per day?
Up to **5**.

## Data questions

### Can I use external data?
No. Train only using the provided datasets.

### Can I use pre-trained models?
Yes, if they are publicly available and do not contain external market data.

### Why are the features anonymized?
To focus the competition on modeling rather than domain-specific feature crafting.

### How should I create a validation set?
`valid.parquet` is provided, but splitting by `seq_ix` (random split or K-fold) is also reasonable because sequences are independent.

## Technical questions

### What are the compute resources for the submission environment?
- 1 CPU core
- 16 GB RAM
- No GPU
- 60-minute total runtime limit

### What is the expected size of the test datasets?
Public and Private test sets are each roughly similar to `valid.parquet` (≈ 1,500 sequences).

### Why is there no GPU?
This reflects common real-world constraints where inference must be fast and CPU-only.

### What Python libraries are available?
Expect a standard ML stack (e.g., `numpy`, `pandas`, `scikit-learn`, `torch`, `onnxruntime`, `tensorflow`).

### Does my code need to be deterministic?
Yes. It should produce the same outputs when run twice on the same data. Set random seeds and avoid nondeterministic behavior.

### My solution has multiple files. How do I submit it?
Include them all in the `.zip` (weights, helpers, configs), with `solution.py` at the root.
