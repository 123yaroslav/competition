This folder contains Kaggle-style runnable scripts.

Suggested order:

1. `python scripts/00_baseline_score.py --data datasets/valid.parquet`
2. `python scripts/01_make_folds.py`
3. `python scripts/02_build_cache.py --parquet datasets/train.parquet --out artifacts/cache/train_raw32`
4. Install torch (CPU) then train:
   - `uv pip install torch`
   - `python scripts/03_train_gru.py --cache artifacts/cache/train_raw32 --fold 0 --out artifacts/models/gru_fold0.pt --use-diff`
5. Train all folds (optional but recommended):
   - `python scripts/08_train_all_folds.py --cache artifacts/cache/train_raw32 --use-diff --epochs 3`
6. Build OOF predictions (for calibration/ensembling diagnostics):
   - `python scripts/06_oof_predict.py --cache artifacts/cache/train_raw32 --models-dir artifacts/models --pattern 'gru_fold{fold}.pt' --n-folds 5`
7. Build a submission zip from trained checkpoints:
   - `python scripts/07_build_submission.py --solution scripts/solution_torch_gru.py --weights artifacts/models/gru_fold0.pt artifacts/models/gru_fold1.pt artifacts/models/gru_fold2.pt artifacts/models/gru_fold3.pt artifacts/models/gru_fold4.pt --out-zip artifacts/submission/submission.zip`
8. Score a built submission zip locally (exact stepwise interface):
   - `python scripts/09_score_submission_zip.py --zip artifacts/submission/submission.zip --data datasets/valid.parquet --max-seq 50`

## TCN (alternative to GRU)
Train a causal TCN (dilated Conv1D) on cached sequences:
- `python scripts/11_train_tcn.py --cache artifacts/cache/train_raw32 --fold 0 --epochs 8 --channels 128 --levels 7 --kernel 3 --use-diff --use-ewma --ewma-alpha 0.05 --device cuda --amp --out artifacts/models/tcn_fold0.pt`
Build and score a submission zip:
- `python scripts/07_build_submission.py --solution scripts/solution_torch_tcn.py --weights artifacts/models/tcn_fold0.pt --out-zip artifacts/submission/submission_tcn.zip`
- `python scripts/09_score_submission_zip.py --zip artifacts/submission/submission_tcn.zip --data datasets/valid.parquet --max-seq 0`

Notes:
- `train.parquet` is large (~10.7M rows). Cache building can take a while on CPU.
- The stepwise scorer lives in `scripts/lib_stepwise.py`.
