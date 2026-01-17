from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

# Allow running as `python scripts/09_score_submission_zip.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from scripts.lib_stepwise import score_stepwise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, required=True, help="Path to submission.zip")
    ap.add_argument("--data", type=str, default="datasets/valid.parquet")
    ap.add_argument(
        "--max-seq",
        type=int,
        default=0,
        help="If >0, score only first N sequences by seq_ix.",
    )
    args = ap.parse_args()

    repo_root = Path(REPO_ROOT)
    data_path = (repo_root / args.data).resolve()
    zip_path = Path(args.zip).resolve()

    workdir = Path(tempfile.mkdtemp(prefix="submission_score_"))

    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(workdir)

        sys.path.insert(0, str(workdir))
        os.chdir(workdir)

        from solution import PredictionModel  # type: ignore

        allowed = None
        if args.max_seq and args.max_seq > 0:
            allowed = set(range(args.max_seq))

        model = PredictionModel()
        res = score_stepwise(str(data_path), model, allowed_seq_ix=allowed)
        print(res)

    finally:
        shutil.rmtree(workdir)


if __name__ == "__main__":
    main()
