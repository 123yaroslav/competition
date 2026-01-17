from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Allow running as `python scripts/07_build_submission.py`
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", type=str, required=True, help="solution.py to ship")
    ap.add_argument(
        "--weights", type=str, nargs="*", default=[], help="model files to include"
    )
    ap.add_argument(
        "--out-zip", type=str, default="artifacts/submission/submission.zip"
    )
    args = ap.parse_args()

    out_zip = Path(args.out_zip)
    stage = out_zip.parent / "stage"

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.solution, stage / "solution.py")
    for w in args.weights:
        p = Path(w)
        shutil.copy2(p, stage / p.name)

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    import zipfile

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in stage.iterdir():
            z.write(p, arcname=p.name)

    print(f"Wrote: {out_zip}")
    with zipfile.ZipFile(out_zip, "r") as z:
        print("Zip contents:")
        for info in z.infolist():
            print(f"  {info.filename} ({info.file_size} bytes)")


if __name__ == "__main__":
    main()
