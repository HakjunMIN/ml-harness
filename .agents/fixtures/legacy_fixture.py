from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODE = sys.argv[1] if len(sys.argv) > 1 else "valid"
ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ["HARNESS_PREDICTIONS_OUTPUT"])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
if MODE == "valid":
    source = ROOT / "valid-predictions.csv"
    if source.resolve() != OUTPUT.resolve():
        shutil.copyfile(source, OUTPUT)
elif MODE == "missing-column":
    source = ROOT / "missing-column-predictions.csv"
    if source.resolve() != OUTPUT.resolve():
        shutil.copyfile(source, OUTPUT)
else:
    raise SystemExit(f"unknown fixture mode: {MODE}")
