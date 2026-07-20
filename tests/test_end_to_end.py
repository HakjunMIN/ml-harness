import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_cli_all_promotes_and_writes_expected_artifacts(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src")
        if not env.get("PYTHONPATH")
        else str(ROOT / "src") + os.pathsep + env["PYTHONPATH"]
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "power_forecasting.cli",
            "all",
            "--output",
            str(tmp_path),
            "--days",
            "45",
            "--plants",
            "2",
            "--seed",
            "21",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "generated" / "promoted_features.py").exists()

    manifest = json.loads(
        (tmp_path / "promotion_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["decision"] == "promote"
    assert manifest["selected_specs"]
    assert manifest["thresholds"] == {
        "minimum_improvement": 0.01,
        "max_plant_regression": 0.03,
    }
    assert (tmp_path / "performance_report.md").stat().st_size > 500
