import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


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

    evidence_block = _readme_bash_block("readme-evidence-check")
    evidence_env = {**env, "OUTPUT_DIR": str(tmp_path)}
    evidence = subprocess.run(
        ["bash", "-c", evidence_block],
        cwd=ROOT,
        env=evidence_env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert evidence.returncode == 0, evidence.stderr + evidence.stdout
    assert "winner_metrics" not in evidence_block
    assert 'manifest["winner"]["metrics"]["nmae"]' in evidence_block
    assert float(evidence.stdout.strip()) == manifest["winner"]["metrics"]["nmae"]


def _readme_bash_block(marker):
    content = README.read_text(encoding="utf-8")
    match = re.search(
        rf"<!-- {re.escape(marker)}: start -->\n"
        r"```bash\n(?P<code>.*?)\n```\n"
        rf"<!-- {re.escape(marker)}: end -->",
        content,
        flags=re.DOTALL,
    )
    assert match, f"README bash block marker not found: {marker}"
    return match.group("code")
