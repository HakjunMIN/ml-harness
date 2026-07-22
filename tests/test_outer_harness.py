from __future__ import annotations

import json
import subprocess
from pathlib import Path

from power_forecasting.research_contracts import load_research_loop_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / ".agents" / "fixtures"
RUNNER = ROOT / ".agents" / "scripts" / "run-research-loop.sh"


def test_research_fixture_uses_config_relative_synthetic_inputs_and_agents_runs_output() -> None:
    config_path = FIXTURES / "research-loop.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = load_research_loop_config(
        payload,
        config_path=config_path,
        repository_root=ROOT,
    )

    assert Path(config.dataset_path) == (FIXTURES / "valid-dataset.csv").resolve()
    assert Path(config.legacy_manifest_path) == (FIXTURES / "promoted-manifest.json").resolve()
    assert Path(config.run_dir) == (ROOT / ".agents" / "runs" / "research-loop-fixture").resolve()
    assert "customer" not in json.dumps(payload).lower()
    assert "generation_mw" not in json.dumps(payload).lower()


def test_research_runner_has_exact_cli_and_does_not_export_environment_secrets() -> None:
    content = RUNNER.read_text(encoding="utf-8")

    assert "uv run python -m power_forecasting.cli research-loop" in content
    assert "export " not in content
    assert "cd \"$repo_root\"" in content


def test_research_runner_accepts_only_help_and_config_options(tmp_path: Path) -> None:
    missing = subprocess.run(
        [str(RUNNER)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    unknown = subprocess.run(
        [str(RUNNER), "--config", str(FIXTURES / "research-loop.json"), "--unexpected"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    help_result = subprocess.run(
        [str(RUNNER), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert missing.returncode == 2
    assert unknown.returncode == 2
    assert help_result.returncode == 0


def test_research_runner_rejects_missing_config_without_creating_output(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(RUNNER), "--config", str(tmp_path / "missing.json")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "config not found" in result.stderr
