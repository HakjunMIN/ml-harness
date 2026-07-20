from __future__ import annotations

import builtins
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from power_forecasting.experiments import ExperimentStore


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dashboard_root(request):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    root = ROOT / "runs" / "pytest-dashboard" / safe_name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dashboard_module_import_does_not_import_streamlit(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "streamlit":
            raise AssertionError("streamlit must not be imported at dashboard module import time")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sys.modules.pop("dashboard.app", None)

    module = importlib.import_module("dashboard.app")

    assert hasattr(module, "discover_artifacts")
    assert hasattr(module, "load_dashboard_data")
    assert hasattr(module, "main")


def test_discover_artifacts_reports_only_expected_existing_files(dashboard_root):
    _write_demo_artifacts(dashboard_root)
    module = importlib.import_module("dashboard.app")

    assert module.discover_artifacts(dashboard_root) == {
        "dataset": dashboard_root / "dataset.csv",
        "database": dashboard_root / "experiments.db",
        "manifest": dashboard_root / "promotion_manifest.json",
        "report": dashboard_root / "performance_report.md",
        "generated_module": dashboard_root / "generated" / "promoted_features.py",
    }


def test_load_dashboard_data_reads_manifest_report_and_experiment_runs(dashboard_root):
    run_id = _write_demo_artifacts(dashboard_root)
    module = importlib.import_module("dashboard.app")

    data = module.load_dashboard_data(dashboard_root)

    assert data["root"] == dashboard_root
    assert data["manifest"]["decision"] == "promote"
    assert data["manifest"]["winner"]["name"] == "hour_sin"
    assert data["report"] == "# Performance report\n\nPromotion decision: promote\n"
    assert [run["id"] for run in data["runs"]] == [run_id]
    assert data["artifacts"]["generated_module"] == dashboard_root / "generated" / "promoted_features.py"


def test_load_dashboard_data_raises_clear_missing_root_and_manifest_errors(dashboard_root):
    module = importlib.import_module("dashboard.app")
    missing_root = dashboard_root / "missing"

    with pytest.raises(FileNotFoundError, match=f"artifact root not found: {re.escape(str(missing_root))}"):
        module.load_dashboard_data(missing_root)

    manifest_path = dashboard_root / "promotion_manifest.json"
    with pytest.raises(FileNotFoundError, match=f"required promotion manifest not found: {re.escape(str(manifest_path))}"):
        module.load_dashboard_data(dashboard_root)


def test_dashboard_help_subprocess_works_without_streamlit():
    result = _run_dashboard("--help")

    assert result.returncode == 0
    assert "--artifacts" in result.stdout
    assert "Streamlit" not in result.stderr


def test_main_returns_two_with_precise_error_for_missing_artifacts(capsys, dashboard_root):
    module = importlib.import_module("dashboard.app")
    missing_root = dashboard_root / "missing"

    exit_code = module.main(["--artifacts", str(missing_root)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert f"ERROR: artifact root not found: {missing_root}" in captured.err


def test_main_without_streamlit_prints_install_command_and_returns_two(
    capsys, dashboard_root, monkeypatch
):
    _write_demo_artifacts(dashboard_root)
    module = importlib.import_module("dashboard.app")
    original_import = builtins.__import__

    def no_streamlit(name, *args, **kwargs):
        if name == "streamlit":
            raise ModuleNotFoundError("No module named 'streamlit'", name="streamlit")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_streamlit)

    exit_code = module.main(["--artifacts", str(dashboard_root)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "pip install 'power-forecasting[dashboard]'" in captured.err


def test_run_dashboard_renders_expected_sections_without_unsafe_html(
    dashboard_root, monkeypatch
):
    _write_demo_artifacts(dashboard_root)
    module = importlib.import_module("dashboard.app")
    fake_streamlit = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    module.run_dashboard(dashboard_root)

    assert ("title", "Power Forecasting Demo Dashboard") in fake_streamlit.calls
    assert any(call[0] == "metric" and call[1] == "Promotion decision" for call in fake_streamlit.calls)
    assert any(call[0] in {"dataframe", "table"} for call in fake_streamlit.calls)
    assert any(call[0] == "markdown" and "Performance report" in call[1] for call in fake_streamlit.calls)
    assert all(call[2].get("unsafe_allow_html") is not True for call in fake_streamlit.calls if call[0] == "markdown")


def _run_dashboard(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src")
        if not env.get("PYTHONPATH")
        else str(ROOT / "src") + os.pathsep + env["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "dashboard/app.py", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def _write_demo_artifacts(root: Path) -> str:
    (root / "dataset.csv").write_text("plant_id,timestamp,generation_mw\n", encoding="utf-8")
    (root / "promotion_manifest.json").write_text(
        json.dumps(_manifest(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "performance_report.md").write_text(
        "# Performance report\n\nPromotion decision: promote\n",
        encoding="utf-8",
    )
    generated = root / "generated" / "promoted_features.py"
    generated.parent.mkdir()
    generated.write_text("# generated feature module\n", encoding="utf-8")

    store = ExperimentStore(root / "experiments.db")
    run_id = store.start_run("aidm-candidate-hour_sin", {"folds": 1})
    store.complete_run(run_id, {"nmae": 0.1, "mae": 1.0}, {"summary": {"candidate": "hour_sin"}})
    return run_id


def _manifest() -> dict:
    return {
        "schema_version": "1",
        "seed": 42,
        "decision": "promote",
        "failed_gates": [],
        "improvement_ratio": 0.5,
        "per_plant_deltas": {"plant_01": -0.1},
        "thresholds": {"minimum_improvement": 0.01, "max_plant_regression": 0.03},
        "baseline": {
            "model": "SPOT",
            "run_id": "baseline-run",
            "metrics": {"mae": 2.0, "rmse": 2.4, "nmae": 0.2},
        },
        "winner": {
            "name": "hour_sin",
            "run_id": "winner-run",
            "metrics": {"mae": 1.0, "rmse": 1.2, "nmae": 0.1},
        },
        "selected_specs": [
            {
                "name": "hour_sin",
                "transform": "cyclic_hour",
                "inputs": ["timestamp"],
                "parameters": {},
                "version": "1",
                "rationale": "daily phase",
            }
        ],
    }


class _FakeStreamlit(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.calls = []

    def title(self, value):
        self.calls.append(("title", value))

    def subheader(self, value):
        self.calls.append(("subheader", value))

    def metric(self, label, value):
        self.calls.append(("metric", label, value))

    def write(self, value):
        self.calls.append(("write", value))

    def json(self, value):
        self.calls.append(("json", value))

    def dataframe(self, value):
        self.calls.append(("dataframe", value))

    def table(self, value):
        self.calls.append(("table", value))

    def markdown(self, value, **kwargs):
        self.calls.append(("markdown", value, kwargs))
