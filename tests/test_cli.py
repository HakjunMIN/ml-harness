from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

from power_forecasting import aidd, aidm, cli, reporting
from power_forecasting.data import DataContractError
from power_forecasting.evaluation import EvaluationResult
from power_forecasting.features import FeatureSpec
from power_forecasting.models import SUPPORTED_MODEL_NAMES


ROOT = Path(__file__).resolve().parents[1]
LEGACY_NAMES = tuple(SUPPORTED_MODEL_NAMES)


def test_run_all_real_e2e_produces_promoted_artifacts_and_valid_report(tmp_path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message=".*encountered in matmul", category=RuntimeWarning
        )
        paths = cli.run_all(tmp_path, days=45, plants=2, seed=13)

    assert list(paths) == [
        "dataset",
        "database",
        "manifest",
        "generated_module",
        "report",
    ]
    assert paths == {
        "dataset": tmp_path / "dataset.csv",
        "database": tmp_path / "experiments.db",
        "manifest": tmp_path / "promotion_manifest.json",
        "generated_module": tmp_path / "generated" / "promoted_features.py",
        "report": tmp_path / "performance_report.md",
    }
    for path in paths.values():
        assert path.exists(), path

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["decision"] == "promote"
    assert aidd.validate_promotion_manifest(manifest)

    report = paths["report"].read_text(encoding="utf-8")
    assert "Promotion decision: promote" in report
    assert "## Ranked AIDM candidates" in report
    assert "## Failed gates" in report
    assert "None" in report
    assert "## Selected feature specs" in report
    legacy_rows = [
        line
        for line in report.splitlines()
        if any(line.startswith(f"| {name} |") for name in LEGACY_NAMES)
    ]
    assert [row.split("|")[1].strip() for row in legacy_rows] == list(LEGACY_NAMES)


def test_render_performance_report_is_deterministic_complete_and_escapes_tables():
    aidm_result = _aidm_result(candidate_name="candidate|pipe", valid_for_aidd=False)
    artifact_paths = {
        "generated|module": Path("artifacts/promoted|features.py"),
        "report": Path("artifacts/performance_report.md"),
    }

    first = reporting.render_performance_report(
        {"rows": 10, "plants": 2, "time_start": "2024|01|01", "time_end": "2024-01-02"},
        _legacy_results(),
        aidm_result,
        artifact_paths,
    )
    second = reporting.render_performance_report(
        {"rows": 10, "plants": 2, "time_start": "2024|01|01", "time_end": "2024-01-02"},
        _legacy_results(),
        aidm_result,
        artifact_paths,
    )

    assert first == second
    assert first.endswith("\n")
    assert "\r\n" not in first
    for name in LEGACY_NAMES:
        assert first.count(f"| {name} |") == 1
    assert "| 1 | candidate\\|pipe | 1.000000 | 1.200000 | 0.100000 |" in first
    assert "| minimum_improvement | 0.010000 |" in first
    assert "Improvement ratio: 0.500000" in first
    assert "plant\\|01" in first
    assert "daily \\| phase" in first
    assert "generated\\|module" in first
    assert "promoted\\|features.py" in first


def test_write_performance_report_preserves_existing_target_on_render_or_replace_failure(
    tmp_path, monkeypatch
):
    target = tmp_path / "nested" / "performance_report.md"
    target.parent.mkdir()
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("keep me\n")
    original_render = reporting.render_performance_report

    def fail_render(*args, **kwargs):
        raise UnicodeEncodeError("utf-8", "x", 0, 1, "injected render failure")

    monkeypatch.setattr(reporting, "render_performance_report", fail_render)
    with pytest.raises(UnicodeEncodeError):
        reporting.write_performance_report(
            {},
            _legacy_results(),
            _aidm_result(),
            {},
            target=target,
        )

    assert target.read_text(encoding="utf-8") == "keep me\n"
    monkeypatch.setattr(reporting, "render_performance_report", original_render)

    def fail_replace(source, destination):
        raise RuntimeError("injected replace failure")

    monkeypatch.setattr(reporting.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="injected replace failure"):
        reporting.write_performance_report(
            {"rows": 1, "plants": 1, "time_start": "a", "time_end": "b"},
            _legacy_results(),
            _aidm_result(),
            {},
            target=target,
        )

    assert target.read_text(encoding="utf-8") == "keep me\n"
    assert not list(target.parent.glob("*.tmp"))


def test_api_artifact_path_conventions_and_manifest_round_trip(tmp_path, monkeypatch):
    dataset = cli.run_generate_data(tmp_path, days=4, plants=1, seed=5)
    assert dataset == tmp_path / "dataset.csv"
    assert dataset.exists()

    captured = {"legacy": []}

    def fake_evaluate(frame, definition, feature_specs, folds):
        captured["legacy"].append(
            (definition.name, folds, pd.api.types.is_datetime64_any_dtype(frame["timestamp"]))
        )
        return _evaluation(0.1)

    monkeypatch.setattr(cli, "evaluate_model", fake_evaluate)
    legacy = cli.run_legacy(tmp_path, dataset=dataset, folds=2)
    assert list(legacy) == list(LEGACY_NAMES)
    assert captured["legacy"] == [(name, 2, True) for name in LEGACY_NAMES]

    def fake_run_aidm(frame, database_path, config):
        captured["database_path"] = Path(database_path)
        captured["aidm_timestamp"] = pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
        captured["aidm_config"] = config
        return _aidm_result(valid_for_aidd=True)

    monkeypatch.setattr(cli, "run_aidm", fake_run_aidm)
    result = cli.run_aidm_workflow(
        tmp_path, dataset=dataset, config=aidm.AIDMConfig(folds=2)
    )
    assert result.manifest["decision"] == "promote"
    assert captured["database_path"] == tmp_path / "experiments.db"
    assert captured["aidm_timestamp"] is True
    assert captured["aidm_config"].folds == 2

    manifest = tmp_path / "promotion_manifest.json"
    assert manifest.exists()
    assert manifest.read_bytes().endswith(b"\n")
    assert aidd.validate_promotion_manifest(json.loads(manifest.read_text(encoding="utf-8")))

    generated = cli.run_aidd_workflow(tmp_path, manifest=manifest)
    assert generated == tmp_path / "generated" / "promoted_features.py"
    assert generated.exists()


def test_missing_or_invalid_provided_dataset_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        cli.run_legacy(tmp_path, dataset=tmp_path / "missing.csv")

    invalid = tmp_path / "invalid.csv"
    with invalid.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("plant_id,timestamp\nplant_01,not-a-date\n")
    with pytest.raises(DataContractError):
        cli.run_legacy(tmp_path, dataset=invalid)


def test_run_all_rejected_decision_writes_report_but_does_not_generate_code(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "run_legacy", lambda output, dataset=None, folds=3: _legacy_results())
    monkeypatch.setattr(
        cli,
        "run_aidm_workflow",
        lambda output, dataset=None, config=aidm.AIDMConfig(): _aidm_result(
            decision="reject",
            failed_gates=["insufficient_improvement:improvement_ratio=0.000000<threshold=0.010000"],
            valid_for_aidd=True,
        ),
    )

    with pytest.raises(RuntimeError, match="AIDM rejected promotion"):
        cli.run_all(tmp_path, days=2, plants=1, seed=3, folds=1)

    assert (tmp_path / "dataset.csv").exists()
    assert json.loads((tmp_path / "promotion_manifest.json").read_text(encoding="utf-8"))[
        "decision"
    ] == "reject"
    assert not (tmp_path / "generated" / "promoted_features.py").exists()
    report = (tmp_path / "performance_report.md").read_text(encoding="utf-8")
    assert "Promotion decision: reject" in report
    assert "insufficient_improvement:improvement_ratio=0.000000" in report


def test_cli_all_module_invocation_succeeds_and_errors_report_to_stderr(tmp_path):
    success_output = tmp_path / "cli-success"
    success = _run_module(
        "all",
        "--output",
        str(success_output),
        "--days",
        "14",
        "--plants",
        "2",
        "--seed",
        "13",
    )

    assert success.returncode == 0, success.stderr + success.stdout
    assert "report:" in success.stdout
    assert (success_output / "performance_report.md").exists()

    failure = _run_module(
        "legacy",
        "--output",
        str(tmp_path / "cli-failure"),
        "--dataset",
        str(tmp_path / "does-not-exist.csv"),
    )

    assert failure.returncode == 2
    assert failure.stderr.startswith("ERROR:")


def test_cli_aidm_rejected_decision_exits_two_after_manifest_and_report(tmp_path):
    output = tmp_path / "aidm-reject"
    dataset = cli.run_generate_data(output, days=4, plants=1, seed=17)

    rejected = _run_module(
        "aidm",
        "--output",
        str(output),
        "--dataset",
        str(dataset),
        "--folds",
        "1",
        "--minimum-improvement",
        "1.0",
        "--max-plant-regression",
        "0.2",
        "--top-single-candidates",
        "1",
        "--seed",
        "17",
    )

    assert rejected.returncode == 2
    assert "ERROR: AIDM rejected promotion" in rejected.stderr
    assert f"manifest: {output / 'promotion_manifest.json'}" in rejected.stdout
    assert f"report: {output / 'performance_report.md'}" in rejected.stdout
    assert "decision: reject" in rejected.stdout
    manifest = json.loads((output / "promotion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision"] == "reject"
    assert (output / "performance_report.md").exists()


def _run_module(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src")
        if not env.get("PYTHONPATH")
        else str(ROOT / "src") + os.pathsep + env["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "-m", "power_forecasting.cli", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )


def _legacy_results():
    return {
        name: _evaluation(0.01 * (index + 1))
        for index, name in enumerate(LEGACY_NAMES)
    }


def _evaluation(nmae):
    return EvaluationResult(
        metrics={"MAE": nmae * 10.0, "RMSE": nmae * 12.0, "NMAE": nmae},
        per_plant={
            "plant_01": {"MAE": nmae * 10.0, "RMSE": nmae * 12.0, "NMAE": nmae}
        },
        fold_metrics=[{"MAE": nmae * 10.0, "RMSE": nmae * 12.0, "NMAE": nmae}],
        predictions=pd.DataFrame({"prediction": [1.0]}),
    )


def _aidm_result(
    *,
    candidate_name="hour_sin",
    decision="promote",
    failed_gates=(),
    valid_for_aidd=True,
):
    specs = (
        FeatureSpec(
            "hour_sin",
            "cyclic_hour",
            ("timestamp",),
            rationale="daily | phase",
        ),
    )
    winner_name = "hour_sin" if valid_for_aidd else candidate_name
    baseline = _candidate("baseline", (), 0.2)
    winner = _candidate(winner_name, specs, 0.1)
    other = _candidate("other_candidate", (), 0.11)
    manifest = {
        "schema_version": "1",
        "seed": 42,
        "baseline": {
            "model": "SPOT",
            "metrics": {"MAE": 2.0, "RMSE": 2.4, "NMAE": 0.2},
            "run_id": "baseline-run",
        },
        "winner": {
            "name": winner.name,
            "metrics": {"MAE": 1.0, "RMSE": 1.2, "NMAE": 0.1},
            "run_id": "winner-run",
        },
        "selected_specs": [spec.to_dict() for spec in specs],
        "per_plant_deltas": {"plant|01": -0.02},
        "thresholds": {
            "minimum_improvement": 0.01,
            "max_plant_regression": 0.03,
        },
        "improvement_ratio": 0.5,
        "decision": decision,
        "failed_gates": list(failed_gates),
    }
    return aidm.AIDMResult(
        baseline=baseline,
        candidates=(winner, other),
        ranking=(winner.name, other.name),
        winner=winner,
        manifest=manifest,
    )


def _candidate(name, specs, nmae):
    return aidm.CandidateResult(
        name=name,
        specs=tuple(specs),
        metrics={"MAE": nmae * 10.0, "RMSE": nmae * 12.0, "NMAE": nmae},
        per_plant={
            "plant|01": {"MAE": nmae * 10.0, "RMSE": nmae * 12.0, "NMAE": nmae}
        },
        run_id=f"{name}-run",
    )
