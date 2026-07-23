from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from power_forecasting import aidd, aidm, cli, reporting
from power_forecasting.data import DataContractError
from power_forecasting.evaluation import EvaluationResult
from power_forecasting.features import FeatureSpec
from power_forecasting.models import SUPPORTED_MODEL_NAMES
from power_forecasting.proposals import ResearchProposal


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

    proposal = ROOT / ".agents" / "fixtures" / "research-proposal.json"
    legacy_predictions = ROOT / ".agents" / "fixtures" / "legacy-predictions.csv"

    def fake_run_aidm(frame, database_path, config, proposal=None, legacy_predictions=None):
        captured["database_path"] = Path(database_path)
        captured["aidm_timestamp"] = pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
        captured["aidm_config"] = config
        captured["proposal"] = proposal
        captured["legacy_predictions"] = legacy_predictions
        return _aidm_result(valid_for_aidd=True)

    monkeypatch.setattr(cli, "run_aidm", fake_run_aidm)
    result = cli.run_aidm_workflow(
        tmp_path, dataset=dataset, config=aidm.AIDMConfig(folds=2)
    )
    assert result.manifest["decision"] == "promote"
    assert captured["database_path"] == tmp_path / "experiments.db"
    assert captured["aidm_timestamp"] is True
    assert captured["aidm_config"].folds == 2

    cli.run_aidm_workflow(
        tmp_path,
        dataset=dataset,
        config=aidm.AIDMConfig(folds=1),
        proposal=proposal,
        legacy_predictions=legacy_predictions,
    )
    assert captured["proposal"] == proposal
    assert captured["legacy_predictions"] == legacy_predictions

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


def test_run_all_defaults_match_standalone_aidm_config(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "run_legacy", lambda output, dataset=None, folds=3: _legacy_results())

    def fake_run_aidm_workflow(output, dataset=None, config=aidm.AIDMConfig()):
        captured["config"] = config
        return _aidm_result(valid_for_aidd=True)

    monkeypatch.setattr(cli, "run_aidm_workflow", fake_run_aidm_workflow)
    monkeypatch.setattr(cli, "run_aidd_workflow", _fake_run_aidd_workflow)

    paths = cli.run_all(tmp_path, days=2, plants=1, seed=7, folds=2)

    assert captured["config"] == aidm.AIDMConfig(folds=2, seed=7)
    assert paths["generated_module"].exists()


def test_run_all_accepts_explicit_gate_overrides(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "run_legacy", lambda output, dataset=None, folds=3: _legacy_results())

    def fake_run_aidm_workflow(output, dataset=None, config=aidm.AIDMConfig()):
        captured["config"] = config
        return _aidm_result(valid_for_aidd=True)

    monkeypatch.setattr(cli, "run_aidm_workflow", fake_run_aidm_workflow)
    monkeypatch.setattr(cli, "run_aidd_workflow", _fake_run_aidd_workflow)

    cli.run_all(
        tmp_path,
        days=2,
        plants=1,
        seed=11,
        folds=1,
        minimum_improvement=0.0,
        max_plant_regression=0.2,
    )

    assert captured["config"] == aidm.AIDMConfig(
        folds=1,
        seed=11,
        minimum_improvement=0.0,
        max_plant_regression=0.2,
    )


def test_cli_all_passes_explicit_gate_overrides_to_run_all(tmp_path, monkeypatch):
    captured = {}

    def fake_run_all(
        output,
        *,
        days=60,
        plants=3,
        seed=42,
        folds=3,
        minimum_improvement=aidm.AIDMConfig().minimum_improvement,
        max_plant_regression=aidm.AIDMConfig().max_plant_regression,
    ):
        captured.update(
            {
                "output": output,
                "days": days,
                "plants": plants,
                "seed": seed,
                "folds": folds,
                "minimum_improvement": minimum_improvement,
                "max_plant_regression": max_plant_regression,
            }
        )
        return {"report": Path(output) / "performance_report.md"}

    monkeypatch.setattr(cli, "run_all", fake_run_all)
    status = cli.main(
        [
            "all",
            "--output",
            str(tmp_path),
            "--days",
            "2",
            "--plants",
            "1",
            "--seed",
            "11",
            "--folds",
            "1",
            "--minimum-improvement",
            "0.0",
            "--max-plant-regression",
            "0.2",
        ]
    )

    assert status == 0
    assert captured == {
        "output": tmp_path,
        "days": 2,
        "plants": 1,
        "seed": 11,
        "folds": 1,
        "minimum_improvement": 0.0,
        "max_plant_regression": 0.2,
    }


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
    assert "generated_module" not in report
    assert "promoted_features.py" not in report


def test_run_all_success_report_is_written_after_generated_module_exists(
    tmp_path, monkeypatch
):
    observed = {}
    original_write = cli.write_performance_report
    monkeypatch.setattr(cli, "run_legacy", lambda output, dataset=None, folds=3: _legacy_results())
    monkeypatch.setattr(
        cli,
        "run_aidm_workflow",
        lambda output, dataset=None, config=aidm.AIDMConfig(): _aidm_result(
            valid_for_aidd=True
        ),
    )
    monkeypatch.setattr(cli, "run_aidd_workflow", _fake_run_aidd_workflow)

    def assert_generated_exists(
        dataset_summary,
        legacy_results,
        aidm_result,
        artifact_paths,
        *,
        target,
    ):
        generated = Path(artifact_paths["generated_module"])
        assert generated.exists()
        observed["artifact_paths"] = dict(artifact_paths)
        return original_write(
            dataset_summary,
            legacy_results,
            aidm_result,
            artifact_paths,
            target=target,
        )

    monkeypatch.setattr(cli, "write_performance_report", assert_generated_exists)

    paths = cli.run_all(tmp_path, days=2, plants=1, seed=3, folds=1)

    assert observed["artifact_paths"]["generated_module"] == paths["generated_module"]


def test_run_all_codegen_failure_report_does_not_claim_generated_artifact(
    tmp_path, monkeypatch
):
    (tmp_path / "generated").write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_legacy", lambda output, dataset=None, folds=3: _legacy_results())
    monkeypatch.setattr(
        cli,
        "run_aidm_workflow",
        lambda output, dataset=None, config=aidm.AIDMConfig(): _aidm_result(
            valid_for_aidd=True
        ),
    )

    with pytest.raises(OSError):
        cli.run_all(tmp_path, days=2, plants=1, seed=3, folds=1)

    report = (tmp_path / "performance_report.md").read_text(encoding="utf-8")
    assert "AIDD generation failed" in report
    assert "generated_module" not in report
    assert "promoted_features.py" not in report


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


def test_cli_aidm_proposal_requires_catalog(tmp_path, monkeypatch, capsys):
    def unexpected_legacy(*args, **kwargs):
        raise AssertionError("proposal without a catalog must fail before AIDM runs")

    monkeypatch.setattr(cli, "run_legacy", unexpected_legacy)

    status = cli.main(
        [
            "aidm",
            "--output",
            str(tmp_path / "aidm-missing-catalog"),
            "--dataset",
            str(ROOT / ".agents" / "fixtures" / "valid-dataset.csv"),
            "--proposal",
            str(ROOT / ".agents" / "fixtures" / "research-proposal.json"),
        ]
    )

    assert status == 2
    assert "--catalog is required when --proposal is supplied" in capsys.readouterr().err


def test_cli_aidm_catalog_validates_and_forwards_proposal(tmp_path, monkeypatch):
    captured = {}

    monkeypatch.setattr(cli, "run_legacy", lambda *args, **kwargs: _legacy_results())

    def fake_run_aidm_workflow(
        output, dataset=None, config=aidm.AIDMConfig(), *, proposal=None, legacy_predictions=None
    ):
        captured["proposal"] = proposal
        return _aidm_result()

    monkeypatch.setattr(cli, "run_aidm_workflow", fake_run_aidm_workflow)

    status = cli.main(
        [
            "aidm",
            "--output",
            str(tmp_path / "aidm-valid-catalog"),
            "--dataset",
            str(ROOT / ".agents" / "fixtures" / "valid-dataset.csv"),
            "--proposal",
            str(ROOT / ".agents" / "fixtures" / "research-proposal.json"),
            "--catalog",
            "configs/optimization-catalog.v1.json",
        ]
    )

    assert status == 0
    assert isinstance(captured["proposal"], ResearchProposal)
    assert captured["proposal"].proposal_id == "fixture-agentic-proposal"


def test_cli_aidm_rejects_estimator_valid_recipe_disallowed_by_catalog(
    tmp_path, monkeypatch, capsys
):
    catalog_dir = ROOT / "outputs" / f"pytest-cli-catalog-{tmp_path.name}"
    catalog_path = catalog_dir / "catalog.json"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_payload = json.loads(
        (ROOT / "configs" / "optimization-catalog.v1.json").read_text(encoding="utf-8")
    )
    catalog_payload["recipes"]["ridge_weather"]["parameters"]["alpha"] = 0.1
    catalog_payload["recipes"]["ridge_weather"]["allowed_parameters"]["alpha"] = [0.1]
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")

    def unexpected_legacy(*args, **kwargs):
        raise AssertionError("catalog-invalid proposal must fail before AIDM runs")

    monkeypatch.setattr(cli, "run_legacy", unexpected_legacy)
    try:
        status = cli.main(
            [
                "aidm",
                "--output",
                str(tmp_path / "aidm-catalog-rejected"),
                "--dataset",
                str(ROOT / ".agents" / "fixtures" / "valid-dataset.csv"),
                "--proposal",
                str(ROOT / ".agents" / "fixtures" / "research-proposal.json"),
                "--catalog",
                str(catalog_path.relative_to(ROOT)),
            ]
        )
    finally:
        catalog_path.unlink(missing_ok=True)
        catalog_dir.rmdir()

    assert status == 2
    assert "model recipe ridge_low contains a value outside catalog policy" in capsys.readouterr().err


def test_cli_aidm_model_search_fixture_runs_with_mocked_optional_model_imports(
    tmp_path, monkeypatch
):
    _install_fake_model_search_modules(monkeypatch)

    def fake_evaluate(frame, definition, feature_specs, folds):
        definition.estimator_factory()
        score = 0.30 if not feature_specs else 0.10
        validation = frame.tail(4).copy()
        prediction = (
            validation["generation_mw"] + score * validation["capacity_mw"]
        ).clip(0, validation["capacity_mw"])
        return EvaluationResult(
            metrics={"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
            per_plant={
                str(plant_id): {
                    "MAE": score * 100.0,
                    "RMSE": score * 120.0,
                    "NMAE": score,
                }
                for plant_id in sorted(frame["plant_id"].unique())
            },
            fold_metrics=[
                {"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score}
            ],
            predictions=pd.DataFrame(
                {
                    "timestamp": validation["timestamp"].to_numpy(),
                    "plant_id": validation["plant_id"].to_numpy(),
                    "actual": validation["generation_mw"].to_numpy(),
                    "prediction": prediction.to_numpy(),
                    "capacity_mw": validation["capacity_mw"].to_numpy(),
                    "fold": 1,
                }
            ),
        )

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    output = tmp_path / "model-search"
    status = cli.main(
        [
            "aidm",
            "--output",
            str(output),
            "--dataset",
            str(ROOT / ".agents" / "fixtures" / "valid-dataset.csv"),
            "--proposal",
            str(ROOT / ".agents" / "fixtures" / "model-search-proposal.json"),
            "--catalog",
            str(ROOT / "configs" / "optimization-catalog.v1.json"),
            "--folds",
            "1",
            "--minimum-improvement",
            "0",
            "--max-plant-regression",
            "1",
            "--seed",
            "7",
        ]
    )

    assert status == 0
    manifest = json.loads((output / "promotion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["proposal"]["proposal_id"] == "fixture-model-search-proposal"
    assert manifest["selected_model_recipe"]["recipe"] in {
        "random_forest",
        "xgboost",
        "lightgbm",
    }
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


def _install_fake_model_search_modules(monkeypatch):
    xgboost = ModuleType("xgboost")

    class FakeXGBRegressor:
        def __init__(self, **parameters):
            self.parameters = dict(parameters)

    lightgbm = ModuleType("lightgbm")

    class FakeLGBMRegressor:
        def __init__(self, **parameters):
            self.parameters = dict(parameters)

    optuna = ModuleType("optuna")
    samplers = ModuleType("optuna.samplers")
    pruners = ModuleType("optuna.pruners")

    class TPESampler:
        def __init__(self, *, seed):
            self.seed = seed

    class NopPruner:
        pass

    class Trial:
        def __init__(self, number):
            self.number = number
            self.params = {}

        def suggest_categorical(self, name, choices):
            value = choices[self.number % len(choices)]
            self.params[name] = value
            return value

    class Study:
        def __init__(self, *, sampler, pruner, direction):
            self.sampler = sampler
            self.pruner = pruner
            self.direction = direction
            self.trials = []

        def optimize(self, objective, n_trials):
            for number in range(n_trials):
                trial = Trial(number)
                trial.value = objective(trial)
                self.trials.append(trial)

    def create_study(*, sampler, pruner, direction):
        return Study(sampler=sampler, pruner=pruner, direction=direction)

    xgboost.XGBRegressor = FakeXGBRegressor
    lightgbm.LGBMRegressor = FakeLGBMRegressor
    samplers.TPESampler = TPESampler
    pruners.NopPruner = NopPruner
    optuna.samplers = samplers
    optuna.pruners = pruners
    optuna.create_study = create_study
    monkeypatch.setitem(sys.modules, "xgboost", xgboost)
    monkeypatch.setitem(sys.modules, "lightgbm", lightgbm)
    monkeypatch.setitem(sys.modules, "optuna", optuna)
    monkeypatch.setitem(sys.modules, "optuna.samplers", samplers)
    monkeypatch.setitem(sys.modules, "optuna.pruners", pruners)


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


def _fake_run_aidd_workflow(output, manifest):
    path = Path(output) / "generated" / "promoted_features.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# generated\n", encoding="utf-8")
    return path


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
