from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from power_forecasting import aidm
from power_forecasting.catalogs import load_optimization_catalog
from power_forecasting.evaluation import EvaluationResult
from power_forecasting.models import SUPPORTED_MODEL_NAMES
from power_forecasting.proposals import load_proposal, proposal_to_dict
from power_forecasting.research_contracts import ResearchLoopConfig
from power_forecasting.research_execution import (
    CHECKSUM_UNAVAILABLE,
    ResearchExecutionError,
    VerificationResult,
    run_experiment_agent,
    run_verifier_agent,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DATASET = REPOSITORY_ROOT / ".agents" / "fixtures" / "valid-dataset.csv"
FIXTURE_MANIFEST = REPOSITORY_ROOT / ".agents" / "fixtures" / "promoted-manifest.json"
FIXTURE_PROPOSAL = REPOSITORY_ROOT / ".agents" / "fixtures" / "research-proposal.json"
FIXTURE_CATALOG = REPOSITORY_ROOT / "configs" / "optimization-catalog.v1.json"


@pytest.fixture
def execution_config(tmp_path: Path) -> ResearchLoopConfig:
    catalog = load_optimization_catalog(FIXTURE_CATALOG, repository_root=REPOSITORY_ROOT)
    return ResearchLoopConfig(
        schema_version="1",
        run_id="research_execution_001",
        dataset_path=str(FIXTURE_DATASET),
        legacy_manifest_path=str(FIXTURE_MANIFEST),
        catalog_path=str(catalog.source_path),
        catalog_sha256=catalog.sha256,
        catalog=catalog,
        run_dir=str(tmp_path / "research-runs" / "research_execution_001"),
        profiles=("safe_weather",),
        max_iterations=2,
        fold_count=1,
        objective="nmae",
        minimum_improvement=0.0,
        max_plant_regression=1.0,
    )


@pytest.fixture
def proposal():
    return load_proposal(FIXTURE_PROPOSAL)


def _evaluation(frame: pd.DataFrame, score: float) -> EvaluationResult:
    validation = frame.tail(4).copy()
    predictions = (
        validation["generation_mw"] + score * validation["capacity_mw"]
    ).clip(0.0, validation["capacity_mw"])
    metrics = {"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score}
    return EvaluationResult(
        metrics=metrics,
        per_plant={
            str(plant_id): dict(metrics)
            for plant_id in sorted(validation["plant_id"].unique())
        },
        fold_metrics=[dict(metrics)],
        predictions=pd.DataFrame(
            {
                "timestamp": validation["timestamp"].to_numpy(),
                "plant_id": validation["plant_id"].to_numpy(),
                "actual": validation["generation_mw"].to_numpy(),
                "prediction": predictions.to_numpy(),
                "capacity_mw": validation["capacity_mw"].to_numpy(),
                "fold": 1,
            }
        ),
    )


def _install_fast_aidm(monkeypatch: pytest.MonkeyPatch, *, candidate_score: float) -> None:
    def fake_evaluate(
        frame: pd.DataFrame,
        definition,
        feature_specs,
        folds: int,
    ) -> EvaluationResult:
        del definition, folds
        return _evaluation(frame, 0.20 if not feature_specs else candidate_score)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)


def _legacy_results() -> dict[str, EvaluationResult]:
    frame = pd.read_csv(FIXTURE_DATASET)
    return {
        name: _evaluation(frame, 0.25)
        for name in SUPPORTED_MODEL_NAMES
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _update_evidence_checksum(experiment, key: str, artifact: Path) -> None:
    evidence_path = experiment.manifest_path.parent / "experiment-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence[key] = _sha256_file(artifact)
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _run_fast_experiment(
    config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_score: float = 0.10,
):
    import power_forecasting.research_execution as execution

    _install_fast_aidm(monkeypatch, candidate_score=candidate_score)
    monkeypatch.setattr(execution, "run_legacy", lambda *args, **kwargs: _legacy_results())
    return run_experiment_agent(config=config, proposal=proposal, iteration=1)


def _run_fast_search_experiment(
    config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = proposal_to_dict(proposal)
    payload["budget"]["max_evaluations"] = 5
    payload["search"] = {
        "sampler": "tpe",
        "seed": 7,
        "n_trials": 2,
        "spaces": {
            "lightgbm": {
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "num_leaves": [15, 31],
                "min_child_samples": [10, 20],
            }
        },
    }
    search_proposal = load_proposal(payload)
    _install_fake_optuna(monkeypatch)

    def fake_evaluate(frame: pd.DataFrame, definition, feature_specs, folds: int):
        del feature_specs, folds
        if definition.name == "SPOT":
            return _evaluation(frame, 0.20)
        if definition.name == "Recipe:lightgbm:selected_lightgbm":
            return _evaluation(frame, 0.05)
        if definition.name.startswith("Recipe:lightgbm:optuna_lightgbm_"):
            return _evaluation(frame, 0.08)
        return _evaluation(frame, 0.10)

    import power_forecasting.research_execution as execution

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(execution, "run_legacy", lambda *args, **kwargs: _legacy_results())
    experiment = run_experiment_agent(
        config=config,
        proposal=search_proposal,
        iteration=1,
    )
    return search_proposal, experiment


def _install_fake_optuna(monkeypatch: pytest.MonkeyPatch) -> None:
    class TPESampler:
        def __init__(self, *, seed: int):
            self.seed = seed

    class NopPruner:
        pass

    class Trial:
        def __init__(self, number: int):
            self.number = number

        def suggest_categorical(self, _name: str, choices: list[object]) -> object:
            return choices[self.number % len(choices)]

    class Study:
        def optimize(self, objective, n_trials: int) -> None:
            for number in range(n_trials):
                objective(Trial(number))

    fake_optuna = SimpleNamespace(
        samplers=SimpleNamespace(TPESampler=TPESampler),
        pruners=SimpleNamespace(NopPruner=NopPruner),
        create_study=lambda **_kwargs: Study(),
    )
    monkeypatch.setattr(aidm, "_import_optuna", lambda: fake_optuna)


def test_experiment_agent_delegates_and_records_isolated_provenance(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    import power_forecasting.research_execution as execution

    _install_fast_aidm(monkeypatch, candidate_score=0.10)
    calls = []
    workflow = execution.run_aidm_workflow

    def spy_workflow(output, dataset=None, config=None, **kwargs):
        calls.append((Path(output), Path(dataset), config, kwargs))
        return workflow(output, dataset=dataset, config=config, **kwargs)

    monkeypatch.setattr(execution, "run_aidm_workflow", spy_workflow)
    monkeypatch.setattr(execution, "run_legacy", lambda *args, **kwargs: _legacy_results())

    result = run_experiment_agent(
        config=execution_config,
        proposal=proposal,
        iteration=1,
    )

    assert len(calls) == 1
    output, dataset, aidm_config, kwargs = calls[0]
    assert dataset == FIXTURE_DATASET
    assert aidm_config.folds == execution_config.fold_count
    assert aidm_config.minimum_improvement == execution_config.minimum_improvement
    assert aidm_config.max_plant_regression == execution_config.max_plant_regression
    assert kwargs["proposal"] == proposal
    assert result.run_id == execution_config.run_id
    assert result.run_state == "promoted"
    assert result.manifest_path == output / "promotion_manifest.json"
    assert result.report_path == output / "performance_report.md"
    assert result.manifest_path.is_file()
    assert result.report_path.is_file()
    assert (output / "experiments.db").is_file()
    assert (output / "research-proposal.json").is_file()
    assert (output / "experiment-evidence.json").is_file()
    assert result.manifest_path.parent.is_relative_to(Path(execution_config.run_dir))

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.selected_candidate_id == manifest["winner"]["name"]
    assert result.selected_recipe_id == manifest["selected_model_recipe"]["name"]
    assert result.selected_candidate_spec_sha256


def test_verifier_confirms_promoted_aidm_evidence(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert isinstance(verification, VerificationResult)
    assert verification.passed is True
    assert verification.checks
    assert all(verification.checks.values())
    assert verification.reasons == ()
    assert verification.report_path == experiment.manifest_path.parent / "verification.json"
    payload = json.loads(verification.report_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["checks"] == dict(verification.checks)
    assert payload["reasons"] == []


def test_verifier_returns_rejected_experiment_as_normal_outcome(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    rejected_config = replace(execution_config, minimum_improvement=0.9)
    experiment = _run_fast_experiment(rejected_config, proposal, monkeypatch)

    verification = run_verifier_agent(
        config=rejected_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert experiment.run_state == "rejected"
    assert verification.passed is False
    assert verification.checks["promoted"] is False
    assert all(
        passed
        for name, passed in verification.checks.items()
        if name != "promoted"
    )
    assert verification.reasons == ("experiment_rejected",)
    payload = json.loads(verification.report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "reject"
    assert payload["passed"] is False


def test_experiment_identity_is_deterministic_from_run_iteration_and_proposal(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    first = _run_fast_experiment(execution_config, proposal, monkeypatch)
    second_config = replace(
        execution_config,
        run_dir=str(tmp_path / "research-runs" / "second-location"),
    )
    second = _run_fast_experiment(second_config, proposal, monkeypatch)
    changed_iteration = run_experiment_agent(
        config=second_config,
        proposal=proposal,
        iteration=2,
    )

    assert first.experiment_id == second.experiment_id
    assert first.experiment_id != changed_iteration.experiment_id
    assert first.manifest_path.parent.name.endswith(first.experiment_id[:16])


@pytest.mark.parametrize(
    ("artifact", "expected_check"),
    [
        ("manifest", "thresholds"),
        ("report", "report_checksum"),
        ("checksum", "evidence_schema"),
        ("evidence_decision", "experiment_identity"),
        ("proposal", "proposal_checksum"),
        ("winner", "selected_candidate"),
        ("recipe", "selected_recipe"),
        ("sqlite", "sqlite_runs"),
    ],
)
def test_verifier_detects_tampered_evidence(
    artifact: str,
    expected_check: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    iteration_dir = experiment.manifest_path.parent

    if artifact == "manifest":
        manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
        manifest["thresholds"]["minimum_improvement"] = 0.4
        experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)
    elif artifact == "report":
        experiment.report_path.write_text("tampered report\n", encoding="utf-8")
    elif artifact == "checksum":
        evidence_path = iteration_dir / "experiment-evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["proposal_sha256"] = "0" * 64
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    elif artifact == "evidence_decision":
        evidence_path = iteration_dir / "experiment-evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["decision"] = "reject"
        evidence["run_state"] = "rejected"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    elif artifact == "proposal":
        proposal_path = iteration_dir / "research-proposal.json"
        payload = json.loads(proposal_path.read_text(encoding="utf-8"))
        payload["rationale"] = "Different bounded proposal rationale."
        proposal_path.write_text(json.dumps(payload), encoding="utf-8")
    elif artifact == "winner":
        manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
        manifest["winner"]["name"] = "ridge_low:unrelated_feature_set"
        experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)
    elif artifact == "recipe":
        manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
        manifest["selected_model_recipe"]["parameters"]["alpha"] = 10.0
        experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)
    else:
        evidence_path = iteration_dir / "experiment-evidence.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        with sqlite3.connect(iteration_dir / "experiments.db") as connection:
            connection.execute(
                "UPDATE runs SET status = 'failed' WHERE id = ?",
                (evidence["selected_candidate"]["database_run_id"],),
            )
        _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks[expected_check] is False


@pytest.mark.parametrize("artifact", ("manifest", "proposal", "evidence"))
def test_verifier_fails_closed_on_malformed_artifacts(
    artifact: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    paths = {
        "manifest": experiment.manifest_path,
        "proposal": experiment.manifest_path.parent / "research-proposal.json",
        "evidence": experiment.manifest_path.parent / "experiment-evidence.json",
    }
    paths[artifact].write_text("{", encoding="utf-8")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.reasons
    assert verification.report_path.is_file()


def test_verifier_invalid_report_has_complete_unavailable_provenance(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    experiment.report_path.unlink()

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    payload = json.loads(verification.report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "invalid"
    assert set(payload["provenance"]) == {
        "proposal_sha256",
        "manifest_sha256",
        "report_sha256",
        "database_sha256",
    }
    assert payload["provenance"]["report_sha256"] == CHECKSUM_UNAVAILABLE
    assert payload["provenance"]["manifest_sha256"] == _sha256_file(
        experiment.manifest_path
    )
    assert payload["provenance"]["database_sha256"] == _sha256_file(
        experiment.manifest_path.parent / "experiments.db"
    )


def test_verifier_report_contains_only_aggregate_provenance(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RESEARCH_EXECUTION_TEST_SECRET", "do-not-log-this-secret")
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    payload = json.loads(verification.report_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, sort_keys=True)
    assert set(payload) == {
        "schema_version",
        "status",
        "passed",
        "checks",
        "reasons",
        "provenance",
    }
    assert "plant-a" not in serialized
    assert "RESEARCH_EXECUTION_TEST_SECRET" not in serialized
    assert "do-not-log-this-secret" not in serialized
    assert str(FIXTURE_DATASET) not in serialized


def test_experiment_agent_enforces_configured_iteration_budget(
    execution_config: ResearchLoopConfig,
    proposal,
):
    with pytest.raises(ValueError, match="max_iterations"):
        run_experiment_agent(
            config=execution_config,
            proposal=proposal,
            iteration=execution_config.max_iterations + 1,
        )

    assert not Path(execution_config.run_dir).exists()


def test_verifier_fails_closed_when_recipe_artifact_contains_nonfinite_json(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["selected_model_recipe"]["parameters"]["alpha"] = float("nan")
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.reasons


def test_verifier_fails_closed_on_overflowing_manifest_threshold(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["thresholds"]["minimum_improvement"] = 10**400
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["thresholds"] is False


def test_verifier_fails_closed_when_manifest_winner_is_null(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["winner"] = None
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.reasons


def test_experiment_agent_records_typed_workflow_failure_evidence(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    import power_forecasting.research_execution as execution

    monkeypatch.setattr(
        execution,
        "run_aidm_workflow",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ResearchExecutionError, match="required evidence"):
        run_experiment_agent(
            config=execution_config,
            proposal=proposal,
            iteration=1,
        )

    failures = list(Path(execution_config.run_dir).rglob("experiment-failure.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert set(failure) == {
        "schema_version",
        "run_id",
        "experiment_id",
        "iteration",
        "run_state",
    }
    assert failure["schema_version"] == "1"
    assert failure["run_id"] == execution_config.run_id
    assert len(failure["experiment_id"]) == 64
    assert failure["iteration"] == 1
    assert failure["run_state"] == "failed"


def test_verifier_does_not_rewrite_source_evidence(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    source_paths = (
        experiment.manifest_path,
        experiment.report_path,
        experiment.manifest_path.parent / "experiments.db",
        experiment.manifest_path.parent / "research-proposal.json",
        experiment.manifest_path.parent / "experiment-evidence.json",
    )
    before = {path: path.read_bytes() for path in source_paths}

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is True
    assert {path: path.read_bytes() for path in source_paths} == before


def test_verifier_rejects_experiment_result_with_wrong_gate_state(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    contradictory_experiment = replace(experiment, run_state="rejected")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=contradictory_experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["experiment_identity"] is False


def test_verifier_detects_baseline_provenance_mismatch(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["baseline"]["model"] = "untrusted_baseline"
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["baseline_provenance"] is False


def test_verifier_rejects_tampered_manifest_seed(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["seed"] = 99
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["seed_provenance"] is False
    assert "check_failed:seed_provenance" in verification.reasons


@pytest.mark.parametrize(
    ("role", "check_name"),
    (
        ("baseline", "baseline_provenance"),
        ("winner", "winner_provenance"),
    ),
)
def test_verifier_rejects_tampered_manifest_run_identity(
    role: str,
    check_name: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest[role]["run_id"] = f"tampered-{role}-run-id"
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks[check_name] is False
    assert f"check_failed:{check_name}" in verification.reasons


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("folds", 2),
        ("seed", 99),
        ("specs", [{"name": "unexpected"}]),
        ("model", "not-spot"),
    ),
)
def test_verifier_rejects_tampered_baseline_sqlite_parameters(
    parameter: str,
    value: object,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    iteration_dir = experiment.manifest_path.parent
    evidence = json.loads(
        (iteration_dir / "experiment-evidence.json").read_text(encoding="utf-8")
    )
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        row = connection.execute(
            "SELECT params_json FROM runs WHERE id = ?",
            (evidence["baseline_database_run_id"],),
        ).fetchone()
        assert row is not None
        params = json.loads(row[0])
        params[parameter] = value
        connection.execute(
            "UPDATE runs SET params_json = ? WHERE id = ?",
            (
                json.dumps(params),
                evidence["baseline_database_run_id"],
            ),
        )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["baseline_provenance"] is False
    assert "check_failed:baseline_provenance" in verification.reasons


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("schema_version", "tampered"),
        ("seed", 99),
        ("model", "Recipe:tampered:model"),
    ),
)
def test_verifier_rejects_tampered_selected_proposal_sqlite_parameters(
    parameter: str,
    value: object,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    iteration_dir = experiment.manifest_path.parent
    evidence = json.loads(
        (iteration_dir / "experiment-evidence.json").read_text(encoding="utf-8")
    )
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        row = connection.execute(
            "SELECT params_json FROM runs WHERE id = ?",
            (evidence["selected_candidate"]["database_run_id"],),
        ).fetchone()
        assert row is not None
        params = json.loads(row[0])
        params[parameter] = value
        connection.execute(
            "UPDATE runs SET params_json = ? WHERE id = ?",
            (
                json.dumps(params),
                evidence["selected_candidate"]["database_run_id"],
            ),
        )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["proposal_runs"] is False
    assert "check_failed:proposal_runs" in verification.reasons


@pytest.mark.parametrize(
    "tamper",
    ("run_name", "unexpected_params_key", "unexpected_artifacts_key"),
)
def test_verifier_rejects_unexpected_selected_proposal_evidence(
    tamper: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    iteration_dir = experiment.manifest_path.parent
    evidence = json.loads(
        (iteration_dir / "experiment-evidence.json").read_text(encoding="utf-8")
    )
    selected_run_id = evidence["selected_candidate"]["database_run_id"]
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        row = connection.execute(
            "SELECT params_json, artifacts_json FROM runs WHERE id = ?",
            (selected_run_id,),
        ).fetchone()
        assert row is not None
        if tamper == "run_name":
            connection.execute(
                "UPDATE runs SET name = ? WHERE id = ?",
                ("aidm-proposal-tampered", selected_run_id),
            )
        elif tamper == "unexpected_params_key":
            params = json.loads(row[0])
            params["unexpected"] = "untrusted"
            connection.execute(
                "UPDATE runs SET params_json = ? WHERE id = ?",
                (json.dumps(params), selected_run_id),
            )
        else:
            artifacts = json.loads(row[1])
            artifacts["unexpected"] = "untrusted"
            connection.execute(
                "UPDATE runs SET artifacts_json = ? WHERE id = ?",
                (json.dumps(artifacts), selected_run_id),
            )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["proposal_runs"] is False
    assert "check_failed:proposal_runs" in verification.reasons


def test_verifier_rejects_unexpected_top_level_manifest_field(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = "untrusted"
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["manifest_schema"] is False
    assert "check_failed:manifest_schema" in verification.reasons


@pytest.mark.parametrize(
    "unexpected_content",
    ("untrusted report content\n", "| unexpected | untrusted |\n"),
)
def test_verifier_rejects_unexpected_report_content(
    unexpected_content: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    experiment.report_path.write_text(
        experiment.report_path.read_text(encoding="utf-8") + unexpected_content,
        encoding="utf-8",
    )
    _update_evidence_checksum(experiment, "report_sha256", experiment.report_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["report_evidence"] is False
    assert "check_failed:report_evidence" in verification.reasons


@pytest.mark.parametrize(
    ("row_prefix", "value_index", "replacement"),
    (
        ("| Rows |", 1, "999"),
        ("| Mean |", 1, "generation_mw: secret"),
        ("| Mean |", 1, "25.000001"),
        ("| 2 |", 2, "99.000000"),
        ("| plant-a |", 1, "0.000000"),
        ("| effective_irradiance |", 5, "generation_mw: secret"),
        ("| database |", 1, "generation_mw: secret"),
    ),
)
def test_verifier_rejects_tampered_allowed_report_row_values(
    row_prefix: str,
    value_index: int,
    replacement: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    lines = experiment.report_path.read_text(encoding="utf-8").splitlines()
    row_index = next(
        index for index, line in enumerate(lines) if line.startswith(row_prefix)
    )
    cells = lines[row_index][2:-2].split(" | ")
    cells[value_index] = replacement
    lines[row_index] = "| " + " | ".join(cells) + " |"
    experiment.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _update_evidence_checksum(experiment, "report_sha256", experiment.report_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["report_evidence"] is False
    assert "check_failed:report_evidence" in verification.reasons


def test_verifier_rejects_report_table_after_empty_section(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    report = experiment.report_path.read_text(encoding="utf-8")
    report = report.replace(
        "## Failed gates\n\nNone\n\n## Selected feature specs",
        (
            "## Failed gates\n\nNone\n| Gate |\n| --- |\n"
            "| generation_mw: secret |\n\n## Selected feature specs"
        ),
    )
    experiment.report_path.write_text(report, encoding="utf-8")
    _update_evidence_checksum(experiment, "report_sha256", experiment.report_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["report_evidence"] is False
    assert "check_failed:report_evidence" in verification.reasons


def test_verifier_preserves_canonical_ranking_with_close_displayed_metrics(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_evaluate(frame: pd.DataFrame, definition, feature_specs, folds: int):
        del feature_specs, folds
        if definition.name == "SPOT":
            return _evaluation(frame, 0.20)
        if definition.name == "Recipe:ridge:ridge_low":
            return _evaluation(frame, 0.1000003)
        return _evaluation(frame, 0.1000004)

    import power_forecasting.research_execution as execution

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(execution, "run_legacy", lambda *args, **kwargs: _legacy_results())
    experiment = run_experiment_agent(
        config=execution_config,
        proposal=proposal,
        iteration=1,
    )

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is True


@pytest.mark.parametrize(
    ("target", "check_name"),
    (
        ("baseline", "manifest_schema"),
        ("legacy_baseline", "manifest_schema"),
        ("baseline_summary", "baseline_provenance"),
        ("selected_summary", "proposal_runs"),
    ),
)
def test_verifier_rejects_unexpected_nested_evidence_field(
    target: str,
    check_name: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    iteration_dir = experiment.manifest_path.parent
    if target in {"baseline", "legacy_baseline"}:
        manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
        if target == "baseline":
            manifest["baseline"]["unexpected"] = "untrusted"
        else:
            manifest["legacy_baseline"] = {"unexpected": "untrusted"}
        experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)
    else:
        evidence = json.loads(
            (iteration_dir / "experiment-evidence.json").read_text(encoding="utf-8")
        )
        run_id = (
            evidence["baseline_database_run_id"]
            if target == "baseline_summary"
            else evidence["selected_candidate"]["database_run_id"]
        )
        with sqlite3.connect(iteration_dir / "experiments.db") as connection:
            row = connection.execute(
                "SELECT artifacts_json FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            assert row is not None
            artifacts = json.loads(row[0])
            artifacts["summary"]["unexpected"] = "untrusted"
            connection.execute(
                "UPDATE runs SET artifacts_json = ? WHERE id = ?",
                (json.dumps(artifacts), run_id),
            )
        _update_evidence_checksum(
            experiment,
            "database_sha256",
            iteration_dir / "experiments.db",
        )

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks[check_name] is False
    assert f"check_failed:{check_name}" in verification.reasons


def test_verifier_binds_report_content_to_manifest_evidence(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    experiment.report_path.write_text(
        "\n".join(
            [
                "# Forecasting Performance Report",
                "",
                "Promotion decision: promote",
                "",
                "### Thresholds",
                "",
                "## Selected feature specs",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _update_evidence_checksum(experiment, "report_sha256", experiment.report_path)

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["report_evidence"] is False


def test_verifier_rejects_incomplete_nonselected_proposal_run(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    iteration_dir = experiment.manifest_path.parent
    evidence = json.loads(
        (iteration_dir / "experiment-evidence.json").read_text(encoding="utf-8")
    )
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        rows = connection.execute("SELECT id, params_json FROM runs").fetchall()
        nonselected_run_id = next(
            run_id
            for run_id, params_json in rows
            if json.loads(params_json).get("proposal_id") == proposal.proposal_id
            and run_id != evidence["selected_candidate"]["database_run_id"]
        )
        connection.execute(
            "UPDATE runs SET status = 'failed' WHERE id = ?",
            (nonselected_run_id,),
        )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["proposal_runs"] is False


def test_verifier_crosschecks_persisted_run_metrics(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    experiment = _run_fast_experiment(execution_config, proposal, monkeypatch)
    iteration_dir = experiment.manifest_path.parent
    evidence = json.loads(
        (iteration_dir / "experiment-evidence.json").read_text(encoding="utf-8")
    )
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        connection.execute(
            "UPDATE runs SET metrics_json = ? WHERE id = ?",
            (
                json.dumps({"mae": 99.0, "rmse": 99.0, "nmae": 0.99}),
                evidence["selected_candidate"]["database_run_id"],
            ),
        )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["metrics_provenance"] is False


def test_verifier_validates_bounded_search_provenance_without_optional_packages(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    payload = proposal_to_dict(proposal)
    payload["budget"]["max_evaluations"] = 5
    payload["search"] = {
        "sampler": "tpe",
        "seed": 7,
        "n_trials": 2,
        "spaces": {
            "lightgbm": {
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "num_leaves": [15, 31],
                "min_child_samples": [10, 20],
            }
        },
    }
    search_proposal = load_proposal(payload)
    _install_fake_optuna(monkeypatch)

    def fake_evaluate(frame: pd.DataFrame, definition, feature_specs, folds: int):
        del feature_specs, folds
        if definition.name == "SPOT":
            return _evaluation(frame, 0.20)
        if definition.name == "Recipe:lightgbm:selected_lightgbm":
            return _evaluation(frame, 0.05)
        if definition.name.startswith("Recipe:lightgbm:optuna_lightgbm_"):
            return _evaluation(frame, 0.08)
        return _evaluation(frame, 0.10)

    import power_forecasting.research_execution as execution

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(execution, "run_legacy", lambda *args, **kwargs: _legacy_results())
    experiment = run_experiment_agent(
        config=execution_config,
        proposal=search_proposal,
        iteration=1,
    )

    verification = run_verifier_agent(
        config=execution_config,
        proposal=search_proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is True
    assert verification.checks["proposal_runs"] is True
    assert experiment.selected_recipe_id == "selected_lightgbm"

    iteration_dir = experiment.manifest_path.parent
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        row = connection.execute(
            """
            SELECT id, params_json
            FROM runs
            WHERE name = 'aidm-selected-lightgbm-safe_solar'
            """
        ).fetchone()
        assert row is not None
        run_id, params_json = row
        params = json.loads(params_json)
        params["search"]["n_trials"] = 3
        connection.execute(
            "UPDATE runs SET params_json = ? WHERE id = ?",
            (json.dumps(params), run_id),
        )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    tampered = run_verifier_agent(
        config=execution_config,
        proposal=search_proposal,
        experiment=experiment,
        iteration=1,
    )

    assert tampered.passed is False
    assert tampered.checks["proposal_runs"] is False


@pytest.mark.parametrize(
    "tamper",
    (
        "missing_candidate_name",
        "extra_field",
        "trial_number",
        "candidate_name",
        "run_id",
        "parameters",
    ),
)
def test_verifier_rejects_tampered_selected_trial_evidence(
    tamper: str,
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    search_proposal, experiment = _run_fast_search_experiment(
        execution_config,
        proposal,
        monkeypatch,
    )
    iteration_dir = experiment.manifest_path.parent
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        row = connection.execute(
            """
            SELECT id, artifacts_json
            FROM runs
            WHERE name = 'aidm-selected-lightgbm-safe_solar'
            """
        ).fetchone()
        assert row is not None
        run_id, artifacts_json = row
        artifacts = json.loads(artifacts_json)
        selected_from_trial = artifacts["selected_from_trial"]
        if tamper == "missing_candidate_name":
            selected_from_trial.pop("candidate_name")
        elif tamper == "extra_field":
            selected_from_trial["unexpected"] = "untrusted"
        elif tamper == "trial_number":
            selected_from_trial["trial_number"] = 99
        elif tamper == "candidate_name":
            selected_from_trial["candidate_name"] = "optuna_lightgbm_999:safe_solar"
        elif tamper == "run_id":
            selected_from_trial["run_id"] = "not-a-recorded-trial"
        else:
            selected_from_trial["parameters"] = {"n_estimators": 999}
        connection.execute(
            "UPDATE runs SET artifacts_json = ? WHERE id = ?",
            (json.dumps(artifacts), run_id),
        )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=search_proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["proposal_runs"] is False
    assert "check_failed:proposal_runs" in verification.reasons


def test_verifier_crosschecks_referenced_selected_trial_evidence(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    search_proposal, experiment = _run_fast_search_experiment(
        execution_config,
        proposal,
        monkeypatch,
    )
    iteration_dir = experiment.manifest_path.parent
    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        selected = connection.execute(
            """
            SELECT artifacts_json
            FROM runs
            WHERE name = 'aidm-selected-lightgbm-safe_solar'
            """
        ).fetchone()
        assert selected is not None
        selected_artifacts = json.loads(selected[0])
        trial_run_id = selected_artifacts["selected_from_trial"]["run_id"]
        trial = connection.execute(
            "SELECT params_json, artifacts_json FROM runs WHERE id = ?",
            (trial_run_id,),
        ).fetchone()
        assert trial is not None
        trial_params = json.loads(trial[0])
        trial_artifacts = json.loads(trial[1])
        trial_params["model_recipe"]["parameters"]["n_estimators"] = 999
        trial_artifacts["summary"]["model_recipe"] = trial_params["model_recipe"]
        connection.execute(
            "UPDATE runs SET params_json = ?, artifacts_json = ? WHERE id = ?",
            (json.dumps(trial_params), json.dumps(trial_artifacts), trial_run_id),
        )
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=execution_config,
        proposal=search_proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["proposal_runs"] is False
    assert "check_failed:proposal_runs" in verification.reasons


def test_verifier_rejects_out_of_space_search_evidence_for_rejected_experiment(
    execution_config: ResearchLoopConfig,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
):
    rejected_config = replace(execution_config, minimum_improvement=0.9)
    search_proposal, experiment = _run_fast_search_experiment(
        rejected_config,
        proposal,
        monkeypatch,
    )
    assert experiment.run_state == "rejected"
    iteration_dir = experiment.manifest_path.parent
    evidence_path = iteration_dir / "experiment-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    manifest = json.loads(experiment.manifest_path.read_text(encoding="utf-8"))
    manifest["selected_model_recipe"]["parameters"]["n_estimators"] = 999
    manifest["selected_model_recipe"]["search"]["selected_trial_parameters"][
        "n_estimators"
    ] = 999
    experiment.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with sqlite3.connect(iteration_dir / "experiments.db") as connection:
        selected_id = evidence["selected_candidate"]["database_run_id"]
        selected = connection.execute(
            "SELECT params_json, artifacts_json FROM runs WHERE id = ?",
            (selected_id,),
        ).fetchone()
        assert selected is not None
        selected_params = json.loads(selected[0])
        selected_artifacts = json.loads(selected[1])
        selected_params["model_recipe"]["parameters"]["n_estimators"] = 999
        selected_params["search"]["selected_trial_parameters"]["n_estimators"] = 999
        selected_artifacts["summary"]["model_recipe"]["parameters"]["n_estimators"] = 999
        selected_artifacts["summary"]["model_recipe"]["search"][
            "selected_trial_parameters"
        ]["n_estimators"] = 999
        selected_artifacts["selected_from_trial"]["parameters"]["n_estimators"] = 999
        connection.execute(
            "UPDATE runs SET params_json = ?, artifacts_json = ? WHERE id = ?",
            (json.dumps(selected_params), json.dumps(selected_artifacts), selected_id),
        )

        trial_id = selected_artifacts["selected_from_trial"]["run_id"]
        trial = connection.execute(
            "SELECT params_json, artifacts_json FROM runs WHERE id = ?",
            (trial_id,),
        ).fetchone()
        assert trial is not None
        trial_params = json.loads(trial[0])
        trial_artifacts = json.loads(trial[1])
        trial_params["model_recipe"]["parameters"]["n_estimators"] = 999
        trial_artifacts["summary"]["model_recipe"]["parameters"]["n_estimators"] = 999
        connection.execute(
            "UPDATE runs SET params_json = ?, artifacts_json = ? WHERE id = ?",
            (json.dumps(trial_params), json.dumps(trial_artifacts), trial_id),
        )

    _update_evidence_checksum(experiment, "manifest_sha256", experiment.manifest_path)
    _update_evidence_checksum(experiment, "database_sha256", iteration_dir / "experiments.db")

    verification = run_verifier_agent(
        config=rejected_config,
        proposal=search_proposal,
        experiment=experiment,
        iteration=1,
    )

    assert verification.passed is False
    assert verification.checks["promoted"] is False
    assert verification.checks["bounded_proposal"] is False
    assert "check_failed:bounded_proposal" in verification.reasons
    assert "experiment_rejected" not in verification.reasons
