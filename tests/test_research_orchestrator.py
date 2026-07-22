from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from power_forecasting.data import REQUIRED_COLUMNS
from power_forecasting import cli
from power_forecasting.proposals import ResearchProposal
from power_forecasting.research_execution import ExperimentResult, VerificationResult
from power_forecasting.research_orchestrator import run_research_loop
from power_forecasting.research_roles import DiagnosticReport, generate_profile_proposal
from power_forecasting.research_state import ResearchStateError


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, profiles: list[str], max_iterations: int) -> Path:
    config_path = tmp_path / "research-loop.json"
    payload = {
        "schema_version": "1",
        "run_id": "test-loop",
        "dataset_path": str(ROOT / ".agents/fixtures/valid-dataset.csv"),
        "legacy_manifest_path": str(ROOT / ".agents/fixtures/promoted-manifest.json"),
        "run_dir": str(tmp_path / "run"),
        "profiles": profiles,
        "max_iterations": max_iterations,
        "fold_count": 1,
        "objective": "NMAE",
        "minimum_improvement": 0.01,
        "max_plant_regression": 0.03,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _diagnosis() -> DiagnosticReport:
    dataset = ROOT / ".agents/fixtures/valid-dataset.csv"
    prediction_columns = tuple(
        column for column in REQUIRED_COLUMNS if column.startswith(("forecast_", "ldaps_"))
    )
    return DiagnosticReport(
        schema_version="1",
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        row_count=4,
        plant_count=1,
        time_start="2024-01-01T00:00:00",
        time_end="2024-01-01T03:00:00",
        missingness={column: 0.0 for column in REQUIRED_COLUMNS},
        drift_summary={column: 0.0 for column in prediction_columns},
        residual_summary={"capacity_utilization_mean": 0.5, "zero_baseline_nmae": 0.5},
        leakage_checks={
            "dataset_schema_valid": True,
            "history_features_strict_prior": True,
            "prediction_inputs_exclude_actual": True,
            "prediction_inputs_exclude_target": True,
        },
        recommended_profiles=("safe_weather", "history_tree", "bounded_search"),
    )


def _fake_agents(monkeypatch, *, decisions: list[str]):
    import power_forecasting.research_orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "run_diagnostic_agent", lambda config: _diagnosis())
    calls = {"experiment": 0}

    def experiment(*, config, proposal, iteration):
        calls["experiment"] += 1
        directory = Path(config.run_dir) / "iterations" / f"{iteration:03d}-evidence"
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in (
            ("promotion_manifest.json", "{}"),
            ("performance_report.md", "aggregate report"),
            ("experiments.db", "database"),
            (
                "experiment-evidence.json",
                json.dumps(
                    {
                        "run_id": config.run_id,
                        "experiment_id": f"experiment-{iteration}",
                        "run_state": "promoted"
                        if decisions[iteration - 1] == "promote"
                        else "rejected",
                        "selected_candidate": None,
                    }
                ),
            ),
        ):
            (directory / name).write_text(content, encoding="utf-8")
        return ExperimentResult(
            config.run_id,
            f"experiment-{iteration}",
            directory / "promotion_manifest.json",
            directory / "performance_report.md",
            "promoted" if decisions[iteration - 1] == "promote" else "rejected",
            None,
            None,
            None,
        )

    def verifier(*, config, proposal, experiment, iteration):
        path = experiment.manifest_path.parent / "verification.json"
        path.write_text("{}", encoding="utf-8")
        if decisions[iteration - 1] == "promote":
            checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
            return VerificationResult(True, checks, (), path)
        checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
        checks["promoted"] = False
        return VerificationResult(
            False,
            checks,
            ("experiment_rejected",),
            path,
        )

    monkeypatch.setattr(orchestrator, "run_experiment_agent", experiment)
    monkeypatch.setattr(orchestrator, "run_verifier_agent", verifier)
    return calls


def test_promoted_result_ends_at_human_review_and_summary_is_private(tmp_path, monkeypatch):
    _fake_agents(monkeypatch, decisions=["promote"])
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "ready_for_human_review"
    assert result["iterations"] == 1
    assert result["used_profiles"] == ["safe_weather"]
    assert "generation_mw" not in json.dumps(result)
    assert Path(result["summary_path"]).is_file()


def test_rejection_uses_each_profile_once_then_exhausts(tmp_path, monkeypatch):
    calls = _fake_agents(monkeypatch, decisions=["reject", "reject"])
    result = run_research_loop(_config(tmp_path, ["safe_weather", "history_tree"], 2))

    assert result["status"] == "exhausted"
    assert result["iterations"] == 2
    assert result["used_profiles"] == ["safe_weather", "history_tree"]
    assert calls["experiment"] == 2


def test_malformed_verifier_fails_closed(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    monkeypatch.setattr(orchestrator, "run_verifier_agent", lambda **kwargs: {"status": "pass"})
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"
    assert result["verifier"]["outcome"] == "invalid"


def test_partial_verifier_result_fails_closed(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])

    def partial_verifier(*, config, proposal, experiment, iteration):
        path = experiment.manifest_path.parent / "verification.json"
        path.write_text("{}", encoding="utf-8")
        return VerificationResult(True, {"promoted": True}, (), path)

    monkeypatch.setattr(orchestrator, "run_verifier_agent", partial_verifier)
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"
    assert result["verifier"]["outcome"] == "invalid"
    assert "verification-failure.json" in json.dumps(result)


def test_verifier_evidence_parse_error_persists_failed_state(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    original_experiment = orchestrator.run_experiment_agent

    def malformed_experiment(**kwargs):
        result = original_experiment(**kwargs)
        result.manifest_path.parent.joinpath("experiment-evidence.json").write_text(
            "{not-json", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(orchestrator, "run_experiment_agent", malformed_experiment)
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"
    state = json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"


def test_resume_rejects_effective_config_changes(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    monkeypatch.setattr(
        orchestrator,
        "run_experiment_agent",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["objective"] = "MAE"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResearchStateError, match="configuration changed"):
        run_research_loop(config, resume=True)


def test_resume_experimenting_fails_without_rerunning_experiment(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    calls = _fake_agents(monkeypatch, decisions=["promote"])
    monkeypatch.setattr(
        orchestrator,
        "run_experiment_agent",
        lambda **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    _fake_agents(monkeypatch, decisions=["promote"])
    result = run_research_loop(config, resume=True)
    assert result["status"] == "failed"
    assert calls["experiment"] == 0


def test_config_json_rejects_duplicate_top_level_and_nested_keys(tmp_path):
    config = _config(tmp_path, ["safe_weather"], 1)
    payload = json.loads(config.read_text(encoding="utf-8"))
    fields = [
        f'"{key}": {json.dumps(value)}'
        for key, value in payload.items()
    ]
    duplicate_top_level = "{" + ",".join(fields + [f'"objective": "NMAE"']) + "}"
    config.write_text(duplicate_top_level, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        run_research_loop(config)

    nested = ",".join(fields + ['"extra": {"key": 1, "key": 2}'])
    config.write_text("{" + nested + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        run_research_loop(config)


def test_cli_research_loop_prints_summary_and_returns_success(tmp_path, monkeypatch, capsys):
    _fake_agents(monkeypatch, decisions=["promote"])
    config = _config(tmp_path, ["safe_weather"], 1)

    assert cli.main(["research-loop", "--config", str(config)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "ready_for_human_review"
    assert output["run_id"] == "test-loop"


def test_resume_requires_matching_artifacts_and_terminal_runs_cannot_resume(
    tmp_path, monkeypatch
):
    _fake_agents(monkeypatch, decisions=["promote"])
    config = _config(tmp_path, ["safe_weather"], 1)
    import power_forecasting.research_orchestrator as orchestrator

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "run_experiment_agent", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    proposal = next((tmp_path / "run").rglob("research-proposal.json"))
    original_proposal = proposal.read_bytes()
    proposal.write_text("tampered", encoding="utf-8")
    with pytest.raises(ResearchStateError, match="checksum mismatch"):
        run_research_loop(config, resume=True)

    proposal.write_bytes(original_proposal)
    result = run_research_loop(config, resume=True)
    assert result["status"] == "failed"
    with pytest.raises(ResearchStateError, match="run state already exists"):
        run_research_loop(config)
    with pytest.raises(ResearchStateError, match="terminal state"):
        run_research_loop(config, resume=True)
