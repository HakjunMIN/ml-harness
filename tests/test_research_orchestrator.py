from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
        proposal_sha256 = hashlib.sha256(
            json.dumps(
                proposal.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        provenance = {
            "proposal_sha256": proposal_sha256,
            "manifest_sha256": hashlib.sha256(
                experiment.manifest_path.read_bytes()
            ).hexdigest(),
            "report_sha256": hashlib.sha256(experiment.report_path.read_bytes()).hexdigest(),
            "database_sha256": hashlib.sha256(
                (experiment.manifest_path.parent / "experiments.db").read_bytes()
            ).hexdigest(),
        }
        if decisions[iteration - 1] == "promote":
            checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
            status = "pass"
            reasons = ()
            passed = True
        else:
            checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
            checks["promoted"] = False
            status = "reject"
            reasons = ("experiment_rejected",)
            passed = False
        payload = {
            "schema_version": "1",
            "status": status,
            "passed": passed,
            "checks": checks,
            "reasons": list(reasons),
            "provenance": provenance,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return VerificationResult(
            passed,
            checks,
            reasons,
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


def test_arbitrary_verifier_report_path_cannot_promote(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])

    def arbitrary_verifier(*, config, proposal, experiment, iteration):
        checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
        path = Path(config.run_dir) / "outside-verification.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": "pass",
                    "passed": True,
                    "checks": checks,
                    "reasons": [],
                    "provenance": {"proposal_sha256": "0" * 64},
                }
            ),
            encoding="utf-8",
        )
        return VerificationResult(True, checks, (), path)

    monkeypatch.setattr(orchestrator, "run_verifier_agent", arbitrary_verifier)
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"


def test_verifier_provenance_mismatch_fails_closed(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])

    def mismatched_verifier(*, config, proposal, experiment, iteration):
        path = experiment.manifest_path.parent / "verification.json"
        checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": "pass",
                    "passed": True,
                    "checks": checks,
                    "reasons": [],
                    "provenance": {
                        "proposal_sha256": "0" * 64,
                        "manifest_sha256": "1" * 64,
                        "report_sha256": "2" * 64,
                        "database_sha256": "3" * 64,
                    },
                }
            ),
            encoding="utf-8",
        )
        return VerificationResult(True, checks, (), path)

    monkeypatch.setattr(orchestrator, "run_verifier_agent", mismatched_verifier)
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"


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


def test_stale_cross_run_diagnosis_fails_before_proposal(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    monkeypatch.setattr(
        orchestrator,
        "generate_profile_proposal",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    run_dir = tmp_path / "run"
    diagnosis_path = run_dir / "diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    diagnosis["run_id"] = "different-run"
    diagnosis_path.write_text(json.dumps(diagnosis), encoding="utf-8")
    checksum = hashlib.sha256(diagnosis_path.read_bytes()).hexdigest()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    diagnosis_key = str(diagnosis_path.resolve())
    state["artifacts"][diagnosis_key] = checksum
    for event in state["transitions"]:
        if event["artifact_path"] == diagnosis_key:
            event["sha256"] = checksum
    state_path.write_text(json.dumps(state), encoding="utf-8")
    journal_path = run_dir / "journal.jsonl"
    journal = json.loads(journal_path.read_text(encoding="utf-8").splitlines()[0])
    journal["sha256"] = checksum
    journal_path.write_text(json.dumps(journal) + "\n", encoding="utf-8")

    result = run_research_loop(config, resume=True)
    assert result["status"] == "failed"
    assert result["verifier"]["reasons"] == ["diagnosis_binding_invalid"]


def test_journal_append_before_state_write_recovers_on_resume(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    original_persist = orchestrator._persist_state
    calls = {"count": 0}

    def interrupt_after_initial(run_dir, state):
        calls["count"] += 1
        if calls["count"] == 1:
            return original_persist(run_dir, state)
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "_persist_state", interrupt_after_initial)
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    run_dir = tmp_path / "run"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "initialized"
    assert len((run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    result = run_research_loop(config, resume=True)
    assert result["status"] == "ready_for_human_review"


def test_journal_recovery_rejects_malformed_tail_timestamp(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    original_persist = orchestrator._persist_state
    calls = {"count": 0}

    def interrupt_after_initial(run_dir, state):
        calls["count"] += 1
        if calls["count"] == 1:
            return original_persist(run_dir, state)
        raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "_persist_state", interrupt_after_initial)
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    run_dir = tmp_path / "run"
    lines = (run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    tail = json.loads(lines[-1])
    tail["timestamp"] = "not-an-iso-timestamp"
    lines[-1] = json.dumps(tail)
    (run_dir / "journal.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    with pytest.raises(ResearchStateError, match="journal"):
        run_research_loop(config, resume=True)


def test_journal_recovery_rejects_partial_multi_artifact_group(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    original_append = orchestrator._append_journal
    calls = {"count": 0}

    def append_partial(path, events):
        calls["count"] += 1
        if calls["count"] == 2:
            return original_append(path, events[:1])
        return original_append(path, events)

    monkeypatch.setattr(orchestrator, "_append_journal", append_partial)
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        # The partial append is followed by an explicit crash simulation.
        original_persist = orchestrator._persist_state
        orchestrator._persist_state = lambda run_dir, state: (
            original_persist(run_dir, state)
            if state.status in {"initialized", "diagnosed"}
            else (_ for _ in ()).throw(KeyboardInterrupt())
        )
        try:
            run_research_loop(config)
        finally:
            orchestrator._persist_state = original_persist

    monkeypatch.setattr(orchestrator, "_append_journal", original_append)
    with pytest.raises(ResearchStateError, match="incomplete artifacts"):
        run_research_loop(config, resume=True)


def test_journal_recovery_accepts_diagnostic_failure_group(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    monkeypatch.setattr(
        orchestrator,
        "run_diagnostic_agent",
        lambda config: (_ for _ in ()).throw(ValueError("diagnostic failure")),
    )
    original_persist = orchestrator._persist_state
    monkeypatch.setattr(
        orchestrator,
        "_persist_state",
        lambda run_dir, state: (
            original_persist(run_dir, state)
            if state.status == "initialized"
            else (_ for _ in ()).throw(KeyboardInterrupt())
        ),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    result = run_research_loop(config, resume=True)
    assert result["status"] == "failed"


def test_journal_recovery_accepts_verifier_report_failure_group(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])

    def invalid_verifier(*, config, proposal, experiment, iteration):
        checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
        checks["promoted"] = False
        reasons = ("invalid_outcome",)
        path = experiment.manifest_path.parent / "verification.json"
        proposal_sha256 = hashlib.sha256(
            json.dumps(proposal.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        payload = {
            "schema_version": "1",
            "status": "invalid",
            "passed": False,
            "checks": checks,
            "reasons": list(reasons),
            "provenance": {
                "proposal_sha256": proposal_sha256,
                "manifest_sha256": hashlib.sha256(
                    experiment.manifest_path.read_bytes()
                ).hexdigest(),
                "report_sha256": hashlib.sha256(
                    experiment.report_path.read_bytes()
                ).hexdigest(),
                "database_sha256": hashlib.sha256(
                    (experiment.manifest_path.parent / "experiments.db").read_bytes()
                ).hexdigest(),
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return VerificationResult(False, checks, reasons, path)

    monkeypatch.setattr(orchestrator, "run_verifier_agent", invalid_verifier)
    original_persist = orchestrator._persist_state
    monkeypatch.setattr(
        orchestrator,
        "_persist_state",
        lambda run_dir, state: (
            original_persist(run_dir, state)
            if state.status != "failed"
            else (_ for _ in ()).throw(KeyboardInterrupt())
        ),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    result = run_research_loop(config, resume=True)
    assert result["status"] == "failed"


def test_journal_recovery_accepts_bound_experiment_failure_group(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    def failed_experiment(**kwargs):
        directory = Path(kwargs["config"].run_dir) / "iterations" / "001-experiment"
        directory.mkdir(parents=True, exist_ok=True)
        failure_path = directory / "experiment-failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "run_id": kwargs["config"].run_id,
                    "experiment_id": "a" * 64,
                    "iteration": kwargs["iteration"],
                    "run_state": "failed",
                }
            ),
            encoding="utf-8",
        )
        raise orchestrator.ResearchExecutionError("experiment failure")

    monkeypatch.setattr(orchestrator, "run_experiment_agent", failed_experiment)
    original_persist = orchestrator._persist_state
    monkeypatch.setattr(
        orchestrator,
        "_persist_state",
        lambda run_dir, state: (
            original_persist(run_dir, state)
            if state.status != "verifying"
            else (_ for _ in ()).throw(KeyboardInterrupt())
        ),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    result = run_research_loop(config, resume=True)
    assert result["status"] == "failed"
    state = json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))
    failure_paths = [
        path for path in state["artifacts"] if path.endswith("experiment-failure.json")
    ]
    assert len(failure_paths) == 1
    assert state["transitions"][-2]["artifact_path"] == failure_paths[0]


def test_resume_rejects_tampered_experiment_failure_artifact(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])

    def failed_experiment(**kwargs):
        directory = Path(kwargs["config"].run_dir) / "iterations" / "001-experiment"
        directory.mkdir(parents=True, exist_ok=True)
        failure_path = directory / "experiment-failure.json"
        failure_path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "run_id": kwargs["config"].run_id,
                    "experiment_id": "b" * 64,
                    "iteration": kwargs["iteration"],
                    "run_state": "failed",
                }
            ),
            encoding="utf-8",
        )
        raise orchestrator.ResearchExecutionError(
            "experiment failure",
            failure_path=failure_path,
        )

    monkeypatch.setattr(orchestrator, "run_experiment_agent", failed_experiment)
    original_persist = orchestrator._persist_state
    monkeypatch.setattr(
        orchestrator,
        "_persist_state",
        lambda run_dir, state: (
            original_persist(run_dir, state)
            if state.status != "failed"
            else (_ for _ in ()).throw(KeyboardInterrupt())
        ),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(KeyboardInterrupt):
        run_research_loop(config)

    failure_path = next((tmp_path / "run").rglob("experiment-failure.json"))
    failure_path.write_text(failure_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    with pytest.raises(ResearchStateError, match="checksum mismatch"):
        run_research_loop(config, resume=True)


def test_excluded_profiles_are_filtered_and_exhausted(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    calls = _fake_agents(monkeypatch, decisions=["promote"])
    monkeypatch.setattr(
        orchestrator,
        "run_diagnostic_agent",
        lambda config: replace(_diagnosis(), recommended_profiles=("safe_weather",)),
    )
    result = run_research_loop(_config(tmp_path, ["history_tree"], 1))

    assert result["status"] == "exhausted"
    assert result["iterations"] == 0
    assert result["used_profiles"] == []
    assert calls["experiment"] == 0


def test_diagnostic_failure_persists_terminal_failed_state(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    monkeypatch.setattr(
        orchestrator,
        "run_diagnostic_agent",
        lambda config: (_ for _ in ()).throw(ValueError("raw diagnostic detail")),
    )
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"
    assert result["verifier"]["reasons"] == ["diagnostic_failed"]
    state = json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert [
        (event["from_status"], event["to_status"])
        for event in state["transitions"]
    ] == [
        ("initialized", "diagnosed"),
        ("diagnosed", "proposed"),
        ("proposed", "experimenting"),
        ("experimenting", "verifying"),
        ("verifying", "failed"),
    ]
    persisted = json.dumps(
        {
            "result": result,
            "state": state,
            "journal": (tmp_path / "run" / "journal.jsonl").read_text(encoding="utf-8"),
        }
    )
    assert "raw diagnostic detail" not in persisted


def test_journal_append_failure_rolls_back_state_transition(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    original_append = orchestrator._append_journal
    monkeypatch.setattr(
        orchestrator,
        "_append_journal",
        lambda path, events: (_ for _ in ()).throw(OSError("journal unavailable")),
    )
    with pytest.raises(OSError, match="journal unavailable"):
        run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    run_dir = tmp_path / "run"
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "initialized"
    assert (run_dir / "journal.jsonl").read_text(encoding="utf-8") == ""

    monkeypatch.setattr(orchestrator, "_append_journal", original_append)
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1), resume=True)
    assert result["status"] == "ready_for_human_review"


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
