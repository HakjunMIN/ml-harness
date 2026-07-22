from __future__ import annotations

import hashlib
import json
import os
import shutil
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


def _run_dir(tmp_path: Path) -> Path:
    return ROOT / ".agents" / "output" / f"pytest-research-{tmp_path.name}"


def _config(
    tmp_path: Path,
    profiles: list[str],
    max_iterations: int,
    *,
    agent_proposals: bool = False,
) -> Path:
    shutil.rmtree(_run_dir(tmp_path), ignore_errors=True)
    config_path = tmp_path / "research-loop.json"
    payload = {
        "schema_version": "1",
        "run_id": "test-loop",
        "dataset_path": str(ROOT / ".agents/fixtures/valid-dataset.csv"),
        "legacy_manifest_path": str(ROOT / ".agents/fixtures/promoted-manifest.json"),
        "run_dir": str(_run_dir(tmp_path)),
        "profiles": profiles,
        "max_iterations": max_iterations,
        "fold_count": 1,
        "objective": "NMAE",
        "minimum_improvement": 0.01,
        "max_plant_regression": 0.03,
        "agent_proposals": agent_proposals,
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


def test_agent_proposal_handoff_resumes_with_catalog_candidate(tmp_path, monkeypatch):
    calls = _fake_agents(monkeypatch, decisions=["promote"])
    config_path = _config(tmp_path, ["safe_weather"], 1, agent_proposals=True)

    handoff = run_research_loop(config_path)

    assert handoff["status"] == "awaiting_proposal"
    proposal_path = Path(handoff["proposal_path"])
    assert Path(handoff["proposal_context_path"]).is_file()
    assert Path(handoff["proposal_catalog_path"]).is_file()
    proposal = generate_profile_proposal(
        "safe_weather",
        run_id="test-loop",
        legacy_manifest_path=ROOT / ".agents/fixtures/promoted-manifest.json",
        fold_count=1,
        objective="NMAE",
        candidate_cap=2,
        diagnosis=_diagnosis(),
    )
    proposal_path.write_text(json.dumps(proposal.to_dict()), encoding="utf-8")

    result = run_research_loop(config_path, resume=True)

    assert result["status"] == "ready_for_human_review"
    assert calls["experiment"] == 1


def test_agent_proposal_handoff_rejects_out_of_catalog_candidate(tmp_path, monkeypatch):
    calls = _fake_agents(monkeypatch, decisions=["promote"])
    config_path = _config(tmp_path, ["safe_weather"], 1, agent_proposals=True)
    handoff = run_research_loop(config_path)
    proposal_path = Path(handoff["proposal_path"])
    proposal = generate_profile_proposal(
        "safe_weather",
        run_id="test-loop",
        legacy_manifest_path=ROOT / ".agents/fixtures/promoted-manifest.json",
        fold_count=1,
        objective="NMAE",
        candidate_cap=2,
        diagnosis=_diagnosis(),
    ).to_dict()
    proposal["model_recipes"][0]["parameters"]["alpha"] = 0.1
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    result = run_research_loop(config_path, resume=True)

    assert result["status"] == "awaiting_proposal"
    assert calls["experiment"] == 0


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


def test_missing_artifact_invalid_verifier_report_is_accepted_fail_closed(
    tmp_path, monkeypatch
):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])

    def missing_artifact_verifier(*, config, proposal, experiment, iteration):
        experiment.report_path.unlink()
        path = experiment.manifest_path.parent / "verification.json"
        checks = {name: True for name in orchestrator._EXPECTED_VERIFIER_CHECKS}
        checks["report_checksum"] = False
        reasons = ("check_failed:report_checksum",)
        provenance = {
            "proposal_sha256": hashlib.sha256(
                json.dumps(
                    proposal.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                experiment.manifest_path.read_bytes()
            ).hexdigest(),
            "report_sha256": "unavailable",
            "database_sha256": hashlib.sha256(
                (experiment.manifest_path.parent / "experiments.db").read_bytes()
            ).hexdigest(),
        }
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1",
                    "status": "invalid",
                    "passed": False,
                    "checks": checks,
                    "reasons": list(reasons),
                    "provenance": provenance,
                }
            ),
            encoding="utf-8",
        )
        return VerificationResult(False, checks, reasons, path)

    monkeypatch.setattr(
        orchestrator, "run_verifier_agent", missing_artifact_verifier
    )
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"
    assert result["verifier"]["outcome"] == "invalid"
    assert "verification-failure.json" not in json.dumps(result)


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
    state = json.loads((_run_dir(tmp_path) / "state.json").read_text(encoding="utf-8"))
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

    run_dir = _run_dir(tmp_path)
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

    run_dir = _run_dir(tmp_path)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "initialized"
    assert len((run_dir / "journal.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    result = run_research_loop(config, resume=True)
    assert result["status"] == "ready_for_human_review"


def test_journal_append_rejects_preexisting_symlink(tmp_path):
    import power_forecasting.research_orchestrator as orchestrator

    target = tmp_path / "target.jsonl"
    target.write_text("keep\n", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    journal.symlink_to(target)

    with pytest.raises(OSError, match="journal"):
        orchestrator._append_journal(
            journal,
            (
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "from_status": "initialized",
                    "to_status": "diagnosed",
                },
            ),
            trusted_root=tmp_path,
        )

    assert target.read_text(encoding="utf-8") == "keep\n"


def test_journal_append_rejects_symlinked_parent(tmp_path):
    import power_forecasting.research_orchestrator as orchestrator

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(OSError, match="journal"):
        orchestrator._append_journal(
            linked_parent / "journal.jsonl",
            (
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "from_status": "initialized",
                    "to_status": "diagnosed",
                },
            ),
            trusted_root=tmp_path,
        )

    assert not (real_parent / "journal.jsonl").exists()


def test_journal_parent_traverses_absolute_components_by_dirfd(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    trusted_root = tmp_path / "trusted" / "run"
    trusted_root.mkdir(parents=True)
    calls: list[tuple[object, int | None]] = []
    original_open = orchestrator.os.open

    def recording_open(path, flags, mode=0o777, *, dir_fd=None):
        calls.append((path, dir_fd))
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(orchestrator.os, "open", recording_open)
    orchestrator._append_journal(
        trusted_root / "journal.jsonl",
        (
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "from_status": "initialized",
                "to_status": "diagnosed",
            },
        ),
        trusted_root=trusted_root,
    )

    assert calls[0] == (os.sep, None)
    assert all(dir_fd is not None for _, dir_fd in calls[1:])
    assert all(not os.path.isabs(os.fspath(path)) for path, _ in calls[1:])


def test_journal_rejects_symlinked_trusted_root_ancestor(tmp_path):
    import power_forecasting.research_orchestrator as orchestrator

    real_parent = tmp_path / "real"
    trusted_root = real_parent / "run"
    trusted_root.mkdir(parents=True)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    journal = linked_parent / "run" / "journal.jsonl"

    with pytest.raises(OSError, match="journal"):
        orchestrator._append_journal(
            journal,
            (
                {
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "from_status": "initialized",
                    "to_status": "diagnosed",
                },
            ),
            trusted_root=journal.parent,
        )

    assert not (trusted_root / "journal.jsonl").exists()


def test_advance_passes_trusted_run_dir_to_journal_append(tmp_path, monkeypatch):
    import power_forecasting.research_orchestrator as orchestrator

    _fake_agents(monkeypatch, decisions=["promote"])
    original_append = orchestrator._append_journal
    calls: list[Path | None] = []

    def append_with_capture(path, events, *, trusted_root=None):
        calls.append(trusted_root)
        return original_append(path, events, trusted_root=trusted_root)

    monkeypatch.setattr(orchestrator, "_append_journal", append_with_capture)
    config = _config(tmp_path, ["safe_weather"], 1)

    run_research_loop(config)

    assert calls
    assert all(root == _run_dir(tmp_path) for root in calls)


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

    run_dir = _run_dir(tmp_path)
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

    def append_partial(path, events, *, trusted_root):
        calls["count"] += 1
        if calls["count"] == 2:
            return original_append(path, events[:1], trusted_root=trusted_root)
        return original_append(path, events, trusted_root=trusted_root)

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
    assert result["used_profiles"] == []
    assert not list(_run_dir(tmp_path).rglob("experiment-failure.json"))


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
    state = json.loads((_run_dir(tmp_path) / "state.json").read_text(encoding="utf-8"))
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

    failure_path = next(_run_dir(tmp_path).rglob("experiment-failure.json"))
    failure_path.write_text(failure_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    with pytest.raises(ResearchStateError, match="checksum mismatch"):
        run_research_loop(config, resume=True)


def test_boolean_experiment_failure_iteration_fails_closed(tmp_path, monkeypatch):
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
                    "experiment_id": "c" * 64,
                    "iteration": True,
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
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"
    state = json.loads((_run_dir(tmp_path) / "state.json").read_text(encoding="utf-8"))
    assert not any(path.endswith("experiment-failure.json") for path in state["artifacts"])
    assert any(path.endswith("verification-failure.json") for path in state["artifacts"])


def test_resume_rejects_nested_experiment_failure_artifact_path(tmp_path, monkeypatch):
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
                    "experiment_id": "d" * 64,
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

    run_dir = _run_dir(tmp_path)
    canonical = next(run_dir.rglob("experiment-failure.json"))
    nested = canonical.parent / "nested" / canonical.name
    nested.parent.mkdir()
    canonical.rename(nested)
    checksum = hashlib.sha256(nested.read_bytes()).hexdigest()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    canonical_key = str(canonical.resolve())
    nested_key = str(nested.resolve())
    state["artifacts"][nested_key] = state["artifacts"].pop(canonical_key)
    for event in state["transitions"]:
        if event["artifact_path"] == canonical_key:
            event["artifact_path"] = nested_key
            event["sha256"] = checksum
    state_path.write_text(json.dumps(state), encoding="utf-8")
    journal_path = run_dir / "journal.jsonl"
    events = []
    for line in journal_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["artifact_path"] == canonical_key:
            event["artifact_path"] = nested_key
            event["sha256"] = checksum
        events.append(json.dumps(event))
    journal_path.write_text("\n".join(events) + "\n", encoding="utf-8")

    monkeypatch.setattr(orchestrator, "_persist_state", original_persist)
    with pytest.raises(ResearchStateError, match="path"):
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

    calls = {"experiment": 0}

    monkeypatch.setattr(
        orchestrator,
        "run_diagnostic_agent",
        lambda config: (_ for _ in ()).throw(ValueError("raw diagnostic detail")),
    )
    monkeypatch.setattr(
        orchestrator,
        "run_experiment_agent",
        lambda **kwargs: calls.__setitem__("experiment", calls["experiment"] + 1),
    )
    result = run_research_loop(_config(tmp_path, ["safe_weather"], 1))

    assert result["status"] == "failed"
    assert calls["experiment"] == 0
    assert result["verifier"]["reasons"] == ["diagnostic_failed"]
    state = json.loads((_run_dir(tmp_path) / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["used_profiles"] == []
    assert state["remaining_profiles"] == ["safe_weather"]
    assert [
        Path(path).name for path in state["artifacts"]
    ] == ["diagnostic-failure.json"]
    assert not list(_run_dir(tmp_path).rglob("experiment-failure.json"))
    assert not list(_run_dir(tmp_path).rglob("research-proposal.json"))
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
            "journal": (_run_dir(tmp_path) / "journal.jsonl").read_text(encoding="utf-8"),
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
        lambda path, events, *, trusted_root: (_ for _ in ()).throw(
            OSError("journal unavailable")
        ),
    )
    config = _config(tmp_path, ["safe_weather"], 1)
    with pytest.raises(OSError, match="journal unavailable"):
        run_research_loop(config)

    run_dir = _run_dir(tmp_path)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "initialized"
    assert (run_dir / "journal.jsonl").read_text(encoding="utf-8") == ""

    monkeypatch.setattr(orchestrator, "_append_journal", original_append)
    result = run_research_loop(config, resume=True)
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

    proposal = next(_run_dir(tmp_path).rglob("research-proposal.json"))
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
