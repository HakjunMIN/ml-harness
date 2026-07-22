from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from power_forecasting.proposals import ResearchProposal, load_proposal
from power_forecasting.research_contracts import (
    ResearchContractError,
    ResearchLoopConfig,
    load_research_loop_config,
)
from power_forecasting.research_execution import (
    ExperimentResult,
    ResearchExecutionError,
    VerificationResult,
    run_experiment_agent,
    run_verifier_agent,
)
from power_forecasting.research_roles import (
    DiagnosticReport,
    generate_profile_proposal,
    run_diagnostic_agent,
)
from power_forecasting.research_state import (
    ResearchState,
    ResearchStateError,
    atomic_write_json,
    initialize_state,
    load_state,
    transition_state,
)


_STATE_NAME = "state.json"
_JOURNAL_NAME = "journal.jsonl"
_CONFIG_NAME = "research-config.json"
_DIAGNOSIS_NAME = "diagnosis.json"
_NOTES_NAME = "research-notes.json"
_PROPOSAL_NAME = "research-proposal.json"
_EVIDENCE_NAME = "experiment-evidence.json"
_VERIFICATION_NAME = "verification.json"
_FAILURE_NAME = "verification-failure.json"
_SUMMARY_NAME = "research-summary.json"
_PROFILE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_:-]{0,127}$")
_CANDIDATE_CAP = 3
_EXPECTED_VERIFIER_CHECKS = frozenset(
    {
        "experiment_identity",
        "artifact_paths",
        "evidence_schema",
        "proposal_artifact",
        "proposal_checksum",
        "manifest_checksum",
        "report_checksum",
        "database_checksum",
        "manifest_schema",
        "thresholds",
        "seed_provenance",
        "baseline_provenance",
        "bounded_proposal",
        "selected_candidate",
        "winner_provenance",
        "selected_specs",
        "selected_recipe",
        "sqlite_runs",
        "proposal_runs",
        "metrics_provenance",
        "gate_outcome",
        "report_evidence",
        "promotion_provenance",
        "promoted",
        "verification_report",
    }
)


class ResearchOrchestratorError(RuntimeError):
    """Raised when a research-loop stage cannot fail closed safely."""


class _DuplicateConfigKey(ValueError):
    """Raised when strict configuration JSON contains a duplicate key."""


def _strict_config_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateConfigKey(f"config contains duplicate key: {key}")
        value[key] = item
    return value


def run_research_loop(config_path: Path, *, resume: bool = False) -> Mapping[str, object]:
    """Run or resume the bounded, evidence-only research loop."""

    config_file = Path(config_path)
    config = _read_config(config_file)
    run_dir = Path(config.run_dir)
    state_path = run_dir / _STATE_NAME
    config_payload = _effective_config(config)
    config_sha256 = _canonical_sha256(config_payload)
    config_snapshot_path = run_dir / _CONFIG_NAME

    if resume:
        if not state_path.is_file():
            raise ResearchStateError("resume requires an existing state.json")
        state = load_state(state_path)
        if state.run_id != config.run_id:
            raise ResearchStateError("state run_id does not match configuration")
        _verify_resume_config(
            state,
            config_snapshot_path,
            config_payload,
            config_sha256,
        )
        if state.status in _TERMINAL_STATUSES:
            raise ResearchStateError(f"cannot resume terminal state {state.status}")
        journal_count = len(state.transitions)
    else:
        if state_path.exists():
            raise ResearchStateError("run state already exists; use --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_snapshot_path, config_payload)
        state = initialize_state(config, config_sha256=config_sha256)
        _persist_state(run_dir, state)
        (run_dir / _JOURNAL_NAME).touch(exist_ok=True)
        journal_count = 0

    verifier_outcome = "pending"
    verifier_reasons: tuple[str, ...] = ()
    initial_status = state.status

    while state.status not in _TERMINAL_STATUSES:
        if state.status == "initialized":
            diagnosis = run_diagnostic_agent(config)
            diagnosis_path = run_dir / _DIAGNOSIS_NAME
            atomic_write_json(diagnosis_path, diagnosis.to_dict())
            state, journal_count = _advance(
                run_dir,
                state,
                to_status="diagnosed",
                artifact_paths={"diagnosis": diagnosis_path},
                journal_count=journal_count,
            )
            continue

        if state.status == "iterate":
            diagnosis_path = _artifact_named(state, _DIAGNOSIS_NAME)
            state, journal_count = _advance(
                run_dir,
                state,
                to_status="diagnosed",
                artifact_paths={"diagnosis": diagnosis_path},
                journal_count=journal_count,
            )
            continue

        if state.status == "diagnosed":
            diagnosis = _load_diagnosis(_artifact_named(state, _DIAGNOSIS_NAME))
            profile = _current_profile(state)
            proposal = generate_profile_proposal(
                profile,
                run_id=config.run_id,
                legacy_manifest_path=Path(config.legacy_manifest_path),
                fold_count=config.fold_count,
                objective=config.objective,
                candidate_cap=_CANDIDATE_CAP,
                diagnosis=diagnosis,
            )
            iteration_dir = _iteration_directory(run_dir, state.iteration, profile)
            proposal_path = iteration_dir / _PROPOSAL_NAME
            notes_path = iteration_dir / _NOTES_NAME
            atomic_write_json(proposal_path, proposal.to_dict())
            atomic_write_json(
                notes_path,
                {
                    "schema_version": "1",
                    "profile": profile,
                    "iteration": state.iteration,
                    "candidate_cap": _CANDIDATE_CAP,
                    "rejected_ideas": [],
                },
            )
            state, journal_count = _advance(
                run_dir,
                state,
                to_status="proposed",
                artifact_paths={
                    "proposal": proposal_path,
                    "notes": notes_path,
                    "config": config_snapshot_path,
                },
                journal_count=journal_count,
            )
            continue

        if state.status == "proposed":
            state, journal_count = _advance(
                run_dir,
                state,
                to_status="experimenting",
                artifact_paths={},
                journal_count=journal_count,
            )
            continue

        if state.status == "experimenting":
            if resume and initial_status == "experimenting":
                failure_path = _write_failure(
                    run_dir,
                    state.iteration,
                    "interrupted_experiment",
                )
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="verifying",
                    artifact_paths={"failure": failure_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                verifier_reasons = (_safe_reason("interrupted_experiment"),)
                continue
            try:
                proposal = _load_proposal(_artifact_named(state, _PROPOSAL_NAME))
                _current_profile(state)
                experiment = run_experiment_agent(
                    config=config,
                    proposal=proposal,
                    iteration=state.iteration,
                )
            except (ResearchExecutionError, OSError, TypeError, ValueError) as exc:
                failure_path = _write_failure(
                    run_dir,
                    state.iteration,
                    "experiment_failed",
                )
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="verifying",
                    artifact_paths={"failure": failure_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                if not verifier_reasons:
                    verifier_reasons = (_safe_reason("experiment_failed"),)
                continue

            try:
                artifact_paths = _experiment_artifacts(experiment)
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="verifying",
                    artifact_paths=artifact_paths,
                    journal_count=journal_count,
                )
            except (ResearchExecutionError, OSError, TypeError, ValueError) as exc:
                _ = exc
                failure_path = _write_failure(
                    run_dir,
                    state.iteration,
                    "experiment_evidence_invalid",
                )
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="verifying",
                    artifact_paths={"failure": failure_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                verifier_reasons = (_safe_reason("experiment_evidence_invalid"),)
            continue

        if state.status == "verifying":
            if any(Path(path).name == _FAILURE_NAME for path in state.artifacts):
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="failed",
                    artifact_paths={},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                verifier_reasons = (_safe_reason("experiment_failed"),)
                continue
            try:
                proposal = _load_proposal(_artifact_named(state, _PROPOSAL_NAME))
                experiment = _load_experiment(state)
                result = run_verifier_agent(
                    config=config,
                    proposal=proposal,
                    experiment=experiment,
                    iteration=state.iteration,
                )
                _validate_verification_result(result)
                classified = _classify_verification(result, experiment)
            except Exception:
                failure_path = _write_failure(
                    run_dir,
                    state.iteration,
                    "verifier_evidence_invalid",
                )
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="failed",
                    artifact_paths={"failure": failure_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                verifier_reasons = (_safe_reason("verifier_evidence_invalid"),)
                continue

            verifier_outcome, verifier_reasons = classified
            if verifier_outcome == "pass":
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="ready_for_human_review",
                    artifact_paths={"verification": result.report_path},
                    journal_count=journal_count,
                )
            elif verifier_outcome == "reject" and state.remaining_profiles:
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="iterate",
                    artifact_paths={"verification": result.report_path},
                    journal_count=journal_count,
                )
            elif verifier_outcome == "reject":
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="exhausted",
                    artifact_paths={"verification": result.report_path},
                    journal_count=journal_count,
                )
            else:
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="failed",
                    artifact_paths={"verification": result.report_path},
                    journal_count=journal_count,
                )

    summary = _summary(
        config,
        state,
        verifier_outcome=verifier_outcome,
        verifier_reasons=verifier_reasons,
    )
    summary_path = run_dir / _SUMMARY_NAME
    atomic_write_json(summary_path, summary)
    result = dict(summary)
    result["summary_path"] = str(summary_path.resolve())
    result["summary_sha256"] = _sha256_file(summary_path)
    return result


_TERMINAL_STATUSES = frozenset({"ready_for_human_review", "exhausted", "failed"})


def _read_config(config_path: Path) -> ResearchLoopConfig:
    if not isinstance(config_path, Path):
        raise TypeError("config_path must be a Path")
    config_file = config_path.resolve()
    try:
        with config_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_strict_config_object)
    except FileNotFoundError:
        raise
    except _DuplicateConfigKey as exc:
        raise ResearchContractError(str(exc)) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchContractError("config must contain valid JSON") from exc
    repository_root = Path(__file__).resolve().parents[2]
    return load_research_loop_config(
        payload,
        config_path=config_file,
        repository_root=repository_root,
    )


def _effective_config(config: ResearchLoopConfig) -> dict[str, object]:
    return {
        "schema_version": config.schema_version,
        "run_id": config.run_id,
        "dataset_path": str(Path(config.dataset_path).resolve()),
        "dataset_sha256": _sha256_file(Path(config.dataset_path)),
        "legacy_manifest_path": str(Path(config.legacy_manifest_path).resolve()),
        "legacy_manifest_sha256": _sha256_file(Path(config.legacy_manifest_path)),
        "run_dir": str(Path(config.run_dir).resolve()),
        "profiles": list(config.profiles),
        "max_iterations": config.max_iterations,
        "fold_count": config.fold_count,
        "objective": config.objective,
        "minimum_improvement": config.minimum_improvement,
        "max_plant_regression": config.max_plant_regression,
    }


def _canonical_sha256(value: Mapping[str, object]) -> str:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_resume_config(
    state: ResearchState,
    snapshot_path: Path,
    current: Mapping[str, object],
    current_sha256: str,
) -> None:
    if state.config_sha256 is None:
        raise ResearchStateError("state has no immutable configuration checksum")
    if state.config_sha256 != current_sha256:
        raise ResearchStateError("configuration changed since run initialization")
    try:
        with snapshot_path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle, object_pairs_hook=_strict_config_object)
    except (_DuplicateConfigKey, OSError, json.JSONDecodeError) as exc:
        raise ResearchStateError("persisted configuration snapshot is invalid") from exc
    if not isinstance(snapshot, Mapping) or dict(snapshot) != dict(current):
        raise ResearchStateError("configuration changed since run initialization")
    if _canonical_sha256(snapshot) != state.config_sha256:
        raise ResearchStateError("persisted configuration checksum mismatch")


def _persist_state(run_dir: Path, state: ResearchState) -> None:
    atomic_write_json(run_dir / _STATE_NAME, state.to_dict())


def _advance(
    run_dir: Path,
    state: ResearchState,
    *,
    to_status: str,
    artifact_paths: Mapping[str, Path],
    journal_count: int,
) -> tuple[ResearchState, int]:
    next_state = transition_state(
        state,
        to_status=to_status,
        artifact_paths=artifact_paths,
    )
    _persist_state(run_dir, next_state)
    events = next_state.transitions[journal_count:]
    if events:
        journal_path = run_dir / _JOURNAL_NAME
        with journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            for event in events:
                handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
    return next_state, len(next_state.transitions)


def _current_profile(state: ResearchState) -> str:
    if not state.used_profiles:
        raise ResearchStateError("state has no selected profile")
    profile = state.used_profiles[-1]
    if not _PROFILE_PATTERN.fullmatch(profile):
        raise ResearchStateError("state contains an invalid profile")
    return profile


def _iteration_directory(run_dir: Path, iteration: int, profile: str) -> Path:
    directory = run_dir / "iterations" / f"{iteration:03d}-{profile}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _artifact_named(state: ResearchState, name: str) -> Path:
    matches = sorted(Path(path) for path in state.artifacts if Path(path).name == name)
    if not matches:
        raise ResearchStateError(f"recorded artifact is missing from state: {name}")
    return matches[-1]


def _load_diagnosis(path: Path) -> DiagnosticReport:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("diagnosis must be an object")
        payload = dict(payload)
        payload["recommended_profiles"] = tuple(payload["recommended_profiles"])
        return DiagnosticReport(**payload)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResearchStateError("recorded diagnosis cannot be loaded") from exc


def _load_proposal(path: Path) -> ResearchProposal:
    try:
        return load_proposal(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResearchStateError("recorded proposal cannot be loaded") from exc


def _experiment_artifacts(experiment: ExperimentResult) -> dict[str, Path]:
    if not isinstance(experiment, ExperimentResult):
        raise ResearchExecutionError("experiment result is malformed")
    paths: dict[str, Path] = {
        "manifest": Path(experiment.manifest_path),
        "report": Path(experiment.report_path),
    }
    parent = Path(experiment.manifest_path).parent
    for key, name in (
        ("database", "experiments.db"),
        ("evidence", _EVIDENCE_NAME),
    ):
        candidate = parent / name
        if candidate.is_file():
            paths[key] = candidate
    return paths


def _load_experiment(state: ResearchState) -> ExperimentResult:
    evidence_path = _artifact_named(state, _EVIDENCE_NAME)
    try:
        with evidence_path.open("r", encoding="utf-8") as handle:
            evidence = json.load(handle)
        if not isinstance(evidence, Mapping):
            raise ValueError
        selected = evidence.get("selected_candidate")
        if selected is not None and not isinstance(selected, Mapping):
            raise ValueError
        return ExperimentResult(
            run_id=str(evidence["run_id"]),
            experiment_id=str(evidence["experiment_id"]),
            manifest_path=evidence_path.parent / "promotion_manifest.json",
            report_path=evidence_path.parent / "performance_report.md",
            run_state=str(evidence["run_state"]),
            selected_candidate_id=None if selected is None else str(selected["id"]),
            selected_candidate_spec_sha256=None
            if selected is None
            else str(selected["spec_sha256"]),
            selected_recipe_id=None
            if selected is None or selected.get("recipe_id") is None
            else str(selected["recipe_id"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResearchStateError("recorded experiment evidence cannot be loaded") from exc


def _validate_verification_result(result: object) -> None:
    if not isinstance(result, VerificationResult):
        raise ResearchExecutionError("verifier result is malformed")
    if type(result.passed) is not bool or not isinstance(result.checks, Mapping):
        raise ResearchExecutionError("verifier result is malformed")
    if set(result.checks) != _EXPECTED_VERIFIER_CHECKS:
        raise ResearchExecutionError("verifier checks are incomplete or unknown")
    if any(type(value) is not bool for value in result.checks.values()):
        raise ResearchExecutionError("verifier checks are malformed")
    if not isinstance(result.reasons, tuple) or any(
        type(reason) is not str for reason in result.reasons
    ):
        raise ResearchExecutionError("verifier reasons are malformed")
    if not isinstance(result.report_path, Path) or not result.report_path.is_file():
        raise ResearchExecutionError("verifier report is missing")


def _classify_verification(
    result: VerificationResult,
    experiment: ExperimentResult,
) -> tuple[str, tuple[str, ...]]:
    reasons = tuple(_safe_reason(reason) for reason in result.reasons)
    if (
        result.passed
        and experiment.run_state == "promoted"
        and all(result.checks.values())
        and not result.reasons
    ):
        return "pass", reasons
    normal_rejection = (
        not result.passed
        and experiment.run_state == "rejected"
        and result.checks.get("promoted") is False
        and result.checks.get("promoted") is not None
        and all(value for name, value in result.checks.items() if name != "promoted")
        and result.reasons == ("experiment_rejected",)
    )
    if normal_rejection:
        return "reject", reasons
    return "invalid", reasons or (_safe_reason("verification_failed"),)


def _write_failure(run_dir: Path, iteration: int, reason: str) -> Path:
    path = run_dir / "iterations" / f"{iteration:03d}-failure" / _FAILURE_NAME
    atomic_write_json(
        path,
        {
            "schema_version": "1",
            "status": "failed",
            "iteration": iteration,
            "reason": _safe_reason(reason),
        },
    )
    return path


def _summary(
    config: ResearchLoopConfig,
    state: ResearchState,
    *,
    verifier_outcome: str,
    verifier_reasons: tuple[str, ...],
) -> dict[str, object]:
    artifacts = {
        path: {"sha256": checksum}
        for path, checksum in sorted(state.artifacts.items())
    }
    return {
        "run_id": config.run_id,
        "status": state.status,
        "iterations": state.iteration,
        "used_profiles": list(state.used_profiles),
        "remaining_profiles": list(state.remaining_profiles),
        "verifier": {
            "outcome": verifier_outcome,
            "reasons": list(verifier_reasons),
        },
        "artifacts": artifacts,
    }


def _safe_reason(value: object) -> str:
    if type(value) is str and _REASON_PATTERN.fullmatch(value):
        return value[:128]
    return "verification_failed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ResearchOrchestratorError", "run_research_loop"]
