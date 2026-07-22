from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
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
_DIAGNOSTIC_FAILURE_NAME = "diagnostic-failure.json"
_NOTES_NAME = "research-notes.json"
_PROPOSAL_NAME = "research-proposal.json"
_EVIDENCE_NAME = "experiment-evidence.json"
_EXPERIMENT_FAILURE_NAME = "experiment-failure.json"
_VERIFICATION_NAME = "verification.json"
_FAILURE_NAME = "verification-failure.json"
_EXHAUSTION_NAME = "exhaustion.json"
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
_VERIFIER_REPORT_KEYS = frozenset(
    {"schema_version", "status", "passed", "checks", "reasons", "provenance"}
)
_VERIFIER_PROVENANCE_KEYS = frozenset(
    {"proposal_sha256", "manifest_sha256", "report_sha256", "database_sha256"}
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
        was_terminal = state.status in _TERMINAL_STATUSES
        state = _reconcile_journal(run_dir, state)
        _validate_recorded_experiment_failures(state, config_payload)
        if was_terminal:
            raise ResearchStateError(f"cannot resume terminal state {state.status}")
        journal_count = len(state.transitions)
    else:
        if state_path.exists():
            raise ResearchStateError("run state already exists; use --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_snapshot_path, config_payload)
        state = initialize_state(config, config_sha256=config_sha256)
        (run_dir / _JOURNAL_NAME).touch(exist_ok=True)
        _persist_state(run_dir, state)
        journal_count = 0

    verifier_outcome = "pending"
    verifier_reasons: tuple[str, ...] = ()
    initial_status = state.status

    while state.status not in _TERMINAL_STATUSES:
        if state.status == "initialized":
            try:
                diagnosis = run_diagnostic_agent(config)
                if not isinstance(diagnosis, DiagnosticReport):
                    raise ResearchOrchestratorError("diagnostic result is malformed")
                if diagnosis.dataset_sha256 != config_payload["dataset_sha256"]:
                    raise ResearchOrchestratorError("diagnostic dataset binding is inconsistent")
                diagnosis_path = run_dir / _DIAGNOSIS_NAME
                diagnosis_payload = diagnosis.to_dict()
                diagnosis_payload.update(
                    {
                        "run_id": config.run_id,
                        "config_sha256": config_sha256,
                        "provenance": {
                            "config_sha256": config_sha256,
                            "dataset_sha256": config_payload["dataset_sha256"],
                            "legacy_manifest_sha256": config_payload[
                                "legacy_manifest_sha256"
                            ],
                        },
                    }
                )
                atomic_write_json(diagnosis_path, diagnosis_payload)
            except Exception:
                failure_path = _write_diagnostic_failure(run_dir, state.iteration)
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="diagnosed",
                    artifact_paths={"diagnostic_failure": failure_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                verifier_reasons = (_diagnostic_failure_reason(state),)
                continue

            recommended = set(diagnosis.recommended_profiles)
            eligible = tuple(
                profile for profile in state.remaining_profiles if profile in recommended
            )
            state = _with_remaining_profiles(state, eligible)
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
            if any(Path(path).name == _DIAGNOSTIC_FAILURE_NAME for path in state.artifacts):
                diagnostic_failure_path = _artifact_named(state, _DIAGNOSTIC_FAILURE_NAME)
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="proposed",
                    artifact_paths={"diagnostic_failure": diagnostic_failure_path},
                    journal_count=journal_count,
                )
                continue
            if not state.used_profiles:
                exhaustion_path = _write_exhaustion(
                    run_dir,
                    state.iteration,
                    "no_recommended_profiles",
                )
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="proposed",
                    artifact_paths={"exhaustion": exhaustion_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "reject"
                verifier_reasons = (_safe_reason("no_recommended_profiles"),)
                continue
            try:
                diagnosis = _load_diagnosis(
                    _artifact_named(state, _DIAGNOSIS_NAME),
                    config=config,
                    config_payload=config_payload,
                )
            except Exception:
                failure_path = _write_diagnostic_failure(
                    run_dir,
                    state.iteration,
                    "diagnosis_binding_invalid",
                )
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="proposed",
                    artifact_paths={"diagnostic_failure": failure_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                verifier_reasons = (_safe_reason("diagnosis_binding_invalid"),)
                continue
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
            if any(
                Path(path).name in {_DIAGNOSTIC_FAILURE_NAME, _EXHAUSTION_NAME}
                for path in state.artifacts
            ):
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="verifying",
                    artifact_paths={},
                    journal_count=journal_count,
                )
                continue
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
                artifact_paths: dict[str, Path]
                experiment_failure = None
                if isinstance(exc, ResearchExecutionError):
                    try:
                        failure_path = exc.failure_path
                        if failure_path is None:
                            failure_path = _locate_experiment_failure_artifact(
                                run_dir,
                                state.iteration,
                            )
                        experiment_failure = _validate_experiment_failure_artifact(
                            failure_path,
                            config=config,
                            iteration=state.iteration,
                        )
                    except (OSError, TypeError, ValueError, ResearchStateError):
                        experiment_failure = None
                if experiment_failure is not None:
                    artifact_paths = {"experiment_failure": experiment_failure}
                else:
                    failure_path = _write_failure(
                        run_dir,
                        state.iteration,
                        "experiment_failed",
                    )
                    artifact_paths = {"failure": failure_path}
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="verifying",
                    artifact_paths=artifact_paths,
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
            if any(Path(path).name == _DIAGNOSTIC_FAILURE_NAME for path in state.artifacts):
                diagnostic_failure_path = _artifact_named(state, _DIAGNOSTIC_FAILURE_NAME)
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="failed",
                    artifact_paths={"diagnostic_failure": diagnostic_failure_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "invalid"
                verifier_reasons = (_diagnostic_failure_reason(state),)
                continue
            if any(Path(path).name == _EXHAUSTION_NAME for path in state.artifacts):
                exhaustion_path = _artifact_named(state, _EXHAUSTION_NAME)
                state, journal_count = _advance(
                    run_dir,
                    state,
                    to_status="exhausted",
                    artifact_paths={"exhaustion": exhaustion_path},
                    journal_count=journal_count,
                )
                verifier_outcome = "reject"
                verifier_reasons = (_safe_reason("no_recommended_profiles"),)
                continue
            if any(
                Path(path).name in {_FAILURE_NAME, _EXPERIMENT_FAILURE_NAME}
                for path in state.artifacts
            ):
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
                proposal_path = _artifact_named(state, _PROPOSAL_NAME)
                proposal = _load_proposal(proposal_path)
                experiment = _load_experiment(state)
                result = run_verifier_agent(
                    config=config,
                    proposal=proposal,
                    experiment=experiment,
                    iteration=state.iteration,
                )
                _validate_verification_result(
                    result,
                    config=config,
                    experiment=experiment,
                    iteration=state.iteration,
                    proposal_path=proposal_path,
                    proposal=proposal,
                    state=state,
                )
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


def _validate_recorded_experiment_failures(
    state: ResearchState,
    config_payload: Mapping[str, object],
) -> None:
    run_dir = Path(str(config_payload["run_dir"])).resolve()
    for artifact_path in state.artifacts:
        path = Path(artifact_path)
        if path.name != _EXPERIMENT_FAILURE_NAME:
            continue
        if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
            raise ResearchStateError("experiment failure artifact is not a regular file")
        try:
            relative = path.resolve().relative_to(run_dir / "iterations")
        except ValueError as exc:
            raise ResearchStateError(
                "experiment failure artifact is outside the iterations directory"
            ) from exc
        if not relative.parts or not re.match(r"^[0-9]{3}-", relative.parts[0]):
            raise ResearchStateError("experiment failure artifact path is invalid")
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle, object_pairs_hook=_strict_config_object)
        except (_DuplicateConfigKey, OSError, json.JSONDecodeError) as exc:
            raise ResearchStateError("experiment failure artifact JSON is invalid") from exc
        if (
            not isinstance(payload, Mapping)
            or set(payload)
            != {"schema_version", "run_id", "experiment_id", "iteration", "run_state"}
            or payload["schema_version"] != "1"
            or payload["run_id"] != config_payload["run_id"]
            or type(payload["experiment_id"]) is not str
            or not _SHA256_PATTERN.fullmatch(payload["experiment_id"])
            or type(payload["iteration"]) is not int
            or payload["iteration"] < 1
            or payload["run_state"] != "failed"
            or not relative.parts[0].startswith(f"{payload['iteration']:03d}-")
        ):
            raise ResearchStateError("experiment failure artifact schema or binding is invalid")


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
    events = next_state.transitions[journal_count:]
    state_path = run_dir / _STATE_NAME
    journal_path = run_dir / _JOURNAL_NAME
    previous_state = state_path.read_bytes() if state_path.exists() else None
    previous_journal = journal_path.read_bytes() if journal_path.exists() else None
    try:
        _append_journal(journal_path, events)
    except Exception:
        _restore_file(journal_path, previous_journal)
        raise
    try:
        _persist_state(run_dir, next_state)
    except Exception:
        _restore_file(journal_path, previous_journal)
        _restore_file(state_path, previous_state)
        raise
    return next_state, len(next_state.transitions)


def _append_journal(path: Path, events: tuple[Mapping[str, object], ...]) -> None:
    if not events:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _restore_file(path: Path, content: bytes | None) -> None:
    if content is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.rollback.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _reconcile_journal(run_dir: Path, state: ResearchState) -> ResearchState:
    journal_path = run_dir / _JOURNAL_NAME
    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ResearchStateError("journal is missing or unreadable") from exc
    try:
        events = [
            json.loads(line, object_pairs_hook=_strict_config_object)
            for line in lines
            if line
        ]
    except (_DuplicateConfigKey, json.JSONDecodeError) as exc:
        raise ResearchStateError("journal JSON is invalid") from exc
    event_keys = {
        "timestamp",
        "from_status",
        "to_status",
        "iteration",
        "profile",
        "artifact_path",
        "sha256",
    }
    for event in events:
        if (
            not isinstance(event, Mapping)
            or set(event) != event_keys
            or type(event["timestamp"]) is not str
            or not _canonical_timestamp(event["timestamp"])
            or (
                event["artifact_path"] is not None
                and type(event["artifact_path"]) is not str
            )
        ):
            raise ResearchStateError("journal event schema is invalid")
    recorded = [dict(event) for event in state.transitions]
    if events[: len(recorded)] != recorded:
        raise ResearchStateError("journal does not match persisted state")
    if len(events) == len(recorded):
        return state

    recovered = state
    for group in _journal_groups(events[len(recorded) :]):
        _validate_recovery_group(group)
        artifact_paths = {
            Path(event["artifact_path"]).name: Path(event["artifact_path"])
            for event in group
            if event["artifact_path"] is not None
        }
        candidate = transition_state(
            recovered,
            to_status=str(group[0]["to_status"]),
            artifact_paths=artifact_paths,
        )
        generated = candidate.transitions[-len(group) :]
        for journal_event, generated_event in zip(group, generated):
            for key in (
                "from_status",
                "to_status",
                "iteration",
                "profile",
                "artifact_path",
                "sha256",
            ):
                if journal_event[key] != generated_event[key]:
                    raise ResearchStateError("journal transition cannot be recovered")
        recovered = replace(
            candidate,
            transitions=recovered.transitions + tuple(group),
        )

    atomic_write_json(run_dir / _STATE_NAME, recovered.to_dict())
    try:
        return load_state(run_dir / _STATE_NAME)
    except ResearchStateError:
        raise


def _journal_groups(
    events: list[Mapping[str, object]],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    groups: list[tuple[Mapping[str, object], ...]] = []
    current: list[Mapping[str, object]] = []
    key: tuple[object, ...] | None = None
    for event in events:
        event_key = (
            event["timestamp"],
            event["from_status"],
            event["to_status"],
            event["iteration"],
            event["profile"],
        )
        if current and event_key != key:
            groups.append(tuple(current))
            current = []
        current.append(event)
        key = event_key
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _canonical_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.isoformat(timespec="microseconds") == value


def _validate_recovery_group(group: tuple[Mapping[str, object], ...]) -> None:
    if not group:
        raise ResearchStateError("journal transition group is empty")
    transition = (group[0]["from_status"], group[0]["to_status"])
    if any(
        (event["from_status"], event["to_status"]) != transition
        or event["timestamp"] != group[0]["timestamp"]
        or event["iteration"] != group[0]["iteration"]
        or event["profile"] != group[0]["profile"]
        for event in group
    ):
        raise ResearchStateError("journal transition group is inconsistent")
    names = [
        Path(event["artifact_path"]).name
        for event in group
        if event["artifact_path"] is not None
    ]
    if len(names) != len(set(names)):
        raise ResearchStateError("journal transition group contains duplicate artifacts")
    expected: frozenset[str] | None
    if transition == ("initialized", "diagnosed") or transition == ("iterate", "diagnosed"):
        expected = frozenset({_DIAGNOSIS_NAME})
        if names == [_DIAGNOSTIC_FAILURE_NAME]:
            expected = frozenset({_DIAGNOSTIC_FAILURE_NAME})
    elif transition == ("diagnosed", "proposed"):
        if names == [_DIAGNOSTIC_FAILURE_NAME]:
            expected = frozenset({_DIAGNOSTIC_FAILURE_NAME})
        elif names == [_EXHAUSTION_NAME]:
            expected = frozenset({_EXHAUSTION_NAME})
        else:
            expected = frozenset({_PROPOSAL_NAME, _NOTES_NAME, _CONFIG_NAME})
    elif transition == ("proposed", "experimenting"):
        expected = frozenset()
    elif transition == ("experimenting", "verifying"):
        if not names:
            expected = frozenset()
        elif names == [_FAILURE_NAME]:
            expected = frozenset({_FAILURE_NAME})
        elif names == [_EXPERIMENT_FAILURE_NAME]:
            expected = frozenset({_EXPERIMENT_FAILURE_NAME})
        else:
            expected = frozenset(
                {
                    "promotion_manifest.json",
                    "performance_report.md",
                    "experiments.db",
                    _EVIDENCE_NAME,
                }
            )
    elif transition[0] == "verifying" and transition[1] in {
        "iterate",
        "ready_for_human_review",
        "exhausted",
    }:
        expected = (
            frozenset({_VERIFICATION_NAME})
            if names != [_EXHAUSTION_NAME]
            else frozenset({_EXHAUSTION_NAME})
        )
    elif transition == ("verifying", "failed"):
        expected = frozenset(names)
        if expected not in {
            frozenset(),
            frozenset({_FAILURE_NAME}),
            frozenset({_EXPERIMENT_FAILURE_NAME}),
            frozenset({_VERIFICATION_NAME}),
            frozenset({_DIAGNOSTIC_FAILURE_NAME}),
        }:
            raise ResearchStateError("journal failure transition artifacts are invalid")
    else:
        raise ResearchStateError("journal failure transition origin is invalid")
    if expected is not None and frozenset(names) != expected:
        raise ResearchStateError("journal transition group has incomplete artifacts")


def _current_profile(state: ResearchState) -> str:
    if not state.used_profiles:
        raise ResearchStateError("state has no selected profile")
    profile = state.used_profiles[-1]
    if not _PROFILE_PATTERN.fullmatch(profile):
        raise ResearchStateError("state contains an invalid profile")
    return profile


def _with_remaining_profiles(
    state: ResearchState,
    remaining_profiles: tuple[str, ...],
) -> ResearchState:
    return ResearchState(
        run_id=state.run_id,
        status=state.status,
        iteration=state.iteration,
        used_profiles=state.used_profiles,
        remaining_profiles=remaining_profiles,
        artifacts=state.artifacts,
        transitions=state.transitions,
        config_sha256=state.config_sha256,
    )


def _iteration_directory(run_dir: Path, iteration: int, profile: str) -> Path:
    directory = run_dir / "iterations" / f"{iteration:03d}-{profile}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _artifact_named(state: ResearchState, name: str) -> Path:
    matches = sorted(Path(path) for path in state.artifacts if Path(path).name == name)
    if not matches:
        raise ResearchStateError(f"recorded artifact is missing from state: {name}")
    return matches[-1]


def _load_diagnosis(
    path: Path,
    *,
    config: ResearchLoopConfig,
    config_payload: Mapping[str, object],
) -> DiagnosticReport:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_strict_config_object)
        if not isinstance(payload, Mapping):
            raise ValueError("diagnosis must be an object")
        payload = dict(payload)
        provenance = payload.pop("provenance")
        if (
            payload.pop("run_id") != config.run_id
            or payload.pop("config_sha256") != _canonical_sha256(config_payload)
            or provenance
            != {
                "config_sha256": _canonical_sha256(config_payload),
                "dataset_sha256": config_payload["dataset_sha256"],
                "legacy_manifest_sha256": config_payload["legacy_manifest_sha256"],
            }
            or payload.get("dataset_sha256") != config_payload["dataset_sha256"]
        ):
            raise ResearchStateError("diagnosis binding does not match this run")
        payload["recommended_profiles"] = tuple(payload["recommended_profiles"])
        return DiagnosticReport(**payload)
    except (_DuplicateConfigKey, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, ResearchStateError):
            raise
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


def _validate_verification_result(
    result: object,
    *,
    config: ResearchLoopConfig,
    experiment: ExperimentResult,
    iteration: int,
    proposal_path: Path,
    proposal: ResearchProposal,
    state: ResearchState,
) -> None:
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
    expected_report = _expected_verification_path(config, experiment, iteration)
    if (
        not isinstance(result.report_path, Path)
        or result.report_path != expected_report
        or result.report_path.is_symlink()
        or not result.report_path.is_file()
    ):
        raise ResearchExecutionError("verifier report is missing")
    _validate_verification_report(
        result,
        experiment,
        proposal_path=proposal_path,
        proposal=proposal,
        state=state,
    )


def _expected_verification_path(
    config: ResearchLoopConfig,
    experiment: ExperimentResult,
    iteration: int,
) -> Path:
    manifest_path = Path(experiment.manifest_path)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ResearchExecutionError("experiment manifest is not a regular file")
    iteration_dir = manifest_path.parent
    if iteration_dir.is_symlink():
        raise ResearchExecutionError("iteration directory is a symlink")
    resolved_iteration = iteration_dir.resolve()
    iterations_root = (Path(config.run_dir).resolve() / "iterations").resolve()
    try:
        relative = resolved_iteration.relative_to(iterations_root)
    except ValueError as exc:
        raise ResearchExecutionError("verification report is outside the iteration directory") from exc
    if not relative.parts or not relative.parts[0].startswith(f"{iteration:03d}-"):
        raise ResearchExecutionError("verification report iteration is invalid")
    return resolved_iteration / _VERIFICATION_NAME


def _validate_verification_report(
    result: VerificationResult,
    experiment: ExperimentResult,
    *,
    proposal_path: Path,
    proposal: ResearchProposal,
    state: ResearchState,
) -> None:
    try:
        with result.report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_strict_config_object)
    except (_DuplicateConfigKey, OSError, json.JSONDecodeError) as exc:
        raise ResearchExecutionError("verification report JSON is invalid") from exc
    if not isinstance(payload, Mapping) or set(payload) != _VERIFIER_REPORT_KEYS:
        raise ResearchExecutionError("verification report schema is invalid")
    if payload["schema_version"] != "1":
        raise ResearchExecutionError("verification report schema is invalid")
    if type(payload["passed"]) is not bool or payload["passed"] != result.passed:
        raise ResearchExecutionError("verification report outcome is inconsistent")
    checks = payload["checks"]
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _EXPECTED_VERIFIER_CHECKS
        or any(type(value) is not bool for value in checks.values())
        or dict(checks) != dict(result.checks)
    ):
        raise ResearchExecutionError("verification report checks are inconsistent")
    reasons = payload["reasons"]
    if (
        not isinstance(reasons, list)
        or any(type(reason) is not str for reason in reasons)
        or reasons != list(result.reasons)
    ):
        raise ResearchExecutionError("verification report reasons are inconsistent")
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != _VERIFIER_PROVENANCE_KEYS:
        raise ResearchExecutionError("verification report provenance is invalid")
    if (
        "proposal_sha256" not in provenance
        or any(
            type(value) is not str or not _SHA256_PATTERN.fullmatch(value)
            for value in provenance.values()
        )
    ):
        raise ResearchExecutionError("verification report provenance is invalid")
    expected_provenance = {
        "proposal_sha256": _proposal_sha256(proposal),
        "manifest_sha256": _bound_artifact_sha256(state, Path(experiment.manifest_path)),
        "report_sha256": _bound_artifact_sha256(state, Path(experiment.report_path)),
        "database_sha256": _bound_artifact_sha256(
            state,
            Path(experiment.manifest_path).parent / "experiments.db",
        ),
    }
    _bound_artifact_sha256(state, proposal_path)
    if dict(provenance) != expected_provenance:
        raise ResearchExecutionError("verification report provenance is inconsistent")
    normal_rejection = (
        not result.passed
        and experiment.run_state == "rejected"
        and result.checks["promoted"] is False
        and all(value for name, value in result.checks.items() if name != "promoted")
        and result.reasons == ("experiment_rejected",)
    )
    expected_status = "pass" if result.passed else "reject" if normal_rejection else "invalid"
    if payload["status"] != expected_status:
        raise ResearchExecutionError("verification report status is inconsistent")


def _bound_artifact_sha256(state: ResearchState, path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ResearchExecutionError("verification evidence is not a regular file")
    resolved = path.resolve()
    recorded = state.artifacts.get(str(resolved))
    if recorded is None:
        raise ResearchExecutionError("verification evidence is not recorded")
    actual = _sha256_file(resolved)
    if actual != recorded:
        raise ResearchExecutionError("verification evidence checksum is inconsistent")
    return actual


def _proposal_sha256(proposal: ResearchProposal) -> str:
    payload = proposal.to_dict()
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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


def _validate_experiment_failure_artifact(
    path: Path | None,
    *,
    config: ResearchLoopConfig,
    iteration: int,
) -> Path | None:
    if not isinstance(path, Path):
        return None
    if (
        path.name != _EXPERIMENT_FAILURE_NAME
        or path.is_symlink()
        or path.parent.is_symlink()
        or not path.is_file()
    ):
        return None
    resolved = path.resolve()
    iterations_root = (Path(config.run_dir).resolve() / "iterations").resolve()
    try:
        relative = resolved.relative_to(iterations_root)
    except ValueError:
        return None
    if (
        len(relative.parts) != 2
        or not relative.parts[0].startswith(f"{iteration:03d}-")
        or relative.parts[1] != _EXPERIMENT_FAILURE_NAME
    ):
        return None
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_strict_config_object)
    except (_DuplicateConfigKey, OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {"schema_version", "run_id", "experiment_id", "iteration", "run_state"}
        or payload["schema_version"] != "1"
        or payload["run_id"] != config.run_id
        or type(payload["experiment_id"]) is not str
        or not _SHA256_PATTERN.fullmatch(payload["experiment_id"])
        or payload["iteration"] != iteration
        or payload["run_state"] != "failed"
    ):
        return None
    return resolved


def _locate_experiment_failure_artifact(run_dir: Path, iteration: int) -> Path | None:
    iterations_root = run_dir / "iterations"
    if iterations_root.is_symlink() or not iterations_root.is_dir():
        return None
    candidates = sorted(
        path
        for path in iterations_root.glob(f"{iteration:03d}-*/{_EXPERIMENT_FAILURE_NAME}")
        if path.is_file() and not path.is_symlink() and not path.parent.is_symlink()
    )
    return candidates[0] if len(candidates) == 1 else None


def _write_diagnostic_failure(run_dir: Path, iteration: int, reason: str = "diagnostic_failed") -> Path:
    path = run_dir / "iterations" / f"{iteration:03d}-failure" / _DIAGNOSTIC_FAILURE_NAME
    atomic_write_json(
        path,
        {
            "schema_version": "1",
            "status": "failed",
            "rejected_conditions": [_safe_reason(reason)],
        },
    )
    return path


def _diagnostic_failure_reason(state: ResearchState) -> str:
    try:
        path = _artifact_named(state, _DIAGNOSTIC_FAILURE_NAME)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        reasons = payload.get("rejected_conditions", ())
        if isinstance(reasons, list) and len(reasons) == 1:
            return _safe_reason(reasons[0])
    except (AttributeError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return _safe_reason("diagnostic_failed")


def _write_exhaustion(run_dir: Path, iteration: int, reason: str) -> Path:
    path = run_dir / _EXHAUSTION_NAME
    atomic_write_json(
        path,
        {
            "schema_version": "1",
            "status": "exhausted",
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
