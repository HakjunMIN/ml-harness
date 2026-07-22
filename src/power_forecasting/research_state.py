from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from power_forecasting.data import REQUIRED_COLUMNS
from power_forecasting.research_contracts import (
    ResearchContractError,
    ResearchLoopConfig,
    SUPPORTED_PROFILES,
    validate_run_id,
)


_STATE_REQUIRED_KEYS = frozenset(
    {
        "run_id",
        "status",
        "iteration",
        "used_profiles",
        "remaining_profiles",
        "artifacts",
        "transitions",
    }
)
_STATE_KEYS = _STATE_REQUIRED_KEYS | frozenset({"config_sha256"})
_EVENT_KEYS = frozenset(
    {
        "timestamp",
        "from_status",
        "to_status",
        "iteration",
        "profile",
        "artifact_path",
        "sha256",
    }
)
_STATUSES = frozenset(
    {
        "initialized",
        "diagnosed",
        "proposed",
        "experimenting",
        "verifying",
        "iterate",
        "ready_for_human_review",
        "exhausted",
        "failed",
    }
)
_TERMINAL_STATUSES = frozenset({"ready_for_human_review", "exhausted", "failed"})
_ALLOWED_TRANSITIONS = {
    "initialized": frozenset({"diagnosed"}),
    "diagnosed": frozenset({"proposed"}),
    "proposed": frozenset({"experimenting"}),
    "experimenting": frozenset({"verifying"}),
    "verifying": frozenset({"iterate", "ready_for_human_review", "exhausted", "failed"}),
    "iterate": frozenset({"diagnosed"}),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EXPERIMENT_ID_PATTERN = _SHA256_PATTERN
_DIAGNOSTIC_FAILURE_NAME = "diagnostic-failure.json"
_EXPERIMENT_FAILURE_NAME = "experiment-failure.json"
_SENSITIVE_TEXT_PATTERN = re.compile(
    r"""(?ix)
    (?:^|[^a-z0-9])
    (?:
        aws[\s_-]?access[\s_-]?key(?:[\s_-]?id)?
        |aws[\s_-]?secret[\s_-]?access[\s_-]?key
        |api[\s_-]?key
        |secret
        |token
        |password
        |credential
        |authorization
    )
    (?:
        \s*(?:=|:)\s*\S+
        |\s+\bis\b\s+\S+
        |\s+(?:bearer|basic)\s+\S+
    )
    """
)
_RAW_SAMPLE_PATTERN = re.compile(
    r"(?i)\b(?:generation_mw|actual_[a-z0-9_]+|target(?:_[a-z0-9_]+)?)\s*(?:=|:)"
)
_DIAGNOSTIC_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_REFERENCE_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
_REFERENCE_PATH_SUFFIXES = frozenset({".json", ".db", ".sqlite", ".md", ".csv"})
_DIAGNOSTIC_ARTIFACT_NAMES = frozenset({"diagnosis", "diagnostic", "diagnostic_report"})
_DIAGNOSTIC_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "config_sha256",
        "provenance",
        "status",
        "dataset",
        "dataset_sha256",
        "dataset_checksum",
        "row_count",
        "plant_count",
        "time_start",
        "time_end",
        "time_coverage",
        "fold_count",
        "chronological_folds",
        "chronological_fold_feasibility",
        "chronological_fold_feasible",
        "fold_feasibility",
        "prediction_time_inputs",
        "available_prediction_time_inputs",
        "available_inputs",
        "history_feature_feasible",
        "history_features_feasible",
        "history_feasible",
        "history_feature_feasibility",
        "baseline_evidence",
        "baseline_reference",
        "baseline_sha256",
        "legacy_evidence",
        "legacy_evidence_reference",
        "legacy_evidence_sha256",
        "missingness",
        "drift_summary",
        "residual_summary",
        "leakage_checks",
        "recommended_profiles",
        "warnings",
        "rejected_conditions",
    }
)
_DIAGNOSTIC_CHECKSUM_KEYS = frozenset(
    {
        "dataset_sha256",
        "dataset_checksum",
        "baseline_sha256",
        "legacy_evidence_sha256",
        "config_sha256",
    }
)
_DIAGNOSTIC_PROVENANCE_KEYS = frozenset(
    {"config_sha256", "dataset_sha256", "legacy_manifest_sha256"}
)
_DIAGNOSTIC_COUNT_KEYS = frozenset({"row_count", "plant_count", "fold_count"})
_DIAGNOSTIC_INPUT_KEYS = frozenset(
    {"prediction_time_inputs", "available_prediction_time_inputs", "available_inputs"}
)
_DIAGNOSTIC_HISTORY_BOOLEAN_KEYS = frozenset(
    {"history_feature_feasible", "history_features_feasible", "history_feasible"}
)
_DIAGNOSTIC_FOLD_BOOLEAN_KEYS = frozenset({"chronological_fold_feasible"})
_DATASET_KEYS = frozenset(
    {
        "sha256",
        "checksum",
        "dataset_sha256",
        "dataset_checksum",
        "row_count",
        "plant_count",
        "time_start",
        "time_end",
        "time_coverage",
    }
)
_TIME_COVERAGE_KEYS = frozenset({"start", "end", "time_start", "time_end"})
_FOLD_FEASIBILITY_KEYS = frozenset({"requested", "available", "fold_count", "feasible", "reason"})
_HISTORY_FEASIBILITY_KEYS = frozenset({"feasible", "reason", "available_rows", "minimum_rows"})
_EVIDENCE_REFERENCE_KEYS = frozenset({"path", "sha256", "checksum", "status"})
_MISSINGNESS_KEYS = frozenset(REQUIRED_COLUMNS)
_DRIFT_SUMMARY_KEYS = frozenset(
    column
    for column in REQUIRED_COLUMNS
    if column.startswith(("forecast_", "ldaps_"))
)
_RESIDUAL_SUMMARY_KEYS = frozenset(
    {"capacity_utilization_mean", "zero_baseline_nmae"}
)
_LEAKAGE_CHECK_KEYS = frozenset(
    {
        "dataset_schema_valid",
        "history_features_strict_prior",
        "prediction_inputs_exclude_actual",
        "prediction_inputs_exclude_target",
    }
)
_PROFILE_ORDER = ("safe_weather", "history_tree", "bounded_search")


class ResearchStateError(ValueError):
    """Raised when persisted research-loop state is invalid or unsafe to resume."""


class _DiagnosticJsonError(ValueError):
    """Raised while parsing diagnostic JSON that is not strict JSON."""


@dataclass(frozen=True)
class ResearchState:
    run_id: str
    status: str
    iteration: int
    used_profiles: tuple[str, ...]
    remaining_profiles: tuple[str, ...]
    artifacts: Mapping[str, str]
    transitions: tuple[Mapping[str, object], ...]
    config_sha256: str | None = None

    def __post_init__(self) -> None:
        _state_run_id(self.run_id)
        if self.config_sha256 is not None and (
            type(self.config_sha256) is not str
            or not _SHA256_PATTERN.fullmatch(self.config_sha256)
        ):
            raise ResearchStateError("config_sha256 must be a lowercase SHA-256 string")
        if type(self.status) is not str or self.status not in _STATUSES:
            raise ResearchStateError("status is not in the research-loop state graph")
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int):
            raise ResearchStateError("iteration must be an integer")
        if self.iteration < 0:
            raise ResearchStateError("iteration must not be negative")

        used_profiles = _profile_tuple(self.used_profiles, "used_profiles")
        remaining_profiles = _profile_tuple(self.remaining_profiles, "remaining_profiles")
        if len(set(used_profiles)) != len(used_profiles):
            raise ResearchStateError("used_profiles contains duplicate profiles")
        if len(set(remaining_profiles)) != len(remaining_profiles):
            raise ResearchStateError("remaining_profiles contains duplicate profiles")
        if set(used_profiles) & set(remaining_profiles):
            raise ResearchStateError("profiles cannot be both used and remaining")
        if self.iteration != len(used_profiles):
            raise ResearchStateError("iteration must equal the number of used_profiles")

        artifacts = _artifact_mapping(self.artifacts)
        transitions = _transition_tuple(self.transitions)
        object.__setattr__(self, "used_profiles", used_profiles)
        object.__setattr__(self, "remaining_profiles", remaining_profiles)
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "transitions", transitions)

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "run_id": self.run_id,
            "status": self.status,
            "iteration": self.iteration,
            "used_profiles": list(self.used_profiles),
            "remaining_profiles": list(self.remaining_profiles),
            "artifacts": dict(self.artifacts),
            "transitions": [dict(event) for event in self.transitions],
        }
        if self.config_sha256 is not None:
            value["config_sha256"] = self.config_sha256
        return value


def load_state(path: Path) -> ResearchState:
    """Load persisted state only when every recorded artifact still matches."""

    state_path = _path_argument(path, "path")
    try:
        with state_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ResearchStateError(f"state JSON is invalid: {exc.msg}") from exc

    if not isinstance(value, Mapping):
        raise ResearchStateError("state must be a JSON object")
    actual_keys = set(value)
    if actual_keys not in {_STATE_REQUIRED_KEYS, _STATE_KEYS}:
        _exact_keys(value, _STATE_KEYS, "state")
    if not isinstance(value["used_profiles"], list):
        raise ResearchStateError("used_profiles must be a list")
    if not isinstance(value["remaining_profiles"], list):
        raise ResearchStateError("remaining_profiles must be a list")
    if not isinstance(value["artifacts"], Mapping):
        raise ResearchStateError("artifacts must be a mapping")
    if not isinstance(value["transitions"], list):
        raise ResearchStateError("transitions must be a list")

    state = ResearchState(
        run_id=value["run_id"],
        status=value["status"],
        iteration=value["iteration"],
        used_profiles=tuple(value["used_profiles"]),
        remaining_profiles=tuple(value["remaining_profiles"]),
        artifacts=value["artifacts"],
        transitions=tuple(value["transitions"]),
        config_sha256=value.get("config_sha256"),
    )
    _validate_transition_history(state)
    verify_recorded_artifacts(state, artifact_paths={})
    return state


def initialize_state(
    config: ResearchLoopConfig,
    *,
    config_sha256: str | None = None,
) -> ResearchState:
    """Create an initial state with an explicit profile/iteration budget."""

    if not isinstance(config, ResearchLoopConfig):
        raise TypeError("config must be a ResearchLoopConfig")
    remaining_profiles = config.profiles[: config.max_iterations]
    return ResearchState(
        run_id=config.run_id,
        status="initialized",
        iteration=0,
        used_profiles=(),
        remaining_profiles=remaining_profiles,
        artifacts={},
        transitions=(),
        config_sha256=config_sha256,
    )


def transition_state(
    state: ResearchState,
    *,
    to_status: str,
    artifact_paths: Mapping[str, Path],
) -> ResearchState:
    """Return the next valid state while retaining checksum-only event evidence."""

    if not isinstance(state, ResearchState):
        raise TypeError("state must be a ResearchState")
    _validate_transition_history(state)
    if state.status in _TERMINAL_STATUSES:
        raise ResearchStateError(f"terminal state {state.status} is immutable")
    if type(to_status) is not str or to_status not in _STATUSES:
        raise ResearchStateError("to_status is not in the research-loop state graph")
    if to_status not in _ALLOWED_TRANSITIONS[state.status]:
        raise ResearchStateError(f"transition {state.status} -> {to_status} is not allowed")

    verify_recorded_artifacts(state, artifact_paths={})
    artifacts = dict(state.artifacts)
    recorded_artifacts = _recorded_artifacts(
        artifact_paths,
        validate_diagnostic=to_status == "diagnosed",
    )
    if to_status == "diagnosed" and not recorded_artifacts:
        raise ResearchStateError("diagnosed transition requires at least one artifact")
    for artifact_path, checksum in recorded_artifacts:
        artifacts[artifact_path] = checksum

    diagnostic_failure = to_status == "diagnosed" and any(
        path.name == _DIAGNOSTIC_FAILURE_NAME for path in artifact_paths.values()
    )
    iteration, used_profiles, remaining_profiles = _advance_cycle(
        state,
        to_status,
        skip_profile=diagnostic_failure,
    )
    profile = used_profiles[-1] if used_profiles else None
    timestamp = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    events = list(state.transitions)
    if recorded_artifacts:
        for artifact_path, checksum in recorded_artifacts:
            events.append(
                _event(
                    timestamp,
                    state.status,
                    to_status,
                    iteration,
                    profile,
                    artifact_path,
                    checksum,
                )
            )
    else:
        events.append(
            _event(
                timestamp,
                state.status,
                to_status,
                iteration,
                profile,
                None,
                None,
            )
        )

    return ResearchState(
        run_id=state.run_id,
        status=to_status,
        iteration=iteration,
        used_profiles=used_profiles,
        remaining_profiles=remaining_profiles,
        artifacts=artifacts,
        transitions=tuple(events),
        config_sha256=state.config_sha256,
    )


def verify_recorded_artifacts(
    state: ResearchState,
    *,
    artifact_paths: Mapping[str, Path],
) -> None:
    """Verify all persisted artifact checksums before state can be resumed."""

    if not isinstance(state, ResearchState):
        raise TypeError("state must be a ResearchState")
    supplied_artifacts = _recorded_artifacts(artifact_paths)
    recorded_paths = set(state.artifacts)
    diagnostic_paths = _diagnostic_artifact_paths(state)

    for artifact_path, _ in supplied_artifacts:
        if artifact_path not in recorded_paths:
            raise ResearchStateError(f"artifact path was not recorded: {artifact_path}")

    for artifact_path, expected_checksum in state.artifacts.items():
        path = Path(artifact_path)
        if not path.exists() or not path.is_file():
            raise ResearchStateError(f"recorded artifact is missing: {artifact_path}")
        actual_checksum = _sha256_file(path)
        if actual_checksum != expected_checksum:
            raise ResearchStateError(f"recorded artifact checksum mismatch: {artifact_path}")
        if artifact_path in diagnostic_paths:
            _validate_diagnostic_artifact(path)
        if path.name == _EXPERIMENT_FAILURE_NAME:
            if path.is_symlink() or path.parent.is_symlink():
                raise ResearchStateError("experiment failure artifact path is unsafe")
            _validate_experiment_failure_artifact(path)


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically replace *path* with finite, canonical JSON written beside it."""

    target = _atomic_destination(path, "path")
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    directory_fd, target_name = _open_destination_directory(target)
    temporary_name: str | None = None
    try:
        _reject_final_target_symlink(directory_fd, target_name)
        temporary_name, temporary_fd = _open_atomic_temporary_file(directory_fd, target_name)
        with os.fdopen(temporary_fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _reject_final_target_symlink(directory_fd, target_name)
        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
        temporary_name = None
    finally:
        if temporary_name is not None:
            os.unlink(temporary_name, dir_fd=directory_fd)
        os.close(directory_fd)


def _advance_cycle(
    state: ResearchState,
    to_status: str,
    *,
    skip_profile: bool = False,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    if skip_profile:
        return state.iteration, state.used_profiles, state.remaining_profiles
    if (state.status, to_status) not in {("initialized", "diagnosed"), ("verifying", "iterate")}:
        return state.iteration, state.used_profiles, state.remaining_profiles
    if not state.remaining_profiles:
        if state.status == "initialized" and to_status == "diagnosed":
            return state.iteration, state.used_profiles, state.remaining_profiles
        raise ResearchStateError("cannot iterate without a remaining profile")
    profile = state.remaining_profiles[0]
    return (
        state.iteration + 1,
        state.used_profiles + (profile,),
        state.remaining_profiles[1:],
    )


def _recorded_artifacts(
    artifact_paths: Mapping[str, Path],
    *,
    validate_diagnostic: bool = False,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(artifact_paths, Mapping):
        raise TypeError("artifact_paths must be a mapping")

    recorded: list[tuple[str, str]] = []
    paths_seen: set[str] = set()
    for name, path in artifact_paths.items():
        if type(name) is not str or not name:
            raise ResearchStateError("artifact names must be nonblank strings")
        if not isinstance(path, Path):
            raise TypeError("artifact paths must be Path values")
        resolved_path = path.resolve()
        artifact_path = str(resolved_path)
        if artifact_path in paths_seen:
            raise ResearchStateError(f"duplicate artifact path: {artifact_path}")
        if not resolved_path.exists() or not resolved_path.is_file():
            raise ResearchStateError(f"artifact path must be an existing file: {artifact_path}")
        if validate_diagnostic or _is_diagnostic_artifact(name):
            _validate_diagnostic_artifact(resolved_path)
        if resolved_path.name == _EXPERIMENT_FAILURE_NAME:
            if resolved_path.is_symlink() or resolved_path.parent.is_symlink():
                raise ResearchStateError("experiment failure artifact path is unsafe")
            _validate_experiment_failure_artifact(resolved_path)
        paths_seen.add(artifact_path)
        recorded.append((artifact_path, _sha256_file(resolved_path)))
    return tuple(sorted(recorded))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_diagnostic_artifact(name: str) -> bool:
    segments = re.split(r"[^a-z0-9]+", name.casefold())
    return bool(_DIAGNOSTIC_ARTIFACT_NAMES & set(segments))


def _validate_experiment_failure_artifact(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_strict_diagnostic_object)
    except (json.JSONDecodeError, UnicodeDecodeError, _DiagnosticJsonError) as exc:
        raise ResearchStateError(f"experiment failure JSON is invalid: {exc}") from exc
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "run_id",
        "experiment_id",
        "iteration",
        "run_state",
    }:
        raise ResearchStateError("experiment failure artifact schema is invalid")
    _state_run_id(value["run_id"])
    if value["schema_version"] != "1":
        raise ResearchStateError("experiment failure schema_version must be exactly '1'")
    if (
        type(value["experiment_id"]) is not str
        or not _EXPERIMENT_ID_PATTERN.fullmatch(value["experiment_id"])
    ):
        raise ResearchStateError("experiment failure experiment_id is invalid")
    if isinstance(value["iteration"], bool) or not isinstance(value["iteration"], int):
        raise ResearchStateError("experiment failure iteration is invalid")
    if value["iteration"] < 1 or value["run_state"] != "failed":
        raise ResearchStateError("experiment failure state is invalid")


def _diagnostic_artifact_paths(state: ResearchState) -> frozenset[str]:
    paths: set[str] = set()
    for event in state.transitions:
        if event["to_status"] != "diagnosed":
            continue
        artifact_path = event["artifact_path"]
        if artifact_path is not None:
            paths.add(artifact_path)
    return frozenset(paths)


def _validate_diagnostic_artifact(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_strict_diagnostic_object,
                parse_constant=_reject_diagnostic_constant,
            )
    except (json.JSONDecodeError, UnicodeDecodeError, _DiagnosticJsonError) as exc:
        raise ResearchStateError(f"diagnostic JSON is invalid: {exc}") from exc

    if not isinstance(value, Mapping):
        raise ResearchStateError("diagnostic artifact must be a JSON object")
    if not value:
        raise ResearchStateError("diagnostic artifact must not be empty")
    _diagnostic_keys(value, _DIAGNOSTIC_KEYS, "diagnostic artifact")

    for field, field_value in value.items():
        _validate_diagnostic_field(field, field_value)


def _strict_diagnostic_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DiagnosticJsonError(f"duplicate diagnostic field: {key}")
        value[key] = item
    return value


def _reject_diagnostic_constant(value: str) -> None:
    raise _DiagnosticJsonError(f"non-finite JSON constant: {value}")


def _diagnostic_keys(value: Mapping[str, object], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ResearchStateError(f"{label} contains forbidden fields: {unknown}")


def _validate_diagnostic_field(field: str, value: object) -> None:
    if field == "schema_version":
        if value != "1":
            raise ResearchStateError("diagnostic schema_version must be exactly '1'")
        return
    if field == "run_id":
        _state_run_id(value)
        return
    if field == "provenance":
        _validate_diagnostic_provenance(value)
        return
    if field == "status":
        _diagnostic_code(value, field)
        return
    if field in _DIAGNOSTIC_CHECKSUM_KEYS:
        _diagnostic_checksum(value, field)
        return
    if field in _DIAGNOSTIC_COUNT_KEYS:
        _diagnostic_count(value, field)
        return
    if field in {"time_start", "time_end"}:
        _diagnostic_timestamp(value, field)
        return
    if field == "time_coverage":
        _validate_time_coverage(value, field)
        return
    if field == "dataset":
        _validate_dataset_aggregate(value)
        return
    if field in {"chronological_folds", "chronological_fold_feasibility", "fold_feasibility"}:
        _validate_fold_feasibility(value, field)
        return
    if field in _DIAGNOSTIC_FOLD_BOOLEAN_KEYS | _DIAGNOSTIC_HISTORY_BOOLEAN_KEYS:
        if type(value) is not bool:
            raise ResearchStateError(f"diagnostic {field} must be a boolean")
        return
    if field == "history_feature_feasibility":
        _validate_history_feasibility(value)
        return
    if field in _DIAGNOSTIC_INPUT_KEYS:
        _validate_prediction_time_inputs(value, field)
        return
    if field == "missingness":
        _validate_ratio_summary(value, field, _MISSINGNESS_KEYS)
        return
    if field == "drift_summary":
        _validate_ratio_summary(value, field, _DRIFT_SUMMARY_KEYS)
        return
    if field == "residual_summary":
        _validate_ratio_summary(value, field, _RESIDUAL_SUMMARY_KEYS)
        return
    if field == "leakage_checks":
        _validate_leakage_checks(value)
        return
    if field == "recommended_profiles":
        _validate_recommended_profiles(value)
        return
    if field in {"warnings", "rejected_conditions"}:
        _validate_diagnostic_codes(value, field)
        return
    if field in {
        "baseline_evidence",
        "baseline_reference",
        "legacy_evidence",
        "legacy_evidence_reference",
    }:
        _validate_evidence_reference(value, field)
        return
    raise ResearchStateError(f"diagnostic artifact contains forbidden field: {field}")


def _validate_dataset_aggregate(value: object) -> None:
    mapping = _diagnostic_mapping(value, _DATASET_KEYS, "diagnostic dataset")
    for field, field_value in mapping.items():
        if field in {"sha256", "checksum", "dataset_sha256", "dataset_checksum"}:
            _diagnostic_checksum(field_value, f"dataset.{field}")
        elif field in {"row_count", "plant_count"}:
            _diagnostic_count(field_value, f"dataset.{field}")
        elif field in {"time_start", "time_end"}:
            _diagnostic_timestamp(field_value, f"dataset.{field}")
        else:
            _validate_time_coverage(field_value, f"dataset.{field}")


def _validate_time_coverage(value: object, label: str) -> None:
    mapping = _diagnostic_mapping(value, _TIME_COVERAGE_KEYS, label)
    if not {"start", "time_start"} & set(mapping) or not {"end", "time_end"} & set(mapping):
        raise ResearchStateError(f"{label} must include a start and end")
    for field, field_value in mapping.items():
        _diagnostic_timestamp(field_value, f"{label}.{field}")


def _validate_fold_feasibility(value: object, label: str) -> None:
    mapping = _diagnostic_mapping(value, _FOLD_FEASIBILITY_KEYS, f"diagnostic {label}")
    for field, field_value in mapping.items():
        if field in {"requested", "available", "fold_count"}:
            _diagnostic_count(field_value, f"{label}.{field}")
        elif field == "feasible":
            if type(field_value) is not bool:
                raise ResearchStateError(f"diagnostic {label}.feasible must be a boolean")
        else:
            _diagnostic_code(field_value, f"{label}.reason")


def _validate_history_feasibility(value: object) -> None:
    mapping = _diagnostic_mapping(value, _HISTORY_FEASIBILITY_KEYS, "diagnostic history_feature_feasibility")
    for field, field_value in mapping.items():
        if field == "feasible":
            if type(field_value) is not bool:
                raise ResearchStateError("diagnostic history_feature_feasibility.feasible must be a boolean")
        elif field in {"available_rows", "minimum_rows"}:
            _diagnostic_count(field_value, f"history_feature_feasibility.{field}")
        else:
            _diagnostic_code(field_value, "history_feature_feasibility.reason")


def _validate_ratio_summary(
    value: object,
    label: str,
    expected_keys: frozenset[str],
) -> None:
    mapping = _diagnostic_mapping(value, expected_keys, f"diagnostic {label}")
    if set(mapping) != set(expected_keys):
        raise ResearchStateError(
            f"diagnostic {label} must contain its exact aggregate keys"
        )
    for key, field_value in mapping.items():
        if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
            raise ResearchStateError(f"diagnostic {label}.{key} must be numeric")
        number = float(field_value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ResearchStateError(
                f"diagnostic {label}.{key} must be finite and between 0 and 1"
            )


def _validate_leakage_checks(value: object) -> None:
    mapping = _diagnostic_mapping(
        value,
        _LEAKAGE_CHECK_KEYS,
        "diagnostic leakage_checks",
    )
    if set(mapping) != set(_LEAKAGE_CHECK_KEYS):
        raise ResearchStateError(
            "diagnostic leakage_checks must contain its exact aggregate keys"
        )
    for key, field_value in mapping.items():
        if type(field_value) is not bool:
            raise ResearchStateError(f"diagnostic leakage_checks.{key} must be a boolean")


def _validate_recommended_profiles(value: object) -> None:
    if not isinstance(value, (list, tuple)):
        raise ResearchStateError("diagnostic recommended_profiles must be a list")
    if len(set(value)) != len(value):
        raise ResearchStateError("diagnostic recommended_profiles contains duplicates")
    if any(type(profile) is not str or profile not in SUPPORTED_PROFILES for profile in value):
        raise ResearchStateError(
            "diagnostic recommended_profiles contains an unsupported profile"
        )
    expected = tuple(profile for profile in _PROFILE_ORDER if profile in value)
    if tuple(value) != expected:
        raise ResearchStateError(
            "diagnostic recommended_profiles must use deterministic supported ordering"
        )


def _validate_evidence_reference(value: object, label: str) -> None:
    if value is None:
        return
    mapping = _diagnostic_mapping(value, _EVIDENCE_REFERENCE_KEYS, f"diagnostic {label}")
    for field, field_value in mapping.items():
        if field in {"sha256", "checksum"}:
            _diagnostic_checksum(field_value, f"{label}.{field}")
        elif field == "path":
            _diagnostic_path(field_value, f"{label}.path")
        else:
            _diagnostic_code(field_value, f"{label}.status")


def _diagnostic_mapping(
    value: object,
    allowed: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ResearchStateError(f"{label} must be a mapping")
    if not value:
        raise ResearchStateError(f"{label} must not be empty")
    _diagnostic_keys(value, allowed, label)
    return value


def _diagnostic_checksum(value: object, label: str) -> None:
    if type(value) is not str or not _SHA256_PATTERN.fullmatch(value):
        raise ResearchStateError(f"diagnostic {label} must be a lowercase SHA-256 string")


def _validate_diagnostic_provenance(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _DIAGNOSTIC_PROVENANCE_KEYS:
        raise ResearchStateError("diagnostic provenance must contain exact binding keys")
    for key, item in value.items():
        _diagnostic_checksum(item, f"diagnostic provenance.{key}")


def _diagnostic_count(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchStateError(f"diagnostic {label} must be a nonnegative integer")


def _validate_prediction_time_inputs(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise ResearchStateError(f"diagnostic {label} must be a list")
    for item in value:
        _diagnostic_code(item, label)
        if item == "generation_mw" or item.startswith(("actual_", "target_")):
            raise ResearchStateError(f"diagnostic {label} contains a target or actual input")


def _validate_diagnostic_codes(value: object, label: str) -> None:
    if not isinstance(value, list):
        raise ResearchStateError(f"diagnostic {label} must be a list")
    for item in value:
        _diagnostic_code(item, label)


def _diagnostic_code(value: object, label: str) -> None:
    if type(value) is not str or not value:
        raise ResearchStateError(f"diagnostic {label} must contain nonblank strings")
    if _SENSITIVE_TEXT_PATTERN.search(value) or _RAW_SAMPLE_PATTERN.search(value):
        raise ResearchStateError(f"diagnostic {label} contains sensitive or raw sample content")
    if not _DIAGNOSTIC_CODE_PATTERN.fullmatch(value):
        raise ResearchStateError(f"diagnostic {label} must contain lowercase aggregate codes")


def _diagnostic_timestamp(value: object, label: str) -> None:
    if type(value) is not str or not value:
        raise ResearchStateError(f"diagnostic {label} must be an ISO timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchStateError(f"diagnostic {label} must be an ISO timestamp") from exc


def _diagnostic_path(value: object, label: str) -> None:
    if type(value) is not str or not value:
        raise ResearchStateError(f"diagnostic {label} must be a nonblank string")
    path = Path(value)
    if _SENSITIVE_TEXT_PATTERN.search(value) or _RAW_SAMPLE_PATTERN.search(value):
        raise ResearchStateError(f"diagnostic {label} contains sensitive or raw sample content")
    if (
        not _REFERENCE_PATH_PATTERN.fullmatch(value)
        or any(part == ".." for part in path.parts)
        or path.suffix.casefold() not in _REFERENCE_PATH_SUFFIXES
    ):
        raise ResearchStateError(f"diagnostic {label} must be a safe artifact file path")


def _state_run_id(value: object) -> str:
    try:
        return validate_run_id(value)
    except ResearchContractError as exc:
        raise ResearchStateError(str(exc)) from exc


def _event(
    timestamp: str,
    from_status: str,
    to_status: str,
    iteration: int,
    profile: str | None,
    artifact_path: str | None,
    checksum: str | None,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "from_status": from_status,
        "to_status": to_status,
        "iteration": iteration,
        "profile": profile,
        "artifact_path": artifact_path,
        "sha256": checksum,
    }


def _validate_transition_history(state: ResearchState) -> None:
    status = "initialized"
    iteration = 0
    used_profiles: tuple[str, ...] = ()
    remaining_profiles = state.used_profiles + state.remaining_profiles
    recorded_paths: set[str] = set()

    for events in _transition_groups(state.transitions):
        event = events[0]
        from_status = event["from_status"]
        to_status = event["to_status"]
        if from_status != status:
            raise ResearchStateError("transition history is not continuous")
        if status in _TERMINAL_STATUSES:
            raise ResearchStateError("transition history changes a terminal state")
        if to_status not in _ALLOWED_TRANSITIONS[status]:
            raise ResearchStateError(f"transition {status} -> {to_status} is not allowed")

        expected_iteration = iteration
        expected_used = used_profiles
        expected_remaining = remaining_profiles
        diagnostic_failure = to_status == "diagnosed" and any(
            artifact_event["artifact_path"] is not None
            and Path(str(artifact_event["artifact_path"])).name == _DIAGNOSTIC_FAILURE_NAME
            for artifact_event in events
        )
        if (
            (status, to_status) in {("initialized", "diagnosed"), ("verifying", "iterate")}
            and not diagnostic_failure
        ):
            if not remaining_profiles:
                if status == "initialized" and to_status == "diagnosed":
                    expected_profile = None
                else:
                    raise ResearchStateError(
                        "transition history iterates without a remaining profile"
                    )
            else:
                expected_iteration += 1
                expected_used = used_profiles + (remaining_profiles[0],)
                expected_remaining = remaining_profiles[1:]
        expected_profile = expected_used[-1] if expected_used else None

        if event["iteration"] != expected_iteration or event["profile"] != expected_profile:
            raise ResearchStateError("transition history has invalid iteration or profile")
        for artifact_event in events:
            if _event_group_key(artifact_event) != _event_group_key(event):
                raise ResearchStateError("transition history splits a transition event")
            artifact_path = artifact_event["artifact_path"]
            checksum = artifact_event["sha256"]
            if artifact_path is not None:
                if state.artifacts.get(artifact_path) != checksum:
                    raise ResearchStateError("transition artifact checksum does not match state")
                recorded_paths.add(artifact_path)
        if to_status == "diagnosed" and not any(
            artifact_event["artifact_path"] is not None for artifact_event in events
        ):
            raise ResearchStateError("diagnosed transition requires artifact evidence")

        status = to_status
        iteration = expected_iteration
        used_profiles = expected_used
        remaining_profiles = expected_remaining

    if (
        status != state.status
        or iteration != state.iteration
        or used_profiles != state.used_profiles
        or remaining_profiles != state.remaining_profiles
    ):
        raise ResearchStateError("transition history does not lead to the recorded state")
    if recorded_paths != set(state.artifacts):
        raise ResearchStateError("transition history does not account for every artifact")


def _transition_groups(
    transitions: tuple[Mapping[str, object], ...],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    groups: list[tuple[Mapping[str, object], ...]] = []
    current: list[Mapping[str, object]] = []
    current_key: tuple[object, ...] | None = None
    for event in transitions:
        event_key = _event_group_key(event)
        if current and event_key != current_key:
            groups.append(tuple(current))
            current = []
        current.append(event)
        current_key = event_key
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _event_group_key(event: Mapping[str, object]) -> tuple[object, ...]:
    return (
        event["timestamp"],
        event["from_status"],
        event["to_status"],
        event["iteration"],
        event["profile"],
    )


def _profile_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ResearchStateError(f"{label} must be a tuple")
    for profile in value:
        if type(profile) is not str or profile not in SUPPORTED_PROFILES:
            raise ResearchStateError(f"{label} contains an unsupported profile")
    return value


def _artifact_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ResearchStateError("artifacts must be a mapping")

    artifacts: dict[str, str] = {}
    for artifact_path, checksum in value.items():
        if type(artifact_path) is not str or not Path(artifact_path).is_absolute():
            raise ResearchStateError("artifact paths must be absolute strings")
        if type(checksum) is not str or not _SHA256_PATTERN.fullmatch(checksum):
            raise ResearchStateError("artifact checksums must be lowercase SHA-256 strings")
        artifacts[artifact_path] = checksum
    return artifacts


def _transition_tuple(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, tuple):
        raise ResearchStateError("transitions must be a tuple")
    return tuple(MappingProxyType(_transition_event(event)) for event in value)


def _transition_event(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ResearchStateError("transition events must be mappings")
    _exact_keys(value, _EVENT_KEYS, "transition event")

    timestamp = value["timestamp"]
    from_status = value["from_status"]
    to_status = value["to_status"]
    iteration = value["iteration"]
    profile = value["profile"]
    artifact_path = value["artifact_path"]
    checksum = value["sha256"]

    if type(timestamp) is not str or not timestamp:
        raise ResearchStateError("transition timestamp must be a nonblank string")
    if type(from_status) is not str or from_status not in _STATUSES:
        raise ResearchStateError("transition from_status is invalid")
    if type(to_status) is not str or to_status not in _STATUSES:
        raise ResearchStateError("transition to_status is invalid")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ResearchStateError("transition iteration is invalid")
    if profile is not None and (type(profile) is not str or profile not in SUPPORTED_PROFILES):
        raise ResearchStateError("transition profile is invalid")
    if (artifact_path is None) != (checksum is None):
        raise ResearchStateError("transition artifact path and checksum must be provided together")
    if artifact_path is not None:
        if type(artifact_path) is not str or not Path(artifact_path).is_absolute():
            raise ResearchStateError("transition artifact_path must be an absolute string")
        if type(checksum) is not str or not _SHA256_PATTERN.fullmatch(checksum):
            raise ResearchStateError("transition sha256 is invalid")

    return {
        "timestamp": timestamp,
        "from_status": from_status,
        "to_status": to_status,
        "iteration": iteration,
        "profile": profile,
        "artifact_path": artifact_path,
        "sha256": checksum,
    }


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ResearchStateError(f"{label} unknown keys: {unknown}")
    raise ResearchStateError(f"{label} missing keys: {missing}")


def _path_argument(value: Path, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be a Path")
    return value.resolve()


def _atomic_destination(value: Path, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be a Path")
    return value if value.is_absolute() else Path.cwd() / value


def _open_destination_directory(target: Path) -> tuple[int, str]:
    if not target.is_absolute() or not target.name or target.name in {".", ".."}:
        raise ResearchStateError("atomic JSON destination must name a file")
    if any(component == ".." for component in target.parts):
        raise ResearchStateError("atomic JSON destination must not contain parent traversal")

    directory_fd = os.open(target.anchor, _directory_open_flags())
    try:
        for component in target.parts[1:-1]:
            child_fd = _open_or_create_directory(directory_fd, component)
            os.close(directory_fd)
            directory_fd = child_fd
    except (OSError, ResearchStateError):
        os.close(directory_fd)
        raise
    return directory_fd, target.name


def _directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ResearchStateError("atomic JSON symlink protection is unavailable")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(info.st_mode):
        raise ResearchStateError(f"atomic JSON destination contains a symlink: {name}")
    if not stat.S_ISDIR(info.st_mode):
        raise ResearchStateError(f"atomic JSON destination parent is not a directory: {name}")
    return os.open(name, _directory_open_flags(), dir_fd=parent_fd)


def _reject_final_target_symlink(directory_fd: int, target_name: str) -> None:
    try:
        info = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode):
        raise ResearchStateError(f"atomic JSON destination contains a symlink: {target_name}")


def _open_atomic_temporary_file(directory_fd: int, target_name: str) -> tuple[str, int]:
    for _ in range(10):
        temporary_name = f".{target_name}.{secrets.token_hex(16)}.tmp"
        try:
            temporary_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                mode=0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            continue
        return temporary_name, temporary_fd
    raise ResearchStateError("unable to create atomic JSON temporary file")


__all__ = [
    "ResearchState",
    "ResearchStateError",
    "atomic_write_json",
    "initialize_state",
    "load_state",
    "transition_state",
    "verify_recorded_artifacts",
]
