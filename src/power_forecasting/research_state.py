from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

from power_forecasting.research_contracts import ResearchLoopConfig, SUPPORTED_PROFILES


_STATE_KEYS = frozenset(
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
    "initialized": frozenset({"diagnosed", "failed"}),
    "diagnosed": frozenset({"proposed", "failed"}),
    "proposed": frozenset({"experimenting", "failed"}),
    "experimenting": frozenset({"verifying", "failed"}),
    "verifying": frozenset({"iterate", "ready_for_human_review", "exhausted", "failed"}),
    "iterate": frozenset({"diagnosed", "failed"}),
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ResearchStateError(ValueError):
    """Raised when persisted research-loop state is invalid or unsafe to resume."""


@dataclass(frozen=True)
class ResearchState:
    run_id: str
    status: str
    iteration: int
    used_profiles: tuple[str, ...]
    remaining_profiles: tuple[str, ...]
    artifacts: Mapping[str, str]
    transitions: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id:
            raise ResearchStateError("run_id must be a nonblank string")
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
        return {
            "run_id": self.run_id,
            "status": self.status,
            "iteration": self.iteration,
            "used_profiles": list(self.used_profiles),
            "remaining_profiles": list(self.remaining_profiles),
            "artifacts": dict(self.artifacts),
            "transitions": [dict(event) for event in self.transitions],
        }


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
    )
    _validate_transition_history(state)
    verify_recorded_artifacts(state, artifact_paths={})
    return state


def initialize_state(config: ResearchLoopConfig) -> ResearchState:
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
    if state.status in _TERMINAL_STATUSES:
        raise ResearchStateError(f"terminal state {state.status} is immutable")
    if type(to_status) is not str or to_status not in _STATUSES:
        raise ResearchStateError("to_status is not in the research-loop state graph")
    if to_status not in _ALLOWED_TRANSITIONS[state.status]:
        raise ResearchStateError(f"transition {state.status} -> {to_status} is not allowed")

    verify_recorded_artifacts(state, artifact_paths={})
    artifacts = dict(state.artifacts)
    recorded_artifacts = _recorded_artifacts(artifact_paths)
    for artifact_path, checksum in recorded_artifacts:
        artifacts[artifact_path] = checksum

    iteration, used_profiles, remaining_profiles = _advance_cycle(state, to_status)
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


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically replace *path* with finite, canonical JSON written beside it."""

    target = _path_argument(path, "path")
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping")
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(value, handle, allow_nan=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _advance_cycle(state: ResearchState, to_status: str) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    if (state.status, to_status) not in {("initialized", "diagnosed"), ("verifying", "iterate")}:
        return state.iteration, state.used_profiles, state.remaining_profiles
    if not state.remaining_profiles:
        raise ResearchStateError("cannot iterate without a remaining profile")
    profile = state.remaining_profiles[0]
    return (
        state.iteration + 1,
        state.used_profiles + (profile,),
        state.remaining_profiles[1:],
    )


def _recorded_artifacts(artifact_paths: Mapping[str, Path]) -> tuple[tuple[str, str], ...]:
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
        paths_seen.add(artifact_path)
        recorded.append((artifact_path, _sha256_file(resolved_path)))
    return tuple(sorted(recorded))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if (status, to_status) in {("initialized", "diagnosed"), ("verifying", "iterate")}:
            if not remaining_profiles:
                raise ResearchStateError("transition history iterates without a remaining profile")
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


__all__ = [
    "ResearchState",
    "ResearchStateError",
    "atomic_write_json",
    "initialize_state",
    "load_state",
    "transition_state",
    "verify_recorded_artifacts",
]
