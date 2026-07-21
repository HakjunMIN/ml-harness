from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from power_forecasting.research_contracts import (
    ResearchContractError,
    SUPPORTED_PROFILES,
    load_research_loop_config,
)
from power_forecasting.research_state import (
    ResearchStateError,
    atomic_write_json,
    initialize_state,
    load_state,
    transition_state,
    verify_recorded_artifacts,
)


@pytest.fixture
def contract_paths(tmp_path: Path) -> dict[str, Path]:
    repository_root = tmp_path / "repository"
    for protected in ("src", "tests", "docs", ".agents/fixtures", ".agents/skills", ".agents/scripts"):
        (repository_root / protected).mkdir(parents=True, exist_ok=True)

    config_path = repository_root / "config" / "research-loop.json"
    input_dir = config_path.parent / "inputs"
    input_dir.mkdir(parents=True)
    dataset_path = input_dir / "dataset.csv"
    dataset_path.write_text("plant_id,timestamp\nplant_01,2026-01-01T00:00:00Z\n", encoding="utf-8")
    legacy_manifest_path = input_dir / "legacy-manifest.json"
    legacy_manifest_path.write_text('{"schema_version":"1"}\n', encoding="utf-8")

    return {
        "repository_root": repository_root,
        "config_path": config_path,
        "dataset_path": dataset_path,
        "legacy_manifest_path": legacy_manifest_path,
    }


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1",
        "run_id": "run_001",
        "dataset_path": "inputs/dataset.csv",
        "legacy_manifest_path": "inputs/legacy-manifest.json",
        "run_dir": "../.agents/runs/run_001",
        "profiles": ["safe_weather", "history_tree"],
        "max_iterations": 2,
        "fold_count": 2,
        "objective": "nmae",
        "minimum_improvement": 0.01,
        "max_plant_regression": 0.03,
    }
    payload.update(overrides)
    return payload


def _load(payload: dict[str, object], paths: dict[str, Path]):
    return load_research_loop_config(
        payload,
        config_path=paths["config_path"],
        repository_root=paths["repository_root"],
    )


def test_valid_config_resolves_relative_paths_from_configuration_directory(
    contract_paths: dict[str, Path],
) -> None:
    config = _load(_payload(), contract_paths)

    assert config.dataset_path == str(contract_paths["dataset_path"].resolve())
    assert config.legacy_manifest_path == str(contract_paths["legacy_manifest_path"].resolve())
    assert config.run_dir == str(
        (contract_paths["repository_root"] / ".agents" / "runs" / "run_001").resolve()
    )
    assert config.profiles == ("safe_weather", "history_tree")


def test_config_rejects_unknown_keys(contract_paths: dict[str, Path]) -> None:
    with pytest.raises(ResearchContractError, match="unknown keys"):
        _load(_payload(unexpected=True), contract_paths)


def test_config_rejects_duplicate_profiles(contract_paths: dict[str, Path]) -> None:
    with pytest.raises(ResearchContractError, match="duplicate"):
        _load(_payload(profiles=["safe_weather", "safe_weather"]), contract_paths)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_iterations", 0),
        ("max_iterations", 11),
        ("fold_count", 0),
        ("fold_count", 11),
    ],
)
def test_config_rejects_iteration_and_fold_counts_outside_bounded_range(
    contract_paths: dict[str, Path], field: str, value: int
) -> None:
    with pytest.raises(ResearchContractError, match=field):
        _load(_payload(**{field: value}), contract_paths)


@pytest.mark.parametrize("run_id", ["../escape", "run/id", "unsafe id", ".", ".."])
def test_config_rejects_unsafe_run_ids(contract_paths: dict[str, Path], run_id: str) -> None:
    with pytest.raises(ResearchContractError, match="run_id"):
        _load(_payload(run_id=run_id), contract_paths)


def test_supported_profiles_are_exact_allowlist(contract_paths: dict[str, Path]) -> None:
    assert SUPPORTED_PROFILES == frozenset({"safe_weather", "history_tree", "bounded_search"})

    with pytest.raises(ResearchContractError, match="unsupported"):
        _load(_payload(profiles=["unbounded_search"]), contract_paths)


def test_config_requires_explicit_existing_input_paths(contract_paths: dict[str, Path]) -> None:
    with pytest.raises(ResearchContractError, match="dataset_path"):
        _load(_payload(dataset_path="inputs/missing.csv"), contract_paths)

    with pytest.raises(ResearchContractError, match="legacy_manifest_path"):
        _load(_payload(legacy_manifest_path="inputs/missing-manifest.json"), contract_paths)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_improvement", math.nan),
        ("minimum_improvement", math.inf),
        ("minimum_improvement", -0.001),
        ("minimum_improvement", 1.001),
        ("minimum_improvement", "0.01"),
        ("minimum_improvement", 10**1000),
        ("max_plant_regression", math.nan),
        ("max_plant_regression", -0.001),
        ("max_plant_regression", 1.001),
        ("max_plant_regression", "0.03"),
        ("max_plant_regression", 10**1000),
    ],
)
def test_config_requires_finite_aidm_compatible_thresholds(
    contract_paths: dict[str, Path], field: str, value: object
) -> None:
    with pytest.raises(ResearchContractError, match=field):
        _load(_payload(**{field: value}), contract_paths)


@pytest.mark.parametrize(
    "run_dir",
    [
        "../src/research-output",
        "../tests/research-output",
        "../docs/research-output",
        "../.agents/fixtures/research-output",
        "../.agents/skills/research-output",
        "../.agents/scripts/research-output",
    ],
)
def test_config_rejects_run_directories_overlapping_protected_repository_content(
    contract_paths: dict[str, Path], run_dir: str
) -> None:
    with pytest.raises(ResearchContractError, match="run_dir"):
        _load(_payload(run_dir=run_dir), contract_paths)


@pytest.mark.parametrize(
    ("run_dir", "is_allowed"),
    [
        ("../.agents/runs/run_001", True),
        ("../.agents/output/run_001", True),
        ("../.agents/cache/run_001", False),
        ("../.agents/research-output", False),
    ],
)
def test_config_permits_only_designated_repository_agents_destinations(
    contract_paths: dict[str, Path], run_dir: str, is_allowed: bool
) -> None:
    payload = _payload(run_dir=run_dir)

    if is_allowed:
        assert _load(payload, contract_paths).run_dir.endswith(run_dir.removeprefix("../"))
    else:
        with pytest.raises(ResearchContractError, match=r"\.agents"):
            _load(payload, contract_paths)


def _config_for_state(contract_paths: dict[str, Path]):
    return _load(_payload(), contract_paths)


def _diagnosis_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "diagnosis.json"
    path.write_text(
        json.dumps(
            {
                "dataset_sha256": "a" * 64,
                "row_count": 3,
                "plant_count": 1,
                "target_samples": ["generation_mw=42"],
                "environment": {"API_TOKEN": "not-for-state"},
            }
        ),
        encoding="utf-8",
    )
    return path


def _state_at_verifying(contract_paths: dict[str, Path], tmp_path: Path):
    state = initialize_state(_config_for_state(contract_paths))
    state = transition_state(
        state,
        to_status="diagnosed",
        artifact_paths={"diagnosis": _diagnosis_artifact(tmp_path)},
    )
    state = transition_state(state, to_status="proposed", artifact_paths={})
    state = transition_state(state, to_status="experimenting", artifact_paths={})
    return transition_state(state, to_status="verifying", artifact_paths={})


def test_state_rejects_transitions_outside_graph(contract_paths: dict[str, Path]) -> None:
    state = initialize_state(_config_for_state(contract_paths))

    with pytest.raises(ResearchStateError, match="transition"):
        transition_state(state, to_status="proposed", artifact_paths={})


def test_iterate_explicitly_allocates_next_bounded_profile(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    state = _state_at_verifying(contract_paths, tmp_path)
    assert state.iteration == 1
    assert state.used_profiles == ("safe_weather",)
    assert state.remaining_profiles == ("history_tree",)

    iterating = transition_state(state, to_status="iterate", artifact_paths={})

    assert iterating.status == "iterate"
    assert iterating.iteration == 2
    assert iterating.used_profiles == ("safe_weather", "history_tree")
    assert iterating.remaining_profiles == ()
    assert transition_state(iterating, to_status="diagnosed", artifact_paths={}).iteration == 2


@pytest.mark.parametrize("terminal_status", ["ready_for_human_review", "exhausted", "failed"])
def test_terminal_states_are_immutable(
    contract_paths: dict[str, Path], terminal_status: str
) -> None:
    state = replace(initialize_state(_config_for_state(contract_paths)), status=terminal_status)

    with pytest.raises(ResearchStateError, match="terminal"):
        transition_state(state, to_status="diagnosed", artifact_paths={})


def test_state_records_only_checksum_metadata_for_diagnostic_artifacts(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    artifact = _diagnosis_artifact(tmp_path)
    state = transition_state(
        initialize_state(_config_for_state(contract_paths)),
        to_status="diagnosed",
        artifact_paths={"diagnosis": artifact},
    )

    event = state.transitions[-1]
    serialized = json.dumps(state.to_dict(), sort_keys=True)

    assert set(event) == {
        "timestamp",
        "from_status",
        "to_status",
        "iteration",
        "profile",
        "artifact_path",
        "sha256",
    }
    assert event["artifact_path"] == str(artifact.resolve())
    assert event["sha256"] == state.artifacts[str(artifact.resolve())]
    assert "generation_mw=42" not in serialized
    assert "not-for-state" not in serialized
    assert "target_samples" not in serialized
    assert "environment" not in serialized


def test_transitions_are_append_only_and_resume_rejects_checksum_mismatch(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    artifact = _diagnosis_artifact(tmp_path)
    diagnosed = transition_state(
        initialize_state(_config_for_state(contract_paths)),
        to_status="diagnosed",
        artifact_paths={"diagnosis": artifact},
    )
    proposed = transition_state(diagnosed, to_status="proposed", artifact_paths={})
    state_path = tmp_path / "state.json"

    atomic_write_json(state_path, proposed.to_dict())

    assert load_state(state_path) == proposed
    assert proposed.transitions[: len(diagnosed.transitions)] == diagnosed.transitions
    verify_recorded_artifacts(proposed, artifact_paths={"diagnosis": artifact})

    artifact.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(ResearchStateError, match="checksum"):
        load_state(state_path)


@pytest.mark.parametrize(
    ("status", "transitions"),
    [
        (
            "ready_for_human_review",
            [
                {
                    "timestamp": "2026-07-22T00:00:00+00:00",
                    "from_status": "initialized",
                    "to_status": "ready_for_human_review",
                    "iteration": 0,
                    "profile": None,
                    "artifact_path": None,
                    "sha256": None,
                }
            ],
        ),
        (
            "diagnosed",
            [
                {
                    "timestamp": "2026-07-22T00:00:00+00:00",
                    "from_status": "initialized",
                    "to_status": "failed",
                    "iteration": 0,
                    "profile": None,
                    "artifact_path": None,
                    "sha256": None,
                },
                {
                    "timestamp": "2026-07-22T00:00:01+00:00",
                    "from_status": "failed",
                    "to_status": "diagnosed",
                    "iteration": 0,
                    "profile": None,
                    "artifact_path": None,
                    "sha256": None,
                },
            ],
        ),
    ],
)
def test_resume_rejects_forged_or_terminal_transition_history(
    contract_paths: dict[str, Path],
    tmp_path: Path,
    status: str,
    transitions: list[dict[str, object]],
) -> None:
    payload = initialize_state(_config_for_state(contract_paths)).to_dict()
    payload["status"] = status
    payload["transitions"] = transitions
    state_path = tmp_path / "forged-state.json"
    atomic_write_json(state_path, payload)

    with pytest.raises(ResearchStateError, match="transition"):
        load_state(state_path)
