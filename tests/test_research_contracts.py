from __future__ import annotations

import hashlib
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
        "run_dir": "../runs/run_001",
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
        (contract_paths["repository_root"] / "runs" / "run_001").resolve()
    )
    assert config.profiles == ("safe_weather", "history_tree")


def test_config_accepts_opt_in_agent_proposals(contract_paths: dict[str, Path]) -> None:
    config = _load(_payload(agent_proposals=True), contract_paths)

    assert config.agent_proposals is True

    with pytest.raises(ResearchContractError, match="agent_proposals"):
        _load(_payload(agent_proposals="true"), contract_paths)


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
        ("../runs", True),
        ("../runs/run_001", True),
        ("../outputs", True),
        ("../outputs/run_001", True),
        ("../.agents/runs/run_001", False),
        ("../.agents/output/run_001", False),
        ("../.agents/cache/run_001", False),
        ("../.agents/research-output", False),
    ],
)
def test_config_permits_only_designated_repository_artifact_destinations(
    contract_paths: dict[str, Path], run_dir: str, is_allowed: bool
) -> None:
    payload = _payload(run_dir=run_dir)

    if is_allowed:
        assert _load(payload, contract_paths).run_dir.endswith(run_dir.removeprefix("../"))
    else:
        with pytest.raises(ResearchContractError, match="run_dir"):
            _load(payload, contract_paths)


@pytest.mark.parametrize(
    "run_dir",
    [
        "..",
        "../.git",
        "../.agents",
        "/outside/research-run",
    ],
)
def test_config_rejects_repository_root_git_and_external_run_directories(
    contract_paths: dict[str, Path], run_dir: str
) -> None:
    with pytest.raises(ResearchContractError, match="run_dir"):
        _load(_payload(run_dir=run_dir), contract_paths)


def test_config_rejects_symlinked_run_destination_and_parent(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    runs = contract_paths["repository_root"] / "runs"
    runs.mkdir(parents=True)

    destination = runs / "linked-destination"
    destination.symlink_to(tmp_path / "destination", target_is_directory=True)
    with pytest.raises(ResearchContractError, match="symlink"):
        _load(_payload(run_dir="../runs/linked-destination"), contract_paths)

    parent = contract_paths["repository_root"] / "linked-parent"
    parent.symlink_to(runs, target_is_directory=True)
    with pytest.raises(ResearchContractError, match="symlink"):
        _load(_payload(run_dir="../linked-parent/run"), contract_paths)


def test_config_rejects_symlinked_allowed_root(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    (contract_paths["repository_root"] / "runs").symlink_to(tmp_path, target_is_directory=True)

    with pytest.raises(ResearchContractError, match="symlink"):
        _load(_payload(run_dir="../runs/run_001"), contract_paths)


def test_config_requires_explicit_run_dir(contract_paths: dict[str, Path]) -> None:
    with pytest.raises(ResearchContractError, match="missing keys"):
        _load(_payload_without("run_dir"), contract_paths)
    with pytest.raises(ResearchContractError, match="run_dir"):
        _load(_payload(run_dir=""), contract_paths)


def _payload_without(key: str) -> dict[str, object]:
    payload = _payload()
    del payload[key]
    return payload


def _config_for_state(contract_paths: dict[str, Path]):
    return _load(_payload(), contract_paths)


def _diagnosis_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1",
        "run_id": "run_001",
        "dataset_sha256": "a" * 64,
        "row_count": 3,
        "plant_count": 1,
        "time_coverage": {
            "start": "2026-01-01T00:00:00+00:00",
            "end": "2026-01-01T02:00:00+00:00",
        },
        "chronological_folds": {"requested": 2, "feasible": True},
        "prediction_time_inputs": ["forecast_irradiance", "forecast_cloud_cover"],
        "history_feature_feasible": True,
        "baseline_evidence": {"sha256": "b" * 64, "status": "available"},
        "legacy_evidence": None,
        "warnings": ["baseline_evidence_available"],
        "rejected_conditions": [],
    }
    payload.update(overrides)
    return payload


def _diagnosis_artifact(tmp_path: Path, **overrides: object) -> Path:
    path = tmp_path / "diagnosis.json"
    path.write_text(json.dumps(_diagnosis_payload(**overrides)), encoding="utf-8")
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


def test_state_graph_is_exact() -> None:
    import power_forecasting.research_state as research_state

    assert research_state._ALLOWED_TRANSITIONS == {
        "initialized": frozenset({"diagnosed"}),
        "diagnosed": frozenset({"proposed", "awaiting_proposal"}),
        "awaiting_proposal": frozenset({"proposed"}),
        "proposed": frozenset({"experimenting"}),
        "experimenting": frozenset({"verifying"}),
        "verifying": frozenset({"iterate", "ready_for_human_review", "exhausted", "failed"}),
        "iterate": frozenset({"diagnosed"}),
    }


@pytest.mark.parametrize("source_status", ["initialized", "diagnosed", "proposed", "experimenting", "iterate"])
def test_only_verifying_may_transition_to_failed(
    contract_paths: dict[str, Path], source_status: str
) -> None:
    state = replace(initialize_state(_config_for_state(contract_paths)), status=source_status)

    with pytest.raises(ResearchStateError, match="transition"):
        transition_state(state, to_status="failed", artifact_paths={})


def test_verifying_may_transition_to_failed(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    failed = transition_state(_state_at_verifying(contract_paths, tmp_path), to_status="failed", artifact_paths={})

    assert failed.status == "failed"


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

    with pytest.raises(ResearchStateError, match="diagnosed.*artifact"):
        transition_state(iterating, to_status="diagnosed", artifact_paths={})


def test_agent_proposal_state_cycles_profiles_to_iteration_budget(
    contract_paths: dict[str, Path],
    tmp_path: Path,
) -> None:
    config = replace(
        _config_for_state(contract_paths),
        profiles=("safe_weather",),
        max_iterations=3,
        agent_proposals=True,
    )
    state = initialize_state(config)

    assert state.remaining_profiles == ("safe_weather", "safe_weather", "safe_weather")

    for expected_iteration in (1, 2, 3):
        state = transition_state(
            state,
            to_status="diagnosed",
            artifact_paths={"diagnosis": _diagnosis_artifact(tmp_path)},
        )
        assert state.iteration == expected_iteration
        if expected_iteration < 3:
            state = transition_state(state, to_status="proposed", artifact_paths={})
            state = transition_state(state, to_status="experimenting", artifact_paths={})
            state = transition_state(state, to_status="verifying", artifact_paths={})
            state = transition_state(state, to_status="iterate", artifact_paths={})

    assert state.used_profiles == ("safe_weather", "safe_weather", "safe_weather")
    assert state.remaining_profiles == ()


@pytest.mark.parametrize("terminal_status", ["ready_for_human_review", "exhausted", "failed"])
def test_terminal_states_are_immutable(
    contract_paths: dict[str, Path], tmp_path: Path, terminal_status: str
) -> None:
    state = transition_state(
        _state_at_verifying(contract_paths, tmp_path),
        to_status=terminal_status,
        artifact_paths={},
    )

    with pytest.raises(ResearchStateError, match="terminal"):
        transition_state(state, to_status="diagnosed", artifact_paths={})


def test_diagnosed_transition_requires_diagnostic_artifact(contract_paths: dict[str, Path]) -> None:
    with pytest.raises(ResearchStateError, match="diagnosed.*artifact"):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={},
        )


def test_resume_rejects_diagnosed_transition_without_artifact_evidence(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    payload = initialize_state(_config_for_state(contract_paths)).to_dict()
    payload["status"] = "diagnosed"
    payload["iteration"] = 1
    payload["used_profiles"] = ["safe_weather"]
    payload["remaining_profiles"] = ["history_tree"]
    payload["transitions"] = [
        {
            "timestamp": "2026-07-22T00:00:00+00:00",
            "from_status": "initialized",
            "to_status": "diagnosed",
            "iteration": 1,
            "profile": "safe_weather",
            "artifact_path": None,
            "sha256": None,
        }
    ]
    state_path = tmp_path / "diagnosed-without-artifact.json"
    atomic_write_json(state_path, payload)

    with pytest.raises(ResearchStateError, match="diagnosed.*artifact"):
        load_state(state_path)


def test_transition_rejects_forged_current_status_and_history(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    forged = replace(_state_at_verifying(contract_paths, tmp_path), status="proposed")

    with pytest.raises(ResearchStateError, match="history"):
        transition_state(forged, to_status="experimenting", artifact_paths={})


def test_atomic_write_json_rejects_final_target_symlink(tmp_path: Path) -> None:
    protected = tmp_path / "protected.json"
    protected.write_text('{"keep":true}\n', encoding="utf-8")
    target = tmp_path / "state.json"
    target.symlink_to(protected)

    with pytest.raises(ResearchStateError, match="symlink"):
        atomic_write_json(target, {"replace": True})

    assert target.is_symlink()
    assert protected.read_text(encoding="utf-8") == '{"keep":true}\n'


def test_atomic_write_json_rejects_parent_directory_symlink(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    link_parent = tmp_path / "link-parent"
    link_parent.symlink_to(protected, target_is_directory=True)
    target = link_parent / "state.json"

    with pytest.raises(ResearchStateError, match="symlink"):
        atomic_write_json(target, {"replace": True})

    assert not (protected / "state.json").exists()


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
    assert "dataset_sha256" not in serialized
    assert "prediction_time_inputs" not in serialized
    assert "baseline_evidence_available" not in serialized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_samples", ["generation_mw=42"]),
        ("rows", [{"plant_id": "plant_01", "generation_mw": 42}]),
        ("records", [{"timestamp": "2026-01-01T00:00:00+00:00"}]),
        ("secret_values", {"api_key": "not-safe"}),
        ("environment", {"API_TOKEN": "not-safe"}),
    ],
)
def test_diagnostic_artifact_rejects_rows_samples_secrets_and_environment_values(
    contract_paths: dict[str, Path], tmp_path: Path, field: str, value: object
) -> None:
    artifact = _diagnosis_artifact(tmp_path, **{field: value})

    with pytest.raises(ResearchStateError, match=field):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={"diagnosis": artifact},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows", [{"plant_id": "plant_01", "generation_mw": 42}]),
        ("secret_values", {"api_key": "not-safe"}),
    ],
)
def test_diagnosed_transition_validates_every_artifact_independent_of_key(
    contract_paths: dict[str, Path], tmp_path: Path, field: str, value: object
) -> None:
    artifact = _diagnosis_artifact(tmp_path, **{field: value})

    with pytest.raises(ResearchStateError, match=field):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={"report": artifact},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rows", [{"plant_id": "plant_01", "generation_mw": 42}]),
        ("secret_values", {"api_key": "not-safe"}),
    ],
)
def test_resume_revalidates_diagnosed_artifacts_independent_of_original_key(
    contract_paths: dict[str, Path], tmp_path: Path, field: str, value: object
) -> None:
    artifact = _diagnosis_artifact(tmp_path)
    state = transition_state(
        initialize_state(_config_for_state(contract_paths)),
        to_status="diagnosed",
        artifact_paths={"report": artifact},
    )
    artifact.write_text(json.dumps(_diagnosis_payload(**{field: value})), encoding="utf-8")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = state.to_dict()
    artifact_path = str(artifact.resolve())
    payload["artifacts"][artifact_path] = checksum
    payload["transitions"][0]["sha256"] = checksum
    state_path = tmp_path / "forged-diagnosed-state.json"
    atomic_write_json(state_path, payload)

    with pytest.raises(ResearchStateError, match=field):
        load_state(state_path)


def test_diagnostic_artifact_requires_strict_json(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    artifact = tmp_path / "diagnosis.json"
    artifact.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(ResearchStateError, match="diagnostic JSON"):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={"diagnosis": artifact},
        )


@pytest.mark.parametrize("artifact_name", ["stage_1_diagnosis", "report_diagnostic"])
def test_diagnostic_artifact_key_segments_enforce_content_safety(
    contract_paths: dict[str, Path], tmp_path: Path, artifact_name: str
) -> None:
    artifact = _diagnosis_artifact(tmp_path, environment={"API_TOKEN": "not-safe"})

    with pytest.raises(ResearchStateError, match="environment"):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={artifact_name: artifact},
        )


def test_hierarchical_diagnostic_artifact_key_enforces_content_safety(
    contract_paths: dict[str, Path], tmp_path: Path
) -> None:
    artifact = _diagnosis_artifact(tmp_path, environment={"API_TOKEN": "not-safe"})

    with pytest.raises(ResearchStateError, match="environment"):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={"diagnosis/raw.json": artifact},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warnings", ["api key: hunter2"]),
        (
            "baseline_evidence",
            {
                "path": "https://example.invalid/report?api_key=hunter2",
                "sha256": "b" * 64,
                "status": "available",
            },
        ),
    ],
)
def test_diagnostic_artifact_rejects_secret_values_in_allowed_fields(
    contract_paths: dict[str, Path], tmp_path: Path, field: str, value: object
) -> None:
    artifact = _diagnosis_artifact(tmp_path, **{field: value})

    with pytest.raises(ResearchStateError, match="sensitive"):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={"diagnosis": artifact},
        )


@pytest.mark.parametrize(
    "warning",
    [
        "AWS_ACCESS_KEY_ID=AKIA_NOT_SAFE",
        "plant_01,2026-01-01T00:00:00Z,42",
    ],
)
def test_diagnostic_artifact_rejects_environment_and_raw_row_text(
    contract_paths: dict[str, Path], tmp_path: Path, warning: str
) -> None:
    artifact = _diagnosis_artifact(tmp_path, warnings=[warning])

    with pytest.raises(ResearchStateError, match="diagnostic warnings"):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={"diagnosis": artifact},
        )


@pytest.mark.parametrize(
    "reference_path",
    [
        "plant_01,2026-01-01T00:00:00Z,42",
        "ghp_0123456789abcdefghijklmno",
    ],
)
def test_diagnostic_artifact_rejects_raw_or_secret_evidence_paths(
    contract_paths: dict[str, Path], tmp_path: Path, reference_path: str
) -> None:
    artifact = _diagnosis_artifact(
        tmp_path,
        baseline_evidence={
            "path": reference_path,
            "sha256": "b" * 64,
            "status": "available",
        },
    )

    with pytest.raises(ResearchStateError, match="baseline_evidence.path"):
        transition_state(
            initialize_state(_config_for_state(contract_paths)),
            to_status="diagnosed",
            artifact_paths={"diagnosis": artifact},
        )


@pytest.mark.parametrize("run_id", ["../unsafe", "unsafe id"])
def test_state_rejects_unsafe_run_ids_during_construction_and_loading(
    contract_paths: dict[str, Path], tmp_path: Path, run_id: str
) -> None:
    initial = initialize_state(_config_for_state(contract_paths))

    with pytest.raises(ResearchStateError, match="run_id"):
        replace(initial, run_id=run_id)

    payload = initial.to_dict()
    payload["run_id"] = run_id
    state_path = tmp_path / "unsafe-state.json"
    atomic_write_json(state_path, payload)

    with pytest.raises(ResearchStateError, match="run_id"):
        load_state(state_path)


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
