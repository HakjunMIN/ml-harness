from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from power_forecasting.data import (
    REQUIRED_COLUMNS,
    generate_synthetic_data,
    parse_timestamps,
    validate_dataset,
)
from power_forecasting.features import apply_feature_specs
from power_forecasting.proposals import ResearchProposal, load_proposal, proposal_to_dict
from power_forecasting.research_contracts import (
    ResearchContractError,
    ResearchLoopConfig,
    SUPPORTED_PROFILES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DATASET = REPOSITORY_ROOT / ".agents" / "fixtures" / "valid-dataset.csv"
FIXTURE_MANIFEST = REPOSITORY_ROOT / ".agents" / "fixtures" / "promoted-manifest.json"
PREDICTION_TIME_COLUMNS = tuple(
    column
    for column in REQUIRED_COLUMNS
    if column.startswith(("forecast_", "ldaps_"))
)


def _roles():
    return importlib.import_module("power_forecasting.research_roles")


def _config() -> ResearchLoopConfig:
    return ResearchLoopConfig(
        schema_version="1",
        run_id="research_roles_001",
        dataset_path=str(FIXTURE_DATASET),
        legacy_manifest_path=str(FIXTURE_MANIFEST),
        run_dir=str(REPOSITORY_ROOT / ".agents" / "runs" / "research_roles_001"),
        profiles=("safe_weather", "history_tree", "bounded_search"),
        max_iterations=3,
        fold_count=2,
        objective="nmae",
        minimum_improvement=0.01,
        max_plant_regression=0.03,
    )


def _diagnosis():
    return _roles().run_diagnostic_agent(_config())


def _candidate_count(proposal: ResearchProposal) -> int:
    search_count = 0
    if proposal.search is not None:
        search_count = int(proposal.search["n_trials"]) + 1
    return len(proposal.feature_sets) * (len(proposal.model_recipes) + search_count)


def _half_drift(frame: pd.DataFrame, column: str) -> float:
    timestamps = parse_timestamps(frame["timestamp"])
    series = (
        pd.DataFrame({"timestamp": timestamps, "value": frame[column]})
        .groupby("timestamp", sort=True)["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    midpoint = len(series) // 2
    early = float(np.mean(series[:midpoint]))
    late = float(np.mean(series[midpoint:]))
    denominator = abs(early) + abs(late)
    return 0.0 if denominator == 0.0 else abs(late - early) / denominator


def test_diagnostic_agent_is_deterministic_and_reports_correct_aggregates():
    roles = _roles()
    first = roles.run_diagnostic_agent(_config())
    second = roles.run_diagnostic_agent(_config())
    frame = pd.read_csv(FIXTURE_DATASET)
    timestamps = parse_timestamps(frame["timestamp"])

    assert isinstance(first, roles.DiagnosticReport)
    assert first == second
    assert first.dataset_sha256 == hashlib.sha256(FIXTURE_DATASET.read_bytes()).hexdigest()
    assert first.row_count == len(frame)
    assert first.plant_count == int(frame["plant_id"].nunique())
    assert first.time_start == timestamps.min().isoformat()
    assert first.time_end == timestamps.max().isoformat()
    assert set(first.missingness) == set(REQUIRED_COLUMNS)
    assert first.missingness == {column: 0.0 for column in REQUIRED_COLUMNS}
    assert set(first.drift_summary) == set(PREDICTION_TIME_COLUMNS)
    assert first.drift_summary == {
        column: pytest.approx(_half_drift(frame, column))
        for column in PREDICTION_TIME_COLUMNS
    }
    capacity = frame["capacity_mw"].to_numpy(dtype=float)
    target = frame["generation_mw"].to_numpy(dtype=float)
    assert first.residual_summary == {
        "capacity_utilization_mean": pytest.approx(float(np.mean(target / capacity))),
        "zero_baseline_nmae": pytest.approx(float(np.sum(np.abs(target)) / np.sum(capacity))),
    }
    assert first.leakage_checks == {
        "dataset_schema_valid": True,
        "history_features_strict_prior": True,
        "prediction_inputs_exclude_actual": True,
        "prediction_inputs_exclude_target": True,
    }
    assert all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for summary in (first.missingness, first.drift_summary, first.residual_summary)
        for value in summary.values()
    )


def test_diagnostic_agent_reports_zero_drift_for_one_timestamp(monkeypatch):
    roles = _roles()
    frame = generate_synthetic_data(days=1, plants=1, seed=7).iloc[[0]].reset_index(
        drop=True
    )
    validate_dataset(frame)
    monkeypatch.setattr(roles, "_load_dataset", lambda path: frame)

    report = roles.run_diagnostic_agent(_config())

    assert report.row_count == 1
    assert all(value == 0.0 for value in report.drift_summary.values())


def test_diagnostic_report_is_immutable_and_serializes_only_aggregate_data(monkeypatch):
    monkeypatch.setenv("RESEARCH_ROLE_TEST_SECRET", "must-not-appear")
    report = _diagnosis()

    with pytest.raises(TypeError):
        report.missingness["forecast_irradiance"] = 1.0

    payload = report.to_dict()
    serialized = json.dumps(payload, sort_keys=True, allow_nan=False)

    assert set(payload) == {
        "schema_version",
        "dataset_sha256",
        "row_count",
        "plant_count",
        "time_start",
        "time_end",
        "missingness",
        "drift_summary",
        "residual_summary",
        "leakage_checks",
        "recommended_profiles",
    }
    assert "plant-a" not in serialized
    assert "RESEARCH_ROLE_TEST_SECRET" not in serialized
    assert "must-not-appear" not in serialized
    assert all(not isinstance(value, list) for value in payload.values())
    assert all(isinstance(value, (float, bool)) for value in payload["missingness"].values())
    assert all(isinstance(value, (float, bool)) for value in payload["drift_summary"].values())
    assert all(isinstance(value, (float, bool)) for value in payload["residual_summary"].values())


def test_diagnostic_report_payload_passes_existing_privacy_artifact_validation():
    from power_forecasting.research_state import _validate_diagnostic_field

    for field, value in _diagnosis().to_dict().items():
        _validate_diagnostic_field(field, value)


def test_diagnostic_recommendations_are_supported_and_history_is_feasible_for_fixture():
    report = _diagnosis()

    assert report.recommended_profiles
    assert set(report.recommended_profiles) <= SUPPORTED_PROFILES
    assert report.recommended_profiles == (
        "safe_weather",
        "history_tree",
        "bounded_search",
    )


@pytest.mark.parametrize(
    ("profile", "expected_recipes"),
    [
        ("safe_weather", ("ridge_weather", "hgb_weather")),
        ("history_tree", ("forest_history", "hgb_history")),
        ("bounded_search", ("forest_search", "xgb_search", "lgbm_search")),
    ],
)
def test_profiles_return_strict_valid_deterministically_ordered_proposals(
    profile: str, expected_recipes: tuple[str, ...]
):
    roles = _roles()
    diagnosis = _diagnosis()

    first = roles.generate_profile_proposal(
        profile,
        run_id="role_run_001",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=6,
        diagnosis=diagnosis,
    )
    second = roles.generate_profile_proposal(
        profile,
        run_id="role_run_001",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=6,
        diagnosis=diagnosis,
    )

    assert isinstance(first, ResearchProposal)
    assert first == second
    assert tuple(recipe.name for recipe in first.model_recipes) == expected_recipes
    assert load_proposal(proposal_to_dict(first)) == first
    assert _candidate_count(first) <= 6
    assert all(
        source != "generation_mw" and not source.startswith("actual_")
        for feature_set in first.feature_sets
        for spec in feature_set.specs
        for source in spec.inputs
    )


@pytest.mark.parametrize(
    ("profile", "candidate_cap"),
    [
        ("safe_weather", 1),
        ("history_tree", 1),
        ("bounded_search", 1),
        ("bounded_search", 5),
    ],
)
def test_candidate_cap_deterministically_trims_templates(
    profile: str, candidate_cap: int
):
    roles = _roles()
    proposal = roles.generate_profile_proposal(
        profile,
        run_id="role_run_002",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=candidate_cap,
        diagnosis=_diagnosis(),
    )

    assert _candidate_count(proposal) <= candidate_cap
    assert proposal.budget["max_evaluations"] == _candidate_count(proposal)
    assert proposal.budget["top_feature_groups"] == 1
    if profile == "bounded_search" and candidate_cap == 5:
        assert proposal.search is not None
        assert proposal.search["n_trials"] == 1


@pytest.mark.parametrize("candidate_cap", [0, -1, True, 1.5])
def test_profile_generation_rejects_invalid_candidate_caps(candidate_cap):
    roles = _roles()

    with pytest.raises(ValueError, match="candidate_cap"):
        roles.generate_profile_proposal(
            "safe_weather",
            run_id="role_run_003",
            legacy_manifest_path=FIXTURE_MANIFEST,
            fold_count=2,
            objective="nmae",
            candidate_cap=candidate_cap,
            diagnosis=_diagnosis(),
        )


def test_profile_generation_fails_closed_for_unsupported_profile():
    roles = _roles()

    with pytest.raises(ResearchContractError, match="unsupported"):
        roles.generate_profile_proposal(
            "unbounded_profile",
            run_id="role_run_004",
            legacy_manifest_path=FIXTURE_MANIFEST,
            fold_count=2,
            objective="nmae",
            candidate_cap=2,
            diagnosis=_diagnosis(),
        )


def test_profile_content_uses_no_raw_target_or_plant_identifier_data():
    roles = _roles()
    diagnosis = _diagnosis()
    changed_aggregates = roles.DiagnosticReport(
        schema_version=diagnosis.schema_version,
        dataset_sha256="f" * 64,
        row_count=diagnosis.row_count,
        plant_count=diagnosis.plant_count,
        time_start=diagnosis.time_start,
        time_end=diagnosis.time_end,
        missingness={key: 0.0 for key in diagnosis.missingness},
        drift_summary={key: 1.0 for key in diagnosis.drift_summary},
        residual_summary={key: 1.0 for key in diagnosis.residual_summary},
        leakage_checks=diagnosis.leakage_checks,
        recommended_profiles=diagnosis.recommended_profiles,
    )

    original = roles.generate_profile_proposal(
        "safe_weather",
        run_id="role_run_005",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=2,
        diagnosis=diagnosis,
    )
    altered = roles.generate_profile_proposal(
        "safe_weather",
        run_id="role_run_005",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=2,
        diagnosis=changed_aggregates,
    )
    serialized = json.dumps(proposal_to_dict(original), sort_keys=True, allow_nan=False)

    assert proposal_to_dict(original) == proposal_to_dict(altered)
    assert "plant-a" not in serialized
    assert "generation_mw" not in serialized


def test_history_tree_features_are_strict_prior_only():
    roles = _roles()
    proposal = roles.generate_profile_proposal(
        "history_tree",
        run_id="role_run_006",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=2,
        diagnosis=_diagnosis(),
    )
    frame = generate_synthetic_data(days=1, plants=1, seed=17)
    specs = proposal.feature_sets[0].specs
    engineered = apply_feature_specs(frame, list(specs))

    assert {spec.transform for spec in specs} == {"lag", "rolling_mean"}
    assert all(
        source.startswith(("forecast_", "ldaps_"))
        for spec in specs
        for source in spec.inputs
    )
    assert all("strictly prior" in spec.rationale.lower() for spec in specs)
    assert math.isnan(engineered.iloc[0, 0])
    assert math.isnan(engineered.iloc[0, 1])
    assert engineered.iloc[1, 0] == frame.iloc[0]["forecast_irradiance"]


def test_bounded_search_uses_existing_optuna_budgets():
    roles = _roles()
    proposal = roles.generate_profile_proposal(
        "bounded_search",
        run_id="role_run_007",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=6,
        diagnosis=_diagnosis(),
    )

    assert proposal.search is not None
    assert proposal.search["sampler"] == "tpe"
    assert 1 <= proposal.search["n_trials"] <= 50
    assert 1 <= proposal.budget["max_evaluations"] <= 50
    assert _candidate_count(proposal) == 6
