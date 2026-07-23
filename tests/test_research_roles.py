from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
from dataclasses import replace
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
from power_forecasting.catalogs import load_optimization_catalog
from power_forecasting.features import apply_feature_specs
from power_forecasting.proposals import ResearchProposal, load_proposal, proposal_to_dict
from power_forecasting.research_contracts import (
    ResearchContractError,
    ResearchLoopConfig,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DATASET = REPOSITORY_ROOT / ".agents" / "fixtures" / "valid-dataset.csv"
FIXTURE_MANIFEST = REPOSITORY_ROOT / ".agents" / "fixtures" / "promoted-manifest.json"
REJECTED_MANIFEST = (
    REPOSITORY_ROOT / ".agents" / "fixtures" / "rejected-promotion-manifest.json"
)
LEAKAGE_MANIFEST = (
    REPOSITORY_ROOT / ".agents" / "fixtures" / "leakage-promotion-manifest.json"
)
MALFORMED_MANIFEST = (
    REPOSITORY_ROOT / ".agents" / "fixtures" / "malformed-promotion-manifest.json"
)
MISSING_THRESHOLDS_MANIFEST = (
    REPOSITORY_ROOT
    / ".agents"
    / "fixtures"
    / "missing-thresholds-promotion-manifest.json"
)
FIXTURE_CATALOG = REPOSITORY_ROOT / "configs" / "optimization-catalog.v1.json"
BASE_DATASET = REPOSITORY_ROOT / ".agents" / "fixtures" / "research-roles-base.csv"
TARGET_VARIANT_DATASET = (
    REPOSITORY_ROOT / ".agents" / "fixtures" / "research-roles-target-variant.csv"
)
PLANT_VARIANT_DATASET = (
    REPOSITORY_ROOT / ".agents" / "fixtures" / "research-roles-plant-variant.csv"
)
PREDICTION_TIME_COLUMNS = tuple(
    column
    for column in REQUIRED_COLUMNS
    if column.startswith(("forecast_", "ldaps_"))
)


def _roles():
    return importlib.import_module("power_forecasting.research_roles")


def _config(
    *,
    dataset_path: Path = FIXTURE_DATASET,
    legacy_manifest_path: Path = FIXTURE_MANIFEST,
) -> ResearchLoopConfig:
    catalog = load_optimization_catalog(FIXTURE_CATALOG, repository_root=REPOSITORY_ROOT)
    return ResearchLoopConfig(
        schema_version="1",
        run_id="research_roles_001",
        dataset_path=str(dataset_path),
        legacy_manifest_path=str(legacy_manifest_path),
        catalog_path=str(catalog.source_path),
        catalog_sha256=catalog.sha256,
        catalog=catalog,
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


def _normalized_diagnosis_for_proposal_comparison(
    roles,
    report,
    reference,
    *,
    replace_residual_summary: bool,
):
    payload = report.to_dict()
    payload["dataset_sha256"] = reference.dataset_sha256
    if replace_residual_summary:
        payload["residual_summary"] = reference.residual_summary
    return roles.DiagnosticReport(**payload)


def _raw_fixture_sensitive_values(
    *dataset_paths: Path,
) -> tuple[set[str], set[str]]:
    plant_ids: set[str] = set()
    target_samples: set[str] = set()
    for dataset_path in dataset_paths:
        with dataset_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                plant_ids.add(row["plant_id"])
                target_samples.add(row["generation_mw"])
    return plant_ids, target_samples


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


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("missingness", 0.0),
        ("drift_summary", 0.0),
        ("residual_summary", 0.0),
        ("leakage_checks", True),
    ],
)
def test_diagnostic_report_mappings_resist_dict_mutation_bypasses(
    attribute: str,
    value: float | bool,
):
    report = _diagnosis()
    mapping = getattr(report, attribute)
    original = dict(mapping)
    key = "__mutation_attempt__"

    assert not isinstance(mapping, dict)
    if attribute == "missingness":
        with pytest.raises(TypeError):
            report.missingness |= {key: value}
    with pytest.raises(TypeError):
        mapping |= {key: value}
    with pytest.raises(TypeError):
        dict.__setitem__(mapping, key, value)
    for mutate in (
        lambda: mapping.update({key: value}),
        lambda: mapping.pop(next(iter(mapping))),
        lambda: mapping.clear(),
    ):
        with pytest.raises((AttributeError, TypeError)):
            mutate()
    for mutate in (
        lambda: dict.update(mapping, {key: value}),
        lambda: dict.pop(mapping, next(iter(mapping))),
        lambda: dict.clear(mapping),
    ):
        with pytest.raises(TypeError):
            mutate()

    assert mapping == original
    assert dict(mapping) == original
    assert report.to_dict()[attribute] == original


def test_diagnostic_report_payload_passes_existing_privacy_artifact_validation():
    from power_forecasting.research_state import _validate_diagnostic_field

    for field, value in _diagnosis().to_dict().items():
        _validate_diagnostic_field(field, value)


def test_diagnostic_recommendations_are_supported_and_history_is_feasible_for_fixture():
    report = _diagnosis()

    assert report.recommended_profiles
    assert set(report.recommended_profiles) <= set(_config().catalog.profile_names)
    assert report.recommended_profiles == (
        "safe_weather",
        "history_tree",
        "bounded_search",
    )


def test_catalog_profile_generation_uses_external_catalog_policy(tmp_path: Path):
    catalog_payload = json.loads(FIXTURE_CATALOG.read_text(encoding="utf-8"))
    catalog_payload["profiles"]["weather_variant"] = {
        "rationale": "Evaluate an externally configured weather variant.",
        "feature_sets": ["safe_weather"],
        "direct_recipes": ["ridge_weather"],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")
    catalog = load_optimization_catalog(catalog_path, repository_root=tmp_path)
    config = replace(
        _config(),
        catalog_path=str(catalog.source_path),
        catalog_sha256=catalog.sha256,
        catalog=catalog,
        profiles=("weather_variant",),
    )
    roles = _roles()
    diagnosis = roles.run_diagnostic_agent(config)

    assert diagnosis.recommended_profiles == ("weather_variant",)
    proposal = roles.generate_profile_proposal(
        "weather_variant",
        catalog=catalog,
        run_id=config.run_id,
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=config.fold_count,
        objective=config.objective,
        candidate_cap=2,
        diagnosis=diagnosis,
    )
    assert tuple(recipe.name for recipe in proposal.model_recipes) == ("ridge_weather",)
    assert proposal.feature_sets[0] == catalog.feature_sets["safe_weather"]


def test_catalog_history_transform_controls_diagnostic_recommendation(tmp_path: Path):
    catalog_payload = json.loads(FIXTURE_CATALOG.read_text(encoding="utf-8"))
    catalog_payload["profiles"]["prior_variant"] = {
        "rationale": "Evaluate an externally configured strict-prior variant.",
        "feature_sets": ["history_tree"],
        "direct_recipes": ["forest_history"],
    }
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_payload), encoding="utf-8")
    catalog = load_optimization_catalog(catalog_path, repository_root=tmp_path)
    short_dataset = tmp_path / "short-dataset.csv"
    pd.read_csv(FIXTURE_DATASET).head(3).to_csv(short_dataset, index=False)
    config = replace(
        _config(),
        dataset_path=str(short_dataset),
        catalog_path=str(catalog.source_path),
        catalog_sha256=catalog.sha256,
        catalog=catalog,
        profiles=("prior_variant",),
    )

    report = _roles().run_diagnostic_agent(config)

    assert report.recommended_profiles == ()


def test_proposal_catalog_records_catalog_identity():
    roles = _roles()
    catalog = _config().catalog

    payload = roles.proposal_catalog(catalog)

    assert payload["catalog_path"] == str(catalog.source_path)
    assert payload["catalog_sha256"] == catalog.sha256


@pytest.mark.parametrize(
    ("manifest_path", "message"),
    [
        (MALFORMED_MANIFEST, "must contain JSON"),
        (REJECTED_MANIFEST, "trusted promoted manifest"),
        (LEAKAGE_MANIFEST, "trusted promoted manifest"),
        (MISSING_THRESHOLDS_MANIFEST, "trusted promoted manifest"),
    ],
)
def test_diagnostic_agent_fails_closed_on_malformed_or_untrusted_promotion_manifests(
    manifest_path: Path,
    message: str,
):
    with pytest.raises(ResearchContractError, match=message):
        _roles().run_diagnostic_agent(_config(legacy_manifest_path=manifest_path))


def test_diagnostic_agent_sanitizes_invalid_manifest_provenance_errors(monkeypatch):
    roles = _roles()
    private_plant_id = "private-plant-identifier-7744"
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    manifest["per_plant_deltas"] = {private_plant_id: 1.1}
    monkeypatch.setattr(roles.json, "load", lambda handle: manifest)

    with pytest.raises(
        ResearchContractError,
        match="trusted promoted manifest",
    ) as exc_info:
        roles.run_diagnostic_agent(_config())

    assert str(exc_info.value) == (
        "legacy_manifest_path must contain a trusted promoted manifest"
    )
    assert private_plant_id not in str(exc_info.value)
    assert "1.1" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("profile", ("safe_weather", "history_tree", "bounded_search"))
def test_profile_generation_accepts_manifest_path_stored_by_research_loop_config(
    profile: str,
):
    roles = _roles()
    config = _config()
    proposal = roles.generate_profile_proposal(
        profile,
        catalog=_config().catalog,
        run_id=config.run_id,
        legacy_manifest_path=config.legacy_manifest_path,
        fold_count=config.fold_count,
        objective=config.objective,
        candidate_cap=6,
        diagnosis=roles.run_diagnostic_agent(config),
    )

    assert isinstance(proposal, ResearchProposal)


@pytest.mark.parametrize("profile", ("safe_weather", "history_tree", "bounded_search"))
def test_profile_generation_uses_only_equal_aggregate_diagnoses(
    profile: str,
):
    roles = _roles()
    base_frame = pd.read_csv(BASE_DATASET)
    target_variant_frame = pd.read_csv(TARGET_VARIANT_DATASET)
    plant_variant_frame = pd.read_csv(PLANT_VARIANT_DATASET)

    pd.testing.assert_frame_equal(
        base_frame.drop(columns="generation_mw"),
        target_variant_frame.drop(columns="generation_mw"),
    )
    pd.testing.assert_frame_equal(
        base_frame.drop(columns="plant_id"),
        plant_variant_frame.drop(columns="plant_id"),
    )
    assert not base_frame["generation_mw"].equals(target_variant_frame["generation_mw"])
    assert not base_frame["plant_id"].equals(plant_variant_frame["plant_id"])

    base = roles.run_diagnostic_agent(_config(dataset_path=BASE_DATASET))
    target_variant = roles.run_diagnostic_agent(
        _config(dataset_path=TARGET_VARIANT_DATASET)
    )
    plant_variant = roles.run_diagnostic_agent(
        _config(dataset_path=PLANT_VARIANT_DATASET)
    )

    assert target_variant.dataset_sha256 != base.dataset_sha256
    assert target_variant.residual_summary != base.residual_summary
    assert target_variant.missingness == base.missingness
    assert target_variant.drift_summary == base.drift_summary
    assert plant_variant.dataset_sha256 != base.dataset_sha256
    assert plant_variant.residual_summary == base.residual_summary
    assert plant_variant.missingness == base.missingness
    assert plant_variant.drift_summary == base.drift_summary

    # The file fingerprint changes with either fixture; only target values alter residuals.
    normalized_target = _normalized_diagnosis_for_proposal_comparison(
        roles,
        target_variant,
        base,
        replace_residual_summary=True,
    )
    normalized_plant = _normalized_diagnosis_for_proposal_comparison(
        roles,
        plant_variant,
        base,
        replace_residual_summary=False,
    )
    assert base == normalized_target == normalized_plant

    plant_ids, target_samples = _raw_fixture_sensitive_values(
        BASE_DATASET,
        TARGET_VARIANT_DATASET,
        PLANT_VARIANT_DATASET,
    )
    serialized_proposals = [
        json.dumps(
            proposal_to_dict(
                roles.generate_profile_proposal(
                    profile,
                    catalog=_config().catalog,
                    run_id="aggregate_only_001",
                    legacy_manifest_path=_config().legacy_manifest_path,
                    fold_count=2,
                    objective="nmae",
                    candidate_cap=6,
                    diagnosis=diagnosis,
                )
            ),
            sort_keys=True,
            allow_nan=False,
        )
        for diagnosis in (base, normalized_target, normalized_plant)
    ]

    assert serialized_proposals[0] == serialized_proposals[1] == serialized_proposals[2]
    for serialized in serialized_proposals:
        assert "generation_mw" not in serialized
        assert all(
            raw_value not in serialized for raw_value in plant_ids | target_samples
        )


def test_profile_generation_rejects_dataset_rows_as_diagnosis():
    roles = _roles()
    with pytest.raises(TypeError, match="DiagnosticReport"):
        roles.generate_profile_proposal(
            "safe_weather",
            catalog=_config().catalog,
            run_id="aggregate_only_001",
            legacy_manifest_path=_config().legacy_manifest_path,
            fold_count=2,
            objective="nmae",
            candidate_cap=2,
            diagnosis=pd.read_csv(BASE_DATASET),
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
        catalog=_config().catalog,
        run_id="role_run_001",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=6,
        diagnosis=diagnosis,
    )
    second = roles.generate_profile_proposal(
        profile,
        catalog=_config().catalog,
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
        catalog=_config().catalog,
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
            catalog=_config().catalog,
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
            catalog=_config().catalog,
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
        catalog=_config().catalog,
        run_id="role_run_005",
        legacy_manifest_path=FIXTURE_MANIFEST,
        fold_count=2,
        objective="nmae",
        candidate_cap=2,
        diagnosis=diagnosis,
    )
    altered = roles.generate_profile_proposal(
        "safe_weather",
        catalog=_config().catalog,
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
        catalog=_config().catalog,
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
        catalog=_config().catalog,
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
