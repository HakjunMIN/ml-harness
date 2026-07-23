from __future__ import annotations

import hashlib
import json
import math
from os import PathLike
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from power_forecasting.aidd import PromotionManifestError, validate_promotion_manifest
from power_forecasting.catalogs import CatalogProfile, OptimizationCatalog
from power_forecasting.profile_names import is_profile_name
from power_forecasting.cli import _load_dataset
from power_forecasting.data import DataContractError, REQUIRED_COLUMNS, parse_timestamps
from power_forecasting.proposals import (
    FeatureSet,
    ModelRecipe,
    ResearchProposal,
    load_proposal,
)
from power_forecasting.research_contracts import (
    ResearchContractError,
    ResearchLoopConfig,
    validate_run_id,
)


_SCHEMA_VERSION = "1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PREDICTION_TIME_COLUMNS = tuple(
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
_STRICT_PRIOR_HISTORY_TRANSFORMS = frozenset({"lag", "rolling_mean"})


@dataclass(frozen=True)
class DiagnosticReport:
    schema_version: str
    dataset_sha256: str
    row_count: int
    plant_count: int
    time_start: str
    time_end: str
    missingness: Mapping[str, float]
    drift_summary: Mapping[str, float]
    residual_summary: Mapping[str, float]
    leakage_checks: Mapping[str, bool]
    recommended_profiles: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ResearchContractError("diagnostic schema_version must be exactly '1'")
        if type(self.dataset_sha256) is not str or not _SHA256_PATTERN.fullmatch(
            self.dataset_sha256
        ):
            raise ResearchContractError(
                "diagnostic dataset_sha256 must be a lowercase SHA-256 string"
            )
        _nonnegative_integer(self.row_count, "diagnostic row_count")
        _nonnegative_integer(self.plant_count, "diagnostic plant_count")
        _iso_timestamp(self.time_start, "diagnostic time_start")
        _iso_timestamp(self.time_end, "diagnostic time_end")

        missingness = _ratio_mapping(
            self.missingness,
            "diagnostic missingness",
            required=frozenset(REQUIRED_COLUMNS),
        )
        drift_summary = _ratio_mapping(
            self.drift_summary,
            "diagnostic drift_summary",
            required=frozenset(_PREDICTION_TIME_COLUMNS),
        )
        residual_summary = _ratio_mapping(
            self.residual_summary,
            "diagnostic residual_summary",
            required=_RESIDUAL_SUMMARY_KEYS,
        )
        leakage_checks = _boolean_mapping(
            self.leakage_checks,
            "diagnostic leakage_checks",
            required=_LEAKAGE_CHECK_KEYS,
        )
        recommended_profiles = _recommended_profiles(self.recommended_profiles)

        object.__setattr__(self, "missingness", MappingProxyType(dict(missingness)))
        object.__setattr__(self, "drift_summary", MappingProxyType(dict(drift_summary)))
        object.__setattr__(
            self,
            "residual_summary",
            MappingProxyType(dict(residual_summary)),
        )
        object.__setattr__(
            self,
            "leakage_checks",
            MappingProxyType(dict(leakage_checks)),
        )
        object.__setattr__(self, "recommended_profiles", recommended_profiles)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_sha256": self.dataset_sha256,
            "row_count": self.row_count,
            "plant_count": self.plant_count,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "missingness": dict(self.missingness),
            "drift_summary": dict(self.drift_summary),
            "residual_summary": dict(self.residual_summary),
            "leakage_checks": dict(self.leakage_checks),
            "recommended_profiles": tuple(self.recommended_profiles),
        }


def run_diagnostic_agent(config: ResearchLoopConfig) -> DiagnosticReport:
    """Return deterministic, aggregate-only facts for a validated research input."""

    if not isinstance(config, ResearchLoopConfig):
        raise TypeError("config must be a ResearchLoopConfig")

    _validate_legacy_manifest(Path(config.legacy_manifest_path))
    dataset_path = Path(config.dataset_path)
    frame = _load_dataset(dataset_path)
    if frame.empty:
        raise DataContractError("dataset must contain at least one row")

    timestamps = parse_timestamps(
        frame["timestamp"],
        error_message="invalid timestamps: unparseable values",
        error_type=DataContractError,
    )
    unique_timestamps = pd.Index(pd.unique(timestamps))
    history_feasible = len(unique_timestamps) >= 4

    return DiagnosticReport(
        schema_version=_SCHEMA_VERSION,
        dataset_sha256=_sha256_file(dataset_path),
        row_count=int(len(frame)),
        plant_count=int(frame["plant_id"].nunique()),
        time_start=timestamps.min().isoformat(),
        time_end=timestamps.max().isoformat(),
        missingness={
            column: float(frame[column].isna().mean()) for column in REQUIRED_COLUMNS
        },
        drift_summary={
            column: _temporal_drift(frame, timestamps, column)
            for column in _PREDICTION_TIME_COLUMNS
        },
        residual_summary=_residual_summary(frame),
        leakage_checks={
            "dataset_schema_valid": True,
            "history_features_strict_prior": True,
            "prediction_inputs_exclude_actual": True,
            "prediction_inputs_exclude_target": True,
        },
        recommended_profiles=tuple(
            profile
            for profile in config.profiles
            if (
                not _profile_requires_strict_prior_history(config.catalog, profile)
                or history_feasible
            )
        ),
    )


def generate_profile_proposal(
    profile: str,
    *,
    catalog: OptimizationCatalog,
    run_id: str,
    legacy_manifest_path: Path,
    fold_count: int,
    objective: str,
    candidate_cap: int,
    diagnosis: DiagnosticReport,
) -> ResearchProposal:
    """Build one fixed, diagnosis-gated proposal without inspecting raw data."""

    _validate_profile_request(
        profile=profile,
        catalog=catalog,
        run_id=run_id,
        legacy_manifest_path=legacy_manifest_path,
        fold_count=fold_count,
        objective=objective,
        candidate_cap=candidate_cap,
        diagnosis=diagnosis,
    )

    catalog_profile = catalog.profile(profile)
    feature_set = _feature_set_for_profile(catalog, catalog_profile)
    recipes, search = _candidate_components(catalog, catalog_profile, candidate_cap)
    candidate_count = len((feature_set,)) * (
        len(recipes) + (int(search["n_trials"]) + 1 if search is not None else 0)
    )
    proposal = ResearchProposal(
        schema_version=_SCHEMA_VERSION,
        proposal_id=f"{run_id}-{profile}-v1",
        rationale=catalog_profile.rationale,
        baseline={"model": "SPOT"},
        feature_sets=(feature_set,),
        model_recipes=recipes,
        budget={"max_evaluations": candidate_count, "top_feature_groups": 1},
        search=search,
    )
    return load_proposal(proposal.to_dict(), catalog=catalog, profile=profile)


def _validate_profile_request(
    *,
    profile: object,
    catalog: object,
    run_id: object,
    legacy_manifest_path: object,
    fold_count: object,
    objective: object,
    candidate_cap: object,
    diagnosis: object,
) -> None:
    if not isinstance(catalog, OptimizationCatalog):
        raise TypeError("catalog must be an OptimizationCatalog")
    if type(profile) is not str or profile not in catalog.profiles:
        raise ResearchContractError(f"unsupported research profile: {profile!r}")
    validate_run_id(run_id)
    _validate_legacy_manifest(
        _path_from_pathlike(legacy_manifest_path, "legacy_manifest_path")
    )
    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or not 1 <= fold_count <= 10
    ):
        raise ResearchContractError("fold_count must be between 1 and 10")
    if type(objective) is not str or not objective.strip():
        raise ResearchContractError("objective must be a nonblank string")
    if (
        isinstance(candidate_cap, bool)
        or not isinstance(candidate_cap, int)
        or candidate_cap <= 0
    ):
        raise ValueError("candidate_cap must be a positive integer")
    if not isinstance(diagnosis, DiagnosticReport):
        raise TypeError("diagnosis must be a DiagnosticReport")
    if profile not in diagnosis.recommended_profiles:
        raise ResearchContractError(
            f"profile {profile!r} is not recommended by the diagnostic report"
        )
    if (
        _profile_requires_strict_prior_history(catalog, profile)
        and not diagnosis.leakage_checks["history_features_strict_prior"]
    ):
        raise ResearchContractError(
            f"{profile} requires strict-prior history features"
        )


def _profile_requires_strict_prior_history(
    catalog: OptimizationCatalog, profile_name: str
) -> bool:
    profile = catalog.profile(profile_name)
    return any(
        spec.transform in _STRICT_PRIOR_HISTORY_TRANSFORMS
        for feature_set_name in profile.feature_set_names
        for spec in catalog.feature_sets[feature_set_name].specs
    )


def _feature_set_for_profile(
    catalog: OptimizationCatalog, profile: CatalogProfile
) -> FeatureSet:
    feature_sets = tuple(
        catalog.feature_sets[name] for name in profile.feature_set_names
    )
    if len(feature_sets) == 1:
        return feature_sets[0]
    return FeatureSet(
        profile.name,
        profile.rationale,
        tuple(spec for feature_set in feature_sets for spec in feature_set.specs),
    )


def _candidate_components(
    catalog: OptimizationCatalog,
    profile: CatalogProfile,
    candidate_cap: int,
) -> tuple[tuple[ModelRecipe, ...], Mapping[str, Any] | None]:
    templates = tuple(
        _catalog_recipe(catalog, recipe_name)
        for recipe_name in profile.direct_recipe_names
    )
    recipes = templates[: min(len(templates), candidate_cap)]
    search: Mapping[str, Any] | None = None

    if profile.search_name is not None:
        remaining = candidate_cap - len(recipes)
        if remaining >= 2:
            source = catalog.searches[profile.search_name]
            search = {
                "sampler": source["sampler"],
                "seed": source["seed"],
                "n_trials": min(int(source["n_trials"]), remaining - 1),
                "spaces": {
                    recipe: {
                        parameter: list(values)
                        for parameter, values in parameters.items()
                    }
                    for recipe, parameters in source["spaces"].items()
                },
            }

    return recipes, search


def _catalog_recipe(catalog: OptimizationCatalog, recipe_name: str) -> ModelRecipe:
    source = catalog.direct_recipes[recipe_name]
    return ModelRecipe(
        source.name,
        source.recipe,
        source.parameters,
        source.rationale,
    )


def proposal_catalog(catalog: OptimizationCatalog) -> dict[str, object]:
    """Return the immutable, privacy-safe candidate catalog for coding agents."""

    if not isinstance(catalog, OptimizationCatalog):
        raise TypeError("catalog must be an OptimizationCatalog")
    return {
        "schema_version": "1",
        "catalog_path": str(catalog.source_path.resolve()),
        "catalog_sha256": catalog.sha256,
        "max_evaluations": 50,
        "profiles": {
            profile_name: {
                "rationale": profile.rationale,
                "feature_sets": [
                    catalog.feature_sets[name].to_dict()
                    for name in profile.feature_set_names
                ],
                "model_recipes": [
                    {
                        "name": recipe.name,
                        "recipe": recipe.recipe,
                        "parameters": dict(recipe.parameters),
                        "allowed_parameters": {
                            key: list(values)
                            for key, values in recipe.allowed_parameters.items()
                        },
                        "rationale": recipe.rationale,
                    }
                    for recipe in (
                        catalog.direct_recipes[name]
                        for name in profile.direct_recipe_names
                    )
                ],
                "search": (
                    None
                    if profile.search_name is None
                    else _json_search(catalog.searches[profile.search_name])
                ),
            }
            for profile_name, profile in catalog.profiles.items()
        },
    }


def _json_search(search: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sampler": search["sampler"],
        "seed": search["seed"],
        "n_trials": search["n_trials"],
        "spaces": {
            recipe: {
                parameter: list(values)
                for parameter, values in parameters.items()
            }
            for recipe, parameters in search["spaces"].items()
        },
    }


def _temporal_drift(
    frame: pd.DataFrame, timestamps: pd.Series, column: str
) -> float:
    means = (
        pd.DataFrame({"timestamp": timestamps, "value": frame[column]})
        .groupby("timestamp", sort=True)["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(means) <= 1:
        return 0.0
    midpoint = len(means) // 2
    early = float(np.mean(means[:midpoint]))
    late = float(np.mean(means[midpoint:]))
    denominator = abs(early) + abs(late)
    if denominator == 0.0:
        return 0.0
    return float(abs(late - early) / denominator)


def _residual_summary(frame: pd.DataFrame) -> dict[str, float]:
    capacity = frame["capacity_mw"].to_numpy(dtype=float)
    target = frame["generation_mw"].to_numpy(dtype=float)
    return {
        "capacity_utilization_mean": float(np.mean(target / capacity)),
        "zero_baseline_nmae": float(np.sum(np.abs(target)) / np.sum(capacity)),
    }


def _validate_legacy_manifest(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise ResearchContractError("legacy_manifest_path must be an existing file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResearchContractError("legacy_manifest_path must contain JSON") from exc
    try:
        validate_promotion_manifest(manifest)
    except PromotionManifestError:
        pass
    else:
        return
    # Raise after handling so raw manifest details are not retained as exception context.
    raise ResearchContractError(
        "legacy_manifest_path must contain a trusted promoted manifest"
    )


def _path_from_pathlike(value: object, label: str) -> Path:
    if not isinstance(value, (str, PathLike)):
        raise TypeError(f"{label} must be a PathLike or string")
    try:
        return Path(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be a PathLike or string") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio_mapping(
    value: Mapping[str, float],
    label: str,
    *,
    required: frozenset[str],
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise ResearchContractError(f"{label} must contain its exact aggregate keys")
    normalized = {}
    for key in sorted(required):
        raw = value[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ResearchContractError(f"{label}.{key} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ResearchContractError(f"{label}.{key} must be finite and between 0 and 1")
        normalized[key] = number
    return normalized


def _boolean_mapping(
    value: Mapping[str, bool],
    label: str,
    *,
    required: frozenset[str],
) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise ResearchContractError(f"{label} must contain its exact aggregate keys")
    normalized = {}
    for key in sorted(required):
        if type(value[key]) is not bool:
            raise ResearchContractError(f"{label}.{key} must be a boolean")
        normalized[key] = value[key]
    return normalized


def _recommended_profiles(value: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ResearchContractError("diagnostic recommended_profiles must be a tuple")
    if len(set(value)) != len(value):
        raise ResearchContractError("diagnostic recommended_profiles contains duplicates")
    if any(not is_profile_name(profile) for profile in value):
        raise ResearchContractError(
            "diagnostic recommended_profiles contains an invalid profile"
        )
    return value


def _nonnegative_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchContractError(f"{label} must be a nonnegative integer")


def _iso_timestamp(value: object, label: str) -> None:
    if type(value) is not str or not value:
        raise ResearchContractError(f"{label} must be an ISO timestamp")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchContractError(f"{label} must be an ISO timestamp") from exc


__all__ = [
    "DiagnosticReport",
    "generate_profile_proposal",
    "proposal_catalog",
    "run_diagnostic_agent",
]
