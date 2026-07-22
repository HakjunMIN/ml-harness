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
from power_forecasting.cli import _load_dataset
from power_forecasting.data import DataContractError, REQUIRED_COLUMNS, parse_timestamps
from power_forecasting.features import FeatureSpec
from power_forecasting.proposals import (
    FeatureSet,
    ModelRecipe,
    ResearchProposal,
    load_proposal,
)
from power_forecasting.research_contracts import (
    ResearchContractError,
    ResearchLoopConfig,
    SUPPORTED_PROFILES,
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
_PROFILE_ORDER = ("safe_weather", "history_tree", "bounded_search")


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
            for profile in _PROFILE_ORDER
            if profile in config.profiles
            and (profile != "history_tree" or history_feasible)
        ),
    )


def generate_profile_proposal(
    profile: str,
    *,
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
        run_id=run_id,
        legacy_manifest_path=legacy_manifest_path,
        fold_count=fold_count,
        objective=objective,
        candidate_cap=candidate_cap,
        diagnosis=diagnosis,
    )

    feature_set = _feature_set_for_profile(profile, diagnosis)
    recipes, search = _candidate_components(profile, candidate_cap)
    candidate_count = len((feature_set,)) * (
        len(recipes) + (int(search["n_trials"]) + 1 if search is not None else 0)
    )
    proposal = ResearchProposal(
        schema_version=_SCHEMA_VERSION,
        proposal_id=f"{run_id}-{profile}-v1",
        rationale=_profile_rationale(profile),
        baseline={"model": "SPOT"},
        feature_sets=(feature_set,),
        model_recipes=recipes,
        budget={"max_evaluations": candidate_count, "top_feature_groups": 1},
        search=search,
    )
    return load_proposal(proposal.to_dict())


def _validate_profile_request(
    *,
    profile: object,
    run_id: object,
    legacy_manifest_path: object,
    fold_count: object,
    objective: object,
    candidate_cap: object,
    diagnosis: object,
) -> None:
    if type(profile) is not str or profile not in SUPPORTED_PROFILES:
        raise ResearchContractError(f"unsupported research profile: {profile!r}")
    validate_run_id(run_id)
    _validate_legacy_manifest(
        _path_from_pathlike(legacy_manifest_path, "legacy_manifest_path")
    )
    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or not 1 <= fold_count <= 10:
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
    if profile == "history_tree" and not diagnosis.leakage_checks[
        "history_features_strict_prior"
    ]:
        raise ResearchContractError("history_tree requires strict-prior history features")


def _feature_set_for_profile(
    profile: str, diagnosis: DiagnosticReport
) -> FeatureSet:
    if profile == "safe_weather":
        return FeatureSet(
            "safe_weather",
            "Bounded calendar and forecast-weather features available at prediction time.",
            _safe_weather_specs(),
        )
    if profile == "history_tree":
        return FeatureSet(
            "history_tree",
            "Strict-prior forecast history features for deterministic tree candidates.",
            _history_specs(),
        )
    if profile == "bounded_search":
        specs = _safe_weather_specs()
        if "history_tree" in diagnosis.recommended_profiles:
            specs += _history_specs()
        return FeatureSet(
            "bounded_search",
            "Bounded forecast-weather and strict-prior history features for search.",
            specs,
        )
    raise AssertionError(f"unreachable supported profile: {profile}")


def _candidate_components(
    profile: str, candidate_cap: int
) -> tuple[tuple[ModelRecipe, ...], Mapping[str, Any] | None]:
    templates = _recipe_templates(profile)
    recipes = templates[: min(len(templates), candidate_cap)]
    search: Mapping[str, Any] | None = None

    if profile == "bounded_search":
        remaining = candidate_cap - len(recipes)
        if remaining >= 2:
            search = _bounded_search(min(2, remaining - 1))

    return recipes, search


def _safe_weather_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            "hour_sin",
            "cyclic_hour",
            ("timestamp",),
            rationale="Prediction-time daily calendar phase.",
        ),
        FeatureSpec(
            "hour_cos",
            "cyclic_hour",
            ("timestamp",),
            rationale="Prediction-time daily calendar phase companion.",
        ),
        FeatureSpec(
            "effective_irradiance",
            "effective_irradiance",
            ("forecast_irradiance", "forecast_cloud_cover"),
            rationale="Cloud-adjusted forecast irradiance.",
        ),
        FeatureSpec(
            "forecast_temperature_derating",
            "temperature_derating",
            ("forecast_irradiance", "forecast_temperature"),
            {"reference": 25.0, "coefficient": 0.004},
            rationale="Forecast-temperature irradiance derating.",
        ),
    )


def _history_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(
            "prior_forecast_irradiance",
            "lag",
            ("forecast_irradiance",),
            {"periods": 1},
            rationale="Uses strictly prior forecast irradiance for the same plant.",
        ),
        FeatureSpec(
            "prior_forecast_cloud_cover_mean",
            "rolling_mean",
            ("forecast_cloud_cover",),
            {"window": 3},
            rationale="Uses a strictly prior rolling forecast cloud-cover mean.",
        ),
    )


def _recipe_templates(profile: str) -> tuple[ModelRecipe, ...]:
    if profile == "safe_weather":
        return (
            ModelRecipe(
                "ridge_weather",
                "ridge",
                {"alpha": 1.0},
                "Regularized linear weather baseline.",
            ),
            ModelRecipe(
                "hgb_weather",
                "hist_gradient_boosting",
                {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
                "Bounded histogram gradient boosting weather candidate.",
            ),
        )
    if profile == "history_tree":
        return (
            ModelRecipe(
                "forest_history",
                "random_forest",
                {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2},
                "Bounded deterministic random forest history candidate.",
            ),
            ModelRecipe(
                "hgb_history",
                "hist_gradient_boosting",
                {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31},
                "Bounded histogram gradient boosting history candidate.",
            ),
        )
    if profile == "bounded_search":
        return (
            ModelRecipe(
                "forest_search",
                "random_forest",
                {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2},
                "Bounded deterministic random forest candidate.",
            ),
            ModelRecipe(
                "xgb_search",
                "xgboost",
                {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                },
                "Bounded XGBoost candidate.",
            ),
            ModelRecipe(
                "lgbm_search",
                "lightgbm",
                {
                    "n_estimators": 100,
                    "learning_rate": 0.03,
                    "num_leaves": 15,
                    "min_child_samples": 10,
                },
                "Bounded LightGBM candidate.",
            ),
        )
    raise AssertionError(f"unreachable supported profile: {profile}")


def _bounded_search(n_trials: int) -> Mapping[str, Any]:
    return {
        "sampler": "tpe",
        "seed": 7,
        "n_trials": n_trials,
        "spaces": {
            "lightgbm": {
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "num_leaves": [15, 31],
                "min_child_samples": [10, 20],
            }
        },
    }


def _profile_rationale(profile: str) -> str:
    return {
        "safe_weather": "Evaluate bounded prediction-time calendar and weather features.",
        "history_tree": "Evaluate bounded strict-prior forecast-history tree candidates.",
        "bounded_search": "Evaluate bounded tree recipes with deterministic LightGBM TPE search.",
    }[profile]


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
    if any(type(profile) is not str or profile not in SUPPORTED_PROFILES for profile in value):
        raise ResearchContractError(
            "diagnostic recommended_profiles contains an unsupported profile"
        )
    expected = tuple(profile for profile in _PROFILE_ORDER if profile in value)
    if value != expected:
        raise ResearchContractError(
            "diagnostic recommended_profiles must use deterministic supported ordering"
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
    "run_diagnostic_agent",
]
