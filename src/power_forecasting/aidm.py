from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from power_forecasting.data import DataContractError, validate_dataset
from power_forecasting.evaluation import evaluate_model
from power_forecasting.experiments import ExperimentStore
from power_forecasting.features import FeatureSpec
from power_forecasting.models import ModelDefinition, model_definition


@dataclass(frozen=True)
class AIDMConfig:
    folds: int = 3
    minimum_improvement: float = 0.01
    max_plant_regression: float = 0.03
    top_single_candidates: int = 3
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.folds, bool) or not isinstance(self.folds, int):
            raise TypeError("folds must be an integer")
        if self.folds < 1:
            raise ValueError("folds must be >= 1")
        if isinstance(self.minimum_improvement, bool) or not isinstance(
            self.minimum_improvement, (int, float)
        ):
            raise TypeError("minimum_improvement must be numeric")
        if not math.isfinite(float(self.minimum_improvement)) or not (
            0.0 <= float(self.minimum_improvement) <= 1.0
        ):
            raise ValueError("minimum_improvement must be between 0 and 1")
        if isinstance(self.max_plant_regression, bool) or not isinstance(
            self.max_plant_regression, (int, float)
        ):
            raise TypeError("max_plant_regression must be numeric")
        if not math.isfinite(float(self.max_plant_regression)) or not (
            0.0 <= float(self.max_plant_regression) <= 1.0
        ):
            raise ValueError("max_plant_regression must be between 0 and 1")
        if isinstance(self.top_single_candidates, bool) or not isinstance(
            self.top_single_candidates, int
        ):
            raise TypeError("top_single_candidates must be an integer")
        if self.top_single_candidates < 1:
            raise ValueError("top_single_candidates must be >= 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


@dataclass(frozen=True)
class CandidateResult:
    name: str
    specs: tuple[FeatureSpec, ...]
    metrics: Mapping[str, float]
    per_plant: Mapping[str, Mapping[str, float]]
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("candidate name must be a non-empty string")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string")

        specs = tuple(sorted(tuple(self.specs), key=lambda spec: spec.name))
        if not all(isinstance(spec, FeatureSpec) for spec in specs):
            raise TypeError("specs must contain FeatureSpec values")

        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "metrics", _normalize_metrics(self.metrics))
        object.__setattr__(self, "per_plant", _normalize_per_plant(self.per_plant))

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "specs": [spec.to_dict() for spec in self.specs],
            "metrics": _copy_metrics(self.metrics),
            "per_plant": _copy_per_plant(self.per_plant),
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class AIDMResult:
    baseline: CandidateResult
    candidates: tuple[CandidateResult, ...]
    ranking: tuple[str, ...]
    winner: CandidateResult | None
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "ranking", tuple(self.ranking))
        object.__setattr__(self, "manifest", _json_safe_mapping(self.manifest))


@dataclass(frozen=True)
class _CandidateGroup:
    name: str
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        specs = tuple(sorted(tuple(self.specs), key=lambda spec: spec.name))
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "name", stable_candidate_name(specs))


def candidate_catalog() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",)),
        FeatureSpec("hour_cos", "cyclic_hour", ("timestamp",)),
        FeatureSpec("doy_sin", "cyclic_day_of_year", ("timestamp",)),
        FeatureSpec("doy_cos", "cyclic_day_of_year", ("timestamp",)),
        FeatureSpec(
            "effective_irradiance",
            "effective_irradiance",
            ("forecast_irradiance", "forecast_cloud_cover"),
        ),
        FeatureSpec(
            "temperature_derating",
            "temperature_derating",
            ("forecast_irradiance", "forecast_temperature"),
        ),
        FeatureSpec("cloud_attenuation", "cloud_attenuation", ("forecast_cloud_cover",)),
        FeatureSpec(
            "irradiance_temperature_interaction",
            "interaction",
            ("forecast_irradiance", "forecast_temperature"),
        ),
    )


def stable_candidate_name(specs: Sequence[FeatureSpec]) -> str:
    names = sorted(spec.name for spec in specs)
    return "+".join(names) if names else "baseline"


def evaluate_promotion_gates(
    baseline: CandidateResult,
    winner: CandidateResult,
    config: AIDMConfig = AIDMConfig(),
) -> dict[str, Any]:
    _validate_config(config)
    baseline_nmae = _nmae(baseline)
    winner_nmae = _nmae(winner)
    improvement_ratio = _improvement_ratio(baseline_nmae, winner_nmae)
    per_plant_deltas = _per_plant_deltas(baseline, winner)

    failed_gates: list[str] = []
    if improvement_ratio < config.minimum_improvement:
        failed_gates.append(
            "insufficient_improvement:"
            f"improvement_ratio={_format_float(improvement_ratio)}"
            f"<threshold={_format_float(config.minimum_improvement)}"
        )

    for plant_id, delta in per_plant_deltas.items():
        if delta > config.max_plant_regression:
            failed_gates.append(
                f"plant_regression:{plant_id}:"
                f"delta={_format_float(delta)}"
                f">threshold={_format_float(config.max_plant_regression)}"
            )

    for source in _unavailable_inputs(winner.specs):
        failed_gates.append(f"unavailable_input:{source}")

    return {
        "decision": "reject" if failed_gates else "promote",
        "failed_gates": failed_gates,
        "per_plant_deltas": per_plant_deltas,
        "improvement_ratio": _round_float(improvement_ratio),
    }


def run_aidm(
    frame: pd.DataFrame,
    database_path,
    config: AIDMConfig = AIDMConfig(),
) -> AIDMResult:
    _validate_config(config)
    if not isinstance(frame, pd.DataFrame):
        raise DataContractError("frame must be a pandas DataFrame")
    validate_dataset(frame)

    store = ExperimentStore(database_path)
    definition = model_definition("SPOT")

    baseline = _evaluate_and_record(
        store=store,
        frame=frame,
        definition=definition,
        specs=(),
        candidate_name="baseline",
        run_name="aidm-baseline-spot",
        config=config,
    )

    single_results = []
    for group in _candidate_groups():
        single_results.append(
            _evaluate_and_record(
                store=store,
                frame=frame,
                definition=definition,
                specs=group.specs,
                candidate_name=group.name,
                run_name=f"aidm-candidate-{group.name}",
                config=config,
            )
        )

    retained = tuple(
        sorted(single_results, key=lambda candidate: (_nmae(candidate), candidate.name))[
            : config.top_single_candidates
        ]
    )

    combination_results = []
    retained_groups = tuple(
        _CandidateGroup(result.name, result.specs) for result in retained
    )
    for group in _combination_groups(retained_groups):
        combination_results.append(
            _evaluate_and_record(
                store=store,
                frame=frame,
                definition=definition,
                specs=group.specs,
                candidate_name=group.name,
                run_name=f"aidm-candidate-{group.name}",
                config=config,
            )
        )

    candidates = tuple(single_results + combination_results)
    ranked = tuple(sorted(candidates, key=lambda candidate: (_nmae(candidate), candidate.name)))
    ranking = tuple(candidate.name for candidate in ranked)
    winner = ranked[0] if ranked else None

    gates = (
        evaluate_promotion_gates(baseline, winner, config)
        if winner is not None
        else {
            "decision": "reject",
            "failed_gates": ["no_candidate"],
            "per_plant_deltas": {},
            "improvement_ratio": 0.0,
        }
    )
    manifest = _manifest(config, baseline, winner, gates)

    return AIDMResult(
        baseline=baseline,
        candidates=candidates,
        ranking=ranking,
        winner=winner,
        manifest=manifest,
    )


def _candidate_groups() -> tuple[_CandidateGroup, ...]:
    specs = {spec.name: spec for spec in candidate_catalog()}
    return (
        _CandidateGroup("hour", (specs["hour_sin"], specs["hour_cos"])),
        _CandidateGroup("day_of_year", (specs["doy_sin"], specs["doy_cos"])),
        _CandidateGroup("effective_irradiance", (specs["effective_irradiance"],)),
        _CandidateGroup("temperature_derating", (specs["temperature_derating"],)),
        _CandidateGroup("cloud_attenuation", (specs["cloud_attenuation"],)),
        _CandidateGroup(
            "irradiance_temperature_interaction",
            (specs["irradiance_temperature_interaction"],),
        ),
    )


def _combination_groups(groups: Sequence[_CandidateGroup]) -> tuple[_CandidateGroup, ...]:
    combinations = []
    seen_names: set[str] = set()
    for size in (2, 3):
        for selected in itertools.combinations(groups, size):
            specs = _dedupe_specs(
                spec for group in selected for spec in group.specs
            )
            name = stable_candidate_name(specs)
            if name in seen_names:
                continue
            seen_names.add(name)
            combinations.append(_CandidateGroup(name, specs))
    return tuple(combinations)


def _dedupe_specs(specs: Iterable[FeatureSpec]) -> tuple[FeatureSpec, ...]:
    by_name = {}
    for spec in specs:
        by_name.setdefault(spec.name, spec)
    return tuple(by_name[name] for name in sorted(by_name))


def _evaluate_and_record(
    *,
    store: ExperimentStore,
    frame: pd.DataFrame,
    definition: ModelDefinition,
    specs: Sequence[FeatureSpec],
    candidate_name: str,
    run_name: str,
    config: AIDMConfig,
) -> CandidateResult:
    specs = tuple(sorted(tuple(specs), key=lambda spec: spec.name))
    params = {
        "schema_version": "1",
        "candidate_name": candidate_name,
        "model": definition.name,
        "folds": config.folds,
        "seed": config.seed,
        "specs": [spec.to_dict() for spec in specs],
    }
    run_id = store.start_run(run_name, params)

    try:
        evaluation = evaluate_model(frame, definition, specs, folds=config.folds)
        candidate = CandidateResult(
            name=candidate_name,
            specs=specs,
            metrics=_normalize_metrics(evaluation.metrics),
            per_plant=_normalize_per_plant(evaluation.per_plant),
            run_id=run_id,
        )
        artifacts = {
            "summary": candidate.summary(),
            "fold_metrics": [
                _normalize_metrics(metrics) for metrics in evaluation.fold_metrics
            ],
            "prediction_rows": int(len(evaluation.predictions)),
        }
        store.complete_run(run_id, candidate.metrics, artifacts)
        return candidate
    except Exception as exc:
        store.fail_run(run_id, str(exc) or exc.__class__.__name__)
        raise


def _manifest(
    config: AIDMConfig,
    baseline: CandidateResult,
    winner: CandidateResult | None,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "seed": config.seed,
        "baseline": {
            "model": "SPOT",
            "metrics": _copy_metrics(baseline.metrics),
            "run_id": baseline.run_id,
        },
        "winner": None
        if winner is None
        else {
            "name": winner.name,
            "metrics": _copy_metrics(winner.metrics),
            "run_id": winner.run_id,
        },
        "selected_specs": []
        if winner is None
        else [spec.to_dict() for spec in winner.specs],
        "per_plant_deltas": dict(gates["per_plant_deltas"]),
        "thresholds": {
            "minimum_improvement": float(config.minimum_improvement),
            "max_plant_regression": float(config.max_plant_regression),
        },
        "improvement_ratio": float(gates["improvement_ratio"]),
        "decision": gates["decision"],
        "failed_gates": list(gates["failed_gates"]),
    }


def _validate_config(config: AIDMConfig) -> None:
    if not isinstance(config, AIDMConfig):
        raise TypeError("config must be an AIDMConfig")


def _normalize_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics must be a mapping")
    normalized = {
        str(key).lower(): _finite_float(f"metric {key}", value)
        for key, value in metrics.items()
    }
    if "nmae" not in normalized:
        raise ValueError("metrics must include nmae")
    return {key: normalized[key] for key in sorted(normalized)}


def _normalize_per_plant(
    per_plant: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    if not isinstance(per_plant, Mapping):
        raise ValueError("per_plant must be a mapping")
    return {
        str(plant_id): _normalize_metrics(metrics)
        for plant_id, metrics in sorted(per_plant.items(), key=lambda item: str(item[0]))
    }


def _copy_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in sorted(metrics)}


def _copy_per_plant(
    per_plant: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    return {
        plant_id: _copy_metrics(metrics)
        for plant_id, metrics in sorted(per_plant.items())
    }


def _finite_float(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nmae(result: CandidateResult) -> float:
    return float(result.metrics["nmae"])


def _improvement_ratio(baseline_nmae: float, winner_nmae: float) -> float:
    if baseline_nmae <= 0:
        return 0.0 if winner_nmae <= baseline_nmae else -1.0
    return (baseline_nmae - winner_nmae) / baseline_nmae


def _per_plant_deltas(
    baseline: CandidateResult, winner: CandidateResult
) -> dict[str, float]:
    baseline_plants = set(baseline.per_plant)
    winner_plants = set(winner.per_plant)
    if baseline_plants != winner_plants:
        raise ValueError("baseline and winner must contain identical per-plant metrics")
    return {
        plant_id: _round_float(
            float(winner.per_plant[plant_id]["nmae"])
            - float(baseline.per_plant[plant_id]["nmae"])
        )
        for plant_id in sorted(baseline_plants)
    }


def _unavailable_inputs(specs: Sequence[FeatureSpec]) -> tuple[str, ...]:
    unavailable = {
        source
        for spec in specs
        for source in spec.inputs
        if source == "generation_mw" or source.startswith("actual_")
    }
    return tuple(sorted(unavailable))


def _round_float(value: float) -> float:
    return float(round(float(value), 12))


def _format_float(value: float) -> str:
    return f"{float(value):.6f}"


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("manifest must be a mapping")
    return _json_safe_value(dict(value))


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("manifest contains non-finite float")
        return float(value)
    return value


__all__ = [
    "AIDMConfig",
    "AIDMResult",
    "CandidateResult",
    "candidate_catalog",
    "evaluate_promotion_gates",
    "run_aidm",
    "stable_candidate_name",
]
