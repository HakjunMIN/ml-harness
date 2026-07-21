from __future__ import annotations

import json
import keyword
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from power_forecasting.aidd import validate_prediction_time_feature_spec
from power_forecasting.features import FeatureSpec


_BASELINE_KEYS = {"model"}
_FEATURE_SPEC_KEYS = {"name", "transform", "inputs", "parameters", "version", "rationale"}
_LIGHTGBM_PARAMETER_VALUES = {
    "n_estimators": {100, 300},
    "learning_rate": {0.03, 0.1},
    "num_leaves": {15, 31},
    "min_child_samples": {10, 20},
}
_LIGHTGBM_SEARCH_KEYS = set(_LIGHTGBM_PARAMETER_VALUES)


class ProposalValidationError(ValueError):
    """Raised when an agent research proposal violates the bounded schema."""


@dataclass(frozen=True)
class FeatureSet:
    name: str
    rationale: str
    specs: tuple[FeatureSpec, ...]

    def __post_init__(self) -> None:
        _identifier_name(self.name, "feature set name")
        _nonblank(self.rationale, f"feature set {self.name}: rationale")
        specs = tuple(self.specs)
        if not specs:
            raise ProposalValidationError(f"feature set {self.name}: specs must be nonempty")
        seen: set[str] = set()
        for spec in specs:
            if not isinstance(spec, FeatureSpec):
                raise ProposalValidationError(f"feature set {self.name}: specs must be FeatureSpec values")
            if spec.name in seen:
                raise ProposalValidationError(f"feature set {self.name}: duplicate feature name {spec.name}")
            seen.add(spec.name)
            validate_prediction_time_feature_spec(spec, error_type=ProposalValidationError)
        object.__setattr__(self, "specs", specs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "specs": [spec.to_dict() for spec in self.specs],
        }


@dataclass(frozen=True)
class ModelRecipe:
    name: str
    recipe: str
    parameters: Mapping[str, Any]
    rationale: str

    def __post_init__(self) -> None:
        _identifier_name(self.name, "model recipe name")
        if self.name.startswith("optuna_lightgbm_"):
            raise ProposalValidationError("model recipe name uses reserved search prefix")
        _nonblank(self.recipe, f"model recipe {self.name}: recipe")
        _nonblank(self.rationale, f"model recipe {self.name}: rationale")
        if not isinstance(self.parameters, Mapping):
            raise ProposalValidationError(f"model recipe {self.name}: parameters must be a mapping")
        params = _validate_recipe_parameters(self.name, self.recipe, self.parameters)
        object.__setattr__(self, "parameters", MappingProxyType(params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "recipe": self.recipe,
            "parameters": dict(self.parameters),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ResearchProposal:
    schema_version: str
    proposal_id: str
    rationale: str
    baseline: Mapping[str, Any]
    feature_sets: tuple[FeatureSet, ...]
    model_recipes: tuple[ModelRecipe, ...]
    budget: Mapping[str, int]
    search: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ProposalValidationError("schema_version must be exactly '1'")
        _nonblank(self.proposal_id, "proposal_id")
        _nonblank(self.rationale, "rationale")
        baseline = _validate_baseline(self.baseline)
        feature_sets = tuple(self.feature_sets)
        model_recipes = tuple(self.model_recipes)
        if not feature_sets:
            raise ProposalValidationError("feature_sets must be nonempty")
        if not model_recipes:
            raise ProposalValidationError("model_recipes must be nonempty")
        _unique_names(feature_sets, "feature set")
        _unique_names(model_recipes, "model recipe")
        search = _validate_search(self.search)
        budget = _validate_budget(self.budget)
        _validate_evaluation_budget(feature_sets, model_recipes, budget, search)
        object.__setattr__(self, "baseline", MappingProxyType(baseline))
        object.__setattr__(self, "feature_sets", feature_sets)
        object.__setattr__(self, "model_recipes", model_recipes)
        object.__setattr__(self, "budget", MappingProxyType(budget))
        object.__setattr__(self, "search", MappingProxyType(search) if search is not None else None)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "rationale": self.rationale,
            "baseline": _json_value(self.baseline),
            "feature_sets": [feature_set.to_dict() for feature_set in self.feature_sets],
            "model_recipes": [recipe.to_dict() for recipe in self.model_recipes],
            "budget": dict(self.budget),
        }
        if self.search is not None:
            payload["search"] = _json_value(self.search)
        return payload


def load_proposal(value: Mapping[str, Any] | Path) -> ResearchProposal:
    if isinstance(value, (str, Path)):
        with Path(value).open("r", encoding="utf-8", newline="") as handle:
            payload = json.load(handle)
    else:
        payload = value
    if not isinstance(payload, Mapping):
        raise ProposalValidationError("proposal must be a mapping")
    _exact_top_level_proposal_keys(payload)
    try:
        feature_sets = tuple(_parse_feature_set(raw) for raw in _require_list(payload, "feature_sets"))
        model_recipes = tuple(_parse_model_recipe(raw) for raw in _require_list(payload, "model_recipes"))
        return ResearchProposal(
            schema_version=payload["schema_version"],
            proposal_id=payload["proposal_id"],
            rationale=payload["rationale"],
            baseline=payload["baseline"],
            feature_sets=feature_sets,
            model_recipes=model_recipes,
            budget=payload["budget"],
            search=payload.get("search"),
        )
    except ProposalValidationError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ProposalValidationError(str(exc) or exc.__class__.__name__) from exc


def proposal_to_dict(proposal: ResearchProposal) -> dict[str, Any]:
    if not isinstance(proposal, ResearchProposal):
        raise TypeError("proposal must be a ResearchProposal")
    return proposal.to_dict()


def _parse_feature_set(raw: Any) -> FeatureSet:
    if not isinstance(raw, Mapping):
        raise ProposalValidationError("feature_sets entries must be mappings")
    _exact_keys(raw, {"name", "rationale", "specs"}, "feature set")
    specs = []
    for spec_payload in _require_list(raw, "specs"):
        if not isinstance(spec_payload, Mapping):
            raise ProposalValidationError("feature specs must be mappings")
        _exact_keys(spec_payload, _FEATURE_SPEC_KEYS, "feature spec")
        specs.append(FeatureSpec.from_dict(spec_payload))
    return FeatureSet(raw["name"], raw["rationale"], tuple(specs))


def _parse_model_recipe(raw: Any) -> ModelRecipe:
    if not isinstance(raw, Mapping):
        raise ProposalValidationError("model_recipes entries must be mappings")
    _exact_keys(raw, {"name", "recipe", "parameters", "rationale"}, "model recipe")
    return ModelRecipe(raw["name"], raw["recipe"], raw["parameters"], raw["rationale"])


def _validate_recipe_parameters(name: str, recipe: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if recipe == "ridge":
        _exact_parameter_keys(parameters, {"alpha"}, name)
        alpha = _number(parameters["alpha"], f"model recipe {name}: alpha")
        if alpha not in {0.1, 1.0, 10.0}:
            raise ProposalValidationError(f"model recipe {name}: alpha outside allowed set")
        return {"alpha": alpha}
    if recipe == "hist_gradient_boosting":
        _exact_parameter_keys(
            parameters,
            {"max_iter", "learning_rate", "max_leaf_nodes"},
            name,
        )
        max_iter = _integer(parameters["max_iter"], f"model recipe {name}: max_iter")
        learning_rate = _number(parameters["learning_rate"], f"model recipe {name}: learning_rate")
        max_leaf_nodes = _integer(parameters["max_leaf_nodes"], f"model recipe {name}: max_leaf_nodes")
        if max_iter not in {50, 100, 200}:
            raise ProposalValidationError(f"model recipe {name}: max_iter outside allowed set")
        if learning_rate not in {0.03, 0.1}:
            raise ProposalValidationError(f"model recipe {name}: learning_rate outside allowed set")
        if max_leaf_nodes not in {15, 31, 63}:
            raise ProposalValidationError(f"model recipe {name}: max_leaf_nodes outside allowed set")
        return {
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
        }
    if recipe == "random_forest":
        _exact_parameter_keys(
            parameters,
            {"n_estimators", "max_depth", "min_samples_leaf"},
            name,
        )
        n_estimators = _integer(parameters["n_estimators"], f"model recipe {name}: n_estimators")
        max_depth = _integer_or_none(parameters["max_depth"], f"model recipe {name}: max_depth")
        min_samples_leaf = _integer(parameters["min_samples_leaf"], f"model recipe {name}: min_samples_leaf")
        if n_estimators not in {100, 200, 400}:
            raise ProposalValidationError(f"model recipe {name}: n_estimators outside allowed set")
        if max_depth not in {8, 12, None}:
            raise ProposalValidationError(f"model recipe {name}: max_depth outside allowed set")
        if min_samples_leaf not in {1, 2, 4}:
            raise ProposalValidationError(f"model recipe {name}: min_samples_leaf outside allowed set")
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
    if recipe == "xgboost":
        _exact_parameter_keys(
            parameters,
            {"n_estimators", "max_depth", "learning_rate", "subsample"},
            name,
        )
        n_estimators = _integer(parameters["n_estimators"], f"model recipe {name}: n_estimators")
        max_depth = _integer(parameters["max_depth"], f"model recipe {name}: max_depth")
        learning_rate = _number(parameters["learning_rate"], f"model recipe {name}: learning_rate")
        subsample = _number(parameters["subsample"], f"model recipe {name}: subsample")
        if n_estimators not in {100, 200, 400}:
            raise ProposalValidationError(f"model recipe {name}: n_estimators outside allowed set")
        if max_depth not in {4, 6, 8}:
            raise ProposalValidationError(f"model recipe {name}: max_depth outside allowed set")
        if learning_rate not in {0.03, 0.1}:
            raise ProposalValidationError(f"model recipe {name}: learning_rate outside allowed set")
        if subsample not in {0.8, 1.0}:
            raise ProposalValidationError(f"model recipe {name}: subsample outside allowed set")
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
        }
    if recipe == "lightgbm":
        _exact_parameter_keys(
            parameters,
            _LIGHTGBM_SEARCH_KEYS,
            name,
        )
        n_estimators = _integer(parameters["n_estimators"], f"model recipe {name}: n_estimators")
        learning_rate = _number(parameters["learning_rate"], f"model recipe {name}: learning_rate")
        num_leaves = _integer(parameters["num_leaves"], f"model recipe {name}: num_leaves")
        min_child_samples = _integer(
            parameters["min_child_samples"],
            f"model recipe {name}: min_child_samples",
        )
        normalized = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
        }
        for parameter, value in normalized.items():
            if value not in _LIGHTGBM_PARAMETER_VALUES[parameter]:
                raise ProposalValidationError(f"model recipe {name}: {parameter} outside allowed set")
        return normalized
    raise ProposalValidationError(f"model recipe {name}: unsupported recipe {recipe}")


def _validate_budget(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise ProposalValidationError("budget must be a mapping")
    _exact_keys(raw, {"max_evaluations", "top_feature_groups"}, "budget")
    max_evaluations = _integer(raw["max_evaluations"], "budget.max_evaluations")
    top_feature_groups = _integer(raw["top_feature_groups"], "budget.top_feature_groups")
    if not 1 <= max_evaluations <= 50:
        raise ProposalValidationError("budget.max_evaluations must be 1..50")
    if not 1 <= top_feature_groups <= 10:
        raise ProposalValidationError("budget.top_feature_groups must be 1..10")
    return {"max_evaluations": max_evaluations, "top_feature_groups": top_feature_groups}


def _validate_evaluation_budget(
    feature_sets: tuple[FeatureSet, ...],
    model_recipes: tuple[ModelRecipe, ...],
    budget: Mapping[str, int],
    search: Mapping[str, Any] | None = None,
) -> None:
    search_trials = int(search["n_trials"]) if search is not None else 0
    combination_count = len(feature_sets) * (len(model_recipes) + search_trials)
    max_evaluations = int(budget["max_evaluations"])
    if combination_count > max_evaluations:
        raise ProposalValidationError(
            "proposal combinations exceed max_evaluations:"
            f" {combination_count}>{max_evaluations}"
        )


def _validate_baseline(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise ProposalValidationError("baseline must be a mapping")
    _exact_keys(raw, _BASELINE_KEYS, "baseline")
    if raw["model"] != "SPOT":
        raise ProposalValidationError("baseline.model must be exactly 'SPOT'")
    return {"model": "SPOT"}


def _exact_top_level_proposal_keys(payload: Mapping[str, Any]) -> None:
    required = {"schema_version", "proposal_id", "rationale", "baseline", "feature_sets", "model_recipes", "budget"}
    allowed = required | {"search"}
    actual = set(payload)
    extra = sorted(actual - allowed)
    missing = sorted(required - actual)
    if extra:
        raise ProposalValidationError(f"proposal unknown keys: {extra}")
    if missing:
        raise ProposalValidationError(f"proposal missing keys: {missing}")


def _validate_search(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ProposalValidationError("search must be a mapping")
    _exact_keys(raw, {"sampler", "seed", "n_trials", "spaces"}, "search")
    if raw["sampler"] != "tpe":
        raise ProposalValidationError("search.sampler must be exactly 'tpe'")
    seed = _integer(raw["seed"], "search.seed")
    if seed < 0:
        raise ProposalValidationError("search.seed must be nonnegative")
    n_trials = _integer(raw["n_trials"], "search.n_trials")
    if not 1 <= n_trials <= 50:
        raise ProposalValidationError("search.n_trials must be 1..50")
    spaces = _validate_search_spaces(raw["spaces"])
    return {
        "sampler": "tpe",
        "seed": seed,
        "n_trials": n_trials,
        "spaces": spaces,
    }


def _validate_search_spaces(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProposalValidationError("search.spaces must be a mapping")
    _exact_keys(raw, {"lightgbm"}, "search.spaces")
    lightgbm = raw["lightgbm"]
    if not isinstance(lightgbm, Mapping):
        raise ProposalValidationError("search.spaces.lightgbm must be a mapping")
    _exact_keys(lightgbm, _LIGHTGBM_SEARCH_KEYS, "search.spaces.lightgbm")
    return {
        "lightgbm": {
            parameter: _validate_discrete_search_values(
                parameter,
                lightgbm[parameter],
                _LIGHTGBM_PARAMETER_VALUES[parameter],
            )
            for parameter in sorted(_LIGHTGBM_SEARCH_KEYS)
        }
    }


def _validate_discrete_search_values(
    parameter: str,
    raw: Any,
    allowed: set[int | float],
) -> list[int | float]:
    if not isinstance(raw, list):
        raise ProposalValidationError(f"search.spaces.lightgbm.{parameter} must be a list")
    if not raw:
        raise ProposalValidationError(f"search.spaces.lightgbm.{parameter} must be nonempty")
    values: list[int | float] = []
    for value in raw:
        if parameter == "learning_rate":
            normalized: int | float = _number(value, f"search.spaces.lightgbm.{parameter}")
        else:
            normalized = _integer(value, f"search.spaces.lightgbm.{parameter}")
        if normalized not in allowed:
            raise ProposalValidationError(
                f"search.spaces.lightgbm.{parameter} contains value outside allowed set"
            )
        values.append(normalized)
    return values


def _exact_keys(mapping: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        if extra:
            raise ProposalValidationError(f"{label} unknown keys: {extra}")
        raise ProposalValidationError(f"{label} missing keys: {missing}")


def _exact_parameter_keys(mapping: Mapping[str, Any], expected: set[str], recipe_name: str) -> None:
    actual = set(mapping)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        if extra:
            raise ProposalValidationError(f"model recipe {recipe_name}: unknown parameters {extra}")
        raise ProposalValidationError(f"model recipe {recipe_name}: missing parameters {missing}")


def _require_list(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping[key]
    if not isinstance(value, list):
        raise ProposalValidationError(f"{key} must be a list")
    return value


def _unique_names(values: tuple[Any, ...], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value.name in seen:
            raise ProposalValidationError(f"duplicate {label} name: {value.name}")
        seen.add(value.name)


def _nonblank(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ProposalValidationError(f"{label} must be a nonblank string")
    return value


def _identifier_name(value: Any, label: str) -> str:
    _nonblank(value, label)
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ProposalValidationError(f"{label} must be an identifier string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProposalValidationError(f"{label} must be an integer")
    return int(value)


def _integer_or_none(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _integer(value, label)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProposalValidationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProposalValidationError(f"{label} must be finite")
    return number


def _json_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposalValidationError(f"{label} must be a mapping")
    return _json_value(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if value is None or type(value) in {str, bool}:
        return value
    if type(value) in {int, float}:
        number = float(value)
        if not math.isfinite(number):
            raise ProposalValidationError("proposal contains non-finite number")
        return value
    raise ProposalValidationError(f"proposal contains non-JSON value {type(value).__name__}")


__all__ = [
    "FeatureSet",
    "ModelRecipe",
    "ProposalValidationError",
    "ResearchProposal",
    "load_proposal",
    "proposal_to_dict",
]
