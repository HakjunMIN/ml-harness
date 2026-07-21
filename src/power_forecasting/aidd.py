from __future__ import annotations

import math
import os
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from power_forecasting.data import REQUIRED_COLUMNS
from power_forecasting.features import (
    FeatureSpec,
    TRANSFORMS,
    _TRANSFORM_ARITY,
    apply_feature_specs,
)


class PromotionManifestError(ValueError):
    """Raised when an AIDM promotion manifest cannot be trusted for rendering."""


_DATETIME_TRANSFORMS = frozenset({"cyclic_hour", "cyclic_day_of_year"})
_HISTORY_TRANSFORMS = frozenset({"lag", "rolling_mean"})
_NUMERIC_TRANSFORMS = frozenset(TRANSFORMS) - _DATETIME_TRANSFORMS
_PREDICTION_TIME_PREFIXES = ("forecast_", "ldaps_")
_NUMERIC_METADATA_INPUTS = frozenset({"capacity_mw", "latitude", "longitude"})


def validate_promotion_manifest(manifest) -> tuple[FeatureSpec, ...]:
    try:
        if not isinstance(manifest, Mapping):
            raise PromotionManifestError("promotion manifest must be a mapping")
        _require_exact(manifest, "schema_version", "1")
        _require_exact(manifest, "decision", "promote")

        selected_specs = _require_key(manifest, "selected_specs")
        if not isinstance(selected_specs, list) or not selected_specs:
            raise PromotionManifestError("selected_specs must be a non-empty list")

        specs = _parse_specs(selected_specs)
        _validate_provenance(manifest, specs)
        return specs
    except PromotionManifestError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PromotionManifestError(str(exc) or exc.__class__.__name__) from exc


def render_promoted_module(manifest, target: Path) -> Path:
    specs = validate_promotion_manifest(manifest)
    _reject_stateful_history_features_for_rendering(specs)
    target = Path(target)
    content = _render_module(
        specs=specs,
        schema_version="1",
        winner_name=_winner_name_for_specs(specs),
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, content)
    return target


def validate_prediction_time_feature_spec(
    spec: FeatureSpec, *, error_type: type[Exception] = PromotionManifestError
) -> None:
    try:
        _validate_spec_primitives(spec)
        _validate_transform_and_arity(spec)
        if spec.transform in _HISTORY_TRANSFORMS:
            _validate_history_parameters(spec)
        _reject_unavailable_inputs(spec)
        _validate_prediction_time_inputs(spec)
        _validate_spec_without_source_existence(spec)
        _canonical_spec_dict(spec)
    except PromotionManifestError as exc:
        if error_type is PromotionManifestError:
            raise
        raise error_type(str(exc)) from exc


def render_model_recipe_patch(manifest, target: Path) -> Path:
    payload = _model_recipe_patch_payload(manifest)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    _atomic_write_text(target, content)
    return target


def _parse_specs(selected_specs: list[Any]) -> tuple[FeatureSpec, ...]:
    parsed: list[FeatureSpec] = []
    seen_names: set[str] = set()
    for index, raw_spec in enumerate(selected_specs):
        if not isinstance(raw_spec, Mapping):
            raise PromotionManifestError(f"selected_specs[{index}] must be a mapping")
        spec = FeatureSpec.from_dict(raw_spec)
        if spec.name in seen_names:
            raise PromotionManifestError(f"duplicate feature name: {spec.name}")
        seen_names.add(spec.name)
        validate_prediction_time_feature_spec(spec)
        parsed.append(spec)
    return tuple(parsed)


def _validate_spec_primitives(spec: FeatureSpec) -> None:
    _require_exact_str_value("feature name", spec.name)
    _require_exact_str_value(f"feature {spec.name}: transform", spec.transform)
    for index, source in enumerate(spec.inputs):
        _require_exact_str_value(f"feature {spec.name}: inputs[{index}]", source)
    _require_exact_str_value(f"feature {spec.name}: version", spec.version)
    _require_exact_str_value(f"feature {spec.name}: rationale", spec.rationale)
    for key, value in spec.parameters.items():
        _require_exact_str_value(f"feature {spec.name}: parameter name", key)
        _validate_primitive_value(f"feature {spec.name}: parameter {key}", value)


def _validate_primitive_value(label: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_exact_str_value(f"{label} key", key)
            _validate_primitive_value(f"{label}.{key}", nested)
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_primitive_value(f"{label}[{index}]", item)
        return
    if value is None or type(value) in {str, bool}:
        return
    if type(value) in {int, float}:
        _finite_number(label, value)
        return
    raise PromotionManifestError(
        f"{label} must contain primitive literal values, got {type(value).__name__}"
    )


def _validate_transform_and_arity(spec: FeatureSpec) -> None:
    if spec.transform not in TRANSFORMS:
        raise PromotionManifestError(
            f"feature {spec.name}: unknown transform {spec.transform}"
        )
    expected_arity = _TRANSFORM_ARITY[spec.transform]
    if len(spec.inputs) != expected_arity:
        raise PromotionManifestError(
            f"feature {spec.name}: {spec.transform} expects {expected_arity} inputs"
        )


def _reject_unavailable_inputs(spec: FeatureSpec) -> None:
    for source in spec.inputs:
        if source == "generation_mw":
            raise PromotionManifestError(
                f"feature {spec.name}: target leakage input {source}"
            )
        if source.startswith("actual_"):
            raise PromotionManifestError(
                f"feature {spec.name}: unavailable actual input {source}"
            )


def _validate_prediction_time_inputs(spec: FeatureSpec) -> None:
    allowed = _prediction_time_inputs_for_transform(spec.transform)
    for source in spec.inputs:
        if source not in allowed:
            raise PromotionManifestError(
                f"feature {spec.name}: unavailable prediction input {source}"
            )


def _prediction_time_inputs_for_transform(transform: str) -> frozenset[str]:
    required = frozenset(REQUIRED_COLUMNS)
    if transform in _DATETIME_TRANSFORMS:
        return frozenset(source for source in required if source == "timestamp")
    if transform in _NUMERIC_TRANSFORMS:
        return frozenset(
            source
            for source in required
            if source.startswith(_PREDICTION_TIME_PREFIXES)
            or source in _NUMERIC_METADATA_INPUTS
        )
    return frozenset()


def _validate_spec_without_source_existence(spec: FeatureSpec) -> None:
    if spec.transform in _HISTORY_TRANSFORMS:
        _validate_history_parameters(spec)
        return
    row = {
        source: _dummy_value_for_transform(spec, position)
        for position, source in enumerate(spec.inputs)
    }
    frame = pd.DataFrame({source: [value] for source, value in row.items()})
    if not row:
        frame = pd.DataFrame(index=[0])
    try:
        apply_feature_specs(frame, [spec])
    except (ValueError, OverflowError) as exc:
        raise PromotionManifestError(str(exc)) from exc


def _dummy_value_for_transform(spec: FeatureSpec, position: int) -> Any:
    if spec.transform in {"cyclic_hour", "cyclic_day_of_year"}:
        return pd.Timestamp("2024-01-01 12:00:00")
    if spec.transform == "ratio" and position == 1:
        epsilon = _finite_float(
            f"feature {spec.name}: parameter epsilon",
            spec.parameters.get("epsilon", 1e-6),
        )
        denominator = math.nextafter(epsilon, math.inf)
        if not math.isfinite(denominator) or denominator <= epsilon:
            raise PromotionManifestError("ratio epsilon is too large for validation")
        return max(2.0, denominator)
    return 1.0


def _validate_history_parameters(spec: FeatureSpec) -> None:
    if spec.transform == "lag":
        _validate_exact_integer_parameter(spec, "periods", {1, 2, 3, 6, 12, 24})
    elif spec.transform == "rolling_mean":
        _validate_exact_integer_parameter(spec, "window", {3, 6, 12, 24})


def _validate_exact_integer_parameter(
    spec: FeatureSpec, key: str, allowed: set[int]
) -> None:
    unexpected = sorted(set(spec.parameters) - {key})
    if unexpected:
        raise PromotionManifestError(f"feature {spec.name}: unexpected parameters: {unexpected}")
    if key not in spec.parameters:
        raise PromotionManifestError(f"feature {spec.name}: missing parameter {key}")
    value = spec.parameters[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromotionManifestError(f"feature {spec.name}: parameter {key} must be an integer")
    if value not in allowed:
        raise PromotionManifestError(f"feature {spec.name}: parameter {key} outside allowed set")


def _reject_stateful_history_features_for_rendering(specs: tuple[FeatureSpec, ...]) -> None:
    for spec in specs:
        if spec.transform in _HISTORY_TRANSFORMS:
            raise PromotionManifestError(
                f"feature {spec.name}: stateful history feature cannot be rendered"
            )


def _validate_provenance(
    manifest: Mapping[str, Any], specs: tuple[FeatureSpec, ...]
) -> None:
    failed_gates = _require_key(manifest, "failed_gates")
    if not isinstance(failed_gates, list):
        raise PromotionManifestError("failed_gates must be a list")
    if failed_gates:
        raise PromotionManifestError("promoted manifest must not contain failed gates")

    baseline = _require_mapping(manifest, "baseline")
    winner = _require_mapping(manifest, "winner")
    thresholds = _require_mapping(manifest, "thresholds")
    per_plant_deltas = _require_mapping(manifest, "per_plant_deltas")

    _require_aidm_seed(manifest)
    baseline_model = _require_nonblank_string(baseline, "model", "baseline.model")
    if baseline_model != "SPOT":
        raise PromotionManifestError("baseline.model must be exactly 'SPOT'")
    _require_nonempty_string(baseline, "run_id", "baseline.run_id")
    _require_nonempty_string(winner, "name", "winner.name")
    _require_nonempty_string(winner, "run_id", "winner.run_id")
    selected_recipe = manifest.get("selected_model_recipe")
    expected_winner_name = (
        _require_nonblank_string(winner, "name", "winner.name")
        if isinstance(selected_recipe, Mapping)
        else _winner_name_for_specs(specs)
    )
    if winner["name"] != expected_winner_name:
        raise PromotionManifestError("winner.name does not match selected_specs")

    baseline_metrics = _metrics(_require_mapping(baseline, "metrics"), "baseline.metrics")
    winner_metrics = _metrics(_require_mapping(winner, "metrics"), "winner.metrics")
    minimum_improvement = _finite_number(
        "thresholds.minimum_improvement",
        _require_key(thresholds, "minimum_improvement"),
    )
    max_plant_regression = _finite_number(
        "thresholds.max_plant_regression",
        _require_key(thresholds, "max_plant_regression"),
    )
    if not 0.0 <= minimum_improvement <= 1.0:
        raise PromotionManifestError("thresholds.minimum_improvement must be between 0 and 1")
    if not 0.0 <= max_plant_regression <= 1.0:
        raise PromotionManifestError("thresholds.max_plant_regression must be between 0 and 1")

    improvement_ratio = _finite_float(
        "improvement_ratio", _require_key(manifest, "improvement_ratio")
    )
    raw_improvement = _raw_improvement_ratio(
        baseline_metrics["nmae"], winner_metrics["nmae"]
    )
    expected_improvement = _round_float(raw_improvement)
    if improvement_ratio != expected_improvement:
        raise PromotionManifestError("improvement_ratio does not match baseline/winner nmae")
    if raw_improvement < minimum_improvement:
        raise PromotionManifestError("improvement_ratio does not satisfy threshold")

    for plant_id, value in per_plant_deltas.items():
        if type(plant_id) is not str or not plant_id:
            raise PromotionManifestError("per_plant_deltas keys must be non-empty strings")
        delta = _finite_float(f"per_plant_deltas.{plant_id}", value)
        if delta > max_plant_regression:
            raise PromotionManifestError(
                f"per_plant_deltas.{plant_id} exceeds max_plant_regression"
            )


def _metrics(metrics: Mapping[str, Any], label: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        if type(key) is not str or not key:
            raise PromotionManifestError(f"{label} keys must be non-empty strings")
        metric_name = key.lower()
        number = _finite_float(f"{label}.{key}", value)
        if metric_name in {"mae", "rmse", "nmae"} and number < 0:
            raise PromotionManifestError(f"{label}.{key} must be non-negative")
        normalized[metric_name] = number
    if "nmae" not in normalized:
        raise PromotionManifestError(f"{label} must include nmae")
    return normalized


def _model_recipe_patch_payload(manifest: Any) -> dict[str, Any]:
    specs = validate_promotion_manifest(manifest)
    if not isinstance(manifest, Mapping):
        raise PromotionManifestError("promotion manifest must be a mapping")
    selected_recipe = _require_mapping(manifest, "selected_model_recipe")
    recipe = _canonical_model_recipe(selected_recipe)
    proposal = _validated_embedded_proposal(_require_mapping(manifest, "proposal"))
    proposal_id = _require_nonblank_string(proposal, "proposal_id", "proposal.proposal_id")
    _reject_unsafe_string("proposal.proposal_id", proposal_id)
    baseline = _require_mapping(manifest, "baseline")
    winner = _require_mapping(manifest, "winner")
    _validate_agentic_model_recipe_binding(proposal, winner, recipe, specs)
    winner_metrics = _metrics(_require_mapping(winner, "metrics"), "winner.metrics")
    failed_gates = _require_key(manifest, "failed_gates")
    if failed_gates != []:
        raise PromotionManifestError("promoted recipe manifest must not contain failed gates")
    return {
        "evidence": {
            "baseline_model": _require_nonblank_string(baseline, "model", "baseline.model"),
            "failed_gates": [],
            "proposal_id": proposal_id,
            "winner_name": _safe_nonblank_string(winner, "name", "winner.name"),
        },
        "manifest_sha256": _sha_json(manifest),
        "schema_version": "1",
        "selected_feature_specs_sha256": _sha_json([spec.to_dict() for spec in specs]),
        "selected_model_recipe": recipe,
        "status": "requires_human_review",
        "winner_metrics": winner_metrics,
    }


def _validated_embedded_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    if _contains_sensitive_key(proposal):
        raise PromotionManifestError("proposal contains unsupported path-like evidence")
    from power_forecasting.proposals import (
        ProposalValidationError,
        load_proposal,
        proposal_to_dict,
    )

    try:
        return proposal_to_dict(load_proposal(proposal))
    except ProposalValidationError as exc:
        raise PromotionManifestError(str(exc)) from exc


def _validate_agentic_model_recipe_binding(
    proposal: Mapping[str, Any],
    winner: Mapping[str, Any],
    selected_recipe: Mapping[str, Any],
    selected_specs: tuple[FeatureSpec, ...],
) -> None:
    feature_sets = _require_list(proposal, "feature_sets", "proposal.feature_sets")
    model_recipes = _require_list(proposal, "model_recipes", "proposal.model_recipes")

    if selected_recipe not in model_recipes:
        raise PromotionManifestError("selected_model_recipe does not match proposal.model_recipes")

    winner_name = _safe_nonblank_string(winner, "name", "winner.name")
    recipe_prefix = f"{selected_recipe['name']}:"
    if not winner_name.startswith(recipe_prefix):
        raise PromotionManifestError("winner.name does not match selected_model_recipe")
    feature_set_name = winner_name[len(recipe_prefix) :]
    if not feature_set_name:
        raise PromotionManifestError("winner.name missing feature set name")

    expected_winner_name = f"{selected_recipe['name']}:{feature_set_name}"
    if winner_name != expected_winner_name:
        raise PromotionManifestError("winner.name does not bind recipe and feature set")

    matching_feature_sets = [feature_set for feature_set in feature_sets if feature_set["name"] == feature_set_name]
    if len(matching_feature_sets) != 1:
        raise PromotionManifestError("winner.name feature set not found in proposal.feature_sets")

    selected_specs_payload = _spec_payloads_by_name([spec.to_dict() for spec in selected_specs])
    feature_set_specs_payload = _spec_payloads_by_name(matching_feature_sets[0]["specs"])
    if selected_specs_payload != feature_set_specs_payload:
        raise PromotionManifestError("selected_specs do not match proposal feature set")


def _spec_payloads_by_name(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(specs, key=lambda spec: spec["name"])


def _canonical_model_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    allowed_top = {"name", "recipe", "parameters", "rationale"}
    extra = set(recipe) - allowed_top
    missing = allowed_top - set(recipe)
    if extra:
        raise PromotionManifestError(f"selected_model_recipe unknown keys: {sorted(extra)}")
    if missing:
        raise PromotionManifestError(f"selected_model_recipe missing keys: {sorted(missing)}")
    name = _require_nonblank_string(recipe, "name", "selected_model_recipe.name")
    recipe_name = _require_nonblank_string(recipe, "recipe", "selected_model_recipe.recipe")
    rationale = _require_nonblank_string(recipe, "rationale", "selected_model_recipe.rationale")
    _reject_unsafe_string("selected_model_recipe.name", name)
    _reject_unsafe_string("selected_model_recipe.recipe", recipe_name)
    _reject_unsafe_string("selected_model_recipe.rationale", rationale)
    parameters = _require_mapping(recipe, "parameters")
    if recipe_name == "ridge":
        _require_exact_keys(parameters, {"alpha"}, "selected_model_recipe.parameters")
        alpha = _finite_number("selected_model_recipe.parameters.alpha", parameters["alpha"])
        if alpha not in {0.1, 1.0, 10.0}:
            raise PromotionManifestError("selected_model_recipe.parameters.alpha outside allowed set")
        canonical_params = {"alpha": alpha}
    elif recipe_name == "hist_gradient_boosting":
        _require_exact_keys(
            parameters,
            {"max_iter", "learning_rate", "max_leaf_nodes"},
            "selected_model_recipe.parameters",
        )
        max_iter = _strict_int("selected_model_recipe.parameters.max_iter", parameters["max_iter"])
        learning_rate = _finite_number(
            "selected_model_recipe.parameters.learning_rate", parameters["learning_rate"]
        )
        max_leaf_nodes = _strict_int(
            "selected_model_recipe.parameters.max_leaf_nodes", parameters["max_leaf_nodes"]
        )
        if max_iter not in {50, 100, 200}:
            raise PromotionManifestError("selected_model_recipe.parameters.max_iter outside allowed set")
        if learning_rate not in {0.03, 0.1}:
            raise PromotionManifestError("selected_model_recipe.parameters.learning_rate outside allowed set")
        if max_leaf_nodes not in {15, 31, 63}:
            raise PromotionManifestError("selected_model_recipe.parameters.max_leaf_nodes outside allowed set")
        canonical_params = {
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
        }
    else:
        raise PromotionManifestError("selected_model_recipe.recipe unsupported")
    return {
        "name": name,
        "parameters": canonical_params,
        "rationale": rationale,
        "recipe": recipe_name,
    }


def _require_exact_keys(mapping: Mapping[str, Any], keys: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != keys:
        raise PromotionManifestError(
            f"{label} keys must be exactly {sorted(keys)}, got {sorted(actual)}"
        )


def _require_list(mapping: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = _require_key(mapping, key)
    if not isinstance(value, list) or not value:
        raise PromotionManifestError(f"{label} must be a non-empty list")
    return value


def _strict_int(label: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromotionManifestError(f"{label} must be an integer")
    return int(value)


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if "path" in lowered or "customer" in lowered:
                return True
            if _contains_sensitive_key(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def _safe_nonblank_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = _require_nonblank_string(mapping, key, label)
    _reject_unsafe_string(label, value)
    return value


def _reject_unsafe_string(label: str, value: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in ("/users/", "\\users\\", "customer")):
        raise PromotionManifestError(f"{label} contains customer path-like content")
    if any(token in lowered for token in ("import ", "exec", "eval(", "subprocess", "os.", "```")):
        raise PromotionManifestError(f"{label} contains executable code-like content")


def _sha_json(value: Any) -> str:
    import hashlib

    canonical = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _winner_name_for_specs(specs: tuple[FeatureSpec, ...]) -> str:
    return "+".join(sorted(spec.name for spec in specs))


def _raw_improvement_ratio(baseline_nmae: float, winner_nmae: float) -> float:
    if baseline_nmae <= 0:
        return 0.0 if winner_nmae <= baseline_nmae else -1.0
    return (baseline_nmae - winner_nmae) / baseline_nmae


def _round_float(value: float) -> float:
    return float(round(float(value), 12))


def _require_exact(manifest: Mapping[str, Any], key: str, expected: str) -> None:
    actual = _require_key(manifest, key)
    if type(actual) is not str or actual != expected:
        raise PromotionManifestError(f"{key} must be exactly {expected!r}")


def _require_key(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise PromotionManifestError(f"missing {key}")
    return mapping[key]


def _require_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _require_key(mapping, key)
    if not isinstance(value, Mapping):
        raise PromotionManifestError(f"{key} must be a mapping")
    return value


def _require_aidm_seed(manifest: Mapping[str, Any]) -> None:
    value = _require_key(manifest, "seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromotionManifestError("seed must be an integer")
    if value < 0:
        raise PromotionManifestError("seed must be >= 0")


def _require_nonblank_string(
    mapping: Mapping[str, Any], key: str, label: str
) -> str:
    if key not in mapping:
        raise PromotionManifestError(f"missing {label}")
    value = mapping[key]
    if type(value) is not str or not value.strip():
        raise PromotionManifestError(f"{label} must be a non-blank string")
    return value


def _require_nonempty_string(
    mapping: Mapping[str, Any], key: str, label: str
) -> None:
    value = _require_key(mapping, key)
    if type(value) is not str or not value:
        raise PromotionManifestError(f"{label} must be a non-empty string")


def _require_exact_str_value(label: str, value: Any) -> None:
    if type(value) is not str:
        raise PromotionManifestError(f"{label} must be a string")


def _finite_float(label: str, value: Any) -> float:
    if isinstance(value, bool):
        raise PromotionManifestError(f"{label} must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise PromotionManifestError(f"{label} must be finite") from exc
    except (TypeError, ValueError) as exc:
        raise PromotionManifestError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise PromotionManifestError(f"{label} must be finite")
    return number


def _finite_number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionManifestError(f"{label} must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise PromotionManifestError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise PromotionManifestError(f"{label} must be finite")
    return number


def _canonical_spec_dict(spec: FeatureSpec) -> dict[str, Any]:
    return _canonical_value(spec.to_dict())


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        canonical = {}
        for key in sorted(value):
            if type(key) is not str:
                raise PromotionManifestError("feature spec keys must be strings")
            canonical[key] = _canonical_value(value[key])
        return canonical
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if type(value) is str or value is None or type(value) is bool:
        return value
    if type(value) is int:
        return int(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise PromotionManifestError("feature specs must contain finite floats")
        return float(value)
    raise PromotionManifestError(
        f"feature specs must contain primitive literal values, got {type(value).__name__}"
    )


def _render_module(
    *, specs: tuple[FeatureSpec, ...], schema_version: str, winner_name: str
) -> str:
    lines = [
        "# This file is generated by power_forecasting.aidd; do not edit.",
        f"# Manifest schema_version: {schema_version!r}",
        f"# Manifest winner: {winner_name!r}",
        "from __future__ import annotations",
        "",
        "import pandas as pd",
        "",
        "from power_forecasting.features import FeatureSpec, apply_feature_specs",
        "",
        "",
        f"MANIFEST_SCHEMA_VERSION = {schema_version!r}",
        f"MANIFEST_WINNER_NAME = {winner_name!r}",
        "PROMOTED_FEATURE_SPECS = (",
    ]
    for spec in specs:
        lines.append("    FeatureSpec.from_dict(")
        lines.extend(_render_literal_lines(_canonical_spec_dict(spec), indent=8))
        lines.append("    ),")
    lines.extend(
        [
            ")",
            "",
            "",
            "def build_promoted_features(frame: pd.DataFrame) -> pd.DataFrame:",
            "    return apply_feature_specs(frame, list(PROMOTED_FEATURE_SPECS))",
            "",
        ]
    )
    return "\n".join(lines)


def _render_literal_lines(value: Any, *, indent: int) -> list[str]:
    rendered = _render_literal(value, indent=indent)
    return rendered.splitlines()


def _render_literal(value: Any, *, indent: int) -> str:
    prefix = " " * indent
    nested_prefix = " " * (indent + 4)
    if isinstance(value, dict):
        if not value:
            return f"{prefix}{{}}"
        lines = [f"{prefix}{{"]
        for key in sorted(value):
            rendered_value = _render_literal(value[key], indent=indent + 4).lstrip()
            lines.append(f"{nested_prefix}{key!r}: {rendered_value},")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return f"{prefix}[]"
        lines = [f"{prefix}["]
        for item in value:
            rendered_item = _render_literal(item, indent=indent + 4).lstrip()
            lines.append(f"{nested_prefix}{rendered_item},")
        lines.append(f"{prefix}]")
        return "\n".join(lines)
    return f"{prefix}{value!r}"


def _atomic_write_text(target: Path, content: str) -> None:
    temp_path: Path | None = None
    created = False
    try:
        for attempt in range(100):
            candidate = target.with_name(f".{target.name}.{os.getpid()}.{attempt}.tmp")
            try:
                with candidate.open("x", encoding="utf-8", newline="\n") as handle:
                    temp_path = candidate
                    created = True
                    handle.write(content)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"could not create temp file for {target}")

        os.replace(temp_path, target)
        created = False
    finally:
        if created and temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "PromotionManifestError",
    "render_model_recipe_patch",
    "render_promoted_module",
    "validate_prediction_time_feature_spec",
    "validate_promotion_manifest",
]
