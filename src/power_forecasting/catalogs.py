from __future__ import annotations

import hashlib
import json
import keyword
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from power_forecasting.aidd import validate_prediction_time_feature_spec
from power_forecasting.features import FeatureSpec
from power_forecasting.proposals import FeatureSet, RECIPE_PARAMETER_VALUES


_SCHEMA_VERSION = "1"
_TOP_LEVEL_KEYS = {"schema_version", "profiles", "feature_sets", "recipes", "searches"}
_FEATURE_SPEC_KEYS = {"name", "transform", "inputs", "parameters", "version", "rationale"}
_RECIPE_KEYS = {"recipe", "parameters", "allowed_parameters", "rationale"}
_SEARCH_KEYS = {"sampler", "seed", "n_trials", "spaces"}
_SUPPORTED_RECIPE_PARAMETERS = {
    "ridge": {"alpha": "number"},
    "hist_gradient_boosting": {
        "max_iter": "integer",
        "learning_rate": "number",
        "max_leaf_nodes": "integer",
    },
    "random_forest": {
        "n_estimators": "integer",
        "max_depth": "integer_or_none",
        "min_samples_leaf": "integer",
    },
    "xgboost": {
        "n_estimators": "integer",
        "max_depth": "integer",
        "learning_rate": "number",
        "subsample": "number",
    },
    "lightgbm": {
        "n_estimators": "integer",
        "learning_rate": "number",
        "num_leaves": "integer",
        "min_child_samples": "integer",
    },
}


class OptimizationCatalogError(ValueError):
    """Raised when an optimization catalog is not a trusted bounded policy."""


@dataclass(frozen=True)
class CatalogRecipe:
    name: str
    recipe: str
    parameters: Mapping[str, int | float | None]
    allowed_parameters: Mapping[str, tuple[int | float | None, ...]]
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(
            self,
            "allowed_parameters",
            MappingProxyType(
                {key: tuple(values) for key, values in self.allowed_parameters.items()}
            ),
        )


@dataclass(frozen=True)
class CatalogProfile:
    name: str
    rationale: str
    feature_set_names: tuple[str, ...]
    direct_recipe_names: tuple[str, ...]
    search_name: str | None = None


@dataclass(frozen=True)
class OptimizationCatalog:
    source_path: Path
    sha256: str
    feature_sets: Mapping[str, FeatureSet]
    direct_recipes: Mapping[str, CatalogRecipe]
    searches: Mapping[str, Mapping[str, Any]]
    profiles: Mapping[str, CatalogProfile]
    profile_names: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_sets", MappingProxyType(dict(self.feature_sets)))
        object.__setattr__(
            self, "direct_recipes", MappingProxyType(dict(self.direct_recipes))
        )
        object.__setattr__(self, "searches", MappingProxyType(dict(self.searches)))
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))
        object.__setattr__(self, "profile_names", tuple(self.profile_names))

    def profile(self, name: str) -> CatalogProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            raise OptimizationCatalogError(f"unknown profile: {name}") from exc


def load_optimization_catalog(path: Path, *, repository_root: Path) -> OptimizationCatalog:
    source_path, catalog_bytes = _read_catalog_bytes(path, repository_root)
    payload = _load_json(catalog_bytes)
    _exact_keys(payload, _TOP_LEVEL_KEYS, "catalog")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise OptimizationCatalogError("schema_version must be exactly '1'")

    feature_sets = _parse_feature_sets(payload["feature_sets"])
    direct_recipes = _parse_recipes(payload["recipes"])
    searches = _parse_searches(payload["searches"], direct_recipes)
    profiles = _parse_profiles(payload["profiles"], feature_sets, direct_recipes, searches)
    return OptimizationCatalog(
        source_path=source_path,
        sha256=hashlib.sha256(catalog_bytes).hexdigest(),
        feature_sets=feature_sets,
        direct_recipes=direct_recipes,
        searches=searches,
        profiles=profiles,
        profile_names=tuple(profiles),
    )


def _catalog_path(
    path: Path, repository_root: Path
) -> tuple[Path, Path, tuple[int, int]]:
    root = Path(repository_root)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise OptimizationCatalogError("repository_root must be an existing directory") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise OptimizationCatalogError("repository_root must be an existing directory")
    try:
        resolved_root = root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise OptimizationCatalogError("repository_root must be an existing directory") from exc

    supplied_path = Path(path)
    if supplied_path.is_absolute():
        try:
            relative_path = supplied_path.relative_to(resolved_root)
        except ValueError as exc:
            raise OptimizationCatalogError("catalog must be inside repository root") from exc
    else:
        relative_path = supplied_path
    if relative_path.suffix != ".json":
        raise OptimizationCatalogError("catalog must be a JSON file")
    candidate = resolved_root
    for component in relative_path.parts:
        if component in {"", "."}:
            continue
        if component == "..":
            raise OptimizationCatalogError("catalog must be inside repository root")
        candidate /= component
        if candidate.is_symlink():
            raise OptimizationCatalogError(
                "catalog must be an existing regular non-symlink file; "
                "catalog path must not contain symlinks"
            )
    return candidate, resolved_root, (root_stat.st_dev, root_stat.st_ino)


def _read_catalog_bytes(path: Path, repository_root: Path) -> tuple[Path, bytes]:
    source_path, resolved_root, root_identity = _catalog_path(path, repository_root)
    relative_path = source_path.relative_to(resolved_root)
    components = tuple(component for component in relative_path.parts if component != ".")
    if not components or not hasattr(os, "O_NOFOLLOW"):
        raise OptimizationCatalogError("catalog must be an existing regular non-symlink file")

    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_DIRECTORY", 0)
        )
        directory_fd = os.open(
            str(repository_root), directory_flags
        )
        opened_root_stat = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened_root_stat.st_mode)
            or (opened_root_stat.st_dev, opened_root_stat.st_ino) != root_identity
        ):
            raise OptimizationCatalogError("repository_root must be an existing directory")

        for component in components[:-1]:
            child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
            if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
                raise OptimizationCatalogError(
                    "catalog must be an existing regular non-symlink file; "
                    "catalog path must not contain symlinks"
                )

        file_fd = os.open(
            components[-1],
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=directory_fd,
        )
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise OptimizationCatalogError(
                "catalog must be an existing regular non-symlink file"
            )
        with os.fdopen(file_fd, "rb", closefd=True) as handle:
            file_fd = None
            return source_path, handle.read()
    except OptimizationCatalogError:
        raise
    except (OSError, ValueError) as exc:
        raise OptimizationCatalogError(
            "catalog must be an existing regular non-symlink file; "
            "catalog path must not contain symlinks"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _load_json(catalog_bytes: bytes) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise OptimizationCatalogError(f"catalog contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            catalog_bytes.decode("utf-8"), object_pairs_hook=reject_duplicates
        )
    except OptimizationCatalogError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise OptimizationCatalogError("catalog must contain valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise OptimizationCatalogError("catalog must be a mapping")
    return payload


def _parse_feature_sets(raw: Any) -> dict[str, FeatureSet]:
    entries = _named_mapping(raw, "feature_sets")
    parsed: dict[str, FeatureSet] = {}
    for name, value in entries.items():
        _identifier(name, "feature set name")
        if not isinstance(value, Mapping):
            raise OptimizationCatalogError(f"feature set {name} must be a mapping")
        _exact_keys(value, {"rationale", "specs"}, f"feature set {name}")
        specs_raw = value["specs"]
        if not isinstance(specs_raw, list) or not specs_raw:
            raise OptimizationCatalogError(f"feature set {name}: specs must be nonempty")
        specs = []
        for spec_raw in specs_raw:
            if not isinstance(spec_raw, Mapping):
                raise OptimizationCatalogError("feature specs must be mappings")
            _exact_keys(spec_raw, _FEATURE_SPEC_KEYS, "feature spec")
            try:
                spec = FeatureSpec.from_dict(spec_raw)
                validate_prediction_time_feature_spec(spec, error_type=OptimizationCatalogError)
            except OptimizationCatalogError:
                raise
            except (TypeError, ValueError, OverflowError) as exc:
                raise OptimizationCatalogError(str(exc) or "invalid feature spec") from exc
            specs.append(spec)
        try:
            parsed[name] = FeatureSet(name, value["rationale"], tuple(specs))
        except (TypeError, ValueError) as exc:
            raise OptimizationCatalogError(str(exc) or "invalid feature set") from exc
    return parsed


def _parse_recipes(raw: Any) -> dict[str, CatalogRecipe]:
    entries = _named_mapping(raw, "recipes")
    parsed: dict[str, CatalogRecipe] = {}
    for name, value in entries.items():
        _identifier(name, "recipe name")
        if not isinstance(value, Mapping):
            raise OptimizationCatalogError(f"recipe {name} must be a mapping")
        _exact_keys(value, _RECIPE_KEYS, f"recipe {name}")
        recipe = value["recipe"]
        if type(recipe) is not str or recipe not in _SUPPORTED_RECIPE_PARAMETERS:
            raise OptimizationCatalogError(f"recipe {name}: unsupported recipe {recipe}")
        rationale = _nonblank(value["rationale"], f"recipe {name}: rationale")
        parameters = _parse_parameter_mapping(
            value["parameters"], name, recipe, "parameters"
        )
        allowed_parameters = _parse_allowed_parameter_mapping(
            value["allowed_parameters"], name, recipe
        )
        for parameter, actual in parameters.items():
            if actual not in allowed_parameters[parameter]:
                raise OptimizationCatalogError(
                    f"recipe {name}: {parameter} is outside allowed values"
                )
            if actual not in RECIPE_PARAMETER_VALUES[recipe][parameter]:
                raise OptimizationCatalogError(
                    f"recipe {name}: {parameter} is outside supported values"
                )
        parsed[name] = CatalogRecipe(
            name=name,
            recipe=recipe,
            parameters=parameters,
            allowed_parameters=allowed_parameters,
            rationale=rationale,
        )
    return parsed


def _parse_parameter_mapping(
    raw: Any, name: str, recipe: str, label: str
) -> dict[str, int | float | None]:
    if not isinstance(raw, Mapping):
        raise OptimizationCatalogError(f"recipe {name}: {label} must be a mapping")
    expected = set(_SUPPORTED_RECIPE_PARAMETERS[recipe])
    actual = set(raw)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        raise OptimizationCatalogError(f"recipe {name}: unknown parameters {extra}")
    if missing:
        raise OptimizationCatalogError(f"recipe {name}: missing parameters {missing}")
    return {
        key: _parameter_value(raw[key], kind, f"recipe {name}: {key}")
        for key, kind in _SUPPORTED_RECIPE_PARAMETERS[recipe].items()
    }


def _parse_allowed_parameter_mapping(
    raw: Any, name: str, recipe: str
) -> dict[str, tuple[int | float | None, ...]]:
    if not isinstance(raw, Mapping):
        raise OptimizationCatalogError(
            f"recipe {name}: allowed_parameters must be a mapping"
        )
    expected = set(_SUPPORTED_RECIPE_PARAMETERS[recipe])
    _exact_keys(raw, expected, f"recipe {name}: allowed_parameters")
    normalized: dict[str, tuple[int | float | None, ...]] = {}
    for key, kind in _SUPPORTED_RECIPE_PARAMETERS[recipe].items():
        values = raw[key]
        if not isinstance(values, list) or not values:
            raise OptimizationCatalogError(
                f"recipe {name}: allowed_parameters.{key} must be a nonempty list"
            )
        parsed = tuple(
            _parameter_value(value, kind, f"recipe {name}: allowed_parameters.{key}")
            for value in values
        )
        if len(set(parsed)) != len(parsed):
            raise OptimizationCatalogError(
                f"recipe {name}: allowed_parameters.{key} contains duplicate values"
            )
        if not set(parsed) <= RECIPE_PARAMETER_VALUES[recipe][key]:
            raise OptimizationCatalogError(
                f"recipe {name}: allowed_parameters.{key} contains values outside supported values"
            )
        normalized[key] = parsed
    return normalized


def _parse_searches(
    raw: Any, recipes: Mapping[str, CatalogRecipe]
) -> dict[str, Mapping[str, Any]]:
    entries = _named_mapping(raw, "searches")
    lightgbm_allowed = _lightgbm_allowed_values(recipes)
    parsed: dict[str, Mapping[str, Any]] = {}
    for name, value in entries.items():
        _identifier(name, "search name")
        if not isinstance(value, Mapping):
            raise OptimizationCatalogError(f"search {name} must be a mapping")
        _exact_keys(value, _SEARCH_KEYS, f"search {name}")
        if value["sampler"] != "tpe":
            raise OptimizationCatalogError(f"search {name}: sampler must be exactly 'tpe'")
        seed = _integer(value["seed"], f"search {name}: seed")
        n_trials = _integer(value["n_trials"], f"search {name}: n_trials")
        if seed < 0 or not 1 <= n_trials <= 50:
            raise OptimizationCatalogError(f"search {name}: invalid bounded TPE settings")
        spaces = value["spaces"]
        if not isinstance(spaces, Mapping):
            raise OptimizationCatalogError(f"search {name}: spaces must be a mapping")
        _exact_keys(spaces, {"lightgbm"}, f"search {name}: spaces")
        lightgbm = spaces["lightgbm"]
        if not isinstance(lightgbm, Mapping):
            raise OptimizationCatalogError(f"search {name}: lightgbm space must be a mapping")
        _exact_keys(lightgbm, set(_SUPPORTED_RECIPE_PARAMETERS["lightgbm"]), f"search {name}: lightgbm space")
        normalized_space: dict[str, tuple[int | float, ...]] = {}
        for parameter, kind in _SUPPORTED_RECIPE_PARAMETERS["lightgbm"].items():
            values = lightgbm[parameter]
            if not isinstance(values, list) or not values:
                raise OptimizationCatalogError(
                    f"search {name}: {parameter} must be a nonempty list"
                )
            normalized = tuple(
                _parameter_value(value, kind, f"search {name}: {parameter}")
                for value in values
            )
            if len(set(normalized)) != len(normalized):
                raise OptimizationCatalogError(
                    f"search {name}: {parameter} contains duplicate values"
                )
            if not set(normalized) <= set(lightgbm_allowed[parameter]):
                raise OptimizationCatalogError(
                    f"search {name}: {parameter} contains values outside allowed values"
                )
            if not set(normalized) <= RECIPE_PARAMETER_VALUES["lightgbm"][parameter]:
                raise OptimizationCatalogError(
                    f"search {name}: {parameter} contains values outside supported values"
                )
            normalized_space[parameter] = normalized
        parsed[name] = MappingProxyType(
            {
                "sampler": "tpe",
                "seed": seed,
                "n_trials": n_trials,
                "spaces": MappingProxyType({"lightgbm": MappingProxyType(normalized_space)}),
            }
        )
    return parsed


def _lightgbm_allowed_values(
    recipes: Mapping[str, CatalogRecipe]
) -> Mapping[str, tuple[int | float | None, ...]]:
    matching = [recipe for recipe in recipes.values() if recipe.recipe == "lightgbm"]
    if not matching:
        raise OptimizationCatalogError("searches require a LightGBM recipe")
    first = matching[0].allowed_parameters
    if any(recipe.allowed_parameters != first for recipe in matching[1:]):
        raise OptimizationCatalogError("LightGBM recipes must share allowed_parameters")
    return first


def _parse_profiles(
    raw: Any,
    feature_sets: Mapping[str, FeatureSet],
    recipes: Mapping[str, CatalogRecipe],
    searches: Mapping[str, Mapping[str, Any]],
) -> dict[str, CatalogProfile]:
    entries = _named_mapping(raw, "profiles")
    parsed: dict[str, CatalogProfile] = {}
    for name, value in entries.items():
        _identifier(name, "profile name")
        if not isinstance(value, Mapping):
            raise OptimizationCatalogError(f"profile {name} must be a mapping")
        allowed = {"rationale", "feature_sets", "direct_recipes", "search"}
        required = allowed - {"search"}
        _exact_keys(value, allowed, f"profile {name}", required=required)
        feature_names = _references(
            value["feature_sets"], "feature set", feature_sets, name
        )
        recipe_names = _references(
            value["direct_recipes"], "direct recipe", recipes, name
        )
        search_name = value.get("search")
        if search_name is not None:
            if type(search_name) is not str or search_name not in searches:
                raise OptimizationCatalogError(f"profile {name}: unknown search")
        parsed[name] = CatalogProfile(
            name=name,
            rationale=_nonblank(value["rationale"], f"profile {name}: rationale"),
            feature_set_names=feature_names,
            direct_recipe_names=recipe_names,
            search_name=search_name,
        )
    return parsed


def _references(
    raw: Any, kind: str, available: Mapping[str, Any], profile_name: str
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise OptimizationCatalogError(
            f"profile {profile_name}: {kind} references must be nonempty"
        )
    if any(type(name) is not str for name in raw) or len(set(raw)) != len(raw):
        raise OptimizationCatalogError(f"profile {profile_name}: invalid {kind} references")
    missing = [name for name in raw if name not in available]
    if missing:
        raise OptimizationCatalogError(f"profile {profile_name}: unknown {kind}")
    return tuple(raw)


def _named_mapping(raw: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise OptimizationCatalogError(f"{label} must be nonempty")
    return raw


def _exact_keys(
    raw: Mapping[str, Any], expected: set[str], label: str, *, required: set[str] | None = None
) -> None:
    required = expected if required is None else required
    actual = set(raw)
    extra = sorted(actual - expected)
    missing = sorted(required - actual)
    if extra:
        raise OptimizationCatalogError(f"{label} unknown keys: {extra}")
    if missing:
        raise OptimizationCatalogError(f"{label} missing keys: {missing}")


def _parameter_value(value: Any, kind: str, label: str) -> int | float | None:
    if kind == "integer":
        return _integer(value, label)
    if kind == "integer_or_none":
        return None if value is None else _integer(value, label)
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise OptimizationCatalogError(f"{label} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise OptimizationCatalogError(f"{label} must be finite")
        return number
    raise AssertionError(f"unsupported parameter kind: {kind}")


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OptimizationCatalogError(f"{label} must be an integer")
    return value


def _identifier(value: Any, label: str) -> str:
    if type(value) is not str or not value or not value.isidentifier() or keyword.iskeyword(value):
        raise OptimizationCatalogError(f"{label} must be an identifier string")
    return value


def _nonblank(value: Any, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise OptimizationCatalogError(f"{label} must be a nonblank string")
    return value


__all__ = [
    "CatalogProfile",
    "CatalogRecipe",
    "OptimizationCatalog",
    "OptimizationCatalogError",
    "load_optimization_catalog",
]
