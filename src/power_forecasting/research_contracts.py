from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from power_forecasting.aidm import AIDMConfig
from power_forecasting.catalogs import (
    OptimizationCatalog,
    OptimizationCatalogError,
    load_optimization_catalog,
)


SCHEMA_VERSION = "1"
_DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "optimization-catalog.v1.json"
)
SUPPORTED_PROFILES = frozenset(
    load_optimization_catalog(
        _DEFAULT_CATALOG_PATH,
        repository_root=_DEFAULT_CATALOG_PATH.parents[1],
    ).profile_names
)

_REQUIRED_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "dataset_path",
        "legacy_manifest_path",
        "catalog_path",
        "run_dir",
        "profiles",
        "max_iterations",
        "fold_count",
        "objective",
        "minimum_improvement",
        "max_plant_regression",
    }
)
_OPTIONAL_CONFIG_KEYS = frozenset({"agent_proposals"})
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ResearchContractError(ValueError):
    """Raised when a research-loop configuration violates its contract."""


@dataclass(frozen=True)
class ResearchLoopConfig:
    schema_version: str
    run_id: str
    dataset_path: str
    legacy_manifest_path: str
    catalog_path: str
    catalog_sha256: str
    catalog: OptimizationCatalog
    run_dir: str
    profiles: tuple[str, ...]
    max_iterations: int
    fold_count: int
    objective: str
    minimum_improvement: float
    max_plant_regression: float
    agent_proposals: bool = False


def load_research_loop_config(
    value: Mapping[str, object],
    *,
    config_path: Path,
    repository_root: Path,
) -> ResearchLoopConfig:
    """Validate a research-loop configuration and resolve its filesystem paths."""

    if not isinstance(value, Mapping):
        raise ResearchContractError("research-loop config must be a mapping")
    _exact_keys(value, _REQUIRED_CONFIG_KEYS, "research-loop config", _OPTIONAL_CONFIG_KEYS)

    config_file = _path_argument(config_path, "config_path")
    root = _repository_root(repository_root)
    if value["schema_version"] != SCHEMA_VERSION:
        raise ResearchContractError("schema_version must be exactly '1'")

    run_id = validate_run_id(value["run_id"])
    config_dir = config_file.parent
    dataset_path = _existing_input_path(value["dataset_path"], config_dir, "dataset_path")
    legacy_manifest_path = _existing_input_path(
        value["legacy_manifest_path"], config_dir, "legacy_manifest_path"
    )
    catalog_path = _path_candidate(value["catalog_path"], config_dir, "catalog_path")
    try:
        catalog = load_optimization_catalog(catalog_path, repository_root=root)
    except OptimizationCatalogError as exc:
        raise ResearchContractError(f"catalog_path must reference an approved catalog: {exc}") from exc
    run_dir = _path_candidate(value["run_dir"], config_dir, "run_dir")
    _validate_run_dir(run_dir, root)
    run_dir = run_dir.resolve()

    profiles = _profiles(value["profiles"], catalog)
    max_iterations = _bounded_integer(value["max_iterations"], "max_iterations")
    fold_count = _bounded_integer(value["fold_count"], "fold_count")
    objective = _nonblank_string(value["objective"], "objective")
    minimum_improvement, max_plant_regression = _aidm_thresholds(
        fold_count,
        value["minimum_improvement"],
        value["max_plant_regression"],
    )
    agent_proposals = value.get("agent_proposals", False)
    if type(agent_proposals) is not bool:
        raise ResearchContractError("agent_proposals must be a boolean")

    return ResearchLoopConfig(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        dataset_path=str(dataset_path),
        legacy_manifest_path=str(legacy_manifest_path),
        catalog_path=str(catalog.source_path),
        catalog_sha256=catalog.sha256,
        catalog=catalog,
        run_dir=str(run_dir),
        profiles=profiles,
        max_iterations=max_iterations,
        fold_count=fold_count,
        objective=objective,
        minimum_improvement=minimum_improvement,
        max_plant_regression=max_plant_regression,
        agent_proposals=agent_proposals,
    )


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    label: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    actual = set(value)
    if expected <= actual <= expected | optional:
        return
    unknown = sorted(actual - expected - optional)
    missing = sorted(expected - actual)
    if unknown:
        raise ResearchContractError(f"{label} unknown keys: {unknown}")
    raise ResearchContractError(f"{label} missing keys: {missing}")


def _path_argument(value: Path, label: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{label} must be a Path")
    return value.resolve()


def _repository_root(value: Path) -> Path:
    root = _path_argument(value, "repository_root")
    if not root.is_dir():
        raise ResearchContractError("repository_root must be an existing directory")
    return root


def validate_run_id(value: object) -> str:
    if type(value) is not str or not _RUN_ID_PATTERN.fullmatch(value) or ".." in value:
        raise ResearchContractError("run_id must be a safe identifier")
    return value


def _nonblank_string(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ResearchContractError(f"{label} must be a nonblank string")
    return value


def _resolved_path(value: object, base: Path, label: str) -> Path:
    return _path_candidate(value, base, label).resolve()


def _path_candidate(value: object, base: Path, label: str) -> Path:
    raw_path = _nonblank_string(value, label)
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _existing_input_path(value: object, base: Path, label: str) -> Path:
    path = _resolved_path(value, base, label)
    if not path.exists():
        raise ResearchContractError(f"{label} must explicitly exist: {path}")
    if not path.is_file():
        raise ResearchContractError(f"{label} must be a file: {path}")
    return path


def _profiles(value: object, catalog: OptimizationCatalog) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ResearchContractError("profiles must be a list")
    if not value:
        raise ResearchContractError("profiles must be nonempty")

    profiles: list[str] = []
    for profile in value:
        if type(profile) is not str:
            raise ResearchContractError("profiles entries must be strings")
        if profile not in catalog.profiles:
            raise ResearchContractError(f"profiles contains unknown catalog profile: {profile}")
        if profile in profiles:
            raise ResearchContractError(f"profiles contains duplicate profile: {profile}")
        profiles.append(profile)
    return tuple(profiles)


def _bounded_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResearchContractError(f"{label} must be an integer")
    if not 1 <= value <= 10:
        raise ResearchContractError(f"{label} must be between 1 and 10")
    return value


def _aidm_thresholds(
    fold_count: int,
    minimum_improvement: object,
    max_plant_regression: object,
) -> tuple[float, float]:
    minimum = _aidm_threshold_value(minimum_improvement, "minimum_improvement")
    regression = _aidm_threshold_value(max_plant_regression, "max_plant_regression")
    aidm_config = AIDMConfig(
        folds=fold_count,
        minimum_improvement=minimum,
        max_plant_regression=regression,
    )

    thresholds = (float(aidm_config.minimum_improvement), float(aidm_config.max_plant_regression))
    if not all(math.isfinite(value) for value in thresholds):
        raise ResearchContractError("research-loop thresholds must be finite")
    return thresholds


def _aidm_threshold_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResearchContractError(f"{label} must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ResearchContractError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ResearchContractError(f"{label} must be finite")
    if not 0.0 <= number <= 1.0:
        raise ResearchContractError(f"{label} must be between 0 and 1")
    return number


def _validate_run_dir(run_dir: Path, repository_root: Path) -> None:
    allowed_destinations = (repository_root / "runs", repository_root / "outputs")
    for path in (run_dir, *allowed_destinations):
        symlink = _first_symlink_component(path)
        if symlink is not None:
            raise ResearchContractError(f"run_dir contains symlinked path component: {symlink}")

    canonical_run_dir = run_dir.resolve()
    canonical_destinations = tuple(destination.resolve() for destination in allowed_destinations)
    if any(_is_within(canonical_run_dir, destination) for destination in canonical_destinations):
        return
    raise ResearchContractError(
        "run_dir must be canonically contained in repository runs or outputs"
    )


def _first_symlink_component(path: Path) -> Path | None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            return current
    return None


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "ResearchContractError",
    "ResearchLoopConfig",
    "SCHEMA_VERSION",
    "SUPPORTED_PROFILES",
    "load_research_loop_config",
    "validate_run_id",
]
