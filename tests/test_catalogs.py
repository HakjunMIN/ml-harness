from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from power_forecasting.catalogs import OptimizationCatalogError, load_optimization_catalog


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPOSITORY_ROOT / "configs" / "optimization-catalog.v1.json"


@pytest.fixture
def catalog_root(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    return root


@pytest.fixture
def write_catalog(catalog_root: Path):
    def write(name: str, payload: object) -> Path:
        path = catalog_root / name
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return write


def _payload() -> dict[str, object]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_loads_default_catalog_with_hash_and_immutable_profile_lookup():
    catalog = load_optimization_catalog(CATALOG_PATH, repository_root=REPOSITORY_ROOT)

    assert catalog.source_path == CATALOG_PATH.resolve()
    assert catalog.sha256 == hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()
    assert catalog.profile_names == ("safe_weather", "history_tree", "bounded_search")
    assert catalog.profile("safe_weather") is catalog.profiles["safe_weather"]
    assert catalog.profile("bounded_search").feature_set_names == (
        "safe_weather",
        "history_tree",
    )
    assert catalog.profile("bounded_search").direct_recipe_names == (
        "forest_search",
        "xgb_search",
        "lgbm_search",
    )
    assert catalog.profile("bounded_search").search_name == "bounded_lightgbm_tpe"
    assert catalog.direct_recipes["ridge_weather"].parameters == {"alpha": 1.0}
    assert catalog.direct_recipes["ridge_weather"].allowed_parameters["alpha"] == (
        0.1,
        1.0,
        10.0,
    )
    assert catalog.searches["bounded_lightgbm_tpe"]["spaces"]["lightgbm"]["num_leaves"] == (
        15,
        31,
    )
    with pytest.raises(TypeError):
        catalog.profiles["other"] = catalog.profile("safe_weather")
    with pytest.raises(OptimizationCatalogError, match="unknown profile"):
        catalog.profile("other")


def test_catalog_policy_and_hash_are_derived_from_a_single_safe_read(monkeypatch):
    expected_bytes = CATALOG_PATH.read_bytes()
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda _path: pytest.fail("catalog must not be read again for its hash"),
    )

    catalog = load_optimization_catalog(CATALOG_PATH, repository_root=REPOSITORY_ROOT)

    assert catalog.sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert catalog.profile_names == ("safe_weather", "history_tree", "bounded_search")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: {**p, "extra": True}, "unknown keys"),
        (lambda p: {**p, "profiles": {}}, "profiles must be nonempty"),
        (
            lambda p: {
                **p,
                "profiles": {
                    **p["profiles"],
                    "safe_weather": {"rationale": "missing references"},
                },
            },
            "profile safe_weather missing keys",
        ),
        (
            lambda p: {
                **p,
                "profiles": {
                    **p["profiles"],
                    "safe_weather": {
                        **p["profiles"]["safe_weather"],
                        "feature_sets": ["not_present"],
                    },
                },
            },
            "unknown feature set",
        ),
        (
            lambda p: {
                **p,
                "profiles": {
                    **p["profiles"],
                    "bounded_search": {
                        **p["profiles"]["bounded_search"],
                        "search": "not_present",
                    },
                },
            },
            "unknown search",
        ),
    ],
)
def test_rejects_unknown_or_malformed_profile_content(
    write_catalog, catalog_root, mutate, message
):
    path = write_catalog("invalid-profile.json", mutate(_payload()))

    with pytest.raises(OptimizationCatalogError, match=message):
        load_optimization_catalog(path, repository_root=catalog_root)


def test_rejects_duplicate_json_named_entities(write_catalog, catalog_root):
    payload = CATALOG_PATH.read_text(encoding="utf-8")
    duplicate = payload.replace(
        '"safe_weather": {',
        '"safe_weather": {',
        1,
    ).replace(
        '"history_tree": {',
        '"safe_weather": {',
        1,
    )
    path = write_catalog("duplicate.json", duplicate)

    with pytest.raises(OptimizationCatalogError, match="duplicate key"):
        load_optimization_catalog(path, repository_root=catalog_root)


def test_rejects_malformed_json_and_non_json_catalog_files(write_catalog, catalog_root):
    malformed = write_catalog("malformed.json", "{")
    non_json = write_catalog("catalog.txt", _payload())

    with pytest.raises(OptimizationCatalogError, match="valid JSON"):
        load_optimization_catalog(malformed, repository_root=catalog_root)
    with pytest.raises(OptimizationCatalogError, match="JSON file"):
        load_optimization_catalog(non_json, repository_root=catalog_root)


def test_rejects_catalog_paths_outside_repository_and_symlinks(catalog_root):
    outside = catalog_root.parent / "outside-catalog.json"
    outside.write_text(json.dumps(_payload()), encoding="utf-8")
    link = catalog_root / "catalog-link.json"
    link.symlink_to(CATALOG_PATH)
    with pytest.raises(OptimizationCatalogError, match="inside repository root"):
        load_optimization_catalog(outside, repository_root=catalog_root)
    with pytest.raises(OptimizationCatalogError, match="regular non-symlink"):
        load_optimization_catalog(link, repository_root=catalog_root)


def test_rejects_catalog_path_with_symlinked_parent_directory(catalog_root):
    catalog_directory = catalog_root / "catalog-directory"
    catalog_directory.mkdir()
    catalog = catalog_directory / "catalog.json"
    catalog.write_text(json.dumps(_payload()), encoding="utf-8")
    link = catalog_root / "catalog-directory-link"
    link.symlink_to(catalog_directory, target_is_directory=True)
    with pytest.raises(OptimizationCatalogError, match="symlink"):
        load_optimization_catalog(
            link / "catalog.json",
            repository_root=catalog_root,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: {
                **p,
                "recipes": {
                    **p["recipes"],
                    "ridge_weather": {
                        **p["recipes"]["ridge_weather"],
                        "parameters": {"fit_intercept": True},
                    },
                },
            },
            "unknown parameters",
        ),
        (
            lambda p: {
                **p,
                "recipes": {
                    **p["recipes"],
                    "ridge_weather": {
                        **p["recipes"]["ridge_weather"],
                        "parameters": {"alpha": 2.0},
                    },
                },
            },
            "outside allowed values",
        ),
        (
            lambda p: {
                **p,
                "feature_sets": {
                    **p["feature_sets"],
                    "safe_weather": {
                        **p["feature_sets"]["safe_weather"],
                        "specs": [
                            {
                                **p["feature_sets"]["safe_weather"]["specs"][0],
                                "transform": "unsupported_transform",
                            }
                        ],
                    },
                },
            },
            "unknown transform",
        ),
    ],
)
def test_rejects_unsupported_recipe_parameter_or_feature_transform(
    write_catalog, catalog_root, mutate, message
):
    path = write_catalog("invalid-policy.json", mutate(_payload()))

    with pytest.raises(OptimizationCatalogError, match=message):
        load_optimization_catalog(path, repository_root=catalog_root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: {
                **p,
                "searches": {
                    **p["searches"],
                    "bounded_lightgbm_tpe": {
                        **p["searches"]["bounded_lightgbm_tpe"],
                        "spaces": {
                            "lightgbm": {
                                **p["searches"]["bounded_lightgbm_tpe"]["spaces"]["lightgbm"],
                                "n_estimators": [100, "300"],
                            }
                        },
                    },
                },
            },
            "must be an integer",
        ),
        (
            lambda p: {
                **p,
                "searches": {
                    **p["searches"],
                    "bounded_lightgbm_tpe": {
                        **p["searches"]["bounded_lightgbm_tpe"],
                        "spaces": {
                            "lightgbm": {
                                **p["searches"]["bounded_lightgbm_tpe"]["spaces"]["lightgbm"],
                                "learning_rate": [0.03, 0.2],
                            }
                        },
                    },
                },
            },
            "outside allowed values",
        ),
    ],
)
def test_rejects_invalid_tpe_values(write_catalog, catalog_root, mutate, message):
    path = write_catalog("invalid-tpe.json", mutate(_payload()))

    with pytest.raises(OptimizationCatalogError, match=message):
        load_optimization_catalog(path, repository_root=catalog_root)


def test_rejects_catalog_values_outside_code_supported_sets(write_catalog, catalog_root):
    payload = _payload()
    ridge = payload["recipes"]["ridge_weather"]
    ridge["parameters"]["alpha"] = 999
    ridge["allowed_parameters"]["alpha"] = [999]
    path = write_catalog("unsupported-ridge-alpha.json", payload)

    with pytest.raises(OptimizationCatalogError, match="outside supported values"):
        load_optimization_catalog(path, repository_root=catalog_root)

    payload = _payload()
    lightgbm = payload["recipes"]["lgbm_search"]
    lightgbm["parameters"]["n_estimators"] = 999
    lightgbm["allowed_parameters"]["n_estimators"] = [999]
    payload["searches"]["bounded_lightgbm_tpe"]["spaces"]["lightgbm"][
        "n_estimators"
    ] = [999]
    path = write_catalog("unsupported-lightgbm-tpe.json", payload)

    with pytest.raises(OptimizationCatalogError, match="outside supported values"):
        load_optimization_catalog(path, repository_root=catalog_root)
