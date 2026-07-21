from __future__ import annotations

import math
import sys
from types import ModuleType

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from power_forecasting.models import SPOT_FEATURES, model_definition_from_recipe
from power_forecasting.proposals import ProposalValidationError, load_proposal, proposal_to_dict


def _proposal(**overrides):
    payload = {
        "schema_version": "1",
        "proposal_id": "proposal-safe-001",
        "rationale": "Evaluate bounded prediction-time feature and model hypotheses.",
        "baseline": {"model": "SPOT"},
        "feature_sets": [
            {
                "name": "safe_solar",
                "rationale": "Use only forecast-time irradiance and cloud cover.",
                "specs": [
                    {
                        "name": "effective_irradiance",
                        "transform": "effective_irradiance",
                        "inputs": ["forecast_irradiance", "forecast_cloud_cover"],
                        "parameters": {},
                        "version": "1",
                        "rationale": "Forecast-time solar attenuation.",
                    }
                ],
            }
        ],
        "model_recipes": [
            {
                "name": "ridge_low",
                "recipe": "ridge",
                "parameters": {"alpha": 1.0},
                "rationale": "Linear regularized baseline.",
            },
            {
                "name": "hgb_small",
                "recipe": "hist_gradient_boosting",
                "parameters": {
                    "max_iter": 50,
                    "learning_rate": 0.1,
                    "max_leaf_nodes": 31,
                },
                "rationale": "Bounded tree model.",
            },
            {
                "name": "forest_small",
                "recipe": "random_forest",
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 8,
                    "min_samples_leaf": 2,
                },
                "rationale": "Bounded deterministic forest.",
            },
            {
                "name": "xgb_small",
                "recipe": "xgboost",
                "parameters": {
                    "n_estimators": 200,
                    "max_depth": 6,
                    "learning_rate": 0.03,
                    "subsample": 0.8,
                },
                "rationale": "Bounded deterministic boosted tree model.",
            },
            {
                "name": "lgbm_small",
                "recipe": "lightgbm",
                "parameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.03,
                    "num_leaves": 15,
                    "min_child_samples": 10,
                },
                "rationale": "Bounded deterministic LightGBM model.",
            },
        ],
        "budget": {"max_evaluations": 10, "top_feature_groups": 3},
    }
    payload.update(overrides)
    return payload


def test_valid_proposal_round_trips_and_model_recipes_build_expected_estimators():
    proposal = load_proposal(_proposal())

    assert proposal.schema_version == "1"
    assert proposal.feature_sets[0].specs[0].name == "effective_irradiance"
    assert proposal_to_dict(proposal) == _proposal()

    ridge = model_definition_from_recipe(proposal.model_recipes[0])
    assert ridge.name == "Recipe:ridge:ridge_low"
    assert ridge.base_features == SPOT_FEATURES
    ridge_steps = ridge.estimator_factory().steps
    assert isinstance(ridge_steps[-1][1], Ridge)
    assert ridge_steps[-1][1].alpha == 1.0

    hgb = model_definition_from_recipe(proposal.model_recipes[1])
    estimator = hgb.estimator_factory().steps[-1][1]
    assert isinstance(estimator, HistGradientBoostingRegressor)
    assert estimator.random_state == 0
    assert estimator.max_iter == 50
    assert estimator.learning_rate == 0.1
    assert estimator.max_leaf_nodes == 31

    forest = model_definition_from_recipe(proposal.model_recipes[2])
    assert forest.name == "Recipe:random_forest:forest_small"
    assert forest.base_features == SPOT_FEATURES
    assert forest.data_availability == "forecast"
    forest_steps = forest.estimator_factory().steps
    assert list(name for name, _ in forest_steps) == ["simpleimputer", "randomforestregressor"]
    assert forest_steps[0][1].strategy == "median"
    forest_estimator = forest_steps[-1][1]
    assert isinstance(forest_estimator, RandomForestRegressor)
    assert forest_estimator.random_state == 0
    assert forest_estimator.n_jobs == 1
    assert forest_estimator.n_estimators == 100
    assert forest_estimator.max_depth == 8
    assert forest_estimator.min_samples_leaf == 2

    xgboost = _import_xgboost_or_skip()
    xgb = model_definition_from_recipe(proposal.model_recipes[3])
    assert xgb.name == "Recipe:xgboost:xgb_small"
    assert xgb.base_features == SPOT_FEATURES
    assert xgb.data_availability == "forecast"
    xgb_steps = xgb.estimator_factory().steps
    assert list(name for name, _ in xgb_steps) == ["simpleimputer", "xgbregressor"]
    assert xgb_steps[0][1].strategy == "median"
    xgb_estimator = xgb_steps[-1][1]
    assert isinstance(xgb_estimator, xgboost.XGBRegressor)
    assert xgb_estimator.random_state == 0
    assert xgb_estimator.n_jobs == 1
    assert xgb_estimator.objective == "reg:squarederror"
    assert xgb_estimator.eval_metric == "rmse"
    assert xgb_estimator.n_estimators == 200
    assert xgb_estimator.max_depth == 6
    assert xgb_estimator.learning_rate == 0.03
    assert xgb_estimator.subsample == 0.8

    lightgbm = _import_lightgbm_or_skip()
    lgbm = model_definition_from_recipe(proposal.model_recipes[4])
    assert lgbm.name == "Recipe:lightgbm:lgbm_small"
    assert lgbm.base_features == SPOT_FEATURES
    assert lgbm.data_availability == "forecast"
    lgbm_steps = lgbm.estimator_factory().steps
    assert list(name for name, _ in lgbm_steps) == ["simpleimputer", "lgbmregressor"]
    assert lgbm_steps[0][1].strategy == "median"
    lgbm_estimator = lgbm_steps[-1][1]
    assert isinstance(lgbm_estimator, lightgbm.LGBMRegressor)
    assert lgbm_estimator.random_state == 0
    assert lgbm_estimator.n_jobs == 1
    assert lgbm_estimator.verbosity == -1
    assert lgbm_estimator.n_estimators == 100
    assert lgbm_estimator.learning_rate == 0.03
    assert lgbm_estimator.num_leaves == 15
    assert lgbm_estimator.min_child_samples == 10


def test_search_proposal_round_trips_without_breaking_legacy_proposals():
    legacy = load_proposal(_proposal(model_recipes=[_proposal()["model_recipes"][0]]))
    assert proposal_to_dict(legacy) == _proposal(model_recipes=[_proposal()["model_recipes"][0]])
    assert legacy.search is None

    proposal = _proposal(
        model_recipes=[_proposal()["model_recipes"][0]],
        search=_lightgbm_search(n_trials=3),
        budget={"max_evaluations": 4, "top_feature_groups": 1},
    )

    loaded = load_proposal(proposal)

    assert loaded.search == {
        "sampler": "tpe",
        "seed": 7,
        "n_trials": 3,
        "spaces": {
            "lightgbm": {
                "n_estimators": [100, 300],
                "learning_rate": [0.03, 0.1],
                "num_leaves": [15, 31],
                "min_child_samples": [10, 20],
            }
        },
    }
    assert proposal_to_dict(loaded) == proposal


def test_search_proposal_rejects_duplicate_lightgbm_choices():
    proposal = _proposal(
        model_recipes=[_proposal()["model_recipes"][0]],
        search=_lightgbm_search(n_trials=3),
        budget={"max_evaluations": 4, "top_feature_groups": 1},
    )
    proposal["search"]["spaces"]["lightgbm"]["n_estimators"] = [100, 100]

    with pytest.raises(ProposalValidationError, match="duplicate"):
        load_proposal(proposal)


def test_proposal_validation_accepts_forecast_history_feature_specs():
    proposal = _proposal()
    proposal["feature_sets"][0]["specs"] = [
        {
            "name": "prior_irradiance",
            "transform": "lag",
            "inputs": ["forecast_irradiance"],
            "parameters": {"periods": 1},
            "version": "1",
            "rationale": "Use strictly prior forecast irradiance from the same plant.",
        },
        {
            "name": "prior_cloud_mean",
            "transform": "rolling_mean",
            "inputs": ["forecast_cloud_cover"],
            "parameters": {"window": 3},
            "version": "1",
            "rationale": "Use strictly prior forecast cloud cover from the same plant.",
        },
    ]

    loaded = load_proposal(proposal)

    assert proposal_to_dict(loaded) == proposal


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: {**p, "extra": True}, "unknown keys"),
        (lambda p: {**p, "schema_version": "2"}, "schema_version"),
        (lambda p: {**p, "proposal_id": " "}, "proposal_id"),
        (lambda p: {**p, "baseline": {"model": "SPOT", "extra": True}}, "baseline unknown keys"),
        (lambda p: {**p, "baseline": {"model": "naive"}}, "baseline.model"),
        (lambda p: {**p, "baseline": {}}, "baseline missing keys"),
        (lambda p: {**p, "feature_sets": []}, "feature_sets"),
        (
            lambda p: {
                **p,
                "feature_sets": [
                    p["feature_sets"][0],
                    {**p["feature_sets"][0], "rationale": "duplicate name"},
                ],
            },
            "duplicate feature set",
        ),
        (
            lambda p: {
                **p,
                "feature_sets": [
                    {
                        **p["feature_sets"][0],
                        "specs": [
                            {
                                **p["feature_sets"][0]["specs"][0],
                                "unknown": True,
                            }
                        ],
                    }
                ],
            },
            "feature spec unknown keys",
        ),
        (
            lambda p: {
                **p,
                "feature_sets": [
                    {
                        **p["feature_sets"][0],
                        "specs": [
                            {
                                key: value
                                for key, value in p["feature_sets"][0]["specs"][0].items()
                                if key != "rationale"
                            }
                        ],
                    }
                ],
            },
            "feature spec missing keys",
        ),
        (
            lambda p: {
                **p,
                "feature_sets": [
                    {
                        **p["feature_sets"][0],
                        "specs": [
                            {
                                **p["feature_sets"][0]["specs"][0],
                                "inputs": ["generation_mw", "forecast_cloud_cover"],
                            }
                        ],
                    }
                ],
            },
            "target leakage",
        ),
        (
            lambda p: {
                **p,
                "feature_sets": [
                    {
                        **p["feature_sets"][0],
                        "specs": [
                            {
                                **p["feature_sets"][0]["specs"][0],
                                "inputs": ["actual_irradiance", "forecast_cloud_cover"],
                            }
                        ],
                    }
                ],
            },
            "actual input",
        ),
        (
            lambda p: {
                **p,
                "feature_sets": [
                    {
                        **p["feature_sets"][0],
                        "specs": [
                            {
                                **p["feature_sets"][0]["specs"][0],
                                "inputs": ["customer_secret", "forecast_cloud_cover"],
                            }
                        ],
                    }
                ],
            },
            "unavailable prediction input",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][0], "recipe": "neural_network"}
                ],
            },
            "unsupported recipe",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][0], "parameters": {"alpha": 1.0, "fit_intercept": True}}
                ],
            },
            "unknown parameters",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][0], "parameters": {"alpha": True}}
                ],
            },
            "alpha",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][0], "parameters": {"alpha": math.inf}}
                ],
            },
            "finite",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][0], "parameters": {"alpha": 2.0}}
                ],
            },
            "allowed",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][1], "parameters": {"max_iter": 50, "learning_rate": 0.2, "max_leaf_nodes": 31}}
                ],
            },
            "learning_rate",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][2], "parameters": {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2, "bootstrap": False}}
                ],
            },
            "unknown parameters",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][2], "parameters": {"n_estimators": 100, "max_depth": 8}}
                ],
            },
            "missing parameters",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][2], "parameters": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 2}}
                ],
            },
            "n_estimators",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][2], "parameters": {"n_estimators": True, "max_depth": 8, "min_samples_leaf": 2}}
                ],
            },
            "n_estimators",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][2], "parameters": {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 2}}
                ],
            },
            "max_depth",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][2], "parameters": {"n_estimators": 100, "max_depth": True, "min_samples_leaf": 2}}
                ],
            },
            "max_depth",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][2], "parameters": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 3}}
                ],
            },
            "min_samples_leaf",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][3], "parameters": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "booster": "gbtree"}}
                ],
            },
            "unknown parameters",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][3], "parameters": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03}}
                ],
            },
            "missing parameters",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][3], "parameters": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8}}
                ],
            },
            "n_estimators",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][3], "parameters": {"n_estimators": 200, "max_depth": False, "learning_rate": 0.03, "subsample": 0.8}}
                ],
            },
            "max_depth",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][3], "parameters": {"n_estimators": 200, "max_depth": 6, "learning_rate": math.nan, "subsample": 0.8}}
                ],
            },
            "finite",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][3], "parameters": {"n_estimators": 200, "max_depth": 6, "learning_rate": True, "subsample": 0.8}}
                ],
            },
            "learning_rate",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {**p["model_recipes"][3], "parameters": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.9}}
                ],
            },
            "subsample",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {
                        **p["model_recipes"][4],
                        "parameters": {
                            "n_estimators": 200,
                            "learning_rate": 0.03,
                            "num_leaves": 15,
                            "min_child_samples": 10,
                        },
                    }
                ],
            },
            "n_estimators",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {
                        **p["model_recipes"][4],
                        "parameters": {
                            "n_estimators": 100,
                            "learning_rate": 0.2,
                            "num_leaves": 15,
                            "min_child_samples": 10,
                        },
                    }
                ],
            },
            "learning_rate",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {
                        **p["model_recipes"][4],
                        "parameters": {
                            "n_estimators": 100,
                            "learning_rate": 0.03,
                            "num_leaves": 63,
                            "min_child_samples": 10,
                        },
                    }
                ],
            },
            "num_leaves",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {
                        **p["model_recipes"][4],
                        "parameters": {
                            "n_estimators": 100,
                            "learning_rate": 0.03,
                            "num_leaves": 15,
                            "min_child_samples": 30,
                        },
                    }
                ],
            },
            "min_child_samples",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {
                        **p["model_recipes"][4],
                        "parameters": {
                            "n_estimators": 100,
                            "learning_rate": 0.03,
                            "num_leaves": 15,
                        },
                    }
                ],
            },
            "missing parameters",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [
                    {
                        **p["model_recipes"][4],
                        "parameters": {
                            "n_estimators": 100,
                            "learning_rate": 0.03,
                            "num_leaves": 15,
                            "min_child_samples": 10,
                            "feature_fraction": 0.8,
                        },
                    }
                ],
            },
            "unknown parameters",
        ),
        (lambda p: {**p, "budget": {"max_evaluations": 0, "top_feature_groups": 1}}, "max_evaluations"),
        (lambda p: {**p, "budget": {"max_evaluations": 1, "top_feature_groups": 11}}, "top_feature_groups"),
        (lambda p: {**p, "search": {**_lightgbm_search(), "sampler": "random"}}, "sampler"),
        (lambda p: {**p, "search": {k: v for k, v in _lightgbm_search().items() if k != "seed"}}, "missing keys"),
        (lambda p: {**p, "search": {**_lightgbm_search(), "seed": -1}}, "seed"),
        (lambda p: {**p, "search": {**_lightgbm_search(), "seed": True}}, "seed"),
        (lambda p: {**p, "search": {**_lightgbm_search(), "n_trials": 0}}, "n_trials"),
        (lambda p: {**p, "search": {**_lightgbm_search(), "n_trials": 51}}, "n_trials"),
        (
            lambda p: {
                **p,
                "search": {
                    **_lightgbm_search(),
                    "spaces": {**_lightgbm_search()["spaces"], "xgboost": {}},
                },
            },
            "spaces",
        ),
        (
            lambda p: {
                **p,
                "search": {
                    **_lightgbm_search(),
                    "spaces": {
                        "lightgbm": {
                            **_lightgbm_search()["spaces"]["lightgbm"],
                            "max_depth": [4],
                        }
                    },
                },
            },
            "unknown keys",
        ),
        (
            lambda p: {
                **p,
                "search": {
                    **_lightgbm_search(),
                    "spaces": {
                        "lightgbm": {
                            **_lightgbm_search()["spaces"]["lightgbm"],
                            "learning_rate": [0.05],
                        }
                    },
                },
            },
            "learning_rate",
        ),
        (
            lambda p: {
                **p,
                "search": {
                    **_lightgbm_search(),
                    "spaces": {
                        "lightgbm": {
                            **_lightgbm_search()["spaces"]["lightgbm"],
                            "n_estimators": [],
                        }
                    },
                },
            },
            "nonempty",
        ),
        (
            lambda p: {
                **p,
                "model_recipes": [p["model_recipes"][0]],
                "search": _lightgbm_search(n_trials=10),
                "budget": {"max_evaluations": 10, "top_feature_groups": 1},
            },
            "max_evaluations",
        ),
    ],
)
def test_strict_proposal_rejections(mutate, message):
    with pytest.raises(ProposalValidationError, match=message):
        load_proposal(mutate(_proposal()))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {
            **p,
            "feature_sets": [{**p["feature_sets"][0], "name": "safe:solar"}],
        },
        lambda p: {
            **p,
            "model_recipes": [{**p["model_recipes"][0], "name": "ridge:low"}],
        },
        lambda p: {
            **p,
            "model_recipes": [{**p["model_recipes"][0], "name": "optuna_lightgbm_0"}],
        },
    ],
)
def test_candidate_identity_parts_reject_colons(mutate):
    with pytest.raises(ProposalValidationError, match="name"):
        load_proposal(mutate(_proposal()))


def test_candidate_identity_collision_shapes_are_rejected():
    proposal = _proposal()
    colliding = {
        **proposal,
        "feature_sets": [
            {**proposal["feature_sets"][0], "name": "gamma"},
            {**proposal["feature_sets"][0], "name": "beta:gamma"},
        ],
        "model_recipes": [
            {**proposal["model_recipes"][0], "name": "alpha:beta"},
            {**proposal["model_recipes"][1], "name": "alpha"},
        ],
    }

    with pytest.raises(ProposalValidationError, match="name"):
        load_proposal(colliding)


def test_xgboost_recipe_reports_clear_message_when_optional_dependency_is_missing(monkeypatch):
    proposal = load_proposal(
        _proposal(
            model_recipes=[
                {
                    "name": "xgb_small",
                    "recipe": "xgboost",
                    "parameters": {
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.03,
                        "subsample": 0.8,
                    },
                    "rationale": "Bounded deterministic boosted tree model.",
                }
            ]
        )
    )
    original_import = __import__

    def fail_xgboost_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ModuleNotFoundError("No module named 'xgboost'", name="xgboost")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_xgboost_import)
    definition = model_definition_from_recipe(proposal.model_recipes[0])

    with pytest.raises(ValueError, match="uv sync --extra model-search"):
        definition.estimator_factory()


def test_xgboost_recipe_preserves_native_runtime_import_failures(monkeypatch):
    proposal = load_proposal(
        _proposal(
            model_recipes=[
                {
                    "name": "xgb_small",
                    "recipe": "xgboost",
                    "parameters": {
                        "n_estimators": 200,
                        "max_depth": 6,
                        "learning_rate": 0.03,
                        "subsample": 0.8,
                    },
                    "rationale": "Bounded deterministic boosted tree model.",
                }
            ]
        )
    )
    original_import = __import__
    runtime_message = "dlopen(libxgboost.dylib): Library not loaded: libomp.dylib"

    def fail_xgboost_runtime_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError(runtime_message)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_xgboost_runtime_import)
    definition = model_definition_from_recipe(proposal.model_recipes[0])

    with pytest.raises(ValueError) as excinfo:
        definition.estimator_factory()

    message = str(excinfo.value)
    assert "XGBoost initialization/native runtime failure" in message
    assert runtime_message in message
    assert "uv sync --extra model-search" not in message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_lightgbm_recipe_reports_clear_message_when_optional_dependency_is_missing(monkeypatch):
    proposal = load_proposal(
        _proposal(
            model_recipes=[
                {
                    "name": "lgbm_small",
                    "recipe": "lightgbm",
                    "parameters": {
                        "n_estimators": 100,
                        "learning_rate": 0.03,
                        "num_leaves": 15,
                        "min_child_samples": 10,
                    },
                    "rationale": "Bounded deterministic LightGBM model.",
                }
            ]
        )
    )
    original_import = __import__

    def fail_lightgbm_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ModuleNotFoundError("No module named 'lightgbm'", name="lightgbm")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_lightgbm_import)
    definition = model_definition_from_recipe(proposal.model_recipes[0])

    with pytest.raises(ValueError, match="uv sync --extra model-search"):
        definition.estimator_factory()


def test_lightgbm_recipe_preserves_native_runtime_import_failures(monkeypatch):
    proposal = load_proposal(
        _proposal(
            model_recipes=[
                {
                    "name": "lgbm_small",
                    "recipe": "lightgbm",
                    "parameters": {
                        "n_estimators": 100,
                        "learning_rate": 0.03,
                        "num_leaves": 15,
                        "min_child_samples": 10,
                    },
                    "rationale": "Bounded deterministic LightGBM model.",
                }
            ]
        )
    )
    original_import = __import__
    runtime_message = "dlopen(lib_lightgbm.dylib): Library not loaded: libomp.dylib"

    def fail_lightgbm_runtime_import(name, *args, **kwargs):
        if name == "lightgbm":
            raise ImportError(runtime_message)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_lightgbm_runtime_import)
    definition = model_definition_from_recipe(proposal.model_recipes[0])

    with pytest.raises(ValueError) as excinfo:
        definition.estimator_factory()

    message = str(excinfo.value)
    assert "LightGBM initialization/native runtime failure" in message
    assert runtime_message in message
    assert "uv sync --extra model-search" not in message
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_lightgbm_factory_configuration_can_be_verified_with_mocked_import(monkeypatch):
    module = ModuleType("lightgbm")

    class FakeLGBMRegressor:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module.LGBMRegressor = FakeLGBMRegressor
    monkeypatch.setitem(sys.modules, "lightgbm", module)
    proposal = load_proposal(
        _proposal(
            model_recipes=[
                {
                    "name": "lgbm_large",
                    "recipe": "lightgbm",
                    "parameters": {
                        "n_estimators": 300,
                        "learning_rate": 0.1,
                        "num_leaves": 31,
                        "min_child_samples": 20,
                    },
                    "rationale": "Bounded deterministic LightGBM model.",
                }
            ]
        )
    )

    estimator = model_definition_from_recipe(proposal.model_recipes[0]).estimator_factory().steps[-1][1]

    assert isinstance(estimator, FakeLGBMRegressor)
    assert estimator.random_state == 0
    assert estimator.n_jobs == 1
    assert estimator.verbosity == -1
    assert estimator.n_estimators == 300
    assert estimator.learning_rate == 0.1
    assert estimator.num_leaves == 31
    assert estimator.min_child_samples == 20


def _import_xgboost_or_skip():
    try:
        import xgboost
    except Exception as exc:
        pytest.skip(f"xgboost import failed in this environment: {exc}")
    return xgboost


def _import_lightgbm_or_skip():
    try:
        import lightgbm
    except Exception as exc:
        pytest.skip(f"lightgbm import failed in this environment: {exc}")
    return lightgbm


def _lightgbm_search(n_trials=2):
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
