from __future__ import annotations

import math

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
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
        (lambda p: {**p, "budget": {"max_evaluations": 0, "top_feature_groups": 1}}, "max_evaluations"),
        (lambda p: {**p, "budget": {"max_evaluations": 1, "top_feature_groups": 11}}, "top_feature_groups"),
    ],
)
def test_strict_proposal_rejections(mutate, message):
    with pytest.raises(ProposalValidationError, match=message):
        load_proposal(mutate(_proposal()))
