from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pandas as pd
import pytest

from power_forecasting import aidm
from power_forecasting.data import generate_synthetic_data
from power_forecasting.experiments import ExperimentStore


@pytest.fixture
def agentic_db_path(request):
    root = Path(__file__).resolve().parents[1] / "runs" / "pytest-agentic-aidm" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root / "experiments.sqlite"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_valid_proposal_evaluates_recipe_feature_products_and_persists_metadata(agentic_db_path, monkeypatch):
    calls = []

    def fake_evaluate(frame, definition, feature_specs, folds):
        calls.append((definition.name, tuple(spec.name for spec in feature_specs)))
        if not feature_specs:
            return _evaluation(frame, 0.20)
        score = {
            ("Recipe:hist_gradient_boosting:hgb_small", ("effective_irradiance",)): 0.11,
            ("Recipe:ridge:ridge_low", ("effective_irradiance",)): 0.10,
        }[(definition.name, tuple(spec.name for spec in feature_specs))]
        return _evaluation(frame, score)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    proposal = _proposal()

    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=3),
        agentic_db_path,
        aidm.AIDMConfig(folds=1, minimum_improvement=0.0, max_plant_regression=1.0),
        proposal=proposal,
    )

    assert calls == [
        ("SPOT", ()),
        ("Recipe:ridge:ridge_low", ("effective_irradiance",)),
        ("Recipe:hist_gradient_boosting:hgb_small", ("effective_irradiance",)),
    ]
    assert result.ranking == ("ridge_low:safe_solar", "hgb_small:safe_solar")
    assert result.winner.name == "ridge_low:safe_solar"
    assert result.winner.model_recipe == proposal["model_recipes"][0]
    assert result.manifest["decision"] == "promote"
    assert result.manifest["proposal"]["proposal_id"] == "proposal-safe-001"
    assert result.manifest["selected_model_recipe"] == proposal["model_recipes"][0]
    assert json.loads(json.dumps(result.manifest, sort_keys=True, allow_nan=False)) == result.manifest

    store = ExperimentStore(agentic_db_path)
    for candidate in result.candidates:
        run = store.get_run(candidate.run_id)
        assert run["params"]["proposal_id"] == "proposal-safe-001"
        assert run["params"]["model_recipe"] == candidate.model_recipe
        assert run["artifacts"]["summary"]["model_recipe"] == candidate.model_recipe
        assert run["artifacts"]["proposal"]["proposal_id"] == "proposal-safe-001"


def test_proposal_budget_fails_before_any_evaluation(agentic_db_path, monkeypatch):
    calls = []
    monkeypatch.setattr(aidm, "evaluate_model", lambda *args, **kwargs: calls.append(args))
    proposal = _proposal(budget={"max_evaluations": 1, "top_feature_groups": 1})

    with pytest.raises(ValueError, match="max_evaluations"):
        aidm.run_aidm(
            generate_synthetic_data(days=4, plants=2, seed=4),
            agentic_db_path,
            aidm.AIDMConfig(folds=1),
            proposal=proposal,
        )

    assert calls == []
    assert not agentic_db_path.exists()


def test_agentic_results_are_deterministic(agentic_db_path, monkeypatch):
    monkeypatch.setattr(aidm, "evaluate_model", lambda frame, definition, feature_specs, folds: _evaluation(frame, 0.2 if not feature_specs else 0.1))
    frame = generate_synthetic_data(days=4, plants=2, seed=6)
    config = aidm.AIDMConfig(folds=1, minimum_improvement=0.0, max_plant_regression=1.0)

    first = aidm.run_aidm(frame, agentic_db_path, config, proposal=_proposal())
    second = aidm.run_aidm(frame, agentic_db_path.parent / "second.sqlite", config, proposal=_proposal())

    assert first.ranking == second.ranking
    assert first.manifest == second.manifest


def test_legacy_predictions_require_exact_coverage_and_can_block_promotion(agentic_db_path, monkeypatch):
    def fake_evaluate(frame, definition, feature_specs, folds):
        return _evaluation(frame, 0.20 if not feature_specs else 0.10)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    frame = generate_synthetic_data(days=4, plants=2, seed=8)
    legacy = _legacy_predictions_for_frame(frame, prediction_mw=frame["generation_mw"].to_numpy())

    result = aidm.run_aidm(
        frame,
        agentic_db_path,
        aidm.AIDMConfig(folds=1, minimum_improvement=0.0, max_plant_regression=1.0),
        proposal=_proposal(),
        legacy_predictions=legacy,
    )

    assert result.manifest["decision"] == "reject"
    assert result.manifest["legacy_baseline"]["metrics"]["nmae"] == 0.0
    assert result.manifest["failed_gates"] == [
        "legacy_regression:winner_nmae=0.100000>legacy_nmae=0.000000"
    ]

    missing = legacy.iloc[:-1]
    with pytest.raises(ValueError, match="coverage"):
        aidm.run_aidm(
            frame,
            agentic_db_path.parent / "missing.sqlite",
            aidm.AIDMConfig(folds=1),
            proposal=_proposal(),
            legacy_predictions=missing,
        )


def test_lightgbm_optuna_search_selects_best_trial_with_deterministic_provenance(agentic_db_path, monkeypatch):
    optuna_state = _install_fake_optuna(monkeypatch)
    _install_fake_lightgbm(monkeypatch)
    calls = []

    def fake_evaluate(frame, definition, feature_specs, folds):
        if not feature_specs:
            return _evaluation(frame, 0.30)
        if definition.name == "Recipe:ridge:ridge_low":
            calls.append((definition.name, None, tuple(spec.name for spec in feature_specs)))
            return _evaluation(frame, 0.11)
        estimator = definition.estimator_factory().steps[-1][1]
        calls.append((definition.name, estimator.parameters, tuple(spec.name for spec in feature_specs)))
        score = {
            (100, 0.03, 15, 10): 0.12,
            (300, 0.1, 31, 20): 0.08,
        }[
            (
                estimator.parameters["n_estimators"],
                estimator.parameters["learning_rate"],
                estimator.parameters["num_leaves"],
                estimator.parameters["min_child_samples"],
            )
        ]
        return _evaluation(frame, score)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    proposal = _proposal(
        model_recipes=[_proposal()["model_recipes"][0]],
        search=_lightgbm_search(n_trials=2),
        budget={"max_evaluations": 3, "top_feature_groups": 1},
    )

    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=12),
        agentic_db_path,
        aidm.AIDMConfig(folds=1, minimum_improvement=0.0, max_plant_regression=1.0),
        proposal=proposal,
    )

    assert optuna_state["sampler_seeds"] == [7]
    assert optuna_state["pruners"] == ["NopPruner"]
    assert result.ranking == ("optuna_lightgbm_1:safe_solar", "ridge_low:safe_solar")
    assert result.winner.name == "optuna_lightgbm_1:safe_solar"
    assert result.winner.model_recipe == {
        "name": "optuna_lightgbm_1",
        "recipe": "lightgbm",
        "parameters": {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 20,
        },
        "rationale": "Optuna TPE trial 1 for bounded LightGBM search.",
        "search": {
            "sampler": "tpe",
            "seed": 7,
            "n_trials": 2,
            "space": proposal["search"]["spaces"]["lightgbm"],
            "trial_number": 1,
            "feature_set": "safe_solar",
        },
    }
    assert result.manifest["selected_model_recipe"] == result.winner.model_recipe

    store = ExperimentStore(agentic_db_path)
    runs = sorted(store.list_runs(), key=lambda run: run["name"])
    trial_runs = [run for run in runs if run["name"].startswith("aidm-optuna-lightgbm")]
    assert [run["name"] for run in trial_runs] == [
        "aidm-optuna-lightgbm-safe_solar-trial-0",
        "aidm-optuna-lightgbm-safe_solar-trial-1",
    ]
    assert trial_runs[1]["params"]["search"] == {
        "sampler": "tpe",
        "seed": 7,
        "n_trials": 2,
        "space": proposal["search"]["spaces"]["lightgbm"],
        "trial_number": 1,
        "feature_set": "safe_solar",
    }
    assert trial_runs[1]["params"]["model_recipe"] == result.winner.model_recipe
    assert trial_runs[1]["artifacts"]["summary"]["model_recipe"] == result.winner.model_recipe
    assert calls[0][0] == "Recipe:ridge:ridge_low"
    assert calls[1][0] == "Recipe:lightgbm:optuna_lightgbm_0"
    assert calls[2][0] == "Recipe:lightgbm:optuna_lightgbm_1"


def test_lightgbm_optuna_search_reuses_duplicate_resolved_parameters(agentic_db_path, monkeypatch):
    _install_fake_optuna(monkeypatch)
    _install_fake_lightgbm(monkeypatch)
    lightgbm_calls = []

    def fake_evaluate(frame, definition, feature_specs, folds):
        if not feature_specs:
            return _evaluation(frame, 0.30)
        if definition.name == "Recipe:ridge:ridge_low":
            return _evaluation(frame, 0.20)
        estimator = definition.estimator_factory().steps[-1][1]
        lightgbm_calls.append((
            definition.name,
            estimator.parameters["n_estimators"],
            estimator.parameters["learning_rate"],
            estimator.parameters["num_leaves"],
            estimator.parameters["min_child_samples"],
            tuple(spec.name for spec in feature_specs),
        ))
        return _evaluation(frame, 0.08)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    singleton_search = _lightgbm_search(n_trials=2)
    singleton_search["spaces"]["lightgbm"] = {
        "n_estimators": [100],
        "learning_rate": [0.03],
        "num_leaves": [15],
        "min_child_samples": [10],
    }
    proposal = _proposal(
        model_recipes=[_proposal()["model_recipes"][0]],
        search=singleton_search,
        budget={"max_evaluations": 3, "top_feature_groups": 1},
    )

    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=16),
        agentic_db_path,
        aidm.AIDMConfig(folds=1, minimum_improvement=0.0, max_plant_regression=1.0),
        proposal=proposal,
    )

    assert lightgbm_calls == [
        (
            "Recipe:lightgbm:optuna_lightgbm_0",
            100,
            0.03,
            15,
            10,
            ("effective_irradiance",),
        )
    ]
    assert result.winner.name == "optuna_lightgbm_0:safe_solar"

    store = ExperimentStore(agentic_db_path)
    trial_runs = [
        run
        for run in store.list_runs()
        if run["name"].startswith("aidm-optuna-lightgbm")
    ]
    assert len(trial_runs) == 1
    actual_run = trial_runs[0]
    assert result.winner.run_id == actual_run["id"]
    assert actual_run["artifacts"]["reused_trials"] == [
        {
            "trial_number": 1,
            "source_trial_number": 0,
            "source_run_id": actual_run["id"],
            "source_candidate_name": "optuna_lightgbm_0:safe_solar",
        }
    ]


def test_search_budget_fails_before_baseline_or_store_creation(agentic_db_path, monkeypatch):
    calls = []
    monkeypatch.setattr(aidm, "evaluate_model", lambda *args, **kwargs: calls.append(args))
    proposal = _proposal(
        model_recipes=[_proposal()["model_recipes"][0]],
        search=_lightgbm_search(n_trials=10),
        budget={"max_evaluations": 10, "top_feature_groups": 1},
    )

    with pytest.raises(ValueError, match="max_evaluations"):
        aidm.run_aidm(
            generate_synthetic_data(days=4, plants=2, seed=13),
            agentic_db_path,
            aidm.AIDMConfig(folds=1),
            proposal=proposal,
        )

    assert calls == []
    assert not agentic_db_path.exists()


def test_failed_optuna_trial_is_recorded_before_exception(agentic_db_path, monkeypatch):
    _install_fake_optuna(monkeypatch)
    _install_fake_lightgbm(monkeypatch)

    def fake_evaluate(frame, definition, feature_specs, folds):
        if not feature_specs:
            return _evaluation(frame, 0.30)
        if definition.name == "Recipe:ridge:ridge_low":
            return _evaluation(frame, 0.20)
        estimator = definition.estimator_factory().steps[-1][1]
        if estimator.parameters["n_estimators"] == 100:
            raise RuntimeError("trial exploded")
        return _evaluation(frame, 0.08)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    proposal = _proposal(
        model_recipes=[_proposal()["model_recipes"][0]],
        search=_lightgbm_search(n_trials=2),
        budget={"max_evaluations": 3, "top_feature_groups": 1},
    )

    with pytest.raises(RuntimeError, match="trial exploded"):
        aidm.run_aidm(
            generate_synthetic_data(days=4, plants=2, seed=14),
            agentic_db_path,
            aidm.AIDMConfig(folds=1),
            proposal=proposal,
        )

    failed = ExperimentStore(agentic_db_path).list_runs(status="failed")
    assert len(failed) == 1
    assert failed[0]["name"] == "aidm-optuna-lightgbm-safe_solar-trial-0"
    assert failed[0]["error"] == "trial exploded"


def test_legacy_gate_still_applies_to_search_winner(agentic_db_path, monkeypatch):
    _install_fake_optuna(monkeypatch)
    _install_fake_lightgbm(monkeypatch)

    def fake_evaluate(frame, definition, feature_specs, folds):
        if not feature_specs:
            return _evaluation(frame, 0.20)
        if definition.name == "Recipe:ridge:ridge_low":
            return _evaluation(frame, 0.15)
        return _evaluation(frame, 0.10)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)
    frame = generate_synthetic_data(days=4, plants=2, seed=15)
    legacy = _legacy_predictions_for_frame(frame, prediction_mw=frame["generation_mw"].to_numpy())
    proposal = _proposal(
        model_recipes=[_proposal()["model_recipes"][0]],
        search=_lightgbm_search(n_trials=1),
        budget={"max_evaluations": 2, "top_feature_groups": 1},
    )

    result = aidm.run_aidm(
        frame,
        agentic_db_path,
        aidm.AIDMConfig(folds=1, minimum_improvement=0.0, max_plant_regression=1.0),
        proposal=proposal,
        legacy_predictions=legacy,
    )

    assert result.winner.name == "optuna_lightgbm_0:safe_solar"
    assert result.manifest["decision"] == "reject"
    assert result.manifest["failed_gates"] == [
        "legacy_regression:winner_nmae=0.100000>legacy_nmae=0.000000"
    ]


def test_no_proposal_mode_keeps_backward_compatible_candidate_shape(agentic_db_path, monkeypatch):
    monkeypatch.setattr(aidm, "evaluate_model", lambda frame, definition, feature_specs, folds: _evaluation(frame, 0.2 if not feature_specs else 0.1))

    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=10),
        agentic_db_path,
        aidm.AIDMConfig(folds=1, top_single_candidates=1, minimum_improvement=0.0, max_plant_regression=1.0),
    )

    assert result.manifest["baseline"]["model"] == "SPOT"
    assert "proposal" not in result.manifest
    assert result.winner.model_recipe is None
    assert "model_recipe" not in result.winner.summary()


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
                "parameters": {"max_iter": 50, "learning_rate": 0.1, "max_leaf_nodes": 31},
                "rationale": "Bounded tree model.",
            },
        ],
        "budget": {"max_evaluations": 10, "top_feature_groups": 3},
    }
    payload.update(overrides)
    return payload


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


def _install_fake_lightgbm(monkeypatch):
    module = ModuleType("lightgbm")

    class FakeLGBMRegressor:
        def __init__(self, **parameters):
            self.parameters = dict(parameters)

    module.LGBMRegressor = FakeLGBMRegressor
    monkeypatch.setitem(sys.modules, "lightgbm", module)


def _install_fake_optuna(monkeypatch):
    state = {"sampler_seeds": [], "pruners": []}
    module = ModuleType("optuna")
    samplers = ModuleType("optuna.samplers")
    pruners = ModuleType("optuna.pruners")

    class TPESampler:
        def __init__(self, *, seed):
            state["sampler_seeds"].append(seed)
            self.seed = seed

    class NopPruner:
        def __init__(self):
            state["pruners"].append("NopPruner")

    class Trial:
        def __init__(self, number):
            self.number = number
            self.params = {}

        def suggest_categorical(self, name, choices):
            value = choices[self.number % len(choices)]
            self.params[name] = value
            return value

    class Study:
        def __init__(self, *, sampler, pruner, direction):
            self.sampler = sampler
            self.pruner = pruner
            self.direction = direction
            self.trials = []

        def optimize(self, objective, n_trials):
            for number in range(n_trials):
                trial = Trial(number)
                value = objective(trial)
                trial.value = value
                self.trials.append(trial)

    def create_study(*, sampler, pruner, direction):
        return Study(sampler=sampler, pruner=pruner, direction=direction)

    samplers.TPESampler = TPESampler
    pruners.NopPruner = NopPruner
    module.samplers = samplers
    module.pruners = pruners
    module.create_study = create_study
    monkeypatch.setitem(sys.modules, "optuna", module)
    monkeypatch.setitem(sys.modules, "optuna.samplers", samplers)
    monkeypatch.setitem(sys.modules, "optuna.pruners", pruners)
    return state


def _evaluation(frame: pd.DataFrame, score: float):
    validation = frame.tail(4).copy()
    prediction = (validation["generation_mw"] + score * validation["capacity_mw"]).clip(
        0, validation["capacity_mw"]
    )
    return SimpleNamespace(
        metrics={"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
        per_plant={
            str(plant_id): {"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score}
            for plant_id in sorted(frame["plant_id"].unique())
        },
        fold_metrics=[{"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score}],
        predictions=pd.DataFrame(
            {
                "timestamp": validation["timestamp"].to_numpy(),
                "plant_id": validation["plant_id"].to_numpy(),
                "actual": validation["generation_mw"].to_numpy(),
                "prediction": prediction.to_numpy(),
                "capacity_mw": validation["capacity_mw"].to_numpy(),
                "fold": 1,
            }
        ),
    )


def _legacy_predictions_for_frame(frame: pd.DataFrame, prediction_mw) -> pd.DataFrame:
    validation = frame.tail(4).copy()
    return pd.DataFrame(
        {
            "plant_id": validation["plant_id"].to_numpy(),
            "timestamp": validation["timestamp"].to_numpy(),
            "prediction_mw": prediction_mw[-4:],
        }
    )
