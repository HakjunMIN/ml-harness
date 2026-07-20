import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from power_forecasting import aidm
from power_forecasting.data import DataContractError, generate_synthetic_data
from power_forecasting.experiments import ExperimentStore
from power_forecasting.features import FeatureSpec


@pytest.fixture
def aidm_db_path(request):
    root = Path(__file__).resolve().parents[1] / "runs" / "pytest-aidm" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root / "experiments.sqlite"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_real_aidm_run_promotes_and_is_deterministic_on_45_day_dataset(aidm_db_path):
    frame = generate_synthetic_data(days=45, plants=2, seed=11)
    config = aidm.AIDMConfig(
        folds=3,
        minimum_improvement=0.0,
        max_plant_regression=0.2,
    )
    second_db = aidm_db_path.parent / "second.sqlite"

    first = aidm.run_aidm(frame, aidm_db_path, config)
    second = aidm.run_aidm(frame, second_db, config)

    assert first.ranking == second.ranking
    assert first.winner is not None
    assert first.manifest["decision"] == "promote"
    assert first.manifest["failed_gates"] == []
    assert json.loads(json.dumps(first.manifest, sort_keys=True, allow_nan=False)) == first.manifest
    assert json.loads(json.dumps(first.winner.summary(), sort_keys=True)) == first.winner.summary()
    assert first.manifest == second.manifest


def test_exact_candidate_catalog_has_no_leakage_and_bounded_combination_count(
    aidm_db_path, monkeypatch
):
    expected_catalog = [
        ("hour_sin", "cyclic_hour", ("timestamp",), {}),
        ("hour_cos", "cyclic_hour", ("timestamp",), {}),
        ("doy_sin", "cyclic_day_of_year", ("timestamp",), {}),
        ("doy_cos", "cyclic_day_of_year", ("timestamp",), {}),
        (
            "effective_irradiance",
            "effective_irradiance",
            ("forecast_irradiance", "forecast_cloud_cover"),
            {},
        ),
        (
            "temperature_derating",
            "temperature_derating",
            ("forecast_irradiance", "forecast_temperature"),
            {},
        ),
        ("cloud_attenuation", "cloud_attenuation", ("forecast_cloud_cover",), {}),
        (
            "irradiance_temperature_interaction",
            "interaction",
            ("forecast_irradiance", "forecast_temperature"),
            {},
        ),
    ]

    catalog = aidm.candidate_catalog()

    assert [
        (spec.name, spec.transform, spec.inputs, dict(spec.parameters))
        for spec in catalog
    ] == expected_catalog
    for spec in catalog:
        assert "generation_mw" not in spec.inputs
        assert not any(source.startswith("actual_") for source in spec.inputs)

    _install_fake_evaluator(monkeypatch)
    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=5),
        aidm_db_path,
        aidm.AIDMConfig(folds=1, top_single_candidates=3),
    )

    assert len(result.candidates) == 10
    assert len({candidate.name for candidate in result.candidates}) == 10
    assert all(len({spec.name for spec in candidate.specs}) == len(candidate.specs) for candidate in result.candidates)


def test_ranking_is_deterministic_and_uses_stable_name_tie_breaks(
    aidm_db_path, monkeypatch
):
    _install_fake_evaluator(monkeypatch, default_score=0.1)

    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=7),
        aidm_db_path,
        aidm.AIDMConfig(folds=1, top_single_candidates=3),
    )

    assert result.ranking == tuple(sorted(candidate.name for candidate in result.candidates))
    assert result.winner.name == result.ranking[0]
    assert result.ranking == (
        "cloud_attenuation",
        "cloud_attenuation+doy_cos+doy_sin",
        "cloud_attenuation+doy_cos+doy_sin+effective_irradiance",
        "cloud_attenuation+effective_irradiance",
        "doy_cos+doy_sin",
        "doy_cos+doy_sin+effective_irradiance",
        "effective_irradiance",
        "hour_cos+hour_sin",
        "irradiance_temperature_interaction",
        "temperature_derating",
    )


def test_promotion_gate_rejects_insufficient_improvement():
    baseline = _candidate("baseline", (), 0.100)
    winner = _candidate("hour_sin", [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",))], 0.099)

    gates = aidm.evaluate_promotion_gates(
        baseline,
        winner,
        aidm.AIDMConfig(minimum_improvement=0.02, max_plant_regression=0.2),
    )

    assert gates["decision"] == "reject"
    assert gates["improvement_ratio"] == pytest.approx(0.01)
    assert gates["failed_gates"] == [
        "insufficient_improvement:improvement_ratio=0.010000<threshold=0.020000"
    ]


def test_promotion_gate_rejects_plant_regression():
    baseline = _candidate("baseline", (), 0.100, plant_scores={"plant_01": 0.10, "plant_02": 0.10})
    winner = _candidate(
        "candidate",
        [FeatureSpec("cloud_attenuation", "cloud_attenuation", ("forecast_cloud_cover",))],
        0.090,
        plant_scores={"plant_01": 0.08, "plant_02": 0.16},
    )

    gates = aidm.evaluate_promotion_gates(
        baseline,
        winner,
        aidm.AIDMConfig(minimum_improvement=0.0, max_plant_regression=0.03),
    )

    assert gates["decision"] == "reject"
    assert gates["per_plant_deltas"] == {"plant_01": -0.02, "plant_02": 0.06}
    assert gates["failed_gates"] == [
        "plant_regression:plant_02:delta=0.060000>threshold=0.030000"
    ]


def test_promotion_gate_rejects_unavailable_actual_or_target_inputs():
    baseline = _candidate("baseline", (), 0.100)
    winner = _candidate(
        "leaky",
        [
            FeatureSpec(
                "leaky_actual",
                "interaction",
                ("actual_irradiance", "forecast_irradiance"),
            ),
            FeatureSpec("leaky_target", "interaction", ("generation_mw", "capacity_mw")),
        ],
        0.050,
    )

    gates = aidm.evaluate_promotion_gates(
        baseline,
        winner,
        aidm.AIDMConfig(minimum_improvement=0.0, max_plant_regression=0.2),
    )

    assert gates["decision"] == "reject"
    assert gates["failed_gates"] == [
        "unavailable_input:actual_irradiance",
        "unavailable_input:generation_mw",
    ]


def test_experiment_store_contains_baseline_and_every_candidate_completed_with_specs(
    aidm_db_path, monkeypatch
):
    _install_fake_evaluator(monkeypatch)

    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=9),
        aidm_db_path,
        aidm.AIDMConfig(folds=1, top_single_candidates=3),
    )

    store = ExperimentStore(aidm_db_path)
    completed_runs = store.list_runs("completed")

    assert len(completed_runs) == 1 + len(result.candidates)
    baseline_run = store.get_run(result.baseline.run_id)
    assert baseline_run["name"] == "aidm-baseline-spot"
    assert baseline_run["params"]["specs"] == []

    for candidate in result.candidates:
        run = store.get_run(candidate.run_id)
        assert run["status"] == "completed"
        assert run["name"] == f"aidm-candidate-{candidate.name}"
        assert run["params"]["candidate_name"] == candidate.name
        assert run["params"]["specs"] == [spec.to_dict() for spec in candidate.specs]
        assert run["metrics"] == candidate.metrics
        assert run["artifacts"]["summary"] == candidate.summary()


def test_invalid_config_and_data_are_rejected(aidm_db_path):
    with pytest.raises(ValueError, match="folds"):
        aidm.AIDMConfig(folds=0)
    with pytest.raises(ValueError, match="minimum_improvement"):
        aidm.AIDMConfig(minimum_improvement=-0.01)
    with pytest.raises(ValueError, match="max_plant_regression"):
        aidm.AIDMConfig(max_plant_regression=-0.01)
    with pytest.raises(ValueError, match="top_single_candidates"):
        aidm.AIDMConfig(top_single_candidates=0)
    with pytest.raises(TypeError, match="seed"):
        aidm.AIDMConfig(seed=1.5)
    with pytest.raises(ValueError, match="seed"):
        aidm.AIDMConfig(seed=-1)

    config = aidm.AIDMConfig()
    with pytest.raises(FrozenInstanceError):
        config.folds = 4

    invalid = generate_synthetic_data(days=2, plants=1, seed=2).drop(columns=["forecast_irradiance"])
    with pytest.raises(DataContractError, match="missing required columns"):
        aidm.run_aidm(invalid, aidm_db_path)


def test_injected_evaluation_failure_records_failed_run_and_re_raises(
    aidm_db_path, monkeypatch
):
    def failing_after_baseline(frame, definition, feature_specs, folds):
        if feature_specs:
            raise RuntimeError("candidate evaluation crashed")
        return _fake_evaluation(0.2)

    monkeypatch.setattr(aidm, "evaluate_model", failing_after_baseline)

    with pytest.raises(RuntimeError, match="candidate evaluation crashed"):
        aidm.run_aidm(
            generate_synthetic_data(days=4, plants=2, seed=13),
            aidm_db_path,
            aidm.AIDMConfig(folds=1),
        )

    store = ExperimentStore(aidm_db_path)
    completed = store.list_runs("completed")
    failed = store.list_runs("failed")
    assert len(completed) == 1
    assert completed[0]["params"]["specs"] == []
    assert len(failed) == 1
    assert failed[0]["error"] == "candidate evaluation crashed"
    assert failed[0]["params"]["candidate_name"] == "hour_cos+hour_sin"
    assert failed[0]["params"]["specs"]


def _install_fake_evaluator(monkeypatch, score_by_name=None, default_score=0.08):
    score_by_name = score_by_name or {}

    def fake_evaluate(frame, definition, feature_specs, folds):
        name = aidm.stable_candidate_name(feature_specs) if feature_specs else "baseline"
        score = score_by_name.get(name, default_score)
        if not feature_specs:
            score = score_by_name.get("baseline", 0.1)
        return _fake_evaluation(score)

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)


def _fake_evaluation(score):
    return SimpleNamespace(
        metrics={"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
        per_plant={
            "plant_01": {"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
            "plant_02": {"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
        },
        fold_metrics=[{"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score}],
        predictions=pd.DataFrame({"prediction": [1.0]}),
    )


def _candidate(name, specs, nmae, plant_scores=None):
    plant_scores = plant_scores or {"plant_01": nmae, "plant_02": nmae}
    return aidm.CandidateResult(
        name=name,
        specs=tuple(specs),
        metrics={"mae": nmae * 100.0, "rmse": nmae * 120.0, "nmae": nmae},
        per_plant={
            plant: {"mae": score * 100.0, "rmse": score * 120.0, "nmae": score}
            for plant, score in plant_scores.items()
        },
        run_id=f"{name}-run",
    )
