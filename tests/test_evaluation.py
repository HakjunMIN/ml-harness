import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone

from power_forecasting.data import generate_synthetic_data
from power_forecasting.evaluation import (
    EvaluationResult,
    chronological_folds,
    compute_metrics,
    evaluate_model,
)
from power_forecasting.features import FeatureSpec
from power_forecasting.models import (
    ModelDefinition,
    PlantHourMeanRegressor,
    model_definition,
)


def test_chronological_folds_keep_timestamps_grouped_and_time_ordered():
    frame = generate_synthetic_data(days=1, plants=3, seed=10)
    frame.index = [f"row_{index}" for index in range(len(frame))]

    fold_pairs = list(chronological_folds(frame, folds=3, minimum_train_fraction=0.5))

    assert len(fold_pairs) == 3
    validation_timestamps = []
    for train_index, validation_index in fold_pairs:
        train = frame.loc[train_index]
        validation = frame.loc[validation_index]

        assert not train.empty
        assert not validation.empty
        assert train["timestamp"].max() < validation["timestamp"].min()
        assert set(train["timestamp"]).isdisjoint(set(validation["timestamp"]))
        for timestamp in validation["timestamp"].unique():
            assert set(frame.index[frame["timestamp"] == timestamp]) <= set(validation_index)
        validation_timestamps.extend(validation["timestamp"].unique())

    assert len(validation_timestamps) == len(set(validation_timestamps))
    expected_validation_timestamps = frame["timestamp"].drop_duplicates().iloc[12:].tolist()
    assert validation_timestamps == expected_validation_timestamps


@pytest.mark.parametrize(
    ("days", "folds", "minimum_train_fraction", "message"),
    [
        (1, 25, 0.5, "insufficient"),
        (1, 0, 0.5, "folds"),
        (1, 2, 0.0, "minimum_train_fraction"),
        (1, 2, 1.0, "minimum_train_fraction"),
    ],
)
def test_chronological_folds_reject_invalid_fold_requests(
    days, folds, minimum_train_fraction, message
):
    frame = generate_synthetic_data(days=days, plants=1, seed=1)

    with pytest.raises(ValueError, match=message):
        list(
            chronological_folds(
                frame, folds=folds, minimum_train_fraction=minimum_train_fraction
            )
        )


def test_compute_metrics_matches_hand_calculated_values():
    metrics = compute_metrics(
        actual=[10.0, 20.0, 40.0],
        prediction=[12.0, 15.0, 37.0],
        capacity_mw=[50.0, 50.0, 100.0],
    )

    assert metrics == {
        "MAE": pytest.approx(10.0 / 3.0),
        "RMSE": pytest.approx(math.sqrt((4.0 + 25.0 + 9.0) / 3.0)),
        "NMAE": pytest.approx(10.0 / 200.0),
    }


@pytest.mark.parametrize(
    ("actual", "prediction", "capacity_mw", "message"),
    [
        ([1.0], [1.0], [0.0], "positive"),
        ([np.nan], [1.0], [1.0], "finite"),
        ([1.0], [np.inf], [1.0], "finite"),
    ],
)
def test_compute_metrics_rejects_invalid_inputs(
    actual, prediction, capacity_mw, message
):
    with pytest.raises(ValueError, match=message):
        compute_metrics(actual, prediction, capacity_mw)


def test_mean_model_evaluation_returns_finite_metrics_and_bounded_predictions():
    frame = generate_synthetic_data(days=4, plants=2, seed=3)

    result = evaluate_model(frame, model_definition("Mean"), feature_specs=[], folds=3)

    assert isinstance(result, EvaluationResult)
    assert set(result.metrics) == {"MAE", "RMSE", "NMAE"}
    assert np.isfinite(list(result.metrics.values())).all()
    assert set(result.per_plant) == set(frame["plant_id"].unique())
    assert len(result.fold_metrics) == 3
    assert np.isfinite(result.predictions["prediction"]).all()
    assert (result.predictions["prediction"] >= 0).all()
    assert (result.predictions["prediction"] <= result.predictions["capacity_mw"]).all()
    assert list(result.predictions.columns) == [
        "timestamp",
        "plant_id",
        "actual",
        "prediction",
        "capacity_mw",
        "fold",
    ]


def test_plant_hour_mean_regressor_uses_plant_hour_and_fallback_means():
    estimator = PlantHourMeanRegressor()
    X = pd.DataFrame(
        {
            "plant_id": ["a", "a", "a", "b"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 01:00",
                    "2024-01-02 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 01:00",
                ]
            ),
        }
    )
    y = np.array([2.0, 4.0, 8.0, 10.0])

    estimator.fit(X, y)
    predictions = estimator.predict(
        pd.DataFrame(
            {
                "plant_id": ["a", "a", "missing"],
                "timestamp": pd.to_datetime(
                    ["2024-01-03 01:00", "2024-01-03 03:00", "2024-01-03 01:00"]
                ),
            }
        )
    )

    np.testing.assert_allclose(predictions, [3.0, 14.0 / 3.0, 6.0])
    assert isinstance(clone(estimator), PlantHourMeanRegressor)


@pytest.mark.parametrize("name", ["Mean", "Weather", "ForecastWeather", "Ldaps", "SPOT"])
def test_all_legacy_model_names_evaluate_successfully(name):
    frame = generate_synthetic_data(days=4, plants=2, seed=11)

    result = evaluate_model(frame, model_definition(name), feature_specs=[], folds=3)

    assert set(result.metrics) == {"MAE", "RMSE", "NMAE"}
    assert np.isfinite(list(result.metrics.values())).all()
    assert len(result.predictions) > 0


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        model_definition("Unknown")


class RecordingRegressor(BaseEstimator, RegressorMixin):
    seen_columns = []

    def fit(self, X, y):
        self.__class__.seen_columns.append(tuple(X.columns))
        self.prediction_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.prediction_)


def test_feature_specs_append_to_base_features_without_mutating_or_duplicating_columns():
    frame = generate_synthetic_data(days=4, plants=2, seed=13)
    original = frame.copy(deep=True)
    RecordingRegressor.seen_columns = []
    definition = ModelDefinition(
        name="Recording",
        base_features=("forecast_irradiance",),
        estimator_factory=RecordingRegressor,
    )
    specs = [
        FeatureSpec(
            "cloud_factor",
            "cloud_attenuation",
            ("forecast_cloud_cover",),
        ),
        FeatureSpec(
            "forecast_irradiance",
            "effective_irradiance",
            ("forecast_irradiance", "forecast_cloud_cover"),
        ),
    ]

    evaluate_model(frame, definition, specs, folds=2)

    assert RecordingRegressor.seen_columns
    assert set(RecordingRegressor.seen_columns) == {
        ("forecast_irradiance", "cloud_factor")
    }
    pd.testing.assert_frame_equal(frame, original)


def test_evaluation_is_deterministic():
    frame = generate_synthetic_data(days=4, plants=2, seed=21)
    definition = model_definition("SPOT")
    specs = [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",))]

    first = evaluate_model(frame, definition, specs, folds=3)
    second = evaluate_model(frame, definition, specs, folds=3)

    assert first.metrics == second.metrics
    assert first.per_plant == second.per_plant
    assert first.fold_metrics == second.fold_metrics
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
