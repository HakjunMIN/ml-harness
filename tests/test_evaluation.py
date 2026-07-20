import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import power_forecasting.models as model_registry
from power_forecasting.data import DataContractError, generate_synthetic_data
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
        train = frame.iloc[train_index]
        validation = frame.iloc[validation_index]

        assert not train.empty
        assert not validation.empty
        assert train["timestamp"].max() < validation["timestamp"].min()
        assert set(train["timestamp"]).isdisjoint(set(validation["timestamp"]))
        for timestamp in validation["timestamp"].unique():
            expected_positions = set(
                np.flatnonzero(frame["timestamp"].to_numpy() == timestamp)
            )
            assert expected_positions <= set(validation_index)
        validation_timestamps.extend(validation["timestamp"].unique())

    assert len(validation_timestamps) == len(set(validation_timestamps))
    expected_validation_timestamps = frame["timestamp"].drop_duplicates().iloc[12:].tolist()
    assert validation_timestamps == expected_validation_timestamps


def test_chronological_folds_and_evaluation_use_positions_with_duplicate_index_labels():
    frame = generate_synthetic_data(days=2, plants=2, seed=27)
    frame.index = pd.Index(["duplicate"] * len(frame))
    unique_timestamps = pd.Index(sorted(pd.to_datetime(frame["timestamp"]).unique()))
    expected_validation_timestamps = unique_timestamps[
        math.ceil(len(unique_timestamps) * 0.5) :
    ]

    fold_pairs = list(chronological_folds(frame, folds=2, minimum_train_fraction=0.5))

    assert len(fold_pairs) == 2
    validation_timestamps = []
    validation_row_count = 0
    for train_positions, validation_positions in fold_pairs:
        assert np.issubdtype(np.asarray(train_positions).dtype, np.integer)
        assert np.issubdtype(np.asarray(validation_positions).dtype, np.integer)
        train = frame.iloc[train_positions]
        validation = frame.iloc[validation_positions]

        assert not train.empty
        assert not validation.empty
        assert train["timestamp"].max() < validation["timestamp"].min()
        assert set(train["timestamp"]).isdisjoint(set(validation["timestamp"]))
        assert len(validation) == len(validation["timestamp"].unique()) * 2
        validation_timestamps.extend(validation["timestamp"].unique())
        validation_row_count += len(validation)

    assert len(validation_timestamps) == len(set(validation_timestamps))
    assert validation_timestamps == list(expected_validation_timestamps)
    assert validation_row_count == len(expected_validation_timestamps) * 2

    result = evaluate_model(frame, model_definition("Mean"), feature_specs=[], folds=2)

    assert len(result.predictions) == validation_row_count
    expected_keys = frame[
        pd.to_datetime(frame["timestamp"]).isin(expected_validation_timestamps)
    ][["timestamp", "plant_id", "generation_mw"]].rename(
        columns={"generation_mw": "actual"}
    )
    actual_keys = result.predictions[["timestamp", "plant_id", "actual"]]
    pd.testing.assert_frame_equal(
        actual_keys.reset_index(drop=True),
        expected_keys.reset_index(drop=True),
    )


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


def test_mean_model_evaluation_accepts_mixed_timestamp_formats():
    frame = generate_synthetic_data(days=4, plants=2, seed=29)
    formatters = (
        lambda timestamp: timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        lambda timestamp: timestamp.strftime("%m/%d/%Y %H:%M"),
        lambda timestamp: timestamp.isoformat(),
    )
    frame["timestamp"] = [
        formatters[index % len(formatters)](timestamp)
        for index, timestamp in enumerate(frame["timestamp"])
    ]

    result = evaluate_model(frame, model_definition("Mean"), feature_specs=[], folds=3)

    assert len(result.predictions) > 0
    assert np.isfinite(result.predictions["prediction"]).all()


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


def test_plant_hour_mean_regressor_is_recognized_by_sklearn_as_regressor():
    assert is_regressor(PlantHourMeanRegressor())


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


def test_supported_model_names_base_features_and_data_availability_are_exact():
    expected_base_features = {
        "Mean": ("plant_id", "timestamp"),
        "Weather": (
            "actual_irradiance",
            "actual_temperature",
            "actual_cloud_cover",
            "actual_wind_speed",
            "capacity_mw",
        ),
        "ForecastWeather": (
            "forecast_irradiance",
            "forecast_temperature",
            "forecast_cloud_cover",
            "forecast_wind_speed",
            "capacity_mw",
        ),
        "Ldaps": (
            "ldaps_irradiance",
            "ldaps_temperature",
            "ldaps_cloud_cover",
            "ldaps_humidity",
            "capacity_mw",
        ),
        "SPOT": (
            "forecast_irradiance",
            "forecast_temperature",
            "forecast_cloud_cover",
            "forecast_wind_speed",
            "capacity_mw",
            "latitude",
            "longitude",
        ),
    }
    expected_data_availability = {
        "Mean": "historical",
        "Weather": "actual",
        "ForecastWeather": "forecast",
        "Ldaps": "forecast",
        "SPOT": "forecast",
    }

    assert model_registry.SUPPORTED_MODEL_NAMES == tuple(expected_base_features)
    for name in model_registry.SUPPORTED_MODEL_NAMES:
        definition = model_definition(name)

        assert definition.name == name
        assert definition.base_features == expected_base_features[name]
        assert definition.data_availability == expected_data_availability[name]

    assert model_definition("Weather").data_availability == "actual"
    assert model_definition("Mean").data_availability != "forecast"


@pytest.mark.parametrize("name", ["Weather", "ForecastWeather", "Ldaps"])
def test_ridge_models_use_median_imputation_scaling_and_ridge_regression(name):
    estimator = model_definition(name).estimator_factory()

    assert isinstance(estimator, Pipeline)
    assert list(estimator.named_steps) == ["simpleimputer", "standardscaler", "ridge"]
    imputer = estimator.named_steps["simpleimputer"]
    scaler = estimator.named_steps["standardscaler"]
    ridge = estimator.named_steps["ridge"]
    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"
    assert isinstance(scaler, StandardScaler)
    assert isinstance(ridge, Ridge)
    assert ridge.alpha == pytest.approx(1.0)


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


class StateIsolationRegressor(BaseEstimator, RegressorMixin):
    fit_instances = []

    def __init__(self, prediction=1.0):
        self.prediction = prediction

    def fit(self, X, y):
        if hasattr(self, "fit_seen_"):
            raise AssertionError("shared estimator instance was reused across folds")
        self.fit_seen_ = True
        self.__class__.fit_instances.append(self)
        return self

    def predict(self, X):
        return np.full(len(X), self.prediction)


def test_evaluate_model_clones_factory_estimator_for_each_fold():
    frame = generate_synthetic_data(days=4, plants=2, seed=17)
    shared_estimator = StateIsolationRegressor(prediction=2.5)
    StateIsolationRegressor.fit_instances = []
    definition = ModelDefinition(
        name="SharedState",
        base_features=("capacity_mw",),
        estimator_factory=lambda: shared_estimator,
    )

    evaluate_model(frame, definition, feature_specs=[], folds=3)

    assert not hasattr(shared_estimator, "fit_seen_")
    assert len(StateIsolationRegressor.fit_instances) == 3
    assert len({id(instance) for instance in StateIsolationRegressor.fit_instances}) == 3


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate keys",
        ),
        (
            lambda frame: frame.assign(
                generation_mw=lambda data: data["capacity_mw"] + 1.0
            ),
            "target outside",
        ),
        (
            lambda frame: frame.assign(forecast_irradiance=np.nan),
            "non-finite numeric",
        ),
    ],
)
def test_evaluate_model_rejects_invalid_or_duplicate_dataset_rows(mutate, message):
    frame = generate_synthetic_data(days=2, plants=1, seed=19)

    with pytest.raises(DataContractError, match=message):
        evaluate_model(mutate(frame), model_definition("Mean"), feature_specs=[], folds=1)


class OutOfBoundsCapacityRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        capacity = X["capacity_mw"].to_numpy(dtype=float)
        predictions = capacity + 5.0
        predictions[::2] = -5.0
        return predictions


def test_evaluate_model_stores_and_scores_clipped_predictions():
    frame = generate_synthetic_data(days=2, plants=2, seed=23)
    definition = ModelDefinition(
        name="OutOfBounds",
        base_features=("capacity_mw",),
        estimator_factory=OutOfBoundsCapacityRegressor,
    )

    result = evaluate_model(frame, definition, feature_specs=[], folds=2)
    expected_predictions = result.predictions.copy()
    clipped_parts = []
    for _, fold in result.predictions.groupby("fold", sort=True):
        capacity = fold["capacity_mw"].to_numpy(dtype=float)
        clipped = capacity.copy()
        clipped[::2] = 0.0
        clipped_parts.append(pd.Series(clipped, index=fold.index))
    expected_predictions["prediction"] = pd.concat(clipped_parts).sort_index()

    pd.testing.assert_series_equal(
        result.predictions["prediction"],
        expected_predictions["prediction"],
        check_names=False,
    )
    assert result.metrics == pytest.approx(
        compute_metrics(
            expected_predictions["actual"],
            expected_predictions["prediction"],
            expected_predictions["capacity_mw"],
        )
    )
    for fold_number, fold in expected_predictions.groupby("fold", sort=True):
        assert result.fold_metrics[fold_number - 1] == pytest.approx(
            compute_metrics(fold["actual"], fold["prediction"], fold["capacity_mw"])
        )
    for plant_id, group in expected_predictions.groupby("plant_id", sort=True):
        assert result.per_plant[str(plant_id)] == pytest.approx(
            compute_metrics(group["actual"], group["prediction"], group["capacity_mw"])
        )


def test_evaluation_is_deterministic():
    frame = generate_synthetic_data(days=4, plants=2, seed=21)
    definition = model_definition("SPOT")
    specs = [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",))]
    estimator = definition.estimator_factory()

    assert isinstance(estimator, Pipeline)
    assert list(estimator.named_steps) == ["simpleimputer", "histgradientboostingregressor"]
    imputer = estimator.named_steps["simpleimputer"]
    regressor = estimator.named_steps["histgradientboostingregressor"]
    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"
    assert isinstance(regressor, HistGradientBoostingRegressor)
    assert regressor.random_state == 0
    assert regressor.max_iter == 100

    first = evaluate_model(frame, definition, specs, folds=3)
    second = evaluate_model(frame, definition, specs, folds=3)

    assert first.metrics == second.metrics
    assert first.per_plant == second.per_plant
    assert first.fold_metrics == second.fold_metrics
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
