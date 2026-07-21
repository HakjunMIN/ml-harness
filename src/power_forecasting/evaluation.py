from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from power_forecasting.data import parse_timestamps, validate_dataset
from power_forecasting.features import FeatureSpec, apply_feature_specs
from power_forecasting.models import ModelDefinition


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    per_plant: dict[str, dict[str, float]]
    fold_metrics: list[dict[str, float]]
    predictions: pd.DataFrame


def chronological_folds(
    frame: pd.DataFrame, folds: int = 3, minimum_train_fraction: float = 0.5
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if not isinstance(folds, int) or folds < 1:
        raise ValueError("folds must be a positive integer")
    if not 0.0 < minimum_train_fraction < 1.0:
        raise ValueError("minimum_train_fraction must be between 0 and 1")
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp column is required")

    timestamps = _timestamp_series(frame).to_numpy()
    unique_timestamps = pd.Index(sorted(pd.unique(timestamps)))
    initial_train_count = math.ceil(len(unique_timestamps) * minimum_train_fraction)
    validation_timestamps = unique_timestamps[initial_train_count:]
    if initial_train_count < 1 or len(validation_timestamps) < folds:
        raise ValueError("insufficient timestamps for requested folds")

    for validation_block in np.array_split(validation_timestamps, folds):
        if len(validation_block) == 0:
            raise ValueError("insufficient timestamps for requested folds")
        validation_values = np.asarray(validation_block)
        validation_start = validation_values[0]
        train_mask = timestamps < validation_start
        validation_mask = np.isin(timestamps, validation_values)
        yield np.flatnonzero(train_mask), np.flatnonzero(validation_mask)


def compute_metrics(
    actual: Sequence[float],
    prediction: Sequence[float],
    capacity_mw: Sequence[float],
) -> dict[str, float]:
    actual_values = _finite_1d("actual", actual)
    prediction_values = _finite_1d("prediction", prediction)
    capacity_values = _finite_1d("capacity_mw", capacity_mw)
    if not (
        len(actual_values) == len(prediction_values) == len(capacity_values)
    ):
        raise ValueError("metric inputs must have the same length")

    denominator = float(np.sum(capacity_values))
    if denominator <= 0:
        raise ValueError("NMAE denominator must be positive")

    error = prediction_values - actual_values
    absolute_error = np.abs(error)
    total_sum_squares = float(np.sum(np.square(actual_values - np.mean(actual_values))))
    residual_sum_squares = float(np.sum(np.square(error)))
    r2 = 0.0 if total_sum_squares == 0.0 else 1.0 - residual_sum_squares / total_sum_squares
    return {
        "MAE": float(np.mean(absolute_error)),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "NMAE": float(np.sum(absolute_error) / denominator),
        "R2": float(r2),
    }


def evaluate_model(
    frame: pd.DataFrame,
    definition: ModelDefinition,
    feature_specs: Sequence[FeatureSpec],
    folds: int = 3,
) -> EvaluationResult:
    validate_dataset(frame)
    specs = list(feature_specs)
    fold_predictions = []
    fold_metrics = []

    for fold_number, (train_index, validation_index) in enumerate(
        chronological_folds(frame, folds=folds), start=1
    ):
        train = frame.iloc[train_index]
        validation = frame.iloc[validation_index]
        x_train = _feature_matrix(train, definition.base_features, specs)
        x_validation = _feature_matrix(validation, definition.base_features, specs)
        y_train = train["generation_mw"].to_numpy(dtype=float)

        estimator = clone(definition.estimator_factory())
        estimator.fit(x_train, y_train)

        raw_predictions = np.asarray(estimator.predict(x_validation), dtype=float)
        if raw_predictions.ndim != 1 or len(raw_predictions) != len(validation):
            raise ValueError("estimator predictions must be one-dimensional")
        if not np.isfinite(raw_predictions).all():
            raise ValueError("estimator predictions must be finite")

        capacity = validation["capacity_mw"].to_numpy(dtype=float)
        actual = validation["generation_mw"].to_numpy(dtype=float)
        clipped_predictions = np.clip(raw_predictions, 0.0, capacity)
        fold_metrics.append(compute_metrics(actual, clipped_predictions, capacity))
        fold_predictions.append(
            pd.DataFrame(
                {
                    "timestamp": validation["timestamp"].to_numpy(),
                    "plant_id": validation["plant_id"].to_numpy(),
                    "actual": actual,
                    "prediction": clipped_predictions,
                    "capacity_mw": capacity,
                    "fold": fold_number,
                }
            )
        )

    predictions = pd.concat(fold_predictions, ignore_index=True)
    metrics = compute_metrics(
        predictions["actual"], predictions["prediction"], predictions["capacity_mw"]
    )
    per_plant = {
        str(plant_id): compute_metrics(
            group["actual"], group["prediction"], group["capacity_mw"]
        )
        for plant_id, group in predictions.groupby("plant_id", sort=True)
    }
    return EvaluationResult(metrics, per_plant, fold_metrics, predictions)


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    return parse_timestamps(frame["timestamp"])


def _feature_matrix(
    frame: pd.DataFrame,
    base_features: Sequence[str],
    feature_specs: Sequence[FeatureSpec],
) -> pd.DataFrame:
    base_columns = _unique_columns(base_features)
    missing = [column for column in base_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing base feature columns: {missing}")

    features = frame.loc[:, base_columns].copy()
    engineered = apply_feature_specs(frame, list(feature_specs))
    for column in engineered.columns:
        if column not in features.columns:
            features[column] = engineered[column].to_numpy()
    return features


def _unique_columns(columns: Sequence[str]) -> list[str]:
    unique = []
    seen = set()
    for column in columns:
        if column not in seen:
            unique.append(column)
            seen.add(column)
    return unique


def _finite_1d(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    return array


__all__ = [
    "EvaluationResult",
    "chronological_folds",
    "compute_metrics",
    "evaluate_model",
]
