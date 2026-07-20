from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    base_features: tuple[str, ...]
    estimator_factory: Callable[[], Any]
    data_availability: str = "forecast"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model name must be a non-empty string")
        object.__setattr__(self, "base_features", tuple(self.base_features))
        if not all(isinstance(column, str) and column for column in self.base_features):
            raise ValueError(f"model {self.name}: base_features must be non-empty strings")
        if not callable(self.estimator_factory):
            raise TypeError(f"model {self.name}: estimator_factory must be callable")
        if self.data_availability not in {"actual", "forecast", "historical"}:
            raise ValueError(
                f"model {self.name}: data_availability must be actual, forecast, or historical"
            )


class PlantHourMeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y: Any) -> "PlantHourMeanRegressor":
        features = self._feature_frame(X)
        target = np.asarray(y, dtype=float)
        if target.ndim != 1 or len(target) != len(features):
            raise ValueError("y must be one-dimensional and match X length")
        if len(target) == 0 or not np.isfinite(target).all():
            raise ValueError("y must contain finite values")

        training = pd.DataFrame(
            {
                "plant_id": features["plant_id"].astype(str).to_numpy(),
                "hour": self._hours(features),
                "target": target,
            }
        )
        self.plant_hour_mean_ = (
            training.groupby(["plant_id", "hour"], sort=False)["target"].mean().to_dict()
        )
        self.plant_mean_ = training.groupby("plant_id", sort=False)["target"].mean().to_dict()
        self.global_mean_ = float(training["target"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, ["plant_hour_mean_", "plant_mean_", "global_mean_"])
        features = self._feature_frame(X)
        plants = features["plant_id"].astype(str).to_numpy()
        hours = self._hours(features)
        return np.asarray(
            [
                self.plant_hour_mean_.get(
                    (plant_id, hour),
                    self.plant_mean_.get(plant_id, self.global_mean_),
                )
                for plant_id, hour in zip(plants, hours)
            ],
            dtype=float,
        )

    @staticmethod
    def _feature_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("PlantHourMeanRegressor requires a pandas DataFrame")
        missing = [column for column in ("plant_id", "timestamp") if column not in X.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        return X.loc[:, ["plant_id", "timestamp"]]

    @staticmethod
    def _hours(features: pd.DataFrame) -> np.ndarray:
        timestamps = pd.to_datetime(features["timestamp"], errors="coerce")
        if timestamps.isna().any():
            raise ValueError("timestamp contains unparseable values")
        return timestamps.dt.hour.to_numpy(dtype=int)


def _ridge_pipeline() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0),
    )


def _spot_pipeline() -> Any:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(random_state=0, max_iter=100),
    )


WEATHER_ORACLE_LEGACY_FEATURES = (
    "actual_irradiance",
    "actual_temperature",
    "actual_cloud_cover",
    "actual_wind_speed",
    "capacity_mw",
)

FORECAST_WEATHER_FEATURES = (
    "forecast_irradiance",
    "forecast_temperature",
    "forecast_cloud_cover",
    "forecast_wind_speed",
    "capacity_mw",
)

LDAPS_FEATURES = (
    "ldaps_irradiance",
    "ldaps_temperature",
    "ldaps_cloud_cover",
    "ldaps_humidity",
    "capacity_mw",
)

SPOT_FEATURES = (
    "forecast_irradiance",
    "forecast_temperature",
    "forecast_cloud_cover",
    "forecast_wind_speed",
    "capacity_mw",
    "latitude",
    "longitude",
)


SUPPORTED_MODEL_NAMES = ("Mean", "Weather", "ForecastWeather", "Ldaps", "SPOT")


_MODEL_DEFINITIONS = {
    "Mean": ModelDefinition(
        "Mean",
        ("plant_id", "timestamp"),
        PlantHourMeanRegressor,
        data_availability="historical",
    ),
    # Diagnostic/oracle legacy only: actual weather is unavailable at forecast time.
    "Weather": ModelDefinition(
        "Weather",
        WEATHER_ORACLE_LEGACY_FEATURES,
        _ridge_pipeline,
        data_availability="actual",
    ),
    "ForecastWeather": ModelDefinition(
        "ForecastWeather",
        FORECAST_WEATHER_FEATURES,
        _ridge_pipeline,
        data_availability="forecast",
    ),
    "Ldaps": ModelDefinition(
        "Ldaps",
        LDAPS_FEATURES,
        _ridge_pipeline,
        data_availability="forecast",
    ),
    "SPOT": ModelDefinition(
        "SPOT",
        SPOT_FEATURES,
        _spot_pipeline,
        data_availability="forecast",
    ),
}


def model_definition(name: str) -> ModelDefinition:
    try:
        return _MODEL_DEFINITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model: {name}") from exc


__all__ = [
    "FORECAST_WEATHER_FEATURES",
    "LDAPS_FEATURES",
    "ModelDefinition",
    "PlantHourMeanRegressor",
    "SPOT_FEATURES",
    "SUPPORTED_MODEL_NAMES",
    "WEATHER_ORACLE_LEGACY_FEATURES",
    "model_definition",
]
