from __future__ import annotations

import math

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "plant_id",
    "timestamp",
    "capacity_mw",
    "latitude",
    "longitude",
    "actual_irradiance",
    "actual_temperature",
    "actual_cloud_cover",
    "actual_wind_speed",
    "forecast_irradiance",
    "forecast_temperature",
    "forecast_cloud_cover",
    "forecast_wind_speed",
    "ldaps_irradiance",
    "ldaps_temperature",
    "ldaps_cloud_cover",
    "ldaps_humidity",
    "generation_mw",
)


NUMERIC_COLUMNS = (
    "capacity_mw",
    "latitude",
    "longitude",
    "actual_irradiance",
    "actual_temperature",
    "actual_cloud_cover",
    "actual_wind_speed",
    "forecast_irradiance",
    "forecast_temperature",
    "forecast_cloud_cover",
    "forecast_wind_speed",
    "ldaps_irradiance",
    "ldaps_temperature",
    "ldaps_cloud_cover",
    "ldaps_humidity",
    "generation_mw",
)


class DataContractError(ValueError):
    """Raised when a data frame violates the forecasting data contract."""


def generate_synthetic_data(days: int, plants: int, seed: int) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be positive")
    if plants <= 0:
        raise ValueError("plants must be positive")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    records = []

    for timestamp in timestamps:
        hour = timestamp.hour
        day_of_year = timestamp.dayofyear
        daylight = max(0.0, math.sin(math.pi * (hour - 6) / 12))
        seasonal = 0.78 + 0.22 * math.cos(2 * math.pi * (day_of_year - 172) / 365)

        for plant_index in range(plants):
            plant_factor = 0.93 + 0.05 * math.sin(plant_index * 1.7)
            capacity_mw = 45.0 + plant_index * 12.5 + (plant_index % 3) * 4.0
            latitude = 32.0 + (plant_index * 3.7) % 17.0
            longitude = -124.0 + (plant_index * 6.3) % 58.0

            daily_cloud_cycle = 0.11 * math.sin(2 * math.pi * (hour + plant_index) / 24)
            synoptic_cloud_cycle = 0.12 * math.sin(
                2 * math.pi * (day_of_year + plant_index * 3) / 9
            )
            actual_cloud_cover = float(
                np.clip(
                    0.38 + daily_cloud_cycle + synoptic_cloud_cycle + rng.normal(0, 0.08),
                    0.0,
                    1.0,
                )
            )

            daily_temperature = 8.0 * math.sin(2 * math.pi * (hour - 8) / 24)
            actual_temperature = (
                22.0
                + 10.0 * seasonal
                + daily_temperature
                - 0.18 * (latitude - 35.0)
                + rng.normal(0, 1.2)
            )
            actual_wind_speed = float(
                np.clip(
                    4.2
                    + 1.4 * math.sin(2 * math.pi * (hour + 4) / 24)
                    + rng.normal(0, 0.8),
                    0.0,
                    None,
                )
            )

            clear_sky_irradiance = 1010.0 * daylight * seasonal
            actual_irradiance = float(
                np.clip(
                    clear_sky_irradiance
                    * (1.0 - 0.72 * actual_cloud_cover)
                    + rng.normal(0, 18.0 * daylight),
                    0.0,
                    None,
                )
            )

            forecast_cloud_cover = float(
                np.clip(actual_cloud_cover + rng.normal(0, 0.11) + 0.02, 0.0, 1.0)
            )
            forecast_temperature = float(actual_temperature + rng.normal(0.3, 2.0))
            forecast_wind_speed = float(
                np.clip(actual_wind_speed + rng.normal(0, 1.1), 0.0, None)
            )
            forecast_irradiance = float(
                np.clip(
                    clear_sky_irradiance
                    * (1.0 - 0.67 * forecast_cloud_cover)
                    + rng.normal(0, 45.0 * daylight),
                    0.0,
                    None,
                )
            )

            ldaps_cloud_cover = float(
                np.clip(actual_cloud_cover + rng.normal(0, 0.07), 0.0, 1.0)
            )
            ldaps_temperature = float(actual_temperature + rng.normal(0, 1.1))
            ldaps_irradiance = float(
                np.clip(actual_irradiance + rng.normal(0, 28.0 * daylight), 0.0, None)
            )
            ldaps_humidity = float(
                np.clip(
                    58.0
                    + 28.0 * actual_cloud_cover
                    - 0.45 * (actual_temperature - 25.0)
                    + rng.normal(0, 4.0),
                    5.0,
                    100.0,
                )
            )

            temperature_derating = min(
                1.04, max(0.72, 1.0 - 0.0045 * max(0.0, actual_temperature - 25.0))
            )
            generation = (
                capacity_mw
                * (actual_irradiance / 1000.0)
                * plant_factor
                * temperature_derating
                * (1.0 - 0.03 * actual_cloud_cover)
            )
            generation_mw = float(np.clip(generation, 0.0, capacity_mw))

            records.append(
                {
                    "plant_id": f"plant_{plant_index + 1:02d}",
                    "timestamp": timestamp,
                    "capacity_mw": capacity_mw,
                    "latitude": latitude,
                    "longitude": longitude,
                    "actual_irradiance": actual_irradiance,
                    "actual_temperature": actual_temperature,
                    "actual_cloud_cover": actual_cloud_cover,
                    "actual_wind_speed": actual_wind_speed,
                    "forecast_irradiance": forecast_irradiance,
                    "forecast_temperature": forecast_temperature,
                    "forecast_cloud_cover": forecast_cloud_cover,
                    "forecast_wind_speed": forecast_wind_speed,
                    "ldaps_irradiance": ldaps_irradiance,
                    "ldaps_temperature": ldaps_temperature,
                    "ldaps_cloud_cover": ldaps_cloud_cover,
                    "ldaps_humidity": ldaps_humidity,
                    "generation_mw": generation_mw,
                }
            )

    frame = pd.DataFrame.from_records(records, columns=REQUIRED_COLUMNS)
    return frame.sort_values(["timestamp", "plant_id"]).reset_index(drop=True)


def validate_dataset(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DataContractError(f"missing required columns: {missing_columns}")

    timestamp_format = None
    if not pd.api.types.is_datetime64_any_dtype(frame["timestamp"]):
        timestamp_format = "mixed"
    parsed_timestamps = pd.to_datetime(
        frame["timestamp"], errors="coerce", format=timestamp_format
    )
    if parsed_timestamps.isna().any():
        raise DataContractError("invalid timestamps: unparseable values")

    keys = pd.DataFrame(
        {"plant_id": frame["plant_id"].to_numpy(), "timestamp": parsed_timestamps.to_numpy()}
    )
    if keys.duplicated().any():
        raise DataContractError("duplicate keys: plant_id,timestamp")

    numeric_values = {}
    non_finite_columns = []
    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        numeric_values[column] = values
        if not np.isfinite(values).all():
            non_finite_columns.append(column)

    if non_finite_columns:
        raise DataContractError(f"non-finite numeric values: {non_finite_columns}")

    capacity = numeric_values["capacity_mw"]
    generation = numeric_values["generation_mw"]
    if (capacity <= 0).any():
        raise DataContractError("non-positive capacity: capacity_mw must be > 0")

    if ((generation < 0) | (generation > capacity)).any():
        raise DataContractError("target outside [0, capacity_mw]: generation_mw")
