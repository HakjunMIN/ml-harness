import numpy as np
import pandas as pd
import pytest

from power_forecasting.data import (
    DataContractError,
    REQUIRED_COLUMNS,
    generate_synthetic_data,
    validate_dataset,
)


def test_generate_synthetic_data_is_deterministic_and_valid():
    first = generate_synthetic_data(days=14, plants=2, seed=7)
    second = generate_synthetic_data(days=14, plants=2, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert set(REQUIRED_COLUMNS).issubset(first.columns)
    assert len(first) == 14 * 24 * 2
    validate_dataset(first)


def test_generate_synthetic_data_bounds_generation_by_capacity():
    frame = generate_synthetic_data(days=7, plants=3, seed=9)

    assert (frame["generation_mw"] >= 0).all()
    assert (frame["generation_mw"] <= frame["capacity_mw"]).all()


@pytest.mark.parametrize(
    ("days", "plants", "message"),
    [
        (0, 1, "days must be positive"),
        (1, 0, "plants must be positive"),
    ],
)
def test_generate_synthetic_data_rejects_non_positive_dimensions(
    days, plants, message
):
    with pytest.raises(ValueError, match=message):
        generate_synthetic_data(days=days, plants=plants, seed=1)


@pytest.fixture
def valid_frame():
    return generate_synthetic_data(days=2, plants=2, seed=3)


def test_validate_dataset_rejects_missing_required_columns(valid_frame):
    frame = valid_frame.drop(columns=["actual_irradiance"])

    with pytest.raises(DataContractError, match="missing required columns"):
        validate_dataset(frame)


def test_validate_dataset_rejects_duplicate_plant_timestamp_keys(valid_frame):
    frame = pd.concat([valid_frame, valid_frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataContractError, match="duplicate keys"):
        validate_dataset(frame)


def test_validate_dataset_rejects_invalid_timestamps(valid_frame):
    frame = valid_frame.copy()
    frame["timestamp"] = frame["timestamp"].astype(object)
    frame.loc[0, "timestamp"] = "not-a-timestamp"

    with pytest.raises(DataContractError, match="invalid timestamps"):
        validate_dataset(frame)


def test_validate_dataset_rejects_non_finite_required_numeric_values(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, "forecast_temperature"] = np.inf

    with pytest.raises(DataContractError, match="non-finite numeric values"):
        validate_dataset(frame)


def test_validate_dataset_rejects_non_positive_capacity(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, "capacity_mw"] = 0.0

    with pytest.raises(DataContractError, match="non-positive capacity"):
        validate_dataset(frame)


def test_validate_dataset_rejects_generation_outside_capacity_bounds(valid_frame):
    frame = valid_frame.copy()
    frame.loc[0, "generation_mw"] = frame.loc[0, "capacity_mw"] + 0.1

    with pytest.raises(DataContractError, match=r"target outside \[0, capacity_mw\]"):
        validate_dataset(frame)
