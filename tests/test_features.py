import math

import numpy as np
import pandas as pd
import pytest

from power_forecasting.data import generate_synthetic_data
from power_forecasting.features import FeatureSpec, apply_feature_specs


def test_apply_feature_specs_generates_hour_sin_and_effective_irradiance_in_order():
    frame = generate_synthetic_data(days=1, plants=1, seed=4)
    specs = [
        FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",)),
        FeatureSpec(
            "effective_irradiance",
            "effective_irradiance",
            ("forecast_irradiance", "forecast_cloud_cover"),
        ),
    ]

    features = apply_feature_specs(frame, specs)

    assert list(features.columns) == ["hour_sin", "effective_irradiance"]
    expected_hour = np.sin(2 * math.pi * frame["timestamp"].dt.hour.to_numpy() / 24)
    expected_effective = frame["forecast_irradiance"].to_numpy() * np.clip(
        1 - frame["forecast_cloud_cover"].to_numpy(), 0, 1
    )
    np.testing.assert_allclose(features["hour_sin"].to_numpy(), expected_hour)
    np.testing.assert_allclose(
        features["effective_irradiance"].to_numpy(), expected_effective
    )
    assert np.isfinite(features.to_numpy()).all()


def test_cyclic_hour_cos_computes_expected_values():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 06:00:00"])})

    features = apply_feature_specs(
        frame, [FeatureSpec("hour_cos", "cyclic_hour", ("timestamp",))]
    )

    np.testing.assert_allclose(features["hour_cos"].to_numpy(), [1.0, 0.0], atol=1e-12)


def test_cyclic_day_of_year_sin_cos_compute_expected_values():
    frame = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01", "2024-07-02"])})

    features = apply_feature_specs(
        frame,
        [
            FeatureSpec("doy_sin", "cyclic_day_of_year", ("timestamp",)),
            FeatureSpec("doy_cos", "cyclic_day_of_year", ("timestamp",)),
        ],
    )

    day_of_year = frame["timestamp"].dt.dayofyear.to_numpy(dtype=float)
    np.testing.assert_allclose(
        features["doy_sin"].to_numpy(), np.sin(2 * math.pi * day_of_year / 365.25)
    )
    np.testing.assert_allclose(
        features["doy_cos"].to_numpy(), np.cos(2 * math.pi * day_of_year / 365.25)
    )


def test_temperature_derating_applies_defaults_and_clips():
    frame = pd.DataFrame({"irradiance": [1000.0, 1000.0, 1000.0], "temperature": [20.0, 30.0, 400.0]})

    features = apply_feature_specs(
        frame,
        [FeatureSpec("derated", "temperature_derating", ("irradiance", "temperature"))],
    )

    np.testing.assert_allclose(features["derated"].to_numpy(), [1000.0, 980.0, 0.0])


def test_cloud_attenuation_clips_cloud_cover():
    frame = pd.DataFrame({"cloud_cover": [-0.25, 0.4, 1.25]})

    features = apply_feature_specs(
        frame, [FeatureSpec("cloud_factor", "cloud_attenuation", ("cloud_cover",))]
    )

    np.testing.assert_allclose(features["cloud_factor"].to_numpy(), [1.0, 0.6, 0.0])


def test_interaction_multiplies_two_numeric_inputs():
    frame = pd.DataFrame({"wind": [2.0, 3.5], "cloud": [0.5, 0.25]})

    features = apply_feature_specs(
        frame, [FeatureSpec("wind_cloud", "interaction", ("wind", "cloud"))]
    )

    np.testing.assert_allclose(features["wind_cloud"].to_numpy(), [1.0, 0.875])


def test_ratio_divides_numeric_inputs():
    frame = pd.DataFrame({"numerator": [10.0, -3.0], "denominator": [2.0, 4.0]})

    features = apply_feature_specs(
        frame, [FeatureSpec("irradiance_per_cloud", "ratio", ("numerator", "denominator"))]
    )

    np.testing.assert_allclose(features["irradiance_per_cloud"].to_numpy(), [5.0, -0.75])


def test_unknown_transform_rejected():
    frame = pd.DataFrame({"source": [1.0]})

    with pytest.raises(ValueError, match="unknown transform"):
        apply_feature_specs(frame, [FeatureSpec("feature", "does_not_exist", ("source",))])


def test_numeric_timestamp_input_rejected():
    frame = pd.DataFrame({"timestamp": [0, 1, 2]})

    with pytest.raises(ValueError, match="invalid datetime input"):
        apply_feature_specs(frame, [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",))])


def test_feature_spec_serialization_roundtrips_and_protects_parameters():
    parameters = {"reference": 20.0, "coefficient": 0.005}
    spec = FeatureSpec(
        "derated",
        "temperature_derating",
        ("irradiance", "temperature"),
        parameters,
        version="2",
        rationale="panel heat lowers output",
    )
    parameters["coefficient"] = 0.5

    payload = spec.to_dict()
    restored = FeatureSpec.from_dict(payload)

    assert spec.parameters["coefficient"] == 0.005
    assert list(payload["parameters"]) == ["coefficient", "reference"]
    assert payload == {
        "name": "derated",
        "transform": "temperature_derating",
        "inputs": ["irradiance", "temperature"],
        "parameters": {"coefficient": 0.005, "reference": 20.0},
        "version": "2",
        "rationale": "panel heat lowers output",
    }
    assert restored == spec
    with pytest.raises(TypeError):
        spec.parameters["coefficient"] = 0.2


def test_feature_spec_defensively_copies_nested_parameters():
    parameters = {"labels": ["morning", "evening"], "bounds": {"low": 0.1}}
    spec = FeatureSpec("cloud_factor", "cloud_attenuation", ("cloud_cover",), parameters)
    parameters["labels"].append("night")
    parameters["bounds"]["low"] = 0.9

    assert spec.to_dict()["parameters"] == {
        "bounds": {"low": 0.1},
        "labels": ["morning", "evening"],
    }
    with pytest.raises(AttributeError):
        spec.parameters["labels"].append("night")


def test_apply_feature_specs_does_not_mutate_input_frame():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
            "cloud_cover": [0.2, 0.4],
        }
    )
    original = frame.copy(deep=True)

    apply_feature_specs(
        frame,
        [
            FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",)),
            FeatureSpec("cloud_factor", "cloud_attenuation", ("cloud_cover",)),
        ],
    )

    pd.testing.assert_frame_equal(frame, original)


def test_duplicate_feature_names_rejected():
    frame = pd.DataFrame({"cloud_cover": [0.2]})

    with pytest.raises(ValueError, match="duplicate feature name"):
        apply_feature_specs(
            frame,
            [
                FeatureSpec("cloud_factor", "cloud_attenuation", ("cloud_cover",)),
                FeatureSpec("cloud_factor", "cloud_attenuation", ("cloud_cover",)),
            ],
        )


def test_missing_source_column_rejected():
    frame = pd.DataFrame({"cloud_cover": [0.2]})

    with pytest.raises(ValueError, match="missing source columns"):
        apply_feature_specs(
            frame, [FeatureSpec("effective", "effective_irradiance", ("irradiance", "cloud_cover"))]
        )


def test_target_leakage_generation_mw_input_rejected():
    frame = pd.DataFrame({"generation_mw": [1.0], "cloud_cover": [0.2]})

    with pytest.raises(ValueError, match="target leakage"):
        apply_feature_specs(
            frame, [FeatureSpec("leaky", "interaction", ("generation_mw", "cloud_cover"))]
        )


def test_invalid_output_name_rejected():
    frame = pd.DataFrame({"cloud_cover": [0.2]})

    with pytest.raises(ValueError, match="invalid feature name"):
        apply_feature_specs(
            frame, [FeatureSpec("not valid", "cloud_attenuation", ("cloud_cover",))]
        )


def test_wrong_transform_arity_rejected():
    frame = pd.DataFrame({"irradiance": [100.0]})

    with pytest.raises(ValueError, match="expects 2 inputs"):
        apply_feature_specs(
            frame, [FeatureSpec("effective", "effective_irradiance", ("irradiance",))]
        )


def test_unexpected_transform_parameter_rejected():
    frame = pd.DataFrame({"cloud_cover": [0.2]})

    with pytest.raises(ValueError, match="unexpected parameters"):
        apply_feature_specs(
            frame,
            [
                FeatureSpec(
                    "cloud_factor",
                    "cloud_attenuation",
                    ("cloud_cover",),
                    {"reference": 25.0},
                )
            ],
        )


def test_invalid_ratio_epsilon_parameter_rejected():
    frame = pd.DataFrame({"numerator": [1.0], "denominator": [2.0]})

    with pytest.raises(ValueError, match="epsilon"):
        apply_feature_specs(
            frame,
            [FeatureSpec("bad_ratio", "ratio", ("numerator", "denominator"), {"epsilon": 0.0})],
        )


def test_nonfinite_ratio_rejected_with_feature_name():
    frame = pd.DataFrame({"numerator": [1.0], "denominator": [0.0]})

    with pytest.raises(ValueError, match="bad_ratio.*denominator near zero"):
        apply_feature_specs(
            frame, [FeatureSpec("bad_ratio", "ratio", ("numerator", "denominator"))]
        )
