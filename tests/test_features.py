import json
import math

import numpy as np
import pandas as pd
import pytest

from power_forecasting.data import generate_synthetic_data
from power_forecasting.features import FeatureSpec, _history_values, apply_feature_specs


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


def test_cyclic_hour_accepts_mixed_parseable_timestamp_formats():
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 00:00:00",
                "01/01/2024 06:00",
                "2024-01-01T18:00:00",
            ]
        }
    )

    features = apply_feature_specs(
        frame, [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",))]
    )

    np.testing.assert_allclose(
        features["hour_sin"].to_numpy(), [0.0, 1.0, -1.0], atol=1e-12
    )


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


def test_lag_rejects_same_timestamp_peers_instead_of_leaking_future_or_other_plants():
    frame = pd.DataFrame(
        {
            "plant_id": ["b", "a", "a", "b", "a", "b", "a", "b"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 01:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 03:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 03:00",
                    "2024-01-01 01:00",
                ]
            ),
            "forecast_irradiance": [10.0, 100.0, 200.0, 30.0, 999.0, 20.0, 300.0, 11.0],
        },
        index=list("hgfedcba"),
    )

    with pytest.raises(ValueError, match="prior_irradiance.*insufficient history"):
        apply_feature_specs(
            frame,
            [
                FeatureSpec(
                    "prior_irradiance",
                    "lag",
                    ("forecast_irradiance",),
                    {"periods": 1},
                )
            ],
        )


def test_history_values_are_strict_prior_per_plant_stably_sorted_and_index_aligned():
    frame = pd.DataFrame(
        {
            "plant_id": ["b", "a", "a", "b", "a", "b", "a", "b"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 01:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 03:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 03:00",
                    "2024-01-01 01:00",
                ]
            ),
            "forecast_irradiance": [10.0, 100.0, 200.0, 30.0, 999.0, 20.0, 300.0, 11.0],
        },
        index=list("hgfedcba"),
    )

    lag_values, lag_insufficient = _history_values(
        frame,
        FeatureSpec("prior_irradiance", "lag", ("forecast_irradiance",), {"periods": 1}),
        window=1,
        reducer=lambda values: values[-1],
    )
    mean_values, mean_insufficient = _history_values(
        frame,
        FeatureSpec("prior_mean", "rolling_mean", ("forecast_irradiance",), {"window": 3}),
        window=3,
        reducer=lambda values: float(np.mean(values)),
        allow_partial=True,
    )

    np.testing.assert_allclose(
        lag_values,
        [np.nan, np.nan, 999.0, 20.0, np.nan, 11.0, 200.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        mean_values,
        [np.nan, np.nan, 999.0, 15.5, np.nan, 11.0, 599.5, np.nan],
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        lag_insufficient,
        [True, True, False, False, True, False, False, True],
    )
    np.testing.assert_array_equal(
        mean_insufficient,
        [True, True, True, True, True, True, True, True],
    )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (FeatureSpec("bad_lag", "lag", ("forecast_irradiance",), {}), "missing parameter periods"),
        (
            FeatureSpec("bad_lag", "lag", ("forecast_irradiance",), {"periods": True}),
            "periods must be an integer",
        ),
        (
            FeatureSpec("bad_lag", "lag", ("forecast_irradiance",), {"periods": 4}),
            "periods outside allowed set",
        ),
        (
            FeatureSpec("bad_lag", "lag", ("forecast_irradiance",), {"periods": 1, "window": 3}),
            "unexpected parameters",
        ),
        (
            FeatureSpec("bad_mean", "rolling_mean", ("forecast_irradiance",), {}),
            "missing parameter window",
        ),
        (
            FeatureSpec("bad_mean", "rolling_mean", ("forecast_irradiance",), {"window": False}),
            "window must be an integer",
        ),
        (
            FeatureSpec("bad_mean", "rolling_mean", ("forecast_irradiance",), {"window": 2}),
            "window outside allowed set",
        ),
    ],
)
def test_history_transform_parameters_are_exact_and_bounded(spec, message):
    frame = pd.DataFrame(
        {
            "plant_id": ["plant_1"],
            "timestamp": pd.to_datetime(["2024-01-01 00:00"]),
            "forecast_irradiance": [1.0],
        }
    )

    with pytest.raises(ValueError, match=message):
        apply_feature_specs(frame, [spec])


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2024-01-01 00:00"]),
                    "forecast_irradiance": [1.0],
                }
            ),
            "plant_id column is required",
        ),
        (
            pd.DataFrame({"plant_id": ["plant_1"], "forecast_irradiance": [1.0]}),
            "timestamp column is required",
        ),
        (
            pd.DataFrame(
                {
                    "plant_id": ["plant_1", "plant_1"],
                    "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"]),
                    "forecast_irradiance": [np.nan, 2.0],
                }
            ),
            "non-finite numeric input forecast_irradiance",
        ),
    ],
)
def test_history_transforms_require_keys_and_finite_sources(frame, message):
    with pytest.raises(ValueError, match=message):
        apply_feature_specs(
            frame,
            [FeatureSpec("prior_irradiance", "lag", ("forecast_irradiance",), {"periods": 1})],
        )


def test_unknown_transform_rejected():
    frame = pd.DataFrame({"source": [1.0]})

    with pytest.raises(ValueError, match="unknown transform"):
        apply_feature_specs(frame, [FeatureSpec("feature", "does_not_exist", ("source",))])


def test_invalid_timestamp_input_rejected_with_feature_context():
    frame = pd.DataFrame({"timestamp": ["2024-01-01 00:00:00", "not-a-timestamp"]})

    with pytest.raises(
        ValueError, match="feature hour_sin: invalid datetime input timestamp"
    ):
        apply_feature_specs(frame, [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",))])


@pytest.mark.parametrize(
    "cloud_cover",
    [
        [0.2, np.nan],
        [0.2, np.inf],
        [0.2, "opaque"],
    ],
)
def test_invalid_numeric_source_inputs_rejected_with_feature_context(cloud_cover):
    frame = pd.DataFrame({"cloud_cover": cloud_cover})

    with pytest.raises(
        ValueError,
        match="feature cloud_factor: non-finite numeric input cloud_cover",
    ):
        apply_feature_specs(
            frame, [FeatureSpec("cloud_factor", "cloud_attenuation", ("cloud_cover",))]
        )


@pytest.mark.parametrize(
    ("feature_name", "transform", "inputs", "parameters", "message"),
    [
        (
            "bad_ratio",
            "ratio",
            ("numerator", "denominator"),
            {"epsilon": "wide"},
            "feature bad_ratio: parameter epsilon must be numeric",
        ),
        (
            "bad_ratio",
            "ratio",
            ("numerator", "denominator"),
            {"epsilon": math.inf},
            "feature bad_ratio: parameter epsilon must be finite",
        ),
        (
            "bad_derated",
            "temperature_derating",
            ("irradiance", "temperature"),
            {"coefficient": np.nan},
            "feature bad_derated: parameter coefficient must be finite",
        ),
    ],
)
def test_non_numeric_and_non_finite_parameters_rejected(
    feature_name, transform, inputs, parameters, message
):
    frame = pd.DataFrame(
        {
            "numerator": [1.0],
            "denominator": [2.0],
            "irradiance": [1000.0],
            "temperature": [25.0],
        }
    )

    with pytest.raises(ValueError, match=message):
        apply_feature_specs(
            frame, [FeatureSpec(feature_name, transform, inputs, parameters)]
        )


def test_negative_temperature_derating_coefficient_rejected():
    frame = pd.DataFrame({"irradiance": [1000.0], "temperature": [30.0]})

    with pytest.raises(
        ValueError, match="feature derated: coefficient must be non-negative"
    ):
        apply_feature_specs(
            frame,
            [
                FeatureSpec(
                    "derated",
                    "temperature_derating",
                    ("irradiance", "temperature"),
                    {"coefficient": -0.001},
                )
            ],
        )


def test_feature_spec_json_roundtrip_and_sorted_serialization_text():
    spec = FeatureSpec(
        "derated",
        "temperature_derating",
        ("irradiance", "temperature"),
        {"reference": 20.0, "coefficient": np.float64(0.005)},
        version="2",
        rationale="panel heat lowers output",
    )

    serialized = json.dumps(spec.to_dict(), sort_keys=True)
    restored = FeatureSpec.from_dict(json.loads(serialized))

    assert serialized == (
        '{"inputs": ["irradiance", "temperature"], "name": "derated", '
        '"parameters": {"coefficient": 0.005, "reference": 20.0}, '
        '"rationale": "panel heat lowers output", '
        '"transform": "temperature_derating", "version": "2"}'
    )
    assert restored == spec
    assert json.dumps(restored.to_dict(), sort_keys=True) == serialized


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
