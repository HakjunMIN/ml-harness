from __future__ import annotations

import copy
import importlib.util
import math
import re
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from power_forecasting import aidd, aidm
from power_forecasting.data import generate_synthetic_data
from power_forecasting.features import FeatureSpec, apply_feature_specs


@pytest.fixture
def aidd_run_dir(request):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    root = Path(__file__).resolve().parents[1] / "runs" / "pytest-aidd" / safe_name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_rendered_hour_sin_module_imports_and_builds_exact_feature_column(aidd_run_dir):
    manifest = _valid_manifest(
        [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",), rationale="daily solar phase")]
    )
    target = aidd_run_dir / "generated" / "promoted_features.py"

    rendered = aidd.render_promoted_module(manifest, target)
    module = _load_module(rendered)
    frame = generate_synthetic_data(days=1, plants=1, seed=4)

    features = module.build_promoted_features(frame)

    assert rendered == target
    assert list(features.columns) == ["hour_sin"]
    expected = np.sin(2 * math.pi * frame["timestamp"].dt.hour.to_numpy() / 24)
    np.testing.assert_allclose(features["hour_sin"].to_numpy(), expected)


def test_generated_output_matches_direct_engine_for_multiple_specs(aidd_run_dir):
    specs = [
        FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",)),
        FeatureSpec(
            "effective_irradiance",
            "effective_irradiance",
            ("forecast_irradiance", "forecast_cloud_cover"),
        ),
    ]
    target = aidd_run_dir / "promoted.py"
    module = _load_module(aidd.render_promoted_module(_valid_manifest(specs), target))
    frame = generate_synthetic_data(days=2, plants=2, seed=8)

    generated = module.build_promoted_features(frame)
    direct = apply_feature_specs(frame, specs)

    pd.testing.assert_frame_equal(generated, direct)


@pytest.mark.parametrize(
    ("epsilon", "denominator"),
    [(3.0, 4.0), (1e16, 2e16)],
)
def test_valid_ratio_spec_with_large_epsilon_is_accepted_and_validates_on_real_sources(
    aidd_run_dir, epsilon, denominator
):
    specs = [
        FeatureSpec(
            "irradiance_cloud_ratio",
            "ratio",
            ("forecast_irradiance", "forecast_cloud_cover"),
            {"epsilon": epsilon},
        )
    ]
    target = aidd.render_promoted_module(_valid_manifest(specs), aidd_run_dir / "promoted.py")
    module = _load_module(target)
    frame = pd.DataFrame(
        {
            "forecast_irradiance": [12.0, 18.0],
            "forecast_cloud_cover": [denominator, denominator * 2.0],
        }
    )

    pd.testing.assert_frame_equal(
        module.build_promoted_features(frame),
        apply_feature_specs(frame, specs),
    )


def test_valid_aidm_manifest_is_accepted_for_promotion_rendering(aidd_run_dir, monkeypatch):
    def fake_evaluate(frame, definition, feature_specs, folds):
        score = 0.2 if not feature_specs else 0.1
        return SimpleNamespace(
            metrics={"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
            per_plant={
                "plant_01": {"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
                "plant_02": {"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score},
            },
            fold_metrics=[{"MAE": score * 100.0, "RMSE": score * 120.0, "NMAE": score}],
            predictions=pd.DataFrame({"prediction": [1.0]}),
        )

    monkeypatch.setattr(aidm, "evaluate_model", fake_evaluate)

    result = aidm.run_aidm(
        generate_synthetic_data(days=4, plants=2, seed=14),
        aidd_run_dir / "experiments.sqlite",
        aidm.AIDMConfig(
            folds=1,
            minimum_improvement=0.0,
            max_plant_regression=0.2,
            top_single_candidates=1,
        ),
    )

    assert result.winner is not None
    assert aidd.validate_promotion_manifest(result.manifest) == result.winner.specs


def test_promotion_manifest_rejects_feature_inputs_outside_prediction_contract():
    manifest = _valid_manifest(
        [
            FeatureSpec(
                "effective_irradiance",
                "effective_irradiance",
                ("forecast_irradiance", "forecast_cloud_cover"),
            )
        ]
    )
    manifest["selected_specs"][0]["inputs"][1] = "definitely_not_a_contract_column"

    with pytest.raises(
        aidd.PromotionManifestError,
        match=r"feature effective_irradiance: unavailable prediction input definitely_not_a_contract_column",
    ):
        aidd.validate_promotion_manifest(manifest)


def test_generated_history_features_match_direct_engine_for_interleaved_plants(aidd_run_dir):
    specs = _history_feature_specs()
    module = _load_module(
        aidd.render_promoted_module(_valid_manifest(specs), aidd_run_dir / "promoted.py")
    )
    frame = pd.DataFrame(
        {
            "plant_id": [
                "plant_b",
                "plant_a",
                "plant_b",
                "plant_a",
                "plant_b",
                "plant_a",
                "plant_a",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:00",
                    "2024-01-01 01:00",
                    "2024-01-01 01:00",
                    "2024-01-01 02:00",
                    "2024-01-01 02:00",
                    "2024-01-01 03:00",
                ]
            ),
            "forecast_irradiance": [20.0, 10.0, 40.0, 30.0, 60.0, 50.0, 70.0],
            "forecast_cloud_cover": [30.0, 40.0, 50.0, 50.0, 70.0, 60.0, 70.0],
        }
    )

    generated = module.build_promoted_features(frame)
    pd.testing.assert_frame_equal(
        generated, apply_feature_specs(frame, specs)
    )
    assert generated.loc[6, "prior_cloud_mean"] == 50.0


def test_generated_history_features_accept_mixed_timestamp_formats_without_mutating_input(
    aidd_run_dir,
):
    specs = _history_feature_specs()
    module = _load_module(
        aidd.render_promoted_module(_valid_manifest(specs), aidd_run_dir / "promoted.py")
    )
    frame = pd.DataFrame(
        {
            "plant_id": ["plant_a"] * 4,
            "timestamp": [
                "2024-01-01 00:00:00",
                "2024/01/01 01:00:00",
                "January 1, 2024 02:00:00",
                "2024-01-01T03:00:00",
            ],
            "forecast_irradiance": [10.0, 20.0, 30.0, 40.0],
            "forecast_cloud_cover": [0.1, 0.2, 0.3, 0.4],
        }
    )
    original = frame.copy(deep=True)

    generated = module.build_promoted_features(frame)

    pd.testing.assert_frame_equal(generated, apply_feature_specs(frame, specs))
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (
            pd.DataFrame({"timestamp": ["2024-01-01"], "forecast_irradiance": [1.0]}),
            "history features require columns: ['plant_id']",
        ),
        (
            pd.DataFrame({"plant_id": ["plant_a"], "forecast_irradiance": [1.0]}),
            "history features require columns: ['timestamp']",
        ),
        (
            pd.DataFrame(
                {
                    "plant_id": [None],
                    "timestamp": ["2024-01-01"],
                    "forecast_irradiance": [1.0],
                }
            ),
            "history features require non-null plant_id values",
        ),
        (
            pd.DataFrame(
                {
                    "plant_id": ["plant_a"],
                    "timestamp": ["not-a-timestamp"],
                    "forecast_irradiance": [1.0],
                }
            ),
            "history features require valid timestamp values",
        ),
        (
            pd.DataFrame(
                {
                    "plant_id": ["plant_a", "plant_a"],
                    "timestamp": ["2024-01-01", "2024-01-01"],
                    "forecast_irradiance": [1.0, 2.0],
                }
            ),
            "history features require unique plant_id/timestamp pairs",
        ),
    ],
)
def test_generated_history_features_validate_runtime_frame_boundary(aidd_run_dir, frame, message):
    module = _load_module(
        aidd.render_promoted_module(
            _valid_manifest(_history_feature_specs()), aidd_run_dir / "promoted.py"
        )
    )

    with pytest.raises(ValueError) as exc_info:
        module.build_promoted_features(frame)

    assert str(exc_info.value) == message


def test_prediction_contract_rejects_zero_input_history_specs_before_stateful_special_case():
    spec = FeatureSpec(
        "prior_irradiance",
        "lag",
        (),
        {"periods": 1},
        rationale="Invalid lag missing its source input.",
    )

    with pytest.raises(aidd.PromotionManifestError, match="lag expects 1 inputs"):
        aidd.validate_prediction_time_feature_spec(spec)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            FeatureSpec("prior_irradiance", "lag", ("forecast_irradiance",), {"periods": True}),
            "periods must be an integer",
        ),
        (
            FeatureSpec("prior_irradiance", "lag", ("forecast_irradiance",), {"periods": 4}),
            "periods outside allowed set",
        ),
        (
            FeatureSpec("prior_irradiance", "lag", ("forecast_irradiance",), {"periods": 1, "window": 3}),
            "unexpected parameters",
        ),
    ],
)
def test_prediction_contract_rejects_invalid_history_parameters(spec, message):
    with pytest.raises(aidd.PromotionManifestError, match=message):
        aidd.validate_prediction_time_feature_spec(spec)


def test_canonical_aidm_selected_specs_are_accepted_by_prediction_contract():
    specs = aidm.candidate_catalog()

    assert aidd.validate_promotion_manifest(_valid_manifest(specs)) == specs


@pytest.mark.parametrize(
    ("case_name", "mutate"),
    [
        ("non_mapping", lambda manifest: []),
        ("invalid_schema", lambda manifest: {**manifest, "schema_version": "2"}),
        ("invalid_decision", lambda manifest: {**manifest, "decision": "reject"}),
        ("empty_specs", lambda manifest: {**manifest, "selected_specs": []}),
        (
            "duplicate_specs",
            lambda manifest: {
                **manifest,
                "selected_specs": manifest["selected_specs"] + [manifest["selected_specs"][0]],
            },
        ),
        (
            "unsupported_transform",
            lambda manifest: _replace_first_spec(manifest, transform="not_a_transform"),
        ),
        (
            "target_input",
            lambda manifest: _replace_first_spec(manifest, inputs=["generation_mw"]),
        ),
        (
            "actual_input",
            lambda manifest: _replace_first_spec(manifest, inputs=["actual_irradiance"]),
        ),
        (
            "missing_provenance",
            lambda manifest: _without_key(manifest, "thresholds"),
        ),
        (
            "nonfinite_improvement",
            lambda manifest: {**manifest, "improvement_ratio": math.inf},
        ),
        (
            "nonfinite_winner_metric",
            lambda manifest: {
                **manifest,
                "winner": {
                    **manifest["winner"],
                    "metrics": {"nmae": math.nan},
                },
            },
        ),
        (
            "winner_name_mismatch",
            lambda manifest: {
                **manifest,
                "winner": {**manifest["winner"], "name": "unrelated_candidate"},
            },
        ),
        (
            "threshold_bypass",
            lambda manifest: {
                **manifest,
                "baseline": {
                    **manifest["baseline"],
                    "metrics": {"nmae": 1.0},
                },
                "winner": {
                    **manifest["winner"],
                    "metrics": {"nmae": 0.990000001},
                },
                "thresholds": {
                    **manifest["thresholds"],
                    "minimum_improvement": 0.01,
                },
                "improvement_ratio": 0.01,
            },
        ),
        (
            "rounded_threshold_bypass",
            lambda manifest: {
                **manifest,
                "baseline": {
                    **manifest["baseline"],
                    "metrics": {"nmae": 1.0},
                },
                "winner": {
                    **manifest["winner"],
                    "metrics": {"nmae": 0.9900000000001},
                },
                "thresholds": {
                    **manifest["thresholds"],
                    "minimum_improvement": 0.01,
                },
                "improvement_ratio": 0.01,
            },
        ),
        (
            "plant_delta_over_threshold",
            lambda manifest: {
                **manifest,
                "per_plant_deltas": {"plant_01": 0.0300000000005},
            },
        ),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_invalid_manifests_are_rejected(case_name, mutate):
    manifest = mutate(_valid_manifest())

    with pytest.raises(aidd.PromotionManifestError):
        aidd.validate_promotion_manifest(manifest)


@pytest.mark.parametrize(
    ("case_name", "mutate"),
    [
        ("missing_seed", lambda manifest: _without_key(manifest, "seed")),
        ("bool_seed", lambda manifest: {**manifest, "seed": True}),
        ("float_seed", lambda manifest: {**manifest, "seed": 42.0}),
        ("negative_seed", lambda manifest: {**manifest, "seed": -1}),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_promotion_manifest_requires_aidm_compatible_seed(case_name, mutate):
    manifest = mutate(_valid_manifest())

    with pytest.raises(aidd.PromotionManifestError, match="seed"):
        aidd.validate_promotion_manifest(manifest)


@pytest.mark.parametrize(
    ("case_name", "mutate"),
    [
        (
            "missing_model",
            lambda manifest: {
                **manifest,
                "baseline": _without_key(manifest["baseline"], "model"),
            },
        ),
        (
            "blank_model",
            lambda manifest: {
                **manifest,
                "baseline": {**manifest["baseline"], "model": ""},
            },
        ),
        (
            "whitespace_model",
            lambda manifest: {
                **manifest,
                "baseline": {**manifest["baseline"], "model": "   "},
            },
        ),
        (
            "non_spot_model",
            lambda manifest: {
                **manifest,
                "baseline": {**manifest["baseline"], "model": "baseline"},
            },
        ),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_promotion_manifest_requires_spot_baseline_model(case_name, mutate):
    manifest = mutate(_valid_manifest())

    with pytest.raises(aidd.PromotionManifestError, match=r"baseline\.model"):
        aidd.validate_promotion_manifest(manifest)


@pytest.mark.parametrize(
    ("case_name", "nmae"),
    [
        ("nan_nmae", math.nan),
        ("infinite_nmae", math.inf),
        ("negative_nmae", -0.001),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_promotion_manifest_rejects_nonfinite_or_negative_baseline_nmae(
    case_name, nmae
):
    manifest = _valid_manifest()
    manifest["baseline"]["metrics"]["nmae"] = nmae

    with pytest.raises(aidd.PromotionManifestError, match=r"baseline\.metrics\.nmae"):
        aidd.validate_promotion_manifest(manifest)


def test_promotion_manifest_rejects_negative_winner_nmae_with_field_context():
    manifest = _valid_manifest()
    manifest["winner"]["metrics"]["nmae"] = -0.001
    manifest["improvement_ratio"] = 1.005

    with pytest.raises(aidd.PromotionManifestError, match=r"winner\.metrics\.nmae"):
        aidd.validate_promotion_manifest(manifest)


@pytest.mark.parametrize(
    ("case_name", "section", "metric_name"),
    [
        ("baseline_mae", "baseline", "MAE"),
        ("winner_rmse", "winner", "RMSE"),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_promotion_manifest_rejects_negative_named_error_metrics(
    case_name, section, metric_name
):
    manifest = _valid_manifest()
    manifest[section]["metrics"][metric_name] = -0.001

    with pytest.raises(
        aidd.PromotionManifestError,
        match=rf"{section}\.metrics\.{metric_name}",
    ):
        aidd.validate_promotion_manifest(manifest)


@pytest.mark.parametrize(
    ("case_name", "mutate", "match"),
    [
        (
            "improvement_ratio",
            lambda manifest: {**manifest, "improvement_ratio": 10**1000},
            r"improvement_ratio",
        ),
        (
            "winner_metric",
            lambda manifest: {
                **manifest,
                "winner": {
                    **manifest["winner"],
                    "metrics": {**manifest["winner"]["metrics"], "nmae": 10**1000},
                },
            },
            r"winner\.metrics\.nmae",
        ),
        (
            "per_plant_delta",
            lambda manifest: {
                **manifest,
                "per_plant_deltas": {"plant_01": 10**1000},
            },
            r"per_plant_deltas\.plant_01",
        ),
        (
            "ratio_epsilon",
            lambda manifest: _valid_manifest(
                [
                    FeatureSpec(
                        "irradiance_cloud_ratio",
                        "ratio",
                        ("forecast_irradiance", "forecast_cloud_cover"),
                        {"epsilon": 10**1000},
                    )
                ]
            ),
            r"feature irradiance_cloud_ratio: parameter epsilon",
        ),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_promotion_manifest_wraps_huge_integer_numeric_fields_with_context(
    case_name, mutate, match
):
    manifest = mutate(_valid_manifest())

    with pytest.raises(aidd.PromotionManifestError, match=match):
        aidd.validate_promotion_manifest(manifest)


@pytest.mark.parametrize(
    ("case_name", "field", "value"),
    [
        ("minimum_improvement_nan", "minimum_improvement", math.nan),
        ("minimum_improvement_below_zero", "minimum_improvement", -0.001),
        ("minimum_improvement_above_one", "minimum_improvement", 1.001),
        ("minimum_improvement_huge_integer", "minimum_improvement", 10**1000),
        ("minimum_improvement_string", "minimum_improvement", "0.01"),
        ("max_plant_regression_infinite", "max_plant_regression", math.inf),
        ("max_plant_regression_below_zero", "max_plant_regression", -0.001),
        ("max_plant_regression_above_one", "max_plant_regression", 1.001),
        ("max_plant_regression_huge_integer", "max_plant_regression", 10**1000),
        ("max_plant_regression_string", "max_plant_regression", "0.03"),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_promotion_manifest_rejects_nonfinite_or_out_of_range_thresholds(
    case_name, field, value
):
    manifest = _valid_manifest()
    manifest["thresholds"][field] = value

    with pytest.raises(aidd.PromotionManifestError, match=rf"thresholds\.{field}"):
        aidd.validate_promotion_manifest(manifest)


def test_rendering_is_deterministic_creates_parent_and_ignores_untrusted_code_fields(aidd_run_dir):
    manifest = _valid_manifest()
    manifest["source"] = "print('do not embed')"
    manifest["selected_specs"][0]["code"] = "raise RuntimeError('do not embed')"
    target = aidd_run_dir / "nested" / "promoted.py"

    first = aidd.render_promoted_module(copy.deepcopy(manifest), target).read_bytes()
    second = aidd.render_promoted_module(copy.deepcopy(manifest), target).read_bytes()
    content = first.decode("utf-8")

    assert target.parent.is_dir()
    assert first == second
    assert "\r\n" not in content
    assert "do not embed" not in content
    assert "PROMOTED_FEATURE_SPECS" in content


def test_render_validation_failure_preserves_existing_target_and_does_not_leave_temp_files(
    aidd_run_dir,
):
    target = aidd_run_dir / "promoted.py"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("keep me\n")
    manifest = _valid_manifest()
    manifest["decision"] = "reject"

    with pytest.raises(aidd.PromotionManifestError):
        aidd.render_promoted_module(manifest, target)

    assert target.read_text(encoding="utf-8") == "keep me\n"
    assert not _temp_files(target.parent)


def test_render_replace_failure_removes_temp_file(aidd_run_dir, monkeypatch):
    target = aidd_run_dir / "promoted.py"

    def fail_replace(source, destination):
        raise RuntimeError("replace failed")

    monkeypatch.setattr(aidd.os, "replace", fail_replace)

    with pytest.raises(RuntimeError, match="replace failed"):
        aidd.render_promoted_module(_valid_manifest(), target)

    assert not target.exists()
    assert not _temp_files(target.parent)


def test_untrusted_string_subclasses_are_rejected_before_rendering(aidd_run_dir):
    class EvilStr(str):
        def __repr__(self):
            return "(__import__('builtins').print('INJECTED') or 'hour_sin')"

    target = aidd_run_dir / "promoted.py"
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("keep me\n")
    manifest = _valid_manifest()
    manifest["winner"]["name"] = EvilStr("hour_sin")
    manifest["selected_specs"][0]["rationale"] = EvilStr("looks safe")

    with pytest.raises(aidd.PromotionManifestError):
        aidd.render_promoted_module(manifest, target)

    assert target.read_text(encoding="utf-8") == "keep me\n"
    assert not _temp_files(target.parent)


def test_render_uses_validated_values_when_manifest_mapping_changes_after_validation(
    aidd_run_dir,
):
    class EvilStr(str):
        def __repr__(self):
            return "(__import__('builtins').print('INJECTED') or 'hour_sin')"

    class FlippingManifest(dict):
        def __init__(self, payload):
            super().__init__(payload)
            self.winner_reads = 0

        def __getitem__(self, key):
            if key == "winner":
                self.winner_reads += 1
                winner = dict(super().__getitem__(key))
                if self.winner_reads > 1:
                    winner["name"] = EvilStr(winner["name"])
                return winner
            return super().__getitem__(key)

    target = aidd.render_promoted_module(
        FlippingManifest(_valid_manifest()), aidd_run_dir / "promoted.py"
    )
    content = target.read_text(encoding="utf-8")

    assert "INJECTED" not in content
    assert "MANIFEST_SCHEMA_VERSION = '1'" in content
    assert "MANIFEST_WINNER_NAME = 'hour_sin'" in content


def test_generated_module_compiles_imports_and_exposes_immutable_public_specs(aidd_run_dir):
    target = aidd.render_promoted_module(_valid_manifest(), aidd_run_dir / "promoted.py")
    content = target.read_text(encoding="utf-8")

    compile(content, str(target), "exec")
    module = _load_module(target)

    assert module.MANIFEST_SCHEMA_VERSION == "1"
    assert module.MANIFEST_WINNER_NAME == "hour_sin"
    assert isinstance(module.PROMOTED_FEATURE_SPECS, tuple)
    assert all(isinstance(spec, FeatureSpec) for spec in module.PROMOTED_FEATURE_SPECS)
    with pytest.raises(TypeError):
        module.PROMOTED_FEATURE_SPECS[0] = FeatureSpec(
            "hour_cos", "cyclic_hour", ("timestamp",)
        )
    with pytest.raises(FrozenInstanceError):
        module.PROMOTED_FEATURE_SPECS[0].name = "changed"


def _valid_manifest(specs=None):
    selected = list(specs or [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",))])
    return {
        "schema_version": "1",
        "seed": 42,
        "baseline": {
            "model": "SPOT",
            "metrics": {"nmae": 0.2},
            "run_id": "baseline-run",
        },
        "winner": {
            "name": "+".join(sorted(spec.name for spec in selected)),
            "metrics": {"nmae": 0.1},
            "run_id": "winner-run",
        },
        "selected_specs": [spec.to_dict() for spec in selected],
        "per_plant_deltas": {"plant_01": -0.02},
        "thresholds": {
            "minimum_improvement": 0.01,
            "max_plant_regression": 0.03,
        },
        "improvement_ratio": 0.5,
        "decision": "promote",
        "failed_gates": [],
    }


def _history_feature_specs():
    return [
        FeatureSpec(
            "prior_irradiance",
            "lag",
            ("forecast_irradiance",),
            {"periods": 1},
            rationale="Use strictly prior forecast irradiance from the same plant.",
        ),
        FeatureSpec(
            "prior_cloud_mean",
            "rolling_mean",
            ("forecast_cloud_cover",),
            {"window": 3},
            rationale="Use a strictly prior cloud-cover window from the same plant.",
        ),
    ]


def _load_module(path):
    module_name = f"_test_promoted_{re.sub(r'[^A-Za-z0-9_]', '_', str(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _replace_first_spec(manifest, **updates):
    updated = copy.deepcopy(manifest)
    updated["selected_specs"][0].update(updates)
    return updated


def _without_key(manifest, key):
    updated = copy.deepcopy(manifest)
    del updated[key]
    return updated


def _temp_files(directory):
    return [path for path in directory.iterdir() if path.name.endswith(".tmp")]
