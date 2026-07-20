from __future__ import annotations

import copy
import importlib.util
import math
import re
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from power_forecasting import aidd
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
