from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from power_forecasting import aidd


@pytest.fixture
def aidd_patch_dir(request):
    root = Path(__file__).resolve().parents[1] / "runs" / "pytest-agentic-aidd" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_render_model_recipe_patch_is_canonical_review_only_and_redacted(aidd_patch_dir):
    manifest = _agentic_manifest()
    target = aidd_patch_dir / "model-recipe-patch.json"

    rendered = aidd.render_model_recipe_patch(manifest, target)
    first = target.read_bytes()
    rendered_again = aidd.render_model_recipe_patch(dict(reversed(list(manifest.items()))), target)
    second = target.read_bytes()

    assert rendered == target
    assert rendered_again == target
    assert first == second
    assert first.endswith(b"\n")
    assert b"\r" not in first
    payload = json.loads(first)
    assert payload == {
        "evidence": {
            "baseline_model": "SPOT",
            "failed_gates": [],
            "proposal_id": "proposal-safe-001",
            "winner_name": "ridge_low:safe_solar",
        },
        "manifest_sha256": _manifest_sha(manifest),
        "schema_version": "1",
        "selected_feature_specs_sha256": _sha_json(manifest["selected_specs"]),
        "selected_model_recipe": manifest["selected_model_recipe"],
        "status": "requires_human_review",
        "winner_metrics": {"nmae": 0.1},
    }
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert "import " not in serialized
    assert "exec" not in serialized
    assert "customer" not in serialized
    assert "/users/" not in serialized


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: {**m, "decision": "reject", "failed_gates": ["insufficient"]},
        lambda m: {k: v for k, v in m.items() if k != "selected_model_recipe"},
        lambda m: {**m, "selected_model_recipe": {"recipe": "ridge", "parameters": {"alpha": 999}}},
        lambda m: {**m, "proposal": {"proposal_id": "proposal-safe-001", "customer_path": "/Users/customer/repo"}},
        lambda m: {**m, "selected_model_recipe": {**m["selected_model_recipe"], "rationale": "import os"}},
        lambda m: {**m, "selected_model_recipe": {**m["selected_model_recipe"], "extra": "x"}},
    ],
)
def test_render_model_recipe_patch_rejects_untrusted_manifests(aidd_patch_dir, mutate):
    target = aidd_patch_dir / "model-recipe-patch.json"
    with pytest.raises(aidd.PromotionManifestError):
        aidd.render_model_recipe_patch(mutate(_agentic_manifest()), target)
    assert not target.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: {**m, "selected_model_recipe": _boosted_recipe()},
        lambda m: {
            **m,
            "selected_model_recipe": _duplicate_ridge_recipe(),
            "proposal": {
                **m["proposal"],
                "model_recipes": [
                    *m["proposal"]["model_recipes"],
                    _duplicate_ridge_recipe(),
                ],
            },
        },
        lambda m: {**m, "winner": {**m["winner"], "name": "ridge_low:alternative_solar"}},
        lambda m: {
            **m,
            "proposal": {
                **m["proposal"],
                "feature_sets": [m["proposal"]["feature_sets"][1]],
            },
        },
        lambda m: {**m, "selected_specs": [_alternative_feature_spec()]},
        lambda m: {
            **m,
            "proposal": {
                **m["proposal"],
                "feature_sets": [
                    {
                        **m["proposal"]["feature_sets"][0],
                        "specs": [_alternative_feature_spec()],
                    },
                    m["proposal"]["feature_sets"][1],
                ],
            },
        },
    ],
)
def test_render_model_recipe_patch_rejects_tampered_agentic_provenance(aidd_patch_dir, mutate):
    target = aidd_patch_dir / "model-recipe-patch.json"
    with pytest.raises(aidd.PromotionManifestError):
        aidd.render_model_recipe_patch(mutate(_agentic_manifest()), target)
    assert not target.exists()


def test_render_model_recipe_patch_rejects_embedded_proposal_over_budget(aidd_patch_dir):
    manifest = _agentic_manifest()
    manifest = {
        **manifest,
        "proposal": {
            **manifest["proposal"],
            "budget": {"max_evaluations": 3, "top_feature_groups": 2},
        },
    }
    target = aidd_patch_dir / "model-recipe-patch.json"

    with pytest.raises(aidd.PromotionManifestError):
        aidd.render_model_recipe_patch(manifest, target)
    assert not target.exists()


def test_render_model_recipe_patch_accepts_aidm_sorted_feature_specs(aidd_patch_dir):
    manifest = _agentic_manifest()
    selected_specs = [_alternative_feature_spec(), _safe_feature_spec()]
    manifest = {
        **manifest,
        "winner": {**manifest["winner"], "name": "ridge_low:combined_solar"},
        "selected_specs": selected_specs,
        "proposal": {
            **manifest["proposal"],
            "feature_sets": [
                {
                    "name": "combined_solar",
                    "rationale": "Forecast-time solar interactions.",
                    "specs": list(reversed(selected_specs)),
                },
                manifest["proposal"]["feature_sets"][1],
            ],
        },
    }
    target = aidd_patch_dir / "model-recipe-patch.json"

    aidd.render_model_recipe_patch(manifest, target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["selected_feature_specs_sha256"] == _sha_json(selected_specs)


def _agentic_manifest():
    selected_spec = _safe_feature_spec()
    return {
        "schema_version": "1",
        "seed": 7,
        "baseline": {
            "model": "SPOT",
            "metrics": {"nmae": 0.2},
            "run_id": "baseline-run",
        },
        "legacy_baseline": None,
        "proposal": {
            "schema_version": "1",
            "proposal_id": "proposal-safe-001",
            "rationale": "Evaluate bounded prediction-time hypotheses.",
            "baseline": {"model": "SPOT"},
            "feature_sets": [
                {
                    "name": "safe_solar",
                    "rationale": "Forecast-time solar attenuation.",
                    "specs": [selected_spec],
                },
                {
                    "name": "alternative_solar",
                    "rationale": "Forecast-time capacity scaling.",
                    "specs": [_alternative_feature_spec()],
                },
            ],
            "model_recipes": [
                {
                    "name": "ridge_low",
                    "recipe": "ridge",
                    "parameters": {"alpha": 1.0},
                    "rationale": "Linear regularized baseline.",
                },
                _boosted_recipe(),
            ],
            "budget": {"max_evaluations": 4, "top_feature_groups": 2},
        },
        "winner": {
            "name": "ridge_low:safe_solar",
            "metrics": {"nmae": 0.1},
            "run_id": "winner-run",
        },
        "selected_model_recipe": {
            "name": "ridge_low",
            "recipe": "ridge",
            "parameters": {"alpha": 1.0},
            "rationale": "Linear regularized baseline.",
        },
        "selected_specs": [selected_spec],
        "per_plant_deltas": {"plant_01": -0.1},
        "thresholds": {"minimum_improvement": 0.0, "max_plant_regression": 1.0},
        "improvement_ratio": 0.5,
        "decision": "promote",
        "failed_gates": [],
    }


def _safe_feature_spec():
    return {
        "name": "effective_irradiance",
        "transform": "effective_irradiance",
        "inputs": ["forecast_irradiance", "forecast_cloud_cover"],
        "parameters": {},
        "version": "1",
        "rationale": "Forecast-time solar attenuation.",
    }


def _alternative_feature_spec():
    return {
        "name": "capacity_scaled_irradiance",
        "transform": "interaction",
        "inputs": ["forecast_irradiance", "capacity_mw"],
        "parameters": {},
        "version": "1",
        "rationale": "Forecast-time capacity scaling.",
    }


def _boosted_recipe():
    return {
        "name": "boosted",
        "recipe": "hist_gradient_boosting",
        "parameters": {"max_iter": 50, "learning_rate": 0.1, "max_leaf_nodes": 15},
        "rationale": "Bounded nonlinear baseline.",
    }


def _duplicate_ridge_recipe():
    return {
        "name": "ridge_low",
        "recipe": "ridge",
        "parameters": {"alpha": 10.0},
        "rationale": "Duplicate-name tampered ridge.",
    }


def _manifest_sha(manifest) -> str:
    return _sha_json(manifest)


def _sha_json(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
