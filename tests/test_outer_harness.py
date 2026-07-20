from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

import pytest

from harness.contract import AdapterConfig, AdapterContractError, load_adapter, run_adapter, sha256_file


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / ".agents" / "fixtures"
SCRIPTS = ROOT / ".agents" / "scripts"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, **overrides: object) -> Path:
    payload = {
        "schema_version": "1",
        "legacy_command": ["python3", str(FIXTURES / "legacy_fixture.py")],
        "input_dataset": "valid-dataset.csv",
        "predictions_output": "generated/predictions.csv",
        "required_prediction_columns": ["plant_id", "timestamp", "prediction_mw"],
        "timeout_seconds": 30,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_load_adapter_accepts_exact_schema_and_resolves_paths() -> None:
    adapter = load_adapter(FIXTURES / "valid-adapter.json")

    assert isinstance(adapter, AdapterConfig)
    assert adapter.schema_version == "1"
    assert adapter.legacy_command[0] == "python3"
    assert adapter.input_dataset == (FIXTURES / "valid-dataset.csv").resolve()
    assert adapter.predictions_output == (FIXTURES / "generated" / "predictions.csv").resolve()
    assert adapter.required_prediction_columns == ("plant_id", "timestamp", "prediction_mw")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema_version": "2"}, "schema_version"),
        ({"legacy_command": []}, "legacy_command"),
        ({"legacy_command": ["python3", ""]}, "non-empty"),
        ({"timeout_seconds": 0}, "timeout"),
        ({"timeout_seconds": 3601}, "timeout"),
        ({"required_prediction_columns": []}, "required_prediction_columns"),
        ({"extra": "field"}, "unknown"),
    ],
)
def test_load_adapter_rejects_invalid_contract_shapes(tmp_path: Path, overrides: dict, message: str) -> None:
    manifest = write_manifest(tmp_path / "adapter.json", **overrides)

    with pytest.raises(AdapterContractError, match=message):
        load_adapter(manifest)


def test_adapter_rejects_path_traversal() -> None:
    with pytest.raises(AdapterContractError, match="escapes"):
        load_adapter(FIXTURES / "invalid-traversal-adapter.json")


def test_run_adapter_executes_fixture_and_writes_redacted_evidence(tmp_path: Path) -> None:
    evidence_path = run_adapter(FIXTURES / "valid-adapter.json", tmp_path, run_id="fixture-run")

    evidence = read_json(evidence_path)
    predictions = FIXTURES / "valid-predictions.csv"
    assert predictions.exists()
    assert evidence["schema_version"] == "1"
    assert evidence["status"] == "success"
    assert evidence["run_id"] == "fixture-run"
    assert evidence["command"][0] == "python3"
    assert evidence["environment"] == sorted([
        "HARNESS_INPUT_DATASET",
        "HARNESS_PREDICTIONS_OUTPUT",
        "HARNESS_RUN_DIR",
    ])
    assert "HARNESS_INPUT_DATASET=" not in json.dumps(evidence)
    assert evidence["input_dataset"]["sha256"] == sha256_file(FIXTURES / "valid-dataset.csv")
    assert evidence["predictions_output"]["sha256"] == sha256_file(predictions)
    assert "customer" not in json.dumps(evidence).lower()


def test_run_adapter_evidence_is_deterministic_except_runtime_seconds(tmp_path: Path) -> None:
    first = read_json(run_adapter(FIXTURES / "valid-adapter.json", tmp_path / "one", run_id="same"))
    second = read_json(run_adapter(FIXTURES / "valid-adapter.json", tmp_path / "two", run_id="same"))
    first.pop("duration_seconds", None)
    second.pop("duration_seconds", None)

    assert first == second


def test_run_adapter_preserves_failure_evidence_for_missing_required_columns(tmp_path: Path) -> None:
    with pytest.raises(AdapterContractError, match="required prediction columns"):
        run_adapter(FIXTURES / "missing-column-adapter.json", tmp_path, run_id="missing-column")

    evidence = read_json(tmp_path / "legacy-evidence.json")
    assert evidence["status"] == "failure"
    assert "required prediction columns" in evidence["error"]
    assert "generation_mw" not in json.dumps(evidence)


def test_run_adapter_rejects_empty_predictions(tmp_path: Path) -> None:
    command = tmp_path / "empty_writer.py"
    command.write_text(
        "from pathlib import Path\nimport os\nPath(os.environ['HARNESS_PREDICTIONS_OUTPUT']).write_text('')\n",
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path / "adapter.json", legacy_command=["python3", str(command)])
    (tmp_path / "valid-dataset.csv").write_text((FIXTURES / "valid-dataset.csv").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(AdapterContractError, match="non-empty"):
        run_adapter(manifest, tmp_path / "run")


def test_run_adapter_rejects_malformed_quoted_predictions(tmp_path: Path) -> None:
    command = tmp_path / "malformed_writer.py"
    command.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['HARNESS_PREDICTIONS_OUTPUT']).write_text("
        "'plant_id,timestamp,prediction_mw\\nplant-1,\"2026-01-01T00:00:00Z,10\\n',"
        " encoding='utf-8')\n",
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path / "adapter.json", legacy_command=["python3", str(command)])
    (tmp_path / "valid-dataset.csv").write_text((FIXTURES / "valid-dataset.csv").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(AdapterContractError, match="malformed prediction"):
        run_adapter(manifest, tmp_path / "run")

    evidence = read_json(tmp_path / "run" / "legacy-evidence.json")
    assert evidence["status"] == "failure"
    assert "malformed prediction" in evidence["error"]


@pytest.mark.parametrize(
    "row",
    [
        "plant-1,2026-01-01T00:00:00Z\n",
        "plant-1,2026-01-01T00:00:00Z,10,extra\n",
    ],
)
def test_run_adapter_rejects_ragged_prediction_rows(tmp_path: Path, row: str) -> None:
    command = tmp_path / "ragged_writer.py"
    csv_content = "plant_id,timestamp,prediction_mw\n" + row
    command.write_text(
        "from pathlib import Path\n"
        "import os\n"
        f"Path(os.environ['HARNESS_PREDICTIONS_OUTPUT']).write_text({csv_content!r}, encoding='utf-8')\n",
        encoding="utf-8",
    )
    manifest = write_manifest(tmp_path / "adapter.json", legacy_command=["python3", str(command)])
    (tmp_path / "valid-dataset.csv").write_text((FIXTURES / "valid-dataset.csv").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(AdapterContractError, match="ragged prediction"):
        run_adapter(manifest, tmp_path / "run")

    evidence = read_json(tmp_path / "run" / "legacy-evidence.json")
    assert evidence["status"] == "failure"
    assert "ragged prediction" in evidence["error"]


def test_cli_module_is_usable_from_shell(tmp_path: Path) -> None:
    completed = subprocess.run(
        ["python3", "-m", "harness.contract", "--adapter", str(FIXTURES / "valid-adapter.json"), "--run-dir", str(tmp_path), "--run-id", "cli"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / ".agents")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "legacy-evidence.json").exists()


def test_run_legacy_script_creates_evidence(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(SCRIPTS / "run-legacy.sh"), "--adapter", str(FIXTURES / "valid-adapter.json"), "--run-dir", str(tmp_path), "--run-id", "script"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "legacy-evidence.json").exists()


def test_run_legacy_script_rejects_unknown_options(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(SCRIPTS / "run-legacy.sh"), "--adapter", str(FIXTURES / "valid-adapter.json"), "--run-dir", str(tmp_path), "--surprise"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "unknown option" in completed.stderr.lower()
    assert not (tmp_path / "legacy-evidence.json").exists()


def test_run_aidm_script_requires_explicit_dataset_and_writes_artifacts(tmp_path: Path) -> None:
    completed = subprocess.run(
        [str(SCRIPTS / "run-aidm.sh"), "--dataset", str(FIXTURES / "valid-dataset.csv"), "--run-dir", str(tmp_path), "--folds", "1", "--minimum-improvement", "0", "--max-plant-regression", "1", "--top-single-candidates", "1", "--seed", "7"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode in {0, 2}, completed.stderr
    assert (tmp_path / "experiments.db").exists()
    assert (tmp_path / "promotion_manifest.json").exists()
    assert (tmp_path / "performance_report.md").exists()


def test_run_aidm_script_forwards_agentic_proposal_and_legacy_predictions(tmp_path: Path) -> None:
    legacy_predictions = tmp_path / "legacy-predictions.csv"
    legacy_predictions.write_text(
        (FIXTURES / "legacy-predictions.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            str(SCRIPTS / "run-aidm.sh"),
            "--dataset",
            str(FIXTURES / "valid-dataset.csv"),
            "--run-dir",
            str(tmp_path),
            "--proposal",
            str(FIXTURES / "research-proposal.json"),
            "--legacy-predictions",
            str(legacy_predictions),
            "--folds",
            "1",
            "--minimum-improvement",
            "0",
            "--max-plant-regression",
            "1",
            "--seed",
            "7",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode in {0, 2}, completed.stderr
    manifest = read_json(tmp_path / "promotion_manifest.json")
    assert manifest["proposal"]["proposal_id"] == "fixture-agentic-proposal"
    assert "legacy_baseline" in manifest
    assert "selected_model_recipe" in manifest


def test_verify_promotion_script_writes_evidence_only_after_success(tmp_path: Path) -> None:
    manifest = tmp_path / "promotion_manifest.json"
    manifest.write_text((FIXTURES / "promoted-manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    generated = tmp_path / "generated" / "promoted_features.py"
    evidence = tmp_path / "promotion-evidence.json"
    assert generated.exists()
    assert evidence.exists()
    payload = read_json(evidence)
    assert payload["status"] == "success"
    assert payload["manifest_sha256"] == sha256_file(manifest)
    assert payload["generated_module_sha256"] == sha256_file(generated)


def test_verify_promotion_script_writes_model_recipe_patch_and_cleans_up_on_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "promotion_manifest.json"
    payload = read_json(FIXTURES / "promoted-manifest.json")
    selected_model_recipe = {
        "name": "ridge_low",
        "recipe": "ridge",
        "parameters": {"alpha": 1.0},
        "rationale": "Linear regularized baseline.",
    }
    payload["proposal"] = {
        "schema_version": "1",
        "proposal_id": "fixture-agentic-proposal",
        "rationale": "Fixture proposal evidence.",
        "baseline": {"model": "SPOT"},
        "feature_sets": [
            {
                "name": "hour",
                "rationale": "Synthetic fixture prediction-time hour signal.",
                "specs": payload["selected_specs"],
            }
        ],
        "model_recipes": [selected_model_recipe],
        "budget": {"max_evaluations": 1, "top_feature_groups": 1},
    }
    payload["winner"]["name"] = "ridge_low:hour"
    payload["selected_model_recipe"] = selected_model_recipe
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = read_json(tmp_path / "promotion-evidence.json")
    assert (tmp_path / "model-recipe-patch.json").exists()
    assert evidence["model_recipe_patch_sha256"] == sha256_file(tmp_path / "model-recipe-patch.json")

    broken = read_json(FIXTURES / "promoted-manifest.json")
    broken["selected_specs"][0]["inputs"] = ["actual_irradiance"]
    manifest.write_text(json.dumps(broken, sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "promotion-evidence.json").exists()
    assert not (tmp_path / "generated" / "promoted_features.py").exists()
    assert not (tmp_path / "model-recipe-patch.json").exists()


def test_verify_promotion_script_fails_closed_for_rejected_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "promotion_manifest.json"
    manifest.write_text((FIXTURES / "rejected-promotion-manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "promote" in completed.stderr.lower()
    assert not (tmp_path / "promotion-evidence.json").exists()
    assert not (tmp_path / "generated" / "promoted_features.py").exists()


def test_verify_promotion_script_fails_closed_for_leakage_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "promotion_manifest.json"
    manifest.write_text((FIXTURES / "leakage-promotion-manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not (tmp_path / "promotion-evidence.json").exists()
    assert not (tmp_path / "generated" / "promoted_features.py").exists()


def test_skill_files_follow_local_conventions() -> None:
    for name in ["legacy-intake", "aidm-experiment", "aidd-promotion", "release-gate"]:
        path = ROOT / ".agents" / "skills" / name / "SKILL.md"
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert f"name: {name}" in content
        assert "description: Use when" in content
        assert "## Prerequisites" in content
        assert "## Workflow" in content
        assert "## Error Table" in content
        assert "## Evidence Output Layout" in content
        assert "## Post-Run Reflection" in content
        assert len(content.splitlines()) <= 500
    assert not list((ROOT / ".agents" / "skills").glob("*/plugin.json"))


def test_release_gate_requires_human_approval_and_fails_closed() -> None:
    content = (ROOT / ".agents" / "skills" / "release-gate" / "SKILL.md").read_text(encoding="utf-8").lower()
    for phrase in ["human approval", "fail closed", "baseline", "aidm", "aidd", "compile", "must not deploy", "must not merge", "must not edit customer systems"]:
        assert phrase in content


def test_gitignore_keeps_agent_sources_but_ignores_runs() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".agents/runs/" in gitignore
    assert ".agents/output/" in gitignore
    assert ".agents/fixtures/" not in gitignore
    assert ".agents/skills/" not in gitignore
    assert ".agents/harness/" not in gitignore


def test_readme_documents_outer_harness() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in [
        ".agents/scripts/run-legacy.sh",
        ".agents/scripts/run-aidm.sh",
        ".agents/scripts/verify-promotion.sh",
        "legacy-evidence.json",
        "promotion-evidence.json",
        "human approval",
        "고객 데이터",
        "거부",
    ]:
        assert phrase in readme
