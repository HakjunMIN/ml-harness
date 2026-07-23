from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from legacy_adapter.contract import AdapterConfig, AdapterContractError, load_adapter, run_adapter, sha256_file
from power_forecasting.proposals import load_proposal, proposal_to_dict


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / ".agents" / "fixtures"
SCRIPTS = ROOT / ".agents" / "scripts"


def safe_run_dir(tmp_path: Path) -> Path:
    path = ROOT / "outputs" / f"pytest-{tmp_path.name}"
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.mark.parametrize("script_name", ["run-legacy.sh", "run-aidm.sh", "verify-promotion.sh"])
def test_legacy_adapter_scripts_run_python_through_uv(script_name: str) -> None:
    content = (SCRIPTS / script_name).read_text(encoding="utf-8")

    assert "uv run python" in content


def test_readme_has_no_stale_direct_python3_uv_invocations() -> None:
    content = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python3 -m" not in content
    assert "python3 -" not in content


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


@pytest.mark.parametrize("module_name", ["legacy_adapter.contract", "harness.contract"])
def test_cli_module_is_usable_from_shell(tmp_path: Path, module_name: str) -> None:
    completed = subprocess.run(
        [
            "python3",
            "-m",
            module_name,
            "--adapter",
            str(FIXTURES / "valid-adapter.json"),
            "--run-dir",
            str(tmp_path / module_name.replace(".", "-")),
            "--run-id",
            "cli",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / ".agents")},
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / module_name.replace(".", "-") / "legacy-evidence.json").exists()


def test_run_legacy_script_creates_evidence(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
    completed = subprocess.run(
        [str(SCRIPTS / "run-legacy.sh"), "--adapter", str(FIXTURES / "valid-adapter.json"), "--run-dir", str(run_dir), "--run-id", "script"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (run_dir / "legacy-evidence.json").exists()


def test_run_legacy_script_rejects_unknown_options(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
    completed = subprocess.run(
        [str(SCRIPTS / "run-legacy.sh"), "--adapter", str(FIXTURES / "valid-adapter.json"), "--run-dir", str(run_dir), "--surprise"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "unknown option" in completed.stderr.lower()
    assert not (run_dir / "legacy-evidence.json").exists()


@pytest.mark.parametrize("script_name", ["run-legacy.sh", "run-aidm.sh", "verify-promotion.sh"])
def test_output_scripts_allow_legacy_local_run_directories(
    tmp_path: Path, script_name: str
) -> None:
    run_dir = tmp_path / "local-output"
    if script_name == "run-legacy.sh":
        arguments = [
            "--adapter",
            str(FIXTURES / "valid-adapter.json"),
            "--run-dir",
            str(run_dir),
        ]
    elif script_name == "run-aidm.sh":
        arguments = [
            "--dataset",
            str(FIXTURES / "valid-dataset.csv"),
            "--run-dir",
            str(run_dir),
            "--folds",
            "1",
            "--minimum-improvement",
            "0",
            "--max-plant-regression",
            "1",
            "--top-single-candidates",
            "1",
        ]
    else:
        run_dir.mkdir()
        (run_dir / "promotion_manifest.json").write_text(
            (FIXTURES / "promoted-manifest.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        arguments = ["--run-dir", str(run_dir)]

    completed = subprocess.run(
        [str(SCRIPTS / script_name), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    if script_name == "run-legacy.sh":
        assert (run_dir / "legacy-evidence.json").is_file()
    elif script_name == "run-aidm.sh":
        assert (run_dir / "promotion_manifest.json").is_file()
    else:
        assert (run_dir / "promotion-evidence.json").is_file()


@pytest.mark.parametrize(
    "protected_path",
    [
        "/",
        ROOT / ".git",
        ROOT / "src",
        ROOT / ".agents" / "harness",
        ROOT / "src" / ".." / "legacy-protected",
    ],
)
@pytest.mark.parametrize("script_name", ["run-legacy.sh", "run-aidm.sh", "verify-promotion.sh"])
def test_output_scripts_reject_unsafe_legacy_destinations_before_side_effects(
    tmp_path: Path, protected_path: Path | str, script_name: str
) -> None:
    if script_name == "run-legacy.sh":
        arguments = [
            "--adapter",
            str(FIXTURES / "valid-adapter.json"),
            "--run-dir",
            str(protected_path),
        ]
    elif script_name == "run-aidm.sh":
        arguments = [
            "--dataset",
            str(FIXTURES / "valid-dataset.csv"),
            "--run-dir",
            str(protected_path),
        ]
    else:
        arguments = ["--run-dir", str(protected_path)]

    completed = subprocess.run(
        [str(SCRIPTS / script_name), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "run-dir" in completed.stderr


def test_verify_promotion_rejects_harness_source_before_side_effects() -> None:
    harness_dir = ROOT / ".agents" / "harness"
    generated = harness_dir / "generated" / "promoted_features.py"
    evidence = harness_dir / "promotion-evidence.json"

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(harness_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "protected repository content" in completed.stderr
    assert not generated.exists()
    assert not evidence.exists()


@pytest.mark.parametrize("script_name", ["run-legacy.sh", "run-aidm.sh", "verify-promotion.sh"])
def test_output_scripts_reject_root_run_directory_with_exit_two(script_name: str) -> None:
    if script_name == "run-legacy.sh":
        arguments = [
            "--adapter",
            str(FIXTURES / "valid-adapter.json"),
            "--run-dir",
            "/",
        ]
    elif script_name == "run-aidm.sh":
        arguments = [
            "--dataset",
            str(FIXTURES / "valid-dataset.csv"),
            "--run-dir",
            "/",
        ]
    else:
        arguments = ["--run-dir", "/"]

    completed = subprocess.run(
        [str(SCRIPTS / script_name), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "run-dir" in completed.stderr


@pytest.mark.parametrize("script_name", ["run-legacy.sh", "run-aidm.sh", "verify-promotion.sh"])
def test_output_scripts_reject_symlinked_run_directories(
    tmp_path: Path, script_name: str
) -> None:
    safe_root = ROOT / ".agents" / "output" / f"pytest-symlink-{tmp_path.name}"
    shutil.rmtree(safe_root, ignore_errors=True)
    safe_root.mkdir(parents=True, exist_ok=True)
    linked = safe_root / "linked"
    linked.symlink_to(tmp_path / "real", target_is_directory=True)
    if script_name == "run-legacy.sh":
        arguments = [
            "--adapter",
            str(FIXTURES / "valid-adapter.json"),
            "--run-dir",
            str(linked),
        ]
    elif script_name == "run-aidm.sh":
        arguments = [
            "--dataset",
            str(FIXTURES / "valid-dataset.csv"),
            "--run-dir",
            str(linked),
        ]
    else:
        arguments = ["--run-dir", str(linked)]

    completed = subprocess.run(
        [str(SCRIPTS / script_name), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "symlink" in completed.stderr


def test_run_aidm_script_requires_explicit_dataset_and_writes_artifacts(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
    completed = subprocess.run(
        [str(SCRIPTS / "run-aidm.sh"), "--dataset", str(FIXTURES / "valid-dataset.csv"), "--run-dir", str(run_dir), "--folds", "1", "--minimum-improvement", "0", "--max-plant-regression", "1", "--top-single-candidates", "1", "--seed", "7"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode in {0, 2}, completed.stderr
    assert (run_dir / "experiments.db").exists()
    assert (run_dir / "promotion_manifest.json").exists()
    assert (run_dir / "performance_report.md").exists()


def test_run_aidm_script_forwards_agentic_proposal_and_legacy_predictions(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
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
            str(run_dir),
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
    manifest = read_json(run_dir / "promotion_manifest.json")
    assert manifest["proposal"]["proposal_id"] == "fixture-agentic-proposal"
    assert "legacy_baseline" in manifest
    assert "selected_model_recipe" in manifest


def test_model_search_fixture_is_valid_bounded_and_leakage_free() -> None:
    proposal_path = FIXTURES / "model-search-proposal.json"

    proposal = load_proposal(proposal_path)
    payload = proposal_to_dict(proposal)

    assert payload["proposal_id"] == "fixture-model-search-proposal"
    assert [feature_set["name"] for feature_set in payload["feature_sets"]] == [
        "safe_temporal",
        "safe_weather",
        "safe_history",
    ]
    assert [recipe["recipe"] for recipe in payload["model_recipes"]] == [
        "random_forest",
        "xgboost",
        "lightgbm",
    ]
    assert payload["search"]["sampler"] == "tpe"
    assert payload["search"]["n_trials"] == 2
    evaluation_count = len(payload["feature_sets"]) * (
        len(payload["model_recipes"]) + payload["search"]["n_trials"] + 1
    )
    assert evaluation_count == 18
    assert evaluation_count <= payload["budget"]["max_evaluations"]
    all_inputs = [
        source
        for feature_set in payload["feature_sets"]
        for spec in feature_set["specs"]
        for source in spec["inputs"]
    ]
    assert "generation_mw" not in all_inputs
    assert not any(source.startswith("actual_") for source in all_inputs)
    assert json.loads(json.dumps(payload, sort_keys=True, allow_nan=False)) == payload


def test_run_aidm_script_already_forwards_fixture_proposal_without_extra_options() -> None:
    content = (SCRIPTS / "run-aidm.sh").read_text(encoding="utf-8")

    assert "--proposal|--legacy-predictions)" in content
    assert 'args+=("$1" "$2")' in content
    assert "model-search" not in content


def test_verify_promotion_script_writes_evidence_only_after_success(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
    manifest = run_dir / "promotion_manifest.json"
    manifest.write_text((FIXTURES / "promoted-manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    generated = run_dir / "generated" / "promoted_features.py"
    evidence = run_dir / "promotion-evidence.json"
    assert generated.exists()
    assert evidence.exists()
    payload = read_json(evidence)
    assert payload["status"] == "success"
    assert payload["manifest_sha256"] == sha256_file(manifest)
    assert payload["generated_module_sha256"] == sha256_file(generated)


def test_verify_promotion_script_writes_model_recipe_patch_and_cleans_up_on_failure(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
    manifest = run_dir / "promotion_manifest.json"
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
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    evidence = read_json(run_dir / "promotion-evidence.json")
    assert (run_dir / "model-recipe-patch.json").exists()
    assert evidence["model_recipe_patch_sha256"] == sha256_file(run_dir / "model-recipe-patch.json")

    broken = read_json(FIXTURES / "promoted-manifest.json")
    broken["selected_specs"][0]["inputs"] = ["actual_irradiance"]
    manifest.write_text(json.dumps(broken, sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not (run_dir / "promotion-evidence.json").exists()
    assert not (run_dir / "generated" / "promoted_features.py").exists()
    assert not (run_dir / "model-recipe-patch.json").exists()


def test_verify_promotion_script_fails_closed_for_rejected_manifest(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
    manifest = run_dir / "promotion_manifest.json"
    manifest.write_text((FIXTURES / "rejected-promotion-manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "promote" in completed.stderr.lower()
    assert not (run_dir / "promotion-evidence.json").exists()
    assert not (run_dir / "generated" / "promoted_features.py").exists()


def test_verify_promotion_script_fails_closed_for_leakage_manifest(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path)
    manifest = run_dir / "promotion_manifest.json"
    manifest.write_text((FIXTURES / "leakage-promotion-manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    completed = subprocess.run(
        [str(SCRIPTS / "verify-promotion.sh"), "--run-dir", str(run_dir)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not (run_dir / "promotion-evidence.json").exists()
    assert not (run_dir / "generated" / "promoted_features.py").exists()


def test_skill_files_follow_local_conventions() -> None:
    for name in ["legacy-intake", "aidm-experiment", "aidd-promotion", "release-gate", "human-review"]:
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
    assert "github pr approval" not in content


def test_human_review_skill_displays_safe_evidence_and_collects_review_decision() -> None:
    content = (ROOT / ".agents" / "skills" / "human-review" / "SKILL.md").read_text(encoding="utf-8").lower()

    for phrase in [
        "vscode_askquestions",
        "performance_report.md",
        "verification.json",
        "promotion_manifest.json",
        "manifest checksum",
        "fail closed",
        "must not deploy",
        "must not merge",
        "request aidd verification",
        "reject or request changes",
    ]:
        assert phrase in content
    for phrase in ["## review tables", "| category | baseline | selected candidate |", "| check | status | detail |", "| artifact | purpose | checksum |"]:
        assert phrase in content
    assert "github pr approval" not in content


def test_aidd_promotion_skill_includes_table_based_human_review_template() -> None:
    content = (ROOT / ".agents" / "skills" / "aidd-promotion" / "SKILL.md").read_text(encoding="utf-8").lower()

    for phrase in ["## human-readable review tables", "| validation | status | reviewer focus |", "| artifact | review purpose | checksum |", "| decision | permitted next step |"]:
        assert phrase in content


def test_gitignore_keeps_agent_sources_but_ignores_root_artifact_directories() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "runs/" in gitignore
    assert "outputs/" in gitignore
    assert ".agents/runs/" not in gitignore
    assert ".agents/output/" not in gitignore
    assert ".agents/fixtures/" not in gitignore
    assert ".agents/skills/" not in gitignore
    assert ".agents/legacy_adapter/" not in gitignore


def test_readme_documents_legacy_adapter_execution() -> None:
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
