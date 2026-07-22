from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from power_forecasting.aidd import (
    PromotionManifestError,
    validate_prediction_time_feature_spec,
    validate_promotion_manifest,
)
from power_forecasting.aidm import (
    AIDMConfig,
    CandidateResult,
    evaluate_promotion_gates,
)
from power_forecasting.cli import run_aidm_workflow, run_legacy
from power_forecasting.data import DataContractError, parse_timestamps, validate_dataset
from power_forecasting.features import FeatureSpec
from power_forecasting.models import model_definition_from_recipe
from power_forecasting.proposals import (
    ProposalValidationError,
    ResearchProposal,
    load_proposal,
    proposal_to_dict,
)
from power_forecasting.reporting import write_performance_report
from power_forecasting.research_contracts import (
    ResearchContractError,
    ResearchLoopConfig,
    validate_run_id,
)
from power_forecasting.research_state import atomic_write_json


_SCHEMA_VERSION = "1"
_MANIFEST_NAME = "promotion_manifest.json"
_DATABASE_NAME = "experiments.db"
_REPORT_NAME = "performance_report.md"
_PROPOSAL_NAME = "research-proposal.json"
_EVIDENCE_NAME = "experiment-evidence.json"
_VERIFICATION_NAME = "verification.json"
_SHA256_LENGTH = 64
_PROPOSAL_RUN_PARAMS_KEYS = frozenset(
    {
        "schema_version",
        "candidate_name",
        "model",
        "folds",
        "seed",
        "specs",
        "model_recipe",
        "proposal_id",
        "proposal",
    }
)
_SEARCH_PROPOSAL_RUN_PARAMS_KEYS = _PROPOSAL_RUN_PARAMS_KEYS | frozenset(
    {"search"}
)
_PROPOSAL_RUN_ARTIFACT_KEYS = frozenset(
    {"summary", "fold_metrics", "prediction_rows", "proposal"}
)
_SELECTED_SEARCH_RUN_ARTIFACT_KEYS = _PROPOSAL_RUN_ARTIFACT_KEYS | frozenset(
    {"selected_from_trial"}
)
_REUSED_SEARCH_RUN_ARTIFACT_KEYS = frozenset(
    {
        "summary",
        "fold_metrics",
        "reused_from_run_id",
        "reused_from_trial_number",
        "reused_from_candidate_name",
    }
)
_VERIFICATION_CHECKS = (
    "experiment_identity",
    "artifact_paths",
    "evidence_schema",
    "proposal_artifact",
    "proposal_checksum",
    "manifest_checksum",
    "report_checksum",
    "database_checksum",
    "manifest_schema",
    "thresholds",
    "seed_provenance",
    "baseline_provenance",
    "bounded_proposal",
    "selected_candidate",
    "winner_provenance",
    "selected_specs",
    "selected_recipe",
    "sqlite_runs",
    "proposal_runs",
    "metrics_provenance",
    "gate_outcome",
    "report_evidence",
    "promotion_provenance",
    "promoted",
    "verification_report",
)


class ResearchExecutionError(RuntimeError):
    """Raised when the bounded AIDM experiment workflow cannot produce evidence."""


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    experiment_id: str
    manifest_path: Path
    report_path: Path
    run_state: str
    selected_candidate_id: str | None
    selected_candidate_spec_sha256: str | None
    selected_recipe_id: str | None


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]
    report_path: Path


def run_experiment_agent(
    *,
    config: ResearchLoopConfig,
    proposal: ResearchProposal,
    iteration: int,
) -> ExperimentResult:
    """Run one bounded proposal through the public AIDM workflow."""

    run_dir = _validated_run_dir(config)
    _validate_iteration(config, iteration)
    validated_proposal = load_proposal(proposal_to_dict(proposal))
    proposal_payload = proposal_to_dict(validated_proposal)
    proposal_sha256 = _sha256_json(proposal_payload)
    experiment_id = _experiment_id(config.run_id, iteration, proposal_sha256)
    iteration_dir = _iteration_dir(run_dir, iteration, experiment_id)
    iteration_dir.mkdir(parents=True, exist_ok=True)

    proposal_path = iteration_dir / _PROPOSAL_NAME
    try:
        atomic_write_json(proposal_path, proposal_payload)
        aidm_config = _research_aidm_config(config)
        dataset_path = Path(config.dataset_path)
        dataset_summary = _dataset_summary(dataset_path)
        aidm_result = run_aidm_workflow(
            iteration_dir,
            dataset=dataset_path,
            config=aidm_config,
            proposal=validated_proposal,
        )
        manifest_path = iteration_dir / _MANIFEST_NAME
        database_path = iteration_dir / _DATABASE_NAME
        report_path = iteration_dir / _REPORT_NAME
        if not manifest_path.is_file() or not database_path.is_file():
            raise ResearchExecutionError("AIDM workflow did not persist required evidence")

        legacy_results = run_legacy(
            iteration_dir,
            dataset=dataset_path,
            folds=aidm_config.folds,
        )
        write_performance_report(
            dataset_summary,
            legacy_results,
            aidm_result,
            {
                "database": database_path,
                "manifest": manifest_path,
                "proposal": proposal_path,
            },
            target=report_path,
        )
        if not report_path.is_file():
            raise ResearchExecutionError("AIDM workflow did not persist a performance report")

        decision = _manifest_decision(aidm_result.manifest)
        winner = aidm_result.winner
        selected_specs = (
            [spec.to_dict() for spec in winner.specs] if winner is not None else None
        )
        selected_recipe = dict(winner.model_recipe) if winner and winner.model_recipe else None
        run_state = "promoted" if decision == "promote" else "rejected"
        evidence = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": config.run_id,
            "experiment_id": experiment_id,
            "iteration": iteration,
            "proposal_id": validated_proposal.proposal_id,
            "proposal_sha256": proposal_sha256,
            "manifest_sha256": _sha256_file(manifest_path),
            "report_sha256": _sha256_file(report_path),
            "database_sha256": _sha256_file(database_path),
            "decision": decision,
            "run_state": run_state,
            "baseline_database_run_id": aidm_result.baseline.run_id,
            "selected_candidate": None
            if winner is None
            else {
                "id": winner.name,
                "database_run_id": winner.run_id,
                "spec_sha256": _sha256_json(selected_specs),
                "recipe_id": None
                if selected_recipe is None
                else selected_recipe.get("name"),
                "recipe_sha256": None
                if selected_recipe is None
                else _sha256_json(selected_recipe),
            },
        }
        atomic_write_json(iteration_dir / _EVIDENCE_NAME, evidence)
    except (
        DataContractError,
        OSError,
        ProposalValidationError,
        ResearchContractError,
        ResearchExecutionError,
        TypeError,
        ValueError,
    ) as exc:
        _write_failure_evidence(iteration_dir, config.run_id, experiment_id, iteration)
        if isinstance(exc, ResearchExecutionError):
            raise
        raise ResearchExecutionError("AIDM experiment workflow failed") from exc

    return ExperimentResult(
        run_id=config.run_id,
        experiment_id=experiment_id,
        manifest_path=manifest_path,
        report_path=report_path,
        run_state=run_state,
        selected_candidate_id=None if winner is None else winner.name,
        selected_candidate_spec_sha256=None
        if selected_specs is None
        else _sha256_json(selected_specs),
        selected_recipe_id=None if selected_recipe is None else selected_recipe.get("name"),
    )


def run_verifier_agent(
    *,
    config: ResearchLoopConfig,
    proposal: ResearchProposal,
    experiment: ExperimentResult,
    iteration: int,
) -> VerificationResult:
    """Independently validate persisted AIDM evidence without modifying it."""

    checks = {name: False for name in _VERIFICATION_CHECKS}
    run_dir = _validated_run_dir(config)
    validated_proposal = load_proposal(proposal_to_dict(proposal))
    proposal_payload = proposal_to_dict(validated_proposal)
    proposal_sha256 = _sha256_json(proposal_payload)
    experiment_id = _experiment_id(config.run_id, iteration, proposal_sha256)
    iteration_dir = _iteration_dir(run_dir, iteration, experiment_id)
    report_path = iteration_dir / _VERIFICATION_NAME
    try:
        iteration_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return _finish_verification(
            checks,
            report_path,
            decision=None,
            proposal_sha256=proposal_sha256,
            observed_checksums={},
        )

    manifest_path = iteration_dir / _MANIFEST_NAME
    performance_report_path = iteration_dir / _REPORT_NAME
    database_path = iteration_dir / _DATABASE_NAME
    proposal_path = iteration_dir / _PROPOSAL_NAME
    evidence_path = iteration_dir / _EVIDENCE_NAME

    checks["artifact_paths"] = _paths_match(
        experiment,
        manifest_path=manifest_path,
        report_path=performance_report_path,
    )

    evidence = _read_json_mapping(evidence_path, iteration_dir)
    checks["evidence_schema"] = _valid_evidence(
        evidence,
        run_id=config.run_id,
        experiment_id=experiment_id,
        iteration=iteration,
        proposal_id=validated_proposal.proposal_id,
        proposal_sha256=proposal_sha256,
    )
    evidence_mapping = evidence if checks["evidence_schema"] else None

    proposal_artifact = _read_json_mapping(proposal_path, iteration_dir)
    artifact_proposal = _load_proposal_safely(proposal_artifact)
    checks["proposal_artifact"] = artifact_proposal is not None
    if artifact_proposal is not None:
        artifact_payload = proposal_to_dict(artifact_proposal)
        artifact_sha256 = _sha256_json(artifact_payload)
        checks["proposal_checksum"] = (
            artifact_payload == proposal_payload
            and artifact_sha256 == proposal_sha256
            and evidence_mapping is not None
            and evidence_mapping["proposal_sha256"] == artifact_sha256
        )

    manifest = _read_json_mapping(manifest_path, iteration_dir)
    checks["manifest_schema"] = _basic_manifest_is_valid(manifest)
    decision = manifest.get("decision") if checks["manifest_schema"] else None
    checks["experiment_identity"] = _experiment_identity_matches(
        experiment,
        run_id=config.run_id,
        experiment_id=experiment_id,
        decision=decision,
    ) and (
        evidence_mapping is None
        or (
            evidence_mapping["decision"] == decision
            and evidence_mapping["run_state"]
            == ("promoted" if decision == "promote" else "rejected")
        )
    )
    manifest_sha256 = _safe_file_sha256(manifest_path, iteration_dir)
    checks["manifest_checksum"] = (
        manifest_sha256 is not None
        and evidence_mapping is not None
        and evidence_mapping["manifest_sha256"] == manifest_sha256
    )

    report_text = _read_text(performance_report_path, iteration_dir)
    report_sha256 = _safe_file_sha256(performance_report_path, iteration_dir)
    checks["report_checksum"] = (
        report_sha256 is not None
        and evidence_mapping is not None
        and evidence_mapping["report_sha256"] == report_sha256
    )

    database_sha256 = _safe_file_sha256(database_path, iteration_dir)
    checks["database_checksum"] = (
        database_sha256 is not None
        and evidence_mapping is not None
        and evidence_mapping["database_sha256"] == database_sha256
    )

    observed_checksums = {
        name: value
        for name, value in {
            "manifest_sha256": manifest_sha256,
            "report_sha256": report_sha256,
            "database_sha256": database_sha256,
        }.items()
        if value is not None
    }

    if checks["manifest_schema"] and isinstance(manifest, Mapping):
        checks["thresholds"] = _thresholds_match(manifest, config)
        checks["seed_provenance"] = _seed_matches_config(manifest, config)
        embedded_proposal = _load_proposal_safely(manifest.get("proposal"))
        checks["bounded_proposal"] = (
            embedded_proposal is not None
            and proposal_to_dict(embedded_proposal) == proposal_payload
            and _selection_is_bounded(manifest, embedded_proposal)
        )
        checks["report_evidence"] = _report_matches_manifest(report_text, manifest)
        checks["promoted"] = decision == "promote"

        if decision == "promote":
            try:
                validate_promotion_manifest(manifest)
            except PromotionManifestError:
                checks["promotion_provenance"] = False
            else:
                checks["promotion_provenance"] = True
        elif decision == "reject":
            checks["promotion_provenance"] = checks["bounded_proposal"]

    if evidence_mapping is not None and isinstance(manifest, Mapping):
        experiment_runs = _read_experiment_runs(
            database_path,
            iteration_dir,
        )
        selected_rows = _selected_runs(
            experiment_runs,
            baseline_run_id=evidence_mapping["baseline_database_run_id"],
            candidate_run_id=evidence_mapping["selected_candidate"]["database_run_id"],
        )
        checks["sqlite_runs"] = _completed_runs(selected_rows)
        checks["proposal_runs"] = _proposal_runs_are_complete(
            experiment_runs,
            proposal=validated_proposal,
            proposal_payload=proposal_payload,
            config=config,
        )
        if selected_rows is not None:
            baseline_row, candidate_row = selected_rows
            checks["metrics_provenance"] = _metrics_match_manifest(
                manifest,
                baseline_row,
                candidate_row,
            )
            checks["baseline_provenance"] = _baseline_matches_manifest(
                manifest,
                baseline_row,
                config,
            )
            checks["selected_candidate"] = _candidate_matches_manifest(
                experiment,
                evidence_mapping,
                manifest,
                candidate_row,
            )
            checks["winner_provenance"] = _winner_run_identity_matches(
                manifest,
                candidate_row,
                config,
            )
            checks["selected_specs"] = _specs_match(
                experiment,
                evidence_mapping,
                manifest,
                candidate_row,
            )
            checks["selected_recipe"] = _recipe_matches(
                experiment,
                evidence_mapping,
                manifest,
                candidate_row,
            )
            checks["gate_outcome"] = _gate_outcome_matches(
                manifest,
                baseline_row,
                candidate_row,
                config,
            )

    return _finish_verification(
        checks,
        report_path,
        decision=decision if decision in {"promote", "reject"} else None,
        proposal_sha256=proposal_sha256,
        observed_checksums=observed_checksums,
    )


def _validated_run_dir(config: ResearchLoopConfig) -> Path:
    if not isinstance(config, ResearchLoopConfig):
        raise TypeError("config must be a ResearchLoopConfig")
    validate_run_id(config.run_id)
    run_dir = Path(config.run_dir).resolve()
    repository_root = Path(__file__).resolve().parents[2]
    protected = tuple(repository_root / name for name in ("src", "tests", "docs"))
    if any(_paths_overlap(run_dir, path) for path in protected):
        raise ResearchContractError("run_dir overlaps protected repository content")
    agents_root = repository_root / ".agents"
    if _paths_overlap(run_dir, agents_root):
        allowed = (agents_root / "runs", agents_root / "output")
        if not any(_is_within(run_dir, path) for path in allowed):
            raise ResearchContractError("run_dir must use an allowed .agents destination")
    return run_dir


def _iteration_dir(run_dir: Path, iteration: int, experiment_id: str) -> Path:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    directory = run_dir / "iterations" / f"{iteration:03d}-{experiment_id[:16]}"
    if not _is_within(directory.resolve(), run_dir):
        raise ResearchExecutionError("iteration directory escapes run_dir")
    return directory


def _validate_iteration(config: ResearchLoopConfig, iteration: int) -> None:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    if iteration > config.max_iterations:
        raise ValueError("iteration must not exceed config.max_iterations")


def _research_aidm_config(config: ResearchLoopConfig) -> AIDMConfig:
    return AIDMConfig(
        folds=config.fold_count,
        minimum_improvement=config.minimum_improvement,
        max_plant_regression=config.max_plant_regression,
        seed=AIDMConfig().seed,
    )


def _experiment_id(run_id: str, iteration: int, proposal_sha256: str) -> str:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValueError("iteration must be a positive integer")
    material = f"{run_id}\n{iteration}\n{proposal_sha256}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _dataset_summary(dataset_path: Path) -> dict[str, object]:
    frame = pd.read_csv(dataset_path)
    validate_dataset(frame)
    timestamps = parse_timestamps(frame["timestamp"])
    return {
        "rows": int(len(frame)),
        "plants": int(frame["plant_id"].nunique()),
        "time_start": timestamps.min().isoformat(),
        "time_end": timestamps.max().isoformat(),
    }


def _manifest_decision(manifest: Mapping[str, Any]) -> str:
    decision = manifest.get("decision")
    if decision not in {"promote", "reject"}:
        raise ResearchExecutionError("AIDM workflow returned an invalid decision")
    return decision


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _safe_json_sha256(value: Any) -> str | None:
    try:
        return _sha256_json(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_failure_evidence(
    iteration_dir: Path,
    run_id: str,
    experiment_id: str,
    iteration: int,
) -> None:
    try:
        atomic_write_json(
            iteration_dir / "experiment-failure.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "iteration": iteration,
                "run_state": "failed",
            },
        )
    except (OSError, TypeError, ValueError):
        pass


def _paths_match(
    experiment: object,
    *,
    manifest_path: Path,
    report_path: Path,
) -> bool:
    if not isinstance(experiment, ExperimentResult):
        return False
    return _path_equals(experiment.manifest_path, manifest_path) and _path_equals(
        experiment.report_path, report_path
    )


def _path_equals(value: object, expected: Path) -> bool:
    if not isinstance(value, Path):
        return False
    try:
        return value.resolve() == expected.resolve()
    except OSError:
        return False


def _valid_evidence(
    evidence: Mapping[str, Any] | None,
    *,
    run_id: str,
    experiment_id: str,
    iteration: int,
    proposal_id: str,
    proposal_sha256: str,
) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    expected_keys = {
        "schema_version",
        "run_id",
        "experiment_id",
        "iteration",
        "proposal_id",
        "proposal_sha256",
        "manifest_sha256",
        "report_sha256",
        "database_sha256",
        "decision",
        "run_state",
        "baseline_database_run_id",
        "selected_candidate",
    }
    if set(evidence) != expected_keys:
        return False
    selected_candidate = evidence["selected_candidate"]
    if not isinstance(selected_candidate, Mapping) or set(selected_candidate) != {
        "id",
        "database_run_id",
        "spec_sha256",
        "recipe_id",
        "recipe_sha256",
    }:
        return False
    return (
        evidence["schema_version"] == _SCHEMA_VERSION
        and evidence["run_id"] == run_id
        and evidence["experiment_id"] == experiment_id
        and evidence["iteration"] == iteration
        and evidence["proposal_id"] == proposal_id
        and evidence["proposal_sha256"] == proposal_sha256
        and evidence["decision"] in {"promote", "reject"}
        and evidence["run_state"]
        == ("promoted" if evidence["decision"] == "promote" else "rejected")
        and all(
            _is_sha256(evidence[name])
            for name in (
                "proposal_sha256",
                "manifest_sha256",
                "report_sha256",
                "database_sha256",
            )
        )
        and type(evidence["baseline_database_run_id"]) is str
        and bool(evidence["baseline_database_run_id"])
        and type(selected_candidate["id"]) is str
        and bool(selected_candidate["id"])
        and type(selected_candidate["database_run_id"]) is str
        and bool(selected_candidate["database_run_id"])
        and _is_sha256(selected_candidate["spec_sha256"])
        and (
            selected_candidate["recipe_id"] is None
            or type(selected_candidate["recipe_id"]) is str
        )
        and (
            selected_candidate["recipe_sha256"] is None
            or _is_sha256(selected_candidate["recipe_sha256"])
        )
    )


def _experiment_identity_matches(
    experiment: object,
    *,
    run_id: str,
    experiment_id: str,
    decision: object,
) -> bool:
    if not isinstance(experiment, ExperimentResult):
        return False
    expected_state = (
        "promoted"
        if decision == "promote"
        else "rejected"
        if decision == "reject"
        else None
    )
    return (
        experiment.run_id == run_id
        and experiment.experiment_id == experiment_id
        and experiment.run_state in {"promoted", "rejected"}
        and (expected_state is None or experiment.run_state == expected_state)
    )


def _read_json_mapping(path: Path, root: Path) -> Mapping[str, Any] | None:
    safe_path = _safe_artifact_file(path, root)
    if safe_path is None:
        return None
    try:
        with safe_path.open("r", encoding="utf-8", newline="") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _read_text(path: Path, root: Path) -> str | None:
    safe_path = _safe_artifact_file(path, root)
    if safe_path is None:
        return None
    try:
        return safe_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _safe_file_sha256(path: Path, root: Path) -> str | None:
    safe_path = _safe_artifact_file(path, root)
    if safe_path is None:
        return None
    try:
        return _sha256_file(safe_path)
    except OSError:
        return None


def _safe_artifact_file(path: Path, root: Path) -> Path | None:
    try:
        if path.is_symlink():
            return None
        resolved_root = root.resolve()
        resolved_path = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not _is_within(resolved_path, resolved_root) or not resolved_path.is_file():
        return None
    return resolved_path


def _load_proposal_safely(value: object) -> ResearchProposal | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return load_proposal(value)
    except (ProposalValidationError, TypeError, ValueError, OverflowError):
        return None


def _basic_manifest_is_valid(manifest: object) -> bool:
    if not isinstance(manifest, Mapping):
        return False
    required = {
        "schema_version",
        "seed",
        "baseline",
        "winner",
        "selected_specs",
        "per_plant_deltas",
        "thresholds",
        "improvement_ratio",
        "decision",
        "failed_gates",
        "proposal",
        "selected_model_recipe",
    }
    if not required.issubset(manifest):
        return False
    if manifest["schema_version"] != _SCHEMA_VERSION:
        return False
    if isinstance(manifest["seed"], bool) or not isinstance(manifest["seed"], int):
        return False
    if manifest["seed"] < 0 or manifest["decision"] not in {"promote", "reject"}:
        return False
    if not isinstance(manifest["baseline"], Mapping) or not isinstance(
        manifest["winner"], Mapping
    ):
        return False
    if not isinstance(manifest["selected_specs"], list) or not manifest["selected_specs"]:
        return False
    if not isinstance(manifest["per_plant_deltas"], Mapping) or not isinstance(
        manifest["thresholds"], Mapping
    ):
        return False
    if not isinstance(manifest["failed_gates"], list) or not all(
        type(value) is str for value in manifest["failed_gates"]
    ):
        return False
    if not isinstance(manifest["proposal"], Mapping) or not isinstance(
        manifest["selected_model_recipe"], Mapping
    ):
        return False
    return all(
        key in manifest["baseline"] for key in ("model", "metrics", "run_id")
    ) and all(key in manifest["winner"] for key in ("name", "metrics", "run_id"))


def _thresholds_match(manifest: Mapping[str, Any], config: ResearchLoopConfig) -> bool:
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != {
        "minimum_improvement",
        "max_plant_regression",
    }:
        return False
    return (
        _same_finite_number(
            thresholds["minimum_improvement"], config.minimum_improvement
        )
        and _same_finite_number(
            thresholds["max_plant_regression"], config.max_plant_regression
        )
    )


def _seed_matches_config(manifest: Mapping[str, Any], config: ResearchLoopConfig) -> bool:
    try:
        aidm_config = _research_aidm_config(config)
    except (TypeError, ValueError, OverflowError):
        return False
    return type(manifest.get("seed")) is int and manifest["seed"] == aidm_config.seed


def _same_finite_number(left: object, right: object) -> bool:
    if isinstance(left, bool) or not isinstance(left, (int, float)):
        return False
    if isinstance(right, bool) or not isinstance(right, (int, float)):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
    except OverflowError:
        return False
    return (
        math.isfinite(left_number)
        and math.isfinite(right_number)
        and left_number == right_number
    )


def _selection_is_bounded(
    manifest: Mapping[str, Any],
    proposal: ResearchProposal,
) -> bool:
    try:
        selected_specs = manifest["selected_specs"]
        if not isinstance(selected_specs, list):
            return False
        parsed_specs = tuple(FeatureSpec.from_dict(spec) for spec in selected_specs)
        for spec in parsed_specs:
            validate_prediction_time_feature_spec(spec)
        selected_by_name = _canonical_specs(selected_specs)
        matching_sets = [
            feature_set
            for feature_set in proposal.feature_sets
            if _canonical_specs(
                [spec.to_dict() for spec in feature_set.specs]
            )
            == selected_by_name
        ]
        if len(matching_sets) != 1:
            return False
        recipe = manifest["selected_model_recipe"]
        winner = manifest["winner"]
        if not isinstance(recipe, Mapping) or not isinstance(winner, Mapping):
            return False
        recipe_name = recipe.get("name")
        feature_set_name = matching_sets[0].name
        if recipe_name == "selected_lightgbm":
            return _selected_search_matches(
                recipe,
                proposal,
                feature_set_name,
                winner.get("name"),
            )
        return (
            any(dict(candidate.to_dict()) == dict(recipe) for candidate in proposal.model_recipes)
            and winner.get("name") == f"{recipe_name}:{feature_set_name}"
        )
    except (KeyError, TypeError, ValueError, PromotionManifestError):
        return False


def _canonical_specs(value: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(value, key=lambda spec: spec["name"])


def _selected_search_matches(
    recipe: Mapping[str, Any],
    proposal: ResearchProposal,
    feature_set_name: str,
    winner_name: object,
) -> bool:
    if proposal.search is None or winner_name != f"selected_lightgbm:{feature_set_name}":
        return False
    search = recipe.get("search")
    if not isinstance(search, Mapping):
        return False
    try:
        feature_index = next(
            index
            for index, feature_set in enumerate(proposal.feature_sets)
            if feature_set.name == feature_set_name
        )
    except StopIteration:
        return False
    expected_space = proposal.search["spaces"]["lightgbm"]
    selected_trial = search.get("selected_trial_number")
    return (
        recipe.get("recipe") == "lightgbm"
        and recipe.get("parameters") == search.get("selected_trial_parameters")
        and search.get("sampler") == proposal.search["sampler"]
        and search.get("seed") == proposal.search["seed"] + feature_index
        and search.get("n_trials") == proposal.search["n_trials"]
        and search.get("space") == expected_space
        and search.get("feature_set") == feature_set_name
        and isinstance(selected_trial, int)
        and not isinstance(selected_trial, bool)
        and 0 <= selected_trial < proposal.search["n_trials"]
        and search.get("selected_trial_candidate_name")
        == f"optuna_lightgbm_{selected_trial}:{feature_set_name}"
    )


def _report_matches_manifest(report: str | None, manifest: Mapping[str, Any]) -> bool:
    if report is None:
        return False
    try:
        thresholds = manifest["thresholds"]
        winner = manifest["winner"]
        selected_specs = manifest["selected_specs"]
        if (
            not isinstance(thresholds, Mapping)
            or not isinstance(winner, Mapping)
            or not isinstance(selected_specs, list)
        ):
            return False
        winner_name = winner["name"]
        if type(winner_name) is not str:
            return False
        metrics = _report_metrics(winner["metrics"])
        threshold_markers = (
            "| max_plant_regression | "
            f"{_report_number(thresholds['max_plant_regression'])} |",
            "| minimum_improvement | "
            f"{_report_number(thresholds['minimum_improvement'])} |",
        )
        winner_marker = (
            f"| 1 | {winner_name} | {_report_number(metrics['mae'])} | "
            f"{_report_number(metrics['rmse'])} | {_report_number(metrics['nmae'])} |"
        )
        spec_markers = []
        for spec in selected_specs:
            if not isinstance(spec, Mapping):
                return False
            name = spec["name"]
            transform = spec["transform"]
            if type(name) is not str or type(transform) is not str:
                return False
            spec_markers.append(f"| {name} | {transform} |")
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        report.startswith("# Forecasting Performance Report\n")
        and f"Promotion decision: {manifest['decision']}\n" in report
        and f"Improvement ratio: {_report_number(manifest['improvement_ratio'])}\n"
        in report
        and "\n### Thresholds\n" in report
        and all(marker in report for marker in threshold_markers)
        and winner_marker in report
        and "## Selected feature specs\n" in report
        and all(marker in report for marker in spec_markers)
    )


def _report_metrics(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("winner metrics must be a mapping")
    normalized = {
        key.lower(): metric
        for key, metric in value.items()
        if type(key) is str
    }
    if set(("mae", "rmse", "nmae")) - set(normalized):
        raise ValueError("winner metrics are incomplete")
    return normalized


def _report_number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("report number must be numeric")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("report number must be finite") from exc
    if not math.isfinite(number):
        raise ValueError("report number must be finite")
    return f"{number:.6f}"


def _read_experiment_runs(
    database_path: Path,
    root: Path,
) -> tuple[dict[str, Any], ...] | None:
    safe_path = _safe_artifact_file(database_path, root)
    if safe_path is None:
        return None
    try:
        with closing(
            sqlite3.connect(
                f"{safe_path.as_uri()}?mode=ro",
                uri=True,
            )
        ) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT id, name, status, params_json, metrics_json, artifacts_json
                FROM runs ORDER BY rowid
                """
            ).fetchall()
    except (OSError, sqlite3.Error, ValueError):
        return None
    payloads = tuple(_decode_run_row(row) for row in rows)
    if any(payload is None for payload in payloads):
        return None
    return tuple(payload for payload in payloads if payload is not None)


def _selected_runs(
    rows: tuple[dict[str, Any], ...] | None,
    *,
    baseline_run_id: object,
    candidate_run_id: object,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if (
        rows is None
        or type(baseline_run_id) is not str
        or type(candidate_run_id) is not str
    ):
        return None
    by_id = {row["id"]: row for row in rows}
    baseline = by_id.get(baseline_run_id)
    candidate = by_id.get(candidate_run_id)
    if baseline is None or candidate is None:
        return None
    return baseline, candidate


def _proposal_runs_are_complete(
    rows: tuple[dict[str, Any], ...] | None,
    *,
    proposal: ResearchProposal,
    proposal_payload: Mapping[str, Any],
    config: ResearchLoopConfig,
) -> bool:
    if rows is None:
        return False
    try:
        aidm_config = _research_aidm_config(config)
        expected = _expected_proposal_runs(proposal)
        proposal_rows = [
            row
            for row in rows
            if row["params"].get("proposal_id") == proposal.proposal_id
        ]
        actual_names = [row["params"].get("candidate_name") for row in proposal_rows]
        if len(proposal_rows) != len(expected) or set(actual_names) != set(expected):
            return False
        for row in proposal_rows:
            params = row["params"]
            artifacts = row["artifacts"]
            candidate_name = params["candidate_name"]
            expected_run = expected[candidate_name]
            summary = artifacts.get("summary")
            if (
                row["status"] != "completed"
                or row["name"] != expected_run["run_name"]
                or not _matches_any_key_set(
                    params,
                    expected_run["params_key_sets"],
                )
                or not _matches_any_key_set(
                    artifacts,
                    expected_run["artifact_key_sets"],
                )
                or params.get("proposal") != proposal_payload
                or params.get("folds") != config.fold_count
                or type(params.get("seed")) is not int
                or params["seed"] != aidm_config.seed
                or params.get("schema_version") != _SCHEMA_VERSION
                or params.get("model") != expected_run["model"]
                or params.get("specs") != expected_run["specs"]
                or not isinstance(summary, Mapping)
                or summary.get("name") != candidate_name
                or summary.get("specs") != expected_run["specs"]
                or summary.get("model_recipe") != params.get("model_recipe")
            ):
                return False
            if expected_run["kind"] == "direct":
                if params.get("model_recipe") != expected_run["recipe"]:
                    return False
            elif not _search_run_is_bounded(
                params,
                expected_run,
                proposal.search,
            ):
                return False
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def _expected_proposal_runs(proposal: ResearchProposal) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for feature_index, feature_set in enumerate(proposal.feature_sets):
        specs = sorted(
            (spec.to_dict() for spec in feature_set.specs),
            key=lambda spec: spec["name"],
        )
        for recipe in proposal.model_recipes:
            candidate_name = f"{recipe.name}:{feature_set.name}"
            expected[candidate_name] = {
                "kind": "direct",
                "recipe": recipe.to_dict(),
                "model": model_definition_from_recipe(recipe).name,
                "run_name": f"aidm-proposal-{candidate_name}",
                "params_key_sets": (_PROPOSAL_RUN_PARAMS_KEYS,),
                "artifact_key_sets": (_PROPOSAL_RUN_ARTIFACT_KEYS,),
                "specs": specs,
            }
        if proposal.search is None:
            continue
        for trial_number in range(int(proposal.search["n_trials"])):
            candidate_name = f"optuna_lightgbm_{trial_number}:{feature_set.name}"
            expected[candidate_name] = {
                "kind": "trial",
                "feature_index": feature_index,
                "feature_set": feature_set.name,
                "model": f"Recipe:lightgbm:optuna_lightgbm_{trial_number}",
                "run_name": (
                    f"aidm-optuna-lightgbm-{feature_set.name}-trial-{trial_number}"
                ),
                "params_key_sets": (_SEARCH_PROPOSAL_RUN_PARAMS_KEYS,),
                "artifact_key_sets": (
                    _PROPOSAL_RUN_ARTIFACT_KEYS,
                    _REUSED_SEARCH_RUN_ARTIFACT_KEYS,
                ),
                "specs": specs,
                "trial_number": trial_number,
            }
        selected_name = f"selected_lightgbm:{feature_set.name}"
        expected[selected_name] = {
            "kind": "selected",
            "feature_index": feature_index,
            "feature_set": feature_set.name,
            "model": "Recipe:lightgbm:selected_lightgbm",
            "run_name": f"aidm-selected-lightgbm-{feature_set.name}",
            "params_key_sets": (_SEARCH_PROPOSAL_RUN_PARAMS_KEYS,),
            "artifact_key_sets": (_SELECTED_SEARCH_RUN_ARTIFACT_KEYS,),
            "specs": specs,
        }
    return expected


def _matches_any_key_set(
    value: object,
    expected_key_sets: tuple[frozenset[str], ...],
) -> bool:
    return isinstance(value, Mapping) and any(
        set(value) == expected_keys for expected_keys in expected_key_sets
    )


def _search_run_is_bounded(
    params: Mapping[str, Any],
    expected: Mapping[str, Any],
    search: Mapping[str, Any] | None,
) -> bool:
    if search is None:
        return False
    recipe = params.get("model_recipe")
    provenance = params.get("search")
    if not isinstance(recipe, Mapping) or not isinstance(provenance, Mapping):
        return False
    if recipe.get("name") not in {
        f"optuna_lightgbm_{expected.get('trial_number')}",
        "selected_lightgbm",
    }:
        return False
    if recipe.get("recipe") != "lightgbm" or recipe.get("search") != provenance:
        return False
    expected_seed = int(search["seed"]) + int(expected["feature_index"])
    if (
        provenance.get("sampler") != search["sampler"]
        or provenance.get("seed") != expected_seed
        or provenance.get("n_trials") != search["n_trials"]
        or provenance.get("feature_set") != expected["feature_set"]
        or provenance.get("space") != search["spaces"]["lightgbm"]
    ):
        return False
    if expected["kind"] == "trial":
        return provenance.get("trial_number") == expected["trial_number"]
    selected_trial = provenance.get("selected_trial_number")
    return (
        isinstance(selected_trial, int)
        and not isinstance(selected_trial, bool)
        and 0 <= selected_trial < int(search["n_trials"])
        and provenance.get("selected_trial_candidate_name")
        == f"optuna_lightgbm_{selected_trial}:{expected['feature_set']}"
        and provenance.get("selected_trial_parameters") == recipe.get("parameters")
    )


def _decode_run_row(row: sqlite3.Row) -> dict[str, Any] | None:
    try:
        params = json.loads(row["params_json"])
        metrics = json.loads(row["metrics_json"])
        artifacts = json.loads(row["artifacts_json"])
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not all(isinstance(value, Mapping) for value in (params, metrics, artifacts)):
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "params": params,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def _completed_runs(
    rows: tuple[dict[str, Any], dict[str, Any]] | None,
) -> bool:
    return rows is not None and all(row["status"] == "completed" for row in rows)


def _candidate_matches_manifest(
    experiment: ExperimentResult,
    evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
) -> bool:
    candidate = evidence["selected_candidate"]
    winner = manifest.get("winner")
    summary = candidate_row["artifacts"].get("summary")
    if not isinstance(winner, Mapping) or not isinstance(summary, Mapping):
        return False
    return (
        candidate["id"] == winner.get("name")
        and experiment.selected_candidate_id == winner.get("name")
        and candidate_row["status"] == "completed"
        and candidate_row["params"].get("candidate_name") == winner.get("name")
        and summary.get("name") == winner.get("name")
        and summary.get("metrics") == winner.get("metrics")
    )


def _baseline_matches_manifest(
    manifest: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    config: ResearchLoopConfig,
) -> bool:
    baseline = manifest.get("baseline")
    summary = baseline_row["artifacts"].get("summary")
    candidate = _candidate_from_run_row(baseline_row)
    try:
        aidm_config = _research_aidm_config(config)
    except (TypeError, ValueError, OverflowError):
        return False
    expected_run_id = _expected_manifest_run_id("baseline", candidate, aidm_config)
    if (
        not isinstance(baseline, Mapping)
        or not isinstance(summary, Mapping)
        or candidate is None
        or expected_run_id is None
    ):
        return False
    return (
        baseline.get("model") == "SPOT"
        and baseline_row["status"] == "completed"
        and baseline_row["name"] == "aidm-baseline-spot"
        and baseline_row["params"] == _baseline_params(aidm_config)
        and summary.get("name") == "baseline"
        and summary.get("specs") == []
        and summary.get("run_id") == baseline_row["id"]
        and "model_recipe" not in summary
        and summary.get("metrics") == baseline.get("metrics")
        and baseline.get("run_id") == expected_run_id
    )


def _baseline_params(config: AIDMConfig) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "candidate_name": "baseline",
        "model": "SPOT",
        "folds": config.folds,
        "seed": config.seed,
        "specs": [],
    }


def _winner_run_identity_matches(
    manifest: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    config: ResearchLoopConfig,
) -> bool:
    winner = manifest.get("winner")
    summary = candidate_row["artifacts"].get("summary")
    candidate = _candidate_from_run_row(candidate_row)
    try:
        aidm_config = _research_aidm_config(config)
    except (TypeError, ValueError, OverflowError):
        return False
    expected_run_id = _expected_manifest_run_id("winner", candidate, aidm_config)
    return (
        isinstance(winner, Mapping)
        and isinstance(summary, Mapping)
        and candidate is not None
        and expected_run_id is not None
        and summary.get("run_id") == candidate_row["id"]
        and candidate.name == winner.get("name")
        and winner.get("run_id") == expected_run_id
    )


def _expected_manifest_run_id(
    role: str,
    candidate: CandidateResult | None,
    config: AIDMConfig,
) -> str | None:
    if candidate is None or role not in {"baseline", "winner"}:
        return None
    try:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "role": role,
            "seed": config.seed,
            "folds": config.folds,
            "thresholds": {
                "minimum_improvement": float(config.minimum_improvement),
                "max_plant_regression": float(config.max_plant_regression),
            },
            "candidate": {
                "name": candidate.name,
                "metrics": {
                    key: float(candidate.metrics[key]) for key in sorted(candidate.metrics)
                },
                "specs": [spec.to_dict() for spec in candidate.specs],
            },
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"power-forecasting/aidm/manifest-run/{canonical}",
        )
    )


def _metrics_match_manifest(
    manifest: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
) -> bool:
    baseline = manifest.get("baseline")
    winner = manifest.get("winner")
    baseline_summary = baseline_row["artifacts"].get("summary")
    candidate_summary = candidate_row["artifacts"].get("summary")
    if not all(
        isinstance(value, Mapping)
        for value in (baseline, winner, baseline_summary, candidate_summary)
    ):
        return False
    return (
        baseline_row["metrics"] == baseline_summary.get("metrics")
        and baseline_row["metrics"] == baseline.get("metrics")
        and candidate_row["metrics"] == candidate_summary.get("metrics")
        and candidate_row["metrics"] == winner.get("metrics")
    )


def _specs_match(
    experiment: ExperimentResult,
    evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
) -> bool:
    candidate = evidence["selected_candidate"]
    summary = candidate_row["artifacts"].get("summary")
    if not isinstance(summary, Mapping):
        return False
    specs = summary.get("specs")
    if not isinstance(specs, list):
        return False
    checksum = _safe_json_sha256(specs)
    if checksum is None:
        return False
    return (
        specs == manifest.get("selected_specs")
        and candidate_row["params"].get("specs") == specs
        and candidate["spec_sha256"] == checksum
        and experiment.selected_candidate_spec_sha256 == checksum
    )


def _recipe_matches(
    experiment: ExperimentResult,
    evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
) -> bool:
    candidate = evidence["selected_candidate"]
    recipe = manifest.get("selected_model_recipe")
    summary = candidate_row["artifacts"].get("summary")
    if not isinstance(recipe, Mapping) or not isinstance(summary, Mapping):
        return False
    recipe_payload = dict(recipe)
    recipe_name = recipe_payload.get("name")
    recipe_sha256 = _safe_json_sha256(recipe_payload)
    if recipe_sha256 is None:
        return False
    return (
        type(recipe_name) is str
        and candidate["recipe_id"] == recipe_name
        and experiment.selected_recipe_id == recipe_name
        and candidate["recipe_sha256"] == recipe_sha256
        and summary.get("model_recipe") == recipe_payload
        and candidate_row["params"].get("model_recipe") == recipe_payload
    )


def _gate_outcome_matches(
    manifest: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    candidate_row: Mapping[str, Any],
    config: ResearchLoopConfig,
) -> bool:
    baseline = _candidate_from_run_row(baseline_row)
    candidate = _candidate_from_run_row(candidate_row)
    if baseline is None or candidate is None:
        return False
    try:
        gates = evaluate_promotion_gates(
            baseline,
            candidate,
            _research_aidm_config(config),
        )
    except (TypeError, ValueError, OverflowError):
        return False
    return (
        manifest.get("decision") == gates["decision"]
        and manifest.get("failed_gates") == gates["failed_gates"]
        and manifest.get("per_plant_deltas") == gates["per_plant_deltas"]
        and manifest.get("improvement_ratio") == gates["improvement_ratio"]
    )


def _candidate_from_run_row(row: Mapping[str, Any]) -> CandidateResult | None:
    summary = row["artifacts"].get("summary")
    if not isinstance(summary, Mapping):
        return None
    specs = summary.get("specs")
    if not isinstance(specs, list):
        return None
    try:
        return CandidateResult(
            name=summary["name"],
            specs=tuple(FeatureSpec.from_dict(spec) for spec in specs),
            metrics=row["metrics"],
            per_plant=summary["per_plant"],
            run_id=row["id"],
            model_recipe=summary.get("model_recipe"),
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _finish_verification(
    checks: dict[str, bool],
    report_path: Path,
    *,
    decision: str | None,
    proposal_sha256: str,
    observed_checksums: Mapping[str, str],
) -> VerificationResult:
    checks["verification_report"] = True
    passed = all(checks.values())
    reasons = _verification_reasons(checks, decision)
    status = (
        "pass"
        if passed
        else "reject"
        if decision == "reject"
        and all(value for name, value in checks.items() if name != "promoted")
        else "invalid"
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "passed": passed,
        "checks": dict(checks),
        "reasons": list(reasons),
        "provenance": {
            "proposal_sha256": proposal_sha256,
            **dict(observed_checksums),
        },
    }
    try:
        atomic_write_json(report_path, payload)
    except (OSError, TypeError, ValueError):
        checks["verification_report"] = False
        passed = False
        reasons = _verification_reasons(checks, decision)
    return VerificationResult(
        passed=passed,
        checks=dict(checks),
        reasons=reasons,
        report_path=report_path,
    )


def _verification_reasons(
    checks: Mapping[str, bool],
    decision: str | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for name in _VERIFICATION_CHECKS:
        if checks.get(name):
            continue
        if name == "promoted" and decision == "reject":
            reasons.append("experiment_rejected")
        else:
            reasons.append(f"check_failed:{name}")
    return tuple(reasons)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "ExperimentResult",
    "ResearchExecutionError",
    "VerificationResult",
    "run_experiment_agent",
    "run_verifier_agent",
]
