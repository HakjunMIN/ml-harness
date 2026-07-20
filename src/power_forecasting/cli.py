from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from power_forecasting.aidd import (
    PromotionManifestError,
    render_promoted_module,
)
from power_forecasting.aidm import AIDMConfig, AIDMResult, run_aidm
from power_forecasting.data import (
    DataContractError,
    generate_synthetic_data,
    parse_timestamps,
    validate_dataset,
)
from power_forecasting.evaluation import EvaluationResult, evaluate_model
from power_forecasting.models import SUPPORTED_MODEL_NAMES, model_definition
from power_forecasting.reporting import write_performance_report


DATASET_NAME = "dataset.csv"
DATABASE_NAME = "experiments.db"
MANIFEST_NAME = "promotion_manifest.json"
GENERATED_MODULE_NAME = "promoted_features.py"
REPORT_NAME = "performance_report.md"


def run_generate_data(output: Path, days: int, plants: int, seed: int) -> Path:
    output_root = _ensure_output_root(output)
    frame = generate_synthetic_data(days=days, plants=plants, seed=seed)
    validate_dataset(frame)
    target = output_root / DATASET_NAME
    _atomic_write_text(target, frame.to_csv(index=False, lineterminator="\n"))
    return target


def run_legacy(
    output: Path, dataset: Path | None = None, folds: int = 3
) -> dict[str, EvaluationResult]:
    output_root = _ensure_output_root(output)
    _validate_folds(folds)
    frame = _load_dataset(_dataset_path(output_root, dataset))
    return {
        name: _evaluate_legacy_model(frame, name, folds)
        for name in SUPPORTED_MODEL_NAMES
    }


def run_aidm_workflow(
    output: Path, dataset: Path | None = None, config: AIDMConfig = AIDMConfig()
) -> AIDMResult:
    output_root = _ensure_output_root(output)
    if not isinstance(config, AIDMConfig):
        raise TypeError("config must be an AIDMConfig")
    frame = _load_dataset(_dataset_path(output_root, dataset))
    result = run_aidm(frame, output_root / DATABASE_NAME, config)
    _write_json_atomic(output_root / MANIFEST_NAME, result.manifest)
    return result


def run_aidd_workflow(output: Path, manifest: Path) -> Path:
    output_root = _ensure_output_root(output)
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        payload = json.load(handle)
    return render_promoted_module(payload, output_root / GENERATED_MODULE_NAME)


def run_all(
    output: Path,
    days: int = 60,
    plants: int = 3,
    seed: int = 42,
    folds: int = 3,
) -> dict[str, Path]:
    output_root = _ensure_output_root(output)
    _validate_folds(folds)
    dataset_path = run_generate_data(output_root, days=days, plants=plants, seed=seed)
    legacy_results = run_legacy(output_root, dataset=dataset_path, folds=folds)
    aidm_result = run_aidm_workflow(
        output_root,
        dataset=dataset_path,
        config=AIDMConfig(
            folds=folds,
            minimum_improvement=0.0,
            max_plant_regression=0.2,
            seed=seed,
        ),
    )
    manifest_path = output_root / MANIFEST_NAME
    _write_json_atomic(manifest_path, aidm_result.manifest)
    if aidm_result.manifest.get("decision") != "promote":
        failed_gates = aidm_result.manifest.get("failed_gates") or ["none reported"]
        raise RuntimeError(
            "AIDM rejected promotion; failed gates: "
            + ", ".join(str(gate) for gate in failed_gates)
        )

    generated_module = run_aidd_workflow(output_root, manifest=manifest_path)
    report_path = output_root / REPORT_NAME
    artifact_paths = {
        "dataset": dataset_path,
        "database": output_root / DATABASE_NAME,
        "manifest": manifest_path,
        "generated_module": generated_module,
        "report": report_path,
    }
    frame = _load_dataset(dataset_path)
    write_performance_report(
        _dataset_summary(frame),
        legacy_results,
        aidm_result,
        artifact_paths,
        target=report_path,
    )
    return artifact_paths


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate-data":
            path = run_generate_data(args.output, args.days, args.plants, args.seed)
            print(f"dataset: {path}")
        elif args.command == "legacy":
            results = run_legacy(args.output, dataset=args.dataset, folds=args.folds)
            for name, result in results.items():
                print(
                    f"{name}: MAE={_metric(result.metrics, 'MAE'):.6f} "
                    f"RMSE={_metric(result.metrics, 'RMSE'):.6f} "
                    f"NMAE={_metric(result.metrics, 'NMAE'):.6f}"
                )
        elif args.command == "aidm":
            config = AIDMConfig(
                folds=args.folds,
                minimum_improvement=args.minimum_improvement,
                max_plant_regression=args.max_plant_regression,
                top_single_candidates=args.top_single_candidates,
                seed=args.seed,
            )
            result = run_aidm_workflow(args.output, dataset=args.dataset, config=config)
            output_root = Path(args.output)
            print(f"database: {output_root / DATABASE_NAME}")
            print(f"manifest: {output_root / MANIFEST_NAME}")
            print(f"decision: {result.manifest['decision']}")
        elif args.command == "aidd":
            path = run_aidd_workflow(args.output, manifest=args.manifest)
            print(f"generated_module: {path}")
        elif args.command == "all":
            paths = run_all(
                args.output,
                days=args.days,
                plants=args.plants,
                seed=args.seed,
                folds=args.folds,
            )
            for key, path in paths.items():
                print(f"{key}: {path}")
        else:
            raise ValueError(f"unknown command: {args.command}")
    except (
        DataContractError,
        PromotionManifestError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="power-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument("--output", required=True, type=Path)

    generate = subparsers.add_parser("generate-data", parents=[output_parent])
    generate.add_argument("--days", type=int, default=60)
    generate.add_argument("--plants", type=int, default=3)
    generate.add_argument("--seed", type=int, default=42)

    legacy = subparsers.add_parser("legacy", parents=[output_parent])
    legacy.add_argument("--dataset", type=Path)
    legacy.add_argument("--folds", type=int, default=3)

    default_aidm = AIDMConfig()
    aidm_parser = subparsers.add_parser("aidm", parents=[output_parent])
    aidm_parser.add_argument("--dataset", type=Path)
    aidm_parser.add_argument("--folds", type=int, default=default_aidm.folds)
    aidm_parser.add_argument(
        "--minimum-improvement", type=float, default=default_aidm.minimum_improvement
    )
    aidm_parser.add_argument(
        "--max-plant-regression", type=float, default=default_aidm.max_plant_regression
    )
    aidm_parser.add_argument(
        "--top-single-candidates", type=int, default=default_aidm.top_single_candidates
    )
    aidm_parser.add_argument("--seed", type=int, default=default_aidm.seed)

    aidd_parser = subparsers.add_parser("aidd", parents=[output_parent])
    aidd_parser.add_argument("--manifest", required=True, type=Path)

    all_parser = subparsers.add_parser("all", parents=[output_parent])
    all_parser.add_argument("--days", type=int, default=60)
    all_parser.add_argument("--plants", type=int, default=3)
    all_parser.add_argument("--seed", type=int, default=42)
    all_parser.add_argument("--folds", type=int, default=3)
    return parser


def _ensure_output_root(output: Path) -> Path:
    output_root = Path(output)
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def _dataset_path(output: Path, dataset: Path | None) -> Path:
    return Path(dataset) if dataset is not None else Path(output) / DATASET_NAME


def _load_dataset(path: Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    if "timestamp" in frame.columns:
        frame = frame.copy()
        frame["timestamp"] = parse_timestamps(
            frame["timestamp"],
            error_message="invalid timestamps: unparseable values",
            error_type=DataContractError,
        )
    validate_dataset(frame)
    return frame


def _dataset_summary(frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = parse_timestamps(
        frame["timestamp"],
        error_message="invalid timestamps: unparseable values",
        error_type=DataContractError,
    )
    return {
        "rows": int(len(frame)),
        "plants": int(frame["plant_id"].nunique()),
        "time_start": timestamps.min().isoformat(),
        "time_end": timestamps.max().isoformat(),
    }


def _write_json_atomic(target: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, content)


def _atomic_write_text(target: Path, content: str) -> None:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    temp_path: Path | None = None
    created = False
    try:
        for attempt in range(100):
            candidate = target.with_name(f".{target.name}.{os.getpid()}.{attempt}.tmp")
            try:
                with candidate.open("x", encoding="utf-8", newline="\n") as handle:
                    temp_path = candidate
                    created = True
                    handle.write(normalized)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"could not create temp file for {target}")
        os.replace(temp_path, target)
        created = False
    finally:
        if created and temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _validate_folds(folds: int) -> None:
    if isinstance(folds, bool) or not isinstance(folds, int):
        raise TypeError("folds must be an integer")
    if folds < 1:
        raise ValueError("folds must be >= 1")


def _evaluate_legacy_model(
    frame: pd.DataFrame, name: str, folds: int
) -> EvaluationResult:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
            module=r"sklearn\.linear_model\._base",
        )
        return evaluate_model(frame, model_definition(name), feature_specs=[], folds=folds)


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    for key, value in metrics.items():
        if str(key).lower() == name.lower():
            return float(value)
    raise ValueError(f"metrics must include {name}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_aidd_workflow",
    "run_aidm_workflow",
    "run_all",
    "run_generate_data",
    "run_legacy",
]
