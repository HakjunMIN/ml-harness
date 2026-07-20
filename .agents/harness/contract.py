from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1"
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "legacy_command",
        "input_dataset",
        "predictions_output",
        "required_prediction_columns",
        "timeout_seconds",
    }
)
HARNESS_ENV_KEYS = (
    "HARNESS_INPUT_DATASET",
    "HARNESS_PREDICTIONS_OUTPUT",
    "HARNESS_RUN_DIR",
)


class AdapterContractError(ValueError):
    """Raised when an adapter manifest or execution violates the harness contract."""


@dataclass(frozen=True)
class AdapterConfig:
    schema_version: str
    legacy_command: tuple[str, ...]
    input_dataset: Path
    predictions_output: Path
    required_prediction_columns: tuple[str, ...]
    timeout_seconds: int
    manifest_path: Path
    manifest_dir: Path
    input_dataset_relative: str
    predictions_output_relative: str


def load_adapter(path: Path | str) -> AdapterConfig:
    manifest_path = Path(path).resolve()
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise AdapterContractError(f"invalid adapter JSON: {exc.msg}") from exc
    except OSError as exc:
        raise AdapterContractError(f"adapter manifest cannot be read: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise AdapterContractError("adapter manifest must be a JSON object")
    unknown = sorted(set(payload) - MANIFEST_FIELDS)
    missing = sorted(MANIFEST_FIELDS - set(payload))
    if unknown:
        raise AdapterContractError(f"unknown adapter fields: {unknown}")
    if missing:
        raise AdapterContractError(f"missing adapter fields: {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise AdapterContractError("schema_version must be exactly '1'")

    command = _string_tuple(payload["legacy_command"], "legacy_command")
    if not command:
        raise AdapterContractError("legacy_command must be a non-empty array")
    columns = _string_tuple(
        payload["required_prediction_columns"], "required_prediction_columns"
    )
    if not columns:
        raise AdapterContractError("required_prediction_columns must be a non-empty array")
    if len(set(columns)) != len(columns):
        raise AdapterContractError("required_prediction_columns must not contain duplicates")

    timeout = payload["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise AdapterContractError("timeout_seconds must be an integer")
    if not 1 <= timeout <= 3600:
        raise AdapterContractError("timeout_seconds must be between 1 and 3600")

    manifest_dir = manifest_path.parent.resolve()
    input_relative = _relative_path_string(payload["input_dataset"], "input_dataset")
    output_relative = _relative_path_string(
        payload["predictions_output"], "predictions_output"
    )
    input_path = _contained_path(manifest_dir, input_relative, "input_dataset")
    output_path = _contained_path(manifest_dir, output_relative, "predictions_output")
    if not input_path.exists():
        raise AdapterContractError(f"input_dataset not found: {input_relative}")
    if not input_path.is_file():
        raise AdapterContractError(f"input_dataset must be a file: {input_relative}")

    return AdapterConfig(
        schema_version=SCHEMA_VERSION,
        legacy_command=command,
        input_dataset=input_path,
        predictions_output=output_path,
        required_prediction_columns=columns,
        timeout_seconds=timeout,
        manifest_path=manifest_path,
        manifest_dir=manifest_dir,
        input_dataset_relative=input_relative,
        predictions_output_relative=output_relative,
    )


def run_adapter(adapter_path: Path | str, run_dir: Path | str, run_id: str | None = None) -> Path:
    adapter = load_adapter(adapter_path)
    run_root = Path(run_dir).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(run_root, 0o700)
    except OSError:
        pass
    evidence_path = run_root / "legacy-evidence.json"
    started = time.monotonic()
    run_name = run_id or "legacy-run"
    output_created = False
    exit_code: int | None = None

    try:
        adapter.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        env = {
            "HARNESS_INPUT_DATASET": str(adapter.input_dataset),
            "HARNESS_PREDICTIONS_OUTPUT": str(adapter.predictions_output),
            "HARNESS_RUN_DIR": str(run_root),
        }
        completed = subprocess.run(
            list(adapter.legacy_command),
            shell=False,
            cwd=str(adapter.manifest_dir),
            env=env,
            text=True,
            capture_output=True,
            timeout=adapter.timeout_seconds,
            check=False,
        )
        exit_code = completed.returncode
        if completed.returncode != 0:
            raise AdapterContractError(
                f"legacy command failed with exit code {completed.returncode}"
            )
        output_created = adapter.predictions_output.exists()
        prediction_info = _validate_predictions(
            adapter.predictions_output,
            adapter.required_prediction_columns,
            adapter.predictions_output_relative,
        )
        evidence = _base_evidence(adapter, run_name, started)
        evidence.update(
            {
                "status": "success",
                "exit_code": completed.returncode,
                "predictions_output": prediction_info,
            }
        )
        _write_json(evidence_path, evidence)
        return evidence_path
    except subprocess.TimeoutExpired as exc:
        evidence = _failure_evidence(adapter, run_name, started, "legacy command timed out")
        evidence["exit_code"] = None
        _write_json(evidence_path, evidence)
        raise AdapterContractError("legacy command timed out") from exc
    except AdapterContractError as exc:
        evidence = _failure_evidence(adapter, run_name, started, str(exc))
        evidence["exit_code"] = exit_code
        if output_created or adapter.predictions_output.exists():
            try:
                evidence["predictions_output"] = _output_checksum_only(
                    adapter.predictions_output, adapter.predictions_output_relative
                )
            except AdapterContractError:
                pass
        _write_json(evidence_path, evidence)
        raise


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise AdapterContractError(f"{label} must be an array")
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str or item == "":
            raise AdapterContractError(f"{label}[{index}] must be a non-empty string")
        result.append(item)
    return tuple(result)


def _relative_path_string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise AdapterContractError(f"{label} must be a non-empty string")
    candidate = Path(value)
    if candidate.is_absolute():
        raise AdapterContractError(f"{label} must be relative to the manifest directory")
    return value


def _contained_path(manifest_dir: Path, relative: str, label: str) -> Path:
    target = (manifest_dir / relative).resolve()
    try:
        target.relative_to(manifest_dir)
    except ValueError as exc:
        raise AdapterContractError(f"{label} escapes manifest directory") from exc
    return target


def _validate_predictions(
    path: Path, required_columns: Sequence[str], relative_path: str
) -> dict[str, Any]:
    if not path.exists():
        raise AdapterContractError("predictions_output was not created")
    if not path.is_file():
        raise AdapterContractError("predictions_output must be a file")
    if path.stat().st_size <= 0:
        raise AdapterContractError("predictions_output must be non-empty")

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except csv.Error as exc:
        raise AdapterContractError(f"predictions_output is not valid CSV: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise AdapterContractError("predictions_output must be UTF-8 CSV") from exc

    if not rows or not rows[0] or any(column == "" for column in rows[0]):
        raise AdapterContractError("predictions_output must contain a non-empty header")
    header = rows[0]
    if len(set(header)) != len(header):
        raise AdapterContractError("predictions_output header columns must be unique")
    data_rows = rows[1:]
    if not data_rows:
        raise AdapterContractError("predictions_output must contain data rows")
    if any(row == header for row in data_rows):
        raise AdapterContractError("predictions_output must contain exactly one header row")
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise AdapterContractError(f"missing required prediction columns: {missing}")
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "columns": header,
        "rows": len(data_rows),
    }


def _output_checksum_only(path: Path, relative_path: str) -> dict[str, Any]:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        raise AdapterContractError("predictions_output is unavailable")
    return {"relative_path": relative_path, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _base_evidence(adapter: AdapterConfig, run_id: str, started: float) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "adapter_manifest": {
            "sha256": sha256_file(adapter.manifest_path),
            "schema_version": adapter.schema_version,
        },
        "command": list(adapter.legacy_command),
        "environment": list(HARNESS_ENV_KEYS),
        "input_dataset": {
            "relative_path": adapter.input_dataset_relative,
            "sha256": sha256_file(adapter.input_dataset),
            "bytes": adapter.input_dataset.stat().st_size,
        },
        "required_prediction_columns": list(adapter.required_prediction_columns),
        "duration_seconds": round(time.monotonic() - started, 6),
    }


def _failure_evidence(adapter: AdapterConfig, run_id: str, started: float, error: str) -> dict[str, Any]:
    evidence = _base_evidence(adapter, run_id, started)
    evidence.update({"status": "failure", "error": error})
    return evidence


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m harness.contract")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        evidence = run_adapter(args.adapter, args.run_dir, run_id=args.run_id)
    except AdapterContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AdapterConfig",
    "AdapterContractError",
    "load_adapter",
    "main",
    "run_adapter",
    "sha256_file",
]
