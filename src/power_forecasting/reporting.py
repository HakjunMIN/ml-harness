from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from power_forecasting.aidm import AIDMResult, CandidateResult
from power_forecasting.evaluation import EvaluationResult
from power_forecasting.features import FeatureSpec
from power_forecasting.models import SUPPORTED_MODEL_NAMES


def render_performance_report(
    dataset_summary: Mapping,
    legacy_results: Mapping[str, EvaluationResult],
    aidm_result: AIDMResult,
    artifact_paths: Mapping[str, Path | str],
) -> str:
    _validate_report_inputs(dataset_summary, legacy_results, aidm_result, artifact_paths)
    lines: list[str] = [
        "# Forecasting Performance Report",
        "",
        "## Data summary",
        "",
    ]
    lines.extend(
        _table(
            ("Field", "Value"),
            (
                ("Rows", str(_summary_value(dataset_summary, ("rows", "row_count")))),
                ("Plants", str(_summary_value(dataset_summary, ("plants", "plant_count")))),
                (
                    "Time range",
                    (
                        f"{_summary_value(dataset_summary, ('time_start', 'start', 'start_time'))}"
                        " to "
                        f"{_summary_value(dataset_summary, ('time_end', 'end', 'end_time'))}"
                    ),
                ),
            ),
        )
    )
    lines.extend(["", "## Legacy model comparison", ""])
    lines.extend(
        _table(
            ("Model", "MAE", "RMSE", "NMAE"),
            tuple(
                (
                    name,
                    _metric(legacy_results[name], "MAE"),
                    _metric(legacy_results[name], "RMSE"),
                    _metric(legacy_results[name], "NMAE"),
                )
                for name in SUPPORTED_MODEL_NAMES
            ),
        )
    )

    lines.extend(["", "## Ranked AIDM candidates", ""])
    candidate_by_name = {candidate.name: candidate for candidate in aidm_result.candidates}
    ranked_rows = []
    for rank, name in enumerate(aidm_result.ranking, start=1):
        candidate = candidate_by_name.get(name)
        if candidate is None:
            raise ValueError(f"AIDM ranking references unknown candidate: {name}")
        ranked_rows.append(
            (
                str(rank),
                candidate.name,
                _candidate_metric(candidate, "MAE"),
                _candidate_metric(candidate, "RMSE"),
                _candidate_metric(candidate, "NMAE"),
            )
        )
    lines.extend(_table(("Rank", "Candidate", "MAE", "RMSE", "NMAE"), tuple(ranked_rows)))

    manifest = aidm_result.manifest
    lines.extend(["", "## Promotion decision", ""])
    lines.append(f"Promotion decision: {_markdown_text(_manifest_value(manifest, 'decision'))}")
    lines.append(
        f"Improvement ratio: {_format_number(_manifest_value(manifest, 'improvement_ratio'))}"
    )
    lines.extend(["", "### Thresholds", ""])
    thresholds = _mapping_value(manifest, "thresholds")
    lines.extend(
        _table(
            ("Threshold", "Value"),
            tuple((key, thresholds[key]) for key in sorted(thresholds)),
        )
    )

    lines.extend(["", "### Per-plant deltas", ""])
    per_plant_deltas = _mapping_value(manifest, "per_plant_deltas")
    if per_plant_deltas:
        lines.extend(
            _table(
                ("Plant", "NMAE delta"),
                tuple(
                    (plant_id, per_plant_deltas[plant_id])
                    for plant_id in sorted(per_plant_deltas, key=str)
                ),
            )
        )
    else:
        lines.append("None")

    lines.extend(["", "## Failed gates", ""])
    failed_gates = tuple(_sequence_value(manifest, "failed_gates"))
    if failed_gates:
        lines.extend(
            _table(
                ("Gate",),
                tuple((gate,) for gate in failed_gates),
            )
        )
    else:
        lines.append("None")

    lines.extend(["", "## Selected feature specs", ""])
    specs = tuple(aidm_result.winner.specs) if aidm_result.winner is not None else ()
    if specs:
        lines.extend(
            _table(
                ("Name", "Transform", "Inputs", "Parameters", "Version", "Rationale"),
                tuple(
                    (
                        spec.name,
                        spec.transform,
                        ", ".join(spec.inputs),
                        _json_for_report(spec.parameters),
                        spec.version,
                        spec.rationale,
                    )
                    for spec in sorted(specs, key=lambda feature: feature.name)
                ),
            )
        )
    else:
        lines.append("None")

    lines.extend(["", "## Artifact paths", ""])
    lines.extend(
        _table(
            ("Artifact", "Path"),
            tuple((key, str(artifact_paths[key])) for key in sorted(artifact_paths, key=str)),
        )
    )
    return "\n".join(lines) + "\n"


def write_performance_report(
    dataset_summary: Mapping,
    legacy_results: Mapping[str, EvaluationResult],
    aidm_result: AIDMResult,
    artifact_paths: Mapping[str, Path | str],
    *,
    target: Path,
) -> Path:
    target = Path(target)
    content = render_performance_report(
        dataset_summary, legacy_results, aidm_result, artifact_paths
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(target, content)
    return target


def _validate_report_inputs(
    dataset_summary: Mapping,
    legacy_results: Mapping[str, EvaluationResult],
    aidm_result: AIDMResult,
    artifact_paths: Mapping[str, Path | str],
) -> None:
    if not isinstance(dataset_summary, Mapping):
        raise TypeError("dataset_summary must be a mapping")
    if not isinstance(legacy_results, Mapping):
        raise TypeError("legacy_results must be a mapping")
    missing = [name for name in SUPPORTED_MODEL_NAMES if name not in legacy_results]
    if missing:
        raise ValueError(f"missing legacy results: {missing}")
    for name in SUPPORTED_MODEL_NAMES:
        if not isinstance(legacy_results[name], EvaluationResult):
            raise TypeError(f"legacy result {name} must be an EvaluationResult")
    if not isinstance(aidm_result, AIDMResult):
        raise TypeError("aidm_result must be an AIDMResult")
    if not isinstance(artifact_paths, Mapping):
        raise TypeError("artifact_paths must be a mapping")


def _table(headers: Sequence[Any], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *(
            "| " + " | ".join(_markdown_cell(value) for value in row) + " |"
            for row in rows
        ),
    ]


def _markdown_cell(value: Any) -> str:
    return _markdown_text(_display_value(value))


def _markdown_text(value: Any) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")
    return text


def _display_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _format_number(value)
    return "" if value is None else str(value)


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("numeric value must not be bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("numeric value must be finite")
    return f"{number:.6f}"


def _summary_value(summary: Mapping, keys: Sequence[str]) -> Any:
    for key in keys:
        if key in summary:
            return summary[key]
    return ""


def _metric(result: EvaluationResult, name: str) -> float:
    return _metric_from_mapping(result.metrics, name)


def _candidate_metric(candidate: CandidateResult, name: str) -> float:
    return _metric_from_mapping(candidate.metrics, name)


def _metric_from_mapping(metrics: Mapping[str, Any], name: str) -> float:
    for key, value in metrics.items():
        if str(key).lower() == name.lower():
            return float(value)
    raise ValueError(f"metrics must include {name}")


def _manifest_value(manifest: Mapping[str, Any], key: str) -> Any:
    if key not in manifest:
        raise ValueError(f"manifest missing {key}")
    return manifest[key]


def _mapping_value(manifest: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = _manifest_value(manifest, key)
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest {key} must be a mapping")
    return value


def _sequence_value(manifest: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = _manifest_value(manifest, key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"manifest {key} must be a sequence")
    return value


def _json_for_report(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, FeatureSpec):
        return value.to_dict()
    return value


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


__all__ = [
    "render_performance_report",
    "write_performance_report",
]
