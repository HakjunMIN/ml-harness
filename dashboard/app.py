from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from power_forecasting.experiments import ExperimentStore


EXPECTED_ARTIFACTS = {
    "dataset": Path("dataset.csv"),
    "database": Path("experiments.db"),
    "manifest": Path("promotion_manifest.json"),
    "report": Path("performance_report.md"),
    "generated_module": Path("generated") / "promoted_features.py",
}
INSTALL_COMMAND = "pip install 'power-forecasting[dashboard]'"


def discover_artifacts(root: Path) -> dict[str, Path]:
    artifact_root = Path(root)
    if not artifact_root.exists():
        raise FileNotFoundError(f"artifact root not found: {artifact_root}")
    if not artifact_root.is_dir():
        raise NotADirectoryError(f"artifact root is not a directory: {artifact_root}")
    return {
        name: artifact_root / relative
        for name, relative in EXPECTED_ARTIFACTS.items()
        if (artifact_root / relative).exists()
    }


def load_dashboard_data(root: Path) -> dict[str, Any]:
    artifact_root = Path(root)
    artifacts = discover_artifacts(artifact_root)
    manifest_path = artifact_root / EXPECTED_ARTIFACTS["manifest"]
    if "manifest" not in artifacts:
        raise FileNotFoundError(f"required promotion manifest not found: {manifest_path}")

    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(f"{manifest_path}: {exc.msg}", exc.doc, exc.pos) from exc

    report = None
    if "report" in artifacts:
        report = artifacts["report"].read_text(encoding="utf-8")

    runs: list[dict[str, Any]] = []
    if "database" in artifacts:
        runs = ExperimentStore(artifacts["database"]).list_runs()

    return {
        "root": artifact_root,
        "artifacts": artifacts,
        "manifest": manifest,
        "report": report,
        "runs": runs,
        "candidate_ranking": _candidate_ranking(runs, manifest),
        "selected_specs": manifest.get("selected_specs", []) if isinstance(manifest, Mapping) else [],
    }


def run_dashboard(root: Path) -> None:
    data = load_dashboard_data(root)
    import streamlit as st

    _render_dashboard(st, data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a Streamlit dashboard for power forecasting demo artifacts."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/demo"),
        help="artifact root produced by the demo workflow (default: artifacts/demo)",
    )
    args = parser.parse_args(argv)

    try:
        run_dashboard(args.artifacts)
    except ModuleNotFoundError as exc:
        if exc.name != "streamlit":
            raise
        print(
            f"ERROR: Streamlit is not installed. Install the dashboard extras with: {INSTALL_COMMAND}",
            file=sys.stderr,
        )
        return 2
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def _render_dashboard(st, data: Mapping[str, Any]) -> None:
    manifest = data["manifest"]
    winner = manifest.get("winner") if isinstance(manifest, Mapping) else None
    winner = winner if isinstance(winner, Mapping) else {}
    metrics = winner.get("metrics", {}) if isinstance(winner, Mapping) else {}
    metrics = metrics if isinstance(metrics, Mapping) else {}

    st.title("Power Forecasting Demo Dashboard")
    st.subheader("Promotion manifest")
    st.metric("Promotion decision", manifest.get("decision", "unknown"))
    st.metric("Winner", winner.get("name", "none"))

    st.subheader("Winner metrics")
    if metrics:
        for name in sorted(metrics):
            st.metric(str(name).upper(), metrics[name])
    else:
        st.write("No winner metrics found in the manifest.")

    st.subheader("AIDM candidate ranking")
    ranking = data.get("candidate_ranking", [])
    if ranking:
        _render_table(st, ranking)
    else:
        st.write("No completed AIDM candidate rankings found.")

    st.subheader("Experiment runs")
    runs = data.get("runs", [])
    if runs:
        _render_table(st, runs)
    else:
        st.write("No experiment database runs found.")

    st.subheader("Performance report")
    report = data.get("report")
    if report:
        st.markdown(report)
    else:
        st.write("No performance_report.md artifact found.")

    st.subheader("Selected feature specs")
    specs = data.get("selected_specs", [])
    if specs:
        st.json(specs)
    else:
        st.write("No selected feature specs recorded.")

    st.subheader("Discovered artifacts")
    _render_table(
        st,
        [
            {"name": name, "path": str(path)}
            for name, path in sorted(data.get("artifacts", {}).items())
        ],
    )


def _render_table(st, rows: list[dict[str, Any]]) -> None:
    if hasattr(st, "dataframe"):
        st.dataframe(rows)
    else:
        st.table(rows)


def _candidate_ranking(runs: list[dict[str, Any]], manifest: Any) -> list[dict[str, Any]]:
    rows = [_ranking_row_from_run(run) for run in runs]
    rows = [row for row in rows if row is not None]

    manifest_row = _ranking_row_from_manifest(manifest)
    if manifest_row is not None and not _has_ranking_row(rows, manifest_row):
        rows.append(manifest_row)

    rows.sort(key=lambda row: (_ranking_sort_value(row.get("nmae")), str(row["candidate"])))
    return [{**row, "rank": rank} for rank, row in enumerate(rows, start=1)]


def _ranking_row_from_run(run: Mapping[str, Any]) -> dict[str, Any] | None:
    if run.get("status") != "completed":
        return None

    candidate = _candidate_name_from_run(run)
    if candidate is None or candidate == "baseline":
        return None

    metrics = _mapping(run.get("metrics"))
    return {
        "rank": 0,
        "candidate": candidate,
        "run_id": _safe_text(run.get("id")),
        "mae": _safe_number(metrics.get("mae")),
        "rmse": _safe_number(metrics.get("rmse")),
        "nmae": _safe_number(metrics.get("nmae")),
        "source": "experiment",
    }


def _ranking_row_from_manifest(manifest: Any) -> dict[str, Any] | None:
    if not isinstance(manifest, Mapping):
        return None
    winner = manifest.get("winner")
    if not isinstance(winner, Mapping):
        return None
    candidate = _safe_text(winner.get("name"))
    if candidate is None:
        return None
    metrics = _mapping(winner.get("metrics"))
    return {
        "rank": 0,
        "candidate": candidate,
        "run_id": _safe_text(winner.get("run_id")),
        "mae": _safe_number(metrics.get("mae")),
        "rmse": _safe_number(metrics.get("rmse")),
        "nmae": _safe_number(metrics.get("nmae")),
        "source": "manifest",
    }


def _candidate_name_from_run(run: Mapping[str, Any]) -> str | None:
    for value in (
        _mapping(run.get("artifacts")).get("summary", {}),
        run.get("params"),
    ):
        mapping = _mapping(value)
        candidate = _first_text(mapping, ("name", "candidate", "candidate_name"))
        if candidate is not None:
            return candidate

    candidate = _first_text(run, ("candidate_name",))
    if candidate is not None:
        return candidate

    name = _safe_text(run.get("name"))
    prefix = "aidm-candidate-"
    if name and name.startswith(prefix):
        return name[len(prefix) :]
    if name == "aidm-baseline-spot":
        return "baseline"
    return None


def _has_ranking_row(rows: list[dict[str, Any]], candidate: Mapping[str, Any]) -> bool:
    return any(
        row.get("run_id") == candidate.get("run_id")
        or row.get("candidate") == candidate.get("candidate")
        for row in rows
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_text(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        text = _safe_text(mapping.get(key))
        if text is not None:
            return text
    return None


def _safe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _safe_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ranking_sort_value(value: Any) -> float:
    number = _safe_number(value)
    return number if number is not None else math.inf


if __name__ == "__main__":
    raise SystemExit(main())
