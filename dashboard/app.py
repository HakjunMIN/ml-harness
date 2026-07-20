from __future__ import annotations

import argparse
import json
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

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = json.load(handle)

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


if __name__ == "__main__":
    raise SystemExit(main())
