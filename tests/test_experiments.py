import math
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from power_forecasting.experiments import ExperimentStore


@pytest.fixture
def experiment_db_path(request):
    root = (
        Path(__file__).resolve().parents[1]
        / "runs"
        / "pytest-experiments"
        / request.node.name
    )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root / "experiments.sqlite"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def assert_utc_iso(timestamp):
    parsed = datetime.fromisoformat(timestamp)

    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def test_completed_run_roundtrips_with_nmae_metric(experiment_db_path):
    store = ExperimentStore(experiment_db_path)

    run_id = store.start_run("mean-baseline", {"folds": 3, "features": ["hour"]})
    store.complete_run(
        run_id,
        {"nmae": 0.12},
        {"predictions": "runs/predictions.csv"},
    )

    run = store.get_run(run_id)

    assert run == {
        "id": run_id,
        "name": "mean-baseline",
        "status": "completed",
        "params": {"folds": 3, "features": ["hour"]},
        "metrics": {"nmae": 0.12},
        "artifacts": {"predictions": "runs/predictions.csv"},
        "error": None,
        "started_at": run["started_at"],
        "completed_at": run["completed_at"],
    }
    assert_utc_iso(run["started_at"])
    assert_utc_iso(run["completed_at"])
    assert datetime.fromisoformat(run["completed_at"]) >= datetime.fromisoformat(
        run["started_at"]
    )
    assert store.list_runs() == [run]


def test_failed_run_records_error_and_terminal_timestamp(experiment_db_path):
    store = ExperimentStore(experiment_db_path)
    run_id = store.start_run("weather-model", {"folds": 5})

    store.fail_run(run_id, "training crashed")

    run = store.get_run(run_id)
    assert run["status"] == "failed"
    assert run["error"] == "training crashed"
    assert run["metrics"] is None
    assert run["artifacts"] is None
    assert_utc_iso(run["completed_at"])


def test_unknown_run_ids_are_rejected(experiment_db_path):
    store = ExperimentStore(experiment_db_path)

    with pytest.raises(KeyError):
        store.get_run("missing")
    with pytest.raises(KeyError):
        store.complete_run("missing", {}, {})
    with pytest.raises(KeyError):
        store.fail_run("missing", "boom")


def test_completed_runs_reject_additional_terminal_transitions(experiment_db_path):
    store = ExperimentStore(experiment_db_path)
    run_id = store.start_run("mean-baseline", {})
    store.complete_run(run_id, {"nmae": 0.2}, {})

    with pytest.raises(ValueError, match="terminal"):
        store.complete_run(run_id, {"nmae": 0.1}, {})
    with pytest.raises(ValueError, match="terminal"):
        store.fail_run(run_id, "late failure")

    assert store.get_run(run_id)["metrics"] == {"nmae": 0.2}


def test_failed_runs_reject_additional_terminal_transitions(experiment_db_path):
    store = ExperimentStore(experiment_db_path)
    run_id = store.start_run("mean-baseline", {})
    store.fail_run(run_id, "training crashed")

    with pytest.raises(ValueError, match="terminal"):
        store.complete_run(run_id, {"nmae": 0.1}, {})
    with pytest.raises(ValueError, match="terminal"):
        store.fail_run(run_id, "second failure")

    assert store.get_run(run_id)["error"] == "training crashed"


def test_json_columns_are_sorted_and_nested_values_roundtrip(experiment_db_path):
    store = ExperimentStore(experiment_db_path)
    params = {
        "z": [{"b": 2, "a": 1}],
        "a": {"d": 4, "c": [True, None, "x"]},
    }
    metrics = {"z": 2.0, "a": {"b": 1.0}}
    artifacts = {"z": [{"path": "later"}], "a": "first"}

    run_id = store.start_run("deterministic-json", params)
    store.complete_run(run_id, metrics, artifacts)

    with sqlite3.connect(experiment_db_path) as connection:
        raw = connection.execute(
            """
            SELECT params_json, metrics_json, artifacts_json
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    assert raw == (
        '{"a": {"c": [true, null, "x"], "d": 4}, "z": [{"a": 1, "b": 2}]}',
        '{"a": {"b": 1.0}, "z": 2.0}',
        '{"a": "first", "z": [{"path": "later"}]}',
    )
    run = store.get_run(run_id)
    assert run["params"] == params
    assert run["metrics"] == metrics
    assert run["artifacts"] == artifacts
    assert list(run["params"]) == ["a", "z"]


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_params_are_rejected(experiment_db_path, bad_value):
    store = ExperimentStore(experiment_db_path)

    with pytest.raises(ValueError):
        store.start_run("bad-params", {"bad": bad_value})

    assert store.list_runs() == []


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_metrics_and_artifacts_are_rejected_without_transition(
    experiment_db_path, bad_value
):
    store = ExperimentStore(experiment_db_path)
    metrics_run_id = store.start_run("bad-metrics", {})
    artifacts_run_id = store.start_run("bad-artifacts", {})

    with pytest.raises(ValueError):
        store.complete_run(metrics_run_id, {"bad": bad_value}, {})
    with pytest.raises(ValueError):
        store.complete_run(artifacts_run_id, {}, {"bad": bad_value})

    assert store.get_run(metrics_run_id)["status"] == "running"
    assert store.get_run(artifacts_run_id)["status"] == "running"


def test_unserializable_params_are_rejected(experiment_db_path):
    store = ExperimentStore(experiment_db_path)

    with pytest.raises(ValueError):
        store.start_run("bad-params", {"bad": object()})


@pytest.mark.parametrize("name", ["", "   ", "\n\t"])
def test_blank_run_names_are_rejected(experiment_db_path, name):
    store = ExperimentStore(experiment_db_path)

    with pytest.raises(ValueError, match="name"):
        store.start_run(name, {})


@pytest.mark.parametrize("error", ["", "   ", "\n\t"])
def test_blank_failure_errors_are_rejected(experiment_db_path, error):
    store = ExperimentStore(experiment_db_path)
    run_id = store.start_run("will-fail", {})

    with pytest.raises(ValueError, match="error"):
        store.fail_run(run_id, error)

    assert store.get_run(run_id)["status"] == "running"


def test_status_filtering_and_newest_first_order(experiment_db_path):
    store = ExperimentStore(experiment_db_path)
    running_id = store.start_run("running", {"index": 1})
    completed_id = store.start_run("completed", {"index": 2})
    store.complete_run(completed_id, {"nmae": 0.2}, {})
    failed_id = store.start_run("failed", {"index": 3})
    store.fail_run(failed_id, "boom")

    assert [run["id"] for run in store.list_runs()] == [
        failed_id,
        completed_id,
        running_id,
    ]
    assert [run["id"] for run in store.list_runs("running")] == [running_id]
    assert [run["id"] for run in store.list_runs("completed")] == [completed_id]
    assert [run["id"] for run in store.list_runs("failed")] == [failed_id]
    with pytest.raises(ValueError, match="status"):
        store.list_runs("queued")


def test_parent_directories_and_independent_connections(experiment_db_path):
    db_path = experiment_db_path.parent / "nested" / "stores" / "experiments.sqlite"
    assert not db_path.parent.exists()

    first_store = ExperimentStore(db_path)
    run_id = first_store.start_run("shared-connection", {"folds": 2})
    second_store = ExperimentStore(db_path)
    second_store.complete_run(run_id, {"nmae": 0.12}, {"report": "report.json"})

    assert db_path.exists()
    assert first_store.get_run(run_id)["status"] == "completed"
    assert second_store.get_run(run_id)["metrics"] == {"nmae": 0.12}
