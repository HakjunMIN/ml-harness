import math
import shutil
import sqlite3
import uuid
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


def test_start_run_returns_uuid4_identifier(experiment_db_path):
    store = ExperimentStore(experiment_db_path)

    run_id = store.start_run("mean-baseline", {"folds": 3})

    parsed = uuid.UUID(run_id)
    assert parsed.version == 4
    assert str(parsed) == run_id


def test_runs_schema_enforces_required_columns_and_status_constraint(
    experiment_db_path,
):
    ExperimentStore(experiment_db_path)

    with sqlite3.connect(experiment_db_path) as connection:
        columns = {
            row[1]: {
                "type": row[2],
                "notnull": row[3],
                "primary_key": row[5],
            }
            for row in connection.execute("PRAGMA table_info(runs)")
        }
        create_sql = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'table' AND name = 'runs'
            """
        ).fetchone()[0]

    assert columns["id"] == {"type": "TEXT", "notnull": 1, "primary_key": 1}
    assert columns["name"]["notnull"] == 1
    assert columns["status"]["notnull"] == 1
    assert columns["started_at"]["notnull"] == 1

    normalized_sql = " ".join(create_sql.upper().split())
    assert "ID TEXT PRIMARY KEY NOT NULL" in normalized_sql
    assert "NAME TEXT NOT NULL" in normalized_sql
    assert "STATUS TEXT NOT NULL CHECK" in normalized_sql
    assert "STARTED_AT TEXT NOT NULL" in normalized_sql
    assert "CHECK (STATUS IN ('RUNNING', 'COMPLETED', 'FAILED'))" in normalized_sql


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


def test_sql_metacharacters_in_name_and_error_remain_data(experiment_db_path):
    store = ExperimentStore(experiment_db_path)
    hostile_name = "name'); DROP TABLE runs; --"
    hostile_error = "error'); ALTER TABLE runs RENAME TO broken; --"

    run_id = store.start_run(hostile_name, {"payload": "value'); VACUUM; --"})
    store.fail_run(run_id, hostile_error)

    run = store.get_run(run_id)
    assert run["name"] == hostile_name
    assert run["error"] == hostile_error

    with sqlite3.connect(experiment_db_path) as connection:
        table_name = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'runs'
            """
        ).fetchone()
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        ]
        row_count = connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    assert table_name == ("runs",)
    assert columns == [
        "id",
        "name",
        "status",
        "params_json",
        "metrics_json",
        "artifacts_json",
        "error",
        "started_at",
        "completed_at",
    ]
    assert row_count == 1


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


def test_mutations_are_committed_before_reopening_connections(experiment_db_path):
    store = ExperimentStore(experiment_db_path)

    completed_id = store.start_run("commit-complete", {"folds": 2})
    with sqlite3.connect(experiment_db_path) as connection:
        assert connection.execute(
            """
            SELECT name, status, params_json
            FROM runs
            WHERE id = ?
            """,
            (completed_id,),
        ).fetchone() == ("commit-complete", "running", '{"folds": 2}')

    store.complete_run(completed_id, {"nmae": 0.12}, {"report": "report.json"})
    with sqlite3.connect(experiment_db_path) as connection:
        assert connection.execute(
            """
            SELECT status, metrics_json, artifacts_json
            FROM runs
            WHERE id = ?
            """,
            (completed_id,),
        ).fetchone() == (
            "completed",
            '{"nmae": 0.12}',
            '{"report": "report.json"}',
        )

    failed_id = store.start_run("commit-fail", {"folds": 1})
    with sqlite3.connect(experiment_db_path) as connection:
        assert connection.execute(
            """
            SELECT name, status, params_json
            FROM runs
            WHERE id = ?
            """,
            (failed_id,),
        ).fetchone() == ("commit-fail", "running", '{"folds": 1}')

    store.fail_run(failed_id, "training crashed")
    with sqlite3.connect(experiment_db_path) as connection:
        assert connection.execute(
            """
            SELECT status, error
            FROM runs
            WHERE id = ?
            """,
            (failed_id,),
        ).fetchone() == ("failed", "training crashed")


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
