import json
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


class ExperimentStore:
    _STATUSES = ("running", "completed", "failed")

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
                    params_json TEXT,
                    metrics_json TEXT,
                    artifacts_json TEXT,
                    error TEXT,
                    started_at TEXT,
                    completed_at TEXT
                )
                """
            )

    def start_run(self, name: str, params: Mapping) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be nonblank")

        run_id = str(uuid.uuid4())
        started_at = _utc_now_iso()
        params_json = _json_dumps(params, "params")
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, name, status, params_json, metrics_json, artifacts_json,
                    error, started_at, completed_at
                )
                VALUES (?, ?, 'running', ?, NULL, NULL, NULL, ?, NULL)
                """,
                (run_id, name, params_json, started_at),
            )
        return run_id

    def complete_run(self, run_id, metrics: Mapping, artifacts: Mapping) -> None:
        completed_at = _utc_now_iso()
        metrics_json = _json_dumps(metrics, "metrics")
        artifacts_json = _json_dumps(artifacts, "artifacts")
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'completed',
                    metrics_json = ?,
                    artifacts_json = ?,
                    completed_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (
                    metrics_json,
                    artifacts_json,
                    completed_at,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                _raise_transition_error(connection, run_id)

    def fail_run(self, run_id, error: str) -> None:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be nonblank")

        completed_at = _utc_now_iso()
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'failed',
                    error = ?,
                    completed_at = ?
                WHERE id = ? AND status = 'running'
                """,
                (error, completed_at, run_id),
            )
            if cursor.rowcount != 1:
                _raise_transition_error(connection, run_id)

    def get_run(self, run_id) -> dict:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, name, status, params_json, metrics_json, artifacts_json,
                       error, started_at, completed_at
                FROM runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return _decode_run(row)

    def list_runs(self, status: str = None) -> list[dict]:
        if status is not None and status not in self._STATUSES:
            raise ValueError("status must be one of running, completed, failed")

        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            if status is None:
                rows = connection.execute(
                    """
                    SELECT id, name, status, params_json, metrics_json, artifacts_json,
                           error, started_at, completed_at
                    FROM runs
                    ORDER BY started_at DESC, rowid DESC
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, name, status, params_json, metrics_json, artifacts_json,
                           error, started_at, completed_at
                    FROM runs
                    WHERE status = ?
                    ORDER BY started_at DESC, rowid DESC
                    """,
                    (status,),
                ).fetchall()
        return [_decode_run(row) for row in rows]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Mapping, label: str) -> str:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label} must be JSON serializable with finite numbers"
        ) from error


def _json_loads(value):
    if value is None:
        return None
    return json.loads(value)


def _decode_run(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "status": row["status"],
        "params": _json_loads(row["params_json"]),
        "metrics": _json_loads(row["metrics_json"]),
        "artifacts": _json_loads(row["artifacts_json"]),
        "error": row["error"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
    }


def _raise_transition_error(connection: sqlite3.Connection, run_id) -> None:
    row = connection.execute(
        "SELECT status FROM runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise KeyError(run_id)
    raise ValueError(f"run {run_id!r} is already terminal ({row[0]})")
