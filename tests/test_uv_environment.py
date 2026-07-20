from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_uv_pytest_console_script_can_import_dashboard_package():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/test_dashboard.py::test_dashboard_module_import_does_not_import_streamlit",
            "-q",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr
