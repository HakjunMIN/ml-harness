import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_promotion_evidence_is_not_ignored():
    assert_not_ignored("artifacts/demo/promotion_manifest.json")
    assert_not_ignored("artifacts/demo/generated/promoted_features.py")


def test_local_demo_artifacts_remain_ignored():
    assert_ignored("artifacts/demo/dataset.csv")
    assert_ignored("artifacts/demo/experiments.db")
    assert_ignored("artifacts/demo/performance_report.md")
    assert_ignored("artifacts/demo/generated/local_debug.py")
    assert_ignored("artifacts/demo/extra_local_artifact.txt")
    assert_ignored("artifacts/other/transient_run.txt")


def assert_ignored(path):
    completed = _git_check_ignore(path)
    assert completed.returncode == 0, _failure_message(path, completed, "ignored")


def assert_not_ignored(path):
    completed = _git_check_ignore(path)
    assert completed.returncode == 1, _failure_message(path, completed, "not ignored")


def _git_check_ignore(path):
    return subprocess.run(
        ["git", "check-ignore", "--quiet", "--", path],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _failure_message(path, completed, expected):
    return (
        f"Expected {path!r} to be {expected}; "
        f"git check-ignore returned {completed.returncode}; stderr={completed.stderr!r}"
    )
