from __future__ import annotations

import json
import subprocess
from pathlib import Path

from power_forecasting.research_contracts import load_research_loop_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / ".agents" / "fixtures"
RUNNER = ROOT / ".agents" / "scripts" / "run-research-loop.sh"


def test_agents_runbook_defines_bounded_repeated_agent_proposals() -> None:
    content = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    for required_text in (
        '"agent_proposals": true',
        '"max_iterations": 10',
        "awaiting_proposal",
        "proposal-context.json",
        "proposal-catalog.json",
        "--resume",
        "50-evaluation budget",
        "ready_for_human_review",
    ):
        assert required_text in content


def test_research_orchestrator_owns_one_request_cycle_and_human_handoff() -> None:
    runbook = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    skill = (
        ROOT / ".agents" / "skills" / "research-orchestrator" / "SKILL.md"
    ).read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for required_text in (
        "single user request",
        "invoke `research-proposal`",
        "repeat without another user request",
        "iteration <current>/<maximum>",
        "evaluations <used>/50",
        "last result",
        "invoke `human-review`",
    ):
        assert required_text in skill

    assert "Continue autonomously until" in runbook
    assert "Do not ask the user to invoke each role" in runbook
    assert "one-shot orchestration" in readme
    assert "manual skill-by-skill path" in readme


def test_docs_define_catalog_authority_and_bound_proposal_examples() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    runbook = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    orchestrator = (
        ROOT / ".agents" / "skills" / "research-orchestrator" / "SKILL.md"
    ).read_text(encoding="utf-8")
    aidm = (ROOT / ".agents" / "skills" / "aidm-experiment" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    default_catalog = "configs/optimization-catalog.v1.json"
    readme_claims = " ".join(readme.split())
    runbook_claims = " ".join(runbook.split())
    orchestrator_claims = " ".join(orchestrator.split())

    for required_text in (
        default_catalog,
        "versioned external owner",
        "profiles, feature sets, direct recipes, allowed parameter values, and bounded TPE space",
        "`catalog_path`",
        "`--catalog`",
        "Python still owns",
        "cannot add code or new estimator capabilities",
        "prediction ensemble",
    ):
        assert required_text in readme_claims

    for required_text in (
        "Do not change the catalog once the run begins",
        "profiles, feature sets, direct recipes, TPE space, folds, gates, and budget",
        "catalog SHA-256",
        "fail closed on resume",
    ):
        assert required_text in runbook_claims

    for required_text in (
        "`catalog_path`",
        "catalog SHA-256",
        "research-config.json",
        "state.json",
        "journal.jsonl",
        "handoff",
        "fail closed on resume",
    ):
        assert required_text in orchestrator_claims

    for document in (readme, aidm):
        commands = " ".join(document.replace("\\\n", " ").split())
        proposal_examples = [
            command
            for command in commands.split(".agents/scripts/run-aidm.sh")[1:]
            if "--proposal" in command
        ]
        assert proposal_examples
        assert all(
            f"--catalog {default_catalog} --proposal" in command
            for command in proposal_examples
        )


def test_readme_is_tutorial_first_agent_skill_framework_overview() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for required_text in (
        "레거시 ML",
        "에이전트 스킬 프레임워크",
        "`AGENTS.md`",
        "`.agents/skills/`",
        "## 5분 시작",
        "one-shot orchestration",
        "manual skill-by-skill path",
        "피처 탐색",
        "하이퍼파라미터 최적화",
        "Ridge",
        "HistGradientBoosting",
        "RandomForest",
        "XGBoost",
        "LightGBM",
        "Optuna TPE",
        "현재 지원하지 않는 범위",
        "prediction ensemble",
    ):
        assert required_text in readme


def test_research_fixture_uses_config_relative_synthetic_inputs_and_root_runs_output() -> None:
    config_path = FIXTURES / "research-loop.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = load_research_loop_config(
        payload,
        config_path=config_path,
        repository_root=ROOT,
    )

    assert Path(config.dataset_path) == (FIXTURES / "valid-dataset.csv").resolve()
    assert Path(config.legacy_manifest_path) == (FIXTURES / "promoted-manifest.json").resolve()
    assert Path(config.run_dir) == (ROOT / "runs" / "research-loop-fixture").resolve()
    assert "customer" not in json.dumps(payload).lower()
    assert "generation_mw" not in json.dumps(payload).lower()


def test_research_runner_has_exact_cli_and_does_not_export_environment_secrets() -> None:
    content = RUNNER.read_text(encoding="utf-8")

    assert "uv run python -m power_forecasting.cli research-loop" in content
    assert "export " not in content
    assert "cd \"$repo_root\"" in content


def test_research_runner_accepts_only_help_and_config_options(tmp_path: Path) -> None:
    missing = subprocess.run(
        [str(RUNNER)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    unknown = subprocess.run(
        [str(RUNNER), "--config", str(FIXTURES / "research-loop.json"), "--unexpected"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    help_result = subprocess.run(
        [str(RUNNER), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert missing.returncode == 2
    assert unknown.returncode == 2
    assert help_result.returncode == 0


def test_research_runner_rejects_missing_config_without_creating_output(tmp_path: Path) -> None:
    result = subprocess.run(
        [str(RUNNER), "--config", str(tmp_path / "missing.json")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "config not found" in result.stderr


def test_research_skill_failure_artifacts_match_orchestrator_contract() -> None:
    orchestrator = (
        ROOT / ".agents" / "skills" / "research-orchestrator" / "SKILL.md"
    ).read_text(encoding="utf-8")
    diagnostic = (
        ROOT / ".agents" / "skills" / "research-diagnostic" / "SKILL.md"
    ).read_text(encoding="utf-8")
    verification = (
        ROOT / ".agents" / "skills" / "research-verification" / "SKILL.md"
    ).read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "diagnostic-failure.json" in orchestrator
    assert "experiment-failure.json" in orchestrator
    assert "diagnostic-failure.json" in diagnostic
    assert "`rejected_conditions`" in diagnostic
    assert "`iteration`, and a safe `reason`" not in diagnostic
    assert "verification.json" in verification
    assert "invalid" in verification
    assert "verification-failure.json" in verification
    assert "malformed" in verification
    assert "diagnostic-failure.json" in readme
    assert "verification-failure.json" in readme
    assert "malformed" in readme
