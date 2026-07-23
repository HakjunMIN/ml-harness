from __future__ import annotations

import ast
import json
from pathlib import Path

from power_forecasting.catalogs import load_optimization_catalog
from power_forecasting.proposals import load_proposal


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
DEFAULT_CATALOG = ROOT / "configs" / "optimization-catalog.v1.json"
EDUCATIONAL_NOTEBOOK = "00_legacy_power_forecasting_models.ipynb"
DEMO_NOTEBOOKS = (
    "01_legacy_baseline.ipynb",
    "01_legacy_baseline-demo.ipynb",
    "02_manual_skill_path.ipynb",
    "03_auto_research_path.ipynb",
)
EXPECTED_NOTEBOOKS = (EDUCATIONAL_NOTEBOOK, *DEMO_NOTEBOOKS)
ALLOWED_CELL_TYPES = {"code", "markdown", "raw"}


def test_expected_notebooks_exist_with_no_extras():
    assert sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")) == sorted(EXPECTED_NOTEBOOKS)


def test_educational_legacy_notebook_explains_models_and_exports_aidm_inputs():
    notebook = _read_notebook(NOTEBOOK_DIR / EDUCATIONAL_NOTEBOOK)
    _assert_valid_nbformat4_notebook(notebook, EDUCATIONAL_NOTEBOOK)
    code = _notebook_code(NOTEBOOK_DIR / EDUCATIONAL_NOTEBOOK)

    for model_name in ("Mean", "Weather", "ForecastWeather", "Ldaps", "SPOT"):
        assert model_name in code
    assert "chronological_holdout" in code
    for artifact_name in (
        "dataset.csv",
        "legacy_predictions.csv",
        "model_metrics.csv",
        "per_plant_metrics.csv",
    ):
        assert artifact_name in code

    tree = ast.parse(code or "\n", filename=EDUCATIONAL_NOTEBOOK)
    compile(tree, EDUCATIONAL_NOTEBOOK, "exec")


def test_demo_notebooks_are_valid_and_compile():
    for name in DEMO_NOTEBOOKS:
        notebook = _read_notebook(NOTEBOOK_DIR / name)
        _assert_valid_nbformat4_notebook(notebook, name)
        assert any(cell.get("cell_type") == "markdown" for cell in notebook["cells"]), name
        code = _notebook_code(NOTEBOOK_DIR / name)
        tree = ast.parse(code or "\n", filename=name)
        compile(tree, name, "exec")


def test_demo_notebooks_tell_the_three_path_story():
    baseline_code = _notebook_code(NOTEBOOK_DIR / "01_legacy_baseline.ipynb")
    assert "generate_legacy_demo_data" in baseline_code
    assert "chronological_holdout" in baseline_code
    assert "SPOT" in baseline_code
    for model_name in ("Mean", "Weather", "ForecastWeather", "Ldaps"):
        assert model_name in baseline_code

    manual_code = _notebook_code(NOTEBOOK_DIR / "02_manual_skill_path.ipynb")
    assert "run-aidm.sh" in manual_code
    assert "verify-promotion.sh" in manual_code
    _assert_aidm_commands_use_default_catalog(manual_code)
    manual_markdown = _notebook_markdown(NOTEBOOK_DIR / "02_manual_skill_path.ipynb")
    for skill in ("legacy-intake", "aidm-experiment", "aidd-promotion", "release-gate"):
        assert skill in manual_markdown

    auto_code = _notebook_code(NOTEBOOK_DIR / "03_auto_research_path.ipynb")
    assert "run-research-loop.sh" in auto_code
    for artifact in (
        "awaiting_proposal",
        "proposal-context.json",
        "proposal-catalog.json",
        "research-proposal.json",
        "--resume",
    ):
        assert artifact in auto_code
    config = _research_loop_config(auto_code)
    assert config["catalog_path"] == "../../configs/optimization-catalog.v1.json"
    assert (
        (ROOT / "runs" / "notebook-03-auto" / config["catalog_path"]).resolve()
        == (ROOT / "configs" / "optimization-catalog.v1.json").resolve()
    )
    assert "result = json.loads(resume.stdout)" in auto_code
    assert "if result['status'] == 'awaiting_proposal':" in auto_code
    auto_markdown = _notebook_markdown(NOTEBOOK_DIR / "03_auto_research_path.ipynb")
    assert "ready_for_human_review" in auto_markdown
    for skill in (
        "research-orchestrator",
        "research-diagnostic",
        "research-proposal",
        "research-verification",
    ):
        assert skill in auto_markdown


def test_manual_notebook_proposal_validates_against_default_catalog():
    proposal = _manual_notebook_proposal(
        NOTEBOOK_DIR / "02_manual_skill_path.ipynb"
    )
    catalog = load_optimization_catalog(DEFAULT_CATALOG, repository_root=ROOT)

    load_proposal(proposal, catalog=catalog)


def test_auto_notebook_proposal_validates_against_its_configured_profile():
    code = _notebook_code(NOTEBOOK_DIR / "03_auto_research_path.ipynb")
    config = _research_loop_config(code)
    catalog = load_optimization_catalog(DEFAULT_CATALOG, repository_root=ROOT)
    proposal = json.loads(
        (ROOT / ".agents" / "fixtures" / "model-search-proposal.json").read_text(
            encoding="utf-8"
        )
    )

    assert (
        "proposal = json.loads((REPO_ROOT / '.agents' / 'fixtures' / "
        "'model-search-proposal.json').read_text(encoding='utf-8'))"
    ) in code
    assert config["profiles"] == ["bounded_search"]
    assert "iteration_dir = OUTPUT / 'iterations' / '001-bounded_search'" in code
    load_proposal(proposal, catalog=catalog, profile=config["profiles"][0])


def _read_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _notebook_code(path: Path) -> str:
    notebook = _read_notebook(path)
    return "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )


def _notebook_markdown(path: Path) -> str:
    notebook = _read_notebook(path)
    return "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "markdown"
    )


def _manual_notebook_proposal(path: Path) -> dict[str, object]:
    tree = ast.parse(_notebook_code(path), filename=path.name)
    assignment = next(
        assignment
        for assignment in tree.body
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "proposal"
            for target in assignment.targets
        )
    )
    proposal = ast.literal_eval(assignment.value)
    assert isinstance(proposal, dict)
    return proposal


def _assert_aidm_commands_use_default_catalog(code: str) -> None:
    tree = ast.parse(code)
    commands = [
        call.args[0]
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "run"
        and call.args
        and isinstance(call.args[0], ast.List)
        and "run-aidm.sh" in ast.unparse(call.args[0])
    ]

    assert commands
    for command in commands:
        catalog_indexes = [
            index
            for index, argument in enumerate(command.elts)
            if isinstance(argument, ast.Constant) and argument.value == "--catalog"
        ]
        assert len(catalog_indexes) == 1, ast.unparse(command)
        assert ast.unparse(command.elts[catalog_indexes[0] + 1]) == "str(CATALOG)"

    catalog_assignment = next(
        assignment
        for assignment in tree.body
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CATALOG"
            for target in assignment.targets
        )
    )
    assert (
        ast.unparse(catalog_assignment.value)
        == "REPO_ROOT / 'configs' / 'optimization-catalog.v1.json'"
    )


def _research_loop_config(code: str) -> dict[str, object]:
    tree = ast.parse(code)
    config_writes = [
        call
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "write_text"
        and call.args
        and isinstance(call.args[0], ast.Call)
        and isinstance(call.args[0].func, ast.Attribute)
        and call.args[0].func.attr == "dumps"
        and call.args[0].args
        and isinstance(call.args[0].args[0], ast.Dict)
    ]
    assert len(config_writes) == 1
    config = ast.literal_eval(config_writes[0].args[0].args[0])
    assert isinstance(config, dict)
    return config


def _assert_valid_nbformat4_notebook(notebook: dict, name: str) -> None:
    assert isinstance(notebook, dict), name
    assert isinstance(notebook.get("cells"), list), name
    assert isinstance(notebook.get("metadata"), dict), name
    assert isinstance(notebook.get("nbformat"), int), name
    assert isinstance(notebook.get("nbformat_minor"), int), name

    for index, cell in enumerate(notebook["cells"]):
        label = f"{name} cell {index}"
        assert isinstance(cell, dict), label
        assert cell.get("cell_type") in ALLOWED_CELL_TYPES, label
        assert isinstance(cell.get("metadata"), dict), label
        assert isinstance(cell.get("source"), (str, list)), label
        if isinstance(cell.get("source"), list):
            assert all(isinstance(line, str) for line in cell["source"]), label

        if cell["cell_type"] == "code":
            execution_count = cell.get("execution_count")
            assert execution_count is None or isinstance(execution_count, int), label
            assert isinstance(cell.get("outputs"), list), label
            assert all(isinstance(output, dict) for output in cell["outputs"]), label


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)
