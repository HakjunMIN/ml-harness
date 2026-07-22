from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
EXPECTED_NOTEBOOKS = (
    "00_legacy_power_forecasting_models.ipynb",
    "01_legacy_baseline.ipynb",
    "02_aidm_feature_discovery.ipynb",
    "03_aidd_promotion.ipynb",
)
WORKFLOW_NOTEBOOKS = EXPECTED_NOTEBOOKS[1:]
EDUCATIONAL_NOTEBOOK = EXPECTED_NOTEBOOKS[0]
ALLOWED_CELL_TYPES = {"code", "markdown", "raw"}


def test_expected_notebooks_exist_with_no_extras():
    assert sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")) == sorted(EXPECTED_NOTEBOOKS)


def test_notebooks_are_minimal_valid_nbformat_workflow_demos():
    for name in WORKFLOW_NOTEBOOKS:
        notebook = _read_notebook(NOTEBOOK_DIR / name)

        _assert_valid_nbformat4_notebook(notebook, name)
        assert notebook["nbformat"] == 4
        assert isinstance(notebook.get("nbformat_minor"), int)
        if notebook["nbformat_minor"] >= 5:
            assert all("id" in cell for cell in notebook["cells"]), name
        assert isinstance(notebook["cells"], list)

        markdown_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"]
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        assert len(markdown_cells) == 1, name
        assert len(code_cells) <= 4, name

        markdown = "\n".join(_source(cell) for cell in markdown_cells)
        assert markdown.lstrip().startswith("#"), name
        assert "Purpose:" in markdown, name
        assert "Assumption:" in markdown, name

        code = "\n".join(_source(cell) for cell in code_cells)
        code_lines = [line for line in code.splitlines() if line.strip()]
        assert len(code_lines) <= 20, name
        assert "power_forecasting" in code, name
        assert 'Path("artifacts/demo")' in code, name

        tree = ast.parse(code or "\n", filename=name)
        forbidden = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        assert not any(isinstance(node, forbidden) for node in ast.walk(tree)), name
        compile(tree, name, "exec")


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


def test_notebooks_use_the_required_workflow_apis_without_running_them():
    expected_snippets = {
        "01_legacy_baseline.ipynb": ("run_generate_data", "run_legacy"),
        "02_aidm_feature_discovery.ipynb": ("AIDMConfig", "run_aidm_workflow"),
        "03_aidd_promotion.ipynb": ("json.loads", "run_aidd_workflow"),
    }

    for name, snippets in expected_snippets.items():
        code = _notebook_code(NOTEBOOK_DIR / name)
        for snippet in snippets:
            assert snippet in code, name


def _read_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _notebook_code(path: Path) -> str:
    notebook = _read_notebook(path)
    return "\n".join(
        _source(cell) for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )


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
