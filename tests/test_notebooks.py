from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
EXPECTED_NOTEBOOKS = (
    "01_legacy_baseline.ipynb",
    "02_aidm_feature_discovery.ipynb",
    "03_aidd_promotion.ipynb",
)


def test_expected_notebooks_exist_with_no_extras():
    assert sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb")) == sorted(EXPECTED_NOTEBOOKS)


def test_notebooks_are_minimal_valid_nbformat_workflow_demos():
    for name in EXPECTED_NOTEBOOKS:
        notebook = _read_notebook(NOTEBOOK_DIR / name)

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


def _source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)
