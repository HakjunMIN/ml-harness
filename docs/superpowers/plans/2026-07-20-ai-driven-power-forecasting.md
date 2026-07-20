# AI-Driven Power Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local, reproducible vertical slice that recreates a manual power-forecasting baseline, discovers better features with AIDM, and promotes validated features into deterministic production code with AIDD.

**Architecture:** Notebooks remain thin workflow clients while tested Python modules own data contracts, features, models, evaluation, experiment persistence, promotion, and reporting. The first version runs locally with deterministic synthetic solar data, SQLite experiment tracking, and a constrained code generator; storage and execution boundaries remain replaceable.

**Tech Stack:** Python 3.9+, NumPy, pandas, scikit-learn, pytest, SQLite, argparse, optional Streamlit

---

## File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, dependencies, pytest configuration, CLI entry point. |
| `.gitignore` | Ignore virtual environments, caches, databases, and generated run artifacts. |
| `README.md` | Explain setup, architecture, workflows, and extension points. |
| `src/power_forecasting/data.py` | Generate deterministic demo data and validate the forecast contract. |
| `src/power_forecasting/features.py` | Define feature specs and compute allowlisted transforms. |
| `src/power_forecasting/models.py` | Build named legacy and candidate model pipelines. |
| `src/power_forecasting/evaluation.py` | Create chronological folds and calculate forecast metrics. |
| `src/power_forecasting/experiments.py` | Persist experiment runs and candidate metrics in SQLite. |
| `src/power_forecasting/aidm.py` | Generate, evaluate, rank, and promote feature candidates. |
| `src/power_forecasting/aidd.py` | Validate manifests and render deterministic feature code. |
| `src/power_forecasting/reporting.py` | Render baseline and promotion reports. |
| `src/power_forecasting/cli.py` | Run `generate-data`, `legacy`, `aidm`, `aidd`, and `all`. |
| `dashboard/app.py` | Display experiment and promotion data when Streamlit is installed. |
| `notebooks/*.ipynb` | Demonstrate the three workflows without embedding business logic. |
| `tests/` | Unit and end-to-end coverage for every production module. |

### Task 1: Package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/power_forecasting/__init__.py`
- Create: `tests/test_package.py`

- [ ] **Step 1: Write the failing package test**

```python
def test_package_exports_version():
    import power_forecasting

    assert power_forecasting.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `python3 -m pytest tests/test_package.py -q`
Expected: FAIL because `power_forecasting` is not installed.

- [ ] **Step 3: Add package configuration and version**

Use a setuptools `src` layout, require Python 3.9, and declare:

```toml
dependencies = [
  "numpy>=1.24,<3",
  "pandas>=2.0,<3",
  "scikit-learn>=1.3,<2",
]

[project.optional-dependencies]
dev = ["pytest>=8,<9"]
dashboard = ["streamlit>=1.36,<2"]

[project.scripts]
power-forecast = "power_forecasting.cli:main"
```

In `__init__.py`, define `__version__ = "0.1.0"`.

- [ ] **Step 4: Install and verify**

Run: `python3 -m pip install -e ".[dev]"`
Expected: installation succeeds.

Run: `python3 -m pytest tests/test_package.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/power_forecasting/__init__.py tests/test_package.py
git commit -m "build: initialize forecasting package"
```

### Task 2: Deterministic forecast dataset

**Files:**
- Create: `src/power_forecasting/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing contract tests**

```python
from power_forecasting.data import REQUIRED_COLUMNS, generate_synthetic_data, validate_dataset


def test_synthetic_data_is_deterministic_and_valid():
    first = generate_synthetic_data(days=14, plants=2, seed=7)
    second = generate_synthetic_data(days=14, plants=2, seed=7)
    assert first.equals(second)
    assert set(REQUIRED_COLUMNS).issubset(first.columns)
    assert len(first) == 14 * 24 * 2
    validate_dataset(first)


def test_generation_respects_capacity():
    frame = generate_synthetic_data(days=7, plants=3, seed=9)
    assert (frame["generation_mw"] >= 0).all()
    assert (frame["generation_mw"] <= frame["capacity_mw"]).all()
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_data.py -q`
Expected: FAIL because `data.py` does not exist.

- [ ] **Step 3: Implement generation and validation**

Define `REQUIRED_COLUMNS`, `DataContractError`, `generate_synthetic_data(days, plants, seed)`, and `validate_dataset(frame)`. Generate hourly daylight and seasonal irradiance, correlated forecast/LDAPS errors, cloud attenuation, temperature derating, plant capacities, and bounded generation. Validation must reject missing columns, duplicate `(plant_id, timestamp)` rows, non-finite numeric values, non-positive capacity, and target values outside `[0, capacity_mw]`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_data.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/power_forecasting/data.py tests/test_data.py
git commit -m "feat: add deterministic forecast dataset"
```

### Task 3: Declarative feature engine

**Files:**
- Create: `src/power_forecasting/features.py`
- Create: `tests/test_features.py`

- [ ] **Step 1: Write failing feature tests**

```python
import numpy as np
from power_forecasting.data import generate_synthetic_data
from power_forecasting.features import FeatureSpec, apply_feature_specs


def test_domain_features_are_finite_and_deterministic():
    frame = generate_synthetic_data(days=3, plants=1, seed=3)
    specs = [
        FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",), {}),
        FeatureSpec("effective_irradiance", "effective_irradiance",
                    ("forecast_irradiance", "forecast_cloud_cover"), {}),
    ]
    result = apply_feature_specs(frame, specs)
    assert list(result.columns) == ["hour_sin", "effective_irradiance"]
    assert np.isfinite(result.to_numpy()).all()


def test_unknown_transform_is_rejected():
    frame = generate_synthetic_data(days=1, plants=1, seed=1)
    spec = FeatureSpec("bad", "arbitrary_python", ("forecast_irradiance",), {})
    with pytest.raises(ValueError, match="Unsupported transform"):
        apply_feature_specs(frame, [spec])
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_features.py -q`
Expected: FAIL because the feature API does not exist.

- [ ] **Step 3: Implement the allowlisted engine**

Create a frozen `FeatureSpec` dataclass with `name`, `transform`, `inputs`,
`parameters`, `version`, and `rationale`. Implement:

```python
TRANSFORMS = {
    "cyclic_hour",
    "cyclic_day_of_year",
    "effective_irradiance",
    "temperature_derating",
    "cloud_attenuation",
    "interaction",
    "ratio",
}
```

`apply_feature_specs` must validate unique output names and source columns, avoid mutating the input, replace infinities with missing values, and reject any resulting missing/non-finite value.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_features.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/power_forecasting/features.py tests/test_features.py
git commit -m "feat: add declarative feature engine"
```

### Task 4: Models and chronological evaluation

**Files:**
- Create: `src/power_forecasting/models.py`
- Create: `src/power_forecasting/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Write failing split and metric tests**

```python
from power_forecasting.data import generate_synthetic_data
from power_forecasting.evaluation import chronological_folds, evaluate_model
from power_forecasting.models import model_definition


def test_folds_never_mix_or_reverse_timestamps():
    frame = generate_synthetic_data(days=20, plants=2, seed=4)
    for train_idx, valid_idx in chronological_folds(frame, folds=3):
        assert frame.loc[train_idx, "timestamp"].max() < frame.loc[valid_idx, "timestamp"].min()


def test_mean_model_returns_finite_metrics():
    frame = generate_synthetic_data(days=30, plants=2, seed=4)
    result = evaluate_model(frame, model_definition("Mean"), feature_specs=[], folds=3)
    assert result.metrics["nmae"] >= 0
    assert set(result.per_plant) == set(frame["plant_id"].unique())
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_evaluation.py -q`
Expected: FAIL because model and evaluation modules do not exist.

- [ ] **Step 3: Implement model adapters**

Create `ModelDefinition(name, base_features, estimator_factory)` and support
`Mean`, `Weather`, `ForecastWeather`, `Ldaps`, and `SPOT`. Use a custom
plant/hour mean estimator for `Mean`, `Ridge` pipelines for weather models, and
`HistGradientBoostingRegressor` for `SPOT`.

- [ ] **Step 4: Implement evaluation**

Create `EvaluationResult(metrics, per_plant, fold_metrics, predictions)`.
Implement timestamp-grouped expanding folds and metrics:

```python
mae = mean(abs(actual - prediction))
rmse = sqrt(mean((actual - prediction) ** 2))
nmae = sum(abs(actual - prediction)) / sum(capacity_mw)
```

Clip predictions to `[0, capacity_mw]`.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m pytest tests/test_evaluation.py -q`
Expected: `2 passed`.

```bash
git add src/power_forecasting/models.py src/power_forecasting/evaluation.py tests/test_evaluation.py
git commit -m "feat: add legacy models and time-safe evaluation"
```

### Task 5: SQLite experiment registry

**Files:**
- Create: `src/power_forecasting/experiments.py`
- Create: `tests/test_experiments.py`

- [ ] **Step 1: Write failing persistence test**

```python
from power_forecasting.experiments import ExperimentStore


def test_store_round_trips_successful_run(tmp_path):
    store = ExperimentStore(tmp_path / "runs.db")
    run_id = store.start_run("baseline", {"model": "SPOT"})
    store.complete_run(run_id, {"nmae": 0.12}, {"features": ["hour_sin"]})
    run = store.get_run(run_id)
    assert run["status"] == "completed"
    assert run["metrics"]["nmae"] == 0.12
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_experiments.py -q`
Expected: FAIL because `ExperimentStore` does not exist.

- [ ] **Step 3: Implement store**

Use SQLite tables `runs(id, name, status, params_json, metrics_json,
artifacts_json, error, started_at, completed_at)`. Use UUID run IDs, UTC ISO
timestamps, parameterized SQL, and JSON serialization with sorted keys. Provide
`start_run`, `complete_run`, `fail_run`, `get_run`, and `list_runs`.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m pytest tests/test_experiments.py -q`
Expected: `1 passed`.

```bash
git add src/power_forecasting/experiments.py tests/test_experiments.py
git commit -m "feat: persist forecasting experiments"
```

### Task 6: AIDM candidate search and promotion gate

**Files:**
- Create: `src/power_forecasting/aidm.py`
- Create: `tests/test_aidm.py`

- [ ] **Step 1: Write failing AIDM integration test**

```python
from power_forecasting.aidm import AIDMConfig, run_aidm
from power_forecasting.data import generate_synthetic_data


def test_aidm_returns_ranked_reproducible_candidates(tmp_path):
    frame = generate_synthetic_data(days=45, plants=2, seed=11)
    config = AIDMConfig(folds=3, minimum_improvement=0.0, max_plant_regression=0.2)
    first = run_aidm(frame, tmp_path / "first.db", config)
    second = run_aidm(frame, tmp_path / "second.db", config)
    assert first.ranking == second.ranking
    assert first.winner is not None
    assert first.manifest["decision"] == "promote"
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_aidm.py -q`
Expected: FAIL because AIDM is not implemented.

- [ ] **Step 3: Implement bounded search**

Define candidate specs for cyclic time, effective irradiance, temperature
derating, cloud attenuation, and irradiance/temperature interaction. Evaluate
single candidates, retain the best three, then evaluate combinations of two and
three. Rank by NMAE with stable feature-name tie breaking.

- [ ] **Step 4: Implement promotion manifest**

Create `AIDMConfig`, `CandidateResult`, and `AIDMResult`. Compare the winner with
the `SPOT` baseline. The manifest must include schema version, seed, baseline and
winner metrics, selected specs, per-plant deltas, threshold values, decision, and
failed gate reasons.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m pytest tests/test_aidm.py -q`
Expected: `1 passed`.

```bash
git add src/power_forecasting/aidm.py tests/test_aidm.py
git commit -m "feat: automate feature discovery and promotion"
```

### Task 7: AIDD deterministic code promotion

**Files:**
- Create: `src/power_forecasting/aidd.py`
- Create: `tests/test_aidd.py`

- [ ] **Step 1: Write failing generation test**

```python
import importlib.util
from power_forecasting.aidd import render_promoted_module
from power_forecasting.data import generate_synthetic_data
from power_forecasting.features import FeatureSpec


def test_generated_module_is_importable_and_equivalent(tmp_path):
    specs = [FeatureSpec("hour_sin", "cyclic_hour", ("timestamp",), {})]
    target = tmp_path / "promoted_features.py"
    render_promoted_module({"decision": "promote", "selected_specs": [s.to_dict() for s in specs]}, target)
    spec = importlib.util.spec_from_file_location("promoted", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.build_promoted_features(generate_synthetic_data(2, 1, 1))
    assert list(result.columns) == ["hour_sin"]
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_aidd.py -q`
Expected: FAIL because AIDD is not implemented.

- [ ] **Step 3: Implement manifest validation and rendering**

Reject non-promoted manifests, unsupported schema versions, empty specs,
duplicate names, and unsupported transforms. Render a module that imports
`FeatureSpec` and `apply_feature_specs`, embeds canonical JSON-compatible spec
data, and exposes `build_promoted_features(frame)`.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m pytest tests/test_aidd.py -q`
Expected: `1 passed`.

```bash
git add src/power_forecasting/aidd.py tests/test_aidd.py
git commit -m "feat: generate promoted feature code"
```

### Task 8: Reports and end-to-end CLI

**Files:**
- Create: `src/power_forecasting/reporting.py`
- Create: `src/power_forecasting/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing end-to-end test**

```python
from power_forecasting.cli import run_all


def test_run_all_creates_complete_artifacts(tmp_path):
    result = run_all(tmp_path, days=45, plants=2, seed=13)
    assert result["dataset"].exists()
    assert result["database"].exists()
    assert result["manifest"].exists()
    assert result["generated_module"].exists()
    assert result["report"].exists()
    assert "Promotion decision: promote" in result["report"].read_text()
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_cli.py -q`
Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement reporting**

Render a Markdown report containing data summary, all five legacy model metrics,
ranked AIDM candidates, promotion thresholds, per-plant deltas, failed gates, and
artifact paths.

- [ ] **Step 4: Implement CLI**

Use `argparse` subcommands `generate-data`, `legacy`, `aidm`, `aidd`, and `all`.
`run_all` writes `dataset.csv`, `experiments.db`, `promotion_manifest.json`,
`generated/promoted_features.py`, and `performance_report.md`. Commands return
non-zero on validation or promotion failure.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m pytest tests/test_cli.py -q`
Expected: `1 passed`.

```bash
git add src/power_forecasting/reporting.py src/power_forecasting/cli.py tests/test_cli.py
git commit -m "feat: add end-to-end forecasting workflow"
```

### Task 9: Notebook clients and dashboard

**Files:**
- Create: `notebooks/01_legacy_baseline.ipynb`
- Create: `notebooks/02_aidm_feature_discovery.ipynb`
- Create: `notebooks/03_aidd_promotion.ipynb`
- Create: `dashboard/app.py`
- Create: `tests/test_notebooks.py`

- [ ] **Step 1: Write failing notebook structure test**

```python
import json
from pathlib import Path


def test_notebooks_are_valid_and_thin():
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text())
        assert notebook["nbformat"] == 4
        code = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert "power_forecasting" in code
        assert len(code.splitlines()) <= 20
```

- [ ] **Step 2: Run and verify failure**

Run: `python3 -m pytest tests/test_notebooks.py -q`
Expected: FAIL because notebooks do not exist.

- [ ] **Step 3: Create thin notebooks**

Each notebook contains a Markdown purpose cell and no more than four code cells.
They import and call package APIs; they do not define feature, model, or
evaluation functions.

- [ ] **Step 4: Create optional dashboard**

`dashboard/app.py` accepts an artifact directory, reads the SQLite run list and
promotion manifest, and displays metrics, candidate ranking, decision, and report.
It must print an actionable error when artifacts are absent.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m pytest tests/test_notebooks.py -q`
Expected: `1 passed`.

```bash
git add notebooks dashboard/app.py tests/test_notebooks.py
git commit -m "feat: add notebook workflows and dashboard"
```

### Task 10: Documentation and full verification

**Files:**
- Create: `README.md`
- Create: `tests/test_end_to_end.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add a production-path smoke test**

```python
import subprocess
import sys


def test_cli_all_command(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "power_forecasting.cli", "all",
         "--output", str(tmp_path), "--days", "45", "--plants", "2", "--seed", "21"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "generated" / "promoted_features.py").exists()
```

- [ ] **Step 2: Document usage and extension points**

README sections must cover setup, quick start, legacy model meanings, architecture,
AIDM search, AIDD safety, artifact layout, dashboard command, customer data
adapter contract, production migration path, and deferred capabilities.

- [ ] **Step 3: Run all checks**

Run: `python3 -m pytest -q`
Expected: all tests pass.

Run: `python3 -m power_forecasting.cli all --output artifacts/demo --days 60 --plants 3 --seed 42`
Expected: exits zero and reports a promoted feature set.

Run: `python3 -m compileall -q src dashboard artifacts/demo/generated`
Expected: exits zero with no output.

- [ ] **Step 4: Inspect generated evidence**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/demo")
manifest = json.loads((root / "promotion_manifest.json").read_text())
assert manifest["decision"] == "promote"
assert manifest["selected_specs"]
assert (root / "performance_report.md").stat().st_size > 500
print(manifest["winner_metrics"]["nmae"])
PY
```

Expected: prints a finite NMAE and exits zero.

- [ ] **Step 5: Commit**

```bash
git add README.md .gitignore tests/test_end_to_end.py
git commit -m "docs: complete forecasting demo guide"
```

## Plan Self-Review

- Every Phase 1 requirement in the design maps to Tasks 2-10.
- Deferred production services remain documented extension points rather than
  incomplete implementations.
- Public names are consistent across tasks: `FeatureSpec`,
  `apply_feature_specs`, `EvaluationResult`, `ExperimentStore`, `AIDMConfig`,
  `AIDMResult`, `run_aidm`, `render_promoted_module`, and `run_all`.
- All code-producing tasks begin with a failing test and include exact test and
  verification commands.
