# Bounded Model Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, reproducible Random Forest, XGBoost, LightGBM, history-aware feature, and bounded Optuna exploration to AIDM.

**Architecture:** Proposal JSON remains the only agent-controlled input. Recipes and search spaces are exact-key allowlists; evaluation remains chronological and promotion still compares SPOT and optional legacy baselines. Optional package imports fail clearly only when their recipe is requested.

**Tech Stack:** Python 3.9+, scikit-learn, XGBoost, LightGBM, Optuna, pandas, SQLite, pytest, uv.

---

### Task 1: Add optional model dependencies and bounded Stage 1 recipes

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/power_forecasting/proposals.py:54-206`
- Modify: `src/power_forecasting/models.py:1-239`
- Modify: `tests/test_proposals.py`

- [ ] **Step 1: Write failing recipe validation and factory tests**

Add tests asserting a valid `random_forest` recipe constructs
`RandomForestRegressor(random_state=0)` and an `xgboost` recipe constructs
`XGBRegressor(random_state=0, n_jobs=1)`. Test exact rejection of unknown,
missing, boolean, non-finite, and out-of-allowlist parameters.

```python
def test_random_forest_recipe_builds_deterministic_estimator():
    recipe = load_proposal(_proposal(model_recipes=[{
        "name": "rf_small", "recipe": "random_forest",
        "parameters": {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2},
        "rationale": "Bounded forest.",
    }])).model_recipes[0]
    estimator = model_definition_from_recipe(recipe).estimator_factory().steps[-1][1]
    assert estimator.random_state == 0
    assert estimator.n_estimators == 200
```

- [ ] **Step 2: Verify tests fail before recipe support exists**

Run:

```bash
uv run pytest tests/test_proposals.py::test_random_forest_recipe_builds_deterministic_estimator -q
```

Expected: FAIL because `random_forest` is unsupported.

- [ ] **Step 3: Declare optional packages**

Add a `model-search` optional dependency extra containing bounded compatible
versions of `xgboost`, `lightgbm`, and `optuna`; regenerate `uv.lock`. Do not
move these packages into core dependencies.

- [ ] **Step 4: Add exact recipe allowlists and factories**

Extend `_validate_recipe_parameters()` and `model_definition_from_recipe()`:

```python
if recipe == "random_forest":
    _exact_parameter_keys(parameters, {"n_estimators", "max_depth", "min_samples_leaf"}, name)
    # n_estimators={100,200,400}; max_depth={8,12,None}; min_samples_leaf={1,2,4}
if recipe == "xgboost":
    _exact_parameter_keys(parameters, {"n_estimators", "max_depth", "learning_rate", "subsample"}, name)
    # finite values from explicitly documented discrete sets only
```

Import optional estimators inside their factory functions and raise
`ProposalValidationError` with the matching extra install command if absent.
Set `random_state=0` and `n_jobs=1`.

- [ ] **Step 5: Run focused Stage 1 recipe tests**

Run:

```bash
uv sync --extra model-search
uv run pytest tests/test_proposals.py tests/test_agentic_aidm.py -q
uv lock --locked
```

Expected: all selected tests pass; lockfile is current.

- [ ] **Step 6: Commit Stage 1 recipe support**

```bash
git add pyproject.toml uv.lock src/power_forecasting/proposals.py src/power_forecasting/models.py tests/test_proposals.py
git commit -m "feat: add bounded forest and xgboost recipes"
```

### Task 2: Add safe history-aware features and R2 reporting

**Files:**
- Modify: `src/power_forecasting/features.py:100-261`
- Modify: `src/power_forecasting/evaluation.py:51-133`
- Modify: `src/power_forecasting/aidd.py`
- Modify: `tests/test_features.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_aidd.py`

- [ ] **Step 1: Write failing time-boundary tests**

Create tests for a per-plant `lag` and `rolling_mean` transform. For a row at
timestamp `t`, assert output uses only source values at timestamps `< t`;
assert same-timestamp rows and future rows do not influence it; assert the
first insufficient-history warm-up rows emit `NaN` so the estimator pipeline can
impute them from training-fold data.

```python
def test_lag_uses_only_strictly_prior_plant_timestamp():
    frame = _history_frame()
    spec = FeatureSpec("prior_forecast", "lag", ("forecast_irradiance",), {"periods": 1})
    values = apply_feature_specs(frame, [spec])
    assert values.loc[frame.index[2], "prior_forecast"] == 10.0
```

- [ ] **Step 2: Verify the new history test fails**

Run:

```bash
uv run pytest tests/test_features.py::test_lag_uses_only_strictly_prior_plant_timestamp -q
```

Expected: FAIL because `lag` is unknown.

- [ ] **Step 3: Implement exact history transforms**

Add `lag` and `rolling_mean` to `TRANSFORMS`, arity, and parameter allowlists.
Require `plant_id` and `timestamp`, sort each plant by parsed timestamp using a
stable sort, shift before any rolling operation, then restore original index.
Require `periods` in `{1,2,3,6,12,24}` and `window` in `{3,6,12,24}`. Reject
target/actual inputs through existing AIDD prediction-time validation.

Do not allow history transforms in `render_promoted_module`; AIDD must reject
them as stateful and retain the human-review patch route.

- [ ] **Step 4: Add R2 without changing promotion gates**

Extend `compute_metrics()` with uppercase `R2` calculated as
`1 - sum((actual-prediction)^2) / sum((actual-mean(actual))^2)`. Raise a clear
error for constant actual targets. Preserve MAE, RMSE, and NMAE unchanged.

- [ ] **Step 5: Run focused history, metric, and AIDD safety tests**

Run:

```bash
uv run pytest tests/test_features.py tests/test_evaluation.py tests/test_aidd.py -q
```

Expected: new strict history boundary and R2 tests pass; AIDD does not render
stateful features.

- [ ] **Step 6: Commit the feature and metric work**

```bash
git add src/power_forecasting/features.py src/power_forecasting/evaluation.py src/power_forecasting/aidd.py tests/test_features.py tests/test_evaluation.py tests/test_aidd.py
git commit -m "feat: add history features and r2 metrics"
```

### Task 3: Add bounded LightGBM and Optuna search contracts

**Files:**
- Modify: `src/power_forecasting/proposals.py`
- Modify: `src/power_forecasting/models.py`
- Modify: `src/power_forecasting/aidm.py`
- Modify: `tests/test_proposals.py`
- Modify: `tests/test_agentic_aidm.py`

- [ ] **Step 1: Write failing search-contract tests**

Add a proposal with a `search` object:

```python
"search": {
    "sampler": "tpe",
    "seed": 17,
    "n_trials": 8,
    "spaces": {
        "lightgbm": {
            "n_estimators": [100, 300],
            "learning_rate": [0.03, 0.1],
            "num_leaves": [15, 31],
        }
    },
}
```

Test exact-key validation, `n_trials` 1..50, fixed sampler, finite discrete
choices, deterministic replay with the same seed, and fail-before-evaluate
when candidate plus trial budget exceeds 50.

- [ ] **Step 2: Verify search tests fail**

Run:

```bash
uv run pytest tests/test_proposals.py::test_valid_optuna_search_contract -q
```

Expected: FAIL because the proposal schema has no `search` field.

- [ ] **Step 3: Implement bounded LightGBM recipe**

Add `lightgbm` exact parameters `n_estimators`, `learning_rate`, `num_leaves`,
and `min_child_samples` with documented discrete allowlists. Its factory uses
`LGBMRegressor(random_state=0, n_jobs=1, verbosity=-1)`, importing LightGBM
lazily and raising a clear missing-extra error when needed.

- [ ] **Step 4: Implement deterministic Optuna execution**

Create an `optuna.samplers.TPESampler(seed=search.seed)` with a
`NopPruner`. Enumerate only declared discrete choices. For every trial:
start an `ExperimentStore` run named `aidm-optuna-<recipe>-trial-<number>`,
record resolved parameters, evaluation metrics, feature specs, proposal ID,
search seed, and trial number; complete or fail the run. Select by NMAE then
canonical parameter JSON; re-evaluate the selected recipe once through the
normal candidate path before gates and manifest creation.

- [ ] **Step 5: Run focused search tests**

Run:

```bash
uv run pytest tests/test_proposals.py tests/test_agentic_aidm.py -q
```

Expected: deterministic search, provenance, budget, and failure tests pass.

- [ ] **Step 6: Commit bounded Stage 2 search**

```bash
git add src/power_forecasting/proposals.py src/power_forecasting/models.py src/power_forecasting/aidm.py tests/test_proposals.py tests/test_agentic_aidm.py
git commit -m "feat: add bounded lightgbm optuna search"
```

### Task 4: Wire CLI, fixtures, documentation, and integration coverage

**Files:**
- Modify: `src/power_forecasting/cli.py`
- Modify: `.agents/scripts/run-aidm.sh`
- Create: `.agents/fixtures/model-search-proposal.json`
- Modify: `.agents/skills/aidm-experiment/SKILL.md`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_outer_harness.py`

- [ ] **Step 1: Write CLI and fixture failing tests**

Add tests for `--proposal` with the new fixture and a five-fold AIDM command.
Assert the harness forwards its JSON proposal unchanged and rejects search
budgets above 50 without writing promotion success evidence.

- [ ] **Step 2: Verify targeted tests fail**

Run:

```bash
uv run pytest tests/test_cli.py tests/test_outer_harness.py -q
```

Expected: FAIL until fixture and documentation-supported command path exist.

- [ ] **Step 3: Add fixture and update command surfaces**

Create a safe proposal fixture containing one temporal/weather/history feature
set, RF/XGBoost/LightGBM recipes, and a bounded Optuna LightGBM search.
Keep the existing CLI flags; add explicit `--search-trials` only if the
proposal contract requires an override, validating it does not exceed the
proposal budget. Document `uv sync --extra model-search`, five-fold usage,
supported models, R2 as diagnostic-only, and stateful AIDD limitation.

- [ ] **Step 4: Run integration tests**

Run:

```bash
uv run pytest tests/test_cli.py tests/test_outer_harness.py tests/test_agentic_aidm.py -q
```

Expected: CLI/harness forwarding and model-search metadata tests pass.

- [ ] **Step 5: Commit integration surfaces**

```bash
git add src/power_forecasting/cli.py .agents/scripts/run-aidm.sh .agents/fixtures/model-search-proposal.json .agents/skills/aidm-experiment/SKILL.md README.md tests/test_cli.py tests/test_outer_harness.py
git commit -m "docs: add bounded model search workflow"
```

### Task 5: Release verification

**Files:**
- Verify: all files above

- [ ] **Step 1: Run diff and locked-environment checks**

Run:

```bash
git diff --check main...HEAD
uv lock --locked
```

Expected: both commands exit 0.

- [ ] **Step 2: Merge and run the full suite once**

After code/spec/quality review and merge to `main`, run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -q -p no:cacheprovider
```

Expected: the entire suite passes.
