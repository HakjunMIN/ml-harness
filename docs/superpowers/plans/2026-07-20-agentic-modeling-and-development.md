# Agentic Modeling and Development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add declarative agent proposals, bounded model recipe search, legacy baseline comparison, and AIDD model patch requests.

**Architecture:** Proposal parsing and model recipes are pure Python contracts. AIDM evaluates only validated combinations through the existing time-safe evaluator; AIDD writes evidence-only patch requests after promotion.

**Tech Stack:** Python 3.9+, existing pandas/scikit-learn/pytest stack

---

### Task 1: Proposal and recipe contracts

**Files:**
- Create: `src/power_forecasting/proposals.py`
- Modify: `src/power_forecasting/models.py`
- Create: `tests/test_proposals.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_proposal_rejects_unknown_recipe_and_leakage():
    with pytest.raises(ProposalValidationError):
        load_proposal({"schema_version": "1", "proposal_id": "x",
                       "rationale": "test", "feature_sets": [], "model_recipes": [
                       {"name": "bad", "recipe": "python_eval", "parameters": {}}],
                       "budget": {"max_evaluations": 1, "top_feature_groups": 1}})
```

- [ ] **Step 2: Implement bounded contracts**

Define frozen proposal, feature-set, and model-recipe dataclasses. Validate exact
JSON keys, feature specs through existing safety checks, recipe parameter
allowlists, unique names, and budget <= 50. Add `model_definition_from_recipe`
for Ridge and HistGradientBoosting factories with validated parameters.

- [ ] **Step 3: Run targeted tests and commit**

Run: `python3 -m pytest tests/test_proposals.py -q`

```bash
git add src/power_forecasting/proposals.py src/power_forecasting/models.py tests/test_proposals.py
git commit -m "feat: add bounded research proposal contracts"
```

### Task 2: Proposal-aware AIDM evaluation

**Files:**
- Modify: `src/power_forecasting/aidm.py`
- Modify: `src/power_forecasting/cli.py`
- Create: `tests/test_agentic_aidm.py`

- [ ] **Step 1: Write failing proposal-run tests**

```python
def test_aidm_evaluates_recipe_feature_candidates(tmp_path):
    result = run_aidm(frame, tmp_path / "runs.db", proposal=proposal)
    assert result.manifest["proposal"]["proposal_id"] == "solar-v1"
    assert result.manifest["winner"]["model_recipe"]["recipe"] in {
        "ridge", "hist_gradient_boosting"
    }
```

- [ ] **Step 2: Implement candidate expansion**

Add optional proposal input while preserving the fixed catalog when absent.
Evaluate validated `recipe × feature_set` candidates within budget. Include recipe
and proposal metadata in experiment params, ranking, manifest, and reports.

- [ ] **Step 3: Add legacy comparison**

Accept a prediction CSV with `plant_id`, `timestamp`, and `prediction_mw`;
validate keys/coverage and compute NMAE. Require no regression against it when
provided. Add `--proposal` and `--legacy-predictions` CLI options.

- [ ] **Step 4: Run targeted tests and commit**

Run: `python3 -m pytest tests/test_agentic_aidm.py -q`

```bash
git add src/power_forecasting/aidm.py src/power_forecasting/cli.py tests/test_agentic_aidm.py
git commit -m "feat: evaluate agentic model and feature proposals"
```

### Task 3: AIDD patch request and agent instructions

**Files:**
- Modify: `src/power_forecasting/aidd.py`
- Modify: `.agents/scripts/verify-promotion.sh`
- Modify: `.agents/skills/aidm-experiment/SKILL.md`
- Modify: `.agents/skills/aidd-promotion/SKILL.md`
- Create: `tests/test_agentic_aidd.py`

- [ ] **Step 1: Write failing patch-request test**

```python
def test_promoted_agentic_manifest_creates_review_only_patch(tmp_path):
    request = render_model_recipe_patch(manifest, tmp_path / "model-recipe-patch.json")
    assert json.loads(request.read_text())["status"] == "requires_human_review"
```

- [ ] **Step 2: Implement deterministic patch request**

Validate promoted recipe evidence, create canonical JSON with checksums and
metrics, and refuse non-promoted or arbitrary recipe content. Update promotion
script and skills to require the proposal and prohibit direct customer edits.

- [ ] **Step 3: Run targeted tests and commit**

Run: `python3 -m pytest tests/test_agentic_aidd.py -q`

```bash
git add src/power_forecasting/aidd.py .agents tests/test_agentic_aidd.py
git commit -m "feat: add human-reviewed model recipe patch requests"
```

### Task 4: Fixture, documentation, and merged verification

**Files:**
- Create: `.agents/fixtures/research-proposal.json`
- Modify: `README.md`
- Modify: `tests/test_outer_harness.py`

- [ ] **Step 1: Add synthetic proposal fixture and harness workflow test**

Test proposal-aware `run-aidm.sh`, selected/rejected evidence, and no generated
customer patch.

- [ ] **Step 2: Document the proposal workflow**

Document JSON proposal shape, allowed recipes, feature safety boundaries, legacy
comparison, patch-request review, and non-deployment guarantee in Korean README.

- [ ] **Step 3: Run targeted tests and commit**

Run: `python3 -m pytest tests/test_proposals.py tests/test_agentic_aidm.py tests/test_agentic_aidd.py tests/test_outer_harness.py -q`

- [ ] **Step 4: Merge then run the full suite once**

Run after merging the worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:.agents python3 -m pytest -q -p no:cacheprovider
```
