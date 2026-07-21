# Autonomous Research Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, evidence-driven autonomous diagnosis, proposal, AIDM experiment, and verification loop without changing existing manual workflows.

**Architecture:** A new `power_forecasting.research` package contains strict run contracts, persisted state, deterministic diagnostic/profile agents, independent verification, and an orchestrator. The existing AIDM workflow remains the experiment engine; new `.agents` assets expose reusable role contracts and a fixture-first runner.

**Tech Stack:** Python 3.9+, pandas, SQLite, existing AIDM/AIDD contracts, argparse, Bash, pytest, uv.

---

### Task 1: Define strict loop contracts and persistent state

**Files:**
- Create: `src/power_forecasting/research/__init__.py`
- Create: `src/power_forecasting/research/contracts.py`
- Create: `src/power_forecasting/research/state.py`
- Create: `tests/test_research_contracts.py`
- Create: `tests/test_research_state.py`

- [ ] **Step 1: Write failing configuration tests**

Add tests for a valid exact-key configuration and rejection of unknown keys,
unsafe IDs, source-directory run paths, missing datasets, invalid thresholds,
unknown profiles, duplicate profiles, folds outside `1..10`, negative seeds,
and iteration budgets outside `1..10`.

```python
def test_valid_research_loop_config_round_trips(tmp_path):
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("plant_id,timestamp\n", encoding="utf-8")
    payload = {
        "schema_version": "1",
        "run_id": "fixture_loop",
        "dataset": str(dataset),
        "run_dir": str(tmp_path / "runs" / "fixture_loop"),
        "legacy_evidence": None,
        "legacy_predictions": None,
        "profiles": ["safe_weather", "history_tree"],
        "folds": 1,
        "seed": 17,
        "max_iterations": 2,
        "minimum_improvement": 0.01,
        "max_plant_regression": 0.03,
    }
    config = load_research_loop_config(payload)
    assert research_loop_config_to_dict(config) == payload
```

- [ ] **Step 2: Run contract tests to verify RED**

Run:

```bash
uv run pytest tests/test_research_contracts.py -q
```

Expected: FAIL because `power_forecasting.research.contracts` does not exist.

- [ ] **Step 3: Implement immutable contracts**

Define:

```python
class ResearchLoopValidationError(ValueError): ...

@dataclass(frozen=True)
class ResearchLoopConfig:
    schema_version: str
    run_id: str
    dataset: Path
    run_dir: Path
    legacy_evidence: Path | None
    legacy_predictions: Path | None
    profiles: tuple[str, ...]
    folds: int
    seed: int
    max_iterations: int
    minimum_improvement: float
    max_plant_regression: float
```

Supported profiles are exactly `safe_weather`, `history_tree`, and
`bounded_search`. Resolve relative paths against the configuration file's
directory. Validate with exact keys, finite numbers, safe identifiers,
explicit existing input paths, and a run directory outside `src`, `tests`,
`docs`, and protected `.agents` content. Permit only `.agents/runs` and
`.agents/output` as repository-local `.agents` destinations.

- [ ] **Step 4: Write failing state-machine tests**

Test valid transitions, invalid skipped transitions, terminal immutability,
atomic `state.json`, append-only `journal.jsonl`, checksum recording, and
resume rejection after artifact tampering.

```python
def test_terminal_state_cannot_transition(tmp_path):
    store = ResearchStateStore(tmp_path, "run_1")
    store.initialize("config.json", "abc")
    store.transition("failed", reason="invalid dataset")
    with pytest.raises(ResearchStateError, match="terminal"):
        store.transition("diagnosed", artifact="diagnosis.json")
```

- [ ] **Step 5: Implement the state store**

Use `tempfile.NamedTemporaryFile` in the destination directory followed by
`os.replace` for `state.json`. Open `journal.jsonl` with append mode and write
one canonical JSON object per transition. Store SHA-256 values for referenced
artifacts. Resume validates every recorded checksum before returning state.

- [ ] **Step 6: Run focused contract/state tests**

Run:

```bash
uv run pytest tests/test_research_contracts.py tests/test_research_state.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the contracts**

```bash
git add src/power_forecasting/research tests/test_research_contracts.py tests/test_research_state.py
git commit -m "feat: add autonomous research loop contracts"
```

### Task 2: Implement diagnostic and proposal agents

**Files:**
- Create: `src/power_forecasting/research/diagnostic.py`
- Create: `src/power_forecasting/research/profiles.py`
- Create: `tests/test_research_diagnostic.py`
- Create: `tests/test_research_profiles.py`

- [ ] **Step 1: Write failing diagnostic tests**

Test deterministic diagnosis for synthetic data, chronological-fold
feasibility, prediction-input names, history feasibility, checksum references,
and absence of raw customer rows, target samples, environment values, and
secrets.

```python
def test_diagnosis_contains_aggregates_not_rows(tmp_path):
    dataset = tmp_path / "dataset.csv"
    frame = generate_synthetic_data(days=5, plants=2, seed=3)
    frame.to_csv(dataset, index=False)
    diagnosis = diagnose_dataset(dataset, folds=1, run_id="run_1")
    assert diagnosis["row_count"] == len(frame)
    assert diagnosis["plant_count"] == 2
    assert "generation_mw" not in diagnosis["prediction_time_inputs"]
    assert "rows" not in diagnosis
```

- [ ] **Step 2: Verify diagnostic RED**

Run:

```bash
uv run pytest tests/test_research_diagnostic.py -q
```

Expected: FAIL because the diagnostic module does not exist.

- [ ] **Step 3: Implement read-only diagnosis**

Reuse `validate_dataset`, `parse_timestamps`, `chronological_folds`, and
`PREDICTION_TIME_INPUTS`. Return canonical JSON-compatible aggregates only.
Validate configured legacy evidence as JSON with `status == "success"` and
record its checksum/path, never its customer contents.

- [ ] **Step 4: Write failing profile tests**

For each supported profile, test deterministic proposal generation, successful
`load_proposal()`, unique IDs derived from run/profile/iteration, profile
preconditions, rejected-idea notes, and no target/actual inputs.

```python
def test_safe_weather_profile_produces_valid_bounded_proposal():
    proposal, notes = build_profile_proposal(
        profile="safe_weather",
        diagnosis=_diagnosis(history_feasible=True),
        run_id="run_1",
        iteration=1,
        seed=17,
    )
    assert load_proposal(proposal).proposal_id == "run_1_safe_weather_001"
    assert notes["rejected_ideas"]
```

- [ ] **Step 5: Implement deterministic profiles**

Implement:

- `safe_weather`: cyclic time, effective irradiance, weather interaction;
  Ridge and HGB recipes.
- `history_tree`: lag/rolling forecast inputs; Random Forest and HGB recipes;
  reject profile when diagnosis says history is infeasible.
- `bounded_search`: safe weather/history features, one direct LightGBM recipe,
  and bounded LightGBM TPE search; reject when the optional model-search extra
  cannot initialize.

Every generated proposal must pass `load_proposal()` before being returned.

- [ ] **Step 6: Run diagnostic/profile tests**

Run:

```bash
uv run pytest tests/test_research_diagnostic.py tests/test_research_profiles.py tests/test_proposals.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit role implementations**

```bash
git add src/power_forecasting/research/diagnostic.py src/power_forecasting/research/profiles.py tests/test_research_diagnostic.py tests/test_research_profiles.py
git commit -m "feat: add diagnostic and proposal agents"
```

### Task 3: Implement experiment and independent verification agents

**Files:**
- Create: `src/power_forecasting/research/experiment.py`
- Create: `src/power_forecasting/research/verification.py`
- Create: `tests/test_research_experiment.py`
- Create: `tests/test_research_verification.py`

- [ ] **Step 1: Write failing experiment-agent tests**

Test that the agent creates an iteration directory, writes the exact validated
proposal and notes atomically, invokes `run_aidm_workflow` with immutable
thresholds/folds/seed, and writes the standard database, manifest, and report.
Test rejected experiments remain valid evidence.

- [ ] **Step 2: Verify experiment RED**

Run:

```bash
uv run pytest tests/test_research_experiment.py -q
```

Expected: FAIL because the experiment module does not exist.

- [ ] **Step 3: Implement the experiment agent**

Define:

```python
@dataclass(frozen=True)
class ResearchIterationResult:
    iteration: int
    profile: str
    directory: Path
    proposal_path: Path
    manifest_path: Path
    database_path: Path
    report_path: Path
```

Call the public CLI workflow functions rather than constructing estimators or
gates. Never catch an experiment failure as success; write a failure artifact
and re-raise a typed research-loop error.

- [ ] **Step 4: Write failing independent-verifier tests**

Test:

- promoted, internally consistent evidence returns `pass`
- valid rejected evidence returns `reject`
- checksum drift, missing report/database, proposal mismatch, selected recipe
  mismatch, threshold mismatch, or incomplete experiment runs returns `invalid`
- verifier never rewrites source evidence

- [ ] **Step 5: Implement verification**

Reload proposal with `load_proposal()`, inspect the SQLite store with
`ExperimentStore`, and validate manifest fields against the run configuration.
For promoted manifests, call existing promotion/provenance validation without
generating code. Write canonical `verification.json` containing status,
accepted checksums, issues, decision, and verifier schema version.

- [ ] **Step 6: Run experiment/verifier tests**

Run:

```bash
uv run pytest tests/test_research_experiment.py tests/test_research_verification.py tests/test_agentic_aidm.py tests/test_agentic_aidd.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit experiment and verification**

```bash
git add src/power_forecasting/research/experiment.py src/power_forecasting/research/verification.py tests/test_research_experiment.py tests/test_research_verification.py
git commit -m "feat: add research experiment verification agents"
```

### Task 4: Implement orchestrator, resume, and CLI

**Files:**
- Create: `src/power_forecasting/research/orchestrator.py`
- Modify: `src/power_forecasting/cli.py`
- Create: `tests/test_research_orchestrator.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing orchestrator transition tests**

Test:

- first promoted/verified profile reaches `ready_for_human_review`
- rejected profile advances to the next unused profile
- iteration exhaustion reaches `exhausted`
- invalid verification reaches `failed`
- interrupted runs resume from the last checksum-valid non-terminal state
- terminal runs reject resume
- no code-generation, worktree, Git, network, or deployment function is called

- [ ] **Step 2: Verify orchestrator RED**

Run:

```bash
uv run pytest tests/test_research_orchestrator.py -q
```

Expected: FAIL because the orchestrator module does not exist.

- [ ] **Step 3: Implement deterministic orchestration**

Implement `run_research_loop(config, *, resume=False)` with injected role
functions for tests. Persist every accepted artifact before its transition.
Use the ordered profile list and stop rules from the design. Produce
`research-summary.json` with terminal state, attempted profiles, iteration
results, final recommendation, and evidence checksums.

- [ ] **Step 4: Add the CLI command**

Add:

```text
power-forecast research-loop --config PATH [--resume]
```

The command loads exact configuration, prints run directory, terminal state,
summary path, and returns:

- `0` for `ready_for_human_review`
- `2` for `exhausted`, `failed`, invalid config, or invalid evidence

Existing commands and parser options must remain unchanged.

- [ ] **Step 5: Run orchestrator and CLI tests**

Run:

```bash
uv run pytest tests/test_research_orchestrator.py tests/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/power_forecasting/research/orchestrator.py src/power_forecasting/cli.py tests/test_research_orchestrator.py tests/test_cli.py
git commit -m "feat: add autonomous research loop command"
```

### Task 5: Add reusable `.agents` role assets and fixture workflow

**Files:**
- Create: `.agents/scripts/run-research-loop.sh`
- Create: `.agents/fixtures/research-loop.json`
- Create: `.agents/skills/research-diagnostic/SKILL.md`
- Create: `.agents/skills/research-proposal/SKILL.md`
- Create: `.agents/skills/research-verification/SKILL.md`
- Create: `.agents/skills/research-orchestrator/SKILL.md`
- Modify: `.agents/skills/aidm-experiment/SKILL.md`
- Modify: `.agents/skills/release-gate/SKILL.md`
- Modify: `README.md`
- Modify: `tests/test_outer_harness.py`

- [ ] **Step 1: Write failing fixture and skill-contract tests**

Test that:

- fixture config parses, resolves paths relative to the fixture file, uses only
  synthetic repository fixtures, and writes under `.agents/runs`
- runner uses `uv run python -m power_forecasting.cli research-loop`
- all four skills contain allowed inputs, artifacts, permissions, stop
  conditions, and forbidden code/deploy actions
- existing scripts are byte-for-byte behavior compatible

- [ ] **Step 2: Verify asset tests RED**

Run:

```bash
uv run pytest tests/test_outer_harness.py -q
```

Expected: FAIL because the new assets do not exist.

- [ ] **Step 3: Add the fixture-first runner**

`run-research-loop.sh` validates `--config`, resolves repository root, exports
only the existing project paths, changes to the root, and executes:

```bash
uv run python -m power_forecasting.cli research-loop --config "$config" "${args[@]}"
```

It accepts only `--config`, `--resume`, and help; unknown/missing options exit
`2`.

- [ ] **Step 4: Write reusable role skills**

Each skill must state:

- exact role responsibility
- allowed input artifact schemas
- output artifact names
- read/write permissions
- evidence and checksum requirements
- iteration/stop conditions
- explicit prohibition on source edits, merge, deployment, gate weakening,
  secrets, and customer-row logging

Update `aidm-experiment` to explain orchestrated invocation remains optional.
Update `release-gate` to reject research-loop summaries as deployment approval.

- [ ] **Step 5: Update Korean documentation**

Document:

- manual workflows remain available and unchanged
- opt-in autonomous Stage 1 flow and state diagram
- fixture command
- artifacts, resume rules, budgets, and terminal states
- explicit boundary before AIDD/code changes/deployment

- [ ] **Step 6: Run targeted integration tests**

Run:

```bash
uv run pytest tests/test_outer_harness.py tests/test_cli.py tests/test_research_contracts.py tests/test_research_state.py tests/test_research_diagnostic.py tests/test_research_profiles.py tests/test_research_experiment.py tests/test_research_verification.py tests/test_research_orchestrator.py -q
git diff --check
uv lock --locked
```

Expected: all tests pass; lockfile is unchanged/current.

- [ ] **Step 7: Commit reusable assets**

```bash
git add .agents README.md tests/test_outer_harness.py
git commit -m "docs: add reusable autonomous research agents"
```

### Task 6: Final review, merge, and one full verification

**Files:**
- Verify: all files above

- [ ] **Step 1: Run independent specification and code-quality review**

Review the entire feature branch against
`docs/superpowers/specs/2026-07-22-autonomous-research-loop-design.md`. Fix and
re-review every material issue before merging.

- [ ] **Step 2: Merge locally**

Merge `feature/autonomous-research-loop` into `main` without changing the
existing untracked PPTX or `.DS_Store` files.

- [ ] **Step 3: Run the full suite once after merge**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -q -p no:cacheprovider
```

Expected: the full suite passes.

- [ ] **Step 4: Clean the worktree**

Remove the merged `.worktrees/autonomous-research-loop` worktree, prune
worktrees, and delete the merged feature branch. Leave generated PPTX files
untracked and uncommitted.
