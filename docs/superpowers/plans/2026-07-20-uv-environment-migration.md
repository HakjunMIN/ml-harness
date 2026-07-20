# uv Environment Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make uv the reproducible, documented environment for development and harness execution.

**Architecture:** `pyproject.toml` remains the sole dependency declaration and
`uv.lock` captures its resolved packages. Local users and harness shell scripts
enter the project environment with `uv run`; documentation describes `uv sync`
as the environment bootstrap command.

**Tech Stack:** Python 3.9+, setuptools, uv, pytest, Bash.

---

### Task 1: Create the reproducible uv lockfile

**Files:**
- Create: `uv.lock`
- Modify: `pyproject.toml` only if `uv lock` requires valid metadata correction
- Test: `uv.lock` resolution through `uv lock --locked`

- [ ] **Step 1: Generate the project lockfile**

Run:

```bash
cd /Users/andy/works/ai/ml-harness
uv lock
```

Expected: creates `uv.lock` using the existing Python requirement
`>=3.9` and the runtime, `dev`, and `dashboard` extras declared in
`pyproject.toml`.

- [ ] **Step 2: Verify the checked-in lock is current**

Run:

```bash
cd /Users/andy/works/ai/ml-harness
uv lock --locked
```

Expected: exits `0` without modifying `uv.lock`.

- [ ] **Step 3: Commit the lockfile**

```bash
git add uv.lock
git commit -m "build: add uv lockfile"
```

### Task 2: Execute outer-harness Python through uv

**Files:**
- Modify: `.agents/scripts/run-legacy.sh:28-33`
- Modify: `.agents/scripts/run-aidm.sh:39-41`
- Modify: `.agents/scripts/verify-promotion.sh:28-57`
- Test: `tests/test_outer_harness.py`

- [ ] **Step 1: Write the failing command-contract test**

Add assertions in `tests/test_outer_harness.py` that each harness script
contains the uv invocation used by its Python process:

```python
def test_harness_scripts_run_python_through_uv() -> None:
    scripts = (
        "run-legacy.sh",
        "run-aidm.sh",
        "verify-promotion.sh",
    )
    for script in scripts:
        contents = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
        assert "uv run python" in contents
```

- [ ] **Step 2: Run the new test to verify it fails**

Run:

```bash
uv run pytest tests/test_outer_harness.py::test_harness_scripts_run_python_through_uv -q
```

Expected: FAIL because the scripts currently invoke `python3`.

- [ ] **Step 3: Replace ambient Python process invocations**

Update each script to preserve its existing argv arrays, `PYTHONPATH`, error
cleanup, and current working directory, but invoke the project interpreter:

```bash
args=(uv run python -m harness.contract --adapter "$adapter" --run-dir "$run_dir")
uv run python -m power_forecasting.cli aidm --output "$run_dir" --dataset "$dataset" "${args[@]}"
uv run python -m power_forecasting.cli aidd --output "$run_dir" --manifest "$manifest"
uv run python -m py_compile "$generated"
```

Use `uv run python -` for each existing stdin Python heredoc in
`verify-promotion.sh`.

- [ ] **Step 4: Run focused harness tests**

Run:

```bash
uv run pytest tests/test_outer_harness.py -q
```

Expected: PASS, including legacy intake, AIDM invocation, promotion verification,
and command-contract coverage.

- [ ] **Step 5: Commit the harness migration**

```bash
git add .agents/scripts/run-legacy.sh .agents/scripts/run-aidm.sh .agents/scripts/verify-promotion.sh tests/test_outer_harness.py
git commit -m "build: run harness scripts through uv"
```

### Task 3: Document uv as the project workflow

**Files:**
- Modify: `README.md:14-79`
- Test: README command snippets checked by inspection

- [ ] **Step 1: Replace bootstrap commands**

Replace the venv and pip commands in the quick start block with:

```bash
uv sync --all-extras
uv run pytest -q
```

- [ ] **Step 2: Prefix documented Python commands**

Update every executable command in the README that currently begins
`python3 -m` to start `uv run python -m`, preserving all CLI arguments. Update
the dashboard command to:

```bash
uv run streamlit run dashboard/app.py
```

- [ ] **Step 3: Add a concise lockfile statement**

Immediately after quick start, state in Korean that `uv.lock` is committed and
`uv sync --all-extras` reproduces the development environment.

- [ ] **Step 4: Verify changed documentation commands**

Run:

```bash
uv run python -m power_forecasting.cli --help
uv run pytest tests/test_cli.py -q
```

Expected: the CLI displays help and all CLI tests pass from the locked
environment.

- [ ] **Step 5: Commit the documentation**

```bash
git add README.md
git commit -m "docs: document uv workflow"
```

### Task 4: Validate the integrated locked environment

**Files:**
- Verify: `pyproject.toml`
- Verify: `uv.lock`
- Verify: `.agents/scripts/run-legacy.sh`
- Verify: `.agents/scripts/run-aidm.sh`
- Verify: `.agents/scripts/verify-promotion.sh`
- Verify: `README.md`

- [ ] **Step 1: Confirm no stale ambient invocation remains**

Run:

```bash
rg -n '(^|[[:space:]])python3[[:space:]]+-m|python3[[:space:]]+-' .agents/scripts README.md
```

Expected: no matches.

- [ ] **Step 2: Confirm the lock is unchanged and run targeted regression coverage**

Run:

```bash
uv lock --locked
uv run pytest tests/test_outer_harness.py tests/test_cli.py -q
```

Expected: both commands exit `0`.

- [ ] **Step 3: Check the final patch**

Run:

```bash
git diff --check HEAD~3..HEAD
git status --short
```

Expected: no whitespace errors and a clean worktree.

- [ ] **Step 4: Commit any final correction**

If validation requires a correction, commit it independently:

```bash
git add <corrected-files>
git commit -m "fix: complete uv environment migration"
```
