# Legacy AIDM/AIDD Outer Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a safe `.agents/` outer harness that executes declared legacy adapters, runs AIDM/AIDD workflows, and produces human-reviewable evidence.

**Architecture:** Python owns manifest parsing, path containment, prediction validation, checksums, and evidence serialization. Shell scripts are thin argument parsers that call the Python helper and existing project CLI; skills orchestrate the scripts but cannot bypass release gates.

**Tech Stack:** Python 3.9+, standard library, Bash, existing `power_forecasting` package, pytest

---

## File Map

| Path | Responsibility |
| --- | --- |
| `.agents/harness/contract.py` | Parse and validate adapter manifests, execute commands safely, validate predictions, and write evidence. |
| `.agents/scripts/run-legacy.sh` | Invoke declared legacy adapter commands. |
| `.agents/scripts/run-aidm.sh` | Run existing AIDM CLI against a selected dataset. |
| `.agents/scripts/verify-promotion.sh` | Validate promoted manifests and write AIDD evidence. |
| `.agents/fixtures/` | Synthetic adapter, data, predictions, and rejection inputs. |
| `.agents/skills/*/SKILL.md` | Agent workflows and non-bypassable release guidance. |
| `tests/test_outer_harness.py` | Validate contract and fixture-driven script behavior. |
| `.gitignore` | Ignore `.agents/runs/` and local harness outputs. |

### Task 1: Harness contract module and fixtures

**Files:**
- Create: `.agents/harness/__init__.py`
- Create: `.agents/harness/contract.py`
- Create: `.agents/fixtures/valid-adapter.json`
- Create: `.agents/fixtures/valid-dataset.csv`
- Create: `.agents/fixtures/valid-predictions.csv`
- Create: `.agents/fixtures/legacy_fixture.py`
- Create: `tests/test_outer_harness.py`

- [ ] **Step 1: Write failing contract tests**

```python
def test_adapter_rejects_path_traversal(tmp_path):
    manifest = tmp_path / "adapter.json"
    manifest.write_text('{"schema_version":"1","legacy_command":["python3"],'
                        '"input_dataset":"../secret.csv","predictions_output":"predictions.csv",'
                        '"required_prediction_columns":["plant_id","timestamp","prediction_mw"],'
                        '"timeout_seconds":60}')
    with pytest.raises(AdapterContractError, match="escapes"):
        load_adapter(manifest)
```

- [ ] **Step 2: Run targeted test to verify failure**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: FAIL because `harness.contract` does not exist.

- [ ] **Step 3: Implement contract helper**

Define `AdapterContractError`, frozen `AdapterConfig`, `load_adapter(path)`,
`run_adapter(adapter_path, run_dir, run_id)`, and `sha256_file(path)`. Require
the exact schema fields and types, resolve paths under the manifest directory,
run commands with `subprocess.run(command_array, shell=False, timeout=...)`,
and emit deterministic `legacy-evidence.json`.

- [ ] **Step 4: Add valid and failure tests**

Test valid fixture execution, missing output, missing required prediction columns,
timeout, checksum stability, and evidence redaction. Fixtures must contain only
synthetic values.

- [ ] **Step 5: Run targeted tests and commit**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: PASS.

```bash
git add .agents/harness .agents/fixtures tests/test_outer_harness.py
git commit -m "feat: add legacy adapter harness contract"
```

### Task 2: Shell workflow entry points

**Files:**
- Create: `.agents/scripts/run-legacy.sh`
- Create: `.agents/scripts/run-aidm.sh`
- Create: `.agents/scripts/verify-promotion.sh`
- Modify: `tests/test_outer_harness.py`

- [ ] **Step 1: Write failing subprocess tests**

```python
def test_run_legacy_script_creates_evidence(tmp_path):
    completed = subprocess.run(
        [".agents/scripts/run-legacy.sh", "--adapter",
         ".agents/fixtures/valid-adapter.json", "--run-dir", str(tmp_path)],
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "legacy-evidence.json").exists()
```

- [ ] **Step 2: Run targeted test to verify failure**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: FAIL because scripts do not exist.

- [ ] **Step 3: Implement scripts**

Use `set -euo pipefail`, explicit option parsing, quoted arguments, and
repository-root discovery. `run-legacy.sh` calls `python3 -m harness.contract`.
`run-aidm.sh` calls the existing project CLI with an explicit output directory.
`verify-promotion.sh` requires `decision == "promote"`, invokes the existing
AIDD CLI, compiles the generated module, and writes checksum-only evidence.

- [ ] **Step 4: Add rejection tests**

Test unknown options, absent input paths, a reject manifest, and that failed AIDD
verification does not create a generated module or success evidence.

- [ ] **Step 5: Run targeted tests and commit**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: PASS.

```bash
git add .agents/scripts tests/test_outer_harness.py
git commit -m "feat: add AIDM AIDD harness scripts"
```

### Task 3: Agent skills and release gate

**Files:**
- Create: `.agents/skills/legacy-intake/SKILL.md`
- Create: `.agents/skills/aidm-experiment/SKILL.md`
- Create: `.agents/skills/aidd-promotion/SKILL.md`
- Create: `.agents/skills/release-gate/SKILL.md`
- Modify: `.gitignore`
- Modify: `tests/test_outer_harness.py`

- [ ] **Step 1: Write failing skill structure tests**

```python
def test_release_gate_requires_human_approval():
    content = Path(".agents/skills/release-gate/SKILL.md").read_text()
    assert "human approval" in content.lower()
    assert "must not deploy" in content.lower()
```

- [ ] **Step 2: Run targeted test to verify failure**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: FAIL because release gate skill does not exist.

- [ ] **Step 3: Author the skills**

Each skill must have frontmatter, trigger language, prerequisites, numbered
workflow, failure table, evidence output, and a post-run reflection. The release
gate must fail closed when baseline, AIDM, AIDD, compilation, or human approval
is absent. No skill may instruct an agent to deploy, merge, or edit customer code
without a human-approved patch request.

- [ ] **Step 4: Ignore local harness run outputs**

Add `.agents/runs/` and `.agents/output/` to `.gitignore`; keep committed
fixtures and skills tracked.

- [ ] **Step 5: Run targeted tests and commit**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: PASS.

```bash
git add .agents/skills .gitignore tests/test_outer_harness.py
git commit -m "feat: add agent workflows for legacy promotion"
```

### Task 4: Documentation and integration check

**Files:**
- Modify: `README.md`
- Modify: `tests/test_outer_harness.py`

- [ ] **Step 1: Write a failing documentation assertion**

```python
def test_readme_documents_outer_harness():
    readme = Path("README.md").read_text()
    assert ".agents/scripts/run-legacy.sh" in readme
    assert "human approval" in readme.lower()
```

- [ ] **Step 2: Run targeted test to verify failure**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: FAIL because README does not document the harness.

- [ ] **Step 3: Document safe integration**

Add Korean documentation covering adapter manifest fields, fixture-first use,
script examples, evidence files, customer-data restrictions, rejection behavior,
and human promotion approval.

- [ ] **Step 4: Run targeted test**

Run: `PYTHONPATH=.agents python3 -m pytest tests/test_outer_harness.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_outer_harness.py
git commit -m "docs: explain legacy outer harness"
```

## Plan Self-Review

- Tasks 1-4 cover contract isolation, execution, AIDM/AIDD workflow, fixtures,
  agent skills, safety gates, and documentation.
- No task adds deployment, automatic merging, customer data, credentials, or
  shell interpolation.
- Public names remain consistent: `AdapterContractError`, `AdapterConfig`,
  `load_adapter`, `run_adapter`, `run-legacy.sh`, `run-aidm.sh`, and
  `verify-promotion.sh`.
- Full-project tests run only once after worktree merge; each task runs its
  targeted outer-harness test file.
