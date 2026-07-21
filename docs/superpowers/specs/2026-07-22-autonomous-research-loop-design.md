# Autonomous Research Loop Design

## Goal

Add an opt-in autonomous research loop that diagnoses a forecasting dataset,
generates bounded research proposals, runs AIDM experiments, independently
verifies evidence, and iterates within fixed budgets. Preserve every existing
manual CLI, script, skill, and promotion workflow unchanged.

## Scope

Stage 1 ends at a verified research recommendation. It may produce AIDM
artifacts and a human-review recommendation, but it does not edit source code,
create a customer patch, merge, deploy, or change production systems.

The existing manual paths remain authoritative:

- `run-legacy.sh`
- `run-aidm.sh`
- `verify-promotion.sh`
- `legacy-intake`
- `aidm-experiment`
- `aidd-promotion`
- `release-gate`

The research loop is a new explicit command and never runs implicitly from
those paths.

## Architecture

### Orchestrator

The orchestrator owns a persisted state machine, iteration budget, stop
conditions, and artifact paths. It does not invent model code or override an
agent result. It invokes role implementations in order and accepts a transition
only when the expected artifact validates.

States:

```text
initialized
  -> diagnosed
  -> proposed
  -> experimenting
  -> verifying
  -> iterate | ready_for_human_review | exhausted | failed
```

Terminal states are immutable. Restarting a terminal run requires a new run ID.
An interrupted non-terminal run may resume only when all recorded artifact
checksums still match.

### Diagnostic agent

The diagnostic role is read-only. It validates the dataset and baseline
evidence, then writes `diagnosis.json` with:

- schema version and run ID
- dataset checksum, row count, plant count, and time coverage
- chronological-fold feasibility
- available prediction-time inputs
- history-feature feasibility
- baseline and legacy-evidence references
- warnings and rejected conditions

It records no customer rows, credentials, environment values, or raw target
samples.

### Research agent

The research role writes proposal JSON only. It selects from versioned bounded
proposal profiles using diagnosis facts:

- forecast/weather interactions
- history features when sufficient chronological coverage exists
- bounded Ridge/HGB/RF/XGBoost/LightGBM recipes
- optional bounded LightGBM Optuna search

Every proposal is validated by the existing `ResearchProposal` contract.
Unknown models, arbitrary code, callbacks, custom search distributions, target
inputs, `actual_*` inputs, or gate changes are impossible outputs.

The role records rationale and rejected ideas in `research-proposal.json` and
`research-notes.json`.

### Experiment agent

The experiment role calls the existing AIDM workflow with the selected
proposal. It may not change promotion thresholds after the run starts. It
writes the normal `experiments.db`, `promotion_manifest.json`, and
`performance_report.md` inside an iteration directory.

Experiment failures are evidence. They do not cause the agent to weaken gates
or edit estimator code.

### Verification agent

The verification role is independent from proposal selection. It validates:

- dataset, diagnosis, proposal, manifest, and report checksums
- proposal budget and selected-recipe provenance
- SPOT and optional legacy-baseline gates
- chronological fold metadata
- completed/failed experiment records
- consistency between winner, selected feature set, and model recipe

It writes `verification.json` with `pass`, `reject`, or `invalid`. Only `pass`
may transition to `ready_for_human_review`.

### Coordinator policy

The coordinator uses deterministic rules:

1. Stop immediately on invalid dataset, missing required baseline evidence, or
   artifact checksum mismatch.
2. If verification passes and the manifest promotes, stop at
   `ready_for_human_review`.
3. If the run is valid but rejected and unused proposal profiles remain,
   advance to the next profile.
4. Stop at `exhausted` when iteration or evaluation budgets are consumed.
5. Never ask an agent to weaken gates, rewrite evidence, or approve its own
   output.

## Run contract

The input `research-loop.json` has exact keys:

- `schema_version`: exactly `"1"`
- `run_id`: safe identifier
- `dataset`: explicit repository-relative or absolute input path
- `run_dir`: explicit non-source output directory
- `legacy_evidence`: optional path
- `legacy_predictions`: optional path
- `profiles`: ordered non-empty list from the supported profile allowlist
- `folds`: integer `1..10`, with five recommended for real data
- `seed`: nonnegative integer
- `max_iterations`: integer `1..10`
- `minimum_improvement` and `max_plant_regression`: finite values validated by
  the existing AIDM configuration contract

Relative paths resolve against the configuration file's directory. The run
directory must not overlap source, tests, documentation, fixtures, skills,
scripts, or harness code. Repository-local `.agents/runs` and `.agents/output`
are the only allowed `.agents` destinations.

The loop may use less than the configured budget but never more. It must not
modify the configuration after `initialized`.

## Artifact layout

```text
<run-dir>/
├── research-loop.json
├── state.json
├── journal.jsonl
├── diagnosis.json
├── research-summary.json
└── iterations/
    └── 001-<profile>/
        ├── research-proposal.json
        ├── research-notes.json
        ├── experiments.db
        ├── promotion_manifest.json
        ├── performance_report.md
        └── verification.json
```

`state.json` is written atomically. `journal.jsonl` is append-only and contains
state transitions, timestamps, artifact paths, and checksums but no customer
row contents.

## Compatibility

Existing AIDM behavior is unchanged when no research-loop command is used.
Existing proposal files remain valid. Existing scripts retain their arguments
and output shapes. The loop reuses public validation and workflow APIs instead
of duplicating model, feature, gate, or provenance logic.

## Safety boundaries

- Local filesystem execution only.
- No source edits, worktree creation, commits, PRs, merges, deployment, network
  calls, scheduler integration, or customer-system writes.
- No agent may both generate a proposal and approve its evidence.
- No shell fragments or dynamic command construction.
- Explicit iteration, candidate, trial, and timeout budgets.
- Fail closed on missing evidence, invalid JSON, unknown profiles, interrupted
  writes, or checksum drift.

## Reusable `.agents` assets

Add reusable role skills for diagnostic, research, experiment, verification,
and orchestration. Add a fixture configuration and script that exercise the
complete loop on synthetic data. Each skill states its allowed inputs,
artifacts, permissions, stop conditions, and forbidden actions so another
project can adapt the contracts without copying repository internals.

## Validation

Test:

- strict loop configuration parsing
- atomic state and append-only journal behavior
- diagnosis redaction and fold feasibility
- deterministic profile selection and proposal generation
- preservation of existing proposal validation
- experiment artifact generation
- independent verification and tamper rejection
- iteration, resume, exhausted, promoted, and failed transitions
- existing manual CLI/script compatibility
- reusable fixture and `.agents` skill documentation

Run targeted tests during implementation and the full suite once after merging.
