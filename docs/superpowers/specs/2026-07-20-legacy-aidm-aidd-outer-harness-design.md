# Legacy AIDM/AIDD Outer Harness Design

## Purpose

Build repository-local agent infrastructure that applies AIDM and AIDD to a
customer legacy forecasting system without modifying, deploying, or directly
coupling to that system. The harness treats the legacy system as an executable
black box and exchanges only declared files.

## Scope

The harness lives in `.agents/` and provides:

- A versioned legacy-adapter contract.
- Safe shell entry points for baseline execution, AIDM experiments, and
  promotion verification.
- Synthetic fixtures for success, rejection, leakage, and malformed-contract
  paths.
- Agent skills for intake, experimentation, promotion, and release gating.
- A human-controlled evidence flow from baseline to generated feature module.

It does not deploy models, merge changes, execute undeclared commands, access
customer systems by default, or store customer data or credentials.

## Contract

Each customer integration supplies an adapter manifest JSON file. It declares:

- `schema_version`: exactly `"1"`.
- `legacy_command`: a non-empty array of executable and literal arguments.
- `input_dataset`: CSV path relative to the manifest directory.
- `predictions_output`: expected CSV path relative to the manifest directory.
- `required_prediction_columns`: required output columns.
- `timeout_seconds`: integer from 1 through 3600.

The runner resolves all relative paths beneath the manifest directory and rejects
path traversal. It invokes the command directly from the JSON array, never
through a shell or string interpolation. The process receives:

- `HARNESS_INPUT_DATASET`
- `HARNESS_PREDICTIONS_OUTPUT`
- `HARNESS_RUN_DIR`

The command must write the declared prediction CSV. The harness validates that it
exists, is non-empty, has exactly one header row, and contains each required
column. It records command metadata and output checksums without recording input
data contents.

## Workflow

```text
adapter manifest + legacy dataset
            |
            v
run-legacy.sh -> baseline evidence
            |
            v
run-aidm.sh -> AIDM experiment manifest and report
            |
            v
verify-promotion.sh -> AIDD validation, generated module, evidence summary
            |
            v
release-gate skill -> human decision: reject, iterate, or create reviewed patch
```

Each run uses a caller-provided or generated run ID under a user-selected output
directory. The output directory is not a repository source path and must be
ignored by Git. A failed command or rejected promotion produces evidence and a
non-zero exit status; it never falls back to a success-shaped result.

## Scripts

### `run-legacy.sh`

Accepts `--adapter`, `--run-dir`, and optional `--run-id`. It validates the
adapter using the Python contract helper, creates a run directory with restrictive
permissions, executes the declared legacy command with timeout enforcement, and
writes `legacy-evidence.json`.

### `run-aidm.sh`

Accepts `--dataset`, `--run-dir`, optional AIDM threshold/fold/seed options, and
invokes `python -m power_forecasting.cli aidm`. It requires a dataset that passes
the existing data contract and leaves `experiments.db`,
`promotion_manifest.json`, and `performance_report.md` in the run directory.

### `verify-promotion.sh`

Accepts `--run-dir`. It requires a promoted `promotion_manifest.json`, runs the
existing AIDD CLI path, and writes a checksum-based `promotion-evidence.json`
only after the generated module compiles. Rejected manifests fail without
generating code.

## Skills

### `legacy-intake`

Guides an agent to inspect a legacy system without editing it, define the adapter
manifest, map customer data into the project data contract, and run only
fixtures until the human explicitly authorizes real customer execution.

### `aidm-experiment`

Requires a baseline evidence directory, selects bounded feature hypotheses,
executes `run-aidm.sh`, compares the promotion manifest with the baseline, and
reports both selected and rejected candidates. It does not weaken gates.

### `aidd-promotion`

Requires a promoted manifest and verified run directory. It invokes
`verify-promotion.sh`, reviews the generated diff, and produces a patch request
instead of editing customer source directly.

### `release-gate`

Checks adapter evidence, AIDM gates, AIDD output, generated-code compilation,
and explicit human approval. It is a stop gate: a missing item rejects release.

## Fixtures

Fixtures are all synthetic:

- A valid adapter invoking a deterministic fixture legacy command.
- A valid dataset and predictions fixture.
- A malformed adapter using a traversal path.
- A predictions fixture missing a required column.
- A rejected promotion manifest fixture.
- A leakage manifest fixture using `actual_*` or `generation_mw`.

Fixtures enable local demonstrations and tests without customer data.

## Error Handling and Safety

- Reject malformed JSON, unknown manifest fields when strictness is required, and
  invalid command array elements.
- Reject paths escaping the manifest directory.
- Reject missing, empty, malformed, or schema-incomplete predictions.
- Redact command environment values from evidence; record only command array,
  exit code, duration, paths, and checksums.
- Do not accept dynamic shell fragments, `eval`, or command substitution.
- Do not generate code on rejected manifests.
- Require explicit human approval before applying any generated patch to a
  customer repository.

## Validation

Targeted harness tests validate contract parsing, path containment, command
execution, failure evidence, prediction schema checks, and promotion evidence.
The existing full project suite runs once after merging the harness worktree,
not after each implementation task.

## Success Criteria

- A new customer integration can be represented by one manifest and file
  contract, with no change to core AIDM/AIDD modules.
- The valid fixture demonstrates baseline -> AIDM -> AIDD evidence flow.
- Every rejection fixture fails closed and leaves inspectable evidence.
- Skills describe a reproducible, human-gated process for agents.
- No fixture, script, or committed evidence contains customer data, credentials,
  or an automated deployment path.
