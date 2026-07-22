---
name: research-diagnostic
description: Use when Stage 1 research-loop needs an aggregate-only diagnosis of a validated synthetic or explicitly approved dataset.
---

# Research Diagnostic

## Exact Responsibility
Produce one deterministic, aggregate-only diagnosis for the current research-loop run. Confirm
the dataset contract, time coverage, missingness, drift, residual aggregates, leakage checks, and
which fixed profiles are feasible. Do not invent candidates or run AIDM.

## Prerequisites
- A validated `research-loop.json` has existing dataset and baseline manifest paths.
- The run directory is under `.agents/runs` or `.agents/output`.

## Allowed Inputs
- `research-config.json` is the effective `schema_version: "1"` configuration snapshot.
- `dataset_path` is the configured CSV satisfying the repository data contract. Read rows only to
  compute aggregates; never copy rows into an artifact.
- `legacy_manifest_path` is an existing `schema_version: "1"` baseline manifest. The diagnostic
  may record only its path, status, and SHA-256.
- Configuration profiles are only `safe_weather`, `history_tree`, and `bounded_search`.

## Workflow
1. Read and checksum the configured inputs.
2. Compute aggregate diagnostics and leakage checks without retaining rows.
3. Write `diagnosis.json` and hand its bound checksum to the orchestrator.

## Output and Permissions
- On success, write only `diagnosis.json`; on a bounded diagnostic failure, write only
  `diagnostic-failure.json` in the run directory and the orchestrator's state/journal transition.
- `diagnosis.json` has `schema_version: "1"`, `dataset_sha256`, nonnegative `row_count` and
  `plant_count`, ISO `time_start`/`time_end`, aggregate `missingness`, `drift_summary`,
  `residual_summary`, boolean `leakage_checks`, and `recommended_profiles`.
- Read the configured inputs and existing run metadata. Do not write source, tests, fixtures,
  skills, scripts, generated code, AIDD artifacts, or customer systems.

## Evidence, Checksums, and Safety
- Record lowercase SHA-256 values for the dataset and baseline reference; bind the diagnosis to
  the effective configuration checksum and run ID.
- Use only aggregate counts, ratios, timestamps, and safe reason codes. Never log a customer row,
  raw target, `actual_*`, `generation_mw`, secret, token, credential, or environment value.
- A failed diagnostic writes bounded privacy-safe `diagnostic-failure.json` with `schema_version:
  "1"`, `status: "failed"`, `iteration`, and a safe `reason`; it never emits partial success
  evidence.

## Bounded Iteration and Stop Conditions
- Run once per loop, before proposal generation. The orchestrator caps total profile iterations
  at `max_iterations` (1..10).
- Stop immediately on missing/invalid inputs, failed leakage checks, checksum mismatch, or invalid
  schema. A successful diagnosis hands control to `research-proposal`; it does not promote.

## Prohibitions
Do not edit source or customer data, merge, deploy, call AIDD, weaken a gate, accept arbitrary
profiles, expose secrets, or log customer rows. A diagnosis is not release or deployment approval.

## Error Table
| Error | Action |
| --- | --- |
| Dataset or baseline missing | Fail closed; report a safe reason code only. |
| Contract or leakage failure | Stop and write `diagnostic-failure.json`; do not propose. |
| Checksum/config binding mismatch | Stop; require a new run. |

## Evidence Output Layout
`diagnosis.json` is bound in `state.json` and `journal.jsonl` on success. A failed diagnostic
binds `diagnostic-failure.json` instead; both artifacts contain only safe checksums/reason codes
and no raw-data sample.

## Post-Run Reflection
Report only status, profile feasibility, aggregate warnings, and checksums. State whether the next
safe step is proposal generation or human review; never claim code or deployment readiness.
