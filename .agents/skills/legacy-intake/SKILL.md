---
name: legacy-intake
description: Use when an agent is asked to connect a customer legacy forecasting system to this repository-local harness, inspect adapter readiness, map data, or prepare a safe baseline run.
---

# Legacy Intake

## Overview
Create a black-box adapter manifest without editing, deploying, or coupling to the customer legacy system. Start fixture-first and require explicit human approval before any customer execution.

## Prerequisites
- Confirm the work is local and read-only for customer systems.
- Use only synthetic fixtures until a human approves a real adapter and dataset.
- Keep credentials and customer data out of `.agents/fixtures/`, commits, logs, and evidence.
- Adapter manifest schema version must be `"1"`.

## Workflow
1. Identify the legacy command as a literal argv array; do not use shell fragments, `eval`, interpolation, or command substitution.
2. Map a dataset into the project data contract outside this skill; keep only relative manifest paths under the manifest directory.
3. Draft `input_dataset`, `predictions_output`, `required_prediction_columns`, and `timeout_seconds` using the strict adapter contract.
4. Run `.agents/scripts/run-legacy.sh --adapter .agents/fixtures/valid-adapter.json --run-dir .agents/runs/fixture-intake` first.
5. Inspect `legacy-evidence.json` for checksums, command argv, status, and required columns; do not expect input contents or environment values.
6. If fixtures pass and the human approves, run the customer adapter in a non-source run directory.
7. Summarize accepted evidence and any rejected contract issues.

## Error Table
| Error | Action |
| --- | --- |
| Path escapes manifest directory | Reject the adapter and ask for contained relative paths. |
| Missing prediction column | Reject baseline evidence and fix the legacy export mapping. |
| Empty CSV or duplicate header | Reject; require a real prediction table with one header. |
| Customer data in fixtures or commits | Stop and remove it before continuing. |

## Evidence Output Layout
- `legacy-evidence.json`: status, run ID, manifest checksum, command argv, named HARNESS environment keys, input checksum, prediction checksum, row count, and errors.
- No customer row contents, secrets, or environment values belong in evidence.

## Post-Run Reflection
Record whether the run stayed fixture-first, whether a human approved customer execution, what failed closed, and what must change before AIDM.
