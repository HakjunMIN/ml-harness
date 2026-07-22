---
name: research-verification
description: Use when Stage 1 research-loop must verify one AIDM experiment's evidence and classify it without changing gates.
---

# Research Verification

## Exact Responsibility
Fail closed on one iteration's evidence. Verify that the proposal, AIDM manifest, performance
report, SQLite experiment database, and experiment evidence describe the same bounded run, then
classify `pass`, `reject`, or `invalid`. This role does not rerun AIDM or approve release.

## Prerequisites
- The current iteration has a validated proposal and complete AIDM evidence artifacts.
- The orchestrator has recorded the expected run ID, profile, and artifact locations.

## Allowed Inputs
- `research-proposal.json`: schema version `1` proposal-first contract.
- `promotion_manifest.json`: schema version `1`, fixed thresholds, baseline SPOT, candidate metrics,
  decision, failed gates, and optional recipe evidence.
- `performance_report.md` and `experiments.db`: artifacts produced by the public AIDM workflow.
- `experiment-evidence.json`: schema version `1` with run identity, artifact paths, candidate,
  metrics, proposal, and seed provenance.
- `state.json` and `journal.jsonl`: schema version `1` state graph records and prior checksums.

## Workflow
1. Recompute and compare every artifact checksum and path binding.
2. Validate all fixed checks and classify the evidence without changing thresholds.
3. Write one verification result and hand the terminal/iteration decision to the orchestrator.

## Output and Permissions
- The verifier writes only `verification.json` in the current iteration directory for valid,
  rejected, or malformed input evidence. Malformed/unsafe input evidence is serialized with
  `status: "invalid"` and `passed: false`; this is the verifier's fail-closed artifact.
- The orchestrator, not the verifier, may write `verification-failure.json` when the verifier
  return value/report is malformed or orchestrator evidence handling fails before a valid
  `verification.json` can be accepted. It contains only a safe reason code and is the
  orchestrator's fail-closed recovery artifact; both artifacts are distinct and neither contains
  raw evidence data.
- `verification.json` has schema version `1`, `status`, boolean `passed`, the complete fixed check
  map, safe `reasons`, and `provenance` with proposal/manifest/report/database checksum values.
  Each provenance key is always present; an unavailable artifact uses the deterministic value
  `unavailable`. The orchestrator accepts that value only for `status: "invalid"` and requires
  real SHA-256 values for valid outcomes.
- Read only the listed run artifacts and configured metadata. Do not write source, tests, fixtures,
  skills, scripts, datasets, generated code, customer systems, or promotion gates.

## Evidence, Checksums, and Safety
- Recompute lowercase SHA-256 for every input and compare it with experiment evidence and state
  bindings. Verify artifact paths stay inside the configured run directory.
- Require the complete verifier check set: identity, paths/schema, proposal and checksums,
  manifest/thresholds/seed/baseline, bounded proposal, selected candidate/spec/recipe,
  SQLite/report metrics, promotion provenance, and `promoted`.
- Never copy raw database rows, customer row values, secrets, credentials, or environment values
  into the report; use aggregate metrics and reason codes only.

## Bounded Iteration and Stop Conditions
- Verify exactly one proposal/experiment per iteration; never retry with relaxed rules.
- `pass` ends the loop at `ready_for_human_review`; `reject` may consume the next configured
  profile only while `max_iterations` (1..10) and the profile budget remain; `invalid` ends in
  `failed`. A terminal state cannot be resumed.

## Prohibitions
Do not edit source or customer data, merge, deploy, call AIDD, generate executable code, weaken
thresholds, accept missing checks, expose secrets, or log customer rows. Verification evidence is
not release/deployment approval.

## Error Table
| Error | Action |
| --- | --- |
| Missing/malformed input evidence | The verifier writes `verification.json` with `status: "invalid"` and fails closed. |
| Malformed verifier result/report or orchestrator evidence handling | The orchestrator writes `verification-failure.json` and fails closed. |
| Checksum/path/provenance mismatch | Mark `invalid`; do not promote. |
| Manifest rejection or gate failure | Mark `reject`; retain the rejection reason. |

## Evidence Output Layout
The iteration contains the AIDM artifacts plus `verification.json`, including invalid/fail-closed
results. The orchestrator records each artifact path and checksum in `state.json` and
`journal.jsonl`.

## Post-Run Reflection
Report the classification, fixed checks that passed/failed, and artifact checksums. State whether
the result is exhausted, failed, or ready only for human review; never call it release-ready.
