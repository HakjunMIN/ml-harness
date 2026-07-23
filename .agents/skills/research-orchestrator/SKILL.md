---
name: research-orchestrator
description: Use when an agent opts into the bounded Stage 1 research-loop state machine and needs safe resume or terminal handling.
---

# Research Orchestrator

## Exact Responsibility
Coordinate diagnostic, proposal, AIDM experiment, and verification roles through the repository's
strict state graph. The only invocation is `.agents/scripts/run-research-loop.sh --config PATH`
with optional `--resume`; the runner invokes exactly `uv run python -m power_forecasting.cli
research-loop`. Orchestration is opt-in and stops before AIDD/code/deploy.

When `agent_proposals: true` is set in the immutable config, the runner stops at
`awaiting_proposal` after diagnostic evidence. It writes a catalog and aggregate-only proposal
context under the assigned iteration directory. Invoke `research-proposal` to create the requested
JSON, then rerun the exact same command with `--resume`.

## Prerequisites
- The runner, project package, and schema-version `1` configuration are present.
- Input artifacts already exist; the orchestrator never creates customer or source inputs.

## Allowed Inputs
- A project-local `research-loop.json`, schema version `1`, with exactly the configured run ID,
  existing dataset and baseline manifest paths, profile allowlist, iteration/fold bounds, objective,
  and AIDM thresholds.
- Existing role artifacts only under the configured root `runs/` or `outputs/` run
  directory: schema-versioned diagnosis/proposal/experiment/verification artifacts.
- `--resume` may read only the matching `state.json`, `journal.jsonl`, immutable
  `research-config.json`, and their bound artifacts.

## Output and Permissions
- Write only under root `runs/` (or the explicitly allowed `outputs/`) and only these
  names: `research-config.json`, `state.json`, `journal.jsonl`, `diagnosis.json`,
  `diagnostic-failure.json`, `research-summary.json`, `verification-failure.json`,
  `exhaustion.json`, plus per-iteration
  `research-proposal.json`, `research-notes.json`, `promotion_manifest.json`, `experiments.db`,
  `performance_report.md`, `experiment-evidence.json`, `experiment-failure.json`, and
  `verification.json`.
- `state.json` schema version `1` records status, iteration, used/remaining profiles, artifact
  paths, transition history, and config SHA-256. Journal events have timestamp, status transition,
  iteration/profile, artifact path, and SHA-256. Summary has run ID, terminal status, iterations,
  profiles, verifier outcome/reasons, and artifact checksums.
- Read project inputs and role artifacts; never write source, tests, fixtures, skills, scripts,
  generated code, customer systems, or any path outside the allowed run directory.

## Evidence, Checksums, and Safety
- Every recorded artifact is atomically written, hashed with lowercase SHA-256, and rebound on
  resume. The effective config includes input checksums; changed config, stale artifacts, malformed
  JSON, incomplete journal groups, and terminal resume all fail closed. Invalid verifier reports
  use the exact provenance keys with `unavailable` for any unavailable artifact checksum; only
  invalid reports may use that representation.
- State transitions are `initialized -> diagnosed -> proposed -> experimenting -> verifying`, or
  for agent proposals, `initialized -> diagnosed -> awaiting_proposal -> proposed ->
  experimenting -> verifying`,
  and then `iterate`, `ready_for_human_review`, `exhausted`, or `failed`.
  A diagnostic failure traverses the required internal states without allocating a profile,
  selecting a proposal, running an experiment, or emitting `experiment-failure.json`; only
  `verifying -> failed` is terminal failure evidence.
- Logs and summary contain aggregate metadata and safe reason codes only; never customer rows,
  target/actual samples, secrets, tokens, credentials, or environment values.

## Bounded Iteration and Stop Conditions
- Profile order is explicit. Default runs use each configured profile at most once; with
  `agent_proposals: true`, rejected iterations deterministically cycle those profiles until
  `max_iterations` is reached. `max_iterations` and `fold_count` are each 1..10; candidate/search
  budgets are validated before AIDM execution.
- `awaiting_proposal` is nonterminal but deliberately does not run AIDM. `ready_for_human_review`,
  `exhausted`, and `failed` are terminal. `--resume` recovers only a
  nonterminal, checksum-consistent state and never reruns an interrupted experiment as success.
- A successful Stage 1 run produces review evidence only. It does not invoke AIDD or create a
  deployable module.

## Workflow
1. Validate the config and initialize an immutable config snapshot and journal.
2. Advance through diagnosis; for `agent_proposals: true`, stop and hand the catalog/context to
   `research-proposal`.
3. Resume only after a catalog-contained proposal is present, then run one bounded experiment and
   verification.
4. Persist every transition and checksum, then stop at a terminal state and write the summary.

## Prohibitions
Do not edit source or customer data, merge, deploy, call AIDD, generate/modify executable code,
weaken thresholds or gates, use unbounded iterations/search, export secrets, or log customer rows.
`research-summary.json` is never release or deployment approval evidence.

## Error Table
| Error | Action |
| --- | --- |
| Missing/unknown option or config | Exit 2; do not create a run. |
| Existing state without `--resume` | Exit 2; preserve the prior run. |
| Changed config, checksum, or journal | Fail closed; do not continue. |
| Terminal state or failed role | Persist terminal evidence; require human decision or a new run. |

## Evidence Output Layout
The canonical terminal artifact is `runs/<run-id>/research-summary.json` (or the configured
`outputs/<run-id>/research-summary.json`), accompanied by
`state.json`, `journal.jsonl`, and all checksummed role/iteration artifacts. Summary status is
`ready_for_human_review`, `exhausted`, or `failed`.

## Post-Run Reflection
Report terminal status, iterations, profiles, safe verifier reasons, and checksums. Hand the
artifacts to a human for the next manual AIDD/review step; never claim automated promotion or
deployment.
