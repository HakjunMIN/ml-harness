# Research-Loop Agent Runbook

Use this runbook only for the optional Stage 1 research loop. The manual
`legacy-intake -> AIDM -> AIDD -> human review` path remains unchanged.

## Starting an Agent-Proposal Run

1. Ask the user for the maximum number of `proposal -> AIDM -> verification`
   cycles. Accept an integer from 1 through 10; when the user does not specify
   one, set `"max_iterations": 10`.
2. Create or update a repository-local research-loop config under `runs/` with
   `"agent_proposals": true` and the selected `"max_iterations"`. Do not change
   the baseline, data paths, gates, profiles, or evaluation budget without
   explicit user approval. Configure artifact output only under root `runs/` or
   `outputs/`; `.agents/` is reserved for framework assets.
3. Run `.agents/scripts/run-research-loop.sh --config <config>`.

## Proposal and Resume Cycle

When the runner returns `awaiting_proposal`:

1. Read only the current iteration's `proposal-context.json`,
   `proposal-catalog.json`, aggregate verification evidence, and the immutable
   config.
2. Invoke the `research-proposal` skill and write exactly one
   `research-proposal.json` in the assigned iteration directory.
3. Use only catalog-contained features, model recipes, and optional TPE search
   space. The proposal must fit both its own budget and the context's remaining
   run-wide 50-evaluation budget.
4. Run the same command with `--resume`.

If verification rejects the experiment and the runner returns a new
`awaiting_proposal`, repeat the cycle. The runner deterministically cycles
configured profiles up to `max_iterations`; do not edit state, journal,
context, catalog, checksums, thresholds, or evidence to force another cycle.

## Stop Conditions

Stop immediately when the runner returns `ready_for_human_review`, `exhausted`,
or `failed`. `ready_for_human_review` is evidence for a human review, not
approval to invoke AIDD, generate code, merge, deploy, or modify a customer
system. Never include raw customer rows, targets, actuals, secrets, or
arbitrary code in proposal artifacts.
