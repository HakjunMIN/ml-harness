# Research-Loop Agent Runbook

Use this runbook only for the optional Stage 1 research loop. The manual
`legacy-intake -> AIDM -> AIDD -> human review` path remains unchanged.

## Starting an Agent-Proposal Run

Treat a request to run the research loop as one-shot orchestration. Continue autonomously until a
terminal Stage 1 state. Do not ask the user to invoke each role or resume command.

1. Ask the user for the maximum number of `proposal -> AIDM -> verification`
   cycles. Accept an integer from 1 through 10; when the user does not specify
   one, set `"max_iterations": 10`.
2. Create or update a repository-local research-loop config under `runs/` with
   `"agent_proposals": true`, the selected `"max_iterations"`, and a
   repository-relative `"catalog_path"` to `configs/optimization-catalog.v1.json`.
   Before starting, display the approved plan: profiles, feature sets, direct recipes,
   TPE space, folds, gates, and budget. Do not change the baseline, data paths, gates,
   profiles, catalog, or evaluation budget without explicit user approval. Configure artifact
   output only under root `runs/` or `outputs/`; `.agents/` is reserved for framework assets.
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
context, catalog, checksums, thresholds, or evidence to force another cycle. Do not change the
catalog once the run begins: its catalog SHA-256 is bound to the research config, state, journal,
and handoffs, and a changed catalog must fail closed on resume.
Do not wait for another user request between cycles. After every runner return,
show `iteration <current>/<maximum> · profile <name> · evaluations <used>/50 ·
last result <status or safe reason>` using aggregate-only values.

## Stop Conditions

Stop immediately when the runner returns `ready_for_human_review`, `exhausted`,
or `failed`. `ready_for_human_review` is evidence for a human review, not
approval to invoke AIDD, generate code, merge, deploy, or modify a customer
system. Never include raw customer rows, targets, actuals, secrets, or
arbitrary code in proposal artifacts.

When the status is `ready_for_human_review`, invoke `human-review` automatically
to display the evidence and ask for the person's next action. Do not invoke
AIDD automatically. The manual skill-by-skill path remains supported when a
person explicitly asks to control each stage.
