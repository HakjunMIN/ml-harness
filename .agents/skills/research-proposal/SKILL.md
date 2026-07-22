---
name: research-proposal
description: Use when Stage 1 research-loop needs one bounded, declarative AIDM proposal from a validated diagnosis.
---

# Research Proposal

## Exact Responsibility
Translate one recommended profile into one proposal-first `ResearchProposal`. With
`agent_proposals: true`, read the runner-generated catalog and aggregate-only context, then choose
the next bounded hypothesis from that catalog. This role never writes estimator code or executes an
unbounded search.

## Prerequisites
- A valid, checksum-bound `diagnosis.json` recommends the requested profile.
- The immutable effective configuration and iteration budget are available.

## Allowed Inputs
- `research-config.json`, `state.json`, and the current `diagnosis.json`, each schema version `1`.
- For agent handoff, the assigned `proposal-context.json` and `proposal-catalog.json`.
- The diagnosis must bind the configured dataset and baseline by lowercase SHA-256 and mark the
  selected profile as recommended.
- A proposal must use schema version `1`, baseline `{"model":"SPOT"}`, prediction-time feature
  specs, supported recipes, and optional bounded LightGBM TPE search only.
- Inputs may name `timestamp`, metadata, forecast/LDAPS columns, and strict prior history
  transforms. `generation_mw`, every `actual_*`, customer-only columns, and target-derived
  values are forbidden.

## Workflow
1. Confirm the loop is `awaiting_proposal` and read only the assigned context/catalog artifacts.
2. Select feature sets, recipes, and optional TPE values from the catalog; use prior aggregate
   results and remaining evaluation budget to justify the choice.
3. Write `research-proposal.json` into the assigned iteration directory. Do not edit the catalog,
   context, or runner-managed notes.
4. Invoke the same research runner with `--resume`; the runner validates the proposal before AIDM.

## Output and Permissions
- Write only the current iteration's `research-proposal.json` under
  `iterations/<iteration>-<profile>/`; the runner writes `research-notes.json`.
- `research-proposal.json` contains `schema_version`, `proposal_id`, `rationale`, `baseline`,
  `feature_sets`, `model_recipes`, `budget`, and optional `search`.
- Read the effective config, diagnosis, and state. Do not write source, tests, fixtures, skills,
  scripts, datasets, generated modules, manifests, customer systems, or gate configuration.

## Evidence, Checksums, and Safety
- Bind proposal provenance to run ID, iteration, diagnosis checksum, baseline checksum, and the
  canonical proposal SHA-256 recorded by the orchestrator.
- `budget.max_evaluations` is 1..50; `top_feature_groups` is 1..10. Search trials are 1..50 and
  only cataloged discrete values may be used. The proposal budget must cover the candidate count
  and fit the context's remaining run-wide evaluation budget.
- Do not include raw rows, target values, secrets, environment values, or arbitrary code in JSON.

## Bounded Iteration and Stop Conditions
- Produce at most one proposal per allocated profile, in configured order, and never reuse a
  profile. Stop when the profile list or `max_iterations` (1..10) is exhausted.
- Reject the proposal and stop the iteration if schema, leakage, checksum, or budget validation
  fails. A rejected candidate is evidence for iteration, not permission to loosen a gate.

## Prohibitions
Do not edit source or customer data, generate code/callbacks, merge, deploy, call AIDD, weaken
thresholds, use unbounded search, expose secrets, or log customer rows. Proposal JSON is not
release approval.

## Error Table
| Error | Action |
| --- | --- |
| Diagnosis is stale or not recommended | Stop; require a fresh diagnostic. |
| Unsupported transform/recipe/search | Reject the proposal and report a safe reason code. |
| Budget exceeded | Stop before AIDM; reduce hypotheses without changing gates. |

## Evidence Output Layout
Each iteration contains `research-proposal.json` and `research-notes.json`; state and journal bind
both paths and checksums. A proposal alone never authorizes AIDD, merge, or deployment.

## Post-Run Reflection
List the selected profile, bounded candidate count, rejected ideas, and checksums. State whether
the next safe step is AIDM experimentation or exhaustion/human review.
