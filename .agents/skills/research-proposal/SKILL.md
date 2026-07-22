---
name: research-proposal
description: Use when Stage 1 research-loop needs one bounded, declarative AIDM proposal from a validated diagnosis.
---

# Research Proposal

## Exact Responsibility
Translate one recommended profile into one proposal-first `ResearchProposal`. Use only the fixed
repository feature transforms, model recipes, and bounded search spaces. This role chooses
hypotheses; it never writes estimator code or executes an unbounded search.

## Prerequisites
- A valid, checksum-bound `diagnosis.json` recommends the requested profile.
- The immutable effective configuration and iteration budget are available.

## Allowed Inputs
- `research-config.json`, `state.json`, and the current `diagnosis.json`, each schema version `1`.
- The diagnosis must bind the configured dataset and baseline by lowercase SHA-256 and mark the
  selected profile as recommended.
- A proposal must use schema version `1`, baseline `{"model":"SPOT"}`, prediction-time feature
  specs, supported recipes, and optional bounded LightGBM TPE search only.
- Inputs may name `timestamp`, metadata, forecast/LDAPS columns, and strict prior history
  transforms. `generation_mw`, every `actual_*`, customer-only columns, and target-derived
  values are forbidden.

## Workflow
1. Select the next unused recommended profile.
2. Render and validate one schema-version `1` proposal plus bounded notes.
3. Persist both artifacts and their canonical checksums before AIDM execution.

## Output and Permissions
- Write only the current iteration's `research-proposal.json` and `research-notes.json` under
  `iterations/<iteration>-<profile>/`.
- `research-proposal.json` contains `schema_version`, `proposal_id`, `rationale`, `baseline`,
  `feature_sets`, `model_recipes`, `budget`, and optional `search`; notes contain schema version,
  profile, iteration, candidate cap, and rejected idea reason codes.
- Read the effective config, diagnosis, and state. Do not write source, tests, fixtures, skills,
  scripts, datasets, generated modules, manifests, customer systems, or gate configuration.

## Evidence, Checksums, and Safety
- Bind proposal provenance to run ID, iteration, diagnosis checksum, baseline checksum, and the
  canonical proposal SHA-256 recorded by the orchestrator.
- `budget.max_evaluations` is 1..50; `top_feature_groups` is 1..10. Search trials are 1..50 and
  only allowed discrete values may be used. Keep the candidate cap fixed at the orchestrator
  limit (three candidates).
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
