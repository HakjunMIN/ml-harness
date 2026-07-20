# Agentic Modeling and Development Design

## Purpose

Extend the forecasting harness from fixed feature-combination search to a safe,
agent-driven research loop. An agent proposes bounded feature and model
experiments; deterministic code validates, evaluates, and promotes only evidence
that passes existing safety gates.

## Safety Boundary

Agents produce declarative JSON only. They cannot submit Python, shell commands,
estimator objects, arbitrary imports, unrestricted hyperparameters, customer
code edits, merges, or deployments. Every proposal is validated before execution.

## Proposal Contract

`research-proposal.json` schema version `"1"` contains:

- `proposal_id`, `rationale`, and `baseline`.
- `feature_sets`: named lists of existing `FeatureSpec` dictionaries.
- `model_recipes`: named allowlisted model recipes with bounded parameters.
- `budget`: maximum candidate evaluations and top feature groups.

Feature specs use the existing transformation allowlist and prediction-time data
contract. Model recipes permit:

- `ridge`: `alpha` from an explicit bounded numeric allowlist.
- `hist_gradient_boosting`: `max_iter`, `learning_rate`, and `max_leaf_nodes`
  from explicit bounded numeric allowlists.

The engine rejects unknown keys, duplicate names, target/actual-data leakage,
unknown recipes, out-of-range parameters, and evaluation budgets above 50.

## AIDM Evaluation

The default baseline remains SPOT with no engineered features. If a legacy
prediction CSV is supplied, the harness validates it against timestamp/plant
keys and compares the selected candidate against both SPOT and legacy NMAE.

For every validated proposal candidate, the engine evaluates:

```text
model recipe × feature set
```

using existing chronological folds. It ranks deterministically by NMAE, recipe
name, and feature-set name. Promotion requires the existing SPOT gates and, when
provided, no regression versus the legacy baseline. Every proposal, rejection,
recipe, feature set, and result is stored in experiment artifacts and the
promotion manifest.

## AIDD Patch Request

For a promoted manifest, AIDD continues to generate the feature module and also
creates `model-recipe-patch.json`. The patch request includes only:

- selected recipe name and validated parameters,
- selected feature-spec checksum,
- manifest checksum,
- evidence paths and performance metrics,
- explicit status `requires_human_review`.

It is not executable code and cannot update customer source. A human uses it to
implement or review an integration patch.

## Agent Workflow

The `aidm-experiment` skill first profiles data and prior evidence, then creates
a proposal using only the contract. It invokes the proposal-aware AIDM command,
reports rejected candidates, and never relaxes gates. The `aidd-promotion` skill
requires a promoted manifest and emits the patch request after compilation.

## Success Criteria

- A valid proposal can add allowed feature sets and model recipes without source
  edits to the evaluator.
- Invalid or unsafe proposals fail before any experiment runs.
- Results identify the selected recipe and compare against SPOT and optional
  legacy baseline.
- AIDD emits a non-executable, human-reviewable model patch request.
- Existing fixed-catalog AIDM remains compatible when no proposal is supplied.
