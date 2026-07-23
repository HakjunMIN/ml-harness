---
name: aidm-experiment
description: Use when an agent has baseline legacy evidence and needs to run bounded AIDM feature discovery, proposal-first model search, compare promotion gates, or explain selected and rejected candidates.
---

# AIDM Experiment

## Overview
Run AIDM only after baseline evidence exists. Agentic experiments are proposal-first: agents may submit bounded JSON feature/model hypotheses, but must never submit estimator code, arbitrary search logic, callbacks, or gate changes to force promotion.

The reusable research orchestrator is optional. The manual workflow below remains valid and is
the default when a human wants to choose each proposal and command.

## Prerequisites
- `legacy-evidence.json` exists with `status: success`, or an explicit legacy prediction CSV is available for optional second-baseline comparison.
- Dataset path is explicit and passes the existing project data contract.
- Optional model-search setup has been installed before using `xgboost`, `lightgbm`, or Optuna search: `uv sync --extra model-search`.
- Agentic proposal JSON, when used, follows schema version `1` and contains only prediction-time feature specs plus supported bounded recipes/search.
- Run directory is outside source paths, under root `runs/` or `outputs/`; `.agents/` is reserved for framework assets.
- Human has approved use of any non-fixture dataset.

## Workflow
1. Start with a fixture command for feature-only recipes: `.agents/scripts/run-aidm.sh --dataset .agents/fixtures/valid-dataset.csv --proposal .agents/fixtures/research-proposal.json --legacy-predictions .agents/fixtures/legacy-predictions.csv --run-dir runs/fixture-aidm --folds 1`.
2. For reusable model-search smoke testing, install `uv sync --extra model-search`, then run `.agents/scripts/run-aidm.sh --dataset .agents/fixtures/valid-dataset.csv --proposal .agents/fixtures/model-search-proposal.json --run-dir runs/fixture-model-search --folds 1 --minimum-improvement 0 --max-plant-regression 1`. The tiny fixture uses one fold; real datasets should use five chronological folds unless a human approves otherwise.
3. For real runs, pass `--dataset` explicitly; never rely on implicit output dataset resolution.
4. Generate or edit proposal JSON only. Do not generate estimator code, customer patches, callbacks, arbitrary Optuna spaces, or gate changes.
5. Keep `--minimum-improvement`, `--max-plant-regression`, folds, proposal budget, recipes, search seed/trials, and optional-extra setup documented in run notes.
6. Inspect `experiments.db`, `promotion_manifest.json`, and `performance_report.md`.
7. Report selected winner, rejected candidates, failed gates, legacy-regression checks, trial evidence, selected re-evaluation evidence, and rejected ideas.
8. Stop if the manifest decision is `reject`; do not edit generated code or bypass AIDD.

## Optional Stage 1 Orchestration
- To opt in to the bounded Stage 1 loop, run `.agents/scripts/run-research-loop.sh --config .agents/fixtures/research-loop.json`.
- The loop may diagnose, create one bounded JSON proposal per profile, run AIDM, and verify evidence.
- It stops at `ready_for_human_review`, `exhausted`, or `failed`; it never invokes AIDD, edits source,
  deploys, merges, or changes gates. Resume only with `--resume` and the unchanged configuration.
- Running the manual steps above remains a supported, equivalent entry point; no orchestration is
  required for AIDM.

## Proposal-First Model Search Contract
- Schema version is exactly `"1"`; top-level keys are `schema_version`, `proposal_id`, `rationale`, `baseline`, `feature_sets`, `model_recipes`, `budget`, and optional `search`.
- Baseline is always `{"model":"SPOT"}`.
- Feature specs must use known deterministic transforms and prediction-time inputs only. `generation_mw`, all `actual_*` columns, customer-only fields, and target-derived aggregates are forbidden.
- Supported direct recipes and parameter sets:
  - `ridge`: `alpha` in `0.1`, `1.0`, `10.0`.
  - `hist_gradient_boosting`: `max_iter` in `50`, `100`, `200`; `learning_rate` in `0.03`, `0.1`; `max_leaf_nodes` in `15`, `31`, `63`.
  - `random_forest`: `n_estimators` in `100`, `200`, `400`; `max_depth` in `8`, `12`, `null`; `min_samples_leaf` in `1`, `2`, `4`.
  - `xgboost`: `n_estimators` in `100`, `200`, `400`; `max_depth` in `4`, `6`, `8`; `learning_rate` in `0.03`, `0.1`; `subsample` in `0.8`, `1.0`.
  - `lightgbm`: `n_estimators` in `100`, `300`; `learning_rate` in `0.03`, `0.1`; `num_leaves` in `15`, `31`; `min_child_samples` in `10`, `20`.
- Supported search is only bounded LightGBM Optuna TPE: `search.sampler` is `tpe`, `search.seed` is nonnegative, `search.n_trials` is `1..50`, and every `search.spaces.lightgbm` list contains unique values from the LightGBM allowed set.
- Budget semantics are exact and preflighted before baseline/store creation: `len(feature_sets) * (len(model_recipes) + search.n_trials + 1 selected_lightgbm re-evaluation) <= budget.max_evaluations`. Without search, use `len(feature_sets) * len(model_recipes)`. `budget.max_evaluations` is `1..50`; `budget.top_feature_groups` is `1..10`.
- Trial evidence is persisted per `aidm-optuna-lightgbm-<feature_set>-trial-<n>` run with parameters, metrics, proposal, search metadata, and reused-trial provenance when parameters duplicate a previous trial.
- The selected LightGBM trial is re-evaluated as `aidm-selected-lightgbm-<feature_set>` and records `selected_from_trial`; this re-evaluation consumes budget.

## Chronology, History, and Leakage Rules
- Use five chronological folds for real evidence. Each validation block must occur after its training rows; do not shuffle or stratify across time.
- History transforms (`lag`, `rolling_mean`) are strict prior per plant and never read the current row, future rows, `generation_mw`, or `actual_*` inputs.
- Missing history warmup values are expected at the start of a plant/fold and are imputed inside the estimator pipeline using training-fold statistics only.
- Fixture `.agents/fixtures/model-search-proposal.json` includes history features that are valid on `.agents/fixtures/valid-dataset.csv`; use `--folds 1` for that tiny dataset.
- AIDD can validate history feature specs but cannot render executable history modules. Any AIDD history-feature promotion is a human-review patch-only request, not an executable module handoff.

## Error Table
| Error | Action |
| --- | --- |
| Missing baseline evidence | Stop; run legacy baseline first. |
| Dataset missing or invalid | Stop; fix data mapping outside AIDM. |
| Optional model-search dependency missing | Install `uv sync --extra model-search` or remove `xgboost`, `lightgbm`, and search from the proposal. |
| XGBoost/LightGBM native runtime failure | Report the surfaced native error details; do not hide them as generic missing dependency failures. |
| Proposal validation fails | Report rejected idea and produce corrected JSON only. |
| Proposal exceeds `max_evaluations` | Reduce feature sets, direct recipes, or search trials; remember selected re-evaluation counts. |
| Legacy prediction coverage mismatch | Regenerate predictions for exactly the evaluation rows. |
| AIDM rejects promotion | Report failed gates and iterate hypotheses. |
| Threshold pressure | Keep gates intact and document the rejection. |
| AIDD rejects history executable generation | Treat as human review patch-only; do not create an executable history module. |

## Evidence Output Layout
- `experiments.db`: run records, metrics, direct recipe runs, search trial runs, selected re-evaluation runs, reused-trial provenance, and failures.
- `promotion_manifest.json`: decision, selected specs, selected model recipe, optional legacy baseline, proposal, thresholds, failed gates, and checksums through later steps.
- `performance_report.md`: baseline, candidate, trial, gate, and artifact summary.

## Post-Run Reflection
Note whether AIDM used the intended dataset and proposal, whether optional extras were installed, whether gates remained unchanged, which trials and ideas were rejected, whether history warmups behaved as strict-prior fold-local imputations, and whether the result is ready for AIDD or needs another JSON-only experiment.
