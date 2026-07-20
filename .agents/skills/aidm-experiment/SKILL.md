---
name: aidm-experiment
description: Use when an agent has baseline legacy evidence and needs to run bounded AIDM feature discovery, compare promotion gates, or explain selected and rejected candidates.
---

# AIDM Experiment

## Overview
Run AIDM only after baseline evidence exists. Agentic experiments are proposal-first: agents may propose bounded JSON feature/model hypotheses, but must never submit code or weaken gates to force promotion.

## Prerequisites
- `legacy-evidence.json` exists with `status: success`, or an explicit legacy prediction CSV is available for optional second-baseline comparison.
- Dataset path is explicit and passes the existing project data contract.
- Agentic proposal JSON, when used, follows schema version `1` and contains only safe prediction-time feature specs plus supported `ridge` or `hist_gradient_boosting` recipes.
- Run directory is outside source paths, preferably under `.agents/runs/` or `.agents/output/`.
- Human has approved use of any non-fixture dataset.

## Workflow
1. Start with fixture command: `.agents/scripts/run-aidm.sh --dataset .agents/fixtures/valid-dataset.csv --proposal .agents/fixtures/research-proposal.json --legacy-predictions .agents/fixtures/legacy-predictions.csv --run-dir .agents/runs/fixture-aidm --folds 1`.
2. For real runs, pass `--dataset` explicitly; never rely on implicit output dataset resolution.
3. Generate or edit proposal JSON only. Do not generate estimator code, customer patches, or gate changes.
4. Keep `--minimum-improvement`, `--max-plant-regression`, folds, proposal budget, recipes, and seed documented in the run notes.
5. Inspect `experiments.db`, `promotion_manifest.json`, and `performance_report.md`.
6. Report selected winner, rejected candidates, failed gates, legacy-regression checks, and rejected ideas.
7. Stop if the manifest decision is `reject`; do not edit generated code or bypass AIDD.

## Error Table
| Error | Action |
| --- | --- |
| Missing baseline evidence | Stop; run legacy baseline first. |
| Dataset missing or invalid | Stop; fix data mapping outside AIDM. |
| Proposal validation fails | Report rejected idea and produce corrected JSON only. |
| Legacy prediction coverage mismatch | Regenerate predictions for exactly the evaluation rows. |
| AIDM rejects promotion | Report failed gates and iterate hypotheses. |
| Threshold pressure | Keep gates intact and document the rejection. |

## Evidence Output Layout
- `experiments.db`: run records and metrics.
- `promotion_manifest.json`: decision, selected specs, selected model recipe, optional legacy baseline, thresholds, failed gates, and checksums through later steps.
- `performance_report.md`: baseline, candidate, and artifact summary.

## Post-Run Reflection
Note whether AIDM used the intended dataset and proposal, whether gates remained unchanged, which ideas were rejected, and whether the result is ready for AIDD or needs another JSON-only experiment.
