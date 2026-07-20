---
name: aidm-experiment
description: Use when an agent has baseline legacy evidence and needs to run bounded AIDM feature discovery, compare promotion gates, or explain selected and rejected candidates.
---

# AIDM Experiment

## Overview
Run AIDM only after baseline evidence exists. Use explicit datasets and thresholds; never weaken gates to force promotion.

## Prerequisites
- `legacy-evidence.json` exists with `status: success`.
- Dataset path is explicit and passes the existing project data contract.
- Run directory is outside source paths, preferably under `.agents/runs/` or `.agents/output/`.
- Human has approved use of any non-fixture dataset.

## Workflow
1. Start with fixture command: `.agents/scripts/run-aidm.sh --dataset .agents/fixtures/valid-dataset.csv --run-dir .agents/runs/fixture-aidm --folds 1 --top-single-candidates 1`.
2. For real runs, pass `--dataset` explicitly; never rely on implicit output dataset resolution.
3. Keep `--minimum-improvement`, `--max-plant-regression`, folds, candidates, and seed documented in the run notes.
4. Inspect `experiments.db`, `promotion_manifest.json`, and `performance_report.md`.
5. Report selected winner, rejected candidates, failed gates, and leakage checks.
6. Stop if the manifest decision is `reject`; do not edit generated code or bypass AIDD.

## Error Table
| Error | Action |
| --- | --- |
| Missing baseline evidence | Stop; run legacy baseline first. |
| Dataset missing or invalid | Stop; fix data mapping outside AIDM. |
| AIDM rejects promotion | Report failed gates and iterate hypotheses. |
| Threshold pressure | Keep gates intact and document the rejection. |

## Evidence Output Layout
- `experiments.db`: run records and metrics.
- `promotion_manifest.json`: decision, selected specs, thresholds, failed gates, and checksums through later steps.
- `performance_report.md`: baseline, candidate, and artifact summary.

## Post-Run Reflection
Note whether AIDM used the intended dataset, whether gates remained unchanged, and whether the result is ready for AIDD or needs another experiment.
