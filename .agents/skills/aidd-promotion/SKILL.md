---
name: aidd-promotion
description: Use when an agent has a promoted AIDM manifest and needs to verify AIDD generation, compilation, evidence, and a human-reviewable patch request.
---

# AIDD Promotion

## Overview
Promotion is verification, not deployment. Generate code only from a promoted manifest. Agentic model recipes produce a non-executable `model-recipe-patch.json` request that requires human review before any customer edit.

## Prerequisites
- Run directory contains `promotion_manifest.json` with `decision: promote`.
- AIDM report and baseline evidence are available for comparison.
- Agentic manifests with `selected_model_recipe` have proposal evidence and selected feature specs in the manifest.
- No customer repository edits, merge, deploy, or production change has been authorized.

## Workflow
1. Run fixture verification first: copy `.agents/fixtures/promoted-manifest.json` to a scratch run directory as `promotion_manifest.json`.
2. Execute `.agents/scripts/verify-promotion.sh --run-dir <run-dir>`.
3. Confirm `generated/promoted_features.py` exists only after AIDD validation and Python compilation pass.
4. If `selected_model_recipe` exists, confirm `model-recipe-patch.json` exists, is JSON only, says `requires_human_review`, and contains no executable code or customer paths.
5. Confirm `promotion-evidence.json` records manifest, generated-module, and model-recipe-patch checksums only after success.
6. Review generated code for deterministic transforms and unavailable-input rejection.
7. Hand `model-recipe-patch.json` to a human reviewer; do not apply it to customer systems yourself.

## Error Table
| Error | Action |
| --- | --- |
| Manifest decision is reject | Fail closed; do not generate code. |
| Leakage input such as `actual_*` or `generation_mw` | Fail closed and report the offending spec. |
| Missing or invalid recipe evidence | Fail closed; do not create a model patch request. |
| Compile fails | Delete generated module and do not write success evidence. |
| Human asks to deploy directly | Refuse; request a reviewed patch path instead. |

## Evidence Output Layout
- `generated/promoted_features.py`: generated module after success only.
- `model-recipe-patch.json`: non-executable human-review request after success only when a selected recipe exists.
- `promotion-evidence.json`: schema version, status, manifest checksum, generated module checksum, optional model recipe patch checksum, and relative artifact paths.

## Post-Run Reflection
State whether validation, generation, compile, and patch rendering passed; identify the exact artifact a human must review before any downstream integration.
