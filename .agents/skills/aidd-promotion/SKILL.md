---
name: aidd-promotion
description: Use when an agent has a promoted AIDM manifest and needs to verify AIDD generation, compilation, evidence, and a human-reviewable patch request.
---

# AIDD Promotion

## Overview
Promotion is verification, not deployment. Generate code only from a promoted manifest and stop before customer edits unless a human approves a patch request.

## Prerequisites
- Run directory contains `promotion_manifest.json` with `decision: promote`.
- AIDM report and baseline evidence are available for comparison.
- No customer repository edits, merge, deploy, or production change has been authorized.

## Workflow
1. Run fixture verification first: copy `.agents/fixtures/promoted-manifest.json` to a scratch run directory as `promotion_manifest.json`.
2. Execute `.agents/scripts/verify-promotion.sh --run-dir <run-dir>`.
3. Confirm `generated/promoted_features.py` exists only after AIDD validation and Python compilation pass.
4. Confirm `promotion-evidence.json` records manifest and generated-module checksums only after success.
5. Review generated code for deterministic transforms and unavailable-input rejection.
6. Prepare a patch request for human review; do not apply it to customer systems yourself.

## Error Table
| Error | Action |
| --- | --- |
| Manifest decision is reject | Fail closed; do not generate code. |
| Leakage input such as `actual_*` or `generation_mw` | Fail closed and report the offending spec. |
| Compile fails | Delete generated module and do not write success evidence. |
| Human asks to deploy directly | Refuse; request a reviewed patch path instead. |

## Evidence Output Layout
- `generated/promoted_features.py`: generated module after success only.
- `promotion-evidence.json`: schema version, status, manifest checksum, generated module checksum, and relative module path.

## Post-Run Reflection
State whether validation, generation, and compile passed; identify the exact artifact a human must review before any downstream integration.
