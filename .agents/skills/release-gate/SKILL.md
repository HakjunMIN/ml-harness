---
name: release-gate
description: Use when an agent is asked whether legacy baseline, AIDM, AIDD, compile, and human approval evidence is sufficient to release or request a patch.
---

# Release Gate

## Overview
This gate must fail closed. Absent baseline, AIDM, AIDD, compile, or human approval means no release.
The Stage 1 research-loop `research-summary.json` is diagnostic/proposal evidence only and is
explicitly not release or deployment approval evidence.

## Prerequisites
- Baseline `legacy-evidence.json` with `status: success`.
- AIDM `promotion_manifest.json`, `experiments.db`, and `performance_report.md`.
- AIDD `promotion-evidence.json` and compiled `generated/promoted_features.py`.
- Explicit human approval for the exact patch or release action.

## Workflow
1. Verify baseline evidence exists and references only checksums, not customer row contents or secret environment values.
2. Verify AIDM completed with the intended explicit dataset and the manifest decision is `promote`.
3. Verify AIDD evidence exists and the generated module checksum matches the compiled file.
4. Verify compile status by rerunning or checking the recorded compile artifact from `verify-promotion.sh`.
5. Verify human approval is explicit, current, and scoped to this run and patch.
6. If any item is missing, say `fail closed` and list the missing evidence.
7. Agents must not deploy, must not merge, and must not edit customer systems. Agents may only prepare a human-reviewed patch request.
8. Reject `research-summary.json`, `state.json`, `journal.jsonl`, or any research-loop terminal
   status as a substitute for AIDM/AIDD/compile evidence or explicit human approval.

## Error Table
| Error | Action |
| --- | --- |
| Missing baseline | Fail closed; run legacy baseline. |
| Missing or rejected AIDM | Fail closed; do not call AIDD. |
| Missing AIDD evidence | Fail closed; run verification. |
| Compile missing or failed | Fail closed; no success evidence is valid. |
| No human approval | Fail closed; do not release. |
| Research-loop summary presented as approval | Reject it; continue with the required release evidence and human approval. |
| Request to deploy, merge, or edit customer systems | Refuse and restate the human-gated patch process. |

## Evidence Output Layout
- Gate summary: baseline path, AIDM artifacts, AIDD artifacts, compile result, human approval reference, and final decision.
- Rejection summary: missing item, reason, and next safe command.

## Post-Run Reflection
Document what evidence was accepted, what failed closed, and whether the next action is reject, iterate, or ask a human to review a patch.
