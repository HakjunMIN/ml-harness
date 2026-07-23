---
name: human-review
description: Use when a research or AIDD run needs a human review decision and reviewers need a safe, in-agent evidence summary.
---

# Human Review

## Exact Responsibility
Render a concise, reviewable evidence summary in the coding agent and collect the reviewer's next action. This skill never changes research-loop state, writes an approval artifact, merges, deploys, or edits a customer system. Agents must not deploy or merge. A coding-agent response records review intent only; it is not release approval.

## Prerequisites
- A Stage 1 run is `ready_for_human_review`, or AIDD validation has completed for a `decision: promote` manifest.
- `performance_report.md`, `verification.json`, and `promotion_manifest.json` exist under the configured run directory.
- For a final patch decision, `promotion-evidence.json` and compiled `generated/promoted_features.py` also exist.

## Allowed Inputs
- `research-summary.json`, `state.json`, `verification.json`, `promotion_manifest.json`, `performance_report.md`, and, after AIDD, `promotion-evidence.json` plus `model-recipe-patch.json`.
- Read only aggregate metrics, decision/reason codes, selected feature names and recipe metadata, relative artifact paths, and SHA-256 checksums.
- Never display rows, target values, actual values, secrets, tokens, credentials, or environment values.

## Workflow
1. Verify that the run artifacts are bound by matching checksums and that `promotion_manifest.json` has `decision: promote`. If any evidence is missing or mismatched, fail closed.
2. Display this review summary in the coding agent using workspace-relative clickable links and the required review tables below. Keep values aggregate-only and use `unavailable` when a permitted value cannot be established.

   ```markdown
   ## Human Review

   **Research decision:** `<promote | reject>`
   **Verification:** `<pass | reject | invalid>`

   | Category | Baseline | Selected candidate |
   | --- | --- | --- |
   | Model / recipe | `<safe baseline metadata>` | `<safe recipe metadata>` |
   | MAE | `<aggregate value>` | `<aggregate value>` |
   | RMSE | `<aggregate value>` | `<aggregate value>` |
   | NMAE | `<aggregate value>` | `<aggregate value>` |
   | Improvement | `-` | `<aggregate ratio or unavailable>` |

   | Check | Status | Detail |
   | --- | --- | --- |
   | Manifest decision | `<pass | fail>` | `<promote | reject>` |
   | Verification | `<pass | reject | invalid>` | `<safe reason codes or none>` |
   | Thresholds | `<pass | fail>` | `<safe gate summary>` |
   | Per-plant regression | `<pass | fail>` | `<aggregate delta summary>` |
   | Evidence checksums | `<pass | fail>` | `<matched artifacts or unavailable>` |

   | Artifact | Purpose | Checksum |
   | --- | --- | --- |
   | [performance_report.md](...) | Aggregate comparison | `<sha256>` |
   | [verification.json](...) | Gate and provenance checks | `<sha256>` |
   | [promotion_manifest.json](...) | Proposed winner and selected specs | `<sha256>` |

   **Selected feature set / recipe:** `<safe metadata only>`
   **Manifest checksum:** `<sha256>`
   **Boundary:** No merge, deploy, customer-system edit, or automatic approval.
   ```

3. Invoke `vscode_askQuestions` with one `Review decision` question and these options:
   - `Request AIDD verification`: proceed only to the separate AIDD verification workflow.
   - `Reject or request changes`: retain the evidence and stop; do not loosen gates.
4. When AIDD evidence and compilation are present, display a second summary that includes `promotion-evidence.json`, `generated/promoted_features.py`, and any non-executable `model-recipe-patch.json`. Ask whether to request release-gate review or reject/request changes.
5. State that a chat answer, research summary, and AIDD success are not release approval. Release-gate still requires explicit human approval for the exact release action.

## Review Tables

- Always render the comparison, gate, and artifact tables in the Stage 1 review. Do not replace them with prose when a source value is available.
- For metric values, show only aggregate MAE, RMSE, NMAE, improvement ratio, and aggregate per-plant delta summary; never show input rows, targets, actuals, or predictions.
- When AIDD evidence exists, add an AIDD artifact row for each available `promotion-evidence.json`, `generated/promoted_features.py`, and `model-recipe-patch.json`. Include the patch status (`requires_human_review`) and safe recipe metadata in the table detail.
- Tables make evidence scannable; they do not grant approval or replace the mandatory `vscode_askQuestions` decision.

## Output and Permissions
- Write no run artifact and do not modify source, tests, datasets, manifests, state, or gates.
- Display only the summary and question result in the coding agent.
- A `Request AIDD verification` result may proceed only to the separate AIDD verification workflow; it must not merge or deploy.

## Evidence, Checksums, and Safety
- Show the manifest checksum with every review request so reviewers can correlate the decision with the evidence.
- Treat missing, stale, or mismatched checksums as `fail closed`.
- Keep model recipe patches non-executable and require their explicit review.

## Error Table
| Error | Action |
| --- | --- |
| Missing or mismatched evidence | Fail closed; show the missing artifact or safe reason code. |
| Manifest decision is not `promote` | Stop; do not request AIDD verification or PR review. |
| AIDD or compile evidence is absent | Allow only `Request AIDD verification`; do not request release-gate review. |
| No explicit human release approval | Fail closed; do not release, merge, deploy, or edit a customer system. |

## Evidence Output Layout
The coding agent displays the required comparison, gate, and artifact tables with links to `performance_report.md`, `verification.json`, and `promotion_manifest.json`. Final review adds rows for `promotion-evidence.json`, `generated/promoted_features.py`, and `model-recipe-patch.json` when present. The review decision is displayed in the agent; the manifest checksum links that decision to the evidence.

## Post-Run Reflection
Report the selected review action, the manifest checksum, and whether the next safe step is AIDD verification, reject/iterate, or release-gate review. Never call a run release-ready before explicit human release approval exists.
