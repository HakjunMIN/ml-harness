# Bounded Model Search Design

## Goal

Extend the forecasting harness in two stages: first, evaluate bounded Random
Forest and XGBoost recipes with safe temporal, weather, lag, rolling, and
interaction features; second, add bounded LightGBM and Optuna searches while
preserving deterministic evidence, promotion gates, and human review.

## Scope and sequencing

### Stage 1: Baseline recipes and history-aware features

- Add optional `xgboost` dependency and `random_forest` / `xgboost` proposal
  recipes with exact, finite allowlisted parameter values.
- Add `lag` and `rolling_mean` feature transforms that operate per plant and
  timestamp with a strict prior-row cutoff. They never use the current row's
  target or any future row.
- Evaluate every recipe and feature set through the existing chronological,
  expanding-window fold evaluator. Five folds remain a configuration value.
- Add R2 to aggregate, fold, and per-plant metrics, while keeping MAE, RMSE,
  and NMAE as promotion criteria.

### Stage 2: Production recipes and bounded optimization

- Add optional `lightgbm` and `optuna` dependencies.
- Add a bounded `lightgbm` recipe and allowlisted search-space definitions.
- Run Optuna with an explicit deterministic sampler seed and a maximum trial
  count. Each trial records parameters, fold metrics, status, and error in the
  existing experiment store.
- Re-evaluate the selected Optuna recipe through the same evaluator before
  promotion. Optuna does not change gate thresholds, choose deployment, or
  create executable customer code.

## Safety and data availability

Feature transforms may use only forecast-time columns and historical values
strictly earlier than the current timestamp. For a validation row, history is
built from earlier timestamps only; ties at the same timestamp are excluded.
Warm-up rows with insufficient prior history emit NaN history features, which
model pipelines impute from the training fold rather than filling from the
target. Any generated feature module remains stateless; history-dependent
features produce a human-review-only recipe patch and are not emitted as
executable AIDD feature code.

Random Forest, XGBoost, LightGBM, and Optuna are optional extras. A proposal
using a missing optional package fails with a clear validation error; core
installation and existing Ridge/HGB workflows keep working.

## Contracts

Proposal schema version advances only if necessary. Recipe validation remains
exact-key, allowlist-based, finite, and budgeted. Search configurations have
exact keys, an integer trial budget of 1 through 50, an explicit seed, and
only named parameter ranges supported by their recipe. The total candidate and
trial budget remains bounded.

Experiment metadata records the resolved package-backed recipe, parameters,
seed, fold count, feature specifications, and search trial identity. Promotion
manifests identify the winning recipe and, when applicable, trial provenance.
The existing SPOT and optional legacy prediction baselines both remain required
gates.

## Non-goals

- No LSTM, GRU, Transformer, stacking, GPU execution, or model serving.
- No arbitrary Python callbacks, arbitrary Optuna search spaces, or unrestricted
  estimator parameters.
- No automatic changes to customer code, training jobs, deployment, or gate
  thresholds.

## Validation

Test recipe validation and estimator construction for all new packages;
time-boundary behavior for lag and rolling features; five-fold chronological
evaluation; R2 calculation; deterministic search replay; trial-budget
enforcement; package-missing errors; experiment and manifest provenance; and
all existing AIDM/AIDD regression tests. Run the full suite once after merging
the completed worktree.
