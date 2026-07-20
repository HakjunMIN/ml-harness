# AI-Driven Power Generation Forecasting Design

## 1. Purpose

Build a production-oriented demonstration that first recreates a conventional,
manually operated power-generation forecasting system and then upgrades it with:

- AI-Driven Modeling (AIDM) for automated feature discovery and evaluation.
- AI-Driven Development (AIDD) for controlled promotion of validated features
  into production-quality source code.
- Reproducible experiment tracking, quality gates, reports, and a lightweight
  dashboard.

The first release focuses on feature engineering. Its interfaces must allow later
expansion to model selection, training logic, diagnostics, deployment, and a
multi-agent self-improving loop.

## 2. Assumptions

- The demonstration forecasts hourly solar photovoltaic generation per plant.
- Customer data is unavailable, so deterministic synthetic data represents plant
  metadata, actual weather, weather forecasts, LDAPS-like forecasts, and output.
- The primary metric is normalized mean absolute error (NMAE). Lower is better.
- All evaluation uses chronological splits. No future observations may influence
  training features.
- The demo runs on one workstation and does not require a feature store,
  Kubernetes, or a workflow orchestrator.
- Notebooks explain and invoke workflows. Reusable behavior lives in importable,
  tested Python modules.

## 3. Scope

### 3.1 Phase 1: Legacy baseline

Create a deliberately manual but valid forecasting workflow with compatible model
names:

- `Mean`: plant and hour-of-day historical mean.
- `Weather`: regression using actual weather observations.
- `ForecastWeather`: regression using forecast weather.
- `Ldaps`: regression using LDAPS-like forecast variables.
- `SPOT`: gradient-boosted model using forecast and plant variables.

The legacy notebook manually selects feature lists, trains models, prints metrics,
and writes predictions. It establishes the baseline that AIDM must beat.

### 3.2 Phase 2: AIDM

Automate candidate generation, time-safe evaluation, ranking, and selection:

- Calendar features: hour, month, day of year, cyclic encodings.
- Weather transforms: clipping, polynomial terms, and pairwise interactions.
- Domain features: clear-sky proxy, temperature derating, cloud attenuation,
  effective irradiance, and capacity-normalized weather.
- Lag and rolling features derived only from information available at prediction
  time.
- Candidate combinations evaluated through expanding-window validation.

### 3.3 Phase 3: AIDD

Promote a selected feature set through a constrained path:

1. AIDM emits a declarative `FeatureSpec`.
2. AIDD validates names, dependencies, and transform parameters.
3. A deterministic renderer writes a generated feature module.
4. Automated tests validate generated code and compare performance.
5. A promotion manifest records evidence and the exact change.

The first release does not allow an LLM to edit arbitrary source files. This
reduces syntax, security, and reproducibility risks while preserving an extension
point for an agent-generated code change request.

### 3.4 Phase 4: Observability

- Store experiment metadata and metrics in a local SQLite registry.
- Store datasets and predictions as CSV or Parquet-compatible artifacts.
- Produce a Markdown performance report.
- Provide a Streamlit dashboard when Streamlit is installed.
- Keep storage and reporting behind interfaces so MLflow or a managed service can
  replace the local implementation.

### 3.5 Deferred

- Continuous production deployment.
- Live drift-triggered retraining.
- Infrastructure monitoring.
- Feature store and model registry services.
- Autonomous multi-agent execution without a human promotion gate.

These are represented as documented extension points, not partially implemented
features.

## 4. Architecture

```text
SyntheticDataGenerator / CustomerDataAdapter
                    |
                    v
              ForecastDataset
                    |
          +---------+----------+
          |                    |
          v                    v
   LegacyModelRunner     CandidateGenerator
          |                    |
          |                    v
          |              FeatureEngine
          |                    |
          +---------+----------+
                    v
             ExperimentRunner
                    |
                    v
               RankingGate
                    |
          +---------+----------+
          |                    |
          v                    v
    ExperimentStore       PromotionManifest
                               |
                               v
                         FeatureRenderer
                               |
                               v
                    GeneratedFeatureModule
```

### 4.1 Modules

| Module | Responsibility |
| --- | --- |
| `data` | Generate deterministic demo data and validate forecast data contracts. |
| `features` | Define feature specifications and compute leakage-safe transforms. |
| `models` | Provide baseline and boosted model adapters with a common interface. |
| `evaluation` | Perform chronological validation and calculate metrics. |
| `experiments` | Persist run metadata, metrics, candidates, and artifacts. |
| `aidm` | Generate candidates, execute experiments, rank results, and select winners. |
| `aidd` | Validate selected specifications and render production feature code. |
| `reporting` | Build comparison tables and Markdown reports. |
| `cli` | Expose repeatable legacy, AIDM, and AIDD commands. |

## 5. Data Contract

Each row represents one plant and forecast timestamp:

- Identity: `plant_id`, `timestamp`.
- Plant: `capacity_mw`, `latitude`, `longitude`.
- Actual weather: `actual_irradiance`, `actual_temperature`,
  `actual_cloud_cover`, `actual_wind_speed`.
- Forecast weather: `forecast_irradiance`, `forecast_temperature`,
  `forecast_cloud_cover`, `forecast_wind_speed`.
- LDAPS-like weather: `ldaps_irradiance`, `ldaps_temperature`,
  `ldaps_cloud_cover`, `ldaps_humidity`.
- Target: `generation_mw`.

Training and evaluation validate required columns, timestamps, uniqueness,
finite numeric values, and non-negative capacity and generation. Customer
integration supplies the same logical contract through an adapter.

## 6. Evaluation and Promotion

### 6.1 Validation

- Sort by timestamp and use expanding-window folds.
- Keep all rows for a timestamp in the same fold.
- Fit transforms and models on training data only.
- Report aggregate and per-plant MAE, RMSE, and NMAE.

### 6.2 Promotion gate

A candidate is promotable only when:

- Its mean NMAE improves over the configured baseline by at least 1%.
- It does not regress any plant by more than 3% NMAE.
- All feature and generated-code tests pass.
- Its required input columns are available at prediction time.
- Repeated runs with the same seed produce the same feature specification.

The default threshold is configurable. The manifest records failed gates as
explicit reasons rather than silently selecting a weaker candidate.

## 7. AIDM Search Strategy

Use a bounded, explainable search rather than an unconstrained combinatorial
search:

1. Generate domain-safe single features.
2. Evaluate each candidate with a low-cost model.
3. Retain the best candidates by validation NMAE.
4. Evaluate bounded combinations of retained candidates.
5. Re-evaluate finalists with the production model adapter.

Every feature carries:

- Stable name and version.
- Transform type and parameters.
- Required source columns.
- Availability timing.
- Human-readable rationale.

This metadata supports reproducibility, leakage checks, code generation, and
future agent reasoning.

## 8. AIDD Change Model

The generated module exposes a pure function:

```python
def build_promoted_features(frame: pandas.DataFrame) -> pandas.DataFrame:
    ...
```

Generation is deterministic from the promotion manifest. The renderer only emits
allowlisted transforms. A future development agent may wrap this output in a pull
request, but it may not bypass tests or promotion evidence.

## 9. Error Handling

- Data-contract failures identify missing columns and invalid rows.
- Feature dependency failures identify the feature and unavailable source.
- Experiment failures are persisted with status `failed` and an error summary.
- AIDD refuses unsupported transforms instead of emitting fallback code.
- Promotion failures produce a report with every failed gate.
- CLI commands exit non-zero on failed validation or generation.

## 10. Testing

- Unit tests for metrics, split boundaries, feature transforms, and manifest
  validation.
- Leakage tests proving that future target changes cannot affect earlier
  features.
- Determinism tests for synthetic data, experiments, and generated code.
- Integration test covering data generation through AIDM selection and AIDD
  rendering.
- Smoke test importing and executing the generated module.
- Regression test confirming that the legacy baseline remains reproducible.

## 11. Project Layout

```text
notebooks/
  01_legacy_baseline.ipynb
  02_aidm_feature_discovery.ipynb
  03_aidd_promotion.ipynb
src/power_forecasting/
  data.py
  features.py
  models.py
  evaluation.py
  experiments.py
  aidm.py
  aidd.py
  reporting.py
  cli.py
tests/
artifacts/
dashboard/
```

Generated artifacts and local databases are ignored by Git. Promotion manifests
and generated production feature modules are version-controlled.

## 12. Success Criteria

- A single command creates deterministic demo data and legacy results.
- A single command runs AIDM and records all evaluated candidates.
- AIDM selects a feature set that satisfies the configured promotion gate.
- AIDD renders importable, tested feature code from the selected manifest.
- The report compares baseline and promoted performance with per-plant detail.
- The full workflow runs locally without external cloud services.
- Customer data, MLflow, managed execution, and deployment agents can be added
  through documented interfaces without replacing the core evaluation logic.

## 13. Future Self-Improving Loop

The next-stage loop consists of independently replaceable agents:

- Diagnostic Agent creates a structured improvement request from drift,
  performance, data-quality, and resource signals.
- AIDM Agent proposes and evaluates feature, model, or logic changes.
- AIDD Agent converts accepted proposals into constrained code changes.
- Validation Agent independently checks quality and operational constraints.
- Deployment Agent promotes only signed, approved manifests.

Every transition uses immutable manifests and evidence. Human approval remains a
configurable gate until operational safety and governance requirements justify a
more autonomous policy.
