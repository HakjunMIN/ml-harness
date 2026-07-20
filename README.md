# Power Forecasting Harness

Power Forecasting Harness is an offline demonstration of AI-assisted power-generation forecasting. It generates or accepts tabular plant/weather data, evaluates legacy baselines, searches deterministic feature candidates with AIDM, promotes a safe manifest through AIDD, and writes auditable artifacts for review.

## Goal and safety scope

The goal is to show a repeatable forecasting workflow from data contract to reportable artifacts. The safety scope is intentionally narrow:

- Runs locally on CSV-like tabular data and a SQLite experiment store.
- Promotes only feature specifications that pass validation gates and manifest checks.
- Generates deterministic Python feature code under the selected output directory.
- Does not deploy models, start production jobs, modify live systems, make autonomous operational decisions, or manage live plant controls.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[dev,dashboard]'
python3 -m pytest -q
```

Run the full demo:

```bash
python3 -m power_forecasting.cli all --output artifacts/demo --days 60 --plants 3 --seed 42
```

## CLI examples

Commands that persist files place them under `--output`. `generate-data`, `aidm`,
`aidd`, and `all` write artifacts; `legacy` reads a dataset and prints metrics
to stdout without persisting a legacy metrics artifact.

Generate a deterministic synthetic dataset:

```bash
python3 -m power_forecasting.cli generate-data --output artifacts/demo --days 60 --plants 3 --seed 42
```

Evaluate legacy models against an existing dataset:

```bash
python3 -m power_forecasting.cli legacy --output artifacts/demo --dataset artifacts/demo/dataset.csv --folds 3
```

Run AIDM feature discovery and write `experiments.db`, `promotion_manifest.json`, and a report:

```bash
python3 -m power_forecasting.cli aidm --output artifacts/demo --dataset artifacts/demo/dataset.csv --folds 3 --seed 42
```

Render the promoted feature module from a promoted manifest:

```bash
python3 -m power_forecasting.cli aidd --output artifacts/demo --manifest artifacts/demo/promotion_manifest.json
```

Run the complete workflow:

```bash
python3 -m power_forecasting.cli all --output artifacts/demo --days 60 --plants 3 --seed 42
```

Optional gate overrides are available for experiments:

```bash
python3 -m power_forecasting.cli all --output artifacts/demo --minimum-improvement 0.01 --max-plant-regression 0.03
```

The production defaults are a minimum NMAE improvement ratio of `0.01` and a maximum per-plant NMAE regression of `0.03`.

## Legacy model meanings

The legacy comparison intentionally mixes simple and richer baselines:

- `Mean`: historical plant/hour mean baseline using plant ID and timestamp.
- `Weather`: actual-weather diagnostic/oracle baseline using `actual_*` columns. It is not a forecast-time production model because actual weather is unavailable before the forecast horizon.
- `ForecastWeather`: linear forecast-weather baseline using forecast irradiance, temperature, cloud cover, wind speed, and capacity.
- `Ldaps`: linear baseline using LDAPS-style forecast columns.
- `SPOT`: gradient-boosted forecast-time baseline using forecast weather, capacity, latitude, and longitude. AIDM compares feature candidates against SPOT.

## Architecture and data flow

```text
synthetic generator or customer adapter
        |
        v
data contract validation
        |
        +--> legacy model evaluation
        |
        +--> AIDM feature search with chronological folds
                  |
                  v
            promotion manifest and SQLite experiment store
                  |
                  v
            AIDD manifest validation and deterministic code generation
                  |
                  v
            performance report, generated/promoted_features.py, dashboard/notebooks
```

Main modules:

- `power_forecasting.data`: synthetic data, timestamp parsing, and contract validation.
- `power_forecasting.models`: legacy model definitions and forecast-time feature sets.
- `power_forecasting.features`: deterministic feature spec transforms.
- `power_forecasting.evaluation`: chronological validation and metrics.
- `power_forecasting.aidm`: candidate catalog, experiment recording, ranking, gates, and manifest creation.
- `power_forecasting.aidd`: safe manifest validation and generated feature-module rendering.
- `power_forecasting.reporting`: Markdown performance report generation.
- `dashboard/app.py`: Streamlit artifact viewer.

## Data contract

Every dataset must contain the required semantic columns below. Extra columns may exist, but the workflow only depends on these names.

| Column | Type and expectation | Meaning |
| --- | --- | --- |
| `plant_id` | non-empty string | Stable plant identifier. |
| `timestamp` | parseable datetime | Forecast/actual timestamp. Rows must be unique by `plant_id,timestamp`. |
| `capacity_mw` | finite number greater than zero | Plant capacity used for clipping and NMAE normalization. |
| `latitude` | finite number | Plant latitude for SPOT. |
| `longitude` | finite number | Plant longitude for SPOT. |
| `actual_irradiance` | finite number | Actual irradiance for diagnostics and synthetic target realism. Not allowed in promoted features. |
| `actual_temperature` | finite number | Actual temperature for diagnostics. Not allowed in promoted features. |
| `actual_cloud_cover` | finite number | Actual cloud cover for diagnostics. Not allowed in promoted features. |
| `actual_wind_speed` | finite number | Actual wind speed for diagnostics. Not allowed in promoted features. |
| `forecast_irradiance` | finite number | Forecast-time irradiance input. |
| `forecast_temperature` | finite number | Forecast-time temperature input. |
| `forecast_cloud_cover` | finite number | Forecast-time cloud-cover input. |
| `forecast_wind_speed` | finite number | Forecast-time wind-speed input. |
| `ldaps_irradiance` | finite number | LDAPS-style forecast irradiance. |
| `ldaps_temperature` | finite number | LDAPS-style forecast temperature. |
| `ldaps_cloud_cover` | finite number | LDAPS-style forecast cloud cover. |
| `ldaps_humidity` | finite number | LDAPS-style forecast humidity. |
| `generation_mw` | finite number in `[0, capacity_mw]` | Target generation. Not allowed in promoted features. |

Validation rejects missing required columns, unparseable timestamps, duplicate `plant_id,timestamp` keys, blank plant IDs, non-finite numeric values, non-positive capacity, and target values outside `[0, capacity_mw]`.

### Customer adapter expectations

A customer adapter should map source systems into the contract before calling the CLI or Python API:

1. Emit one row per plant/timestamp with stable plant IDs and consistent timestamp semantics.
2. Provide all required numeric columns as finite values with MW units for capacity and generation.
3. Keep forecast-time columns separate from `actual_*` diagnostic columns and from `generation_mw`.
4. Use only information available at forecast time for features intended for promotion.
5. Run `validate_dataset` before evaluation and fail fast on contract errors.
6. Preserve chronological order or allow the workflow to sort/validate timestamps before folds.

## AIDM discovery, validation, gates, and experiment store

AIDM starts with SPOT as the forecast-time baseline. It evaluates a bounded catalog of deterministic feature specs:

- hour sine/cosine
- day-of-year sine/cosine
- effective irradiance
- temperature derating
- cloud attenuation
- irradiance-temperature interaction

It first scores single feature groups, retains the configured top single candidates, then evaluates two- and three-group combinations. Evaluation uses chronological folds: each validation block is later than its training data, preventing future-to-past leakage.

Every baseline and candidate run is stored in `experiments.db` with parameters, metrics, artifacts, status, and errors. The winner must pass all promotion gates:

- improvement ratio: `(baseline_nmae - winner_nmae) / baseline_nmae >= 0.01` by default
- per-plant regression: each winner-vs-baseline NMAE delta must be `<= 0.03` by default
- feature availability: promoted specs cannot use `generation_mw` or any `actual_*` input

The manifest records the seed, baseline and winner metrics, selected specs, thresholds, improvement ratio, per-plant deltas, decision, and failed gates.

## AIDD constrained deterministic safety

AIDD reads only a promotion manifest and writes `generated/promoted_features.py`. It validates:

- schema version and `decision == "promote"`
- non-empty selected feature specs
- known deterministic transforms and primitive literal parameters
- no duplicate feature names
- no target leakage or actual-weather inputs
- baseline model provenance is exactly `SPOT`
- improvement ratio and per-plant deltas satisfy manifest thresholds
- winner name matches the selected specs

The generated module contains `PROMOTED_FEATURE_SPECS` and `build_promoted_features(frame)`. It applies the same deterministic transforms as the runtime feature engine and does not include training, deployment, network access, or autonomous behavior.

## Artifact layout

A successful `all` run writes:

```text
artifacts/demo/
├── dataset.csv
├── experiments.db
├── promotion_manifest.json
├── performance_report.md
└── generated/
    └── promoted_features.py
```

The canonical review evidence is versioned at
`artifacts/demo/promotion_manifest.json` and
`artifacts/demo/generated/promoted_features.py`. These files make the promoted
feature decision and generated production feature module reviewable in code
review; the manifest uses deterministic logical run identifiers for stable
regeneration. Other demo artifacts, including the dataset, SQLite experiment
DB, performance report, and transient run outputs, remain local and ignored.

On rejection, `dataset.csv`, `experiments.db`, `promotion_manifest.json`, and `performance_report.md` are still useful for diagnosis. `generated/promoted_features.py` is written only after a promoted manifest passes AIDD validation.

## Notebooks and dashboard

The notebooks in `notebooks/` are short workflow demonstrations that assume `Path("artifacts/demo")`:

- `01_legacy_baseline.ipynb`
- `02_aidm_feature_discovery.ipynb`
- `03_aidd_promotion.ipynb`

After creating artifacts, launch the dashboard:

```bash
streamlit run dashboard/app.py -- --artifacts artifacts/demo
```

The dashboard reads local artifacts, shows the promotion decision, winner metrics, AIDM ranking, experiment runs, report text, selected specs, and discovered artifact paths.

## Production migration and extension points not included

The repository is a local harness, not a production platform. The following are extension points and are not included:

- MLflow or managed experiment tracking in place of SQLite.
- A customer data adapter connected to live source systems.
- A deployment agent, scheduler, model registry, serving endpoint, or rollout mechanism.
- Live monitoring, alerting, rollback, or production incident automation.
- An automated diagnosis/self-improving loop that changes production behavior without review.

Production migration should add those pieces explicitly, keep the data contract boundary, retain chronological validation, and require human review before deployment.

## Testing

Use the existing test suite:

```bash
python3 -m pytest -q
python3 -m compileall -q src dashboard artifacts/demo/generated
```

The end-to-end smoke test runs the production CLI path:

```bash
python3 -m power_forecasting.cli all --output <tmp> --days 45 --plants 2 --seed 21
```

It asserts a promoted decision, default `0.01`/`0.03` gates, non-empty selected specs, `generated/promoted_features.py`, and a non-trivial report.

After a successful run, validate the promotion evidence. The snippet checks
`artifacts/demo` by default; set `OUTPUT_DIR` to inspect another output
directory:

<!-- readme-evidence-check: start -->
```bash
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/demo}" python3 - <<'PY'
import json
import math
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_DIR"])
manifest = json.loads((root / "promotion_manifest.json").read_text(encoding="utf-8"))
assert manifest["decision"] == "promote"
assert manifest["selected_specs"]
assert (root / "performance_report.md").stat().st_size > 500
winner_nmae = manifest["winner"]["metrics"]["nmae"]
assert math.isfinite(winner_nmae)
print(winner_nmae)
PY
```
<!-- readme-evidence-check: end -->

Expected: prints the promoted winner NMAE and exits zero.

## Troubleshooting and reject behavior

- `ERROR: dataset not found`: run `generate-data` first or pass the correct `--dataset`.
- `ERROR: missing required columns` or `invalid timestamps`: fix the customer adapter output to satisfy the data contract.
- `ERROR: AIDM rejected promotion`: inspect `promotion_manifest.json`, `failed_gates`, `experiments.db`, and `performance_report.md`. Rejection is expected when a candidate does not meet improvement, per-plant regression, or feature-availability gates.
- Missing `generated/promoted_features.py`: the manifest was rejected or AIDD validation failed. The workflow intentionally avoids generated code unless promotion is safe.
- Streamlit import error: install dashboard extras with `python3 -m pip install -e '.[dashboard]'`.
