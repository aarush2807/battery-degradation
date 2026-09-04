# Battery Degradation Analytics & Forecasting

End-to-end Python toolkit for lithium-ion **capacity / State of Health (SOH)** prediction, multi-horizon degradation forecasting, **Remaining Useful Life (RUL)** / End of Life (EOL) estimation, diagnostics, and an interactive Streamlit dashboard.

> **Important limitations.** Results depend on data quality and operating conditions (chemistry, temperature, C-rate, depth of discharge, calendar aging, duty cycle). Models trained on one profile often fail to transfer to another. Tree models interpolate well but extrapolate poorly; trend / linear models extrapolate better but can miss nonlinear fade and knees. Uncertainty bands are **approximate** (ensemble / tree disagreement), not calibrated confidence intervals. Knee detection is **experimental**. The bundled CSV is **synthetic demo data only** — not a validated physics simulation.

## Features

- Leakage-safe lag, rolling, slope, acceleration, and cumulative features
- Direct multi-horizon forecasting (primary) plus recursive and hybrid trajectories
- Baselines, regularized linear models, Random Forest / ExtraTrees / HistGradientBoosting, optional XGBoost
- Per-horizon hybrid weighted ensembles (ML + trend), selected on **validation RMSE only**
- Chronological within-battery (and optional cross-battery) splits — no shuffled leakage
- RUL / EOL estimation, experimental knee detection, engineering diagnostic flags
- CLI, `BatteryPredictor` API, Streamlit + Plotly dashboard, Matplotlib static reports

## Architecture

```
battery_degradation/   # core library (schema, features, models, pipeline, …)
dashboard/             # Streamlit UI
scripts/               # synthetic data + thin CLI wrappers
tests/                 # unit + leakage + pipeline tests
config/default.yaml    # horizons, models, EOL, validation, dashboard
data/raw/              # cycle CSVs (synthetic demo included)
data/processed/        # prepared cycles written by train
models/                # joblib estimators + metadata.json + metrics.csv
outputs/{figures,predictions,reports}/
main.py                # CLI entrypoint
```

**Pipeline flow:** load & schema-normalize → SOH / reference capacity → features → supervised matrices per horizon → time-aware train/val/test → fit candidates → weighted ensemble → select on validation → refit train+val → untouched test → save models / metrics / figures → forecast / RUL.

## Installation

```bash
cd battery_degradation
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional: `pip install xgboost` then set `models.xgboost: true` in `config/default.yaml`.

## Quick start (acceptance workflow)

```bash
python scripts/generate_synthetic_data.py
python main.py train --data data/raw/synthetic_battery_cycles.csv --no-tuning
python main.py evaluate --data data/raw/synthetic_battery_cycles.csv
python main.py forecast --data data/raw/synthetic_battery_cycles.csv --battery-id BATTERY_001 --cycles 100
python main.py dashboard
# or: streamlit run dashboard/app.py
pytest -q
```

A demo synthetic CSV and trained model metadata are committed so you can skip generate/train and go straight to evaluate / forecast / dashboard after `pip install`.

### Training speed notes

Default config keeps demos practical:

| Setting | Default | Why |
|--------|---------|-----|
| `tuning.enabled` | `false` | RandomizedSearchCV over trees is slow |
| `models.polynomial_regression` | `false` | Degree ≥2 on ~100 features explodes |
| `models.stacking` | `false` | Stacking CV over trees is expensive |
| Tree `n_estimators` | ~80 | Enough for demo; raise for research |

Enable deeper search with `tuning.enabled: true` and `tuning.n_iter` (e.g. 10–15), or pass nothing and edit the YAML. CLI: `--no-tuning` forces tuning off.

## Dataset format

Required: a **cycle index** and a **capacity-like** column. Optional: temperature, internal resistance, voltage/current stats, energy, efficiency, `battery_id`, timestamps.

Aliases are resolved in `battery_degradation/schema.py` (e.g. `cycle` → `cycle_number`, `Capacity` → `capacity_ah`). Ambiguous columns are logged; override via `data.column_overrides` in config or the dashboard sidebar.

## SOH and RUL

\[
\mathrm{SOH} = \frac{\text{current capacity}}{\text{reference capacity}}
\]

Default reference = mean capacity over the first **5** cycles (`first_n_average`). Alternatives: first cycle or user-rated capacity (`config/default.yaml` → `battery`).

Default EOL threshold: SOH ≤ **0.80**.

\[
\mathrm{RUL}(t) = \max\bigl(0,\ \widehat{\mathrm{EOL\ cycle}} - t\bigr)
\]

EOL is estimated by interpolating where the forecast (or historical) SOH trajectory crosses the threshold.

## Hybrid ensemble design

For each forecast horizon:

1. Fit baselines, linear/regularized, and tree models on the chronological **train** split.
2. Score on **validation** only (never peek at test for selection).
3. Build a nonnegative weighted ensemble (weights ∝ 1/RMSE, sum to 1) mixing strong ML and trend models — **horizon-specific** weights so short vs long horizons can prefer different families.
4. Optionally compare stacking (when enabled).
5. Refit the winner on train+validation; report once on untouched **test**.

## Why time-series validation matters

Random `train_test_split` leaks future cycle information into training. This project uses chronological within-battery splits. Feature rows at cycle \(t\) never use measurements from cycles \(> t\). Leakage tests mutate/remove future cycles and assert past features are unchanged.

## Python API

```python
from battery_degradation.pipeline import BatteryPredictor

predictor = BatteryPredictor()
predictor.fit("data/raw/synthetic_battery_cycles.csv")
forecast = predictor.forecast(battery_id="BATTERY_001", future_cycles=100)
print(forecast.head())
print(forecast.attrs.get("eol"))  # eol_cycle, rul, …
predictor.save()  # models/ + metadata.json
```

## CLI

```bash
python main.py train --data path/to.csv [--no-tuning] [--target capacity|soh] [--eol 0.80]
python main.py evaluate --data path/to.csv
python main.py forecast --data path/to.csv --battery-id BATTERY_001 --cycles 100 [--method direct|recursive|hybrid]
python main.py dashboard
```

## Dashboard

```bash
source .venv/bin/activate
streamlit run dashboard/app.py
# or
python main.py dashboard
```

**Important:** Use the **Navigation** radio in the left sidebar (Overview, Battery Explorer, …). Page modules live under `dashboard/views/` (not Streamlit’s special `pages/` folder) so the app is a single entrypoint with working charts.

Pages: **Overview**, **Battery Explorer**, **Forecasting**, **Model Performance**, **Diagnostics**, **Data Quality**.

Upload a CSV in the sidebar or rely on `data/raw/synthetic_battery_cycles.csv` plus saved `models/` after training. A green sidebar status line shows cycle/battery counts when data loaded successfully.


## Testing

```bash
pytest -q
```

Coverage includes schema aliases, feature correctness, leakage, RUL/EOL/knee, model smoke tests, forecasting helpers, and pipeline fit → forecast → save → load (tests write only under pytest `tmp_path`).

## Train on a real CSV

1. Ensure cycle + capacity columns (or set `data.column_overrides`).
2. `python main.py train --data /path/to/your.csv --no-tuning`
3. Open the dashboard and upload the same CSV, or use saved artifacts under `models/` and `data/processed/`.

## Regenerating demo artifacts

```bash
python scripts/generate_synthetic_data.py
python main.py train --data data/raw/synthetic_battery_cycles.csv --no-tuning
```

This refreshes `data/raw/synthetic_battery_cycles.csv`, `models/*.joblib`, `models/metadata.json`, `models/metrics.csv`, and `data/processed/prepared_cycles.csv`.

## Configuration

See [`config/default.yaml`](config/default.yaml) for horizons, lags, EOL SOH, model toggles, validation fractions, uncertainty percentiles, knee detector knobs, and dashboard defaults.

## License

Demo / engineering toolkit — use at your own risk for research and development. Not a substitute for OEM BMS validation or laboratory testing.
