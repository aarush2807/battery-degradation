# Battery Degradation Analytics & Forecasting

Production-style Python system for predicting lithium-ion battery capacity / State of Health (SOH), degradation trajectories, Remaining Useful Life (RUL), and End of Life (EOL) from historical charge/discharge cycle data.

> **Disclaimer:** Predictions depend on data quality and operating conditions (chemistry, temperature, C-rate, DoD, calendar aging, duty cycle, etc.). Models trained on one profile may not generalize to another. Long-range forecasts are more uncertain. Tree models interpolate well but extrapolate poorly; trend models extrapolate better but may miss nonlinear fade. Uncertainty bands are **approximate**, not calibrated confidence intervals. Degradation knee detection is **experimental**. Synthetic data is for software demo only — not a validated physics simulation.

## Features

- Leakage-safe lag / rolling / slope / acceleration / cumulative feature engineering
- Direct multi-horizon forecasting (primary) + recursive and hybrid trajectories
- Baselines, regularized linear, RF / ExtraTrees / HistGradientBoosting, optional XGBoost
- Per-horizon hybrid weighted ensembles (ML + trend) selected on **validation only**
- Chronological and expanding-window validation (no shuffled splits)
- RUL / EOL estimation, experimental knee detection, engineering diagnostic flags
- CLI + `BatteryPredictor` Python API + Streamlit / Plotly dashboard
- Matplotlib static reports under `outputs/figures/`

## Architecture

```
battery_degradation/   # core library (schema, features, models, pipeline, …)
dashboard/             # Streamlit UI calling the library
scripts/               # synthetic data + thin CLI wrappers
tests/                 # pytest including leakage tests
config/default.yaml
data/{raw,processed}/
models/
outputs/{figures,predictions,reports}/
main.py
```

## Installation

```bash
cd battery_degradation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: `pip install xgboost` then set `models.xgboost: true` in config.

## Quick start

```bash
python scripts/generate_synthetic_data.py
python main.py train --data data/raw/synthetic_battery_cycles.csv
python main.py evaluate --data data/raw/synthetic_battery_cycles.csv
python main.py forecast --data data/raw/synthetic_battery_cycles.csv --battery-id BATTERY_001 --cycles 100
streamlit run dashboard/app.py
pytest
```

## Dataset format

Flexible CSV schema. Required: a cycle index + a capacity-like column. Optional: temperature, resistance, voltage/current stats, energy, efficiency, `battery_id`, timestamps.

Aliases are resolved via `battery_degradation/schema.py` (e.g. `cycle` → `cycle_number`, `Capacity` → `capacity_ah`). Ambiguous mappings are logged and can be overridden in config (`data.column_overrides`) or the Streamlit sidebar.

## SOH and RUL

\[
\mathrm{SOH} = \frac{\text{current capacity}}{\text{reference capacity}}
\]

Default reference = average capacity over the first **5** cycles (`first_n_average`). Alternatives: first cycle or user-rated capacity.

Default EOL: SOH ≤ **0.80**.

\[
\mathrm{RUL}(t) = \max(0,\ \widehat{\text{EOL cycle}} - t)
\]

## Hybrid ensemble design

For each horizon, candidates are scored with time-aware validation RMSE. A nonnegative weighted ensemble mixes strong tree models with trend extrapolators (weights ∝ 1/RMSE, sum to 1). Weights are **horizon-specific** so short-horizon vs long-horizon tradeoffs are learned from validation—not hard-coded. Stacking is compared when enabled; the winner is refit on train+val and evaluated once on untouched test.

## Why time-series validation matters

Random `train_test_split` leaks future cycle information into training. This project uses chronological within-battery splits (and optional cross-battery holdout). Feature rows at cycle \(t\) never use measurements from cycles \(> t\).

## Python API

```python
from battery_degradation.pipeline import BatteryPredictor

predictor = BatteryPredictor()
predictor.fit("data/raw/battery_cycles.csv")
forecast = predictor.forecast(battery_id="BATTERY_001", future_cycles=100)
print(forecast.head())
predictor.save()
```

## CLI

```bash
python main.py train --data path/to.csv
python main.py evaluate --data path/to.csv
python main.py forecast --data path/to.csv --battery-id BATTERY_001 --cycles 100
python main.py dashboard
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

Pages: Overview, Battery Explorer, Forecasting, Model Performance, Diagnostics, Data Quality.

## Testing

```bash
pytest -q
```

Leakage tests verify that mutating/removing future cycles does not change past features and that target columns never enter `X`.

## Train on a real CSV

1. Ensure columns include cycle + capacity (or set overrides).
2. `python main.py train --data /path/to/your.csv --no-tuning` (or enable tuning in config).
3. `streamlit run dashboard/app.py` and upload the same CSV, or rely on saved `models/` + `data/processed/`.

## Configuration

See [`config/default.yaml`](config/default.yaml) for horizons, lags, EOL, model toggles, and dashboard defaults.

## License

Demo / engineering toolkit — use at your own risk for research and development.
