"""Pipeline integration tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from battery_degradation.config import load_config
from battery_degradation.pipeline import BatteryPredictor


def _small_multi_battery():
    rows = []
    for bid, fade in [("BATTERY_001", 0.0015), ("BATTERY_002", 0.0025)]:
        for c in range(1, 120):
            rows.append(
                {
                    "battery_id": bid,
                    "cycle_number": c,
                    "capacity_ah": 4.5 - fade * c + np.random.normal(0, 0.002),
                    "temperature_mean": 25 + np.random.normal(0, 0.5),
                    "internal_resistance": 0.02 + 0.00002 * c,
                    "energy_wh": 15.0,
                    "charge_time": 3600,
                    "discharge_time": 3400,
                }
            )
    return pd.DataFrame(rows)


def test_pipeline_fit_forecast_save_load(tmp_path):
    cfg = load_config()
    cfg["tuning"]["enabled"] = False
    cfg["forecasting"]["horizons"] = [1, 5, 10]
    cfg["models"] = {
        "last_value": True,
        "historical_linear": True,
        "linear_regression": True,
        "polynomial_regression": False,
        "ridge": True,
        "lasso": False,
        "elastic_net": False,
        "random_forest": True,
        "extra_trees": False,
        "gradient_boosting": False,
        "xgboost": False,
        "ensemble": True,
        "stacking": False,
    }
    cfg["features"]["rolling_windows"] = [3, 5, 10]
    cfg["features"]["slope_windows"] = [5, 10]
    cfg["features"]["lag_cycles"] = [1, 2, 3]
    cfg["paths"]["models_dir"] = str(tmp_path / "models")
    cfg["paths"]["figures_dir"] = str(tmp_path / "figures")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")

    pred = BatteryPredictor(config=cfg)
    df = _small_multi_battery()
    pred.fit(df)
    assert pred.metrics_df_ is not None
    assert 1 in pred.models_by_horizon_
    fc = pred.forecast(battery_id="BATTERY_001", future_cycles=20)
    assert len(fc) == 20
    assert {"predicted_capacity", "predicted_soh", "estimated_rul"}.issubset(fc.columns)
    assert (fc["estimated_rul"] >= 0).all()

    pred.save(tmp_path / "models")
    pred2 = BatteryPredictor(config=cfg).load(tmp_path / "models")
    assert pred2.models_by_horizon_
    fc2 = pred2.forecast(battery_id="BATTERY_001", future_cycles=10)
    assert len(fc2) == 10


def test_single_battery_without_id():
    cfg = load_config()
    cfg["tuning"]["enabled"] = False
    cfg["forecasting"]["horizons"] = [1, 5]
    cfg["models"] = {
        "last_value": True,
        "historical_linear": True,
        "linear_regression": True,
        "polynomial_regression": False,
        "ridge": False,
        "lasso": False,
        "elastic_net": False,
        "random_forest": True,
        "extra_trees": False,
        "gradient_boosting": False,
        "xgboost": False,
        "ensemble": False,
        "stacking": False,
    }
    rows = []
    for c in range(1, 100):
        rows.append({"cycle": c, "capacity": 4.0 - 0.002 * c, "temp": 25.0})
    df = pd.DataFrame(rows)
    pred = BatteryPredictor(config=cfg)
    pred.fit(df)
    fc = pred.forecast(future_cycles=5)
    assert len(fc) == 5
