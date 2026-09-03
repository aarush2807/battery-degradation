"""Evaluation metrics and model comparison tables."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from battery_degradation.utils import get_logger

logger = get_logger(__name__)


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-8
    if not mask.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def max_abs_error(y_true, y_pred) -> float:
    return float(np.max(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "max_abs_error": max_abs_error(y_true, y_pred),
    }


def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str,
    horizon: int,
    split: str = "validation",
) -> dict[str, Any]:
    metrics = regression_metrics(y_true, y_pred)
    metrics.update({"model": model_name, "horizon": horizon, "split": split})
    return metrics


def metrics_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # rank within horizon+split by rmse
    df["rank"] = df.groupby(["horizon", "split"])["rmse"].rank(method="min")
    return df.sort_values(["horizon", "split", "rmse"]).reset_index(drop=True)


def best_model_per_horizon(metrics_df: pd.DataFrame, split: str = "validation") -> pd.DataFrame:
    sub = metrics_df[metrics_df["split"] == split]
    if sub.empty:
        return sub
    idx = sub.groupby("horizon")["rmse"].idxmin()
    return sub.loc[idx].reset_index(drop=True)


def per_battery_errors(
    meta: pd.DataFrame,
    y_true,
    y_pred,
) -> pd.DataFrame:
    tmp = meta.copy()
    tmp["y_true"] = np.asarray(y_true, dtype=float)
    tmp["y_pred"] = np.asarray(y_pred, dtype=float)
    tmp["abs_err"] = (tmp["y_true"] - tmp["y_pred"]).abs()
    tmp["sq_err"] = (tmp["y_true"] - tmp["y_pred"]) ** 2
    rows = []
    for bid, g in tmp.groupby("battery_id"):
        rows.append(
            {
                "battery_id": bid,
                "mae": float(g["abs_err"].mean()),
                "rmse": float(np.sqrt(g["sq_err"].mean())),
                "n": int(len(g)),
            }
        )
    out = pd.DataFrame(rows).sort_values("rmse")
    return out
