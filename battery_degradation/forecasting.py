"""Multi-step forecasting: direct, recursive, and hybrid post-processing."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from battery_degradation.features import build_features_for_battery
from battery_degradation.rul import estimate_eol_from_trajectory, rul_at_cycle
from battery_degradation.uncertainty import (
    attach_soh_bounds,
    ensemble_disagreement_intervals,
    tree_prediction_intervals,
)
from battery_degradation.ensemble import WeightedEnsemble
from battery_degradation.utils import clamp, get_logger

logger = get_logger(__name__)


def apply_capacity_constraints(
    preds: np.ndarray,
    reference_capacity: float,
    capacity_upper_factor: float = 1.05,
    soh_min: float = 0.0,
    soh_max: float = 1.05,
) -> np.ndarray:
    upper = reference_capacity * capacity_upper_factor
    lower = reference_capacity * soh_min
    return np.clip(preds, lower, upper)


def smooth_forecast(values: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()
    s = pd.Series(values).rolling(window=window, min_periods=1, center=True).mean()
    return s.to_numpy(dtype=float)


def soft_monotonic_degradation(values: np.ndarray, max_increase: float = 0.002) -> np.ndarray:
    """
    Soft monotonicity: allow tiny increases (noise/recovery) but prevent large rises.
    Does not force every step to decrease.
    """
    out = values.copy()
    if len(out) == 0:
        return out
    running_max_allowed = out[0] + max_increase
    for i in range(1, len(out)):
        # Cap upward jumps relative to previous
        if out[i] > out[i - 1] + max_increase:
            out[i] = out[i - 1] + max_increase
        # Also gently enforce overall non-increase trend envelope
        if out[i] > running_max_allowed:
            out[i] = running_max_allowed
        running_max_allowed = max(running_max_allowed - 1e-6, out[i])
    return out


def direct_multi_horizon_forecast(
    feature_row: pd.DataFrame,
    models_by_horizon: dict[int, Any],
    horizons: list[int],
    reference_capacity: float,
    current_cycle: float,
    current_capacity: float,
    eol_soh: float = 0.80,
    uncertainty_cfg: Optional[dict] = None,
    constraints_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Question A style: use dedicated models for each horizon from state at t.
    Fills intermediate cycles by interpolation between anchored horizon predictions.
    """
    uncertainty_cfg = uncertainty_cfg or {}
    constraints_cfg = constraints_cfg or {}
    anchors: dict[int, dict[str, float]] = {0: {"capacity": current_capacity}}

    for h in sorted(horizons):
        model = models_by_horizon.get(h)
        if model is None:
            continue
        X = feature_row
        mean = np.asarray(model.predict(X), dtype=float)[0]
        lower = mean
        upper = mean
        if uncertainty_cfg.get("enabled", True):
            lp = uncertainty_cfg.get("lower_percentile", 5)
            up = uncertainty_cfg.get("upper_percentile", 95)
            try:
                if isinstance(model, WeightedEnsemble):
                    mean_a, lo, hi = ensemble_disagreement_intervals(model, X, lp, up)
                else:
                    mean_a, lo, hi = tree_prediction_intervals(model, X, lp, up)
                mean, lower, upper = float(mean_a[0]), float(lo[0]), float(hi[0])
            except Exception:
                pass
        if constraints_cfg.get("apply_constraints", True):
            mean = float(
                apply_capacity_constraints(
                    np.array([mean]),
                    reference_capacity,
                    constraints_cfg.get("capacity_upper_factor", 1.05),
                    constraints_cfg.get("soh_min", 0.0),
                    constraints_cfg.get("soh_max", 1.05),
                )[0]
            )
            lower = float(min(lower, mean))
            upper = float(max(upper, mean))
        anchors[h] = {"capacity": mean, "lower": lower, "upper": upper}

    max_h = max(horizons) if horizons else 0
    rows = []
    sorted_anchors = sorted(anchors.keys())
    for step in range(1, max_h + 1):
        # interpolate between nearest anchors
        left = max([a for a in sorted_anchors if a <= step])
        right_candidates = [a for a in sorted_anchors if a >= step]
        right = right_candidates[0] if right_candidates else left
        if right == left:
            cap = anchors[left]["capacity"]
            lo = anchors[left].get("lower", cap)
            hi = anchors[left].get("upper", cap)
        else:
            w = (step - left) / (right - left)
            cap = (1 - w) * anchors[left]["capacity"] + w * anchors[right]["capacity"]
            lo = (1 - w) * anchors[left].get("lower", anchors[left]["capacity"]) + w * anchors[
                right
            ].get("lower", anchors[right]["capacity"])
            hi = (1 - w) * anchors[left].get("upper", anchors[left]["capacity"]) + w * anchors[
                right
            ].get("upper", anchors[right]["capacity"])
        soh = cap / max(reference_capacity, 1e-12)
        rows.append(
            {
                "cycle_number": current_cycle + step,
                "predicted_capacity": cap,
                "predicted_soh": soh,
                "lower_capacity": lo,
                "upper_capacity": hi,
                "lower_soh": lo / max(reference_capacity, 1e-12),
                "upper_soh": hi / max(reference_capacity, 1e-12),
                "horizon_step": step,
                "method": "direct",
            }
        )
    forecast = pd.DataFrame(rows)
    if len(forecast):
        eol = estimate_eol_from_trajectory(
            np.concatenate([[current_cycle], forecast["cycle_number"].to_numpy()]),
            np.concatenate(
                [[current_capacity / max(reference_capacity, 1e-12)], forecast["predicted_soh"].to_numpy()]
            ),
            eol_soh=eol_soh,
            current_cycle=current_cycle,
        )
        forecast["estimated_rul"] = [
            rul_at_cycle(eol.get("eol_cycle"), c) for c in forecast["cycle_number"]
        ]
        forecast.attrs["eol"] = eol
    return forecast


def recursive_forecast(
    history_df: pd.DataFrame,
    model_h1: Any,
    feature_cols: list[str],
    n_steps: int,
    config: dict[str, Any],
    reference_capacity: float,
    exogenous_strategy: str = "carry_last",
    future_profile: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Question B style recursive trajectory using t+1 model repeatedly.
    Accumulates error; exogenous unknowns handled via strategy.
    """
    feat_cfg = config.get("features", {})
    battery_cfg = config.get("battery", {})
    high_temp = float(battery_cfg.get("high_temperature_threshold", 40.0))
    work = history_df.sort_values("cycle_number").copy()
    rows = []
    future_profile = future_profile or {}

    for step in range(1, n_steps + 1):
        featured = build_features_for_battery(
            work,
            lag_cycles=feat_cfg.get("lag_cycles", [1, 2, 3, 5, 10]),
            rolling_windows=feat_cfg.get("rolling_windows", [3, 5, 10, 20, 50]),
            slope_windows=feat_cfg.get("slope_windows", [5, 10, 20, 50]),
            high_temp_threshold=high_temp,
        )
        last = featured.iloc[[-1]]
        X = last.reindex(columns=feature_cols)
        # fill missing feature cols
        X = X.fillna(0.0)
        pred = float(np.asarray(model_h1.predict(X), dtype=float)[0])
        pred = float(
            apply_capacity_constraints(
                np.array([pred]),
                reference_capacity,
                config.get("forecasting", {}).get("capacity_upper_factor", 1.05),
            )[0]
        )
        new_cycle = float(work["cycle_number"].iloc[-1]) + 1
        new_row = work.iloc[[-1]].copy()
        new_row["cycle_number"] = new_cycle
        new_row["capacity_ah"] = pred
        new_row["soh"] = pred / max(reference_capacity, 1e-12)
        new_row["soh_pct"] = new_row["soh"] * 100

        # exogenous handling
        for col in ("temperature_mean", "current_mean", "internal_resistance"):
            if col not in work.columns:
                continue
            if col in future_profile:
                new_row[col] = future_profile[col]
            elif exogenous_strategy == "carry_last":
                new_row[col] = work[col].iloc[-1]
            elif exogenous_strategy == "rolling_mean":
                new_row[col] = work[col].tail(10).mean()
            elif exogenous_strategy == "omit":
                new_row[col] = np.nan

        work = pd.concat([work, new_row], ignore_index=True)
        rows.append(
            {
                "cycle_number": new_cycle,
                "predicted_capacity": pred,
                "predicted_soh": pred / max(reference_capacity, 1e-12),
                "lower_capacity": pred,
                "upper_capacity": pred,
                "lower_soh": pred / max(reference_capacity, 1e-12),
                "upper_soh": pred / max(reference_capacity, 1e-12),
                "horizon_step": step,
                "method": "recursive",
            }
        )

    forecast = pd.DataFrame(rows)
    eol_soh = float(config.get("battery", {}).get("eol_soh", 0.80))
    current_cycle = float(history_df["cycle_number"].max())
    if len(forecast):
        eol = estimate_eol_from_trajectory(
            forecast["cycle_number"].to_numpy(),
            forecast["predicted_soh"].to_numpy(),
            eol_soh=eol_soh,
            current_cycle=current_cycle,
        )
        forecast["estimated_rul"] = [
            rul_at_cycle(eol.get("eol_cycle"), c) for c in forecast["cycle_number"]
        ]
        forecast.attrs["eol"] = eol
    return forecast


def postprocess_forecast(
    forecast: pd.DataFrame,
    smoothing_window: int = 5,
    apply_monotonic: bool = False,
) -> pd.DataFrame:
    out = forecast.copy()
    if "predicted_capacity" in out.columns:
        out["predicted_capacity_raw"] = out["predicted_capacity"]
        out["predicted_capacity_smoothed"] = smooth_forecast(
            out["predicted_capacity"].to_numpy(), smoothing_window
        )
        if apply_monotonic:
            out["predicted_capacity_monotonic"] = soft_monotonic_degradation(
                out["predicted_capacity_smoothed"].to_numpy()
            )
        else:
            out["predicted_capacity_monotonic"] = out["predicted_capacity_smoothed"]
    return out
