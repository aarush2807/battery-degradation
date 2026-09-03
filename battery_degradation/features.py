"""Leakage-safe feature engineering for battery degradation."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from battery_degradation.utils import get_logger

logger = get_logger(__name__)

OPTIONAL_SIGNAL_COLUMNS = [
    "temperature_mean",
    "internal_resistance",
    "charge_time",
    "discharge_time",
    "voltage_mean",
    "voltage_min",
    "voltage_max",
    "current_mean",
    "current_min",
    "current_max",
    "energy_wh",
    "coulombic_efficiency",
    "charge_capacity_ah",
    "discharge_capacity_ah",
]


def _window_slope(y: np.ndarray) -> float:
    """Least-squares slope of y vs 0..n-1. Uses only provided window values."""
    n = len(y)
    if n < 2 or np.any(~np.isfinite(y)):
        return np.nan
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def _rolling_stats(series: pd.Series, window: int, prefix: str) -> pd.DataFrame:
    """
    Leakage-safe rolling stats for predicting the NEXT step from history ≤ t.

    Uses expanding shift(1) so the value at index t uses only observations ≤ t-1
    when targeting t, OR we attach features at row t using history ≤ t for
    predicting t+h. We use closed='left' via shift after rolling on raw series
    including current point for "state at t" features.

    Convention in this project:
      Features at cycle t may include measurements at cycle t (known at end of cycle t)
      and must not use cycles > t. Targets are capacity/SOH at t+h.
    """
    rolled = series.rolling(window=window, min_periods=max(1, min(3, window)))
    return pd.DataFrame(
        {
            f"{prefix}_rolling_mean_{window}": rolled.mean(),
            f"{prefix}_rolling_std_{window}": rolled.std(),
            f"{prefix}_rolling_min_{window}": rolled.min(),
            f"{prefix}_rolling_max_{window}": rolled.max(),
        },
        index=series.index,
    )


def _slope_series(series: pd.Series, window: int, name: str) -> pd.Series:
    values = series.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        if len(window_vals) >= 2:
            out[i] = _window_slope(window_vals)
    return pd.Series(out, index=series.index, name=name)


def _lags(series: pd.Series, lags: Sequence[int], prefix: str) -> pd.DataFrame:
    data = {f"{prefix}_lag_{lag}": series.shift(lag) for lag in lags}
    return pd.DataFrame(data, index=series.index)


def build_features_for_battery(
    battery_df: pd.DataFrame,
    lag_cycles: Sequence[int] = (1, 2, 3, 5, 10),
    rolling_windows: Sequence[int] = (3, 5, 10, 20, 50),
    slope_windows: Sequence[int] = (5, 10, 20, 50),
    high_temp_threshold: float = 40.0,
) -> pd.DataFrame:
    """Build leakage-safe features for a single battery (sorted by cycle)."""
    df = battery_df.sort_values("cycle_number").reset_index(drop=True).copy()
    parts: list[pd.DataFrame] = [df]

    cycle = df["cycle_number"].astype(float)
    capacity = df["capacity_ah"].astype(float)
    soh = df["soh"].astype(float)

    # Cycle position nonlinearities
    parts.append(
        pd.DataFrame(
            {
                "cycle_number_squared": cycle**2,
                "sqrt_cycle": np.sqrt(np.clip(cycle, 0, None)),
                "log1p_cycle": np.log1p(np.clip(cycle, 0, None)),
            },
            index=df.index,
        )
    )

    # Lags
    parts.append(_lags(capacity, lag_cycles, "capacity"))
    parts.append(_lags(soh, lag_cycles, "SOH"))

    # Rolling
    for w in rolling_windows:
        parts.append(_rolling_stats(capacity, w, "capacity"))
        parts.append(
            pd.DataFrame(
                {
                    f"SOH_rolling_mean_{w}": soh.rolling(w, min_periods=max(1, min(3, w))).mean(),
                    f"SOH_rolling_std_{w}": soh.rolling(w, min_periods=max(1, min(3, w))).std(),
                },
                index=df.index,
            )
        )

    # Slopes and deltas
    for w in slope_windows:
        parts.append(_slope_series(capacity, w, f"capacity_slope_{w}").to_frame())
        if w in (5, 10, 20):
            parts.append(_slope_series(soh, w, f"SOH_slope_{w}").to_frame())

    for d in (1, 5, 10):
        parts.append(
            pd.DataFrame(
                {
                    f"delta_capacity_{d}": capacity.diff(d),
                    f"delta_SOH_{d}": soh.diff(d),
                },
                index=df.index,
            )
        )
        # percentage degradation relative to value d cycles ago
        prev_c = capacity.shift(d)
        parts.append(
            pd.DataFrame(
                {
                    f"pct_degradation_capacity_{d}": np.where(
                        prev_c.abs() > 1e-12, (capacity - prev_c) / prev_c, np.nan
                    )
                },
                index=df.index,
            )
        )

    # Velocity / acceleration (based on slope_5)
    slope5 = _slope_series(capacity, 5, "capacity_slope_5_tmp")
    slope10 = _slope_series(capacity, 10, "capacity_slope_10_tmp")
    vel = slope5
    acc = slope5.diff(5)
    parts.append(
        pd.DataFrame(
            {
                "degradation_velocity": vel,
                "degradation_acceleration": acc,
                "change_in_capacity_slope": slope10 - slope5,
            },
            index=df.index,
        )
    )

    # Optional signals
    if "temperature_mean" in df.columns:
        temp = pd.to_numeric(df["temperature_mean"], errors="coerce")
        parts.append(_lags(temp, [1, 5], "temperature"))
        for w in (5, 10):
            parts.append(
                pd.DataFrame(
                    {f"temperature_rolling_mean_{w}": temp.rolling(w, min_periods=1).mean()},
                    index=df.index,
                )
            )
        high = (temp > high_temp_threshold).astype(float)
        parts.append(
            pd.DataFrame(
                {"cumulative_high_temperature_exposure": high.cumsum()},
                index=df.index,
            )
        )

    if "internal_resistance" in df.columns:
        ir = pd.to_numeric(df["internal_resistance"], errors="coerce")
        parts.append(_lags(ir, [1, 5], "resistance"))
        for w in (5, 10):
            parts.append(
                pd.DataFrame(
                    {f"resistance_rolling_mean_{w}": ir.rolling(w, min_periods=1).mean()},
                    index=df.index,
                )
            )
        ir_slope5 = _slope_series(ir, 5, "ir_slope_5")
        ir_slope10 = _slope_series(ir, 10, "ir_slope_10")
        parts.append(
            pd.DataFrame(
                {
                    "change_in_resistance_slope": ir_slope10 - ir_slope5,
                },
                index=df.index,
            )
        )

    # Cumulative features
    if "energy_wh" in df.columns:
        energy = pd.to_numeric(df["energy_wh"], errors="coerce").fillna(0.0)
        parts.append(pd.DataFrame({"cumulative_energy_throughput": energy.cumsum()}, index=df.index))
    if "charge_capacity_ah" in df.columns:
        cc = pd.to_numeric(df["charge_capacity_ah"], errors="coerce").fillna(0.0)
        parts.append(pd.DataFrame({"cumulative_charge_throughput": cc.cumsum()}, index=df.index))
    if "discharge_capacity_ah" in df.columns:
        dc = pd.to_numeric(df["discharge_capacity_ah"], errors="coerce").fillna(0.0)
        parts.append(pd.DataFrame({"cumulative_discharge_throughput": dc.cumsum()}, index=df.index))
    if "charge_time" in df.columns or "discharge_time" in df.columns:
        ct = pd.to_numeric(df.get("charge_time", 0), errors="coerce").fillna(0.0)
        dt = pd.to_numeric(df.get("discharge_time", 0), errors="coerce").fillna(0.0)
        parts.append(pd.DataFrame({"cumulative_time": (ct + dt).cumsum()}, index=df.index))

    feat = pd.concat(parts, axis=1)
    # Deduplicate columns if any overlap
    feat = feat.loc[:, ~feat.columns.duplicated()]
    return feat


def build_features(
    df: pd.DataFrame,
    config: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    config = config or {}
    feat_cfg = config.get("features", {})
    battery_cfg = config.get("battery", {})
    lag_cycles = feat_cfg.get("lag_cycles", [1, 2, 3, 5, 10])
    rolling_windows = feat_cfg.get("rolling_windows", [3, 5, 10, 20, 50])
    slope_windows = feat_cfg.get("slope_windows", [5, 10, 20, 50])
    high_temp = float(battery_cfg.get("high_temperature_threshold", 40.0))

    frames = []
    for _, group in df.groupby("battery_id", sort=False):
        frames.append(
            build_features_for_battery(
                group,
                lag_cycles=lag_cycles,
                rolling_windows=rolling_windows,
                slope_windows=slope_windows,
                high_temp_threshold=high_temp,
            )
        )
    out = pd.concat(frames, axis=0).reset_index(drop=True)
    logger.info("Built feature matrix: %s rows x %s cols", out.shape[0], out.shape[1])
    return out


def add_targets(df: pd.DataFrame, horizons: Sequence[int] = (1, 5, 10, 25, 50, 100)) -> pd.DataFrame:
    """Create capacity/SOH targets at t+h using future shift within each battery."""
    out = df.copy()
    for h in horizons:
        out[f"capacity_t_plus_{h}"] = out.groupby("battery_id")["capacity_ah"].shift(-h)
        out[f"soh_t_plus_{h}"] = out.groupby("battery_id")["soh"].shift(-h)
    return out


def get_feature_columns(df: pd.DataFrame, horizons: Sequence[int]) -> list[str]:
    """Return model feature columns excluding IDs, targets, and raw passthrough metadata."""
    target_cols = set()
    for h in horizons:
        target_cols.add(f"capacity_t_plus_{h}")
        target_cols.add(f"soh_t_plus_{h}")

    exclude = target_cols | {
        "battery_id",
        "timestamp",
        "soh_pct",
        "reference_capacity",
    }
    # Keep numeric features only
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("capacity_t_plus_") or col.startswith("soh_t_plus_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols


def make_supervised_matrix(
    featured: pd.DataFrame,
    horizon: int,
    target: str = "capacity",
    feature_cols: Optional[list[str]] = None,
    horizons: Sequence[int] = (1, 5, 10, 25, 50, 100),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Return X, y, meta for a horizon.
    meta retains battery_id and cycle_number for validation splitting.
    """
    target_col = f"{target}_t_plus_{horizon}" if target in ("capacity", "soh") else target
    if target == "capacity":
        target_col = f"capacity_t_plus_{horizon}"
    elif target == "soh":
        target_col = f"soh_t_plus_{horizon}"

    if feature_cols is None:
        feature_cols = get_feature_columns(featured, horizons)

    # Hard guard: never allow target columns in features
    forbidden = [c for c in feature_cols if "t_plus_" in c]
    if forbidden:
        raise ValueError(f"Target leakage in feature list: {forbidden}")

    work = featured.dropna(subset=[target_col]).copy()
    X = work[feature_cols].copy()
    y = work[target_col].copy()
    meta = work[["battery_id", "cycle_number"]].copy()
    return X, y, meta
