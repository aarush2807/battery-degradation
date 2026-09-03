"""Preprocessing: SOH, reference capacity, sorting, cleaning helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from battery_degradation.utils import get_logger, safe_div

logger = get_logger(__name__)


def sort_battery_cycles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)
    return out


def compute_reference_capacity(
    battery_df: pd.DataFrame,
    method: str = "first_n_average",
    n: int = 5,
    rated_capacity: Optional[float] = None,
) -> float:
    caps = battery_df["capacity_ah"].dropna()
    if caps.empty:
        raise ValueError("Cannot compute reference capacity: no capacity values.")
    if method == "rated":
        if rated_capacity is None or rated_capacity <= 0:
            raise ValueError("rated_capacity must be provided when method='rated'")
        return float(rated_capacity)
    if method == "first_cycle":
        return float(caps.iloc[0])
    # default first_n_average
    n = max(1, int(n))
    return float(caps.iloc[:n].mean())


def add_soh(
    df: pd.DataFrame,
    method: str = "first_n_average",
    n: int = 5,
    rated_capacity: Optional[float] = None,
    references: Optional[dict[str, float]] = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Add soh and reference_capacity columns per battery."""
    out = sort_battery_cycles(df)
    refs: dict[str, float] = dict(references or {})
    soh_parts: list[pd.Series] = []
    ref_parts: list[pd.Series] = []

    for battery_id, group in out.groupby("battery_id", sort=False):
        if battery_id not in refs:
            refs[str(battery_id)] = compute_reference_capacity(
                group, method=method, n=n, rated_capacity=rated_capacity
            )
        ref = refs[str(battery_id)]
        soh = group["capacity_ah"].astype(float) / ref
        soh_parts.append(soh)
        ref_parts.append(pd.Series(ref, index=group.index))

    out["soh"] = pd.concat(soh_parts).sort_index()
    out["reference_capacity"] = pd.concat(ref_parts).sort_index()
    out["soh_pct"] = out["soh"] * 100.0
    return out, refs


def basic_clean(df: pd.DataFrame, drop_invalid_capacity: bool = False) -> pd.DataFrame:
    """Light cleaning without silently deleting large amounts of data."""
    out = sort_battery_cycles(df)
    n_before = len(out)
    # Drop rows missing cycle or capacity (required for modeling)
    out = out.dropna(subset=["cycle_number", "capacity_ah"])
    if drop_invalid_capacity:
        invalid = out["capacity_ah"] <= 0
        if invalid.any():
            logger.warning("Dropping %s rows with non-positive capacity", int(invalid.sum()))
            out = out.loc[~invalid]
    dropped = n_before - len(out)
    if dropped:
        logger.info("Dropped %s rows missing cycle/capacity", dropped)
    return out.reset_index(drop=True)


def prepare_dataset(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    battery_cfg = config.get("battery", {})
    cleaned = basic_clean(df)
    prepared, refs = add_soh(
        cleaned,
        method=battery_cfg.get("reference_capacity_method", "first_n_average"),
        n=int(battery_cfg.get("reference_cycles", 5)),
        rated_capacity=battery_cfg.get("rated_capacity"),
    )
    return prepared, refs
