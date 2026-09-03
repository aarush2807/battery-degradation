"""Data quality checks and engineering diagnostics (heuristics)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from battery_degradation.utils import get_logger

logger = get_logger(__name__)


def data_quality_report(df: pd.DataFrame, min_cycles: int = 20) -> dict[str, Any]:
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "warnings": [],
        "errors": [],
    }
    if "battery_id" in df.columns:
        batteries = df["battery_id"].nunique()
        report["n_batteries"] = int(batteries)
        cycle_counts = df.groupby("battery_id")["cycle_number"].agg(["min", "max", "count"])
        report["cycle_ranges"] = cycle_counts.reset_index().to_dict(orient="records")
        short = cycle_counts[cycle_counts["count"] < min_cycles]
        if len(short):
            report["warnings"].append(
                f"{len(short)} batteries have fewer than {min_cycles} cycles."
            )
            report["insufficient_history_batteries"] = short.index.astype(str).tolist()
    else:
        report["n_batteries"] = 1

    missing = df.isna().sum()
    report["missing_values"] = missing[missing > 0].to_dict()

    if "cycle_number" in df.columns and "battery_id" in df.columns:
        dup = df.duplicated(subset=["battery_id", "cycle_number"], keep=False)
        report["duplicate_cycle_rows"] = int(dup.sum())
        if dup.any():
            report["warnings"].append(
                f"Found {int(dup.sum())} rows with duplicate battery_id/cycle_number."
            )

        nonmono = []
        for bid, g in df.groupby("battery_id"):
            cycles = g["cycle_number"].to_numpy()
            if len(cycles) > 1 and np.any(np.diff(cycles) < 0):
                nonmono.append(str(bid))
        report["non_monotonic_batteries"] = nonmono
        if nonmono:
            report["warnings"].append(
                f"Non-monotonic cycle numbers for batteries: {nonmono}"
            )

    if "capacity_ah" in df.columns:
        neg = (df["capacity_ah"] < 0).sum()
        report["negative_capacity_rows"] = int(neg)
        if neg:
            report["errors"].append(f"{neg} rows have negative capacity.")
        infs = np.isinf(pd.to_numeric(df["capacity_ah"], errors="coerce")).sum()
        if infs:
            report["errors"].append(f"{infs} infinite capacity values.")

    if "soh" in df.columns:
        weird = ((df["soh"] < 0) | (df["soh"] > 1.2)).sum()
        if weird:
            report["warnings"].append(f"{weird} rows have SOH outside [0, 1.2].")

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        report["invalid_timestamps"] = int(ts.isna().sum())
        if "battery_id" in df.columns:
            ts_dup = df.assign(_ts=ts).duplicated(subset=["battery_id", "_ts"], keep=False)
            report["duplicate_timestamps"] = int(ts_dup.sum())

    return report


def compute_degradation_metrics(df: pd.DataFrame, slope_window: int = 20) -> pd.DataFrame:
    """Add rolling degradation velocity/acceleration columns for diagnostics."""
    out = df.sort_values(["battery_id", "cycle_number"]).copy()
    vel_list = []
    acc_list = []
    for _, g in out.groupby("battery_id", sort=False):
        cap = g["capacity_ah"].astype(float)
        # simple rolling slope proxy via diff / window
        slope = cap.diff(slope_window) / float(slope_window)
        vel_list.append(slope)
        acc_list.append(slope.diff(slope_window))
    out["diag_degradation_velocity"] = pd.concat(vel_list).sort_index()
    out["diag_degradation_acceleration"] = pd.concat(acc_list).sort_index()
    return out


def engineering_flags(
    battery_df: pd.DataFrame,
    eol_soh: float = 0.80,
    high_temp_threshold: float = 40.0,
    uncertainty_width: Optional[float] = None,
    min_history: int = 20,
) -> list[dict[str, str]]:
    """
    Rule-based engineering diagnostic indicators (not failure claims).
    """
    flags: list[dict[str, str]] = []
    df = battery_df.sort_values("cycle_number")
    if len(df) < min_history:
        flags.append(
            {
                "code": "insufficient_history",
                "severity": "warning",
                "message": f"Insufficient historical data (<{min_history} cycles).",
            }
        )
        return flags

    soh = df["soh"].astype(float)
    current_soh = float(soh.iloc[-1])
    if current_soh <= eol_soh + 0.05:
        flags.append(
            {
                "code": "approaching_eol",
                "severity": "warning",
                "message": f"Approaching EOL (current SOH={current_soh:.3f}, threshold={eol_soh}).",
            }
        )

    # Accelerated degradation: recent slope steeper than early slope
    mid = len(df) // 2
    early_slope = (soh.iloc[mid] - soh.iloc[0]) / max(1, mid)
    late_slope = (soh.iloc[-1] - soh.iloc[mid]) / max(1, len(df) - mid)
    if late_slope < early_slope * 1.5 and late_slope < -1e-5:
        flags.append(
            {
                "code": "accelerated_degradation",
                "severity": "info",
                "message": "Accelerated degradation detected (heuristic late-life slope).",
            }
        )

    if "internal_resistance" in df.columns:
        ir = pd.to_numeric(df["internal_resistance"], errors="coerce").dropna()
        if len(ir) >= 10:
            change = (ir.iloc[-1] - ir.iloc[0]) / max(abs(ir.iloc[0]), 1e-9)
            if change > 0.2:
                flags.append(
                    {
                        "code": "elevated_resistance_growth",
                        "severity": "info",
                        "message": f"Elevated resistance growth (~{change*100:.1f}% from start).",
                    }
                )

    if "temperature_mean" in df.columns:
        temp = pd.to_numeric(df["temperature_mean"], errors="coerce")
        high_frac = float((temp > high_temp_threshold).mean())
        if high_frac > 0.2:
            flags.append(
                {
                    "code": "high_temperature_exposure",
                    "severity": "info",
                    "message": f"High-temperature exposure in {high_frac*100:.1f}% of cycles.",
                }
            )

    # Unexpected capacity recovery
    if len(soh) >= 10:
        recent = soh.iloc[-10:]
        if recent.iloc[-1] - recent.iloc[0] > 0.02:
            flags.append(
                {
                    "code": "unexpected_capacity_recovery",
                    "severity": "info",
                    "message": "Unexpected capacity recovery in recent cycles (may be noise/rest).",
                }
            )

    if uncertainty_width is not None and uncertainty_width > 0.05:
        flags.append(
            {
                "code": "high_model_uncertainty",
                "severity": "warning",
                "message": "High model uncertainty on latest forecast.",
            }
        )

    for f in flags:
        f["label"] = "Engineering diagnostic indicator"
    return flags


def battery_summary_table(
    df: pd.DataFrame,
    eol_estimates: Optional[dict[str, dict[str, Any]]] = None,
) -> pd.DataFrame:
    rows = []
    eol_estimates = eol_estimates or {}
    for bid, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_number")
        ref = float(g["reference_capacity"].iloc[0]) if "reference_capacity" in g else float("nan")
        cur_cap = float(g["capacity_ah"].iloc[-1])
        cur_soh = float(g["soh"].iloc[-1]) if "soh" in g else cur_cap / ref
        fade = (1.0 - cur_soh) * 100.0
        est = eol_estimates.get(str(bid), {})
        row = {
            "battery_id": bid,
            "current_cycle": int(g["cycle_number"].iloc[-1]),
            "reference_capacity": ref,
            "current_capacity": cur_cap,
            "current_soh": cur_soh,
            "capacity_fade_pct": fade,
            "estimated_rul": est.get("rul"),
            "estimated_eol_cycle": est.get("eol_cycle"),
            "estimated_knee_cycle": est.get("knee_cycle"),
        }
        if "temperature_mean" in g.columns:
            row["mean_temperature"] = float(pd.to_numeric(g["temperature_mean"], errors="coerce").mean())
        if "internal_resistance" in g.columns:
            ir = pd.to_numeric(g["internal_resistance"], errors="coerce").dropna()
            if len(ir) >= 2:
                row["resistance_change_pct"] = float((ir.iloc[-1] - ir.iloc[0]) / max(abs(ir.iloc[0]), 1e-9) * 100)
        rows.append(row)
    return pd.DataFrame(rows)
