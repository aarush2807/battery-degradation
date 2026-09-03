"""RUL / EOL estimation and experimental knee detection."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from battery_degradation.utils import clamp, get_logger

logger = get_logger(__name__)


def estimate_eol_from_trajectory(
    cycles: np.ndarray,
    soh: np.ndarray,
    eol_soh: float = 0.80,
    current_cycle: Optional[float] = None,
) -> dict[str, Any]:
    """
    Find first cycle where SOH crosses eol_soh.
    If never crosses, linearly extrapolate from recent slope.
    """
    cycles = np.asarray(cycles, dtype=float)
    soh = np.asarray(soh, dtype=float)
    mask = np.isfinite(cycles) & np.isfinite(soh)
    cycles, soh = cycles[mask], soh[mask]
    if len(cycles) == 0:
        return {"eol_cycle": None, "rul": 0.0, "crossed": False}

    if current_cycle is None:
        current_cycle = float(cycles[-1])

    below = np.where(soh <= eol_soh)[0]
    if len(below):
        eol_cycle = float(cycles[below[0]])
        rul = max(0.0, eol_cycle - float(current_cycle))
        return {"eol_cycle": eol_cycle, "rul": rul, "crossed": True}

    # Extrapolate using last up to 50 points
    n = min(50, len(cycles))
    x = cycles[-n:]
    y = soh[-n:]
    if len(x) < 2:
        return {"eol_cycle": None, "rul": 0.0, "crossed": False}
    coef = np.polyfit(x, y, 1)
    slope, intercept = float(coef[0]), float(coef[1])
    if slope >= -1e-12:
        # flat or increasing — cannot reliably estimate EOL
        return {"eol_cycle": None, "rul": None, "crossed": False, "note": "non_decreasing_trend"}
    eol_cycle = (eol_soh - intercept) / slope
    if eol_cycle < current_cycle:
        eol_cycle = float(current_cycle)
    rul = max(0.0, eol_cycle - float(current_cycle))
    return {"eol_cycle": float(eol_cycle), "rul": float(rul), "crossed": False}


def rul_at_cycle(estimated_eol_cycle: Optional[float], current_cycle: float) -> float:
    if estimated_eol_cycle is None:
        return 0.0
    return max(0.0, float(estimated_eol_cycle) - float(current_cycle))


def detect_degradation_knee(
    cycles: np.ndarray,
    soh: np.ndarray,
    smoothing_window: int = 15,
    slope_window: int = 20,
    acceleration_threshold: float = 1.5e-5,
    persistence_cycles: int = 10,
) -> dict[str, Any]:
    """
    Experimental degradation knee estimate.

    Method: smooth SOH, compute local slope, detect persistent acceleration
    (slope becoming more negative). Not scientifically validated.
    """
    cycles = np.asarray(cycles, dtype=float)
    soh = np.asarray(soh, dtype=float)
    if len(cycles) < max(smoothing_window, slope_window) + persistence_cycles:
        return {
            "estimated_knee_cycle": None,
            "knee_confidence": 0.0,
            "heuristic_score": 0.0,
            "label": "Experimental degradation knee estimate",
        }

    s = pd.Series(soh).rolling(smoothing_window, min_periods=max(3, smoothing_window // 2), center=True).mean()
    s = s.bfill().ffill()
    slope = s.diff(slope_window) / float(slope_window)
    accel = slope.diff(slope_window)  # more negative => acceleration of fade

    # Flag where acceleration is strongly negative (capacity fading faster)
    flag = accel < -abs(acceleration_threshold)
    knee_idx = None
    run = 0
    for i, f in enumerate(flag.fillna(False).to_numpy()):
        if i < len(cycles) // 3:
            # ignore very early life
            run = 0
            continue
        if f:
            run += 1
            if run >= persistence_cycles:
                knee_idx = i - persistence_cycles + 1
                break
        else:
            run = 0

    if knee_idx is None:
        # fallback: argmin of acceleration in latter half
        latter = accel.iloc[len(accel) // 2 :]
        if latter.notna().any():
            knee_idx = int(latter.idxmin())
            conf = 0.3
        else:
            return {
                "estimated_knee_cycle": None,
                "knee_confidence": 0.0,
                "heuristic_score": 0.0,
                "label": "Experimental degradation knee estimate",
            }
    else:
        conf = 0.6

    score = float(abs(accel.iloc[knee_idx])) if np.isfinite(accel.iloc[knee_idx]) else 0.0
    return {
        "estimated_knee_cycle": float(cycles[knee_idx]),
        "knee_confidence": conf,
        "heuristic_score": score,
        "label": "Experimental degradation knee estimate",
    }
