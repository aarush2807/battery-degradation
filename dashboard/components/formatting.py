"""Metric formatting helpers for Streamlit."""

from __future__ import annotations

from typing import Any, Optional


def fmt_soh(soh: Optional[float]) -> str:
    if soh is None:
        return "N/A"
    return f"{soh * 100:.2f}%"


def fmt_capacity(cap: Optional[float]) -> str:
    if cap is None:
        return "N/A"
    return f"{cap:.3f} Ah"


def fmt_rul(rul: Optional[float]) -> str:
    if rul is None:
        return "N/A"
    return f"{int(max(0, round(rul)))} cycles"


def fmt_cycle(cycle: Optional[float]) -> str:
    if cycle is None:
        return "N/A"
    return f"Cycle {int(round(cycle))}"


def fmt_rmse(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f} Ah"
