"""Shared Streamlit metric cards."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.components.formatting import fmt_capacity, fmt_cycle, fmt_rul, fmt_soh


def render_kpi_row(kpis: dict[str, Any]) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Cycle", kpis.get("current_cycle", "N/A"))
    c2.metric("Current Capacity", fmt_capacity(kpis.get("current_capacity")))
    c3.metric("Current SOH", fmt_soh(kpis.get("current_soh")))
    c4.metric("Estimated RUL", fmt_rul(kpis.get("estimated_rul")))
    c5.metric("Estimated EOL", fmt_cycle(kpis.get("estimated_eol_cycle")))

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Knee Cycle (experimental)", fmt_cycle(kpis.get("estimated_knee_cycle")))
    rate = kpis.get("degradation_rate")
    c7.metric("Degradation Rate", f"{rate:.5f} Ah/cycle" if rate is not None else "N/A")
    ir = kpis.get("latest_internal_resistance")
    c8.metric("Internal Resistance", f"{ir:.4f}" if ir is not None else "N/A")
    temp = kpis.get("average_temperature")
    c9.metric("Avg Temperature", f"{temp:.1f} °C" if temp is not None else "N/A")
