"""Streamlit entrypoint — Battery Degradation Analytics & Forecasting.

Page modules live in ``dashboard/views/`` (not ``pages/``) so Streamlit does not
auto-register them as blank multipage routes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.sidebar import render_sidebar
from dashboard.views import (
    battery_explorer,
    data_quality,
    diagnostics,
    forecasting,
    model_performance,
    overview,
)

st.set_page_config(
    page_title="Battery Degradation Analytics",
    page_icon="🔋",
    layout="wide",
)

st.title("Battery Degradation Analytics & Forecasting")
st.markdown(
    "Machine-learning-based battery health, degradation, and remaining useful life analysis."
)
st.caption(
    "Predictions depend on training data quality and operating conditions. "
    "Long-range forecasts have greater uncertainty. Knee detection is experimental. "
    "Uncertainty bands are approximate, not calibrated confidence intervals."
)

ctx = render_sidebar()
if not ctx:
    st.error(
        "No dataset loaded. Upload a CSV in the sidebar, or ensure "
        "`data/raw/synthetic_battery_cycles.csv` exists "
        "(run `python scripts/generate_synthetic_data.py`)."
    )
    st.stop()

prepared = ctx.get("prepared")
if prepared is not None and not prepared.empty:
    st.sidebar.success(
        f"{len(prepared):,} cycles · {prepared['battery_id'].nunique()} batteries · "
        f"{ctx.get('battery_id', '')}"
    )

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Battery Explorer",
        "Forecasting",
        "Model Performance",
        "Diagnostics",
        "Data Quality",
    ],
)

PAGE_RENDERERS = {
    "Overview": overview.render,
    "Battery Explorer": battery_explorer.render,
    "Forecasting": forecasting.render,
    "Model Performance": model_performance.render,
    "Diagnostics": diagnostics.render,
    "Data Quality": data_quality.render,
}

try:
    PAGE_RENDERERS[page](ctx)
except Exception as exc:
    st.error(f"This page failed to render: {exc}")
    st.exception(exc)
