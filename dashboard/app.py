"""Streamlit entrypoint — Battery Degradation Analytics & Forecasting."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.components.sidebar import render_sidebar
from dashboard.pages import (
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
    st.stop()

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

if page == "Overview":
    overview.render(ctx)
elif page == "Battery Explorer":
    battery_explorer.render(ctx)
elif page == "Forecasting":
    forecasting.render(ctx)
elif page == "Model Performance":
    model_performance.render(ctx)
elif page == "Diagnostics":
    diagnostics.render(ctx)
elif page == "Data Quality":
    data_quality.render(ctx)
