"""Interactive battery explorer page."""

from __future__ import annotations

import streamlit as st

from battery_degradation.diagnostics import battery_summary_table
from battery_degradation.features import build_features_for_battery
from dashboard.components.charts import (
    plot_capacity_history,
    plot_degradation_rate,
    plot_resistance_history,
    plot_soh_history,
    plot_temperature_history,
)
from dashboard.components.tables import download_csv_button, show_dataframe
import plotly.express as px


def render(ctx: dict) -> None:
    st.header("Battery Explorer")
    prepared = ctx.get("prepared")
    if prepared is None:
        st.error("No data loaded.")
        return

    batteries = st.multiselect(
        "Compare batteries",
        sorted(prepared["battery_id"].astype(str).unique()),
        default=[ctx["battery_id"]],
    )
    if not batteries:
        st.warning("Select at least one battery.")
        return

    data = prepared[prepared["battery_id"].isin(batteries)].copy()
    c0, c1 = ctx["cycle_range"]
    # For multi compare, use full range unless single
    if len(batteries) == 1:
        data = data[(data["cycle_number"] >= c0) & (data["cycle_number"] <= c1)]

    numeric_cols = [c for c in data.columns if c not in ("battery_id", "timestamp") and data[c].dtype != "O"]
    y_var = st.selectbox(
        "Y variable",
        [
            "capacity_ah",
            "soh",
            "temperature_mean",
            "internal_resistance",
            "voltage_mean",
            "energy_wh",
            "coulombic_efficiency",
        ],
        index=0,
    )
    rolling = st.slider("Rolling window (display)", 1, 50, 5)
    use_smooth = st.checkbox("Show smoothed values", value=False)

    plot_df = data.copy()
    if use_smooth and y_var in plot_df.columns:
        plot_df[y_var] = plot_df.groupby("battery_id")[y_var].transform(
            lambda s: s.rolling(rolling, min_periods=1).mean()
        )

    if y_var == "soh":
        st.plotly_chart(plot_soh_history(plot_df, eol_soh=ctx["eol"]), use_container_width=True)
    elif y_var == "capacity_ah":
        st.plotly_chart(plot_capacity_history(plot_df), use_container_width=True)
    elif y_var == "temperature_mean" and "temperature_mean" in plot_df.columns:
        st.plotly_chart(plot_temperature_history(plot_df), use_container_width=True)
    elif y_var == "internal_resistance" and "internal_resistance" in plot_df.columns:
        st.plotly_chart(plot_resistance_history(plot_df), use_container_width=True)
    elif y_var in plot_df.columns:
        fig = px.line(plot_df, x="cycle_number", y=y_var, color="battery_id")
        fig.update_layout(template="plotly_white", title=f"{y_var} vs Cycle")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Column '{y_var}' not available in this dataset.")

    # Degradation rate for primary battery
    primary = data[data["battery_id"] == batteries[0]]
    featured = build_features_for_battery(primary)
    st.plotly_chart(plot_degradation_rate(featured), use_container_width=True)

    st.subheader("Battery summary table")
    summary = battery_summary_table(prepared[prepared["battery_id"].isin(batteries)])
    show_dataframe(summary)
    download_csv_button(summary, "battery_summary.csv", "Download battery summary CSV")
