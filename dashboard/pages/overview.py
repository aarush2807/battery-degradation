"""Overview page — executive battery health KPIs."""

from __future__ import annotations

import streamlit as st

from battery_degradation.features import build_features_for_battery
from battery_degradation.rul import detect_degradation_knee
from dashboard.components.charts import plot_capacity_history, plot_degradation_rate, plot_rul, plot_soh_history
from dashboard.components.metrics import render_kpi_row
from dashboard.components.tables import download_csv_button, show_dataframe


def render(ctx: dict) -> None:
    st.header("Overview")
    st.caption("Executive battery-health summary for the selected cell.")

    prepared = ctx.get("prepared")
    if prepared is None or prepared.empty:
        st.error("No data loaded.")
        return

    bid = ctx["battery_id"]
    c0, c1 = ctx["cycle_range"]
    hist = prepared[(prepared["battery_id"] == bid) & (prepared["cycle_number"] >= c0) & (prepared["cycle_number"] <= c1)]
    hist = hist.sort_values("cycle_number")

    predictor = ctx.get("predictor")
    if predictor is not None and predictor.models_by_horizon_:
        if predictor.prepared_ is None:
            predictor.prepared_ = prepared
            predictor.reference_capacities_ = ctx.get("refs", {})
        try:
            kpis = predictor.battery_kpis(bid)
        except Exception as exc:
            st.warning(f"KPI forecast unavailable: {exc}")
            kpis = _fallback_kpis(hist, ctx["eol"])
    else:
        st.info("Train models (Model Performance page or CLI) for full RUL/EOL KPIs.")
        kpis = _fallback_kpis(hist, ctx["eol"])

    render_kpi_row(kpis)

    knee = kpis.get("estimated_knee_cycle")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_soh_history(hist, eol_soh=ctx["eol"], knee_cycle=knee, color=None), use_container_width=True)
    with col2:
        st.plotly_chart(plot_capacity_history(hist, color=None), use_container_width=True)

    featured = build_features_for_battery(hist)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_degradation_rate(featured), use_container_width=True)
    with c4:
        if predictor is not None and predictor.models_by_horizon_:
            try:
                fc = predictor.forecast(battery_id=bid, future_cycles=min(100, ctx["forecast_cycles"]))
                if "estimated_rul" in fc.columns:
                    st.plotly_chart(plot_rul(fc), use_container_width=True)
            except Exception as exc:
                st.warning(str(exc))
        else:
            st.info("RUL trend available after training.")

    n_bat = prepared["battery_id"].nunique()
    if n_bat > 1:
        st.subheader("Multi-battery SOH comparison")
        st.plotly_chart(plot_soh_history(prepared, eol_soh=ctx["eol"]), use_container_width=True)

    if kpis.get("flags"):
        st.subheader("Engineering diagnostic indicators")
        for flag in kpis["flags"]:
            getattr(st, flag.get("severity", "info"), st.info)(flag["message"])

    summary = hist.tail(1)[["battery_id", "cycle_number", "capacity_ah", "soh"]]
    download_csv_button(summary, f"battery_{bid}_snapshot.csv", "Download battery snapshot CSV")


def _fallback_kpis(hist, eol):
    knee = detect_degradation_knee(hist["cycle_number"].to_numpy(), hist["soh"].to_numpy())
    return {
        "current_cycle": int(hist["cycle_number"].iloc[-1]),
        "current_capacity": float(hist["capacity_ah"].iloc[-1]),
        "current_soh": float(hist["soh"].iloc[-1]),
        "estimated_rul": None,
        "estimated_eol_cycle": None,
        "estimated_knee_cycle": knee.get("estimated_knee_cycle"),
        "degradation_rate": None,
        "flags": [],
    }
