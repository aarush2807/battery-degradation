"""Diagnostics page — engineering indicators and explainability."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from battery_degradation.diagnostics import compute_degradation_metrics, engineering_flags
from battery_degradation.features import build_features_for_battery
from battery_degradation.rul import detect_degradation_knee
from dashboard.components.charts import plot_feature_importance, plot_degradation_rate


def render(ctx: dict) -> None:
    st.header("Diagnostics")
    st.caption("Engineering diagnostic indicators — heuristic flags for inspection, not failure claims.")

    prepared = ctx.get("prepared")
    if prepared is None:
        st.error("No data loaded.")
        return

    bid = ctx["battery_id"]
    hist = prepared[prepared["battery_id"] == bid].sort_values("cycle_number")
    featured = build_features_for_battery(hist)
    diag = compute_degradation_metrics(hist)
    knee = detect_degradation_knee(hist["cycle_number"].to_numpy(), hist["soh"].to_numpy())

    flags = engineering_flags(hist, eol_soh=ctx["eol"])
    st.subheader("Engineering Flags")
    if not flags:
        st.success("No heuristic flags triggered.")
    for f in flags:
        getattr(st, f.get("severity", "info"), st.info)(f["message"] + " _(Engineering diagnostic indicator)_")

    st.info(
        f"Experimental degradation knee estimate: "
        f"{knee.get('estimated_knee_cycle')} (confidence={knee.get('knee_confidence')})"
    )

    # Dual axis capacity + resistance
    if "internal_resistance" in hist.columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=hist["cycle_number"], y=hist["capacity_ah"], name="Capacity"),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=hist["cycle_number"], y=hist["internal_resistance"], name="Resistance"),
            secondary_y=True,
        )
        fig.update_layout(title="Capacity & Resistance", template="plotly_white")
        fig.update_yaxes(title_text="Capacity (Ah)", secondary_y=False)
        fig.update_yaxes(title_text="Resistance", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_degradation_rate(featured), use_container_width=True)
    with c2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=diag["cycle_number"],
                y=diag["diag_degradation_acceleration"],
                name="Acceleration",
            )
        )
        if knee.get("estimated_knee_cycle"):
            fig2.add_vline(x=knee["estimated_knee_cycle"], line_dash="dash", annotation_text="Knee")
        fig2.update_layout(title="SOH/Capacity acceleration (heuristic)", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    if "temperature_mean" in hist.columns:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=hist["temperature_mean"], y=hist["soh"], mode="markers", name="SOH vs Temp"))
        fig3.update_layout(title="Temperature vs SOH", template="plotly_white", xaxis_title="Temp", yaxis_title="SOH")
        st.plotly_chart(fig3, use_container_width=True)

    predictor = ctx.get("predictor")
    if predictor and predictor.feature_importances_:
        h = int(ctx.get("horizon", 1))
        imp = predictor.feature_importances_.get(h) or next(iter(predictor.feature_importances_.values()))
        st.plotly_chart(plot_feature_importance(imp), use_container_width=True)

    if predictor and predictor.ensemble_weights_:
        st.subheader("Ensemble component weights")
        st.json({str(k): v for k, v in predictor.ensemble_weights_.items()})
