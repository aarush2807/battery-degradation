"""Forecasting page with interactive Plotly forecast."""

from __future__ import annotations

import streamlit as st

from battery_degradation.rul import detect_degradation_knee
from dashboard.components.charts import plot_forecast
from dashboard.components.formatting import fmt_capacity, fmt_cycle, fmt_rul, fmt_soh
from dashboard.components.tables import download_csv_button


def render(ctx: dict) -> None:
    st.header("Forecasting")
    st.caption("Direct / recursive / hybrid trajectory forecasts with uncertainty and EOL markers.")

    prepared = ctx.get("prepared")
    predictor = ctx.get("predictor")
    if prepared is None:
        st.error("No data loaded.")
        return
    if predictor is None or not predictor.models_by_horizon_:
        st.warning("No trained models found. Run training from CLI or the Model Performance page.")
        st.code("python main.py train --data data/raw/synthetic_battery_cycles.csv")
        return

    if predictor.prepared_ is None:
        predictor.prepared_ = prepared
        predictor.reference_capacities_ = ctx.get("refs", predictor.reference_capacities_)

    bid = ctx["battery_id"]
    hist = prepared[prepared["battery_id"] == bid].sort_values("cycle_number")
    if len(hist) < 20:
        st.warning("At least 20 historical cycles are required to generate this forecast.")
        return

    c1, c2, c3, c4 = st.columns(4)
    method = c1.radio("Forecast method", ["direct", "recursive", "hybrid"], horizontal=True)
    pred_type = c2.radio("Prediction type", ["capacity", "soh"], horizontal=True)
    start_cycle = c3.number_input(
        "Starting cycle (history end)",
        min_value=int(hist["cycle_number"].min()),
        max_value=int(hist["cycle_number"].max()),
        value=int(hist["cycle_number"].max()),
    )
    future_cycles = c4.slider("Forecast length", 1, 500, int(ctx["forecast_cycles"]))

    show_raw = st.checkbox("Display raw model prediction", value=True)
    show_smooth = st.checkbox("Display smoothed prediction", value=ctx.get("smoothing", True))
    show_mono = st.checkbox("Display monotonic trend", value=False)
    show_unc = st.checkbox("Uncertainty band", value=ctx.get("uncertainty", True))

    st.subheader("Scenario analysis (experimental)")
    st.caption("Scenario-based model forecasts — not guaranteed outcomes.")
    use_scenario = st.checkbox("Enable temperature scenario comparison")
    temp_a = temp_b = None
    if use_scenario and "temperature_mean" in hist.columns:
        col_a, col_b = st.columns(2)
        temp_a = col_a.number_input("Scenario A temperature (°C)", value=30.0)
        temp_b = col_b.number_input("Scenario B temperature (°C)", value=40.0)

    try:
        with st.spinner("Generating forecast..."):
            forecast = predictor.forecast(
                battery_id=bid,
                future_cycles=future_cycles,
                method=method,
                start_cycle=int(start_cycle),
                smoothing_window=int(ctx.get("smooth_window", 5)),
                apply_monotonic=show_mono,
                show_uncertainty=show_unc,
                future_profile={"temperature_mean": temp_a} if (use_scenario and temp_a is not None) else None,
            )
    except Exception as exc:
        st.error(str(exc))
        return

    hist_use = hist[hist["cycle_number"] <= start_cycle]
    knee = detect_degradation_knee(hist_use["cycle_number"].to_numpy(), hist_use["soh"].to_numpy())
    eol = forecast.attrs.get("eol", {})

    # Summary cards
    cur_soh = float(hist_use["soh"].iloc[-1])
    def soh_at(n):
        row = forecast[forecast["horizon_step"] == n]
        if len(row):
            return float(row["predicted_soh"].iloc[0])
        if len(forecast) >= n:
            return float(forecast.iloc[n - 1]["predicted_soh"])
        return None

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Current SOH", fmt_soh(cur_soh))
    m2.metric("SOH @ +25", fmt_soh(soh_at(25)))
    m3.metric("SOH @ +50", fmt_soh(soh_at(50)))
    m4.metric("SOH @ +100", fmt_soh(soh_at(100)))
    m5.metric("Estimated EOL", fmt_cycle(eol.get("eol_cycle")))
    m6.metric("Estimated RUL", fmt_rul(eol.get("rul")))

    fig = plot_forecast(
        hist_use,
        forecast,
        eol_soh=ctx["eol"],
        eol_cycle=eol.get("eol_cycle"),
        knee_cycle=knee.get("estimated_knee_cycle"),
        show_uncertainty=show_unc,
        show_smoothed=show_smooth,
        show_monotonic=show_mono,
        y_col=pred_type,
    )
    st.plotly_chart(fig, use_container_width=True)

    if use_scenario and temp_b is not None:
        fc_b = predictor.forecast(
            battery_id=bid,
            future_cycles=future_cycles,
            method=method,
            start_cycle=int(start_cycle),
            future_profile={"temperature_mean": temp_b},
            show_uncertainty=False,
        )
        st.write(f"Scenario B ({temp_b}°C) EOL: {fc_b.attrs.get('eol', {}).get('eol_cycle')}")

    download_csv_button(forecast, f"forecast_{bid}.csv", "Download forecast CSV")
