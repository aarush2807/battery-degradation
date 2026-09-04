"""Shared sidebar controls."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

from battery_degradation.config import load_config
from battery_degradation.data_loader import load_and_normalize, load_default_or_upload
from battery_degradation.pipeline import BatteryPredictor
from battery_degradation.preprocessing import prepare_dataset
from battery_degradation.utils import project_path


def init_session_state() -> None:
    defaults = {
        "config": load_config(),
        "raw_df": None,
        "prepared_df": None,
        "schema_result": None,
        "source_name": None,
        "predictor": None,
        "selected_battery": None,
        "column_overrides": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_predictor_if_available() -> Optional[BatteryPredictor]:
    models_dir = project_path("models")
    if (models_dir / "metadata.json").exists():
        try:
            pred = BatteryPredictor(config=st.session_state.config)
            pred.load(models_dir)
            if pred.prepared_ is None and st.session_state.get("prepared_df") is not None:
                pred.prepared_ = st.session_state.prepared_df
                # rebuild refs
                if "reference_capacity" in pred.prepared_.columns:
                    refs = (
                        pred.prepared_.groupby("battery_id")["reference_capacity"].first().astype(float).to_dict()
                    )
                    pred.reference_capacities_ = {str(k): float(v) for k, v in refs.items()}
            return pred
        except Exception as exc:
            st.warning(f"Could not load saved models: {exc}")
    return st.session_state.get("predictor")


def render_sidebar() -> dict[str, Any]:
    init_session_state()
    st.sidebar.title("Controls")
    uploaded = st.sidebar.file_uploader("Upload cycle CSV", type=["csv"])

    overrides = st.session_state.column_overrides
    with st.sidebar.expander("Column mapping overrides"):
        st.caption("Provide canonical → original column if aliases are ambiguous.")
        cycle_override = st.text_input("cycle_number column", value=overrides.get("cycle_number", ""))
        cap_override = st.text_input("capacity_ah column", value=overrides.get("capacity_ah", ""))
        if st.button("Apply mapping overrides"):
            new_over = dict(overrides)
            if cycle_override:
                new_over["cycle_number"] = cycle_override
            if cap_override:
                new_over["capacity_ah"] = cap_override
            st.session_state.column_overrides = new_over

    try:
        if uploaded is not None:
            raw_upload = pd.read_csv(uploaded)
            df, schema = load_and_normalize(raw_upload, column_overrides=st.session_state.column_overrides)
            source_name = uploaded.name
        else:
            df, schema, source_name = load_default_or_upload(
                column_overrides=st.session_state.column_overrides
            )
        prepared, refs = prepare_dataset(df, st.session_state.config)
        st.session_state.raw_df = df
        st.session_state.prepared_df = prepared
        st.session_state.schema_result = schema
        st.session_state.source_name = source_name
        st.session_state.reference_capacities = refs
    except Exception as exc:
        st.sidebar.error(f"Data load failed: {exc}")
        return {}

    st.sidebar.caption(f"Loaded: **{st.session_state.source_name}**")
    if schema.ambiguities:
        st.sidebar.warning(f"Ambiguous columns: {list(schema.ambiguities.keys())}")

    batteries = sorted(prepared["battery_id"].astype(str).unique())
    selected_battery = st.sidebar.selectbox("Battery", batteries, index=0)
    st.session_state.selected_battery = selected_battery

    bat_df = prepared[prepared["battery_id"] == selected_battery]
    cmin, cmax = int(bat_df["cycle_number"].min()), int(bat_df["cycle_number"].max())
    cycle_range = st.sidebar.slider("Cycle range", cmin, cmax, (cmin, cmax))

    target = st.sidebar.selectbox("Target variable", ["capacity", "soh"])
    horizon = st.sidebar.selectbox(
        "Forecast horizon (model)",
        st.session_state.config.get("forecasting", {}).get("horizons", [1, 5, 10, 25, 50, 100]),
    )
    forecast_cycles = st.sidebar.slider(
        "Forecast cycles",
        1,
        int(st.session_state.config.get("dashboard", {}).get("max_forecast_cycles", 500)),
        int(st.session_state.config.get("dashboard", {}).get("default_forecast_cycles", 100)),
    )
    ref_method = st.sidebar.selectbox(
        "Reference capacity method",
        ["first_n_average", "first_cycle", "rated"],
        index=0,
    )
    eol = st.sidebar.slider("EOL SOH threshold", 0.70, 0.90, float(st.session_state.config["battery"]["eol_soh"]), 0.01)
    st.session_state.config["battery"]["eol_soh"] = eol
    st.session_state.config["battery"]["reference_capacity_method"] = ref_method

    model_choice = st.sidebar.selectbox(
        "Model selection",
        ["Best saved model", "WeightedEnsemble", "RandomForest", "Ridge", "HistoricalLinear"],
    )
    smoothing = st.sidebar.checkbox("Smoothing", value=True)
    uncertainty = st.sidebar.checkbox(
        "Show uncertainty",
        value=bool(st.session_state.config.get("dashboard", {}).get("show_uncertainty", True)),
    )
    raw_toggle = st.sidebar.checkbox("Show raw processed table preview", value=False)
    smooth_window = st.sidebar.slider(
        "Smoothing window",
        1,
        50,
        int(st.session_state.config.get("dashboard", {}).get("default_smoothing_window", 5)),
    )

    predictor = load_predictor_if_available()
    # Always attach the in-memory prepared frame so forecast/KPIs work without
    # relying solely on a possibly stale prepared_path in metadata.json.
    if predictor is not None and prepared is not None:
        predictor.prepared_ = prepared
        predictor.reference_capacities_ = {
            str(k): float(v) for k, v in (refs or {}).items()
        }
    st.session_state.predictor = predictor

    if raw_toggle:
        st.sidebar.caption(f"Prepared shape: {prepared.shape[0]} × {prepared.shape[1]}")

    return {
        "prepared": prepared,
        "battery_id": selected_battery,
        "cycle_range": cycle_range,
        "target": target,
        "horizon": horizon,
        "forecast_cycles": forecast_cycles,
        "eol": eol,
        "model_choice": model_choice,
        "smoothing": smoothing,
        "uncertainty": uncertainty,
        "raw_toggle": raw_toggle,
        "smooth_window": smooth_window,
        "predictor": predictor,
        "schema": schema,
        "refs": refs,
    }
