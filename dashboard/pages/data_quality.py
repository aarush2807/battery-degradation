"""Data quality dashboard page."""

from __future__ import annotations

import streamlit as st

from battery_degradation.diagnostics import data_quality_report
from dashboard.components.charts import plot_missing_values
from dashboard.components.tables import download_csv_button, show_dataframe
import pandas as pd
import plotly.express as px


def render(ctx: dict) -> None:
    st.header("Data Quality")
    prepared = ctx.get("prepared")
    if prepared is None:
        st.error("No data loaded.")
        return

    report = data_quality_report(prepared)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", report["n_rows"])
    c2.metric("Columns", report["n_columns"])
    c3.metric("Batteries", report.get("n_batteries", 1))
    c4.metric("Duplicate cycle rows", report.get("duplicate_cycle_rows", 0))

    for err in report.get("errors", []):
        st.error(err)
    for warn in report.get("warnings", []):
        st.warning(warn)
    if not report.get("errors") and not report.get("warnings"):
        st.success("No critical data-quality issues detected.")

    missing = report.get("missing_values") or {}
    if missing:
        st.subheader("Missing values")
        st.plotly_chart(plot_missing_values(missing), use_container_width=True)
    else:
        st.info("No missing values in prepared dataset.")

    if report.get("cycle_ranges"):
        ranges = pd.DataFrame(report["cycle_ranges"])
        st.subheader("Cycle-count distribution")
        fig = px.bar(ranges, x="battery_id", y="count", title="Cycles per battery")
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Battery-level summary")
        show_dataframe(ranges)
        download_csv_button(ranges, "cycle_ranges.csv", "Download cycle-range CSV")

    schema = ctx.get("schema")
    if schema is not None:
        with st.expander("Schema mapping"):
            st.json(
                {
                    "mapping": schema.mapping,
                    "ambiguities": schema.ambiguities,
                    "unmapped": schema.unmapped,
                    "warnings": schema.warnings,
                }
            )

    if ctx.get("raw_toggle"):
        st.subheader("Prepared data preview")
        show_dataframe(prepared.head(200))
        download_csv_button(prepared, "processed_features_preview.csv", "Download prepared data CSV")
