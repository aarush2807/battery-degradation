"""Table helpers."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def show_dataframe(df: pd.DataFrame, height: int = 360) -> None:
    st.dataframe(df, use_container_width=True, height=height)


def download_csv_button(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )
