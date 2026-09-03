"""Reusable Plotly chart components for the dashboard."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _base_layout(title: str, xlab: str, ylab: str) -> dict[str, Any]:
    return dict(
        title=title,
        xaxis_title=xlab,
        yaxis_title=ylab,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )


def plot_capacity_history(df: pd.DataFrame, color: str = "battery_id") -> go.Figure:
    fig = px.line(df, x="cycle_number", y="capacity_ah", color=color if color in df.columns else None)
    fig.update_layout(**_base_layout("Capacity vs Cycle", "Cycle", "Capacity (Ah)"))
    return fig


def plot_soh_history(
    df: pd.DataFrame,
    eol_soh: float = 0.80,
    knee_cycle: Optional[float] = None,
    color: str = "battery_id",
) -> go.Figure:
    plot_df = df.copy()
    plot_df["soh_pct"] = plot_df["soh"] * 100
    fig = px.line(plot_df, x="cycle_number", y="soh_pct", color=color if color in plot_df.columns else None)
    fig.add_hline(y=eol_soh * 100, line_dash="dot", line_color="#c0392b", annotation_text="EOL")
    if knee_cycle is not None:
        fig.add_vline(x=knee_cycle, line_dash="dash", line_color="#27ae60", annotation_text="Knee")
    fig.update_layout(**_base_layout("SOH vs Cycle", "Cycle", "SOH (%)"))
    return fig


def plot_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    eol_soh: float = 0.80,
    eol_cycle: Optional[float] = None,
    knee_cycle: Optional[float] = None,
    show_uncertainty: bool = True,
    show_smoothed: bool = False,
    show_monotonic: bool = False,
    y_col: str = "capacity",
) -> go.Figure:
    fig = go.Figure()
    if y_col == "soh":
        hist_y = history["soh"] * 100
        pred_y = forecast["predicted_soh"] * 100
        lower = forecast.get("lower_soh", forecast["predicted_soh"]) * 100
        upper = forecast.get("upper_soh", forecast["predicted_soh"]) * 100
        ylab = "SOH (%)"
        eol_y = eol_soh * 100
    else:
        hist_y = history["capacity_ah"]
        pred_y = forecast["predicted_capacity"]
        lower = forecast.get("lower_capacity", forecast["predicted_capacity"])
        upper = forecast.get("upper_capacity", forecast["predicted_capacity"])
        ylab = "Capacity (Ah)"
        eol_y = None

    fig.add_trace(
        go.Scatter(
            x=history["cycle_number"],
            y=hist_y,
            mode="lines",
            name="Historical",
            line=dict(color="#1f4e79", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["cycle_number"],
            y=pred_y,
            mode="lines",
            name="Forecast",
            line=dict(color="#c0392b", width=2, dash="dash"),
            customdata=np.column_stack(
                [
                    forecast.get("predicted_capacity", pred_y),
                    forecast.get("predicted_soh", pred_y),
                    forecast.get("lower_capacity", lower),
                    forecast.get("upper_capacity", upper),
                    forecast.get("estimated_rul", np.full(len(forecast), np.nan)),
                ]
            ),
            hovertemplate=(
                "Cycle %{x}<br>Pred Cap %{customdata[0]:.4f}<br>Pred SOH %{customdata[1]:.4f}"
                "<br>Lower %{customdata[2]:.4f}<br>Upper %{customdata[3]:.4f}"
                "<br>RUL %{customdata[4]:.0f}<extra></extra>"
            ),
        )
    )
    if show_uncertainty:
        fig.add_trace(
            go.Scatter(
                x=list(forecast["cycle_number"]) + list(forecast["cycle_number"][::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(192,57,43,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Uncertainty",
                hoverinfo="skip",
            )
        )
    if show_smoothed and "predicted_capacity_smoothed" in forecast.columns:
        y = forecast["predicted_capacity_smoothed"] if y_col == "capacity" else forecast["predicted_capacity_smoothed"] / max(
            history["reference_capacity"].iloc[-1], 1e-12
        ) * (100 if y_col == "soh" else 1)
        if y_col == "soh":
            y = forecast["predicted_capacity_smoothed"] / max(history["reference_capacity"].iloc[-1], 1e-12) * 100
        fig.add_trace(go.Scatter(x=forecast["cycle_number"], y=y, name="Smoothed", line=dict(color="#8e44ad")))
    if show_monotonic and "predicted_capacity_monotonic" in forecast.columns:
        y = forecast["predicted_capacity_monotonic"]
        if y_col == "soh":
            y = y / max(history["reference_capacity"].iloc[-1], 1e-12) * 100
        fig.add_trace(go.Scatter(x=forecast["cycle_number"], y=y, name="Monotonic trend", line=dict(color="#16a085")))
    if eol_y is not None:
        fig.add_hline(y=eol_y, line_dash="dot", line_color="#c0392b", annotation_text="EOL")
    if eol_cycle is not None:
        fig.add_vline(x=eol_cycle, line_dash="dot", line_color="#8e44ad", annotation_text="EOL cycle")
    if knee_cycle is not None:
        fig.add_vline(x=knee_cycle, line_dash="dash", line_color="#27ae60", annotation_text="Knee")
    fig.update_layout(**_base_layout("Degradation Forecast", "Cycle", ylab))
    return fig


def plot_rul(forecast: pd.DataFrame) -> go.Figure:
    fig = px.line(forecast, x="cycle_number", y="estimated_rul")
    fig.update_layout(**_base_layout("Estimated RUL", "Cycle", "RUL (cycles)"))
    return fig


def plot_degradation_rate(df: pd.DataFrame, slope_col: str = "capacity_slope_20") -> go.Figure:
    if slope_col not in df.columns:
        # compute simple diff
        tmp = df.sort_values("cycle_number").copy()
        tmp["slope"] = tmp["capacity_ah"].diff(10) / 10
        fig = px.line(tmp, x="cycle_number", y="slope")
    else:
        fig = px.line(df, x="cycle_number", y=slope_col)
    fig.update_layout(**_base_layout("Degradation Rate", "Cycle", "Capacity slope"))
    return fig


def plot_model_comparison(metrics_df: pd.DataFrame, metric: str = "rmse", split: str = "validation") -> go.Figure:
    sub = metrics_df[metrics_df["split"] == split]
    fig = px.bar(sub, x="model", y=metric, color="horizon", barmode="group")
    fig.update_layout(**_base_layout(f"{metric.upper()} by Model", "Model", metric.upper()))
    return fig


def plot_horizon_performance(metrics_df: pd.DataFrame, metric: str = "rmse", split: str = "validation") -> go.Figure:
    sub = metrics_df[metrics_df["split"] == split]
    fig = px.line(sub, x="horizon", y=metric, color="model", markers=True)
    fig.update_layout(**_base_layout(f"{metric.upper()} vs Horizon", "Horizon", metric.upper()))
    return fig


def plot_actual_vs_predicted(y_true, y_pred) -> go.Figure:
    df = pd.DataFrame({"actual": y_true, "predicted": y_pred})
    fig = px.scatter(df, x="actual", y="predicted", opacity=0.5)
    lims = [df.min().min(), df.max().max()]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="y=x", line=dict(dash="dash", color="#c0392b")))
    fig.update_layout(**_base_layout("Actual vs Predicted", "Actual", "Predicted"))
    return fig


def plot_residuals(cycles, y_true, y_pred) -> go.Figure:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig = px.scatter(x=cycles, y=resid)
    fig.add_hline(y=0, line_dash="dash", line_color="#c0392b")
    fig.update_layout(**_base_layout("Residuals vs Cycle", "Cycle", "Residual"))
    return fig


def plot_feature_importance(importance: dict[str, float], top_n: int = 20) -> go.Figure:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    df = pd.DataFrame(items, columns=["feature", "importance"])
    fig = px.bar(df.iloc[::-1], x="importance", y="feature", orientation="h")
    fig.update_layout(**_base_layout("Feature Importance", "Importance", "Feature"))
    return fig


def plot_temperature_history(df: pd.DataFrame) -> go.Figure:
    fig = px.line(df, x="cycle_number", y="temperature_mean", color="battery_id" if "battery_id" in df.columns else None)
    fig.update_layout(**_base_layout("Temperature vs Cycle", "Cycle", "Temperature"))
    return fig


def plot_resistance_history(df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        df, x="cycle_number", y="internal_resistance", color="battery_id" if "battery_id" in df.columns else None
    )
    fig.update_layout(**_base_layout("Internal Resistance vs Cycle", "Cycle", "Resistance"))
    return fig


def plot_ensemble_weights(weights: dict[str, float], title: str = "Ensemble Weights") -> go.Figure:
    df = pd.DataFrame({"model": list(weights.keys()), "weight": list(weights.values())})
    fig = px.bar(df, x="model", y="weight")
    fig.update_layout(**_base_layout(title, "Model", "Weight"))
    return fig


def plot_missing_values(missing: dict[str, int]) -> go.Figure:
    df = pd.DataFrame({"column": list(missing.keys()), "missing": list(missing.values())})
    fig = px.bar(df, x="column", y="missing")
    fig.update_layout(**_base_layout("Missing Values", "Column", "Count"))
    return fig
