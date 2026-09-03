"""Matplotlib static visualizations for reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_degradation.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def _save(fig, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", path)
    return path


def plot_capacity_history(
    df: pd.DataFrame,
    path: Path,
    pred: Optional[pd.DataFrame] = None,
    battery_id: Optional[str] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = df if battery_id is None else df[df["battery_id"] == battery_id]
    ax.plot(data["cycle_number"], data["capacity_ah"], label="Actual", color="#1f4e79")
    if pred is not None and len(pred):
        ax.plot(pred["cycle_number"], pred["predicted_capacity"], "--", label="Predicted", color="#c0392b")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Capacity vs Cycle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_soh_history(
    df: pd.DataFrame,
    path: Path,
    eol_soh: float = 0.80,
    battery_id: Optional[str] = None,
    knee_cycle: Optional[float] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = df if battery_id is None else df[df["battery_id"] == battery_id]
    ax.plot(data["cycle_number"], data["soh"] * 100, label="SOH", color="#1f4e79")
    ax.axhline(eol_soh * 100, color="#c0392b", linestyle=":", label=f"EOL {eol_soh*100:.0f}%")
    if knee_cycle is not None:
        ax.axvline(knee_cycle, color="#27ae60", linestyle="--", label="Knee (experimental)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH (%)")
    ax.set_title("SOH vs Cycle")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    path: Path,
    eol_soh: float = 0.80,
    eol_cycle: Optional[float] = None,
    knee_cycle: Optional[float] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["cycle_number"], history["capacity_ah"], "-", color="#1f4e79", label="Historical")
    ax.plot(
        forecast["cycle_number"],
        forecast["predicted_capacity"],
        "--",
        color="#c0392b",
        label="Forecast",
    )
    if {"lower_capacity", "upper_capacity"}.issubset(forecast.columns):
        ax.fill_between(
            forecast["cycle_number"],
            forecast["lower_capacity"],
            forecast["upper_capacity"],
            color="#c0392b",
            alpha=0.2,
            label="Uncertainty",
        )
    if eol_cycle is not None:
        ax.axvline(eol_cycle, color="#8e44ad", linestyle=":", label="Est. EOL")
    if knee_cycle is not None:
        ax.axvline(knee_cycle, color="#27ae60", linestyle="--", label="Knee")
    ax.set_title("Future Capacity Forecast")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_model_comparison(metrics_df: pd.DataFrame, path: Path, metric: str = "rmse") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    sub = metrics_df[metrics_df["split"] == "validation"].copy()
    if sub.empty:
        sub = metrics_df.copy()
    pivot = sub.pivot_table(index="model", values=metric, aggfunc="mean").sort_values(metric)
    pivot.plot(kind="barh", ax=ax, legend=False, color="#1f4e79")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Comparison ({metric.upper()})")
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, path)


def plot_residuals(cycles, y_true, y_pred, path: Path) -> Path:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(cycles, resid, s=12, alpha=0.6, color="#1f4e79")
    ax.axhline(0, color="#c0392b", linestyle="--")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs Cycle")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_feature_importance(names: list[str], importances: np.ndarray, path: Path, top_n: int = 20) -> Path:
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.array(names)[order][::-1], np.array(importances)[order][::-1], color="#1f4e79")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    return _save(fig, path)


def plot_rul(cycles, rul, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cycles, rul, color="#1f4e79")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Estimated RUL (cycles)")
    ax.set_title("RUL vs Cycle")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_degradation_rate(cycles, slopes, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cycles, slopes, color="#1f4e79")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Capacity slope")
    ax.set_title("Degradation Rate vs Cycle")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_actual_vs_predicted(y_true, y_pred, path: Path) -> Path:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, color="#1f4e79")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#c0392b", label="y = x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_horizon_error(metrics_df: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sub = metrics_df[metrics_df["split"] == "validation"]
    for model, g in sub.groupby("model"):
        g = g.sort_values("horizon")
        ax.plot(g["horizon"], g["rmse"], marker="o", label=model)
    ax.set_xlabel("Horizon (cycles)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs Forecast Horizon")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def generate_all_static_figures(
    prepared: pd.DataFrame,
    metrics_df: pd.DataFrame,
    forecast: Optional[pd.DataFrame],
    figures_dir: Path,
    battery_id: str,
    eol_soh: float,
    knee_cycle: Optional[float] = None,
    eol_cycle: Optional[float] = None,
    feature_importance: Optional[tuple[list[str], np.ndarray]] = None,
    residual_data: Optional[dict[str, Any]] = None,
) -> list[Path]:
    ensure_dir(figures_dir)
    hist = prepared[prepared["battery_id"] == battery_id].sort_values("cycle_number")
    paths = []
    paths.append(plot_capacity_history(hist, figures_dir / "capacity_vs_cycle.png", battery_id=battery_id))
    paths.append(
        plot_soh_history(
            hist, figures_dir / "soh_vs_cycle.png", eol_soh=eol_soh, battery_id=battery_id, knee_cycle=knee_cycle
        )
    )
    if forecast is not None and len(forecast):
        paths.append(
            plot_forecast(
                hist,
                forecast,
                figures_dir / "future_forecast.png",
                eol_soh=eol_soh,
                eol_cycle=eol_cycle,
                knee_cycle=knee_cycle,
            )
        )
        if "estimated_rul" in forecast.columns:
            paths.append(
                plot_rul(
                    forecast["cycle_number"],
                    forecast["estimated_rul"],
                    figures_dir / "rul_vs_cycle.png",
                )
            )
    if metrics_df is not None and len(metrics_df):
        paths.append(plot_model_comparison(metrics_df, figures_dir / "model_comparison_rmse.png"))
        paths.append(plot_horizon_error(metrics_df, figures_dir / "horizon_rmse.png"))
    if feature_importance is not None:
        names, imps = feature_importance
        paths.append(plot_feature_importance(names, imps, figures_dir / "feature_importance.png"))
    if residual_data is not None:
        paths.append(
            plot_residuals(
                residual_data["cycles"],
                residual_data["y_true"],
                residual_data["y_pred"],
                figures_dir / "residuals.png",
            )
        )
        paths.append(
            plot_actual_vs_predicted(
                residual_data["y_true"], residual_data["y_pred"], figures_dir / "actual_vs_predicted.png"
            )
        )
    if "capacity_slope_20" in hist.columns:
        paths.append(
            plot_degradation_rate(
                hist["cycle_number"], hist["capacity_slope_20"], figures_dir / "degradation_rate.png"
            )
        )
    return paths
