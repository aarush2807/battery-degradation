"""Text/CSV reporting after training and evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from battery_degradation.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def write_metrics_csv(metrics_df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    metrics_df.to_csv(path, index=False)
    return path


def write_model_summary(
    path: Path,
    dataset_info: dict[str, Any],
    best_by_horizon: pd.DataFrame,
    metrics_df: pd.DataFrame,
    battery_snapshot: Optional[dict[str, Any]] = None,
    warnings: Optional[list[str]] = None,
    assumptions: Optional[list[str]] = None,
) -> Path:
    ensure_dir(path.parent)
    lines: list[str] = []
    lines.append("=" * 50)
    lines.append("BATTERY DEGRADATION MODEL")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("Dataset:")
    for k, v in dataset_info.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Best models by validation RMSE (test reported but not used for selection):")
    if best_by_horizon is not None and len(best_by_horizon):
        for _, row in best_by_horizon.iterrows():
            lines.append(
                f"  Horizon {int(row['horizon'])}: {row['model']} | "
                f"Val RMSE={row['rmse']:.6f}"
            )
    lines.append("")
    if battery_snapshot:
        lines.append("Current battery snapshot:")
        for k, v in battery_snapshot.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
    if warnings:
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")
    lines.append("Assumptions:")
    for a in assumptions or [
        "Features use only information available at or before cycle t.",
        "Model selection uses validation metrics only; test is final holdout.",
        "Uncertainty bands are approximate, not calibrated confidence intervals.",
        "Degradation knee detection is experimental.",
        "Synthetic data is for software demo only, not a validated physics model.",
    ]:
        lines.append(f"  - {a}")
    lines.append("")
    lines.append("Dashboard:")
    lines.append("  streamlit run dashboard/app.py")
    lines.append("=" * 50)

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report %s", path)
    return path


def print_training_banner(
    dataset_info: dict[str, Any],
    best_by_horizon: pd.DataFrame,
    battery_snapshot: Optional[dict[str, Any]] = None,
) -> None:
    print("=" * 50)
    print("BATTERY DEGRADATION MODEL")
    print("=" * 50)
    print("")
    print("Dataset:")
    for k, v in dataset_info.items():
        print(f"  {k}: {v}")
    print("")
    if best_by_horizon is not None and len(best_by_horizon):
        for _, row in best_by_horizon.iterrows():
            h = int(row["horizon"])
            print(f"Best model — {h} cycle{'s' if h != 1 else ''}:")
            print(f"  {row['model']}")
            print(f"  Validation RMSE: {row['rmse']:.4f}")
            if "test_rmse" in row and pd.notna(row.get("test_rmse")):
                print(f"  Test RMSE: {row['test_rmse']:.4f}")
            print("")
    if battery_snapshot:
        print("Current battery:")
        for k, v in battery_snapshot.items():
            print(f"  {k}: {v}")
        print("")
    print("Dashboard:")
    print("  streamlit run dashboard/app.py")
    print("=" * 50)
