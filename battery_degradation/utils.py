"""Shared utilities: paths, seeding, formatting, logging helpers."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_logger(name: str = "battery_degradation") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def format_soh(soh: float, as_percent: bool = False) -> str:
    if as_percent:
        return f"{soh * 100:.2f}%"
    return f"{soh:.4f}"


def format_capacity(capacity: float, unit: str = "Ah") -> str:
    return f"{capacity:.3f} {unit}"


def format_rul(rul: float) -> str:
    return f"{int(max(0, round(rul)))} cycles"


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def safe_div(numer: float, denom: float, default: float = np.nan) -> float:
    if denom is None or denom == 0 or np.isnan(denom):
        return default
    return float(numer / denom)


def flatten_unique(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
