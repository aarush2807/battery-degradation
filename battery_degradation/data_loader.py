"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from battery_degradation.schema import SchemaMappingResult, apply_schema, resolve_schema
from battery_degradation.utils import get_logger, project_path

logger = get_logger(__name__)


def load_csv(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info("Loaded %s rows x %s cols from %s", len(df), df.shape[1], path.name)
    return df


def load_and_normalize(
    source: Union[str, Path, pd.DataFrame],
    column_overrides: Optional[dict[str, str]] = None,
    allow_ambiguous: bool = False,
) -> tuple[pd.DataFrame, SchemaMappingResult]:
    """Load CSV or DataFrame and normalize to canonical schema."""
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        raw = load_csv(source)

    result = resolve_schema(
        list(raw.columns),
        overrides=column_overrides,
        allow_ambiguous=allow_ambiguous,
    )
    if not result.ok:
        raise ValueError("Schema resolution failed:\n" + "\n".join(result.errors))

    normalized = apply_schema(raw, result.mapping)
    if "battery_id" not in normalized.columns:
        normalized["battery_id"] = "BATTERY_001"
        logger.info("No battery_id column found; assigning single id BATTERY_001")

    # Coerce numeric cycle
    normalized["cycle_number"] = pd.to_numeric(normalized["cycle_number"], errors="coerce")
    if "capacity_ah" in normalized.columns:
        normalized["capacity_ah"] = pd.to_numeric(normalized["capacity_ah"], errors="coerce")

    return normalized, result


def default_synthetic_path() -> Path:
    return project_path("data", "raw", "synthetic_battery_cycles.csv")


def load_default_or_upload(
    uploaded: Optional[pd.DataFrame] = None,
    column_overrides: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, SchemaMappingResult, str]:
    """Prefer uploaded frame; else synthetic CSV. Returns (df, schema, source_name)."""
    if uploaded is not None and len(uploaded):
        df, schema = load_and_normalize(uploaded, column_overrides=column_overrides)
        return df, schema, "uploaded.csv"
    path = default_synthetic_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No upload provided and synthetic data missing at {path}. "
            "Run: python scripts/generate_synthetic_data.py"
        )
    df, schema = load_and_normalize(path, column_overrides=column_overrides)
    return df, schema, path.name
