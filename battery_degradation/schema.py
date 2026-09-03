"""Centralized column schema and alias normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from battery_degradation.utils import get_logger

logger = get_logger(__name__)

# Canonical name -> accepted aliases (lowercase match)
COLUMN_ALIASES: dict[str, list[str]] = {
    "battery_id": ["battery_id", "battery", "cell_id", "cell", "id", "batteryid"],
    "cycle_number": [
        "cycle_number",
        "cycle",
        "cycle_index",
        "cycle_num",
        "cycle_count",
        "n_cycle",
        "cycles",
    ],
    "capacity_ah": [
        "capacity_ah",
        "capacity",
        "capacity_Ah",
        "discharge_capacity",
        "q_discharge",
        "cap",
        "Capacity",
    ],
    "discharge_capacity_ah": [
        "discharge_capacity_ah",
        "discharge_capacity",
        "qd",
        "q_d",
    ],
    "charge_capacity_ah": [
        "charge_capacity_ah",
        "charge_capacity",
        "qc",
        "q_c",
    ],
    "voltage_mean": ["voltage_mean", "v_mean", "avg_voltage", "voltage_avg", "voltage"],
    "voltage_min": ["voltage_min", "v_min", "min_voltage"],
    "voltage_max": ["voltage_max", "v_max", "max_voltage"],
    "current_mean": ["current_mean", "i_mean", "avg_current", "current_avg", "current"],
    "current_min": ["current_min", "i_min", "min_current"],
    "current_max": ["current_max", "i_max", "max_current"],
    "temperature_mean": [
        "temperature_mean",
        "temp",
        "temperature",
        "avg_temperature",
        "temp_mean",
        "t_mean",
    ],
    "temperature_min": ["temperature_min", "temp_min", "t_min", "min_temperature"],
    "temperature_max": ["temperature_max", "temp_max", "t_max", "max_temperature"],
    "charge_time": ["charge_time", "t_charge", "charge_duration"],
    "discharge_time": ["discharge_time", "t_discharge", "discharge_duration"],
    "internal_resistance": [
        "internal_resistance",
        "resistance",
        "ir",
        "dcr",
        "internal_r",
    ],
    "energy_wh": ["energy_wh", "energy", "discharge_energy", "wh"],
    "coulombic_efficiency": [
        "coulombic_efficiency",
        "ce",
        "coulomb_efficiency",
        "efficiency",
    ],
    "timestamp": ["timestamp", "time", "datetime", "date", "date_time"],
}

REQUIRED_CANONICAL = {"cycle_number"}
CAPACITY_CANDIDATES = ("capacity_ah", "discharge_capacity_ah", "charge_capacity_ah")


@dataclass
class SchemaMappingResult:
    mapping: dict[str, str]  # canonical -> original column
    ambiguities: dict[str, list[str]] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def has_capacity(self) -> bool:
        return any(c in self.mapping for c in CAPACITY_CANDIDATES)

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def build_alias_lookup() -> dict[str, list[str]]:
    """Map normalized alias -> list of canonical names that claim it."""
    lookup: dict[str, list[str]] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalize_name(alias)
            lookup.setdefault(key, [])
            if canonical not in lookup[key]:
                lookup[key].append(canonical)
    return lookup


def resolve_schema(
    columns: list[str],
    overrides: Optional[dict[str, str]] = None,
    allow_ambiguous: bool = False,
) -> SchemaMappingResult:
    """
    Resolve raw CSV columns to canonical names.

    overrides: mapping of canonical_name -> original column name (user forced).
    Ambiguous aliases are reported and not auto-assigned unless overridden.
    """
    overrides = overrides or {}
    lookup = build_alias_lookup()
    col_by_norm = {_normalize_name(c): c for c in columns}

    mapping: dict[str, str] = {}
    ambiguities: dict[str, list[str]] = {}
    claimed_originals: set[str] = set()
    warnings: list[str] = []
    errors: list[str] = []

    # Apply explicit overrides first
    for canonical, original in overrides.items():
        if original not in columns:
            errors.append(f"Override column '{original}' for '{canonical}' not found in data.")
            continue
        mapping[canonical] = original
        claimed_originals.add(original)

    # Auto-map unambiguous aliases
    for original in columns:
        if original in claimed_originals:
            continue
        norm = _normalize_name(original)
        candidates = lookup.get(norm, [])
        if not candidates:
            continue
        if len(candidates) > 1:
            ambiguities[original] = candidates
            msg = (
                f"Ambiguous column '{original}' maps to {candidates}. "
                "Provide an explicit override in config or dashboard."
            )
            warnings.append(msg)
            logger.warning(msg)
            if allow_ambiguous:
                # Prefer first listed candidate not already mapped
                for cand in candidates:
                    if cand not in mapping:
                        mapping[cand] = original
                        claimed_originals.add(original)
                        break
            continue
        canonical = candidates[0]
        if canonical in mapping:
            warnings.append(
                f"Multiple columns map to '{canonical}': "
                f"keeping '{mapping[canonical]}', ignoring '{original}'."
            )
            continue
        mapping[canonical] = original
        claimed_originals.add(original)

    unmapped = [c for c in columns if c not in claimed_originals]

    for req in REQUIRED_CANONICAL:
        if req not in mapping:
            errors.append(f"Required column '{req}' could not be resolved from {columns}.")

    if not any(c in mapping for c in CAPACITY_CANDIDATES):
        errors.append(
            "No capacity column found. Need one of: capacity_ah, discharge_capacity_ah, "
            "charge_capacity_ah (or aliases)."
        )

    return SchemaMappingResult(
        mapping=mapping,
        ambiguities=ambiguities,
        unmapped=unmapped,
        warnings=warnings,
        errors=errors,
    )


def apply_schema(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename mapped columns to canonical names; keep unmapped extras."""
    rename = {original: canonical for canonical, original in mapping.items()}
    out = df.rename(columns=rename).copy()
    # Ensure primary capacity_ah exists
    if "capacity_ah" not in out.columns:
        for alt in ("discharge_capacity_ah", "charge_capacity_ah"):
            if alt in out.columns:
                out["capacity_ah"] = out[alt]
                logger.info("Using %s as capacity_ah", alt)
                break
    return out


def schema_summary(result: SchemaMappingResult) -> dict[str, Any]:
    return {
        "mapping": result.mapping,
        "ambiguities": result.ambiguities,
        "unmapped": result.unmapped,
        "warnings": result.warnings,
        "errors": result.errors,
        "has_capacity": result.has_capacity,
        "ok": result.ok,
    }
