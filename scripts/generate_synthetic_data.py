#!/usr/bin/env python3
"""Generate synthetic lithium-ion battery cycle data for demo/testing.

This is NOT a physically validated battery simulation — software demo only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def simulate_battery(
    battery_id: str,
    n_cycles: int,
    rng: np.random.Generator,
    initial_capacity: float = 4.5,
    linear_fade: float = 3.5e-4,
    nonlinear_scale: float = 0.0,
    knee_start: float | None = None,
    knee_strength: float = 0.0,
    base_temp: float = 25.0,
    temp_noise: float = 1.5,
    resistance_growth: float = 2e-5,
    noise_std: float = 0.008,
) -> pd.DataFrame:
    cycles = np.arange(1, n_cycles + 1)
    # Base linear + mild nonlinear fade
    fade = linear_fade * cycles + nonlinear_scale * (cycles / n_cycles) ** 2
    if knee_start is not None:
        knee = np.maximum(0, cycles - knee_start)
        fade = fade + knee_strength * (knee / max(n_cycles - knee_start, 1)) ** 1.8

    capacity = initial_capacity * (1.0 - fade)
    capacity = capacity + rng.normal(0, noise_std, size=n_cycles)
    # rare small recovery blips
    recovery_idx = rng.choice(n_cycles, size=max(1, n_cycles // 80), replace=False)
    capacity[recovery_idx] += rng.uniform(0.005, 0.02, size=len(recovery_idx))
    capacity = np.clip(capacity, 0.5, initial_capacity * 1.02)

    temperature = base_temp + rng.normal(0, temp_noise, size=n_cycles)
    # mild correlation: hotter cycles slightly more fade already in trajectory; add measurement link
    temperature += 0.002 * (initial_capacity - capacity) * 100

    internal_resistance = 0.018 + resistance_growth * cycles + rng.normal(0, 0.0004, size=n_cycles)
    internal_resistance += 0.01 * (1 - capacity / initial_capacity)

    voltage_mean = 3.6 - 0.15 * (1 - capacity / initial_capacity) + rng.normal(0, 0.01, n_cycles)
    voltage_min = voltage_mean - rng.uniform(0.15, 0.35, n_cycles)
    voltage_max = voltage_mean + rng.uniform(0.2, 0.4, n_cycles)
    current_mean = rng.normal(-1.5, 0.2, n_cycles)
    current_min = current_mean - rng.uniform(0.5, 1.5, n_cycles)
    current_max = current_mean + rng.uniform(1.0, 2.5, n_cycles)
    charge_time = rng.normal(3600, 200, n_cycles)
    discharge_time = rng.normal(3400, 220, n_cycles)
    energy_wh = capacity * voltage_mean * rng.uniform(0.95, 1.05, n_cycles)
    charge_capacity = capacity * rng.uniform(0.98, 1.02, n_cycles)
    discharge_capacity = capacity.copy()
    coulombic_efficiency = np.clip(discharge_capacity / np.maximum(charge_capacity, 1e-6), 0.9, 1.02)

    return pd.DataFrame(
        {
            "battery_id": battery_id,
            "cycle_number": cycles,
            "capacity_ah": capacity,
            "discharge_capacity_ah": discharge_capacity,
            "charge_capacity_ah": charge_capacity,
            "voltage_mean": voltage_mean,
            "voltage_min": voltage_min,
            "voltage_max": voltage_max,
            "current_mean": current_mean,
            "current_min": current_min,
            "current_max": current_max,
            "temperature_mean": temperature,
            "temperature_min": temperature - rng.uniform(1, 3, n_cycles),
            "temperature_max": temperature + rng.uniform(1, 4, n_cycles),
            "charge_time": charge_time,
            "discharge_time": discharge_time,
            "internal_resistance": internal_resistance,
            "energy_wh": energy_wh,
            "coulombic_efficiency": coulombic_efficiency,
            "timestamp": pd.date_range("2020-01-01", periods=n_cycles, freq="D"),
        }
    )


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = [
        dict(battery_id="BATTERY_001", n_cycles=900, linear_fade=2.2e-4, noise_std=0.006),  # slow
        dict(battery_id="BATTERY_002", n_cycles=800, linear_fade=3.5e-4, nonlinear_scale=0.05),  # moderate
        dict(
            battery_id="BATTERY_003",
            n_cycles=1000,
            linear_fade=2.8e-4,
            knee_start=650,
            knee_strength=0.25,
        ),  # late knee
        dict(battery_id="BATTERY_004", n_cycles=750, linear_fade=3.2e-4, base_temp=38.0, temp_noise=2.5),
        dict(battery_id="BATTERY_005", n_cycles=850, linear_fade=3.0e-4, resistance_growth=5e-5),
        dict(battery_id="BATTERY_006", n_cycles=700, linear_fade=3.8e-4, noise_std=0.02),  # noisy
        dict(battery_id="BATTERY_007", n_cycles=1100, linear_fade=1.8e-4, initial_capacity=4.8),
        dict(
            battery_id="BATTERY_008",
            n_cycles=950,
            linear_fade=3.0e-4,
            knee_start=700,
            knee_strength=0.18,
            base_temp=32.0,
        ),
    ]
    frames = [simulate_battery(rng=rng, **p) for p in profiles]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic battery cycle CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "synthetic_battery_cycles.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows, {df['battery_id'].nunique()} batteries -> {args.output}")
    print("Note: synthetic demo data only — not a validated physics model.")


if __name__ == "__main__":
    main()
