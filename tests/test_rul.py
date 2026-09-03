"""SOH / RUL tests."""

import numpy as np
import pandas as pd

from battery_degradation.preprocessing import add_soh, compute_reference_capacity
from battery_degradation.rul import detect_degradation_knee, estimate_eol_from_trajectory, rul_at_cycle


def test_soh_first_n_average():
    df = pd.DataFrame(
        {
            "battery_id": ["A"] * 10,
            "cycle_number": range(1, 11),
            "capacity_ah": [5, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1],
        }
    )
    ref = compute_reference_capacity(df, method="first_n_average", n=5)
    assert np.isclose(ref, np.mean([5, 4.9, 4.8, 4.7, 4.6]))
    out, refs = add_soh(df, method="first_n_average", n=5)
    assert np.isclose(out.iloc[-1]["soh"], 4.1 / ref)


def test_rul_clamped_at_zero():
    assert rul_at_cycle(100, 120) == 0.0
    assert rul_at_cycle(100, 80) == 20.0
    assert rul_at_cycle(None, 50) == 0.0


def test_eol_crossing():
    cycles = np.arange(1, 101)
    soh = np.linspace(1.0, 0.7, 100)
    res = estimate_eol_from_trajectory(cycles, soh, eol_soh=0.8, current_cycle=50)
    assert res["eol_cycle"] is not None
    assert res["rul"] >= 0


def test_knee_detector_runs():
    cycles = np.arange(1, 301)
    soh = 1.0 - 0.0003 * cycles
    soh = soh - np.maximum(0, cycles - 200) ** 1.5 * 1e-5
    out = detect_degradation_knee(cycles, soh)
    assert "estimated_knee_cycle" in out
    assert out["label"].startswith("Experimental")
