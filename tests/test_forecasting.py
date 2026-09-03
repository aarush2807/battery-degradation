"""Forecasting helper tests."""

import numpy as np
import pandas as pd

from battery_degradation.forecasting import apply_capacity_constraints, soft_monotonic_degradation, smooth_forecast


def test_constraints():
    preds = np.array([5.0, -1.0, 3.0])
    out = apply_capacity_constraints(preds, reference_capacity=4.0, capacity_upper_factor=1.05)
    assert out.max() <= 4.0 * 1.05 + 1e-9
    assert out.min() >= 0.0


def test_smooth_and_monotonic():
    vals = np.array([4.0, 3.9, 4.05, 3.8, 3.7])
    sm = smooth_forecast(vals, window=3)
    assert len(sm) == len(vals)
    mono = soft_monotonic_degradation(vals, max_increase=0.01)
    assert mono[2] <= vals[1] + 0.01 + 1e-9
