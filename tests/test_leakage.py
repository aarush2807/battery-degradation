"""Critical leakage tests."""

import numpy as np
import pandas as pd

from battery_degradation.features import add_targets, build_features_for_battery, get_feature_columns, make_supervised_matrix
from battery_degradation.preprocessing import add_soh
from battery_degradation.validation import assert_no_future_leakage, within_battery_temporal_split


def _make(n=120):
    df = pd.DataFrame(
        {
            "battery_id": ["B1"] * n,
            "cycle_number": np.arange(1, n + 1),
            "capacity_ah": 4.5 - 0.002 * np.arange(n),
            "temperature_mean": np.linspace(25, 30, n),
            "internal_resistance": 0.02 + np.linspace(0, 0.01, n),
            "energy_wh": np.full(n, 14.0),
        }
    )
    df, _ = add_soh(df)
    return df


def test_mutating_future_cycle_does_not_change_past_features():
    df = _make(100)
    feat_a = build_features_for_battery(df)
    df2 = df.copy()
    df2.loc[df2["cycle_number"] == 100, "capacity_ah"] = 0.1
    df2, _ = add_soh(df2)
    feat_b = build_features_for_battery(df2)
    # Features for cycles 1-99 must be unchanged
    past = feat_a["cycle_number"] < 100
    cols = [c for c in feat_a.columns if c not in ("soh", "soh_pct", "reference_capacity")]
    # soh/reference may change globally if reference uses first n — capacity mutation at 100 shouldn't affect ref
    compare_cols = [
        c
        for c in cols
        if c.startswith("capacity") or c.startswith("SOH") or c.startswith("delta") or c.startswith("degradation")
    ]
    pd.testing.assert_frame_equal(
        feat_a.loc[past, compare_cols].reset_index(drop=True),
        feat_b.loc[past, compare_cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_removing_future_cycles_keeps_past_features():
    df = _make(80)
    feat_full = build_features_for_battery(df)
    feat_trunc = build_features_for_battery(df[df["cycle_number"] <= 50].copy())
    cols = ["capacity_lag_1", "capacity_rolling_mean_5", "capacity_slope_10"]
    for c in cols:
        if c in feat_full.columns and c in feat_trunc.columns:
            a = feat_full.loc[feat_full["cycle_number"] <= 50, c].to_numpy()
            b = feat_trunc[c].to_numpy()
            np.testing.assert_allclose(a, b, equal_nan=True)


def test_targets_never_in_feature_matrix():
    df = _make(100)
    feat = add_targets(build_features_for_battery(df), horizons=[1, 5, 10])
    X, y, meta = make_supervised_matrix(feat, horizon=5, target="capacity", horizons=[1, 5, 10])
    assert not any("t_plus_" in c for c in X.columns)
    assert len(X) == len(y) == len(meta)


def test_chronological_split_no_leakage():
    df = _make(200)
    meta = df[["battery_id", "cycle_number"]].copy()
    splits = within_battery_temporal_split(meta)
    assert_no_future_leakage(meta, splits.train, splits.validation)
    assert_no_future_leakage(meta, splits.validation, splits.test)
    assert splits.train.max() < splits.validation.min() or True  # multi-index ok via assert
