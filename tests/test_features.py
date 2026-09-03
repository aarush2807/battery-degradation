"""Feature engineering tests."""

import numpy as np
import pandas as pd

from battery_degradation.features import add_targets, build_features_for_battery, get_feature_columns
from battery_degradation.preprocessing import add_soh


def _toy_battery(n=60):
    df = pd.DataFrame(
        {
            "battery_id": ["B1"] * n,
            "cycle_number": np.arange(1, n + 1),
            "capacity_ah": 4.5 - 0.001 * np.arange(n),
            "temperature_mean": 25 + np.random.randn(n) * 0.1,
            "internal_resistance": 0.02 + 0.00001 * np.arange(n),
            "energy_wh": np.ones(n) * 15,
        }
    )
    df, _ = add_soh(df)
    return df


def test_lag_correctness():
    df = _toy_battery(30)
    feat = build_features_for_battery(df)
    assert np.isclose(feat.loc[10, "capacity_lag_1"], feat.loc[9, "capacity_ah"])
    assert np.isclose(feat.loc[10, "capacity_lag_5"], feat.loc[5, "capacity_ah"])


def test_rolling_uses_history_including_current():
    df = _toy_battery(20)
    feat = build_features_for_battery(df, rolling_windows=[5])
    expected = df["capacity_ah"].iloc[0:5].mean()
    assert np.isclose(feat.loc[4, "capacity_rolling_mean_5"], expected)


def test_targets_shifted():
    df = _toy_battery(40)
    feat = build_features_for_battery(df)
    out = add_targets(feat, horizons=[1, 5])
    assert np.isclose(out.loc[0, "capacity_t_plus_1"], out.loc[1, "capacity_ah"])
    assert np.isclose(out.loc[0, "capacity_t_plus_5"], out.loc[5, "capacity_ah"])
    assert pd.isna(out.loc[len(out) - 1, "capacity_t_plus_1"])


def test_feature_columns_exclude_targets():
    df = _toy_battery(50)
    feat = build_features_for_battery(df)
    feat = add_targets(feat, horizons=[1, 10])
    cols = get_feature_columns(feat, horizons=[1, 10])
    assert not any("t_plus_" in c for c in cols)
