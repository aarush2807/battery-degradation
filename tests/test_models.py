"""Model smoke tests."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from battery_degradation.models import HistoricalLinearExtrapolation, LastValueBaseline, build_candidate_models


def test_baselines_predict():
    X = pd.DataFrame(
        {
            "capacity_ah": [4.0, 3.9],
            "capacity_slope_10": [-0.001, -0.002],
        }
    )
    y = np.array([3.99, 3.88])
    lv = LastValueBaseline().fit(X, y)
    np.testing.assert_allclose(lv.predict(X), X["capacity_ah"])
    hl = HistoricalLinearExtrapolation(horizon=10).fit(X, y)
    pred = hl.predict(X)
    assert pred.shape == (2,)


def test_candidate_models_build():
    cfg = {
        "models": {
            "last_value": True,
            "historical_linear": True,
            "linear_regression": True,
            "polynomial_regression": False,
            "ridge": True,
            "lasso": False,
            "elastic_net": False,
            "random_forest": True,
            "extra_trees": False,
            "gradient_boosting": False,
            "xgboost": False,
            "ensemble": True,
        }
    }
    specs = build_candidate_models(cfg, horizon=5, seed=0)
    names = [s.name for s in specs]
    assert "LastValue" in names
    assert "RandomForest" in names
    # fit one quickly
    X = pd.DataFrame(np.random.randn(40, 3), columns=["capacity_ah", "a", "b"])
    y = X["capacity_ah"] - 0.01
    for s in specs:
        m = clone(s.estimator)
        m.fit(X, y)
        assert len(m.predict(X)) == 40
