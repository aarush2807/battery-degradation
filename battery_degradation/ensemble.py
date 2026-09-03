"""Weighted hybrid ensembles and stacking."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from battery_degradation.utils import get_logger

logger = get_logger(__name__)


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """Nonnegative weights summing to 1, set from validation performance."""

    def __init__(self, estimators: Optional[dict[str, Any]] = None, weights: Optional[dict[str, float]] = None):
        self.estimators = estimators or {}
        self.weights = weights or {}

    def fit(self, X, y):
        for name, est in self.estimators.items():
            est.fit(X, y)
        return self

    def predict(self, X):
        if not self.estimators:
            raise RuntimeError("No estimators in ensemble")
        preds = []
        ws = []
        for name, est in self.estimators.items():
            preds.append(est.predict(X))
            ws.append(self.weights.get(name, 0.0))
        w = np.asarray(ws, dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
        stacked = np.vstack(preds)
        return np.average(stacked, axis=0, weights=w)

    def predict_components(self, X) -> dict[str, np.ndarray]:
        return {name: est.predict(X) for name, est in self.estimators.items()}


def weights_from_validation_rmse(rmses: dict[str, float], eps: float = 1e-8) -> dict[str, float]:
    """Inverse-RMSE weights, nonnegative, sum to 1."""
    inv = {k: 1.0 / max(v, eps) for k, v in rmses.items()}
    total = sum(inv.values())
    return {k: v / total for k, v in inv.items()}


def build_weighted_ensemble(
    fitted_models: dict[str, Any],
    val_rmses: dict[str, float],
    prefer_names: Optional[list[str]] = None,
    top_k: int = 4,
) -> WeightedEnsemble:
    """Select top-k by validation RMSE and weight them."""
    ranked = sorted(val_rmses.items(), key=lambda kv: kv[1])
    if prefer_names:
        # keep only available preferred, else top_k
        selected = [n for n, _ in ranked if n in prefer_names][:top_k]
        if len(selected) < 2:
            selected = [n for n, _ in ranked[:top_k]]
    else:
        selected = [n for n, _ in ranked[:top_k]]
    selected = [n for n in selected if n in fitted_models]
    subset_rmse = {n: val_rmses[n] for n in selected}
    weights = weights_from_validation_rmse(subset_rmse)
    estimators = {n: fitted_models[n] for n in selected}
    logger.info("Ensemble weights: %s", weights)
    return WeightedEnsemble(estimators=estimators, weights=weights)


def build_stacking_ensemble(base_estimators: list[tuple[str, Any]], seed: int = 42) -> StackingRegressor:
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=Pipeline(
            [("scaler", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 2, 20)))]
        ),
        passthrough=False,
        n_jobs=-1,
    )
