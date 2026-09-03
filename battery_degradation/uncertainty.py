"""Uncertainty estimation helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from battery_degradation.ensemble import WeightedEnsemble


def tree_prediction_intervals(
    model: Any,
    X,
    lower_percentile: float = 5,
    upper_percentile: float = 95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate predictive intervals from tree ensemble member predictions.
    Returns mean, lower, upper.
    """
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps.get("model", model)

    if isinstance(est, (RandomForestRegressor, ExtraTreesRegressor)):
        # collect per-tree predictions
        all_preds = np.stack([t.predict(np.asarray(X)) for t in est.estimators_], axis=0)
        mean = all_preds.mean(axis=0)
        lower = np.percentile(all_preds, lower_percentile, axis=0)
        upper = np.percentile(all_preds, upper_percentile, axis=0)
        return mean, lower, upper

    # fallback: point prediction with empty band
    pred = np.asarray(model.predict(X), dtype=float)
    return pred, pred.copy(), pred.copy()


def ensemble_disagreement_intervals(
    ensemble: WeightedEnsemble,
    X,
    lower_percentile: float = 5,
    upper_percentile: float = 95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate intervals from component model disagreement."""
    comps = ensemble.predict_components(X)
    stacked = np.vstack(list(comps.values()))
    mean = np.average(
        stacked,
        axis=0,
        weights=[ensemble.weights.get(n, 0) for n in comps.keys()],
    )
    # If weights degenerate, use simple mean
    if not np.isfinite(mean).all():
        mean = stacked.mean(axis=0)
    lower = np.percentile(stacked, lower_percentile, axis=0)
    upper = np.percentile(stacked, upper_percentile, axis=0)
    # Also scale by std as soft band if components agree too tightly
    std = stacked.std(axis=0)
    lower = np.minimum(lower, mean - std)
    upper = np.maximum(upper, mean + std)
    return mean, lower, upper


def attach_soh_bounds(
    capacity_mean: np.ndarray,
    capacity_lower: np.ndarray,
    capacity_upper: np.ndarray,
    reference_capacity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = max(float(reference_capacity), 1e-12)
    return capacity_mean / ref, capacity_lower / ref, capacity_upper / ref
