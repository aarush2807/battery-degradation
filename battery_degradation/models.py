"""Model zoo for battery degradation forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from battery_degradation.utils import get_logger, set_global_seed

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    HAS_XGBOOST = False
    XGBRegressor = None  # type: ignore


class LastValueBaseline(BaseEstimator, RegressorMixin):
    """Predict future capacity ≈ current capacity (requires capacity_ah in X)."""

    def __init__(self, capacity_col: str = "capacity_ah"):
        self.capacity_col = capacity_col
        self.feature_names_in_: Optional[list[str]] = None

    def fit(self, X, y=None):
        self.feature_names_in_ = list(getattr(X, "columns", range(X.shape[1])))
        return self

    def predict(self, X):
        if hasattr(X, "columns") and self.capacity_col in X.columns:
            return X[self.capacity_col].to_numpy(dtype=float)
        # fallback: first column
        arr = np.asarray(X, dtype=float)
        return arr[:, 0]


class HistoricalLinearExtrapolation(BaseEstimator, RegressorMixin):
    """
    Uses capacity_slope_w and capacity_ah + cycle to extrapolate.
    At predict time: pred = capacity_ah + slope * horizon.
    Horizon is stored at fit/construction.
    """

    def __init__(self, horizon: int = 1, slope_col: str = "capacity_slope_10"):
        self.horizon = horizon
        self.slope_col = slope_col

    def fit(self, X, y=None):
        self.feature_names_in_ = list(getattr(X, "columns", []))
        return self

    def predict(self, X):
        if hasattr(X, "columns"):
            cap = X["capacity_ah"].to_numpy(dtype=float) if "capacity_ah" in X.columns else np.asarray(X)[:, 0]
            if self.slope_col in X.columns:
                slope = X[self.slope_col].to_numpy(dtype=float)
            else:
                # try any capacity_slope
                slope_cols = [c for c in X.columns if c.startswith("capacity_slope_")]
                slope = X[slope_cols[0]].to_numpy(dtype=float) if slope_cols else np.zeros(len(X))
        else:
            arr = np.asarray(X, dtype=float)
            cap = arr[:, 0]
            slope = np.zeros(len(arr))
        slope = np.nan_to_num(slope, nan=0.0)
        return cap + slope * float(self.horizon)


@dataclass
class ModelSpec:
    name: str
    estimator: Any
    needs_scaling: bool = False
    supports_uncertainty: bool = False
    family: str = "ml"  # baseline | trend | ml | ensemble


def build_candidate_models(config: dict[str, Any], horizon: int, seed: int = 42) -> list[ModelSpec]:
    set_global_seed(seed)
    mcfg = config.get("models", {})
    specs: list[ModelSpec] = []

    if mcfg.get("last_value", True):
        specs.append(ModelSpec("LastValue", LastValueBaseline(), family="baseline"))
    if mcfg.get("historical_linear", True):
        specs.append(
            ModelSpec(
                "HistoricalLinear",
                HistoricalLinearExtrapolation(horizon=horizon),
                family="trend",
            )
        )
    if mcfg.get("linear_regression", True):
        specs.append(
            ModelSpec(
                "LinearRegression",
                Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
                needs_scaling=True,
                family="trend",
            )
        )
    if mcfg.get("polynomial_regression", True):
        for deg in mcfg.get("polynomial_degrees", [2, 3]):
            specs.append(
                ModelSpec(
                    f"PolynomialDeg{deg}",
                    Pipeline(
                        [
                            ("poly", PolynomialFeatures(degree=int(deg), include_bias=False)),
                            ("scaler", StandardScaler()),
                            ("model", LinearRegression()),
                        ]
                    ),
                    family="trend",
                )
            )
    if mcfg.get("ridge", True):
        specs.append(
            ModelSpec(
                "Ridge",
                Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=seed))]),
                family="trend",
            )
        )
    if mcfg.get("lasso", True):
        specs.append(
            ModelSpec(
                "Lasso",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Lasso(alpha=0.0005, max_iter=5000, random_state=seed)),
                    ]
                ),
                family="trend",
            )
        )
    if mcfg.get("elastic_net", True):
        specs.append(
            ModelSpec(
                "ElasticNet",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=5000, random_state=seed)),
                    ]
                ),
                family="trend",
            )
        )
    if mcfg.get("random_forest", True):
        specs.append(
            ModelSpec(
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
                supports_uncertainty=True,
                family="ml",
            )
        )
    if mcfg.get("extra_trees", True):
        specs.append(
            ModelSpec(
                "ExtraTrees",
                ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
                supports_uncertainty=True,
                family="ml",
            )
        )
    if mcfg.get("gradient_boosting", True):
        specs.append(
            ModelSpec(
                "HistGradientBoosting",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=200,
                    random_state=seed,
                ),
                family="ml",
            )
        )
    if mcfg.get("xgboost", False) and HAS_XGBOOST:
        specs.append(
            ModelSpec(
                "XGBoost",
                XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=seed,
                    n_jobs=-1,
                    objective="reg:squarederror",
                ),
                family="ml",
            )
        )
    elif mcfg.get("xgboost", False) and not HAS_XGBOOST:
        logger.info("xgboost requested but not installed; skipping.")

    return specs


RF_PARAM_DIST = {
    "n_estimators": [100, 200, 400, 600],
    "max_depth": [None, 5, 10, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": [1.0, "sqrt", "log2"],
}


def get_estimator_for_tuning(name: str, seed: int = 42):
    if name == "RandomForest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1), RF_PARAM_DIST
    if name == "ExtraTrees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1), RF_PARAM_DIST
    if name == "Ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=seed))]), {
            "model__alpha": np.logspace(-3, 2, 20)
        }
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=seed), {
            "max_depth": [3, 5, 6, 8, None],
            "learning_rate": [0.03, 0.05, 0.08, 0.1],
            "max_iter": [100, 200, 300],
        }
    return None, None


def unwrap_model(estimator: Any) -> Any:
    if isinstance(estimator, Pipeline):
        return estimator.named_steps.get("model", estimator)
    return estimator
