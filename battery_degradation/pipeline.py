"""End-to-end BatteryPredictor pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from battery_degradation.config import load_config
from battery_degradation.data_loader import load_and_normalize
from battery_degradation.diagnostics import battery_summary_table, data_quality_report, engineering_flags
from battery_degradation.ensemble import WeightedEnsemble, build_stacking_ensemble, build_weighted_ensemble
from battery_degradation.evaluation import (
    best_model_per_horizon,
    evaluate_predictions,
    metrics_table,
    per_battery_errors,
    regression_metrics,
)
from battery_degradation.features import add_targets, build_features, get_feature_columns, make_supervised_matrix
from battery_degradation.forecasting import (
    direct_multi_horizon_forecast,
    postprocess_forecast,
    recursive_forecast,
)
from battery_degradation.models import build_candidate_models, get_estimator_for_tuning, unwrap_model
from battery_degradation.preprocessing import prepare_dataset
from battery_degradation.reporting import print_training_banner, write_metrics_csv, write_model_summary
from battery_degradation.rul import detect_degradation_knee, estimate_eol_from_trajectory
from battery_degradation.utils import ensure_dir, get_logger, project_path, set_global_seed
from battery_degradation.validation import make_split
from battery_degradation.visualization import generate_all_static_figures

logger = get_logger(__name__)


class BatteryPredictor:
    """High-level API for training, evaluating, forecasting, and persistence."""

    def __init__(self, config: Optional[dict[str, Any]] = None, config_path: Optional[Union[str, Path]] = None):
        self.config = config or load_config(config_path)
        self.prepared_: Optional[pd.DataFrame] = None
        self.featured_: Optional[pd.DataFrame] = None
        self.schema_mapping_: dict[str, str] = {}
        self.reference_capacities_: dict[str, float] = {}
        self.feature_cols_: list[str] = []
        self.models_by_horizon_: dict[int, Any] = {}
        self.model_names_by_horizon_: dict[int, str] = {}
        self.all_fitted_: dict[int, dict[str, Any]] = {}
        self.metrics_df_: Optional[pd.DataFrame] = None
        self.best_by_horizon_: Optional[pd.DataFrame] = None
        self.quality_report_: dict[str, Any] = {}
        self.ensemble_weights_: dict[int, dict[str, float]] = {}
        self.feature_importances_: dict[int, dict[str, float]] = {}
        self.target_name_: str = "capacity"
        self.training_timestamp_: Optional[str] = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        data: Union[str, Path, pd.DataFrame],
        column_overrides: Optional[dict[str, str]] = None,
        target: str = "capacity",
    ) -> "BatteryPredictor":
        set_global_seed(int(self.config.get("random_seed", 42)))
        self.target_name_ = target
        raw, schema = load_and_normalize(
            data,
            column_overrides=column_overrides or self.config.get("data", {}).get("column_overrides"),
        )
        self.schema_mapping_ = schema.mapping
        self.quality_report_ = data_quality_report(raw, min_cycles=self.config.get("features", {}).get("min_history_cycles", 20))

        prepared, refs = prepare_dataset(raw, self.config)
        self.prepared_ = prepared
        self.reference_capacities_ = refs

        featured = build_features(prepared, self.config)
        horizons = list(self.config.get("forecasting", {}).get("horizons", [1, 5, 10, 25, 50, 100]))
        featured = add_targets(featured, horizons=horizons)
        self.featured_ = featured
        self.feature_cols_ = get_feature_columns(featured, horizons)

        metric_rows: list[dict[str, Any]] = []
        self.models_by_horizon_ = {}
        self.model_names_by_horizon_ = {}
        self.all_fitted_ = {}
        self.ensemble_weights_ = {}
        self.feature_importances_ = {}

        val_cfg = self.config.get("validation", {})
        seed = int(self.config.get("random_seed", 42))

        for horizon in horizons:
            logger.info("Training horizon t+%s", horizon)
            X, y, meta = make_supervised_matrix(
                featured, horizon=horizon, target=target, feature_cols=self.feature_cols_, horizons=horizons
            )
            # Drop rows with too many NaNs in features
            valid = X.notna().mean(axis=1) > 0.7
            X, y, meta = X.loc[valid].fillna(0.0), y.loc[valid], meta.loc[valid]
            # Reset index for positional splits
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)
            meta = meta.reset_index(drop=True)

            splits = make_split(
                meta,
                mode=val_cfg.get("mode", "within_battery"),
                train_fraction=val_cfg.get("train_fraction", 0.70),
                validation_fraction=val_cfg.get("validation_fraction", 0.15),
                test_fraction=val_cfg.get("test_fraction", 0.15),
                seed=seed,
            )

            X_train, y_train = X.loc[splits.train], y.loc[splits.train]
            X_val, y_val = X.loc[splits.validation], y.loc[splits.validation]
            X_test, y_test = X.loc[splits.test], y.loc[splits.test]
            meta_test = meta.loc[splits.test]

            candidates = build_candidate_models(self.config, horizon=horizon, seed=seed)
            fitted: dict[str, Any] = {}
            val_rmse: dict[str, float] = {}

            for spec in candidates:
                logger.info("  Fitting %s (horizon %s)", spec.name, horizon)
                model = clone(spec.estimator)
                model = self._maybe_tune(spec.name, model, X_train, y_train)
                model.fit(X_train, y_train)
                pred_val = model.predict(X_val)
                m_val = evaluate_predictions(y_val, pred_val, spec.name, horizon, "validation")
                metric_rows.append(m_val)
                pred_test = model.predict(X_test)
                m_test = evaluate_predictions(y_test, pred_test, spec.name, horizon, "test")
                metric_rows.append(m_test)
                fitted[spec.name] = model
                val_rmse[spec.name] = m_val["rmse"]

            # Weighted hybrid ensemble from validation RMSE
            if self.config.get("models", {}).get("ensemble", True) and len(fitted) >= 2:
                # Prefer mix of ml + trend families when available
                prefer = [
                    n
                    for n in [
                        "RandomForest",
                        "ExtraTrees",
                        "HistGradientBoosting",
                        "Ridge",
                        "LinearRegression",
                        "HistoricalLinear",
                        "XGBoost",
                    ]
                    if n in fitted
                ]
                ensemble = build_weighted_ensemble(fitted, val_rmse, prefer_names=prefer, top_k=4)
                # Refit ensemble components already fitted on train; predict val/test
                pred_val = ensemble.predict(X_val)
                m_val = evaluate_predictions(y_val, pred_val, "WeightedEnsemble", horizon, "validation")
                metric_rows.append(m_val)
                pred_test = ensemble.predict(X_test)
                metric_rows.append(evaluate_predictions(y_test, pred_test, "WeightedEnsemble", horizon, "test"))
                fitted["WeightedEnsemble"] = ensemble
                val_rmse["WeightedEnsemble"] = m_val["rmse"]
                self.ensemble_weights_[horizon] = dict(ensemble.weights)

            # Stacking on train (optional)
            if self.config.get("models", {}).get("stacking", True) and len(fitted) >= 3:
                base = []
                for name in ["RandomForest", "ExtraTrees", "Ridge", "HistGradientBoosting"]:
                    if name in fitted and name != "WeightedEnsemble":
                        base.append((name, clone(fitted[name])))
                if len(base) >= 2:
                    try:
                        stacker = build_stacking_ensemble(base, seed=seed)
                        stacker.fit(X_train, y_train)
                        pred_val = stacker.predict(X_val)
                        m_val = evaluate_predictions(y_val, pred_val, "StackingEnsemble", horizon, "validation")
                        metric_rows.append(m_val)
                        pred_test = stacker.predict(X_test)
                        metric_rows.append(
                            evaluate_predictions(y_test, pred_test, "StackingEnsemble", horizon, "test")
                        )
                        fitted["StackingEnsemble"] = stacker
                        val_rmse["StackingEnsemble"] = m_val["rmse"]
                    except Exception as exc:
                        logger.warning("Stacking failed for horizon %s: %s", horizon, exc)

            # Select best by validation RMSE only
            best_name = min(val_rmse, key=val_rmse.get)
            # Retrain selected on train+val
            X_trval = pd.concat([X_train, X_val], axis=0)
            y_trval = pd.concat([y_train, y_val], axis=0)
            final_model = self._refit_selected(best_name, fitted, X_trval, y_trval, seed=seed, horizon=horizon)

            # Final untouched test evaluation for selected model
            final_test_pred = final_model.predict(X_test)
            final_test_metrics = evaluate_predictions(
                y_test, final_test_pred, best_name, horizon, "test_final"
            )
            metric_rows.append(final_test_metrics)

            self.models_by_horizon_[horizon] = final_model
            self.model_names_by_horizon_[horizon] = best_name
            self.all_fitted_[horizon] = fitted
            self._store_importance(horizon, final_model, self.feature_cols_)

            # Per-battery error for final test
            try:
                bat_err = per_battery_errors(meta_test, y_test, final_test_pred)
                logger.info(
                    "Horizon %s best=%s val_rmse=%.5f | best battery=%s worst=%s",
                    horizon,
                    best_name,
                    val_rmse[best_name],
                    bat_err.iloc[0]["battery_id"] if len(bat_err) else None,
                    bat_err.iloc[-1]["battery_id"] if len(bat_err) else None,
                )
            except Exception:
                pass

        self.metrics_df_ = metrics_table(metric_rows)
        self.best_by_horizon_ = best_model_per_horizon(self.metrics_df_, split="validation")
        # Attach final test rmse for banner
        if self.best_by_horizon_ is not None and len(self.best_by_horizon_):
            test_final = self.metrics_df_[self.metrics_df_["split"] == "test_final"]
            test_map = {
                int(r["horizon"]): r["rmse"]
                for _, r in test_final.iterrows()
                if r["model"] == self.model_names_by_horizon_.get(int(r["horizon"]))
            }
            self.best_by_horizon_["test_rmse"] = self.best_by_horizon_["horizon"].map(
                lambda h: test_map.get(int(h))
            )

        self.training_timestamp_ = datetime.now(timezone.utc).isoformat()
        self._write_outputs()
        return self

    def _maybe_tune(self, name: str, model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        tcfg = self.config.get("tuning", {})
        if not tcfg.get("enabled", False):
            return model
        if name not in ("RandomForest", "ExtraTrees", "Ridge", "HistGradientBoosting"):
            return model
        est, dist = get_estimator_for_tuning(name, seed=int(self.config.get("random_seed", 42)))
        if est is None or dist is None:
            return model
        n_iter = int(tcfg.get("n_iter", 15))
        n_splits = min(int(tcfg.get("cv_splits", 3)), max(2, len(X_train) // 50))
        try:
            search = RandomizedSearchCV(
                est,
                param_distributions=dist,
                n_iter=n_iter,
                cv=TimeSeriesSplit(n_splits=n_splits),
                scoring="neg_root_mean_squared_error",
                random_state=int(self.config.get("random_seed", 42)),
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            logger.info("Tuned %s best_score=%.5f params=%s", name, -search.best_score_, search.best_params_)
            return search.best_estimator_
        except Exception as exc:
            logger.warning("Tuning failed for %s: %s", name, exc)
            return model

    def _refit_selected(
        self,
        best_name: str,
        fitted: dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        seed: int,
        horizon: int,
    ) -> Any:
        if best_name == "WeightedEnsemble":
            # Rebuild weights from already known component models refit on tr+val
            base = {k: clone(v) if not isinstance(v, WeightedEnsemble) else v for k, v in fitted.items() if k != "WeightedEnsemble" and k != "StackingEnsemble"}
            # Fit each
            fitted_new = {}
            val_proxy = {}
            # Use last 20% of X as pseudo-val for weight stability (still chronological within concat)
            n = len(X)
            cut = max(1, int(n * 0.8))
            X_a, y_a = X.iloc[:cut], y.iloc[:cut]
            X_b, y_b = X.iloc[cut:], y.iloc[cut:]
            for name, est in base.items():
                m = clone(est) if not isinstance(est, WeightedEnsemble) else est
                try:
                    m.fit(X_a, y_a)
                    pred = m.predict(X_b)
                    val_proxy[name] = float(np.sqrt(np.mean((np.asarray(y_b) - pred) ** 2)))
                    m.fit(X, y)
                    fitted_new[name] = m
                except Exception:
                    continue
            if len(fitted_new) >= 2:
                ens = build_weighted_ensemble(fitted_new, val_proxy, top_k=4)
                self.ensemble_weights_[horizon] = dict(ens.weights)
                return ens
        model = fitted[best_name]
        if isinstance(model, WeightedEnsemble):
            return model
        try:
            return clone(model).fit(X, y)
        except Exception:
            model.fit(X, y)
            return model

    def _store_importance(self, horizon: int, model: Any, feature_cols: list[str]) -> None:
        try:
            est = unwrap_model(model)
            if hasattr(est, "feature_importances_"):
                imps = est.feature_importances_
                self.feature_importances_[horizon] = {
                    feature_cols[i]: float(imps[i]) for i in range(min(len(feature_cols), len(imps)))
                }
            elif hasattr(est, "coef_"):
                coef = np.ravel(est.coef_)
                self.feature_importances_[horizon] = {
                    feature_cols[i]: float(abs(coef[i])) for i in range(min(len(feature_cols), len(coef)))
                }
            elif isinstance(model, WeightedEnsemble):
                # average importances across tree members if available
                agg: dict[str, float] = {}
                for name, est2 in model.estimators.items():
                    u = unwrap_model(est2)
                    if hasattr(u, "feature_importances_"):
                        for i, col in enumerate(feature_cols):
                            if i < len(u.feature_importances_):
                                agg[col] = agg.get(col, 0.0) + float(u.feature_importances_[i]) * model.weights.get(
                                    name, 0
                                )
                if agg:
                    self.feature_importances_[horizon] = agg
        except Exception as exc:
            logger.debug("Could not extract importances: %s", exc)

    def _write_outputs(self) -> None:
        reports_dir = project_path(self.config.get("paths", {}).get("reports_dir", "outputs/reports"))
        figures_dir = project_path(self.config.get("paths", {}).get("figures_dir", "outputs/figures"))
        ensure_dir(reports_dir)
        ensure_dir(figures_dir)

        dataset_info = {
            "Batteries": int(self.prepared_["battery_id"].nunique()) if self.prepared_ is not None else 0,
            "Cycles": int(len(self.prepared_)) if self.prepared_ is not None else 0,
            "Target": self.target_name_,
            "Features": len(self.feature_cols_),
        }
        if self.quality_report_:
            dataset_info["MissingColumns"] = len(self.quality_report_.get("missing_values", {}))

        battery_snapshot = None
        knee_cycle = None
        eol_cycle = None
        forecast = None
        if self.prepared_ is not None and len(self.prepared_):
            bid = str(self.prepared_["battery_id"].iloc[0])
            try:
                forecast = self.forecast(battery_id=bid, future_cycles=100)
                hist = self.prepared_[self.prepared_["battery_id"] == bid]
                knee = detect_degradation_knee(
                    hist["cycle_number"].to_numpy(),
                    hist["soh"].to_numpy(),
                    **{k: v for k, v in self.config.get("knee", {}).items() if k in (
                        "smoothing_window", "slope_window", "acceleration_threshold", "persistence_cycles"
                    )},
                )
                knee_cycle = knee.get("estimated_knee_cycle")
                eol = forecast.attrs.get("eol") or {}
                eol_cycle = eol.get("eol_cycle")
                battery_snapshot = {
                    "Cycle": int(hist["cycle_number"].iloc[-1]),
                    "Capacity": f"{hist['capacity_ah'].iloc[-1]:.2f} Ah",
                    "SOH": f"{hist['soh'].iloc[-1]*100:.1f}%",
                    "Estimated EOL": f"Cycle {eol_cycle:.0f}" if eol_cycle else "N/A",
                    "Estimated RUL": f"{eol.get('rul'):.0f} cycles" if eol.get("rul") is not None else "N/A",
                }
            except Exception as exc:
                logger.warning("Post-train forecast for report failed: %s", exc)

        write_metrics_csv(self.metrics_df_, reports_dir / "model_metrics.csv")
        write_model_summary(
            reports_dir / "model_summary.txt",
            dataset_info=dataset_info,
            best_by_horizon=self.best_by_horizon_,
            metrics_df=self.metrics_df_,
            battery_snapshot=battery_snapshot,
            warnings=self.quality_report_.get("warnings", []),
        )
        print_training_banner(dataset_info, self.best_by_horizon_, battery_snapshot)

        # Static figures
        residual_data = None
        fi = None
        if self.feature_importances_:
            h = min(self.feature_importances_)
            items = sorted(self.feature_importances_[h].items(), key=lambda kv: kv[1], reverse=True)[:20]
            fi = ([k for k, _ in items], np.array([v for _, v in items]))
        try:
            bid = str(self.prepared_["battery_id"].iloc[0])
            generate_all_static_figures(
                self.featured_ if self.featured_ is not None else self.prepared_,
                self.metrics_df_,
                forecast,
                figures_dir,
                battery_id=bid,
                eol_soh=float(self.config.get("battery", {}).get("eol_soh", 0.80)),
                knee_cycle=knee_cycle,
                eol_cycle=eol_cycle,
                feature_importance=fi,
                residual_data=residual_data,
            )
        except Exception as exc:
            logger.warning("Figure generation issue: %s", exc)

        self.save()

    # -------------------------------------------------------------- evaluate
    def evaluate(self) -> pd.DataFrame:
        if self.metrics_df_ is None:
            raise RuntimeError("No metrics available. Call fit() or load() first.")
        return self.metrics_df_.copy()

    # -------------------------------------------------------------- forecast
    def forecast(
        self,
        battery_id: Optional[str] = None,
        future_cycles: int = 100,
        method: str = "direct",
        start_cycle: Optional[int] = None,
        smoothing_window: int = 5,
        apply_monotonic: bool = False,
        future_profile: Optional[dict[str, float]] = None,
        show_uncertainty: bool = True,
    ) -> pd.DataFrame:
        if self.prepared_ is None or not self.models_by_horizon_:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        df = self.prepared_
        if battery_id is None:
            battery_id = str(df["battery_id"].iloc[0])
        hist = df[df["battery_id"] == battery_id].sort_values("cycle_number").copy()
        if start_cycle is not None:
            hist = hist[hist["cycle_number"] <= start_cycle]
        if len(hist) < self.config.get("features", {}).get("min_history_cycles", 20):
            raise ValueError(
                f"At least {self.config.get('features', {}).get('min_history_cycles', 20)} "
                "historical cycles are required to generate this forecast."
            )

        ref = float(self.reference_capacities_.get(str(battery_id), hist["reference_capacity"].iloc[-1]))
        current_cycle = float(hist["cycle_number"].iloc[-1])
        current_capacity = float(hist["capacity_ah"].iloc[-1])
        eol_soh = float(self.config.get("battery", {}).get("eol_soh", 0.80))
        horizons = sorted(self.models_by_horizon_.keys())

        # Build feature row at last cycle
        featured_hist = build_features(
            hist.assign(battery_id=battery_id) if "battery_id" not in hist.columns else hist,
            self.config,
        )
        # Rebuild properly:
        from battery_degradation.features import build_features_for_battery

        feat_cfg = self.config.get("features", {})
        featured_hist = build_features_for_battery(
            hist,
            lag_cycles=feat_cfg.get("lag_cycles", [1, 2, 3, 5, 10]),
            rolling_windows=feat_cfg.get("rolling_windows", [3, 5, 10, 20, 50]),
            slope_windows=feat_cfg.get("slope_windows", [5, 10, 20, 50]),
            high_temp_threshold=float(self.config.get("battery", {}).get("high_temperature_threshold", 40.0)),
        )
        last_row = featured_hist.iloc[[-1]].reindex(columns=self.feature_cols_).fillna(0.0)

        unc = dict(self.config.get("uncertainty", {}))
        unc["enabled"] = show_uncertainty and unc.get("enabled", True)
        constraints = dict(self.config.get("forecasting", {}))

        if method == "recursive":
            model_h1 = self.models_by_horizon_.get(1) or next(iter(self.models_by_horizon_.values()))
            forecast = recursive_forecast(
                hist,
                model_h1,
                self.feature_cols_,
                n_steps=future_cycles,
                config=self.config,
                reference_capacity=ref,
                exogenous_strategy=self.config.get("forecasting", {}).get("recursive_exogenous_strategy", "carry_last"),
                future_profile=future_profile,
            )
        else:
            # direct or hybrid: use direct anchors then optionally blend with recursive trend
            use_horizons = [h for h in horizons if h <= future_cycles]
            if max(horizons) < future_cycles:
                use_horizons = horizons
            forecast = direct_multi_horizon_forecast(
                last_row,
                self.models_by_horizon_,
                horizons=use_horizons or horizons,
                reference_capacity=ref,
                current_cycle=current_cycle,
                current_capacity=current_capacity,
                eol_soh=eol_soh,
                uncertainty_cfg=unc,
                constraints_cfg=constraints,
            )
            if future_cycles < len(forecast):
                forecast = forecast.iloc[:future_cycles].copy()
            elif future_cycles > len(forecast) and 1 in self.models_by_horizon_:
                # extend with recursive beyond last direct horizon
                extra = recursive_forecast(
                    hist,
                    self.models_by_horizon_[1],
                    self.feature_cols_,
                    n_steps=future_cycles,
                    config=self.config,
                    reference_capacity=ref,
                    future_profile=future_profile,
                )
                forecast = extra.iloc[:future_cycles].copy()
                forecast["method"] = "hybrid" if method == "hybrid" else forecast.get("method", "direct")

            if method == "hybrid" and 1 in self.models_by_horizon_:
                rec = recursive_forecast(
                    hist,
                    self.models_by_horizon_[1],
                    self.feature_cols_,
                    n_steps=min(future_cycles, len(forecast)),
                    config=self.config,
                    reference_capacity=ref,
                    future_profile=future_profile,
                )
                # Blend: short horizon favor direct/tree, long favor recursive/trend already in ensemble
                n = min(len(forecast), len(rec))
                weights = np.linspace(0.8, 0.4, n)  # still mostly direct; ensemble already hybridized
                forecast = forecast.iloc[:n].copy()
                forecast["predicted_capacity"] = (
                    weights * forecast["predicted_capacity"].to_numpy()
                    + (1 - weights) * rec["predicted_capacity"].to_numpy()[:n]
                )
                forecast["predicted_soh"] = forecast["predicted_capacity"] / max(ref, 1e-12)
                forecast["method"] = "hybrid"

        forecast = postprocess_forecast(forecast, smoothing_window=smoothing_window, apply_monotonic=apply_monotonic)
        # Ensure RUL/EOL attrs
        if "attrs" not in dir(forecast) or not forecast.attrs.get("eol"):
            eol = estimate_eol_from_trajectory(
                np.concatenate([[current_cycle], forecast["cycle_number"].to_numpy()]),
                np.concatenate([[current_capacity / max(ref, 1e-12)], forecast["predicted_soh"].to_numpy()]),
                eol_soh=eol_soh,
                current_cycle=current_cycle,
            )
            forecast.attrs["eol"] = eol
            if "estimated_rul" not in forecast.columns:
                from battery_degradation.rul import rul_at_cycle

                forecast["estimated_rul"] = [rul_at_cycle(eol.get("eol_cycle"), c) for c in forecast["cycle_number"]]
        return forecast

    def plot_forecast(self, battery_id: Optional[str] = None, future_cycles: int = 100, path: Optional[Path] = None):
        from battery_degradation.visualization import plot_forecast as _plot

        forecast = self.forecast(battery_id=battery_id, future_cycles=future_cycles)
        if battery_id is None:
            battery_id = str(self.prepared_["battery_id"].iloc[0])
        hist = self.prepared_[self.prepared_["battery_id"] == battery_id]
        path = path or project_path("outputs", "figures", f"forecast_{battery_id}.png")
        eol = forecast.attrs.get("eol", {})
        return _plot(hist, forecast, path, eol_cycle=eol.get("eol_cycle"))

    def battery_kpis(self, battery_id: str) -> dict[str, Any]:
        if self.prepared_ is None:
            raise RuntimeError("No data loaded")
        hist = self.prepared_[self.prepared_["battery_id"] == battery_id].sort_values("cycle_number")
        if hist.empty:
            raise ValueError(f"Unknown battery_id: {battery_id}")
        ref = float(hist["reference_capacity"].iloc[-1])
        cur_cycle = int(hist["cycle_number"].iloc[-1])
        cur_cap = float(hist["capacity_ah"].iloc[-1])
        cur_soh = float(hist["soh"].iloc[-1])
        knee = detect_degradation_knee(hist["cycle_number"].to_numpy(), hist["soh"].to_numpy())
        try:
            fc = self.forecast(battery_id=battery_id, future_cycles=100)
            eol = fc.attrs.get("eol", {})
        except Exception:
            eol = estimate_eol_from_trajectory(
                hist["cycle_number"].to_numpy(), hist["soh"].to_numpy(),
                eol_soh=float(self.config.get("battery", {}).get("eol_soh", 0.80)),
            )
        slope = np.nan
        if len(hist) >= 10:
            slope = float(np.polyfit(hist["cycle_number"].tail(20), hist["capacity_ah"].tail(20), 1)[0])
        kpis = {
            "current_cycle": cur_cycle,
            "current_capacity": cur_cap,
            "current_soh": cur_soh,
            "current_soh_pct": cur_soh * 100,
            "capacity_fade_pct": (1 - cur_soh) * 100,
            "estimated_rul": eol.get("rul"),
            "estimated_eol_cycle": eol.get("eol_cycle"),
            "estimated_knee_cycle": knee.get("estimated_knee_cycle"),
            "knee_confidence": knee.get("knee_confidence"),
            "degradation_rate": slope,
            "reference_capacity": ref,
        }
        if "internal_resistance" in hist.columns:
            kpis["latest_internal_resistance"] = float(pd.to_numeric(hist["internal_resistance"], errors="coerce").iloc[-1])
        if "temperature_mean" in hist.columns:
            kpis["average_temperature"] = float(pd.to_numeric(hist["temperature_mean"], errors="coerce").mean())
        kpis["flags"] = engineering_flags(
            hist,
            eol_soh=float(self.config.get("battery", {}).get("eol_soh", 0.80)),
            high_temp_threshold=float(self.config.get("battery", {}).get("high_temperature_threshold", 40.0)),
        )
        return kpis

    # ----------------------------------------------------------- persistence
    def save(self, models_dir: Optional[Union[str, Path]] = None) -> Path:
        models_dir = Path(models_dir) if models_dir else project_path(
            self.config.get("paths", {}).get("models_dir", "models")
        )
        ensure_dir(models_dir)
        for horizon, model in self.models_by_horizon_.items():
            joblib.dump(model, models_dir / f"horizon_{horizon}_best.joblib")
        meta = {
            "feature_cols": self.feature_cols_,
            "model_names_by_horizon": {str(k): v for k, v in self.model_names_by_horizon_.items()},
            "reference_capacities": self.reference_capacities_,
            "schema_mapping": self.schema_mapping_,
            "eol_soh": self.config.get("battery", {}).get("eol_soh", 0.80),
            "training_timestamp": self.training_timestamp_,
            "target_name": self.target_name_,
            "ensemble_weights": {str(k): v for k, v in self.ensemble_weights_.items()},
            "feature_importances": {str(k): v for k, v in self.feature_importances_.items()},
            "config": self.config,
            "metrics_records": self.metrics_df_.to_dict(orient="records") if self.metrics_df_ is not None else [],
            "cycle_range": {
                "min": float(self.prepared_["cycle_number"].min()) if self.prepared_ is not None else None,
                "max": float(self.prepared_["cycle_number"].max()) if self.prepared_ is not None else None,
            },
        }
        # Persist prepared data for dashboard convenience
        if self.prepared_ is not None:
            processed_cfg = self.config.get("paths", {}).get("processed_dir", "data/processed")
            processed = Path(processed_cfg) if Path(processed_cfg).is_absolute() else project_path(processed_cfg)
            ensure_dir(processed)
            prepared_path = processed / "prepared_cycles.csv"
            self.prepared_.to_csv(prepared_path, index=False)
            meta["prepared_path"] = str(prepared_path)
        with (models_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        if self.metrics_df_ is not None:
            self.metrics_df_.to_csv(models_dir / "metrics.csv", index=False)
        logger.info("Saved models to %s", models_dir)
        return models_dir

    def load(self, models_dir: Optional[Union[str, Path]] = None) -> "BatteryPredictor":
        models_dir = Path(models_dir) if models_dir else project_path(
            self.config.get("paths", {}).get("models_dir", "models")
        )
        meta_path = models_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json in {models_dir}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_cols_ = meta.get("feature_cols", [])
        self.model_names_by_horizon_ = {int(k): v for k, v in meta.get("model_names_by_horizon", {}).items()}
        self.reference_capacities_ = meta.get("reference_capacities", {})
        self.schema_mapping_ = meta.get("schema_mapping", {})
        self.training_timestamp_ = meta.get("training_timestamp")
        self.target_name_ = meta.get("target_name", "capacity")
        self.ensemble_weights_ = {int(k): v for k, v in meta.get("ensemble_weights", {}).items()}
        self.feature_importances_ = {int(k): v for k, v in meta.get("feature_importances", {}).items()}
        if meta.get("config"):
            self.config = meta["config"]
        self.models_by_horizon_ = {}
        for horizon in self.model_names_by_horizon_:
            path = models_dir / f"horizon_{horizon}_best.joblib"
            if path.exists():
                self.models_by_horizon_[horizon] = joblib.load(path)
        if meta.get("metrics_records"):
            self.metrics_df_ = pd.DataFrame(meta["metrics_records"])
            self.best_by_horizon_ = best_model_per_horizon(self.metrics_df_, split="validation")
        prepared_path = meta.get("prepared_path")
        if prepared_path and Path(prepared_path).exists():
            self.prepared_ = pd.read_csv(prepared_path)
        logger.info("Loaded models from %s", models_dir)
        return self
