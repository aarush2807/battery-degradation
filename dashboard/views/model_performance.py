"""Model performance comparison page."""

from __future__ import annotations

import streamlit as st

from battery_degradation.pipeline import BatteryPredictor
from dashboard.components.charts import (
    plot_ensemble_weights,
    plot_feature_importance,
    plot_horizon_performance,
    plot_model_comparison,
)
from dashboard.components.tables import download_csv_button, show_dataframe


def render(ctx: dict) -> None:
    st.header("Model Performance")
    st.caption("Validation-based model selection; test metrics are holdout-only.")

    prepared = ctx.get("prepared")
    predictor = ctx.get("predictor")

    with st.expander("Train models from dashboard", expanded=predictor is None or not getattr(predictor, "models_by_horizon_", None)):
        st.write("Training runs only when explicitly triggered.")
        eol = st.number_input("EOL SOH", 0.7, 0.95, float(ctx.get("eol", 0.8)), 0.01)
        enable_tuning = st.checkbox("Enable hyperparameter tuning", value=False)
        families = st.multiselect(
            "Model families",
            ["random_forest", "extra_trees", "gradient_boosting", "ridge", "ensemble"],
            default=["random_forest", "extra_trees", "gradient_boosting", "ridge", "ensemble"],
        )
        if st.button("Train Models"):
            if prepared is None:
                st.error("Load data first.")
            else:
                cfg = dict(st.session_state.config)
                cfg.setdefault("battery", {})["eol_soh"] = eol
                cfg.setdefault("tuning", {})["enabled"] = enable_tuning
                # Disable unselected
                for key in ["random_forest", "extra_trees", "gradient_boosting", "ridge", "ensemble", "lasso", "elastic_net", "polynomial_regression", "stacking"]:
                    cfg.setdefault("models", {})[key] = key in families or key in (
                        "lasso",
                        "elastic_net",
                        "polynomial_regression",
                        "linear_regression",
                        "last_value",
                        "historical_linear",
                    )
                # Keep baselines always
                cfg["models"]["last_value"] = True
                cfg["models"]["historical_linear"] = True
                cfg["models"]["linear_regression"] = True
                progress = st.progress(0)
                status = st.status("Training...", expanded=True)
                try:
                    pred = BatteryPredictor(config=cfg)
                    # Save prepared path via fit on dataframe
                    with status:
                        st.write("Fitting leakage-safe feature pipeline and models...")
                        pred.fit(prepared, target=ctx.get("target", "capacity"))
                        progress.progress(100)
                    st.session_state.predictor = pred
                    st.session_state.config = cfg
                    st.success("Training complete. Models saved to models/.")
                    predictor = pred
                except Exception as exc:
                    status.update(label="Training failed", state="error")
                    st.error(str(exc))

    predictor = st.session_state.get("predictor") or predictor
    if predictor is None or predictor.metrics_df_ is None or predictor.metrics_df_.empty:
        st.warning("No metrics available. Train models to populate this page.")
        return

    metrics = predictor.metrics_df_.copy()
    val = metrics[metrics["split"] == "validation"]

    # Best model cards
    horizons_show = [1, 10, 50, 100]
    cols = st.columns(len(horizons_show))
    for col, h in zip(cols, horizons_show):
        sub = val[val["horizon"] == h]
        if sub.empty:
            col.metric(f"Best @ {h} cycles", "N/A")
            continue
        best = sub.loc[sub["rmse"].idxmin()]
        col.metric(f"Best @ {h} cycles", str(best["model"]), delta=f"RMSE {best['rmse']:.4f}")

    metric_choice = st.selectbox("Metric", ["rmse", "mae", "r2"])
    models_filter = st.multiselect("Filter models", sorted(val["model"].unique()), default=sorted(val["model"].unique()))
    horizons_filter = st.multiselect(
        "Filter horizons", sorted(val["horizon"].unique()), default=sorted(val["horizon"].unique())
    )
    filtered = metrics[
        (metrics["model"].isin(models_filter))
        & (metrics["horizon"].isin(horizons_filter))
        & (metrics["split"].isin(["validation", "test", "test_final"]))
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_model_comparison(filtered[filtered["split"] == "validation"], metric=metric_choice),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            plot_horizon_performance(filtered[filtered["split"] == "validation"], metric=metric_choice),
            use_container_width=True,
        )

    st.subheader("Model ranking table")
    st.caption("Ranks use validation RMSE. Test columns are for final reporting only.")
    show_dataframe(filtered.sort_values(["horizon", "split", "rmse"]))
    download_csv_button(filtered, "model_metrics.csv", "Download model metrics CSV")

    # Ensemble weights / importance
    if predictor.ensemble_weights_:
        h = int(ctx.get("horizon", 50))
        if h in predictor.ensemble_weights_:
            st.subheader(f"Ensemble weights — horizon {h}")
            st.plotly_chart(plot_ensemble_weights(predictor.ensemble_weights_[h]), use_container_width=True)
            st.json(predictor.ensemble_weights_[h])

    if predictor.feature_importances_:
        h = int(ctx.get("horizon", 1))
        if h in predictor.feature_importances_:
            st.subheader(f"Feature importance — horizon {h}")
            st.plotly_chart(plot_feature_importance(predictor.feature_importances_[h]), use_container_width=True)
