#!/usr/bin/env python3
"""CLI entrypoint for battery degradation analytics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from battery_degradation.config import load_config
from battery_degradation.pipeline import BatteryPredictor
from battery_degradation.utils import ensure_dir, get_logger, project_path, set_global_seed

logger = get_logger("main")


def cmd_train(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else load_config()
    if args.eol is not None:
        config.setdefault("battery", {})["eol_soh"] = args.eol
    if args.no_tuning:
        config.setdefault("tuning", {})["enabled"] = False
    set_global_seed(config.get("random_seed", 42))
    predictor = BatteryPredictor(config=config)
    predictor.fit(args.data, target=args.target)
    print(f"Models saved under: {project_path('models')}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    predictor = BatteryPredictor(config=load_config(args.config) if args.config else load_config())
    models_dir = Path(args.models) if args.models else project_path("models")
    if (models_dir / "metadata.json").exists():
        predictor.load(models_dir)
        # Optionally refresh evaluation by refitting quickly is expensive; use saved metrics
        metrics = predictor.evaluate()
    else:
        predictor.fit(args.data)
        metrics = predictor.evaluate()
    out = Path(args.output) if args.output else project_path("outputs", "reports", "eval_metrics.csv")
    ensure_dir(out.parent)
    metrics.to_csv(out, index=False)
    print(metrics.head(30).to_string(index=False))
    print(f"\nWrote metrics -> {out}")


def cmd_forecast(args: argparse.Namespace) -> None:
    predictor = BatteryPredictor(config=load_config(args.config) if args.config else load_config())
    models_dir = project_path("models")
    if (models_dir / "metadata.json").exists():
        predictor.load(models_dir)
        # Ensure data context
        if predictor.prepared_ is None:
            from battery_degradation.data_loader import load_and_normalize
            from battery_degradation.preprocessing import prepare_dataset

            raw, _ = load_and_normalize(args.data)
            prepared, refs = prepare_dataset(raw, predictor.config)
            predictor.prepared_ = prepared
            predictor.reference_capacities_ = refs or predictor.reference_capacities_
    else:
        logger.info("No saved models found; training first...")
        predictor.fit(args.data)

    forecast = predictor.forecast(
        battery_id=args.battery_id,
        future_cycles=args.cycles,
        method=args.method,
    )
    out = Path(args.output) if args.output else project_path(
        "outputs", "predictions", f"forecast_{args.battery_id or 'default'}.csv"
    )
    ensure_dir(out.parent)
    forecast.to_csv(out, index=False)
    eol = forecast.attrs.get("eol", {})
    print(forecast.head(10).to_string(index=False))
    print(f"\n... {len(forecast)} forecast rows")
    print(f"Estimated EOL cycle: {eol.get('eol_cycle')}")
    print(f"Estimated RUL: {eol.get('rul')}")
    print(f"Wrote forecast -> {out}")
    try:
        fig = predictor.plot_forecast(battery_id=args.battery_id, future_cycles=args.cycles)
        print(f"Wrote figure -> {fig}")
    except Exception as exc:
        logger.warning("Could not write forecast figure: %s", exc)


def cmd_dashboard(args: argparse.Namespace) -> None:
    app = project_path("dashboard", "app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Battery Degradation Analytics CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train models")
    p_train.add_argument("--data", required=True, help="Path to cycle CSV")
    p_train.add_argument("--config", default=None)
    p_train.add_argument("--target", default="capacity", choices=["capacity", "soh"])
    p_train.add_argument("--eol", type=float, default=None)
    p_train.add_argument("--no-tuning", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate models")
    p_eval.add_argument("--data", required=True)
    p_eval.add_argument("--config", default=None)
    p_eval.add_argument("--models", default=None)
    p_eval.add_argument("--output", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_fc = sub.add_parser("forecast", help="Forecast future degradation")
    p_fc.add_argument("--data", required=True)
    p_fc.add_argument("--battery-id", default="BATTERY_001")
    p_fc.add_argument("--cycles", type=int, default=100)
    p_fc.add_argument("--method", default="direct", choices=["direct", "recursive", "hybrid"])
    p_fc.add_argument("--output", default=None)
    p_fc.add_argument("--config", default=None)
    p_fc.set_defaults(func=cmd_forecast)

    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
