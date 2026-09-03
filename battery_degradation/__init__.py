"""Battery Degradation Prediction and Analytics System."""

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "BatteryPredictor":
        from battery_degradation.pipeline import BatteryPredictor

        return BatteryPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BatteryPredictor", "__version__"]
