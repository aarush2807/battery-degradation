"""Dashboard view modules (Streamlit UI pages).

Named ``views`` instead of ``pages`` so Streamlit multipage auto-discovery
does not treat these as standalone blank routes.
"""

from . import (
    battery_explorer,
    data_quality,
    diagnostics,
    forecasting,
    model_performance,
    overview,
)

__all__ = [
    "battery_explorer",
    "data_quality",
    "diagnostics",
    "forecasting",
    "model_performance",
    "overview",
]
