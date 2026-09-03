"""Configuration loading and helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml

from battery_degradation.utils import PROJECT_ROOT, get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: Optional[Path | str] = None, overrides: Optional[dict] = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if overrides:
        config = deep_merge(config, overrides)
    return config


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
