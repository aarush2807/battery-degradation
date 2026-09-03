"""Time-series aware validation splits (no random shuffling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from battery_degradation.utils import get_logger

logger = get_logger(__name__)


@dataclass
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_split_indices(
    n: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitIndices:
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("train/validation/test fractions must sum to 1")
    if n < 10:
        raise ValueError(f"Need at least 10 samples for chronological split, got {n}")
    n_train = int(n * train_fraction)
    n_val = int(n * validation_fraction)
    # remainder to test
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        # fallback minimal sizes
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)
        n_train = n - n_val - n_test
    train = np.arange(0, n_train)
    validation = np.arange(n_train, n_train + n_val)
    test = np.arange(n_train + n_val, n)
    return SplitIndices(train=train, validation=validation, test=test)


def within_battery_temporal_split(
    meta: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitIndices:
    """
    Split each battery chronologically, then concatenate index positions
    relative to the provided meta row order (assumed sorted by battery, cycle).
    """
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for _, group in meta.groupby("battery_id", sort=False):
        positions = group.index.to_numpy()
        # Use positional order within group after sorting by cycle
        order = np.argsort(group["cycle_number"].to_numpy())
        ordered_positions = positions[order]
        splits = chronological_split_indices(
            len(ordered_positions), train_fraction, validation_fraction, test_fraction
        )
        train_idx.extend(ordered_positions[splits.train].tolist())
        val_idx.extend(ordered_positions[splits.validation].tolist())
        test_idx.extend(ordered_positions[splits.test].tolist())

    return SplitIndices(
        train=np.array(train_idx, dtype=int),
        validation=np.array(val_idx, dtype=int),
        test=np.array(test_idx, dtype=int),
    )


def cross_battery_split(
    meta: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Hold out entire battery IDs for val/test generalization checks."""
    batteries = sorted(meta["battery_id"].astype(str).unique())
    if len(batteries) < 3:
        logger.warning(
            "Fewer than 3 batteries; falling back to within-battery temporal split."
        )
        return within_battery_temporal_split(
            meta, train_fraction, validation_fraction, test_fraction
        )
    rng = np.random.default_rng(seed)
    shuffled = list(batteries)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * validation_fraction))
    train_b = set(shuffled[:n_train])
    val_b = set(shuffled[n_train : n_train + n_val])
    test_b = set(shuffled[n_train + n_val :])
    if not test_b:
        # ensure nonempty test
        moved = val_b.pop()
        test_b.add(moved)

    def mask(ids: set[str]) -> np.ndarray:
        return meta.index[meta["battery_id"].astype(str).isin(ids)].to_numpy()

    return SplitIndices(train=mask(train_b), validation=mask(val_b), test=mask(test_b))


def make_split(
    meta: pd.DataFrame,
    mode: str = "within_battery",
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    if mode == "cross_battery":
        return cross_battery_split(
            meta, train_fraction, validation_fraction, test_fraction, seed=seed
        )
    return within_battery_temporal_split(
        meta, train_fraction, validation_fraction, test_fraction
    )


def expanding_window_splits(
    n: int,
    n_splits: int = 5,
    min_train_size: Optional[int] = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) using sklearn TimeSeriesSplit semantics."""
    if n_splits < 2:
        n_splits = 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tscv.split(np.arange(n)):
        if min_train_size and len(train_idx) < min_train_size:
            continue
        yield train_idx, val_idx


def assert_no_future_leakage(meta: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray) -> None:
    """For within-battery checks: max train cycle < min val cycle per battery."""
    train_meta = meta.loc[train_idx]
    val_meta = meta.loc[val_idx]
    for battery_id in val_meta["battery_id"].unique():
        t_max = train_meta.loc[train_meta["battery_id"] == battery_id, "cycle_number"]
        v_min = val_meta.loc[val_meta["battery_id"] == battery_id, "cycle_number"]
        if t_max.empty or v_min.empty:
            continue
        if t_max.max() >= v_min.min():
            raise AssertionError(
                f"Temporal leakage for battery {battery_id}: "
                f"train max cycle {t_max.max()} >= val min cycle {v_min.min()}"
            )
