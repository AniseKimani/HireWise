"""Small shared helpers for the Day 3 synthetic-data generator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def apportion_with_floor(weights: pd.Series, total: int, floor: int = 0) -> pd.Series:
    """Deterministic largest-remainder apportionment of `total` integer
    units across `weights.index`, in proportion to `weights`, with every
    entry guaranteed at least `floor` units. Raises if `floor * n > total`.

    Used for both "N clients per country" (floor=1, so every country with
    at least one request gets a client) and "N workers per category"
    (floor=20, per docs/experimental_design.md Section 8).
    """
    n = len(weights)
    if floor * n > total:
        raise ValueError(f"floor {floor} * {n} entries exceeds total {total}")

    remaining = total - floor * n
    if remaining == 0 or weights.sum() == 0:
        shares = pd.Series(0.0, index=weights.index)
    else:
        shares = weights / weights.sum() * remaining

    base = np.floor(shares).astype(int) + floor
    leftover = total - base.sum()
    if leftover > 0:
        fractional = shares - np.floor(shares)
        order = sorted(weights.index, key=lambda k: (-fractional[k], k))
        for key in order[:leftover]:
            base[key] += 1
    return base


def standardize(values: np.ndarray) -> np.ndarray:
    """Z-score standardisation; constant input maps to all zeros rather
    than dividing by zero."""
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.mean()) / std


def stable_id(prefix: str, index: int, width: int = 6) -> str:
    return f"{prefix}{index + 1:0{width}d}"
