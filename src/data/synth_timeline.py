"""Simulated timeline and temporal train/validation/test split
(docs/experimental_design.md Sections 7.6 and 11).

The real dataset's collection window is too short and too bursty to use
directly (docs/experimental_design.md Section 2, Limitation 3), so every
request gets a deterministic simulated day on a `simulated_days`-day
calendar: each client has an activity start day, and its requests are
placed uniformly between that start day and the end of the calendar.

The train/validation/test split is a **global temporal split on this
simulated day**, not a random split after the fact: sort all requests by
simulated day, then cut at the configured ratios. This guarantees no
validation/test request can appear "before" a training request.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.config import DataConfig
from src.data.generator_config import GeneratorConfig
from src.data.synth_rng import stage_rng


def assign_timeline(
    assignment_df: pd.DataFrame, clients_df: pd.DataFrame, config: DataConfig, gen_config: GeneratorConfig
) -> pd.DataFrame:
    """Return assignment_df with an added `simulated_day` column (float,
    range [client's activity_start_day, gen_config.simulated_days])."""
    rng = stage_rng(config.seed, "timeline")
    merged = assignment_df.merge(
        clients_df[["client_id", "activity_start_day"]], on="client_id", how="left"
    ).sort_values("link").reset_index(drop=True)

    merged["simulated_day"] = rng.uniform(
        merged["activity_start_day"].to_numpy(),
        gen_config.simulated_days,
    )
    return merged.drop(columns=["activity_start_day"])


def compute_temporal_split(timeline_df: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, dict]:
    """Assign split='train'/'validation'/'test' by simulated_day rank
    (ties broken by `link` for determinism), at the configured ratios.
    Returns (timeline_df_with_split, boundaries), where boundaries records
    the simulated day at each cut -- `validation_start_day` is used as the
    cold-start cohort's join day (Section 12)."""
    n = len(timeline_df)
    n_train = round(n * config.train_ratio)
    n_validation = round(n * config.validation_ratio)
    n_test = n - n_train - n_validation

    ordered = timeline_df.sort_values(["simulated_day", "link"]).reset_index(drop=True)
    split = np.empty(n, dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_validation] = "validation"
    split[n_train + n_validation:] = "test"
    ordered["split"] = split

    boundaries = {
        "n_train": int(n_train),
        "n_validation": int(n_validation),
        "n_test": int(n_test),
        "train_end_day": float(ordered.loc[n_train - 1, "simulated_day"]),
        "validation_start_day": float(ordered.loc[n_train, "simulated_day"]),
        "validation_end_day": float(ordered.loc[n_train + n_validation - 1, "simulated_day"]),
        "test_start_day": float(ordered.loc[n_train + n_validation, "simulated_day"]),
    }
    return ordered.sort_values("link").reset_index(drop=True), boundaries
