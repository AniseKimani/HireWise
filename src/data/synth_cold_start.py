"""New-entrant cold-start cohort (docs/experimental_design.md Section 12).

A stratified random 10% of workers (by primary category) form the
new-entrant cohort. Their `join_day` is set to the first day of the
validation period, so the chronological interaction simulation never
places them in a training-period applicant pool -- they have zero
training interactions **by construction**, not by deleting history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.config import DataConfig
from src.data.synth_util import apportion_with_floor
from src.data.synth_rng import stage_rng

ESTABLISHED_JOIN_DAY = 0.0


def select_cold_start_cohort(workers_df: pd.DataFrame, config: DataConfig) -> set[str]:
    category_counts = workers_df["primary_category"].value_counts()
    n_cold_start = round(len(workers_df) * config.cold_start_ratio)
    quota = apportion_with_floor(category_counts, total=n_cold_start, floor=0)

    rng = stage_rng(config.seed, "cold_start_cohort")
    chosen: set[str] = set()
    for category in sorted(quota.index):
        n = int(quota[category])
        if n <= 0:
            continue
        pool = workers_df.loc[workers_df["primary_category"] == category, "worker_id"].sort_values().to_numpy()
        picked = rng.choice(pool, size=min(n, len(pool)), replace=False)
        chosen.update(picked.tolist())
    return chosen


def assign_worker_join_days(
    workers_df: pd.DataFrame, cold_start_ids: set[str], validation_start_day: float
) -> pd.DataFrame:
    workers_df = workers_df.copy()
    workers_df["is_cold_start"] = workers_df["worker_id"].isin(cold_start_ids)
    workers_df["join_day"] = np.where(workers_df["is_cold_start"], validation_start_day, ESTABLISHED_JOIN_DAY)
    return workers_df
