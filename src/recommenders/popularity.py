"""Training-history-only Popularity baseline (Day 4 brief Section 6).

Popularity signal: count of TRAINING-split `HIRED` interactions per
worker (docs/experimental_design.md Section 15's "Popularity (training
hire counts)"). Validation/test interactions are never counted. Workers
with zero training hires -- including, by construction, every cold-start
worker -- tie at 0 and are broken deterministically by worker_id.
"""
from __future__ import annotations

import pandas as pd

from src.recommenders.base import BaseRecommender


class PopularityBaseline(BaseRecommender):
    name = "popularity"

    def __init__(self, interactions: pd.DataFrame, requests: pd.DataFrame):
        train_links = set(requests.loc[requests["split"] == "train", "link"])
        train_hires = interactions.loc[
            (interactions["event"] == "HIRED") & interactions["link"].isin(train_links)
        ]
        self.hire_counts: dict[str, int] = train_hires.groupby("worker_id").size().to_dict()

    def rank(self, link: str, candidates: list[str], k: int) -> list[str]:
        ranked = sorted(candidates, key=lambda w: (-self.hire_counts.get(w, 0), w))
        return ranked[:k]
