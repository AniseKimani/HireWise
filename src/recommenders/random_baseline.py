"""Deterministic Random baseline (Day 4 brief Section 5).

Provides the chance-performance floor: candidates are shuffled with a
generator seeded from a hash of (project master seed, request link), so
rankings are fully reproducible across runs and independent of call
order or batching, without ever touching generator-private data.
"""
from __future__ import annotations

import hashlib

import numpy as np

from src.recommenders.base import BaseRecommender


def _link_seed(master_seed: int, link: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{link}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class RandomBaseline(BaseRecommender):
    name = "random"

    def __init__(self, seed: int):
        self.seed = seed

    def rank(self, link: str, candidates: list[str], k: int) -> list[str]:
        if not candidates:
            return []
        rng = np.random.default_rng(np.random.SeedSequence(_link_seed(self.seed, link)))
        order = rng.permutation(len(candidates))
        return [candidates[i] for i in order][:k]
