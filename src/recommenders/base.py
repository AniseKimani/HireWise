"""Shared recommender interface. Every Day 4+ recommender (Random,
Popularity, CBF, and later CF/KG/Hybrid) implements `rank`, taking a
request identifier and its candidate universe and returning an ordered
list of up to `k` worker IDs, most-recommended first, with no duplicates.

Recommenders never receive generator-private data: only what the caller
(the evaluator, via src/evaluation/protocol.py) passes to `rank`, which
is built exclusively from model-visible files.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    name: str

    @abstractmethod
    def rank(self, link: str, candidates: list[str], k: int) -> list[str]:
        """Return up to `k` worker IDs from `candidates`, ranked
        best-first, with no duplicates. `candidates` is already the full
        eligible pool for `link` (see src/evaluation/protocol.py); this
        method only ranks and truncates, it does not re-filter
        eligibility."""
        raise NotImplementedError

    def rank_batch(self, requests: dict[str, list[str]], k: int) -> dict[str, list[str]]:
        """Convenience batch wrapper; recommenders needing precomputation
        across the whole batch (e.g. Popularity) may override this."""
        return {link: self.rank(link, candidates, k) for link, candidates in requests.items()}
