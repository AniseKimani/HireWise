"""Content-Based Filtering recommender (Day 4 brief Section 7).

Uses only model-visible request and worker content -- never latent
quality, true skills/proficiency, generator utility, client taste, future
interactions, or oracle truth:

    CBF = w_skill * skill_similarity + w_category * category_compatibility
        + w_price * price_compatibility

- **skill_similarity**: Jaccard similarity between the request's
  controlled-vocabulary skills (`skills_vocab`) and the worker's
  *declared* skills (never true/private skills), bounded [0, 1].
- **category_compatibility**: 1.0 if the request's category is the
  worker's primary category, 0.6 if it is one of the worker's secondary
  categories, else 0.0 (an observable profile field, not a private one).
- **price_compatibility**: 1 - |worker.price_tier - request.price_tier|,
  bounded [0, 1]; a neutral score (configurable) is used when the request
  has no stated price (`price_tier` missing/unspecified pay type) or the
  worker has no price_tier. This is a deliberately different functional
  form from Day 3's generator budget-fit component (a step function with
  exponential decay), so CBF is not silently approximating the
  generator's own scoring rule.

No request-side experience signal exists in the Day 2 data (requests
carry no experience-level field), so experience is intentionally *not*
included as a CBF feature -- see docs/day4_recommender_report.md.

Weights are supplied by the caller (validated to sum to 1.0 by
src/evaluation/config.py's EvaluationConfig), never hard-coded here.
"""
from __future__ import annotations

import pandas as pd

from src.recommenders.base import BaseRecommender


def _split_set(cell: object) -> frozenset[str]:
    if isinstance(cell, str) and cell:
        return frozenset(cell.split("|"))
    return frozenset()


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class ContentBasedRecommender(BaseRecommender):
    name = "cbf"

    def __init__(self, workers: pd.DataFrame, requests: pd.DataFrame, weights: dict[str, float], price_neutral_score: float):
        required = {"skill", "category", "price"}
        if set(weights) != required:
            raise ValueError(f"weights must have exactly keys {required}, got {weights}")
        self.weights = weights
        self.price_neutral_score = price_neutral_score

        self._worker_skills = {
            wid: _split_set(skills) for wid, skills in zip(workers["worker_id"], workers["declared_skills"])
        }
        self._worker_primary = dict(zip(workers["worker_id"], workers["primary_category"]))
        self._worker_secondary = {
            wid: _split_set(secs) for wid, secs in zip(workers["worker_id"], workers["secondary_categories"])
        }
        self._worker_price_tier = dict(zip(workers["worker_id"], workers["price_tier"]))

        self._request_skills = {
            link: _split_set(skills) for link, skills in zip(requests["link"], requests["skills_vocab"])
        }
        self._request_category = dict(zip(requests["link"], requests["category"]))
        self._request_price_tier = dict(zip(requests["link"], requests["price_tier"]))

    def _category_compatibility(self, request_category: str, worker_id: str) -> float:
        if self._worker_primary.get(worker_id) == request_category:
            return 1.0
        if request_category in self._worker_secondary.get(worker_id, frozenset()):
            return 0.6
        return 0.0

    def _price_compatibility(self, request_price_tier: float, worker_id: str) -> float:
        if pd.isna(request_price_tier):
            return self.price_neutral_score
        worker_price_tier = self._worker_price_tier.get(worker_id)
        if worker_price_tier is None or pd.isna(worker_price_tier):
            return self.price_neutral_score
        return 1.0 - abs(worker_price_tier - request_price_tier)

    def score(self, link: str, worker_id: str) -> float:
        request_skills = self._request_skills.get(link, frozenset())
        worker_skills = self._worker_skills.get(worker_id, frozenset())
        skill_sim = jaccard_similarity(request_skills, worker_skills)

        category_compat = self._category_compatibility(self._request_category.get(link), worker_id)
        price_compat = self._price_compatibility(self._request_price_tier.get(link), worker_id)

        return (
            self.weights["skill"] * skill_sim
            + self.weights["category"] * category_compat
            + self.weights["price"] * price_compat
        )

    def rank(self, link: str, candidates: list[str], k: int) -> list[str]:
        scored = [(w, self.score(link, w)) for w in candidates]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))  # deterministic tie-break by worker_id
        return [w for w, _ in scored[:k]]
