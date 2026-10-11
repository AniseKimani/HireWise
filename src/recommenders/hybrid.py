"""Weighted hybrid recommender (Day 8).

    Hybrid(r, w) = w_cbf * CBF_norm + w_cf * CF_norm + w_kg * KG_norm

Each component score comes from the FROZEN standalone recommender's own
scoring interface (no algorithm is reimplemented here):

- CBF: `ContentBasedRecommender.score(link, worker_id)` (Day 4).
- CF: the raw collaborative score from `CollaborativeFiltering._cf_scores`
  (Day 5); a candidate with no collaborative evidence scores 0.
  Standalone CF's popularity fallback is a tie-breaking rule for its own
  ranking, not collaborative evidence, so it is not part of the CF score.
- KG: the final KG score from `KnowledgeGraphRecommender.scored` (Day 7),
  which also enforces the KG temporal guards.

Normalisation: each component is min-max normalised across the request's
own candidate pool, independently. A component that is constant across
the pool carries no ranking information for that request and contributes
exactly 0 for every candidate. No label, global statistic, or future
data enters normalisation. Ties are broken by worker_id.

The hybrid sees only (link, candidates) and the three component scores:
never interactions, relevance labels, splits, or generator-private data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.evaluation.evaluator import evaluate
from src.recommenders.base import BaseRecommender
from src.recommenders.hybrid_config import EXPECTED_COMPONENTS, balance

COMPONENTS = tuple(EXPECTED_COMPONENTS)
COMPONENT_LABELS = {
    "cbf": "profile match (CBF)",
    "cf": "collaborative history (CF)",
    "kg": "knowledge-graph evidence (KG)",
}
CONSTANT_TOLERANCE = 1e-12
METRIC_KEY = {"ndcg_at_10": "ndcg_at_k", "precision_at_10": "precision_at_k",
              "recall_at_10": "recall_at_k", "f1_at_10": "f1_at_k"}

ScoreFn = Callable[[str, list[str]], np.ndarray]


# ---------------------------------------------------------------------------
# Adapters onto the frozen standalone scoring interfaces
# ---------------------------------------------------------------------------

def cbf_score_fn(cbf) -> ScoreFn:
    return lambda link, candidates: np.array([cbf.score(link, w) for w in candidates], dtype=float)


def cf_score_fn(cf) -> ScoreFn:
    def scores(link, candidates):
        evidence = cf._cf_scores(link, candidates)
        return np.array([evidence.get(w, 0.0) for w in candidates], dtype=float)
    return scores


def kg_score_fn(kg) -> ScoreFn:
    return lambda link, candidates: np.asarray(kg.scored(link, candidates)[2], dtype=float)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def minmax_normalize(raw: np.ndarray) -> np.ndarray:
    """Column-wise min-max over the candidate pool; constant columns -> 0."""
    raw = np.asarray(raw, dtype=float)
    lo = raw.min(axis=0)
    span = raw.max(axis=0) - lo
    constant = span <= CONSTANT_TOLERANCE
    return np.where(constant, 0.0, (raw - lo) / np.where(constant, 1.0, span))


def validate_weights(weights: dict[str, float]) -> None:
    if set(weights) != set(COMPONENTS):
        raise ValueError(f"weights must have exactly keys {COMPONENTS}, got {weights}")
    if any(v < 0 for v in weights.values()):
        raise ValueError(f"weights must be non-negative, got {weights}")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {weights}")


# ---------------------------------------------------------------------------
# Hybrid recommender
# ---------------------------------------------------------------------------

@dataclass
class HybridExplanation:
    link: str
    worker_id: str
    rank: int
    score: float
    weights: dict[str, float]
    raw_scores: dict[str, float]
    normalized_scores: dict[str, float]
    contributions: dict[str, float]
    cited_components: list[str]      # positive contributions, largest first
    dominant_component: str | None
    text: str
    kg_summary: str | None = None
    kg_paths: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "link": self.link, "worker_id": self.worker_id, "rank": self.rank, "score": self.score,
            "weights": self.weights, "raw_scores": self.raw_scores, "normalized_scores": self.normalized_scores,
            "contributions": self.contributions, "cited_components": self.cited_components,
            "dominant_component": self.dominant_component, "text": self.text,
            "kg_summary": self.kg_summary, "kg_paths": self.kg_paths,
        }


class HybridScoreCache:
    """Raw component matrices per request; weights never change raw scores."""

    def __init__(self):
        self._store: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self._store)


class HybridRecommender(BaseRecommender):
    name = "hybrid"

    def __init__(self, score_fns: dict[str, ScoreFn], weights: dict[str, float], cache: HybridScoreCache | None = None):
        if set(score_fns) != set(COMPONENTS):
            raise ValueError(f"score_fns must have exactly keys {COMPONENTS}")
        validate_weights(weights)
        self.score_fns = score_fns
        self.weights = dict(weights)
        self.weight_vector = np.array([weights[c] for c in COMPONENTS], dtype=float)
        self.cache = cache if cache is not None else HybridScoreCache()

    def with_weights(self, weights: dict[str, float]) -> "HybridRecommender":
        return HybridRecommender(self.score_fns, weights, self.cache)

    def raw_matrix(self, link: str, candidates: list[str]) -> np.ndarray:
        key = tuple(candidates)
        cached = self.cache._store.get(link)
        if cached is None or cached[0] != key:
            raw = np.column_stack([self.score_fns[c](link, list(candidates)) for c in COMPONENTS])
            self.cache._store[link] = (key, raw)
            cached = self.cache._store[link]
        return cached[1]

    def scored(self, link: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(raw, normalized, contributions, score) per candidate."""
        raw = self.raw_matrix(link, candidates)
        normalized = minmax_normalize(raw)
        contributions = normalized * self.weight_vector
        return raw, normalized, contributions, contributions.sum(axis=1)

    def _order(self, candidates: list[str], score: np.ndarray) -> list[int]:
        return sorted(range(len(candidates)), key=lambda i: (-score[i], candidates[i]))

    def rank(self, link: str, candidates: list[str], k: int) -> list[str]:
        if not candidates:
            return []
        _, _, _, score = self.scored(link, candidates)
        return [candidates[i] for i in self._order(candidates, score)[:k]]

    def explain(self, link: str, candidates: list[str], k: int, kg_recommender=None, kg_paths: int = 3) -> list[HybridExplanation]:
        """Top-k with each recommendation's actual contribution breakdown.
        With `kg_recommender`, the strongest KG paths (from the frozen Day 7
        explainer, same candidate pool) are attached for context."""
        if not candidates:
            return []
        raw, normalized, contributions, score = self.scored(link, candidates)
        order = self._order(candidates, score)[:k]
        kg_by_worker = {}
        if kg_recommender is not None:
            kg_by_worker = {e.worker_id: e for e in kg_recommender.explain(link, candidates, len(candidates))}

        out = []
        for rank, i in enumerate(order, start=1):
            contrib = {c: float(contributions[i, j]) for j, c in enumerate(COMPONENTS)}
            cited = sorted((c for c in COMPONENTS if contrib[c] > 0), key=lambda c: (-contrib[c], COMPONENTS.index(c)))
            worker_id = candidates[i]
            kg_exp = kg_by_worker.get(worker_id)
            kg_summary = kg_exp.summary if kg_exp is not None else None
            kg_path_dicts = []
            if kg_exp is not None:
                top_paths = sorted(kg_exp.paths, key=lambda p: (-p.contribution, p.path_type, tuple(n for _, n in p.nodes)))
                kg_path_dicts = [
                    {"path_type": p.path_type, "text": p.text, "kg_internal_contribution": p.contribution,
                     "nodes": [{"label": label, "id": node_id} for label, node_id in p.nodes], "edges": list(p.edges)}
                    for p in top_paths[:kg_paths]
                ]
            out.append(HybridExplanation(
                link=link, worker_id=worker_id, rank=rank, score=float(score[i]), weights=dict(self.weights),
                raw_scores={c: float(raw[i, j]) for j, c in enumerate(COMPONENTS)},
                normalized_scores={c: float(normalized[i, j]) for j, c in enumerate(COMPONENTS)},
                contributions=contrib, cited_components=cited,
                dominant_component=cited[0] if cited else None,
                text=explanation_text(contrib, cited, float(score[i]), kg_summary),
                kg_summary=kg_summary, kg_paths=kg_path_dicts,
            ))
        return out


def explanation_text(contributions: dict[str, float], cited: list[str], score: float, kg_summary: str | None) -> str:
    """Template filled only with the actual contributions (largest first)."""
    if not cited:
        return f"No component distinguishes this worker within the eligible pool (score {score:.2f})"
    parts = [f"{COMPONENT_LABELS[c]} contributed {contributions[c]:.2f}" for c in cited]
    silent = [COMPONENT_LABELS[c] for c in COMPONENTS if c not in cited]
    text = "Recommended because " + ", ".join(parts) + f" (hybrid score {score:.2f})"
    if silent:
        text += "; no contribution from " + " or ".join(silent)
    if kg_summary and "kg" in cited:
        text += f". Strongest KG evidence: {kg_summary}"
    return text


# ---------------------------------------------------------------------------
# Validation-only tuning
# ---------------------------------------------------------------------------

def tune_hybrid(base: HybridRecommender, weight_grid: list[dict], eval_requests, k: int,
                primary_metric: str, tiebreak_metric: str) -> tuple[dict, list[dict]]:
    """Evaluate every predeclared weight vector on the given (validation)
    requests. Selection: primary metric desc, tiebreak metric desc, then
    the more balanced vector (smaller sum |w - 1/3|), then grid order."""
    primary, tiebreak = METRIC_KEY[primary_metric], METRIC_KEY[tiebreak_metric]
    rows, best, best_key = [], None, None
    for index, weights in enumerate(weight_grid):
        _, summary = evaluate(base.with_weights(weights), eval_requests, k)
        rows.append({"grid_index": index, **{f"w_{c}": weights[c] for c in COMPONENTS},
                     "balance": balance(weights), **summary})
        key = (summary[primary], summary[tiebreak], -balance(weights), -index)
        if best_key is None or key > best_key:
            best_key, best = key, {"grid_index": index, "weights": dict(weights)}
    for row in rows:
        row["selected"] = row["grid_index"] == best["grid_index"]
    return best, rows
