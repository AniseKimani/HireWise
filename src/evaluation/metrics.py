"""Ranking metrics for recommender evaluation (Day 4 brief Section 9).

Every metric takes a ranked list of worker IDs and a relevance mapping
(`worker_id -> grade`, grade 0 meaning "not relevant" and absent keys
treated as 0) and returns a float. Metrics never look at anything beyond
these two arguments, so they are usable identically for Random,
Popularity, CBF, and any future recommender.

Binary metrics (Precision/Recall/F1) treat any grade >= 1 as relevant,
matching docs/experimental_design.md Section 15 ("Binary metrics treat
grade >= 1 as relevant"). NDCG@K uses the full graded relevance.
"""
from __future__ import annotations

from math import log2


def _validate_ranking(ranked_worker_ids: list[str]) -> None:
    if len(ranked_worker_ids) != len(set(ranked_worker_ids)):
        raise ValueError(f"ranked_worker_ids contains duplicates: {ranked_worker_ids}")


def precision_at_k(ranked_worker_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Relevant items in the top-k, divided by min(k, len(ranking)) --
    so a ranking shorter than k (a small candidate pool) is not
    unfairly penalised for items that were never offered."""
    _validate_ranking(ranked_worker_ids)
    top_k = ranked_worker_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for w in top_k if relevance.get(w, 0) >= 1)
    return hits / min(k, len(ranked_worker_ids))


def recall_at_k(ranked_worker_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Relevant items in the top-k, divided by the total number of
    relevant items available (anywhere in the candidate universe, not
    just the top-k). 0.0 if there are no relevant items at all."""
    _validate_ranking(ranked_worker_ids)
    total_relevant = sum(1 for grade in relevance.values() if grade >= 1)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_worker_ids[:k]
    hits = sum(1 for w in top_k if relevance.get(w, 0) >= 1)
    return hits / total_relevant


def f1_at_k(ranked_worker_ids: list[str], relevance: dict[str, int], k: int) -> float:
    p = precision_at_k(ranked_worker_ids, relevance, k)
    r = recall_at_k(ranked_worker_ids, relevance, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_worker_ids: list[str], relevance: dict[str, int], k: int) -> float:
    """Graded NDCG@K. 0.0 if no relevant item exists anywhere in the
    candidate universe (ideal DCG would be 0)."""
    _validate_ranking(ranked_worker_ids)
    top_k = ranked_worker_ids[:k]
    actual_grades = [relevance.get(w, 0) for w in top_k]
    dcg = _dcg(actual_grades)

    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = _dcg(ideal_grades)
    if idcg == 0:
        return 0.0
    return dcg / idcg


METRIC_FUNCTIONS = {
    "precision_at_10": lambda ranked, relevance: precision_at_k(ranked, relevance, 10),
    "recall_at_10": lambda ranked, relevance: recall_at_k(ranked, relevance, 10),
    "f1_at_10": lambda ranked, relevance: f1_at_k(ranked, relevance, 10),
    "ndcg_at_10": lambda ranked, relevance: ndcg_at_k(ranked, relevance, 10),
}


def compute_all_at_k(ranked_worker_ids: list[str], relevance: dict[str, int], k: int) -> dict[str, float]:
    return {
        "precision_at_k": precision_at_k(ranked_worker_ids, relevance, k),
        "recall_at_k": recall_at_k(ranked_worker_ids, relevance, k),
        "f1_at_k": f1_at_k(ranked_worker_ids, relevance, k),
        "ndcg_at_k": ndcg_at_k(ranked_worker_ids, relevance, k),
    }


def macro_average(per_request_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Simple unweighted mean across requests -- every included request
    contributes equally regardless of its candidate pool size or number
    of relevant items."""
    if not per_request_metrics:
        return {}
    keys = per_request_metrics[0].keys()
    return {key: sum(m[key] for m in per_request_metrics) / len(per_request_metrics) for key in keys}
