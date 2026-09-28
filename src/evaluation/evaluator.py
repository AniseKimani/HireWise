"""Runs a recommender across a set of evaluation requests and aggregates
metrics (Day 4 brief Section 9). Reusable across Random, Popularity, CBF,
and any future recommender that implements src/recommenders/base.py's
`BaseRecommender`.
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import compute_all_at_k, macro_average
from src.evaluation.protocol import EvalRequest
from src.recommenders.base import BaseRecommender


def evaluate(recommender: BaseRecommender, eval_requests: list[EvalRequest], k: int) -> tuple[pd.DataFrame, dict]:
    """Evaluate `recommender` over every non-excluded request in
    `eval_requests`. Returns (per_request_df, summary_dict). Excluded
    requests are skipped but counted in the summary, per
    docs/experimental_design.md Section 15."""
    rows = []
    metrics_only = []
    for req in eval_requests:
        if req.excluded:
            continue
        ranked = recommender.rank(req.link, req.candidates, k)
        _validate_ranking(ranked, req.candidates, k)
        metrics = compute_all_at_k(ranked, req.relevance, k)
        rows.append({"link": req.link, "n_candidates": req.n_candidates, **metrics})
        metrics_only.append(metrics)

    per_request_df = pd.DataFrame(rows)
    summary = macro_average(metrics_only) if metrics_only else {
        "precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0, "ndcg_at_k": 0.0,
    }
    summary["n_included"] = len(rows)
    summary["n_excluded"] = sum(1 for r in eval_requests if r.excluded)
    summary["n_excluded_pool_too_small"] = sum(1 for r in eval_requests if r.excluded_reason == "pool_too_small")
    summary["n_excluded_no_relevant_workers"] = sum(1 for r in eval_requests if r.excluded_reason == "no_relevant_workers")
    return per_request_df, summary


def _validate_ranking(ranked: list[str], candidates: list[str], k: int) -> None:
    if len(ranked) != len(set(ranked)):
        raise ValueError(f"recommender returned duplicate worker IDs: {ranked}")
    if len(ranked) > k:
        raise ValueError(f"recommender returned more than k={k} recommendations: {len(ranked)}")
    candidate_set = set(candidates)
    invalid = [w for w in ranked if w not in candidate_set]
    if invalid:
        raise ValueError(f"recommender returned worker(s) outside the candidate universe: {invalid}")


def cold_start_requests(eval_requests: list[EvalRequest], cold_start_worker_ids: set[str]) -> list[EvalRequest]:
    """Subset of non-excluded requests whose relevant set (grade >= 1)
    includes at least one new-entrant cold-start worker (Day 4 brief
    Section 10)."""
    return [
        req for req in eval_requests
        if not req.excluded
        and any(worker_id in cold_start_worker_ids for worker_id, grade in req.relevance.items() if grade >= 1)
    ]
