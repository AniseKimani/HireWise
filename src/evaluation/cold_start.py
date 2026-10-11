"""Cold-start-worker diagnostics (docs/experimental_design.md Section 12),
shared by Day 8+. Same definitions as the Day 7 diagnostic table
(results/day7/cold_start_worker_diagnostics.csv):

- cold_recall_at_k: among requests whose relevant set contains at least
  one cold-start worker, the fraction of those cold-start workers ranked
  in the top K, macro-averaged over such requests;
- cold_exposure_ratio: share of top-K slots given to cold-start workers
  divided by their share of relevant items (1 = proportional visibility).

The cold-start request subset itself is
src/evaluation/evaluator.cold_start_requests, unchanged since Day 4.
"""
from __future__ import annotations

import numpy as np


def cold_worker_diagnostics(rankings: dict[str, list[str]], eval_requests, cold_start_ids: set[str]) -> dict:
    slots = cold_slots = relevant = relevant_cold = 0
    cold_recalls = []
    for req in eval_requests:
        if req.excluded:
            continue
        ranked = rankings[req.link]
        slots += len(ranked)
        cold_slots += sum(1 for w in ranked if w in cold_start_ids)
        rel = [w for w, g in req.relevance.items() if g >= 1]
        cold_rel = [w for w in rel if w in cold_start_ids]
        relevant += len(rel)
        relevant_cold += len(cold_rel)
        if cold_rel:
            top = set(ranked)
            cold_recalls.append(sum(1 for w in cold_rel if w in top) / len(cold_rel))
    cold_share_top_k = cold_slots / slots if slots else 0.0
    cold_share_relevant = relevant_cold / relevant if relevant else 0.0
    return {
        "n_cold_start_requests": len(cold_recalls),
        "cold_recall_at_k": float(np.mean(cold_recalls)) if cold_recalls else 0.0,
        "cold_share_of_top_k_slots": cold_share_top_k,
        "cold_share_of_relevant_items": cold_share_relevant,
        "cold_exposure_ratio": cold_share_top_k / cold_share_relevant if cold_share_relevant else 0.0,
    }
