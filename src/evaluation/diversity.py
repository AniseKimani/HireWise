"""Diversity and catalogue-coverage metrics over top-K recommendation lists
(docs/experimental_design.md Section 15, "Diversity").

All functions take `rankings` = {request link: [worker_id, ...]} for one
method over one evaluation split, and never look at relevance.

- catalogue_coverage: share of the eligible catalogue (workers in at
  least one included request's candidate pool) appearing in any top-K
  list.
- exposure Gini: Gini coefficient of top-K appearance counts over the
  eligible catalogue, zeros included (0 = perfectly even exposure,
  -> 1 = all exposure on one worker).
- top_share: share of all top-K slots taken by the most-recommended
  ceil(fraction * catalogue size) workers.
- intra-list diversity: 1 - mean pairwise Jaccard similarity of declared
  skills within a list, averaged over lists (the Section 15 definition).
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np


def exposure_counts(rankings: dict[str, list[str]]) -> Counter:
    return Counter(w for ranked in rankings.values() for w in ranked)


def catalogue_coverage(rankings: dict[str, list[str]], eligible_workers: set[str]) -> float:
    if not eligible_workers:
        return 0.0
    recommended = {w for ranked in rankings.values() for w in ranked}
    return len(recommended & eligible_workers) / len(eligible_workers)


def gini(values) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * np.sum(ranks * x)) / (n * x.sum()) - (n + 1) / n)


def top_share(counts: Counter, catalogue_size: int, fraction: float) -> tuple[int, float]:
    total = sum(counts.values())
    n_top = max(1, math.ceil(fraction * catalogue_size))
    if total == 0:
        return n_top, 0.0
    top = sorted(counts.values(), reverse=True)[:n_top]
    return n_top, sum(top) / total


def jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def intra_list_diversity(rankings: dict[str, list[str]], worker_skills: dict[str, frozenset]) -> float:
    per_list = []
    for ranked in rankings.values():
        if len(ranked) < 2:
            continue
        sims = [jaccard(worker_skills.get(a, frozenset()), worker_skills.get(b, frozenset())) for a, b in combinations(ranked, 2)]
        per_list.append(1.0 - sum(sims) / len(sims))
    return float(np.mean(per_list)) if per_list else 0.0


def diversity_summary(rankings: dict[str, list[str]], eligible_workers: set[str],
                      worker_skills: dict[str, frozenset], top_fraction: float) -> dict:
    counts = exposure_counts(rankings)
    n_top, share = top_share(counts, len(eligible_workers), top_fraction)
    exposure = [counts.get(w, 0) for w in sorted(eligible_workers)]
    return {
        "n_lists": len(rankings),
        "n_slots": sum(counts.values()),
        "eligible_catalogue_size": len(eligible_workers),
        "unique_workers_recommended": len(counts),
        "catalogue_coverage": catalogue_coverage(rankings, eligible_workers),
        "exposure_gini": gini(exposure),
        "top_share_n_workers": n_top,
        "top_share_of_slots": share,
        "max_appearances_single_worker": max(counts.values()) if counts else 0,
        "intra_list_diversity": intra_list_diversity(rankings, worker_skills),
    }
