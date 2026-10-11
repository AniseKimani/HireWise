"""Structural explainability metrics for graph-path explanations
(docs/experimental_design.md Section 15, "Explainability").

Implemented here (Day 7, KG-only):

- path coverage: share of recommendations with at least one valid
  request-anchored KG path of length <= `max_path_length`;
- path coverage excluding category paths: every eligible candidate
  already matches the request's category (shared candidate protocol), so
  a category path exists for every recommendation and the plain coverage
  figure is trivially 100%;
- mean supporting-path count and mean path length;
- a KG-internal fidelity sanity check: the first cited component is the
  largest contributor, and path + reputation contributions reconstruct
  the score exactly.

Not implemented here: the documented explanation-fidelity metric, which
is defined against the *hybrid* score's largest contributing factor and
therefore belongs to Day 8.

Metric functions read only `Explanation` objects; they never re-score.
"""
from __future__ import annotations

from statistics import mean, median

import numpy as np

from src.kg.explanations import PATH_TYPES, Explanation

SCORE_TOLERANCE = 1e-9


def _safe_mean(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def explainability_summary(explanations: list[Explanation], max_path_length: int = 3) -> dict:
    n = len(explanations)
    if n == 0:
        return {"n_recommendations": 0}

    covered = sum(1 for e in explanations if any(l <= max_path_length for l in e.path_lengths))
    covered_non_category = sum(1 for e in explanations if e.n_non_category_paths > 0)
    supporting = [e.n_supporting_paths for e in explanations]
    supporting_non_category = [e.n_non_category_paths for e in explanations]
    per_rec_mean_length = [mean(e.path_lengths) for e in explanations if e.path_lengths]
    pooled_lengths = [l for e in explanations for l in e.path_lengths]

    first_cited_is_largest = []
    reconstructs_score = []
    displayed_share = []
    for e in explanations:
        positive = {c: v for c, v in e.component_contributions.items() if v > 0}
        if positive:
            largest = max(positive.values())
            first_cited_is_largest.append(
                bool(e.cited_components) and abs(e.component_contributions[e.cited_components[0]] - largest) <= SCORE_TOLERANCE
            )
        reputation = e.reputation.contribution if e.reputation is not None else 0.0
        reconstructs_score.append(abs(e.total_path_contribution + reputation - e.score) <= SCORE_TOLERANCE * max(1.0, abs(e.score)))
        if e.score > 0:
            displayed_share.append((sum(p.contribution for p in e.paths) + reputation) / e.score)

    path_type_presence = {t: sum(1 for e in explanations if e.n_paths_by_type.get(t, 0) > 0) / n for t in PATH_TYPES}
    mean_paths_by_type = {t: float(mean(e.n_paths_by_type.get(t, 0) for e in explanations)) for t in PATH_TYPES}

    return {
        "n_recommendations": n,
        "max_path_length": max_path_length,
        "path_coverage": covered / n,
        "path_coverage_excluding_category": covered_non_category / n,
        "mean_supporting_paths": float(mean(supporting)),
        "median_supporting_paths": float(median(supporting)),
        "mean_supporting_paths_excluding_category": float(mean(supporting_non_category)),
        "mean_path_length_per_recommendation": _safe_mean(per_rec_mean_length),
        "mean_path_length_pooled": _safe_mean(pooled_lengths),
        "max_observed_path_length": max(pooled_lengths) if pooled_lengths else None,
        "share_recommendations_with_path_type": path_type_presence,
        "mean_paths_by_type": mean_paths_by_type,
        "fidelity_sanity": {
            "n_with_positive_contribution": len(first_cited_is_largest),
            "share_first_cited_component_is_largest": (
                sum(first_cited_is_largest) / len(first_cited_is_largest) if first_cited_is_largest else None
            ),
            "share_contributions_reconstruct_score": sum(reconstructs_score) / n,
            "mean_displayed_contribution_share": _safe_mean(displayed_share),
        },
    }


def component_contribution_summary(explanations: list[Explanation], components: tuple[str, ...]) -> list[dict]:
    """Per component: mean absolute contribution, mean share of the final
    score, and how often it is the largest contributor, over the given
    recommendations."""
    rows = []
    positive_scores = [e for e in explanations if e.score > 0]
    for c in components:
        contributions = [e.component_contributions[c] for e in explanations]
        shares = [e.component_contributions[c] / e.score for e in positive_scores]
        top = sum(1 for e in explanations if e.cited_components and e.cited_components[0] == c)
        rows.append({
            "component": c,
            "mean_contribution": float(mean(contributions)) if contributions else 0.0,
            "mean_share_of_score": float(mean(shares)) if shares else 0.0,
            "share_positive": sum(1 for v in contributions if v > 0) / len(contributions) if contributions else 0.0,
            "share_largest_component": top / len(explanations) if explanations else 0.0,
        })
    return rows


def deterministic_sample(explanations: list[Explanation], n: int, seed: int) -> list[Explanation]:
    """`n` explanations chosen without replacement by a generator seeded
    from the project master seed, over a population sorted by
    (link, rank) so the draw never depends on input order."""
    population = sorted(explanations, key=lambda e: (e.link, e.rank))
    if n >= len(population):
        return population
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    chosen = sorted(rng.choice(len(population), size=n, replace=False).tolist())
    return [population[i] for i in chosen]
