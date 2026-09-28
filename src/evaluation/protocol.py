"""Day 4 evaluation protocol (docs/experimental_design.md Section 15,
already approved before Day 3): candidate universe construction,
relevance extraction, and per-split evaluation-request assembly.

This module reads only **model-visible** files (Day 2's
`sampled_requests.csv`, Day 3's `workers.csv`/`requests.csv`/
`interactions.csv`). It never reads `data/processed/synthetic_private/`.
Relevance comes entirely from the historical `interactions.csv` record
(HIRED/SHORTLISTED), which is legitimate observed ground truth for
offline evaluation -- not the same thing as a recommender consuming
generator-private latent variables, which never happens here.

Candidate universe (Section 15): a worker is eligible for a request if
their primary OR secondary category matches the request's category, they
had joined by the request's simulated day, and (if the request carries a
location restriction) their country satisfies it. This mirrors Day 3's
own generation-time eligibility rule exactly -- verified empirically
before implementation, so that every historically relevant worker always
falls inside the reconstructed candidate universe (Recall@K is never
structurally capped below 1.0 by a mismatched eligibility definition).
Capacity (Day 3's 30-day/4-hire rolling limit) is a generation-time
mechanic, not a platform search constraint, and is intentionally not
reapplied here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.evaluation.config import EvaluationConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DAY2_PROCESSED = REPO_ROOT / "data" / "processed"
DAY3_SYNTHETIC = DAY2_PROCESSED / "synthetic"


@dataclass(frozen=True)
class EvalRequest:
    link: str
    split: str
    category: str
    candidates: list[str]  # sorted worker_ids, deterministic order
    relevance: dict[str, int]  # worker_id -> grade, restricted to candidates
    n_candidates: int
    excluded: bool
    excluded_reason: str | None  # None, "pool_too_small", "no_relevant_workers"


@dataclass
class ProtocolData:
    workers: pd.DataFrame
    requests: pd.DataFrame  # Day 3 requests.csv merged with Day 2 request content
    interactions: pd.DataFrame
    known_countries: list[str] = field(default_factory=list)


def load_protocol_data(day2_dir: Path = DAY2_PROCESSED, day3_dir: Path = DAY3_SYNTHETIC) -> ProtocolData:
    workers = pd.read_csv(day3_dir / "workers.csv")
    requests = pd.read_csv(day3_dir / "requests.csv")
    interactions = pd.read_csv(day3_dir / "interactions.csv")
    sampled = pd.read_csv(day2_dir / "sampled_requests.csv")[
        ["link", "category", "skills_vocab", "price_tier", "pay_type", "location_requirement"]
    ]
    requests = requests.merge(sampled, on="link", how="left")
    known_countries = sorted(workers["country"].dropna().unique().tolist())
    return ProtocolData(workers=workers, requests=requests, interactions=interactions, known_countries=known_countries)


def extract_required_countries(location_requirement: object, known_countries: list[str]) -> list[str]:
    """Country names (from the worker country pool) mentioned in a
    request's free-text location requirement, matched on word
    boundaries. Independent re-implementation of the same rule Day 3's
    generator uses (src/data/synth_workers.extract_required_countries),
    kept here rather than imported so src/evaluation does not depend on
    the data-generation package's internals."""
    if not isinstance(location_requirement, str) or not location_requirement:
        return []
    return [c for c in known_countries if re.search(rf"\b{re.escape(c)}\b", location_requirement)]


class EligibilityIndex:
    """Precomputed per-category worker positions for O(pool size)
    candidate-universe construction."""

    def __init__(self, workers: pd.DataFrame, known_countries: list[str]):
        self.workers = workers.reset_index(drop=True)
        self.worker_id = self.workers["worker_id"].to_numpy()
        self.join_day = self.workers["join_day"].to_numpy()
        self.country = self.workers["country"].to_numpy()
        self.known_countries = known_countries

        primary: dict[str, list[int]] = {}
        secondary: dict[str, list[int]] = {}
        for pos, row in enumerate(self.workers.itertuples()):
            primary.setdefault(row.primary_category, []).append(pos)
            secs = row.secondary_categories.split("|") if isinstance(row.secondary_categories, str) and row.secondary_categories else []
            for cat in secs:
                secondary.setdefault(cat, []).append(pos)
        self._primary = primary
        self._secondary = secondary

    def eligible_worker_ids(self, category: str, day: float, location_requirement: object) -> list[str]:
        positions = set(self._primary.get(category, [])) | set(self._secondary.get(category, []))
        if not positions:
            return []
        positions = sorted(positions)
        required = extract_required_countries(location_requirement, self.known_countries)
        eligible = [
            self.worker_id[p]
            for p in positions
            if self.join_day[p] <= day and (not required or self.country[p] in required)
        ]
        return sorted(eligible)


def relevance_for_link(interactions: pd.DataFrame, config: EvaluationConfig) -> dict[str, dict[str, int]]:
    """Precompute link -> {worker_id: grade} for every request that has
    interactions, once, for reuse across all evaluated requests."""
    grade_by_event = {
        "HIRED": config.relevance_hired_grade,
        "SHORTLISTED": config.relevance_shortlisted_grade,
    }
    result: dict[str, dict[str, int]] = {}
    for row in interactions.itertuples():
        result.setdefault(row.link, {})[row.worker_id] = grade_by_event[row.event]
    return result


def build_eval_requests(
    data: ProtocolData, index: EligibilityIndex, split: str, config: EvaluationConfig
) -> list[EvalRequest]:
    """One EvalRequest per request in `split`, with exclusion flags per
    docs/experimental_design.md Section 15: pool < min_candidate_pool_size,
    or (defensively) no relevant worker inside the candidate universe."""
    relevance_by_link = relevance_for_link(data.interactions, config)
    subset = data.requests.loc[data.requests["split"] == split]

    eval_requests = []
    for row in subset.itertuples():
        candidates = index.eligible_worker_ids(row.category, row.simulated_day, row.location_requirement)
        n_candidates = len(candidates)
        full_relevance = relevance_by_link.get(row.link, {})
        relevance = {w: g for w, g in full_relevance.items() if w in candidates}

        if n_candidates < config.min_candidate_pool_size:
            excluded, reason = True, "pool_too_small"
        elif not any(g >= 1 for g in relevance.values()):
            excluded, reason = True, "no_relevant_workers"
        else:
            excluded, reason = False, None

        eval_requests.append(EvalRequest(
            link=row.link, split=split, category=row.category, candidates=candidates,
            relevance=relevance, n_candidates=n_candidates, excluded=excluded, excluded_reason=reason,
        ))
    return eval_requests
