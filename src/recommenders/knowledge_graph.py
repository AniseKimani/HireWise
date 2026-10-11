"""Knowledge Graph-only recommender (Day 7).

Scores each eligible worker for a request purely from paths in the
training-only graph snapshot (`src/kg/graph_view.py`), using the
degree-weighted path count (DWPC; Himmelstein & Baranzini 2015):

    path_weight = base_weight * PDP ** (-w)
    PDP         = product over the path's edges of deg(u) * deg(v),

where deg counts a node's edges of that edge's relationship type, and w
is the damping exponent (w = 0 gives a plain path count; larger w damps
paths through hub nodes such as very common skills or very active
clients). Path types, anchors, and lengths are listed in
src/kg/explanations.py. Raw component scores are sums of path weights:

    skill         = DWPC over (required Skill)<-[:HAS_SKILL]-(Worker)
    related_skill = DWPC over (required Skill)-[:RELATED_TO]-(Skill)<-[:HAS_SKILL]-(Worker),
                    base weight = background-corpus NPMI; the intermediate
                    skill must not itself be required (no double-counting
                    of direct matches)
    category      = DWPC over (ServiceCategory)<-[:SPECIALIZES_IN]-(Worker),
                    base weight = configured primary/secondary role weight
    history       = DWPC over repeat_hire + hired_similar_skill + co_hire,
                    all through TRAINING HIRED edges only (distinct
                    client-worker pairs). repeat_hire/co_hire need the
                    candidate's own training hires (never available for
                    cold-start workers); hired_similar_skill reaches any
                    worker sharing a declared skill with one the client
                    hired in training
    reputation    = Bayesian-smoothed mean of the worker's TRAINING
                    REVIEWED ratings (neutral global prior when none)

Each component is min-max normalized within the request's candidate
pool, and the final score is the configured weighted sum. Ties are broken
by worker_id. There is no popularity or CF fallback: KG-only means only
graph evidence ranks workers.

`co_hire` overlaps conceptually with collaborative filtering (shared
clients), but it is an explicit enumeration of graph paths over the
training HIRED edges, not the Day 5 CF similarity matrix; this module
does not import src/recommenders/collaborative.py.

Location is not a score component (it is a hard eligibility filter in the
shared candidate protocol); price/rate is not a score component (it
belongs to CBF); INTERESTED_IN is not a score component (for an eligible
pool it is identical for every candidate of a request).

Temporal guards (raise `TemporalLeakageError` rather than silently
continue): a request may only be scored if it is strictly later than
every historical edge in the snapshot, and its own link must not appear
among the snapshot's historical edges. Validation and test requests
satisfy both by construction; training requests are rejected.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.kg.explanations import (
    COMPONENT_LABELS, PATH_SPECS, ExplainedPath, Explanation, ReputationEvidence,
    build_summary, path_text, reputation_text,
)
from src.kg.graph_view import TRAIN_SPLIT, TrainingGraphView
from src.recommenders.base import BaseRecommender
from src.recommenders.kg_config import EXPECTED_COMPONENTS

COMPONENTS = tuple(EXPECTED_COMPONENTS)
COMPONENT_INDEX = {c: i for i, c in enumerate(COMPONENTS)}
PATH_COMPONENT_INDEX = {t: COMPONENT_INDEX[spec.component] for t, spec in PATH_SPECS.items()}
REPUTATION_INDEX = COMPONENT_INDEX["reputation"]


class TemporalLeakageError(RuntimeError):
    """Scoring this request would let historical graph edges at or after
    the request (including its own outcome) influence the ranking."""


class KGIndex:
    """Adjacency and degree lookups derived once from a TrainingGraphView."""

    def __init__(self, view: TrainingGraphView):
        self.view = view
        self.skills = frozenset(view.skills)

        worker_skills: dict[str, list[str]] = defaultdict(list)
        for worker_id, skill in view.has_skill:
            worker_skills[worker_id].append(skill)
        self.worker_skills = {w: tuple(sorted(s)) for w, s in worker_skills.items()}
        self.worker_skill_sets = {w: frozenset(s) for w, s in self.worker_skills.items()}
        self.skill_worker_degree: dict[str, int] = defaultdict(int)
        for _, skill in view.has_skill:
            self.skill_worker_degree[skill] += 1

        self.worker_categories: dict[str, dict[str, str]] = defaultdict(dict)
        self.category_worker_degree: dict[str, int] = defaultdict(int)
        for worker_id, category, role in view.specializes_in:
            self.worker_categories[worker_id][category] = role
            self.category_worker_degree[category] += 1

        related: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self.related_props: dict[tuple[str, str], tuple[float, int]] = {}
        for a, b, npmi, support in view.related_to:
            related[a].append((b, npmi))
            related[b].append((a, npmi))
            self.related_props[(a, b)] = (npmi, support)
        self.related = {s: tuple(sorted(n)) for s, n in related.items()}
        self.related_degree = {s: len(n) for s, n in self.related.items()}

        hire_links: dict[tuple[str, str], list[str]] = defaultdict(list)
        for client_id, worker_id, link, _ in view.train_hires:
            hire_links[(client_id, worker_id)].append(link)
        self.hire_links = {pair: tuple(sorted(links)) for pair, links in hire_links.items()}
        hired: dict[str, list[str]] = defaultdict(list)
        hirers: dict[str, list[str]] = defaultdict(list)
        for client_id, worker_id in self.hire_links:
            hired[client_id].append(worker_id)
            hirers[worker_id].append(client_id)
        self.hired = {c: tuple(sorted(ws)) for c, ws in hired.items()}
        self.hired_sets = {c: frozenset(ws) for c, ws in self.hired.items()}
        self.hirers = {w: tuple(sorted(cs)) for w, cs in hirers.items()}

        reviews: dict[str, list[int]] = defaultdict(list)
        for _, worker_id, _, rating, _ in view.train_reviews:
            reviews[worker_id].append(rating)
        self.review_count = {w: len(r) for w, r in reviews.items()}
        self.review_sum = {w: sum(r) for w, r in reviews.items()}
        n_reviews = len(view.train_reviews)
        self.mean_train_rating = (sum(r for _, _, _, r, _ in view.train_reviews) / n_reviews) if n_reviews else 0.0

        self.history_links = frozenset(l for _, _, l, _ in view.train_hires) | frozenset(
            l for _, _, l, _, _ in view.train_reviews
        )
        days = [d for _, _, _, d in view.train_hires] + [d for _, _, _, _, d in view.train_reviews]
        self.max_history_day = max(days) if days else float("-inf")


@dataclass(frozen=True)
class RequestContext:
    link: str
    client_id: str
    category: str
    simulated_day: float
    skills: tuple[str, ...]                                  # required skills present in the graph, sorted
    related_targets: dict[str, tuple[tuple[str, float], ...]]  # non-required skill -> ((required skill, npmi), ...)


def _split_skills(cell: object) -> tuple[str, ...]:
    if isinstance(cell, str) and cell:
        return tuple(sorted(set(cell.split("|"))))
    return ()


def enumerate_paths(index: KGIndex, ctx: RequestContext, worker_id: str, role_weights: dict[str, float]):
    """Every request-anchored path instance ending at `worker_id`, in a
    deterministic order, as (path_type, node_ids, base_weight, PDP).

    This is the single source of truth for both scoring and explanation.
    """
    paths = []
    w_skills = index.worker_skills.get(worker_id, ())
    w_skill_set = index.worker_skill_sets.get(worker_id, frozenset())
    w_skill_deg = len(w_skills)
    skill_deg = index.skill_worker_degree

    for s in ctx.skills:
        if s in w_skill_set:
            paths.append(("skill", (s, worker_id), 1.0, skill_deg[s] * w_skill_deg))

    related_deg = index.related_degree
    for s_prime in w_skills:
        for s, npmi in ctx.related_targets.get(s_prime, ()):
            pdp = related_deg[s] * related_deg[s_prime] * skill_deg[s_prime] * w_skill_deg
            paths.append(("related_skill", (s, s_prime, worker_id), npmi, pdp))

    role = index.worker_categories.get(worker_id, {}).get(ctx.category)
    if role is not None:
        pdp = index.category_worker_degree[ctx.category] * len(index.worker_categories[worker_id])
        paths.append(("category", (ctx.category, worker_id), role_weights[role], pdp))

    c = ctx.client_id
    client_hired = index.hired.get(c, ())
    if client_hired:
        c_deg = len(client_hired)
        w_hirers = index.hirers.get(worker_id, ())
        w_hire_deg = len(w_hirers)
        if worker_id in index.hired_sets[c]:
            paths.append(("repeat_hire", (c, worker_id), 1.0, c_deg * w_hire_deg))
        for w_prime in client_hired:
            if w_prime == worker_id:
                continue
            wp_hirers = index.hirers[w_prime]
            wp_hire_deg = len(wp_hirers)
            wp_skills = index.worker_skills.get(w_prime, ())
            wp_skill_deg = len(wp_skills)
            for s in wp_skills:
                if s in w_skill_set:
                    pdp = c_deg * wp_hire_deg * wp_skill_deg * skill_deg[s] * skill_deg[s] * w_skill_deg
                    paths.append(("hired_similar_skill", (c, w_prime, s, worker_id), 1.0, pdp))
            if w_hire_deg:
                for c_prime in wp_hirers:
                    if c_prime == c or worker_id not in index.hired_sets[c_prime]:
                        continue
                    cp_deg = len(index.hired[c_prime])
                    pdp = c_deg * wp_hire_deg * wp_hire_deg * cp_deg * cp_deg * w_hire_deg
                    paths.append(("co_hire", (c, w_prime, c_prime, worker_id), 1.0, pdp))
    return paths


def path_weight(base_weight: float, pdp: float, damping: float) -> float:
    if damping == 0.0:
        return base_weight
    return base_weight * pdp ** (-damping)


def minmax_columns(raw: np.ndarray) -> np.ndarray:
    lo = raw.min(axis=0)
    span = raw.max(axis=0) - lo
    safe = np.where(span > 1e-12, span, 1.0)
    return np.where(span > 1e-12, (raw - lo) / safe, 0.0)


class KGComponentCache:
    """Raw component matrices per request, computed once for every
    damping value in `dampings` (weights never change raw scores, so the
    weight grid reuses this cache)."""

    def __init__(self, dampings: tuple[float, ...]):
        self.dampings = tuple(dampings)
        self._store: dict[str, tuple[tuple[str, ...], dict[float, np.ndarray]]] = {}

    def __len__(self) -> int:
        return len(self._store)


class KnowledgeGraphRecommender(BaseRecommender):
    name = "kg"

    def __init__(
        self,
        index: KGIndex,
        requests: pd.DataFrame,
        damping: float,
        weights: dict[str, float],
        role_weights: dict[str, float],
        reputation_prior_strength: float,
        max_paths_per_component: int = 3,
        cache: KGComponentCache | None = None,
    ):
        if set(weights) != set(COMPONENTS):
            raise ValueError(f"weights must have exactly keys {COMPONENTS}, got {weights}")
        self.index = index
        self.damping = float(damping)
        self.weights = dict(weights)
        self.weight_vector = np.array([weights[c] for c in COMPONENTS], dtype=float)
        self.role_weights = dict(role_weights)
        self.reputation_prior_strength = float(reputation_prior_strength)
        self.max_paths_per_component = max_paths_per_component
        self.cache = cache if cache is not None else KGComponentCache((self.damping,))
        if self.damping not in self.cache.dampings:
            raise ValueError(f"damping {self.damping} not covered by cache dampings {self.cache.dampings}")

        self._request_rows = {
            row.link: (row.client_id, row.category, float(row.simulated_day), row.skills_vocab)
            for row in requests[["link", "client_id", "category", "simulated_day", "skills_vocab"]].itertuples(index=False)
        }
        self._contexts: dict[str, RequestContext] = {}

    def with_config(self, damping: float, weights: dict[str, float]) -> "KnowledgeGraphRecommender":
        """Same index, requests, and cache; different damping/weights."""
        clone = object.__new__(KnowledgeGraphRecommender)
        clone.__dict__.update(self.__dict__)
        clone.damping = float(damping)
        clone.weights = dict(weights)
        clone.weight_vector = np.array([weights[c] for c in COMPONENTS], dtype=float)
        if clone.damping not in clone.cache.dampings:
            raise ValueError(f"damping {clone.damping} not covered by cache dampings {clone.cache.dampings}")
        return clone

    # ------------------------------------------------------------------
    # Request context and temporal guards
    # ------------------------------------------------------------------
    def context(self, link: str) -> RequestContext:
        ctx = self._contexts.get(link)
        if ctx is not None:
            return ctx
        if link not in self._request_rows:
            raise KeyError(f"unknown request link {link!r}")
        if link in self.index.history_links:
            raise TemporalLeakageError(
                f"request {link!r} has its own outcome among the graph's historical edges; "
                f"only requests after the {TRAIN_SPLIT!r} period may be scored"
            )
        client_id, category, day, skills_cell = self._request_rows[link]
        if not day > self.index.max_history_day:
            raise TemporalLeakageError(
                f"request {link!r} (day {day}) is not strictly after the latest historical graph edge "
                f"(day {self.index.max_history_day})"
            )
        skills = tuple(s for s in _split_skills(skills_cell) if s in self.index.skills)
        required = frozenset(skills)
        targets: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for s in skills:
            for s_prime, npmi in self.index.related.get(s, ()):
                if s_prime not in required:
                    targets[s_prime].append((s, npmi))
        ctx = RequestContext(
            link=link, client_id=client_id, category=category, simulated_day=day, skills=skills,
            related_targets={k: tuple(v) for k, v in targets.items()},
        )
        self._contexts[link] = ctx
        return ctx

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def reputation(self, worker_id: str) -> float:
        n = self.index.review_count.get(worker_id, 0)
        total = self.index.review_sum.get(worker_id, 0)
        m = self.reputation_prior_strength
        return (total + m * self.index.mean_train_rating) / (n + m)

    def raw_components(self, link: str, candidates: list[str]) -> np.ndarray:
        """(n_candidates, n_components) raw scores at this recommender's damping."""
        key = tuple(candidates)
        cached = self.cache._store.get(link)
        if cached is None or cached[0] != key:
            ctx = self.context(link)
            dampings = self.cache.dampings
            raws = {d: np.zeros((len(candidates), len(COMPONENTS))) for d in dampings}
            for i, worker_id in enumerate(candidates):
                for path_type, _, base, pdp in enumerate_paths(self.index, ctx, worker_id, self.role_weights):
                    col = PATH_COMPONENT_INDEX[path_type]
                    for d in dampings:
                        raws[d][i, col] += path_weight(base, pdp, d)
                rep = self.reputation(worker_id)
                for d in dampings:
                    raws[d][i, REPUTATION_INDEX] = rep
            self.cache._store[link] = (key, raws)
            cached = self.cache._store[link]
        else:
            self.context(link)  # re-run the temporal guards even on a cache hit
        return cached[1][self.damping]

    def scored(self, link: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(raw, contributions, score) for every candidate."""
        raw = self.raw_components(link, candidates)
        contributions = minmax_columns(raw) * self.weight_vector
        return raw, contributions, contributions.sum(axis=1)

    def rank(self, link: str, candidates: list[str], k: int) -> list[str]:
        if not candidates:
            return []
        _, _, score = self.scored(link, candidates)
        order = sorted(range(len(candidates)), key=lambda i: (-score[i], candidates[i]))
        return [candidates[i] for i in order[:k]]

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------
    def _edge_properties(self, path_type: str, node_ids: tuple[str, ...]) -> tuple[dict, ...]:
        def hired(client_id: str, worker_id: str) -> dict:
            links = self.index.hire_links[(client_id, worker_id)]
            return {"type": "HIRED", "split": TRAIN_SPLIT, "n_hires": len(links), "links": list(links)}

        def related(a: str, b: str) -> dict:
            npmi, support = self.index.related_props[(min(a, b), max(a, b))]
            return {"type": "RELATED_TO", "npmi": npmi, "support": support}

        has_skill = {"type": "HAS_SKILL"}
        if path_type == "skill":
            return (has_skill,)
        if path_type == "related_skill":
            return (related(node_ids[0], node_ids[1]), has_skill)
        if path_type == "category":
            category, worker_id = node_ids
            return ({"type": "SPECIALIZES_IN", "role": self.index.worker_categories[worker_id][category]},)
        if path_type == "repeat_hire":
            return (hired(node_ids[0], node_ids[1]),)
        if path_type == "hired_similar_skill":
            return (hired(node_ids[0], node_ids[1]), has_skill, has_skill)
        if path_type == "co_hire":
            c, w_prime, c_prime, worker_id = node_ids
            return (hired(c, w_prime), hired(c_prime, w_prime), hired(c_prime, worker_id))
        raise ValueError(f"unknown path type {path_type!r}")

    def explain(self, link: str, candidates: list[str], k: int) -> list[Explanation]:
        """Top-k ranking with one Explanation per recommended worker, built
        from the same enumerated paths that produced the score."""
        if not candidates:
            return []
        raw, contributions, score = self.scored(link, candidates)
        ctx = self.context(link)
        order = sorted(range(len(candidates)), key=lambda i: (-score[i], candidates[i]))[:k]
        position = {w: i for i, w in enumerate(candidates)}

        explanations = []
        for rank, i in enumerate(order, start=1):
            worker_id = candidates[i]
            all_paths: list[ExplainedPath] = []
            recomputed = np.zeros(len(COMPONENTS))
            records = enumerate_paths(self.index, ctx, worker_id, self.role_weights)
            weights = [path_weight(base, pdp, self.damping) for _, _, base, pdp in records]
            for (path_type, _, _, _), weight in zip(records, weights):
                recomputed[PATH_COMPONENT_INDEX[path_type]] += weight
            for col in range(len(COMPONENTS)):
                if col == REPUTATION_INDEX:
                    continue
                if abs(recomputed[col] - raw[i, col]) > 1e-9 * max(1.0, abs(raw[i, col])):
                    raise AssertionError(
                        f"explanation paths do not reproduce component {COMPONENTS[col]!r} for {worker_id}: "
                        f"{recomputed[col]} vs {raw[i, col]}"
                    )
            for (path_type, node_ids, _, _), weight in zip(records, weights):
                spec = PATH_SPECS[path_type]
                col = PATH_COMPONENT_INDEX[path_type]
                share = contributions[i, col] * weight / raw[i, col] if raw[i, col] > 0 else 0.0
                edges = self._edge_properties(path_type, node_ids)
                all_paths.append(ExplainedPath(
                    path_type=path_type, component=spec.component,
                    nodes=tuple(zip(spec.labels, node_ids)), edges=edges, length=spec.length,
                    path_weight=weight, contribution=float(share), text=path_text(path_type, node_ids, edges),
                ))

            component_contributions = {c: float(contributions[i, COMPONENT_INDEX[c]]) for c in COMPONENTS}
            cited = sorted(
                (c for c in COMPONENTS if component_contributions[c] > 0),
                key=lambda c: (-component_contributions[c], COMPONENT_INDEX[c]),
            )
            displayed = []
            for component in COMPONENTS:
                component_paths = sorted(
                    (p for p in all_paths if p.component == component),
                    key=lambda p: (-p.contribution, p.path_type, tuple(n for _, n in p.nodes)),
                )
                displayed.extend(component_paths[: self.max_paths_per_component])
            component_rank = {c: r for r, c in enumerate(cited)}
            displayed.sort(key=lambda p: (component_rank.get(p.component, len(COMPONENTS)), -p.contribution,
                                          p.path_type, tuple(n for _, n in p.nodes)))

            n_reviews = self.index.review_count.get(worker_id, 0)
            mean_rating = self.index.review_sum[worker_id] / n_reviews if n_reviews else None
            smoothed = float(raw[i, REPUTATION_INDEX])
            reputation = ReputationEvidence(
                n_reviews=n_reviews, mean_rating=mean_rating, smoothed_rating=smoothed,
                prior_mean=self.index.mean_train_rating, prior_used_only=n_reviews == 0,
                contribution=component_contributions["reputation"],
                text=reputation_text(n_reviews, mean_rating, smoothed, self.index.mean_train_rating),
            )
            n_paths_by_type = {t: 0 for t in PATH_SPECS}
            for p in all_paths:
                n_paths_by_type[p.path_type] += 1

            explanations.append(Explanation(
                link=link, worker_id=worker_id, rank=rank, score=float(score[position[worker_id]]),
                component_contributions=component_contributions,
                component_raw={c: float(raw[i, COMPONENT_INDEX[c]]) for c in COMPONENTS},
                cited_components=cited, n_paths_by_type=n_paths_by_type,
                path_lengths=[p.length for p in all_paths],
                total_path_contribution=float(sum(p.contribution for p in all_paths)),
                paths=displayed, reputation=reputation,
                summary=build_summary(cited, displayed, reputation),
            ))
        return explanations


__all__ = [
    "COMPONENTS", "COMPONENT_LABELS", "KGComponentCache", "KGIndex", "KnowledgeGraphRecommender",
    "RequestContext", "TemporalLeakageError", "enumerate_paths", "minmax_columns", "path_weight",
]
