"""Graph-path explanation model for the Day 7 KG recommender.

An explanation is never written after the fact: it is assembled from the
exact path instances that `src/recommenders/knowledge_graph.py` sums into
each score component, and every displayed path carries the share of the
final score it contributed. Text is a fixed template per path type,
filled only with node identifiers and edge properties from that path.

Path anchors. There is no Request node (Day 6 design), so a request is
represented in the graph by its anchors: its required Skill nodes, its
ServiceCategory node, and its Client node. Path length is the number of
real graph edges from an anchor to the recommended worker; no virtual
edge is counted.

| Path type | Pattern | Length | Component |
|---|---|---|---|
| skill | (Skill)<-[:HAS_SKILL]-(Worker) | 1 | skill |
| related_skill | (Skill)-[:RELATED_TO]-(Skill)<-[:HAS_SKILL]-(Worker) | 2 | related_skill |
| category | (ServiceCategory)<-[:SPECIALIZES_IN]-(Worker) | 1 | category |
| repeat_hire | (Client)-[:HIRED]->(Worker) | 1 | history |
| hired_similar_skill | (Client)-[:HIRED]->(Worker)-[:HAS_SKILL]->(Skill)<-[:HAS_SKILL]-(Worker) | 3 | history |
| co_hire | (Client)-[:HIRED]->(Worker)<-[:HIRED]-(Client)-[:HIRED]->(Worker) | 3 | history |

Reputation is worker-level evidence (training `REVIEWED` edges into the
worker), not a path from a request anchor, so it is reported separately
and excluded from path-coverage statistics.

Cold-start workers. `repeat_hire` and `co_hire` end in a HIRED edge into
the recommended worker, so a worker without training hires never has
one, and reputation falls back to the neutral prior. `hired_similar_skill`
CAN reach a cold-start worker: it uses the requesting client's training
hires and the cold-start worker's declared skills, both genuine
training-time graph facts -- no history is attributed to the worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.kg import schema


@dataclass(frozen=True)
class PathSpec:
    component: str
    labels: tuple[str, ...]
    relationships: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.relationships)


PATH_SPECS: dict[str, PathSpec] = {
    "skill": PathSpec("skill", (schema.LABEL_SKILL, schema.LABEL_WORKER), (schema.REL_HAS_SKILL,)),
    "related_skill": PathSpec(
        "related_skill", (schema.LABEL_SKILL, schema.LABEL_SKILL, schema.LABEL_WORKER),
        (schema.REL_RELATED_TO, schema.REL_HAS_SKILL),
    ),
    "category": PathSpec("category", (schema.LABEL_CATEGORY, schema.LABEL_WORKER), (schema.REL_SPECIALIZES_IN,)),
    "repeat_hire": PathSpec("history", (schema.LABEL_CLIENT, schema.LABEL_WORKER), (schema.REL_HIRED,)),
    "hired_similar_skill": PathSpec(
        "history", (schema.LABEL_CLIENT, schema.LABEL_WORKER, schema.LABEL_SKILL, schema.LABEL_WORKER),
        (schema.REL_HIRED, schema.REL_HAS_SKILL, schema.REL_HAS_SKILL),
    ),
    "co_hire": PathSpec(
        "history", (schema.LABEL_CLIENT, schema.LABEL_WORKER, schema.LABEL_CLIENT, schema.LABEL_WORKER),
        (schema.REL_HIRED, schema.REL_HIRED, schema.REL_HIRED),
    ),
}

PATH_TYPES = tuple(PATH_SPECS)
CATEGORY_PATH_TYPES = ("category",)
# Paths whose final edge is a HIRED edge INTO the recommended worker, i.e.
# that require the worker's own interaction history. A worker with no
# training hire (every cold-start worker) can never have one.
# `hired_similar_skill` is deliberately absent: it reaches the worker
# through a declared skill, using only the *client's* training hires.
WORKER_HISTORY_PATH_TYPES = ("repeat_hire", "co_hire")

COMPONENT_LABELS = {
    "skill": "direct skill match",
    "related_skill": "related-skill match",
    "category": "category specialization",
    "history": "your hiring history",
    "reputation": "training-period reviews",
}


@dataclass(frozen=True)
class ExplainedPath:
    path_type: str
    component: str
    nodes: tuple[tuple[str, str], ...]   # (label, id), anchor first, recommended worker last
    edges: tuple[dict, ...]               # relationship type + properties, in path order
    length: int
    path_weight: float                    # base weight * DWPC degree damping
    contribution: float                   # share of the final score attributed to this path
    text: str


@dataclass(frozen=True)
class ReputationEvidence:
    n_reviews: int
    mean_rating: float | None
    smoothed_rating: float
    prior_mean: float
    prior_used_only: bool
    contribution: float
    text: str


@dataclass
class Explanation:
    link: str
    worker_id: str
    rank: int
    score: float
    component_contributions: dict[str, float]
    component_raw: dict[str, float]
    cited_components: list[str]           # components with positive contribution, largest first
    n_paths_by_type: dict[str, int]       # untruncated counts
    path_lengths: list[int]               # one per request-anchored path, untruncated
    total_path_contribution: float        # sum over ALL paths, untruncated
    paths: list[ExplainedPath] = field(default_factory=list)  # displayed (top per component)
    reputation: ReputationEvidence | None = None
    summary: str = ""

    @property
    def n_supporting_paths(self) -> int:
        return sum(self.n_paths_by_type.values())

    @property
    def n_non_category_paths(self) -> int:
        return sum(n for t, n in self.n_paths_by_type.items() if t not in CATEGORY_PATH_TYPES)

    def to_dict(self) -> dict:
        return {
            "link": self.link, "worker_id": self.worker_id, "rank": self.rank, "score": self.score,
            "summary": self.summary,
            "component_contributions": self.component_contributions,
            "component_raw": self.component_raw,
            "cited_components": self.cited_components,
            "n_paths_by_type": self.n_paths_by_type,
            "n_supporting_paths": self.n_supporting_paths,
            "total_path_contribution": self.total_path_contribution,
            "paths": [
                {
                    "path_type": p.path_type, "component": p.component,
                    "nodes": [{"label": label, "id": node_id} for label, node_id in p.nodes],
                    "edges": list(p.edges), "length": p.length,
                    "path_weight": p.path_weight, "contribution": p.contribution, "text": p.text,
                }
                for p in self.paths
            ],
            "reputation": None if self.reputation is None else {
                "n_reviews": self.reputation.n_reviews, "mean_rating": self.reputation.mean_rating,
                "smoothed_rating": self.reputation.smoothed_rating, "prior_mean": self.reputation.prior_mean,
                "prior_used_only": self.reputation.prior_used_only,
                "contribution": self.reputation.contribution, "text": self.reputation.text,
            },
        }


def path_text(path_type: str, node_ids: tuple[str, ...], edges: tuple[dict, ...]) -> str:
    """Fixed template per path type, filled only from the path itself."""
    if path_type == "skill":
        return f"Has required skill '{node_ids[0]}'"
    if path_type == "related_skill":
        return (
            f"Has '{node_ids[1]}', related to required skill '{node_ids[0]}' "
            f"(background-corpus NPMI {edges[0]['npmi']:.2f})"
        )
    if path_type == "category":
        return f"Specializes in '{node_ids[0]}' ({edges[0]['role']} category)"
    if path_type == "repeat_hire":
        n = edges[0]["n_hires"]
        return f"You hired this worker before ({n} training-period hire{'s' if n != 1 else ''})"
    if path_type == "hired_similar_skill":
        return f"Shares skill '{node_ids[2]}' with {node_ids[1]}, whom you hired before"
    if path_type == "co_hire":
        return f"{node_ids[2]} also hired {node_ids[1]} (whom you hired) and this worker"
    raise ValueError(f"unknown path type {path_type!r}")


def reputation_text(n_reviews: int, mean_rating: float | None, smoothed: float, prior_mean: float) -> str:
    if n_reviews == 0:
        return f"No training-period reviews yet; neutral prior rating {prior_mean:.2f} used"
    return (
        f"Average training-period rating {mean_rating:.2f}/5 from {n_reviews} review{'s' if n_reviews != 1 else ''} "
        f"(smoothed {smoothed:.2f})"
    )


def build_summary(cited_components: list[str], paths: list[ExplainedPath], reputation: ReputationEvidence | None, max_items: int = 3) -> str:
    """One line per cited component (largest contribution first), each
    using that component's highest-contribution displayed path."""
    parts = []
    for component in cited_components[:max_items]:
        if component == "reputation":
            if reputation is not None:
                parts.append(reputation.text)
            continue
        top = next((p for p in paths if p.component == component), None)
        if top is not None:
            parts.append(top.text)
    if not parts:
        return "No positive graph evidence beyond eligibility"
    return "; ".join(parts)
