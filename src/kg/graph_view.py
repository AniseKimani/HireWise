"""Training-only, in-memory snapshot of the HireWise knowledge graph
(Day 7 scoring engine).

The Day 7 KG recommender never queries Neo4j per request. Instead, the
model-visible, temporally safe part of the graph is extracted ONCE into a
`TrainingGraphView`, from either of two sources that must agree exactly:

- `load_view_from_neo4j`: read-only Cypher run in READ-access
  transactions against the live Day 6 graph;
- `load_view_from_kg_csvs`: the deterministic Day 6 CSV artifacts under
  `data/processed/kg/` (hash-checked by scripts/verify_day6.py).

`TrainingGraphView.fingerprint()` hashes a canonical serialization of
every table, so Neo4j/CSV parity (and rerun reproducibility) is a single
string comparison.

Temporal safety (docs/experimental_design.md Sections 11 and 14):

- `HIRED` edges are restricted to `split = TRAIN_SPLIT`. The split is a
  module constant, not a parameter, so no caller can widen the view to
  validation/test edges.
- `REVIEWED` edges have no `split` property (Day 6 schema). They are kept
  only when joined, on (client, worker, link), to a training `HIRED`
  edge -- a plain `MATCH ()-[:REVIEWED]->()` would include validation and
  test ratings.
- `REQUESTED_SERVICE` is not extracted at all: no Day 7 score component
  needs it, so it cannot leak.
- `INTERESTED_IN` is training-derived by construction (Day 6) and is kept
  for graph inspection/statistics only; it is not a score component.
- Worker nodes contribute only their `worker_id`. No other node property
  (in particular not `is_cold_start` or `join_day`) is extracted:
  eligibility by join day is the shared evaluation protocol's job.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.kg import schema

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG_CSV_DIR = REPO_ROOT / "data" / "processed" / "kg"

TRAIN_SPLIT = "train"

TABLES = (
    "workers", "skills", "categories", "has_skill", "specializes_in",
    "related_to", "train_hires", "train_reviews", "interested_in",
)


@dataclass(frozen=True)
class TrainingGraphView:
    """Canonical, sorted tuples; equality and fingerprint ignore `source`."""

    workers: tuple[str, ...]
    skills: tuple[str, ...]
    categories: tuple[str, ...]
    has_skill: tuple[tuple[str, str], ...]                     # (worker_id, skill)
    specializes_in: tuple[tuple[str, str, str], ...]           # (worker_id, category, role)
    related_to: tuple[tuple[str, str, float, int], ...]        # (skill_a, skill_b, npmi, support), skill_a < skill_b
    train_hires: tuple[tuple[str, str, str, float], ...]       # (client_id, worker_id, link, simulated_day)
    train_reviews: tuple[tuple[str, str, str, int, float], ...]  # (client_id, worker_id, link, rating, hire_simulated_day)
    interested_in: tuple[tuple[str, str, int], ...]            # (client_id, category, training_request_count)
    source: str = field(default="", compare=False)

    def canonical_tables(self) -> dict[str, list]:
        return {name: [list(row) if isinstance(row, tuple) else row for row in getattr(self, name)] for name in TABLES}

    def table_fingerprints(self) -> dict[str, str]:
        return {name: _sha256_json(rows) for name, rows in self.canonical_tables().items()}

    def fingerprint(self) -> str:
        return _sha256_json(self.table_fingerprints())

    def stats(self) -> dict:
        hire_pairs = {(c, w) for c, w, _, _ in self.train_hires}
        role_counts = Counter(role for _, _, role in self.specializes_in)
        return {
            "n_workers": len(self.workers),
            "n_skills": len(self.skills),
            "n_categories": len(self.categories),
            "n_has_skill": len(self.has_skill),
            "n_specializes_in": len(self.specializes_in),
            "n_specializes_in_primary": role_counts.get("primary", 0),
            "n_specializes_in_secondary": role_counts.get("secondary", 0),
            "n_related_to": len(self.related_to),
            "n_train_hires": len(self.train_hires),
            "n_train_hire_distinct_pairs": len(hire_pairs),
            "n_clients_with_train_hires": len({c for c, _ in hire_pairs}),
            "n_workers_with_train_hires": len({w for _, w in hire_pairs}),
            "n_train_reviews": len(self.train_reviews),
            "n_workers_with_train_reviews": len({w for _, w, _, _, _ in self.train_reviews}),
            "mean_train_rating": (
                sum(r for _, _, _, r, _ in self.train_reviews) / len(self.train_reviews) if self.train_reviews else None
            ),
            "max_train_hire_simulated_day": max((d for _, _, _, d in self.train_hires), default=None),
            "n_interested_in": len(self.interested_in),
        }


def _sha256_json(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_view(
    workers, skills, categories, has_skill, specializes_in, related_to,
    train_hires, train_reviews, interested_in, source: str,
) -> TrainingGraphView:
    """Normalize types and sort every table so that the same graph always
    yields the same view regardless of source or row order."""
    return TrainingGraphView(
        workers=tuple(sorted({str(w) for w in workers})),
        skills=tuple(sorted({str(s) for s in skills})),
        categories=tuple(sorted({str(c) for c in categories})),
        has_skill=tuple(sorted({(str(w), str(s)) for w, s in has_skill})),
        specializes_in=tuple(sorted({(str(w), str(c), str(r)) for w, c, r in specializes_in})),
        related_to=tuple(sorted((str(a), str(b), float(n), int(s)) for a, b, n, s in related_to)),
        train_hires=tuple(sorted((str(c), str(w), str(l), float(d)) for c, w, l, d in train_hires)),
        train_reviews=tuple(sorted((str(c), str(w), str(l), int(r), float(d)) for c, w, l, r, d in train_reviews)),
        interested_in=tuple(sorted((str(c), str(cat), int(n)) for c, cat, n in interested_in)),
        source=source,
    )


# ---------------------------------------------------------------------------
# Neo4j source (read-only Cypher, READ-access transactions)
# ---------------------------------------------------------------------------
# Every query ends in ORDER BY so row order is never left to the server,
# and the training split is always a bound parameter fed from TRAIN_SPLIT.

EXTRACTION_QUERIES = {
    "workers": f"MATCH (w:{schema.LABEL_WORKER}) RETURN w.worker_id AS worker_id ORDER BY worker_id",
    "skills": f"MATCH (s:{schema.LABEL_SKILL}) RETURN s.name AS name ORDER BY name",
    "categories": f"MATCH (cat:{schema.LABEL_CATEGORY}) RETURN cat.name AS name ORDER BY name",
    "has_skill": (
        f"MATCH (w:{schema.LABEL_WORKER})-[:{schema.REL_HAS_SKILL}]->(s:{schema.LABEL_SKILL}) "
        "RETURN w.worker_id AS worker_id, s.name AS skill ORDER BY worker_id, skill"
    ),
    "specializes_in": (
        f"MATCH (w:{schema.LABEL_WORKER})-[r:{schema.REL_SPECIALIZES_IN}]->(cat:{schema.LABEL_CATEGORY}) "
        "RETURN w.worker_id AS worker_id, cat.name AS category, r.role AS role ORDER BY worker_id, category"
    ),
    "related_to": (
        f"MATCH (a:{schema.LABEL_SKILL})-[r:{schema.REL_RELATED_TO}]->(b:{schema.LABEL_SKILL}) "
        "RETURN a.name AS skill_a, b.name AS skill_b, r.npmi AS npmi, r.support AS support ORDER BY skill_a, skill_b"
    ),
    "train_hires": (
        f"MATCH (c:{schema.LABEL_CLIENT})-[h:{schema.REL_HIRED}]->(w:{schema.LABEL_WORKER}) "
        "WHERE h.split = $split "
        "RETURN c.client_id AS client_id, w.worker_id AS worker_id, h.link AS link, h.simulated_day AS simulated_day "
        "ORDER BY link, worker_id, client_id"
    ),
    "train_reviews": (
        f"MATCH (c:{schema.LABEL_CLIENT})-[rv:{schema.REL_REVIEWED}]->(w:{schema.LABEL_WORKER}) "
        f"MATCH (c)-[h:{schema.REL_HIRED} {{link: rv.link}}]->(w) "
        "WHERE h.split = $split "
        "RETURN c.client_id AS client_id, w.worker_id AS worker_id, rv.link AS link, rv.rating AS rating, "
        "h.simulated_day AS simulated_day ORDER BY link, worker_id, client_id"
    ),
    "interested_in": (
        f"MATCH (c:{schema.LABEL_CLIENT})-[i:{schema.REL_INTERESTED_IN}]->(cat:{schema.LABEL_CATEGORY}) "
        "RETURN c.client_id AS client_id, cat.name AS category, i.training_request_count AS training_request_count "
        "ORDER BY client_id, category"
    ),
}


def load_view_from_neo4j(client) -> TrainingGraphView:
    """`client` is an open `src.kg.neo4j_client.Neo4jClient`."""
    params = {"split": TRAIN_SPLIT}
    rows = {name: client.run_read(query, params if "$split" in query else None) for name, query in EXTRACTION_QUERIES.items()}
    return build_view(
        workers=[r["worker_id"] for r in rows["workers"]],
        skills=[r["name"] for r in rows["skills"]],
        categories=[r["name"] for r in rows["categories"]],
        has_skill=[(r["worker_id"], r["skill"]) for r in rows["has_skill"]],
        specializes_in=[(r["worker_id"], r["category"], r["role"]) for r in rows["specializes_in"]],
        related_to=[(r["skill_a"], r["skill_b"], r["npmi"], r["support"]) for r in rows["related_to"]],
        train_hires=[(r["client_id"], r["worker_id"], r["link"], r["simulated_day"]) for r in rows["train_hires"]],
        train_reviews=[
            (r["client_id"], r["worker_id"], r["link"], r["rating"], r["simulated_day"]) for r in rows["train_reviews"]
        ],
        interested_in=[(r["client_id"], r["category"], r["training_request_count"]) for r in rows["interested_in"]],
        source="neo4j",
    )


# ---------------------------------------------------------------------------
# CSV source (deterministic Day 6 artifacts)
# ---------------------------------------------------------------------------

def load_view_from_kg_csvs(kg_dir: str | Path = DEFAULT_KG_CSV_DIR) -> TrainingGraphView:
    kg_dir = Path(kg_dir)
    workers = pd.read_csv(kg_dir / "nodes_worker.csv", usecols=["worker_id"])
    skills = pd.read_csv(kg_dir / "nodes_skill.csv")
    categories = pd.read_csv(kg_dir / "nodes_category.csv")
    has_skill = pd.read_csv(kg_dir / "rel_has_skill.csv")
    specializes_in = pd.read_csv(kg_dir / "rel_specializes_in.csv")
    related_to = pd.read_csv(kg_dir / "rel_related_to.csv")
    hired = pd.read_csv(kg_dir / "rel_hired.csv")
    reviewed = pd.read_csv(kg_dir / "rel_reviewed.csv")
    interested_in = pd.read_csv(kg_dir / "rel_interested_in.csv")

    train_hires = hired.loc[hired["split"] == TRAIN_SPLIT, ["client_id", "worker_id", "link", "simulated_day"]]
    train_reviews = reviewed.merge(train_hires, on=["client_id", "worker_id", "link"], how="inner")

    return build_view(
        workers=workers["worker_id"].tolist(),
        skills=skills["name"].tolist(),
        categories=categories["name"].tolist(),
        has_skill=zip(has_skill["worker_id"], has_skill["skill"]),
        specializes_in=zip(specializes_in["worker_id"], specializes_in["category"], specializes_in["role"]),
        related_to=zip(related_to["skill_a"], related_to["skill_b"], related_to["npmi"], related_to["support"]),
        train_hires=zip(train_hires["client_id"], train_hires["worker_id"], train_hires["link"], train_hires["simulated_day"]),
        train_reviews=zip(
            train_reviews["client_id"], train_reviews["worker_id"], train_reviews["link"],
            train_reviews["rating"], train_reviews["simulated_day"],
        ),
        interested_in=zip(interested_in["client_id"], interested_in["category"], interested_in["training_request_count"]),
        source="csv",
    )


def compare_views(a: TrainingGraphView, b: TrainingGraphView) -> dict[str, bool]:
    """Per-table equality, for a parity report more informative than a
    single fingerprint mismatch."""
    return {name: getattr(a, name) == getattr(b, name) for name in TABLES}
