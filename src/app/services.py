"""Recommendation service: the only layer between HTTP routes and the
frozen recommender stack (Day 9).

Built ONCE per application: model-visible data, the shared eligibility
index, the frozen Day 4/5/7 models (src/recommenders/frozen_models.py,
which also checks the KG snapshot fingerprint), and the frozen Day 8
hybrid with the weights recorded in results/day8/selected_hybrid_config.json.

For a recommendation the service:
1. looks up the request by its `link` (the repository's request id);
2. rebuilds the eligible candidate universe with the SAME function the
   evaluation protocol uses (EligibilityIndex.eligible_worker_ids);
   relevance labels are never computed or read here;
3. ranks with the frozen hybrid and takes its actual contribution-based
   explanations, with KG path evidence from the frozen Day 7 explainer;
4. returns plain, allow-listed data (src/app/serializers.py).

Only validation/test-period requests are recommendable: the frozen KG
component (and therefore the hybrid) only scores requests strictly after
its training-only history, so training-period requests are rejected
rather than scored with their own outcomes visible.

No recommender module imports Flask; this module does not either.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.app.serializers import serialize_recommendation, serialize_request, serialize_worker
from src.data.config import load_config
from src.data.io import sha256_of
from src.evaluation.protocol import EligibilityIndex, load_protocol_data
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import COMPONENTS, HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn
from src.recommenders.hybrid_config import load_hybrid_config
from src.recommenders.knowledge_graph import TemporalLeakageError

REPO_ROOT = Path(__file__).resolve().parents[2]
DAY2_PROCESSED = REPO_ROOT / "data" / "processed"
DAY8_RESULTS = REPO_ROOT / "results" / "day8"
KG_PREPARE_MANIFEST = DAY2_PROCESSED / "kg" / "kg_prepare_manifest.json"
RECOMMENDABLE_SPLITS = ("validation", "test")
MODEL_NAME = "hirewise-hybrid-day8"


class UnknownRequest(LookupError):
    pass


class UnknownWorker(LookupError):
    pass


class RequestNotRecommendable(ValueError):
    pass


class FrozenArtifactMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelVersion:
    name: str
    weights: dict
    normalization: str
    selected_config_sha256: str
    hybrid_config_sha256: str

    def to_dict(self) -> dict:
        return {
            "name": self.name, "weights": dict(self.weights), "normalization": self.normalization,
            "selected_config_sha256": self.selected_config_sha256[:12],
            "hybrid_config_sha256": self.hybrid_config_sha256[:12],
        }


def load_selected_hybrid() -> tuple[dict, ModelVersion]:
    """Read and integrity-check the frozen Day 8 selection."""
    selected_path = DAY8_RESULTS / "selected_hybrid_config.json"
    with open(selected_path, encoding="utf-8") as fh:
        selected = json.load(fh)
    with open(DAY8_RESULTS / "day8_manifest.json", encoding="utf-8") as fh:
        manifest = json.load(fh)
    recorded = {e["path"]: e["sha256"] for e in manifest["files"]}
    selected_sha = sha256_of(selected_path)
    if recorded.get("results/day8/selected_hybrid_config.json") != selected_sha:
        raise FrozenArtifactMismatch("results/day8/selected_hybrid_config.json differs from the Day 8 manifest")
    hybrid_sha = sha256_of(REPO_ROOT / "config" / "hybrid_config.yaml")
    if hybrid_sha != selected["hybrid_config_sha256"]:
        raise FrozenArtifactMismatch("config/hybrid_config.yaml differs from the configuration Day 8 selected from")
    if selected["weights"] not in load_hybrid_config().weight_grid:
        raise FrozenArtifactMismatch("selected hybrid weights are not in the predeclared grid")
    return selected, ModelVersion(MODEL_NAME, selected["weights"], selected["normalization"], selected_sha, hybrid_sha)


def expected_graph_counts() -> dict[str, int]:
    """Day 6 node/relationship totals, from the Day 6 manifest row counts."""
    with open(KG_PREPARE_MANIFEST, encoding="utf-8") as fh:
        files = json.load(fh)["files"]
    return {
        "nodes": sum(e["rows"] for e in files if "/nodes_" in e["path"]),
        "relationships": sum(e["rows"] for e in files if "/rel_" in e["path"]),
    }


class RecommendationService:
    def __init__(self, data, eligibility: EligibilityIndex, models: dict, hybrid: HybridRecommender,
                 model_version: ModelVersion, titles: dict[str, str], startup_seconds: float):
        self.data = data
        self.eligibility = eligibility
        self.models = models
        self.hybrid = hybrid
        self.model_version = model_version
        self.startup_seconds = startup_seconds
        self._lock = threading.Lock()  # component caches are plain dicts

        requests = data.requests.copy()
        requests["title"] = requests["link"].map(titles)
        self._requests = {row["link"]: row for row in requests.to_dict(orient="records")}
        self._workers = {row["worker_id"]: row for row in data.workers.to_dict(orient="records")}
        self._kg_index = models["kg"].index

        # Browsable requests (Day 10): only the recommendable ones, in a
        # fixed chronological order (simulated day, then link).
        browsable = sorted((r for r in self._requests.values() if r["split"] in RECOMMENDABLE_SPLITS),
                           key=lambda r: (r["simulated_day"], r["link"]))
        self._browsable = [serialize_request(r) for r in browsable]
        self._browsable_categories = sorted({r["category"] for r in self._browsable})

    @classmethod
    def build(cls) -> "RecommendationService":
        t0 = time.perf_counter()
        _, version = load_selected_hybrid()
        data = load_protocol_data()
        eligibility = EligibilityIndex(data.workers, data.known_countries)
        models = build_frozen_models(data, load_config().seed, load_cf_config().min_similarity)
        hybrid = HybridRecommender(
            {"cbf": cbf_score_fn(models["cbf"]), "cf": cf_score_fn(models["cf"]), "kg": kg_score_fn(models["kg"])},
            version.weights,
        )
        titles = pd.read_csv(DAY2_PROCESSED / "sampled_requests.csv", usecols=["link", "title"])
        return cls(data, eligibility, models, hybrid, version, dict(zip(titles["link"], titles["title"])),
                   time.perf_counter() - t0)

    # ------------------------------------------------------------------
    def _request_row(self, link: str) -> dict:
        row = self._requests.get(link)
        if row is None:
            raise UnknownRequest(link)
        return row

    def get_request(self, link: str) -> dict:
        return serialize_request(self._request_row(link))

    def list_requests(self, page: int, per_page: int, q: str | None = None, category: str | None = None,
                      skill: str | None = None) -> dict:
        """One page of recommendable (validation/test) requests, optionally
        filtered by a case-insensitive title substring, an exact category,
        and/or a required skill (case-insensitive exact match). Items use
        the same allow-listed request serializer as the single lookup, plus
        the size of the eligible candidate pool."""
        q_lower = q.lower() if q else None
        skill_lower = skill.lower() if skill else None
        matches = [
            r for r in self._browsable
            if (q_lower is None or q_lower in (r["title"] or "").lower())
            and (category is None or r["category"] == category)
            and (skill_lower is None or any(s.lower() == skill_lower for s in r["required_skills"]))
        ]
        total = len(matches)
        start = (page - 1) * per_page
        items = [
            {**r, "n_eligible_workers": len(self.candidates_for(r["link"]))}
            for r in matches[start:start + per_page]
        ]
        return {
            "items": items, "page": page, "per_page": per_page, "total": total,
            "total_pages": (total + per_page - 1) // per_page,
            "filters": {"q": q, "category": category, "skill": skill},
            "available_categories": list(self._browsable_categories),
        }

    def worker_profile(self, worker_id: str) -> dict:
        row = self._workers.get(worker_id)
        if row is None:
            raise UnknownWorker(worker_id)
        return serialize_worker(row, self._kg_index.review_count.get(worker_id, 0), self._kg_index.review_sum.get(worker_id, 0))

    def has_worker(self, worker_id: str) -> bool:
        return worker_id in self._workers

    def candidates_for(self, link: str) -> list[str]:
        row = self._request_row(link)
        return self.eligibility.eligible_worker_ids(row["category"], row["simulated_day"], row["location_requirement"])

    def recommend(self, link: str, k: int, kg_evidence_paths: int = 3) -> dict:
        row = self._request_row(link)
        if row["split"] not in RECOMMENDABLE_SPLITS:
            raise RequestNotRecommendable(
                "only validation/test-period requests can be recommended; the frozen models' history covers the training period"
            )
        candidates = self.candidates_for(link)
        with self._lock:
            try:
                explanations = self.hybrid.explain(link, candidates, k, kg_recommender=self.models["kg"],
                                                   kg_paths=kg_evidence_paths)
            except TemporalLeakageError as exc:
                raise RequestNotRecommendable(str(exc)) from exc
        return {
            "request": serialize_request(row),
            "model": self.model_version.to_dict(),
            "k": k,
            "n_candidates": len(candidates),
            "n_returned": len(explanations),
            "components": list(COMPONENTS),
            "recommendations": [
                serialize_recommendation(e, self.worker_profile(e.worker_id), kg_evidence_paths) for e in explanations
            ],
        }


# ---------------------------------------------------------------------------
# Neo4j (read-only)
# ---------------------------------------------------------------------------

WORKER_GRAPH_QUERIES = {
    "skills": "MATCH (w:Worker {worker_id: $worker_id})-[:HAS_SKILL]->(s:Skill) RETURN s.name AS skill ORDER BY skill",
    "categories": ("MATCH (w:Worker {worker_id: $worker_id})-[r:SPECIALIZES_IN]->(c:ServiceCategory) "
                   "RETURN c.name AS category, r.role AS role ORDER BY role, category"),
    "location": ("MATCH (w:Worker {worker_id: $worker_id})-[:LOCATED_IN]->(l:Location) "
                 "RETURN l.name AS location ORDER BY location"),
}
COUNT_QUERIES = {
    "nodes": "MATCH (n) RETURN count(n) AS c",
    "relationships": "MATCH ()-[r]->() RETURN count(r) AS c",
}


class Neo4jUnavailable(RuntimeError):
    pass


def probe_neo4j() -> dict:
    """Read-only availability probe. Never returns error text (it may
    contain connection details)."""
    from src.kg.neo4j_client import Neo4jClient

    try:
        with Neo4jClient() as client:
            counts = {name: client.run_read(q)[0]["c"] for name, q in COUNT_QUERIES.items()}
    except Exception:
        return {"available": False, "graph_matches_day6_counts": None}
    return {"available": True, "graph_matches_day6_counts": counts == expected_graph_counts()}


def worker_graph_from_neo4j(worker_id: str) -> dict:
    """Static profile neighbourhood of one worker from the live graph, via
    fixed parameterized read-only Cypher. Interaction edges (HIRED,
    REVIEWED) are deliberately not returned: they include
    validation/test outcomes."""
    from src.kg.neo4j_client import Neo4jClient

    try:
        with Neo4jClient() as client:
            rows = {name: client.run_read(q, {"worker_id": worker_id}) for name, q in WORKER_GRAPH_QUERIES.items()}
    except Exception as exc:
        raise Neo4jUnavailable("Neo4j is not reachable") from exc
    return {
        "worker_id": worker_id,
        "skills": [r["skill"] for r in rows["skills"]],
        "categories": [{"category": r["category"], "role": r["role"]} for r in rows["categories"]],
        "location": rows["location"][0]["location"] if rows["location"] else None,
        "source": "neo4j",
    }
