"""Day 9 API integration tests against the real frozen stack: routes,
frozen-hybrid parity, determinism, and field allow-lists. The live Neo4j
checks are READ-ONLY and skip cleanly when Neo4j is unavailable. Skips
entirely when the processed data or Day 8 results are absent."""
import json
import unittest
from pathlib import Path

import pandas as pd

from src.app import create_app
from src.app.serializers import RECOMMENDATION_FIELDS, REQUEST_FIELDS, WORKER_FIELDS
from src.app.services import RecommendationService, expected_graph_counts, probe_neo4j, worker_graph_from_neo4j
from src.data.config import load_config
from src.evaluation.config import load_evaluation_config
from src.evaluation.protocol import EligibilityIndex, build_eval_requests
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
DAY8 = REPO_ROOT / "results" / "day8"

ALLOWED_KEYS = (
    set(REQUEST_FIELDS) | set(WORKER_FIELDS) | set(RECOMMENDATION_FIELDS)
    | {"request", "model", "k", "n_candidates", "n_returned", "components", "recommendations",
       "name", "weights", "normalization", "selected_config_sha256", "hybrid_config_sha256",
       "cbf", "cf", "kg", "path_type", "text"}
)
FORBIDDEN_KEY_MARKERS = ("quality", "proficiency", "true_skill", "taste", "utility", "noise", "oracle", "cold_start",
                         "relevance", "grade", "event", "shortlist", "hired", "outcome", "private")


def all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from all_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from all_keys(item)


def _neo4j_available() -> bool:
    return probe_neo4j()["available"]


@unittest.skipUnless(SYNTHETIC.exists() and (DAY8 / "selected_hybrid_config.json").exists(), "processed data / Day 8 results absent")
class TestDay9Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = RecommendationService.build()
        cls.client = create_app("testing", service=cls.service,
                                neo4j_probe=lambda: {"available": True, "graph_matches_day6_counts": True}).test_client()
        eval_cfg = load_evaluation_config()
        index = EligibilityIndex(cls.service.data.workers, cls.service.data.known_countries)
        cls.eval_requests = [r for s in ("validation", "test") for r in build_eval_requests(cls.service.data, index, s, eval_cfg)]
        included = [r for r in cls.eval_requests if not r.excluded]
        cls.sample = included[::170]
        cls.link = cls.sample[0].link
        requests = cls.service.data.requests
        cls.train_link = requests.loc[requests["split"] == "train", "link"].iloc[0]

    def get(self, path, **params):
        return self.client.get(path, query_string=params)

    # ---- request / worker routes ---------------------------------------
    def test_request_lookup(self):
        r = self.get("/api/requests", link=self.link)
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(tuple(sorted(body)), tuple(sorted(REQUEST_FIELDS)))
        self.assertEqual(body["link"], self.link)
        self.assertEqual(self.get("/api/requests", link="https://example.invalid/none").status_code, 404)

    def test_worker_lookup(self):
        worker_id = self.service.data.workers["worker_id"].iloc[0]
        r = self.client.get(f"/api/workers/{worker_id}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.get_json()), set(WORKER_FIELDS))
        self.assertEqual(self.client.get("/api/workers/W999999").status_code, 404)

    # ---- recommendations --------------------------------------------------
    def test_default_k_and_structure(self):
        body = self.get("/api/recommendations", link=self.link).get_json()
        self.assertEqual(body["k"], 10)
        self.assertEqual(body["n_returned"], 10)
        recs = body["recommendations"]
        self.assertEqual([r["rank"] for r in recs], list(range(1, 11)))
        self.assertEqual(len({r["worker"]["worker_id"] for r in recs}), 10)
        scores = [r["score"] for r in recs]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(body["model"]["weights"], {"cbf": 0.4, "cf": 0.1, "kg": 0.5})
        for r in recs:
            self.assertEqual(set(r), set(RECOMMENDATION_FIELDS))
            self.assertAlmostEqual(sum(r["contributions"].values()), r["score"], places=12)
            self.assertTrue(r["explanation"].startswith(("Recommended because", "No component")))
            if r["contributions"]["kg"] > 0:
                self.assertTrue(r["kg_summary"])
                self.assertIn("Strongest KG evidence", r["explanation"])
                self.assertGreaterEqual(len(r["kg_evidence"]), 1)

    def test_custom_k_and_invalid_k(self):
        self.assertEqual(self.get("/api/recommendations", link=self.link, k="3").get_json()["n_returned"], 3)
        big = self.get("/api/recommendations", link=self.link, k="50").get_json()
        self.assertEqual(big["n_returned"], min(50, big["n_candidates"]))
        self.assertEqual(self.get("/api/recommendations", link=self.link, k="0").status_code, 400)
        self.assertEqual(self.get("/api/recommendations", link=self.link, k="x").status_code, 400)

    def test_unknown_and_training_requests(self):
        self.assertEqual(self.get("/api/recommendations", link="https://example.invalid/none").status_code, 404)
        self.assertEqual(self.get("/api/recommendations", link=self.train_link).status_code, 409)

    # ---- frozen hybrid parity -----------------------------------------------
    def test_api_matches_direct_frozen_day8_hybrid(self):
        with open(DAY8 / "selected_hybrid_config.json", encoding="utf-8") as fh:
            weights = json.load(fh)["weights"]
        models = build_frozen_models(self.service.data, load_config().seed, load_cf_config().min_similarity)
        direct = HybridRecommender({"cbf": cbf_score_fn(models["cbf"]), "cf": cf_score_fn(models["cf"]),
                                    "kg": kg_score_fn(models["kg"])}, weights)
        for req in self.sample:
            self.assertEqual(self.service.candidates_for(req.link), req.candidates)
            body = self.get("/api/recommendations", link=req.link).get_json()
            self.assertEqual(body["n_candidates"], req.n_candidates)
            expected = direct.explain(req.link, req.candidates, 10)
            self.assertEqual([r["worker"]["worker_id"] for r in body["recommendations"]], [e.worker_id for e in expected])
            self.assertEqual([r["score"] for r in body["recommendations"]], [e.score for e in expected])
            self.assertEqual([r["contributions"] for r in body["recommendations"]], [e.contributions for e in expected])

    # ---- determinism ----------------------------------------------------------
    def test_repeated_and_fresh_service_responses_identical(self):
        first = self.get("/api/recommendations", link=self.link).get_data()
        self.assertEqual(first, self.get("/api/recommendations", link=self.link).get_data())
        fresh = create_app("testing", service=RecommendationService.build()).test_client()
        self.assertEqual(first, fresh.get("/api/recommendations", query_string={"link": self.link}).get_data())

    # ---- leakage ------------------------------------------------------------------
    def test_responses_only_contain_allow_listed_keys(self):
        bodies = [self.get("/api/recommendations", link=r.link).get_json() for r in self.sample[:5]]
        bodies.append(self.get("/api/requests", link=self.link).get_json())
        for body in bodies:
            keys = set(all_keys(body))
            self.assertTrue(keys <= ALLOWED_KEYS, keys - ALLOWED_KEYS)
            for key in keys:
                for marker in FORBIDDEN_KEY_MARKERS:
                    self.assertNotIn(marker, key.lower())

    def test_request_lookup_does_not_reveal_outcome(self):
        body = self.get("/api/requests", link=self.link).get_json()
        hired = set(self.service.data.interactions.loc[self.service.data.interactions["link"] == self.link, "worker_id"])
        self.assertFalse(any(w in json.dumps(body) for w in hired))

    # ---- Neo4j (read-only, live) ----------------------------------------------------
    @unittest.skipUnless(_neo4j_available(), "Neo4j unavailable; read-only live checks skipped")
    def test_live_probe_and_worker_graph_read_only(self):
        status = probe_neo4j()
        self.assertEqual(status, {"available": True, "graph_matches_day6_counts": True})
        worker = self.service.data.workers.iloc[0]
        graph = worker_graph_from_neo4j(worker["worker_id"])
        self.assertEqual(graph["skills"], sorted(worker["declared_skills"].split("|")))
        self.assertEqual(probe_neo4j(), status)  # nothing changed in the graph
        self.assertEqual(expected_graph_counts(), {"nodes": 6420, "relationships": 79165})


if __name__ == "__main__":
    unittest.main()
