"""Day 9 Flask app unit tests: factory/config, health reporting, input
validation, JSON error handling, serializers, and read-only/no-Flask
boundaries. Uses fakes only -- no data files, no Neo4j."""
import ast
import inspect
import json
import math
import unittest
from pathlib import Path

import numpy as np

from src.app import create_app, routes, services
from src.app.config import ProductionConfig, resolve_config
from src.app.serializers import json_safe, serialize_worker, split_pipe
from src.app.services import Neo4jUnavailable, RequestNotRecommendable, UnknownRequest, UnknownWorker

REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeVersion:
    def to_dict(self):
        return {"name": "fake", "weights": {"cbf": 0.4, "cf": 0.1, "kg": 0.5}}


class FakeService:
    model_version = FakeVersion()

    def __init__(self, fail=None):
        self.fail = fail
        self.calls = []

    def get_request(self, link):
        if link == "missing":
            raise UnknownRequest(link)
        return {"link": link}

    def worker_profile(self, worker_id):
        if worker_id == "W404":
            raise UnknownWorker(worker_id)
        return {"worker_id": worker_id}

    def has_worker(self, worker_id):
        return worker_id != "W404"

    def recommend(self, link, k, kg_evidence_paths=3):
        self.calls.append((link, k))
        if self.fail == "boom":
            raise RuntimeError("secret path C:/x/.env leaked?")
        if link == "missing":
            raise UnknownRequest(link)
        if link == "train":
            raise RequestNotRecommendable("training period")
        return {"k": k, "recommendations": []}


def app_with(service=None, probe=None, reader=None, config="testing"):
    return create_app(config, service=service, neo4j_probe=probe, worker_graph_reader=reader).test_client()


class TestFactoryAndConfig(unittest.TestCase):
    def test_testing_app_creates_without_loading_anything(self):
        app = create_app("testing")
        self.assertTrue(app.config["TESTING"])
        self.assertNotIn("hirewise_service", app.extensions)

    def test_dict_overrides_apply(self):
        app = create_app({"MAX_K": 20, "LOAD_RECOMMENDER_ON_STARTUP": False, "NEO4J_HEALTHCHECK_ENABLED": False})
        self.assertEqual(app.config["MAX_K"], 20)

    def test_default_environment_is_production_without_debug(self):
        import os
        if not os.environ.get("HIREWISE_ENV"):
            self.assertIs(resolve_config(None), ProductionConfig)
        self.assertFalse(ProductionConfig.DEBUG)
        with self.assertRaises(ValueError):
            resolve_config("staging")

    def test_override_of_max_k_is_enforced(self):
        client = create_app({"MAX_K": 5, "LOAD_RECOMMENDER_ON_STARTUP": False, "NEO4J_HEALTHCHECK_ENABLED": False},
                            service=FakeService()).test_client()
        self.assertEqual(client.get("/api/recommendations?link=x&k=6").status_code, 400)
        self.assertEqual(client.get("/api/recommendations?link=x&k=5").status_code, 200)


class TestHealth(unittest.TestCase):
    def test_ok_when_recommender_and_neo4j_ready(self):
        r = app_with(FakeService(), probe=lambda: {"available": True, "graph_matches_day6_counts": True}).get("/api/health")
        body = r.get_json()
        self.assertEqual((r.status_code, body["status"]), (200, "ok"))
        self.assertTrue(body["recommender"]["ready"])
        self.assertFalse(body["recommendations_require_neo4j"])

    def test_degraded_when_neo4j_unavailable(self):
        body = app_with(FakeService(), probe=lambda: {"available": False, "graph_matches_day6_counts": None}).get("/api/health").get_json()
        self.assertEqual(body["status"], "degraded")
        self.assertFalse(body["neo4j"]["available"])

    def test_unavailable_without_recommender(self):
        r = app_with(None).get("/api/health")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.get_json()["status"], "unavailable")
        self.assertFalse(r.get_json()["neo4j"]["checked"])

    def test_health_exposes_no_secrets(self):
        text = app_with(FakeService(), probe=lambda: {"available": True, "graph_matches_day6_counts": True}).get("/api/health").get_data(as_text=True)
        for marker in ("password", "bolt://", "neo4j_uri", "NEO4J_", ".env", "C:\\", "/Users/"):
            self.assertNotIn(marker, text)


class TestValidationAndErrors(unittest.TestCase):
    def setUp(self):
        self.service = FakeService()
        self.client = app_with(self.service)

    def test_default_and_custom_k(self):
        self.client.get("/api/recommendations?link=x")
        self.client.get("/api/recommendations?link=x&k=25")
        self.assertEqual(self.service.calls, [("x", 10), ("x", 25)])

    def test_invalid_k_rejected(self):
        for k in ("0", "-1", "51", "abc", "5.0", "+5", " 5", "", "1e3", "999999999"):
            r = self.client.get("/api/recommendations", query_string={"link": "x", "k": k})
            self.assertEqual(r.status_code, 400, k)
            self.assertEqual(r.get_json()["error"]["code"], "invalid_k")
        self.assertEqual(self.service.calls, [])

    def test_link_validation(self):
        self.assertEqual(self.client.get("/api/recommendations").get_json()["error"]["code"], "missing_link")
        self.assertEqual(self.client.get("/api/requests?link=%20").status_code, 400)
        long_link = "x" * 3000
        self.assertEqual(self.client.get("/api/requests", query_string={"link": long_link}).get_json()["error"]["code"], "invalid_link")

    def test_unknown_request_and_worker(self):
        self.assertEqual(self.client.get("/api/requests?link=missing").status_code, 404)
        self.assertEqual(self.client.get("/api/recommendations?link=missing").get_json()["error"]["code"], "unknown_request")
        self.assertEqual(self.client.get("/api/workers/W404").get_json()["error"]["code"], "unknown_worker")

    def test_malformed_worker_id(self):
        r = self.client.get("/api/workers/bad$id")
        self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (400, "invalid_worker_id"))

    def test_training_request_not_recommendable(self):
        r = self.client.get("/api/recommendations?link=train")
        self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (409, "request_not_recommendable"))

    def test_recommender_unavailable(self):
        r = app_with(None).get("/api/recommendations?link=x")
        self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (503, "recommender_unavailable"))

    def test_neo4j_unavailable_on_graph_route(self):
        def reader(worker_id):
            raise Neo4jUnavailable("down")
        r = app_with(FakeService(), reader=reader).get("/api/workers/W1/graph")
        self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (503, "neo4j_unavailable"))
        self.assertEqual(app_with(FakeService(), reader=reader).get("/api/workers/W404/graph").status_code, 404)

    def test_unknown_route_and_method_are_json(self):
        self.assertEqual(self.client.get("/api/nope").get_json()["error"]["code"], "not_found")
        self.assertEqual(self.client.post("/api/health").get_json()["error"]["code"], "method_not_allowed")

    def test_internal_error_hides_details(self):
        client = app_with(FakeService(fail="boom"))
        with self.assertLogs("src.app.errors", level="ERROR"):  # logged server-side only
            r = client.get("/api/recommendations?link=x")
        self.assertEqual((r.status_code, r.get_json()), (500, {"error": {"code": "internal_error", "message": "An unexpected internal error occurred."}}))
        self.assertNotIn("Traceback", r.get_data(as_text=True))
        self.assertNotIn(".env", r.get_data(as_text=True))


class TestSerialization(unittest.TestCase):
    def test_json_safe(self):
        self.assertIsNone(json_safe(float("nan")))
        self.assertIsNone(json_safe(np.float64("inf")))
        self.assertEqual(json_safe(np.int64(3)), 3)
        self.assertIsInstance(json_safe(np.float32(0.5)), float)
        self.assertIs(json_safe(np.bool_(True)), True)
        with self.assertRaises(TypeError):
            json_safe(object())

    def test_split_pipe_is_sorted_and_deduplicated(self):
        self.assertEqual(split_pipe("b|a|b"), ["a", "b"])
        self.assertEqual(split_pipe(float("nan")), [])

    def test_worker_serializer_ignores_unlisted_columns(self):
        row = {"worker_id": "W1", "primary_category": "Cat", "secondary_categories": float("nan"), "declared_skills": "x|y",
               "experience_level": "entry", "experience_years": np.float64(1.5), "price_tier": float("nan"), "rate": 10,
               "country": "Kenya", "join_day": 0.0, "is_cold_start": True, "quality": 0.9, "true_skills": "z"}
        out = serialize_worker(row, 0, 0)
        self.assertNotIn("is_cold_start", out)
        self.assertNotIn("quality", out)
        self.assertNotIn("true_skills", out)
        self.assertIsNone(out["price_tier"])
        self.assertIsNone(out["training_mean_rating"])
        json.dumps(out, allow_nan=False)

    def test_strict_json_provider_refuses_nan(self):
        app = create_app("testing")
        with app.app_context(), self.assertRaises(ValueError):
            app.json.dumps({"x": math.nan})


def _code_only(module) -> str:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body[0].value.value = ""
    return ast.unparse(tree)


class TestBoundaries(unittest.TestCase):
    def test_service_cypher_is_fixed_and_read_only(self):
        for query in list(services.WORKER_GRAPH_QUERIES.values()) + list(services.COUNT_QUERIES.values()):
            upper = query.upper()
            for keyword in ("CREATE", "MERGE", "SET ", "DELETE", "REMOVE", "DETACH", "LOAD CSV", "CALL "):
                self.assertNotIn(keyword, upper)
        code = _code_only(services)
        self.assertIn("run_read", code)
        for forbidden in ("run_write", "run_batched", ".run("):
            self.assertNotIn(forbidden, code)

    def test_no_client_supplied_cypher(self):
        code = _code_only(routes)
        self.assertNotIn("cypher", code.lower())
        self.assertNotIn("run_read", code)

    def test_model_layers_do_not_depend_on_flask(self):
        for package in ("recommenders", "kg", "evaluation", "data"):
            for path in (REPO_ROOT / "src" / package).glob("*.py"):
                self.assertNotIn("flask", path.read_text(encoding="utf-8").lower(), str(path))

    def test_routes_contain_no_scoring_logic(self):
        code = _code_only(routes)
        for marker in ("minmax", "weights", "score_fn", "enumerate_paths", "jaccard"):
            self.assertNotIn(marker, code)


if __name__ == "__main__":
    unittest.main()
