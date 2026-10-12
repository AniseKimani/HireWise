"""Day 10 frontend tests: page rendering, static-asset safety, the
JavaScript <-> API field contract, the request-listing extension, and an
end-to-end browse -> recommend -> worker -> graph flow through the Flask
test client (no browser automation). Data-dependent tests skip when the
processed data / Day 8 results are absent."""
import re
import unittest
from pathlib import Path

from src.app import create_app
from src.app.serializers import RECOMMENDATION_FIELDS, REQUEST_FIELDS, WORKER_FIELDS
from src.app.services import Neo4jUnavailable

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC = REPO_ROOT / "src" / "app" / "static"
JS = {p.name: p.read_text(encoding="utf-8") for p in (STATIC / "js").glob("*.js")}
SYNTHETIC = REPO_ROOT / "data" / "processed" / "synthetic"
DAY8 = REPO_ROOT / "results" / "day8"
PRIVATE_MARKERS = ("is_cold_start", "cold_start", "quality", "proficiency", "true_skill", "taste", "utility",
                   "oracle", "relevance", "shortlist", "NEO4J_URI", "NEO4J_PASSWORD", "password", "bolt://")
LIST_KEYS = {"items", "page", "per_page", "total", "total_pages", "filters", "available_categories"}
ITEM_KEYS = set(REQUEST_FIELDS) | {"n_eligible_workers"}


class TestPage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app("testing").test_client()
        cls.response = cls.client.get("/")
        cls.html = cls.response.get_data(as_text=True)

    def test_root_renders_main_sections(self):
        self.assertEqual(self.response.status_code, 200)
        self.assertEqual(self.response.mimetype, "text/html")
        for element_id in ("health-status", "request-browser", "request-filters", "request-list", "page-prev", "page-next",
                           "selected-request", "recommend-form", "k-select", "recommend-button", "results",
                           "recommendation-list", "worker-dialog", "worker-profile", "graph-load", "worker-graph"):
            self.assertIn(f'id="{element_id}"', self.html, element_id)
        self.assertIn("<h1", self.html)
        self.assertIn('lang="en"', self.html)

    def test_assets_referenced_and_served(self):
        self.assertIn("/static/css/style.css", self.html)
        self.assertIn('type="module" src="/static/js/app.js"', self.html)
        for asset in ("css/style.css", "js/app.js", "js/api.js", "js/format.js", "js/dom.js"):
            self.assertEqual(self.client.get(f"/static/{asset}").status_code, 200, asset)

    def test_controls_are_labelled(self):
        for control in ("filter-q", "filter-category", "filter-skill", "k-select"):
            self.assertIn(f'for="{control}"', self.html)

    def test_default_k_is_ten(self):
        self.assertRegex(self.html, r'<option value="10" selected>')

    def test_no_inline_script_or_style_and_csp_present(self):
        self.assertNotRegex(self.html, r"<script(?![^>]*\bsrc=)[^>]*>")
        self.assertNotIn("style=", self.html)
        self.assertNotRegex(self.html, r"\son[a-z]+=")
        csp = self.response.headers["Content-Security-Policy"]
        self.assertIn("script-src 'self'", csp)
        self.assertNotIn("unsafe-inline", csp)
        self.assertEqual(self.response.headers["X-Content-Type-Options"], "nosniff")

    def test_no_data_or_private_fields_embedded_in_html(self):
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker.lower(), self.html.lower())
        self.assertNotIn("upwork.com/jobs", self.html)  # request data is fetched, never server-rendered


class TestJavaScriptSafety(unittest.TestCase):
    def test_no_unsafe_dom_sinks(self):
        for name, source in JS.items():
            for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
                self.assertNotIn(sink, source, f"{name} uses {sink}")

    def test_fetch_is_centralised_in_api_client(self):
        for name, source in JS.items():
            if name != "api.js":
                self.assertNotIn("fetch(", source, name)
        self.assertIn('from "./api.js"', JS["app.js"])

    def test_no_secrets_cypher_or_cohort_flag(self):
        for name, source in JS.items():
            lowered = source.lower()
            for marker in ("neo4j_uri", "neo4j_user", "neo4j_password", "password", "bolt://", "match (", "is_cold_start", "cold_start"):
                self.assertNotIn(marker, lowered, f"{name} mentions {marker!r}")

    def test_links_are_url_encoded_via_url_search_params(self):
        self.assertIn("url.searchParams.set", JS["api.js"])
        self.assertIn("encodeURIComponent(workerId)", JS["api.js"])

    def test_explanations_are_not_generated_in_javascript(self):
        # Explanation text is only ever copied from the API response.
        self.assertIn("text: rec.explanation", JS["app.js"])
        self.assertNotIn("Recommended because", "".join(JS.values()))

    def test_fields_read_by_app_exist_in_api_contract(self):
        source = JS["app.js"]
        contract = {
            "worker": set(WORKER_FIELDS),
            "rec": set(RECOMMENDATION_FIELDS),
            "item": set(REQUEST_FIELDS) | {"n_eligible_workers"},
            "result": LIST_KEYS | {"n_returned", "n_candidates", "model", "recommendations"},
            "graph": {"skills", "categories", "location", "worker_id", "source"},
            "evidence": {"path_type", "text"},
            "health": {"status", "recommender", "neo4j", "recommendations_require_neo4j"},
        }
        for variable, allowed in contract.items():
            used = set(re.findall(rf"(?<![\w.]){variable}\.([a-z0-9_]+)", source))  # skips e.g. ui.graph.append
            self.assertTrue(used, f"no {variable}.<field> use found")
            self.assertTrue(used <= allowed, f"app.js reads {variable}.{used - allowed} not in the API contract")

    def test_every_api_error_code_has_a_friendly_message(self):
        for code in ("network_error", "unknown_request", "request_not_recommendable", "recommender_unavailable",
                     "neo4j_unavailable", "unknown_worker", "invalid_k", "invalid_page", "invalid_per_page", "invalid_q"):
            self.assertIn(f"{code}:", JS["format.js"], code)


@unittest.skipUnless(SYNTHETIC.exists() and (DAY8 / "selected_hybrid_config.json").exists(), "processed data / Day 8 results absent")
class TestRequestListingAndEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.app.services import RecommendationService
        from src.evaluation.config import load_evaluation_config
        from src.evaluation.protocol import EligibilityIndex, build_eval_requests

        cls.service = RecommendationService.build()

        def unavailable_graph(worker_id):
            raise Neo4jUnavailable("down")

        cls.client = create_app("testing", service=cls.service,
                                neo4j_probe=lambda: {"available": False, "graph_matches_day6_counts": None},
                                worker_graph_reader=unavailable_graph).test_client()
        index = EligibilityIndex(cls.service.data.workers, cls.service.data.known_countries)
        cls.pool_size = {r.link: r.n_candidates for s in ("validation", "test")
                         for r in build_eval_requests(cls.service.data, index, s, load_evaluation_config())}

    def get(self, path, **params):
        return self.client.get(path, query_string=params)

    def test_default_listing(self):
        body = self.get("/api/requests").get_json()
        self.assertEqual(set(body), LIST_KEYS)
        self.assertEqual((body["page"], body["per_page"], len(body["items"])), (1, 20, 20))
        self.assertEqual(body["total"], 3600)  # all validation (1,200) + test (2,400) requests
        self.assertEqual(body["total_pages"], 180)
        self.assertEqual(len(body["available_categories"]), 88)

    def test_listing_is_deterministic_and_chronological(self):
        a = self.get("/api/requests", page="3", per_page="50").get_data()
        self.assertEqual(a, self.get("/api/requests", page="3", per_page="50").get_data())
        items = [i for p in range(1, 4) for i in self.get("/api/requests", page=str(p), per_page="50").get_json()["items"]]
        keys = [(i["simulated_day"], i["link"]) for i in items]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len({i["link"] for i in items}), 150)

    def test_listing_items_are_allow_listed_and_recommendable(self):
        for item in self.get("/api/requests", per_page="50").get_json()["items"]:
            self.assertEqual(set(item), ITEM_KEYS)
            self.assertIn(item["split"], ("validation", "test"))
            self.assertEqual(item["n_eligible_workers"], self.pool_size[item["link"]])
        text = self.get("/api/requests", per_page="50").get_data(as_text=True).lower()
        for marker in ("cold_start", "relevance", "shortlist", "hired", "quality"):
            self.assertNotIn(f'"{marker}', text)

    def test_pagination_validation(self):
        for params in ({"page": "0"}, {"page": "-1"}, {"page": "x"}, {"per_page": "0"}, {"per_page": "51"},
                       {"q": "a" * 101}, {"skill": "b" * 101}):
            self.assertEqual(self.get("/api/requests", **params).status_code, 400, params)
        self.assertEqual(self.get("/api/requests", page="181").get_json()["items"], [])

    def test_filters(self):
        q = self.get("/api/requests", q="LOGO", per_page="50").get_json()
        self.assertTrue(q["total"] > 0 and all("logo" in i["title"].lower() for i in q["items"]))
        category = q["available_categories"][0]
        c = self.get("/api/requests", category=category, per_page="50").get_json()
        self.assertTrue(c["total"] > 0 and all(i["category"] == category for i in c["items"]))
        s = self.get("/api/requests", skill="python", per_page="50").get_json()
        self.assertTrue(s["total"] > 0 and all("python" in [x.lower() for x in i["required_skills"]] for i in s["items"]))
        self.assertEqual(self.get("/api/requests", q="zzzz-no-such-title").get_json()["total"], 0)

    def test_single_lookup_still_works(self):
        link = self.get("/api/requests").get_json()["items"][0]["link"]
        self.assertEqual(self.get("/api/requests", link=link).get_json()["link"], link)

    def test_end_to_end_flow_with_neo4j_unavailable(self):
        health = self.client.get("/api/health").get_json()
        self.assertEqual((health["status"], health["recommender"]["ready"]), ("degraded", True))
        item = self.get("/api/requests", per_page="1").get_json()["items"][0]
        recs = self.get("/api/recommendations", link=item["link"]).get_json()
        self.assertEqual(recs["n_returned"], min(10, item["n_eligible_workers"]))
        first = recs["recommendations"][0]
        self.assertEqual(set(first["contributions"]), {"cbf", "cf", "kg"})
        self.assertAlmostEqual(sum(first["contributions"].values()), first["score"], places=12)
        worker = self.client.get(f"/api/workers/{first['worker']['worker_id']}").get_json()
        self.assertEqual(worker, first["worker"])
        graph = self.client.get(f"/api/workers/{first['worker']['worker_id']}/graph")
        self.assertEqual((graph.status_code, graph.get_json()["error"]["code"]), (503, "neo4j_unavailable"))
        # recommendations keep working while the graph is down
        self.assertEqual(self.get("/api/recommendations", link=item["link"]).status_code, 200)


if __name__ == "__main__":
    unittest.main()
