"""Day 7 leakage and temporal-safety guards for the KG recommender.

Static checks scan executable code only (docstrings/comments stripped),
so honest prose naming what a module does NOT use cannot cause false
alarms. Runtime checks inject validation/test edges and confirm the
graph view and scores do not move.
"""
import ast
import inspect
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.kg import explanations, graph_view
from src.kg.graph_view import EXTRACTION_QUERIES, TRAIN_SPLIT, load_view_from_kg_csvs
from src.recommenders import knowledge_graph
from src.recommenders.knowledge_graph import KGIndex, KnowledgeGraphRecommender, TemporalLeakageError
from tests.kg_fixtures import EQUAL_WEIGHTS, HIRED_ALL, REVIEWED_ALL, ROLE_WEIGHTS, toy_requests, write_toy_kg_csvs

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY7_MODULES = (graph_view, knowledge_graph, explanations)
PRIVATE_MARKERS = (
    "quality", "true_skills", "proficiency", "utility", "noise", "taste_", "oracle", "standardized",
    "synthetic_private", "clients_private", "workers_private", "truth.csv",
)
COHORT_MARKERS = ("is_cold_start", "cold_start", "join_day")
WRITE_KEYWORDS = ("CREATE", "MERGE", "SET ", "DELETE", "REMOVE", "DETACH", "LOAD CSV", "CALL ")


def _code_only_source(module) -> str:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body[0].value.value = ""
    return ast.unparse(tree)


class TestStaticCypherSafety(unittest.TestCase):
    def test_every_hired_query_filters_on_the_train_split_parameter(self):
        for name, query in EXTRACTION_QUERIES.items():
            if ":HIRED" in query:
                self.assertIn("h.split = $split", query, f"{name} reads HIRED without a split filter")

    def test_reviewed_is_only_read_through_a_join_to_training_hired(self):
        reviewed_queries = {n: q for n, q in EXTRACTION_QUERIES.items() if ":REVIEWED" in q}
        self.assertEqual(set(reviewed_queries), {"train_reviews"})
        query = reviewed_queries["train_reviews"]
        self.assertIn("[h:HIRED {link: rv.link}]", query)
        self.assertIn("h.split = $split", query)

    def test_requested_service_is_never_extracted(self):
        for query in EXTRACTION_QUERIES.values():
            self.assertNotIn("REQUESTED_SERVICE", query)
        self.assertNotIn("rel_requested_service", _code_only_source(graph_view))

    def test_queries_are_read_only_and_explicitly_ordered(self):
        for name, query in EXTRACTION_QUERIES.items():
            upper = query.upper()
            for keyword in WRITE_KEYWORDS:
                self.assertNotIn(keyword, upper, f"{name} contains write keyword {keyword!r}")
            self.assertIn("ORDER BY", upper, f"{name} leaves row order to the server")

    def test_queries_never_look_up_an_evaluated_request(self):
        for query in EXTRACTION_QUERIES.values():
            self.assertNotIn("$link", query)

    def test_extraction_uses_read_transactions_only(self):
        code = _code_only_source(graph_view)
        self.assertIn("run_read", code)
        for forbidden in ("run_write", "run_batched", ".run("):
            self.assertNotIn(forbidden, code)

    def test_train_split_is_a_fixed_constant(self):
        self.assertEqual(TRAIN_SPLIT, "train")
        params = inspect.signature(graph_view.load_view_from_neo4j).parameters
        self.assertEqual(list(params), ["client"])
        self.assertEqual(list(inspect.signature(graph_view.load_view_from_kg_csvs).parameters), ["kg_dir"])

    def test_worker_nodes_contribute_only_their_id(self):
        self.assertEqual(EXTRACTION_QUERIES["workers"].split("RETURN")[1].strip(), "w.worker_id AS worker_id ORDER BY worker_id")


class TestStaticModelInputs(unittest.TestCase):
    def test_no_generator_private_markers(self):
        for module in DAY7_MODULES:
            code = _code_only_source(module).lower()
            for marker in PRIVATE_MARKERS:
                self.assertNotIn(marker, code, f"{module.__name__} code mentions {marker!r}")

    def test_cold_start_cohort_is_not_a_model_feature(self):
        for module in DAY7_MODULES:
            code = _code_only_source(module).lower()
            for marker in COHORT_MARKERS:
                self.assertNotIn(marker, code, f"{module.__name__} code mentions {marker!r}")

    def test_no_price_or_location_feature(self):
        code = _code_only_source(knowledge_graph).lower()
        for marker in ("price_tier", "rate", "country", "location", "interested_in"):
            self.assertIsNone(re.search(rf"\b{re.escape(marker)}\b", code), f"knowledge_graph.py uses {marker!r}")

    def test_does_not_import_other_recommenders(self):
        code = _code_only_source(knowledge_graph)
        for module_name in ("collaborative", "content_based", "popularity", "random_baseline"):
            self.assertNotIn(module_name, code)


class TestRuntimeTemporalSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _scores(self, kg_dir: Path) -> np.ndarray:
        rec = KnowledgeGraphRecommender(KGIndex(load_view_from_kg_csvs(kg_dir)), toy_requests(), 0.4,
                                        EQUAL_WEIGHTS, ROLE_WEIGHTS, 5)
        return rec.scored("r1", ["W1", "W2", "W3", "W4"])[2]

    def test_adding_validation_and_test_edges_changes_neither_view_nor_scores(self):
        train_only_hired = [h for h in HIRED_ALL if h[4] == "train"]
        train_only_reviewed = [r for r in REVIEWED_ALL if r[2] in {"t1", "t3"}]
        clean = write_toy_kg_csvs(self.root / "clean", hired=train_only_hired, reviewed=train_only_reviewed)
        polluted = write_toy_kg_csvs(self.root / "polluted")  # includes validation/test hires + reviews + orphan review
        self.assertEqual(load_view_from_kg_csvs(clean).fingerprint(), load_view_from_kg_csvs(polluted).fingerprint())
        np.testing.assert_array_equal(self._scores(clean), self._scores(polluted))

    def test_evaluated_requests_own_future_hire_is_invisible(self):
        # r1 (validation) gets its own HIRED + REVIEWED outcome written into the graph.
        hired = HIRED_ALL + [("C1", "W2", "r1", 10.0, "validation")]
        reviewed = REVIEWED_ALL + [("C1", "W2", "r1", 5, True)]
        with_outcome = write_toy_kg_csvs(self.root / "with_outcome", hired=hired, reviewed=reviewed)
        without = write_toy_kg_csvs(self.root / "without")
        np.testing.assert_array_equal(self._scores(with_outcome), self._scores(without))
        view = load_view_from_kg_csvs(with_outcome)
        self.assertNotIn("r1", {l for _, _, l, _ in view.train_hires})

    def test_review_of_a_test_hire_is_excluded_even_if_mislabelled_hire_pair_exists(self):
        # Same client/worker pair has a TRAIN hire (t1) and a TEST hire; only the
        # review joined by link to the training hire may survive.
        hired = HIRED_ALL + [("C1", "W3", "x2", 40.0, "test")]
        reviewed = REVIEWED_ALL + [("C1", "W3", "x2", 1, True)]
        view = load_view_from_kg_csvs(write_toy_kg_csvs(self.root / "pair", hired=hired, reviewed=reviewed))
        self.assertEqual({l for _, _, l, _, _ in view.train_reviews}, {"t1", "t3"})

    def test_training_requests_cannot_be_scored(self):
        rec = KnowledgeGraphRecommender(KGIndex(load_view_from_kg_csvs(write_toy_kg_csvs(self.root / "g"))),
                                        toy_requests(), 0.0, EQUAL_WEIGHTS, ROLE_WEIGHTS, 5)
        for link in ("r_early", "t1"):
            with self.assertRaises(TemporalLeakageError):
                rec.rank(link, ["W1"], 10)


SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"
KG_DIR = REPO_ROOT / "data" / "processed" / "kg"


@unittest.skipUnless(KG_DIR.exists() and SYNTHETIC_DIR.exists(), "Day 3/Day 6 artifacts not generated locally")
class TestRealSnapshotTemporalSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.view = load_view_from_kg_csvs(KG_DIR)
        cls.requests = pd.read_csv(SYNTHETIC_DIR / "requests.csv")
        cls.cold = set(pd.read_csv(SYNTHETIC_DIR / "cold_start_cohort.csv")["worker_id"])

    def test_snapshot_edges_are_exactly_the_training_ones(self):
        train_links = set(self.requests.loc[self.requests["split"] == "train", "link"])
        self.assertEqual(len(self.view.train_hires), 7117)
        self.assertEqual(len(self.view.train_reviews), 6454)
        self.assertTrue({l for _, _, l, _ in self.view.train_hires} <= train_links)
        self.assertTrue({l for _, _, l, _, _ in self.view.train_reviews} <= train_links)

    def test_no_cold_start_worker_has_a_historical_edge(self):
        self.assertFalse(self.cold & {w for _, w, _, _ in self.view.train_hires})
        self.assertFalse(self.cold & {w for _, w, _, _, _ in self.view.train_reviews})

    def test_history_ends_before_every_validation_and_test_request(self):
        index = KGIndex(self.view)
        later = self.requests.loc[self.requests["split"].isin(["validation", "test"]), "simulated_day"]
        self.assertLess(index.max_history_day, later.min())
        self.assertFalse(index.history_links & set(self.requests.loc[self.requests["split"] != "train", "link"]))


if __name__ == "__main__":
    unittest.main()
