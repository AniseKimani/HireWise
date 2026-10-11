"""TrainingGraphView construction, determinism, and source parity
(Day 7). Live Neo4j parity is checked read-only in
tests/test_day7_integration.py."""
import tempfile
import unittest
from pathlib import Path

from src.kg.graph_view import (
    EXTRACTION_QUERIES, TABLES, TRAIN_SPLIT, build_view, compare_views, load_view_from_kg_csvs, load_view_from_neo4j,
)
from tests.kg_fixtures import HIRED_ALL, toy_view, write_toy_kg_csvs


class FakeReadClient:
    """Records every query/parameter pair and answers from toy rows, so
    the Neo4j extraction path can be tested without a database."""

    def __init__(self):
        self.calls = []

    def run_read(self, query, parameters=None):
        self.calls.append((query, parameters))
        name = next(n for n, q in EXTRACTION_QUERIES.items() if q == query)
        view = toy_view()
        if name == "workers":
            return [{"worker_id": w} for w in view.workers]
        if name in ("skills", "categories"):
            return [{"name": v} for v in getattr(view, name)]
        if name == "has_skill":
            return [{"worker_id": w, "skill": s} for w, s in view.has_skill]
        if name == "specializes_in":
            return [{"worker_id": w, "category": c, "role": r} for w, c, r in view.specializes_in]
        if name == "related_to":
            return [{"skill_a": a, "skill_b": b, "npmi": n, "support": s} for a, b, n, s in view.related_to]
        if name == "train_hires":
            return [{"client_id": c, "worker_id": w, "link": l, "simulated_day": d}
                    for c, w, l, d, s in HIRED_ALL if s == parameters["split"]]
        if name == "train_reviews":
            return [{"client_id": c, "worker_id": w, "link": l, "rating": r, "simulated_day": d}
                    for c, w, l, r, d in view.train_reviews]
        if name == "interested_in":
            return [{"client_id": c, "category": cat, "training_request_count": n} for c, cat, n in view.interested_in]
        raise AssertionError(name)


class TestCsvView(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.kg_dir = write_toy_kg_csvs(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_training_hires_are_kept(self):
        view = load_view_from_kg_csvs(self.kg_dir)
        self.assertEqual({l for _, _, l, _ in view.train_hires}, {"t1", "t2", "t3"})

    def test_reviews_are_restricted_by_join_to_training_hires(self):
        view = load_view_from_kg_csvs(self.kg_dir)
        self.assertEqual({l for _, _, l, _, _ in view.train_reviews}, {"t1", "t3"})
        # the hire day is carried onto the review from the joined HIRED edge
        self.assertEqual({(l, d) for _, _, l, _, d in view.train_reviews}, {("t1", 1.0), ("t3", 3.0)})

    def test_csv_view_equals_hand_built_view(self):
        view = load_view_from_kg_csvs(self.kg_dir)
        self.assertEqual(view, toy_view())
        self.assertTrue(all(compare_views(view, toy_view()).values()))
        self.assertEqual(view.fingerprint(), toy_view().fingerprint())

    def test_view_does_not_carry_worker_cohort_or_join_day(self):
        view = load_view_from_kg_csvs(self.kg_dir)
        self.assertEqual(set(view.workers), {"W1", "W2", "W3", "W4"})
        self.assertNotIn("is_cold_start", repr(view))
        self.assertNotIn("join_day", repr(view))


class TestDeterminism(unittest.TestCase):
    def test_fingerprint_is_independent_of_row_order(self):
        a = toy_view()
        b = build_view(
            workers=reversed(a.workers), skills=reversed(a.skills), categories=a.categories,
            has_skill=reversed(a.has_skill), specializes_in=reversed(a.specializes_in),
            related_to=reversed(a.related_to), train_hires=reversed(a.train_hires),
            train_reviews=reversed(a.train_reviews), interested_in=a.interested_in, source="other",
        )
        self.assertEqual(a, b)
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_fingerprint_changes_when_any_table_changes(self):
        a = toy_view()
        b = build_view(
            workers=a.workers, skills=a.skills, categories=a.categories, has_skill=a.has_skill[:-1],
            specializes_in=a.specializes_in, related_to=a.related_to, train_hires=a.train_hires,
            train_reviews=a.train_reviews, interested_in=a.interested_in, source="x",
        )
        self.assertNotEqual(a.fingerprint(), b.fingerprint())
        self.assertFalse(compare_views(a, b)["has_skill"])

    def test_table_fingerprints_cover_every_table(self):
        self.assertEqual(set(toy_view().table_fingerprints()), set(TABLES))

    def test_stats(self):
        stats = toy_view().stats()
        self.assertEqual(stats["n_train_hires"], 3)
        self.assertEqual(stats["n_train_reviews"], 2)
        self.assertEqual(stats["mean_train_rating"], 4.0)
        self.assertEqual(stats["max_train_hire_simulated_day"], 3.0)


class TestNeo4jExtractionWithFakeClient(unittest.TestCase):
    def test_extraction_uses_read_transactions_and_train_split_parameter(self):
        client = FakeReadClient()
        view = load_view_from_neo4j(client)
        self.assertEqual(view, toy_view())
        self.assertEqual(view.source, "neo4j")
        for query, params in client.calls:
            if "$split" in query:
                self.assertEqual(params, {"split": TRAIN_SPLIT})
            else:
                self.assertIsNone(params)
        self.assertEqual(TRAIN_SPLIT, "train")


if __name__ == "__main__":
    unittest.main()
