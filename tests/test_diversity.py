"""Diversity / catalogue coverage and cold-start-worker diagnostics (Day 8)."""
import unittest
from collections import Counter

from src.evaluation.cold_start import cold_worker_diagnostics
from src.evaluation.diversity import (
    catalogue_coverage, diversity_summary, gini, intra_list_diversity, jaccard, top_share,
)
from src.evaluation.protocol import EvalRequest

RANKINGS = {"r1": ["A", "B"], "r2": ["A", "C"]}
SKILLS = {"A": frozenset({"x", "y"}), "B": frozenset({"x", "y"}), "C": frozenset({"z"})}


class TestDiversity(unittest.TestCase):
    def test_catalogue_coverage(self):
        self.assertEqual(catalogue_coverage(RANKINGS, {"A", "B", "C", "D"}), 0.75)
        self.assertEqual(catalogue_coverage({}, set()), 0.0)

    def test_gini(self):
        self.assertEqual(gini([1, 1, 1, 1]), 0.0)
        self.assertAlmostEqual(gini([0, 0, 0, 4]), 0.75)
        self.assertEqual(gini([0, 0]), 0.0)

    def test_top_share(self):
        n_top, share = top_share(Counter({"A": 2, "B": 1, "C": 1}), catalogue_size=4, fraction=0.25)
        self.assertEqual((n_top, share), (1, 0.5))

    def test_intra_list_diversity(self):
        self.assertEqual(jaccard(SKILLS["A"], SKILLS["B"]), 1.0)
        # r1: identical skills -> 0 diversity; r2: disjoint -> 1
        self.assertAlmostEqual(intra_list_diversity(RANKINGS, SKILLS), 0.5)
        self.assertEqual(intra_list_diversity({"r": ["A"]}, SKILLS), 0.0)

    def test_summary(self):
        s = diversity_summary(RANKINGS, {"A", "B", "C", "D"}, SKILLS, 0.25)
        self.assertEqual(s["unique_workers_recommended"], 3)
        self.assertEqual(s["n_slots"], 4)
        self.assertEqual(s["max_appearances_single_worker"], 2)


class TestColdWorkerDiagnostics(unittest.TestCase):
    def test_definitions(self):
        reqs = [
            EvalRequest("r1", "test", "c", ["A", "B", "C"], {"A": 2, "C": 1}, 3, False, None),
            EvalRequest("r2", "test", "c", ["A", "B", "C"], {"B": 1}, 3, False, None),
            EvalRequest("r3", "test", "c", ["A", "B", "C"], {"C": 1}, 3, True, "pool_too_small"),
        ]
        out = cold_worker_diagnostics({"r1": ["C", "B"], "r2": ["A", "B"]}, reqs, cold_start_ids={"C"})
        self.assertEqual(out["n_cold_start_requests"], 1)
        self.assertEqual(out["cold_recall_at_k"], 1.0)
        self.assertEqual(out["cold_share_of_top_k_slots"], 0.25)
        self.assertAlmostEqual(out["cold_share_of_relevant_items"], 1 / 3)
        self.assertAlmostEqual(out["cold_exposure_ratio"], 0.75)


if __name__ == "__main__":
    unittest.main()
