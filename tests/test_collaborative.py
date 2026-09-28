"""Unit tests for src/recommenders/collaborative.py, including a
hand-calculated user-based/item-based scoring scenario (values verified
independently before being encoded here)."""
import math
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.recommenders.collaborative import (
    CollaborativeFiltering,
    build_neighbor_lists,
    build_training_matrix,
    cosine_similarity_matrix,
)
from src.recommenders.popularity import PopularityBaseline

WEIGHTS = {"HIRED": 2.0, "SHORTLISTED": 1.0}


def make_scenario():
    """C1 interacted with W1 (HIRED) and W2 (SHORTLISTED); C2 with W1
    (HIRED) and W3 (SHORTLISTED); C3 with W4 (HIRED) -- all in training.
    Matrix rows=[C1,C2,C3], cols=[W1,W2,W3,W4]:
        C1: [2, 1, 0, 0]
        C2: [2, 0, 1, 0]
        C3: [0, 0, 0, 2]
    """
    interactions = pd.DataFrame([
        {"link": "r1", "worker_id": "W1", "client_id": "C1", "event": "HIRED"},
        {"link": "r2", "worker_id": "W2", "client_id": "C1", "event": "SHORTLISTED"},
        {"link": "r3", "worker_id": "W1", "client_id": "C2", "event": "HIRED"},
        {"link": "r4", "worker_id": "W3", "client_id": "C2", "event": "SHORTLISTED"},
        {"link": "r5", "worker_id": "W4", "client_id": "C3", "event": "HIRED"},
    ])
    requests = pd.DataFrame([
        {"link": "r1", "client_id": "C1", "split": "train"},
        {"link": "r2", "client_id": "C1", "split": "train"},
        {"link": "r3", "client_id": "C2", "split": "train"},
        {"link": "r4", "client_id": "C2", "split": "train"},
        {"link": "r5", "client_id": "C3", "split": "train"},
    ])
    return interactions, requests


class TestBuildTrainingMatrix(unittest.TestCase):
    def test_matrix_shape_and_values(self):
        interactions, requests = make_scenario()
        tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        self.assertEqual(tm.clients, ["C1", "C2", "C3"])
        self.assertEqual(tm.workers, ["W1", "W2", "W3", "W4"])
        expected = np.array([[2, 1, 0, 0], [2, 0, 1, 0], [0, 0, 0, 2]], dtype=float)
        np.testing.assert_array_equal(tm.matrix.toarray(), expected)

    def test_training_only_validation_and_test_excluded(self):
        interactions, requests = make_scenario()
        requests.loc[requests["link"] == "r1", "split"] = "validation"
        requests.loc[requests["link"] == "r3", "split"] = "test"
        tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        # r1 (C1-W1) and r3 (C2-W1) removed; C1 now only has W2, C2 only W3.
        self.assertNotIn("W1", tm.workers)

    def test_repeated_pair_sum_aggregation(self):
        interactions = pd.DataFrame([
            {"link": "r1", "worker_id": "W1", "client_id": "C1", "event": "SHORTLISTED"},
            {"link": "r2", "worker_id": "W1", "client_id": "C1", "event": "HIRED"},
        ])
        requests = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "split": "train"},
            {"link": "r2", "client_id": "C1", "split": "train"},
        ])
        tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        self.assertEqual(tm.matrix.toarray().tolist(), [[3.0]])  # 1 + 2

    def test_repeated_pair_max_aggregation(self):
        interactions = pd.DataFrame([
            {"link": "r1", "worker_id": "W1", "client_id": "C1", "event": "SHORTLISTED"},
            {"link": "r2", "worker_id": "W1", "client_id": "C1", "event": "HIRED"},
        ])
        requests = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "split": "train"},
            {"link": "r2", "client_id": "C1", "split": "train"},
        ])
        tm = build_training_matrix(interactions, requests, WEIGHTS, "max")
        self.assertEqual(tm.matrix.toarray().tolist(), [[2.0]])  # max(1, 2)

    def test_sparse_representation(self):
        interactions, requests = make_scenario()
        tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        self.assertIsInstance(tm.matrix, csr_matrix)

    def test_invalid_aggregation_raises(self):
        interactions, requests = make_scenario()
        with self.assertRaises(ValueError):
            build_training_matrix(interactions, requests, WEIGHTS, "average")


class TestCosineSimilarity(unittest.TestCase):
    def test_hand_calculated_similarity(self):
        m = csr_matrix(np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=float))
        sim = cosine_similarity_matrix(m)
        expected = np.array([[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])
        np.testing.assert_allclose(sim, expected)

    def test_zero_vector_row_gives_zero_similarity(self):
        m = csr_matrix(np.array([[1, 1], [0, 0]], dtype=float))
        sim = cosine_similarity_matrix(m)
        self.assertEqual(sim[1, 0], 0.0)
        self.assertEqual(sim[1, 1], 0.0)

    def test_self_similarity_is_one_for_nonzero_row(self):
        m = csr_matrix(np.array([[3, 4]], dtype=float))
        sim = cosine_similarity_matrix(m)
        self.assertAlmostEqual(sim[0, 0], 1.0)


class TestNeighborLists(unittest.TestCase):
    def test_self_excluded(self):
        sim = np.array([[1, 0.5], [0.5, 1]])
        neighbors = build_neighbor_lists(sim, min_similarity=0.0)
        self.assertNotIn(0, neighbors[0].tolist())
        self.assertNotIn(1, neighbors[1].tolist())

    def test_only_positive_similarity_included(self):
        sim = np.array([[1, 0, -0.5], [0, 1, 0.3], [-0.5, 0.3, 1]])
        neighbors = build_neighbor_lists(sim, min_similarity=0.0)
        self.assertEqual(neighbors[0].tolist(), [])  # only 0 and -0.5 available, neither > 0
        self.assertEqual(neighbors[1].tolist(), [2])

    def test_deterministic_descending_order(self):
        sim = np.array([[1, 0.9, 0.3, 0.6], [0.9, 1, 0, 0], [0.3, 0, 1, 0], [0.6, 0, 0, 1]])
        neighbors = build_neighbor_lists(sim, min_similarity=0.0)
        self.assertEqual(neighbors[0].tolist(), [1, 3, 2])


class TestCollaborativeFilteringScoring(unittest.TestCase):
    def setUp(self):
        interactions, requests = make_scenario()
        self.tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        self.fallback = PopularityBaseline(interactions, requests)
        self.link_to_client = {"eval_req": "C1"}

    def test_user_based_hand_calculated_scores(self):
        # sim(C1,C2) = (2*2+1*0+0*1)/(sqrt(5)*sqrt(5)) = 4/5 = 0.8; sim(C1,C3)=0 (excluded)
        # score(W1) = 0.8 * C2[W1]=2 -> 1.6; score(W2) = 0.8*0 -> 0
        # score(W3) = 0.8 * C2[W3]=1 -> 0.8; W4 absent from C2's contribution -> 0
        cf = CollaborativeFiltering(self.tm, "user", k_neighbors=10, fallback=self.fallback, link_to_client=self.link_to_client)
        scores = cf._cf_scores("eval_req", ["W1", "W2", "W3", "W4"])
        self.assertAlmostEqual(scores["W1"], 1.6)
        self.assertAlmostEqual(scores["W2"], 0.0)
        self.assertAlmostEqual(scores["W3"], 0.8)
        self.assertAlmostEqual(scores["W4"], 0.0)

    def test_user_based_ranks_expected_neighbor_worker_first(self):
        cf = CollaborativeFiltering(self.tm, "user", k_neighbors=10, fallback=self.fallback, link_to_client=self.link_to_client)
        ranked = cf.rank("eval_req", ["W4", "W3", "W2", "W1"], k=10)
        self.assertEqual(ranked[0], "W1")  # highest CF score

    def test_item_based_hand_calculated_scores(self):
        # sim(W1,W2) = sim(W1,W3) = 1/sqrt(2) ~ 0.7071; sim(W2,W3)=sim(*,W4)=0
        # C1 history: W1(2), W2(1). score(W1) via neighbor W2 in history: 0.7071*1
        # score(W2) via neighbor W1 in history: 0.7071*2; score(W3) via neighbor W1: 0.7071*2; score(W4)=0
        cf = CollaborativeFiltering(self.tm, "item", k_neighbors=10, fallback=self.fallback, link_to_client=self.link_to_client)
        scores = cf._cf_scores("eval_req", ["W1", "W2", "W3", "W4"])
        self.assertAlmostEqual(scores["W1"], 1 / math.sqrt(2), places=6)
        self.assertAlmostEqual(scores["W2"], 2 / math.sqrt(2), places=6)
        self.assertAlmostEqual(scores["W3"], 2 / math.sqrt(2), places=6)
        self.assertAlmostEqual(scores["W4"], 0.0)

    def test_client_with_no_training_history_gets_empty_scores(self):
        cf = CollaborativeFiltering(self.tm, "user", k_neighbors=10, fallback=self.fallback, link_to_client={"req": "C_UNKNOWN"})
        self.assertEqual(cf._cf_scores("req", ["W1", "W2"]), {})

    def test_candidate_worker_with_no_training_history_scores_zero_or_absent(self):
        cf = CollaborativeFiltering(self.tm, "user", k_neighbors=10, fallback=self.fallback, link_to_client=self.link_to_client)
        scores = cf._cf_scores("eval_req", ["W_NEVER_SEEN"])
        self.assertEqual(scores.get("W_NEVER_SEEN", 0.0), 0.0)

    def test_no_content_feature_dependency(self):
        # The CF scorer must work correctly even though we never pass it
        # anything resembling declared_skills/category/price/country --
        # it only ever sees the interaction matrix.
        cf = CollaborativeFiltering(self.tm, "item", k_neighbors=10, fallback=self.fallback, link_to_client=self.link_to_client)
        import inspect
        source = inspect.getsource(type(cf)).lower()
        for marker in ("declared_skills", "category", "price_tier", "country", "skills_vocab"):
            self.assertNotIn(marker, source)


class TestFallback(unittest.TestCase):
    def setUp(self):
        interactions, requests = make_scenario()
        self.tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        self.fallback = PopularityBaseline(interactions, requests)

    def test_no_history_client_uses_pure_fallback_ordering(self):
        cf = CollaborativeFiltering(self.tm, "user", k_neighbors=10, fallback=self.fallback, link_to_client={"req": "C_UNKNOWN"})
        ranked, diag = cf.rank_with_diagnostics("req", ["W4", "W1", "W3", "W2"], k=10)
        # No CF signal at all -> pure fallback (popularity) order: W1 has 2
        # training hires, W4 has 1, W2/W3 tie at 0 -> worker_id order.
        self.assertEqual(ranked, ["W1", "W4", "W2", "W3"])
        self.assertTrue(diag["client_has_history"] is False)
        self.assertTrue(diag["request_fully_fallback"])
        self.assertEqual(diag["n_recommendations_via_fallback"], 4)

    def test_zero_score_candidate_falls_back_but_ranks_below_genuine_cf_score(self):
        cf = CollaborativeFiltering(self.tm, "user", k_neighbors=10, fallback=self.fallback, link_to_client={"req": "C1"})
        ranked, diag = cf.rank_with_diagnostics("req", ["W1", "W2", "W3", "W4"], k=10)
        # W1 (cf=1.6) and W3 (cf=0.8) have genuine positive CF scores and
        # must rank above W2/W4 (cf=0, fallback-only).
        self.assertEqual(ranked[0], "W1")
        self.assertEqual(ranked[1], "W3")
        self.assertEqual(set(ranked[2:]), {"W2", "W4"})
        self.assertFalse(diag["request_fully_fallback"])
        self.assertEqual(diag["n_recommendations_via_fallback"], 2)

    def test_fallback_uses_training_only_popularity(self):
        # Popularity fallback must itself never see validation/test hires.
        interactions, requests = make_scenario()
        requests2 = requests.copy()
        requests2.loc[requests2["link"] == "r5", "split"] = "test"
        fallback = PopularityBaseline(interactions, requests2)
        self.assertNotIn("W4", fallback.hire_counts)  # W4's only hire (r5) is now in test


class TestDeterminismAndValidity(unittest.TestCase):
    def test_deterministic_ranking(self):
        interactions, requests = make_scenario()
        tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        fallback = PopularityBaseline(interactions, requests)
        link_to_client = {"req": "C1"}
        cf1 = CollaborativeFiltering(tm, "item", k_neighbors=10, fallback=fallback, link_to_client=link_to_client)
        cf2 = CollaborativeFiltering(tm, "item", k_neighbors=10, fallback=fallback, link_to_client=link_to_client)
        self.assertEqual(cf1.rank("req", ["W4", "W1", "W3", "W2"], 10), cf2.rank("req", ["W4", "W1", "W3", "W2"], 10))

    def test_no_duplicates_and_respects_k(self):
        interactions, requests = make_scenario()
        tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        fallback = PopularityBaseline(interactions, requests)
        cf = CollaborativeFiltering(tm, "user", k_neighbors=10, fallback=fallback, link_to_client={"req": "C1"})
        ranked = cf.rank("req", ["W1", "W2", "W3", "W4"], k=2)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(len(ranked), len(set(ranked)))

    def test_invalid_method_raises(self):
        interactions, requests = make_scenario()
        tm = build_training_matrix(interactions, requests, WEIGHTS, "sum")
        fallback = PopularityBaseline(interactions, requests)
        with self.assertRaises(ValueError):
            CollaborativeFiltering(tm, "graph", k_neighbors=10, fallback=fallback, link_to_client={})


if __name__ == "__main__":
    unittest.main()
