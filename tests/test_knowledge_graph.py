"""Day 7 KG recommender: DWPC path scores against hand-computed values on
the toy graph (tests/kg_fixtures.py), normalization, ranking, tie-breaks,
cold-start behaviour, and temporal guards."""
import unittest

import numpy as np

from src.recommenders.knowledge_graph import (
    COMPONENTS, KGComponentCache, KGIndex, KnowledgeGraphRecommender, TemporalLeakageError,
    enumerate_paths, minmax_columns, path_weight,
)
from tests.kg_fixtures import EQUAL_WEIGHTS, ROLE_WEIGHTS, toy_requests, toy_view

CANDIDATES = ["W1", "W2", "W3", "W4"]


def make_recommender(damping=0.0, weights=None, cache=None, requests=None):
    return KnowledgeGraphRecommender(
        KGIndex(toy_view()), toy_requests() if requests is None else requests, damping,
        weights or EQUAL_WEIGHTS, ROLE_WEIGHTS, reputation_prior_strength=5, max_paths_per_component=3, cache=cache,
    )


class TestRawComponentsPathCount(unittest.TestCase):
    """damping w = 0: DWPC is a plain (base-weighted) path count."""

    def setUp(self):
        self.raw = make_recommender().raw_components("r1", CANDIDATES)

    def test_skill_component(self):
        np.testing.assert_allclose(self.raw[:, COMPONENTS.index("skill")], [1, 0, 1, 0])

    def test_related_skill_component_uses_npmi_and_excludes_required_intermediates(self):
        # W1 and W4 hold B, related to required A (npmi 0.5). W2 holds C,
        # related only to B, which is not required -> no path.
        np.testing.assert_allclose(self.raw[:, COMPONENTS.index("related_skill")], [0.5, 0, 0, 0.5])

    def test_category_component_uses_role_weights(self):
        np.testing.assert_allclose(self.raw[:, COMPONENTS.index("category")], [1.0, 0.5, 1.0, 1.0])

    def test_history_component(self):
        # W1: hired_similar (C1->W3 shares A) + co_hire (C1->W3<-C2->W1) = 2
        # W3: repeat_hire only (W3 is the hired worker itself, so no W'=W3 paths) = 1
        np.testing.assert_allclose(self.raw[:, COMPONENTS.index("history")], [2, 0, 1, 0])

    def test_reputation_component_is_smoothed_training_mean(self):
        # mu = (5 + 3) / 2 = 4; m = 5
        np.testing.assert_allclose(self.raw[:, COMPONENTS.index("reputation")], [23 / 6, 4.0, 25 / 6, 4.0])


class TestDWPCDamping(unittest.TestCase):
    def test_path_degree_products(self):
        rec = make_recommender()
        paths = {(t, n): pdp for t, n, _, pdp in enumerate_paths(rec.index, rec.context("r1"), "W1", ROLE_WEIGHTS)}
        self.assertEqual(paths[("skill", ("A", "W1"))], 2 * 2)                     # deg(A)=2, |S_W1|=2
        self.assertEqual(paths[("related_skill", ("A", "B", "W1"))], 1 * 2 * 2 * 2)  # RT(A)=1, RT(B)=2, HS(B)=2, |S_W1|=2
        self.assertEqual(paths[("category", ("Cat", "W1"))], 4 * 1)                # 4 workers in Cat, W1 has 1 category
        self.assertEqual(paths[("hired_similar_skill", ("C1", "W3", "A", "W1"))], 1 * 2 * 1 * 2 * 2 * 2)
        self.assertEqual(paths[("co_hire", ("C1", "W3", "C2", "W1"))], 1 * 2 * 2 * 2 * 2 * 1)

    def test_damped_skill_score(self):
        rec = make_recommender(damping=0.5, cache=KGComponentCache((0.5,)))
        raw = rec.raw_components("r1", CANDIDATES)
        self.assertAlmostEqual(raw[0, COMPONENTS.index("skill")], 4 ** -0.5)
        self.assertAlmostEqual(raw[0, COMPONENTS.index("related_skill")], 0.5 * 8 ** -0.5)
        self.assertAlmostEqual(raw[0, COMPONENTS.index("history")], 2 * 16 ** -0.5)

    def test_path_weight(self):
        self.assertEqual(path_weight(0.7, 100, 0.0), 0.7)
        self.assertAlmostEqual(path_weight(1.0, 16, 0.5), 0.25)


class TestNormalizationAndRanking(unittest.TestCase):
    def test_minmax_constant_column_is_zero(self):
        out = minmax_columns(np.array([[1.0, 2.0], [1.0, 4.0]]))
        np.testing.assert_allclose(out, [[0.0, 0.0], [0.0, 1.0]])

    def test_scores_and_ranking_match_hand_computation(self):
        rec = make_recommender()
        _, contributions, score = rec.scored("r1", CANDIDATES)
        np.testing.assert_allclose(score, [0.8, 0.1, 0.7, 0.5])
        np.testing.assert_allclose(contributions.sum(axis=1), score)
        self.assertEqual(rec.rank("r1", CANDIDATES, 10), ["W1", "W3", "W4", "W2"])
        self.assertEqual(rec.rank("r1", CANDIDATES, 2), ["W1", "W3"])

    def test_ties_break_by_worker_id(self):
        rec = make_recommender(weights={"skill": 0.0, "related_skill": 0.0, "category": 1.0, "history": 0.0, "reputation": 0.0})
        # W1, W3, W4 all primary (tied), W2 secondary
        self.assertEqual(rec.rank("r1", ["W4", "W3", "W2", "W1"], 10), ["W1", "W3", "W4", "W2"])

    def test_ranking_is_independent_of_candidate_order(self):
        rec = make_recommender()
        self.assertEqual(rec.rank("r1", CANDIDATES, 10), make_recommender().rank("r1", list(reversed(CANDIDATES)), 10))

    def test_empty_candidates(self):
        self.assertEqual(make_recommender().rank("r1", [], 10), [])

    def test_with_config_shares_cache_and_rejects_uncached_damping(self):
        cache = KGComponentCache((0.0, 0.4))
        rec = make_recommender(cache=cache)
        rec.rank("r1", CANDIDATES, 10)
        other = rec.with_config(0.4, EQUAL_WEIGHTS)
        other.rank("r1", CANDIDATES, 10)
        self.assertEqual(len(cache), 1)
        with self.assertRaises(ValueError):
            rec.with_config(0.9, EQUAL_WEIGHTS)

    def test_bad_weights_rejected(self):
        with self.assertRaises(ValueError):
            make_recommender(weights={"skill": 1.0})


class TestColdStartWorker(unittest.TestCase):
    def test_worker_without_history_gets_no_history_paths_and_neutral_prior(self):
        rec = make_recommender()
        paths = enumerate_paths(rec.index, rec.context("r1"), "W4", ROLE_WEIGHTS)
        self.assertEqual({t for t, _, _, _ in paths}, {"related_skill", "category"})
        self.assertEqual(rec.reputation("W4"), rec.index.mean_train_rating)


class TestTemporalGuards(unittest.TestCase):
    def test_request_not_after_history_is_rejected(self):
        with self.assertRaises(TemporalLeakageError):
            make_recommender().rank("r_early", CANDIDATES, 10)

    def test_request_whose_own_outcome_is_a_graph_edge_is_rejected(self):
        requests = toy_requests([{"link": "t3", "client_id": "C2", "category": "Cat", "simulated_day": 50.0,
                                  "skills_vocab": "A", "split": "validation"}])
        with self.assertRaises(TemporalLeakageError):
            make_recommender(requests=requests).rank("t3", CANDIDATES, 10)

    def test_guard_runs_on_cache_hits_too(self):
        rec = make_recommender()
        rec.rank("r1", CANDIDATES, 10)
        rec._contexts.clear()
        rec.index.max_history_day = 99.0  # simulate a snapshot that now overlaps r1
        with self.assertRaises(TemporalLeakageError):
            rec.rank("r1", CANDIDATES, 10)

    def test_unknown_link(self):
        with self.assertRaises(KeyError):
            make_recommender().rank("missing", CANDIDATES, 10)


if __name__ == "__main__":
    unittest.main()
