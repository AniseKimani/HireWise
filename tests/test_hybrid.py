"""Day 8 hybrid: normalisation, weighted combination, ranking, explanations,
and validation-only selection, on hand-checkable toy component scores."""
import unittest

import numpy as np

from src.evaluation.protocol import EvalRequest
from src.recommenders.hybrid import (
    COMPONENTS, HybridRecommender, HybridScoreCache, explanation_text, minmax_normalize, tune_hybrid, validate_weights,
)

CANDIDATES = ["W1", "W2", "W3", "W4"]
TOY = {
    "cbf": {"W1": 0.9, "W2": 0.5, "W3": 0.1, "W4": 0.5},
    "cf": {"W1": 0.0, "W2": 0.0, "W3": 0.0, "W4": 0.0},   # constant -> contributes 0
    "kg": {"W1": 0.2, "W2": 0.8, "W3": 0.6, "W4": 0.2},
}
EQUAL = {"cbf": 0.34, "cf": 0.33, "kg": 0.33}


def toy_fns(table=TOY, calls=None):
    def make(component):
        def fn(link, candidates):
            if calls is not None:
                calls.append((component, link))
            return np.array([table[component][w] for w in candidates], dtype=float)
        return fn
    return {c: make(c) for c in COMPONENTS}


def hybrid(weights=EQUAL, table=TOY, calls=None):
    return HybridRecommender(toy_fns(table, calls), weights)


class TestNormalisation(unittest.TestCase):
    def test_minmax_correctness(self):
        out = minmax_normalize(np.array([[2.0], [4.0], [3.0]]))
        np.testing.assert_allclose(out[:, 0], [0.0, 1.0, 0.5])

    def test_constant_column_is_zero(self):
        out = minmax_normalize(np.array([[5.0, 1.0], [5.0, 3.0]]))
        np.testing.assert_allclose(out[:, 0], [0.0, 0.0])
        np.testing.assert_allclose(out[:, 1], [0.0, 1.0])

    def test_single_candidate_is_zero(self):
        np.testing.assert_allclose(minmax_normalize(np.array([[0.7, 0.2, 0.1]])), [[0.0, 0.0, 0.0]])

    def test_deterministic_and_scale_invariant(self):
        raw = np.array([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]])
        np.testing.assert_array_equal(minmax_normalize(raw), minmax_normalize(raw.copy()))
        np.testing.assert_allclose(minmax_normalize(raw)[:, 0], minmax_normalize(raw)[:, 1])


class TestWeightedSum(unittest.TestCase):
    def test_scores_match_hand_computation(self):
        raw, normalized, contributions, score = hybrid().scored("r", CANDIDATES)
        np.testing.assert_allclose(normalized[:, 0], [1.0, 0.5, 0.0, 0.5])
        np.testing.assert_allclose(normalized[:, 1], [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(normalized[:, 2], [0.0, 1.0, 4 / 6, 0.0])
        np.testing.assert_allclose(score, [0.34, 0.34 * 0.5 + 0.33, 0.33 * 4 / 6, 0.17])
        np.testing.assert_allclose(contributions.sum(axis=1), score)

    def test_weights_validated(self):
        for bad in ({"cbf": 0.5, "cf": 0.5}, {"cbf": 0.6, "cf": 0.3, "kg": 0.3}, {"cbf": -0.1, "cf": 0.6, "kg": 0.5}):
            with self.assertRaises(ValueError):
                validate_weights(bad)
        validate_weights(EQUAL)

    def test_ranking_preserves_candidate_set_and_k(self):
        ranked = hybrid().rank("r", CANDIDATES, 10)
        self.assertEqual(sorted(ranked), sorted(CANDIDATES))
        self.assertEqual(ranked, ["W2", "W1", "W3", "W4"])
        self.assertEqual(hybrid().rank("r", CANDIDATES, 2), ["W2", "W1"])
        self.assertEqual(hybrid().rank("r", [], 10), [])

    def test_ties_break_by_worker_id_regardless_of_candidate_order(self):
        tied = {c: {w: 1.0 for w in CANDIDATES} for c in COMPONENTS}
        self.assertEqual(hybrid(table=tied).rank("r", ["W4", "W2", "W3", "W1"], 10), ["W1", "W2", "W3", "W4"])
        self.assertEqual(hybrid().rank("r", list(reversed(CANDIDATES)), 10), hybrid().rank("r", CANDIDATES, 10))

    def test_single_component_hybrid_reproduces_component_ranking(self):
        for c in ("cbf", "kg"):
            weights = {x: (1.0 if x == c else 0.0) for x in COMPONENTS}
            expected = sorted(CANDIDATES, key=lambda w: (-TOY[c][w], w))
            self.assertEqual(hybrid(weights).rank("r", CANDIDATES, 10), expected)

    def test_cache_reused_across_weights(self):
        calls = []
        base = hybrid(calls=calls)
        base.rank("r", CANDIDATES, 10)
        base.with_weights({"cbf": 0.6, "cf": 0.2, "kg": 0.2}).rank("r", CANDIDATES, 10)
        self.assertEqual(len(calls), 3)  # each component scored once for request r


class TestExplanations(unittest.TestCase):
    def test_contributions_reconstruct_score_and_order(self):
        for e in hybrid().explain("r", CANDIDATES, 4):
            self.assertAlmostEqual(sum(e.contributions.values()), e.score, places=12)
            if e.cited_components:
                self.assertEqual(e.dominant_component, max(e.contributions, key=e.contributions.get))
            self.assertNotIn("cf", e.cited_components)  # constant CF contributes nothing
            self.assertIn("collaborative history (CF)", e.text.split("no contribution from")[1])

    def test_explanations_follow_ranking(self):
        h = hybrid()
        self.assertEqual([e.worker_id for e in h.explain("r", CANDIDATES, 3)], h.rank("r", CANDIDATES, 3))

    def test_text_uses_actual_contributions(self):
        text = explanation_text({"cbf": 0.31, "cf": 0.08, "kg": 0.44}, ["kg", "cbf", "cf"], 0.83, "Has required skill 'Python'")
        self.assertTrue(text.startswith("Recommended because knowledge-graph evidence (KG) contributed 0.44"))
        self.assertIn("Strongest KG evidence: Has required skill 'Python'", text)
        self.assertIn("No component", explanation_text({c: 0.0 for c in COMPONENTS}, [], 0.0, None))


def _request(link, relevance):
    return EvalRequest(link=link, split="validation", category="Cat", candidates=CANDIDATES, relevance=relevance,
                       n_candidates=4, excluded=False, excluded_reason=None)


class TestTuning(unittest.TestCase):
    def test_selects_best_and_breaks_exact_ties_by_balance(self):
        grid = [{"cbf": 0.5, "cf": 0.25, "kg": 0.25}, {"cbf": 0.34, "cf": 0.33, "kg": 0.33},
                {"cbf": 0.2, "cf": 0.2, "kg": 0.6}]
        requests = [_request("r", {"W2": 2})]
        best, rows = tune_hybrid(hybrid(), grid, requests, 2, "ndcg_at_10", "recall_at_10")
        self.assertEqual(sum(r["selected"] for r in rows), 1)
        best_row = max(rows, key=lambda r: (r["ndcg_at_k"], r["recall_at_k"], -r["balance"], -r["grid_index"]))
        self.assertEqual(best["grid_index"], best_row["grid_index"])

    def test_tuning_only_sees_the_requests_it_is_given(self):
        calls = []
        requests = [_request("val_1", {"W1": 1})]
        tune_hybrid(hybrid(calls=calls), [EQUAL], requests, 2, "ndcg_at_10", "recall_at_10")
        self.assertEqual({link for _, link in calls}, {"val_1"})


if __name__ == "__main__":
    unittest.main()
