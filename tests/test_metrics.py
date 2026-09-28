"""Hand-calculated unit tests for src/evaluation/metrics.py."""
import unittest

from src.evaluation.metrics import (
    f1_at_k,
    macro_average,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionRecallF1(unittest.TestCase):
    def setUp(self):
        self.ranked = ["a", "b", "c", "d", "e"]
        self.relevance = {"a": 0, "b": 2, "c": 1, "d": 0, "e": 0}

    def test_precision_at_3_hand_calculated(self):
        # top3 = a,b,c; relevant (grade>=1) among them: b, c -> 2/3
        self.assertAlmostEqual(precision_at_k(self.ranked, self.relevance, 3), 2 / 3)

    def test_recall_at_3_hand_calculated(self):
        # total relevant anywhere = b, c = 2; both in top3 -> 2/2 = 1.0
        self.assertAlmostEqual(recall_at_k(self.ranked, self.relevance, 3), 1.0)

    def test_f1_at_3_hand_calculated(self):
        # P=2/3, R=1.0 -> F1 = 2*P*R/(P+R) = 2*(2/3)/(5/3) = 4/5 = 0.8
        self.assertAlmostEqual(f1_at_k(self.ranked, self.relevance, 3), 0.8)

    def test_precision_at_k_full_ranking(self):
        # top5 = all 5; relevant = b,c -> 2/5
        self.assertAlmostEqual(precision_at_k(self.ranked, self.relevance, 5), 2 / 5)

    def test_no_relevant_items_gives_zero_everywhere(self):
        relevance = {w: 0 for w in self.ranked}
        self.assertEqual(precision_at_k(self.ranked, relevance, 3), 0.0)
        self.assertEqual(recall_at_k(self.ranked, relevance, 3), 0.0)
        self.assertEqual(f1_at_k(self.ranked, relevance, 3), 0.0)

    def test_multiple_relevant_items(self):
        ranked = ["a", "b", "c", "d"]
        relevance = {"a": 1, "b": 1, "c": 1, "d": 0}
        self.assertAlmostEqual(precision_at_k(ranked, relevance, 4), 3 / 4)
        self.assertAlmostEqual(recall_at_k(ranked, relevance, 4), 1.0)

    def test_fewer_than_k_recommendations(self):
        ranked = ["a", "b"]
        relevance = {"a": 1, "b": 0}
        # divisor is min(k, len(ranking)) = 2, not k = 5
        self.assertAlmostEqual(precision_at_k(ranked, relevance, 5), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, relevance, 5), 1.0)

    def test_empty_ranking(self):
        self.assertEqual(precision_at_k([], {"a": 1}, 5), 0.0)
        self.assertEqual(recall_at_k([], {"a": 1}, 5), 0.0)

    def test_duplicate_recommendations_raise(self):
        with self.assertRaises(ValueError):
            precision_at_k(["a", "a", "b"], {"a": 1}, 3)
        with self.assertRaises(ValueError):
            recall_at_k(["a", "a", "b"], {"a": 1}, 3)
        with self.assertRaises(ValueError):
            ndcg_at_k(["a", "a", "b"], {"a": 1}, 3)


class TestNdcg(unittest.TestCase):
    def test_ndcg_at_3_hand_calculated(self):
        ranked = ["a", "b", "c", "d", "e"]
        relevance = {"a": 0, "b": 2, "c": 1, "d": 0, "e": 0}
        # DCG@3 = (2^0-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
        #       = 0 + 3/1.5849625 + 1/2 = 1.892789 + 0.5 = 2.392789
        # ideal order top3 by grade desc = [2,1,0]
        # IDCG@3 = (2^2-1)/log2(2) + (2^1-1)/log2(3) + (2^0-1)/log2(4)
        #        = 3/1 + 1/1.5849625 + 0 = 3.630930
        # NDCG@3 = 2.392789 / 3.630930 = 0.659;
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, 3), 0.6590, places=3)

    def test_ndcg_perfect_ranking_is_one(self):
        ranked = ["a", "b", "c"]
        relevance = {"a": 2, "b": 1, "c": 0}
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, 3), 1.0)

    def test_ndcg_no_relevant_items_is_zero(self):
        ranked = ["a", "b", "c"]
        relevance = {"a": 0, "b": 0, "c": 0}
        self.assertEqual(ndcg_at_k(ranked, relevance, 3), 0.0)

    def test_ndcg_worst_ranking_less_than_best(self):
        relevance = {"a": 2, "b": 1, "c": 0}
        best = ndcg_at_k(["a", "b", "c"], relevance, 3)
        worst = ndcg_at_k(["c", "b", "a"], relevance, 3)
        self.assertGreater(best, worst)

    def test_ndcg_fewer_than_k_items(self):
        ranked = ["a", "b"]
        relevance = {"a": 1, "b": 0}
        # DCG = (2^1-1)/log2(2) + (2^0-1)/log2(3) = 1 + 0 = 1
        # IDCG = same ideal (only one relevant item) = 1
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, 5), 1.0)


class TestMacroAverage(unittest.TestCase):
    def test_simple_average(self):
        per_request = [
            {"precision_at_k": 1.0, "ndcg_at_k": 1.0},
            {"precision_at_k": 0.0, "ndcg_at_k": 0.5},
        ]
        result = macro_average(per_request)
        self.assertAlmostEqual(result["precision_at_k"], 0.5)
        self.assertAlmostEqual(result["ndcg_at_k"], 0.75)

    def test_empty_input_returns_empty_dict(self):
        self.assertEqual(macro_average([]), {})


if __name__ == "__main__":
    unittest.main()
