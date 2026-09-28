import unittest

import pandas as pd

from src.recommenders.content_based import ContentBasedRecommender, jaccard_similarity

WEIGHTS = {"skill": 0.5, "category": 0.25, "price": 0.25}


def make_workers(rows):
    defaults = {
        "worker_id": None, "primary_category": "Writing", "secondary_categories": "",
        "declared_skills": "", "price_tier": 0.5,
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def make_requests(rows):
    defaults = {"link": None, "category": "Writing", "skills_vocab": "", "price_tier": 0.5}
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestJaccardSimilarity(unittest.TestCase):
    def test_identical_sets(self):
        self.assertEqual(jaccard_similarity(frozenset({"a", "b"}), frozenset({"a", "b"})), 1.0)

    def test_disjoint_sets(self):
        self.assertEqual(jaccard_similarity(frozenset({"a"}), frozenset({"b"})), 0.0)

    def test_partial_overlap(self):
        # intersection {b} = 1, union {a,b,c} = 3 -> 1/3
        self.assertAlmostEqual(jaccard_similarity(frozenset({"a", "b"}), frozenset({"b", "c"})), 1 / 3)

    def test_both_empty(self):
        self.assertEqual(jaccard_similarity(frozenset(), frozenset()), 0.0)


class TestSkillScoring(unittest.TestCase):
    def test_strong_skill_match_scores_higher_than_unrelated(self):
        workers = make_workers([
            {"worker_id": "W_match", "declared_skills": "Python|Django|SQL"},
            {"worker_id": "W_unrelated", "declared_skills": "Graphic Design|Adobe Photoshop"},
        ])
        requests = make_requests([{"link": "r1", "skills_vocab": "Python|Django|SQL"}])
        cbf = ContentBasedRecommender(workers, requests, WEIGHTS, price_neutral_score=0.5)
        self.assertGreater(cbf.score("r1", "W_match"), cbf.score("r1", "W_unrelated"))

    def test_exact_skill_match_ranks_first(self):
        workers = make_workers([
            {"worker_id": "W_exact", "declared_skills": "Python|Django|SQL"},
            {"worker_id": "W_partial", "declared_skills": "Python"},
            {"worker_id": "W_none", "declared_skills": "Welding"},
        ])
        requests = make_requests([{"link": "r1", "skills_vocab": "Python|Django|SQL"}])
        cbf = ContentBasedRecommender(workers, requests, WEIGHTS, price_neutral_score=0.5)
        ranked = cbf.rank("r1", ["W_none", "W_partial", "W_exact"], k=10)
        self.assertEqual(ranked[0], "W_exact")

    def test_missing_skills_treated_as_empty_not_error(self):
        workers = make_workers([{"worker_id": "W1", "declared_skills": None}])
        requests = make_requests([{"link": "r1", "skills_vocab": "Python"}])
        cbf = ContentBasedRecommender(workers, requests, WEIGHTS, price_neutral_score=0.5)
        self.assertEqual(cbf.score("r1", "W1"), WEIGHTS["category"] * 1.0 + WEIGHTS["price"] * 1.0)


class TestCategoryCompatibility(unittest.TestCase):
    def test_primary_category_scores_full(self):
        workers = make_workers([{"worker_id": "W1", "primary_category": "Writing"}])
        requests = make_requests([{"link": "r1", "category": "Writing"}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 1, "price": 0}, price_neutral_score=0.5)
        self.assertEqual(cbf.score("r1", "W1"), 1.0)

    def test_secondary_category_scores_partial(self):
        workers = make_workers([{"worker_id": "W1", "primary_category": "Design", "secondary_categories": "Writing|Marketing"}])
        requests = make_requests([{"link": "r1", "category": "Writing"}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 1, "price": 0}, price_neutral_score=0.5)
        self.assertEqual(cbf.score("r1", "W1"), 0.6)

    def test_no_category_match_scores_zero(self):
        workers = make_workers([{"worker_id": "W1", "primary_category": "Design", "secondary_categories": ""}])
        requests = make_requests([{"link": "r1", "category": "Writing"}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 1, "price": 0}, price_neutral_score=0.5)
        self.assertEqual(cbf.score("r1", "W1"), 0.0)

    def test_primary_beats_secondary_ranking(self):
        workers = make_workers([
            {"worker_id": "W_primary", "primary_category": "Writing"},
            {"worker_id": "W_secondary", "primary_category": "Design", "secondary_categories": "Writing"},
        ])
        requests = make_requests([{"link": "r1", "category": "Writing"}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 1, "price": 0}, price_neutral_score=0.5)
        ranked = cbf.rank("r1", ["W_secondary", "W_primary"], k=10)
        self.assertEqual(ranked[0], "W_primary")


class TestPriceCompatibility(unittest.TestCase):
    def test_bounded_between_zero_and_one(self):
        workers = make_workers([
            {"worker_id": "W_cheap", "price_tier": 0.0},
            {"worker_id": "W_expensive", "price_tier": 1.0},
        ])
        requests = make_requests([{"link": "r1", "price_tier": 0.5}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 0, "price": 1}, price_neutral_score=0.5)
        for worker_id in ("W_cheap", "W_expensive"):
            score = cbf.score("r1", worker_id)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_exact_price_tier_match_scores_one(self):
        workers = make_workers([{"worker_id": "W1", "price_tier": 0.7}])
        requests = make_requests([{"link": "r1", "price_tier": 0.7}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 0, "price": 1}, price_neutral_score=0.5)
        self.assertAlmostEqual(cbf.score("r1", "W1"), 1.0)

    def test_far_price_tier_scores_near_zero(self):
        workers = make_workers([{"worker_id": "W1", "price_tier": 0.0}])
        requests = make_requests([{"link": "r1", "price_tier": 1.0}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 0, "price": 1}, price_neutral_score=0.5)
        self.assertAlmostEqual(cbf.score("r1", "W1"), 0.0)

    def test_unspecified_request_price_uses_neutral_score(self):
        workers = make_workers([{"worker_id": "W1", "price_tier": 0.0}])
        requests = make_requests([{"link": "r1", "price_tier": float("nan")}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 0, "price": 1}, price_neutral_score=0.42)
        self.assertAlmostEqual(cbf.score("r1", "W1"), 0.42)

    def test_missing_worker_price_tier_uses_neutral_score(self):
        workers = make_workers([{"worker_id": "W1", "price_tier": float("nan")}])
        requests = make_requests([{"link": "r1", "price_tier": 0.5}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0, "category": 0, "price": 1}, price_neutral_score=0.42)
        self.assertAlmostEqual(cbf.score("r1", "W1"), 0.42)


class TestNoPrivateFieldsConsumed(unittest.TestCase):
    def test_works_without_any_private_columns_present(self):
        # workers/requests frames here carry ONLY the observable columns
        # CBF documents using; if it silently depended on a private
        # column (quality, true_skills, proficiency, taste_*), this
        # would raise a KeyError.
        workers = make_workers([{"worker_id": "W1", "declared_skills": "Python"}])
        requests = make_requests([{"link": "r1", "skills_vocab": "Python"}])
        self.assertNotIn("quality", workers.columns)
        self.assertNotIn("true_skills", workers.columns)
        cbf = ContentBasedRecommender(workers, requests, WEIGHTS, price_neutral_score=0.5)
        score = cbf.score("r1", "W1")
        self.assertIsInstance(score, float)


class TestDeterminismAndValidity(unittest.TestCase):
    def test_deterministic_ranking(self):
        workers = make_workers([
            {"worker_id": f"W{i}", "declared_skills": "Python" if i % 2 == 0 else "Design"} for i in range(20)
        ])
        requests = make_requests([{"link": "r1", "skills_vocab": "Python"}])
        cbf1 = ContentBasedRecommender(workers, requests, WEIGHTS, price_neutral_score=0.5)
        cbf2 = ContentBasedRecommender(workers, requests, WEIGHTS, price_neutral_score=0.5)
        candidates = list(workers["worker_id"])
        self.assertEqual(cbf1.rank("r1", candidates, k=10), cbf2.rank("r1", candidates, k=10))

    def test_no_duplicates_and_respects_k(self):
        workers = make_workers([{"worker_id": f"W{i}"} for i in range(20)])
        requests = make_requests([{"link": "r1"}])
        cbf = ContentBasedRecommender(workers, requests, WEIGHTS, price_neutral_score=0.5)
        candidates = list(workers["worker_id"])
        ranked = cbf.rank("r1", candidates, k=10)
        self.assertEqual(len(ranked), 10)
        self.assertEqual(len(ranked), len(set(ranked)))

    def test_invalid_weight_keys_raise(self):
        workers = make_workers([{"worker_id": "W1"}])
        requests = make_requests([{"link": "r1"}])
        with self.assertRaises(ValueError):
            ContentBasedRecommender(workers, requests, {"skill": 1.0}, price_neutral_score=0.5)


if __name__ == "__main__":
    unittest.main()
