"""End-to-end Day 4 checks against the real Day 2/Day 3 data: all three
recommenders produce valid top-10 outputs and the evaluator runs
end-to-end. Skips cleanly if the synthetic dataset hasn't been
generated locally.
"""
import unittest
from pathlib import Path

from src.data.config import load_config
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import evaluate
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityBaseline
from src.recommenders.random_baseline import RandomBaseline

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"


@unittest.skipUnless(SYNTHETIC_DIR.exists(), f"{SYNTHETIC_DIR} not found; run scripts/generate_synthetic_data.py first")
class TestDay4Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_cfg = load_config()
        cls.eval_cfg = load_evaluation_config()
        cls.data = load_protocol_data()
        cls.index = EligibilityIndex(cls.data.workers, cls.data.known_countries)
        cls.val_requests = build_eval_requests(cls.data, cls.index, "validation", cls.eval_cfg)
        cls.recommenders = {
            "random": RandomBaseline(seed=cls.data_cfg.seed),
            "popularity": PopularityBaseline(cls.data.interactions, cls.data.requests),
            "cbf": ContentBasedRecommender(
                cls.data.workers, cls.data.requests,
                {"skill": 0.5, "category": 0.25, "price": 0.25}, cls.eval_cfg.cbf_price_neutral_score,
            ),
        }

    def test_all_three_recommenders_produce_valid_top10(self):
        included = [r for r in self.val_requests if not r.excluded][:50]
        for name, rec in self.recommenders.items():
            for req in included:
                ranked = rec.rank(req.link, req.candidates, self.eval_cfg.k)
                self.assertLessEqual(len(ranked), self.eval_cfg.k)
                self.assertEqual(len(ranked), len(set(ranked)), f"{name} produced duplicates")
                self.assertTrue(set(ranked).issubset(set(req.candidates)), f"{name} recommended non-candidate worker")

    def test_evaluator_runs_end_to_end_for_all_recommenders(self):
        for name, rec in self.recommenders.items():
            df, summary = evaluate(rec, self.val_requests, self.eval_cfg.k)
            self.assertGreater(summary["n_included"], 0)
            for metric in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"):
                self.assertGreaterEqual(summary[metric], 0.0)
                self.assertLessEqual(summary[metric], 1.0)

    def test_phase1_population_counts_unchanged(self):
        self.assertEqual(len(self.data.workers), 3000)
        self.assertEqual(self.data.requests["link"].nunique(), 12000)
        counts = self.data.requests["split"].value_counts()
        self.assertEqual(counts["train"], 8400)
        self.assertEqual(counts["validation"], 1200)
        self.assertEqual(counts["test"], 2400)

    def test_frozen_generator_hash_unchanged(self):
        from src.data.io import sha256_of
        with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
            expected = fh.read().split()[0]
        actual = sha256_of(REPO_ROOT / "config" / "generator_v1.yaml")
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
