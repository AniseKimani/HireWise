"""End-to-end Day 5 checks against the real Day 2/3/4 data: CF trains,
evaluates, and integrates with the shared protocol correctly. Skips
cleanly if the synthetic dataset hasn't been generated locally.
"""
import unittest
from pathlib import Path

import pandas as pd

from src.data.config import load_config
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import evaluate
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.recommenders.cf_config import load_cf_config
from src.recommenders.collaborative import CollaborativeFiltering, build_training_matrix
from src.recommenders.popularity import PopularityBaseline

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"


@unittest.skipUnless(SYNTHETIC_DIR.exists(), f"{SYNTHETIC_DIR} not found; run scripts/generate_synthetic_data.py first")
class TestDay5Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_cfg = load_config()
        cls.eval_cfg = load_evaluation_config()
        cls.cf_cfg = load_cf_config()
        cls.data = load_protocol_data()
        cls.index = EligibilityIndex(cls.data.workers, cls.data.known_countries)
        cls.val_requests = build_eval_requests(cls.data, cls.index, "validation", cls.eval_cfg)
        cls.link_to_client = dict(zip(cls.data.requests["link"], cls.data.requests["client_id"]))
        cls.fallback = PopularityBaseline(cls.data.interactions, cls.data.requests)
        cls.matrix = build_training_matrix(cls.data.interactions, cls.data.requests, cls.cf_cfg.event_weights, "sum")
        cls.cf = CollaborativeFiltering(cls.matrix, "user", 20, cls.fallback, cls.link_to_client, cls.cf_cfg.min_similarity)

    def test_matrix_built_from_training_only(self):
        train_links = set(self.data.requests.loc[self.data.requests["split"] == "train", "link"])
        non_train_links = set(self.data.requests["link"]) - train_links
        # Sanity: matrix worker/client sets are derived only from training
        # interactions -- cross-check against an independent recomputation.
        train_interactions = self.data.interactions.loc[self.data.interactions["link"].isin(train_links)]
        self.assertEqual(set(self.matrix.workers), set(train_interactions["worker_id"].unique()))

    def test_cf_produces_valid_top10_on_real_requests(self):
        included = [r for r in self.val_requests if not r.excluded][:100]
        for req in included:
            ranked = self.cf.rank(req.link, req.candidates, self.eval_cfg.k)
            self.assertLessEqual(len(ranked), self.eval_cfg.k)
            self.assertEqual(len(ranked), len(set(ranked)))
            self.assertTrue(set(ranked).issubset(set(req.candidates)))

    def test_evaluator_runs_end_to_end_for_cf(self):
        df, summary = evaluate(self.cf, self.val_requests, self.eval_cfg.k)
        self.assertGreater(summary["n_included"], 0)
        for metric in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"):
            self.assertGreaterEqual(summary[metric], 0.0)
            self.assertLessEqual(summary[metric], 1.0)

    def test_cold_start_workers_absent_from_training_matrix(self):
        cold_start_ids = set(pd.read_csv(SYNTHETIC_DIR / "cold_start_cohort.csv")["worker_id"])
        overlap = cold_start_ids & set(self.matrix.worker_index.keys())
        self.assertEqual(overlap, set())

    def test_phase1_population_counts_unchanged(self):
        self.assertEqual(len(self.data.workers), 3000)
        counts = self.data.requests["split"].value_counts()
        self.assertEqual(counts["train"], 8400)
        self.assertEqual(counts["validation"], 1200)
        self.assertEqual(counts["test"], 2400)

    def test_frozen_generator_hash_unchanged(self):
        from src.data.io import sha256_of
        with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
            expected = fh.read().split()[0]
        self.assertEqual(sha256_of(REPO_ROOT / "config" / "generator_v1.yaml"), expected)

    def test_day4_selected_cbf_config_unchanged(self):
        day4_config_path = REPO_ROOT / "results" / "day4" / "selected_cbf_config.json"
        if not day4_config_path.exists():
            self.skipTest("Day 4 results not generated locally")
        import json
        with open(day4_config_path, encoding="utf-8") as fh:
            config = json.load(fh)
        self.assertEqual(config["weights"], {"category": 0.25, "price": 0.25, "skill": 0.5})


if __name__ == "__main__":
    unittest.main()
