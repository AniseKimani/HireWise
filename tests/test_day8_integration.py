"""End-to-end Day 8 checks against real Day 2/3/6/7 data: frozen models
rebuild exactly, the hybrid integrates with the unchanged protocol, and
the cold-start subset is the established one. Skips if the synthetic data
or Day 7 results are absent locally."""
import json
import unittest
from pathlib import Path

import pandas as pd

from src.data.config import load_config
from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import cold_start_requests, evaluate
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.recommenders.cf_config import load_cf_config
from src.recommenders.frozen_models import build_frozen_models
from src.recommenders.hybrid import HybridRecommender, cbf_score_fn, cf_score_fn, kg_score_fn
from src.recommenders.hybrid_config import load_hybrid_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"
DAY7 = REPO_ROOT / "results" / "day7"
DAY8 = REPO_ROOT / "results" / "day8"


@unittest.skipUnless(SYNTHETIC_DIR.exists() and (DAY7 / "selected_kg_config.json").exists(), "Day 3/Day 7 artifacts not present")
class TestDay8Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_cfg = load_evaluation_config()
        cls.hybrid_cfg = load_hybrid_config()
        cls.data = load_protocol_data()
        index = EligibilityIndex(cls.data.workers, cls.data.known_countries)
        cls.val = build_eval_requests(cls.data, index, "validation", cls.eval_cfg)
        cls.test = build_eval_requests(cls.data, index, "test", cls.eval_cfg)
        cls.subset = [r for r in cls.val if not r.excluded][:60]
        cls.models = build_frozen_models(cls.data, load_config().seed, load_cf_config().min_similarity)
        cls.fns = {"cbf": cbf_score_fn(cls.models["cbf"]), "cf": cf_score_fn(cls.models["cf"]), "kg": kg_score_fn(cls.models["kg"])}

    def test_rebuilt_kg_reproduces_frozen_validation_metrics(self):
        frozen = pd.read_csv(DAY7 / "validation_metrics.csv").set_index("method")
        _, summary = evaluate(self.models["kg"], self.val, self.eval_cfg.k)
        for m in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"):
            self.assertAlmostEqual(summary[m], frozen.loc["kg", m], places=12)

    def test_hybrid_rankings_valid(self):
        h = HybridRecommender(self.fns, self.hybrid_cfg.weight_grid[0])
        for req in self.subset:
            ranked = h.rank(req.link, req.candidates, self.eval_cfg.k)
            self.assertEqual(len(ranked), min(10, len(req.candidates)))
            self.assertEqual(len(set(ranked)), len(ranked))
            self.assertTrue(set(ranked) <= set(req.candidates))

    def test_single_component_hybrid_matches_cbf_and_kg(self):
        for name in ("cbf", "kg"):
            h = HybridRecommender(self.fns, {c: (1.0 if c == name else 0.0) for c in ("cbf", "cf", "kg")})
            for req in self.subset[:30]:
                self.assertEqual(h.rank(req.link, req.candidates, 10), self.models[name].rank(req.link, req.candidates, 10))

    def test_cold_start_subset_is_the_established_one(self):
        cold = set(pd.read_csv(SYNTHETIC_DIR / "cold_start_cohort.csv")["worker_id"])
        self.assertEqual(len(cold_start_requests(self.val, cold)), 442)
        self.assertEqual(len(cold_start_requests(self.test, cold)), 924)

    def test_protocol_request_sets_unchanged(self):
        self.assertEqual(sum(not r.excluded for r in self.val), 1126)
        self.assertEqual(sum(not r.excluded for r in self.test), 2275)

    def test_recorded_day8_selection_is_from_grid(self):
        path = DAY8 / "selected_hybrid_config.json"
        if not path.exists():
            self.skipTest("results/day8 not generated yet")
        with open(path, encoding="utf-8") as fh:
            selected = json.load(fh)
        self.assertIn(selected["weights"], self.hybrid_cfg.weight_grid)
        self.assertAlmostEqual(sum(selected["weights"].values()), 1.0)


if __name__ == "__main__":
    unittest.main()
