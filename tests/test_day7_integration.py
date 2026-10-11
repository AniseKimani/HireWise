"""End-to-end Day 7 checks against the real Day 2/3/6 data: the KG
recommender builds from the training-only snapshot, integrates with the
shared evaluation protocol, explains its rankings faithfully, and is
deterministic. The live Neo4j parity check is READ-ONLY and skips cleanly
if Neo4j is not configured/reachable. Skips entirely if the synthetic
dataset hasn't been generated locally.
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.config import load_evaluation_config
from src.evaluation.evaluator import evaluate
from src.evaluation.protocol import EligibilityIndex, build_eval_requests, load_protocol_data
from src.kg.explanations import WORKER_HISTORY_PATH_TYPES
from src.kg.graph_view import compare_views, load_view_from_kg_csvs
from src.recommenders.kg_config import load_kg_config
from src.recommenders.knowledge_graph import KGComponentCache, KGIndex, KnowledgeGraphRecommender

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"
KG_DIR = REPO_ROOT / "data" / "processed" / "kg"
DAY7_RESULTS = REPO_ROOT / "results" / "day7"


def _neo4j_available() -> bool:
    try:
        from src.kg.neo4j_client import Neo4jClient, Neo4jConfigError

        with Neo4jClient():
            pass
        return True
    except Exception:
        return False


@unittest.skipUnless(SYNTHETIC_DIR.exists() and KG_DIR.exists(), "Day 3/Day 6 artifacts not generated locally")
class TestDay7Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_cfg = load_evaluation_config()
        cls.kg_cfg = load_kg_config()
        cls.data = load_protocol_data()
        cls.index = EligibilityIndex(cls.data.workers, cls.data.known_countries)
        cls.val_requests = build_eval_requests(cls.data, cls.index, "validation", cls.eval_cfg)
        cls.included = [r for r in cls.val_requests if not r.excluded][:60]
        cls.view = load_view_from_kg_csvs(KG_DIR)
        cls.cold = set(pd.read_csv(SYNTHETIC_DIR / "cold_start_cohort.csv")["worker_id"])

    def _recommender(self, damping=0.4, grid_index=0):
        return KnowledgeGraphRecommender(
            KGIndex(self.view), self.data.requests, damping, self.kg_cfg.weight_grid[grid_index],
            self.kg_cfg.category_role_weights, self.kg_cfg.reputation_prior_strength,
            self.kg_cfg.explanation_max_paths_per_component, KGComponentCache((damping,)),
        )

    def test_valid_top10_on_real_requests(self):
        rec = self._recommender()
        for req in self.included:
            ranked = rec.rank(req.link, req.candidates, self.eval_cfg.k)
            self.assertEqual(len(ranked), min(self.eval_cfg.k, len(req.candidates)))
            self.assertEqual(len(ranked), len(set(ranked)))
            self.assertTrue(set(ranked) <= set(req.candidates))

    def test_rankings_are_deterministic_across_fresh_instances(self):
        a, b = self._recommender(), self._recommender()
        for req in self.included[:30]:
            self.assertEqual(a.rank(req.link, req.candidates, 10), b.rank(req.link, req.candidates, 10))

    def test_evaluator_runs_end_to_end(self):
        _, summary = evaluate(self._recommender(), self.included, self.eval_cfg.k)
        self.assertEqual(summary["n_included"], len(self.included))
        for metric in ("precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"):
            self.assertTrue(0.0 <= summary[metric] <= 1.0)

    def test_explanations_are_faithful_and_cold_workers_have_no_history_paths(self):
        rec = self._recommender()
        history_types = WORKER_HISTORY_PATH_TYPES
        for req in self.included[:20]:
            ranked = rec.rank(req.link, req.candidates, 10)
            exps = rec.explain(req.link, req.candidates, 10)
            self.assertEqual([e.worker_id for e in exps], ranked)
            for e in exps:
                self.assertAlmostEqual(e.total_path_contribution + e.reputation.contribution, e.score, places=9)
                self.assertTrue(all(l <= 3 for l in e.path_lengths))
                self.assertGreaterEqual(e.n_paths_by_type["category"], 1)  # eligibility guarantees a category path
                if e.worker_id in self.cold:
                    self.assertEqual(sum(e.n_paths_by_type[t] for t in history_types), 0)
                    self.assertTrue(e.reputation.prior_used_only)

    def test_recorded_snapshot_fingerprint_matches_csv_view(self):
        snapshot_path = DAY7_RESULTS / "graph_snapshot.json"
        if not snapshot_path.exists():
            self.skipTest("results/day7 not generated yet")
        with open(snapshot_path, encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["fingerprint"], self.view.fingerprint())

    @unittest.skipUnless(_neo4j_available(), "Neo4j not configured/reachable; read-only parity check skipped")
    def test_neo4j_view_equals_csv_view_read_only(self):
        from src.kg.graph_view import load_view_from_neo4j
        from src.kg.neo4j_client import Neo4jClient

        with Neo4jClient() as client:
            neo4j_view = load_view_from_neo4j(client)
        self.assertTrue(all(compare_views(neo4j_view, self.view).values()), compare_views(neo4j_view, self.view))
        self.assertEqual(neo4j_view.fingerprint(), self.view.fingerprint())

    def test_frozen_artifacts_unchanged(self):
        from src.data.io import sha256_of

        with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
            self.assertEqual(sha256_of(REPO_ROOT / "config" / "generator_v1.yaml"), fh.read().split()[0])
        with open(REPO_ROOT / "results" / "day5" / "day5_manifest.json", encoding="utf-8") as fh:
            day5 = json.load(fh)
        self.assertEqual(sha256_of(REPO_ROOT / "results" / "day4" / "selected_cbf_config.json"), day5["day4_cbf_config_sha256"])
        self.assertEqual(sha256_of(REPO_ROOT / "config" / "cf_config.yaml"), day5["cf_config_sha256"])
        for entry in day5["files"]:
            self.assertEqual(sha256_of(REPO_ROOT / entry["path"]), entry["sha256"], entry["path"])


if __name__ == "__main__":
    unittest.main()
