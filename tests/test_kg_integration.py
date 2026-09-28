"""End-to-end Day 6 checks against the real Day 2/Day 3 data: prepare_all()
produces a complete, internally consistent KG dataset. Skips cleanly if
the synthetic dataset hasn't been generated locally.
"""
import unittest
from pathlib import Path

from src.data.config import load_config
from src.kg.prepare import prepare_all

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"


@unittest.skipUnless(SYNTHETIC_DIR.exists(), f"{SYNTHETIC_DIR} not found; run scripts/generate_synthetic_data.py first")
class TestKGIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = prepare_all()

    def test_population_counts(self):
        self.assertEqual(len(self.kg.clients), 2000)
        self.assertEqual(len(self.kg.workers), 3000)
        self.assertEqual(len(self.kg.skills), 1187)
        self.assertEqual(len(self.kg.categories), 88)

    def test_all_relationship_endpoints_reference_existing_nodes(self):
        worker_ids = set(self.kg.workers["worker_id"])
        client_ids = set(self.kg.clients["client_id"])
        skill_names = set(self.kg.skills["name"])
        category_names = set(self.kg.categories["name"])
        location_names = set(self.kg.locations["name"])

        self.assertTrue(set(self.kg.has_skill["worker_id"]).issubset(worker_ids))
        self.assertTrue(set(self.kg.has_skill["skill"]).issubset(skill_names))
        self.assertTrue(set(self.kg.specializes_in["worker_id"]).issubset(worker_ids))
        self.assertTrue(set(self.kg.specializes_in["category"]).issubset(category_names))
        self.assertTrue(set(self.kg.located_in_worker["worker_id"]).issubset(worker_ids))
        self.assertTrue(set(self.kg.located_in_worker["location"]).issubset(location_names))
        self.assertTrue(set(self.kg.located_in_client["client_id"]).issubset(client_ids))
        self.assertTrue(set(self.kg.requested_service["client_id"]).issubset(client_ids))
        self.assertTrue(set(self.kg.requested_service["category"]).issubset(category_names))
        self.assertTrue(set(self.kg.hired["client_id"]).issubset(client_ids))
        self.assertTrue(set(self.kg.hired["worker_id"]).issubset(worker_ids))
        self.assertTrue(set(self.kg.reviewed["client_id"]).issubset(client_ids))
        self.assertTrue(set(self.kg.reviewed["worker_id"]).issubset(worker_ids))
        self.assertTrue(set(self.kg.related_to["skill_a"]).issubset(skill_names))
        self.assertTrue(set(self.kg.related_to["skill_b"]).issubset(skill_names))

    def test_no_duplicate_domain_ids(self):
        self.assertTrue(self.kg.clients["client_id"].is_unique)
        self.assertTrue(self.kg.workers["worker_id"].is_unique)
        self.assertTrue(self.kg.skills["name"].is_unique)
        self.assertTrue(self.kg.categories["name"].is_unique)
        self.assertTrue(self.kg.locations["name"].is_unique)
        self.assertTrue(self.kg.requested_service["link"].is_unique)
        self.assertTrue(self.kg.hired["link"].is_unique)

    def test_deterministic_rerun(self):
        rerun = prepare_all()
        self.assertTrue(self.kg.clients.equals(rerun.clients))
        self.assertTrue(self.kg.hired.equals(rerun.hired))
        self.assertTrue(self.kg.related_to.equals(rerun.related_to))

    def test_frozen_generator_hash_unchanged(self):
        from src.data.io import sha256_of
        with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
            expected = fh.read().split()[0]
        self.assertEqual(sha256_of(REPO_ROOT / "config" / "generator_v1.yaml"), expected)

    def test_day4_and_day5_artifacts_unchanged(self):
        import json

        day4_path = REPO_ROOT / "results" / "day4" / "selected_cbf_config.json"
        day5_path = REPO_ROOT / "results" / "day5" / "selected_cf_config.json"
        if day4_path.exists():
            with open(day4_path, encoding="utf-8") as fh:
                cbf = json.load(fh)
            self.assertEqual(cbf["weights"], {"category": 0.25, "price": 0.25, "skill": 0.5})
        if day5_path.exists():
            with open(day5_path, encoding="utf-8") as fh:
                cf = json.load(fh)
            self.assertEqual((cf["aggregation"], cf["method"], cf["k_neighbors"]), ("sum", "user", 20))

    def test_phase1_split_counts_unchanged(self):
        data_cfg = load_config()
        split_counts = self.kg.requested_service["split"].value_counts()
        self.assertEqual(split_counts["train"], round(data_cfg.model_requests * data_cfg.train_ratio))
        self.assertEqual(split_counts["validation"], round(data_cfg.model_requests * data_cfg.validation_ratio))
        self.assertEqual(split_counts["test"], round(data_cfg.model_requests * data_cfg.test_ratio))


if __name__ == "__main__":
    unittest.main()
