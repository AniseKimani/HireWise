"""config/kg_config.yaml loading and validation (Day 7)."""
import tempfile
import unittest
from pathlib import Path

import yaml

from src.recommenders.kg_config import DEFAULT_KG_CONFIG_PATH, EXPECTED_COMPONENTS, load_kg_config


class TestKGConfig(unittest.TestCase):
    def setUp(self):
        with open(DEFAULT_KG_CONFIG_PATH, encoding="utf-8") as fh:
            self.raw = yaml.safe_load(fh)

    def _load(self, raw):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kg.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            return load_kg_config(path)

    def test_project_config_loads(self):
        cfg = load_kg_config()
        self.assertEqual(cfg.components, EXPECTED_COMPONENTS)
        self.assertEqual(cfg.tuning_primary_metric, "ndcg_at_10")
        self.assertEqual(cfg.tuning_tiebreak_metric, "recall_at_10")
        self.assertEqual(cfg.dwpc_damping_grid, [0.0, 0.4])
        self.assertEqual(len(cfg.weight_grid), 6)

    def test_project_config_has_no_price_or_location_component(self):
        cfg = load_kg_config()
        for entry in cfg.weight_grid:
            self.assertFalse({"price", "rate", "location", "interested_in"} & set(entry))

    def test_weights_must_sum_to_one(self):
        raw = dict(self.raw)
        raw["weight_grid"] = [{c: 0.5 for c in EXPECTED_COMPONENTS}]
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_weights_must_use_exact_components(self):
        raw = dict(self.raw)
        raw["weight_grid"] = [{"skill": 0.5, "price": 0.5}]
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_damping_grid_must_be_increasing_and_non_negative(self):
        for grid in ([0.4, 0.0], [-0.1], []):
            raw = dict(self.raw)
            raw["dwpc_damping_grid"] = grid
            with self.assertRaises(ValueError):
                self._load(raw)

    def test_role_weights_validated(self):
        raw = dict(self.raw)
        raw["category_role_weights"] = {"primary": 0.4, "secondary": 0.5}
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_unknown_metric_rejected(self):
        raw = dict(self.raw)
        raw["tuning_primary_metric"] = "mrr"
        with self.assertRaises(ValueError):
            self._load(raw)


if __name__ == "__main__":
    unittest.main()
