"""config/hybrid_config.yaml loading and validation (Day 8)."""
import copy
import tempfile
import unittest
from pathlib import Path

import yaml

from src.recommenders.hybrid_config import DEFAULT_HYBRID_CONFIG_PATH, balance, load_hybrid_config


class TestHybridConfig(unittest.TestCase):
    def setUp(self):
        with open(DEFAULT_HYBRID_CONFIG_PATH, encoding="utf-8") as fh:
            self.raw = yaml.safe_load(fh)

    def _load(self, raw):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "h.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            return load_hybrid_config(path)

    def test_project_config(self):
        cfg = load_hybrid_config()
        self.assertEqual(cfg.components, ["cbf", "cf", "kg"])
        self.assertEqual(cfg.normalization, "minmax_within_candidate_pool")
        self.assertEqual(len(cfg.weight_grid), 12)
        self.assertEqual(cfg.bootstrap["n_resamples"], 1000)
        self.assertEqual(set(cfg.bootstrap["baselines"]), {"kg", "cbf", "cf", "popularity"})
        for entry in cfg.weight_grid:
            self.assertAlmostEqual(sum(entry.values()), 1.0)
            self.assertGreaterEqual(sum(1 for v in entry.values() if v > 0), 2)

    def test_single_component_entry_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["weight_grid"].append({"cbf": 1.0, "cf": 0.0, "kg": 0.0})
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_bad_sum_and_duplicates_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw["weight_grid"] = [{"cbf": 0.5, "cf": 0.5, "kg": 0.5}]
        with self.assertRaises(ValueError):
            self._load(raw)
        raw = copy.deepcopy(self.raw)
        raw["weight_grid"] = raw["weight_grid"][:1] * 2
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_bootstrap_validated(self):
        raw = copy.deepcopy(self.raw)
        raw["bootstrap"]["baselines"] = ["oracle"]
        with self.assertRaises(ValueError):
            self._load(raw)

    def test_balance(self):
        self.assertAlmostEqual(balance({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), 0.0)
        self.assertGreater(balance({"a": 0.6, "b": 0.2, "c": 0.2}), balance({"a": 0.5, "b": 0.25, "c": 0.25}))


if __name__ == "__main__":
    unittest.main()
