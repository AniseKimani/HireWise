import unittest
from dataclasses import replace

import pandas as pd

from src.data.config import load_config
from src.data.generator_config import load_generator_config
from src.data.synth_timeline import assign_timeline, compute_temporal_split

GEN_CFG = load_generator_config("config/generator_v1.yaml")


class TestAssignTimeline(unittest.TestCase):
    def test_day_within_client_window(self):
        cfg = load_config()
        clients_df = pd.DataFrame({"client_id": ["C1", "C2"], "activity_start_day": [0.0, 300.0]})
        assignment_df = pd.DataFrame({"link": ["a", "b", "c"], "client_id": ["C1", "C1", "C2"]})
        result = assign_timeline(assignment_df, clients_df, cfg, GEN_CFG)
        self.assertTrue((result.loc[result["client_id"] == "C2", "simulated_day"] >= 300.0).all())
        self.assertTrue((result["simulated_day"] <= GEN_CFG.simulated_days).all())

    def test_deterministic(self):
        cfg = load_config()
        clients_df = pd.DataFrame({"client_id": ["C1", "C2"], "activity_start_day": [0.0, 300.0]})
        assignment_df = pd.DataFrame({"link": ["a", "b", "c"], "client_id": ["C1", "C1", "C2"]})
        r1 = assign_timeline(assignment_df, clients_df, cfg, GEN_CFG)
        r2 = assign_timeline(assignment_df, clients_df, cfg, GEN_CFG)
        self.assertTrue(r1.equals(r2))


class TestComputeTemporalSplit(unittest.TestCase):
    def test_exact_counts_at_configured_ratios(self):
        cfg = replace(load_config(), train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2)
        n = 1000
        timeline_df = pd.DataFrame({"link": [f"r{i}" for i in range(n)], "client_id": "C1", "simulated_day": range(n)})
        split_df, boundaries = compute_temporal_split(timeline_df, cfg)
        counts = split_df["split"].value_counts()
        self.assertEqual(counts["train"], 700)
        self.assertEqual(counts["validation"], 100)
        self.assertEqual(counts["test"], 200)
        self.assertEqual(boundaries["n_train"], 700)

    def test_no_future_leaks_into_earlier_split(self):
        cfg = replace(load_config(), train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2)
        n = 500
        timeline_df = pd.DataFrame({"link": [f"r{i}" for i in range(n)], "client_id": "C1", "simulated_day": range(n)})
        split_df, _ = compute_temporal_split(timeline_df, cfg)
        merged = split_df.set_index("link")
        train_max = merged.loc[merged["split"] == "train", "simulated_day"].max()
        val_min = merged.loc[merged["split"] == "validation", "simulated_day"].min()
        val_max = merged.loc[merged["split"] == "validation", "simulated_day"].max()
        test_min = merged.loc[merged["split"] == "test", "simulated_day"].min()
        self.assertLessEqual(train_max, val_min)
        self.assertLessEqual(val_max, test_min)

    def test_splits_are_non_overlapping_and_cover_all_requests(self):
        cfg = replace(load_config(), train_ratio=0.7, validation_ratio=0.1, test_ratio=0.2)
        n = 300
        timeline_df = pd.DataFrame({"link": [f"r{i}" for i in range(n)], "client_id": "C1", "simulated_day": range(n)})
        split_df, _ = compute_temporal_split(timeline_df, cfg)
        self.assertEqual(set(split_df["split"].unique()), {"train", "validation", "test"})
        self.assertEqual(len(split_df), n)
        self.assertEqual(split_df["link"].nunique(), n)


if __name__ == "__main__":
    unittest.main()
