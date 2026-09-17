"""End-to-end invariant checks against the real generated Day 3 synthetic
dataset (data/processed/synthetic + synthetic_private). Skips cleanly if
those outputs don't exist -- run scripts/generate_synthetic_data.py
first. Mirrors tests/test_integration.py's pattern from Day 2.
"""
import unittest
from pathlib import Path

import pandas as pd

from src.data.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"
PRIVATE_DIR = REPO_ROOT / "data" / "processed" / "synthetic_private"
DAY2_DIR = REPO_ROOT / "data" / "processed"


@unittest.skipUnless(SYNTHETIC_DIR.exists(), f"{SYNTHETIC_DIR} not found; run scripts/generate_synthetic_data.py first")
class TestSyntheticDatasetInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.clients = pd.read_csv(SYNTHETIC_DIR / "clients.csv")
        cls.workers = pd.read_csv(SYNTHETIC_DIR / "workers.csv")
        cls.requests = pd.read_csv(SYNTHETIC_DIR / "requests.csv")
        cls.interactions = pd.read_csv(SYNTHETIC_DIR / "interactions.csv")
        cls.ratings = pd.read_csv(SYNTHETIC_DIR / "ratings.csv")
        cls.cold_start = pd.read_csv(SYNTHETIC_DIR / "cold_start_cohort.csv")
        cls.sampled_requests = pd.read_csv(DAY2_DIR / "sampled_requests.csv")
        cls.vocab = set(pd.read_csv(DAY2_DIR / "skills_vocab.csv")["skill"])

    def test_population_counts(self):
        self.assertEqual(len(self.clients), self.config.synthetic_clients)
        self.assertEqual(len(self.workers), self.config.synthetic_workers)
        self.assertEqual(len(self.requests), self.config.model_requests)
        self.assertEqual(len(self.cold_start), round(self.config.synthetic_workers * self.config.cold_start_ratio))

    def test_ids_are_unique_and_stably_formatted(self):
        self.assertTrue(self.clients["client_id"].is_unique)
        self.assertTrue(self.workers["worker_id"].is_unique)
        self.assertTrue(self.clients["client_id"].str.match(r"^C\d{6}$").all())
        self.assertTrue(self.workers["worker_id"].str.match(r"^W\d{6}$").all())

    def test_no_orphan_foreign_keys(self):
        self.assertTrue(self.interactions["client_id"].isin(self.clients["client_id"]).all())
        self.assertTrue(self.interactions["worker_id"].isin(self.workers["worker_id"]).all())
        self.assertTrue(self.interactions["link"].isin(self.requests["link"]).all())
        self.assertTrue(self.ratings["worker_id"].isin(self.workers["worker_id"]).all())
        self.assertTrue(self.ratings["link"].isin(self.requests["link"]).all())

    def test_requests_map_to_day2_sampled_requests(self):
        self.assertEqual(set(self.requests["link"]), set(self.sampled_requests["link"]))

    def test_worker_declared_skills_belong_to_controlled_vocabulary(self):
        for cell in self.workers["declared_skills"].dropna():
            skills = cell.split("|") if cell else []
            self.assertTrue(set(skills).issubset(self.vocab), f"out-of-vocab skill in {skills}")

    def test_cold_start_workers_have_no_training_interactions(self):
        cold_ids = set(self.cold_start["worker_id"])
        train_links = set(self.requests.loc[self.requests["split"] == "train", "link"])
        train_interactions = self.interactions.loc[self.interactions["link"].isin(train_links)]
        self.assertEqual(train_interactions["worker_id"].isin(cold_ids).sum(), 0)

    def test_temporal_ordering_is_valid(self):
        merged = self.requests.set_index("link")
        train_max = merged.loc[merged["split"] == "train", "simulated_day"].max()
        val_min = merged.loc[merged["split"] == "validation", "simulated_day"].min()
        val_max = merged.loc[merged["split"] == "validation", "simulated_day"].max()
        test_min = merged.loc[merged["split"] == "test", "simulated_day"].min()
        self.assertLessEqual(train_max, val_min)
        self.assertLessEqual(val_max, test_min)

    def test_splits_are_non_overlapping_and_exhaustive(self):
        self.assertEqual(set(self.requests["split"].unique()), {"train", "validation", "test"})
        self.assertEqual(self.requests["link"].nunique(), len(self.requests))

    def test_no_worker_interacts_before_its_join_day(self):
        join_day = self.workers.set_index("worker_id")["join_day"]
        merged = self.interactions.copy()
        merged["join_day"] = merged["worker_id"].map(join_day)
        self.assertTrue((merged["join_day"] <= merged["simulated_day"]).all())

    def test_rating_bounds_are_valid(self):
        self.assertTrue(self.ratings["rating"].between(1, 5).all())

    def test_ratings_only_exist_for_hired_interactions(self):
        hired = set(zip(
            self.interactions.loc[self.interactions["event"] == "HIRED", "link"],
            self.interactions.loc[self.interactions["event"] == "HIRED", "worker_id"],
        ))
        rating_pairs = set(zip(self.ratings["link"], self.ratings["worker_id"]))
        self.assertTrue(rating_pairs.issubset(hired))

    def test_at_most_one_hire_per_request(self):
        hires = self.interactions.loc[self.interactions["event"] == "HIRED"]
        self.assertTrue((hires.groupby("link").size() <= 1).all())

    def test_worker_capacity_not_exceeded_in_any_30_day_window(self):
        hires = self.interactions.loc[self.interactions["event"] == "HIRED"]
        for worker_id, group in hires.groupby("worker_id"):
            days = sorted(group["simulated_day"])
            for i in range(len(days)):
                window_count = sum(1 for d in days if days[i] - 30 < d <= days[i])
                self.assertLessEqual(window_count, 4, f"{worker_id} exceeds capacity around day {days[i]}")

    def test_no_ineligible_hire_for_location_restricted_requests(self):
        restricted = self.sampled_requests.loc[
            self.sampled_requests["location_requirement"].notna(), ["link", "location_requirement"]
        ]
        if restricted.empty:
            self.skipTest("no location-restricted requests in this sample")
        worker_country = self.workers.set_index("worker_id")["country"]
        hires = self.interactions.loc[self.interactions["event"] == "HIRED"].merge(restricted, on="link", how="inner")
        for row in hires.itertuples():
            country = worker_country[row.worker_id]
            self.assertIn(country, row.location_requirement)


if __name__ == "__main__":
    unittest.main()
