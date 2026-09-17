"""Unit tests for src/data/sample.py using a small synthetic eligible frame."""
import unittest
from dataclasses import replace

import pandas as pd

from src.data.config import load_config
from src.data.sample import background_corpus, sample_requests


def make_eligible_frame(n_a: int, n_b: int) -> pd.DataFrame:
    rows = []
    for i in range(n_a):
        rows.append({"link": f"a-{i:03d}", "category": "A", "eligible_for_model": True})
    for i in range(n_b):
        rows.append({"link": f"b-{i:03d}", "category": "B", "eligible_for_model": True})
    return pd.DataFrame(rows)


class TestSampleRequests(unittest.TestCase):
    def test_sample_size_matches_config_exactly(self):
        cfg = replace(load_config(), model_requests=10)
        df = make_eligible_frame(60, 40)
        sampled = sample_requests(df, cfg)
        self.assertEqual(int(sampled["is_sampled_request"].sum()), 10)

    def test_sampling_is_deterministic_given_same_seed(self):
        cfg = replace(load_config(), model_requests=10)
        df = make_eligible_frame(60, 40)
        a = sample_requests(df, cfg)["is_sampled_request"]
        b = sample_requests(df, cfg)["is_sampled_request"]
        self.assertTrue((a.values == b.values).all())

    def test_different_seed_can_change_the_sample(self):
        cfg_a = replace(load_config(), model_requests=10, seed=42)
        cfg_b = replace(load_config(), model_requests=10, seed=43)
        df = make_eligible_frame(60, 40)
        a = sample_requests(df, cfg_a)["is_sampled_request"]
        b = sample_requests(df, cfg_b)["is_sampled_request"]
        self.assertFalse((a.values == b.values).all())

    def test_sample_and_background_are_disjoint_and_cover_eligible_population(self):
        cfg = replace(load_config(), model_requests=10)
        df = make_eligible_frame(60, 40)
        sampled = sample_requests(df, cfg)
        background = background_corpus(sampled)
        sampled_links = set(sampled.loc[sampled["is_sampled_request"], "link"])
        background_links = set(background["link"])
        self.assertTrue(sampled_links.isdisjoint(background_links))
        self.assertEqual(len(sampled_links) + len(background_links), int(df["eligible_for_model"].sum()))

    def test_ineligible_rows_are_never_sampled(self):
        cfg = replace(load_config(), model_requests=5)
        df = make_eligible_frame(10, 10)
        df.loc[0, "eligible_for_model"] = False
        sampled = sample_requests(df, cfg)
        self.assertFalse(bool(sampled.loc[0, "is_sampled_request"]))


if __name__ == "__main__":
    unittest.main()
