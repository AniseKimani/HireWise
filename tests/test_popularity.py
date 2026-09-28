import unittest

import pandas as pd

from src.recommenders.popularity import PopularityBaseline


def make_data(interactions_rows, requests_rows):
    interactions = pd.DataFrame(interactions_rows)
    requests = pd.DataFrame(requests_rows)
    return interactions, requests


class TestPopularityBaseline(unittest.TestCase):
    def test_ranks_by_training_hire_count_descending(self):
        interactions, requests = make_data(
            [
                {"link": "r1", "worker_id": "W1", "event": "HIRED"},
                {"link": "r2", "worker_id": "W1", "event": "HIRED"},
                {"link": "r3", "worker_id": "W2", "event": "HIRED"},
            ],
            [
                {"link": "r1", "split": "train"},
                {"link": "r2", "split": "train"},
                {"link": "r3", "split": "train"},
            ],
        )
        rec = PopularityBaseline(interactions, requests)
        ranked = rec.rank("any", ["W1", "W2", "W3"], k=10)
        self.assertEqual(ranked, ["W1", "W2", "W3"])  # W1:2 hires, W2:1, W3:0

    def test_validation_and_test_interactions_are_ignored(self):
        interactions, requests = make_data(
            [
                {"link": "r1", "worker_id": "W1", "event": "HIRED"},   # train
                {"link": "r2", "worker_id": "W2", "event": "HIRED"},   # validation
                {"link": "r3", "worker_id": "W3", "event": "HIRED"},   # test
            ],
            [
                {"link": "r1", "split": "train"},
                {"link": "r2", "split": "validation"},
                {"link": "r3", "split": "test"},
            ],
        )
        rec = PopularityBaseline(interactions, requests)
        self.assertEqual(rec.hire_counts, {"W1": 1})

    def test_shortlisted_not_hired_does_not_count(self):
        interactions, requests = make_data(
            [
                {"link": "r1", "worker_id": "W1", "event": "SHORTLISTED"},
                {"link": "r1", "worker_id": "W2", "event": "HIRED"},
            ],
            [{"link": "r1", "split": "train"}],
        )
        rec = PopularityBaseline(interactions, requests)
        self.assertEqual(rec.hire_counts, {"W2": 1})

    def test_deterministic_tie_break_by_worker_id(self):
        interactions = pd.DataFrame(columns=["link", "worker_id", "event"])
        requests = pd.DataFrame(columns=["link", "split"])
        rec = PopularityBaseline(interactions, requests)
        ranked = rec.rank("any", ["W3", "W1", "W2"], k=10)
        self.assertEqual(ranked, ["W1", "W2", "W3"])  # all tied at 0 -> ascending worker_id

    def test_cold_start_worker_with_no_training_history_gets_no_boost(self):
        interactions, requests = make_data(
            [{"link": "r1", "worker_id": "W1", "event": "HIRED"}],
            [{"link": "r1", "split": "train"}],
        )
        rec = PopularityBaseline(interactions, requests)
        # W_COLD never appears in training interactions at all.
        self.assertNotIn("W_COLD", rec.hire_counts)
        ranked = rec.rank("any", ["W_COLD", "W1"], k=10)
        self.assertEqual(ranked, ["W1", "W_COLD"])

    def test_no_duplicates_and_respects_k(self):
        interactions = pd.DataFrame(columns=["link", "worker_id", "event"])
        requests = pd.DataFrame(columns=["link", "split"])
        rec = PopularityBaseline(interactions, requests)
        candidates = [f"W{i}" for i in range(20)]
        ranked = rec.rank("any", candidates, k=10)
        self.assertEqual(len(ranked), 10)
        self.assertEqual(len(ranked), len(set(ranked)))


if __name__ == "__main__":
    unittest.main()
