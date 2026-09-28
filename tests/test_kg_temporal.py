"""Temporal safety tests for src/kg/prepare.py (Day 6 brief Section 5):
train/validation/test properties preserved on historical edges,
cold-start workers never get a training HIRED edge, and INTERESTED_IN
can never become timeless/leak validation or test knowledge.
"""
import unittest

import pandas as pd

from src.kg.prepare import prepare_hired_edges, prepare_interested_in_edges, prepare_requested_service_edges


class TestHistoricalEdgesPreserveSplit(unittest.TestCase):
    def test_hired_edges_carry_the_correct_split_per_request(self):
        interactions = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "worker_id": "W1", "event": "HIRED", "simulated_day": 1.0},
            {"link": "r2", "client_id": "C1", "worker_id": "W2", "event": "HIRED", "simulated_day": 300.0},
        ])
        requests = pd.DataFrame([
            {"link": "r1", "split": "train"},
            {"link": "r2", "split": "test"},
        ])
        out = prepare_hired_edges(interactions, requests)
        split_by_link = dict(zip(out["link"], out["split"]))
        self.assertEqual(split_by_link, {"r1": "train", "r2": "test"})

    def test_requested_service_edges_carry_the_correct_split(self):
        requests = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "simulated_day": 1.0, "split": "train"},
            {"link": "r2", "client_id": "C1", "simulated_day": 300.0, "split": "validation"},
        ])
        sampled = pd.DataFrame([{"link": "r1", "category": "Writing"}, {"link": "r2", "category": "Writing"}])
        out = prepare_requested_service_edges(requests, sampled)
        split_by_link = dict(zip(out["link"], out["split"]))
        self.assertEqual(split_by_link, {"r1": "train", "r2": "validation"})


class TestColdStartNoTrainingHireEdge(unittest.TestCase):
    def test_cold_start_worker_join_day_after_training_window_yields_no_training_hired_edge(self):
        # This mirrors Day 3's actual mechanism: a cold-start worker's
        # join_day is set to the validation boundary, so no training
        # request could ever have hired them -- prepare_hired_edges just
        # passes through whatever split the source data recorded, and
        # this test documents that a cold-start worker with (by
        # construction) no training-split HIRED interaction produces no
        # training HIRED edge.
        interactions = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "worker_id": "W_COLD", "event": "HIRED", "simulated_day": 330.0},
        ])
        requests = pd.DataFrame([{"link": "r1", "split": "validation"}])
        out = prepare_hired_edges(interactions, requests)
        self.assertTrue((out["split"] != "train").all())


class TestInterestedInCannotLeakValidationOrTest(unittest.TestCase):
    def test_validation_and_test_requests_never_contribute_to_interested_in(self):
        requested_service = pd.DataFrame([
            {"client_id": "C1", "category": "Writing", "link": "r1", "simulated_day": 1.0, "split": "validation"},
            {"client_id": "C1", "category": "Writing", "link": "r2", "simulated_day": 2.0, "split": "validation"},
            {"client_id": "C1", "category": "Writing", "link": "r3", "simulated_day": 3.0, "split": "test"},
            {"client_id": "C1", "category": "Writing", "link": "r4", "simulated_day": 4.0, "split": "test"},
        ])
        # 4 total requests in the category (would clear the >=2
        # threshold if split were ignored), but zero of them are training.
        out = prepare_interested_in_edges(requested_service)
        self.assertEqual(len(out), 0)

    def test_mixed_split_only_counts_training_requests(self):
        requested_service = pd.DataFrame([
            {"client_id": "C1", "category": "Writing", "link": "r1", "simulated_day": 1.0, "split": "train"},
            {"client_id": "C1", "category": "Writing", "link": "r2", "simulated_day": 2.0, "split": "validation"},
            {"client_id": "C1", "category": "Writing", "link": "r3", "simulated_day": 3.0, "split": "test"},
        ])
        # Only 1 training request -- below the >=2 threshold, even though
        # 3 total requests exist across all splits.
        out = prepare_interested_in_edges(requested_service)
        self.assertEqual(len(out), 0)

    def test_adding_a_validation_request_never_creates_or_removes_an_interested_in_edge(self):
        base = pd.DataFrame([
            {"client_id": "C1", "category": "Writing", "link": "r1", "simulated_day": 1.0, "split": "train"},
            {"client_id": "C1", "category": "Writing", "link": "r2", "simulated_day": 2.0, "split": "train"},
        ])
        with_extra_validation = pd.concat([base, pd.DataFrame([
            {"client_id": "C1", "category": "Writing", "link": "r3", "simulated_day": 300.0, "split": "validation"},
        ])], ignore_index=True)

        out_base = prepare_interested_in_edges(base)
        out_extra = prepare_interested_in_edges(with_extra_validation)
        self.assertTrue(out_base.equals(out_extra))


if __name__ == "__main__":
    unittest.main()
