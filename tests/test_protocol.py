import unittest

import pandas as pd

from src.evaluation.config import load_evaluation_config
from src.evaluation.protocol import (
    EligibilityIndex,
    ProtocolData,
    build_eval_requests,
    extract_required_countries,
    relevance_for_link,
)

EVAL_CFG = load_evaluation_config()


def make_workers(rows):
    defaults = {
        "worker_id": None, "primary_category": "Writing", "secondary_categories": "",
        "join_day": 0.0, "country": "United States",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestExtractRequiredCountries(unittest.TestCase):
    def test_matches_known_country(self):
        text = "Only freelancers located in the United States may apply."
        self.assertEqual(extract_required_countries(text, ["United States", "United Kingdom"]), ["United States"])

    def test_no_restriction_returns_empty(self):
        self.assertEqual(extract_required_countries(None, ["United States"]), [])
        self.assertEqual(extract_required_countries(float("nan"), ["United States"]), [])


class TestEligibilityIndex(unittest.TestCase):
    def test_primary_and_secondary_category_included(self):
        workers = make_workers([
            {"worker_id": "W_primary", "primary_category": "Writing"},
            {"worker_id": "W_secondary", "primary_category": "Design", "secondary_categories": "Writing"},
            {"worker_id": "W_unrelated", "primary_category": "Design"},
        ])
        index = EligibilityIndex(workers, ["United States"])
        eligible = index.eligible_worker_ids("Writing", day=100.0, location_requirement=None)
        self.assertEqual(set(eligible), {"W_primary", "W_secondary"})

    def test_temporal_eligibility_excludes_workers_who_joined_after_request(self):
        workers = make_workers([
            {"worker_id": "W_early", "join_day": 0.0},
            {"worker_id": "W_late", "join_day": 200.0},
        ])
        index = EligibilityIndex(workers, ["United States"])
        eligible = index.eligible_worker_ids("Writing", day=100.0, location_requirement=None)
        self.assertEqual(eligible, ["W_early"])

    def test_worker_joining_exactly_on_request_day_is_eligible(self):
        workers = make_workers([{"worker_id": "W1", "join_day": 100.0}])
        index = EligibilityIndex(workers, ["United States"])
        eligible = index.eligible_worker_ids("Writing", day=100.0, location_requirement=None)
        self.assertEqual(eligible, ["W1"])

    def test_location_restriction_filters_by_country(self):
        workers = make_workers([
            {"worker_id": "W_us", "country": "United States"},
            {"worker_id": "W_uk", "country": "United Kingdom"},
        ])
        index = EligibilityIndex(workers, ["United States", "United Kingdom"])
        eligible = index.eligible_worker_ids(
            "Writing", day=100.0, location_requirement="Only freelancers located in the United Kingdom may apply."
        )
        self.assertEqual(eligible, ["W_uk"])

    def test_no_location_restriction_includes_all_countries(self):
        workers = make_workers([
            {"worker_id": "W_us", "country": "United States"},
            {"worker_id": "W_uk", "country": "United Kingdom"},
        ])
        index = EligibilityIndex(workers, ["United States", "United Kingdom"])
        eligible = index.eligible_worker_ids("Writing", day=100.0, location_requirement=None)
        self.assertEqual(set(eligible), {"W_us", "W_uk"})

    def test_no_matching_category_returns_empty(self):
        workers = make_workers([{"worker_id": "W1", "primary_category": "Design"}])
        index = EligibilityIndex(workers, ["United States"])
        self.assertEqual(index.eligible_worker_ids("Writing", day=100.0, location_requirement=None), [])

    def test_result_is_sorted_deterministic(self):
        workers = make_workers([{"worker_id": w} for w in ["W3", "W1", "W2"]])
        index = EligibilityIndex(workers, ["United States"])
        eligible = index.eligible_worker_ids("Writing", day=100.0, location_requirement=None)
        self.assertEqual(eligible, ["W1", "W2", "W3"])


class TestRelevanceForLink(unittest.TestCase):
    def test_hired_and_shortlisted_grades(self):
        interactions = pd.DataFrame([
            {"link": "r1", "worker_id": "W1", "event": "HIRED"},
            {"link": "r1", "worker_id": "W2", "event": "SHORTLISTED"},
        ])
        relevance = relevance_for_link(interactions, EVAL_CFG)
        self.assertEqual(relevance["r1"], {"W1": 2, "W2": 1})

    def test_no_future_leakage_only_uses_given_interactions(self):
        # relevance_for_link has no notion of "future" itself -- the
        # caller (build_eval_requests) must pass only the appropriate
        # interactions. This test documents that relevance_for_link is a
        # pure function of its input, so leakage prevention is entirely
        # the caller's responsibility (tested in TestBuildEvalRequests).
        interactions = pd.DataFrame([{"link": "r1", "worker_id": "W1", "event": "HIRED"}])
        relevance = relevance_for_link(interactions, EVAL_CFG)
        self.assertEqual(set(relevance.keys()), {"r1"})


class TestBuildEvalRequests(unittest.TestCase):
    def _protocol_data(self):
        workers = make_workers([{"worker_id": f"W{i}"} for i in range(30)])
        requests = pd.DataFrame([
            {"link": "r_train", "client_id": "C1", "simulated_day": 10.0, "split": "train",
             "category": "Writing", "skills_vocab": "Python", "price_tier": 0.5,
             "pay_type": "fixed", "location_requirement": None},
            {"link": "r_val_small_pool", "client_id": "C1", "simulated_day": 300.0, "split": "validation",
             "category": "Writing", "skills_vocab": "Python", "price_tier": 0.5,
             "pay_type": "fixed", "location_requirement": "Only freelancers located in the United Kingdom may apply."},
        ])
        interactions = pd.DataFrame([
            {"link": "r_train", "worker_id": "W1", "event": "HIRED"},
            {"link": "r_val_small_pool", "worker_id": "W2", "event": "HIRED"},
        ])
        return ProtocolData(workers=workers, requests=requests, interactions=interactions, known_countries=["United States", "United Kingdom"])

    def test_excludes_requests_with_pool_below_minimum(self):
        data = self._protocol_data()
        index = EligibilityIndex(data.workers, data.known_countries)
        reqs = build_eval_requests(data, index, "validation", EVAL_CFG)
        self.assertEqual(len(reqs), 1)
        self.assertTrue(reqs[0].excluded)
        self.assertEqual(reqs[0].excluded_reason, "pool_too_small")  # UK-only pool is empty (all workers are US)

    def test_included_request_has_correct_relevance_and_candidates(self):
        data = self._protocol_data()
        # Make the training request's pool large enough to be included.
        data.workers.loc[:, "country"] = "United States"
        index = EligibilityIndex(data.workers, data.known_countries)
        reqs = build_eval_requests(data, index, "train", EVAL_CFG)
        self.assertEqual(len(reqs), 1)
        req = reqs[0]
        self.assertFalse(req.excluded)
        self.assertEqual(req.relevance, {"W1": 2})
        self.assertIn("W1", req.candidates)

    def test_no_relevant_workers_reason_when_relevant_worker_outside_pool(self):
        workers = make_workers([{"worker_id": "W_outside", "primary_category": "Design"}] +
                                [{"worker_id": f"W{i}"} for i in range(25)])
        requests = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "simulated_day": 10.0, "split": "train",
             "category": "Writing", "skills_vocab": "Python", "price_tier": 0.5,
             "pay_type": "fixed", "location_requirement": None},
        ])
        # The only historically relevant worker has a different category,
        # so it is not in the reconstructed candidate universe.
        interactions = pd.DataFrame([{"link": "r1", "worker_id": "W_outside", "event": "HIRED"}])
        data = ProtocolData(workers=workers, requests=requests, interactions=interactions, known_countries=["United States"])
        index = EligibilityIndex(data.workers, data.known_countries)
        reqs = build_eval_requests(data, index, "train", EVAL_CFG)
        self.assertEqual(len(reqs), 1)
        self.assertTrue(reqs[0].excluded)
        self.assertEqual(reqs[0].excluded_reason, "no_relevant_workers")


if __name__ == "__main__":
    unittest.main()
