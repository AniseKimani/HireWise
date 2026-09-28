"""Unit tests for src/kg/prepare.py using small synthetic fixtures."""
import unittest

import pandas as pd

from src.kg.prepare import (
    normalize_location,
    prepare_category_nodes,
    prepare_client_nodes,
    prepare_has_skill_edges,
    prepare_hired_edges,
    prepare_interested_in_edges,
    prepare_located_in_edges,
    prepare_requested_service_edges,
    prepare_reviewed_edges,
    prepare_skill_nodes,
    prepare_specializes_in_edges,
    prepare_worker_nodes,
)


class TestNormalizeLocation(unittest.TestCase):
    def test_strips_and_collapses_whitespace(self):
        self.assertEqual(normalize_location("  United   States  "), "United States")

    def test_decodes_html_entities(self):
        self.assertEqual(normalize_location("C&amp;ote d&#39;Ivoire"), "C&ote d'Ivoire")

    def test_none_for_missing(self):
        self.assertIsNone(normalize_location(None))
        self.assertIsNone(normalize_location(float("nan")))

    def test_none_for_blank(self):
        self.assertIsNone(normalize_location("   "))

    def test_deterministic(self):
        self.assertEqual(normalize_location("United States"), normalize_location("United States"))

    def test_case_and_whitespace_variants_collapse_to_the_same_key(self):
        # Not literal case-folding (country strings are already
        # platform-controlled), but whitespace variants must not create
        # duplicate keys.
        self.assertEqual(normalize_location("United  States"), normalize_location("United States"))


class TestNodePreparation(unittest.TestCase):
    def test_client_nodes_deterministic_ids_and_columns(self):
        clients = pd.DataFrame([
            {"client_id": "C002", "preferred_pay_type": "fixed", "median_price_tier": 0.5, "country": "X"},
            {"client_id": "C001", "preferred_pay_type": "hourly", "median_price_tier": 0.3, "country": "Y"},
        ])
        out = prepare_client_nodes(clients)
        self.assertEqual(list(out["client_id"]), ["C001", "C002"])  # sorted
        self.assertEqual(set(out.columns), {"client_id", "preferred_pay_type", "median_price_tier"})

    def test_worker_nodes_exclude_private_columns_even_if_present_upstream(self):
        workers = pd.DataFrame([{
            "worker_id": "W1", "experience_level": "entry", "experience_years": 1.0,
            "price_tier": 0.5, "rate": 10.0, "join_day": 0.0, "is_cold_start": False,
            "quality": 0.9, "true_skills": "Python", "declared_skills": "Python",
            "primary_category": "Writing", "secondary_categories": "",
        }])
        out = prepare_worker_nodes(workers)
        self.assertNotIn("quality", out.columns)
        self.assertNotIn("true_skills", out.columns)
        self.assertNotIn("declared_skills", out.columns)

    def test_skill_nodes_include_full_vocabulary_not_only_connected_ones(self):
        vocab = pd.DataFrame({"skill": ["Python", "Django", "Welding"], "frequency": [10, 10, 10]})
        out = prepare_skill_nodes(vocab)
        self.assertEqual(set(out["name"]), {"Python", "Django", "Welding"})

    def test_category_nodes_sorted_and_unique(self):
        category_stats = pd.DataFrame({"category": ["Writing", "Design", "Writing"]})
        out = prepare_category_nodes(category_stats)
        self.assertEqual(list(out["name"]), ["Design", "Writing"])


class TestRelationshipPreparation(unittest.TestCase):
    def test_has_skill_only_vocabulary_skills(self):
        workers = pd.DataFrame([{"worker_id": "W1", "declared_skills": "Python|ObscureSkill"}])
        out = prepare_has_skill_edges(workers, vocab_skills={"Python"})
        self.assertEqual(list(out["skill"]), ["Python"])

    def test_has_skill_deduplicates(self):
        workers = pd.DataFrame([{"worker_id": "W1", "declared_skills": "Python|Python"}])
        out = prepare_has_skill_edges(workers, vocab_skills={"Python"})
        self.assertEqual(len(out), 1)

    def test_specializes_in_primary_and_secondary_roles(self):
        workers = pd.DataFrame([{"worker_id": "W1", "primary_category": "Writing", "secondary_categories": "Design|Marketing"}])
        out = prepare_specializes_in_edges(workers, category_names={"Writing", "Design", "Marketing"})
        roles = dict(zip(out["category"], out["role"]))
        self.assertEqual(roles["Writing"], "primary")
        self.assertEqual(roles["Design"], "secondary")
        self.assertEqual(roles["Marketing"], "secondary")

    def test_specializes_in_ignores_out_of_scope_category(self):
        workers = pd.DataFrame([{"worker_id": "W1", "primary_category": "Writing", "secondary_categories": "OutOfScope"}])
        out = prepare_specializes_in_edges(workers, category_names={"Writing"})
        self.assertEqual(list(out["category"]), ["Writing"])

    def test_located_in_drops_missing_location(self):
        entities = pd.DataFrame([{"worker_id": "W1", "country": "France"}, {"worker_id": "W2", "country": None}])
        out = prepare_located_in_edges(entities, "worker_id")
        self.assertEqual(list(out["worker_id"]), ["W1"])

    def test_requested_service_carries_temporal_properties(self):
        requests = pd.DataFrame([{"link": "r1", "client_id": "C1", "simulated_day": 5.0, "split": "train"}])
        sampled = pd.DataFrame([{"link": "r1", "category": "Writing"}])
        out = prepare_requested_service_edges(requests, sampled)
        row = out.iloc[0]
        self.assertEqual(row["client_id"], "C1")
        self.assertEqual(row["category"], "Writing")
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["simulated_day"], 5.0)

    def test_requested_service_one_edge_per_request_not_merged(self):
        # Same client, same category, two different requests -> two rows.
        requests = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "simulated_day": 1.0, "split": "train"},
            {"link": "r2", "client_id": "C1", "simulated_day": 2.0, "split": "train"},
        ])
        sampled = pd.DataFrame([{"link": "r1", "category": "Writing"}, {"link": "r2", "category": "Writing"}])
        out = prepare_requested_service_edges(requests, sampled)
        self.assertEqual(len(out), 2)
        self.assertEqual(out["link"].nunique(), 2)

    def test_interested_in_requires_at_least_two_training_requests(self):
        requested_service = pd.DataFrame([
            {"client_id": "C1", "category": "Writing", "link": "r1", "simulated_day": 1.0, "split": "train"},
            {"client_id": "C1", "category": "Writing", "link": "r2", "simulated_day": 2.0, "split": "train"},
            {"client_id": "C2", "category": "Design", "link": "r3", "simulated_day": 1.0, "split": "train"},
        ])
        out = prepare_interested_in_edges(requested_service)
        self.assertEqual(list(zip(out["client_id"], out["category"])), [("C1", "Writing")])

    def test_interested_in_ignores_validation_and_test_requests(self):
        requested_service = pd.DataFrame([
            {"client_id": "C1", "category": "Writing", "link": "r1", "simulated_day": 1.0, "split": "validation"},
            {"client_id": "C1", "category": "Writing", "link": "r2", "simulated_day": 2.0, "split": "test"},
        ])
        out = prepare_interested_in_edges(requested_service)
        self.assertEqual(len(out), 0)

    def test_hired_edges_carry_link_and_split(self):
        interactions = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "worker_id": "W1", "event": "HIRED", "simulated_day": 1.0},
            {"link": "r1", "client_id": "C1", "worker_id": "W2", "event": "SHORTLISTED", "simulated_day": 1.0},
        ])
        requests = pd.DataFrame([{"link": "r1", "split": "train"}])
        out = prepare_hired_edges(interactions, requests)
        self.assertEqual(len(out), 1)  # SHORTLISTED excluded
        self.assertEqual(out.iloc[0]["worker_id"], "W1")
        self.assertEqual(out.iloc[0]["split"], "train")

    def test_reviewed_edges_carry_rating_and_has_review(self):
        ratings = pd.DataFrame([{"link": "r1", "worker_id": "W1", "rating": 5, "has_review": True}])
        requests = pd.DataFrame([{"link": "r1", "client_id": "C1"}])
        out = prepare_reviewed_edges(ratings, requests)
        row = out.iloc[0]
        self.assertEqual(row["rating"], 5)
        self.assertTrue(row["has_review"])
        self.assertEqual(row["client_id"], "C1")


if __name__ == "__main__":
    unittest.main()
