import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from src.data.config import load_config
from src.data.generator_config import load_generator_config
from src.data.synth_interactions import (
    build_neighbor_indices,
    build_proficiency_matrix,
    build_skill_relatedness,
    CandidateIndex,
    compute_skill_fit,
    estimate_utility_moments,
    simulate,
)

GEN_CFG = load_generator_config("config/generator_v1.yaml")


class TestSkillFit(unittest.TestCase):
    def test_direct_proficiency_used_when_no_neighbors(self):
        matrix = np.array([[0.0, 0.8], [0.9, 0.0]])
        fit = compute_skill_fit(np.array([0, 1]), [1], matrix, {}, lam=0.5)
        np.testing.assert_allclose(fit, [0.8, 0.0])

    def test_neighbor_credit_applied_with_lambda(self):
        # Worker has no direct proficiency in skill 0, but proficiency 1.0
        # in its neighbor skill 1; with lambda=0.5, credit = 0.5.
        matrix = np.array([[0.0, 1.0]])
        neighbor_indices = {0: np.array([1])}
        fit = compute_skill_fit(np.array([0]), [0], matrix, neighbor_indices, lam=0.5)
        np.testing.assert_allclose(fit, [0.5])

    def test_direct_beats_neighbor_when_higher(self):
        matrix = np.array([[0.9, 1.0]])
        neighbor_indices = {0: np.array([1])}
        fit = compute_skill_fit(np.array([0]), [0], matrix, neighbor_indices, lam=0.5)
        np.testing.assert_allclose(fit, [0.9])  # max(0.9, 0.5*1.0)

    def test_empty_pool_returns_empty(self):
        matrix = np.zeros((3, 2))
        fit = compute_skill_fit(np.array([], dtype=int), [0], matrix, {}, lam=0.5)
        self.assertEqual(len(fit), 0)


class TestBuildSkillRelatedness(unittest.TestCase):
    def test_topn_and_undirected(self):
        cooc = pd.DataFrame({
            "skill_a": ["A", "A", "B"], "skill_b": ["B", "C", "C"],
            "support": [10, 8, 5], "npmi": [0.5, 0.3, 0.9],
        })
        relatedness = build_skill_relatedness(cooc, topn=1)
        self.assertEqual(relatedness["A"], [("B", 0.5)])
        self.assertEqual(relatedness["B"][0][0], "C")  # 0.9 > 0.5
        self.assertEqual(relatedness["C"][0][0], "B")


class TestCandidateIndex(unittest.TestCase):
    def _workers(self):
        return pd.DataFrame({
            "worker_id": ["W1", "W2", "W3"],
            "primary_category": ["Writing", "Design", "Writing"],
            "secondary_categories": ["", "Writing", ""],
            "country": ["United States", "United Kingdom", "United States"],
            "join_day": [0.0, 0.0, 100.0],
        })

    def test_primary_and_secondary_pool(self):
        idx = CandidateIndex(self._workers(), ["United States", "United Kingdom"])
        pool, fit = idx.eligible_pool("Writing", day=50.0, location_requirement=None, blocked=None)
        # W3 excluded: joins on day 100, after this request's day 50.
        self.assertEqual(sorted(pool.tolist()), [0, 1])  # W1 (primary), W2 (secondary)

    def test_location_requirement_filters_country(self):
        idx = CandidateIndex(self._workers(), ["United States", "United Kingdom"])
        pool, fit = idx.eligible_pool(
            "Writing", day=200.0, location_requirement="Only freelancers located in the United Kingdom may apply.", blocked=None
        )
        self.assertEqual(pool.tolist(), [1])  # only W2 is in the UK

    def test_blocked_workers_excluded(self):
        idx = CandidateIndex(self._workers(), ["United States", "United Kingdom"])
        pool, fit = idx.eligible_pool("Writing", day=200.0, location_requirement=None, blocked=np.array([0]))
        self.assertNotIn(0, pool.tolist())


class TestSimulateSmallScale(unittest.TestCase):
    def _small_setup(self):
        cfg = replace(load_config(), seed=42)
        gen_cfg = replace(GEN_CFG, applicant_pool_max=5, shortlist_size=2)

        workers_private = pd.DataFrame({
            "worker_id": [f"W{i}" for i in range(10)],
            "quality": np.linspace(0.1, 0.9, 10),
            "true_skills": ["Copywriting|English"] * 10,
            "proficiency": ["0.8|0.6"] * 10,
            "taste_0": np.zeros(10), "taste_1": np.zeros(10),
        })
        workers_df = pd.DataFrame({
            "worker_id": [f"W{i}" for i in range(10)],
            "primary_category": ["Writing"] * 10,
            "secondary_categories": [""] * 10,
            "price_tier": np.linspace(0.1, 0.9, 10),
            "country": ["United States"] * 10,
            "join_day": [0.0] * 10,
        })
        clients_private = pd.DataFrame({"client_id": ["C0"], "taste_0": [0.0], "taste_1": [0.0]})
        timeline_df = pd.DataFrame({
            "link": [f"r{i}" for i in range(20)], "client_id": "C0", "simulated_day": np.arange(20),
        })
        requests_info = pd.DataFrame({
            "link": [f"r{i}" for i in range(20)], "category": "Writing", "price_tier": 0.5,
            "skills_vocab": "Copywriting|English", "location_requirement": None,
        })
        vocab = ["Copywriting", "English"]
        cooc = pd.DataFrame({"skill_a": ["Copywriting"], "skill_b": ["English"], "support": [10], "npmi": [0.5]})

        relatedness = build_skill_relatedness(cooc, gen_cfg.skill_relatedness_topn)
        proficiency_matrix, skill_index = build_proficiency_matrix(workers_private, vocab)
        neighbor_indices = build_neighbor_indices(relatedness, skill_index)
        candidate_index = CandidateIndex(workers_df, ["United States"])

        return (timeline_df, requests_info, clients_private, workers_private, candidate_index,
                proficiency_matrix, skill_index, neighbor_indices, cfg, gen_cfg)

    def _simulate(self, args):
        (timeline_df, requests_info, clients_private, workers_private, candidate_index,
         proficiency_matrix, skill_index, neighbor_indices, cfg, gen_cfg) = args
        calibration = estimate_utility_moments(
            timeline_df, requests_info, clients_private, workers_private, candidate_index,
            proficiency_matrix, skill_index, neighbor_indices, cfg, gen_cfg,
        )
        return simulate(
            timeline_df, requests_info, clients_private, workers_private, candidate_index,
            proficiency_matrix, skill_index, neighbor_indices, calibration, cfg, gen_cfg,
        )

    def test_end_to_end_produces_valid_interactions(self):
        interactions_df, truth_df = self._simulate(self._small_setup())

        self.assertGreater(len(interactions_df), 0)
        self.assertTrue(set(interactions_df["event"]).issubset({"SHORTLISTED", "HIRED"}))
        # No private column leaks into the model-visible interactions frame.
        for col in ("quality", "utility", "noise", "S_raw"):
            self.assertNotIn(col, interactions_df.columns)
        # Truth frame carries the private utility decomposition.
        for col in ("S_raw", "utility", "noise", "is_hired"):
            self.assertIn(col, truth_df.columns)

    def test_deterministic(self):
        setup = self._small_setup()
        i1, t1 = self._simulate(setup)
        i2, t2 = self._simulate(setup)
        self.assertTrue(i1.equals(i2))
        self.assertTrue(t1.equals(t2))

    def test_at_most_one_hire_per_request(self):
        interactions_df, _ = self._simulate(self._small_setup())
        hires_per_request = interactions_df.loc[interactions_df["event"] == "HIRED"].groupby("link").size()
        self.assertTrue((hires_per_request <= 1).all())


if __name__ == "__main__":
    unittest.main()
