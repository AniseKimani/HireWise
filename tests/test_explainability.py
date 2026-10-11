"""Structural explainability metrics (Day 7)."""
import unittest

from src.evaluation.explainability import component_contribution_summary, deterministic_sample, explainability_summary
from src.kg.explanations import Explanation, ReputationEvidence
from src.recommenders.knowledge_graph import COMPONENTS
from tests.test_knowledge_graph import CANDIDATES, make_recommender


def _explanation(link, rank, n_paths_by_type, lengths, contributions):
    score = sum(contributions.values())
    rep = contributions.get("reputation", 0.0)
    cited = sorted((c for c in COMPONENTS if contributions.get(c, 0) > 0), key=lambda c: -contributions[c])
    return Explanation(
        link=link, worker_id=f"W{rank}", rank=rank, score=score,
        component_contributions={c: contributions.get(c, 0.0) for c in COMPONENTS},
        component_raw={c: 0.0 for c in COMPONENTS}, cited_components=cited,
        n_paths_by_type=n_paths_by_type, path_lengths=lengths, total_path_contribution=score - rep,
        paths=[], reputation=ReputationEvidence(0, None, 4.0, 4.0, True, rep, ""),
    )


class TestExplainabilitySummary(unittest.TestCase):
    def setUp(self):
        self.exps = [
            _explanation("a", 1, {"skill": 2, "category": 1}, [1, 1, 1], {"skill": 0.4, "category": 0.1}),
            _explanation("a", 2, {"category": 1}, [1], {"category": 0.2, "reputation": 0.1}),
        ]

    def test_coverage_with_and_without_category(self):
        s = explainability_summary(self.exps)
        self.assertEqual(s["path_coverage"], 1.0)
        self.assertEqual(s["path_coverage_excluding_category"], 0.5)

    def test_path_counts_and_lengths(self):
        s = explainability_summary(self.exps)
        self.assertEqual(s["mean_supporting_paths"], 2.0)
        self.assertEqual(s["mean_supporting_paths_excluding_category"], 1.0)
        self.assertEqual(s["mean_path_length_pooled"], 1.0)
        self.assertEqual(s["max_observed_path_length"], 1)

    def test_fidelity_sanity(self):
        s = explainability_summary(self.exps)["fidelity_sanity"]
        self.assertEqual(s["share_first_cited_component_is_largest"], 1.0)
        self.assertEqual(s["share_contributions_reconstruct_score"], 1.0)

    def test_fidelity_sanity_detects_a_broken_explanation(self):
        broken = _explanation("b", 1, {"skill": 1}, [1], {"skill": 0.1, "category": 0.5})
        broken.cited_components = ["skill", "category"]
        broken.total_path_contribution = 0.0
        s = explainability_summary([broken])["fidelity_sanity"]
        self.assertEqual(s["share_first_cited_component_is_largest"], 0.0)
        self.assertEqual(s["share_contributions_reconstruct_score"], 0.0)

    def test_real_explanations_pass_sanity(self):
        exps = make_recommender().explain("r1", CANDIDATES, 4)
        s = explainability_summary(exps)
        self.assertEqual(s["fidelity_sanity"]["share_contributions_reconstruct_score"], 1.0)
        self.assertEqual(s["fidelity_sanity"]["share_first_cited_component_is_largest"], 1.0)
        self.assertLessEqual(s["max_observed_path_length"], 3)

    def test_empty(self):
        self.assertEqual(explainability_summary([]), {"n_recommendations": 0})


class TestComponentContributions(unittest.TestCase):
    def test_shares(self):
        exps = [_explanation("a", 1, {"skill": 1}, [1], {"skill": 0.3, "category": 0.1})]
        rows = {r["component"]: r for r in component_contribution_summary(exps, COMPONENTS)}
        self.assertAlmostEqual(rows["skill"]["mean_share_of_score"], 0.75)
        self.assertEqual(rows["skill"]["share_largest_component"], 1.0)
        self.assertEqual(rows["history"]["share_positive"], 0.0)


class TestDeterministicSample(unittest.TestCase):
    def test_same_seed_same_sample_regardless_of_input_order(self):
        exps = [_explanation(f"l{i}", 1, {"skill": 1}, [1], {"skill": 0.1}) for i in range(50)]
        a = deterministic_sample(exps, 10, seed=42)
        b = deterministic_sample(list(reversed(exps)), 10, seed=42)
        self.assertEqual([e.link for e in a], [e.link for e in b])
        self.assertEqual(len(a), 10)
        self.assertNotEqual([e.link for e in a], [e.link for e in deterministic_sample(exps, 10, seed=7)])

    def test_small_population_returned_whole(self):
        exps = [_explanation("x", 1, {"skill": 1}, [1], {"skill": 0.1})]
        self.assertEqual(len(deterministic_sample(exps, 30, seed=42)), 1)


if __name__ == "__main__":
    unittest.main()
