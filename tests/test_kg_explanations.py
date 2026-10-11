"""Day 7 graph-path explanations: built from the scored paths, ordered by
contribution, faithful to the score, and correctly templated."""
import json
import unittest

from src.kg.explanations import PATH_SPECS, build_summary, path_text, reputation_text
from src.recommenders.knowledge_graph import COMPONENTS
from tests.test_knowledge_graph import CANDIDATES, make_recommender


class TestPathSpecs(unittest.TestCase):
    def test_every_path_is_at_most_three_edges_and_labels_align(self):
        for spec in PATH_SPECS.values():
            self.assertLessEqual(spec.length, 3)
            self.assertEqual(len(spec.labels), spec.length + 1)
            self.assertEqual(spec.labels[-1], "Worker")
            self.assertIn(spec.component, COMPONENTS)

    def test_reputation_is_not_a_request_anchored_path(self):
        self.assertNotIn("reputation", {s.component for s in PATH_SPECS.values()})


class TestTemplates(unittest.TestCase):
    def test_texts_use_only_path_content(self):
        self.assertEqual(path_text("skill", ("Python", "W1"), ({"type": "HAS_SKILL"},)), "Has required skill 'Python'")
        self.assertIn("NPMI 0.62", path_text("related_skill", ("Django", "Flask", "W1"),
                                             ({"type": "RELATED_TO", "npmi": 0.62, "support": 9}, {"type": "HAS_SKILL"})))
        self.assertIn("secondary", path_text("category", ("Cat", "W1"), ({"type": "SPECIALIZES_IN", "role": "secondary"},)))
        self.assertIn("2 training-period hires", path_text("repeat_hire", ("C1", "W1"), ({"type": "HIRED", "n_hires": 2},)))

    def test_reputation_text_never_invents_reviews(self):
        self.assertIn("No training-period reviews", reputation_text(0, None, 4.4, 4.4))
        self.assertIn("from 3 reviews", reputation_text(3, 4.0, 4.2, 4.4))

    def test_unknown_path_type(self):
        with self.assertRaises(ValueError):
            path_text("nope", (), ())

    def test_empty_summary(self):
        self.assertEqual(build_summary([], [], None), "No positive graph evidence beyond eligibility")


class TestExplanations(unittest.TestCase):
    def setUp(self):
        self.rec = make_recommender()
        self.explanations = self.rec.explain("r1", CANDIDATES, 4)

    def test_explanations_follow_the_ranking(self):
        self.assertEqual([e.worker_id for e in self.explanations], self.rec.rank("r1", CANDIDATES, 4))
        self.assertEqual([e.rank for e in self.explanations], [1, 2, 3, 4])

    def test_contributions_reconstruct_score_exactly(self):
        for e in self.explanations:
            self.assertAlmostEqual(e.total_path_contribution + e.reputation.contribution, e.score, places=12)
            self.assertAlmostEqual(sum(e.component_contributions.values()), e.score, places=12)

    def test_first_cited_component_is_largest(self):
        for e in self.explanations:
            if e.cited_components:
                top = e.cited_components[0]
                self.assertEqual(e.component_contributions[top], max(e.component_contributions.values()))

    def test_top_worker_paths_and_edge_properties(self):
        w1 = self.explanations[0]
        self.assertEqual(w1.worker_id, "W1")
        self.assertEqual(w1.n_paths_by_type, {"skill": 1, "related_skill": 1, "category": 1, "repeat_hire": 0,
                                              "hired_similar_skill": 1, "co_hire": 1})
        self.assertEqual(sorted(w1.path_lengths), [1, 1, 2, 3, 3])
        co_hire = next(p for p in w1.paths if p.path_type == "co_hire")
        self.assertEqual(co_hire.nodes, (("Client", "C1"), ("Worker", "W3"), ("Client", "C2"), ("Worker", "W1")))
        self.assertTrue(all(edge["type"] == "HIRED" and edge["split"] == "train" for edge in co_hire.edges))
        related = next(p for p in w1.paths if p.path_type == "related_skill")
        self.assertEqual(related.edges[0], {"type": "RELATED_TO", "npmi": 0.5, "support": 10})

    def test_history_paths_never_cite_validation_or_test_links(self):
        for e in self.explanations:
            for p in e.paths:
                for edge in p.edges:
                    for link in edge.get("links", []):
                        self.assertIn(link, {"t1", "t2", "t3"})

    def test_cold_worker_explanation(self):
        w4 = next(e for e in self.explanations if e.worker_id == "W4")
        self.assertEqual(w4.n_paths_by_type["repeat_hire"] + w4.n_paths_by_type["hired_similar_skill"]
                         + w4.n_paths_by_type["co_hire"], 0)
        self.assertTrue(w4.reputation.prior_used_only)
        self.assertIn("No training-period reviews", w4.reputation.text)

    def test_displayed_paths_are_capped_per_component(self):
        rec = make_recommender()
        rec.max_paths_per_component = 1
        for e in rec.explain("r1", CANDIDATES, 4):
            components = [p.component for p in e.paths]
            for c in set(components):
                self.assertLessEqual(components.count(c), 1)

    def test_explanations_are_deterministic_and_serializable(self):
        again = make_recommender().explain("r1", CANDIDATES, 4)
        first = json.dumps([e.to_dict() for e in self.explanations], sort_keys=True)
        self.assertEqual(first, json.dumps([e.to_dict() for e in again], sort_keys=True))


if __name__ == "__main__":
    unittest.main()
