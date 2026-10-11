"""Day 8 leakage and frozen-model integrity guards for the hybrid."""
import ast
import inspect
import json
import re
import unittest
from pathlib import Path

import numpy as np

from src.data.io import sha256_of
from src.evaluation import cold_start, diversity, significance
from src.recommenders import frozen_models, hybrid
from tests.test_hybrid import CANDIDATES, hybrid as make_hybrid

REPO_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_MARKERS = ("quality", "true_skills", "proficiency", "utility", "noise", "taste_", "oracle",
                   "synthetic_private", "clients_private", "workers_private", "truth.csv", "is_cold_start")


def _code_only_source(module) -> str:
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body[0].value.value = ""
    return ast.unparse(tree)


class TestStaticHybridInputs(unittest.TestCase):
    def test_no_generator_private_markers(self):
        for module in (hybrid, frozen_models, significance, diversity, cold_start):
            code = _code_only_source(module).lower()
            for marker in PRIVATE_MARKERS:
                self.assertNotIn(marker, code, f"{module.__name__} mentions {marker!r}")

    def test_hybrid_scoring_never_reads_labels_or_outcomes(self):
        code = _code_only_source(hybrid)
        scoring = code.split("def tune_hybrid")[0]  # everything except the validation tuning loop
        for marker in ("relevance", "interactions", "split", "HIRED", "SHORTLISTED", "rating", "test"):
            self.assertIsNone(re.search(rf"\b{marker}\b", scoring), f"hybrid scoring code mentions {marker!r}")

    def test_hybrid_does_not_reimplement_components(self):
        code = _code_only_source(hybrid)
        for marker in ("jaccard", "cosine", "enumerate_paths", "path_weight", "price_tier", "declared_skills"):
            self.assertNotIn(marker, code)
        for call in ("cbf.score(", "cf._cf_scores(", "kg.scored("):
            self.assertIn(call, code)

    def test_tuning_function_has_no_test_split_reference(self):
        source = inspect.getsource(hybrid.tune_hybrid).split('"""')[-1]
        self.assertNotIn("test", source)


class TestRuntimeNoLabelInfluence(unittest.TestCase):
    def test_scores_depend_only_on_component_scores(self):
        # Two different relevance assignments cannot change hybrid scores:
        # the hybrid API has no channel through which labels could enter.
        a = make_hybrid().scored("r", CANDIDATES)[3]
        b = make_hybrid().scored("r", CANDIDATES)[3]
        np.testing.assert_array_equal(a, b)
        params = inspect.signature(hybrid.HybridRecommender.rank).parameters
        self.assertEqual(list(params), ["self", "link", "candidates", "k"])


class TestFrozenModelIntegrity(unittest.TestCase):
    def test_standalone_configs_and_results_unchanged_since_day7(self):
        manifest_path = REPO_ROOT / "results" / "day7" / "day7_manifest.json"
        if not manifest_path.exists():
            self.skipTest("Day 7 results not present")
        with open(manifest_path, encoding="utf-8") as fh:
            day7 = json.load(fh)
        for path, digest in day7["frozen_inputs_sha256"].items():
            self.assertEqual(sha256_of(REPO_ROOT / path), digest, path)
        self.assertEqual(sha256_of(REPO_ROOT / "config" / "kg_config.yaml"), day7["kg_config_sha256"])
        for entry in day7["files"]:
            self.assertEqual(sha256_of(REPO_ROOT / entry["path"]), entry["sha256"], entry["path"])

    def test_frozen_selected_values(self):
        def load(p):
            with open(REPO_ROOT / p, encoding="utf-8") as fh:
                return json.load(fh)
        cbf = load("results/day4/selected_cbf_config.json")
        cf = load("results/day5/selected_cf_config.json")
        kg = load("results/day7/selected_kg_config.json")
        self.assertEqual(cbf["weights"], {"skill": 0.5, "category": 0.25, "price": 0.25})
        self.assertEqual((cf["method"], cf["k_neighbors"], cf["aggregation"]), ("user", 20, "sum"))
        self.assertEqual(cf["event_weights"], {"HIRED": 2.0, "SHORTLISTED": 1.0})
        self.assertEqual(cf["fallback"], "popularity_training_only")
        self.assertEqual(kg["damping"], 0.4)
        self.assertEqual(kg["weights"], {"skill": 0.2, "related_skill": 0.15, "category": 0.1, "history": 0.2, "reputation": 0.35})

    def test_generator_unchanged(self):
        with open(REPO_ROOT / "config" / "generator_v1.yaml.sha256", encoding="utf-8") as fh:
            self.assertEqual(sha256_of(REPO_ROOT / "config" / "generator_v1.yaml"), fh.read().split()[0])


if __name__ == "__main__":
    unittest.main()
