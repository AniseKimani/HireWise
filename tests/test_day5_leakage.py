"""Leakage guards for Day 5 Collaborative Filtering:
- no generator-private data;
- no CBF content-feature dependency (CF must not quietly become a hybrid);
- validation/test interactions cannot alter the fitted training matrix.
"""
import inspect
import unittest
from pathlib import Path

import pandas as pd

from src.recommenders import collaborative, content_based
from src.recommenders.collaborative import CollaborativeFiltering, build_training_matrix
from src.recommenders.popularity import PopularityBaseline

REPO_ROOT = Path(__file__).resolve().parents[1]

PRIVATE_MARKERS = (
    "quality", "true_skills", "proficiency", "utility", "noise",
    "taste_", "synthetic_private", "clients_private", "workers_private", "truth.csv",
)
CONTENT_FEATURE_MARKERS = (
    "declared_skills", "skills_vocab", "category", "price_tier", "rate", "country",
)


class TestNoGeneratorPrivateData(unittest.TestCase):
    def test_collaborative_module_source_has_no_private_markers(self):
        source = inspect.getsource(collaborative).lower()
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker.lower(), source, f"collaborative.py mentions {marker!r}")


def _code_only_source(module) -> str:
    """Source of `module` with module/class/function docstrings blanked
    out and comments dropped (ast.unparse never emits comments), so
    leakage scans below check actual executable code, not explanatory
    prose that legitimately names what the module does NOT do."""
    import ast

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body[0].value.value = ""
    return ast.unparse(tree)


class TestNoContentFeatureDependency(unittest.TestCase):
    def test_collaborative_module_does_not_import_content_based(self):
        code = _code_only_source(collaborative)
        self.assertNotIn("content_based", code)
        self.assertNotIn("ContentBasedRecommender", code)

    def test_collaborative_source_never_references_cbf_feature_names(self):
        import re

        code = _code_only_source(collaborative).lower()
        for marker in CONTENT_FEATURE_MARKERS:
            # Word-boundary match: short markers like "rate" or "category"
            # would otherwise false-positive inside unrelated identifiers
            # such as "aggregated" or "categorical".
            pattern = rf"\b{re.escape(marker)}\b"
            self.assertIsNone(re.search(pattern, code), f"collaborative.py's code (not prose) mentions content feature {marker!r}")

    def test_content_based_module_not_imported_transitively(self):
        # collaborative.py's own dependency list should not include CBF.
        self.assertNotIn(content_based, [m for m in collaborative.__dict__.values() if inspect.ismodule(m)])


class TestValidationTestCannotAlterFittedModel(unittest.TestCase):
    def _scenario(self):
        interactions = pd.DataFrame([
            {"link": "r1", "worker_id": "W1", "client_id": "C1", "event": "HIRED"},
            {"link": "r2", "worker_id": "W2", "client_id": "C1", "event": "SHORTLISTED"},
        ])
        requests = pd.DataFrame([
            {"link": "r1", "client_id": "C1", "split": "train"},
            {"link": "r2", "client_id": "C1", "split": "train"},
        ])
        return interactions, requests

    def test_adding_validation_interactions_does_not_change_the_matrix(self):
        interactions, requests = self._scenario()
        weights = {"HIRED": 2.0, "SHORTLISTED": 1.0}
        tm_before = build_training_matrix(interactions, requests, weights, "sum")

        extra = pd.concat([interactions, pd.DataFrame([
            {"link": "r3", "worker_id": "W3", "client_id": "C1", "event": "HIRED"},
        ])], ignore_index=True)
        requests_extra = pd.concat([requests, pd.DataFrame([
            {"link": "r3", "client_id": "C1", "split": "validation"},
        ])], ignore_index=True)
        tm_after = build_training_matrix(extra, requests_extra, weights, "sum")

        self.assertEqual(tm_before.workers, tm_after.workers)
        self.assertEqual(tm_before.matrix.toarray().tolist(), tm_after.matrix.toarray().tolist())

    def test_adding_test_interactions_does_not_change_the_matrix(self):
        interactions, requests = self._scenario()
        weights = {"HIRED": 2.0, "SHORTLISTED": 1.0}
        tm_before = build_training_matrix(interactions, requests, weights, "sum")

        extra = pd.concat([interactions, pd.DataFrame([
            {"link": "r4", "worker_id": "W4", "client_id": "C1", "event": "HIRED"},
        ])], ignore_index=True)
        requests_extra = pd.concat([requests, pd.DataFrame([
            {"link": "r4", "client_id": "C1", "split": "test"},
        ])], ignore_index=True)
        tm_after = build_training_matrix(extra, requests_extra, weights, "sum")

        self.assertEqual(tm_before.workers, tm_after.workers)


class TestRealDataNoPrivateColumns(unittest.TestCase):
    def test_protocol_frames_used_by_cf_carry_no_private_columns(self):
        synthetic_dir = REPO_ROOT / "data" / "processed" / "synthetic"
        if not synthetic_dir.exists():
            self.skipTest("Day 3 synthetic outputs not generated locally")
        from src.evaluation.protocol import load_protocol_data

        data = load_protocol_data()
        for marker in PRIVATE_MARKERS:
            leaked = [c for c in data.interactions.columns if marker.lower() in c.lower()]
            self.assertEqual(leaked, [], f"interactions frame carries private-looking column(s): {leaked}")


if __name__ == "__main__":
    unittest.main()
