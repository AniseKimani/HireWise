"""Leakage guards for the knowledge graph: preparation never reads
generator-private truth, prohibited fields never appear in a prepared
table, and no private marker appears in the loader's source code.
"""
import inspect
import unittest
from pathlib import Path

import pandas as pd

from src.kg import loader, prepare, schema

REPO_ROOT = Path(__file__).resolve().parents[1]

PRIVATE_MARKERS = (
    "quality", "true_skills", "proficiency", "taste_", "synthetic_private",
    "utility", "noise", "oracle", "standardized",
)


def _code_only_source(module) -> str:
    """Source of `module` with module/class/function docstrings blanked
    out (comments are already dropped by ast.unparse), so leakage scans
    check actual executable code, not explanatory prose that legitimately
    names what the module does NOT read/use."""
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


class TestPrepareModuleNeverReferencesPrivatePaths(unittest.TestCase):
    def test_prepare_code_never_opens_synthetic_private(self):
        code = _code_only_source(prepare)
        self.assertNotIn("synthetic_private", code)

    def test_prepare_code_has_no_private_markers(self):
        code = _code_only_source(prepare).lower()
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, code, f"prepare.py's code (not prose) mentions {marker!r}")

    def test_loader_code_has_no_private_markers(self):
        code = _code_only_source(loader).lower()
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, code, f"loader.py's code (not prose) mentions {marker!r}")


class TestPreparedTablesNeverCarryForbiddenColumns(unittest.TestCase):
    def test_allow_list_rejects_a_forbidden_column(self):
        from src.kg.prepare import _assert_allowed_columns

        df = pd.DataFrame({"worker_id": ["W1"], "quality": [0.9]})
        with self.assertRaises(ValueError):
            _assert_allowed_columns(df, schema.ALLOWED_NODE_PROPERTIES[schema.LABEL_WORKER], {"worker_id"}, "test")

    def test_worker_node_preparation_cannot_smuggle_a_private_column(self):
        workers = pd.DataFrame([{
            "worker_id": "W1", "experience_level": "entry", "experience_years": 1.0,
            "price_tier": 0.5, "rate": 10.0, "join_day": 0.0, "is_cold_start": False,
            "quality": 0.99,  # present upstream, must never reach the output
        }])
        out = prepare.prepare_worker_nodes(workers)
        self.assertNotIn("quality", out.columns)


class TestRealDataNeverExposesPrivateColumns(unittest.TestCase):
    def test_prepared_kg_tables_carry_no_private_columns(self):
        kg_dir = REPO_ROOT / "data" / "processed" / "kg"
        if not kg_dir.exists():
            self.skipTest("KG artifacts not prepared locally")
        for filename in prepare.KG_FILENAMES.values():
            path = kg_dir / filename
            if not path.exists():
                continue
            columns = pd.read_csv(path, nrows=1).columns
            for marker in PRIVATE_MARKERS:
                leaked = [c for c in columns if marker in c.lower()]
                self.assertEqual(leaked, [], f"{filename} carries private-looking column(s): {leaked}")

    def test_prepare_all_never_opens_a_synthetic_private_path(self):
        # Monkeypatch pandas.read_csv to record every path it's asked to
        # open, then run prepare_all() and assert none of them touch
        # synthetic_private.
        import pandas as pd_module

        opened_paths = []
        original_read_csv = pd_module.read_csv

        def spy_read_csv(path, *args, **kwargs):
            opened_paths.append(str(path))
            return original_read_csv(path, *args, **kwargs)

        day3_dir = REPO_ROOT / "data" / "processed" / "synthetic"
        if not day3_dir.exists():
            self.skipTest("Day 3 synthetic outputs not generated locally")

        pd_module.read_csv = spy_read_csv
        try:
            prepare.prepare_all()
        finally:
            pd_module.read_csv = original_read_csv

        leaked = [p for p in opened_paths if "synthetic_private" in p]
        self.assertEqual(leaked, [])


if __name__ == "__main__":
    unittest.main()
