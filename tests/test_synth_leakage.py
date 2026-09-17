"""Leakage guard: no generator-private column may appear in any
model-visible synthetic output file. Runs against the real generated
outputs (skip cleanly if absent) and, defensively, encodes the exact
private-column names so this test fails loudly if a future refactor ever
renames a private field into a shape that could collide with a public
one.
"""
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "data" / "processed" / "synthetic"
PRIVATE_DIR = REPO_ROOT / "data" / "processed" / "synthetic_private"

PRIVATE_COLUMN_MARKERS = {
    "quality", "true_skills", "proficiency", "utility", "noise",
    "S_raw", "T_raw", "Q_raw", "B_raw", "C_raw",
    "S_std", "T_std", "Q_std", "B_std", "C_std",
}
PRIVATE_COLUMN_PREFIXES = ("taste_",)

MODEL_VISIBLE_FILES = ["clients.csv", "workers.csv", "requests.csv", "interactions.csv", "ratings.csv", "cold_start_cohort.csv"]


def _is_private(column: str) -> bool:
    return column in PRIVATE_COLUMN_MARKERS or any(column.startswith(p) for p in PRIVATE_COLUMN_PREFIXES)


@unittest.skipUnless(SYNTHETIC_DIR.exists(), f"{SYNTHETIC_DIR} not found; run scripts/generate_synthetic_data.py first")
class TestNoPrivateLeakageInGeneratedOutputs(unittest.TestCase):
    def test_no_model_visible_file_carries_a_private_column(self):
        for filename in MODEL_VISIBLE_FILES:
            path = SYNTHETIC_DIR / filename
            if not path.exists():
                continue
            columns = pd.read_csv(path, nrows=1).columns
            leaked = [c for c in columns if _is_private(c)]
            self.assertEqual(leaked, [], f"{filename} leaks private column(s): {leaked}")

    def test_private_truth_files_do_carry_the_private_columns(self):
        workers_private = pd.read_csv(PRIVATE_DIR / "workers_private.csv", nrows=1)
        self.assertIn("quality", workers_private.columns)
        truth = pd.read_csv(PRIVATE_DIR / "truth.csv", nrows=1)
        self.assertIn("utility", truth.columns)

    def test_workers_and_clients_do_not_share_row_identity_with_private_frames_beyond_id(self):
        # Sanity: the observable frame's ID column joins cleanly to the
        # private frame's ID column (same population), but no other
        # columns collide.
        workers = pd.read_csv(SYNTHETIC_DIR / "workers.csv", nrows=1)
        workers_private = pd.read_csv(PRIVATE_DIR / "workers_private.csv", nrows=1)
        shared = set(workers.columns) & set(workers_private.columns)
        self.assertEqual(shared, {"worker_id"})


if __name__ == "__main__":
    unittest.main()
