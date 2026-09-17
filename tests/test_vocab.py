"""Unit tests for src/data/vocab.py using a small synthetic frame."""
import unittest
from dataclasses import replace

import pandas as pd

from src.data.config import load_config
from src.data.vocab import build_vocabulary


def make_flagged_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {"is_duplicate": False, "has_no_skills": False, "is_in_scope_category": True}
    return pd.DataFrame([{**defaults, **r} for r in rows])


class TestBuildVocabulary(unittest.TestCase):
    def test_threshold_is_exact_frequency_cutoff(self):
        cfg = replace(load_config(), min_skill_frequency=3)
        rows = [{"skills_raw": ["A", "B"]} for _ in range(3)] + [{"skills_raw": ["C"]} for _ in range(2)]
        df = make_flagged_frame(rows)
        vocab, vocab_table, rare_table = build_vocabulary(df, cfg)
        self.assertEqual(vocab, {"A", "B"})
        self.assertEqual(set(vocab_table["skill"]), {"A", "B"})
        self.assertEqual(set(rare_table["skill"]), {"C"})

    def test_duplicates_and_out_of_scope_rows_excluded_from_frequency_count(self):
        cfg = replace(load_config(), min_skill_frequency=2)
        rows = [
            {"skills_raw": ["A"], "is_duplicate": True},
            {"skills_raw": ["A"], "is_in_scope_category": False},
            {"skills_raw": ["A"], "has_no_skills": False},
        ]
        df = make_flagged_frame(rows)
        vocab, _, _ = build_vocabulary(df, cfg)
        # Only the third row counts (the other two are excluded from the
        # in-scope modelling population), so frequency is 1, below threshold 2.
        self.assertEqual(vocab, set())

    def test_rare_skills_are_not_discarded_from_the_job_but_from_vocab_only(self):
        cfg = replace(load_config(), min_skill_frequency=2)
        rows = [{"skills_raw": ["Common", "Rare"]}, {"skills_raw": ["Common"]}]
        df = make_flagged_frame(rows)
        vocab, vocab_table, rare_table = build_vocabulary(df, cfg)
        self.assertEqual(vocab, {"Common"})
        self.assertIn("Rare", rare_table["skill"].tolist())
        self.assertEqual(int(rare_table.loc[rare_table["skill"] == "Rare", "frequency"].iloc[0]), 1)


if __name__ == "__main__":
    unittest.main()
