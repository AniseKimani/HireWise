"""Unit tests for src/data/clean.py using a small synthetic raw frame that
mirrors the real CSV's schema, not real Upwork postings."""
import unittest

import pandas as pd

from src.data.clean import finalize_flags, parse_and_flag
from src.data.config import load_config

CONFIG = load_config()


def make_footer(category: str, skills: list[str]) -> str:
    skills_block = ",     ".join(skills)
    return (
        "Job body. "
        "Posted On: February 17, 2024 09:09 UTC\n"
        f"Category: {category}\n"
        f"Skills:{skills_block}\n"
        f"Skills:{skills_block}\n"
        "click to apply"
    )


def make_raw(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "title": "Some Job",
        "link": None,
        "description": "",
        "published_date": "2024-02-17T09:09:00Z",
        "is_hourly": False,
        "hourly_low": None,
        "hourly_high": None,
        "budget": 100.0,
        "country": "United States",
    }
    full_rows = []
    for i, row in enumerate(rows):
        merged = {**defaults, **row}
        if merged["link"] is None:
            merged["link"] = f"https://example.com/job-{i}"
        full_rows.append(merged)
    return pd.DataFrame(full_rows)


class TestDuplicateFlagging(unittest.TestCase):
    def test_completeness_wins_tie_break(self):
        desc = make_footer("Writing", ["Python"] * 25)  # >= min_skill_frequency irrelevant here
        raw = make_raw([
            {"title": "Dup Job", "description": desc, "link": "https://z.com/1",
             "budget": 100.0},  # more complete? both have budget; equal completeness -> link tie-break
            {"title": "Dup Job", "description": desc, "link": "https://a.com/2", "budget": 100.0},
        ])
        cfg = load_config()
        flagged = parse_and_flag(raw, cfg)
        # Equal completeness -> smallest link kept ("https://a.com/2").
        kept = flagged.loc[~flagged["is_duplicate"], "link"].tolist()
        self.assertEqual(kept, ["https://a.com/2"])
        self.assertEqual(int(flagged["is_duplicate"].sum()), 1)

    def test_more_complete_row_is_kept_even_if_link_is_larger(self):
        desc = make_footer("Writing", ["Python"] * 25)
        raw = make_raw([
            {"title": "Dup Job", "description": desc, "link": "https://a.com/1",
             "budget": None, "is_hourly": None},  # less complete
            {"title": "Dup Job", "description": desc, "link": "https://z.com/2",
             "budget": 100.0},  # more complete
        ])
        cfg = load_config()
        flagged = parse_and_flag(raw, cfg)
        kept = flagged.loc[~flagged["is_duplicate"], "link"].tolist()
        self.assertEqual(kept, ["https://z.com/2"])

    def test_unique_rows_are_never_flagged_duplicate(self):
        raw = make_raw([
            {"title": "Job A", "description": make_footer("Writing", ["Python"])},
            {"title": "Job B", "description": make_footer("Design", ["Figma"])},
        ])
        cfg = load_config()
        flagged = parse_and_flag(raw, cfg)
        self.assertEqual(int(flagged["is_duplicate"].sum()), 0)


class TestZeroSkillAndScope(unittest.TestCase):
    def test_zero_skill_job_is_flagged(self):
        desc = (
            "Body. Posted On: February 17, 2024 09:09 UTC\n"
            "Category: Data Entry\n"
            "click to apply"
        )
        raw = make_raw([{"description": desc}])
        cfg = load_config()
        flagged = parse_and_flag(raw, cfg)
        self.assertTrue(bool(flagged.loc[0, "has_no_skills"]))

    def test_category_below_threshold_is_out_of_scope(self):
        # One category with many jobs (in scope), one with a single job
        # (out of scope, given min_category_jobs=100 in the real config;
        # here we use a tiny config-equivalent threshold via monkeypatch).
        big_desc = make_footer("Popular Category", ["Python"])
        small_desc = make_footer("Rare Category", ["Java"])
        rows = [{"title": f"Big {i}", "description": big_desc} for i in range(5)]
        rows.append({"title": "Small", "description": small_desc})
        raw = make_raw(rows)

        from dataclasses import replace
        cfg = replace(load_config(), min_category_jobs=5, min_skill_frequency=1)
        flagged = parse_and_flag(raw, cfg)

        self.assertTrue(flagged.loc[flagged["category"] == "Popular Category", "is_in_scope_category"].all())
        self.assertFalse(flagged.loc[flagged["category"] == "Rare Category", "is_in_scope_category"].any())


class TestExclusionPrecedence(unittest.TestCase):
    def test_precedence_order_duplicate_beats_others(self):
        from dataclasses import replace
        cfg = replace(load_config(), min_category_jobs=1, min_skill_frequency=1)

        desc = make_footer("Writing", ["Python"])
        raw = make_raw([
            {"title": "Dup", "description": desc, "link": "https://a.com/1"},
            {"title": "Dup", "description": desc, "link": "https://a.com/2"},
        ])
        flagged = parse_and_flag(raw, cfg)
        vocab = {"Python"}
        finalized = finalize_flags(flagged, vocab, cfg)
        duplicate_row = finalized.loc[finalized["is_duplicate"]].iloc[0]
        self.assertEqual(duplicate_row["exclusion_reason"], "duplicate")

    def test_eligible_row_has_no_exclusion_reason(self):
        from dataclasses import replace
        cfg = replace(load_config(), min_category_jobs=1, min_skill_frequency=1)
        desc = make_footer("Writing", ["Python"])
        raw = make_raw([{"description": desc}])
        flagged = parse_and_flag(raw, cfg)
        finalized = finalize_flags(flagged, {"Python"}, cfg)
        self.assertTrue(bool(finalized.loc[0, "eligible_for_model"]))
        self.assertIsNone(finalized.loc[0, "exclusion_reason"])

    def test_no_vocab_skill_excludes_even_with_skills_present(self):
        from dataclasses import replace
        cfg = replace(load_config(), min_category_jobs=1, min_skill_frequency=1)
        desc = make_footer("Writing", ["ObscureSkillXYZ"])
        raw = make_raw([{"description": desc}])
        flagged = parse_and_flag(raw, cfg)
        finalized = finalize_flags(flagged, {"Python"}, cfg)  # vocab doesn't include ObscureSkillXYZ
        self.assertFalse(bool(finalized.loc[0, "eligible_for_model"]))
        self.assertEqual(finalized.loc[0, "exclusion_reason"], "no_vocab_skill")


class TestPayTypeAndPriceTier(unittest.TestCase):
    def test_pay_type_derivation(self):
        from dataclasses import replace
        cfg = replace(load_config(), min_category_jobs=1, min_skill_frequency=1)
        desc = make_footer("Writing", ["Python"])
        raw = make_raw([
            {"description": desc, "is_hourly": True, "hourly_low": 10.0, "hourly_high": 20.0, "budget": None},
            {"description": desc, "is_hourly": False, "budget": 500.0},
            {"description": desc, "is_hourly": None, "budget": None, "hourly_low": None, "hourly_high": None},
        ])
        flagged = parse_and_flag(raw, cfg)
        self.assertEqual(flagged["pay_type"].tolist(), ["hourly", "fixed", "unspecified"])
        self.assertAlmostEqual(flagged.loc[0, "pay_amount"], 15.0)
        self.assertAlmostEqual(flagged.loc[1, "pay_amount"], 500.0)
        self.assertTrue(pd.isna(flagged.loc[2, "pay_amount"]))

    def test_stored_budget_value_is_never_modified(self):
        from dataclasses import replace
        cfg = replace(load_config(), min_category_jobs=1, min_skill_frequency=1)
        desc = make_footer("Writing", ["Python"])
        raw = make_raw([{"description": desc, "budget": 987654.0}])
        flagged = parse_and_flag(raw, cfg)
        self.assertEqual(flagged.loc[0, "budget"], 987654.0)


if __name__ == "__main__":
    unittest.main()
