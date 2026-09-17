"""Aggregate integration tests against the local Upwork CSV.

These re-verify the Day 1 approved targets end-to-end through the real
pipeline modules (not just the parser). They skip cleanly, rather than
failing, when the raw dataset is not present locally -- it is git-ignored
and must be supplied by whoever runs the tests.
"""
import unittest
from pathlib import Path

from src.data.clean import finalize_flags, parse_and_flag
from src.data.config import load_config
from src.data.io import load_raw
from src.data.sample import background_corpus, sample_requests
from src.data.vocab import build_vocabulary

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "upwork-jobs.csv"


@unittest.skipUnless(RAW_PATH.exists(), f"raw dataset not found at {RAW_PATH}; skipping integration test")
class TestFullPipelineAgainstRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        raw = load_raw(RAW_PATH)
        cls.flagged = parse_and_flag(raw, cls.config)
        cls.vocab, cls.vocab_table, cls.rare_table = build_vocabulary(cls.flagged, cls.config)
        cls.finalized = finalize_flags(cls.flagged, cls.vocab, cls.config)
        cls.sampled = sample_requests(cls.finalized, cls.config)
        cls.background = background_corpus(cls.sampled)

    def test_raw_row_count(self):
        self.assertEqual(len(self.flagged), 53058)

    def test_duplicate_count(self):
        self.assertEqual(int(self.flagged["is_duplicate"].sum()), 22)

    def test_zero_skill_count(self):
        self.assertEqual(int(self.flagged["has_no_skills"].sum()), 515)

    def test_in_scope_category_count(self):
        n_categories = self.flagged.loc[self.flagged["is_in_scope_category"], "category"].nunique()
        self.assertEqual(n_categories, 88)

    def test_scope_retention_percentage(self):
        dedup_devskill = (~self.flagged["is_duplicate"] & ~self.flagged["has_no_skills"])
        in_scope = dedup_devskill & self.flagged["is_in_scope_category"]
        retention = round(int(in_scope.sum()) / int(dedup_devskill.sum()) * 100, 2)
        self.assertAlmostEqual(retention, 92.34, places=2)

    def test_controlled_vocabulary_size(self):
        self.assertEqual(len(self.vocab), 1187)

    def test_eligible_modelling_jobs_count(self):
        self.assertEqual(int(self.finalized["eligible_for_model"].sum()), 47938)

    def test_sampled_request_count_matches_config(self):
        self.assertEqual(int(self.sampled["is_sampled_request"].sum()), self.config.model_requests)

    def test_background_and_sample_partition_eligible_population(self):
        n_sampled = int(self.sampled["is_sampled_request"].sum())
        n_background = len(self.background)
        n_eligible = int(self.sampled["eligible_for_model"].sum())
        self.assertEqual(n_sampled + n_background, n_eligible)

    def test_no_vocab_skill_leaks_out_of_scope_category_skills(self):
        # A vocabulary skill must have been observed in the in-scope
        # modelling population, by construction of build_vocabulary.
        from src.data.vocab import in_scope_modelling_mask
        in_scope_skills = set(
            s for lst in self.flagged.loc[in_scope_modelling_mask(self.flagged), "skills_raw"] for s in lst
        )
        self.assertTrue(self.vocab.issubset(in_scope_skills))

    def test_raw_file_is_not_modified_by_running_the_pipeline(self):
        size_before = RAW_PATH.stat().st_size
        # Re-run the whole pipeline once more.
        load_raw(RAW_PATH)
        self.assertEqual(RAW_PATH.stat().st_size, size_before)


if __name__ == "__main__":
    unittest.main()
