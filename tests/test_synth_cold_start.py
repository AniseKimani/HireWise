import unittest
from dataclasses import replace

import pandas as pd

from src.data.config import load_config
from src.data.synth_cold_start import assign_worker_join_days, select_cold_start_cohort


def make_workers(n_per_category: int, categories: list[str]) -> pd.DataFrame:
    rows = []
    i = 0
    for cat in categories:
        for _ in range(n_per_category):
            rows.append({"worker_id": f"W{i:04d}", "primary_category": cat})
            i += 1
    return pd.DataFrame(rows)


class TestColdStartCohort(unittest.TestCase):
    def test_cohort_size_matches_ratio(self):
        cfg = replace(load_config(), cold_start_ratio=0.10)
        workers_df = make_workers(100, ["A", "B", "C"])  # 300 workers
        cohort = select_cold_start_cohort(workers_df, cfg)
        self.assertEqual(len(cohort), 30)

    def test_cohort_is_stratified_across_categories(self):
        cfg = replace(load_config(), cold_start_ratio=0.10)
        workers_df = make_workers(100, ["A", "B", "C"])
        cohort = select_cold_start_cohort(workers_df, cfg)
        by_cat = workers_df.set_index("worker_id").loc[list(cohort), "primary_category"].value_counts()
        self.assertTrue((by_cat == 10).all())

    def test_deterministic(self):
        cfg = replace(load_config(), cold_start_ratio=0.10)
        workers_df = make_workers(100, ["A", "B", "C"])
        cohort1 = select_cold_start_cohort(workers_df, cfg)
        cohort2 = select_cold_start_cohort(workers_df, cfg)
        self.assertEqual(cohort1, cohort2)

    def test_assign_join_days(self):
        cfg = replace(load_config(), cold_start_ratio=0.10)
        workers_df = make_workers(100, ["A", "B", "C"])
        cohort = select_cold_start_cohort(workers_df, cfg)
        result = assign_worker_join_days(workers_df, cohort, validation_start_day=200.0)
        cold = result.loc[result["worker_id"].isin(cohort)]
        established = result.loc[~result["worker_id"].isin(cohort)]
        self.assertTrue((cold["join_day"] == 200.0).all())
        self.assertTrue((established["join_day"] == 0.0).all())
        self.assertTrue(cold["is_cold_start"].all())
        self.assertFalse(established["is_cold_start"].any())


if __name__ == "__main__":
    unittest.main()
