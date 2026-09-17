import unittest
from dataclasses import replace

import pandas as pd

from src.data.config import load_config
from src.data.generator_config import load_generator_config
from src.data.synth_workers import apply_location_guarantee, extract_required_countries, generate_workers

GEN_CFG = load_generator_config("config/generator_v1.yaml")


def make_day2_frames():
    category_stats = pd.DataFrame({"category": ["Writing", "Design"], "n_jobs": [100, 50]})
    category_similarity = pd.DataFrame({
        "category": ["Writing"] * 3 + ["Design"] * 3,
        "similar_category": ["Design", "Writing", "Writing"] * 2,
        "rank": [1, 2, 3, 1, 2, 3],
        "cosine_similarity": [0.5, 0.4, 0.3] * 2,
    })
    rows = []
    for i in range(100):
        rows.append({
            "link": f"w{i}", "category": "Writing",
            "skills_vocab": "Copywriting|English|Content Writing",
            "price_tier": (i % 10) / 10.0, "pay_amount": 50.0 + i,
            "country_clean": "United States" if i % 2 == 0 else "United Kingdom",
        })
    for i in range(50):
        rows.append({
            "link": f"d{i}", "category": "Design",
            "skills_vocab": "Graphic Design|Adobe Photoshop",
            "price_tier": (i % 10) / 10.0, "pay_amount": 30.0 + i,
            "country_clean": "United States",
        })
    eligible_jobs = pd.DataFrame(rows)
    return category_stats, category_similarity, eligible_jobs


class TestGenerateWorkers(unittest.TestCase):
    def test_worker_count_and_category_floor(self):
        cfg = replace(load_config(), synthetic_workers=100)
        gen_cfg = replace(GEN_CFG, worker_category_floor=20)
        cat_stats, cat_sim, eligible = make_day2_frames()
        workers_df, private_df = generate_workers(cat_stats, cat_sim, eligible, cfg, gen_cfg)
        self.assertEqual(len(workers_df), 100)
        counts = workers_df["primary_category"].value_counts()
        self.assertTrue((counts >= 20).all())

    def test_skills_come_from_category_pool_only(self):
        cfg = replace(load_config(), synthetic_workers=100)
        gen_cfg = replace(GEN_CFG, worker_category_floor=20, worker_secondary_category_min=0, worker_secondary_category_max=0)
        cat_stats, cat_sim, eligible = make_day2_frames()
        workers_df, private_df = generate_workers(cat_stats, cat_sim, eligible, cfg, gen_cfg)
        writing_pool = {"Copywriting", "English", "Content Writing"}
        design_pool = {"Graphic Design", "Adobe Photoshop"}
        for row in workers_df.itertuples():
            true_skills = set(
                private_df.loc[private_df["worker_id"] == row.worker_id, "true_skills"].iloc[0].split("|")
            )
            expected_pool = writing_pool if row.primary_category == "Writing" else design_pool
            self.assertTrue(true_skills.issubset(expected_pool))

    def test_deterministic(self):
        cfg = replace(load_config(), synthetic_workers=100)
        gen_cfg = replace(GEN_CFG, worker_category_floor=20)
        cat_stats, cat_sim, eligible = make_day2_frames()
        w1, p1 = generate_workers(cat_stats, cat_sim, eligible, cfg, gen_cfg)
        w2, p2 = generate_workers(cat_stats, cat_sim, eligible, cfg, gen_cfg)
        self.assertTrue(w1.equals(w2))
        self.assertTrue(p1.equals(p2))

    def test_private_columns_never_in_observable_frame(self):
        cfg = replace(load_config(), synthetic_workers=100)
        gen_cfg = replace(GEN_CFG, worker_category_floor=20)
        cat_stats, cat_sim, eligible = make_day2_frames()
        workers_df, private_df = generate_workers(cat_stats, cat_sim, eligible, cfg, gen_cfg)
        for col in ("quality", "true_skills", "proficiency"):
            self.assertNotIn(col, workers_df.columns)
            self.assertIn(col, private_df.columns)


class TestExtractRequiredCountries(unittest.TestCase):
    def test_matches_known_country(self):
        text = "Only freelancers located in the United States may apply."
        result = extract_required_countries(text, ["United States", "United Kingdom"])
        self.assertEqual(result, ["United States"])

    def test_no_match_returns_empty(self):
        self.assertEqual(extract_required_countries("No restriction here", ["United States"]), [])

    def test_non_string_returns_empty(self):
        self.assertEqual(extract_required_countries(float("nan"), ["United States"]), [])

    def test_does_not_confuse_similar_country_names(self):
        text = "Only freelancers located in the United Arab Emirates may apply."
        result = extract_required_countries(text, ["United States", "United Arab Emirates"])
        self.assertEqual(result, ["United Arab Emirates"])


class TestLocationGuarantee(unittest.TestCase):
    def test_shortfall_pairs_get_workers_reassigned(self):
        cfg = replace(load_config(), synthetic_workers=100)
        gen_cfg = replace(GEN_CFG, worker_category_floor=20, worker_location_min_eligible=10)
        cat_stats, cat_sim, eligible = make_day2_frames()
        workers_df, _ = generate_workers(cat_stats, cat_sim, eligible, cfg, gen_cfg)

        sampled_requests = pd.DataFrame({
            "location_requirement": ["Only freelancers located in the United Kingdom may apply."] * 5,
            "category": ["Writing"] * 5,
        })
        fixed_df, report = apply_location_guarantee(workers_df, sampled_requests, ["United States", "United Kingdom"], cfg, gen_cfg)
        eligible_uk = fixed_df.loc[(fixed_df["primary_category"] == "Writing") & (fixed_df["country"] == "United Kingdom")]
        self.assertGreaterEqual(len(eligible_uk), gen_cfg.worker_location_min_eligible)
        self.assertEqual(report["pairs_checked"], 1)


if __name__ == "__main__":
    unittest.main()
