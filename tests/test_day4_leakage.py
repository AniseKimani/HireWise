"""Leakage guard for Day 4 recommenders: none of Random, Popularity, or
CBF may read generator-private data. Static source inspection (no
recommender module may import synthetic_private paths or reference
private column names) plus a runtime check that the real workers/requests
frames handed to recommenders carry no private columns.
"""
import inspect
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.protocol import load_protocol_data
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityBaseline
from src.recommenders.random_baseline import RandomBaseline

REPO_ROOT = Path(__file__).resolve().parents[1]
RECOMMENDER_MODULES = [RandomBaseline, PopularityBaseline, ContentBasedRecommender]

PRIVATE_MARKERS = (
    "quality", "true_skills", "proficiency", "utility", "noise",
    "taste_", "synthetic_private", "clients_private", "workers_private", "truth.csv",
)


class TestRecommenderSourceHasNoPrivateReferences(unittest.TestCase):
    def test_no_recommender_source_mentions_private_markers(self):
        for cls in RECOMMENDER_MODULES:
            source = inspect.getsource(cls)
            lowered = source.lower()
            for marker in PRIVATE_MARKERS:
                self.assertNotIn(marker.lower(), lowered, f"{cls.__name__} source mentions {marker!r}")


class TestRecommendersToleratePrivateColumnAbsence(unittest.TestCase):
    """If a recommender secretly depended on a private column, removing
    it from the input frame would raise -- these recommenders should
    never receive it in the first place, but this also proves they don't
    implicitly require it."""

    def test_cbf_works_with_only_documented_observable_columns(self):
        workers = pd.DataFrame([{
            "worker_id": "W1", "primary_category": "Writing", "secondary_categories": "",
            "declared_skills": "Python", "price_tier": 0.5,
        }])
        requests = pd.DataFrame([{"link": "r1", "category": "Writing", "skills_vocab": "Python", "price_tier": 0.5}])
        cbf = ContentBasedRecommender(workers, requests, {"skill": 0.5, "category": 0.25, "price": 0.25}, 0.5)
        self.assertIsInstance(cbf.score("r1", "W1"), float)


class TestRealDataHasNoPrivateColumnsWhereRecommendersReadIt(unittest.TestCase):
    def test_protocol_workers_and_requests_frames_carry_no_private_columns(self):
        if not (REPO_ROOT / "data" / "processed" / "synthetic").exists():
            self.skipTest("Day 3 synthetic outputs not generated locally")
        data = load_protocol_data()
        for marker in PRIVATE_MARKERS:
            leaked_worker_cols = [c for c in data.workers.columns if marker.lower() in c.lower()]
            leaked_request_cols = [c for c in data.requests.columns if marker.lower() in c.lower()]
            self.assertEqual(leaked_worker_cols, [], f"workers frame carries private-looking column(s): {leaked_worker_cols}")
            self.assertEqual(leaked_request_cols, [], f"requests frame carries private-looking column(s): {leaked_request_cols}")


if __name__ == "__main__":
    unittest.main()
