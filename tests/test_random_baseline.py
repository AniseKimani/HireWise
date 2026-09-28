import unittest

from src.recommenders.random_baseline import RandomBaseline


class TestRandomBaseline(unittest.TestCase):
    def test_deterministic_across_instances(self):
        candidates = [f"W{i:03d}" for i in range(30)]
        r1 = RandomBaseline(seed=42).rank("link-a", candidates, k=10)
        r2 = RandomBaseline(seed=42).rank("link-a", candidates, k=10)
        self.assertEqual(r1, r2)

    def test_different_links_get_different_orderings(self):
        candidates = [f"W{i:03d}" for i in range(30)]
        rec = RandomBaseline(seed=42)
        r_a = rec.rank("link-a", candidates, k=10)
        r_b = rec.rank("link-b", candidates, k=10)
        self.assertNotEqual(r_a, r_b)

    def test_different_seed_changes_ranking(self):
        candidates = [f"W{i:03d}" for i in range(30)]
        r1 = RandomBaseline(seed=42).rank("link-a", candidates, k=10)
        r2 = RandomBaseline(seed=43).rank("link-a", candidates, k=10)
        self.assertNotEqual(r1, r2)

    def test_only_returns_eligible_candidates(self):
        candidates = [f"W{i:03d}" for i in range(5)]
        ranked = RandomBaseline(seed=42).rank("link-a", candidates, k=10)
        self.assertTrue(set(ranked).issubset(set(candidates)))

    def test_no_duplicates(self):
        candidates = [f"W{i:03d}" for i in range(50)]
        ranked = RandomBaseline(seed=42).rank("link-a", candidates, k=10)
        self.assertEqual(len(ranked), len(set(ranked)))

    def test_returns_at_most_k(self):
        candidates = [f"W{i:03d}" for i in range(50)]
        ranked = RandomBaseline(seed=42).rank("link-a", candidates, k=10)
        self.assertEqual(len(ranked), 10)

    def test_fewer_candidates_than_k_returns_all(self):
        candidates = [f"W{i:03d}" for i in range(4)]
        ranked = RandomBaseline(seed=42).rank("link-a", candidates, k=10)
        self.assertEqual(sorted(ranked), sorted(candidates))

    def test_empty_candidates_returns_empty(self):
        self.assertEqual(RandomBaseline(seed=42).rank("link-a", [], k=10), [])

    def test_is_a_permutation_not_a_subset_selection_bias(self):
        # All candidates should be reachable in rank-1 position across many links.
        candidates = [f"W{i:03d}" for i in range(5)]
        rec = RandomBaseline(seed=42)
        first_picks = {rec.rank(f"link-{i}", candidates, k=10)[0] for i in range(200)}
        self.assertEqual(first_picks, set(candidates))


if __name__ == "__main__":
    unittest.main()
