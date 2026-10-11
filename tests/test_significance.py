"""Paired bootstrap significance (Day 8) on small synthetic examples."""
import unittest

import numpy as np
import pandas as pd

from src.evaluation.significance import align_per_request, bootstrap_indices, holm_adjust, paired_bootstrap


class TestPairedBootstrap(unittest.TestCase):
    def test_deterministic_for_same_seed(self):
        rng = np.random.default_rng(0)
        a, b = rng.random(200), rng.random(200)
        self.assertEqual(paired_bootstrap(a, b, 500, [42, 8]), paired_bootstrap(a, b, 500, [42, 8]))
        self.assertNotEqual(paired_bootstrap(a, b, 500, [42, 8])["ci_low"], paired_bootstrap(a, b, 500, [42, 9])["ci_low"])

    def test_identical_methods(self):
        a = np.linspace(0, 1, 50)
        r = paired_bootstrap(a, a.copy(), 1000, 1)
        self.assertEqual(r["mean_difference"], 0.0)
        self.assertEqual((r["ci_low"], r["ci_high"]), (0.0, 0.0))
        self.assertEqual(r["p_value"], 1.0)

    def test_clear_known_difference_is_detected(self):
        rng = np.random.default_rng(1)
        b = rng.random(300)
        a = b + 0.1 + rng.normal(0, 0.01, 300)
        r = paired_bootstrap(a, b, 1000, 7)
        self.assertAlmostEqual(r["mean_difference"], float(np.mean(a - b)))
        self.assertTrue(r["ci_low"] > 0.09 and r["ci_high"] < 0.11)
        self.assertAlmostEqual(r["p_value"], 1 / 1001)

    def test_pure_noise_is_not_significant(self):
        rng = np.random.default_rng(2)
        b = rng.random(400)
        noise = rng.normal(0, 0.2, 400)
        a = b + (noise - noise.mean())  # zero-mean paired difference
        r = paired_bootstrap(a, b, 1000, 3)
        self.assertLess(r["ci_low"], 0.0)
        self.assertGreater(r["ci_high"], 0.0)
        self.assertGreater(r["p_value"], 0.5)

    def test_sign_follows_order(self):
        a, b = np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0])
        self.assertGreater(paired_bootstrap(a, b, 200, 1)["mean_difference"], 0)
        self.assertLess(paired_bootstrap(b, a, 200, 1)["mean_difference"], 0)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            paired_bootstrap([1, 2], [1], 200, 1)
        with self.assertRaises(ValueError):
            paired_bootstrap([], [], 200, 1)

    def test_indices_shape_and_range(self):
        idx = bootstrap_indices(10, 50, 3)
        self.assertEqual(idx.shape, (50, 10))
        self.assertTrue(idx.min() >= 0 and idx.max() <= 9)


class TestHolm(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(holm_adjust([0.5, 0.9]), [1.0, 1.0])
        self.assertEqual(holm_adjust([]), [])


class TestAlignment(unittest.TestCase):
    def test_pairs_by_link(self):
        a = pd.DataFrame({"link": ["x", "y"], "ndcg_at_k": [0.1, 0.2]})
        b = pd.DataFrame({"link": ["y", "x"], "ndcg_at_k": [0.5, 0.4]})
        va, vb, links = align_per_request(a, b, "ndcg_at_k")
        self.assertEqual(links, ["x", "y"])
        np.testing.assert_allclose(va, [0.1, 0.2])
        np.testing.assert_allclose(vb, [0.4, 0.5])

    def test_mismatched_requests_rejected(self):
        a = pd.DataFrame({"link": ["x"], "ndcg_at_k": [0.1]})
        b = pd.DataFrame({"link": ["y"], "ndcg_at_k": [0.5]})
        with self.assertRaises(ValueError):
            align_per_request(a, b, "ndcg_at_k")


if __name__ == "__main__":
    unittest.main()
