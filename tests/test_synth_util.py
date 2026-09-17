import unittest

import pandas as pd

from src.data.synth_util import apportion_with_floor, stable_id


class TestApportionWithFloor(unittest.TestCase):
    def test_sums_to_exact_total(self):
        weights = pd.Series({"a": 10, "b": 20, "c": 70})
        result = apportion_with_floor(weights, total=33, floor=0)
        self.assertEqual(int(result.sum()), 33)

    def test_respects_floor(self):
        weights = pd.Series({"a": 1, "b": 1, "c": 1000})
        result = apportion_with_floor(weights, total=100, floor=20)
        self.assertTrue((result >= 20).all())
        self.assertEqual(int(result.sum()), 100)

    def test_proportional_to_weights_roughly(self):
        weights = pd.Series({"a": 1, "b": 9})
        result = apportion_with_floor(weights, total=100, floor=0)
        self.assertLess(result["a"], result["b"])

    def test_floor_exceeding_total_raises(self):
        weights = pd.Series({"a": 1, "b": 1, "c": 1})
        with self.assertRaises(ValueError):
            apportion_with_floor(weights, total=2, floor=1)

    def test_deterministic_tie_break(self):
        weights = pd.Series({"z": 1, "a": 1, "m": 1})
        r1 = apportion_with_floor(weights, total=4, floor=0)
        r2 = apportion_with_floor(weights, total=4, floor=0)
        self.assertTrue(r1.equals(r2))


class TestStableId(unittest.TestCase):
    def test_format(self):
        self.assertEqual(stable_id("W", 0), "W000001")
        self.assertEqual(stable_id("C", 999), "C001000")

    def test_width_parameter(self):
        self.assertEqual(stable_id("X", 0, width=3), "X001")


if __name__ == "__main__":
    unittest.main()
