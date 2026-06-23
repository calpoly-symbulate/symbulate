"""Tests for symbulate.index_sets.

Covers membership testing, item lookup, and equality for all four
index set classes: Reals, Naturals, DiscreteTimeSequence, and Integers.
"""
import unittest

from symbulate.index_sets import (
    Reals,
    Naturals,
    DiscreteTimeSequence,
    Integers,
)


class TestReals(unittest.TestCase):

    def test_positive_real_is_contained(self):
        self.assertIn(1.5, Reals())

    def test_negative_real_is_contained(self):
        self.assertIn(-3.14, Reals())

    def test_zero_is_contained(self):
        self.assertIn(0, Reals())

    def test_integer_is_contained(self):
        self.assertIn(5, Reals())

    def test_positive_infinity_not_contained(self):
        self.assertNotIn(float('inf'), Reals())

    def test_negative_infinity_not_contained(self):
        self.assertNotIn(float('-inf'), Reals())

    def test_string_not_contained(self):
        self.assertNotIn('hello', Reals())

    def test_none_not_contained(self):
        self.assertNotIn(None, Reals())

    def test_getitem_returns_value(self):
        self.assertEqual(Reals()[2.5], 2.5)

    def test_getitem_raises_for_infinity(self):
        with self.assertRaises(KeyError):
            Reals()[float('inf')]

    def test_equality_same_type(self):
        self.assertEqual(Reals(), Reals())

    def test_inequality_different_type(self):
        self.assertNotEqual(Reals(), Naturals())


class TestNaturals(unittest.TestCase):

    def test_zero_is_contained(self):
        self.assertIn(0, Naturals())

    def test_positive_integer_is_contained(self):
        self.assertIn(3, Naturals())

    def test_negative_integer_not_contained(self):
        self.assertNotIn(-1, Naturals())

    def test_float_whole_number_is_contained(self):
        # 2.0 is an integer-valued float so it qualifies
        self.assertIn(2.0, Naturals())

    def test_non_integer_float_not_contained(self):
        self.assertNotIn(1.5, Naturals())

    def test_string_not_contained(self):
        self.assertNotIn('3', Naturals())

    def test_none_not_contained(self):
        self.assertNotIn(None, Naturals())

    def test_getitem_returns_value(self):
        self.assertEqual(Naturals()[3], 3)

    def test_getitem_raises_for_negative(self):
        with self.assertRaises(KeyError):
            Naturals()[-1]

    def test_getitem_raises_for_non_integer(self):
        with self.assertRaises(KeyError):
            Naturals()[1.5]

    def test_equality_same_type(self):
        self.assertEqual(Naturals(), Naturals())

    def test_inequality_different_type(self):
        self.assertNotEqual(Naturals(), Reals())


class TestDiscreteTimeSequence(unittest.TestCase):

    def test_getitem_converts_index_to_time(self):
        ts = DiscreteTimeSequence(fs=4)
        self.assertEqual(ts[2], 0.5)

    def test_getitem_zero_returns_zero(self):
        ts = DiscreteTimeSequence(fs=4)
        self.assertEqual(ts[0], 0.0)

    def test_getitem_larger_index(self):
        ts = DiscreteTimeSequence(fs=4)
        self.assertEqual(ts[8], 2.0)

    def test_grid_point_is_contained(self):
        ts = DiscreteTimeSequence(fs=4)
        self.assertIn(0.25, ts)
        self.assertIn(0.5, ts)
        self.assertIn(1.0, ts)

    def test_zero_is_contained(self):
        self.assertIn(0, DiscreteTimeSequence(fs=4))

    def test_non_grid_point_not_contained(self):
        ts = DiscreteTimeSequence(fs=4)
        self.assertNotIn(0.1, ts)
        self.assertNotIn(0.3, ts)

    def test_equality_same_fs(self):
        self.assertEqual(DiscreteTimeSequence(4), DiscreteTimeSequence(4))

    def test_inequality_different_fs(self):
        self.assertNotEqual(DiscreteTimeSequence(4), DiscreteTimeSequence(8))

    def test_inequality_different_type(self):
        self.assertNotEqual(DiscreteTimeSequence(4), Reals())


class TestIntegers(unittest.TestCase):

    def test_is_discrete_time_sequence(self):
        self.assertIsInstance(Integers(), DiscreteTimeSequence)

    def test_getitem_returns_integer_as_float(self):
        # Integers has fs=1, so index n maps to n/1 = n
        self.assertEqual(Integers()[3], 3.0)

    def test_zero_is_contained(self):
        self.assertIn(0, Integers())

    def test_positive_integer_is_contained(self):
        self.assertIn(5, Integers())

    def test_non_integer_not_contained(self):
        self.assertNotIn(1.5, Integers())

    def test_negative_integer_is_contained(self):
        # DiscreteTimeSequence.__contains__ only checks is_integer, not sign,
        # so negative integers are accepted even though the docstring says
        # "non-negative integers (0, 1, 2, ...)".
        self.assertIn(-1, Integers())

    def test_equality_same_type(self):
        self.assertEqual(Integers(), Integers())


if __name__ == '__main__':
    unittest.main()
