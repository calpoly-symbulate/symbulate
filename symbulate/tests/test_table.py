"""Tests for symbulate.table.Table.

Tests focus on user-observable behavior: correct lookups, expected
display output, arithmetic operations, and integration with .tabulate().
"""
import unittest
import numpy as np

from symbulate import *
from symbulate import distributions
from symbulate.table import Table


class TestTableConstruction(unittest.TestCase):

    def test_lookup_by_outcome(self):
        t = Table({'heads': 55, 'tails': 45})
        self.assertEqual(t['heads'], 55)
        self.assertEqual(t['tails'], 45)

    def test_missing_outcome_filled_with_zero(self):
        t = Table({'A': 3}, outcomes=['A', 'B', 'C'])
        self.assertEqual(t['B'], 0)
        self.assertEqual(t['C'], 0)

    def test_missing_key_raises_key_error(self):
        t = Table({'a': 1, 'b': 2})
        with self.assertRaises(KeyError):
            _ = t['c']

    def test_normalize_values_sum_to_one(self):
        t = Table({'heads': 55, 'tails': 45}, normalize=True)
        self.assertAlmostEqual(sum(t.values()), 1.0)

    def test_normalize_correct_proportions(self):
        t = Table({'heads': 55, 'tails': 45}, normalize=True)
        self.assertAlmostEqual(t['heads'], 0.55)
        self.assertAlmostEqual(t['tails'], 0.45)

    def test_normalize_with_missing_outcomes(self):
        # zeros added for missing entries should not affect normalization
        t = Table({'A': 80, 'B': 20}, outcomes=['A', 'B', 'C'], normalize=True)
        self.assertAlmostEqual(t['A'], 0.80)
        self.assertAlmostEqual(t['B'], 0.20)
        self.assertAlmostEqual(t['C'], 0.0)


class TestTableDisplay(unittest.TestCase):

    def test_repr_shows_outcomes_and_counts(self):
        t = Table({'heads': 55, 'tails': 45})
        r = repr(t)
        self.assertIn('heads', r)
        self.assertIn('tails', r)
        self.assertIn('55', r)
        self.assertIn('45', r)

    def test_repr_shows_correct_total(self):
        t = Table({'a': 30, 'b': 70})
        r = repr(t)
        self.assertIn('Total', r)
        self.assertIn('100', r)

    def test_repr_sorted_by_default(self):
        t = Table({'cat': 1, 'ant': 3, 'bee': 2})
        r = repr(t)
        self.assertLess(r.index('ant'), r.index('bee'))
        self.assertLess(r.index('bee'), r.index('cat'))

    def test_repr_outcomes_order_preserved(self):
        t = Table({'bee': 2, 'ant': 3}, outcomes=['bee', 'ant'])
        r = repr(t)
        self.assertLess(r.index('bee'), r.index('ant'))

    def test_repr_shows_frequency_header(self):
        t = Table({'a': 1, 'b': 2})
        self.assertIn('Frequency', repr(t))

    def test_repr_shows_relative_frequency_when_normalized(self):
        t = Table({'a': 1, 'b': 2}, normalize=True)
        self.assertIn('Relative Frequency', repr(t))

    def test_repr_shows_custom_outcome_column(self):
        t = Table({'a': 1}, outcome_column='Category')
        self.assertIn('Category', repr(t))

    def test_html_structure(self):
        t = Table({'a': 1, 'b': 2})
        html = t._repr_html_()
        self.assertIn('<table>', html)
        self.assertIn('</table>', html)
        self.assertIn('Total', html)

    def test_large_table_truncates(self):
        t = Table({i: i for i in range(25)})
        self.assertIn('...', repr(t))

    def test_small_table_no_truncation(self):
        t = Table({i: i for i in range(5)})
        self.assertNotIn('...', repr(t))


class TestTableArithmetic(unittest.TestCase):

    def test_add_scalar(self):
        t = Table({'a': 10, 'b': 20})
        result = t + 5
        self.assertEqual(result['a'], 15)
        self.assertEqual(result['b'], 25)

    def test_multiply_scalar(self):
        t = Table({'a': 10, 'b': 20})
        result = t * 2
        self.assertEqual(result['a'], 20)
        self.assertEqual(result['b'], 40)

    def test_divide_scalar(self):
        t = Table({'a': 10, 'b': 20})
        result = t / 2
        self.assertEqual(result['a'], 5.0)
        self.assertEqual(result['b'], 10.0)

    def test_subtract_scalar(self):
        t = Table({'a': 10, 'b': 20})
        result = t - 3
        self.assertEqual(result['a'], 7)
        self.assertEqual(result['b'], 17)

    def test_result_is_table(self):
        t = Table({'a': 1, 'b': 2})
        self.assertIsInstance(t * 2, Table)

    def test_arithmetic_preserves_custom_column_in_display(self):
        t = Table({'ant': 10, 'bee': 20}, outcome_column='Category')
        result = t * 2
        self.assertIn('Category', repr(result))

    def test_arithmetic_preserves_outcomes_order_in_display(self):
        t = Table({'bee': 10, 'ant': 20}, outcomes=['bee', 'ant'])
        result = t + 1
        r = repr(result)
        self.assertLess(r.index('bee'), r.index('ant'))


class TestTableFromTabulate(unittest.TestCase):

    def test_tabulate_returns_table(self):
        distributions.rng = np.random.default_rng(42)
        result = RV(Bernoulli(0.5)).sim(100).tabulate()
        self.assertIsInstance(result, Table)

    def test_tabulate_counts_sum_to_n(self):
        distributions.rng = np.random.default_rng(42)
        n = 200
        result = RV(Bernoulli(0.5)).sim(n).tabulate()
        self.assertEqual(sum(result.values()), n)

    def test_tabulate_keys_are_outcomes(self):
        distributions.rng = np.random.default_rng(42)
        result = RV(Bernoulli(0.5)).sim(100).tabulate()
        self.assertIn(0, result)
        self.assertIn(1, result)

    def test_tabulate_normalize_sums_to_one(self):
        distributions.rng = np.random.default_rng(42)
        result = RV(Bernoulli(0.5)).sim(100).tabulate(normalize=True)
        self.assertAlmostEqual(sum(result.values()), 1.0)


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

class TestTableNormalizeErrors(unittest.TestCase):

    def test_normalize_zero_sum_raises_value_error(self):
        with self.assertRaises(ValueError):
            Table({'a': 0, 'b': 0}, normalize=True)

    def test_normalize_zero_sum_message(self):
        with self.assertRaisesRegex(ValueError, "sum to zero"):
            Table({'a': 0, 'b': 0}, normalize=True)

    def test_normalize_empty_hash_map_raises_value_error(self):
        with self.assertRaises(ValueError):
            Table({}, normalize=True)


class TestTableEmptyRepr(unittest.TestCase):

    def test_repr_empty_table_does_not_crash(self):
        repr(Table({}))

    def test_repr_empty_table_contains_empty_marker(self):
        self.assertIn('(empty)', repr(Table({})))

    def test_repr_empty_table_shows_outcome_header(self):
        self.assertIn('Outcome', repr(Table({})))

    def test_repr_empty_table_shows_frequency_header(self):
        self.assertIn('Frequency', repr(Table({})))

    def test_repr_empty_table_custom_outcome_column(self):
        self.assertIn('Category', repr(Table({}, outcome_column='Category')))

    def test_repr_empty_table_normalized_header(self):
        # normalize=True on all-zero table raises before repr is called,
        # but a normalized table that happens to be empty after construction
        # should still show Relative Frequency — verify via outcomes kwarg
        t = Table({'a': 1}, outcomes=[], normalize=False)
        t.value_column = 'Relative Frequency'
        self.assertIn('Relative Frequency', repr(t))


if __name__ == '__main__':
    unittest.main()
