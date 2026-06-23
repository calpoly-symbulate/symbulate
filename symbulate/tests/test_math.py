"""Tests for symbulate.math.

Covers scalar math operations, statistical summary functions,
quantile/order statistics, counting helpers, and TypeError/ValueError
guards. All tests use concrete lists with known correct answers.
"""
import math
import unittest

from symbulate.math import (
    sqrt, exp, log, sin, cos, factorial,
    mean, var, sd, median, min_max_diff, med_abs_dev,
    quantile, iqr, orderstatistics,
    skewness, kurtosis, moment, trimmed_mean,
    count, count_eq, count_neq, count_lt, count_gt, count_geq, count_leq,
    comparefun,
    interarrival_times, arrival_times, states,
)


class TestScalarMath(unittest.TestCase):

    def test_sqrt_scalar(self):
        self.assertAlmostEqual(sqrt(9), 3.0)

    def test_exp_scalar(self):
        self.assertAlmostEqual(exp(1), math.e)

    def test_log_natural(self):
        self.assertAlmostEqual(log(math.e), 1.0)

    def test_log_base_2(self):
        self.assertAlmostEqual(log(8, 2), 3.0)

    def test_log_base_10(self):
        self.assertAlmostEqual(log(100, 10), 2.0)

    def test_sin_zero(self):
        self.assertAlmostEqual(sin(0), 0.0)

    def test_cos_zero(self):
        self.assertAlmostEqual(cos(0), 1.0)

    def test_factorial_five(self):
        self.assertEqual(factorial(5), 120)


class TestStatisticalFunctions(unittest.TestCase):

    DATA = [2, 4, 4, 4, 5, 5, 7, 9]
    SIMPLE = [1, 2, 3, 4, 5]

    def test_mean(self):
        self.assertAlmostEqual(mean(self.SIMPLE), 3.0)

    def test_mean_docstring_example(self):
        self.assertAlmostEqual(mean([1, 2, 3, 4, 5]), 3.0)

    def test_var_docstring_example(self):
        self.assertAlmostEqual(var(self.DATA), 4.0)

    def test_sd_docstring_example(self):
        self.assertAlmostEqual(sd(self.DATA), 2.0)

    def test_median_odd_length(self):
        self.assertAlmostEqual(median(self.SIMPLE), 3.0)

    def test_median_even_length(self):
        self.assertAlmostEqual(median([1, 2, 3, 4]), 2.5)

    def test_min_max_diff(self):
        self.assertEqual(min_max_diff(self.SIMPLE), 4)

    def test_med_abs_dev(self):
        self.assertAlmostEqual(med_abs_dev(self.SIMPLE), 1.0)

    def test_sd_equals_sqrt_var(self):
        data = [3, 7, 7, 19]
        self.assertAlmostEqual(sd(data), math.sqrt(var(data)))


class TestStatisticalErrors(unittest.TestCase):

    def test_mean_single_value_raises(self):
        with self.assertRaises(TypeError):
            mean(3)

    def test_median_single_value_raises(self):
        with self.assertRaises(TypeError):
            median(3)

    def test_min_max_diff_single_value_raises(self):
        with self.assertRaises(TypeError):
            min_max_diff(3)

    def test_iqr_single_value_raises(self):
        with self.assertRaises(TypeError):
            iqr(3)

    def test_skewness_single_value_raises(self):
        with self.assertRaises(TypeError):
            skewness(3)

    def test_kurtosis_single_value_raises(self):
        with self.assertRaises(TypeError):
            kurtosis(3)


class TestQuantileFunctions(unittest.TestCase):

    DATA = [1, 2, 3, 4, 5]

    def test_quantile_25th(self):
        self.assertAlmostEqual(quantile(0.25)(self.DATA), 2.0)

    def test_quantile_50th(self):
        self.assertAlmostEqual(quantile(0.5)(self.DATA), 3.0)

    def test_quantile_75th(self):
        self.assertAlmostEqual(quantile(0.75)(self.DATA), 4.0)

    def test_quantile_0_is_min(self):
        self.assertAlmostEqual(quantile(0)(self.DATA), 1.0)

    def test_quantile_1_is_max(self):
        self.assertAlmostEqual(quantile(1)(self.DATA), 5.0)

    def test_iqr(self):
        self.assertAlmostEqual(iqr(self.DATA), 2.0)

    def test_orderstatistics_first(self):
        self.assertEqual(orderstatistics(1)([5, 3, 1, 4, 2]), 1)

    def test_orderstatistics_second(self):
        self.assertEqual(orderstatistics(2)([5, 3, 1, 4, 2]), 2)

    def test_orderstatistics_zero_raises(self):
        with self.assertRaises(ValueError):
            orderstatistics(0)

    def test_orderstatistics_negative_raises(self):
        with self.assertRaises(ValueError):
            orderstatistics(-1)


class TestHigherOrderStats(unittest.TestCase):

    def test_skewness_symmetric_is_zero(self):
        # perfectly symmetric data has skewness 0
        self.assertAlmostEqual(skewness([1, 2, 3, 4, 5]), 0.0, places=10)

    def test_skewness_returns_float(self):
        result = skewness([1, 1, 2, 3, 5, 8])
        self.assertIsInstance(result, float)

    def test_kurtosis_uniform_is_negative(self):
        # uniform-like data has negative excess kurtosis
        self.assertLess(kurtosis([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 0)

    def test_moment_second_equals_var(self):
        data = [1, 2, 3, 4, 5]
        self.assertAlmostEqual(moment(2)(data), var(data))

    def test_moment_first_is_zero_for_symmetric(self):
        # first central moment is always 0
        self.assertAlmostEqual(moment(1)([1, 2, 3, 4, 5]), 0.0, places=10)

    def test_trimmed_mean_removes_extremes(self):
        # trimming the top and bottom 20% of [1,2,3,4,5] leaves [2,3,4]
        self.assertAlmostEqual(trimmed_mean(0.2)([1, 2, 3, 4, 5]), 3.0)


class TestCountingFunctions(unittest.TestCase):

    DATA = [1, 2, 3, 3, 5]
    RANGE = [1, 2, 3, 4, 5]

    def test_count_eq(self):
        self.assertEqual(count_eq(3)(self.DATA), 2)

    def test_count_neq(self):
        self.assertEqual(count_neq(3)(self.DATA), 3)

    def test_count_lt(self):
        self.assertEqual(count_lt(3)(self.RANGE), 2)

    def test_count_gt(self):
        self.assertEqual(count_gt(3)(self.RANGE), 2)

    def test_count_geq(self):
        self.assertEqual(count_geq(3)(self.RANGE), 3)

    def test_count_leq(self):
        self.assertEqual(count_leq(3)(self.RANGE), 3)

    def test_count_with_predicate(self):
        self.assertEqual(count(lambda x: x > 3)(self.RANGE), 2)

    def test_count_default_counts_all(self):
        self.assertEqual(count()(self.RANGE), 5)

    def test_comparefun_gt(self):
        import operator
        self.assertEqual(comparefun(self.RANGE, operator.gt, 3), 2)


class TestProcessHelperTypeErrors(unittest.TestCase):

    def test_interarrival_times_rejects_plain_list(self):
        with self.assertRaises(TypeError):
            interarrival_times([1, 2, 3])

    def test_arrival_times_rejects_plain_list(self):
        with self.assertRaises(TypeError):
            arrival_times([1, 2, 3])

    def test_states_rejects_plain_list(self):
        with self.assertRaises(TypeError):
            states([1, 2, 3])


if __name__ == '__main__':
    unittest.main()
