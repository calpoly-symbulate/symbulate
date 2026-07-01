"""Tests for symbulate.math.

Covers scalar math operations, statistical summary functions,
quantile/order statistics, counting helpers, and TypeError/ValueError
guards. All tests use concrete lists with known correct answers.
"""

import math
import unittest

from symbulate import RV, Normal, Binomial

from symbulate.math import (
    sqrt,
    exp,
    log,
    sin,
    cos,
    factorial,
    mean,
    var,
    sd,
    median,
    min_max_diff,
    med_abs_dev,
    quantile,
    quartiles,
    deciles,
    is_discrete,
    iqr,
    orderstatistics,
    skewness,
    kurtosis,
    moment,
    trimmed_mean,
    count,
    count_eq,
    count_neq,
    count_lt,
    count_gt,
    count_geq,
    count_leq,
    comparefun,
    interarrival_times,
    arrival_times,
    states,
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

    def test_log_base_zero_raises(self):
        with self.assertRaises(ValueError):
            log(8, 0)

    def test_log_base_one_raises(self):
        with self.assertRaises(ValueError):
            log(8, 1)

    def test_log_base_negative_raises(self):
        with self.assertRaises(ValueError):
            log(8, -2)

    def test_log_base_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            log(8, "e")


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

    def test_mean_empty_raises(self):
        with self.assertRaises(ValueError):
            mean([])

    def test_mean_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            mean(None)

    def test_mean_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            mean(["a", "b"])

    def test_median_empty_raises(self):
        with self.assertRaises(ValueError):
            median([])

    def test_median_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            median(None)

    def test_median_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            median(["a", "b"])

    def test_min_max_diff_empty_raises(self):
        with self.assertRaises(ValueError):
            min_max_diff([])

    def test_min_max_diff_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            min_max_diff(None)

    def test_min_max_diff_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            min_max_diff(["a", "b"])

    def test_var_empty_raises(self):
        with self.assertRaises(ValueError):
            var([])

    def test_var_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            var(None)

    def test_var_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            var(["a", "b"])

    def test_sd_empty_raises(self):
        with self.assertRaises(ValueError):
            sd([])

    def test_sd_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            sd(None)

    def test_sd_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            sd(["a", "b"])

    def test_iqr_empty_raises(self):
        with self.assertRaises(ValueError):
            iqr([])

    def test_iqr_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            iqr(None)

    def test_iqr_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            iqr(["a", "b"])

    def test_skewness_empty_raises(self):
        with self.assertRaises(ValueError):
            skewness([])

    def test_skewness_too_few_raises(self):
        with self.assertRaises(ValueError):
            skewness([1, 2])

    def test_skewness_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            skewness(None)

    def test_skewness_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            skewness(["a", "b", "c"])

    def test_kurtosis_empty_raises(self):
        with self.assertRaises(ValueError):
            kurtosis([])

    def test_kurtosis_too_few_raises(self):
        with self.assertRaises(ValueError):
            kurtosis([1, 2, 3])

    def test_kurtosis_non_iterable_raises(self):
        with self.assertRaises(TypeError):
            kurtosis(None)

    def test_kurtosis_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            kurtosis(["a", "b", "c", "d"])


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

    def test_quantile_above_one_raises(self):
        with self.assertRaises(ValueError):
            quantile(25)

    def test_quantile_negative_raises(self):
        with self.assertRaises(ValueError):
            quantile(-0.1)

    def test_quartiles_keys(self):
        self.assertEqual(set(quartiles(self.DATA).keys()), {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_quartiles_values(self):
        q = quartiles(self.DATA)
        self.assertAlmostEqual(q[0.0], 1.0)
        self.assertAlmostEqual(q[0.25], 2.0)
        self.assertAlmostEqual(q[0.5], 3.0)
        self.assertAlmostEqual(q[0.75], 4.0)
        self.assertAlmostEqual(q[1.0], 5.0)

    def test_deciles_keys(self):
        self.assertEqual(
            set(deciles(self.DATA).keys()),
            {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
        )

    def test_deciles_min_max(self):
        d = deciles(self.DATA)
        self.assertAlmostEqual(d[0.0], 1.0)
        self.assertAlmostEqual(d[1.0], 5.0)

    def test_deciles_median(self):
        d = deciles(self.DATA)
        self.assertAlmostEqual(d[0.5], 3.0)

    def test_quartiles_empty_raises(self):
        with self.assertRaises(ValueError):
            quartiles([])

    def test_deciles_empty_raises(self):
        with self.assertRaises(ValueError):
            deciles([])

    def test_quartiles_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            quartiles([1, 2, "a"])

    def test_deciles_non_numeric_raises(self):
        with self.assertRaises(TypeError):
            deciles([1, 2, "a"])


class TestIsDiscrete(unittest.TestCase):

    def test_integer_list_is_discrete(self):
        self.assertTrue(is_discrete([0, 1, 1, 0, 1]))

    def test_all_unique_values_not_discrete(self):
        # no repeats -> 0% of distinct values have count > 1
        self.assertFalse(is_discrete([0.0, 1.0, 2.0]))

    def test_repeated_values_discrete(self):
        # majority of distinct values appear more than once
        self.assertTrue(is_discrete([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]))

    def test_single_element_not_discrete(self):
        # one value cannot repeat
        self.assertFalse(is_discrete([1]))

    def test_unhashable_values_raises(self):
        with self.assertRaises(TypeError):
            is_discrete([[1, 2], [3, 4]])

    def test_rv_input_raises(self):
        with self.assertRaises(TypeError):
            is_discrete(RV(Normal(0, 1)))

    def test_scalar_input_raises(self):
        with self.assertRaises(TypeError):
            is_discrete(3)

    def test_none_input_raises(self):
        with self.assertRaises(TypeError):
            is_discrete(None)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            is_discrete([])

    def test_non_numeric_hashable_works(self):
        # heuristic is type-agnostic; strings with repeats are treated as discrete
        self.assertTrue(is_discrete(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"]))

    def test_binomial_simulation_is_discrete(self):
        results = RV(Binomial(n=10, p=0.5)).sim(500)
        self.assertTrue(is_discrete(results))

    def test_normal_simulation_not_discrete(self):
        results = RV(Normal(0, 1)).sim(500)
        self.assertFalse(is_discrete(results))


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

    def test_trimmed_mean_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            trimmed_mean(0.6)

    def test_moment_non_integer_k_raises(self):
        with self.assertRaises(TypeError):
            moment(2.5)

    def test_moment_negative_k_raises(self):
        with self.assertRaises(ValueError):
            moment(-1)


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

    def test_count_non_callable_raises_type_error(self):
        with self.assertRaises(TypeError):
            count(5)

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


if __name__ == "__main__":
    unittest.main()
