"""Tests for symbulate.results.

Covers:
- _is_hashable: hashable vs unhashable objects
- _is_boolean_vector: all-bool vs mixed vs non-bool
- _sim_with_progress: draw count/order, no output when fast, throttled
  redraws (at most ~101 total) when slow, instead of one per iteration
- Results: init, len, iter, get, apply, __getitem__, filter, _get_counts,
  tabulate, arithmetic ops, comparison ops, logical ops, _statistic_factory
  raises, plot raises, __repr__, _repr_html_
- RVResults: init (dim and index_set detection), _set_array, univariate
  statistics (mean, std, var), multivariate statistics (cov), standardize,
  tabulate
"""

import contextlib
import io
import sys
import unittest

import numpy as np

from symbulate.result import Scalar, Vector
from symbulate.results import (
    Results,
    RVResults,
    _is_boolean_vector,
    _is_hashable,
    _sim_with_progress,
)

SIM_ID = 42.0


def make_results(values, sim_id=SIM_ID):
    return Results(values, sim_id=sim_id)


# ---------------------------------------------------------------------------
# _is_hashable
# ---------------------------------------------------------------------------


class TestIsHashable(unittest.TestCase):

    def test_int(self):
        self.assertTrue(_is_hashable(1))

    def test_string(self):
        self.assertTrue(_is_hashable("hello"))

    def test_tuple(self):
        self.assertTrue(_is_hashable((1, 2)))

    def test_list_is_not_hashable(self):
        self.assertFalse(_is_hashable([1, 2]))

    def test_none(self):
        self.assertTrue(_is_hashable(None))


# ---------------------------------------------------------------------------
# _is_boolean_vector
# ---------------------------------------------------------------------------


class TestIsBooleanVector(unittest.TestCase):

    def test_all_bool(self):
        self.assertTrue(_is_boolean_vector([True, False, True]))

    def test_numpy_bool(self):
        self.assertTrue(_is_boolean_vector([np.bool_(True), np.bool_(False)]))

    def test_mixed_bool_and_int(self):
        self.assertFalse(_is_boolean_vector([True, 1, False]))

    def test_all_int(self):
        self.assertFalse(_is_boolean_vector([0, 1, 0]))

    def test_empty(self):
        self.assertTrue(_is_boolean_vector([]))


# ---------------------------------------------------------------------------
# _sim_with_progress
# ---------------------------------------------------------------------------


class TestSimWithProgress(unittest.TestCase):
    """draw_func is called n times regardless of whether the bar shows, and
    the bar (once showing) redraws only when the displayed percentage
    changes -- not on every draw, which was measured to add up to ~40x
    overhead on cheap draws from the write+flush call alone."""

    def _run_capturing_stderr(self, n, progress_delay):
        buf = io.StringIO()
        real_stderr = sys.stderr
        sys.stderr = buf
        try:
            draws = _sim_with_progress(lambda: 1, n, progress_delay=progress_delay)
        finally:
            sys.stderr = real_stderr
        return draws, buf.getvalue()

    def test_returns_all_draws_in_order(self):
        counter = iter(range(100))
        draws = _sim_with_progress(lambda: next(counter), 100, progress_delay=999)
        self.assertEqual(draws, list(range(100)))

    def test_no_output_when_faster_than_progress_delay(self):
        draws, output = self._run_capturing_stderr(50, progress_delay=999)
        self.assertEqual(len(draws), 50)
        self.assertEqual(output, "")

    def test_shows_bar_reaching_100_percent_when_slower_than_delay(self):
        draws, output = self._run_capturing_stderr(50, progress_delay=0)
        self.assertEqual(len(draws), 50)
        self.assertIn("100%", output)
        self.assertTrue(output.endswith("\n"))

    def test_redraws_are_throttled_not_once_per_iteration(self):
        """Regression test: redraw count must stay near the number of
        distinct percentage points (~101), not scale with n."""
        n = 10_000
        _, output = self._run_capturing_stderr(n, progress_delay=0)
        n_redraws = output.count("\r")
        self.assertLess(
            n_redraws, 110, f"{n_redraws} redraws for n={n} -- throttling isn't working"
        )
        self.assertGreater(n_redraws, 90)

    def test_redraw_count_scales_with_percent_points_not_n(self):
        """A 10x larger n should not produce meaningfully more redraws,
        since both are throttled to ~one redraw per percentage point."""
        _, output_small = self._run_capturing_stderr(1_000, progress_delay=0)
        _, output_large = self._run_capturing_stderr(10_000, progress_delay=0)
        self.assertLess(output_large.count("\r") - output_small.count("\r"), 10)


# ---------------------------------------------------------------------------
# Results.__init__
# ---------------------------------------------------------------------------


class TestResultsInit(unittest.TestCase):

    def test_stores_list(self):
        r = Results([1, 2, 3])
        self.assertEqual(r.results, [1, 2, 3])

    def test_explicit_sim_id(self):
        r = Results([1, 2], sim_id=7.0)
        self.assertEqual(r.sim_id, 7.0)

    def test_auto_sim_id_is_float(self):
        r = Results([1, 2])
        self.assertIsInstance(r.sim_id, float)

    def test_from_generator(self):
        r = Results(x for x in range(3))
        self.assertEqual(len(r), 3)


# ---------------------------------------------------------------------------
# Results.__len__ and __iter__
# ---------------------------------------------------------------------------


class TestResultsLenIter(unittest.TestCase):

    def test_len(self):
        self.assertEqual(len(make_results([10, 20, 30])), 3)

    def test_len_empty(self):
        self.assertEqual(len(make_results([])), 0)

    def test_iter(self):
        self.assertEqual(list(make_results([1, 2, 3])), [1, 2, 3])


# ---------------------------------------------------------------------------
# Results.get
# ---------------------------------------------------------------------------


class TestResultsGet(unittest.TestCase):

    def test_get_int(self):
        r = make_results([10, 20, 30])
        self.assertEqual(r.get(0), 10)
        self.assertEqual(r.get(2), 30)

    def test_get_slice(self):
        r = make_results([10, 20, 30, 40])
        self.assertEqual(r.get(slice(1, 3)), [20, 30])

    def test_get_numeric_vector(self):
        r = make_results([10, 20, 30, 40])
        sub = r.get([0, 2])
        self.assertIsInstance(sub, Results)
        self.assertEqual(list(sub), [10, 30])


# ---------------------------------------------------------------------------
# Results.apply
# ---------------------------------------------------------------------------


class TestResultsApply(unittest.TestCase):

    def test_apply_doubles(self):
        result = make_results([1, 2, 3]).apply(lambda x: x * 2)
        self.assertEqual(list(result), [2, 4, 6])

    def test_apply_returns_results(self):
        result = make_results([1, 2, 3]).apply(lambda x: x)
        self.assertIsInstance(result, Results)

    def test_apply_preserves_sim_id(self):
        r = Results([1, 2, 3], sim_id=99.0)
        self.assertEqual(r.apply(lambda x: x).sim_id, 99.0)

    def test_apply_non_callable_raises_type_error(self):
        with self.assertRaises(TypeError):
            make_results([1, 2, 3]).apply(5)


# ---------------------------------------------------------------------------
# Results.__getitem__
# ---------------------------------------------------------------------------


class TestResultsGetItem(unittest.TestCase):

    def test_getitem_int_indexes_into_each_result(self):
        r = Results([[10, 20], [30, 40], [50, 60]], sim_id=SIM_ID)
        self.assertEqual(list(r[0]), [10, 30, 50])

    def test_getitem_boolean_results_mask(self):
        r = make_results([1, 2, 3, 4, 5])
        mask = Results([True, False, True, False, True], sim_id=SIM_ID)
        self.assertEqual(list(r[mask]), [1, 3, 5])


# ---------------------------------------------------------------------------
# Results.filter
# ---------------------------------------------------------------------------


class TestResultsFilter(unittest.TestCase):

    def test_filter_callable(self):
        r = make_results([1, 2, 3, 4, 5])
        self.assertEqual(list(r.filter(lambda x: x > 3)), [4, 5])

    def test_filter_boolean_results(self):
        r = make_results([10, 20, 30])
        mask = Results([True, False, True], sim_id=SIM_ID)
        self.assertEqual(list(r.filter(mask)), [10, 30])

    def test_filter_wrong_sim_id_raises(self):
        r = Results([1, 2, 3], sim_id=1.0)
        mask = Results([True, False, True], sim_id=2.0)
        with self.assertRaises(Exception):
            r.filter(mask)

    def test_filter_wrong_length_raises(self):
        r = make_results([1, 2, 3])
        mask = Results([True, False], sim_id=SIM_ID)
        with self.assertRaises(ValueError):
            r.filter(mask)

    def test_filter_non_boolean_results_raises(self):
        r = make_results([1, 2, 3])
        mask = Results([1, 0, 1], sim_id=SIM_ID)
        with self.assertRaises(ValueError):
            r.filter(mask)

    def test_filter_bad_type_raises(self):
        with self.assertRaises(TypeError):
            make_results([1, 2, 3]).filter("not callable or Results")


# ---------------------------------------------------------------------------
# Results._get_counts
# ---------------------------------------------------------------------------


class TestResultsGetCounts(unittest.TestCase):

    def test_counts_hashable(self):
        counts = make_results(["H", "T", "H"])._get_counts()
        self.assertEqual(counts["H"], 2)
        self.assertEqual(counts["T"], 1)

    def test_counts_list_of_hashables_becomes_tuple(self):
        counts = make_results([[1, 2], [1, 2], [3, 4]])._get_counts()
        self.assertEqual(counts[(1, 2)], 2)
        self.assertEqual(counts[(3, 4)], 1)

    def test_counts_total_matches_length(self):
        r = make_results([1, 2, 1, 3, 2, 1])
        self.assertEqual(sum(r._get_counts().values()), 6)


# ---------------------------------------------------------------------------
# Results.tabulate
# ---------------------------------------------------------------------------


class TestResultsTabulate(unittest.TestCase):

    def test_tabulate_counts(self):
        table = make_results(["H", "T", "H", "H"]).tabulate()
        self.assertEqual(table["H"], 3)
        self.assertEqual(table["T"], 1)

    def test_tabulate_normalize(self):
        table = make_results(["H", "T", "H"]).tabulate(normalize=True)
        self.assertAlmostEqual(table["H"], 2 / 3)

    def test_tabulate_with_outcomes_includes_zeros(self):
        table = make_results(["H", "H"]).tabulate(outcomes=["H", "T"])
        self.assertEqual(table["H"], 2)
        self.assertEqual(table["T"], 0)

    def test_tabulate_bin_creates_ten_equal_width_bins(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(bin=True)
        self.assertEqual(len(table), 10)

    def test_tabulate_bin_total_equals_n(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(bin=True)
        self.assertEqual(sum(table.values()), 100)

    def test_tabulate_bin_uses_bin_column_and_string_labels(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(bin=True)
        self.assertEqual(table.outcome_column, "Bin")
        self.assertTrue(all(isinstance(key, str) for key in table.keys()))

    def test_tabulate_bin_normalize_sums_to_one(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(
            bin=True, normalize=True
        )
        self.assertAlmostEqual(sum(table.values()), 1.0)

    def test_tabulate_bin_equal_counts_for_uniform_data(self):
        # 100 evenly-spaced floats → 10 equal-width bins should each get 10 counts
        data = [float(i) for i in range(0, 100)]
        table = make_results(data).tabulate(bin=True)
        counts = list(table.values())
        self.assertEqual(min(counts), max(counts))

    def test_tabulate_bin_last_label_closed(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(bin=True)
        labels = list(table.keys())
        self.assertTrue(labels[-1].endswith("]"))

    def test_tabulate_bin_non_last_labels_half_open(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(bin=True)
        labels = list(table.keys())
        self.assertTrue(all(label.endswith(")") for label in labels[:-1]))

    def test_tabulate_nbins_creates_correct_number_of_bins(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(
            bin=True, nbins=5
        )
        self.assertEqual(len(table), 5)

    def test_tabulate_nbins_total_equals_n(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(
            bin=True, nbins=5
        )
        self.assertEqual(sum(table.values()), 100)

    def test_tabulate_binwidth_creates_correct_number_of_bins(self):
        # range is 1..100 = 99, binwidth=10 → ceil(99/10) = 10 bins
        table = make_results([float(i) for i in range(1, 101)]).tabulate(
            bin=True, binwidth=10
        )
        self.assertEqual(len(table), 10)

    def test_tabulate_binwidth_total_equals_n(self):
        table = make_results([float(i) for i in range(1, 101)]).tabulate(
            bin=True, binwidth=10
        )
        self.assertEqual(sum(table.values()), 100)

    def test_tabulate_nbins_and_binwidth_raises(self):
        with self.assertRaises(ValueError):
            make_results([float(i) for i in range(1, 101)]).tabulate(
                bin=True, nbins=5, binwidth=10
            )

    def test_tabulate_nbins_without_bin_warns(self):
        with self.assertWarns(UserWarning):
            make_results([float(i) for i in range(1, 101)]).tabulate(nbins=5)

    def test_tabulate_binwidth_without_bin_warns(self):
        with self.assertWarns(UserWarning):
            make_results([float(i) for i in range(1, 101)]).tabulate(binwidth=10)


# ---------------------------------------------------------------------------
# Results arithmetic
# ---------------------------------------------------------------------------


class TestResultsArithmetic(unittest.TestCase):

    def test_add_scalar(self):
        self.assertEqual(list(make_results([1, 2, 3]) + 10), [11, 12, 13])

    def test_add_results(self):
        r1 = make_results([1, 2, 3])
        r2 = make_results([4, 5, 6])
        self.assertEqual(list(r1 + r2), [5, 7, 9])

    def test_add_mismatched_length_raises(self):
        with self.assertRaises(Exception):
            make_results([1, 2, 3]) + make_results([4, 5])

    def test_add_different_sim_id_raises(self):
        r1 = Results([1, 2, 3], sim_id=1.0)
        r2 = Results([4, 5, 6], sim_id=2.0)
        with self.assertRaises(Exception):
            r1 + r2

    def test_sub(self):
        self.assertEqual(list(make_results([5, 7, 9]) - 2), [3, 5, 7])

    def test_mul(self):
        self.assertEqual(list(make_results([2, 3, 4]) * 3), [6, 9, 12])


# ---------------------------------------------------------------------------
# Results comparison
# ---------------------------------------------------------------------------


class TestResultsComparison(unittest.TestCase):

    def test_gt_scalar(self):
        mask = make_results([1, 2, 3, 4, 5]) > 3
        self.assertEqual(list(mask), [False, False, False, True, True])

    def test_lt_scalar(self):
        mask = make_results([1, 2, 3]) < 2
        self.assertEqual(list(mask), [True, False, False])

    def test_eq_scalar(self):
        mask = make_results([1, 2, 3]) == 2
        self.assertEqual(list(mask), [False, True, False])


# ---------------------------------------------------------------------------
# Results logical
# ---------------------------------------------------------------------------


class TestResultsLogical(unittest.TestCase):

    def _bools(self, values):
        return Results(values, sim_id=SIM_ID)

    def test_and(self):
        result = self._bools([True, False, True]) & self._bools([True, True, False])
        self.assertEqual(list(result), [True, False, False])

    def test_or(self):
        result = self._bools([True, False, False]) | self._bools([False, True, False])
        self.assertEqual(list(result), [True, True, False])

    def test_not(self):
        result = ~self._bools([True, False, True])
        self.assertEqual(list(result), [False, True, False])

    def test_and_non_boolean_self_raises(self):
        with self.assertRaises(ValueError):
            make_results([1, 0, 1]) & self._bools([True, False, True])

    def test_and_non_boolean_other_raises(self):
        with self.assertRaises(ValueError):
            self._bools([True, False, True]) & make_results([1, 0, 1])

    def test_and_non_results_other_raises(self):
        with self.assertRaises(TypeError):
            self._bools([True, False]) & True

    def test_and_different_sim_id_raises(self):
        b1 = Results([True, False], sim_id=1.0)
        b2 = Results([True, True], sim_id=2.0)
        with self.assertRaises(Exception):
            b1 & b2


# ---------------------------------------------------------------------------
# Results._statistic_factory raises
# ---------------------------------------------------------------------------


class TestResultsStatisticFactory(unittest.TestCase):

    def test_mean_raises(self):
        with self.assertRaises(Exception):
            make_results([1, 2, 3]).mean()

    def test_std_raises(self):
        with self.assertRaises(Exception):
            make_results([1, 2, 3]).std()


# ---------------------------------------------------------------------------
# Results.plot raises
# ---------------------------------------------------------------------------


class TestResultsPlot(unittest.TestCase):

    def test_plot_raises(self):
        with self.assertRaises(Exception):
            make_results(["H", "T"]).plot()


# ---------------------------------------------------------------------------
# Results.__repr__ and _repr_html_
# ---------------------------------------------------------------------------


class TestResultsRepr(unittest.TestCase):

    def test_repr_contains_header(self):
        s = repr(make_results([1, 2, 3]))
        self.assertIn("Index", s)
        self.assertIn("Result", s)

    def test_repr_contains_values(self):
        s = repr(make_results([10, 20]))
        self.assertIn("10", s)
        self.assertIn("20", s)

    def test_repr_long_truncates(self):
        # repr dots match max_index_length (2 digits for indices 0-19 → "..")
        r = make_results(list(range(20)))
        s = repr(r)
        line_count = s.count("\n") + 1
        # header + 9 data rows + ellipsis line + last row = 12 lines (< 21)
        self.assertLess(line_count, 21)

    def test_repr_eleven_shows_all(self):
        s = repr(make_results(list(range(11))))
        self.assertNotIn("...", s)

    def test_repr_html_contains_table_tags(self):
        html = make_results([10, 20])._repr_html_()
        self.assertIn("<table>", html)
        self.assertIn("10", html)

    def test_repr_html_long_shows_ellipsis(self):
        html = make_results(list(range(20)))._repr_html_()
        self.assertIn("...", html)


# ---------------------------------------------------------------------------
# RVResults.plot
# ---------------------------------------------------------------------------


class TestRVResultsPlot(unittest.TestCase):

    def test_plot_invalid_type_raises(self):
        rvr = RVResults([1.0, 2.0, 3.0])
        with self.assertRaises(Exception):
            rvr.plot(type=42)


# ---------------------------------------------------------------------------
# RVResults.__init__
# ---------------------------------------------------------------------------


class TestRVResultsInit(unittest.TestCase):

    def test_dim_1_for_scalars(self):
        self.assertEqual(RVResults([1.0, 2.0, 3.0]).dim, 1)

    def test_dim_2_for_length_2_vectors(self):
        rvr = RVResults([Vector([1.0, 2.0]), Vector([3.0, 4.0])])
        self.assertEqual(rvr.dim, 2)

    def test_dim_none_for_strings(self):
        self.assertIsNone(RVResults(["H", "T", "H"]).dim)

    def test_dim_none_for_inconsistent(self):
        self.assertIsNone(RVResults([1.0, Vector([1.0, 2.0])]).dim)

    def test_dim_none_for_empty(self):
        self.assertIsNone(RVResults([]).dim)

    def test_index_set_none_for_scalars(self):
        self.assertIsNone(RVResults([1.0, 2.0]).index_set)


# ---------------------------------------------------------------------------
# RVResults._set_array
# ---------------------------------------------------------------------------


class TestRVResultsSetArray(unittest.TestCase):

    def test_set_array_scalars(self):
        rvr = RVResults([1.0, 2.0, 3.0])
        rvr._set_array()
        np.testing.assert_array_equal(rvr.array, [1.0, 2.0, 3.0])

    def test_set_array_caches(self):
        rvr = RVResults([1.0, 2.0, 3.0])
        rvr._set_array()
        arr = rvr.array
        rvr._set_array()
        self.assertIs(rvr.array, arr)

    def test_set_array_raises_for_non_numeric(self):
        with self.assertRaises(Exception):
            RVResults(["H", "T"])._set_array()


# ---------------------------------------------------------------------------
# RVResults statistics (univariate)
# ---------------------------------------------------------------------------


class TestRVResultsUnivariate(unittest.TestCase):

    def setUp(self):
        self.values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.rvr = RVResults(self.values)

    def test_mean_returns_scalar(self):
        self.assertIsInstance(self.rvr.mean(), Scalar)

    def test_mean_value(self):
        self.assertAlmostEqual(float(self.rvr.mean()), 3.0)

    def test_std_value(self):
        self.assertAlmostEqual(float(self.rvr.std()), np.std(self.values))

    def test_var_value(self):
        self.assertAlmostEqual(float(self.rvr.var()), np.var(self.values))

    def test_mean_2d_returns_vector(self):
        rvr = RVResults([Vector([1.0, 2.0]), Vector([3.0, 4.0]), Vector([5.0, 6.0])])
        mean = rvr.mean()
        self.assertIsInstance(mean, Vector)
        self.assertAlmostEqual(mean[0], 3.0)
        self.assertAlmostEqual(mean[1], 4.0)

    def test_sum_value(self):
        self.assertAlmostEqual(float(self.rvr.sum()), 15.0)

    def test_min_value(self):
        self.assertAlmostEqual(float(self.rvr.min()), 1.0)

    def test_max_value(self):
        self.assertAlmostEqual(float(self.rvr.max()), 5.0)


# ---------------------------------------------------------------------------
# RVResults statistics (multivariate)
# ---------------------------------------------------------------------------


class TestRVResultsMultivariate(unittest.TestCase):

    def test_cov_positive_for_perfectly_correlated(self):
        # dim0 == dim1 → cov must be positive
        vals = [float(i) for i in range(1, 11)]
        rvr = RVResults([Vector([v, v]) for v in vals])
        self.assertGreater(float(rvr.cov()), 0)

    def test_cov_zero_for_uncorrelated(self):
        # alternating pattern: dim0 constant, dim1 alternating → cov ≈ 0
        rvr = RVResults([Vector([1.0, float(i % 2)]) for i in range(20)])
        self.assertAlmostEqual(float(rvr.cov()), 0.0, places=10)

    def test_cov_raises_for_1d(self):
        with self.assertRaises(Exception):
            RVResults([1.0, 2.0, 3.0]).cov()


# ---------------------------------------------------------------------------
# RVResults.standardize
# ---------------------------------------------------------------------------


class TestRVResultsStandardize(unittest.TestCase):

    def test_standardized_mean_is_zero(self):
        rvr = RVResults([float(i) for i in range(1, 101)])
        std = rvr.standardize()
        self.assertAlmostEqual(float(std.mean()), 0.0, places=10)

    def test_standardized_std_is_one(self):
        rvr = RVResults([float(i) for i in range(1, 101)])
        std = rvr.standardize()
        self.assertAlmostEqual(float(std.std()), 1.0, places=10)

    def test_standardize_raises_for_non_numeric(self):
        with self.assertRaises(Exception):
            RVResults(["H", "T"]).standardize()

    def test_standardize_constant_data_raises_clear_message(self):
        # Regression test: standardizing zero-variance data used to raise
        # a raw ZeroDivisionError instead of an explanatory message.
        rvr = RVResults([5.0] * 20)
        with self.assertRaisesRegex(Exception, "no variability"):
            rvr.standardize()

    def test_standardize_single_draw_raises_clear_message(self):
        # A single value also has zero sample standard deviation.
        rvr = RVResults([3.0])
        with self.assertRaisesRegex(Exception, "no variability"):
            rvr.standardize()


# ---------------------------------------------------------------------------
# RVResults.tabulate
# ---------------------------------------------------------------------------


class TestRVResultsTabulate(unittest.TestCase):

    def test_tabulate_counts(self):
        table = RVResults([1, 1, 2, 3]).tabulate()
        self.assertEqual(table[1], 2)
        self.assertEqual(table[2], 1)

    def test_tabulate_normalize(self):
        table = RVResults([1, 2, 2]).tabulate(normalize=True)
        self.assertAlmostEqual(table[2], 2 / 3)

    def test_tabulate_bin_creates_ten_equal_width_bins(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(bin=True)
        self.assertEqual(len(table), 10)
        self.assertEqual(sum(table.values()), 100)

    def test_tabulate_bin_uses_bin_column(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(bin=True)
        self.assertEqual(table.outcome_column, "Bin")

    def test_tabulate_bin_normalize_sums_to_one(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(
            bin=True, normalize=True
        )
        self.assertAlmostEqual(sum(table.values()), 1.0)

    def test_tabulate_bin_equal_counts_for_uniform_data(self):
        data = [float(i) for i in range(0, 100)]
        table = RVResults(data).tabulate(bin=True)
        counts = list(table.values())
        self.assertEqual(min(counts), max(counts))

    def test_tabulate_bin_labels_span_data_range(self):
        data = [float(i) for i in range(0, 100)]
        table = RVResults(data).tabulate(bin=True)
        labels = list(table.keys())
        # first label starts at 0 and last label ends at 99
        self.assertTrue(labels[0].startswith("[0"))
        self.assertTrue(labels[-1].endswith("]"))

    def test_tabulate_bin_last_label_closed(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(bin=True)
        labels = list(table.keys())
        self.assertTrue(labels[-1].endswith("]"))

    def test_tabulate_bin_non_last_labels_half_open(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(bin=True)
        labels = list(table.keys())
        self.assertTrue(all(label.endswith(")") for label in labels[:-1]))

    def test_tabulate_nbins_creates_correct_number_of_bins(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(bin=True, nbins=5)
        self.assertEqual(len(table), 5)

    def test_tabulate_nbins_total_equals_n(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(bin=True, nbins=5)
        self.assertEqual(sum(table.values()), 100)

    def test_tabulate_binwidth_creates_correct_number_of_bins(self):
        # range is 1..100 = 99, binwidth=10 → ceil(99/10) = 10 bins
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(
            bin=True, binwidth=10
        )
        self.assertEqual(len(table), 10)

    def test_tabulate_binwidth_total_equals_n(self):
        table = RVResults([float(i) for i in range(1, 101)]).tabulate(
            bin=True, binwidth=10
        )
        self.assertEqual(sum(table.values()), 100)

    def test_tabulate_nbins_and_binwidth_raises(self):
        with self.assertRaises(ValueError):
            RVResults([float(i) for i in range(1, 101)]).tabulate(
                bin=True, nbins=5, binwidth=10
            )

    def test_tabulate_nbins_without_bin_warns(self):
        with self.assertWarns(UserWarning):
            RVResults([float(i) for i in range(1, 101)]).tabulate(nbins=5)

    def test_tabulate_binwidth_without_bin_warns(self):
        with self.assertWarns(UserWarning):
            RVResults([float(i) for i in range(1, 101)]).tabulate(binwidth=10)


if __name__ == "__main__":
    unittest.main()
