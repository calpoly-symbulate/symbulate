"""Tests for symbulate.base — Arithmetic, Comparable, Statistical, Logical, Filterable,
and Transformable mixins.

Covers (tested through concrete classes that implement each mixin):

Arithmetic  — via RV (inherits Arithmetic)
  All 13 operators: +, radd, -, rsub, unary -, *, rmul, /, rtruediv, **, rpow, ^ (alias
  for **), and r^ (alias for r**).  Degenerate RV(BoxModel([v])) always draws v, giving
  exact, seed-independent checks.  One RV+RV test exercises the same-space branch.

Comparable  — via RV (inherits Comparable)
  All 6 comparison operators: ==, !=, <, <=, >, >=.  Comparisons on an RV return an
  Event; calling .draw() on it yields a deterministic bool from the degenerate box.

Statistical — via RVResults (returned by RV.sim())
  sum, mean, var, std, sd (alias), median, quantile, percentile (alias), iqr,
  skew, skewness (alias), kurtosis, max, min, min_max_diff, cov, corr, corrcoef (alias).
  Alias methods are verified to return the same value as their primary method.

Logical     — via Event (inherits Logical) and Results (inherits Logical)
  &, |, ~ operators checked for correct combined-event probabilities via .sim().mean().
  Results-level logical ops tested via element-wise boolean partitions.

Filterable  — via RVResults (inherits Filterable through Results)
  All 6 filter methods (eq, neq, lt, leq, gt, geq) and 7 count methods (count,
  count_eq, count_neq, count_lt, count_leq, count_gt, count_geq).  Partition identities
  (e.g. count_lt + count_geq == N) and content-validity checks (all x in filter_lt(v)
  satisfy x < v) give exact, seed-independent assertions.

Transformable — via RV (inherits Transformable)
  abs, round (no ndigits), round (with ndigits), math.floor, math.ceil — all checked
  against the degenerate BoxModel([v]) for exact draw results.

Note on base.py fix applied alongside these tests:
  quantile() previously used `lambda **kwargs: np.quantile(q=q, **kwargs)` which cannot
  accept the positional `op(self.array)` call inside RVResults._statistic_factory.
  Fixed to `lambda a, axis=None: np.quantile(a, q=q, axis=axis)`.
"""

import math
import unittest
import numpy as np
import scipy.stats as stats

from symbulate import (
    RV,
    Normal,
    Exponential,
    BivariateNormal,
    BoxModel,
    ProbabilitySpace,
)
from symbulate import seed
from symbulate import distributions
from symbulate.probability_space import Event

Nsim = 10000


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def degenerate(value):
    """Return an RV that always draws *value* (single-element BoxModel)."""
    return RV(BoxModel([value]))


# ===========================================================================
# Arithmetic mixin
# ===========================================================================


class TestArithmetic(unittest.TestCase):
    """Arithmetic mixin — all 13 operators via degenerate RV(BoxModel([5]))."""

    def setUp(self):
        self.X = degenerate(5)  # always draws 5

    # --- RV + scalar / scalar + RV ---

    def test_add_scalar(self):
        self.assertEqual((self.X + 3).draw(), 8)

    def test_radd_scalar(self):
        self.assertEqual((3 + self.X).draw(), 8)

    # --- RV - scalar / scalar - RV ---

    def test_sub_scalar(self):
        self.assertEqual((self.X - 2).draw(), 3)

    def test_rsub_scalar(self):
        self.assertEqual((3 - self.X).draw(), -2)

    # --- unary negation ---

    def test_neg(self):
        self.assertEqual((-self.X).draw(), -5)

    # --- RV * scalar / scalar * RV ---

    def test_mul_scalar(self):
        self.assertEqual((self.X * 2).draw(), 10)

    def test_rmul_scalar(self):
        self.assertEqual((2 * self.X).draw(), 10)

    # --- RV / scalar / scalar / RV ---

    def test_truediv_scalar(self):
        self.assertAlmostEqual(float((self.X / 5).draw()), 1.0)

    def test_rtruediv_scalar(self):
        self.assertAlmostEqual(float((10 / self.X).draw()), 2.0)

    # --- RV ** scalar / scalar ** RV ---

    def test_pow_scalar(self):
        self.assertEqual((self.X**2).draw(), 25)

    def test_rpow_scalar(self):
        self.assertEqual((2**self.X).draw(), 32)

    # --- ^ is an alias for ** ---

    def test_xor_same_result_as_pow(self):
        self.assertEqual((self.X ^ 2).draw(), (self.X**2).draw())

    def test_rxor_same_result_as_rpow(self):
        self.assertEqual((2 ^ self.X).draw(), (2**self.X).draw())

    # --- RV + RV on the same probability space ---

    def test_add_two_rvs_same_space(self):
        """(X + Y) where X, Y on the same N(0,1)^2 space has mean ≈ 0."""
        seed(42)
        X, Y = RV(Normal(0, 1) ** 2)
        self.assertAlmostEqual(float((X + Y).sim(Nsim).mean()), 0.0, delta=0.1)


# ===========================================================================
# Comparable mixin
# ===========================================================================


class TestComparable(unittest.TestCase):
    """Comparable mixin — all 6 comparisons via RV.draw() on degenerate RV(BoxModel([5])).

    X = RV(BoxModel([5])) always draws 5, so each comparison result is deterministic.
    _comparison_factory creates Event objects; Event.draw() returns bool.
    """

    def setUp(self):
        self.X = degenerate(5)  # always draws 5

    def test_eq_true(self):
        self.assertTrue((self.X == 5).draw())

    def test_eq_false(self):
        self.assertFalse((self.X == 4).draw())

    def test_ne_true(self):
        self.assertTrue((self.X != 4).draw())

    def test_ne_false(self):
        self.assertFalse((self.X != 5).draw())

    def test_lt_true(self):
        self.assertTrue((self.X < 6).draw())

    def test_lt_false(self):
        self.assertFalse((self.X < 5).draw())

    def test_le_true(self):
        self.assertTrue((self.X <= 5).draw())

    def test_le_false(self):
        self.assertFalse((self.X <= 4).draw())

    def test_gt_true(self):
        self.assertTrue((self.X > 4).draw())

    def test_gt_false(self):
        self.assertFalse((self.X > 5).draw())

    def test_ge_true(self):
        self.assertTrue((self.X >= 5).draw())

    def test_ge_false(self):
        self.assertFalse((self.X >= 6).draw())


# ===========================================================================
# Statistical mixin
# ===========================================================================


class TestStatistical(unittest.TestCase):
    """Statistical mixin — all 18 methods via RVResults from RV(Normal(0,1)).sim()."""

    @classmethod
    def setUpClass(cls):
        seed(42)
        cls.sims = RV(Normal(0, 1)).sim(Nsim)
        # Bivariate sims for cov / corr tests (corr = 0.7, sd1=sd2=1)
        seed(42)
        X, Y = RV(BivariateNormal(mean1=0, mean2=0, sd1=1, sd2=1, corr=0.7))
        cls.biv_sims = (X & Y).sim(Nsim)

    # --- mean, sum ---

    def test_mean_normal(self):
        self.assertAlmostEqual(float(self.sims.mean()), 0.0, delta=0.1)

    def test_sum_approx_zero(self):
        # sum ≈ N * mean ≈ 0
        self.assertAlmostEqual(float(self.sims.sum()), 0.0, delta=Nsim * 0.1)

    # --- var, std, sd alias ---

    def test_var_normal(self):
        self.assertAlmostEqual(float(self.sims.var()), 1.0, delta=0.05)

    def test_std_normal(self):
        self.assertAlmostEqual(float(self.sims.std()), 1.0, delta=0.05)

    def test_sd_alias_equals_std(self):
        self.assertAlmostEqual(float(self.sims.sd()), float(self.sims.std()))

    # --- median, quantile, percentile alias, iqr ---

    def test_median_normal(self):
        self.assertAlmostEqual(float(self.sims.median()), 0.0, delta=0.05)

    def test_quantile_half_is_near_zero(self):
        self.assertAlmostEqual(float(self.sims.quantile(0.5)), 0.0, delta=0.05)

    def test_quantile_quarter_is_negative(self):
        self.assertLess(float(self.sims.quantile(0.25)), 0.0)

    def test_percentile_alias_equals_quantile(self):
        q = 0.3
        self.assertAlmostEqual(
            float(self.sims.percentile(q)), float(self.sims.quantile(q))
        )

    def test_iqr_normal(self):
        # Normal IQR ≈ 2 * 0.6745 ≈ 1.35
        self.assertAlmostEqual(float(self.sims.iqr()), 1.35, delta=0.1)

    def test_iqr_equals_q75_minus_q25(self):
        q75 = float(self.sims.quantile(0.75))
        q25 = float(self.sims.quantile(0.25))
        self.assertAlmostEqual(float(self.sims.iqr()), q75 - q25)

    # --- skew, skewness alias, kurtosis ---

    def test_skew_normal_near_zero(self):
        self.assertAlmostEqual(float(self.sims.skew()), 0.0, delta=0.1)

    def test_skewness_alias_equals_skew(self):
        self.assertAlmostEqual(float(self.sims.skewness()), float(self.sims.skew()))

    def test_kurtosis_normal_near_zero(self):
        # scipy.stats.kurtosis uses excess kurtosis; Normal ≈ 0
        self.assertAlmostEqual(float(self.sims.kurtosis()), 0.0, delta=0.2)

    # --- max, min, min_max_diff ---

    def test_max_exceeds_mean(self):
        self.assertGreater(float(self.sims.max()), float(self.sims.mean()))

    def test_min_below_mean(self):
        self.assertLess(float(self.sims.min()), float(self.sims.mean()))

    def test_min_max_diff_positive(self):
        self.assertGreater(float(self.sims.min_max_diff()), 0.0)

    def test_min_max_diff_equals_max_minus_min(self):
        self.assertAlmostEqual(
            float(self.sims.min_max_diff()),
            float(self.sims.max()) - float(self.sims.min()),
        )

    # --- cov, corr, corrcoef alias ---

    def test_cov_bivariate(self):
        # Cov(X, Y) = ρ * σX * σY = 0.7 * 1 * 1 = 0.7
        self.assertAlmostEqual(float(self.biv_sims.cov()), 0.7, delta=0.05)

    def test_corr_bivariate(self):
        self.assertAlmostEqual(float(self.biv_sims.corr()), 0.7, delta=0.05)

    def test_corrcoef_alias_equals_corr(self):
        self.assertAlmostEqual(
            float(self.biv_sims.corrcoef()), float(self.biv_sims.corr())
        )

    def test_cov_raises_for_univariate(self):
        """cov() requires at least 2 dimensions."""
        self.assertRaises(Exception, self.sims.cov)

    def test_corr_raises_for_univariate(self):
        """corr() requires at least 2 dimensions."""
        self.assertRaises(Exception, self.sims.corr)

    def test_quantile_out_of_range_raises(self):
        """quantile(q) raises ValueError when q is outside [0, 1]."""
        with self.assertRaises(ValueError):
            self.sims.quantile(25)

    def test_percentile_out_of_range_raises(self):
        """percentile(q) raises ValueError when q is outside [0, 1]."""
        with self.assertRaises(ValueError):
            self.sims.percentile(2.0)


# ===========================================================================
# Logical mixin
# ===========================================================================


class TestLogical(unittest.TestCase):
    """Logical mixin — &, |, ~ via Event and via boolean Results.

    Event tests: use RV comparisons on BoxModel([1..6]) to create Events with
    known exact probabilities.

    Results tests: create boolean Results via element-wise RV comparisons on a
    single simulation, then verify partition and identity relationships.
    """

    # --- via Event ---

    @classmethod
    def setUpClass(cls):
        P = BoxModel([1, 2, 3, 4, 5, 6])
        cls.X = RV(P)
        # A = {5, 6}, P(A) = 1/3
        # B = {1, 2}, P(B) = 1/3
        # A and B are disjoint
        cls.A = cls.X > 4
        cls.B = cls.X < 3

    def test_event_and_disjoint_events_prob_zero(self):
        """A & B where A = {5,6} and B = {1,2} are disjoint → P ≈ 0.

        Event.sim() returns a plain Results (not RVResults), so .mean() is
        unavailable.  Use Filterable.count() to measure the proportion instead.
        """
        result = (self.A & self.B).sim(Nsim)
        prob = result.count(lambda x: x) / Nsim
        self.assertAlmostEqual(prob, 0.0, delta=0.02)

    def test_event_or_union_prob(self):
        """A | B where A = {5,6}, B = {1,2} → P ≈ 4/6 ≈ 0.667."""
        result = (self.A | self.B).sim(Nsim)
        prob = result.count(lambda x: x) / Nsim
        self.assertAlmostEqual(prob, 4 / 6, delta=0.05)

    def test_event_invert_complement_prob(self):
        """~A where A = {5,6} → P ≈ 4/6 ≈ 0.667."""
        result = (~self.A).sim(Nsim)
        prob = result.count(lambda x: x) / Nsim
        self.assertAlmostEqual(prob, 4 / 6, delta=0.05)

    def test_event_and_returns_event_instance(self):
        self.assertIsInstance(self.A & self.B, Event)

    def test_event_or_returns_event_instance(self):
        self.assertIsInstance(self.A | self.B, Event)

    def test_event_invert_returns_event_instance(self):
        self.assertIsInstance(~self.A, Event)

    def test_event_and_different_prob_space_raises(self):
        """Events from different prob_spaces cannot be combined."""
        Y = RV(BoxModel([1, 2, 3]))
        C = Y > 2
        self.assertRaises(Exception, lambda: self.A & C)

    def test_event_bool_raises_type_error(self):
        """Casting an Event to bool is explicitly blocked."""
        self.assertRaises(TypeError, bool, self.A)

    # --- via Results (boolean element-wise) ---

    def test_results_and_is_intersection(self):
        """(sims > 0.5) & (sims < -0.5) is always False (mutually exclusive)."""
        seed(42)
        sims = RV(Normal(0, 1)).sim(Nsim)
        high = sims > 0.5
        low = sims < -0.5
        and_result = high & low
        self.assertEqual(and_result.count(lambda x: x), 0)

    def test_results_or_is_union(self):
        """(sims > 0.5) | (sims < -0.5) count == count_gt(0.5) + count_lt(-0.5)."""
        seed(42)
        sims = RV(Normal(0, 1)).sim(Nsim)
        high = sims > 0.5
        low = sims < -0.5
        or_result = high | low
        expected = high.count(lambda x: x) + low.count(lambda x: x)
        self.assertEqual(or_result.count(lambda x: x), expected)

    def test_results_invert_complement(self):
        """~high and high together cover all N outcomes."""
        seed(42)
        sims = RV(Normal(0, 1)).sim(Nsim)
        high = sims > 0.5
        not_high = ~high
        self.assertEqual(high.count(lambda x: x) + not_high.count(lambda x: x), Nsim)

    def test_results_and_non_boolean_raises_value_error(self):
        """Logical ops on non-boolean Results raise ValueError."""
        seed(42)
        sims = RV(Normal(0, 1)).sim(Nsim)
        # sims contains floats, not booleans
        self.assertRaises(ValueError, lambda: sims & sims)

    def test_results_and_non_results_raises_type_error(self):
        """Logical AND with a non-Results raises TypeError."""
        seed(42)
        sims = RV(Normal(0, 1)).sim(Nsim)
        high = sims > 0.5
        self.assertRaises(TypeError, lambda: high & True)


# ===========================================================================
# Filterable mixin
# ===========================================================================


class TestFilterable(unittest.TestCase):
    """Filterable mixin — all filter_* and count_* methods via RVResults.

    Uses partition identities (count_lt + count_geq == N) and content-validity
    checks (every x in filter_lt(0) satisfies x < 0) that hold regardless of
    the specific simulated values — no seed needed.

    All assertions are exact (no floating-point tolerance).
    """

    N = 500

    @classmethod
    def setUpClass(cls):
        # threshold value chosen well within the Normal support
        cls.threshold = 0.0
        cls.sims = RV(Normal(0, 1)).sim(cls.N)

    # --- partition identities ---

    def test_count_lt_plus_count_geq_equals_N(self):
        v = self.threshold
        self.assertEqual(self.sims.count_lt(v) + self.sims.count_geq(v), self.N)

    def test_count_leq_plus_count_gt_equals_N(self):
        v = self.threshold
        self.assertEqual(self.sims.count_leq(v) + self.sims.count_gt(v), self.N)

    def test_count_eq_plus_count_neq_equals_N(self):
        v = self.threshold
        self.assertEqual(self.sims.count_eq(v) + self.sims.count_neq(v), self.N)

    def test_count_no_arg_equals_N(self):
        """count() with no argument counts every outcome."""
        self.assertEqual(self.sims.count(), self.N)

    def test_count_with_func_matches_count_gt(self):
        """count(lambda x: x > 0) == count_gt(0)."""
        self.assertEqual(
            self.sims.count(lambda x: x > self.threshold),
            self.sims.count_gt(self.threshold),
        )

    # --- content validity of filter results ---

    def test_filter_lt_contains_only_values_below_threshold(self):
        v = self.threshold
        filtered = self.sims.filter_lt(v)
        self.assertTrue(all(x < v for x in filtered))

    def test_filter_leq_contains_only_values_at_most_threshold(self):
        v = self.threshold
        filtered = self.sims.filter_leq(v)
        self.assertTrue(all(x <= v for x in filtered))

    def test_filter_gt_contains_only_values_above_threshold(self):
        v = self.threshold
        filtered = self.sims.filter_gt(v)
        self.assertTrue(all(x > v for x in filtered))

    def test_filter_geq_contains_only_values_at_least_threshold(self):
        v = self.threshold
        filtered = self.sims.filter_geq(v)
        self.assertTrue(all(x >= v for x in filtered))

    # --- length consistency between filter and count ---

    def test_count_lt_matches_len_filter_lt(self):
        v = self.threshold
        self.assertEqual(self.sims.count_lt(v), len(self.sims.filter_lt(v)))

    def test_count_leq_matches_len_filter_leq(self):
        v = self.threshold
        self.assertEqual(self.sims.count_leq(v), len(self.sims.filter_leq(v)))

    def test_count_gt_matches_len_filter_gt(self):
        v = self.threshold
        self.assertEqual(self.sims.count_gt(v), len(self.sims.filter_gt(v)))

    def test_count_geq_matches_len_filter_geq(self):
        v = self.threshold
        self.assertEqual(self.sims.count_geq(v), len(self.sims.filter_geq(v)))

    def test_count_neq_matches_len_filter_neq(self):
        v = self.threshold
        self.assertEqual(self.sims.count_neq(v), len(self.sims.filter_neq(v)))

    # --- filter_eq with discrete distribution ---

    def test_filter_eq_content_validity(self):
        """Every element returned by filter_eq(3) is exactly 3."""
        seed(42)
        disc_sims = RV(BoxModel([1, 2, 3, 4, 5])).sim(200)
        filtered = disc_sims.filter_eq(3)
        self.assertTrue(all(x == 3 for x in filtered))

    def test_count_eq_matches_len_filter_eq(self):
        """count_eq(3) == len(filter_eq(3))."""
        seed(42)
        disc_sims = RV(BoxModel([1, 2, 3, 4, 5])).sim(200)
        self.assertEqual(disc_sims.count_eq(3), len(disc_sims.filter_eq(3)))

    def test_count_non_callable_raises_type_error(self):
        """count() raises TypeError when passed a non-callable."""
        with self.assertRaises(TypeError):
            self.sims.count(5)


# ===========================================================================
# Multivariate filter / count
# ===========================================================================


class TestMultivariateFilterCount(unittest.TestCase):
    """Multivariate count() and filter() via joint RV distributions.

    Uses a fixed-seed joint Normal(0,1) & Binomial(10,0.5) simulation so
    that every assertion is exact and reproducible without relying on
    specific probability values.
    """

    N = 1000

    @classmethod
    def setUpClass(cls):
        seed(0)
        X, Y = RV(Normal(0, 1) ** 2)
        cls.sims = (X & Y).sim(cls.N)

    # --- callable mode ---

    def test_count_callable_mode_matches_lambda_with_index(self):
        """count(f0, f1) agrees with count(lambda x: f0(x[0]) and f1(x[1]))."""
        tx, ty = 0.0, 0.0
        per_component = self.sims.count(
            lambda x: x > tx,
            lambda y: y > ty,
        )
        combined = self.sims.count(lambda v: v[0] > tx and v[1] > ty)
        self.assertEqual(per_component, combined)

    def test_count_callable_mode_none_skips_component(self):
        """count(None, f1) ignores component 0 and filters only on component 1."""
        ty = 0.5
        via_none = self.sims.count(None, lambda y: y > ty)
        direct = self.sims.count(lambda v: v[1] > ty)
        self.assertEqual(via_none, direct)

    # --- op-tuple mode ---

    def test_count_op_tuple_mode_matches_callable_mode(self):
        """count(('>', t), ('>', u)) == count(lambda x: x>t, lambda y: y>u)."""
        tx, ty = 0.0, 0.5
        via_tuples = self.sims.count((">", tx), (">", ty))
        via_callables = self.sims.count(
            lambda x: x > tx,
            lambda y: y > ty,
        )
        self.assertEqual(via_tuples, via_callables)

    def test_count_op_tuple_all_operators(self):
        """Each operator string produces a result consistent with a direct lambda."""
        tx = 0.5
        for op_str, func in [
            ("<", lambda x, t=tx: x < t),
            ("<=", lambda x, t=tx: x <= t),
            (">", lambda x, t=tx: x > t),
            (">=", lambda x, t=tx: x >= t),
        ]:
            with self.subTest(op=op_str):
                via_tuple = self.sims.count((op_str, tx), None)
                via_lambda = self.sims.count(lambda v, f=func: f(v[0]))
                self.assertEqual(via_tuple, via_lambda)

    def test_count_op_tuple_none_skips_component(self):
        """count(None, ('>', v)) applies condition only to component 1."""
        ty = 0.5
        via_none = self.sims.count(None, (">", ty))
        direct = self.sims.count(lambda v: v[1] > ty)
        self.assertEqual(via_none, direct)

    def test_count_op_tuple_invalid_operator_raises_value_error(self):
        """An unrecognised operator string raises ValueError."""
        with self.assertRaises(ValueError):
            self.sims.count(("??", 0), (">", 0.3))

    # --- filter consistency ---

    def test_filter_mv_length_matches_count_mv(self):
        """len(filter(conds)) == count(conds) for both callable and tuple modes."""
        tx, ty = 0.0, 0.5
        self.assertEqual(
            len(self.sims.filter(lambda x: x > tx, lambda y: y > ty)),
            self.sims.count(lambda x: x > tx, lambda y: y > ty),
        )
        self.assertEqual(
            len(self.sims.filter((">", tx), (">", ty))),
            self.sims.count((">", tx), (">", ty)),
        )

    def test_filter_mv_content_validity(self):
        """Every outcome returned by filter(('>', 0), ('>', 0.5)) satisfies both conditions."""
        filtered = self.sims.filter((">", 0.0), (">", 0.5))
        self.assertTrue(all(v[0] > 0.0 and v[1] > 0.5 for v in filtered))

    # --- type-error guard ---

    def test_count_mixed_callable_and_tuple_raises_type_error(self):
        """Mixing callables and op-tuples in a single count() call raises TypeError."""
        with self.assertRaises(TypeError):
            self.sims.count(lambda x: x > 0, (">", 3))


# ===========================================================================
# Transformable mixin
# ===========================================================================


class TestTransformable(unittest.TestCase):
    """Transformable mixin — abs, round, floor, ceil via degenerate RV.draw()."""

    def test_abs_of_negative(self):
        X = degenerate(-2.7)
        self.assertAlmostEqual(float(abs(X).draw()), 2.7)

    def test_abs_of_positive(self):
        X = degenerate(2.3)
        self.assertAlmostEqual(float(abs(X).draw()), 2.3)

    def test_round_no_ndigits_negative(self):
        """round(-2.7) == -3."""
        X = degenerate(-2.7)
        self.assertEqual(round(X).draw(), -3)

    def test_round_no_ndigits_positive(self):
        """round(2.3) == 2."""
        X = degenerate(2.3)
        self.assertEqual(round(X).draw(), 2)

    def test_round_with_ndigits(self):
        """round(-2.7, 1) keeps one decimal place."""
        X = degenerate(-2.7)
        self.assertAlmostEqual(float(round(X, 1).draw()), -2.7, places=10)

    def test_floor_negative(self):
        """math.floor(-2.7) == -3."""
        X = degenerate(-2.7)
        self.assertEqual(math.floor(X).draw(), -3)

    def test_floor_positive(self):
        """math.floor(2.3) == 2."""
        X = degenerate(2.3)
        self.assertEqual(math.floor(X).draw(), 2)

    def test_ceil_negative(self):
        """math.ceil(-2.7) == -2."""
        X = degenerate(-2.7)
        self.assertEqual(math.ceil(X).draw(), -2)

    def test_ceil_positive(self):
        """math.ceil(2.3) == 3."""
        X = degenerate(2.3)
        self.assertEqual(math.ceil(X).draw(), 3)

    def test_round_invalid_ndigits_raises(self):
        """round(X, ndigits) raises ValueError when ndigits is not a non-negative int."""
        X = degenerate(2.3)
        with self.assertRaises(ValueError):
            round(X, -1)


if __name__ == "__main__":
    unittest.main()
