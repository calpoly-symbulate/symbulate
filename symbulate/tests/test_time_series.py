"""Tests for symbulate.time_series.

Covers the moving-average process: the sample-path result object, the
probability space, and the two facts an MA(q) is taught for -- every value
has the same distribution (there is no startup transient), and the
correlation between values cuts off to exactly zero once they are more than
``q`` apart.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_random_walk.py.
"""

import unittest

import numpy as np

from symbulate import *
from symbulate import distributions
from symbulate.time_series import MA, MAResult, MAProbabilitySpace
from symbulate.result import InfiniteVector

Nsim = 10000


def seed(value=42):
    """Reseed the generator that distribution draws route through."""
    distributions.rng = np.random.default_rng(value)


def sample_paths(process, length, n_paths):
    """Return an (n_paths, length) array of simulated values."""
    return np.array(
        [[float(v) for v in process.draw()[:length]] for _ in range(n_paths)]
    )


def theoretical_variance(coefs, noise_sd=1.0):
    """Var(X[n]) = s**2 * (1 + sum of squared coefficients)."""
    return noise_sd**2 * (1 + sum(c * c for c in coefs))


def theoretical_correlation(coefs, lag):
    """Correlation between X[n] and X[n + lag]; exactly 0 beyond lag q."""
    w = [1.0] + list(coefs)
    if lag >= len(w):
        return 0.0
    return sum(w[i] * w[i + lag] for i in range(len(w) - lag)) / sum(v * v for v in w)


class TestMAResult(unittest.TestCase):

    def test_is_infinite_vector(self):
        seed()
        self.assertIsInstance(MA(coefs=[0.5]).draw(), InfiniteVector)

    def test_value_is_weighted_sum_of_shocks(self):
        # With every shock equal to 1, X[n] is just 1 + sum(coefs).
        seed()
        path = MA(coefs=[0.8, 0.5], noise_dist=Bernoulli(1)).draw()
        for n in range(6):
            self.assertAlmostEqual(float(path[n]), 1 + 0.8 + 0.5)

    def test_mean_shifts_every_value(self):
        seed()
        path = MA(coefs=[0.5], noise_dist=Bernoulli(1), mean=10).draw()
        self.assertAlmostEqual(float(path[0]), 10 + 1 + 0.5)

    def test_path_is_cached_and_stable(self):
        seed()
        path = MA(coefs=[0.8, 0.5]).draw()
        first = [float(path[n]) for n in range(12)]
        self.assertEqual(first, [float(path[n]) for n in range(12)])

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed()
        path = MA(coefs=[0.8, 0.5]).draw()
        early = [float(path[n]) for n in range(8)]
        path[400]
        self.assertEqual([float(path[n]) for n in range(8)], early)

    def test_matches_hand_computed_formula(self):
        # X[n] = mean + shock[n] + coefs[0]*shock[n-1] + coefs[1]*shock[n-2],
        # where shocks[k] holds the shock at time k - q.
        seed()
        coefs = [0.8, 0.5]
        q = len(coefs)
        path = MA(coefs=coefs, mean=3).draw()
        shocks = path.get_shocks()
        weights = [1.0] + coefs
        for n in range(10):
            expected = 3 + sum(
                weights[i] * float(shocks[n + q - i]) for i in range(q + 1)
            )
            self.assertAlmostEqual(float(path[n]), expected)

    def test_result_constructed_directly(self):
        # The result object weights whatever shocks it is handed.
        shocks = InfiniteVector(lambda n: 1)
        path = MAResult(shocks, coefs=[0.5, 0.25], mean=0)
        for n in range(4):
            self.assertAlmostEqual(float(path[n]), 1 + 0.5 + 0.25)


class TestMAProbabilitySpace(unittest.TestCase):

    def test_stores_parameters(self):
        space = MAProbabilitySpace(coefs=[0.3], mean=2)
        self.assertEqual(space.coefs, [0.3])
        self.assertEqual(space.mean, 2)

    def test_default_noise_is_standard_normal(self):
        space = MAProbabilitySpace(coefs=[0.5])
        self.assertIsInstance(space.noise_dist, Normal)

    def test_draw_returns_result(self):
        seed()
        self.assertIsInstance(MAProbabilitySpace(coefs=[0.5]).draw(), MAResult)


class TestMA(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(MA(coefs=[0.5]), RV)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(MA(coefs=[0.5])[3], RV)

    def test_stores_parameters(self):
        X = MA(coefs=[0.8, 0.5], mean=4)
        self.assertEqual(X.coefs, [0.8, 0.5])
        self.assertEqual(X.mean, 4)

    def test_order_is_length_of_coefs(self):
        self.assertEqual(len(MA(coefs=[0.1, 0.2, 0.3]).coefs), 3)

    def test_reproducible_under_same_seed(self):
        seed(7)
        first = [float(v) for v in MA(coefs=[0.8, 0.5]).draw()[:15]]
        seed(7)
        self.assertEqual(first, [float(v) for v in MA(coefs=[0.8, 0.5]).draw()[:15]])

    def test_accepts_an_rv_as_noise(self):
        seed()
        path = MA(coefs=[0.5], noise_dist=RV(Bernoulli(0.5)) * 2 - 1).draw()
        # Shocks are +/-1, so each value is one of 1.5, 0.5, -0.5, -1.5.
        for n in range(10):
            self.assertIn(round(float(path[n]), 6), [1.5, 0.5, -0.5, -1.5])


class TestMANoStartupTransient(unittest.TestCase):
    """Every value has the same distribution, including the very first.

    An MA(q) is defined over all time and is stationary by construction, so
    the shocks from before time 0 are simulated too. Without them the first
    ``q`` values would be too small -- these tests would fail.
    """

    def test_variance_is_the_same_at_time_zero_and_later(self):
        seed()
        coefs = [0.8, 0.5]
        X = MA(coefs=coefs)
        expected = theoretical_variance(coefs)
        for n in [0, 1, 2, 10, 50]:
            self.assertAlmostEqual(
                float(X[n].sim(Nsim).var()),
                expected,
                delta=0.15,
                msg=f"Var(X[{n}]) differs from the stationary variance",
            )

    def test_first_value_has_full_variance(self):
        # The sharpest form of the same check: X[0] must not be the
        # single-shock variance of 1.0 that a silent start would give.
        seed()
        coefs = [0.8, 0.5]
        v = float(MA(coefs=coefs)[0].sim(Nsim).var())
        self.assertAlmostEqual(v, theoretical_variance(coefs), delta=0.15)
        self.assertGreater(v, 1.4)

    def test_mean_is_the_mean_parameter_everywhere(self):
        seed()
        X = MA(coefs=[0.8, 0.5], mean=7)
        for n in [0, 30]:
            self.assertAlmostEqual(float(X[n].sim(Nsim).mean()), 7.0, delta=0.1)


class TestMATheory(unittest.TestCase):
    """The closed-form facts an MA(q) is taught with."""

    def test_variance_matches_closed_form(self):
        seed()
        for coefs in [[], [0.5], [0.8, 0.5], [0.2, -0.4, 0.6]]:
            v = float(MA(coefs=coefs)[20].sim(Nsim).var())
            self.assertAlmostEqual(v, theoretical_variance(coefs), delta=0.15)

    def test_noise_scale_scales_the_variance(self):
        seed()
        coefs = [0.5]
        v = float(MA(coefs=coefs, noise_dist=Normal(0, 3))[10].sim(Nsim).var())
        self.assertAlmostEqual(v, theoretical_variance(coefs, noise_sd=3), delta=1.5)

    def test_correlation_cuts_off_after_lag_q(self):
        # The signature of a moving-average process: correlation is real up
        # to lag q, then exactly zero.
        seed()
        coefs = [0.8, 0.5]
        paths = sample_paths(MA(coefs=coefs), length=8, n_paths=20000)
        for lag in range(1, 6):
            measured = np.corrcoef(paths[:, 0], paths[:, lag])[0, 1]
            self.assertAlmostEqual(
                measured,
                theoretical_correlation(coefs, lag),
                delta=0.04,
                msg=f"correlation at lag {lag} is wrong",
            )

    def test_ma1_with_unit_coefficient_has_lag_one_correlation_one_half(self):
        # An exact, memorable case: theta = 1 gives 1 / (1 + 1) = 0.5.
        seed()
        paths = sample_paths(MA(coefs=[1.0]), length=4, n_paths=20000)
        self.assertAlmostEqual(
            np.corrcoef(paths[:, 0], paths[:, 1])[0, 1], 0.5, delta=0.03
        )
        self.assertAlmostEqual(
            np.corrcoef(paths[:, 0], paths[:, 2])[0, 1], 0.0, delta=0.03
        )

    def test_no_coefficients_is_independent_noise(self):
        seed()
        paths = sample_paths(MA(coefs=[]), length=4, n_paths=20000)
        self.assertAlmostEqual(paths[:, 0].var(), 1.0, delta=0.06)
        self.assertAlmostEqual(
            np.corrcoef(paths[:, 0], paths[:, 1])[0, 1], 0.0, delta=0.03
        )

    def test_correlation_does_not_depend_on_where_you_measure(self):
        # Stationarity again: the lag-1 correlation is the same starting from
        # time 0 as from time 20.
        seed()
        paths = sample_paths(MA(coefs=[0.8, 0.5]), length=25, n_paths=20000)
        early = np.corrcoef(paths[:, 0], paths[:, 1])[0, 1]
        late = np.corrcoef(paths[:, 20], paths[:, 21])[0, 1]
        self.assertAlmostEqual(early, late, delta=0.04)


class TestMAErrors(unittest.TestCase):

    def test_non_numeric_coefs_raise_type_error(self):
        for bad in ["abc", [0.5, "x"], 5, None]:
            self.assertRaisesRegex(
                TypeError, "coefs must be", lambda v=bad: MA(coefs=v)
            )

    def test_bad_noise_dist_raises_type_error(self):
        for bad in [5, "normal", [0, 1]]:
            self.assertRaisesRegex(
                TypeError,
                "noise_dist must be",
                lambda v=bad: MA(coefs=[0.5], noise_dist=v),
            )

    def test_non_numeric_mean_raises_type_error(self):
        self.assertRaisesRegex(
            TypeError, "mean must be a number", lambda: MA(coefs=[0.5], mean="x")
        )


if __name__ == "__main__":
    unittest.main()
