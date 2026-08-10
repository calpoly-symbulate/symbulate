"""Tests for symbulate.time_series.

Covers the moving-average process and the autoregressive/ARMA family.

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
from symbulate.time_series import (
    MA,
    MAResult,
    MAProbabilitySpace,
    AR,
    ARMA,
    ARMAResult,
    ARMAProbabilitySpace,
    ARCH,
    GARCH,
    GARCHResult,
    GARCHProbabilitySpace,
)
from symbulate.result import InfiniteVector

Nsim = 10000


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
        seed(42)
        self.assertIsInstance(MA(coefs=[0.5]).draw(), InfiniteVector)

    def test_value_is_weighted_sum_of_shocks(self):
        # With every shock equal to 1, X[n] is just 1 + sum(coefs).
        seed(42)
        path = MA(coefs=[0.8, 0.5], noise_dist=Bernoulli(1)).draw()
        for n in range(6):
            self.assertAlmostEqual(float(path[n]), 1 + 0.8 + 0.5)

    def test_mean_shifts_every_value(self):
        seed(42)
        path = MA(coefs=[0.5], noise_dist=Bernoulli(1), mean=10).draw()
        self.assertAlmostEqual(float(path[0]), 10 + 1 + 0.5)

    def test_path_is_cached_and_stable(self):
        seed(42)
        path = MA(coefs=[0.8, 0.5]).draw()
        first = [float(path[n]) for n in range(12)]
        self.assertEqual(first, [float(path[n]) for n in range(12)])

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed(42)
        path = MA(coefs=[0.8, 0.5]).draw()
        early = [float(path[n]) for n in range(8)]
        path[400]
        self.assertEqual([float(path[n]) for n in range(8)], early)

    def test_matches_hand_computed_formula(self):
        # X[n] = mean + shock[n] + coefs[0]*shock[n-1] + coefs[1]*shock[n-2],
        # where shocks[k] holds the shock at time k - q.
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
        coefs = [0.8, 0.5]
        v = float(MA(coefs=coefs)[0].sim(Nsim).var())
        self.assertAlmostEqual(v, theoretical_variance(coefs), delta=0.15)
        self.assertGreater(v, 1.4)

    def test_mean_is_the_mean_parameter_everywhere(self):
        seed(42)
        X = MA(coefs=[0.8, 0.5], mean=7)
        for n in [0, 30]:
            self.assertAlmostEqual(float(X[n].sim(Nsim).mean()), 7.0, delta=0.1)


class TestMATheory(unittest.TestCase):
    """The closed-form facts an MA(q) is taught with."""

    def test_variance_matches_closed_form(self):
        seed(42)
        for coefs in [[], [0.5], [0.8, 0.5], [0.2, -0.4, 0.6]]:
            v = float(MA(coefs=coefs)[20].sim(Nsim).var())
            self.assertAlmostEqual(v, theoretical_variance(coefs), delta=0.15)

    def test_noise_scale_scales_the_variance(self):
        seed(42)
        coefs = [0.5]
        v = float(MA(coefs=coefs, noise_dist=Normal(0, 3))[10].sim(Nsim).var())
        self.assertAlmostEqual(v, theoretical_variance(coefs, noise_sd=3), delta=1.5)

    def test_correlation_cuts_off_after_lag_q(self):
        # The signature of a moving-average process: correlation is real up
        # to lag q, then exactly zero.
        seed(42)
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
        seed(42)
        paths = sample_paths(MA(coefs=[1.0]), length=4, n_paths=20000)
        self.assertAlmostEqual(
            np.corrcoef(paths[:, 0], paths[:, 1])[0, 1], 0.5, delta=0.03
        )
        self.assertAlmostEqual(
            np.corrcoef(paths[:, 0], paths[:, 2])[0, 1], 0.0, delta=0.03
        )

    def test_no_coefficients_is_independent_noise(self):
        seed(42)
        paths = sample_paths(MA(coefs=[]), length=4, n_paths=20000)
        self.assertAlmostEqual(paths[:, 0].var(), 1.0, delta=0.06)
        self.assertAlmostEqual(
            np.corrcoef(paths[:, 0], paths[:, 1])[0, 1], 0.0, delta=0.03
        )

    def test_correlation_does_not_depend_on_where_you_measure(self):
        # Stationarity again: the lag-1 correlation is the same starting from
        # time 0 as from time 20.
        seed(42)
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


class TestARMAResult(unittest.TestCase):

    def test_is_infinite_vector(self):
        seed(42)
        self.assertIsInstance(AR(coefs=[0.5]).draw(), InfiniteVector)

    def test_recursion_matches_hand_computation(self):
        # Every shock is 1 and the process starts at 0, so
        # X[0] = 1, X[n] = 0.5 * X[n-1] + 1.
        seed(42)
        path = AR(coefs=[0.5], noise_dist=Bernoulli(1), initial=0).draw()
        expected = [1.0, 1.5, 1.75, 1.875, 1.9375, 1.96875]
        self.assertEqual([float(path[n]) for n in range(6)], expected)

    def test_presample_values_are_used(self):
        # Starting at 10 with unit shocks: X[0] = 0.5 * 10 + 1 = 6.
        seed(42)
        path = AR(coefs=[0.5], noise_dist=Bernoulli(1), initial=10).draw()
        self.assertAlmostEqual(float(path[0]), 6.0)

    def test_second_order_uses_both_previous_values(self):
        # X[0] = 0.5 * X[-1] + 0.25 * X[-2] + 1, with X[-2]=2, X[-1]=4.
        seed(42)
        path = AR(coefs=[0.5, 0.25], noise_dist=Bernoulli(1), initial=[2, 4]).draw()
        self.assertAlmostEqual(float(path[0]), 0.5 * 4 + 0.25 * 2 + 1)

    def test_does_not_clobber_the_infinite_vector_cache(self):
        # Regression test: the internal list must not be called `values`,
        # which InfiniteTuple already uses for its own cache. When it was,
        # both appended to the same list and every value came out twice.
        seed(42)
        path = AR(coefs=[0.5], noise_dist=Bernoulli(1), initial=0).draw()
        first = [float(path[n]) for n in range(8)]
        self.assertEqual(len(set(first)), 8, "values are repeating")
        self.assertEqual(first, [float(path[n]) for n in range(8)])

    def test_path_is_cached_and_stable(self):
        seed(42)
        path = ARMA(ar_coefs=[0.5], ma_coefs=[0.3]).draw()
        first = [float(path[n]) for n in range(12)]
        self.assertEqual(first, [float(path[n]) for n in range(12)])

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed(42)
        path = AR(coefs=[0.6]).draw()
        early = [float(path[n]) for n in range(8)]
        path[300]
        self.assertEqual([float(path[n]) for n in range(8)], early)

    def test_get_shocks_returns_the_shocks(self):
        seed(42)
        self.assertIsInstance(AR(coefs=[0.5]).draw().get_shocks(), InfiniteVector)


class TestARMAProbabilitySpace(unittest.TestCase):

    def test_stores_parameters(self):
        space = ARMAProbabilitySpace(ar_coefs=[0.5], ma_coefs=[0.2], mean=3)
        self.assertEqual(space.ar_coefs, [0.5])
        self.assertEqual(space.ma_coefs, [0.2])
        self.assertEqual(space.mean, 3)

    def test_default_noise_is_standard_normal(self):
        self.assertIsInstance(ARMAProbabilitySpace(ar_coefs=[0.5]).noise_dist, Normal)

    def test_draw_returns_result(self):
        seed(42)
        self.assertIsInstance(ARMAProbabilitySpace(ar_coefs=[0.5]).draw(), ARMAResult)


class TestARMAConstruction(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(ARMA(ar_coefs=[0.5]), RV)
        self.assertIsInstance(AR(coefs=[0.5]), RV)

    def test_ar_is_an_arma(self):
        self.assertIsInstance(AR(coefs=[0.5]), ARMA)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(ARMA(ar_coefs=[0.5])[3], RV)

    def test_ar_exposes_coefs_alias(self):
        X = AR(coefs=[0.5, 0.2])
        self.assertEqual(X.coefs, [0.5, 0.2])
        self.assertEqual(X.coefs, X.ar_coefs)
        self.assertEqual(X.ma_coefs, [])

    def test_reproducible_under_same_seed(self):
        seed(11)
        first = [float(v) for v in ARMA(ar_coefs=[0.5], ma_coefs=[0.3]).draw()[:15]]
        seed(11)
        self.assertEqual(
            first, [float(v) for v in ARMA(ar_coefs=[0.5], ma_coefs=[0.3]).draw()[:15]]
        )


class TestARMATheory(unittest.TestCase):
    """The closed-form facts an AR/ARMA is taught with."""

    def test_ar1_stationary_variance(self):
        # Var = s**2 / (1 - phi**2), the same at every time when started
        # from the stationary distribution.
        seed(42)
        phi = 0.7
        expected = 1 / (1 - phi**2)
        X = AR(coefs=[phi], initial="stationary")
        for n in [0, 1, 5, 30]:
            self.assertAlmostEqual(float(X[n].sim(Nsim).var()), expected, delta=0.25)

    def test_ar1_autocorrelation_is_phi_to_the_k(self):
        # The AR fingerprint: correlation decays geometrically and never
        # reaches zero, unlike an MA's hard cutoff.
        seed(42)
        phi = 0.7
        paths = sample_paths(AR(coefs=[phi], initial="stationary"), 6, 20000)
        for k in [1, 2, 3]:
            measured = np.corrcoef(paths[:, 0], paths[:, k])[0, 1]
            self.assertAlmostEqual(measured, phi**k, delta=0.04)

    def test_ar2_matches_yule_walker(self):
        # phi = [0.5, 0.3] gives stationary variance 2.2436 and lag-1
        # correlation phi1 / (1 - phi2) = 0.7143.
        seed(42)
        paths = sample_paths(AR(coefs=[0.5, 0.3], initial="stationary"), 4, 20000)
        self.assertAlmostEqual(paths[:, 0].var(), 2.2436, delta=0.2)
        self.assertAlmostEqual(
            np.corrcoef(paths[:, 0], paths[:, 1])[0, 1], 0.5 / (1 - 0.3), delta=0.04
        )

    def test_mean_is_the_process_mean(self):
        seed(42)
        X = AR(coefs=[0.6], mean=20, initial=20)
        self.assertAlmostEqual(float(X[40].sim(Nsim).mean()), 20.0, delta=0.3)

    def test_unit_coefficient_is_a_random_walk(self):
        # phi = 1 removes the pull home, so the variance grows with n.
        seed(42)
        X = AR(coefs=[1.0], initial=0)
        for n in [10, 20, 40]:
            self.assertAlmostEqual(
                float(X[n].sim(6000).var()) / (n + 1), 1.0, delta=0.2
            )

    def test_no_ar_terms_reproduces_ma(self):
        # ARMA with an empty AR part is exactly a moving-average process.
        seed(42)
        coefs = [0.8, 0.5]
        arma = sample_paths(ARMA(ar_coefs=[], ma_coefs=coefs), 6, 20000)
        expected_var = 1 + sum(c * c for c in coefs)
        self.assertAlmostEqual(arma[:, 0].var(), expected_var, delta=0.15)
        # ...including the hard cutoff past lag q.
        self.assertAlmostEqual(
            np.corrcoef(arma[:, 0], arma[:, 3])[0, 1], 0.0, delta=0.04
        )

    def test_noise_scale_scales_the_variance(self):
        seed(42)
        phi, sd = 0.5, 3.0
        X = AR(coefs=[phi], noise_dist=Normal(0, sd), initial="stationary")
        self.assertAlmostEqual(
            float(X[10].sim(Nsim).var()), sd**2 / (1 - phi**2), delta=1.5
        )


class TestARMAInitialConditions(unittest.TestCase):
    """The `initial` parameter's accepted forms."""

    def test_number_starts_every_path_there(self):
        seed(42)
        path = AR(coefs=[0.5], noise_dist=Bernoulli(1), initial=10).draw()
        self.assertAlmostEqual(float(path[0]), 6.0)

    def test_sequence_of_p_values(self):
        seed(42)
        path = AR(coefs=[0.5, 0.25], noise_dist=Bernoulli(1), initial=[2, 4]).draw()
        self.assertAlmostEqual(float(path[0]), 0.5 * 4 + 0.25 * 2 + 1)

    def test_distribution_start_is_random_per_path(self):
        seed(42)
        X = AR(coefs=[0.5], noise_dist=Bernoulli(1), initial=Normal(0, 5))
        starts = [float(X.draw()[0]) for _ in range(50)]
        self.assertGreater(len(set(starts)), 40)

    def test_multivariate_start_is_drawn_jointly(self):
        # For p > 1 the starting values are correlated, which only a
        # multivariate distribution can express.
        seed(42)
        cov = [[2.0, 1.5], [1.5, 2.0]]
        X = AR(coefs=[0.5, 0.3], initial=MultivariateNormal([0, 0], cov))
        self.assertEqual(len(X.draw().presample), 2)

    def test_stationary_has_no_transient(self):
        # The variance is already the long-run one at time 0.
        seed(42)
        phi = 0.7
        X = AR(coefs=[phi], initial="stationary")
        expected = 1 / (1 - phi**2)
        self.assertAlmostEqual(float(X[0].sim(Nsim).var()), expected, delta=0.25)

    def test_fixed_start_does_have_a_transient(self):
        # The counterpart: starting at 0 gives noise variance at time 0,
        # growing toward the stationary variance.
        seed(42)
        X = AR(coefs=[0.7], initial=0)
        early = float(X[0].sim(Nsim).var())
        late = float(X[30].sim(Nsim).var())
        self.assertAlmostEqual(early, 1.0, delta=0.15)
        self.assertGreater(late, early + 0.5)

    def test_stationary_works_for_ar2(self):
        seed(42)
        X = AR(coefs=[0.5, 0.3], initial="stationary")
        self.assertAlmostEqual(float(X[0].sim(Nsim).var()), 2.2436, delta=0.25)


class TestARMAErrors(unittest.TestCase):

    def test_bad_coefficients_raise_type_error(self):
        for bad in ["abc", [0.5, "x"], 5, None]:
            self.assertRaisesRegex(
                TypeError, "ar_coefs must be", lambda v=bad: ARMA(ar_coefs=v)
            )
        for bad in ["abc", [0.5, "x"], 5]:
            self.assertRaisesRegex(
                TypeError,
                "ma_coefs must be",
                lambda v=bad: ARMA(ar_coefs=[0.5], ma_coefs=v),
            )

    def test_bad_noise_dist_raises_type_error(self):
        self.assertRaisesRegex(
            TypeError, "noise_dist must be", lambda: ARMA(ar_coefs=[0.5], noise_dist=5)
        )

    def test_non_numeric_mean_raises_type_error(self):
        self.assertRaisesRegex(
            TypeError, "mean must be a number", lambda: ARMA(ar_coefs=[0.5], mean="x")
        )

    def test_wrong_length_initial_raises_value_error(self):
        self.assertRaisesRegex(
            ValueError,
            "needs 2",
            lambda: AR(coefs=[0.5, 0.2], initial=[1, 2, 3]),
        )

    def test_stationary_with_ma_terms_raises(self):
        # No simple closed form once moving-average terms are present.
        self.assertRaisesRegex(
            ValueError,
            "pure autoregressive",
            lambda: ARMA(ar_coefs=[0.5], ma_coefs=[0.3], initial="stationary"),
        )

    def test_stationary_with_non_normal_noise_raises(self):
        self.assertRaisesRegex(
            ValueError,
            "normal shocks",
            lambda: AR(coefs=[0.5], noise_dist=Exponential(1), initial="stationary"),
        )

    def test_stationary_with_explosive_coefficients_raises(self):
        # No long-run distribution exists to start from.
        self.assertRaisesRegex(
            ValueError,
            "does not settle down",
            lambda: AR(coefs=[1.5], initial="stationary"),
        )

    def test_explosive_process_still_simulates(self):
        # Explosive is allowed -- only "stationary" rejects it.
        seed(42)
        path = AR(coefs=[1.5], noise_dist=Bernoulli(1), initial=1).draw()
        self.assertGreater(float(path[10]), float(path[5]))


class TestGARCHResult(unittest.TestCase):

    def test_is_infinite_vector(self):
        seed(42)
        self.assertIsInstance(GARCH(omega=0.2, arch_coefs=[0.1]).draw(), InfiniteVector)

    def test_recursion_matches_hand_computation(self):
        # Starting variance 4, every shock 1. The recursion reproduces 4
        # exactly: 0.2 + 0.1 * 4 + 0.85 * 4 = 4, so every value is sqrt(4).
        seed(42)
        path = GARCH(
            omega=0.2,
            arch_coefs=[0.1],
            garch_coefs=[0.85],
            noise_dist=Bernoulli(1),
            initial=4,
        ).draw()
        for n in range(5):
            self.assertAlmostEqual(float(path[n]), 2.0)

    def test_variance_recursion_is_followed(self):
        # With a fixed starting variance and unit shocks the whole variance
        # path can be worked out by hand.
        seed(42)
        omega, a, b, v0 = 0.5, 0.2, 0.3, 1.0
        path = GARCH(
            omega=omega,
            arch_coefs=[a],
            garch_coefs=[b],
            noise_dist=Bernoulli(1),
            initial=v0,
        ).draw()
        path[3]
        variances = path.get_variances()
        expected, prev_sq, prev_var = [], v0, v0
        for _ in range(4):
            v = omega + a * prev_sq + b * prev_var
            expected.append(v)
            prev_var, prev_sq = v, v  # shock is 1, so X**2 == variance
        for got, want in zip(variances, expected):
            self.assertAlmostEqual(got, want)

    def test_does_not_clobber_the_infinite_vector_cache(self):
        seed(42)
        path = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85]).draw()
        first = [float(path[n]) for n in range(8)]
        self.assertEqual(first, [float(path[n]) for n in range(8)])

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed(42)
        path = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85]).draw()
        early = [float(path[n]) for n in range(8)]
        path[200]
        self.assertEqual([float(path[n]) for n in range(8)], early)

    def test_variances_are_positive(self):
        seed(42)
        path = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85]).draw()
        path[50]
        self.assertTrue(all(v > 0 for v in path.get_variances()))


class TestGARCHProbabilitySpace(unittest.TestCase):

    def test_default_initial_is_the_long_run_variance(self):
        space = GARCHProbabilitySpace(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85])
        self.assertAlmostEqual(space.initial, 0.2 / (1 - 0.95))

    def test_non_stationary_falls_back_to_omega(self):
        # No long-run variance exists, so there is nothing better to use.
        space = GARCHProbabilitySpace(omega=0.3, arch_coefs=[0.5], garch_coefs=[0.6])
        self.assertEqual(space.initial, 0.3)

    def test_draw_returns_result(self):
        seed(42)
        self.assertIsInstance(
            GARCHProbabilitySpace(omega=0.2, arch_coefs=[0.1]).draw(), GARCHResult
        )


class TestGARCHConstruction(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(GARCH(omega=0.2, arch_coefs=[0.1]), RV)
        self.assertIsInstance(ARCH(omega=0.5, coefs=[0.5]), RV)

    def test_arch_is_a_garch(self):
        self.assertIsInstance(ARCH(omega=0.5, coefs=[0.5]), GARCH)

    def test_arch_has_no_persistence_terms(self):
        X = ARCH(omega=0.5, coefs=[0.5])
        self.assertEqual(X.garch_coefs, [])
        self.assertEqual(X.coefs, X.arch_coefs)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(GARCH(omega=0.2, arch_coefs=[0.1])[3], RV)

    def test_reproducible_under_same_seed(self):
        seed(5)
        first = [float(v) for v in GARCH(omega=0.2, arch_coefs=[0.1]).draw()[:12]]
        seed(5)
        self.assertEqual(
            first, [float(v) for v in GARCH(omega=0.2, arch_coefs=[0.1]).draw()[:12]]
        )


class TestGARCHTheory(unittest.TestCase):
    """The facts a GARCH is taught for."""

    def test_long_run_variance_matches_closed_form(self):
        # omega / (1 - sum of coefficients), and the default start means it
        # holds from time 0 with no warm-up.
        seed(42)
        omega, a, b = 0.2, 0.1, 0.85
        X = GARCH(omega=omega, arch_coefs=[a], garch_coefs=[b])
        expected = omega / (1 - a - b)
        for n in [0, 1, 5, 30]:
            self.assertAlmostEqual(float(X[n].sim(Nsim).var()), expected, delta=0.8)

    def test_values_are_centered_at_zero(self):
        seed(42)
        X = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85])
        self.assertAlmostEqual(float(X[20].sim(Nsim).mean()), 0.0, delta=0.15)

    def test_values_are_uncorrelated_but_their_sizes_are_not(self):
        # The signature of a GARCH: no correlation between the values
        # themselves, but positive correlation between their squares --
        # volatility clustering.
        seed(42)
        paths = sample_paths(
            GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85]), 6, 20000
        )
        plain = np.corrcoef(paths[:, 3], paths[:, 4])[0, 1]
        squared = np.corrcoef(paths[:, 3] ** 2, paths[:, 4] ** 2)[0, 1]
        self.assertAlmostEqual(plain, 0.0, delta=0.05)
        self.assertGreater(squared, 0.03)

    def test_arch_long_run_variance(self):
        seed(42)
        omega, a = 0.5, 0.5
        X = ARCH(omega=omega, coefs=[a])
        self.assertAlmostEqual(float(X[20].sim(Nsim).var()), omega / (1 - a), delta=0.3)

    def test_no_coefficients_is_plain_noise(self):
        # Variance is just omega, and there is nothing to cluster.
        seed(42)
        X = GARCH(omega=2.0, arch_coefs=[])
        self.assertAlmostEqual(float(X[10].sim(Nsim).var()), 2.0, delta=0.2)

    def test_starting_low_shows_a_warm_up(self):
        # The counterpart to the default: start below the long-run variance
        # and it climbs toward it.
        seed(42)
        X = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85], initial=0.2)
        early = float(X[0].sim(Nsim).var())
        late = float(X[30].sim(Nsim).var())
        self.assertLess(early, 1.0)
        self.assertGreater(late, early * 2)

    def test_explosive_variance_grows(self):
        # Coefficients summing past 1 are allowed; the variance just grows.
        seed(42)
        X = GARCH(omega=0.2, arch_coefs=[0.2], garch_coefs=[0.9])
        self.assertGreater(
            float(X[25].sim(3000).var()), float(X[5].sim(3000).var()) * 2
        )


class TestGARCHErrors(unittest.TestCase):

    def test_bad_coefficients_raise_type_error(self):
        for bad in ["abc", [0.1, "x"], 5]:
            self.assertRaisesRegex(
                TypeError,
                "arch_coefs must be",
                lambda v=bad: GARCH(omega=0.2, arch_coefs=v),
            )

    def test_negative_coefficients_raise_value_error(self):
        self.assertRaisesRegex(
            ValueError,
            "cannot contain negative",
            lambda: GARCH(omega=0.2, arch_coefs=[-0.1]),
        )
        self.assertRaisesRegex(
            ValueError,
            "cannot contain negative",
            lambda: GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[-0.5]),
        )

    def test_non_positive_omega_raises(self):
        self.assertRaisesRegex(
            ValueError,
            "omega must be positive",
            lambda: GARCH(omega=0, arch_coefs=[0.1]),
        )
        self.assertRaisesRegex(
            TypeError,
            "omega must be a number",
            lambda: GARCH(omega="x", arch_coefs=[0.1]),
        )

    def test_non_positive_initial_raises(self):
        self.assertRaisesRegex(
            ValueError,
            "initial must be positive",
            lambda: GARCH(omega=0.2, arch_coefs=[0.1], initial=0),
        )

    def test_bad_noise_dist_raises_type_error(self):
        self.assertRaisesRegex(
            TypeError,
            "noise_dist must be",
            lambda: GARCH(omega=0.2, arch_coefs=[0.1], noise_dist=5),
        )

    def test_stationary_start_on_explosive_process_raises(self):
        self.assertRaisesRegex(
            ValueError,
            "never settles",
            lambda: GARCH(
                omega=0.2, arch_coefs=[0.5], garch_coefs=[0.6], initial="stationary"
            ),
        )


if __name__ == "__main__":
    unittest.main()
