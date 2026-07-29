"""Tests for symbulate.gaussian_process.

Covers GaussianProcess (construction, path evaluation, lazy caching,
statistical properties), BrownianMotion (deterministic start, mean,
variance, drift, and scale parameters), and OrnsteinUhlenbeck (both
initial-value parameterizations, mean reversion, the long-run
distribution, and the Brownian-motion limit).
"""

import unittest
import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import gaussian_process
from symbulate.index_sets import DiscreteTimeSequence

Nsim = 1000


def seed(value=42):
    gaussian_process.rng = np.random.default_rng(value)


class TestGaussianProcessConstruction(unittest.TestCase):

    def test_is_random_process(self):
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        self.assertIsInstance(X, RandomProcess)

    def test_is_rv(self):
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        self.assertIsInstance(X, RV)

    def test_draw_returns_callable_path(self):
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        path = X.draw()
        result = path(1.0)
        self.assertIsInstance(result, float)

    def test_probability_space_type(self):
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        self.assertIsInstance(X.prob_space, GaussianProcessProbabilitySpace)


class TestGaussianProcessPaths(unittest.TestCase):

    def test_same_time_returns_cached_value(self):
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        path = X.draw()
        self.assertEqual(path(1.0), path(1.0))

    def test_subsequent_times_are_consistent(self):
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        path = X.draw()
        v0 = path(0.5)
        v1 = path(1.0)
        v2 = path(2.0)
        self.assertTrue(np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2))

    def test_cached_value_unchanged_after_new_time(self):
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        path = X.draw()
        v_before = path(1.0)
        path(2.0)
        v_after = path(1.0)
        self.assertEqual(v_before, v_after)

    def test_different_draws_produce_different_paths(self):
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        values = {X.draw()(1.0) for _ in range(10)}
        self.assertGreater(len(values), 1)

    def test_zero_variance_is_deterministic(self):
        # when cov(t, t) = 0, the path value equals the mean exactly
        X = GaussianProcess(lambda t: 5.0, lambda s, t: 0.0)
        for _ in range(10):
            self.assertEqual(X.draw()(1.0), 5.0)

    def test_invalid_time_raises_key_error(self):
        # infinity is not in the Reals index set
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        path = X.draw()
        with self.assertRaises(KeyError):
            path(float("inf"))


class TestGaussianProcessStatistics(unittest.TestCase):

    def setUp(self):
        seed()

    def test_mean_function_respected(self):
        X = GaussianProcess(lambda t: 3.0, lambda s, t: min(s, t))
        simulated_mean = X[1.0].sim(Nsim).mean()
        self.assertAlmostEqual(simulated_mean, 3.0, delta=0.2)

    def test_variance_respected(self):
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        simulated_var = X[1.0].sim(Nsim).var()
        self.assertAlmostEqual(simulated_var, 1.0, delta=0.2)

    def test_squared_exponential_kernel_variance(self):
        # SE kernel: cov(t, t) = 1, so marginal variance at any t is 1
        se_cov = lambda s, t: np.exp(-0.5 * (s - t) ** 2)
        X = GaussianProcess(lambda t: 0, se_cov)
        simulated_var = X[1.0].sim(Nsim).var()
        self.assertAlmostEqual(simulated_var, 1.0, delta=0.2)


class TestGaussianProcessDiscreteTime(unittest.TestCase):

    def test_discrete_time_path_is_subscriptable(self):
        seed()
        X = GaussianProcess(
            lambda t: 0, lambda s, t: min(s, t), DiscreteTimeSequence(fs=4)
        )
        path = X.draw()
        # index 4 maps to time 4/4 = 1.0
        result = path[4]
        self.assertIsInstance(result, float)

    def test_discrete_time_caches_index(self):
        seed()
        X = GaussianProcess(
            lambda t: 0, lambda s, t: min(s, t), DiscreteTimeSequence(fs=4)
        )
        path = X.draw()
        self.assertEqual(path[4], path[4])


class TestGaussianProcessProbabilitySpace(unittest.TestCase):

    def test_draw_gives_path(self):
        seed()
        P = GaussianProcessProbabilitySpace(lambda t: 0, lambda s, t: min(s, t))
        path = P.draw()
        self.assertIsInstance(path(1.0), float)

    def test_two_draws_differ(self):
        seed()
        P = GaussianProcessProbabilitySpace(lambda t: 0, lambda s, t: min(s, t))
        values = {P.draw()(1.0) for _ in range(10)}
        self.assertGreater(len(values), 1)


class TestBrownianMotion(unittest.TestCase):

    def test_is_random_process(self):
        B = BrownianMotion()
        self.assertIsInstance(B, RandomProcess)

    def test_is_rv(self):
        B = BrownianMotion()
        self.assertIsInstance(B, RV)

    def test_starts_at_zero(self):
        # cov(0, 0) = scale**2 * min(0, 0) = 0, so path value is exactly mean(0) = 0
        seed()
        B = BrownianMotion()
        for _ in range(10):
            self.assertEqual(B.draw()(0), 0)

    def test_starts_at_zero_with_drift(self):
        seed()
        B = BrownianMotion(drift=3.0)
        for _ in range(10):
            self.assertEqual(B.draw()(0), 0)

    def test_mean_is_zero_at_t(self):
        seed()
        B = BrownianMotion()
        simulated_mean = B[1.0].sim(Nsim).mean()
        self.assertAlmostEqual(simulated_mean, 0.0, delta=0.1)

    def test_variance_equals_t(self):
        seed()
        B = BrownianMotion()
        simulated_var = B[1.0].sim(Nsim).var()
        self.assertAlmostEqual(simulated_var, 1.0, delta=0.2)

    def test_variance_scales_with_t(self):
        seed()
        B = BrownianMotion()
        simulated_var = B[2.0].sim(Nsim).var()
        self.assertAlmostEqual(simulated_var, 2.0, delta=0.3)

    def test_drift_shifts_mean(self):
        seed()
        drift = 2.0
        B = BrownianMotion(drift=drift)
        simulated_mean = B[1.0].sim(Nsim).mean()
        self.assertAlmostEqual(simulated_mean, drift, delta=0.2)

    def test_scale_affects_variance(self):
        seed()
        scale = 2.0
        B = BrownianMotion(scale=scale)
        simulated_var = B[1.0].sim(Nsim).var()
        self.assertAlmostEqual(simulated_var, scale**2, delta=0.5)

    def test_different_paths_from_same_process(self):
        seed()
        B = BrownianMotion()
        values = {B.draw()(1.0) for _ in range(10)}
        self.assertGreater(len(values), 1)

    def test_probability_space_type(self):
        B = BrownianMotion()
        self.assertIsInstance(B.prob_space, BrownianMotionProbabilitySpace)


class TestBrownianMotionProbabilitySpace(unittest.TestCase):

    def test_draw_gives_path(self):
        seed()
        P = BrownianMotionProbabilitySpace()
        path = P.draw()
        self.assertIsInstance(path(1.0), float)

    def test_starts_at_zero(self):
        seed()
        P = BrownianMotionProbabilitySpace()
        for _ in range(10):
            self.assertEqual(P.draw()(0), 0)


class TestGaussianProcessErrors(unittest.TestCase):
    """Error handling for GaussianProcessProbabilitySpace and GaussianProcess."""

    def test_non_callable_mean_func_raises_type_error(self):
        """mean_func must be callable, not a plain value."""
        with self.assertRaises(TypeError):
            GaussianProcess(mean_func=0, cov_func=lambda s, t: min(s, t))

    def test_non_callable_cov_func_raises_type_error(self):
        """cov_func must be callable, not a plain value."""
        with self.assertRaises(TypeError):
            GaussianProcess(mean_func=lambda t: 0, cov_func=1)

    def test_invalid_index_set_raises_type_error(self):
        """index_set must be Reals or DiscreteTimeSequence, not a string."""
        with self.assertRaises(TypeError):
            GaussianProcess(
                mean_func=lambda t: 0,
                cov_func=lambda s, t: min(s, t),
                index_set="not_an_index_set",
            )

    def test_invalid_index_set_integer_raises_type_error(self):
        """index_set must be Reals or DiscreteTimeSequence, not an int."""
        with self.assertRaises(TypeError):
            GaussianProcess(
                mean_func=lambda t: 0,
                cov_func=lambda s, t: min(s, t),
                index_set=5,
            )

    def test_invalid_time_raises_key_error(self):
        """Evaluating a path outside the index set raises KeyError."""
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        path = X.draw()
        with self.assertRaises(KeyError):
            path(float("inf"))

    def test_valid_construction_does_not_raise(self):
        """Callable mean/cov and a valid index set must not raise."""
        seed()
        X = GaussianProcess(
            mean_func=lambda t: 0,
            cov_func=lambda s, t: min(s, t),
            index_set=DiscreteTimeSequence(fs=4),
        )
        path = X.draw()
        self.assertIsInstance(path[4], float)


class TestBrownianMotionErrors(unittest.TestCase):
    """Error handling for BrownianMotionProbabilitySpace and BrownianMotion."""

    def test_non_numeric_drift_raises_type_error(self):
        """drift must be a number."""
        with self.assertRaises(TypeError):
            BrownianMotion(drift="fast")

    def test_non_numeric_scale_raises_type_error(self):
        """scale must be a number."""
        with self.assertRaises(TypeError):
            BrownianMotion(scale="large")

    def test_zero_scale_raises_value_error(self):
        """scale=0 would produce a constant process — not allowed."""
        with self.assertRaises(ValueError):
            BrownianMotion(scale=0)

    def test_negative_scale_raises_value_error(self):
        """Negative scale is not physically meaningful."""
        with self.assertRaises(ValueError):
            BrownianMotion(scale=-1)

    def test_valid_negative_drift_does_not_raise(self):
        """Negative drift is valid; only scale must be positive."""
        seed()
        B = BrownianMotion(drift=-2.0, scale=0.5)
        path = B.draw()
        self.assertIsInstance(path(1.0), float)

    def test_valid_large_scale_does_not_raise(self):
        """Any positive scale is valid."""
        seed()
        B = BrownianMotion(drift=0, scale=10.0)
        path = B.draw()
        self.assertIsInstance(path(1.0), float)


class TestOrnsteinUhlenbeckConstruction(unittest.TestCase):

    def test_is_random_process(self):
        X = OrnsteinUhlenbeck()
        self.assertIsInstance(X, RandomProcess)

    def test_is_rv(self):
        X = OrnsteinUhlenbeck()
        self.assertIsInstance(X, RV)

    def test_probability_space_type(self):
        X = OrnsteinUhlenbeck()
        self.assertIsInstance(X.prob_space, OrnsteinUhlenbeckProbabilitySpace)

    def test_is_gaussian_process_probability_space(self):
        # OU is built as a Gaussian process, not new machinery.
        X = OrnsteinUhlenbeck()
        self.assertIsInstance(X.prob_space, GaussianProcessProbabilitySpace)

    def test_parameters_stored_on_probability_space(self):
        P = OrnsteinUhlenbeckProbabilitySpace(
            reversion_rate=2.0, mean=3.0, scale=0.5, initial_value=7.0
        )
        self.assertEqual(P.reversion_rate, 2.0)
        self.assertEqual(P.mean, 3.0)
        self.assertEqual(P.scale, 0.5)
        self.assertEqual(P.initial_value, 7.0)


class TestOrnsteinUhlenbeckPaths(unittest.TestCase):

    def test_starts_at_zero_by_default(self):
        # cov(0, 0) = 0 for the fixed-start parameterization, so the path
        # value at 0 is exactly mean_func(0) = initial_value.
        seed()
        X = OrnsteinUhlenbeck()
        for _ in range(10):
            self.assertEqual(X.draw()(0), 0)

    def test_starts_at_initial_value(self):
        seed()
        X = OrnsteinUhlenbeck(mean=0, initial_value=5.0)
        for _ in range(10):
            self.assertEqual(X.draw()(0), 5.0)

    def test_same_time_returns_cached_value(self):
        seed()
        path = OrnsteinUhlenbeck().draw()
        self.assertEqual(path(1.0), path(1.0))

    def test_cached_value_unchanged_after_new_time(self):
        seed()
        path = OrnsteinUhlenbeck().draw()
        before = path(1.0)
        path(2.0)
        self.assertEqual(before, path(1.0))

    def test_different_draws_produce_different_paths(self):
        seed()
        X = OrnsteinUhlenbeck()
        values = {X.draw()(1.0) for _ in range(10)}
        self.assertGreater(len(values), 1)

    def test_stationary_start_is_random(self):
        # Started from the long-run distribution, time 0 is not deterministic.
        seed()
        X = OrnsteinUhlenbeck(initial_value="stationary")
        values = {X.draw()(0.0) for _ in range(10)}
        self.assertGreater(len(values), 1)


class TestOrnsteinUhlenbeckStatistics(unittest.TestCase):
    """Simulated moments against the closed-form OU mean and variance."""

    def setUp(self):
        seed()
        self.reversion_rate = 1.5
        self.mean = 4.0
        self.scale = 2.0
        self.initial_value = 10.0
        self.long_run_var = self.scale**2 / (2 * self.reversion_rate)

    def _process(self):
        return OrnsteinUhlenbeck(
            reversion_rate=self.reversion_rate,
            mean=self.mean,
            scale=self.scale,
            initial_value=self.initial_value,
        )

    def test_mean_travels_from_initial_value_toward_mean(self):
        X = self._process()
        for t in [0.25, 1.0, 3.0]:
            expected = self.mean + (self.initial_value - self.mean) * np.exp(
                -self.reversion_rate * t
            )
            self.assertAlmostEqual(X[t].sim(Nsim).mean(), expected, delta=0.15)

    def test_variance_climbs_to_long_run_variance(self):
        X = self._process()
        for t in [0.25, 1.0, 3.0]:
            expected = self.long_run_var * (1 - np.exp(-2 * self.reversion_rate * t))
            self.assertAlmostEqual(X[t].sim(Nsim).var(), expected, delta=0.2)

    def test_long_run_mean_is_mean_parameter(self):
        # Far enough out, the transient has died away.
        X = self._process()
        self.assertAlmostEqual(X[10.0].sim(Nsim).mean(), self.mean, delta=0.2)

    def test_long_run_variance(self):
        X = self._process()
        self.assertAlmostEqual(X[10.0].sim(Nsim).var(), self.long_run_var, delta=0.25)

    def test_reverts_toward_mean_from_above(self):
        # Started above the mean, the process must come down toward it.
        X = self._process()
        early = X[0.25].sim(Nsim).mean()
        late = X[3.0].sim(Nsim).mean()
        self.assertGreater(early, late)
        self.assertGreater(early, self.mean)
        self.assertAlmostEqual(late, self.mean, delta=0.2)

    def test_larger_reversion_rate_stays_closer_to_mean(self):
        # A harder pull means a smaller long-run variance.
        loose = OrnsteinUhlenbeck(reversion_rate=0.5, scale=1)
        tight = OrnsteinUhlenbeck(reversion_rate=5.0, scale=1)
        self.assertGreater(loose[5.0].sim(Nsim).var(), tight[5.0].sim(Nsim).var())


class TestOrnsteinUhlenbeckStationary(unittest.TestCase):
    """The stationary parameterization looks the same at every time."""

    def setUp(self):
        seed()
        self.reversion_rate = 1.0
        self.mean = 4.0
        self.scale = 2.0
        self.long_run_var = self.scale**2 / (2 * self.reversion_rate)
        self.X = OrnsteinUhlenbeck(
            reversion_rate=self.reversion_rate,
            mean=self.mean,
            scale=self.scale,
            initial_value="stationary",
        )

    def test_mean_is_constant_over_time(self):
        for t in [0.0, 2.0, 7.0]:
            self.assertAlmostEqual(self.X[t].sim(Nsim).mean(), self.mean, delta=0.2)

    def test_variance_is_constant_over_time(self):
        for t in [0.0, 2.0, 7.0]:
            self.assertAlmostEqual(
                self.X[t].sim(Nsim).var(), self.long_run_var, delta=0.3
            )

    def test_covariance_decays_exponentially_with_gap(self):
        for s, t in [(1.0, 1.5), (1.0, 3.0)]:
            expected = self.long_run_var * np.exp(-self.reversion_rate * abs(s - t))
            simulated = (self.X[s] & self.X[t]).sim(5000).cov()
            self.assertAlmostEqual(simulated, expected, delta=0.15)

    def test_covariance_depends_only_on_the_gap(self):
        # Stationary: two pairs the same distance apart have the same
        # covariance, wherever they sit on the clock.
        first = (self.X[1.0] & self.X[1.5]).sim(5000).cov()
        second = (self.X[6.0] & self.X[6.5]).sim(5000).cov()
        self.assertAlmostEqual(first, second, delta=0.2)

    def test_defined_at_negative_times(self):
        # The stationary process runs over all of the real line, so negative
        # times are meaningful and have the same variance as positive ones.
        self.assertAlmostEqual(
            self.X[-5.0].sim(Nsim).var(), self.long_run_var, delta=0.3
        )


class TestOrnsteinUhlenbeckRelationships(unittest.TestCase):

    def test_covariance_approaches_brownian_motion_as_pull_vanishes(self):
        # With almost no pull toward the mean, OU's covariance collapses to
        # Brownian motion's scale**2 * min(s, t). Checked on the closed forms
        # rather than by simulation, since both are exact.
        scale, s, t = 1.0, 1.0, 2.0
        brownian = scale**2 * min(s, t)
        previous_gap = float("inf")
        for reversion_rate in [1.0, 0.1, 1e-3, 1e-5]:
            _, cov_func = gaussian_process._ornstein_uhlenbeck_funcs(
                reversion_rate, 0, scale, 0
            )
            gap = abs(cov_func(s, t) - brownian)
            self.assertLess(gap, previous_gap)
            previous_gap = gap
        self.assertAlmostEqual(previous_gap, 0.0, places=4)

    def test_stationary_marginal_is_normal(self):
        # The long-run distribution is Normal(mean, sd=sqrt(long-run var)).
        seed()
        reversion_rate, mean, scale = 2.0, 1.0, 1.5
        long_run_sd = np.sqrt(scale**2 / (2 * reversion_rate))
        X = OrnsteinUhlenbeck(
            reversion_rate=reversion_rate,
            mean=mean,
            scale=scale,
            initial_value="stationary",
        )
        sims = X[3.0].sim(2000)
        pvalue = stats.kstest(list(sims), stats.norm(mean, long_run_sd).cdf).pvalue
        self.assertGreater(pvalue, 0.01)


class TestOrnsteinUhlenbeckErrors(unittest.TestCase):
    """Error handling for OrnsteinUhlenbeck and its probability space."""

    def test_non_numeric_reversion_rate_raises_type_error(self):
        with self.assertRaises(TypeError):
            OrnsteinUhlenbeck(reversion_rate="fast")

    def test_non_numeric_mean_raises_type_error(self):
        with self.assertRaises(TypeError):
            OrnsteinUhlenbeck(mean="middle")

    def test_non_numeric_scale_raises_type_error(self):
        with self.assertRaises(TypeError):
            OrnsteinUhlenbeck(scale="large")

    def test_unknown_initial_value_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            OrnsteinUhlenbeck(initial_value="steady")

    def test_zero_reversion_rate_raises_value_error(self):
        # No pull toward the mean is Brownian motion, not an OU process, and
        # the long-run variance would divide by zero.
        with self.assertRaises(ValueError):
            OrnsteinUhlenbeck(reversion_rate=0)

    def test_negative_reversion_rate_raises_value_error(self):
        with self.assertRaises(ValueError):
            OrnsteinUhlenbeck(reversion_rate=-1)

    def test_zero_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            OrnsteinUhlenbeck(scale=0)

    def test_negative_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            OrnsteinUhlenbeck(scale=-1)

    def test_reversion_rate_error_message_points_to_brownian_motion(self):
        with self.assertRaisesRegex(ValueError, "BrownianMotion"):
            OrnsteinUhlenbeck(reversion_rate=0)

    def test_negative_mean_and_initial_value_are_valid(self):
        # Only reversion_rate and scale are restricted in sign.
        seed()
        X = OrnsteinUhlenbeck(reversion_rate=1, mean=-3.0, initial_value=-8.0)
        self.assertEqual(X.draw()(0), -8.0)
        self.assertIsInstance(X.draw()(1.0), float)

    def test_probability_space_validates_too(self):
        with self.assertRaises(ValueError):
            OrnsteinUhlenbeckProbabilitySpace(scale=-1)


if __name__ == "__main__":
    unittest.main()
