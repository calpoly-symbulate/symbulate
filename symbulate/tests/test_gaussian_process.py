"""Tests for symbulate.gaussian_process.

Covers GaussianProcess (construction, path evaluation, lazy caching,
statistical properties), BrownianMotion (deterministic start, mean,
variance, drift, and scale parameters), OrnsteinUhlenbeck (both
initial-value parameterizations, mean reversion, the long-run
distribution, and the Brownian-motion limit), BrownianBridge (both
endpoints pinned, the widest-in-the-middle spread, and the limited time
domain), FractionalBrownianMotion (variance growth, self-similarity, the
sign of the increment correlation either side of hurst=0.5),
GeometricBrownianMotion (positivity, the log-normal marginal, and that it is
exactly the exponential of its own Brownian path), and the TimeInterval index
set the bridge is defined over.
"""

import unittest
import numpy as np
import scipy.stats as stats

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt

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


class TestBrownianBridgeConstruction(unittest.TestCase):

    def test_is_random_process(self):
        self.assertIsInstance(BrownianBridge(), RandomProcess)

    def test_is_rv(self):
        self.assertIsInstance(BrownianBridge(), RV)

    def test_probability_space_type(self):
        X = BrownianBridge()
        self.assertIsInstance(X.prob_space, BrownianBridgeProbabilitySpace)

    def test_is_gaussian_process_probability_space(self):
        X = BrownianBridge()
        self.assertIsInstance(X.prob_space, GaussianProcessProbabilitySpace)

    def test_parameters_stored_on_probability_space(self):
        P = BrownianBridgeProbabilitySpace(
            end_time=4, initial_value=2, final_value=10, scale=1.5
        )
        self.assertEqual(P.end_time, 4)
        self.assertEqual(P.initial_value, 2)
        self.assertEqual(P.final_value, 10)
        self.assertEqual(P.scale, 1.5)


class TestBrownianBridgeEndpoints(unittest.TestCase):
    """Both ends are pinned exactly, which is what makes it a bridge."""

    def test_starts_and_ends_at_zero_by_default(self):
        seed()
        X = BrownianBridge()
        for _ in range(10):
            path = X.draw()
            self.assertEqual(path(0), 0)
            self.assertEqual(path(1), 0)

    def test_hits_both_given_endpoints_exactly(self):
        seed()
        X = BrownianBridge(end_time=4, initial_value=2, final_value=10)
        for _ in range(10):
            path = X.draw()
            self.assertEqual(path(0), 2)
            self.assertEqual(path(4), 10)

    def test_endpoints_exact_even_after_visiting_other_times(self):
        # Pinning must survive conditioning on interior points.
        seed()
        path = BrownianBridge(end_time=2, final_value=5).draw()
        path(0.5)
        path(1.0)
        path(1.5)
        self.assertEqual(path(0), 0)
        self.assertEqual(path(2), 5)

    def test_endpoints_have_no_randomness(self):
        seed()
        X = BrownianBridge(end_time=3, initial_value=1, final_value=7)
        self.assertEqual({X.draw()(0) for _ in range(10)}, {1})
        self.assertEqual({X.draw()(3) for _ in range(10)}, {7})

    def test_interior_is_random(self):
        seed()
        X = BrownianBridge()
        self.assertGreater(len({X.draw()(0.5) for _ in range(10)}), 1)


class TestBrownianBridgeStatistics(unittest.TestCase):

    def setUp(self):
        seed()
        self.end_time = 4.0
        self.initial_value = 2.0
        self.final_value = 10.0
        self.scale = 1.5
        self.X = BrownianBridge(
            end_time=self.end_time,
            initial_value=self.initial_value,
            final_value=self.final_value,
            scale=self.scale,
        )

    def _expected_mean(self, t):
        slope = (self.final_value - self.initial_value) / self.end_time
        return self.initial_value + slope * t

    def _expected_var(self, t):
        return self.scale**2 * t * (self.end_time - t) / self.end_time

    def test_mean_is_a_straight_line_between_the_ends(self):
        for t in [0.5, 2.0, 3.5]:
            self.assertAlmostEqual(
                self.X[t].sim(Nsim).mean(), self._expected_mean(t), delta=0.15
            )

    def test_variance_matches_closed_form(self):
        for t in [0.5, 2.0, 3.5]:
            self.assertAlmostEqual(
                self.X[t].sim(Nsim).var(), self._expected_var(t), delta=0.3
            )

    def test_variance_is_widest_in_the_middle(self):
        middle = self.X[2.0].sim(Nsim).var()
        near_start = self.X[0.5].sim(Nsim).var()
        near_end = self.X[3.5].sim(Nsim).var()
        self.assertGreater(middle, near_start)
        self.assertGreater(middle, near_end)

    def test_variance_is_symmetric_about_the_middle(self):
        # t and end_time - t have the same spread.
        self.assertAlmostEqual(
            self.X[0.5].sim(Nsim).var(), self.X[3.5].sim(Nsim).var(), delta=0.3
        )

    def test_covariance_matches_closed_form(self):
        for s, t in [(1.0, 3.0), (0.5, 1.0)]:
            expected = self.scale**2 * (min(s, t) - s * t / self.end_time)
            self.assertAlmostEqual(
                (self.X[s] & self.X[t]).sim(5000).cov(), expected, delta=0.2
            )

    def test_scale_widens_the_middle(self):
        seed()
        narrow = BrownianBridge(scale=1)[0.5].sim(Nsim).var()
        wide = BrownianBridge(scale=3)[0.5].sim(Nsim).var()
        self.assertGreater(wide, narrow)


class TestBrownianBridgeDomain(unittest.TestCase):
    """The bridge exists only between its two ends."""

    def test_index_set_is_the_bridge_interval(self):
        X = BrownianBridge(end_time=3)
        self.assertEqual(X.draw().index_set, TimeInterval(0, 3))

    def test_time_after_end_raises_key_error(self):
        seed()
        path = BrownianBridge(end_time=1).draw()
        with self.assertRaises(KeyError):
            path(1.5)

    def test_time_before_start_raises_key_error(self):
        seed()
        path = BrownianBridge(end_time=1).draw()
        with self.assertRaises(KeyError):
            path(-0.5)

    def test_error_message_names_the_interval(self):
        seed()
        path = BrownianBridge(end_time=1).draw()
        with self.assertRaisesRegex(KeyError, r"TimeInterval\(0, 1\)"):
            path(1.5)

    def test_the_two_ends_themselves_are_allowed(self):
        seed()
        path = BrownianBridge(end_time=1).draw()
        self.assertEqual(path(0), 0)
        self.assertEqual(path(1), 0)


class TestBrownianBridgePlot(unittest.TestCase):
    """The plot's time axis defaults to the bridge's own interval."""

    def _last_xdata(self):
        return plt.gca().lines[-1].get_xdata()

    def test_plot_defaults_to_the_bridge_interval(self):
        seed()
        plt.figure()
        BrownianBridge(end_time=3).draw().plot()
        xs = self._last_xdata()
        self.assertAlmostEqual(xs[0], 0)
        self.assertAlmostEqual(xs[-1], 3)
        plt.close("all")

    def test_plot_does_not_raise_key_error(self):
        # Regression: the old default tmax=10 evaluated past the bridge's
        # domain and raised KeyError.
        seed()
        plt.figure()
        BrownianBridge(end_time=2).draw().plot()  # must not raise
        plt.close("all")

    def test_explicit_tmax_overrides_the_default(self):
        seed()
        plt.figure()
        BrownianBridge(end_time=3).draw().plot(tmax=1.5)
        self.assertAlmostEqual(self._last_xdata()[-1], 1.5)
        plt.close("all")


class TestBrownianBridgeRelationships(unittest.TestCase):

    def test_is_brownian_motion_with_the_far_end_subtracted(self):
        # B(t) - (t / T) * B(T) is a standard Brownian bridge, so its
        # variance at t must match the bridge's t * (T - t) / T.
        seed()
        end_time = 2.0
        B = BrownianMotion()
        bridge_like = B[1.0] - (1.0 / end_time) * B[end_time]
        expected = 1.0 * (end_time - 1.0) / end_time
        self.assertAlmostEqual(bridge_like.sim(2000).var(), expected, delta=0.15)

    def test_standard_bridge_marginal_is_normal(self):
        seed()
        X = BrownianBridge()
        sims = X[0.5].sim(2000)
        # Var at the midpoint of a standard bridge is 0.5 * 0.5 / 1 = 0.25.
        cdf = stats.norm(0, np.sqrt(0.25)).cdf
        self.assertGreater(stats.kstest(list(sims), cdf).pvalue, 0.01)


class TestBrownianBridgeErrors(unittest.TestCase):

    def test_non_numeric_end_time_raises_type_error(self):
        with self.assertRaises(TypeError):
            BrownianBridge(end_time="later")

    def test_non_numeric_initial_value_raises_type_error(self):
        with self.assertRaises(TypeError):
            BrownianBridge(initial_value="low")

    def test_non_numeric_final_value_raises_type_error(self):
        with self.assertRaises(TypeError):
            BrownianBridge(final_value="high")

    def test_non_numeric_scale_raises_type_error(self):
        with self.assertRaises(TypeError):
            BrownianBridge(scale="wide")

    def test_zero_end_time_raises_value_error(self):
        with self.assertRaises(ValueError):
            BrownianBridge(end_time=0)

    def test_negative_end_time_raises_value_error(self):
        with self.assertRaises(ValueError):
            BrownianBridge(end_time=-1)

    def test_zero_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            BrownianBridge(scale=0)

    def test_negative_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            BrownianBridge(scale=-1)

    def test_negative_endpoint_values_are_valid(self):
        # Only end_time and scale are restricted in sign.
        seed()
        path = BrownianBridge(initial_value=-5, final_value=-2).draw()
        self.assertEqual(path(0), -5)
        self.assertEqual(path(1), -2)

    def test_probability_space_validates_too(self):
        with self.assertRaises(ValueError):
            BrownianBridgeProbabilitySpace(end_time=0)


class TestTimeInterval(unittest.TestCase):
    """The index set backing the Brownian bridge."""

    def test_contains_interior_and_both_ends(self):
        times = TimeInterval(0, 1)
        for t in [0, 0.5, 1]:
            self.assertIn(t, times)

    def test_excludes_times_outside(self):
        times = TimeInterval(0, 1)
        for t in [-0.5, 1.5, float("inf")]:
            self.assertNotIn(t, times)

    def test_excludes_non_numeric(self):
        self.assertNotIn("half", TimeInterval(0, 1))

    def test_is_continuous_time(self):
        # Subclassing Reals is what makes a process over it continuous-time.
        self.assertIsInstance(TimeInterval(0, 1), Reals)

    def test_equality_compares_bounds(self):
        self.assertEqual(TimeInterval(0, 1), TimeInterval(0, 1))
        self.assertNotEqual(TimeInterval(0, 1), TimeInterval(0, 2))
        self.assertNotEqual(TimeInterval(0, 1), Reals())

    def test_repr_shows_bounds(self):
        self.assertEqual(repr(TimeInterval(0, 2)), "TimeInterval(0, 2)")

    def test_base_repr_is_the_class_name(self):
        self.assertEqual(repr(Reals()), "Reals")

    def test_non_numeric_bounds_raise_type_error(self):
        with self.assertRaises(TypeError):
            TimeInterval("a", 1)
        with self.assertRaises(TypeError):
            TimeInterval(0, "b")

    def test_end_not_after_start_raises_value_error(self):
        with self.assertRaises(ValueError):
            TimeInterval(1, 1)
        with self.assertRaises(ValueError):
            TimeInterval(2, 1)

    def test_usable_directly_with_gaussian_process(self):
        seed()
        X = GaussianProcess(
            lambda t: 0, lambda s, t: min(s, t), index_set=TimeInterval(0, 5)
        )
        path = X.draw()
        self.assertIsInstance(path(2.0), float)
        with self.assertRaises(KeyError):
            path(6.0)


class TestFractionalBrownianMotionConstruction(unittest.TestCase):

    def test_is_random_process(self):
        self.assertIsInstance(FractionalBrownianMotion(), RandomProcess)

    def test_is_rv(self):
        self.assertIsInstance(FractionalBrownianMotion(), RV)

    def test_probability_space_type(self):
        X = FractionalBrownianMotion()
        self.assertIsInstance(X.prob_space, FractionalBrownianMotionProbabilitySpace)

    def test_is_gaussian_process_probability_space(self):
        X = FractionalBrownianMotion()
        self.assertIsInstance(X.prob_space, GaussianProcessProbabilitySpace)

    def test_parameters_stored_on_probability_space(self):
        P = FractionalBrownianMotionProbabilitySpace(hurst=0.8, scale=2.0)
        self.assertEqual(P.hurst, 0.8)
        self.assertEqual(P.scale, 2.0)


class TestFractionalBrownianMotionPaths(unittest.TestCase):

    def test_starts_at_zero(self):
        # cov(0, 0) = 0 for every hurst, so time 0 is deterministic.
        seed()
        for hurst in [0.2, 0.5, 0.9]:
            X = FractionalBrownianMotion(hurst=hurst)
            for _ in range(5):
                self.assertEqual(X.draw()(0), 0)

    def test_same_time_returns_cached_value(self):
        seed()
        path = FractionalBrownianMotion(hurst=0.8).draw()
        self.assertEqual(path(1.0), path(1.0))

    def test_cached_value_unchanged_after_new_time(self):
        seed()
        path = FractionalBrownianMotion(hurst=0.8).draw()
        before = path(1.0)
        path(2.0)
        self.assertEqual(before, path(1.0))

    def test_different_draws_produce_different_paths(self):
        seed()
        X = FractionalBrownianMotion(hurst=0.7)
        self.assertGreater(len({X.draw()(1.0) for _ in range(10)}), 1)


class TestFractionalBrownianMotionStatistics(unittest.TestCase):

    def test_variance_matches_closed_form(self):
        # Var(X(t)) = scale**2 * t**(2 * hurst).
        scale = 1.5
        for hurst in [0.3, 0.5, 0.8]:
            seed()
            X = FractionalBrownianMotion(hurst=hurst, scale=scale)
            for t in [0.5, 2.0]:
                expected = scale**2 * t ** (2 * hurst)
                self.assertAlmostEqual(
                    X[t].sim(Nsim).var(), expected, delta=0.15 + 0.1 * expected
                )

    def test_mean_is_zero(self):
        seed()
        X = FractionalBrownianMotion(hurst=0.8)
        self.assertAlmostEqual(X[2.0].sim(Nsim).mean(), 0.0, delta=0.3)

    def test_larger_hurst_spreads_faster(self):
        seed()
        low = FractionalBrownianMotion(hurst=0.3)[4.0].sim(Nsim).var()
        high = FractionalBrownianMotion(hurst=0.8)[4.0].sim(Nsim).var()
        self.assertGreater(high, low)

    def test_covariance_matches_closed_form(self):
        seed()
        hurst, scale = 0.7, 1.0
        X = FractionalBrownianMotion(hurst=hurst, scale=scale)
        for s, t in [(1.0, 2.0), (0.5, 1.5)]:
            expected = (scale**2 / 2) * (
                abs(s) ** (2 * hurst)
                + abs(t) ** (2 * hurst)
                - abs(s - t) ** (2 * hurst)
            )
            self.assertAlmostEqual((X[s] & X[t]).sim(5000).cov(), expected, delta=0.25)

    def test_self_similar(self):
        # Stretching time by a scales the spread by a**(2 * hurst).
        seed()
        hurst = 0.7
        X = FractionalBrownianMotion(hurst=hurst)
        at_one = X[1.0].sim(Nsim).var()
        at_four = X[4.0].sim(Nsim).var()
        self.assertAlmostEqual(at_four / at_one, 4 ** (2 * hurst), delta=0.6)

    def test_defined_at_negative_times(self):
        # The absolute values in the covariance keep it valid below 0.
        seed()
        X = FractionalBrownianMotion(hurst=0.7)
        self.assertAlmostEqual(X[-2.0].sim(Nsim).var(), 2 ** (2 * 0.7), delta=0.5)


class TestFractionalBrownianMotionMemory(unittest.TestCase):
    """The Hurst parameter is exactly the sign of the increment correlation."""

    def _increment_cov(self, hurst):
        seed()
        X = FractionalBrownianMotion(hurst=hurst)
        return (X[1.0] & (X[2.0] - X[1.0])).sim(5000).cov()

    def test_hurst_above_half_keeps_going_the_same_way(self):
        self.assertGreater(self._increment_cov(0.8), 0.1)

    def test_hurst_below_half_reverses(self):
        self.assertLess(self._increment_cov(0.3), -0.1)

    def test_hurst_half_has_independent_increments(self):
        self.assertAlmostEqual(self._increment_cov(0.5), 0.0, delta=0.1)

    def test_increment_covariance_matches_closed_form(self):
        # Cov(X(1), X(2) - X(1)) = scale**2 * (2**(2 * hurst - 1) - 1).
        for hurst in [0.3, 0.8]:
            expected = 2 ** (2 * hurst - 1) - 1
            self.assertAlmostEqual(self._increment_cov(hurst), expected, delta=0.2)


class TestFractionalBrownianMotionRelationships(unittest.TestCase):

    def test_hurst_half_is_brownian_motion_covariance(self):
        # At hurst = 0.5 the covariance is exactly scale**2 * min(s, t).
        P = FractionalBrownianMotionProbabilitySpace(hurst=0.5, scale=2.0)
        cov_func = P.cov_func
        for s, t in [(1.0, 2.0), (0.5, 3.0), (2.0, 2.0)]:
            self.assertAlmostEqual(cov_func(s, t), 4.0 * min(s, t), places=12)

    def test_hurst_half_matches_brownian_motion_variance(self):
        seed()
        X = FractionalBrownianMotion(hurst=0.5)
        self.assertAlmostEqual(X[3.0].sim(Nsim).var(), 3.0, delta=0.5)

    def test_marginal_is_normal(self):
        seed()
        hurst = 0.7
        X = FractionalBrownianMotion(hurst=hurst)
        sims = X[2.0].sim(2000)
        sd = np.sqrt(2 ** (2 * hurst))
        pvalue = stats.kstest(list(sims), stats.norm(0, sd).cdf).pvalue
        self.assertGreater(pvalue, 0.01)


class TestFractionalBrownianMotionErrors(unittest.TestCase):

    def test_non_numeric_hurst_raises_type_error(self):
        with self.assertRaises(TypeError):
            FractionalBrownianMotion(hurst="smooth")

    def test_non_numeric_scale_raises_type_error(self):
        with self.assertRaises(TypeError):
            FractionalBrownianMotion(scale="big")

    def test_hurst_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            FractionalBrownianMotion(hurst=0)

    def test_hurst_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            FractionalBrownianMotion(hurst=1)

    def test_hurst_above_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            FractionalBrownianMotion(hurst=1.5)

    def test_negative_hurst_raises_value_error(self):
        with self.assertRaises(ValueError):
            FractionalBrownianMotion(hurst=-0.2)

    def test_hurst_error_message_explains_the_range(self):
        with self.assertRaisesRegex(ValueError, "strictly between 0 and 1"):
            FractionalBrownianMotion(hurst=2)

    def test_zero_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            FractionalBrownianMotion(scale=0)

    def test_negative_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            FractionalBrownianMotion(scale=-1)

    def test_hurst_near_the_ends_is_valid(self):
        seed()
        for hurst in [0.01, 0.99]:
            path = FractionalBrownianMotion(hurst=hurst).draw()
            self.assertIsInstance(path(1.0), float)

    def test_probability_space_validates_too(self):
        with self.assertRaises(ValueError):
            FractionalBrownianMotionProbabilitySpace(hurst=0)


class TestGeometricBrownianMotionConstruction(unittest.TestCase):

    def test_is_random_process(self):
        self.assertIsInstance(GeometricBrownianMotion(), RandomProcess)

    def test_is_rv(self):
        self.assertIsInstance(GeometricBrownianMotion(), RV)

    def test_probability_space_type(self):
        X = GeometricBrownianMotion()
        self.assertIsInstance(X.prob_space, GeometricBrownianMotionProbabilitySpace)

    def test_is_not_a_gaussian_process(self):
        # It is the exponential of one, which is log-normal, not normal.
        X = GeometricBrownianMotion()
        self.assertNotIsInstance(X.prob_space, GaussianProcessProbabilitySpace)

    def test_parameters_stored_on_probability_space(self):
        P = GeometricBrownianMotionProbabilitySpace(
            initial_value=100, growth_rate=0.05, scale=0.2
        )
        self.assertEqual(P.initial_value, 100)
        self.assertEqual(P.growth_rate, 0.05)
        self.assertEqual(P.scale, 0.2)


class TestGeometricBrownianMotionPaths(unittest.TestCase):

    def test_starts_at_initial_value(self):
        seed()
        for initial_value in [1, 100, 0.5]:
            X = GeometricBrownianMotion(initial_value=initial_value)
            for _ in range(5):
                self.assertAlmostEqual(float(X.draw()(0)), float(initial_value))

    def test_same_time_returns_cached_value(self):
        seed()
        path = GeometricBrownianMotion(initial_value=100).draw()
        self.assertEqual(path(1.0), path(1.0))

    def test_cached_value_unchanged_after_zooming_in(self):
        seed()
        path = GeometricBrownianMotion(initial_value=100).draw()
        before = float(path(1.0)), float(path(2.0))
        for t in [1.1, 1.25, 1.5, 1.75, 1.9]:
            path(t)
        self.assertEqual((float(path(1.0)), float(path(2.0))), before)

    def test_different_draws_produce_different_paths(self):
        seed()
        X = GeometricBrownianMotion(initial_value=100)
        self.assertGreater(len({float(X.draw()(1.0)) for _ in range(10)}), 1)

    def test_stays_positive(self):
        # Multiplying by a positive factor can never reach 0.
        seed()
        X = GeometricBrownianMotion(initial_value=100, growth_rate=0, scale=0.8)
        self.assertTrue(all(value > 0 for value in X[5.0].sim(500)))

    def test_negative_time_raises_value_error(self):
        seed()
        path = GeometricBrownianMotion().draw()
        with self.assertRaisesRegex(ValueError, "only defined for t >= 0"):
            path(-1.0)


class TestGeometricBrownianMotionStatistics(unittest.TestCase):

    def setUp(self):
        seed()
        self.initial_value = 100.0
        self.growth_rate = 0.05
        self.scale = 0.3
        self.X = GeometricBrownianMotion(
            initial_value=self.initial_value,
            growth_rate=self.growth_rate,
            scale=self.scale,
        )

    def _expected_mean(self, t):
        return self.initial_value * np.exp(self.growth_rate * t)

    def test_mean_grows_exponentially_at_the_growth_rate(self):
        for t in [1.0, 3.0]:
            expected = self._expected_mean(t)
            self.assertAlmostEqual(
                self.X[t].sim(Nsim).mean(), expected, delta=0.05 * expected
            )

    def test_variance_matches_closed_form(self):
        for t in [1.0, 3.0]:
            expected = self._expected_mean(t) ** 2 * (np.exp(self.scale**2 * t) - 1)
            self.assertAlmostEqual(
                self.X[t].sim(Nsim).var(), expected, delta=0.35 * expected
            )

    def test_zero_growth_rate_keeps_the_mean_flat(self):
        # growth_rate is the growth rate of the mean, so 0 means no growth.
        seed()
        X = GeometricBrownianMotion(initial_value=50, growth_rate=0, scale=0.2)
        self.assertAlmostEqual(X[5.0].sim(Nsim).mean(), 50.0, delta=2.0)

    def test_larger_scale_spreads_wider(self):
        seed()
        calm = GeometricBrownianMotion(initial_value=100, scale=0.1)
        wild = GeometricBrownianMotion(initial_value=100, scale=0.5)
        self.assertGreater(wild[3.0].sim(Nsim).var(), calm[3.0].sim(Nsim).var())


class TestGeometricBrownianMotionRelationships(unittest.TestCase):

    def test_log_of_the_path_is_normal(self):
        # log(value(t) / initial_value) is Normal with mean
        # (growth_rate - scale**2 / 2) * t and sd scale * sqrt(t).
        seed()
        initial_value, growth_rate, scale, t = 100.0, 0.05, 0.3, 2.0
        X = GeometricBrownianMotion(
            initial_value=initial_value, growth_rate=growth_rate, scale=scale
        )
        values = np.array(list(X[t].sim(2000)), dtype=float)
        logs = np.log(values / initial_value)
        cdf = stats.norm((growth_rate - scale**2 / 2) * t, scale * np.sqrt(t)).cdf
        self.assertGreater(stats.kstest(logs, cdf).pvalue, 0.01)

    def test_path_is_exactly_the_exponential_of_its_brownian_motion(self):
        # Not an approximation: the value is the closed-form transform of the
        # underlying Brownian path, to the last floating-point bit.
        seed()
        initial_value, growth_rate, scale = 100.0, 0.05, 0.3
        path = GeometricBrownianMotion(
            initial_value=initial_value, growth_rate=growth_rate, scale=scale
        ).draw()
        for t in [0.4, 1.7, 5.0]:
            expected = initial_value * np.exp(
                (growth_rate - scale**2 / 2) * t + scale * path.brownian_path(t)
            )
            self.assertEqual(float(path(t)), float(expected))

    def test_underlying_brownian_path_starts_at_zero(self):
        seed()
        path = GeometricBrownianMotion(initial_value=100).draw()
        path(1.0)
        self.assertEqual(path.brownian_path(0), 0)

    def test_median_falls_while_the_mean_rises(self):
        # When growth_rate < scale**2 / 2 the mean still climbs, pulled up by
        # rare very large values, while a typical path drifts toward 0.
        #
        # The two closed forms are checked directly, because at this much
        # skew the *sample* mean is useless as evidence: the standard error of
        # the mean here is around 8, so a simulated mean can easily land below
        # the starting value even though the true mean is above it. The
        # median is robust to the skew, so that is the half that is checked
        # against simulation.
        seed()
        initial_value, growth_rate, scale, t = 100.0, 0.01, 0.5, 10.0
        true_mean = initial_value * np.exp(growth_rate * t)
        true_median = initial_value * np.exp((growth_rate - scale**2 / 2) * t)
        self.assertGreater(true_mean, initial_value)
        self.assertLess(true_median, initial_value)

        X = GeometricBrownianMotion(
            initial_value=initial_value, growth_rate=growth_rate, scale=scale
        )
        values = np.array(list(X[t].sim(2000)), dtype=float)
        self.assertLess(np.median(values), initial_value)
        self.assertAlmostEqual(np.median(values), true_median, delta=0.3 * true_median)


class TestGeometricBrownianMotionErrors(unittest.TestCase):

    def test_non_numeric_initial_value_raises_type_error(self):
        with self.assertRaises(TypeError):
            GeometricBrownianMotion(initial_value="high")

    def test_non_numeric_growth_rate_raises_type_error(self):
        with self.assertRaises(TypeError):
            GeometricBrownianMotion(growth_rate="fast")

    def test_non_numeric_scale_raises_type_error(self):
        with self.assertRaises(TypeError):
            GeometricBrownianMotion(scale="wide")

    def test_zero_initial_value_raises_value_error(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(initial_value=0)

    def test_negative_initial_value_raises_value_error(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(initial_value=-5)

    def test_initial_value_error_explains_why(self):
        # Called through the older `initial_value` spelling on purpose, so
        # this doubles as alias coverage; the message itself now uses the
        # current name.
        with self.assertRaisesRegex(ValueError, "initial must be positive"):
            GeometricBrownianMotion(initial_value=0)

    def test_zero_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(scale=0)

    def test_negative_scale_raises_value_error(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotion(scale=-1)

    def test_negative_growth_rate_is_valid(self):
        # A shrinking price is perfectly meaningful.
        seed()
        X = GeometricBrownianMotion(initial_value=100, growth_rate=-0.1)
        self.assertLess(X[5.0].sim(Nsim).mean(), 100)

    def test_probability_space_validates_too(self):
        with self.assertRaises(ValueError):
            GeometricBrownianMotionProbabilitySpace(initial_value=0)


class TestInitialNaming(unittest.TestCase):
    """`initial` is the shared name; the older spellings still work.

    Every process is standardising on `initial` for its starting condition
    (MODEL-DECISIONS.md, "One Name for a Process's Starting Condition").
    `initial_value` -- and `final_value` on the Brownian bridge -- are kept
    as accepted aliases so existing code and notebooks do not break. The
    other 75-odd references in this file still use the old names, which is
    itself the compatibility test.
    """

    def test_ornstein_uhlenbeck_both_names(self):
        self.assertEqual(
            float(OrnsteinUhlenbeck(initial=5).draw()(0)),
            float(OrnsteinUhlenbeck(initial_value=5).draw()(0)),
        )

    def test_ornstein_uhlenbeck_stationary_still_accepted(self):
        # The sentinel has to survive the rename.
        path = OrnsteinUhlenbeck(reversion_rate=1, scale=1, initial="stationary").draw()
        self.assertIsInstance(float(path(1.0)), float)

    def test_ornstein_uhlenbeck_both_names_at_once_raises(self):
        self.assertRaisesRegex(
            ValueError,
            "not both",
            lambda: OrnsteinUhlenbeck(initial=1, initial_value=2),
        )

    def test_brownian_bridge_both_names(self):
        a = BrownianBridge(initial=2, final=9).draw()
        b = BrownianBridge(initial_value=2, final_value=9).draw()
        self.assertEqual(float(a(0)), float(b(0)))
        # The bridge is pinned at both ends, whichever spelling was used.
        self.assertAlmostEqual(float(a(1.0)), 9.0, places=6)
        self.assertAlmostEqual(float(b(1.0)), 9.0, places=6)

    def test_brownian_bridge_final_alias_alone(self):
        path = BrownianBridge(initial=1, final_value=4).draw()
        self.assertAlmostEqual(float(path(0)), 1.0, places=6)
        self.assertAlmostEqual(float(path(1.0)), 4.0, places=6)

    def test_geometric_brownian_motion_both_names(self):
        self.assertEqual(
            float(GeometricBrownianMotion(initial=100).draw()(0)),
            float(GeometricBrownianMotion(initial_value=100).draw()(0)),
        )

    def test_geometric_brownian_motion_default_is_still_one(self):
        self.assertEqual(float(GeometricBrownianMotion().draw()(0)), 1.0)

    def test_initial_attribute_readable(self):
        self.assertEqual(OrnsteinUhlenbeck(initial=3).initial, 3)
        self.assertEqual(BrownianBridge(initial=3, final=4).final, 4)
        self.assertEqual(GeometricBrownianMotion(initial=7).initial, 7)


if __name__ == "__main__":
    unittest.main()
