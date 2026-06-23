"""Tests for symbulate.gaussian_process.

Covers GaussianProcess (construction, path evaluation, lazy caching,
statistical properties) and BrownianMotion (deterministic start, mean,
variance, drift, and scale parameters).
"""
import unittest
import numpy as np

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
            path(float('inf'))


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
            lambda t: 0,
            lambda s, t: min(s, t),
            DiscreteTimeSequence(fs=4)
        )
        path = X.draw()
        # index 4 maps to time 4/4 = 1.0
        result = path[4]
        self.assertIsInstance(result, float)

    def test_discrete_time_caches_index(self):
        seed()
        X = GaussianProcess(
            lambda t: 0,
            lambda s, t: min(s, t),
            DiscreteTimeSequence(fs=4)
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
        self.assertAlmostEqual(simulated_var, scale ** 2, delta=0.5)

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


if __name__ == '__main__':
    unittest.main()
