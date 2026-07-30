"""Tests for symbulate.diffusion_process.

Covers DiffusionProcess (construction, the starting value, lazy caching and
zoom consistency, error handling) plus the two cases where the
approximation has a known exact answer to check against: constant
coefficients, which reduce to Brownian motion, and a mean-reverting drift,
which reproduces the Ornstein-Uhlenbeck long-run variance.

Also pins the seeding behavior: the module draws from its own ``rng``, so
seeding that is what makes a path reproducible. ``numpy.random.seed`` does
not, which was a real bug in this module's demo.
"""

import unittest
import numpy as np

from symbulate import *
from symbulate import diffusion_process

Nsim = 1000


def seed(value=42):
    diffusion_process.rng = np.random.default_rng(value)


def brownian_like(scale=1.0, x0=0, tol=1e-2):
    """A diffusion with constant coefficients -- Brownian motion."""
    return DiffusionProcess(
        drift=lambda x, t: 0, diffusion=lambda x, t: scale, x0=x0, tol=tol
    )


class TestDiffusionProcessConstruction(unittest.TestCase):

    def test_is_random_process(self):
        self.assertIsInstance(brownian_like(), RandomProcess)

    def test_is_rv(self):
        self.assertIsInstance(brownian_like(), RV)

    def test_probability_space_type(self):
        self.assertIsInstance(
            brownian_like().prob_space, DiffusionProcessProbabilitySpace
        )

    def test_probability_space_draws_a_path(self):
        seed()
        P = DiffusionProcessProbabilitySpace(
            drift=lambda x, t: 0, diffusion=lambda x, t: 1
        )
        self.assertIsInstance(float(P.draw()(1.0)), float)


class TestDiffusionProcessPaths(unittest.TestCase):

    def test_starts_at_x0(self):
        seed()
        for x0 in [0, 100, -5.5]:
            path = brownian_like(x0=x0).draw()
            self.assertEqual(path(0), float(x0))

    def test_same_time_returns_cached_value(self):
        seed()
        path = brownian_like().draw()
        self.assertEqual(path(1.0), path(1.0))

    def test_cached_value_unchanged_after_a_later_time(self):
        seed()
        path = brownian_like().draw()
        before = path(1.0)
        path(5.0)
        self.assertEqual(before, path(1.0))

    def test_cached_value_unchanged_after_zooming_in(self):
        # The point of the bridge sampling: filling in detail between two
        # known times must not rewrite either of them.
        seed()
        path = brownian_like().draw()
        at_one, at_two = path(1.0), path(2.0)
        for t in [1.1, 1.5, 1.9, 1.25, 1.75]:
            path(t)
        self.assertEqual(path(1.0), at_one)
        self.assertEqual(path(2.0), at_two)

    def test_zoomed_values_lie_between_their_neighbors_in_time(self):
        # Every bridge value is inserted in time order, so the cached times
        # stay sorted no matter what order they were asked for.
        seed()
        path = brownian_like().draw()
        for t in [3.0, 0.5, 2.0, 1.0, 2.5]:
            path(t)
        self.assertEqual(path.times, sorted(path.times))

    def test_different_draws_produce_different_paths(self):
        seed()
        X = brownian_like()
        self.assertGreater(len({float(X.draw()(1.0)) for _ in range(10)}), 1)

    def test_negative_time_raises_value_error(self):
        seed()
        path = brownian_like().draw()
        with self.assertRaises(ValueError):
            path(-1.0)

    def test_negative_time_message_explains_the_start(self):
        seed()
        path = brownian_like().draw()
        with self.assertRaisesRegex(ValueError, "only defined for t >= 0"):
            path(-1.0)


class TestDiffusionProcessSeeding(unittest.TestCase):
    """Seeding the module's own rng is what makes a path reproducible.

    Regression test for the demo bug fixed alongside merging this module:
    ``numpy.random.seed`` seeds the legacy global generator, which
    ``numpy.random.default_rng`` does not read.
    """

    def _one_draw(self):
        X = brownian_like(scale=0.4, x0=100)
        return float(X.draw()(1.0))

    def test_seeding_module_rng_is_reproducible(self):
        values = []
        for _ in range(3):
            seed(42)
            values.append(self._one_draw())
        self.assertEqual(len(set(values)), 1)

    def test_different_seeds_give_different_paths(self):
        seed(1)
        first = self._one_draw()
        seed(2)
        second = self._one_draw()
        self.assertNotEqual(first, second)

    def test_numpy_random_seed_does_not_control_the_path(self):
        # If this ever starts passing, the module has been switched back to
        # the legacy global generator.
        seed()
        values = []
        for _ in range(3):
            np.random.seed(42)
            values.append(self._one_draw())
        self.assertGreater(len(set(values)), 1)


class TestDiffusionProcessBrownianCase(unittest.TestCase):
    """Constant drift and diffusion make the scheme exact."""

    def test_variance_matches_brownian_motion(self):
        for scale in [1.0, 2.0]:
            seed()
            X = brownian_like(scale=scale)
            for t in [1.0, 3.0]:
                expected = scale**2 * t
                self.assertAlmostEqual(
                    X[t].sim(Nsim).var(), expected, delta=0.2 + 0.15 * expected
                )

    def test_mean_stays_at_x0_without_drift(self):
        seed()
        X = brownian_like(x0=10)
        self.assertAlmostEqual(X[2.0].sim(Nsim).mean(), 10.0, delta=0.2)

    def test_constant_drift_shifts_the_mean(self):
        seed()
        X = DiffusionProcess(
            drift=lambda x, t: 2.0, diffusion=lambda x, t: 1.0, x0=0, tol=1e-2
        )
        self.assertAlmostEqual(X[3.0].sim(Nsim).mean(), 6.0, delta=0.3)

    def test_zero_diffusion_is_deterministic(self):
        # With no randomness the path is the ODE solution x0 + drift * t.
        seed()
        X = DiffusionProcess(
            drift=lambda x, t: 1.0, diffusion=lambda x, t: 0.0, x0=0, tol=1e-2
        )
        values = {round(float(X.draw()(2.0)), 6) for _ in range(5)}
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values.pop(), 2.0, places=6)


class TestDiffusionProcessKnownModels(unittest.TestCase):

    def test_ornstein_uhlenbeck_long_run_variance(self):
        # A mean-reverting drift settles at variance scale**2 / (2 * rate).
        seed()
        rate, scale = 1.5, 2.0
        X = DiffusionProcess(
            drift=lambda x, t: -rate * x,
            diffusion=lambda x, t: scale,
            x0=0,
            tol=1e-2,
        )
        expected = scale**2 / (2 * rate)
        self.assertAlmostEqual(X[8.0].sim(Nsim).var(), expected, delta=0.3)

    def test_ornstein_uhlenbeck_reverts_toward_its_mean(self):
        seed()
        X = DiffusionProcess(
            drift=lambda x, t: -1.0 * (x - 5.0),
            diffusion=lambda x, t: 1.0,
            x0=20.0,
            tol=1e-2,
        )
        self.assertAlmostEqual(X[6.0].sim(Nsim).mean(), 5.0, delta=0.3)

    def test_geometric_brownian_motion_stays_positive(self):
        # Both coefficients vanish at 0, so a path started above 0 cannot
        # cross it -- the standard reason GBM models prices.
        seed()
        rate, vol = 0.05, 0.3
        X = DiffusionProcess(
            drift=lambda x, t: rate * x,
            diffusion=lambda x, t: vol * x,
            x0=100,
            tol=1e-2,
        )
        sims = X[2.0].sim(200)
        self.assertTrue(all(value > 0 for value in sims))

    def test_geometric_brownian_motion_mean_grows_exponentially(self):
        seed()
        rate, vol, x0, t = 0.05, 0.3, 100.0, 2.0
        X = DiffusionProcess(
            drift=lambda x, t: rate * x,
            diffusion=lambda x, t: vol * x,
            x0=x0,
            tol=1e-2,
        )
        expected = x0 * np.exp(rate * t)
        self.assertAlmostEqual(X[t].sim(Nsim).mean(), expected, delta=0.1 * expected)


class TestDiffusionProcessErrors(unittest.TestCase):

    def test_non_callable_drift_raises_type_error(self):
        with self.assertRaises(TypeError):
            DiffusionProcess(drift=0, diffusion=lambda x, t: 1)

    def test_non_callable_diffusion_raises_type_error(self):
        with self.assertRaises(TypeError):
            DiffusionProcess(drift=lambda x, t: 0, diffusion=1)

    def test_non_numeric_x0_raises_type_error(self):
        with self.assertRaises(TypeError):
            DiffusionProcess(drift=lambda x, t: 0, diffusion=lambda x, t: 1, x0="start")

    def test_zero_tol_raises_value_error(self):
        with self.assertRaises(ValueError):
            DiffusionProcess(drift=lambda x, t: 0, diffusion=lambda x, t: 1, tol=0)

    def test_negative_tol_raises_value_error(self):
        with self.assertRaises(ValueError):
            DiffusionProcess(drift=lambda x, t: 0, diffusion=lambda x, t: 1, tol=-1)

    def test_error_messages_name_the_parameter(self):
        with self.assertRaisesRegex(TypeError, "drift must be callable"):
            DiffusionProcess(drift=0, diffusion=lambda x, t: 1)
        with self.assertRaisesRegex(TypeError, "diffusion must be callable"):
            DiffusionProcess(drift=lambda x, t: 0, diffusion=1)

    def test_probability_space_validates_too(self):
        with self.assertRaises(TypeError):
            DiffusionProcessProbabilitySpace(drift=0, diffusion=lambda x, t: 1)


if __name__ == "__main__":
    unittest.main()
