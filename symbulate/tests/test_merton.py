"""Tests for the Merton jump-diffusion in symbulate.diffusion_process.

The model is geometric Brownian motion plus jumps arriving at Poisson times,
each multiplying the value by ``exp(J)`` with ``J`` normal. It is composed
exactly from those two pieces, so there are closed forms to check against.

Writing ``k1 = exp(jump_mean + jump_sd**2 / 2)`` and
``k2 = exp(2 * jump_mean + 2 * jump_sd**2)``:

    mean = initial * exp(growth_rate * t)
    var  = mean**2 * (exp(scale**2 * t + jump_rate * t * (k2 - 2*k1 + 1)) - 1)

The headline claim is the mean: it does **not** move when the jump settings
change, because the jumps' average multiplicative effect is corrected out of
the drift. ``TestMertonCompensator`` pins that.
"""

import unittest
import numpy as np

from symbulate import *
from symbulate import diffusion_process, gaussian_process, renewal_process

Nsim = 2000


def seed(value=42):
    """Seed every module a Merton path draws from."""
    diffusion_process.rng = np.random.default_rng(value)
    gaussian_process.rng = np.random.default_rng(value + 1)
    renewal_process.rng = np.random.default_rng(value + 2)


def expected_mean(initial, growth_rate, t):
    return initial * np.exp(growth_rate * t)


def expected_var(initial, growth_rate, scale, jump_rate, jump_mean, jump_sd, t):
    k1 = np.exp(jump_mean + jump_sd**2 / 2)
    k2 = np.exp(2 * jump_mean + 2 * jump_sd**2)
    mean = expected_mean(initial, growth_rate, t)
    return mean**2 * (np.exp(scale**2 * t + jump_rate * t * (k2 - 2 * k1 + 1)) - 1)


class TestMertonConstruction(unittest.TestCase):

    def test_is_random_process(self):
        self.assertIsInstance(MertonJumpDiffusion(), RandomProcess)

    def test_is_rv(self):
        self.assertIsInstance(MertonJumpDiffusion(), RV)

    def test_probability_space_type(self):
        self.assertIsInstance(
            MertonJumpDiffusion().prob_space, MertonJumpDiffusionProbabilitySpace
        )

    def test_parameters_stored_on_probability_space(self):
        P = MertonJumpDiffusionProbabilitySpace(
            initial=100,
            growth_rate=0.05,
            scale=0.2,
            jump_rate=2,
            jump_mean=-0.1,
            jump_sd=0.15,
        )
        self.assertEqual(P.initial, 100)
        self.assertEqual(P.growth_rate, 0.05)
        self.assertEqual(P.scale, 0.2)
        self.assertEqual(P.jump_rate, 2)
        self.assertEqual(P.jump_mean, -0.1)
        self.assertEqual(P.jump_sd, 0.15)


class TestMertonPaths(unittest.TestCase):

    def test_starts_at_initial(self):
        seed()
        for initial in [1, 100, 0.5]:
            X = MertonJumpDiffusion(initial=initial)
            for _ in range(5):
                self.assertAlmostEqual(float(X.draw()(0)), float(initial))

    def test_same_time_returns_cached_value(self):
        seed()
        path = MertonJumpDiffusion(initial=100).draw()
        self.assertEqual(path(1.0), path(1.0))

    def test_cached_value_unchanged_after_zooming_in(self):
        seed()
        path = MertonJumpDiffusion(initial=100).draw()
        before = float(path(1.0)), float(path(2.0))
        for t in [1.2, 1.5, 1.8]:
            path(t)
        self.assertEqual((float(path(1.0)), float(path(2.0))), before)

    def test_different_draws_differ(self):
        seed()
        X = MertonJumpDiffusion(initial=100)
        self.assertGreater(len({float(X.draw()(1.0)) for _ in range(10)}), 1)

    def test_stays_positive(self):
        # Jumps multiply, so however large or downward they are the value
        # cannot reach 0.
        seed()
        X = MertonJumpDiffusion(
            initial=100, scale=0.5, jump_rate=5, jump_mean=-0.5, jump_sd=0.5
        )
        self.assertTrue(all(value > 0 for value in X[3.0].sim(500)))

    def test_negative_time_raises_value_error(self):
        seed()
        path = MertonJumpDiffusion().draw()
        with self.assertRaisesRegex(ValueError, "only defined for t >= 0"):
            path(-1.0)

    def test_path_exposes_its_two_pieces(self):
        seed()
        path = MertonJumpDiffusion(initial=100).draw()
        path(2.0)
        self.assertEqual(float(path.brownian_path(0)), 0.0)
        self.assertEqual(float(path.jump_path(0)), 0.0)


class TestMertonCompensator(unittest.TestCase):
    """growth_rate stays the growth rate of the mean, whatever the jumps do."""

    def test_mean_matches_closed_form_for_every_jump_setting(self):
        initial, growth_rate, scale, t = 100.0, 0.05, 0.2, 2.0
        target = expected_mean(initial, growth_rate, t)

        for jump_rate, jump_mean, jump_sd in [
            (1, 0.0, 0.1),
            (1, -0.1, 0.15),
            (5, -0.3, 0.4),
            (2, 0.2, 0.3),
        ]:
            seed()
            X = MertonJumpDiffusion(
                initial=initial,
                growth_rate=growth_rate,
                scale=scale,
                jump_rate=jump_rate,
                jump_mean=jump_mean,
                jump_sd=jump_sd,
            )
            observed = X[t].sim(Nsim).mean()
            self.assertAlmostEqual(
                observed,
                target,
                delta=0.10 * target,
                msg=(
                    f"rate={jump_rate} mean={jump_mean} sd={jump_sd} gave "
                    f"{observed}, expected about {target}"
                ),
            )

    def test_turning_up_the_jumps_widens_the_spread_not_the_mean(self):
        initial, growth_rate, scale, t = 100.0, 0.05, 0.2, 2.0
        settings = dict(initial=initial, growth_rate=growth_rate, scale=scale)
        seed()
        calm = MertonJumpDiffusion(
            jump_rate=1, jump_mean=-0.05, jump_sd=0.1, **settings
        )[t].sim(Nsim)
        seed()
        wild = MertonJumpDiffusion(
            jump_rate=5, jump_mean=-0.3, jump_sd=0.4, **settings
        )[t].sim(Nsim)

        target = expected_mean(initial, growth_rate, t)
        self.assertAlmostEqual(calm.mean(), target, delta=0.10 * target)
        self.assertAlmostEqual(wild.mean(), target, delta=0.10 * target)
        self.assertGreater(wild.var(), calm.var())

    def test_variance_matches_closed_form(self):
        initial, growth_rate, scale, t = 100.0, 0.05, 0.2, 2.0
        jump_rate, jump_mean, jump_sd = 1, -0.1, 0.15
        seed()
        X = MertonJumpDiffusion(
            initial=initial,
            growth_rate=growth_rate,
            scale=scale,
            jump_rate=jump_rate,
            jump_mean=jump_mean,
            jump_sd=jump_sd,
        )
        target = expected_var(
            initial, growth_rate, scale, jump_rate, jump_mean, jump_sd, t
        )
        self.assertAlmostEqual(X[t].sim(Nsim).var(), target, delta=0.35 * target)

    def test_jumps_add_variance_on_top_of_the_wiggle(self):
        initial, growth_rate, scale, t = 100.0, 0.05, 0.2, 2.0
        seed()
        with_jumps = MertonJumpDiffusion(
            initial=initial,
            growth_rate=growth_rate,
            scale=scale,
            jump_rate=3,
            jump_mean=-0.2,
            jump_sd=0.3,
        )[t].sim(Nsim)
        seed()
        without = GeometricBrownianMotion(
            initial=initial, growth_rate=growth_rate, scale=scale
        )[t].sim(Nsim)
        self.assertGreater(with_jumps.var(), without.var())


class TestMertonComparedToGeometricBrownianMotion(unittest.TestCase):

    def test_almost_no_jumps_behaves_like_plain_geometric_brownian_motion(self):
        initial, growth_rate, scale, t = 100.0, 0.05, 0.2, 2.0
        seed()
        barely = MertonJumpDiffusion(
            initial=initial,
            growth_rate=growth_rate,
            scale=scale,
            jump_rate=1e-6,
            jump_mean=0.0,
            jump_sd=0.1,
        )[t].sim(Nsim)
        target_var = expected_var(initial, growth_rate, scale, 0, 0, 0.1, t)
        self.assertAlmostEqual(barely.var(), target_var, delta=0.35 * target_var)

    def test_downward_jumps_make_the_lower_tail_fatter(self):
        # The reason the model exists: plain geometric Brownian motion is
        # criticised for making a crash too unlikely.
        initial, growth_rate, scale, t = 100.0, 0.0, 0.2, 2.0
        floor = 60.0
        seed()
        merton = MertonJumpDiffusion(
            initial=initial,
            growth_rate=growth_rate,
            scale=scale,
            jump_rate=1,
            jump_mean=-0.3,
            jump_sd=0.2,
        )[t].sim(Nsim)
        seed()
        smooth = GeometricBrownianMotion(
            initial=initial, growth_rate=growth_rate, scale=scale
        )[t].sim(Nsim)
        self.assertGreater(merton.count_lt(floor), smooth.count_lt(floor))


class TestMertonErrors(unittest.TestCase):

    def test_non_numeric_parameters_raise_type_error(self):
        for kwargs in [
            {"initial": "high"},
            {"growth_rate": "fast"},
            {"scale": "wide"},
            {"jump_rate": "often"},
            {"jump_mean": "down"},
            {"jump_sd": "varied"},
        ]:
            with self.assertRaises(TypeError):
                MertonJumpDiffusion(**kwargs)

    def test_non_positive_initial_raises_value_error(self):
        for value in [0, -100]:
            with self.assertRaises(ValueError):
                MertonJumpDiffusion(initial=value)

    def test_non_positive_scale_raises_value_error(self):
        for value in [0, -0.2]:
            with self.assertRaises(ValueError):
                MertonJumpDiffusion(scale=value)

    def test_non_positive_jump_rate_raises_value_error(self):
        for value in [0, -1]:
            with self.assertRaises(ValueError):
                MertonJumpDiffusion(jump_rate=value)

    def test_jump_rate_error_points_to_geometric_brownian_motion(self):
        with self.assertRaisesRegex(ValueError, "GeometricBrownianMotion"):
            MertonJumpDiffusion(jump_rate=0)

    def test_non_positive_jump_sd_raises_value_error(self):
        for value in [0, -0.1]:
            with self.assertRaises(ValueError):
                MertonJumpDiffusion(jump_sd=value)

    def test_negative_growth_rate_and_jump_mean_are_valid(self):
        # Only the four scale-like parameters are restricted in sign.
        seed()
        X = MertonJumpDiffusion(
            initial=100, growth_rate=-0.1, jump_mean=-0.5, scale=0.2
        )
        self.assertAlmostEqual(float(X.draw()(0)), 100.0)

    def test_probability_space_validates_too(self):
        with self.assertRaises(ValueError):
            MertonJumpDiffusionProbabilitySpace(jump_sd=0)


if __name__ == "__main__":
    unittest.main()
