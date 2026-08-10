"""Tests for the CIR process in symbulate.diffusion_process.

CIR has closed forms to check against, so these are correctness checks rather
than plausibility checks. Given the value ``x`` now, the value ``dt`` later has

    mean = x * decay + mean * (1 - decay)
    var  = x * scale**2 / rate * (decay - decay**2)
           + mean * scale**2 / (2 * rate) * (1 - decay)**2

with ``decay = exp(-rate * dt)``, and left alone the process settles into a
``Gamma(2 * rate * mean / scale**2, rate=2 * rate / scale**2)``.

The headline claim -- that this is simulated *exactly*, not by small steps --
is tested by ``TestCIRExactness``: reaching a time in one jump must give the
same distribution as reaching it in fifty, which is precisely what an
Euler-Maruyama scheme would fail.
"""

import unittest
import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import diffusion_process

Nsim = 2000


def transition_mean(x, elapsed, rate, mean, scale):
    """Exact mean of the CIR value ``elapsed`` after being at ``x``."""
    decay = np.exp(-rate * elapsed)
    return x * decay + mean * (1 - decay)


def transition_var(x, elapsed, rate, mean, scale):
    """Exact variance of the CIR value ``elapsed`` after being at ``x``."""
    decay = np.exp(-rate * elapsed)
    return (
        x * scale**2 / rate * (decay - decay**2)
        + mean * scale**2 / (2 * rate) * (1 - decay) ** 2
    )


class TestCIRConstruction(unittest.TestCase):

    def test_is_random_process(self):
        self.assertIsInstance(CIR(), RandomProcess)

    def test_is_rv(self):
        self.assertIsInstance(CIR(), RV)

    def test_probability_space_type(self):
        self.assertIsInstance(CIR().prob_space, CIRProbabilitySpace)

    def test_parameters_stored_on_probability_space(self):
        P = CIRProbabilitySpace(reversion_rate=2.0, mean=0.05, scale=0.1, initial=0.03)
        self.assertEqual(P.reversion_rate, 2.0)
        self.assertEqual(P.mean, 0.05)
        self.assertEqual(P.scale, 0.1)
        self.assertEqual(P.initial, 0.03)

    def test_initial_defaults_to_mean(self):
        P = CIRProbabilitySpace(mean=0.07)
        self.assertEqual(P.initial, 0.07)

    def test_older_initial_value_name_still_accepted(self):
        # CIR shipped with initial_value before the package standardised on
        # initial, so the old spelling has to keep working.
        self.assertEqual(
            CIRProbabilitySpace(mean=0.05, initial_value=0.02).initial, 0.02
        )
        self.assertEqual(float(CIR(mean=0.05, initial_value=0.02).draw()(0)), 0.02)

    def test_giving_both_names_raises_value_error(self):
        with self.assertRaises(ValueError):
            CIR(mean=0.05, initial=0.01, initial_value=0.02)

    def test_initial_is_reported_on_the_process(self):
        self.assertEqual(CIR(mean=0.05, initial=0.02).initial, 0.02)


class TestCIRPaths(unittest.TestCase):

    def test_starts_at_mean_by_default(self):
        seed(42)
        X = CIR(reversion_rate=1, mean=0.05, scale=0.1)
        for _ in range(5):
            self.assertEqual(float(X.draw()(0)), 0.05)

    def test_starts_at_initial_when_given(self):
        seed(42)
        X = CIR(reversion_rate=1, mean=0.05, scale=0.1, initial=0.02)
        for _ in range(5):
            self.assertEqual(float(X.draw()(0)), 0.02)

    def test_can_start_at_zero(self):
        seed(42)
        X = CIR(reversion_rate=1, mean=0.05, scale=0.1, initial=0)
        path = X.draw()
        self.assertEqual(float(path(0)), 0.0)
        # It is pushed straight back up, since the pull at 0 is upward.
        self.assertGreater(float(path(1.0)), 0.0)

    def test_same_time_returns_cached_value(self):
        seed(42)
        path = CIR(mean=0.05, scale=0.1).draw()
        self.assertEqual(path(1.0), path(1.0))

    def test_cached_value_unchanged_after_a_later_time(self):
        seed(42)
        path = CIR(mean=0.05, scale=0.1).draw()
        before = path(1.0)
        path(5.0)
        self.assertEqual(before, path(1.0))

    def test_cached_values_unchanged_after_filling_in_between(self):
        seed(42)
        path = CIR(mean=0.05, scale=0.1).draw()
        at_one, at_two = path(1.0), path(2.0)
        for t in [1.25, 1.5, 1.75]:
            path(t)
        self.assertEqual(path(1.0), at_one)
        self.assertEqual(path(2.0), at_two)

    def test_times_stay_sorted(self):
        seed(42)
        path = CIR(mean=0.05, scale=0.1).draw()
        for t in [3.0, 0.5, 2.0, 1.0]:
            path(t)
        self.assertEqual(path.times, sorted(path.times))

    def test_different_draws_differ(self):
        seed(42)
        X = CIR(mean=0.05, scale=0.1)
        self.assertGreater(len({float(X.draw()(1.0)) for _ in range(10)}), 1)

    def test_negative_time_raises_value_error(self):
        seed(42)
        path = CIR().draw()
        with self.assertRaisesRegex(ValueError, "only defined for t >= 0"):
            path(-1.0)


class TestCIRTransitionLaw(unittest.TestCase):
    """Simulated moments against the exact transition mean and variance."""

    def setUp(self):
        self.rate, self.mean, self.scale = 1.5, 0.05, 0.1
        self.start = 0.02
        self.X = CIR(
            reversion_rate=self.rate,
            mean=self.mean,
            scale=self.scale,
            initial=self.start,
        )

    def test_mean_matches_closed_form(self):
        for t in [0.2, 1.0, 5.0]:
            seed(42)
            expected = transition_mean(self.start, t, self.rate, self.mean, self.scale)
            self.assertAlmostEqual(
                self.X[t].sim(Nsim).mean(), expected, delta=0.06 * expected
            )

    def test_variance_matches_closed_form(self):
        for t in [0.2, 1.0, 5.0]:
            seed(42)
            expected = transition_var(self.start, t, self.rate, self.mean, self.scale)
            self.assertAlmostEqual(
                self.X[t].sim(Nsim).var(), expected, delta=0.25 * expected
            )

    def test_climbs_toward_mean_from_below(self):
        seed(42)
        early = self.X[0.2].sim(Nsim).mean()
        seed(42)
        late = self.X[5.0].sim(Nsim).mean()
        self.assertLess(early, late)
        self.assertAlmostEqual(late, self.mean, delta=0.005)

    def test_falls_toward_mean_from_above(self):
        seed(42)
        X = CIR(
            reversion_rate=self.rate,
            mean=self.mean,
            scale=self.scale,
            initial=0.20,
        )
        self.assertGreater(X[0.2].sim(Nsim).mean(), X[5.0].sim(Nsim).mean())


class TestCIRExactness(unittest.TestCase):
    """Reaching a time in one jump equals reaching it in many."""

    def _value_after_steps(self, n_steps, rate, mean, scale, start, horizon, n):
        seed(42)
        values = []
        for _ in range(n):
            path = CIR(
                reversion_rate=rate,
                mean=mean,
                scale=scale,
                initial=start,
            ).draw()
            value = start
            for i in range(1, n_steps + 1):
                value = path(horizon * i / n_steps)
            values.append(float(value))
        return np.array(values)

    def test_one_jump_matches_many_forward_steps(self):
        # This is what separates an exact transition law from an
        # Euler-Maruyama scheme, whose answer drifts with the step count.
        rate, mean, scale, start, horizon = 1.5, 0.05, 0.1, 0.02, 5.0
        expected_mean = transition_mean(start, horizon, rate, mean, scale)
        expected_var = transition_var(start, horizon, rate, mean, scale)

        for n_steps in [1, 5, 50]:
            values = self._value_after_steps(
                n_steps, rate, mean, scale, start, horizon, 1500
            )
            self.assertAlmostEqual(
                values.mean(),
                expected_mean,
                delta=0.06 * expected_mean,
                msg=f"{n_steps} steps gave mean {values.mean()}",
            )
            self.assertAlmostEqual(
                values.var(),
                expected_var,
                delta=0.3 * expected_var,
                msg=f"{n_steps} steps gave var {values.var()}",
            )


class TestCIRStationaryLaw(unittest.TestCase):

    def test_settles_into_a_gamma(self):
        rate, mean, scale = 1.5, 0.05, 0.1
        seed(42)
        X = CIR(reversion_rate=rate, mean=mean, scale=scale)
        values = np.array(list(X[40.0].sim(Nsim)), dtype=float)
        shape = 2 * rate * mean / scale**2
        gamma_rate = 2 * rate / scale**2
        pvalue = stats.kstest(
            values, stats.gamma(shape, scale=1 / gamma_rate).cdf
        ).pvalue
        self.assertGreater(pvalue, 0.01)

    def test_long_run_mean_is_the_mean_parameter(self):
        seed(42)
        X = CIR(reversion_rate=1.5, mean=0.05, scale=0.1)
        self.assertAlmostEqual(X[40.0].sim(Nsim).mean(), 0.05, delta=0.004)

    def test_long_run_variance_matches_closed_form(self):
        rate, mean, scale = 1.5, 0.05, 0.1
        seed(42)
        X = CIR(reversion_rate=rate, mean=mean, scale=scale)
        expected = mean * scale**2 / (2 * rate)
        self.assertAlmostEqual(X[40.0].sim(Nsim).var(), expected, delta=0.25 * expected)

    def test_a_stronger_pull_holds_it_closer(self):
        seed(42)
        loose = CIR(reversion_rate=0.5, mean=0.05, scale=0.1)
        tight = CIR(reversion_rate=5.0, mean=0.05, scale=0.1)
        self.assertGreater(loose[20.0].sim(Nsim).var(), tight[20.0].sim(Nsim).var())


class TestCIRPositivity(unittest.TestCase):
    """The whole point of CIR: it does not go negative."""

    def test_never_negative(self):
        seed(42)
        X = CIR(reversion_rate=1.0, mean=0.05, scale=0.1)
        values = np.array(list(X[5.0].sim(Nsim)), dtype=float)
        self.assertTrue(np.all(values >= 0))

    def test_strictly_positive_when_feller_condition_holds(self):
        # 2 * rate * mean >= scale ** 2, so the path never reaches 0.
        rate, mean, scale = 1.0, 0.05, 0.1
        self.assertGreaterEqual(2 * rate * mean, scale**2)
        seed(42)
        X = CIR(reversion_rate=rate, mean=mean, scale=scale)
        values = np.array(list(X[5.0].sim(Nsim)), dtype=float)
        self.assertTrue(np.all(values > 0))

    def test_reaches_the_boundary_when_feller_condition_fails(self):
        # A large scale relative to the pull lets the path get down to 0 --
        # but never below it.
        rate, mean, scale = 1.0, 0.02, 0.4
        self.assertLess(2 * rate * mean, scale**2)
        seed(42)
        X = CIR(reversion_rate=rate, mean=mean, scale=scale)
        values = np.array(list(X[3.0].sim(Nsim)), dtype=float)
        self.assertTrue(np.all(values >= 0))
        self.assertLess(values.min(), 1e-6)

    def test_degrees_of_freedom_reported_on_the_path(self):
        rate, mean, scale = 1.0, 0.05, 0.1
        path = CIR(reversion_rate=rate, mean=mean, scale=scale).draw()
        self.assertAlmostEqual(path.degrees_of_freedom, 4 * rate * mean / scale**2)

    def test_values_stay_non_negative_when_filling_in_between(self):
        # The bridge is the one approximate step; it must still respect the
        # process's floor at 0.
        seed(42)
        path = CIR(reversion_rate=1.0, mean=0.02, scale=0.4).draw()
        path(0.0)
        path(2.0)
        for t in [0.25, 0.5, 0.75, 1.0, 1.5]:
            self.assertGreaterEqual(float(path(t)), 0.0)


class TestCIRComparedToOrnsteinUhlenbeck(unittest.TestCase):

    def test_same_mean_reversion(self):
        # Both are pulled toward mean the same way, so the mean of each at a
        # given time follows the same exponential approach.
        rate, mean, start = 1.0, 0.05, 0.02
        seed(42)
        cir = CIR(reversion_rate=rate, mean=mean, scale=0.1, initial=start)
        expected = transition_mean(start, 1.0, rate, mean, 0.1)
        self.assertAlmostEqual(
            cir[1.0].sim(Nsim).mean(), expected, delta=0.06 * expected
        )

    def test_ornstein_uhlenbeck_can_go_negative_but_cir_cannot(self):
        seed(42)
        ou = OrnsteinUhlenbeck(reversion_rate=1.0, mean=0.05, scale=0.2, initial=0.05)
        ou_values = np.array(list(ou[3.0].sim(Nsim)), dtype=float)
        self.assertTrue(np.any(ou_values < 0))

        seed(42)
        cir = CIR(reversion_rate=1.0, mean=0.05, scale=0.2)
        cir_values = np.array(list(cir[3.0].sim(Nsim)), dtype=float)
        self.assertTrue(np.all(cir_values >= 0))


class TestCIRErrors(unittest.TestCase):

    def test_non_numeric_parameters_raise_type_error(self):
        for kwargs in [
            {"reversion_rate": "fast"},
            {"mean": "middle"},
            {"scale": "wide"},
            {"initial": "low"},
        ]:
            with self.assertRaises(TypeError):
                CIR(**kwargs)

    def test_non_positive_reversion_rate_raises_value_error(self):
        for value in [0, -1]:
            with self.assertRaises(ValueError):
                CIR(reversion_rate=value)

    def test_non_positive_mean_raises_value_error(self):
        for value in [0, -0.05]:
            with self.assertRaises(ValueError):
                CIR(mean=value)

    def test_non_positive_scale_raises_value_error(self):
        for value in [0, -0.1]:
            with self.assertRaises(ValueError):
                CIR(scale=value)

    def test_negative_initial_raises_value_error(self):
        with self.assertRaises(ValueError):
            CIR(initial=-0.01)

    def test_initial_error_explains_the_floor(self):
        with self.assertRaisesRegex(ValueError, "never goes below 0"):
            CIR(initial=-0.01)

    def test_probability_space_validates_too(self):
        with self.assertRaises(ValueError):
            CIRProbabilitySpace(scale=0)


if __name__ == "__main__":
    unittest.main()
