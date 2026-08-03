"""Tests for symbulate.hitting_times.

Brownian motion is the anchor here, because its hitting times have exact
closed forms to check against:

    P(reach level a by time t) = 2 * Phi(-a / (scale * sqrt(t)))

and, with drift, the inverse-Gaussian form used in
``TestHittingTimeWithDrift``. Those give a real correctness check rather than
a plausibility check.

The two accuracy claims in the module are tested separately, because they are
different claims:

- whether the level was reached, and in which ``step``-sized stretch, is
  exact for Brownian motion at any ``step`` (``TestHittingTimeExactness``);
- where inside that stretch is approximate and leans late, so the reported
  time is only good to about one ``step`` (``TestHittingTimeTiming``).
"""

import unittest
import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import gaussian_process, hitting_times


def seed(value=42):
    gaussian_process.rng = np.random.default_rng(value)
    hitting_times.rng = np.random.default_rng(value + 1)


def reach_probability(level, scale, time):
    """Exact chance a Brownian motion reaches ``level`` by ``time``."""
    return 2 * stats.norm.cdf(-level / (scale * np.sqrt(time)))


def simulate(process, n, **kwargs):
    """Simulate ``n`` hitting times as a plain float array."""
    return np.array(list(hitting_time(process, **kwargs).sim(n)), dtype=float)


class TestHittingTimeAPI(unittest.TestCase):

    def test_process_gives_a_random_variable(self):
        T = hitting_time(BrownianMotion(), level=1)
        self.assertIsInstance(T, RV)

    def test_path_gives_a_number(self):
        seed()
        path = BrownianMotion().draw()
        result = hitting_time(path, level=1, max_time=5.0)
        self.assertIsInstance(result, float)

    def test_random_variable_can_be_simulated(self):
        seed()
        values = simulate(BrownianMotion(), 50, level=1, max_time=4.0, step=1.0)
        self.assertEqual(len(values), 50)

    def test_starting_on_the_level_returns_the_start_time(self):
        # Brownian motion starts at 0, so level 0 is reached immediately.
        seed()
        path = BrownianMotion().draw()
        self.assertEqual(hitting_time(path, level=0), 0.0)

    def test_already_on_the_level_at_start_time_returns_start_time(self):
        seed()
        path = BrownianMotion().draw()
        where_it_is = float(path(2.0))
        self.assertEqual(hitting_time(path, level=where_it_is, start_time=2.0), 2.0)

    def test_start_time_skips_earlier_crossings(self):
        # Searching from later on must report a later time, even though the
        # path has already been to the level before then.
        seed()
        path = BrownianMotion().draw()
        first = hitting_time(path, level=0.3, max_time=8.0, step=0.5)
        self.assertTrue(np.isfinite(first))
        later = hitting_time(
            path, level=0.3, max_time=8.0, step=0.5, start_time=first + 0.5
        )
        self.assertGreater(later, first)

    def test_unreachable_level_returns_infinity(self):
        # A level far away is not reached in a short window.
        seed()
        values = simulate(BrownianMotion(), 30, level=50, max_time=1.0, step=0.5)
        self.assertTrue(np.all(np.isinf(values)))

    def test_reported_time_is_inside_the_search_window(self):
        seed()
        values = simulate(BrownianMotion(), 200, level=0.5, max_time=4.0, step=1.0)
        finite = values[np.isfinite(values)]
        self.assertGreater(len(finite), 0)
        self.assertTrue(np.all(finite > 0))
        self.assertTrue(np.all(finite <= 4.0))


class TestHittingTimeExactness(unittest.TestCase):
    """Whether the level was reached is exact for Brownian motion."""

    def test_reach_probability_matches_closed_form(self):
        level, scale, max_time, n = 1.0, 1.0, 4.0, 2000
        seed()
        values = simulate(
            BrownianMotion(scale=scale),
            n,
            level=level,
            max_time=max_time,
            step=1.0,
        )
        expected = reach_probability(level, scale, max_time)
        observed = np.mean(np.isfinite(values))
        standard_error = np.sqrt(expected * (1 - expected) / n)
        self.assertLess(abs(observed - expected), 3.5 * standard_error)

    def test_scale_is_respected(self):
        level, scale, max_time, n = 1.0, 2.0, 2.0, 2000
        seed()
        values = simulate(
            BrownianMotion(scale=scale),
            n,
            level=level,
            max_time=max_time,
            step=0.5,
        )
        expected = reach_probability(level, scale, max_time)
        observed = np.mean(np.isfinite(values))
        standard_error = np.sqrt(expected * (1 - expected) / n)
        self.assertLess(abs(observed - expected), 3.5 * standard_error)

    def test_step_size_does_not_change_the_answer(self):
        # The crossing formula is exact for Brownian motion, so scanning in
        # one big stride must agree with scanning in small ones. This is the
        # property that distinguishes it from just checking sampled values,
        # which would miss crossings and depend heavily on the step.
        level, max_time, n = 1.0, 4.0, 2000
        expected = reach_probability(level, 1.0, max_time)
        standard_error = np.sqrt(expected * (1 - expected) / n)
        for step in [max_time, 1.0, 0.25]:
            seed()
            values = simulate(
                BrownianMotion(), n, level=level, max_time=max_time, step=step
            )
            observed = np.mean(np.isfinite(values))
            self.assertLess(
                abs(observed - expected),
                3.5 * standard_error,
                msg=f"step={step} gave {observed}, expected {expected}",
            )

    def test_hidden_crossings_are_detected(self):
        # With one huge stride, the endpoint values almost never sit above the
        # level, so nearly every crossing found has to be one that happened
        # and reversed in between. Checking only sampled values would report
        # far fewer.
        level, max_time, n = 1.0, 4.0, 1000
        seed()
        values = simulate(
            BrownianMotion(), n, level=level, max_time=max_time, step=max_time
        )
        found = np.mean(np.isfinite(values))
        ends_above = np.mean(
            [BrownianMotion().draw()(max_time) >= level for _ in range(n)]
        )
        self.assertGreater(found, ends_above + 0.1)
        self.assertAlmostEqual(
            found, reach_probability(level, 1.0, max_time), delta=0.05
        )


class TestHittingTimeTiming(unittest.TestCase):
    """Where inside a stretch is approximate; the stretch itself is not."""

    def test_cdf_is_accurate_at_a_step_boundary(self):
        # A crossing is attributed to the correct stride exactly, so the
        # distribution is accurate at multiples of step even though placement
        # inside a stride is not.
        level, step, n = 1.0, 1.0, 2000
        seed()
        values = simulate(BrownianMotion(), n, level=level, max_time=4.0, step=step)
        for boundary in [1.0, 2.0, 3.0]:
            expected = reach_probability(level, 1.0, boundary)
            standard_error = np.sqrt(expected * (1 - expected) / n)
            observed = np.mean(values <= boundary)
            self.assertLess(
                abs(observed - expected),
                3.5 * standard_error,
                msg=f"at t={boundary}: {observed} vs {expected}",
            )

    def test_a_finer_step_sharpens_the_reported_time(self):
        # Inside a stride the placement leans late, so a coarse step
        # understates how many crossings happened early. A finer step should
        # get closer to the truth.
        level, n, early = 1.0, 2000, 0.3
        expected = reach_probability(level, 1.0, early)

        errors = []
        for step in [1.0, 0.1]:
            seed()
            values = simulate(BrownianMotion(), n, level=level, max_time=2.0, step=step)
            errors.append(abs(np.mean(values <= early) - expected))
        self.assertLess(errors[1], errors[0])


class TestHittingTimeWithDrift(unittest.TestCase):
    """Drift is handled for free: a bridge's law does not depend on it."""

    def test_reach_probability_matches_inverse_gaussian(self):
        drift, level, scale, max_time, n = 0.5, 1.0, 1.0, 3.0, 2000
        seed()
        values = simulate(
            BrownianMotion(drift=drift, scale=scale),
            n,
            level=level,
            max_time=max_time,
            step=0.5,
        )
        root = scale * np.sqrt(max_time)
        expected = stats.norm.cdf((drift * max_time - level) / root) + np.exp(
            2 * drift * level / scale**2
        ) * stats.norm.cdf(-(drift * max_time + level) / root)
        observed = np.mean(np.isfinite(values))
        standard_error = np.sqrt(expected * (1 - expected) / n)
        self.assertLess(abs(observed - expected), 3.5 * standard_error)

    def test_upward_drift_reaches_a_level_sooner(self):
        seed()
        drifting = simulate(
            BrownianMotion(drift=1.0), 400, level=1.0, max_time=4.0, step=0.5
        )
        seed()
        driftless = simulate(BrownianMotion(), 400, level=1.0, max_time=4.0, step=0.5)
        self.assertGreater(
            np.mean(np.isfinite(drifting)), np.mean(np.isfinite(driftless))
        )


class TestHittingTimeDirection(unittest.TestCase):
    """Falling to a level works the same as rising to one."""

    def test_level_below_the_start_is_found(self):
        level, max_time, n = -1.0, 4.0, 2000
        seed()
        values = simulate(BrownianMotion(), n, level=level, max_time=max_time, step=1.0)
        # By symmetry the chance of reaching -1 equals that of reaching +1.
        expected = reach_probability(1.0, 1.0, max_time)
        observed = np.mean(np.isfinite(values))
        standard_error = np.sqrt(expected * (1 - expected) / n)
        self.assertLess(abs(observed - expected), 3.5 * standard_error)

    def test_symmetric_levels_agree(self):
        seed()
        up = simulate(BrownianMotion(), 600, level=1.0, max_time=4.0, step=1.0)
        seed()
        down = simulate(BrownianMotion(), 600, level=-1.0, max_time=4.0, step=1.0)
        self.assertAlmostEqual(
            np.mean(np.isfinite(up)), np.mean(np.isfinite(down)), delta=0.06
        )


class TestHittingTimeMonotonicity(unittest.TestCase):

    def test_a_further_level_takes_longer_to_reach(self):
        seed()
        near = simulate(BrownianMotion(), 500, level=0.5, max_time=4.0, step=1.0)
        seed()
        far = simulate(BrownianMotion(), 500, level=2.0, max_time=4.0, step=1.0)
        self.assertGreater(np.mean(np.isfinite(near)), np.mean(np.isfinite(far)))

    def test_a_longer_window_reaches_the_level_more_often(self):
        seed()
        short = simulate(BrownianMotion(), 500, level=1.0, max_time=1.0, step=0.5)
        seed()
        long = simulate(BrownianMotion(), 500, level=1.0, max_time=8.0, step=0.5)
        self.assertGreater(np.mean(np.isfinite(long)), np.mean(np.isfinite(short)))


class TestHittingTimeOtherGaussianProcesses(unittest.TestCase):
    """Approximate for these, so behavior rather than exact numbers."""

    def test_works_for_ornstein_uhlenbeck(self):
        seed()
        values = simulate(
            OrnsteinUhlenbeck(reversion_rate=1, mean=0, scale=1),
            300,
            level=1.0,
            max_time=5.0,
            step=0.25,
        )
        finite = values[np.isfinite(values)]
        self.assertGreater(len(finite), 0)
        self.assertTrue(np.all(finite <= 5.0))

    def test_mean_reversion_makes_a_far_level_harder_to_reach(self):
        # A stronger pull back toward 0 should reach a high level less often.
        seed()
        loose = simulate(
            OrnsteinUhlenbeck(reversion_rate=0.2),
            300,
            level=1.5,
            max_time=5.0,
            step=0.25,
        )
        seed()
        tight = simulate(
            OrnsteinUhlenbeck(reversion_rate=5.0),
            300,
            level=1.5,
            max_time=5.0,
            step=0.25,
        )
        self.assertGreater(np.mean(np.isfinite(loose)), np.mean(np.isfinite(tight)))

    def test_works_for_a_brownian_bridge(self):
        # The bridge is pinned back to 0 at end_time, so a level it can only
        # touch in the middle is reached sometimes but not always.
        seed()
        values = simulate(
            BrownianBridge(end_time=1.0),
            300,
            level=0.5,
            max_time=1.0,
            step=0.1,
        )
        proportion = np.mean(np.isfinite(values))
        self.assertGreater(proportion, 0.0)
        self.assertLess(proportion, 1.0)

    def test_works_for_a_custom_gaussian_process(self):
        seed()
        X = GaussianProcess(lambda t: 0, lambda s, t: min(s, t))
        values = simulate(X, 200, level=1.0, max_time=4.0, step=1.0)
        self.assertGreater(np.mean(np.isfinite(values)), 0.0)


class TestLocalVarianceRate(unittest.TestCase):
    """The bridge-matching rate that lets one formula serve every process."""

    def test_exact_for_brownian_motion(self):
        for scale in [1.0, 2.5]:
            cov_func = lambda s, t, scale=scale: scale**2 * min(s, t)
            for t0, t1 in [(0.0, 1.0), (3.0, 3.5), (7.0, 20.0)]:
                self.assertAlmostEqual(
                    hitting_times._local_variance_rate(cov_func, t0, t1),
                    scale**2,
                    places=9,
                )

    def test_exact_for_a_brownian_bridge(self):
        cov_func = lambda s, t: min(s, t) - s * t / 1.0
        for t0, t1 in [(0.1, 0.4), (0.5, 0.9)]:
            self.assertAlmostEqual(
                hitting_times._local_variance_rate(cov_func, t0, t1), 1.0, places=9
            )

    def test_converges_for_ornstein_uhlenbeck(self):
        # Not exact, but closer the shorter the stretch.
        long_run_var = 0.5
        cov_func = lambda s, t: long_run_var * (np.exp(-abs(s - t)) - np.exp(-(s + t)))
        errors = [
            abs(hitting_times._local_variance_rate(cov_func, 1.0, 1.0 + gap) - 1.0)
            for gap in [2.0, 0.5, 0.1]
        ]
        self.assertTrue(errors[0] > errors[1] > errors[2])
        self.assertLess(errors[-1], 0.01)

    def test_zero_when_there_is_no_time_between(self):
        cov_func = lambda s, t: min(s, t)
        self.assertEqual(hitting_times._local_variance_rate(cov_func, 1.0, 1.0), 0.0)


class TestCrossingProbability(unittest.TestCase):

    def test_certain_when_an_endpoint_has_reached_the_level(self):
        self.assertEqual(hitting_times._crossing_probability(1.0, 0.0, 1.0, 1.0), 1.0)
        self.assertEqual(hitting_times._crossing_probability(-1.0, 2.0, 1.0, 1.0), 1.0)

    def test_impossible_when_the_process_cannot_move(self):
        self.assertEqual(hitting_times._crossing_probability(1.0, 1.0, 0.0, 1.0), 0.0)
        self.assertEqual(hitting_times._crossing_probability(1.0, 1.0, 1.0, 0.0), 0.0)

    def test_between_zero_and_one_otherwise(self):
        probability = hitting_times._crossing_probability(0.5, 0.5, 1.0, 1.0)
        self.assertGreater(probability, 0.0)
        self.assertLess(probability, 1.0)

    def test_larger_gaps_are_less_likely_to_be_crossed(self):
        near = hitting_times._crossing_probability(0.2, 0.2, 1.0, 1.0)
        far = hitting_times._crossing_probability(2.0, 2.0, 1.0, 1.0)
        self.assertGreater(near, far)

    def test_more_time_makes_a_crossing_more_likely(self):
        brief = hitting_times._crossing_probability(1.0, 1.0, 1.0, 0.1)
        long = hitting_times._crossing_probability(1.0, 1.0, 1.0, 10.0)
        self.assertGreater(long, brief)


class TestHittingTimeGeometricBrownianMotion(unittest.TestCase):
    """Exact too, by moving the question onto the log scale."""

    def test_reach_probability_matches_closed_form(self):
        # A price reaching L is its log reaching log(L / initial_value), and
        # that log is a Brownian motion with drift, so the inverse-Gaussian
        # form applies exactly.
        initial_value, growth_rate, scale = 100.0, 0.05, 0.3
        level, max_time, n = 120.0, 5.0, 2000

        barrier = np.log(level / initial_value)
        drift = growth_rate - scale**2 / 2
        root = scale * np.sqrt(max_time)
        expected = stats.norm.cdf((drift * max_time - barrier) / root) + np.exp(
            2 * drift * barrier / scale**2
        ) * stats.norm.cdf(-(drift * max_time + barrier) / root)

        seed()
        values = simulate(
            GeometricBrownianMotion(
                initial_value=initial_value, growth_rate=growth_rate, scale=scale
            ),
            n,
            level=level,
            max_time=max_time,
            step=1.0,
        )
        observed = np.mean(np.isfinite(values))
        standard_error = np.sqrt(expected * (1 - expected) / n)
        self.assertLess(abs(observed - expected), 3.5 * standard_error)

    def test_step_size_does_not_change_the_answer(self):
        level, max_time, n = 120.0, 5.0, 1500
        results = []
        for step in [max_time, 1.0]:
            seed()
            values = simulate(
                GeometricBrownianMotion(initial_value=100, growth_rate=0.05, scale=0.3),
                n,
                level=level,
                max_time=max_time,
                step=step,
            )
            results.append(np.mean(np.isfinite(values)))
        self.assertAlmostEqual(results[0], results[1], delta=0.05)

    def test_a_higher_target_takes_longer(self):
        seed()
        near = simulate(
            GeometricBrownianMotion(initial_value=100, scale=0.3),
            300,
            level=110,
            max_time=5.0,
            step=1.0,
        )
        seed()
        far = simulate(
            GeometricBrownianMotion(initial_value=100, scale=0.3),
            300,
            level=200,
            max_time=5.0,
            step=1.0,
        )
        self.assertGreater(np.mean(np.isfinite(near)), np.mean(np.isfinite(far)))

    def test_falling_to_a_lower_price_works(self):
        seed()
        values = simulate(
            GeometricBrownianMotion(initial_value=100, scale=0.3),
            300,
            level=80,
            max_time=5.0,
            step=1.0,
        )
        self.assertGreater(np.mean(np.isfinite(values)), 0.0)

    def test_non_positive_level_raises_value_error(self):
        path = GeometricBrownianMotion(initial_value=100).draw()
        for level in [0, -10]:
            with self.assertRaises(ValueError):
                hitting_time(path, level=level)

    def test_non_positive_level_message_explains_why(self):
        path = GeometricBrownianMotion(initial_value=100).draw()
        with self.assertRaisesRegex(ValueError, "never reaches 0"):
            hitting_time(path, level=0)


class TestHittingTimeUnsupportedProcesses(unittest.TestCase):
    """Tiers A and C are not built, and the error says so."""

    def test_diffusion_process_path_raises_not_implemented(self):
        path = DiffusionProcess(drift=lambda x, t: 0, diffusion=lambda x, t: 1).draw()
        with self.assertRaises(NotImplementedError):
            hitting_time(path, level=1.0)

    def test_random_walk_path_raises_not_implemented(self):
        path = RandomWalk(p=0.5).draw()
        with self.assertRaises(NotImplementedError):
            hitting_time(path, level=1.0)

    def test_error_message_names_what_is_supported(self):
        path = RandomWalk(p=0.5).draw()
        with self.assertRaisesRegex(NotImplementedError, "BrownianMotion"):
            hitting_time(path, level=1.0)


class TestHittingTimeErrors(unittest.TestCase):

    def test_non_numeric_level_raises_type_error(self):
        with self.assertRaises(TypeError):
            hitting_time(BrownianMotion(), level="high")

    def test_non_numeric_step_raises_type_error(self):
        with self.assertRaises(TypeError):
            hitting_time(BrownianMotion(), level=1, step="small")

    def test_max_time_not_after_start_time_raises_value_error(self):
        with self.assertRaises(ValueError):
            hitting_time(BrownianMotion(), level=1, max_time=2.0, start_time=2.0)

    def test_negative_start_time_raises_value_error(self):
        with self.assertRaises(ValueError):
            hitting_time(BrownianMotion(), level=1, start_time=-1.0)

    def test_zero_step_raises_value_error(self):
        with self.assertRaises(ValueError):
            hitting_time(BrownianMotion(), level=1, step=0)

    def test_negative_tol_raises_value_error(self):
        with self.assertRaises(ValueError):
            hitting_time(BrownianMotion(), level=1, tol=-1)

    def test_errors_are_raised_before_any_simulating(self):
        # Validation happens up front, so a bad argument fails immediately
        # rather than once a path is drawn.
        with self.assertRaises(ValueError):
            hitting_time(BrownianMotion(), level=1, max_time=0.0)


if __name__ == "__main__":
    unittest.main()
