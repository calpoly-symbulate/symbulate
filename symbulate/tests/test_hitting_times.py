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

The jump and discrete-time processes (Tier A) are exact, so they are tested
against exact answers rather than against tolerances: the k-th arrival time of
a Poisson process *is* the time its count reaches k, on every single path, and
the mean hitting times of an asymmetric random walk, a two-state Markov chain,
and a two-state continuous-time Markov chain all have closed forms
(``1 / (2p - 1)``, ``1 / a``, ``1 / q``). Their tests also pin the two things
that make them different from the Gaussian case: reaching a level means
reaching it *or passing it*, and neither ``step`` nor ``tol`` changes the
answer.
"""

import unittest
from unittest import mock

import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import distributions, gaussian_process, hitting_times, markov_chains


def seed(value=42):
    gaussian_process.rng = np.random.default_rng(value)
    hitting_times.rng = np.random.default_rng(value + 1)
    distributions.rng = np.random.default_rng(value + 2)
    markov_chains.rng = np.random.default_rng(value + 3)


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
    """Tier C is not built, and the error says so."""

    def test_diffusion_process_path_raises_not_implemented(self):
        path = DiffusionProcess(drift=lambda x, t: 0, diffusion=lambda x, t: 1).draw()
        with self.assertRaises(NotImplementedError):
            hitting_time(path, level=1.0)

    def test_cir_path_raises_not_implemented(self):
        path = CIR().draw()
        with self.assertRaises(NotImplementedError):
            hitting_time(path, level=2.0)

    def test_error_message_names_what_is_supported(self):
        path = CIR().draw()
        with self.assertRaisesRegex(NotImplementedError, "BrownianMotion"):
            hitting_time(path, level=2.0)

    def test_error_message_mentions_the_diffusion_gap(self):
        path = CIR().draw()
        with self.assertRaisesRegex(NotImplementedError, "not built yet"):
            hitting_time(path, level=2.0)

    def test_multi_compartment_path_raises_not_implemented(self):
        # An epidemic path is several counts at once, so there is no single
        # level for it to reach.
        seed()
        path = SIR(population=100, infection_rate=2, recovery_rate=1).draw()
        with self.assertRaises(NotImplementedError):
            hitting_time(path, level=50)


# --- Tier A: jump and discrete-time processes ----------------------------


class TestHittingTimeJumpProcessesAreExact(unittest.TestCase):
    """A pure-jump path can only reach a level at a jump, so this is exact."""

    def test_poisson_count_reaches_k_at_the_kth_arrival(self):
        # Not a tolerance check: the time the count reaches k is the k-th
        # arrival time, exactly, on every path.
        seed()
        path = PoissonProcess(rate=2).draw()
        for k in [1, 2, 5]:
            expected = float(path.get_arrival_times()[k - 1])
            self.assertEqual(hitting_time(path, level=k, max_time=100), expected)

    def test_renewal_count_reaches_k_at_the_kth_arrival(self):
        seed()
        path = RenewalProcess(Gamma(shape=2, rate=2)).draw()
        for k in [1, 3]:
            expected = float(path.get_arrival_times()[k - 1])
            self.assertEqual(hitting_time(path, level=k, max_time=100), expected)

    def test_poisson_mean_matches_the_gamma_closed_form(self):
        # The k-th arrival of a rate-r Poisson process is Gamma(k, r), so the
        # mean hitting time of level k is exactly k / r.
        seed()
        values = simulate(PoissonProcess(rate=2), 2000, level=4, max_time=1000)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values.mean(), 4 / 2, delta=0.06)

    def test_level_zero_is_reached_immediately(self):
        seed()
        path = PoissonProcess(rate=1).draw()
        self.assertEqual(hitting_time(path, level=0, max_time=10), 0.0)

    def test_unreached_level_returns_infinity(self):
        seed()
        values = simulate(PoissonProcess(rate=1), 50, level=100, max_time=5)
        self.assertTrue(np.all(np.isinf(values)))

    def test_a_level_between_two_counts_is_reached_by_passing_it(self):
        # A count goes 3, 4 and is never at 3.5, so reaching 3.5 means the
        # moment it got to 4.
        seed()
        path = PoissonProcess(rate=1).draw()
        self.assertEqual(
            hitting_time(path, level=3.5, max_time=100),
            hitting_time(path, level=4, max_time=100),
        )

    def test_continuous_time_markov_chain_mean_matches_the_exponential(self):
        # Starting in state 0 of a two-state chain, the time until state 1 is
        # Exponential(rate=q01), so its mean is 1 / q01.
        seed()
        chain = ContinuousTimeMarkovChain([[-2, 2], [1, -1]], [1.0, 0.0])
        values = simulate(chain, 2000, level=1, max_time=1000)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values.mean(), 1 / 2, delta=0.03)

    def test_continuous_time_markov_chain_hits_at_a_jump(self):
        seed()
        path = ContinuousTimeMarkovChain([[-1, 1], [2, -2]], [1.0, 0.0]).draw()
        reached = hitting_time(path, level=1, max_time=100)
        self.assertEqual(reached, float(path.get_arrival_times()[0]))

    def test_queue_reaches_one_customer_when_the_first_arrives(self):
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=1.2)).draw()
        self.assertEqual(
            hitting_time(path, level=1, max_time=1000),
            float(path.customer_arrival_times[0]),
        )

    def test_queue_length_hitting_time_is_a_jump_time(self):
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=1.5)).draw()
        reached = hitting_time(path, level=4, max_time=500)
        self.assertTrue(np.isfinite(reached))
        jump_times = [float(path.get_arrival_times()[n]) for n in range(400)]
        self.assertIn(reached, jump_times)

    def test_compound_poisson_ruin_time_is_a_jump_time(self):
        # A surplus falling to a level is the ruin question, and it is one use
        # of this same mechanism rather than a separate utility.
        seed()
        surplus = CompoundPoissonProcess(rate=1, jump_dist=Normal(mean=-1, sd=2))
        path = surplus.draw()
        ruin = hitting_time(path, level=-5, max_time=200)
        self.assertTrue(np.isfinite(ruin))
        jump_times = [float(path.get_arrival_times()[n]) for n in range(300)]
        self.assertIn(ruin, jump_times)

    def test_compound_poisson_falls_to_a_negative_level(self):
        seed()
        surplus = CompoundPoissonProcess(rate=2, jump_dist=Normal(mean=-1, sd=1))
        values = simulate(surplus, 500, level=-10, max_time=100)
        self.assertTrue(np.all(np.isfinite(values)))
        # Losing about 2 per unit time, so about 10 is lost by time 5.
        self.assertAlmostEqual(values.mean(), 5.0, delta=0.4)

    def test_time_varying_rate_counts_say_why_they_are_not_supported(self):
        # A non-homogeneous Poisson or Cox count moves in jumps, but knows its
        # jumps on the expected-count scale rather than on the clock, so it
        # gets its own message rather than the generic one.
        seed()
        for process in [
            NonHomogeneousPoissonProcess(rate=lambda t: 2 * t),
            CoxProcess(intensity=Gamma(shape=2, rate=1)),
        ]:
            with self.assertRaises(NotImplementedError) as context:
                hitting_time(process.draw(), level=3, max_time=100)
            message = str(context.exception)
            self.assertIn("expected number of events", message)
            self.assertIn("count itself at a time", message)


class TestHittingTimeDiscreteTimeProcessesAreExact(unittest.TestCase):
    """A discrete-time path has nothing between its steps, so this is exact."""

    def test_a_walk_that_only_rises_reaches_the_level_at_that_step(self):
        seed()
        path = RandomWalk(p=1).draw()
        self.assertEqual(hitting_time(path, level=5, max_time=50), 5.0)

    def test_a_level_between_two_steps_is_reached_by_passing_it(self):
        # A +1 walk goes 4, 5 and is never at 4.5, so reaching 4.5 means step 5.
        seed()
        path = RandomWalk(p=1).draw()
        self.assertEqual(hitting_time(path, level=4.5, max_time=50), 5.0)

    def test_reported_time_is_a_whole_number_of_steps(self):
        seed()
        values = simulate(RandomWalk(p=0.6), 200, level=3, max_time=200)
        finite = values[np.isfinite(values)]
        self.assertGreater(len(finite), 0)
        self.assertTrue(np.all(finite == np.round(finite)))

    def test_mean_matches_the_asymmetric_walk_closed_form(self):
        # For a +-1 walk with p > 1/2, E[T_1] = 1 / (2p - 1).
        seed()
        values = simulate(RandomWalk(p=0.7), 4000, level=1, max_time=2000)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values.mean(), 1 / (2 * 0.7 - 1), delta=0.1)

    def test_chance_of_ever_reaching_matches_the_closed_form(self):
        # A downward-drifting walk reaches +1 with probability p / (1 - p).
        # The crossings that happen, happen early, so a 600-step window is
        # long enough for the truncation to be lost in the noise.
        p, n = 0.3, 1000
        seed()
        values = simulate(RandomWalk(p=p), n, level=1, max_time=600)
        expected = p / (1 - p)
        observed = np.mean(np.isfinite(values))
        standard_error = np.sqrt(expected * (1 - expected) / n)
        self.assertLess(abs(observed - expected), 4 * standard_error)

    def test_walk_falls_to_a_negative_level(self):
        # Drifting downward, the walk reaches -1 with probability 1, and
        # E[T] = 1 / (q - p) exactly as in the rising case.
        seed()
        values = simulate(RandomWalk(p=0.3), 2000, level=-1, max_time=2000)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values.mean(), 1 / (2 * 0.7 - 1), delta=0.25)

    def test_markov_chain_mean_matches_the_geometric(self):
        # Starting in state 0, the first step in state 1 is geometric with
        # success probability 0.25, so its mean is 1 / 0.25 = 4.
        seed()
        chain = MarkovChain([[0.75, 0.25], [0.5, 0.5]], [1.0, 0.0])
        values = simulate(chain, 4000, level=1, max_time=500)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertAlmostEqual(values.mean(), 4.0, delta=0.15)

    def test_moving_average_process_is_read_step_by_step(self):
        seed()
        path = MA(coefs=[0.5], noise_dist=Normal(mean=0, sd=1)).draw()
        reached = hitting_time(path, level=1.5, max_time=200)
        self.assertTrue(np.isfinite(reached))
        self.assertEqual(reached, float(int(reached)))
        self.assertGreaterEqual(float(path[int(reached)]), 1.5)

    def test_max_time_counts_steps(self):
        # A walk that rises by 1 per step cannot reach 20 within 10 steps.
        seed()
        path = RandomWalk(p=1).draw()
        self.assertEqual(hitting_time(path, level=20, max_time=10), float("inf"))
        self.assertEqual(hitting_time(path, level=20, max_time=25), 20.0)

    def test_a_faster_sampling_rate_is_reported_in_time_not_steps(self):
        # A discrete-time process sampled twice per unit time reaches a level
        # at a time, not at a step index.
        walk = RandomProcess(
            Bernoulli(p=1) ** inf,
            DiscreteTimeSequence(fs=2),
            lambda outcome, n: n,
        )
        path = walk.draw()
        self.assertEqual(hitting_time(path, level=4, max_time=10), 2.0)


class TestHittingTimeTierADoesNotUseStepOrTol(unittest.TestCase):
    """`step` and `tol` deal with what a continuous path does in between."""

    def test_step_does_not_change_a_jump_answer(self):
        seed()
        path = PoissonProcess(rate=2).draw()
        answers = {
            hitting_time(path, level=3, max_time=100, step=size)
            for size in [1e-4, 1.0, 100.0]
        }
        self.assertEqual(len(answers), 1)

    def test_tol_does_not_change_a_jump_answer(self):
        seed()
        path = PoissonProcess(rate=2).draw()
        answers = {
            hitting_time(path, level=3, max_time=100, tol=size) for size in [1e-12, 0.5]
        }
        self.assertEqual(len(answers), 1)

    def test_step_does_not_change_a_discrete_time_answer(self):
        seed()
        path = RandomWalk(p=0.6).draw()
        answers = {
            hitting_time(path, level=2, max_time=100, step=size)
            for size in [1e-4, 1.0, 100.0]
        }
        self.assertEqual(len(answers), 1)


class TestHittingTimeTierAStartTime(unittest.TestCase):
    """Searching from a later time, for both Tier A families."""

    def test_already_past_the_level_looks_for_a_fall_back_to_it(self):
        # The direction is taken from where the path is at start_time, so a
        # count already above the level is waiting to come back down to it --
        # which a count never does.
        seed()
        path = PoissonProcess(rate=2).draw()
        first = hitting_time(path, level=3, max_time=100)
        self.assertTrue(np.isfinite(first))
        self.assertEqual(
            hitting_time(path, level=3, max_time=100, start_time=first + 5),
            float("inf"),
        )

    def test_start_time_inside_a_stretch_reports_a_later_crossing(self):
        seed()
        path = PoissonProcess(rate=2).draw()
        first = hitting_time(path, level=5, max_time=100)
        later = hitting_time(path, level=5, max_time=100, start_time=first / 2)
        self.assertEqual(later, first)

    def test_already_on_the_level_at_start_time_returns_start_time(self):
        seed()
        path = PoissonProcess(rate=1).draw()
        reached = hitting_time(path, level=2, max_time=100)
        self.assertEqual(
            hitting_time(path, level=2, max_time=100, start_time=reached + 0.01),
            reached + 0.01,
        )

    def test_start_time_skips_earlier_steps_of_a_walk(self):
        seed()
        path = RandomWalk(p=0.6).draw()
        first = hitting_time(path, level=1, max_time=100)
        self.assertTrue(np.isfinite(first))
        later = hitting_time(path, level=1, max_time=100, start_time=first + 1)
        self.assertGreater(later, first)


class TestHittingTimeTierAErrors(unittest.TestCase):

    def test_named_markov_chain_states_raise_type_error(self):
        seed()
        chain = MarkovChain(
            [[0.5, 0.5], [0.3, 0.7]], [1.0, 0.0], state_labels=["sun", "rain"]
        )
        with self.assertRaises(TypeError) as context:
            hitting_time(chain.draw(), level=1)
        message = str(context.exception)
        self.assertIn("not a number", message)
        self.assertIn("state_labels", message)

    def test_named_continuous_time_markov_chain_states_raise_type_error(self):
        seed()
        chain = ContinuousTimeMarkovChain(
            [[-1, 1], [2, -2]], [1.0, 0.0], state_labels=["calm", "busy"]
        )
        with self.assertRaises(TypeError):
            hitting_time(chain.draw(), level=1)

    def test_too_many_jumps_raises_value_error(self):
        seed()
        path = PoissonProcess(rate=100).draw()
        with mock.patch.object(hitting_times, "MAX_JUMPS", 20):
            with self.assertRaises(ValueError) as context:
                hitting_time(path, level=1000, max_time=100)
        self.assertIn("jumps", str(context.exception))


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
