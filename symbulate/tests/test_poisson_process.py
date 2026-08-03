"""Tests for symbulate.poisson_process.

Covers the sample-path result object (counts, monotonicity, states),
the probability space, and the PoissonProcess random process — including
a regression test for the continuous-time index set (a Poisson process
must accept non-integer times) and the Poisson(rate * t) marginal.

Also covers NonHomogeneousPoissonProcess: the cumulative rate (the rate
function integrated over time) against integrals known in closed form, the
Poisson(cumulative_rate(t)) marginal, and the validation of a user-supplied
rate function — including a rate that only goes negative partway through, and
one that cannot be integrated at all, both of which must raise rather than
return a plausible-looking number.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_distributions.py.
"""

import unittest

import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import distributions
from symbulate.poisson_process import (
    PoissonProcess,
    PoissonProcessResult,
    PoissonProcessProbabilitySpace,
    NonHomogeneousPoissonProcess,
    NonHomogeneousPoissonProcessResult,
    NonHomogeneousPoissonProcessProbabilitySpace,
)
from symbulate.index_sets import Reals
from symbulate.result import ContinuousTimeFunction, DiscreteValued

Nsim = 10000


def seed(value=42):
    """Reseed the generator that distribution draws route through."""
    distributions.rng = np.random.default_rng(value)


class TestPoissonProcessResult(unittest.TestCase):

    def test_is_continuous_time_function(self):
        path = PoissonProcess(rate=1).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_is_discrete_valued(self):
        path = PoissonProcess(rate=1).draw()
        self.assertIsInstance(path, DiscreteValued)

    def test_starts_at_zero(self):
        seed()
        path = PoissonProcess(rate=5).draw()
        self.assertEqual(path(0), 0)

    def test_counts_are_nonnegative_integers(self):
        seed()
        path = PoissonProcess(rate=2).draw()
        for t in [0.5, 1.0, 2.0, 5.0]:
            value = path(t)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_counts_are_nondecreasing(self):
        seed()
        path = PoissonProcess(rate=2).draw()
        values = [path(t) for t in range(0, 11)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_getitem_matches_call(self):
        seed()
        path = PoissonProcess(rate=2).draw()
        for t in [0.5, 1.0, 3.5]:
            self.assertEqual(path[t], path(t))

    def test_get_states_counts_up_from_zero(self):
        path = PoissonProcess(rate=1).draw()
        states = path.get_states()
        self.assertEqual([states[i] for i in range(5)], [0, 1, 2, 3, 4])

    def test_result_constructed_from_interarrival_times(self):
        # One arrival at t=1, another at t=3 (cumulative): N(t) jumps 0 -> 1 -> 2.
        path = PoissonProcessResult([1.0, 2.0, 5.0])
        self.assertEqual(path(0.5), 0)
        self.assertEqual(path(1.5), 1)
        self.assertEqual(path(4.0), 2)


class TestPoissonProcessProbabilitySpace(unittest.TestCase):

    def test_rate_stored(self):
        self.assertEqual(PoissonProcessProbabilitySpace(rate=3).rate, 3)

    def test_draw_returns_result(self):
        seed()
        path = PoissonProcessProbabilitySpace(rate=2).draw()
        self.assertIsInstance(path, PoissonProcessResult)

    def test_draw_path_starts_at_zero(self):
        seed()
        path = PoissonProcessProbabilitySpace(rate=2).draw()
        self.assertEqual(path(0), 0)


class TestPoissonProcess(unittest.TestCase):

    def test_rate_stored(self):
        self.assertEqual(PoissonProcess(rate=4).rate, 4)

    def test_is_random_process_and_rv(self):
        N = PoissonProcess(rate=1)
        self.assertIsInstance(N, RandomProcess)
        self.assertIsInstance(N, RV)

    def test_index_set_is_reals(self):
        # Regression: a Poisson process is continuous-time, so its index
        # set must be the reals, not the natural-number RandomProcess default.
        self.assertIsInstance(PoissonProcess(rate=1).index_set, Reals)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(PoissonProcess(rate=1)[2], RV)

    def test_call_returns_rv(self):
        self.assertIsInstance(PoissonProcess(rate=1)(2.5), RV)

    def test_setitem_at_integer_time(self):
        N = PoissonProcess(rate=1)
        N[2] = 5
        self.assertIn(2, N.rvs)

    def test_setitem_at_continuous_time(self):
        # Regression: setting a value at a non-integer time must NOT raise
        # (previously failed with KeyError due to a Naturals index set).
        N = PoissonProcess(rate=1)
        try:
            N[1.5] = 5
        except KeyError:
            self.fail("Setting a value at continuous time 1.5 raised KeyError.")
        self.assertIn(1.5, N.rvs)

    def test_setitem_constant_value(self):
        N = PoissonProcess(rate=1)
        N[1.5] = 7
        self.assertTrue(all(v == 7 for v in N[1.5].sim(50)))

    def test_reproducible_under_same_seed(self):
        N = PoissonProcess(rate=2)
        seed(123)
        first = list(N(3.0).sim(50))
        seed(123)
        second = list(N(3.0).sim(50))
        self.assertEqual(first, second)


class TestMarginalDistribution(unittest.TestCase):

    def test_marginal_is_poisson(self):
        # N(t) is Poisson(rate * t); here rate=1, t=3 -> Poisson(3).
        seed()
        N = PoissonProcess(rate=1)
        simulated = N(3).sim(Nsim).tabulate()
        exp_list, obs_list = [], []
        for k in range(0, 15):
            expected = Nsim * stats.poisson(mu=3).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k] if k in simulated else 0)
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_marginal_mean_scales_with_rate(self):
        seed()
        N = PoissonProcess(rate=2)
        # E[N(4)] = rate * t = 8
        self.assertAlmostEqual(N(4).sim(Nsim).mean(), 8.0, delta=0.3)


class TestPoissonProcessErrors(unittest.TestCase):
    """Error handling for PoissonProcessProbabilitySpace and PoissonProcess."""

    def test_string_rate_raises_type_error(self):
        """rate must be a number, not a string."""
        with self.assertRaises(TypeError):
            PoissonProcess(rate="fast")

    def test_none_rate_raises_type_error(self):
        """rate must be a number, not None."""
        with self.assertRaises(TypeError):
            PoissonProcess(rate=None)

    def test_zero_rate_raises_value_error(self):
        """rate=0 would mean no events ever occur — not allowed."""
        with self.assertRaises(ValueError):
            PoissonProcess(rate=0)

    def test_negative_rate_raises_value_error(self):
        """Negative rate is not physically meaningful."""
        with self.assertRaises(ValueError):
            PoissonProcess(rate=-1)

    def test_probability_space_string_rate_raises_type_error(self):
        """PoissonProcessProbabilitySpace also validates rate type."""
        with self.assertRaises(TypeError):
            PoissonProcessProbabilitySpace(rate="high")

    def test_probability_space_zero_rate_raises_value_error(self):
        """PoissonProcessProbabilitySpace also validates rate > 0."""
        with self.assertRaises(ValueError):
            PoissonProcessProbabilitySpace(rate=0)

    def test_valid_integer_rate_does_not_raise(self):
        """A positive integer rate is valid."""
        seed()
        N = PoissonProcess(rate=3)
        path = N.draw()
        self.assertGreaterEqual(path(1.0), 0)

    def test_valid_float_rate_does_not_raise(self):
        """A positive float rate is valid."""
        seed()
        N = PoissonProcess(rate=0.5)
        path = N.draw()
        self.assertGreaterEqual(path(1.0), 0)


def growing_rate(t):
    """A rate that grows steadily: the integral from 0 to t is t squared."""
    return 2 * t


def shift_change_rate(t):
    """A rate that jumps at time 5: the integral from 0 to 10 is 55."""
    return 1 if t < 5 else 10


class TestNonHomogeneousPoissonProcessResult(unittest.TestCase):

    def test_is_continuous_time_function(self):
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_is_discrete_valued(self):
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        self.assertIsInstance(path, DiscreteValued)

    def test_starts_at_zero(self):
        seed()
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        self.assertEqual(path(0), 0)

    def test_counts_are_nonnegative_integers(self):
        seed()
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        for t in [0.5, 1.0, 2.0, 5.0]:
            value = path(t)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_counts_are_nondecreasing(self):
        seed()
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        values = [path(t) for t in range(0, 11)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_getitem_matches_call(self):
        seed()
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        for t in [0.5, 1.0, 3.5]:
            self.assertEqual(path[t], path(t))

    def test_get_states_counts_up_from_zero(self):
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        states = path.get_states()
        self.assertEqual([states[i] for i in range(5)], [0, 1, 2, 3, 4])

    def test_result_constructed_from_rate_one_arrivals(self):
        # Rate-1 arrivals at 1, 3, and 8 on the expected-count scale, read onto
        # the clock through a cumulative rate of t squared: the counts change
        # at the times where t squared passes 1, 3, and 8.
        path = NonHomogeneousPoissonProcessResult(
            [1.0, 2.0, 5.0], lambda t: max(t, 0) ** 2
        )
        self.assertEqual(path(0.5), 0)
        self.assertEqual(path(1.5), 1)
        self.assertEqual(path(2.5), 2)


class TestNonHomogeneousPoissonProcessProbabilitySpace(unittest.TestCase):

    def test_rate_stored_as_given(self):
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        self.assertIs(space.rate, growing_rate)

    def test_draw_returns_result(self):
        seed()
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        self.assertIsInstance(space.draw(), NonHomogeneousPoissonProcessResult)

    def test_draw_path_starts_at_zero(self):
        seed()
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        self.assertEqual(space.draw()(0), 0)

    def test_paths_share_one_cumulative_rate(self):
        # The rate function is integrated once for the whole space, not once
        # per sample path, so every draw must be handed the same object.
        seed()
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        self.assertIs(space.draw().cumulative_rate, space.draw().cumulative_rate)

    def test_successive_draws_are_independent(self):
        seed()
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first(t) for t in range(1, 20)], [second(t) for t in range(1, 20)]
        )


class TestNonHomogeneousPoissonProcess(unittest.TestCase):

    def test_rate_stored_as_given(self):
        self.assertIs(
            NonHomogeneousPoissonProcess(rate=growing_rate).rate, growing_rate
        )

    def test_is_random_process_and_rv(self):
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertIsInstance(N, RandomProcess)
        self.assertIsInstance(N, RV)

    def test_index_set_is_reals(self):
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertIsInstance(N.index_set, Reals)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(NonHomogeneousPoissonProcess(rate=growing_rate)[2], RV)

    def test_call_returns_rv(self):
        self.assertIsInstance(NonHomogeneousPoissonProcess(rate=growing_rate)(2.5), RV)

    def test_setitem_at_continuous_time(self):
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        try:
            N[1.5] = 5
        except KeyError:
            self.fail("Setting a value at continuous time 1.5 raised KeyError.")
        self.assertIn(1.5, N.rvs)

    def test_reproducible_under_same_seed(self):
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        seed(123)
        first = list(N(3.0).sim(50))
        seed(123)
        second = list(N(3.0).sim(50))
        self.assertEqual(first, second)


class TestCumulativeRate(unittest.TestCase):
    """The rate function integrated over time, against known integrals."""

    def test_zero_at_time_zero(self):
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertEqual(N.cumulative_rate(0), 0.0)

    def test_zero_before_time_zero(self):
        # Negative times are before the process starts, matching how
        # PoissonProcess reports a count of 0 there.
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertEqual(N.cumulative_rate(-5), 0.0)

    def test_growing_rate_integral(self):
        # The integral of 2t from 0 to 3 is 9.
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertAlmostEqual(N.cumulative_rate(3), 9.0, places=6)

    def test_constant_number_rate_integral(self):
        # A rate given as a number is a rate that never changes.
        N = NonHomogeneousPoissonProcess(rate=2)
        self.assertAlmostEqual(N.cumulative_rate(4), 8.0, places=6)

    def test_jumpy_rate_integral(self):
        # 1 for five units of time, then 10 for five more: 5 + 50 = 55.
        N = NonHomogeneousPoissonProcess(rate=shift_change_rate)
        self.assertAlmostEqual(N.cumulative_rate(10), 55.0, places=6)

    def test_decaying_rate_integral(self):
        # The integral of exp(-t) from 0 to 5 is 1 - exp(-5).
        N = NonHomogeneousPoissonProcess(rate=lambda t: np.exp(-t))
        self.assertAlmostEqual(N.cumulative_rate(5), 1 - np.exp(-5), places=6)

    def test_repeated_calls_agree(self):
        """Values are cached, so a repeated call must give the same answer."""
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertEqual(N.cumulative_rate(3), N.cumulative_rate(3))

    def test_out_of_order_calls_stay_consistent(self):
        # Each new time is filled in by integrating from the nearest earlier
        # known time, so asking out of order must not corrupt the earlier ones.
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        late = N.cumulative_rate(10)
        self.assertAlmostEqual(N.cumulative_rate(2), 4.0, places=6)
        self.assertAlmostEqual(N.cumulative_rate(5), 25.0, places=6)
        self.assertEqual(N.cumulative_rate(10), late)
        self.assertAlmostEqual(late, 100.0, places=6)

    def test_rapidly_oscillating_rate_is_integrated_by_subdividing(self):
        # A single integration over this range is not accurate enough, so it is
        # retried over smaller pieces rather than refused. The integral of
        # 1 + sin(1000t) from 0 to 100 is 100 + (1 - cos(100000)) / 1000.
        N = NonHomogeneousPoissonProcess(rate=lambda t: 1 + np.sin(1000 * t))
        expected = 100 + (1 - np.cos(100000)) / 1000
        self.assertAlmostEqual(N.cumulative_rate(100), expected, places=3)


class TestNonHomogeneousMarginalDistribution(unittest.TestCase):

    def test_marginal_is_poisson_with_cumulative_rate_mean(self):
        # N(t) is Poisson(cumulative_rate(t)); with rate 2t and t=3 that is
        # Poisson(9).
        seed()
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        simulated = N(3).sim(Nsim).tabulate()
        exp_list, obs_list = [], []
        for k in range(0, 30):
            expected = Nsim * stats.poisson(mu=9).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k] if k in simulated else 0)
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_mean_matches_cumulative_rate(self):
        seed()
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertAlmostEqual(N(3).sim(Nsim).mean(), 9.0, delta=0.3)

    def test_constant_rate_matches_ordinary_poisson_process(self):
        # A rate that never changes must reproduce PoissonProcess: both have
        # E[N(4)] = rate * t = 8.
        seed()
        N = NonHomogeneousPoissonProcess(rate=2)
        self.assertAlmostEqual(N(4).sim(Nsim).mean(), 8.0, delta=0.3)

    def test_events_concentrate_where_the_rate_is_high(self):
        # With the rate jumping from 1 to 10 at time 5, the second five units
        # of time should see about ten times as many events as the first five.
        seed()
        N = NonHomogeneousPoissonProcess(rate=shift_change_rate)
        before = N(5).sim(1000).mean()
        seed()
        during = (N(10) - N(5)).sim(1000).mean()
        self.assertAlmostEqual(before, 5.0, delta=0.5)
        self.assertAlmostEqual(during, 50.0, delta=1.5)

    def test_decaying_rate_gives_finitely_many_events(self):
        # With rate exp(-t) the total expected count is 1, so paths stay small
        # forever rather than the count running away.
        seed()
        N = NonHomogeneousPoissonProcess(rate=lambda t: np.exp(-t))
        self.assertAlmostEqual(N(50).sim(1000).mean(), 1.0, delta=0.15)


class TestNonHomogeneousPoissonProcessValidation(unittest.TestCase):
    """Validation of the user-supplied rate, for both the space and process."""

    def test_none_rate_raises_type_error(self):
        with self.assertRaises(TypeError):
            NonHomogeneousPoissonProcess(rate=None)

    def test_string_rate_raises_type_error(self):
        with self.assertRaises(TypeError):
            NonHomogeneousPoissonProcess(rate="fast")

    def test_boolean_rate_raises_type_error(self):
        with self.assertRaises(TypeError):
            NonHomogeneousPoissonProcess(rate=True)

    def test_two_argument_function_raises_type_error(self):
        """A rate function takes one argument: the time."""
        with self.assertRaises(TypeError):
            NonHomogeneousPoissonProcess(rate=lambda t, extra: t)

    def test_non_numeric_return_raises_type_error(self):
        with self.assertRaises(TypeError):
            NonHomogeneousPoissonProcess(rate=lambda t: "fast")

    def test_zero_number_rate_raises_value_error(self):
        with self.assertRaises(ValueError):
            NonHomogeneousPoissonProcess(rate=0)

    def test_negative_number_rate_raises_value_error(self):
        with self.assertRaises(ValueError):
            NonHomogeneousPoissonProcess(rate=-2)

    def test_negative_rate_function_raises_value_error(self):
        with self.assertRaises(ValueError):
            NonHomogeneousPoissonProcess(rate=lambda t: -1)

    def test_infinite_rate_at_time_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            NonHomogeneousPoissonProcess(rate=lambda t: np.inf)

    def test_error_message_suggests_a_rate_function(self):
        with self.assertRaises(TypeError) as context:
            NonHomogeneousPoissonProcess(rate=None)
        self.assertIn("lambda t", str(context.exception))

    def test_probability_space_also_validates(self):
        with self.assertRaises(TypeError):
            NonHomogeneousPoissonProcessProbabilitySpace(None)
        with self.assertRaises(ValueError):
            NonHomogeneousPoissonProcessProbabilitySpace(-2)

    def test_rate_going_negative_later_raises_when_reached(self):
        # A rate that is fine at time 0 and turns negative at time 5 is a real
        # rate function up to time 5, so it is accepted and then refused for
        # the times where it is not — rather than silently counting events
        # against a negative rate.
        N = NonHomogeneousPoissonProcess(rate=lambda t: 5 - t)
        self.assertAlmostEqual(N.cumulative_rate(3), 10.5, places=6)
        with self.assertRaises(ValueError):
            N.cumulative_rate(8)

    def test_non_integrable_rate_raises_instead_of_guessing(self):
        # Numerical integration returns a confidently wrong number for 1/t
        # near 0 (a finite value, with a large error estimate). That must be
        # rejected, not reported as an expected number of events.
        N = NonHomogeneousPoissonProcess(rate=lambda t: 1 / t if t > 0 else 1.0)
        with self.assertRaises(ValueError):
            N.cumulative_rate(1)

    def test_non_integrable_rate_error_mentions_the_rate_function(self):
        N = NonHomogeneousPoissonProcess(rate=lambda t: 1 / t if t > 0 else 1.0)
        with self.assertRaises(ValueError) as context:
            N.cumulative_rate(1)
        self.assertIn("rate function", str(context.exception))

    def test_rate_that_is_zero_for_a_while_is_allowed(self):
        # Events that only start at time 5 are a legitimate model, unlike a
        # constant rate of 0, which would mean no events ever.
        seed()
        N = NonHomogeneousPoissonProcess(rate=lambda t: 0 if t < 5 else 2)
        self.assertEqual(N.cumulative_rate(5), 0.0)
        self.assertEqual(N.draw()(4.0), 0)
        self.assertAlmostEqual(N(10).sim(1000).mean(), 10.0, delta=0.7)


if __name__ == "__main__":
    unittest.main()
