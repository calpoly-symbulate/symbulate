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

Also covers CoxProcess, whose intensity is itself random: the two exact
cases (one random rate held for a whole path, giving negative binomial counts,
and a rate that switches between regimes with a Markov chain), the approximate
one (a continuously varying intensity path, read on a grid), and the properties
the grid exists to protect -- that the expected count is one increasing
function of time however it is asked about, so the count of events can never go
backwards.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_distributions.py, plus the
generator of whichever module supplies the intensity (``markov_chains.rng``,
``diffusion_process.rng``, ``gaussian_process.rng``) -- none of which read
NumPy's global generator.
"""

import unittest
from unittest import mock

import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import distributions
from symbulate import diffusion_process, gaussian_process, markov_chains
from symbulate import poisson_process
from symbulate.poisson_process import (
    PoissonProcess,
    PoissonProcessResult,
    PoissonProcessProbabilitySpace,
    NonHomogeneousPoissonProcess,
    NonHomogeneousPoissonProcessResult,
    NonHomogeneousPoissonProcessProbabilitySpace,
    CoxProcess,
    CoxProcessResult,
    CoxProcessProbabilitySpace,
    _PathCumulativeRate,
    _StepCumulativeRate,
    _cumulative_rate_for,
)
from symbulate.index_sets import Reals
from symbulate.result import ContinuousTimeFunction, DiscreteValued, InfiniteVector

Nsim = 10000


class TestPoissonProcessResult(unittest.TestCase):

    def test_is_continuous_time_function(self):
        path = PoissonProcess(rate=1).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_is_discrete_valued(self):
        path = PoissonProcess(rate=1).draw()
        self.assertIsInstance(path, DiscreteValued)

    def test_starts_at_zero(self):
        seed(42)
        path = PoissonProcess(rate=5).draw()
        self.assertEqual(path(0), 0)

    def test_counts_are_nonnegative_integers(self):
        seed(42)
        path = PoissonProcess(rate=2).draw()
        for t in [0.5, 1.0, 2.0, 5.0]:
            value = path(t)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_counts_are_nondecreasing(self):
        seed(42)
        path = PoissonProcess(rate=2).draw()
        values = [path(t) for t in range(0, 11)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_getitem_matches_call(self):
        seed(42)
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
        seed(42)
        path = PoissonProcessProbabilitySpace(rate=2).draw()
        self.assertIsInstance(path, PoissonProcessResult)

    def test_draw_path_starts_at_zero(self):
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
        N = PoissonProcess(rate=3)
        path = N.draw()
        self.assertGreaterEqual(path(1.0), 0)

    def test_valid_float_rate_does_not_raise(self):
        """A positive float rate is valid."""
        seed(42)
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
        seed(42)
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        self.assertEqual(path(0), 0)

    def test_counts_are_nonnegative_integers(self):
        seed(42)
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        for t in [0.5, 1.0, 2.0, 5.0]:
            value = path(t)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_counts_are_nondecreasing(self):
        seed(42)
        path = NonHomogeneousPoissonProcess(rate=growing_rate).draw()
        values = [path(t) for t in range(0, 11)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_getitem_matches_call(self):
        seed(42)
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
        seed(42)
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        self.assertIsInstance(space.draw(), NonHomogeneousPoissonProcessResult)

    def test_draw_path_starts_at_zero(self):
        seed(42)
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        self.assertEqual(space.draw()(0), 0)

    def test_paths_share_one_cumulative_rate(self):
        # The rate function is integrated once for the whole space, not once
        # per sample path, so every draw must be handed the same object.
        seed(42)
        space = NonHomogeneousPoissonProcessProbabilitySpace(growing_rate)
        self.assertIs(space.draw().cumulative_rate, space.draw().cumulative_rate)

    def test_successive_draws_are_independent(self):
        seed(42)
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
        seed(42)
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
        seed(42)
        N = NonHomogeneousPoissonProcess(rate=growing_rate)
        self.assertAlmostEqual(N(3).sim(Nsim).mean(), 9.0, delta=0.3)

    def test_constant_rate_matches_ordinary_poisson_process(self):
        # A rate that never changes must reproduce PoissonProcess: both have
        # E[N(4)] = rate * t = 8.
        seed(42)
        N = NonHomogeneousPoissonProcess(rate=2)
        self.assertAlmostEqual(N(4).sim(Nsim).mean(), 8.0, delta=0.3)

    def test_events_concentrate_where_the_rate_is_high(self):
        # With the rate jumping from 1 to 10 at time 5, the second five units
        # of time should see about ten times as many events as the first five.
        seed(42)
        N = NonHomogeneousPoissonProcess(rate=shift_change_rate)
        before = N(5).sim(1000).mean()
        seed(42)
        during = (N(10) - N(5)).sim(1000).mean()
        self.assertAlmostEqual(before, 5.0, delta=0.5)
        self.assertAlmostEqual(during, 50.0, delta=1.5)

    def test_decaying_rate_gives_finitely_many_events(self):
        # With rate exp(-t) the total expected count is 1, so paths stay small
        # forever rather than the count running away.
        seed(42)
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
        seed(42)
        N = NonHomogeneousPoissonProcess(rate=lambda t: 0 if t < 5 else 2)
        self.assertEqual(N.cumulative_rate(5), 0.0)
        self.assertEqual(N.draw()(4.0), 0)
        self.assertAlmostEqual(N(10).sim(1000).mean(), 10.0, delta=0.7)


# --- Cox process ---------------------------------------------------------


def two_state_chain(rates, initial=(1.0, 0.0)):
    """A two-state continuous-time Markov chain whose states are ``rates``."""
    return ContinuousTimeMarkovChain(
        [[-1, 1], [2, -2]], list(initial), state_labels=list(rates)
    )


class FakeStepPath(DiscreteValued):
    """A piecewise-constant intensity path built by hand, for testing.

    Holds ``states[n]`` for ``holding_times[n]`` units of time, repeating the
    last of each forever, which is all ``_StepCumulativeRate`` reads.
    """

    def __init__(self, states, holding_times):
        self.states = InfiniteVector(lambda n: states[min(n, len(states) - 1)])
        self.interarrival_times = InfiniteVector(
            lambda n: holding_times[min(n, len(holding_times) - 1)]
        )


class TestCoxProcessValidation(unittest.TestCase):

    def test_intensity_stored_as_given(self):
        intensity = Gamma(shape=2, rate=1)
        self.assertIs(CoxProcess(intensity=intensity).intensity, intensity)

    def test_step_defaults_and_is_stored(self):
        self.assertEqual(CoxProcess(Gamma(shape=2, rate=1)).step, 0.01)
        self.assertEqual(CoxProcess(Gamma(shape=2, rate=1), step=0.5).step, 0.5)

    def test_is_random_process_and_rv(self):
        N = CoxProcess(Gamma(shape=2, rate=1))
        self.assertIsInstance(N, RandomProcess)
        self.assertIsInstance(N, RV)

    def test_index_set_is_reals(self):
        self.assertIsInstance(CoxProcess(Gamma(shape=2, rate=1)).index_set, Reals)

    def test_intensity_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError) as context:
            CoxProcess("busy")
        message = str(context.exception)
        self.assertIn("intensity", message)
        self.assertIn("distribution", message)
        self.assertIn("random process", message)

    def test_boolean_intensity_raises_type_error(self):
        with self.assertRaises(TypeError):
            CoxProcess(True)

    def test_intensity_that_can_be_negative_raises_value_error(self):
        # A Normal intensity would mean a negative rate of events.
        with self.assertRaises(ValueError) as context:
            CoxProcess(Normal(mean=5, sd=1))
        message = str(context.exception)
        self.assertIn("negative", message)
        self.assertIn("Normal", message)

    def test_multivariate_intensity_raises_type_error(self):
        with self.assertRaises(TypeError) as context:
            CoxProcess(MultivariateNormal(mean=[1, 1], cov=[[1, 0], [0, 1]]))
        self.assertIn("single rate", str(context.exception))

    def test_nonpositive_step_raises_value_error(self):
        for bad_step in [0, -0.5]:
            with self.assertRaises(ValueError):
                CoxProcess(Gamma(shape=2, rate=1), step=bad_step)

    def test_step_of_wrong_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            CoxProcess(Gamma(shape=2, rate=1), step="small")

    def test_deterministic_number_intensity_must_be_positive(self):
        with self.assertRaises(ValueError):
            CoxProcess(-3)


class TestCoxProcessResult(unittest.TestCase):

    def test_is_continuous_time_function(self):
        seed(42)
        self.assertIsInstance(
            CoxProcess(Gamma(shape=2, rate=1)).draw(), ContinuousTimeFunction
        )

    def test_is_discrete_valued(self):
        seed(42)
        self.assertIsInstance(CoxProcess(Gamma(shape=2, rate=1)).draw(), DiscreteValued)

    def test_is_a_nonhomogeneous_poisson_path(self):
        # Conditional on its intensity, a Cox process *is* a non-homogeneous
        # Poisson process, and the counting is literally that class's.
        seed(42)
        self.assertIsInstance(
            CoxProcess(Gamma(shape=2, rate=1)).draw(),
            NonHomogeneousPoissonProcessResult,
        )

    def test_starts_at_zero(self):
        seed(42)
        self.assertEqual(CoxProcess(Gamma(shape=2, rate=1)).draw()(0), 0)

    def test_counts_are_nonnegative_integers(self):
        seed(42)
        path = CoxProcess(Gamma(shape=2, rate=1)).draw()
        for t in [0.5, 1.0, 2.0, 5.0]:
            value = path(t)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_counts_are_nondecreasing(self):
        seed(42)
        path = CoxProcess(Gamma(shape=2, rate=1)).draw()
        values = [path(t) for t in range(0, 11)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_getitem_matches_call(self):
        seed(42)
        path = CoxProcess(Gamma(shape=2, rate=1)).draw()
        for t in [0.5, 1.0, 3.5]:
            self.assertEqual(path[t], path(t))

    def test_get_states_counts_up_from_zero(self):
        seed(42)
        states = CoxProcess(Gamma(shape=2, rate=1)).draw().get_states()
        self.assertEqual([states[i] for i in range(5)], [0, 1, 2, 3, 4])

    def test_drawn_intensity_is_kept_on_the_path(self):
        # The whole point of a Cox process: each path knows the rate it was
        # generated at, so it can be looked at and plotted.
        seed(42)
        path = CoxProcess(Gamma(shape=2, rate=1)).draw()
        self.assertIsInstance(path.intensity, float)
        self.assertGreater(path.intensity, 0)

    def test_drawn_intensity_is_the_whole_path_for_a_process(self):
        seed(42)
        seed(3)
        path = CoxProcess(two_state_chain([1, 5])).draw()
        self.assertIsInstance(path.intensity, DiscreteValued)

    def test_result_constructed_from_rate_one_arrivals(self):
        # Rate-1 arrivals at 1, 3, and 8 on the expected-count scale, read onto
        # the clock at a constant intensity of 2: the counts change at 0.5,
        # 1.5, and 4.
        path = CoxProcessResult([1.0, 2.0, 5.0], lambda t: 2 * max(t, 0), 2.0)
        self.assertEqual(path(0.25), 0)
        self.assertEqual(path(1.0), 1)
        self.assertEqual(path(2.0), 2)
        self.assertEqual(path.intensity, 2.0)


class TestCoxProcessProbabilitySpace(unittest.TestCase):

    def test_intensity_stored_as_given(self):
        intensity = Gamma(shape=2, rate=1)
        self.assertIs(CoxProcessProbabilitySpace(intensity).intensity, intensity)

    def test_draw_returns_result(self):
        seed(42)
        space = CoxProcessProbabilitySpace(Gamma(shape=2, rate=1))
        self.assertIsInstance(space.draw(), CoxProcessResult)

    def test_each_path_gets_its_own_intensity(self):
        seed(42)
        space = CoxProcessProbabilitySpace(Gamma(shape=2, rate=1))
        self.assertNotEqual(space.draw().intensity, space.draw().intensity)

    def test_each_path_gets_its_own_cumulative_rate(self):
        # A random intensity means a different expected count for every path,
        # unlike the non-homogeneous Poisson process, where one is shared.
        seed(42)
        space = CoxProcessProbabilitySpace(Gamma(shape=2, rate=1))
        self.assertIsNot(space.draw().cumulative_rate, space.draw().cumulative_rate)

    def test_paths_share_one_cumulative_rate_when_intensity_is_not_random(self):
        seed(42)
        space = CoxProcessProbabilitySpace(growing_rate)
        self.assertIs(space.draw().cumulative_rate, space.draw().cumulative_rate)

    def test_successive_draws_are_independent(self):
        seed(42)
        space = CoxProcessProbabilitySpace(Gamma(shape=2, rate=1))
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first(t) for t in range(1, 20)], [second(t) for t in range(1, 20)]
        )


class TestMixedPoissonProcess(unittest.TestCase):
    """A single random rate, held for the whole path -- the exact case.

    With a Gamma(shape=r, rate=beta) intensity, the count by time t is
    negative binomial with r successes and probability beta / (beta + t), so
    the mean, the variance, and the chance of no events at all are all known
    in closed form.
    """

    shape = 3.0
    rate = 2.0
    time = 4.0

    def cox_process(self):
        return CoxProcess(intensity=Gamma(shape=self.shape, rate=self.rate))

    def test_cumulative_rate_is_exactly_rate_times_time(self):
        seed(42)
        path = self.cox_process().draw()
        for t in [0.0, 0.5, 3.0, 10.0]:
            self.assertAlmostEqual(path.cumulative_rate(t), path.intensity * t)

    def test_cumulative_rate_is_zero_before_time_zero(self):
        seed(42)
        self.assertEqual(self.cox_process().draw().cumulative_rate(-1), 0.0)

    def test_mean_matches_negative_binomial(self):
        seed(42)
        counts = self.cox_process()[self.time].sim(Nsim)
        expected = self.shape * self.time / self.rate
        self.assertAlmostEqual(counts.mean(), expected, delta=0.15)

    def test_variance_is_overdispersed_and_matches_theory(self):
        # var = mean + (variance of the random expected count), which is what
        # makes a mixed Poisson process wider than a Poisson process.
        seed(42)
        counts = self.cox_process()[self.time].sim(Nsim)
        mean = self.shape * self.time / self.rate
        expected = mean + self.shape * (self.time / self.rate) ** 2
        self.assertAlmostEqual(counts.var(), expected, delta=1.2)
        self.assertGreater(counts.var(), counts.mean())

    def test_chance_of_no_events_matches_negative_binomial(self):
        seed(42)
        counts = self.cox_process()[self.time].sim(Nsim)
        p = self.rate / (self.rate + self.time)
        expected = stats.nbinom.pmf(0, self.shape, p)
        observed = np.mean([count == 0 for count in counts])
        self.assertAlmostEqual(observed, expected, delta=0.01)

    def test_step_is_irrelevant_when_the_intensity_never_changes(self):
        seed(42)
        coarse = CoxProcess(Gamma(shape=self.shape, rate=self.rate), step=1000).draw()
        coarse_counts = [coarse(t) for t in range(1, 10)]
        seed(42)
        fine = CoxProcess(Gamma(shape=self.shape, rate=self.rate), step=1e-4).draw()
        self.assertEqual(coarse_counts, [fine(t) for t in range(1, 10)])


class TestMarkovModulatedPoissonProcess(unittest.TestCase):
    """A rate that switches between regimes -- also an exact case.

    A continuous-time Markov chain holds one rate at a time, so the expected
    count is a sum of rate times holding time with nothing approximated.
    """

    def test_cumulative_rate_matches_states_times_holding_times(self):
        seed(42)
        seed(11)
        path = CoxProcess(two_state_chain([1, 5])).draw()
        intensity = path.intensity
        # Add the intensity up by hand over the first few stretches it holds.
        total = 0.0
        elapsed = 0.0
        for n in range(6):
            total += intensity.states[n] * intensity.interarrival_times[n]
            elapsed += intensity.interarrival_times[n]
        self.assertAlmostEqual(path.cumulative_rate(elapsed), total, places=10)

    def test_cumulative_rate_interpolates_within_a_stretch(self):
        seed(42)
        seed(5)
        path = CoxProcess(two_state_chain([1, 5])).draw()
        intensity = path.intensity
        first_switch = intensity.interarrival_times[0]
        midpoint = first_switch / 2
        self.assertAlmostEqual(
            path.cumulative_rate(midpoint), intensity.states[0] * midpoint, places=10
        )

    def test_equal_rates_reduce_to_an_ordinary_poisson_process(self):
        # If both regimes have the same rate, the switching is invisible and
        # the count must have a Poisson(rate * t) distribution.
        seed(42)
        seed(4)
        counts = CoxProcess(two_state_chain([3, 3]))[2].sim(Nsim)
        self.assertAlmostEqual(counts.mean(), 6.0, delta=0.15)
        self.assertAlmostEqual(counts.var(), 6.0, delta=0.4)

    def test_switching_rates_are_overdispersed(self):
        seed(42)
        seed(6)
        counts = CoxProcess(two_state_chain([1, 9]))[2].sim(Nsim)
        self.assertGreater(counts.var(), 1.5 * counts.mean())

    def test_step_is_irrelevant_when_the_intensity_holds_one_value_at_a_time(self):
        seed(42)
        seed(9)
        coarse = CoxProcess(two_state_chain([1, 5]), step=1000).draw()
        coarse_counts = [coarse(t) for t in range(1, 8)]
        seed(42)
        seed(9)
        fine = CoxProcess(two_state_chain([1, 5]), step=1e-3).draw()
        self.assertEqual(coarse_counts, [fine(t) for t in range(1, 8)])

    def test_non_numeric_states_raise_type_error(self):
        seed(42)
        seed(2)
        chain = ContinuousTimeMarkovChain(
            [[-1, 1], [2, -2]], [1.0, 0.0], state_labels=["low", "high"]
        )
        with self.assertRaises(TypeError) as context:
            CoxProcess(chain).draw()(2.0)
        message = str(context.exception)
        self.assertIn("not a number", message)
        self.assertIn("state_labels", message)

    def test_a_count_process_can_be_an_intensity(self):
        # A Poisson process is nonnegative and holds one value at a time, so
        # it is a legitimate (rising) intensity, added up exactly.
        seed(42)
        path = CoxProcess(PoissonProcess(rate=1)).draw()
        intensity = path.intensity
        first_arrival = intensity.interarrival_times[0]
        self.assertEqual(path.cumulative_rate(first_arrival), 0.0)


class TestStepCumulativeRate(unittest.TestCase):
    """The exact sum for an intensity that holds one value at a time."""

    def test_matches_a_hand_computed_sum(self):
        # Intensity 2 for 1 unit of time, then 6 for 2 units, then 0 forever.
        cumulative_rate = _StepCumulativeRate(FakeStepPath([2, 6, 0], [1.0, 2.0, inf]))
        self.assertEqual(cumulative_rate(0), 0.0)
        self.assertAlmostEqual(cumulative_rate(0.5), 1.0)
        self.assertAlmostEqual(cumulative_rate(1.0), 2.0)
        self.assertAlmostEqual(cumulative_rate(2.0), 8.0)
        self.assertAlmostEqual(cumulative_rate(3.0), 14.0)

    def test_flattens_once_the_intensity_is_zero_forever(self):
        # An intensity of 0 held for an infinite time is 0 further events, not
        # an undefined number: 0 times infinity must not leak out as a nan.
        cumulative_rate = _StepCumulativeRate(FakeStepPath([4, 0], [2.0, inf]))
        self.assertAlmostEqual(cumulative_rate(2.0), 8.0)
        self.assertAlmostEqual(cumulative_rate(100.0), 8.0)
        self.assertFalse(np.isnan(cumulative_rate(1e6)))

    def test_times_can_be_asked_about_in_any_order(self):
        cumulative_rate = _StepCumulativeRate(FakeStepPath([2, 6], [1.0, 2.0]))
        late = cumulative_rate(2.5)
        self.assertAlmostEqual(cumulative_rate(0.5), 1.0)
        self.assertAlmostEqual(cumulative_rate(2.5), late)

    def test_too_many_jumps_raises_value_error(self):
        cumulative_rate = _StepCumulativeRate(FakeStepPath([1], [1e-9]))
        with mock.patch.object(poisson_process, "_MAX_INTENSITY_JUMPS", 20):
            with self.assertRaises(ValueError) as context:
                cumulative_rate(1.0)
        self.assertIn("jumps", str(context.exception))


class TestPathCumulativeRate(unittest.TestCase):
    """The grid sum for an intensity that changes continuously."""

    def test_is_the_left_hand_sum_on_the_grid(self):
        # Intensity 2t read every 0.25: the left-hand sum up to 1 is
        # 0.25 * (0 + 0.5 + 1 + 1.5) = 0.75.
        cumulative_rate = _PathCumulativeRate(lambda t: 2 * t, 0.25)
        self.assertAlmostEqual(cumulative_rate(1.0), 0.75)

    def test_gets_closer_to_the_true_integral_as_the_step_shrinks(self):
        errors = [
            abs(_PathCumulativeRate(lambda t: 2 * t, step)(3.0) - 9.0)
            for step in [0.1, 0.01, 0.001]
        ]
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertLess(errors[-1], 0.01)

    def test_reads_the_intensity_once_per_grid_point(self):
        times_read = []

        def intensity(t):
            times_read.append(t)
            return 1.0

        cumulative_rate = _PathCumulativeRate(intensity, 0.5)
        cumulative_rate(2.0)
        cumulative_rate(1.0)
        cumulative_rate(2.0)
        self.assertEqual(sorted(times_read), sorted(set(times_read)))

    def test_never_goes_backwards_along_a_rough_path(self):
        # The grid is fixed and anchored at 0 precisely so that this is one
        # well-defined increasing function of t, whatever order it is asked
        # about in -- which is what keeps the count of events from going
        # backwards. A path whose values swing about is the case that would
        # break if the sum used the value at t itself.
        seed(42)
        seed(13)
        cumulative_rate = _PathCumulativeRate(CIR(scale=0.9).draw(), 0.01)
        values = [cumulative_rate(t) for t in np.linspace(0, 5, 401)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_same_answer_whatever_order_times_are_asked_in(self):
        seed(42)
        seed(21)
        path = CIR(scale=0.9).draw()
        forwards = _PathCumulativeRate(path, 0.01)
        backwards = _PathCumulativeRate(path, 0.01)
        near_first = forwards(1.0)
        far_first = backwards(5.0)
        self.assertEqual(backwards(1.0), near_first)
        self.assertEqual(forwards(5.0), far_first)

    def test_too_many_grid_points_raises_value_error_suggesting_a_step(self):
        cumulative_rate = _PathCumulativeRate(lambda t: 1.0, 0.01)
        with self.assertRaises(ValueError) as context:
            cumulative_rate(1e9)
        message = str(context.exception)
        self.assertIn("step", message)


class TestCoxProcessWithVaryingIntensity(unittest.TestCase):
    """A continuously varying intensity path -- the one approximate case."""

    def test_mean_matches_the_average_intensity(self):
        # A CIR intensity started at its long-run mean has expected value that
        # mean at every time, so the expected count by time t is mean * t.
        seed(42)
        seed(17)
        N = CoxProcess(CIR(reversion_rate=1, mean=2, scale=0.5), step=0.05)
        self.assertAlmostEqual(N[2].sim(2000).mean(), 4.0, delta=0.3)

    def test_intensity_path_is_kept_and_can_be_evaluated(self):
        seed(42)
        seed(19)
        path = CoxProcess(CIR(mean=2)).draw()
        self.assertGreater(path.intensity(1.0), 0)

    def test_a_process_that_goes_negative_raises_naming_the_time(self):
        seed(42)
        seed(1)
        with self.assertRaises(ValueError) as context:
            CoxProcess(BrownianMotion())[5.0].draw()
        message = str(context.exception)
        self.assertIn("negative", message)
        self.assertIn("At time", message)

    def test_counts_are_nondecreasing_along_a_rough_intensity(self):
        seed(42)
        seed(23)
        path = CoxProcess(CIR(mean=3, scale=0.9)).draw()
        values = [path(t) for t in np.linspace(0, 5, 101)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)


class TestCoxProcessWithARateThatIsNotRandom(unittest.TestCase):
    """An intensity that is not random reduces to the Poisson family."""

    def test_constant_intensity_matches_an_ordinary_poisson_process(self):
        seed(42)
        counts = CoxProcess(3)[2].sim(Nsim)
        self.assertAlmostEqual(counts.mean(), 6.0, delta=0.15)
        self.assertAlmostEqual(counts.var(), 6.0, delta=0.4)

    def test_rate_function_matches_the_nonhomogeneous_poisson_process(self):
        seed(42)
        N = CoxProcess(growing_rate)
        self.assertAlmostEqual(N.draw().cumulative_rate(3), 9.0, places=6)
        self.assertAlmostEqual(N[3].sim(Nsim).mean(), 9.0, delta=0.2)

    def test_drawn_intensity_is_the_rate_as_given(self):
        seed(42)
        self.assertIs(CoxProcess(growing_rate).draw().intensity, growing_rate)


class TestCumulativeRateDispatch(unittest.TestCase):
    """Which way a drawn intensity is added up over time."""

    def test_a_number_is_added_up_exactly(self):
        cumulative_rate = _cumulative_rate_for(2.5, 0.01)
        self.assertAlmostEqual(cumulative_rate(4), 10.0)
        self.assertEqual(cumulative_rate(-1), 0.0)

    def test_a_negative_number_raises_value_error(self):
        with self.assertRaises(ValueError):
            _cumulative_rate_for(-2.5, 0.01)

    def test_a_piecewise_constant_path_uses_the_exact_sum(self):
        self.assertIsInstance(
            _cumulative_rate_for(FakeStepPath([1], [1.0]), 0.01), _StepCumulativeRate
        )

    def test_any_other_path_uses_the_grid_sum(self):
        self.assertIsInstance(
            _cumulative_rate_for(lambda t: 1.0, 0.01), _PathCumulativeRate
        )

    def test_something_that_is_not_an_intensity_raises_type_error(self):
        with self.assertRaises(TypeError) as context:
            _cumulative_rate_for("busy", 0.01)
        self.assertIn("neither a number nor a sample path", str(context.exception))


if __name__ == "__main__":
    unittest.main()
