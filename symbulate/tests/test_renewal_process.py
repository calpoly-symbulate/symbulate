"""Tests for symbulate.renewal_process.

Covers the sample-path result object (counts, monotonicity, states), the
probability space, and the RenewalProcess random process — including the
continuous-time index set, the exact E[N(t)] = sum_k P(S_k <= t) renewal
mean, the Poisson special case (exponential interarrival times), and the
validation of the interarrival distribution.

Also covers CompoundPoissonProcess, which lives in the same module: the
running-total sample path and its states, the E[X(t)] = rate * t * E[Y] and
Var[X(t)] = rate * t * E[Y^2] moments, the unit-jump special case that
recovers the Poisson count itself, and the validation of rate and jump_dist
(where, unlike an interarrival time, a negative jump is legitimate).

The negative-support tests are regression tests: nothing used to stop a
distribution like Normal from being used as an interarrival time, which
silently broke the nondecreasing-count invariant a renewal process assumes.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_distributions.py.
"""

import unittest

import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import distributions
from symbulate.renewal_process import (
    RenewalProcess,
    RenewalProcessResult,
    RenewalProcessProbabilitySpace,
    CompoundPoissonProcess,
    CompoundPoissonProcessResult,
    CompoundPoissonProcessProbabilitySpace,
)
from symbulate.index_sets import Reals
from symbulate.result import ContinuousTimeFunction, DiscreteValued

Nsim = 10000


class TestRenewalProcessResult(unittest.TestCase):

    def test_is_continuous_time_function(self):
        path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_is_discrete_valued(self):
        path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
        self.assertIsInstance(path, DiscreteValued)

    def test_starts_at_zero(self):
        seed(42)
        path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
        self.assertEqual(path(0), 0)

    def test_counts_are_nonnegative_integers(self):
        seed(42)
        path = RenewalProcess(Uniform(a=0, b=2)).draw()
        for t in [0.5, 1.0, 2.0, 5.0]:
            value = path(t)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_counts_are_nondecreasing(self):
        seed(42)
        path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
        values = [path(t) for t in range(0, 11)]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(earlier, later)

    def test_getitem_matches_call(self):
        seed(42)
        path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
        for t in [0.5, 1.0, 3.5]:
            self.assertEqual(path[t], path(t))

    def test_get_states_counts_up_from_zero(self):
        path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
        states = path.get_states()
        self.assertEqual([states[i] for i in range(5)], [0, 1, 2, 3, 4])

    def test_result_constructed_from_interarrival_times(self):
        # Arrivals at t=1, 3, and 8 (cumulative): N(t) jumps 0 -> 1 -> 2 -> 3.
        path = RenewalProcessResult([1.0, 2.0, 5.0])
        self.assertEqual(path(0.5), 0)
        self.assertEqual(path(1.5), 1)
        self.assertEqual(path(4.0), 2)


class TestRenewalProcessProbabilitySpace(unittest.TestCase):

    def test_interarrival_dist_stored(self):
        dist = Gamma(shape=2, rate=1)
        space = RenewalProcessProbabilitySpace(dist)
        self.assertIs(space.interarrival_dist, dist)

    def test_draw_returns_result(self):
        seed(42)
        path = RenewalProcessProbabilitySpace(Gamma(shape=2, rate=1)).draw()
        self.assertIsInstance(path, RenewalProcessResult)

    def test_draw_path_starts_at_zero(self):
        seed(42)
        path = RenewalProcessProbabilitySpace(Gamma(shape=2, rate=1)).draw()
        self.assertEqual(path(0), 0)

    def test_successive_draws_are_independent(self):
        # The interarrival-time space is built once and reused, so each draw
        # must still give a fresh path rather than repeating the first one.
        seed(42)
        space = RenewalProcessProbabilitySpace(Exponential(rate=1))
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first(t) for t in range(1, 20)], [second(t) for t in range(1, 20)]
        )


class TestRenewalProcess(unittest.TestCase):

    def test_interarrival_dist_stored(self):
        dist = Gamma(shape=2, rate=1)
        self.assertIs(RenewalProcess(dist).interarrival_dist, dist)

    def test_accepts_keyword_argument(self):
        dist = Gamma(shape=2, rate=1)
        self.assertIs(RenewalProcess(interarrival_dist=dist).interarrival_dist, dist)

    def test_is_random_process_and_rv(self):
        N = RenewalProcess(Gamma(shape=2, rate=1))
        self.assertIsInstance(N, RandomProcess)
        self.assertIsInstance(N, RV)

    def test_index_set_is_reals(self):
        # A renewal process is continuous-time, so its index set must be the
        # reals, not the natural-number RandomProcess default.
        self.assertIsInstance(RenewalProcess(Exponential(rate=1)).index_set, Reals)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(RenewalProcess(Gamma(shape=2, rate=1))[2], RV)

    def test_call_returns_rv(self):
        self.assertIsInstance(RenewalProcess(Gamma(shape=2, rate=1))(2.5), RV)

    def test_setitem_at_continuous_time(self):
        N = RenewalProcess(Gamma(shape=2, rate=1))
        try:
            N[1.5] = 5
        except KeyError:
            self.fail("Setting a value at continuous time 1.5 raised KeyError.")
        self.assertIn(1.5, N.rvs)

    def test_reproducible_under_same_seed(self):
        N = RenewalProcess(Gamma(shape=2, rate=1))
        seed(123)
        first = list(N(3.0).sim(50))
        seed(123)
        second = list(N(3.0).sim(50))
        self.assertEqual(first, second)


class TestMarginalDistribution(unittest.TestCase):

    def test_exponential_interarrivals_give_poisson_counts(self):
        # Exponential interarrival times make a renewal process a Poisson
        # process, so N(3) is Poisson(rate * t) = Poisson(3).
        seed(42)
        N = RenewalProcess(Exponential(rate=1))
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

    def test_exponential_interarrival_mean_matches_rate_times_t(self):
        seed(42)
        N = RenewalProcess(Exponential(rate=2))
        # E[N(4)] = rate * t = 8
        self.assertAlmostEqual(N(4).sim(Nsim).mean(), 8.0, delta=0.3)

    def test_gamma_interarrival_mean_matches_renewal_function(self):
        # For any renewal process, E[N(t)] = sum_k P(S_k <= t), where S_k is
        # the sum of the first k interarrival times. Gamma(shape=2, rate=1)
        # interarrivals make S_k a Gamma(shape=2k, rate=1).
        seed(42)
        t = 10
        expected = sum(stats.gamma(a=2 * k, scale=1).cdf(t) for k in range(1, 60))
        N = RenewalProcess(Gamma(shape=2, rate=1))
        self.assertAlmostEqual(N(t).sim(Nsim).mean(), expected, delta=0.1)

    def test_uniform_interarrival_mean_matches_renewal_function(self):
        # Uniform(a=0, b=2) interarrivals: S_k is an Irwin-Hall sum, so use
        # the elementary renewal bound instead -- E[N(t)] ~ t / E[X] = t.
        seed(42)
        N = RenewalProcess(Uniform(a=0, b=2))
        self.assertAlmostEqual(N(50).sim(1000).mean(), 50, delta=1.5)


class TestRenewalProcessValidation(unittest.TestCase):
    """Validation of interarrival_dist, for both the space and the process."""

    def test_number_raises_type_error(self):
        """A rate is not a distribution."""
        with self.assertRaises(TypeError):
            RenewalProcess(2)

    def test_number_error_suggests_poisson_process(self):
        """The message points a student who passed a rate to PoissonProcess."""
        with self.assertRaises(TypeError) as context:
            RenewalProcess(2)
        self.assertIn("PoissonProcess", str(context.exception))

    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            RenewalProcess("Gamma")

    def test_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            RenewalProcess(None)

    def test_uninstantiated_distribution_class_raises_type_error(self):
        """Passing the class Gamma rather than Gamma(shape=2, rate=1)."""
        with self.assertRaises(TypeError):
            RenewalProcess(Gamma)

    def test_multivariate_distribution_raises_type_error(self):
        """A vector per draw is not a single waiting time."""
        with self.assertRaises(TypeError):
            RenewalProcess(BivariateNormal(mean1=1, mean2=1))

    def test_normal_raises_value_error(self):
        """Regression: Normal can produce negative interarrival times."""
        with self.assertRaises(ValueError):
            RenewalProcess(Normal(mean=5, sd=1))

    def test_negative_support_error_names_the_distribution(self):
        """The message says which distribution was rejected and what to use."""
        with self.assertRaises(ValueError) as context:
            RenewalProcess(Normal(mean=5, sd=1))
        message = str(context.exception)
        self.assertIn("Normal", message)
        self.assertIn("Exponential", message)

    def test_partly_negative_uniform_raises_value_error(self):
        """Regression: Uniform(a=-1, b=3) is negative on part of its range."""
        with self.assertRaises(ValueError):
            RenewalProcess(Uniform(a=-1, b=3))

    def test_negative_discrete_uniform_raises_value_error(self):
        with self.assertRaises(ValueError):
            RenewalProcess(DiscreteUniform(-3, 4))

    def test_cauchy_raises_value_error(self):
        with self.assertRaises(ValueError):
            RenewalProcess(Cauchy())

    def test_probability_space_also_validates(self):
        """The guard lives in the space, so it fires there too."""
        with self.assertRaises(ValueError):
            RenewalProcessProbabilitySpace(Normal(mean=5, sd=1))
        with self.assertRaises(TypeError):
            RenewalProcessProbabilitySpace(2)

    def test_always_zero_distribution_raises_value_error(self):
        """Poisson(0) is a point mass at 0: infinitely many events at time 0."""
        with self.assertRaises(ValueError):
            RenewalProcess(Poisson(0))

    def test_zero_support_distribution_raises_value_error(self):
        """DiscreteUniform(0, 0) can only ever produce 0."""
        with self.assertRaises(ValueError):
            RenewalProcess(DiscreteUniform(0, 0))

    def test_nonnegative_continuous_distributions_accepted(self):
        for dist in [
            Exponential(rate=1),
            Gamma(shape=2, rate=1),
            Uniform(a=0, b=2),
            LogNormal(0, 1),
            Weibull(shape=2, scale=1),
            Beta(2, 3),
        ]:
            with self.subTest(dist=type(dist).__name__):
                seed(42)
                path = RenewalProcess(dist).draw()
                self.assertGreaterEqual(path(1.0), 0)

    def test_nonnegative_discrete_distribution_accepted(self):
        """A discrete interarrival time is unusual but legitimate."""
        seed(42)
        path = RenewalProcess(Poisson(2)).draw()
        self.assertGreaterEqual(path(5.0), 0)

    def test_point_mass_without_support_accepted(self):
        """LogNormal(mu, 0) has no scipy support to read; its quantile does."""
        seed(42)
        # A point mass at exp(0) = 1, so the count at t=3.5 is exactly 3.
        path = RenewalProcess(LogNormal(0, 0)).draw()
        self.assertEqual(path(3.5), 3)


class TestCompoundPoissonProcessResult(unittest.TestCase):

    def test_is_continuous_time_function(self):
        path = CompoundPoissonProcess(1, Exponential(rate=1)).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_is_discrete_valued(self):
        path = CompoundPoissonProcess(1, Exponential(rate=1)).draw()
        self.assertIsInstance(path, DiscreteValued)

    def test_starts_at_zero(self):
        seed(42)
        path = CompoundPoissonProcess(1, Exponential(rate=1)).draw()
        self.assertEqual(path(0), 0)

    def test_result_accumulates_jumps_at_arrival_times(self):
        # Arrivals at t=1, 3, and 8 (cumulative), with jumps 10, 20, 30:
        # the total steps 0 -> 10 -> 30 -> 60.
        path = CompoundPoissonProcessResult([1.0, 2.0, 5.0], [10, 20, 30])
        self.assertEqual(path(0.5), 0)
        self.assertEqual(path(1.5), 10)
        self.assertEqual(path(4.0), 30)

    def test_jump_is_included_at_its_own_arrival_time(self):
        """The path is right-continuous: the jump at time 1 counts at time 1."""
        path = CompoundPoissonProcessResult([1.0, 2.0, 5.0], [10, 20, 30])
        self.assertEqual(path(1.0), 10)

    def test_total_can_decrease_when_jumps_are_negative(self):
        # Unlike a renewal count, a compound total is not monotone -- a
        # negative jump is a legitimate modeling choice, not an error.
        path = CompoundPoissonProcessResult([1.0, 1.0, 1.0], [5, -3, 2])
        self.assertEqual([path(t) for t in [0.5, 1.0, 2.0, 2.5]], [0, 5, 2, 2])

    def test_getitem_matches_call(self):
        seed(42)
        path = CompoundPoissonProcess(1, Exponential(rate=1)).draw()
        for t in [0.5, 1.0, 3.5]:
            self.assertEqual(path[t], path(t))

    def test_states_are_running_totals_starting_at_zero(self):
        path = CompoundPoissonProcessResult([1.0, 2.0, 5.0], [10, 20, 30])
        self.assertEqual([path.states[n] for n in range(4)], [0, 10, 30, 60])

    def test_get_states_returns_the_running_totals(self):
        path = CompoundPoissonProcessResult([1.0, 2.0, 5.0], [10, 20, 30])
        self.assertEqual([path.get_states()[n] for n in range(3)], [0, 10, 30])

    def test_states_match_the_path_between_arrivals(self):
        """The path sits at states[n] during the n-th waiting period."""
        path = CompoundPoissonProcessResult([1.0, 2.0, 5.0], [10, 20, 30])
        self.assertEqual(path(0.5), path.states[0])
        self.assertEqual(path(2.0), path.states[1])
        self.assertEqual(path(4.0), path.states[2])

    def test_get_interarrival_and_arrival_times(self):
        # A Vector rather than a plain list, since arrival times are the
        # cumulative sum of the interarrival times.
        path = CompoundPoissonProcessResult(Vector([1.0, 2.0, 5.0]), [10, 20, 30])
        self.assertEqual(list(path.get_interarrival_times()), [1.0, 2.0, 5.0])
        self.assertEqual(list(path.get_arrival_times()), [1.0, 3.0, 8.0])

    def test_jump_sizes_are_stored(self):
        path = CompoundPoissonProcessResult([1.0, 2.0, 5.0], [10, 20, 30])
        self.assertEqual(list(path.jump_sizes), [10, 20, 30])


class TestCompoundPoissonProcessProbabilitySpace(unittest.TestCase):

    def test_rate_and_jump_dist_stored(self):
        jumps = Exponential(rate=1)
        space = CompoundPoissonProcessProbabilitySpace(2, jumps)
        self.assertEqual(space.rate, 2)
        self.assertIs(space.jump_dist, jumps)

    def test_draw_returns_result(self):
        seed(42)
        space = CompoundPoissonProcessProbabilitySpace(2, Exponential(rate=1))
        self.assertIsInstance(space.draw(), CompoundPoissonProcessResult)

    def test_draw_path_starts_at_zero(self):
        seed(42)
        space = CompoundPoissonProcessProbabilitySpace(2, Exponential(rate=1))
        self.assertEqual(space.draw()(0), 0)

    def test_successive_draws_are_independent(self):
        # Both sequences are built once and reused, so each draw must still
        # give a fresh path rather than repeating the first one.
        seed(42)
        space = CompoundPoissonProcessProbabilitySpace(1, Exponential(rate=1))
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first(t) for t in range(1, 20)], [second(t) for t in range(1, 20)]
        )


class TestCompoundPoissonProcess(unittest.TestCase):

    def test_rate_and_jump_dist_stored(self):
        jumps = Exponential(rate=1)
        X = CompoundPoissonProcess(2, jumps)
        self.assertEqual(X.rate, 2)
        self.assertIs(X.jump_dist, jumps)

    def test_accepts_keyword_arguments(self):
        jumps = Exponential(rate=1)
        X = CompoundPoissonProcess(rate=2, jump_dist=jumps)
        self.assertEqual(X.rate, 2)
        self.assertIs(X.jump_dist, jumps)

    def test_is_random_process_and_rv(self):
        X = CompoundPoissonProcess(2, Exponential(rate=1))
        self.assertIsInstance(X, RandomProcess)
        self.assertIsInstance(X, RV)

    def test_index_set_is_reals(self):
        X = CompoundPoissonProcess(2, Exponential(rate=1))
        self.assertIsInstance(X.index_set, Reals)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(CompoundPoissonProcess(2, Exponential(rate=1))[2], RV)

    def test_call_returns_rv(self):
        self.assertIsInstance(CompoundPoissonProcess(2, Exponential(rate=1))(2.5), RV)

    def test_reproducible_under_same_seed(self):
        X = CompoundPoissonProcess(2, Exponential(rate=1))
        seed(123)
        first = list(X(3.0).sim(50))
        seed(123)
        second = list(X(3.0).sim(50))
        self.assertEqual(first, second)


class TestCompoundPoissonMoments(unittest.TestCase):

    def test_mean_is_rate_times_time_times_mean_jump(self):
        # E[X(t)] = rate * t * E[Y] = 2 * 4 * 1 = 8.
        seed(42)
        X = CompoundPoissonProcess(2, Exponential(rate=1))
        self.assertAlmostEqual(X(4).sim(Nsim).mean(), 8.0, delta=0.3)

    def test_variance_is_rate_times_time_times_second_moment(self):
        # Var[X(t)] = rate * t * E[Y^2]. For Exponential(rate=1) jumps,
        # E[Y^2] = 2, so Var[X(4)] = 2 * 4 * 2 = 16 -- not 8 * Var(Y) = 8,
        # since the number of jumps varies too.
        seed(42)
        X = CompoundPoissonProcess(2, Exponential(rate=1))
        self.assertAlmostEqual(X(4).sim(Nsim).var(), 16.0, delta=1.0)

    def test_gamma_jumps_mean(self):
        # E[Y] = shape / rate = 2, so E[X(5)] = 3 * 5 * 2 = 30.
        seed(42)
        X = CompoundPoissonProcess(3, Gamma(shape=2, rate=1))
        self.assertAlmostEqual(X(5).sim(Nsim).mean(), 30.0, delta=0.5)

    def test_negative_jumps_give_a_mean_of_zero(self):
        # Symmetric jumps: the total drifts nowhere on average.
        seed(42)
        X = CompoundPoissonProcess(2, Normal(mean=0, sd=1))
        self.assertAlmostEqual(X(5).sim(Nsim).mean(), 0.0, delta=0.2)

    def test_unit_jumps_give_poisson_counts(self):
        # A jump of exactly 1 at every event makes the total the count
        # itself, so X(5) is Poisson(rate * t) = Poisson(10).
        seed(42)
        X = CompoundPoissonProcess(2, Uniform(a=1, b=1))
        simulated = X(5).sim(Nsim).tabulate()
        exp_list, obs_list = [], []
        for k in range(0, 30):
            expected = Nsim * stats.poisson(mu=10).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[float(k)] if float(k) in simulated else 0)
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)


class TestCompoundPoissonProcessValidation(unittest.TestCase):
    """Validation of rate and jump_dist, for both the space and the process."""

    def test_string_rate_raises_type_error(self):
        with self.assertRaises(TypeError):
            CompoundPoissonProcess("2", Exponential(rate=1))

    def test_boolean_rate_raises_type_error(self):
        with self.assertRaises(TypeError):
            CompoundPoissonProcess(True, Exponential(rate=1))

    def test_zero_rate_raises_value_error(self):
        with self.assertRaises(ValueError):
            CompoundPoissonProcess(0, Exponential(rate=1))

    def test_negative_rate_raises_value_error(self):
        with self.assertRaises(ValueError):
            CompoundPoissonProcess(-1, Exponential(rate=1))

    def test_negative_rate_error_points_at_jump_dist(self):
        """A student wanting downward jumps is sent to jump_dist, not rate."""
        with self.assertRaises(ValueError) as context:
            CompoundPoissonProcess(-1, Exponential(rate=1))
        self.assertIn("jump_dist", str(context.exception))

    def test_number_jump_dist_raises_type_error(self):
        with self.assertRaises(TypeError):
            CompoundPoissonProcess(2, 5)

    def test_number_jump_dist_error_shows_how_to_fix_a_fixed_jump(self):
        with self.assertRaises(TypeError) as context:
            CompoundPoissonProcess(2, 5)
        self.assertIn("Uniform(a=5, b=5)", str(context.exception))

    def test_string_jump_dist_raises_type_error(self):
        with self.assertRaises(TypeError):
            CompoundPoissonProcess(2, "Exponential")

    def test_none_jump_dist_raises_type_error(self):
        with self.assertRaises(TypeError):
            CompoundPoissonProcess(2, None)

    def test_uninstantiated_distribution_class_raises_type_error(self):
        """Passing the class Exponential rather than Exponential(rate=1)."""
        with self.assertRaises(TypeError):
            CompoundPoissonProcess(2, Exponential)

    def test_multivariate_jump_dist_raises_type_error(self):
        """A vector per event is not a single jump size."""
        with self.assertRaises(TypeError):
            CompoundPoissonProcess(2, BivariateNormal(mean1=1, mean2=1))

    def test_probability_space_also_validates(self):
        """The guard lives in the space, so it fires there too."""
        with self.assertRaises(ValueError):
            CompoundPoissonProcessProbabilitySpace(0, Exponential(rate=1))
        with self.assertRaises(TypeError):
            CompoundPoissonProcessProbabilitySpace(2, 5)

    def test_negative_support_jump_dist_accepted(self):
        """Unlike an interarrival time, a jump may be negative."""
        seed(42)
        path = CompoundPoissonProcess(2, Normal(mean=0, sd=1)).draw()
        # Symmetric jumps over 50 time units: the total goes below 0 at some
        # point, which a renewal count never could.
        self.assertLess(min(path(t) for t in range(1, 51)), 0)

    def test_discrete_jump_dist_accepted(self):
        seed(42)
        path = CompoundPoissonProcess(2, Poisson(3)).draw()
        self.assertGreaterEqual(path(5.0), 0)


if __name__ == "__main__":
    unittest.main()
