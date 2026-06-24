"""Tests for symbulate.poisson_process.

Covers the sample-path result object (counts, monotonicity, states),
the probability space, and the PoissonProcess random process — including
a regression test for the continuous-time index set (a Poisson process
must accept non-integer times) and the Poisson(rate * t) marginal.

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


if __name__ == "__main__":
    unittest.main()
