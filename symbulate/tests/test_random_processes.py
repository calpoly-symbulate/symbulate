"""Tests for symbulate.random_processes.RandomProcess.

Covers construction, the three access forms (``X[t]``, ``X(t)``, and the
underlying ``draw``), assignment via ``X[t] = value`` (scalars, strings,
RVs, and silently-ignored values), index-set validation, custom value
functions, custom index sets, reproducibility, and the marginal
distribution of the process.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching the convention used in
test_distributions.py.
"""

import unittest

import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import distributions
from symbulate.index_sets import (
    Naturals,
    Integers,
    DiscreteTimeSequence,
)
from symbulate.result import TimeFunction

Nsim = 10000


def seed(value=42):
    """Reseed the generator that distribution draws route through."""
    distributions.rng = np.random.default_rng(value)


def bernoulli_process(p=0.5):
    """A RandomProcess of i.i.d. Bernoulli(p) values indexed by the naturals."""
    return RandomProcess(Bernoulli(p=p) ** inf)


class TestRandomProcessInit(unittest.TestCase):

    def test_is_RV_subclass(self):
        self.assertIsInstance(bernoulli_process(), RV)

    def test_default_index_set_is_naturals(self):
        X = bernoulli_process()
        self.assertIsInstance(X.index_set, Naturals)

    def test_rvs_starts_empty(self):
        self.assertEqual(bernoulli_process().rvs, {})

    def test_custom_index_set_is_stored(self):
        index_set = DiscreteTimeSequence(2)
        X = RandomProcess(Bernoulli(p=0.5) ** inf, index_set)
        self.assertIs(X.index_set, index_set)

    def test_has_prob_space(self):
        self.assertTrue(hasattr(bernoulli_process(), "prob_space"))


class TestGetItem(unittest.TestCase):

    def test_returns_RV(self):
        self.assertIsInstance(bernoulli_process()[3], RV)

    def test_unset_time_draws_from_process(self):
        seed()
        X = bernoulli_process()
        self.assertTrue(set(X[3].sim(Nsim)) <= {0, 1})

    def test_unset_time_marginal_mean(self):
        seed()
        X = bernoulli_process(p=0.3)
        self.assertAlmostEqual(X[5].sim(Nsim).mean(), 0.3, delta=0.03)

    def test_different_times_are_independent_draws(self):
        # Two distinct times should not be perfectly identical streams.
        seed()
        X = bernoulli_process()
        s0 = list(X[0].sim(Nsim))
        s1 = list(X[1].sim(Nsim))
        self.assertNotEqual(s0, s1)

    def test_set_rv_is_returned_by_getitem(self):
        X = bernoulli_process()
        rv = RV(Bernoulli(p=1))
        X[1] = rv
        self.assertIs(X[1], rv)

    def test_set_rv_simulates_from_that_rv(self):
        X = bernoulli_process()
        X[1] = RV(Bernoulli(p=1))
        self.assertTrue(all(v == 1 for v in X[1].sim(100)))


class TestSetItemScalar(unittest.TestCase):

    def test_scalar_is_constant(self):
        X = bernoulli_process()
        X[0] = 1
        self.assertTrue(all(v == 1 for v in X[0].sim(100)))

    def test_scalar_stored_in_rvs(self):
        X = bernoulli_process()
        X[2] = 5
        self.assertIn(2, X.rvs)
        self.assertIsInstance(X.rvs[2], RV)

    def test_string_is_treated_as_scalar(self):
        X = bernoulli_process()
        X[0] = "a"
        self.assertIn(0, X.rvs)
        self.assertTrue(all(v == "a" for v in X[0].sim(20)))

    def test_overwrite_replaces_value(self):
        X = bernoulli_process()
        X[0] = 1
        X[0] = 0
        self.assertTrue(all(v == 0 for v in X[0].sim(50)))

    def test_non_scalar_non_rv_raises_typeerror(self):
        # Regression test: a list is neither a scalar nor an RV. This used
        # to be silently ignored (X.rvs stayed empty, no error, no effect)
        # instead of raising -- now it raises a clear TypeError instead.
        X = bernoulli_process()
        with self.assertRaises(TypeError):
            X[0] = [1, 2, 3]
        self.assertEqual(X.rvs, {})

    def test_bad_value_error_names_accepted_types(self):
        X = bernoulli_process()
        with self.assertRaisesRegex(TypeError, "RV.*scalar"):
            X[0] = [1, 2, 3]


class TestSetItemValidation(unittest.TestCase):

    def test_non_integer_time_raises_keyerror(self):
        X = bernoulli_process()  # index set is the naturals
        with self.assertRaises(KeyError):
            X[1.5] = 1

    def test_negative_time_raises_keyerror(self):
        X = bernoulli_process()
        with self.assertRaises(KeyError):
            X[-1] = 1

    def test_valid_natural_time_does_not_raise(self):
        X = bernoulli_process()
        try:
            X[10] = 1
        except KeyError:
            self.fail("Setting a valid natural-number time raised KeyError.")


class TestCall(unittest.TestCase):

    def test_returns_RV(self):
        self.assertIsInstance(bernoulli_process()(3), RV)

    def test_values_from_process(self):
        seed()
        X = bernoulli_process()
        self.assertTrue(set(X(3).sim(Nsim)) <= {0, 1})

    def test_reflects_scalar_assignment(self):
        X = bernoulli_process()
        X[0] = 1
        self.assertTrue(all(v == 1 for v in X(0).sim(50)))

    def test_call_matches_getitem_under_same_seed(self):
        X = bernoulli_process()
        seed(5)
        from_getitem = list(X[3].sim(50))
        seed(5)
        from_call = list(X(3).sim(50))
        self.assertEqual(from_getitem, from_call)


class TestDraw(unittest.TestCase):

    def test_draw_returns_time_function(self):
        self.assertIsInstance(bernoulli_process().draw(), TimeFunction)

    def test_draw_is_callable_at_a_time(self):
        seed()
        path = bernoulli_process().draw()
        self.assertIn(path(0), (0, 1))


class TestIndexSets(unittest.TestCase):

    def test_integers_accepts_integer_time(self):
        X = RandomProcess(Bernoulli(p=0.5) ** inf, Integers())
        X[3] = 1
        self.assertIn(3, X.rvs)

    def test_integers_rejects_non_integer_time(self):
        X = RandomProcess(Bernoulli(p=0.5) ** inf, Integers())
        with self.assertRaises(KeyError):
            X[3.5] = 1

    def test_discrete_time_accepts_grid_point(self):
        X = RandomProcess(Bernoulli(p=0.5) ** inf, DiscreteTimeSequence(2))
        X[0.5] = 1  # 0.5 lands on the 1/2-spaced grid
        self.assertIn(0.5, X.rvs)

    def test_discrete_time_rejects_off_grid_point(self):
        X = RandomProcess(Bernoulli(p=0.5) ** inf, DiscreteTimeSequence(2))
        with self.assertRaises(KeyError):
            X[0.3] = 1


class TestCustomFunc(unittest.TestCase):

    def test_custom_func_is_applied(self):
        seed()
        X = RandomProcess(
            Bernoulli(p=0.5) ** inf, Naturals(), lambda outcome, t: outcome[t] * 10
        )
        self.assertTrue(set(X[2].sim(Nsim)) <= {0, 10})

    def test_custom_func_reflected_through_call(self):
        seed()
        X = RandomProcess(
            Bernoulli(p=0.5) ** inf, Naturals(), lambda outcome, t: outcome[t] + 100
        )
        self.assertTrue(set(X(4).sim(Nsim)) <= {100, 101})


class TestReproducibility(unittest.TestCase):

    def test_same_seed_gives_same_sims(self):
        X = bernoulli_process()
        seed(123)
        first = list(X[3].sim(50))
        seed(123)
        second = list(X[3].sim(50))
        self.assertEqual(first, second)


class TestMarginalDistribution(unittest.TestCase):

    def test_bernoulli_marginal_goodness_of_fit(self):
        seed()
        X = bernoulli_process(p=0.4)
        simulated = X[7].sim(Nsim).tabulate()
        exp_list, obs_list = [], []
        for k in range(2):
            expected = Nsim * stats.binom(n=1, p=0.4).pmf(k)
            exp_list.append(expected)
            obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)


class TestRandomProcessErrors(unittest.TestCase):
    """Error handling for RandomProcess construction."""

    def test_non_callable_func_raises_type_error(self):
        """func must be callable, not a plain value."""
        with self.assertRaises(TypeError):
            RandomProcess(Bernoulli(p=0.5) ** inf, Naturals(), func=5)

    def test_string_func_raises_type_error(self):
        """func must be callable, not a string."""
        with self.assertRaises(TypeError):
            RandomProcess(Bernoulli(p=0.5) ** inf, Naturals(), func="outcome[t]")

    def test_string_index_set_raises_type_error(self):
        """index_set must be an IndexSet instance, not a string."""
        with self.assertRaises(TypeError):
            RandomProcess(Bernoulli(p=0.5) ** inf, index_set="naturals")

    def test_integer_index_set_raises_type_error(self):
        """index_set must be an IndexSet instance, not a bare int."""
        with self.assertRaises(TypeError):
            RandomProcess(Bernoulli(p=0.5) ** inf, index_set=10)

    def test_valid_custom_func_does_not_raise(self):
        """A valid callable func and valid index_set must not raise."""
        X = RandomProcess(
            Bernoulli(p=0.5) ** inf,
            Naturals(),
            lambda outcome, t: 1 - outcome[t],
        )
        self.assertIsInstance(X, RandomProcess)

    def test_valid_discrete_time_index_set_does_not_raise(self):
        """DiscreteTimeSequence is a valid index_set."""
        X = RandomProcess(Bernoulli(p=0.5) ** inf, DiscreteTimeSequence(4))
        self.assertIsInstance(X, RandomProcess)


if __name__ == "__main__":
    unittest.main()
