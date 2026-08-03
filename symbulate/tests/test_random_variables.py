"""Tests for symbulate.random_variables.RV and RVConditional.

Covers:
- RV construction (with and without a custom func)
- draw() and sim()
- __call__ (applies func and emits UserWarning)
- apply()
- __iter__ (unpacking random vectors; raises on scalar RVs)
- __getitem__ (int, list, slice, and RV index)
- Arithmetic operations (+, -, *, /, **) between two RVs and between RV and scalar
- Comparison operators (returns Event)
- Joint distributions via & and scalar & RV
- Conditional distributions via |
- RVConditional construction and statistical correctness
- check_same_prob_space raises on mismatched spaces
"""

import unittest
import warnings

import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import distributions
from symbulate.random_variables import RVConditional
from symbulate.results import RVResults

Nsim = 10000


def seed(value=42):
    """Reseed the generator that distribution draws route through."""
    distributions.rng = np.random.default_rng(value)


class TestRVInit(unittest.TestCase):

    def test_stores_prob_space(self):
        P = Normal(mean=0, var=1)
        X = RV(P)
        self.assertIs(X.prob_space, P)

    def test_default_func_is_identity(self):
        seed()
        X = RV(Normal(mean=5, sd=1))
        val = X.draw()
        self.assertIsInstance(val, float)

    def test_custom_func_applied_sum(self):
        seed()
        X = RV(BoxModel([0, 1], size=5), sum)
        val = X.draw()
        self.assertIn(val, range(6))

    def test_custom_func_stored(self):
        f = lambda x: x * 2
        X = RV(Normal(mean=0, sd=1), f)
        self.assertIs(X.func, f)

    def test_non_callable_func_raises_type_error(self):
        with self.assertRaises(TypeError):
            RV(Normal(mean=0, sd=1), 5)


class TestRVDraw(unittest.TestCase):

    def test_draw_returns_numeric_from_normal(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        val = X.draw()
        self.assertIsInstance(val, float)

    def test_draw_in_support_bernoulli(self):
        seed()
        X = RV(Bernoulli(p=0.5))
        for _ in range(200):
            self.assertIn(X.draw(), (0, 1))

    def test_draw_constant_p1(self):
        seed()
        X = RV(Bernoulli(p=1))
        for _ in range(50):
            self.assertEqual(X.draw(), 1)

    def test_draw_uses_custom_func(self):
        seed()
        X = RV(BoxModel([1, 2, 3, 4, 5], size=5), sum)
        val = X.draw()
        self.assertIn(val, range(5, 26))


class TestRVSim(unittest.TestCase):

    def test_sim_correct_length(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        sims = X.sim(200)
        self.assertEqual(len(sims), 200)

    def test_sim_returns_rv_results(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        sims = X.sim(100)
        self.assertIsInstance(sims, RVResults)

    def test_sim_values_in_support_bernoulli(self):
        seed()
        X = RV(Bernoulli(p=0.5))
        sims = X.sim(Nsim)
        self.assertTrue(all(v in (0, 1) for v in sims))

    def test_sim_distribution_normal(self):
        seed()
        X = RV(Normal(mean=3, sd=2))
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_sim_distribution_exponential(self):
        seed()
        X = RV(Exponential(rate=2))
        sims = X.sim(Nsim)
        cdf = stats.expon(scale=0.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_sim_negative_n_raises(self):
        X = RV(Normal(mean=0, sd=1))
        with self.assertRaises(ValueError):
            X.sim(-1)

    def test_sim_float_n_raises(self):
        X = RV(Normal(mean=0, sd=1))
        with self.assertRaises(ValueError):
            X.sim(2.5)


class TestRVCall(unittest.TestCase):

    def test_call_emits_user_warning(self):
        X = RV(Normal(mean=0, sd=1))
        with self.assertWarns(UserWarning):
            X(0.5)

    def test_call_identity_func(self):
        X = RV(Normal(mean=0, sd=1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertAlmostEqual(X(3.14), 3.14)

    def test_call_custom_func(self):
        X = RV(Normal(mean=0, sd=1), lambda x: x * 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertAlmostEqual(X(3.0), 6.0)

    def test_call_sum_func(self):
        X = RV(BoxModel([0, 1], size=5), sum)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assertEqual(X((1, 1, 1, 0, 0)), 3)


class TestRVApply(unittest.TestCase):

    def test_apply_returns_rv(self):
        X = RV(Normal(mean=0, sd=1))
        Y = X.apply(abs)
        self.assertIsInstance(Y, RV)

    def test_apply_abs_non_negative(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        sims = X.apply(abs).sim(Nsim)
        self.assertTrue(all(v >= 0 for v in sims))

    def test_apply_square_gives_chi2(self):
        seed()
        X = RV(Normal(mean=0, var=1))
        sims = X.apply(lambda x: x**2).sim(Nsim)
        cdf = stats.chi2(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_apply_log_uniform_gives_exponential(self):
        seed()
        X = RV(Uniform(a=0, b=1))
        sims = X.apply(lambda x: -log(x)).sim(Nsim)
        cdf = stats.expon(scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_apply_preserves_prob_space(self):
        P = Normal(mean=0, sd=1)
        X = RV(P)
        Y = X.apply(lambda x: x + 1)
        self.assertIs(Y.prob_space, P)


class TestRVIter(unittest.TestCase):

    def test_unpack_two_components(self):
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        self.assertIsInstance(X, RV)
        self.assertIsInstance(Y, RV)

    def test_unpack_three_components(self):
        X, Y, Z = RV(Normal(mean=0, var=1) ** 3)
        self.assertIsInstance(X, RV)
        self.assertIsInstance(Y, RV)
        self.assertIsInstance(Z, RV)

    def test_unpack_scalar_raises(self):
        X = RV(Normal(mean=0, sd=1))
        with self.assertRaises(Exception):
            a, b = X

    def test_components_independent(self):
        seed(1)
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        s1 = list(X.sim(Nsim))
        seed(2)
        s2 = list(Y.sim(Nsim))
        self.assertNotEqual(s1, s2)

    def test_unpacked_components_correct_marginals(self):
        seed()
        X, Y = RV(Uniform(a=0, b=1) ** 2)
        sims_x = X.sim(Nsim)
        cdf = stats.uniform(loc=0, scale=1).cdf
        pval = stats.kstest(sims_x, cdf).pvalue
        self.assertTrue(pval > 0.01)


class TestRVGetItem(unittest.TestCase):

    def test_int_index_returns_rv(self):
        X = RV(BoxModel([1, 2, 3], size=3))
        self.assertIsInstance(X[0], RV)

    def test_int_index_value_in_support(self):
        seed()
        X = RV(BoxModel([10, 20, 30], size=3))
        sims = X[0].sim(200)
        self.assertTrue(all(v in (10, 20, 30) for v in sims))

    def test_list_index_returns_vector_length(self):
        seed()
        X = RV(BoxModel([1, 2, 3, 4, 5], size=5))
        sub = X[[0, 2]].draw()
        self.assertEqual(len(sub), 2)

    def test_slice_index_returns_vector_length(self):
        seed()
        X = RV(BoxModel([1, 2, 3, 4, 5], size=5))
        sub = X[0:3].draw()
        self.assertEqual(len(sub), 3)

    def test_rv_index_returns_rv(self):
        P = BoxModel(list(range(5)), size=5)
        X = RV(P)
        I = RV(P, lambda x: 0)
        result = X[I]
        self.assertIsInstance(result, RV)

    def test_getitem_vector_component(self):
        seed()
        Z = RV(Bernoulli(p=1) ** 2)
        sims = Z[0].sim(100)
        self.assertTrue(all(v == 1 for v in sims))


class TestRVArithmetic(unittest.TestCase):

    def test_add_rv_rv_normal(self):
        seed()
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X + Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=np.sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_add_rv_scalar_shift(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        sims = (X + 5).sim(Nsim)
        cdf = stats.norm(loc=5, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_sub_rv_rv_normal(self):
        seed()
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X - Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=np.sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_mul_scalar_rv(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        sims = (3 * X).sim(Nsim)
        cdf = stats.norm(loc=0, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_div_rv_scalar(self):
        seed()
        X = RV(Normal(mean=0, sd=2))
        sims = (X / 2).sim(Nsim)
        cdf = stats.norm(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_pow_rv_int_chi2(self):
        seed()
        X = RV(Normal(mean=0, var=1))
        sims = (X**2).sim(Nsim)
        cdf = stats.chi2(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_add_poisson_additive(self):
        seed()
        X, Y = RV(Poisson(lam=3) * Poisson(lam=5))
        sims = (X + Y).sim(Nsim)
        simulated = sims.tabulate()
        exp_list, obs_list = [], []
        for k in range(20):
            expected = Nsim * stats.poisson(mu=8).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_add_different_prob_spaces_raises(self):
        X = RV(Normal(mean=0, sd=1))
        Y = RV(Normal(mean=0, sd=1))
        with self.assertRaises(Exception):
            X + Y


class TestRVComparison(unittest.TestCase):

    def test_gt_scalar_returns_event(self):
        from symbulate.probability_space import Event

        X = RV(Normal(mean=0, sd=1))
        event = X > 0
        self.assertIsInstance(event, Event)

    def test_lt_scalar_returns_event(self):
        from symbulate.probability_space import Event

        X = RV(Normal(mean=0, sd=1))
        event = X < 0
        self.assertIsInstance(event, Event)

    def test_eq_scalar_returns_event(self):
        from symbulate.probability_space import Event

        X = RV(Bernoulli(p=0.5))
        event = X == 1
        self.assertIsInstance(event, Event)

    def test_comparison_rv_rv_same_space(self):
        from symbulate.probability_space import Event

        X, Y = RV(Normal(mean=0, var=1) ** 2)
        event = X < Y
        self.assertIsInstance(event, Event)

    def test_conditional_gt_all_positive(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        sims = (X | (X > 0)).sim(500)
        self.assertTrue(all(v > 0 for v in sims))

    def test_comparison_non_scalar_raises(self):
        X = RV(Normal(mean=0, sd=1))
        with self.assertRaises((NotImplementedError, Exception)):
            _ = X > [1, 2, 3]


class TestRVJoint(unittest.TestCase):

    def test_and_rv_rv_returns_rv(self):
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        Z = X & Y
        self.assertIsInstance(Z, RV)

    def test_and_rv_rv_draw_length(self):
        seed()
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        Z = X & Y
        val = Z.draw()
        self.assertEqual(len(val), 2)

    def test_and_rv_scalar_draw_length(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        Z = X & 5
        val = Z.draw()
        self.assertEqual(len(val), 2)

    def test_rand_scalar_first_component_constant(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        Z = 3 & X
        sims = Z.sim(100)
        self.assertTrue(all(v[0] == 3 for v in sims))

    def test_rand_scalar_draw_length(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        Z = 3 & X
        val = Z.draw()
        self.assertEqual(len(val), 2)

    def test_and_rv_timefunc_returns_rv(self):
        X = RV(Normal(mean=0, sd=1))
        t = InfiniteVector(lambda n: n**2)
        Z = X & t
        self.assertIsInstance(Z, RV)

    def test_and_rv_timefunc_draw_length(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        t = InfiniteVector(lambda n: n**2)
        Z = X & t
        val = Z.draw()
        self.assertEqual(len(val), 2)

    def test_and_rv_timefunc_second_component_constant(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        t = InfiniteVector(lambda n: n**2)
        Z = X & t
        sims = Z.sim(100)
        self.assertTrue(all(v[1] is t for v in sims))

    def test_rand_timefunc_first_component_constant(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        t = InfiniteVector(lambda n: n**2)
        Z = t & X
        sims = Z.sim(100)
        self.assertTrue(all(v[0] is t for v in sims))

    def test_rand_timefunc_draw_length(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        t = InfiniteVector(lambda n: n**2)
        Z = t & X
        val = Z.draw()
        self.assertEqual(len(val), 2)

    def test_and_non_rv_non_scalar_raises(self):
        X = RV(Normal(mean=0, sd=1))
        with self.assertRaises(Exception):
            _ = X & [1, 2, 3]

    def test_and_sum_matches_sum_rv(self):
        seed(7)
        X, Y = RV(Bernoulli(p=0.5) ** 2)
        sims_and = (X & Y).sim(Nsim).apply(sum)
        seed(7)
        sims_add = (X + Y).sim(Nsim)
        self.assertAlmostEqual(
            np.mean(list(sims_and)), np.mean(list(sims_add)), delta=0.05
        )


class TestRVConditionalConstruction(unittest.TestCase):

    def test_or_returns_rvconditional(self):
        X = RV(Normal(mean=0, sd=1))
        cond = X | (X > 0)
        self.assertIsInstance(cond, RVConditional)

    def test_rvconditional_is_rv_subclass(self):
        X = RV(Normal(mean=0, sd=1))
        cond = X | (X > 0)
        self.assertIsInstance(cond, RV)

    def test_conditional_stores_condition_event(self):
        X = RV(Normal(mean=0, sd=1))
        event = X > 0
        cond = X | event
        self.assertIs(cond.condition_event, event)

    def test_or_non_event_raises(self):
        X = RV(Normal(mean=0, sd=1))
        with self.assertRaises((NotImplementedError, Exception)):
            _ = X | True


class TestRVConditionalDraw(unittest.TestCase):

    def test_draw_satisfies_gt_condition(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        cond = X | (X > 0)
        for _ in range(200):
            self.assertGreater(cond.draw(), 0)

    def test_draw_satisfies_lt_condition(self):
        seed()
        X = RV(Normal(mean=0, sd=1))
        cond = X | (X < 0)
        for _ in range(200):
            self.assertLess(cond.draw(), 0)

    def test_sim_satisfies_condition(self):
        seed()
        X = RV(Exponential(rate=1))
        sims = (X | (X > 1)).sim(500)
        self.assertTrue(all(v > 1 for v in sims))


class TestRVConditionalDistribution(unittest.TestCase):

    def test_uniform_given_positive_half(self):
        # U(0,1) | U > 0.5  ~  Uniform(0.5, 1)
        seed()
        X = RV(Uniform(a=0, b=1))
        sims = (X | (X > 0.5)).sim(2000)
        cdf = stats.uniform(loc=0.5, scale=0.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_normal_truncated_mean(self):
        # Mean of N(0,1) | X > 0 is sqrt(2/pi) ≈ 0.798
        seed()
        X = RV(Normal(mean=0, sd=1))
        sims = list((X | (X > 0)).sim(Nsim))
        self.assertAlmostEqual(np.mean(sims), np.sqrt(2 / np.pi), delta=0.05)

    def test_binomial_conditional_sum(self):
        # X~Bin(5,p), Y~Bin(5,p): X | (X+Y==5) ~ Hypergeometric(N=10, K=5, n=5)
        # (the p cancels; result is Hypergeometric, not Binomial)
        seed()
        X, Y = RV(Binomial(n=5, p=0.4) * Binomial(n=5, p=0.4))
        sims = (X | (X + Y == 5)).sim(Nsim)
        simulated = sims.tabulate()
        exp_list, obs_list = [], []
        for k in range(6):
            expected = Nsim * stats.hypergeom(M=10, n=5, N=5).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_poisson_conditional_binomial(self):
        # X~Pois(6), Y~Pois(7): X | (X+Y==12) ~ Bin(12, 6/13)
        seed()
        X, Y = RV(Poisson(lam=6) * Poisson(lam=7))
        sims = (X | (X + Y == 12)).sim(Nsim)
        simulated = sims.tabulate()
        exp_list, obs_list = [], []
        for k in range(13):
            expected = Nsim * stats.binom(n=12, p=6 / 13).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_exponential_truncated_distribution(self):
        # Exp(1) | X > 1 ~ 1 + Exp(1)  (memoryless property)
        seed()
        X = RV(Exponential(rate=1))
        sims = (X | (X > 1)).sim(Nsim)
        shifted = [v - 1 for v in sims]
        cdf = stats.expon(scale=1).cdf
        pval = stats.kstest(shifted, cdf).pvalue
        self.assertTrue(pval > 0.01)


class TestCheckSameProbSpace(unittest.TestCase):

    def test_same_space_no_raise(self):
        P = Normal(mean=0, sd=1) ** 2
        X, Y = RV(P)
        try:
            X.check_same_prob_space(Y)
        except Exception:
            self.fail("check_same_prob_space raised for same probability space.")

    def test_different_spaces_raises(self):
        X = RV(Normal(mean=0, sd=1))
        Y = RV(Normal(mean=0, sd=1))
        with self.assertRaises(Exception):
            X.check_same_prob_space(Y)

    def test_no_prob_space_attr_ignored(self):
        X = RV(Normal(mean=0, sd=1))
        try:
            X.check_same_prob_space(42)
        except Exception:
            self.fail("check_same_prob_space raised for non-RV scalar.")

    def test_no_prob_space_attr_string_ignored(self):
        X = RV(Normal(mean=0, sd=1))
        try:
            X.check_same_prob_space("hello")
        except Exception:
            self.fail("check_same_prob_space raised for string argument.")


if __name__ == "__main__":
    unittest.main()
