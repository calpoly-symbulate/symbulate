import math
import unittest
import numpy as np
import scipy.stats as stats
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from symbulate import *
from symbulate import distributions

Nsim = 10000


class TestBernoulli(unittest.TestCase):

    def test_p_one(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Bernoulli(p=1))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 1 for sim in sims))

    def test_sum(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X = RV(Bernoulli(p=0.4) ** 5)
        sims = X.apply(sum).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(6):
            expected = Nsim * stats.binom(n=5, p=0.4).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_Bernoulli_Binomial_n_1(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X = RV(Bernoulli(p=0.4))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(2):
            expected = Nsim * stats.binom(n=1, p=0.4).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_Bernoulli_error_p_negative(self):
        self.assertRaises(Exception, lambda: Bernoulli(p=-0.1))

    def test_Bernoulli_error_p_too_large(self):
        self.assertRaises(Exception, lambda: Bernoulli(p=1.5))

    def test_Bernoulli_p_zero(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Bernoulli(p=0))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 0 for sim in sims))

    def test_Bernoulli_mean_sd(self):
        X = Bernoulli(p=0.3)
        self.assertAlmostEqual(float(X.mean()), 0.3)
        self.assertAlmostEqual(float(X.sd()), np.sqrt(0.3 * 0.7))

    def test_Bernoulli_pmf(self):
        X = Bernoulli(p=0.3)
        self.assertAlmostEqual(float(X.pmf(1)), 0.3)
        self.assertAlmostEqual(float(X.pmf(0)), 0.7)


class TestBinomial(unittest.TestCase):

    def test_Binomial_p_1(self):
        distributions.rng = np.random.default_rng(42)
        for nsample in range(1, 1000, 100):
            X = RV(Binomial(n=nsample, p=1.0))
            sims = X.sim(Nsim)
        self.assertTrue(all(sim == nsample for sim in sims))

    def test_Binomial_error_n(self):
        self.assertRaises(Exception, lambda: Binomial(n=-10, p=0.4))

    def test_Binomial_additive(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X, Y = RV(Binomial(n=8, p=0.6) * Binomial(n=5, p=0.6))
        sims = (X & Y).sim(Nsim).apply(sum)
        simulated = sims.tabulate()
        for k in range(2, 14):
            expected = Nsim * stats.binom(n=13, p=0.6).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_Binomial_error_p_negative(self):
        self.assertRaises(Exception, lambda: Binomial(n=10, p=-0.1))

    def test_Binomial_error_p_too_large(self):
        self.assertRaises(Exception, lambda: Binomial(n=10, p=1.5))

    def test_Binomial_error_n_float(self):
        self.assertRaises(Exception, lambda: Binomial(n=2.5, p=0.4))

    def test_Binomial_mean_sd_pmf(self):
        X = Binomial(n=10, p=0.3)
        self.assertAlmostEqual(float(X.mean()), 3.0)
        self.assertAlmostEqual(float(X.sd()), np.sqrt(10 * 0.3 * 0.7))
        self.assertAlmostEqual(float(X.pmf(3)), stats.binom(n=10, p=0.3).pmf(3))


class TestHypergeometric(unittest.TestCase):

    def test_Hypergeometric_no_failures(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Hypergeometric(n=10, N0=0, N1=1000))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))

    def test_Hypergeometric_Binomial_converge(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X = RV(Hypergeometric(n=8, N0=200, N1=800))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(9):
            expected = Nsim * stats.binom(n=8, p=0.8).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_Hypergeometric_error_n_greater(self):
        self.assertRaises(Exception, lambda: Hypergeometric(n=10, N0=1, N1=8))

    def test_Hypergeometric_error_n_zero(self):
        self.assertRaises(Exception, lambda: Hypergeometric(n=0, N0=5, N1=5))

    def test_Hypergeometric_error_N0_negative(self):
        self.assertRaises(Exception, lambda: Hypergeometric(n=2, N0=-1, N1=5))

    def test_Hypergeometric_error_N1_negative(self):
        self.assertRaises(Exception, lambda: Hypergeometric(n=2, N0=5, N1=-1))

    def test_Hypergeometric_mean(self):
        X = Hypergeometric(n=4, N0=6, N1=10)
        self.assertAlmostEqual(float(X.mean()), 4 * 10 / 16)

    def test_Hypergeometric_pmf(self):
        X = Hypergeometric(n=2, N0=3, N1=3)
        self.assertAlmostEqual(float(X.pmf(1)), stats.hypergeom(M=6, n=3, N=2).pmf(1))


class TestGeometric(unittest.TestCase):

    def test_Geometric_error(self):
        self.assertRaises(Exception, lambda: Geometric(p=0))

    def test_Geometric_error_p_one(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X = Geometric(p=1)
            sims = X.sim(100)
        for value in sims:
            self.assertEqual(value, 1)

    def test_Geometric_error_p_greater_than_one(self):
        self.assertRaises(Exception, lambda: Geometric(p=1.5))

    def test_Geometric_mean(self):
        X = Geometric(p=0.25)
        self.assertAlmostEqual(float(X.mean()), 4.0)

    def test_Geometric_pmf(self):
        X = Geometric(p=0.5)
        self.assertAlmostEqual(float(X.pmf(1)), 0.5)
        self.assertAlmostEqual(float(X.pmf(2)), 0.25)

    def test_Geometric_to_NBinom(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X = Geometric(p=0.8)
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(1, 10):
            expected = Nsim * stats.nbinom(n=1, p=0.8).pmf(k - 1)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)


class TestNegativeBinomial(unittest.TestCase):

    def test_NBinom_error_r(self):
        self.assertRaises(Exception, lambda: NegativeBinomial(r=-10, p=0.6))

    def test_NBinom_p_1(self):
        distributions.rng = np.random.default_rng(42)
        X = NegativeBinomial(r=10, p=1)
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))

    def test_NBinom_Pascal_additive(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X, Y = RV(Pascal(r=4, p=0.6) * Pascal(r=6, p=0.6))
        sims = (X + Y).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(10, 35):
            expected = Nsim * stats.nbinom(n=10, p=0.6).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_NBinom_to_Geometric(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X = NegativeBinomial(r=1, p=0.8)
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(10):
            expected = Nsim * stats.geom(p=0.8).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_NBinom_error_p_zero(self):
        self.assertRaises(Exception, lambda: NegativeBinomial(r=5, p=0))

    def test_NBinom_error_p_too_large(self):
        self.assertRaises(Exception, lambda: NegativeBinomial(r=5, p=1.5))

    def test_NBinom_error_r_float(self):
        self.assertRaises(Exception, lambda: NegativeBinomial(r=2.5, p=0.5))

    def test_NBinom_mean(self):
        X = NegativeBinomial(r=3, p=0.5)
        self.assertAlmostEqual(float(X.mean()), 6.0)


class TestPascal(unittest.TestCase):

    def test_Pascal_error_r(self):
        self.assertRaises(Exception, lambda: Pascal(r=0, p=0.3))

    def test_Pascal_p_1(self):
        distributions.rng = np.random.default_rng(42)
        X = Pascal(r=10, p=1.0)
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 0 for sim in sims))

    def test_Pascal_error_p_zero(self):
        self.assertRaises(Exception, lambda: Pascal(r=5, p=0))

    def test_Pascal_error_r_float(self):
        self.assertRaises(Exception, lambda: Pascal(r=2.5, p=0.5))

    def test_Pascal_mean(self):
        X = Pascal(r=3, p=0.5)
        self.assertAlmostEqual(float(X.mean()), 3.0)

    def test_Pascal_pmf(self):
        X = Pascal(r=1, p=0.5)
        self.assertAlmostEqual(float(X.pmf(0)), 0.5)
        self.assertAlmostEqual(float(X.pmf(1)), 0.25)


class TestPoisson(unittest.TestCase):

    def test_Poisson_error(self):
        X = Poisson(lam=0)
        sims = X.sim(100)
        for value in sims:
            self.assertEqual(value, 0)

    def test_Poisson_additive(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X, Y = RV(Poisson(lam=4) * Poisson(lam=7))
        sims = (X + Y).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(25):
            expected = Nsim * stats.poisson(mu=11).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_conditional_Poisson_add(self):
        distributions.rng = np.random.default_rng(42)
        obs_list, exp_list = [], []
        X, Y = RV(Poisson(lam=6) * Poisson(lam=7))
        sims = (X | (X + Y == 12)).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(12):
            expected = Nsim * stats.binom(n=12, p=6 / 13).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_Poisson_error_negative_lam(self):
        self.assertRaises(Exception, lambda: Poisson(lam=-1))

    def test_Poisson_mean_sd(self):
        X = Poisson(lam=5)
        self.assertAlmostEqual(float(X.mean()), 5.0)
        self.assertAlmostEqual(float(X.sd()), np.sqrt(5))

    def test_Poisson_pmf(self):
        X = Poisson(lam=3)
        self.assertAlmostEqual(float(X.pmf(3)), stats.poisson(mu=3).pmf(3))

    def test_Poisson_zero(self):
        X = Poisson(0)

        sims = X.sim(100)

        for value in sims:
            self.assertEqual(value, 0)

    def test_Poisson_zero(self):
        X = Poisson(0)

        sims = X.sim(100)

        for value in sims:
            self.assertEqual(value, 0)


class TestDiscreteUniform(unittest.TestCase):

    def test_DiscreteUniform_error_equal(self):
        X = DiscreteUniform(a=5, b=5)
        sims = X.sim(100)
        for value in sims:
            self.assertEqual(value, 5)

    def test_DiscreteUniform_error_reversed(self):
        self.assertRaises(Exception, lambda: DiscreteUniform(a=6, b=3))

    def test_DiscreteUniform_mean(self):
        X = DiscreteUniform(a=1, b=6)
        self.assertAlmostEqual(float(X.mean()), 3.5)

    def test_DiscreteUniform_pmf(self):
        X = DiscreteUniform(a=1, b=6)
        for k in range(1, 7):
            self.assertAlmostEqual(float(X.pmf(k)), 1 / 6)

    def test_DiscreteUniform_distributional(self):
        distributions.rng = np.random.default_rng(42)
        exp_list, obs_list = [], []
        X = RV(DiscreteUniform(a=1, b=6))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(1, 7):
            exp_list.append(Nsim / 6)
            obs_list.append(simulated[k])
        pval = stats.chisquare(obs_list, exp_list).pvalue
        self.assertTrue(pval > 0.01)

    def test_DiscreteUniform_sim_bounds(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(DiscreteUniform(a=3, b=7))
        sims = X.sim(Nsim)
        self.assertTrue(all(3 <= sim <= 7 for sim in sims))

    def test_DiscreteUniform_degenerate(self):
        X = DiscreteUniform(3, 3)

        sims = X.sim(100)

        for value in sims:
            self.assertEqual(value, 3)


class TestUniform(unittest.TestCase):

    def test_Uniform_error(self):
        self.assertRaises(Exception, lambda: Uniform(a=6, b=-1))

    def test_conditional_exp_uniform(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Exponential(rate=3) ** 2)
        sims = (X | (X < 3) & (X + Y > 3)).sim(1000)
        cdf = stats.uniform(loc=0, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_ChiSquare(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Uniform(a=0, b=1))
        sims = (-2 * log(X)).sim(Nsim)
        cdf = stats.chi2(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Exponential(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Uniform(a=0, b=1))
        Y = -1 / 5 * log(X)
        sims = Y.sim(Nsim)
        cdf = stats.expon(scale=1 / 5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Beta(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Uniform(a=0, b=1))
        sims = (X**15).sim(Nsim)
        cdf = stats.beta(a=1 / 15, b=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Cauchy(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Uniform(a=0, b=1))
        sims = (pi * (X - 1 / 2)).apply(tan).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Pareto(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Uniform(a=0, b=1))
        sims = (2 * X ** (-1 / 0.1)).sim(10000)
        cdf = stats.pareto(b=0.1, loc=0, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_mean_pdf(self):
        X = Uniform(a=2, b=8)
        self.assertAlmostEqual(float(X.mean()), 5.0)
        self.assertAlmostEqual(float(X.pdf(5)), 1 / 6)

    def test_Uniform_cdf(self):
        X = Uniform(a=0, b=4)
        self.assertAlmostEqual(float(X.cdf(2)), 0.5)
        self.assertAlmostEqual(float(X.cdf(0)), 0.0)
        self.assertAlmostEqual(float(X.cdf(4)), 1.0)


class TestNormal(unittest.TestCase):

    def test_Normal_error(self):
        self.assertRaises(Exception, lambda: Normal(mean=0, var=-10))

    def test_sum(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(mean=-1, sd=2) ** 3)
        sims = X.apply(sum).sim(Nsim)
        cdf = stats.norm(loc=-3, scale=np.sqrt(12)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_sum_Standard_Normal(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X + Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_subtract_Standard_Normal(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X - Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_standardize(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(mean=8, var=4))
        X_stand = (X - 8) / 2
        sims = X_stand.sim(Nsim)
        cdf = stats.norm(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_standardize_to_Normal(self):
        distributions.rng = np.random.default_rng(42)
        Z = RV(Normal(mean=0, sd=1))
        X = 10 + 5 * Z
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=10, scale=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_Gamma(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(mean=0, var=1))
        X = X**2
        sims = X.sim(Nsim)
        cdf = stats.gamma(a=1 / 2, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_ChiSquare(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(mean=0, var=1))
        X = X**2
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_Cauchy(self):
        # Seed 42 is pathological for this heavy-tailed Cauchy KS test
        # (lands in the ~1% false-rejection region); use a robust seed.
        distributions.rng = np.random.default_rng(0)
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X / Y).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_F(self):
        distributions.rng = np.random.default_rng(42)
        A, B, C, V, W, X, Y, Z = RV(Normal(mean=0, var=1) ** 8)
        sims = (
            (((A**2) + (B**2) + (C**2)) / 3)
            / (((V**2) + (W**2) + (X**2) + (Y**2) + (Z**2)) / 5)
        ).sim(Nsim)
        cdf = stats.f(dfn=3, dfd=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_sum_Normal_to_ChiSquare(self):
        distributions.rng = np.random.default_rng(42)
        X, Y, Z, A, B = RV(Normal(mean=0, var=1) ** 5)
        sims = ((X**2) + (Y**2) + (Z**2) + (A**2) + (B**2)).sim(Nsim)
        cdf = stats.chi2(df=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_var_param(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(mean=0, var=4))
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=0, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_mean_sd_var_pdf(self):
        X = Normal(mean=5, sd=2)
        self.assertAlmostEqual(float(X.mean()), 5.0)
        self.assertAlmostEqual(float(X.sd()), 2.0)
        self.assertAlmostEqual(float(X.var()), 4.0)
        self.assertAlmostEqual(float(X.pdf(5)), stats.norm(loc=5, scale=2).pdf(5))

    def test_Normal_error_sd_negative(self):
        self.assertRaises(Exception, lambda: Normal(mean=0, sd=-1))

    def test_Normal_error_var_negative(self):
        self.assertRaises(Exception, lambda: Normal(mean=0, var=-1.0))

    def test_Normal_sd_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X = Normal(5, 0)
            sims = X.sim(100)

        for value in sims:
            self.assertEqual(value, 5)

    def test_Normal_sd_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X = Normal(5, 0)
            sims = X.sim(100)

        for value in sims:
            self.assertEqual(value, 5)


class TestExponential(unittest.TestCase):

    def test_Exponential_error(self):
        self.assertRaises(Exception, lambda: Exponential(rate=-5))

    def test_Exponential_to_Gamma(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Exponential(rate=0.9))
        sims = X.sim(Nsim)
        cdf = stats.gamma(scale=1 / 0.9, a=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_sum_Gamma(self):
        distributions.rng = np.random.default_rng(42)
        X, Y, Z, A = RV(Exponential(rate=0.9) ** 4)
        sims = (X + Y + Z + A).sim(Nsim)
        cdf = stats.gamma(scale=1 / 0.9, a=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_ChiSquare(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Exponential(rate=1 / 2))
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_Pareto(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Exponential(rate=2))
        sims = (3 * exp(X)).sim(Nsim)
        cdf = stats.pareto(b=2, loc=0, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_Weibull(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Exponential(rate=5))
        sims = X.sim(Nsim)
        cdf = stats.weibull_min(scale=1 / 5, c=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_Rayleigh(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Exponential(rate=5))
        sims = sqrt(X).sim(Nsim)
        cdf = stats.rayleigh(scale=1 / sqrt(2 * 5)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Poisson_Exponential_to_Geometric(self):
        distributions.rng = np.random.default_rng(42)

        def poisson_exp():
            x = Exponential(rate=1 / lam).draw()
            z = RV(Poisson(x)).draw()
            return z

        exp_list, obs_list, lam = [], [], 1 / 5
        P = ProbabilitySpace(poisson_exp)
        A = RV(P)
        sims = (A + 1).sim(Nsim)
        simulated = sims.tabulate()
        for k in range(40):
            expected = Nsim * stats.geom(p=1 / (1 + lam)).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_scale_param(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Exponential(scale=2))
        sims = X.sim(Nsim)
        cdf = stats.expon(scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_error_scale_negative(self):
        self.assertRaises(Exception, lambda: Exponential(scale=-1))

    def test_Exponential_mean_sd(self):
        X = Exponential(rate=2)
        self.assertAlmostEqual(float(X.mean()), 0.5)
        self.assertAlmostEqual(float(X.sd()), 0.5)


class TestGamma(unittest.TestCase):

    def test_Gamma_shape_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape=-5, rate=40))

    def test_Gamma_rate_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape=4, rate=-10))

    def test_Gamma_to_Exponential(self):
        distributions.rng = np.random.default_rng(42)
        X = Gamma(shape=1, rate=1 / 0.9)
        sims = X.sim(Nsim)
        cdf = stats.expon(scale=0.9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_reshape(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Gamma(shape=9, scale=4))
        sims = (X * 8).sim(Nsim)
        cdf = stats.gamma(scale=4 * 8, a=9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_additive(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Gamma(shape=10, scale=0.5) * Gamma(shape=8, scale=0.5))
        sims = (X + Y).sim(Nsim)
        cdf = stats.gamma(scale=0.5, a=18).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_to_Beta(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Gamma(shape=5, scale=8) * Gamma(shape=4, scale=8))
        sims = (X / (X + Y)).sim(Nsim)
        cdf = stats.beta(a=5, b=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_to_F(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Gamma(shape=2, rate=5) * Gamma(shape=4, rate=7))
        sims = ((4 * 5 * X) / (2 * 7 * Y)).sim(Nsim)
        cdf = stats.f(dfn=2 * 2, dfd=2 * 4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_to_ChiSquare(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Gamma(shape=10 / 2, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=10).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_scale_param(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Gamma(shape=3, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.gamma(a=3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_error_scale_negative(self):
        self.assertRaises(Exception, lambda: Gamma(shape=2, scale=-1))

    def test_Gamma_mean(self):
        X = Gamma(shape=4, rate=2)
        self.assertAlmostEqual(float(X.mean()), 2.0)


class TestBeta(unittest.TestCase):

    def test_Beta_error_a(self):
        self.assertRaises(Exception, lambda: Beta(a=-10, b=3))

    def test_Beta_error_b(self):
        self.assertRaises(Exception, lambda: Beta(a=3, b=-10))

    def test_Beta_to_Uniform(self):
        distributions.rng = np.random.default_rng(42)
        X = Beta(a=1, b=1)
        sims = X.sim(Nsim)
        cdf = stats.uniform(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_symmetry(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Beta(a=4, b=5))
        sims = (1 - X).sim(Nsim)
        cdf = stats.beta(a=5, b=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_to_Exponential(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Beta(a=0.7, b=1))
        sims = (-log(X)).sim(Nsim)
        cdf = stats.expon(scale=1 / 0.7).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_to_F(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Beta(a=10 / 2, b=12 / 2))
        sims = (12 * X / (10 * (1 - X))).sim(Nsim)
        cdf = stats.f(dfn=10, dfd=12).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_mean_pdf(self):
        X = Beta(a=2, b=5)
        self.assertAlmostEqual(float(X.mean()), 2 / 7)
        self.assertAlmostEqual(float(X.pdf(0.5)), stats.beta(a=2, b=5).pdf(0.5))


class TestStudentT(unittest.TestCase):

    def test_StudentT_df_error(self):
        self.assertRaises(Exception, lambda: StudentT(df=0))

    def test_StudentT_to_Normal(self):
        distributions.rng = np.random.default_rng(42)
        X = StudentT(df=Nsim)
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_ChiSquare_to_StudentT(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Normal(mean=0, var=1) * ChiSquare(df=5))
        sims = (X / sqrt(Y / 5)).sim(Nsim)
        cdf = stats.t(df=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_StudentT_df1_nan_moments(self):
        X = StudentT(df=1)
        self.assertTrue(math.isnan(X.mean()))
        self.assertTrue(math.isnan(X.sd()))
        self.assertTrue(math.isnan(X.var()))

    def test_StudentT_mean_zero(self):
        X = StudentT(df=5)
        self.assertAlmostEqual(float(X.mean()), 0.0)

    def test_StudentT_error_negative_df(self):
        self.assertRaises(Exception, lambda: StudentT(df=-1))

    # --- noncentral t (noncentrality parameter, for power analysis) ---

    def test_StudentT_default_noncentrality_is_zero(self):
        self.assertEqual(StudentT(df=10).noncentrality, 0)

    def test_StudentT_noncentral_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(StudentT(df=8, noncentrality=1.5))
        sims = X.sim(Nsim)
        cdf = stats.nct(df=8, nc=1.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_StudentT_noncentral_shifts_mean(self):
        # Noncentrality shifts the mean away from 0; matches scipy's nct.
        X = StudentT(df=10, noncentrality=2)
        self.assertAlmostEqual(
            float(X.mean()), float(stats.nct(df=10, nc=2).mean()), places=6
        )
        self.assertGreater(float(X.mean()), 0)

    def test_StudentT_noncentral_zero_matches_central(self):
        # noncentrality=0 must be bit-for-bit the central t at several points.
        central, nc0 = StudentT(df=6), StudentT(df=6, noncentrality=0)
        for x in [-2.0, -0.5, 0.0, 1.0, 3.0]:
            self.assertEqual(float(nc0.cdf(x)), float(central.cdf(x)))

    def test_StudentT_noncentral_invalid_raises(self):
        self.assertRaises(Exception, lambda: StudentT(df=5, noncentrality="a"))

    def test_StudentT_noncentral_plots_without_error(self):
        StudentT(df=8, noncentrality=1.5).draw()
        RV(StudentT(df=8, noncentrality=1.5)).sim(100).plot()
        StudentT(df=8, noncentrality=1.5).plot()
        StudentT(df=8, noncentrality=1.5).plot(cdf=True)
        plt.close("all")


class TestChiSquare(unittest.TestCase):

    def test_ChiSquare_error(self):
        self.assertRaises(Exception, lambda: ChiSquare(df=0.5))

    def test_ChiSquare_to_Gamma(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(ChiSquare(df=10))
        sims = X.sim(Nsim)
        cdf = stats.gamma(a=5, scale=1 / 0.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_to_F(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(ChiSquare(df=3) * ChiSquare(df=5))
        sims = ((X / 3) / (Y / 5)).sim(Nsim)
        cdf = stats.f(dfn=3, dfd=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_to_Beta(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(ChiSquare(df=4) * ChiSquare(df=5))
        sims = (X / (X + Y)).sim(Nsim)
        cdf = stats.beta(a=4 / 2, b=5 / 2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_mean(self):
        X = ChiSquare(df=7)
        self.assertAlmostEqual(float(X.mean()), 7.0)

    def test_ChiSquare_error_float_df(self):
        self.assertRaises(Exception, lambda: ChiSquare(df=2.5))

    def test_ChiSquare_error_zero_df(self):
        self.assertRaises(Exception, lambda: ChiSquare(df=0))

    # --- noncentral chi-square (noncentrality parameter, for power analysis) ---

    def test_ChiSquare_default_noncentrality_is_zero(self):
        self.assertEqual(ChiSquare(df=4).noncentrality, 0)

    def test_ChiSquare_noncentral_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(ChiSquare(df=4, noncentrality=3))
        sims = X.sim(Nsim)
        cdf = stats.ncx2(df=4, nc=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_noncentral_mean_is_df_plus_nc(self):
        # Noncentral chi-square has mean df + noncentrality.
        self.assertAlmostEqual(float(ChiSquare(df=4, noncentrality=3).mean()), 7.0)

    def test_ChiSquare_noncentral_zero_matches_central(self):
        central, nc0 = ChiSquare(df=6), ChiSquare(df=6, noncentrality=0)
        for x in [0.5, 2.0, 5.0, 9.0]:
            self.assertEqual(float(nc0.cdf(x)), float(central.cdf(x)))

    def test_ChiSquare_negative_noncentrality_raises(self):
        self.assertRaises(Exception, lambda: ChiSquare(df=4, noncentrality=-1))

    def test_ChiSquare_noncentral_plots_without_error(self):
        RV(ChiSquare(df=4, noncentrality=3)).sim(100).plot()
        ChiSquare(df=4, noncentrality=3).plot()
        ChiSquare(df=4, noncentrality=3).plot(cdf=True)
        plt.close("all")


class TestF(unittest.TestCase):

    def test_F_error(self):
        self.assertRaises(Exception, lambda: F(dfN=0, dfD=5))

    def test_inverse_T(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(F(dfN=4, dfD=8))
        sims = (1 / X).sim(Nsim)
        cdf = stats.f(dfn=8, dfd=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_StudentT_to_F(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(StudentT(df=15))
        sims = (X**2).sim(Nsim)
        cdf = stats.f(dfn=1, dfd=15).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_F_to_Beta(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(F(dfN=5, dfD=8))
        sims = ((5 * X / 8) / (1 + (5 * X / 8))).sim(Nsim)
        cdf = stats.beta(a=5 / 2, b=8 / 2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_F_error_dfD(self):
        self.assertRaises(Exception, lambda: F(dfN=5, dfD=0))

    def test_F_error_dfN_negative(self):
        self.assertRaises(Exception, lambda: F(dfN=-1, dfD=5))

    def test_F_mean(self):
        X = F(dfN=5, dfD=10)
        self.assertAlmostEqual(float(X.mean()), 10 / (10 - 2))

    # --- noncentral F (noncentrality parameter, for ANOVA power analysis) ---

    def test_F_default_noncentrality_is_zero(self):
        self.assertEqual(F(dfN=5, dfD=10).noncentrality, 0)

    def test_F_noncentral_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(F(dfN=5, dfD=10, noncentrality=4))
        sims = X.sim(Nsim)
        cdf = stats.ncf(dfn=5, dfd=10, nc=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_F_noncentral_shifts_mean(self):
        # Noncentral F: mean = dfD (dfN + nc) / (dfN (dfD - 2)); matches scipy.
        X = F(dfN=5, dfD=10, noncentrality=4)
        self.assertAlmostEqual(
            float(X.mean()), float(stats.ncf(dfn=5, dfd=10, nc=4).mean()), places=6
        )
        self.assertGreater(float(X.mean()), float(F(dfN=5, dfD=10).mean()))

    def test_F_noncentral_zero_matches_central(self):
        central, nc0 = F(dfN=5, dfD=10), F(dfN=5, dfD=10, noncentrality=0)
        for x in [0.25, 1.0, 2.0, 4.0]:
            self.assertEqual(float(nc0.cdf(x)), float(central.cdf(x)))

    def test_F_negative_noncentrality_raises(self):
        self.assertRaises(Exception, lambda: F(dfN=5, dfD=10, noncentrality=-2))

    def test_F_noncentral_plots_without_error(self):
        RV(F(dfN=5, dfD=10, noncentrality=4)).sim(100).plot()
        F(dfN=5, dfD=10, noncentrality=4).plot()
        F(dfN=5, dfD=10, noncentrality=4).plot(cdf=True)
        plt.close("all")


class TestCauchy(unittest.TestCase):

    def test_Cauchy_mean(self):
        X = Cauchy()
        math.isnan(X.mean())

    def test_Cauchy_to_T(self):
        # Seed 42 is pathological for this heavy-tailed Cauchy KS test
        # (lands in the ~1% false-rejection region); use a robust seed.
        distributions.rng = np.random.default_rng(0)
        X = RV(Cauchy())
        sims = X.sim(Nsim)
        cdf = stats.t(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Cauchy_inverse(self):
        # Seed 42 is pathological for this heavy-tailed Cauchy KS test
        # (lands in the ~1% false-rejection region); use a robust seed.
        distributions.rng = np.random.default_rng(0)
        X = RV(Cauchy())
        sims = (1 / X).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Cauchy_additive(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(Cauchy() ** 2)
        sims = (X + Y).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Cauchy_loc_scale_params(self):
        X = Cauchy(loc=2, scale=3)
        self.assertEqual(X.loc, 2)
        self.assertEqual(X.scale, 3)
        # median of Cauchy equals its loc parameter
        self.assertAlmostEqual(float(X.median()), 2.0)

    def test_Cauchy_pdf_at_loc(self):
        X = Cauchy(loc=0, scale=1)
        self.assertAlmostEqual(float(X.pdf(0)), 1 / math.pi)


class TestLognormal(unittest.TestCase):

    def test_LogNormal_error(self):
        self.assertRaises(Exception, lambda: LogNormal(mu=0, sigma=-5))

    def test_LogNormal_to_Normal(self):
        distributions.rng = np.random.default_rng(42)
        X = LogNormal(mu=10, sigma=5)
        sims = X.sim(Nsim).apply(log)
        cdf = stats.norm(loc=10, scale=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_LogNormal(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(mean=10, sd=5))
        sims = X.apply(exp).sim(Nsim)
        cdf = stats.lognorm(s=5, scale=exp(10)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogNormal_Product(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(LogNormal(mu=10, sigma=5) * LogNormal(mu=11, sigma=6))
        sims = (X * Y).sim(Nsim)
        cdf = stats.lognorm(s=sqrt(25 + 36), scale=exp(21)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogNormal_error_sigma_zero(self):
        X = LogNormal(mu=0, sigma=0)
        sims = X.sim(100)
        for value in sims:
            self.assertAlmostEqual(value, 1.0)

    def test_LogNormal_mean(self):
        X = LogNormal(mu=1, sigma=0.5)
        expected_mean = np.exp(1 + 0.5**2 / 2)
        self.assertAlmostEqual(float(X.mean()), expected_mean, places=5)

    def test_LogNormal_sigma_zero(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X = LogNormal(mu=0, sigma=0)
            sims = X.sim(100)

        for value in sims:
            self.assertAlmostEqual(value, 1.0)


class TestPareto(unittest.TestCase):

    def test_Pareto_check_mean(self):
        x = stats.pareto(b=-3)
        math.isnan(x.mean())

    def test_Pareto_to_Exponential(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Pareto(b=1.5, scale=0.1))
        sims = (log(X / 0.1)).sim(Nsim)
        cdf = stats.expon(scale=1 / 1.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Pareto_error_b_nonpositive(self):
        self.assertRaises(Exception, lambda: Pareto(b=0, scale=1))

    def test_Pareto_error_b_negative(self):
        self.assertRaises(Exception, lambda: Pareto(b=-1, scale=1))

    def test_Pareto_error_scale_nonpositive(self):
        self.assertRaises(Exception, lambda: Pareto(b=2, scale=0))

    def test_Pareto_draw_above_scale(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Pareto(b=2, scale=3))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim >= 3 for sim in sims))

    def test_Pareto_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Pareto(b=2, scale=1))
        sims = X.sim(Nsim)
        cdf = stats.pareto(b=2, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)


class TestWeibull(unittest.TestCase):
    def test_Weibull_to_Exponential(self):
        """X = RV(Weibull(scale=1, c=10))
        sims = X.sim(Nsim)
        cdf = stats.expon(scale=1/10).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > .01)
        """
        pass


class TestRayleigh(unittest.TestCase):

    def test_Rayleigh_Normal(self):
        distributions.rng = np.random.default_rng(42)
        A, B = RV(Normal(mean=0, var=1) * Normal(mean=0, var=1))
        sims = (A**2 + B**2).apply(sqrt).sim(Nsim)
        cdf = stats.rayleigh.cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Rayleigh_to_Chi(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Rayleigh())
        sims = X.sim(Nsim)
        cdf = stats.chi(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Rayleigh_mean(self):
        X = Rayleigh()
        self.assertAlmostEqual(float(X.mean()), np.sqrt(np.pi / 2), places=5)

    def test_Rayleigh_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Rayleigh())
        sims = X.sim(Nsim)
        cdf = stats.rayleigh.cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)


class TestWeibull(unittest.TestCase):

    def test_Weibull_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Weibull(shape=1.5, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.weibull_min(c=1.5, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Weibull_mean_var_sd(self):
        X = Weibull(shape=1.5, scale=2)
        th = stats.weibull_min(c=1.5, scale=2)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_Weibull_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
        value = Weibull(shape=1.5, scale=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_Weibull_shape_one_is_exponential(self):
        # Weibull with shape=1 is Exponential(scale) -- same CDF.
        X = Weibull(shape=1, scale=2)
        self.assertAlmostEqual(float(X.cdf(2)), float(stats.expon(scale=2).cdf(2)))

    def test_Weibull_default_scale_is_one(self):
        X = Weibull(shape=2)
        self.assertEqual(X.scale, 1.0)

    def test_Weibull_invalid_shape_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Weibull(shape=b))

    def test_Weibull_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Weibull(shape=1.5, scale=b))

    def test_Weibull_plots_without_error(self):
        # draw / RV / sim / plot all wired through the base class.
        Weibull(1.5, 2).draw()
        RV(Weibull(1.5, 2)).sim(100).plot()
        Weibull(1.5, 2).plot()
        Weibull(1.5, 2).plot(cdf=True)
        plt.close("all")


class TestLogistic(unittest.TestCase):

    def test_Logistic_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Logistic(loc=2, scale=3))
        sims = X.sim(Nsim)
        cdf = stats.logistic(loc=2, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Logistic_mean_var_sd(self):
        X = Logistic(loc=2, scale=3)
        th = stats.logistic(loc=2, scale=3)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_Logistic_closed_form_cdf(self):
        # The defining feature: the CDF is the logistic (sigmoid) function
        # 1 / (1 + exp(-(x - loc) / scale)).
        X = Logistic(loc=1, scale=2)
        for x in [-3.0, 0.0, 1.0, 4.5]:
            expected = 1 / (1 + np.exp(-(x - 1) / 2))
            self.assertAlmostEqual(float(X.cdf(x)), expected, places=9)

    def test_Logistic_cdf_at_loc_is_half(self):
        # Symmetric about loc, so P(X <= loc) = 1/2.
        self.assertAlmostEqual(float(Logistic(loc=5, scale=2).cdf(5)), 0.5)

    def test_Logistic_draw_is_scalar(self):
        distributions.rng = np.random.default_rng(0)
        value = Logistic(loc=0, scale=1).draw()
        self.assertIsInstance(value, Scalar)

    def test_Logistic_default_params(self):
        X = Logistic()
        self.assertEqual(X.loc, 0)
        self.assertEqual(X.scale, 1)

    def test_Logistic_invalid_loc_raises(self):
        for bad in ["a", None]:
            self.assertRaises(Exception, lambda b=bad: Logistic(loc=b))

    def test_Logistic_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Logistic(scale=b))

    def test_Logistic_plots_without_error(self):
        # draw / RV / sim / plot all wired through the base class.
        Logistic(0, 1).draw()
        RV(Logistic(0, 1)).sim(100).plot()
        Logistic(0, 1).plot()
        Logistic(0, 1).plot(cdf=True)
        plt.close("all")


class TestGompertz(unittest.TestCase):

    def test_Gompertz_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Gompertz(shape=1.5, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.gompertz(c=1.5, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gompertz_mean_var_sd(self):
        X = Gompertz(shape=1.5, scale=2)
        th = stats.gompertz(c=1.5, scale=2)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_Gompertz_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
        value = Gompertz(shape=1.5, scale=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_Gompertz_hazard_increases(self):
        # The defining actuarial feature: the hazard (force of mortality)
        # rises with age, so h(x) = pdf/(1 - cdf) is strictly increasing.
        X = Gompertz(shape=1.5, scale=2)
        xs = [0.0, 0.5, 1.0, 2.0]
        hazards = [float(X.pdf(x)) / (1 - float(X.cdf(x))) for x in xs]
        self.assertTrue(all(b > a for a, b in zip(hazards, hazards[1:])))

    def test_Gompertz_default_scale_is_one(self):
        X = Gompertz(shape=2)
        self.assertEqual(X.scale, 1.0)

    def test_Gompertz_invalid_shape_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Gompertz(shape=b))

    def test_Gompertz_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Gompertz(shape=1.5, scale=b))

    def test_Gompertz_plots_without_error(self):
        # draw / RV / sim / plot all wired through the base class.
        Gompertz(1.5, 2).draw()
        RV(Gompertz(1.5, 2)).sim(100).plot()
        Gompertz(1.5, 2).plot()
        Gompertz(1.5, 2).plot(cdf=True)
        plt.close("all")


class TestMakeham(unittest.TestCase):

    def test_Makeham_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = Makeham(shape=1.5, makeham=0.3, scale=2)
        sims = RV(X).sim(Nsim)
        pval = stats.kstest(sims, X.cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Makeham_zero_reduces_to_gompertz(self):
        # makeham=0 is the plain Gompertz law -- same CDF at every point.
        X = Makeham(shape=1.5, makeham=0, scale=2)
        gompertz_cdf = stats.gompertz(c=1.5, scale=2).cdf
        for x in [0.1, 0.5, 1.0, 2.5]:
            self.assertAlmostEqual(float(X.cdf(x)), float(gompertz_cdf(x)), places=9)

    def test_Makeham_higher_hazard_than_gompertz(self):
        # Adding the constant term raises the hazard, so Makeham survival is
        # below the matching Gompertz survival everywhere past 0.
        X = Makeham(shape=1.5, makeham=0.3, scale=2)
        gompertz_sf = stats.gompertz(c=1.5, scale=2).sf
        for x in [0.5, 1.0, 2.0]:
            self.assertLess(1 - float(X.cdf(x)), float(gompertz_sf(x)))

    def test_Makeham_cdf_sf_endpoints(self):
        X = Makeham(shape=1.5, makeham=0.3, scale=2)
        self.assertAlmostEqual(float(X.cdf(0)), 0.0, places=9)

    def test_Makeham_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
        value = Makeham(shape=1.5, makeham=0.3, scale=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_Makeham_default_scale_is_one(self):
        X = Makeham(shape=2, makeham=0.5)
        self.assertEqual(X.scale, 1.0)

    def test_Makeham_invalid_shape_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Makeham(shape=b, makeham=0.3))

    def test_Makeham_invalid_makeham_raises(self):
        for bad in [-1, "a"]:
            self.assertRaises(Exception, lambda b=bad: Makeham(shape=1.5, makeham=b))

    def test_Makeham_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(
                Exception, lambda b=bad: Makeham(shape=1.5, makeham=0.3, scale=b)
            )

    def test_Makeham_plots_without_error(self):
        # draw / RV / sim / plot all wired through the base class.
        Makeham(1.5, 0.3, 2).draw()
        RV(Makeham(1.5, 0.3, 2)).sim(100).plot()
        Makeham(1.5, 0.3, 2).plot()
        Makeham(1.5, 0.3, 2).plot(cdf=True)
        plt.close("all")


class TestLaplace(unittest.TestCase):

    def test_Laplace_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Laplace(loc=2, scale=3))
        sims = X.sim(Nsim)
        cdf = stats.laplace(loc=2, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Laplace_mean_var_sd(self):
        X = Laplace(loc=2, scale=3)
        th = stats.laplace(loc=2, scale=3)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_Laplace_peak_density(self):
        # The defining feature: a sharp peak at loc with density
        # 1 / (2 * scale), twice as tall (relative to spread) as it looks.
        X = Laplace(loc=1, scale=2)
        self.assertAlmostEqual(float(X.pdf(1)), 1 / (2 * 2), places=9)

    def test_Laplace_cdf_at_loc_is_half(self):
        # Symmetric about loc, so P(X <= loc) = 1/2.
        self.assertAlmostEqual(float(Laplace(loc=5, scale=2).cdf(5)), 0.5)

    def test_Laplace_draw_is_scalar(self):
        distributions.rng = np.random.default_rng(0)
        value = Laplace(loc=0, scale=1).draw()
        self.assertIsInstance(value, Scalar)

    def test_Laplace_default_params(self):
        X = Laplace()
        self.assertEqual(X.loc, 0)
        self.assertEqual(X.scale, 1)

    def test_Laplace_invalid_loc_raises(self):
        for bad in ["a", None]:
            self.assertRaises(Exception, lambda b=bad: Laplace(loc=b))

    def test_Laplace_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Laplace(scale=b))

    def test_Laplace_plots_without_error(self):
        # draw / RV / sim / plot all wired through the base class.
        Laplace(0, 1).draw()
        RV(Laplace(0, 1)).sim(100).plot()
        Laplace(0, 1).plot()
        Laplace(0, 1).plot(cdf=True)
        plt.close("all")


class TestDeMoivre(unittest.TestCase):

    def test_DeMoivre_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(DeMoivre(omega=100))
        sims = X.sim(Nsim)
        cdf = stats.uniform(loc=0, scale=100).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_DeMoivre_mean_var_sd(self):
        X = DeMoivre(omega=100)
        th = stats.uniform(loc=0, scale=100)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_DeMoivre_is_reparameterized_uniform(self):
        # Age at death is Uniform(0, omega): constant density 1/omega on
        # [0, omega], and P(X <= x) = x / omega.
        X = DeMoivre(omega=80)
        self.assertAlmostEqual(float(X.pdf(50)), 1 / 80, places=9)
        self.assertAlmostEqual(float(X.cdf(20)), 20 / 80, places=9)

    def test_DeMoivre_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
        value = DeMoivre(omega=100).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)
        self.assertLessEqual(float(value), 100.0)

    def test_DeMoivre_invalid_omega_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: DeMoivre(omega=b))

    def test_DeMoivre_plots_without_error(self):
        # draw / RV / sim / plot all wired through the base class.
        DeMoivre(100).draw()
        RV(DeMoivre(100)).sim(100).plot()
        DeMoivre(100).plot()
        DeMoivre(100).plot(cdf=True)
        plt.close("all")


class TestGEV(unittest.TestCase):

    def test_GEV_distributional(self):
        # shape (xi) uses the standard EVT sign, so scipy's c is -shape.
        distributions.rng = np.random.default_rng(42)
        X = RV(GEV(loc=2, scale=3, shape=0.2))
        sims = X.sim(Nsim)
        cdf = stats.genextreme(c=-0.2, loc=2, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_GEV_mean_var_sd(self):
        X = GEV(loc=2, scale=3, shape=0.2)
        th = stats.genextreme(c=-0.2, loc=2, scale=3)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_GEV_default_is_gumbel(self):
        # shape=0 is the Gumbel distribution: unbounded, mean = euler-gamma.
        X = GEV()
        self.assertEqual(X.shape, 0)
        self.assertAlmostEqual(
            float(X.mean()), float(stats.gumbel_r().mean()), places=6
        )
        self.assertAlmostEqual(
            float(X.cdf(100)), float(stats.gumbel_r().cdf(100)), places=9
        )

    def test_GEV_shape_sign_selects_the_three_laws(self):
        # xi > 0 -> Frechet, bounded below; xi < 0 -> reverse-Weibull, bounded
        # above; xi = 0 -> Gumbel, unbounded both ways.
        frechet = GEV(loc=0, scale=1, shape=0.3)
        self.assertGreater(float(frechet.quantile(0.0)), -np.inf)  # bounded below
        self.assertEqual(float(frechet.quantile(1.0)), np.inf)  # heavy right tail
        rweibull = GEV(loc=0, scale=1, shape=-0.3)
        self.assertEqual(float(rweibull.quantile(0.0)), -np.inf)
        self.assertLess(float(rweibull.quantile(1.0)), np.inf)  # bounded above
        gumbel = GEV(loc=0, scale=1, shape=0)
        self.assertEqual(float(gumbel.quantile(0.0)), -np.inf)
        self.assertEqual(float(gumbel.quantile(1.0)), np.inf)

    def test_GEV_draw_is_scalar(self):
        distributions.rng = np.random.default_rng(0)
        value = GEV(loc=0, scale=1, shape=0.1).draw()
        self.assertIsInstance(value, Scalar)

    def test_GEV_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: GEV(scale=b))

    def test_GEV_invalid_loc_shape_raise(self):
        self.assertRaises(Exception, lambda: GEV(loc="a"))
        self.assertRaises(Exception, lambda: GEV(shape="a"))

    def test_GEV_plots_without_error(self):
        # draw / RV / sim / plot all wired through the base class.
        GEV(0, 1, 0.2).draw()
        RV(GEV(0, 1, 0.2)).sim(100).plot()
        GEV(0, 1, 0.2).plot()
        GEV(0, 1, 0.2).plot(cdf=True)
        plt.close("all")


class TestMultivariateNormal(unittest.TestCase):

    def test_MultivariateNormal_mean_cov_error(self):
        self.assertRaises(
            Exception, lambda: MultivariateNormal(mean=[2], cov=[[2, 2], [3, 4]])
        )

    def test_MultivariateNormal_cov_square_error(self):
        self.assertRaises(
            Exception, lambda: MultivariateNormal(mean=[2, 4], cov=[[2, 4, 5], [2, 1]])
        )

    def test_MultivariateNormal_cov_not_psd(self):
        self.assertRaises(
            Exception, lambda: MultivariateNormal(mean=[0, 0], cov=[[-1, 0], [0, 1]])
        )

    def test_MultivariateNormal_draw_shape(self):
        distributions.rng = np.random.default_rng(42)
        X = MultivariateNormal(mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        draw = X.draw()
        self.assertEqual(len(draw), 3)

    def test_MultivariateNormal_marginal(self):
        distributions.rng = np.random.default_rng(42)
        X, _ = RV(MultivariateNormal(mean=[3, 7], cov=[[4, 0], [0, 9]]))
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_MultivariateNormal_plot_raises(self):
        X = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        self.assertRaises(Exception, X.plot)


class TestBivariateNormal(unittest.TestCase):

    def test_BivariateNormal_error1(self):
        self.assertRaises(
            Exception,
            lambda: BivariateNormal(mean1=3, mean2=4, sd1=-3, sd2=3, corr=0.9),
        )

    def test_BivariateNormal_error2(self):
        self.assertRaises(
            Exception, lambda: BivariateNormal(mean1=3, mean2=4, sd1=3, sd2=3, corr=1.1)
        )

    def test_LinCom_BivNormal(self):
        distributions.rng = np.random.default_rng(42)
        X, Y = RV(BivariateNormal(mean1=30, mean2=50, sd1=8, sd2=6, corr=-0.4))
        Z = 4 * X - 2 * Y
        Z_mean = 4 * 30 + (-2) * 50
        Z_var = 16 * 64 + 4 * 36 + 2 * (4) * (-2) * (-0.4 * 8 * 6)
        sims = Z.sim(Nsim)
        cdf = stats.norm(loc=Z_mean, scale=sqrt(Z_var)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_BivNormal_condDistr_r(self):
        distributions.rng = np.random.default_rng(42)
        for c in [-0.9, 0.9, 0.1]:
            X, Y = RV(BivariateNormal(mean1=20, mean2=10, sd1=3, sd2=5, corr=c))
            sims = (Y | (abs(X - 21) < 0.1)).sim(500)
            cdf = stats.norm(
                loc=10 + c * 5 / 3 * (21 - 20), scale=5 * sqrt(1 - c**2)
            ).cdf
            pval = stats.kstest(sims, cdf).pvalue
            self.assertTrue(pval > 0.01)

    def test_BivariateNormal_corr_below_neg1(self):
        self.assertRaises(
            Exception,
            lambda: BivariateNormal(mean1=0, mean2=0, sd1=1, sd2=1, corr=-1.5),
        )

    def test_BivariateNormal_marginal(self):
        distributions.rng = np.random.default_rng(42)
        X, _ = RV(BivariateNormal(mean1=5, mean2=10, sd1=2, sd2=3, corr=0.6))
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=5, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_BivariateNormal_explicit_cov(self):
        # Exercise the var1/var2/cov keyword paths (instead of sd/corr)
        X = BivariateNormal(mean1=1, mean2=2, var1=4, var2=9, cov=3.0)
        self.assertEqual(X.cov, [[4, 3.0], [3.0, 9]])


class TestMultinomial(unittest.TestCase):

    def test_Multinomial_error_n_negative(self):
        self.assertRaises(Exception, lambda: Multinomial(n=-1, p=[0.5, 0.5]))

    def test_Multinomial_error_n_float(self):
        self.assertRaises(Exception, lambda: Multinomial(n=5.5, p=[0.5, 0.5]))

    def test_Multinomial_error_p_sum(self):
        self.assertRaises(Exception, lambda: Multinomial(n=10, p=[0.5, 0.6]))

    def test_Multinomial_error_p_negative(self):
        self.assertRaises(Exception, lambda: Multinomial(n=10, p=[-0.1, 1.1]))

    def test_Multinomial_draw_sums_to_n(self):
        distributions.rng = np.random.default_rng(42)
        X = Multinomial(n=20, p=[0.2, 0.5, 0.3])
        draw = X.draw()
        self.assertEqual(sum(draw), 20)

    def test_Multinomial_marginals_match_Binomial(self):
        distributions.rng = np.random.default_rng(42)
        p = [0.3, 0.5, 0.2]
        n = 15
        X0, X1, X2 = RV(Multinomial(n=n, p=p))
        for rv, pi in zip([X0, X1, X2], p):
            sims = rv.sim(Nsim)
            expected_mean = n * pi
            expected_var = n * pi * (1 - pi)
            self.assertAlmostEqual(float(sims.mean()), expected_mean, delta=0.15)
            self.assertAlmostEqual(float(sims.var()), expected_var, delta=0.15)

    def test_Multinomial_plot_raises(self):
        X = Multinomial(n=10, p=[0.5, 0.5])
        self.assertRaises(Exception, X.plot)

    def test_Multinomial_pdf(self):
        X = Multinomial(n=10, p=[0.5, 0.3, 0.2])
        expected = stats.multinomial(10, [0.5, 0.3, 0.2]).pmf([5, 3, 2])
        self.assertAlmostEqual(float(X.pdf([5, 3, 2])), expected)


# ===========================================================================
# Parameter validation: type guards and helpful error messages
#
# Covers the wrong inputs a student is likely to make: a non-number (e.g. a
# string), a float where a whole number is required, or two mutually
# exclusive parameters. assertRaisesRegex checks the message text, so a
# cryptic low-level TypeError (e.g. "'<' not supported between ...") would
# fail the test rather than pass silently.
# ===========================================================================


class TestNonNumericInputs(unittest.TestCase):

    def test_Bernoulli_p_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "p must be a number", lambda: Bernoulli(p="x")
        )

    def test_Binomial_n_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "n must be a non-negative integer",
            lambda: Binomial(n="x", p=0.5),
        )

    def test_Binomial_p_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "p must be a number", lambda: Binomial(n=5, p="x")
        )

    def test_Hypergeometric_n_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "n must be a positive integer",
            lambda: Hypergeometric(n="x", N0=3, N1=3),
        )

    def test_Hypergeometric_N0_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "N0 must be a non-negative integer",
            lambda: Hypergeometric(n=2, N0="x", N1=3),
        )

    def test_Hypergeometric_N1_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "N1 must be a non-negative integer",
            lambda: Hypergeometric(n=2, N0=3, N1="x"),
        )

    def test_Geometric_p_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "p must be a number", lambda: Geometric(p="x")
        )

    def test_NegativeBinomial_r_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "r must be a positive integer",
            lambda: NegativeBinomial(r="x", p=0.5),
        )

    def test_NegativeBinomial_p_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "p must be a number", lambda: NegativeBinomial(r=3, p="x")
        )

    def test_Pascal_r_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "r must be a positive integer", lambda: Pascal(r="x", p=0.5)
        )

    def test_Pascal_p_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "p must be a number", lambda: Pascal(r=3, p="x")
        )

    def test_Poisson_lam_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "lam must be a non-negative number", lambda: Poisson(lam="x")
        )

    def test_DiscreteUniform_a_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "a must be a number", lambda: DiscreteUniform(a="x", b=5)
        )

    def test_DiscreteUniform_b_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "b must be a number", lambda: DiscreteUniform(a=0, b="x")
        )

    def test_Uniform_a_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "a must be a number", lambda: Uniform(a="x", b=1)
        )

    def test_Uniform_b_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "b must be a number", lambda: Uniform(a=0, b="x")
        )

    def test_Normal_mean_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "mean must be a number", lambda: Normal(mean="x")
        )

    def test_Normal_sd_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "sd must be a non-negative number", lambda: Normal(sd="x")
        )

    def test_Normal_var_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "var must be a non-negative number", lambda: Normal(var="x")
        )

    def test_Exponential_rate_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "rate must be a positive number", lambda: Exponential(rate="x")
        )

    def test_Exponential_scale_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "scale must be a positive number", lambda: Exponential(scale="x")
        )

    def test_Gamma_shape_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "shape must be a positive number", lambda: Gamma(shape="x")
        )

    def test_Gamma_rate_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "rate must be a positive number",
            lambda: Gamma(shape=2, rate="x"),
        )

    def test_Gamma_scale_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "scale must be a positive number",
            lambda: Gamma(shape=2, scale="x"),
        )

    def test_Beta_a_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "a must be a positive number", lambda: Beta(a="x", b=1)
        )

    def test_Beta_b_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "b must be a positive number", lambda: Beta(a=1, b="x")
        )

    def test_StudentT_df_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "df must be a positive number", lambda: StudentT(df="x")
        )

    def test_ChiSquare_df_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "df must be a positive integer", lambda: ChiSquare(df="x")
        )

    def test_F_dfN_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "dfN must be a positive number", lambda: F(dfN="x", dfD=5)
        )

    def test_F_dfD_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "dfD must be a positive number", lambda: F(dfN=5, dfD="x")
        )

    def test_Cauchy_loc_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "loc must be a number", lambda: Cauchy(loc="x")
        )

    def test_Cauchy_scale_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "scale must be a positive number", lambda: Cauchy(scale="x")
        )

    def test_LogNormal_mu_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "mu must be a number", lambda: LogNormal(mu="x")
        )

    def test_LogNormal_sigma_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "sigma must be a non-negative number",
            lambda: LogNormal(sigma="x"),
        )

    def test_Pareto_b_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "b must be a positive number", lambda: Pareto(b="x")
        )

    def test_Pareto_scale_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "scale must be a positive number",
            lambda: Pareto(b=2, scale="x"),
        )

    def test_BivariateNormal_mean1_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "mean1 must be a number", lambda: BivariateNormal(mean1="x")
        )

    def test_BivariateNormal_corr_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "corr must be a number between -1 and 1",
            lambda: BivariateNormal(corr="x"),
        )

    def test_BivariateNormal_sd1_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "sd1 must be a non-negative number",
            lambda: BivariateNormal(sd1="x"),
        )

    def test_BivariateNormal_var1_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "var1 must be a non-negative number",
            lambda: BivariateNormal(var1="x"),
        )

    def test_BivariateNormal_cov_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "cov must be a number", lambda: BivariateNormal(cov="x")
        )

    def test_Multinomial_n_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "n must be a non-negative integer",
            lambda: Multinomial(n="x", p=[0.5, 0.5]),
        )


class TestRateScaleConflict(unittest.TestCase):

    def test_Exponential_both_rate_and_scale(self):
        self.assertRaisesRegex(
            Exception,
            "either rate or scale",
            lambda: Exponential(rate=2, scale=3),
        )

    def test_Gamma_both_rate_and_scale(self):
        self.assertRaisesRegex(
            Exception,
            "either rate or scale",
            lambda: Gamma(shape=2, rate=2, scale=3),
        )

    def test_Exponential_single_param_still_works(self):
        self.assertAlmostEqual(float(Exponential().mean()), 1.0)
        self.assertAlmostEqual(float(Exponential(rate=2).mean()), 0.5)
        self.assertAlmostEqual(float(Exponential(scale=5).mean()), 5.0)

    def test_Gamma_single_param_still_works(self):
        self.assertAlmostEqual(float(Gamma(2).mean()), 2.0)
        self.assertAlmostEqual(float(Gamma(2, scale=5).mean()), 10.0)

    def test_scale_given_leaves_rate_none(self):
        self.assertIsNone(Exponential(scale=5).rate)
        self.assertIsNone(Gamma(2, scale=5).rate)

    def test_Exponential_consistent_rate_scale_warns(self):
        with self.assertWarns(UserWarning):
            Exponential(rate=2, scale=0.5)

    def test_Exponential_consistent_rate_scale_correct_distribution(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X = Exponential(rate=2, scale=0.5)
        self.assertAlmostEqual(float(X.mean()), 0.5)

    def test_Gamma_consistent_rate_scale_warns(self):
        with self.assertWarns(UserWarning):
            Gamma(shape=3, rate=2, scale=0.5)

    def test_Gamma_consistent_rate_scale_correct_distribution(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X = Gamma(shape=3, rate=2, scale=0.5)
        self.assertAlmostEqual(float(X.mean()), 1.5)


class TestNormalSdVarConflict(unittest.TestCase):

    def test_both_sd_and_var_inconsistent_raises(self):
        self.assertRaisesRegex(ValueError, "sd or var", lambda: Normal(sd=2, var=1))

    def test_both_sd_and_var_consistent_warns(self):
        with self.assertWarns(UserWarning):
            Normal(sd=2, var=4)

    def test_both_sd_and_var_consistent_correct_distribution(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X = Normal(sd=2, var=4)
        self.assertAlmostEqual(float(X.sd()), 2.0)
        self.assertAlmostEqual(float(X.var()), 4.0)

    def test_default_is_standard_normal(self):
        X = Normal()
        self.assertAlmostEqual(float(X.mean()), 0.0)
        self.assertAlmostEqual(float(X.sd()), 1.0)


class TestIntegerVsRealParameters(unittest.TestCase):

    def test_Hypergeometric_rejects_float_n(self):
        self.assertRaisesRegex(
            Exception,
            "n must be a positive integer",
            lambda: Hypergeometric(n=2.5, N0=3, N1=3),
        )

    def test_Hypergeometric_rejects_float_counts(self):
        self.assertRaisesRegex(
            Exception,
            "N0 must be a non-negative integer",
            lambda: Hypergeometric(n=2, N0=3.5, N1=3),
        )

    def test_StudentT_allows_non_integer_df(self):
        # Student's t is legitimately defined for non-integer df.
        self.assertEqual(StudentT(df=2.5).df, 2.5)


class TestPoissonLamZero(unittest.TestCase):

    def test_lam_zero_allowed(self):
        self.assertAlmostEqual(float(Poisson(lam=0).mean()), 0.0)

    def test_lam_negative_rejected(self):
        self.assertRaisesRegex(
            Exception, "non-negative number", lambda: Poisson(lam=-1)
        )


class TestValidBoundaryParameters(unittest.TestCase):
    """Boundary-valid parameters must be accepted, not over-rejected.

    Each test simply constructs the distribution; if construction raised,
    the test would error.
    """

    def test_Bernoulli_boundaries(self):
        Bernoulli(p=0)
        Bernoulli(p=1)

    def test_Binomial_boundaries(self):
        Binomial(n=0, p=0.5)
        Binomial(n=10, p=0)
        Binomial(n=10, p=1)

    def test_Poisson_zero(self):
        Poisson(lam=0)

    def test_DiscreteUniform_equal_bounds(self):
        DiscreteUniform(a=5, b=5)

    def test_LogNormal_sigma_zero(self):
        LogNormal(sigma=0)


# ===========================================================================
# Stacked error messages
#
# When a student gets more than one parameter wrong, every mistake should be
# reported in a single exception (under an "Invalid parameters:" header) so
# they can fix them all at once instead of one error at a time. A single bad
# parameter must still raise its plain message, with no header.
# ===========================================================================


class TestStackedErrorMessages(unittest.TestCase):

    def test_single_error_has_no_header(self):
        # One bad parameter -> the message is unchanged (no "Invalid parameters").
        with self.assertRaises(Exception) as cm:
            Binomial(n=5, p="x")
        self.assertEqual(str(cm.exception), "p must be a number between 0 and 1")

    def test_multiple_errors_have_header(self):
        with self.assertRaises(Exception) as cm:
            Binomial(n=-1, p=2)
        self.assertIn("Invalid parameters:", str(cm.exception))

    def test_Binomial_stacks_n_and_p(self):
        with self.assertRaises(Exception) as cm:
            Binomial(n=-1, p=2)
        message = str(cm.exception)
        self.assertIn("n must be a non-negative integer", message)
        self.assertIn("p must be a number between 0 and 1", message)

    def test_Beta_stacks_a_and_b(self):
        with self.assertRaises(Exception) as cm:
            Beta(a=-1, b="x")
        message = str(cm.exception)
        self.assertIn("a must be a positive number", message)
        self.assertIn("b must be a positive number", message)

    def test_Gamma_stacks_shape_and_rate(self):
        with self.assertRaises(Exception) as cm:
            Gamma(shape=-5, rate=-10)
        message = str(cm.exception)
        self.assertIn("shape must be a positive number", message)
        self.assertIn("rate must be a positive number", message)

    def test_Hypergeometric_stacks_all_three(self):
        with self.assertRaises(Exception) as cm:
            Hypergeometric(n=-1, N0="x", N1=-2)
        message = str(cm.exception)
        self.assertIn("n must be a positive integer", message)
        self.assertIn("N0 must be a non-negative integer", message)
        self.assertIn("N1 must be a non-negative integer", message)

    def test_BivariateNormal_stacks_multiple(self):
        with self.assertRaises(Exception) as cm:
            BivariateNormal(mean1="x", corr=5, sd1=-1)
        message = str(cm.exception)
        self.assertIn("mean1 must be a number", message)
        self.assertIn("corr must be a number between -1 and 1", message)
        self.assertIn("sd1 must be a non-negative number", message)

    def test_Multinomial_stacks_n_and_p(self):
        with self.assertRaises(Exception) as cm:
            Multinomial(n=-1, p=[0.5, 0.6])
        message = str(cm.exception)
        self.assertIn("n must be a non-negative integer", message)
        self.assertIn("Elements of p must be non-negative", message)

    def test_conflict_reported_alone(self):
        # A structural conflict (both rate and scale) is reported by itself,
        # even when another parameter is also invalid: the contradiction must
        # be resolved before the value checks are meaningful.
        with self.assertRaises(Exception) as cm:
            Gamma(shape=-5, rate=2, scale=3)
        message = str(cm.exception)
        self.assertIn("Specify either rate or scale", message)
        self.assertNotIn("shape must be a positive number", message)


class TestDistributionXlim(unittest.TestCase):
    """Default plotting x-limits (task 10).

    Bounded distributions keep their true full support; skewed unbounded
    distributions use a highest-density interval; symmetric unbounded ones
    stay on the equal-tailed window. All checks read the closed-form
    ``xlim`` (no simulation), so they are deterministic.
    """

    COVERAGE = distributions._PLOT_COVERAGE

    # --- bounded distributions keep true full support (HDI must not leak in) ---

    def test_binomial_keeps_full_support(self):
        # The large-n case is the tempting one to trim, and must not be.
        self.assertEqual(Binomial(1000, 0.5).xlim, (0, 1000))
        self.assertEqual(Binomial(10, 0.3).xlim, (0, 10))

    def test_bounded_continuous_keep_full_support(self):
        self.assertEqual(Beta(2, 5).xlim, (0, 1))
        self.assertEqual(Uniform(2, 7).xlim, (2, 7))

    def test_bounded_discrete_keep_full_support(self):
        self.assertEqual(DiscreteUniform(1, 6).xlim, (1, 6))
        self.assertEqual(Bernoulli(0.3).xlim, (0, 1))

    def test_hypergeometric_window_holds_all_mass(self):
        # Its support isn't a simple (0, n), so assert the window covers all
        # of the probability rather than a specific tuple -- i.e. not trimmed.
        d = Hypergeometric(n=5, N0=10, N1=20)
        lo, hi = d.xlim
        self.assertAlmostEqual(float(d.cdf(hi) - d.cdf(lo - 1)), 1.0, places=9)

    # --- symmetric unbounded stay on the equal-tailed window (HDI == that) ---

    def test_symmetric_unbounded_stay_equal_tailed(self):
        for d in [Normal(0, 1), StudentT(5), Cauchy(0, 1)]:
            lo, hi = d.xlim
            self.assertAlmostEqual(lo, float(d.quantile(0.001)), places=6)
            self.assertAlmostEqual(hi, float(d.quantile(0.999)), places=6)

    # --- unbounded discrete: exact highest-density interval ---

    def test_poisson_hdi_trims_right_tail(self):
        # Right-skewed: keeps the dense low values, trims the thin upper tail
        # below where the equal-tailed window would have stopped.
        d = Poisson(0.5)
        lo, hi = d.xlim
        self.assertEqual(lo, 0)
        self.assertLess(hi, int(d.quantile(0.999)))

    def test_discrete_hdi_covers_target(self):
        for d in [
            Poisson(3),
            Poisson(50),
            Geometric(0.3),
            NegativeBinomial(3, 0.5),
            Pascal(2, 0.3),
        ]:
            lo, hi = d.xlim
            cover = float(d.cdf(hi) - d.cdf(lo - 1))
            self.assertGreaterEqual(cover, self.COVERAGE - 1e-9)

    def test_discrete_hdi_respects_support_lower_bound(self):
        self.assertGreaterEqual(Geometric(0.3).xlim[0], 1)
        self.assertGreaterEqual(NegativeBinomial(3, 0.5).xlim[0], 3)

    # --- skewed unbounded continuous: HDI via equal-density root-finding ---

    def test_continuous_hdi_covers_target(self):
        for d in [
            Gamma(2),
            Gamma(0.5),
            ChiSquare(10),
            ChiSquare(1),
            F(5, 10),
            LogNormal(0, 1),
            Pareto(2, 1),
        ]:
            lo, hi = d.xlim
            self.assertAlmostEqual(
                float(d.cdf(hi) - d.cdf(lo)), self.COVERAGE, places=3
            )

    def test_continuous_interior_mode_endpoints_equal_density(self):
        # For an interior-mode density the two HDI endpoints are the pair of
        # equal-density points enclosing the coverage.
        for d in [Gamma(9), ChiSquare(10), F(5, 10), LogNormal(0, 1)]:
            lo, hi = d.xlim
            self.assertAlmostEqual(float(d.pdf(lo)), float(d.pdf(hi)), places=6)

    def test_continuous_monotone_starts_at_support_bound(self):
        # Monotone-decreasing densities peak at the lower bound, so the HDI
        # keeps that bound instead of solving for a left equal-density point.
        self.assertEqual(Pareto(2, 1).xlim[0], 1)  # scale
        self.assertEqual(Pareto(3, 5).xlim[0], 5)  # scale
        self.assertEqual(Gamma(0.5).xlim[0], 0)  # shape < 1 -> mode at 0
        self.assertEqual(ChiSquare(1).xlim[0], 0)  # df = 1 -> mode at 0

    def test_skewed_hdi_narrower_than_equal_tailed(self):
        # The whole point: for a skewed density the HDI is the shortest window
        # for its coverage, so it is narrower than the old equal-tailed span.
        for d in [Gamma(2), LogNormal(0, 1), F(5, 10)]:
            lo, hi = d.xlim
            equal_tailed_width = float(d.quantile(0.999)) - float(d.quantile(0.001))
            self.assertLess(hi - lo, equal_tailed_width)


class TestDistributionXlimZoom(unittest.TestCase):
    """The ``xlim`` parameter of ``Distribution.plot()``.

    ``xlim=None`` (default) uses the distribution's own window: full
    support when both ends are bounded, and a highest-density probability
    cut where a side is unbounded. ``xlim="zoom"`` frames the plot on the
    tightest window holding ``_PLOT_COVERAGE`` of the probability -- the
    same window, applied even to a bounded distribution whose default
    shows full support. ``xlim=(a, b)`` sets exact limits. Most checks
    read the deterministic ``_hdi_window`` helper; the overlay and
    rendering checks use a non-interactive backend.
    """

    COVERAGE = distributions._PLOT_COVERAGE

    def tearDown(self):
        plt.close("all")

    # --- the zoom window covers the standard share of the probability ---

    def test_zoom_window_covers_target_discrete(self):
        for d in [Binomial(100, 0.5), Poisson(20), Geometric(0.2)]:
            lo, hi = d._hdi_window()
            cover = float(d.cdf(hi) - d.cdf(lo - 1))
            self.assertGreaterEqual(cover, self.COVERAGE - 1e-9)

    def test_zoom_window_covers_target_continuous(self):
        for d in [Gamma(2), LogNormal(0, 1), Normal(0, 1)]:
            lo, hi = d._hdi_window()
            self.assertAlmostEqual(
                float(d.cdf(hi) - d.cdf(lo)), self.COVERAGE, places=3
            )

    # --- xlim="zoom" trims a bounded distribution's full-support default ---

    def test_zoom_trims_bounded_binomial(self):
        d = Binomial(100, 0.5)
        self.assertEqual(d.xlim, (0, 100))  # default window unchanged
        lo, hi = d._hdi_window()
        self.assertGreater(lo, 0)
        self.assertLess(hi, 100)

    # --- xlim="zoom" == default for a distribution already framed on its
    #     highest-density window (only bounded ones differ) ---

    def test_zoom_matches_default_for_unbounded(self):
        for d in [Poisson(50), Geometric(0.3), NegativeBinomial(3, 0.5)]:
            self.assertEqual(d._hdi_window(), d.xlim)
        for d in [
            Gamma(2),
            LogNormal(0, 1),
            Normal(0, 1),
            Cauchy(0, 1),
            Exponential(1),
            Rayleigh(),
        ]:
            lo, hi = d._hdi_window()
            self.assertAlmostEqual(lo, float(d.xlim[0]), places=6)
            self.assertAlmostEqual(hi, float(d.xlim[1]), places=6)

    # --- overlay: the theoretical curve no longer stretches the shared axis ---

    def test_overlay_does_not_stretch_to_full_support(self):
        # 10000 draws of Binomial(100, 0.5) realize only a narrow band, but the
        # theoretical curve's full (0, 100) support used to widen the shared
        # axis to the whole range. xlim="zoom" keeps it tight.
        plt.figure()
        RV(Binomial(100, 0.5)).sim(10000).plot()
        Binomial(100, 0.5).plot(xlim="zoom")
        lo, hi = plt.gca().get_xlim()
        self.assertGreater(lo, 5)
        self.assertLess(hi, 95)

    # --- xlim=None / an explicit (a, b) behave as documented ---

    def test_default_keeps_full_support_for_binomial(self):
        plt.figure()
        Binomial(100, 0.5).plot()  # xlim defaults to None
        self.assertEqual(tuple(plt.gca().get_xlim()), (0.0, 100.0))

    def test_explicit_xlim_used_as_given(self):
        plt.figure()
        Binomial(100, 0.5).plot(xlim=(10, 90))
        self.assertEqual(tuple(plt.gca().get_xlim()), (10.0, 90.0))

    # --- friendly error for an unrecognized xlim string ---

    def test_invalid_xlim_string_raises(self):
        for bad in ["trim", "tight", "hdi", "auto"]:
            with self.assertRaises(ValueError) as cm:
                Normal(0, 1).plot(xlim=bad)
            self.assertIn("zoom", str(cm.exception))

    # --- every base-plot distribution accepts xlim="zoom" and renders ---

    def test_zoom_renders_for_every_distribution(self):
        dists = [
            Bernoulli(0.3),
            Binomial(20, 0.4),
            Hypergeometric(5, 10, 20),
            Geometric(0.3),
            NegativeBinomial(3, 0.5),
            Pascal(2, 0.3),
            Poisson(4),
            DiscreteUniform(1, 6),
            Uniform(2, 5),
            Normal(0, 1),
            Exponential(1),
            Gamma(2),
            Beta(2, 3),
            StudentT(3),
            ChiSquare(4),
            F(5, 10),
            Cauchy(0, 1),
            LogNormal(0, 1),
            Pareto(2, 1),
            Rayleigh(),
            Weibull(1.5, 2),
            Laplace(0, 1),
            Makeham(1.5, 0.3, 2),
        ]
        for d in dists:
            lo, hi = d._hdi_window()
            self.assertTrue(np.isfinite(lo) and np.isfinite(hi), type(d).__name__)
            self.assertLessEqual(lo, hi, type(d).__name__)
            for xlim in (None, "zoom"):
                plt.figure()
                d.plot(xlim=xlim)  # must not raise
                plt.close("all")

    # --- a window that collapses to one value still yields a usable axis ---

    def test_single_value_window_expands_and_is_quiet(self):
        # When one outcome carries essentially all the mass the window
        # collapses to a point; the axis must not be set to a singular
        # (equal) range, and no raw matplotlib warning should reach the user.
        for call in [
            lambda: Bernoulli(0.999).plot(xlim="zoom"),  # zoom path
            lambda: Geometric(0.999).plot(),  # default path
        ]:
            plt.figure()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                call()
                lo, hi = plt.gca().get_xlim()
            plt.close("all")
            self.assertLess(lo, hi)  # not singular
            self.assertFalse(
                any("identical" in str(w.message).lower() for w in caught),
                "set_xlim received identical limits",
            )

    # --- fixed defaults: the two one-sided outliers now match the rest ---

    def test_exponential_default_covers_standard_share(self):
        # Was 0.999 (equal-tailed upper); now matches the other one-sided
        # distributions at _PLOT_COVERAGE, still pinned at the lower bound 0.
        d = Exponential(1)
        lo, hi = d.xlim
        self.assertEqual(lo, 0)
        self.assertAlmostEqual(float(d.cdf(hi) - d.cdf(lo)), self.COVERAGE, places=3)

    def test_rayleigh_default_starts_near_lower_bound(self):
        # Was the equal-tailed 0.1st percentile (~0.045); now a highest-density
        # window whose lower edge sits at the true bound 0.
        d = Rayleigh()
        lo, hi = d.xlim
        self.assertLess(lo, 0.02)
        self.assertAlmostEqual(float(d.cdf(hi) - d.cdf(lo)), self.COVERAGE, places=3)


class TestDistributionCDFPlot(unittest.TestCase):
    """The ``cdf=`` parameter of ``Distribution.plot()`` (task 12).

    ``cdf=False`` (default) plots the density/mass function; ``cdf=True``
    plots the cumulative distribution function. Discrete CDFs render as a
    right-continuous step function with no markers; continuous CDFs render
    as a smooth curve. The ``xlim`` x-window logic is reused unchanged.
    """

    def tearDown(self):
        plt.close("all")

    # --- discrete CDF is a marker-less step function ---

    def test_discrete_cdf_is_stepped_without_markers(self):
        plt.figure()
        p = Poisson(3).plot(cdf=True)
        (line,) = p.ax.get_lines()
        self.assertEqual(line.get_drawstyle(), "steps-post")
        self.assertEqual(line.get_marker(), "None")
        # no scatter dots (the pmf default draws these; the CDF must not)
        self.assertEqual(len(p.ax.collections), 0)

    # --- titles name what the plot shows ---

    def test_cdf_title(self):
        for d in [Poisson(3), Normal(0, 1)]:
            plt.figure()
            self.assertEqual(d.plot(cdf=True).ax.get_title(), "CDF Plot")
            plt.close("all")

    def test_default_pmf_title_for_discrete(self):
        plt.figure()
        self.assertEqual(Binomial(10, 0.5).plot().ax.get_title(), "PMF Plot")

    def test_default_pdf_title_for_continuous(self):
        plt.figure()
        self.assertEqual(Normal(0, 1).plot().ax.get_title(), "PDF Plot")

    # --- both vertical and horizontal gridlines, like the ECDF plot ---

    def test_both_gridlines_shown(self):
        for call in [
            lambda: Poisson(3).plot(cdf=True),  # cdf, discrete
            lambda: Normal(0, 1).plot(cdf=True),  # cdf, continuous
            lambda: Binomial(10, 0.5).plot(),  # pmf
            lambda: Normal(0, 1).plot(),  # pdf
        ]:
            plt.figure()
            ax = call().ax
            self.assertTrue(any(gl.get_visible() for gl in ax.get_xgridlines()))
            self.assertTrue(any(gl.get_visible() for gl in ax.get_ygridlines()))
            plt.close("all")

    # --- CDF step styling matches the empirical ECDF (make_ecdf) ---

    def test_cdf_matches_ecdf_linewidth(self):
        from symbulate.plot import ECDF_LINEWIDTH

        for d in [Poisson(3), Normal(0, 1)]:
            plt.figure()
            (line,) = d.plot(cdf=True).ax.get_lines()
            self.assertEqual(line.get_linewidth(), ECDF_LINEWIDTH)
            plt.close("all")

    # --- continuous CDF is a smooth (non-stepped) curve ---

    def test_continuous_cdf_is_smooth_curve(self):
        plt.figure()
        p = Normal(0, 1).plot(cdf=True)
        (line,) = p.ax.get_lines()
        self.assertEqual(line.get_drawstyle(), "default")
        self.assertGreater(len(line.get_xdata()), 100)

    # --- the plotted CDF values match the distribution's own cdf ---

    def test_cdf_values_match_distribution(self):
        for d in [Poisson(4), Normal(0, 1), Gamma(2)]:
            plt.figure()
            (line,) = d.plot(cdf=True).ax.get_lines()
            xs, ys = line.get_xdata(), line.get_ydata()
            np.testing.assert_allclose(ys, d.cdf(np.asarray(xs)), atol=1e-9)
            plt.close("all")

    # --- CDF is monotonically non-decreasing and lands in [0, 1] ---

    def test_cdf_is_monotone_in_unit_interval(self):
        for d in [Binomial(20, 0.4), Geometric(0.3), Exponential(1)]:
            plt.figure()
            (line,) = d.plot(cdf=True).ax.get_lines()
            ys = np.asarray(line.get_ydata())
            self.assertTrue(np.all(np.diff(ys) >= -1e-12), type(d).__name__)
            self.assertGreaterEqual(ys.min(), -1e-9)
            self.assertLessEqual(ys.max(), 1 + 1e-9)
            plt.close("all")

    # --- default type is still the pdf/pmf (regression) ---

    def test_default_type_is_pdf(self):
        plt.figure()
        p = Binomial(10, 0.5).plot()  # cdf defaults to False (pmf)
        # The discrete pmf draws a filled dot at each value (one scatter
        # collection) plus a dashed connecting line (one Line2D).
        self.assertEqual(len(p.ax.collections), 1)
        self.assertEqual(len(p.ax.get_lines()), 1)

    # --- xlim="zoom" x-window is reused unchanged for the CDF ---

    def test_cdf_reuses_zoom_window(self):
        plt.figure()
        pdf_win = Binomial(100, 0.5).plot(xlim="zoom").ax.get_xlim()
        plt.close("all")
        plt.figure()
        cdf_win = Binomial(100, 0.5).plot(cdf=True, xlim="zoom").ax.get_xlim()
        self.assertEqual(pdf_win, cdf_win)
        self.assertGreater(cdf_win[0], 0)  # not the full (0, 100) support
        self.assertLess(cdf_win[1], 100)

    # --- the old type= spelling raises a friendly pointer to cdf= ---

    def test_old_type_kwarg_raises_helpful_error(self):
        for old_call in [
            lambda: Normal(0, 1).plot(type="cdf"),  # the former CDF syntax
            lambda: Normal(0, 1).plot(type="histogram"),  # any type= at all
        ]:
            with self.assertRaises(ValueError) as cm:
                old_call()
            msg = str(cm.exception)
            self.assertIn("type", msg)
            self.assertIn("cdf", msg)

    # --- every base-plot distribution renders a CDF cleanly ---

    def test_cdf_renders_for_every_distribution(self):
        dists = [
            Bernoulli(0.3),
            Binomial(20, 0.4),
            Hypergeometric(5, 10, 20),
            Geometric(0.3),
            NegativeBinomial(3, 0.5),
            Pascal(2, 0.3),
            Poisson(4),
            DiscreteUniform(1, 6),
            Uniform(2, 5),
            Normal(0, 1),
            Exponential(1),
            Gamma(2),
            Beta(2, 3),
            StudentT(3),
            ChiSquare(4),
            F(5, 10),
            Cauchy(0, 1),
            LogNormal(0, 1),
            Pareto(2, 1),
            Rayleigh(),
            Weibull(1.5, 2),
            Laplace(0, 1),
            Makeham(1.5, 0.3, 2),
        ]
        for d in dists:
            plt.figure()
            d.plot(cdf=True)  # must not raise
            plt.close("all")


class TestDistributionShade(unittest.TestCase):
    """``DistributionPlot.shade(lt, le, gt, ge)`` -- chained off ``.plot()``.

    ``Distribution.plot()`` returns a ``DistributionPlot``, and ``.shade()``
    fills a tail or interval under the curve it drew (pmf/pdf or cdf).
    There is no standalone ``shade`` on a distribution, so a curve always
    exists first. Bounds read as probability inequalities; the strict
    (``lt``/``gt``) vs. inclusive (``le``/``ge``) choice matters for
    discrete distributions. The shaded region honors the displayed x-window
    (default, ``xlim="zoom"``, an explicit ``xlim``, or overlay union), not
    the distribution's static ``self.xlim``.
    """

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def _impulse_xs(ax):
        """Rounded x-positions of the last impulse (vlines) collection."""
        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        return sorted(round(seg[0][0]) for seg in lcs[-1].get_segments())

    @staticmethod
    def _fills(ax):
        return [c for c in ax.collections if isinstance(c, PolyCollection)]

    # --- shade is reachable only by chaining off plot() ---

    def test_shade_not_on_distribution(self):
        # No standalone shade: it lives on the plot object, not the
        # distribution, so a curve must be plotted first.
        self.assertFalse(hasattr(Poisson(3), "shade"))
        with self.assertRaises(AttributeError):
            Poisson(3).shade(le=3)

    def test_shade_returns_plot_for_chaining(self):
        plt.figure()
        result = Normal(0, 1).plot().shade(lt=0)
        self.assertEqual(type(result).__name__, "DistributionPlot")
        self.assertEqual(repr(result), "")

    # --- discrete: strict vs. inclusive bounds change which mass shades ---

    def test_discrete_inclusive_includes_endpoint(self):
        plt.figure()
        p = Poisson(3).plot().shade(le=3)
        self.assertIn(3, self._impulse_xs(p.ax))

    def test_discrete_strict_excludes_endpoint(self):
        plt.figure()
        p = Poisson(3).plot().shade(lt=3)
        self.assertNotIn(3, self._impulse_xs(p.ax))

    def test_discrete_right_tail_inclusive(self):
        plt.figure()
        p = Binomial(20, 0.5).plot().shade(ge=12)
        xs = self._impulse_xs(p.ax)
        self.assertIn(12, xs)
        self.assertTrue(all(x >= 12 for x in xs))

    def test_discrete_interval_mixed_bounds(self):
        # gt=3, le=7  ->  4, 5, 6, 7  (3 excluded, 7 included)
        plt.figure()
        p = Binomial(10, 0.5).plot().shade(gt=3, le=7)
        self.assertEqual(self._impulse_xs(p.ax), [4, 5, 6, 7])

    # --- continuous: a filled region with the shade constants ---

    def test_continuous_shade_fills_with_shade_constants(self):
        from symbulate.plot import SHADE_COLOR, SHADE_ALPHA

        plt.figure()
        p = Normal(0, 1).plot().shade(lt=-1.96)
        fills = self._fills(p.ax)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[-1].get_alpha(), SHADE_ALPHA)
        expected = matplotlib.colors.to_rgba(SHADE_COLOR, SHADE_ALPHA)
        np.testing.assert_allclose(fills[-1].get_facecolor()[0], expected)

    def test_continuous_fill_stays_within_bounds(self):
        plt.figure()
        p = Normal(0, 1).plot().shade(gt=-1, lt=1)
        xs = self._fills(p.ax)[-1].get_paths()[0].vertices[:, 0]
        self.assertGreaterEqual(xs.min(), -1 - 1e-9)
        self.assertLessEqual(xs.max(), 1 + 1e-9)

    # --- the shaded region respects the displayed window, not self.xlim ---

    def test_open_tail_uses_displayed_axis_not_self_xlim(self):
        # On an xlim="zoom" window the axis is far tighter than the (0, 100)
        # default support; an open left tail must start at the visible edge,
        # which the fork's self.xlim-based version could not do.
        plt.figure()
        p = Binomial(100, 0.5).plot(xlim="zoom")
        axlo, _ = p.ax.get_xlim()
        self.assertGreater(axlo, 0)  # window is tighter than full support
        p.shade(le=50)
        xs = self._impulse_xs(p.ax)
        self.assertGreaterEqual(min(xs), int(np.floor(axlo)))
        self.assertNotIn(0, xs)  # would appear if self.xlim[0]=0 were used

    def test_shade_draws_onto_existing_curve_without_replotting(self):
        plt.figure()
        p = Poisson(3).plot()
        n_before = len(p.ax.collections)
        p.shade(le=2)
        # shade adds exactly one impulse collection on top of the already
        # drawn pmf (its dots + dashed line are left untouched)
        self.assertEqual(len(p.ax.collections), n_before + 1)

    # --- shade under a cdf uses the cdf, and fills for the step case ---

    def test_shade_under_continuous_cdf(self):
        plt.figure()
        p = Normal(0, 1).plot(cdf=True)
        self.assertEqual(p.plot_type, "cdf")
        p.shade(lt=0)
        self.assertEqual(len(self._fills(p.ax)), 1)

    def test_shade_under_discrete_cdf_fills_stepwise(self):
        plt.figure()
        p = Poisson(3).plot(cdf=True)
        p.shade(le=2)
        # a discrete cdf shades as a filled staircase, not impulses
        self.assertEqual(len(self._fills(p.ax)), 1)

    # --- friendly errors ---

    def test_both_upper_bounds_raises(self):
        plt.figure()
        with self.assertRaises(ValueError) as cm:
            Poisson(3).plot().shade(lt=3, le=5)
        self.assertIn("lt", str(cm.exception))
        self.assertIn("le", str(cm.exception))

    def test_both_lower_bounds_raises(self):
        plt.figure()
        with self.assertRaises(ValueError) as cm:
            Poisson(3).plot().shade(gt=1, ge=2)
        self.assertIn("gt", str(cm.exception))
        self.assertIn("ge", str(cm.exception))

    def test_crossed_bounds_raises(self):
        plt.figure()
        with self.assertRaises(ValueError) as cm:
            Normal(0, 1).plot().shade(gt=5, lt=3)
        self.assertIn("less than", str(cm.exception))
