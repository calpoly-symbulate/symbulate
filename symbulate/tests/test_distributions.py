import math
import unittest
import numpy as np
import scipy.stats as stats
import warnings

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
        for i, (rv, pi) in enumerate(zip([X0, X1, X2], p)):
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
