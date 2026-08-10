import inspect
import math
import re
import time
import unittest
import numpy as np
import scipy.stats as stats
import scipy.special as special
import warnings

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from symbulate import *
from symbulate import distributions

# Benford is deliberately not exported from symbulate/__init__.py yet -- the
# public-API addition is awaiting team sign-off -- so it is imported from its
# module directly rather than coming in through the star import above.
from symbulate.distributions import Benford
from symbulate.plot import THEORETICAL_CDF_PDF_OVERLAY_ERROR

# InverseGaussian is likewise not exported yet -- awaiting team sign-off on the
# public-API addition -- so it too is imported straight from its module.
from symbulate.distributions import InverseGaussian

# The sentinel RV.sim() compares `func` against to decide whether anything has
# been composed on top of the distribution (see TestFastSim). Private, so it
# comes from its module rather than the public API.
from symbulate.random_variables import _IDENTITY
from symbulate.result import Scalar
from symbulate.results import RVResults

Nsim = 10000


class TestBernoulli(unittest.TestCase):

    def test_p_one(self):
        seed(42)
        X = RV(Bernoulli(p=1))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 1 for sim in sims))

    def test_sum(self):
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
        for nsample in range(1, 1000, 100):
            X = RV(Binomial(n=nsample, p=1.0))
            sims = X.sim(Nsim)
        self.assertTrue(all(sim == nsample for sim in sims))

    def test_Binomial_error_n(self):
        self.assertRaises(Exception, lambda: Binomial(n=-10, p=0.4))

    def test_Binomial_additive(self):
        seed(42)
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


class TestBetaBinomial(unittest.TestCase):

    def test_BetaBinomial_distributional(self):
        seed(42)
        exp_list, obs_list = [], []
        X = RV(BetaBinomial(n=10, shape1=2, shape2=3))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(11):
            expected = Nsim * stats.betabinom(10, 2, 3).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_BetaBinomial_mean_var_sd(self):
        X = BetaBinomial(n=10, shape1=2, shape2=3)
        th = stats.betabinom(10, 2, 3)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_BetaBinomial_pmf(self):
        X = BetaBinomial(n=10, shape1=2, shape2=3)
        for k in [0, 3, 7, 10]:
            self.assertAlmostEqual(
                float(X.pmf(k)), float(stats.betabinom(10, 2, 3).pmf(k))
            )

    def test_BetaBinomial_uniform_when_shapes_equal_one(self):
        # shape1 = shape2 = 1 makes the shared probability Uniform(0, 1), so
        # every count from 0 to n is equally likely -- a DiscreteUniform.
        X = BetaBinomial(n=10, shape1=1, shape2=1)
        for k in range(11):
            self.assertAlmostEqual(float(X.pmf(k)), 1 / 11, places=9)

    def test_BetaBinomial_n1_is_bernoulli(self):
        # With a single trial the shared probability integrates out to its
        # mean shape1 / (shape1 + shape2), so BetaBinomial(1, shape1, shape2)
        # is Bernoulli(shape1 / (shape1 + shape2)).
        X = BetaBinomial(n=1, shape1=2, shape2=3)
        self.assertAlmostEqual(float(X.pmf(1)), 2 / 5, places=9)
        self.assertAlmostEqual(float(X.pmf(0)), 3 / 5, places=9)

    def test_BetaBinomial_approaches_binomial(self):
        # As the shapes grow with shape1 / (shape1 + shape2) fixed, the beta
        # concentrates on a single probability and the beta-binomial
        # approaches Binomial(n, p).
        X = BetaBinomial(n=10, shape1=300, shape2=700)
        binom = stats.binom(n=10, p=0.3)
        for k in range(11):
            self.assertAlmostEqual(float(X.pmf(k)), float(binom.pmf(k)), places=2)

    def test_BetaBinomial_draw_is_scalar_in_support(self):
        seed(0)
        value = BetaBinomial(n=10, shape1=2, shape2=3).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)
        self.assertLessEqual(float(value), 10.0)

    def test_BetaBinomial_error_n_negative(self):
        self.assertRaises(Exception, lambda: BetaBinomial(n=-1, shape1=2, shape2=3))

    def test_BetaBinomial_error_n_float(self):
        self.assertRaises(Exception, lambda: BetaBinomial(n=2.5, shape1=2, shape2=3))

    def test_BetaBinomial_error_shape1_not_positive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: BetaBinomial(n=10, shape1=v, shape2=3)
            )

    def test_BetaBinomial_error_shape2_not_positive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: BetaBinomial(n=10, shape1=2, shape2=v)
            )

    def test_BetaBinomial_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        BetaBinomial(n=10, shape1=2, shape2=3).draw()
        RV(BetaBinomial(n=10, shape1=2, shape2=3)).sim(100).plot()
        BetaBinomial(n=10, shape1=2, shape2=3).plot()
        plt.figure()
        BetaBinomial(n=10, shape1=2, shape2=3).plot(cdf=True)
        plt.close("all")


class TestBetaNegativeBinomial(unittest.TestCase):

    def test_BetaNegativeBinomial_distributional(self):
        seed(42)
        X = RV(BetaNegativeBinomial(r=5, shape1=4, shape2=3))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        exp_list, obs_list = [], []
        for k in range(60):
            expected = Nsim * stats.betanbinom(5, 4, 3).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_BetaNegativeBinomial_mean_var_sd(self):
        X = BetaNegativeBinomial(r=5, shape1=4, shape2=3)
        th = stats.betanbinom(5, 4, 3)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_BetaNegativeBinomial_pmf(self):
        X = BetaNegativeBinomial(r=5, shape1=3, shape2=2)
        for k in [0, 1, 5, 12]:
            self.assertAlmostEqual(
                float(X.pmf(k)), float(stats.betanbinom(5, 3, 2).pmf(k))
            )

    def test_BetaNegativeBinomial_approaches_pascal(self):
        # As the shapes grow with shape1 / (shape1 + shape2) fixed, the beta
        # concentrates on a single probability and the beta-negative binomial
        # approaches a Pascal(r, p) (failures before the r-th success).
        X = BetaNegativeBinomial(r=5, shape1=600, shape2=400)  # ratio = 0.6
        pascal = Pascal(r=5, p=0.6)
        for k in range(15):
            self.assertAlmostEqual(float(X.pmf(k)), float(pascal.pmf(k)), places=2)

    def test_BetaNegativeBinomial_draw_is_scalar_in_support(self):
        seed(0)
        value = BetaNegativeBinomial(r=5, shape1=3, shape2=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_BetaNegativeBinomial_error_r(self):
        for bad in [0, -1, 2.5, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: BetaNegativeBinomial(r=v, shape1=3, shape2=2)
            )

    def test_BetaNegativeBinomial_error_shape1_not_positive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: BetaNegativeBinomial(r=5, shape1=v, shape2=2)
            )

    def test_BetaNegativeBinomial_error_shape2_not_positive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: BetaNegativeBinomial(r=5, shape1=3, shape2=v)
            )

    def test_BetaNegativeBinomial_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        BetaNegativeBinomial(r=5, shape1=3, shape2=2).draw()
        RV(BetaNegativeBinomial(r=5, shape1=3, shape2=2)).sim(100).plot()
        BetaNegativeBinomial(r=5, shape1=3, shape2=2).plot()
        plt.figure()
        BetaNegativeBinomial(r=5, shape1=3, shape2=2).plot(cdf=True)
        plt.close("all")


class TestHypergeometric(unittest.TestCase):

    def test_Hypergeometric_no_failures(self):
        seed(42)
        X = RV(Hypergeometric(n=10, N0=0, N1=1000))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))

    def test_Hypergeometric_Binomial_converge(self):
        seed(42)
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


class TestNegativeHypergeometric(unittest.TestCase):

    def test_NegativeHypergeometric_distributional(self):
        seed(42)
        X = RV(NegativeHypergeometric(r=3, N0=7, N1=5))
        sims = X.sim(Nsim)
        expected = stats.nhypergeom(M=12, n=5, r=3)
        counts = np.bincount(np.array(list(sims), dtype=int), minlength=6)[:6]
        pval = stats.chisquare(counts, len(sims) * expected.pmf(np.arange(6))).pvalue
        self.assertTrue(pval > 0.01)

    def test_NegativeHypergeometric_matches_urn_simulation(self):
        # Draw without replacement from an urn of N0 zeros and N1 ones until
        # r zeros appear, counting the ones. This is the definition, checked
        # independently of scipy's parameterization.
        r, N0, N1 = 3, 7, 5
        rng = np.random.default_rng(0)
        urn = np.array([1] * N1 + [0] * N0)
        outcomes = []
        for _ in range(40000):
            rng.shuffle(urn)
            zeros = ones = 0
            for ball in urn:
                if ball == 0:
                    zeros += 1
                    if zeros == r:
                        break
                else:
                    ones += 1
            outcomes.append(ones)
        empirical = np.bincount(outcomes, minlength=N1 + 1) / len(outcomes)
        X = NegativeHypergeometric(r=r, N0=N0, N1=N1)
        theoretical = np.array([float(X.pmf(k)) for k in range(N1 + 1)])
        self.assertTrue(np.max(np.abs(empirical - theoretical)) < 0.01)

    def test_NegativeHypergeometric_mean_closed_form(self):
        # mean = r * N1 / (N0 + 1)
        for r, N0, N1 in [(3, 7, 5), (1, 5, 4), (2, 10, 8), (7, 7, 5)]:
            X = NegativeHypergeometric(r=r, N0=N0, N1=N1)
            self.assertAlmostEqual(float(X.mean()), r * N1 / (N0 + 1), places=8)

    def test_NegativeHypergeometric_pmf_sums_to_one(self):
        X = NegativeHypergeometric(r=3, N0=7, N1=5)
        total = sum(float(X.pmf(k)) for k in range(6))
        self.assertAlmostEqual(total, 1.0, places=8)

    def test_NegativeHypergeometric_pmf_matches_scipy(self):
        X = NegativeHypergeometric(r=3, N0=7, N1=5)
        th = stats.nhypergeom(M=12, n=5, r=3)
        for k in range(6):
            self.assertAlmostEqual(float(X.pmf(k)), float(th.pmf(k)), places=10)

    def test_NegativeHypergeometric_support_is_zero_to_N1(self):
        X = NegativeHypergeometric(r=3, N0=7, N1=5)
        self.assertEqual(X.xlim, (0, 5))
        self.assertGreater(float(X.pmf(5)), 0.0)
        self.assertAlmostEqual(float(X.pmf(6)), 0.0, places=12)

    def test_NegativeHypergeometric_r_equals_one(self):
        # Stopping at the first zero: mean = N1 / (N0 + 1).
        X = NegativeHypergeometric(r=1, N0=7, N1=5)
        self.assertAlmostEqual(float(X.mean()), 5 / 8, places=8)

    def test_NegativeHypergeometric_no_successes_is_point_mass_at_zero(self):
        X = NegativeHypergeometric(r=3, N0=7, N1=0)
        self.assertAlmostEqual(float(X.mean()), 0.0, places=10)
        self.assertAlmostEqual(float(X.pmf(0)), 1.0, places=10)

    def test_NegativeHypergeometric_draws_within_support(self):
        seed(0)
        sims = RV(NegativeHypergeometric(r=3, N0=7, N1=5)).sim(1000)
        self.assertTrue(all(0 <= int(v) <= 5 for v in sims))

    def test_NegativeHypergeometric_r_greater_than_N0_raises(self):
        # scipy silently returns nan here, so this must be caught up front.
        self.assertRaises(Exception, lambda: NegativeHypergeometric(r=8, N0=7, N1=5))

    def test_NegativeHypergeometric_invalid_r_raises(self):
        for bad in [0, -1, 2.5, "a"]:
            self.assertRaises(
                Exception, lambda b=bad: NegativeHypergeometric(r=b, N0=7, N1=5)
            )

    def test_NegativeHypergeometric_invalid_N0_raises(self):
        for bad in [0, -1, 2.5, "a"]:
            self.assertRaises(
                Exception, lambda b=bad: NegativeHypergeometric(r=1, N0=b, N1=5)
            )

    def test_NegativeHypergeometric_invalid_N1_raises(self):
        for bad in [-1, 2.5, "a"]:
            self.assertRaises(
                Exception, lambda b=bad: NegativeHypergeometric(r=3, N0=7, N1=b)
            )

    def test_NegativeHypergeometric_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        NegativeHypergeometric(r=3, N0=7, N1=5).draw()
        RV(NegativeHypergeometric(r=3, N0=7, N1=5)).sim(100).plot()
        NegativeHypergeometric(r=3, N0=7, N1=5).plot()
        plt.figure()
        NegativeHypergeometric(r=3, N0=7, N1=5).plot(cdf=True)
        plt.close("all")


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
        seed(42)
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
        seed(42)
        X = NegativeBinomial(r=10, p=1)
        sims = X.sim(Nsim)
        self.assertTrue(all(sim == 10 for sim in sims))

    def test_NBinom_Pascal_additive(self):
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
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
        seed(42)
        X = RV(DiscreteUniform(a=3, b=7))
        sims = X.sim(Nsim)
        self.assertTrue(all(3 <= sim <= 7 for sim in sims))

    def test_DiscreteUniform_degenerate(self):
        X = DiscreteUniform(3, 3)

        sims = X.sim(100)

        for value in sims:
            self.assertEqual(value, 3)


class TestZipf(unittest.TestCase):
    """The finite-support Zipf distribution on the ranks 1, ..., n."""

    # --- expected probabilities ---

    def test_Zipf_pmf_is_normalized_power_law(self):
        # pmf(k) = (1 / k ** a) / H(n, a), computed here from the definition
        # rather than from scipy, so the test would catch a wrong
        # normalizing constant.
        a, n = 1.5, 8
        harmonic = math.fsum(1 / k**a for k in range(1, n + 1))
        X = Zipf(shape=a, n=n)
        for k in range(1, n + 1):
            self.assertAlmostEqual(float(X.pmf(k)), (1 / k**a) / harmonic, places=12)

    def test_Zipf_pmf_sums_to_one(self):
        X = Zipf(shape=1, n=5)
        self.assertAlmostEqual(float(np.sum(X.pmf(np.arange(1, 6)))), 1.0, places=12)

    def test_Zipf_classic_shape_one_probabilities(self):
        # a = 1 is Zipf's law itself: rank k has probability proportional to
        # 1 / k, normalized by the 5th harmonic number 137 / 60.
        X = Zipf(shape=1, n=5)
        for k in range(1, 6):
            self.assertAlmostEqual(float(X.pmf(k)), (1 / k) / (137 / 60), places=12)

    def test_Zipf_mean_var_sd(self):
        X = Zipf(shape=1.2, n=10)
        th = stats.zipfian(1.2, 10)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    # --- support ---

    def test_Zipf_support_is_one_through_n(self):
        # Finite support: nothing below 1, nothing above n.
        X = Zipf(shape=1.3, n=6)
        self.assertEqual(float(X.pmf(0)), 0.0)
        self.assertEqual(float(X.pmf(7)), 0.0)
        self.assertAlmostEqual(float(X.cdf(6)), 1.0, places=12)
        self.assertEqual(float(X.quantile(1.0)), 6.0)

    def test_Zipf_xlim_is_full_support(self):
        # Bounded at both ends, so the default plotting window is the whole
        # support -- no probability trimmed, like Binomial and DiscreteUniform.
        self.assertEqual(Zipf(shape=1.1, n=50).xlim, (1, 50))

    def test_Zipf_pmf_is_decreasing_in_rank(self):
        # Zipf's law: rank 1 is the most common and probability falls off
        # from there.
        probs = Zipf(shape=1.1, n=20).pmf(np.arange(1, 21))
        self.assertTrue(all(np.diff(probs) < 0))

    # --- wraps zipfian (finite), not zipf (the infinite-support zeta) ---

    def test_Zipf_wraps_scipy_zipfian(self):
        X = Zipf(shape=2, n=10)
        for k in [1, 2, 5, 10]:
            self.assertAlmostEqual(
                float(X.pmf(k)), float(stats.zipfian.pmf(k, 2, 10)), places=12
            )

    def test_Zipf_is_not_the_zeta_distribution(self):
        # Regression guard for scipy's naming trap: scipy.stats.zipf is the
        # infinite-support zeta distribution, NOT this one. The finite
        # normalizing constant makes every probability strictly larger.
        X = Zipf(shape=2, n=10)
        for k in [1, 2, 5, 10]:
            self.assertGreater(float(X.pmf(k)), float(stats.zipf.pmf(k, 2)))
        # The zeta distribution puts mass above n; the finite Zipf does not.
        self.assertGreater(float(stats.zipf.pmf(11, 2)), 0.0)
        self.assertEqual(float(X.pmf(11)), 0.0)

    def test_Zipf_defined_for_shape_below_one(self):
        # The infinite sum diverges for a <= 1, so the zeta distribution does
        # not exist there -- but a finite sum always converges, so the
        # finite-support Zipf does.
        for a in [0, 0.5, 1]:
            probs = Zipf(shape=a, n=6).pmf(np.arange(1, 7))
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=12)

    def test_Zipf_large_n_approaches_zeta(self):
        # With a > 1 fixed, growing n approaches the infinite-support zeta,
        # because the ranks past n carry almost no probability.
        X = Zipf(shape=3, n=10000)
        for k in [1, 2, 5]:
            self.assertAlmostEqual(
                float(X.pmf(k)), float(stats.zipf.pmf(k, 3)), places=6
            )

    # --- relationships with distributions already in Symbulate ---

    def test_Zipf_shape_zero_is_discrete_uniform(self):
        # a = 0 makes every rank equally likely: DiscreteUniform(1, n).
        X = Zipf(shape=0, n=5)
        for k in range(1, 6):
            self.assertAlmostEqual(
                float(X.pmf(k)), float(DiscreteUniform(a=1, b=5).pmf(k)), places=12
            )

    def test_Zipf_n_two_is_shifted_bernoulli(self):
        # With only two ranks, Zipf(a, 2) is 1 + Bernoulli(p) where
        # p = 2 ** -a / (1 + 2 ** -a) is the chance of landing on rank 2.
        a = 1.4
        p = 2**-a / (1 + 2**-a)
        X = Zipf(shape=a, n=2)
        self.assertAlmostEqual(float(X.pmf(1)), float(Bernoulli(p).pmf(0)), places=12)
        self.assertAlmostEqual(float(X.pmf(2)), float(Bernoulli(p).pmf(1)), places=12)

    def test_Zipf_n_one_is_point_mass(self):
        X = Zipf(shape=1.5, n=1)
        self.assertEqual(float(X.pmf(1)), 1.0)
        self.assertAlmostEqual(float(X.mean()), 1.0)

    # --- sampling ---

    def test_Zipf_distributional(self):
        seed(42)
        exp_list, obs_list = [], []
        X = RV(Zipf(shape=1.2, n=10))
        sims = X.sim(Nsim)
        simulated = sims.tabulate()
        for k in range(1, 11):
            expected = Nsim * float(stats.zipfian(1.2, 10).pmf(k))
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(simulated[k])
        pval = stats.chisquare(
            obs_list, np.array(exp_list) * sum(obs_list) / sum(exp_list)
        ).pvalue
        self.assertTrue(pval > 0.01)

    def test_Zipf_sim_stays_in_support(self):
        seed(42)
        sims = RV(Zipf(shape=0.7, n=12)).sim(Nsim)
        self.assertTrue(all(1 <= sim <= 12 for sim in sims))

    def test_Zipf_draw_is_scalar_in_support(self):
        seed(0)
        value = Zipf(shape=1.2, n=10).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 1.0)
        self.assertLessEqual(float(value), 10.0)

    # --- reproducibility ---

    def test_Zipf_same_seed_gives_same_sims(self):
        seed(2024)
        first = list(RV(Zipf(shape=1.2, n=10)).sim(500))
        seed(2024)
        second = list(RV(Zipf(shape=1.2, n=10)).sim(500))
        self.assertEqual(first, second)

    def test_Zipf_different_seed_gives_different_sims(self):
        seed(1)
        first = list(RV(Zipf(shape=1.2, n=10)).sim(500))
        seed(2)
        second = list(RV(Zipf(shape=1.2, n=10)).sim(500))
        self.assertNotEqual(first, second)

    # --- parameter validation ---

    def test_Zipf_error_shape_negative(self):
        self.assertRaisesRegex(
            Exception,
            "shape must be a non-negative number",
            lambda: Zipf(shape=-1, n=10),
        )

    def test_Zipf_error_shape_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "shape must be a non-negative number",
            lambda: Zipf(shape="x", n=10),
        )

    def test_Zipf_error_n_not_positive(self):
        for bad in [0, -3]:
            self.assertRaisesRegex(
                Exception,
                "n must be a positive integer",
                lambda v=bad: Zipf(shape=1, n=v),
            )

    def test_Zipf_error_n_float(self):
        self.assertRaisesRegex(
            Exception, "n must be a positive integer", lambda: Zipf(shape=1, n=10.5)
        )

    def test_Zipf_error_n_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "n must be a positive integer", lambda: Zipf(shape=1, n="x")
        )

    def test_Zipf_stacks_shape_and_n(self):
        # Both parameters wrong -> both mistakes reported at once.
        with self.assertRaises(Exception) as cm:
            Zipf(shape=-1, n=0)
        message = str(cm.exception)
        self.assertIn("Invalid parameters:", message)
        self.assertIn("shape must be a non-negative number", message)
        self.assertIn("n must be a positive integer", message)

    def test_Zipf_boundary_parameters_accepted(self):
        # a = 0 (uniform) and n = 1 (point mass) are valid, not errors.
        Zipf(shape=0, n=1)
        Zipf(shape=0, n=10)
        Zipf(shape=1, n=1)

    def test_Zipf_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Zipf(shape=1.2, n=10).draw()
        RV(Zipf(shape=1.2, n=10)).sim(100).plot()
        Zipf(shape=1.2, n=10).plot()
        plt.figure()
        Zipf(shape=1.2, n=10).plot(cdf=True)
        plt.figure()
        Zipf(shape=1.2, n=500).plot()
        plt.close("all")


class TestZeta(unittest.TestCase):

    def test_Zeta_pmf_matches_formula(self):
        # P(X = k) = 1 / (k ** shape * zeta(shape))
        from scipy.special import zeta

        shape = 2.5
        X = Zeta(shape=shape)
        for k in [1, 2, 3, 10, 50]:
            expected = 1 / (k**shape * zeta(shape))
            self.assertAlmostEqual(float(X.pmf(k)), float(expected), places=12)

    def test_Zeta_pmf_matches_scipy_zipf(self):
        # scipy's `zipf` is the zeta distribution, NOT the finite Zipf.
        X = Zeta(shape=3.0)
        th = stats.zipf(a=3.0)
        for k in range(1, 11):
            self.assertAlmostEqual(float(X.pmf(k)), float(th.pmf(k)), places=12)

    def test_Zeta_distributional(self):
        seed(42)
        X = RV(Zeta(shape=2.5))
        sims = X.sim(Nsim)
        # Compare the head of the distribution, lumping the tail into one bin
        # so every expected count is large enough for a chi-square test.
        values = np.array(list(sims), dtype=float)
        edges = [1, 2, 3, 4, 5]
        observed = [int((values == v).sum()) for v in edges]
        observed.append(int((values > edges[-1]).sum()))
        th = stats.zipf(a=2.5)
        expected = [Nsim * float(th.pmf(v)) for v in edges]
        expected.append(Nsim * float(th.sf(edges[-1])))
        pval = stats.chisquare(observed, expected).pvalue
        self.assertTrue(pval > 0.01)

    def test_Zeta_pmf_agrees_with_cdf(self):
        # Summing the pmf up to k must give the cdf at k. This is exact, so it
        # holds regardless of how much probability is left in the tail -- which
        # matters here, since the tail beyond any cutoff is never negligible.
        X = Zeta(shape=2.5)
        for k in [1, 5, 20, 500]:
            head = sum(float(X.pmf(j)) for j in range(1, k + 1))
            self.assertAlmostEqual(head, float(X.cdf(k)), places=10)

    def test_Zeta_total_probability_is_one(self):
        # The head plus the survival function beyond it accounts for all of it.
        X = Zeta(shape=2.5)
        head = sum(float(X.pmf(j)) for j in range(1, 501))
        tail = float(stats.zipf(a=2.5).sf(500))
        self.assertAlmostEqual(head + tail, 1.0, places=10)

    def test_Zeta_is_decreasing(self):
        X = Zeta(shape=2.0)
        probs = [float(X.pmf(k)) for k in range(1, 30)]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_Zeta_mean_finite_only_above_two(self):
        from scipy.special import zeta

        # shape > 2: finite, equal to zeta(shape - 1) / zeta(shape).
        for shape in [2.5, 3.0, 4.0]:
            X = Zeta(shape=shape)
            expected = float(zeta(shape - 1) / zeta(shape))
            self.assertAlmostEqual(float(X.mean()), expected, places=8)
        # shape <= 2: the mean diverges.
        for shape in [1.5, 2.0]:
            self.assertTrue(np.isinf(float(Zeta(shape=shape).mean())))

    def test_Zeta_var_finite_only_above_three(self):
        self.assertTrue(np.isfinite(float(Zeta(shape=3.5).var())))
        for shape in [1.5, 2.5, 3.0]:
            self.assertTrue(np.isinf(float(Zeta(shape=shape).var())))

    def test_Zeta_larger_shape_concentrates_on_one(self):
        probs = [float(Zeta(shape=s).pmf(1)) for s in [1.5, 2.0, 2.5, 4.0]]
        self.assertEqual(probs, sorted(probs))

    def test_Zeta_is_limit_of_Zipf(self):
        # Zeta is the n -> infinity limit of the finite Zipf.
        shape = 2.5
        target = float(Zeta(shape=shape).pmf(1))
        approx = [float(Zipf(shape=shape, n=n).pmf(1)) for n in [10, 100, 1000, 100000]]
        gaps = [abs(a - target) for a in approx]
        self.assertEqual(gaps, sorted(gaps, reverse=True))
        self.assertAlmostEqual(approx[-1], target, places=7)

    def test_Zeta_xlim_is_bounded_head_window(self):
        # Deliberately a small fixed window: the usual highest-density helper
        # would enumerate out to an astronomically large quantile for a power
        # law near shape = 1.
        for shape in [1.1, 1.5, 2.5, 4.0]:
            self.assertEqual(Zeta(shape=shape).xlim, (1, 20))

    def test_Zeta_draws_are_positive_integers(self):
        seed(0)
        sims = RV(Zeta(shape=2.5)).sim(500)
        values = [float(v) for v in sims]
        self.assertTrue(all(v >= 1 for v in values))
        self.assertTrue(all(v == int(v) for v in values))

    def test_Zeta_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(Zeta(shape=2.5).draw(), Scalar)

    def test_Zeta_invalid_shape_raises(self):
        # shape must be strictly greater than 1; at 1 or below the
        # probabilities cannot be normalized.
        for bad in [1, 1.0, 0.5, 0, -2, "a"]:
            self.assertRaises(Exception, lambda b=bad: Zeta(shape=b))

    def test_Zeta_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Zeta(shape=2.5).draw()
        RV(Zeta(shape=2.5)).sim(100).plot()
        Zeta(shape=2.5).plot()
        plt.figure()
        Zeta(shape=2.5).plot(cdf=True)
        plt.figure()
        Zeta(shape=1.5).plot()
        plt.close("all")


class TestBenford(unittest.TestCase):

    def test_Benford_pmf_matches_formula(self):
        # P(X = d) = log_base(1 + 1/d), computed here the long way round
        # (a ratio of plain logs) so the test does not just repeat the
        # log1p/change-of-base spelling the implementation uses.
        for base in [10, 2, 8, 16]:
            X = Benford(base=base)
            for d in range(1, base):
                expected = math.log(1 + 1 / d) / math.log(base)
                self.assertAlmostEqual(float(X.pmf(d)), expected, places=12)

    def test_Benford_total_probability_is_one(self):
        # The support is finite, so this is an exact sum, not a head-plus-tail
        # approximation the way Zeta's has to be.
        for base in [2, 10, 16]:
            total = sum(float(Benford(base=base).pmf(d)) for d in range(1, base))
            self.assertAlmostEqual(total, 1.0, places=12)

    def test_Benford_leading_one_is_about_thirty_percent(self):
        # The fact the law is known for: a leading 1 turns up about 30.1% of
        # the time, not the 1-in-9 (11.1%) a uniform guess would give.
        self.assertAlmostEqual(float(Benford().pmf(1)), 0.30103, places=5)
        self.assertGreater(float(Benford().pmf(1)), 2.5 * (1 / 9))

    def test_Benford_pmf_is_decreasing(self):
        # Every digit is likelier than the one after it -- 1 leads about six
        # times as often as 9.
        probs = [float(Benford().pmf(d)) for d in range(1, 10)]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertAlmostEqual(probs[0] / probs[-1], 6.579, places=3)

    def test_Benford_support_bounds(self):
        # Digits run 1 .. base - 1: there is no leading 0, and no digit as
        # large as the base itself.
        for base in [2, 10, 16]:
            X = Benford(base=base)
            self.assertEqual(float(X.pmf(0)), 0.0)
            self.assertEqual(float(X.pmf(base)), 0.0)
            self.assertAlmostEqual(float(X.cdf(base - 1)), 1.0, places=12)

    def test_Benford_xlim_comes_from_the_automatic_mechanism(self):
        # No custom xlim override on this class: the support is bounded at
        # both ends, so the shared rule uses it in full.
        self.assertEqual(Benford().xlim, (1, 9))
        self.assertEqual(Benford(base=16).xlim, (1, 15))
        self.assertEqual(Benford(base=2).xlim, (1, 1))

    def test_Benford_mean(self):
        # sum(d * log10(1 + 1/d)) over d = 1..9.
        self.assertAlmostEqual(float(Benford().mean()), 3.440237, places=6)

    def test_Benford_base_two_is_a_point_mass_at_one(self):
        # Degenerate but legitimate: in binary every number leads with a 1.
        X = Benford(base=2)
        self.assertAlmostEqual(float(X.pmf(1)), 1.0, places=12)
        self.assertAlmostEqual(float(X.mean()), 1.0, places=12)
        self.assertAlmostEqual(float(X.var()), 0.0, places=12)
        seed(0)
        self.assertTrue(all(v == 1 for v in RV(X).sim(200)))

    def test_Benford_distributional(self):
        seed(42)
        sims = RV(Benford()).sim(Nsim)
        values = np.array(list(sims), dtype=float)
        observed = [int((values == d).sum()) for d in range(1, 10)]
        expected = [Nsim * float(Benford().pmf(d)) for d in range(1, 10)]
        pval = stats.chisquare(observed, expected).pvalue
        self.assertTrue(pval > 0.01)

    def test_Benford_draws_are_digits(self):
        seed(0)
        values = [float(v) for v in RV(Benford()).sim(500)]
        self.assertTrue(all(1 <= v <= 9 for v in values))
        self.assertTrue(all(v == int(v) for v in values))

    def test_Benford_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(Benford().draw(), Scalar)

    def test_Benford_whole_valued_float_base_accepted(self):
        # 10.0 is a whole number, so it is accepted and stored as an int --
        # the same leniency DiscreteUniform's bounds have.
        X = Benford(base=10.0)
        self.assertEqual(X.base, 10)
        self.assertIsInstance(X.base, int)

    def test_Benford_invalid_base_raises(self):
        # base must be an integer of at least 2: base 1 has no digits to
        # lead with, and a fractional base is not a base.
        for bad in [1, 1.0, 0, -3, 10.5, "ten", None]:
            self.assertRaises(Exception, lambda b=bad: Benford(base=b))

    def test_Benford_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Benford().draw()
        RV(Benford()).sim(100).plot()
        Benford().plot()
        plt.figure()
        Benford().plot(cdf=True)
        plt.figure()
        Benford(base=2).plot()
        plt.close("all")


class TestUniform(unittest.TestCase):

    def test_Uniform_error(self):
        self.assertRaises(Exception, lambda: Uniform(a=6, b=-1))

    def test_conditional_exp_uniform(self):
        seed(42)
        X, Y = RV(Exponential(rate=3) ** 2)
        sims = (X | (X < 3) & (X + Y > 3)).sim(1000)
        cdf = stats.uniform(loc=0, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_ChiSquare(self):
        seed(42)
        X = RV(Uniform(a=0, b=1))
        sims = (-2 * log(X)).sim(Nsim)
        cdf = stats.chi2(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Exponential(self):
        seed(42)
        X = RV(Uniform(a=0, b=1))
        Y = -1 / 5 * log(X)
        sims = Y.sim(Nsim)
        cdf = stats.expon(scale=1 / 5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Beta(self):
        seed(42)
        X = RV(Uniform(a=0, b=1))
        sims = (X**15).sim(Nsim)
        cdf = stats.beta(a=1 / 15, b=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Cauchy(self):
        seed(42)
        X = RV(Uniform(a=0, b=1))
        sims = (pi * (X - 1 / 2)).apply(tan).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Uniform_to_Pareto(self):
        seed(42)
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


class TestIrwinHall(unittest.TestCase):

    def test_IrwinHall_distributional(self):
        seed(42)
        X = RV(IrwinHall(n=5))
        sims = X.sim(Nsim)
        cdf = stats.irwinhall(5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_IrwinHall_general_bounds_distributional(self):
        # Sum of n uniforms on [a, b], not just on [0, 1].
        seed(42)
        X = RV(IrwinHall(n=4, a=10, b=20))
        sims = X.sim(Nsim)
        cdf = stats.irwinhall(4, loc=4 * 10, scale=20 - 10).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_IrwinHall_matches_sum_of_uniforms(self):
        # The defining property: adding n independent uniforms gives this
        # distribution.
        seed(42)
        U1, U2, U3 = RV(Uniform(0, 1) ** 3)
        sims = (U1 + U2 + U3).sim(Nsim)
        pval = stats.kstest(sims, IrwinHall(n=3).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_IrwinHall_mean_var_sd(self):
        # n times the mean and variance of a single uniform, since the
        # pieces are independent.
        for n, a, b in [(1, 0, 1), (5, 0, 1), (12, 0, 1), (4, 10, 20), (3, -2, 2)]:
            X = IrwinHall(n=n, a=a, b=b)
            self.assertAlmostEqual(float(X.mean()), n * (a + b) / 2, places=8)
            self.assertAlmostEqual(float(X.var()), n * (b - a) ** 2 / 12, places=8)
            self.assertAlmostEqual(
                float(X.sd()), math.sqrt(n * (b - a) ** 2 / 12), places=8
            )

    def test_IrwinHall_n_12_has_variance_one(self):
        # The classic normal-approximation case: mean 6, variance exactly 1.
        X = IrwinHall(n=12)
        self.assertAlmostEqual(float(X.mean()), 6.0, places=8)
        self.assertAlmostEqual(float(X.var()), 1.0, places=8)

    def test_IrwinHall_n_1_is_Uniform(self):
        X = IrwinHall(n=1, a=2, b=8)
        Y = Uniform(a=2, b=8)
        for x in [2, 3.5, 5, 8]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=8)
            self.assertAlmostEqual(float(X.cdf(x)), float(Y.cdf(x)), places=8)

    def test_IrwinHall_n_2_is_Triangular(self):
        # Two uniforms on [a, b] sum to a triangle on [2a, 2b] peaking at a + b.
        X = IrwinHall(n=2, a=0, b=1)
        Y = Triangular(low=0, mode=1, high=2)
        for x in [0, 0.5, 1, 1.5, 2]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=8)
            self.assertAlmostEqual(float(X.cdf(x)), float(Y.cdf(x)), places=8)

    def test_IrwinHall_over_n_is_Bates(self):
        # The sum divided by n is the average of the n uniforms, which is
        # exactly what Bates describes.
        for n, a, b in [(5, 0, 1), (4, 10, 20)]:
            X = IrwinHall(n=n, a=a, b=b)
            Y = Bates(n=n, a=a, b=b)
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                # The scaled sum and the Bates variable share every quantile.
                self.assertAlmostEqual(
                    float(X.quantile(q)) / n, float(Y.quantile(q)), places=8
                )

    def test_IrwinHall_symmetric_about_mean(self):
        X = IrwinHall(n=7)
        for offset in [0.5, 1.5, 3.0]:
            self.assertAlmostEqual(
                float(X.pdf(3.5 - offset)), float(X.pdf(3.5 + offset)), places=8
            )

    def test_IrwinHall_density_peaks_at_mean(self):
        X = IrwinHall(n=6)
        xs = np.linspace(0, 6, 12001)
        self.assertAlmostEqual(float(xs[np.argmax(X.pdf(xs))]), 3.0, places=3)

    def test_IrwinHall_no_probability_outside_support(self):
        X = IrwinHall(n=4, a=10, b=20)
        self.assertAlmostEqual(float(X.pdf(39)), 0.0, places=8)
        self.assertAlmostEqual(float(X.pdf(81)), 0.0, places=8)
        self.assertAlmostEqual(float(X.cdf(40)), 0.0, places=8)
        self.assertAlmostEqual(float(X.cdf(80)), 1.0, places=8)

    def test_IrwinHall_approaches_Normal(self):
        # The Central Limit Theorem, which is what this distribution is for:
        # the standardized density gets closer to the standard normal one as
        # n grows.
        zs = np.linspace(-4, 4, 4001)
        errors = []
        for n in [1, 2, 6, 12, 30]:
            X = IrwinHall(n=n)
            sd = math.sqrt(n / 12)
            # Largest gap between the standardized sum's CDF and the standard
            # normal one. Compared on the CDF rather than the density: the
            # densities agree at the center for small n by coincidence, while
            # this whole-curve distance shrinks with every step up in n.
            errors.append(np.max(np.abs(X.cdf(n / 2 + zs * sd) - stats.norm.cdf(zs))))
        self.assertTrue(all(x > y for x, y in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 0.005)

    def test_IrwinHall_draws_within_bounds(self):
        seed(0)
        sims = RV(IrwinHall(n=4, a=10, b=20)).sim(1000)
        self.assertTrue(all(40 <= float(v) <= 80 for v in sims))

    def test_IrwinHall_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(IrwinHall(n=5).draw(), Scalar)

    def test_IrwinHall_xlim_is_full_support(self):
        self.assertEqual(IrwinHall(n=5).xlim, (0, 5))
        self.assertEqual(IrwinHall(n=4, a=10, b=20).xlim, (40, 80))

    def test_IrwinHall_invalid_n_raises(self):
        for bad in [0, -3, 2.5, "5", None]:
            self.assertRaises(Exception, lambda b=bad: IrwinHall(n=b))

    def test_IrwinHall_invalid_bounds_raise(self):
        self.assertRaises(Exception, lambda: IrwinHall(n=3, a=6, b=-1))
        self.assertRaises(Exception, lambda: IrwinHall(n=3, a="a", b=1))
        self.assertRaises(Exception, lambda: IrwinHall(n=3, a=0, b="b"))

    def test_IrwinHall_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        IrwinHall(5).draw()
        RV(IrwinHall(5)).sim(100).plot()
        IrwinHall(5).plot()
        plt.figure()
        IrwinHall(5).plot(cdf=True)
        plt.figure()
        IrwinHall(30).plot()  # zooms itself off the (0, 30) support
        plt.close("all")


class TestBates(unittest.TestCase):

    def test_Bates_distributional(self):
        seed(42)
        X = RV(Bates(n=5, a=2, b=8))
        sims = X.sim(Nsim)
        cdf = stats.irwinhall(5, loc=2, scale=(8 - 2) / 5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Bates_mean_var_sd(self):
        X = Bates(n=5, a=2, b=8)
        # mean = (a + b) / 2, var = (b - a)**2 / (12 n)
        self.assertAlmostEqual(float(X.mean()), 5.0, places=6)
        self.assertAlmostEqual(float(X.var()), (8 - 2) ** 2 / (12 * 5), places=6)
        self.assertAlmostEqual(
            float(X.sd()), np.sqrt((8 - 2) ** 2 / (12 * 5)), places=6
        )

    def test_Bates_support(self):
        # Bates lives on [a, b].
        X = Bates(n=4, a=1, b=3)
        self.assertEqual(X.xlim, (1, 3))
        self.assertAlmostEqual(float(X.cdf(1)), 0.0, places=9)
        self.assertAlmostEqual(float(X.cdf(3)), 1.0, places=9)

    def test_Bates_n1_is_uniform(self):
        # The average of a single uniform is that uniform.
        X = Bates(n=1, a=2, b=5)
        for x in [2.5, 3.0, 4.7]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(Uniform(a=2, b=5).pdf(x)), places=9
            )

    def test_Bates_n2_is_triangular(self):
        # The average of two U(0, 1) is triangular on [0, 1], peak 2 at 0.5.
        X = Bates(n=2, a=0, b=1)
        self.assertAlmostEqual(float(X.pdf(0.5)), 2.0, places=6)
        self.assertAlmostEqual(float(X.pdf(0.25)), 1.0, places=6)

    def test_Bates_is_mean_of_uniforms(self):
        # Bates(n) is the distribution of the average of n Uniform(0, 1).
        seed(42)
        n = 5
        U = RV(Uniform(0, 1) ** n)
        sims = U.apply(lambda u: sum(u) / n).sim(Nsim)
        cdf = stats.irwinhall(n, loc=0, scale=1 / n).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Bates_draw_is_scalar_in_support(self):
        seed(0)
        value = Bates(n=5, a=0, b=1).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)
        self.assertLessEqual(float(value), 1.0)

    def test_Bates_error_n(self):
        for bad in [0, -1, 2.5, "a"]:
            self.assertRaises(Exception, lambda v=bad: Bates(n=v))

    def test_Bates_error_reversed_bounds(self):
        self.assertRaises(Exception, lambda: Bates(n=3, a=5, b=1))

    def test_Bates_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Bates(n=5).draw()
        RV(Bates(n=5)).sim(100).plot()
        Bates(n=5).plot()
        plt.figure()
        Bates(n=5).plot(cdf=True)
        plt.close("all")


class TestLogUniform(unittest.TestCase):

    def test_LogUniform_distributional(self):
        seed(42)
        X = RV(LogUniform(a=1, b=100))
        sims = X.sim(Nsim)
        cdf = stats.loguniform(1, 100).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogUniform_mean_median_sd(self):
        X = LogUniform(a=1, b=100)
        th = stats.loguniform(1, 100)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)
        # median is the geometric mean sqrt(a * b)
        self.assertAlmostEqual(float(X.median()), (1 * 100) ** 0.5, places=6)

    def test_LogUniform_pdf(self):
        X = LogUniform(a=1, b=100)
        for x in [1, 10, 100]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(stats.loguniform(1, 100).pdf(x))
            )

    def test_LogUniform_support(self):
        X = LogUniform(a=2, b=50)
        self.assertEqual(X.xlim, (2, 50))
        self.assertAlmostEqual(float(X.cdf(2)), 0.0, places=9)
        self.assertAlmostEqual(float(X.cdf(50)), 1.0, places=9)

    def test_LogUniform_log_is_uniform(self):
        # If X ~ LogUniform(a, b), then log(X) ~ Uniform(log a, log b).
        seed(42)
        X = RV(LogUniform(a=1, b=100))
        sims = log(X).sim(Nsim)
        cdf = stats.uniform(0, np.log(100)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogUniform_exp_of_uniform(self):
        # exp(Uniform(log a, log b)) has a LogUniform(a, b) distribution.
        seed(42)
        U = RV(Uniform(np.log(1), np.log(100)))
        sims = exp(U).sim(Nsim)
        cdf = stats.loguniform(1, 100).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogUniform_draw_is_scalar_in_support(self):
        seed(0)
        value = LogUniform(a=1, b=100).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 1.0)
        self.assertLessEqual(float(value), 100.0)

    def test_LogUniform_error_a_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(Exception, lambda v=bad: LogUniform(a=v, b=10))

    def test_LogUniform_error_b_not_greater(self):
        self.assertRaises(Exception, lambda: LogUniform(a=10, b=2))
        self.assertRaises(Exception, lambda: LogUniform(a=5, b=5))

    def test_LogUniform_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        LogUniform(a=1, b=100).draw()
        RV(LogUniform(a=1, b=100)).sim(100).plot()
        LogUniform(a=1, b=100).plot()
        plt.figure()
        LogUniform(a=1, b=100).plot(cdf=True)
        plt.close("all")


class TestReciprocal(unittest.TestCase):

    def test_Reciprocal_is_loguniform(self):
        # Reciprocal is an alias for LogUniform: same distribution.
        X = Reciprocal(a=1, b=100)
        self.assertIsInstance(X, LogUniform)
        Y = LogUniform(a=1, b=100)
        for x in [1, 10, 100]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=12)
        self.assertEqual(float(X.median()), float(Y.median()))

    def test_Reciprocal_plots_without_error(self):
        Reciprocal(a=1, b=100).draw()
        RV(Reciprocal(a=1, b=100)).sim(100).plot()
        Reciprocal(a=1, b=100).plot()
        plt.close("all")


class TestNormal(unittest.TestCase):

    def test_Normal_error(self):
        self.assertRaises(Exception, lambda: Normal(mean=0, var=-10))

    def test_sum(self):
        seed(42)
        X = RV(Normal(mean=-1, sd=2) ** 3)
        sims = X.apply(sum).sim(Nsim)
        cdf = stats.norm(loc=-3, scale=np.sqrt(12)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_sum_Standard_Normal(self):
        seed(42)
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X + Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_subtract_Standard_Normal(self):
        seed(42)
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X - Y).sim(Nsim)
        cdf = stats.norm(loc=0, scale=sqrt(2)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_standardize(self):
        seed(42)
        X = RV(Normal(mean=8, var=4))
        X_stand = (X - 8) / 2
        sims = X_stand.sim(Nsim)
        cdf = stats.norm(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_standardize_to_Normal(self):
        seed(42)
        Z = RV(Normal(mean=0, sd=1))
        X = 10 + 5 * Z
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=10, scale=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_Gamma(self):
        seed(42)
        X = RV(Normal(mean=0, var=1))
        X = X**2
        sims = X.sim(Nsim)
        cdf = stats.gamma(a=1 / 2, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_ChiSquare(self):
        seed(42)
        X = RV(Normal(mean=0, var=1))
        X = X**2
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_Cauchy(self):
        # Seed 42 is pathological for this heavy-tailed Cauchy KS test
        # (lands in the ~1% false-rejection region); use a robust seed.
        seed(0)
        X, Y = RV(Normal(mean=0, var=1) ** 2)
        sims = (X / Y).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_F(self):
        seed(42)
        A, B, C, V, W, X, Y, Z = RV(Normal(mean=0, var=1) ** 8)
        sims = (
            (((A**2) + (B**2) + (C**2)) / 3)
            / (((V**2) + (W**2) + (X**2) + (Y**2) + (Z**2)) / 5)
        ).sim(Nsim)
        cdf = stats.f(dfn=3, dfd=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_sum_Normal_to_ChiSquare(self):
        seed(42)
        X, Y, Z, A, B = RV(Normal(mean=0, var=1) ** 5)
        sims = ((X**2) + (Y**2) + (Z**2) + (A**2) + (B**2)).sim(Nsim)
        cdf = stats.chi2(df=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_var_param(self):
        seed(42)
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


class TestTruncatedNormal(unittest.TestCase):

    def test_TruncatedNormal_error_bounds(self):
        self.assertRaises(Exception, lambda: TruncatedNormal(a=2, b=1))

    def test_TruncatedNormal_error_bounds_equal(self):
        self.assertRaises(Exception, lambda: TruncatedNormal(a=1, b=1))

    def test_TruncatedNormal_error_sd_negative(self):
        self.assertRaises(Exception, lambda: TruncatedNormal(sd=-1, a=-2, b=2))

    def test_TruncatedNormal_error_sd_zero(self):
        self.assertRaises(Exception, lambda: TruncatedNormal(sd=0, a=-2, b=2))

    def test_TruncatedNormal_error_var_negative(self):
        self.assertRaises(Exception, lambda: TruncatedNormal(var=-1.0, a=-2, b=2))

    def test_TruncatedNormal_sd_var_conflict(self):
        self.assertRaises(
            ValueError, lambda: TruncatedNormal(mean=0, sd=2, var=9, a=-1, b=1)
        )

    def test_TruncatedNormal_distribution(self):
        seed(42)
        X = RV(TruncatedNormal(mean=1, sd=2, a=-1, b=4))
        sims = X.sim(Nsim)
        a_std, b_std = (-1 - 1) / 2, (4 - 1) / 2
        cdf = stats.truncnorm(a_std, b_std, loc=1, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_TruncatedNormal_pdf_matches_scipy(self):
        X = TruncatedNormal(mean=1, sd=2, a=-1, b=4)
        a_std, b_std = (-1 - 1) / 2, (4 - 1) / 2
        th = stats.truncnorm(a_std, b_std, loc=1, scale=2)
        for x in [-1, 0, 1, 2, 4]:
            self.assertAlmostEqual(float(X.pdf(x)), float(th.pdf(x)), places=12)

    def test_TruncatedNormal_pdf_zero_outside_bounds(self):
        X = TruncatedNormal(mean=0, sd=1, a=-2, b=2)
        self.assertEqual(float(X.pdf(-3)), 0.0)
        self.assertEqual(float(X.pdf(3)), 0.0)

    def test_TruncatedNormal_mean_var_match_scipy(self):
        X = TruncatedNormal(mean=1, sd=2, a=-1, b=4)
        a_std, b_std = (-1 - 1) / 2, (4 - 1) / 2
        th = stats.truncnorm(a_std, b_std, loc=1, scale=2)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()))
        self.assertAlmostEqual(float(X.var()), float(th.var()))

    def test_TruncatedNormal_draws_within_bounds(self):
        seed(42)
        X = TruncatedNormal(mean=0, sd=3, a=-1, b=2)
        for _ in range(500):
            self.assertTrue(-1 <= X.draw() <= 2)

    def test_TruncatedNormal_var_param(self):
        # Passing var instead of sd gives the same distribution.
        X = TruncatedNormal(mean=0, var=4, a=-3, b=3)
        Y = TruncatedNormal(mean=0, sd=2, a=-3, b=3)
        for x in [-3, -1, 0, 1, 3]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=12)

    def test_TruncatedNormal_lower_truncation_is_halfnormal(self):
        # A standard normal truncated below at its mean is a half-normal.
        seed(42)
        X = RV(TruncatedNormal(mean=0, sd=1, a=0))
        sims = X.sim(Nsim)
        cdf = stats.halfnorm(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_TruncatedNormal_infinite_bounds_is_normal(self):
        # With no truncation the density equals the underlying normal's.
        X = TruncatedNormal(mean=2, sd=3)
        Y = Normal(mean=2, sd=3)
        for x in [-4, 0, 2, 5, 8]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=12)

    def test_TruncatedNormal_xlim_uses_finite_bounds(self):
        X = TruncatedNormal(mean=0, sd=1, a=-2, b=2)
        self.assertEqual(X.xlim, (-2, 2))

    def test_TruncatedNormal_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        TruncatedNormal(mean=0, sd=1, a=-2, b=2).draw()
        RV(TruncatedNormal(mean=0, sd=1, a=-2, b=2)).sim(100).plot()
        TruncatedNormal(mean=0, sd=1, a=-2, b=2).plot()
        plt.figure()
        TruncatedNormal(mean=0, sd=1, a=-2, b=2).plot(cdf=True)
        plt.figure()
        TruncatedNormal(mean=0, sd=1, a=0).plot()
        plt.close("all")


class TestSkewNormal(unittest.TestCase):

    @staticmethod
    def delta(shape):
        # The tilt on a 0-to-1 scale, which every summary formula runs through.
        return shape / math.sqrt(1 + shape**2)

    def test_SkewNormal_distributional(self):
        seed(42)
        X = RV(SkewNormal(loc=0, scale=1, shape=4))
        sims = X.sim(Nsim)
        cdf = stats.skewnorm(a=4, loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_SkewNormal_general_params_distributional(self):
        seed(42)
        X = RV(SkewNormal(loc=70, scale=12, shape=-3))
        sims = X.sim(Nsim)
        cdf = stats.skewnorm(a=-3, loc=70, scale=12).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_SkewNormal_shape_zero_is_Normal(self):
        # No tilt at all, so it should agree with the normal exactly.
        X = SkewNormal(loc=5, scale=2, shape=0)
        Y = Normal(mean=5, sd=2)
        xs = np.linspace(-5, 15, 501)
        self.assertTrue(np.allclose(X.pdf(xs), Y.pdf(xs)))
        self.assertTrue(np.allclose(X.cdf(xs), Y.cdf(xs)))
        self.assertAlmostEqual(float(X.mean()), 5.0, places=8)
        self.assertAlmostEqual(float(X.sd()), 2.0, places=8)

    def test_SkewNormal_mean_var_sd(self):
        # Tilting moves the center and shrinks the spread, both by amounts set
        # by delta.
        for loc, scale, shape in [
            (0, 1, 0),
            (0, 1, 4),
            (0, 1, -4),
            (2, 3, 1),
            (70, 12, -3),
        ]:
            X = SkewNormal(loc=loc, scale=scale, shape=shape)
            d = self.delta(shape)
            expected_mean = loc + scale * d * math.sqrt(2 / math.pi)
            expected_var = scale**2 * (1 - 2 * d**2 / math.pi)
            self.assertAlmostEqual(float(X.mean()), expected_mean, places=8)
            self.assertAlmostEqual(float(X.var()), expected_var, places=8)
            self.assertAlmostEqual(float(X.sd()), math.sqrt(expected_var), places=8)

    def test_SkewNormal_mean_is_not_loc_and_sd_is_not_scale(self):
        # The trap worth pinning down: with a tilt, loc and scale are not the
        # mean and sd -- the mean rises above loc and the sd falls below scale.
        X = SkewNormal(loc=0, scale=1, shape=4)
        self.assertGreater(float(X.mean()), 0)
        self.assertLess(float(X.sd()), 1)
        self.assertAlmostEqual(float(X.mean()), 0.7740617226, places=8)
        self.assertAlmostEqual(float(X.var()), 0.4008284495, places=8)

    def test_SkewNormal_stores_params(self):
        X = SkewNormal(loc=2, scale=3, shape=-1.5)
        self.assertEqual(X.loc, 2)
        self.assertEqual(X.scale, 3)
        self.assertEqual(X.shape, -1.5)
        self.assertAlmostEqual(X.params["a"], -1.5, places=8)
        self.assertEqual(X.params["loc"], 2)
        self.assertEqual(X.params["scale"], 3)

    def test_SkewNormal_skew_direction(self):
        # Positive shape stretches the right tail, so the mean is pulled past
        # the median; negative shape does the mirror image.
        right = SkewNormal(loc=0, scale=1, shape=4)
        self.assertLess(float(right.median()), float(right.mean()))
        left = SkewNormal(loc=0, scale=1, shape=-4)
        self.assertGreater(float(left.median()), float(left.mean()))
        none = SkewNormal(loc=0, scale=1, shape=0)
        self.assertAlmostEqual(float(none.median()), float(none.mean()), places=8)

    def test_SkewNormal_negative_shape_is_mirror_image(self):
        # Centered at 0, shape and -shape are reflections of one another.
        xs = np.linspace(-4, 4, 801)
        X = SkewNormal(loc=0, scale=1, shape=3)
        Y = SkewNormal(loc=0, scale=1, shape=-3)
        self.assertTrue(np.allclose(X.pdf(xs), Y.pdf(-xs)))
        self.assertAlmostEqual(float(X.mean()), -float(Y.mean()), places=8)
        self.assertAlmostEqual(float(X.sd()), float(Y.sd()), places=8)

    def test_SkewNormal_matches_stochastic_representation(self):
        # The mixture the density is built from: a one-sided half-normal piece
        # weighted by delta, plus an ordinary normal piece.
        seed(42)
        shape = 4
        d = self.delta(shape)
        Z, W = RV(HalfNormal(scale=1) * Normal(mean=0, sd=1))
        sims = (d * Z + math.sqrt(1 - d**2) * W).sim(Nsim)
        cdf = SkewNormal(loc=0, scale=1, shape=shape).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_SkewNormal_approaches_HalfNormal_as_shape_grows(self):
        # A big tilt squeezes the left half away, leaving a half-normal.
        xs = np.linspace(-3, 5, 2001)
        errors = []
        for shape in [1, 2, 5, 10, 100]:
            X = SkewNormal(loc=0, scale=1, shape=shape)
            errors.append(np.max(np.abs(X.cdf(xs) - stats.halfnorm.cdf(xs))))
        self.assertTrue(all(x > y for x, y in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 0.01)

    def test_SkewNormal_skewness_is_capped(self):
        # However far the shape is pushed, the skewness stops near 0.995 --
        # the docstring tells students to switch distributions rather than
        # keep raising shape, so pin the ceiling down.
        skewnesses = []
        for shape in [1, 5, 100, 10000]:
            X = SkewNormal(loc=0, scale=1, shape=shape)
            skewness = float(stats.skewnorm(**X.params).stats("s"))
            self.assertLess(skewness, 0.9953)
            skewnesses.append(skewness)
        self.assertTrue(all(x < y for x, y in zip(skewnesses, skewnesses[1:])))
        self.assertGreater(skewnesses[-1], 0.995)

    def test_SkewNormal_supported_on_all_real_numbers(self):
        # The thin tail is thin, not missing: there is density on both sides
        # for any shape.
        X = SkewNormal(loc=0, scale=1, shape=8)
        self.assertGreater(float(X.pdf(-2)), 0.0)
        self.assertGreater(float(X.cdf(-2)), 0.0)
        self.assertLess(float(X.cdf(6)), 1.0)

    def test_SkewNormal_cdf_quantile_roundtrip(self):
        X = SkewNormal(loc=70, scale=12, shape=-3)
        for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            self.assertAlmostEqual(float(X.cdf(X.quantile(q))), q, places=6)

    def test_SkewNormal_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(SkewNormal().draw(), Scalar)

    def test_SkewNormal_default_is_standard_normal(self):
        X = SkewNormal()
        self.assertEqual((X.loc, X.scale, X.shape), (0, 1, 0))
        self.assertAlmostEqual(float(X.pdf(0)), float(Normal().pdf(0)), places=8)

    def test_SkewNormal_xlim_is_equal_tailed_default(self):
        # Unbounded on both sides, so both ends are quantile cuts.
        X = SkewNormal(loc=0, scale=1, shape=4)
        self.assertAlmostEqual(X.xlim[0], float(X.quantile(0.001)), places=8)
        self.assertAlmostEqual(X.xlim[1], float(X.quantile(0.999)), places=8)

    def test_SkewNormal_invalid_params_raise(self):
        for bad in [0, -1, "1", None]:
            self.assertRaises(Exception, lambda b=bad: SkewNormal(scale=b))
        for bad in ["1", None]:
            self.assertRaises(Exception, lambda b=bad: SkewNormal(loc=b))
            self.assertRaises(Exception, lambda b=bad: SkewNormal(shape=b))

    def test_SkewNormal_shape_error_message_explains_the_parameter(self):
        # shape has no invalid numbers, so the message should say what the
        # values mean rather than just rejecting the input.
        with self.assertRaises(Exception) as caught:
            SkewNormal(shape="a lot")
        message = str(caught.exception)
        self.assertIn("shape = 0", message)
        self.assertIn("right tail", message)
        self.assertIn("left tail", message)

    def test_SkewNormal_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        SkewNormal().draw()
        RV(SkewNormal(loc=0, scale=1, shape=4)).sim(100).plot()
        SkewNormal(loc=0, scale=1, shape=4).plot()
        plt.figure()
        SkewNormal(loc=0, scale=1, shape=-4).plot(cdf=True)
        plt.figure()
        SkewNormal(loc=0, scale=1, shape=20).plot()
        plt.close("all")


class TestExponential(unittest.TestCase):

    def test_Exponential_error(self):
        self.assertRaises(Exception, lambda: Exponential(rate=-5))

    def test_Exponential_to_Gamma(self):
        seed(42)
        X = RV(Exponential(rate=0.9))
        sims = X.sim(Nsim)
        cdf = stats.gamma(scale=1 / 0.9, a=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_sum_Gamma(self):
        seed(42)
        X, Y, Z, A = RV(Exponential(rate=0.9) ** 4)
        sims = (X + Y + Z + A).sim(Nsim)
        cdf = stats.gamma(scale=1 / 0.9, a=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_ChiSquare(self):
        seed(42)
        X = RV(Exponential(rate=1 / 2))
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_Pareto(self):
        seed(42)
        X = RV(Exponential(rate=2))
        sims = (3 * exp(X)).sim(Nsim)
        cdf = stats.pareto(b=2, loc=0, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_Weibull(self):
        seed(42)
        X = RV(Exponential(rate=5))
        sims = X.sim(Nsim)
        cdf = stats.weibull_min(scale=1 / 5, c=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Exponential_to_Rayleigh(self):
        seed(42)
        X = RV(Exponential(rate=5))
        sims = sqrt(X).sim(Nsim)
        cdf = stats.rayleigh(scale=1 / sqrt(2 * 5)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Poisson_Exponential_to_Geometric(self):
        seed(42)

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
        seed(42)
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


class TestExponentiallyModifiedGaussian(unittest.TestCase):

    def test_EMG_distributional(self):
        seed(42)
        X = RV(ExponentiallyModifiedGaussian(mean=0, sd=1, rate=1))
        sims = X.sim(Nsim)
        cdf = stats.exponnorm(K=1 / (1 * 1), loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_EMG_general_params_distributional(self):
        seed(42)
        X = RV(ExponentiallyModifiedGaussian(mean=500, sd=50, rate=0.01))
        sims = X.sim(Nsim)
        cdf = stats.exponnorm(K=1 / (0.01 * 50), loc=500, scale=50).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_EMG_matches_normal_plus_exponential(self):
        # The defining property: adding an independent normal value and an
        # exponential value gives this distribution.
        seed(42)
        Z, W = RV(Normal(mean=2, sd=3) * Exponential(rate=0.5))
        sims = (Z + W).sim(Nsim)
        cdf = ExponentiallyModifiedGaussian(mean=2, sd=3, rate=0.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_EMG_mean_var_sd(self):
        # Means add and variances add, since the two pieces are independent.
        for mean, sd, rate in [
            (0, 1, 1),
            (0, 1, 5),
            (2, 0.5, 0.25),
            (-3, 4, 2),
            (500, 50, 0.01),
        ]:
            X = ExponentiallyModifiedGaussian(mean=mean, sd=sd, rate=rate)
            self.assertAlmostEqual(float(X.mean()), mean + 1 / rate, places=8)
            self.assertAlmostEqual(float(X.var()), sd**2 + 1 / rate**2, places=8)
            self.assertAlmostEqual(
                float(X.sd()), math.sqrt(sd**2 + 1 / rate**2), places=8
            )

    def test_EMG_mean_is_not_the_mean_argument(self):
        # The trap worth pinning down: the exponential part shifts the center,
        # so mean() is not what was passed in as mean.
        X = ExponentiallyModifiedGaussian(mean=0, sd=1, rate=1)
        self.assertAlmostEqual(float(X.mean()), 1.0, places=8)
        self.assertAlmostEqual(float(X.var()), 2.0, places=8)

    def test_EMG_stores_component_params(self):
        # Stored under loc/scale/rate so the mean/sd methods are not shadowed.
        X = ExponentiallyModifiedGaussian(mean=2, sd=0.5, rate=0.25)
        self.assertEqual(X.loc, 2)
        self.assertEqual(X.scale, 0.5)
        self.assertEqual(X.rate, 0.25)
        self.assertTrue(callable(X.mean) and callable(X.sd))

    def test_EMG_scipy_shape_param(self):
        # K is the exponential's average delay measured in sds of the normal.
        X = ExponentiallyModifiedGaussian(mean=2, sd=0.5, rate=0.25)
        self.assertAlmostEqual(X.params["K"], 1 / (0.25 * 0.5), places=8)
        self.assertEqual(X.params["loc"], 2)
        self.assertEqual(X.params["scale"], 0.5)

    def test_EMG_is_right_skewed(self):
        # Always skewed right: the exponential part only ever adds to the
        # value, so the mean is pulled past the median.
        for mean, sd, rate in [(0, 1, 1), (2, 0.5, 0.25), (-3, 4, 2)]:
            X = ExponentiallyModifiedGaussian(mean=mean, sd=sd, rate=rate)
            self.assertLess(float(X.median()), float(X.mean()))

    def test_EMG_approaches_Normal_as_rate_grows(self):
        # A large rate means short exponential delays, so little is added and
        # the shape closes in on the normal part it started from.
        xs = np.linspace(-5, 5, 2001)
        errors = []
        for rate in [1, 2, 5, 10, 100]:
            X = ExponentiallyModifiedGaussian(mean=0, sd=1, rate=rate)
            # Largest gap between the two CDFs, which shrinks with every step
            # up in rate.
            errors.append(np.max(np.abs(X.cdf(xs) - stats.norm.cdf(xs))))
        self.assertTrue(all(x > y for x, y in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 0.01)

    def test_EMG_approaches_Exponential_as_sd_shrinks(self):
        # A tiny sd leaves the normal part all but constant, so what remains
        # is the exponential part shifted by mean (here mean = 0).
        xs = np.linspace(-2, 8, 2001)
        errors = []
        for sd in [1, 0.5, 0.1, 0.01]:
            X = ExponentiallyModifiedGaussian(mean=0, sd=sd, rate=1)
            errors.append(np.max(np.abs(X.cdf(xs) - stats.expon.cdf(xs))))
        self.assertTrue(all(x > y for x, y in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 0.01)

    def test_EMG_supported_on_all_real_numbers(self):
        # The normal part can reach any value, so there is density everywhere
        # -- including well below the mean argument.
        X = ExponentiallyModifiedGaussian(mean=0, sd=1, rate=1)
        self.assertGreater(float(X.pdf(-6)), 0.0)
        self.assertGreater(float(X.cdf(-6)), 0.0)
        self.assertLess(float(X.cdf(20)), 1.0)

    def test_EMG_cdf_quantile_roundtrip(self):
        X = ExponentiallyModifiedGaussian(mean=500, sd=50, rate=0.01)
        for q in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            self.assertAlmostEqual(float(X.cdf(X.quantile(q))), q, places=6)

    def test_EMG_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(ExponentiallyModifiedGaussian().draw(), Scalar)

    def test_EMG_xlim_is_equal_tailed_default(self):
        # Unbounded on both sides, so both ends are quantile cuts.
        X = ExponentiallyModifiedGaussian(mean=0, sd=1, rate=1)
        self.assertAlmostEqual(X.xlim[0], float(X.quantile(0.001)), places=8)
        self.assertAlmostEqual(X.xlim[1], float(X.quantile(0.999)), places=8)

    def test_EMG_invalid_params_raise(self):
        for bad in [0, -1, "1", None]:
            self.assertRaises(
                Exception, lambda b=bad: ExponentiallyModifiedGaussian(sd=b)
            )
            self.assertRaises(
                Exception, lambda b=bad: ExponentiallyModifiedGaussian(rate=b)
            )
        self.assertRaises(Exception, lambda: ExponentiallyModifiedGaussian(mean="0"))

    def test_EMG_error_message_points_to_simpler_distribution(self):
        # sd = 0 and rate = 0 are the two limiting cases, and each has its own
        # distribution already; the message should say which one.
        with self.assertRaises(Exception) as caught:
            ExponentiallyModifiedGaussian(sd=0)
        self.assertIn("Exponential", str(caught.exception))
        with self.assertRaises(Exception) as caught:
            ExponentiallyModifiedGaussian(rate=0)
        self.assertIn("Normal", str(caught.exception))

    def test_EMG_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        ExponentiallyModifiedGaussian().draw()
        RV(ExponentiallyModifiedGaussian()).sim(100).plot()
        ExponentiallyModifiedGaussian().plot()
        plt.figure()
        ExponentiallyModifiedGaussian().plot(cdf=True)
        plt.figure()
        ExponentiallyModifiedGaussian(mean=0, sd=1, rate=0.05).plot()
        plt.close("all")


class TestGamma(unittest.TestCase):

    def test_Gamma_shape_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape=-5, rate=40))

    def test_Gamma_rate_error(self):
        self.assertRaises(Exception, lambda: Gamma(shape=4, rate=-10))

    def test_Gamma_to_Exponential(self):
        seed(42)
        X = Gamma(shape=1, rate=1 / 0.9)
        sims = X.sim(Nsim)
        cdf = stats.expon(scale=0.9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_reshape(self):
        seed(42)
        X = RV(Gamma(shape=9, scale=4))
        sims = (X * 8).sim(Nsim)
        cdf = stats.gamma(scale=4 * 8, a=9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_additive(self):
        seed(42)
        X, Y = RV(Gamma(shape=10, scale=0.5) * Gamma(shape=8, scale=0.5))
        sims = (X + Y).sim(Nsim)
        cdf = stats.gamma(scale=0.5, a=18).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_to_Beta(self):
        seed(42)
        X, Y = RV(Gamma(shape=5, scale=8) * Gamma(shape=4, scale=8))
        sims = (X / (X + Y)).sim(Nsim)
        cdf = stats.beta(a=5, b=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_to_F(self):
        seed(42)
        X, Y = RV(Gamma(shape=2, rate=5) * Gamma(shape=4, rate=7))
        sims = ((4 * 5 * X) / (2 * 7 * Y)).sim(Nsim)
        cdf = stats.f(dfn=2 * 2, dfd=2 * 4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_to_ChiSquare(self):
        seed(42)
        X = RV(Gamma(shape=10 / 2, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.chi2(df=10).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gamma_scale_param(self):
        seed(42)
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


class TestInverseGamma(unittest.TestCase):

    def test_InverseGamma_distributional(self):
        seed(42)
        X = RV(InverseGamma(shape=3, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.invgamma(3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_InverseGamma_mean_var_sd(self):
        X = InverseGamma(shape=3, scale=2)
        th = stats.invgamma(3, scale=2)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_InverseGamma_pdf(self):
        X = InverseGamma(shape=3, scale=2)
        for x in [0.5, 1, 3]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(stats.invgamma(3, scale=2).pdf(x))
            )

    def test_InverseGamma_is_reciprocal_of_gamma(self):
        # If X ~ Gamma(shape, rate=scale) then 1/X ~ InverseGamma(shape, scale).
        seed(42)
        X = RV(Gamma(shape=3, rate=2))
        sims = (1 / X).sim(Nsim)
        cdf = stats.invgamma(3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_InverseGamma_draw_is_scalar_in_support(self):
        seed(0)
        value = InverseGamma(shape=3, scale=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreater(float(value), 0.0)

    def test_InverseGamma_error_shape_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(Exception, lambda v=bad: InverseGamma(shape=v, scale=2))

    def test_InverseGamma_error_scale_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(Exception, lambda v=bad: InverseGamma(shape=3, scale=v))

    def test_InverseGamma_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        InverseGamma(shape=3, scale=2).draw()
        RV(InverseGamma(shape=3, scale=2)).sim(100).plot()
        InverseGamma(shape=3, scale=2).plot()
        plt.figure()
        InverseGamma(shape=3, scale=2).plot(cdf=True)
        plt.close("all")


class TestScaledInverseChiSquare(unittest.TestCase):

    def test_ScaledInverseChiSquare_distributional(self):
        seed(42)
        X = RV(ScaledInverseChiSquare(df=6, scale=2))
        sims = X.sim(Nsim)
        # equivalent inverse gamma: shape=df/2, scale=df*scale/2
        cdf = stats.invgamma(3, scale=6).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ScaledInverseChiSquare_mean_var_sd(self):
        X = ScaledInverseChiSquare(df=6, scale=2)
        # mean = df*scale/(df-2); var = 2 df^2 scale^2 / ((df-2)^2 (df-4))
        self.assertAlmostEqual(float(X.mean()), 6 * 2 / (6 - 2), places=6)
        self.assertAlmostEqual(
            float(X.var()), 2 * 6**2 * 2**2 / ((6 - 2) ** 2 * (6 - 4)), places=6
        )
        self.assertAlmostEqual(float(X.sd()), 3.0, places=6)

    def test_ScaledInverseChiSquare_is_inverse_gamma(self):
        # ScaledInverseChiSquare(df, scale) == InverseGamma(df/2, df*scale/2).
        X = ScaledInverseChiSquare(df=6, scale=2)
        self.assertIsInstance(X, InverseGamma)
        Y = InverseGamma(shape=3, scale=6)
        for x in [0.5, 2, 5, 12]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=9)

    def test_ScaledInverseChiSquare_pdf(self):
        X = ScaledInverseChiSquare(df=6, scale=2)
        for x in [0.5, 2, 5]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(stats.invgamma(3, scale=6).pdf(x))
            )

    def test_ScaledInverseChiSquare_chisquare_relationship(self):
        # If X ~ ScaledInvChiSq(df, scale), then df*scale/X ~ ChiSquare(df).
        seed(42)
        X = RV(ScaledInverseChiSquare(df=6, scale=2))
        sims = (6 * 2 / X).sim(Nsim)
        cdf = stats.chi2(df=6).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ScaledInverseChiSquare_draw_is_scalar_in_support(self):
        seed(0)
        value = ScaledInverseChiSquare(df=6, scale=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreater(float(value), 0.0)

    def test_ScaledInverseChiSquare_default_scale(self):
        X = ScaledInverseChiSquare(df=6)
        self.assertEqual(X.df, 6)
        self.assertEqual(X.scale, 1.0)

    def test_ScaledInverseChiSquare_error_df_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: ScaledInverseChiSquare(df=v, scale=2)
            )

    def test_ScaledInverseChiSquare_error_scale_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: ScaledInverseChiSquare(df=6, scale=v)
            )

    def test_ScaledInverseChiSquare_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all inherited from InverseGamma / base class.
        ScaledInverseChiSquare(df=6, scale=2).draw()
        RV(ScaledInverseChiSquare(df=6, scale=2)).sim(100).plot()
        ScaledInverseChiSquare(df=6, scale=2).plot()
        plt.figure()
        ScaledInverseChiSquare(df=6, scale=2).plot(cdf=True)
        plt.close("all")


class TestLogGamma(unittest.TestCase):

    def test_LogGamma_distributional(self):
        seed(42)
        X = RV(LogGamma(shape=2))
        sims = X.sim(Nsim)
        cdf = stats.loggamma(c=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogGamma_is_log_of_Gamma(self):
        # The defining property: log(Gamma) has this distribution. Note the
        # direction -- this is the opposite of LogNormal, where the variable's
        # own log is normal.
        seed(42)
        G = RV(Gamma(shape=2, scale=1))
        sims = G.apply(log).sim(Nsim)
        pval = stats.kstest(sims, stats.loggamma(c=2).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogGamma_exp_is_Gamma(self):
        # The same fact read the other way round: exponentiating gives a Gamma.
        seed(7)
        X = RV(LogGamma(shape=3))
        sims = X.apply(exp).sim(Nsim)
        pval = stats.kstest(sims, stats.gamma(a=3).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogGamma_mean_var_are_digamma_trigamma(self):
        from scipy.special import digamma, polygamma

        for shape in [0.5, 1.0, 2.0, 5.0]:
            X = LogGamma(shape=shape)
            self.assertAlmostEqual(float(X.mean()), float(digamma(shape)), places=8)
            self.assertAlmostEqual(float(X.var()), float(polygamma(1, shape)), places=8)

    def test_LogGamma_support_includes_negative_values(self):
        # Unlike LogNormal, a log-gamma variable ranges over all real numbers.
        X = LogGamma(shape=2)
        self.assertLess(float(X.quantile(0.01)), 0.0)
        self.assertGreater(float(X.pdf(-3)), 0.0)

    def test_LogGamma_is_left_skewed(self):
        for shape in [0.5, 1.0, 2.0]:
            sk = float(stats.loggamma(c=shape).stats(moments="s"))
            self.assertLess(sk, 0.0)

    def test_LogGamma_loc_scale_shift_and_stretch(self):
        from scipy.special import digamma

        X = LogGamma(shape=2, loc=3, scale=2)
        self.assertAlmostEqual(float(X.mean()), 3 + 2 * float(digamma(2)), places=8)

    def test_LogGamma_shape_one_mirrors_Gumbel(self):
        # shape=1 is the smallest-extreme-value distribution, which is the
        # reflection of this package's Gumbel (that one models maxima).
        X = LogGamma(shape=1)
        G = Gumbel(loc=0, scale=1)
        for x in [-1.5, -0.5, 0.0, 0.5, 1.5]:
            self.assertAlmostEqual(float(X.pdf(x)), float(G.pdf(-x)), places=8)

    def test_LogGamma_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(LogGamma(shape=2).draw(), Scalar)

    def test_LogGamma_defaults(self):
        X = LogGamma(shape=2)
        self.assertEqual(X.loc, 0)
        self.assertEqual(X.scale, 1)

    def test_LogGamma_invalid_shape_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: LogGamma(shape=b))

    def test_LogGamma_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: LogGamma(shape=2, scale=b))

    def test_LogGamma_invalid_loc_raises(self):
        self.assertRaises(Exception, lambda: LogGamma(shape=2, loc="a"))

    def test_LogGamma_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        LogGamma(shape=2).draw()
        RV(LogGamma(shape=2)).sim(100).plot()
        LogGamma(shape=2).plot()
        plt.figure()
        LogGamma(shape=2).plot(cdf=True)
        plt.close("all")


class TestInverseGaussian(unittest.TestCase):

    # The (mean, shape) pairs the formula checks sweep over.
    PAIRS = [(1.0, 1.0), (2.0, 3.0), (0.5, 4.0), (5.0, 0.5), (3.0, 10.0)]

    def test_InverseGaussian_distributional(self):
        seed(42)
        X = RV(InverseGaussian(mean=2, shape=3))
        sims = X.sim(Nsim)
        # scipy's own parameters, translated: mu = mean / shape, scale = shape.
        cdf = stats.invgauss(mu=2 / 3, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_InverseGaussian_mean_matches_parameter(self):
        for mean, shape in self.PAIRS:
            X = InverseGaussian(mean=mean, shape=shape)
            self.assertAlmostEqual(float(X.mean()), mean, places=8)

    def test_InverseGaussian_variance_is_mean_cubed_over_shape(self):
        for mean, shape in self.PAIRS:
            X = InverseGaussian(mean=mean, shape=shape)
            self.assertAlmostEqual(float(X.var()), mean**3 / shape, places=8)

    def test_InverseGaussian_mean_param_agrees_with_mean_method(self):
        # `mean` is the method inherited from Distribution; the parameter is
        # kept under `mean_param` so the two do not collide.
        X = InverseGaussian(mean=2, shape=3)
        self.assertEqual(X.mean_param, 2)
        self.assertAlmostEqual(float(X.mean()), X.mean_param, places=8)

    def test_InverseGaussian_pdf_matches_scipy(self):
        for mean, shape in self.PAIRS:
            X = InverseGaussian(mean=mean, shape=shape)
            ref = stats.invgauss(mu=mean / shape, scale=shape)
            for x in [0.1, 0.5, 1.0, 2.0, 5.0]:
                self.assertAlmostEqual(float(X.pdf(x)), float(ref.pdf(x)), places=8)

    @staticmethod
    def _max_normal_gap(mean, shape):
        """Largest cdf gap to Normal(mean, mean ** 3 / shape), over +/- 2 sd."""
        X = InverseGaussian(mean=mean, shape=shape)
        sd = (mean**3 / shape) ** 0.5
        normal = Normal(mean=mean, sd=sd)
        return max(
            abs(float(X.cdf(mean + z * sd)) - float(normal.cdf(mean + z * sd)))
            for z in [-2, -1, 0, 1, 2]
        )

    def test_InverseGaussian_approaches_normal_as_shape_grows(self):
        # With the mean fixed, a large shape shrinks the variance and the skew
        # washes out, leaving Normal(mean, mean ** 3 / shape). This is exact
        # arithmetic, not a simulation, so the gap below is reproducible: it is
        # about 2.8e-4 at shape = 1e6, comfortably inside the tolerance.
        self.assertLess(self._max_normal_gap(2.0, 1_000_000), 1e-3)

    def test_InverseGaussian_normal_gap_shrinks_with_shape(self):
        # The limit itself, not just one point on the way to it: the gap has to
        # keep closing as the shape grows.
        gaps = [self._max_normal_gap(2.0, s) for s in [1e3, 1e4, 1e5, 1e6, 1e7]]
        for earlier, later in zip(gaps, gaps[1:]):
            self.assertLess(later, earlier)

    def test_InverseGaussian_defaults(self):
        X = InverseGaussian()
        self.assertEqual(X.mean_param, 1.0)
        self.assertEqual(X.shape, 1.0)

    def test_InverseGaussian_is_positive(self):
        X = InverseGaussian(mean=2, shape=3)
        self.assertEqual(float(X.cdf(0)), 0.0)
        self.assertGreater(float(X.quantile(0.001)), 0.0)

    def test_InverseGaussian_is_right_skewed(self):
        for mean, shape in self.PAIRS:
            sk = float(stats.invgauss(mu=mean / shape, scale=shape).stats(moments="s"))
            self.assertGreater(sk, 0.0)

    def test_InverseGaussian_xlim_starts_at_zero(self):
        # Support is (0, inf): the fixed lower bound is kept, the unbounded
        # upper end is cut at the 0.999 quantile. No per-distribution code.
        X = InverseGaussian(mean=2, shape=3)
        low, high = X.xlim
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, float(X.quantile(0.999)), places=8)

    def test_InverseGaussian_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(InverseGaussian(mean=2, shape=3).draw(), Scalar)

    def test_InverseGaussian_invalid_mean_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: InverseGaussian(mean=b))

    def test_InverseGaussian_invalid_shape_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: InverseGaussian(shape=b))

    def test_InverseGaussian_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        InverseGaussian(mean=2, shape=3).draw()
        RV(InverseGaussian(mean=2, shape=3)).sim(100).plot()
        InverseGaussian(mean=2, shape=3).plot()
        plt.figure()
        InverseGaussian(mean=2, shape=3).plot(cdf=True)
        plt.close("all")


class TestBeta(unittest.TestCase):

    def test_Beta_error_shape1(self):
        self.assertRaises(Exception, lambda: Beta(shape1=-10, shape2=3))

    def test_Beta_error_shape2(self):
        self.assertRaises(Exception, lambda: Beta(shape1=3, shape2=-10))

    def test_Beta_to_Uniform(self):
        seed(42)
        X = Beta(shape1=1, shape2=1)
        sims = X.sim(Nsim)
        cdf = stats.uniform(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_symmetry(self):
        seed(42)
        X = RV(Beta(shape1=4, shape2=5))
        sims = (1 - X).sim(Nsim)
        cdf = stats.beta(a=5, b=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_to_Exponential(self):
        seed(42)
        X = RV(Beta(shape1=0.7, shape2=1))
        sims = (-log(X)).sim(Nsim)
        cdf = stats.expon(scale=1 / 0.7).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_to_F(self):
        seed(42)
        X = RV(Beta(shape1=10 / 2, shape2=12 / 2))
        sims = (12 * X / (10 * (1 - X))).sim(Nsim)
        cdf = stats.f(dfn=10, dfd=12).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_mean_pdf(self):
        X = Beta(shape1=2, shape2=5)
        self.assertAlmostEqual(float(X.mean()), 2 / 7)
        self.assertAlmostEqual(float(X.pdf(0.5)), stats.beta(a=2, b=5).pdf(0.5))

    # --- xmin / xmax: the beta stretched onto an interval other than [0, 1] ---

    def test_Beta_default_bounds_are_zero_one(self):
        X = Beta(shape1=2, shape2=5)
        self.assertEqual(X.xmin, 0.0)
        self.assertEqual(X.xmax, 1.0)

    def test_Beta_defaults_reproduce_standard_beta(self):
        # The bounds are new, so the default case has to be bit-for-bit what
        # it was: loc=0, scale=1 is scipy's no-op.
        for a, b in [(1, 1), (2, 5), (0.5, 0.5), (3, 1), (7.5, 2.25)]:
            X = Beta(shape1=a, shape2=b)
            ref = stats.beta(a=a, b=b)
            self.assertAlmostEqual(float(X.mean()), float(ref.mean()), places=12)
            self.assertAlmostEqual(float(X.var()), float(ref.var()), places=12)
            for x in [0.01, 0.25, 0.5, 0.75, 0.99]:
                self.assertAlmostEqual(float(X.pdf(x)), float(ref.pdf(x)), places=12)
                self.assertAlmostEqual(float(X.cdf(x)), float(ref.cdf(x)), places=12)

    def test_Beta_with_bounds_matches_hand_stretched_beta(self):
        # The defining identity: xmin + (xmax - xmin) * Beta(a, b) has this
        # distribution. Compared against a stretched RV, not against scipy, so
        # the test states the identity rather than restating the loc/scale call.
        xmin, xmax = 10, 20
        seed(42)
        stretched = (xmin + (xmax - xmin) * RV(Beta(shape1=2, shape2=5))).sim(Nsim)
        cdf = Beta(shape1=2, shape2=5, xmin=xmin, xmax=xmax).cdf
        pval = stats.kstest(stretched, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_bounds_shift_and_stretch_the_mean(self):
        for a, b, xmin, xmax in [
            (1, 1, 2, 4),
            (2, 5, 10, 20),
            (3, 1, -5, 5),
            (0.5, 0.5, 0, 100),
        ]:
            X = Beta(shape1=a, shape2=b, xmin=xmin, xmax=xmax)
            expected = xmin + (xmax - xmin) * a / (a + b)
            self.assertAlmostEqual(float(X.mean()), expected, places=8)
            # Variance stretches by the square of the width.
            standard_var = a * b / ((a + b) ** 2 * (a + b + 1))
            self.assertAlmostEqual(
                float(X.var()), (xmax - xmin) ** 2 * standard_var, places=8
            )

    def test_Beta_doctest_mean_is_three(self):
        # Beta(1, 1) is Uniform(0, 1) with mean 0.5, so on [2, 4] the mean is
        # 2 + 2 * 0.5 = 3. This is the value the class docstring shows.
        self.assertEqual(float(Beta(1, 1, xmin=2, xmax=4).mean()), 3.0)

    def test_Beta_support_follows_bounds(self):
        # No per-distribution window code: the support comes from scipy, so
        # xlim follows the bounds automatically.
        X = Beta(shape1=2, shape2=5, xmin=10, xmax=20)
        self.assertEqual(X._support(), (10.0, 20.0))
        self.assertEqual(X.xlim, (10.0, 20.0))

    def test_Beta_bounds_can_be_negative(self):
        X = Beta(shape1=2, shape2=2, xmin=-3, xmax=-1)
        self.assertEqual(X._support(), (-3.0, -1.0))
        self.assertAlmostEqual(float(X.mean()), -2.0, places=8)

    def test_Beta_error_xmax_not_greater_than_xmin(self):
        for xmin, xmax in [(1, 1), (5, 2), (0, -1)]:
            self.assertRaises(
                Exception,
                lambda lo=xmin, hi=xmax: Beta(shape1=2, shape2=5, xmin=lo, xmax=hi),
            )

    def test_Beta_error_non_numeric_bounds(self):
        # A bad type must produce the friendly message, not a TypeError raised
        # from inside the xmax > xmin comparison.
        for bad in ["a", None, [0, 1]]:
            with self.assertRaises(Exception) as cm:
                Beta(shape1=2, shape2=5, xmin=bad)
            self.assertIn("xmin must be a number", str(cm.exception))
            with self.assertRaises(Exception) as cm:
                Beta(shape1=2, shape2=5, xmax=bad)
            self.assertIn("xmax must be a number", str(cm.exception))

    def test_Beta_very_narrow_interval_is_well_behaved(self):
        # A tiny width is legal; check nothing degenerates into nan or escapes
        # the interval.
        xmin, width = 1.0, 1e-10
        X = Beta(shape1=2, shape2=5, xmin=xmin, xmax=xmin + width)
        self.assertAlmostEqual(float(X.mean()), xmin + width * 2 / 7, places=12)
        self.assertFalse(math.isnan(float(X.var())))
        self.assertGreater(float(X.var()), 0.0)
        seed(0)
        sims = np.array(RV(X).sim(1000), dtype=float)
        self.assertFalse(np.isnan(sims).any())
        self.assertTrue(((sims >= xmin) & (sims <= xmin + width)).all())

    def test_Beta_plots_with_non_default_bounds(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        Beta(shape1=2, shape2=5, xmin=10, xmax=20).draw()
        RV(Beta(shape1=2, shape2=5, xmin=10, xmax=20)).sim(100).plot()
        Beta(shape1=2, shape2=5, xmin=10, xmax=20).plot()
        plt.figure()
        Beta(shape1=2, shape2=5, xmin=10, xmax=20).plot(cdf=True)
        plt.close("all")

    def test_Beta_internal_uses_are_unaffected(self):
        # Beta is reused inside Dirichlet (and stretched the same way by PERT),
        # neither of which passes bounds. Pin that they still get [0, 1].
        marginal = Dirichlet([2.0, 3.0, 5.0])._marginal_1d(0)
        self.assertIsInstance(marginal, Beta)
        self.assertEqual((marginal.xmin, marginal.xmax), (0.0, 1.0))
        self.assertEqual(marginal._support(), (0.0, 1.0))


class TestPERT(unittest.TestCase):

    def test_PERT_distributional(self):
        seed(42)
        X = PERT(low=1, mode=2, high=10)
        sims = X.sim(Nsim)
        cdf = stats.beta(X.alpha, X.beta, loc=1, scale=9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_PERT_mean_is_classic_formula(self):
        # The whole point of PERT: mean = (low + 4 * mode + high) / 6.
        for low, mode, high in [(1, 2, 10), (0, 5, 10), (0, 9, 10), (-5, 0, 5)]:
            X = PERT(low=low, mode=mode, high=high)
            self.assertAlmostEqual(
                float(X.mean()), (low + 4 * mode + high) / 6, places=8
            )

    def test_PERT_mean_uses_weight(self):
        # A general weight generalizes the /6 formula to /(weight + 2).
        low, mode, high, w = 0, 2, 10, 7.0
        X = PERT(low=low, mode=mode, high=high, weight=w)
        self.assertAlmostEqual(
            float(X.mean()), (low + w * mode + high) / (w + 2), places=8
        )

    def test_PERT_density_peaks_at_mode(self):
        # The shape parameters are chosen so the peak lands exactly on mode.
        for low, mode, high in [(1, 2, 10), (0, 9, 10), (-5, 0, 5)]:
            X = PERT(low=low, mode=mode, high=high)
            xs = np.linspace(low, high, 200001)
            self.assertAlmostEqual(float(xs[np.argmax(X.pdf(xs))]), mode, places=3)

    def test_PERT_shape_parameters(self):
        X = PERT(low=0, mode=2, high=10, weight=4)
        self.assertAlmostEqual(X.alpha, 1 + 4 * (2 - 0) / 10, places=10)
        self.assertAlmostEqual(X.beta, 1 + 4 * (10 - 2) / 10, places=10)

    def test_PERT_symmetric_case_is_Beta_3_3(self):
        # low=0, mode=0.5, high=1 with the default weight is exactly Beta(3, 3).
        X = PERT(low=0, mode=0.5, high=1)
        B = Beta(shape1=3, shape2=3)
        self.assertAlmostEqual(X.alpha, 3.0, places=10)
        self.assertAlmostEqual(X.beta, 3.0, places=10)
        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            self.assertAlmostEqual(float(X.pdf(x)), float(B.pdf(x)), places=8)
            self.assertAlmostEqual(float(X.cdf(x)), float(B.cdf(x)), places=8)

    def test_PERT_mode_at_endpoints_allowed(self):
        # mode == low gives alpha = 1; mode == high gives beta = 1. Both are
        # valid Beta distributions and need no special-casing.
        X = PERT(low=0, mode=0, high=10)
        self.assertAlmostEqual(X.alpha, 1.0, places=10)
        Y = PERT(low=0, mode=10, high=10)
        self.assertAlmostEqual(Y.beta, 1.0, places=10)

    def test_PERT_weight_increases_concentration(self):
        sds = [float(PERT(0, 5, 10, weight=w).sd()) for w in [1, 2, 4, 8, 20]]
        self.assertEqual(sds, sorted(sds, reverse=True))

    def test_PERT_draws_within_bounds(self):
        seed(0)
        sims = RV(PERT(low=1, mode=2, high=10)).sim(1000)
        values = [float(v) for v in sims]
        self.assertTrue(all(1 <= v <= 10 for v in values))

    def test_PERT_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(PERT(low=1, mode=2, high=10).draw(), Scalar)

    def test_PERT_xlim_is_full_support(self):
        self.assertEqual(PERT(low=1, mode=2, high=10).xlim, (1, 10))

    def test_PERT_default_weight_is_four(self):
        self.assertEqual(PERT(low=0, mode=1, high=2).weight, 4.0)

    def test_PERT_invalid_bounds_raise(self):
        # high must be strictly greater than low.
        self.assertRaises(Exception, lambda: PERT(low=5, mode=5, high=5))
        self.assertRaises(Exception, lambda: PERT(low=10, mode=5, high=1))

    def test_PERT_mode_outside_bounds_raises(self):
        self.assertRaises(Exception, lambda: PERT(low=0, mode=-1, high=10))
        self.assertRaises(Exception, lambda: PERT(low=0, mode=11, high=10))

    def test_PERT_invalid_weight_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: PERT(0, 5, 10, weight=b))

    def test_PERT_non_numeric_raises(self):
        self.assertRaises(Exception, lambda: PERT(low="a", mode=5, high=10))
        self.assertRaises(Exception, lambda: PERT(low=0, mode="a", high=10))
        self.assertRaises(Exception, lambda: PERT(low=0, mode=5, high="a"))

    def test_PERT_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        PERT(1, 2, 10).draw()
        RV(PERT(1, 2, 10)).sim(100).plot()
        PERT(1, 2, 10).plot()
        plt.figure()
        PERT(1, 2, 10).plot(cdf=True)
        plt.close("all")


class TestTriangular(unittest.TestCase):

    def test_Triangular_distributional(self):
        seed(42)
        X = Triangular(low=1, mode=2, high=10)
        sims = X.sim(Nsim)
        cdf = stats.triang((2 - 1) / 9, loc=1, scale=9).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Triangular_mean_is_average_of_three_points(self):
        for low, mode, high in [(1, 2, 10), (0, 5, 10), (0, 9, 10), (-5, 0, 5)]:
            X = Triangular(low=low, mode=mode, high=high)
            self.assertAlmostEqual(float(X.mean()), (low + mode + high) / 3, places=8)

    def test_Triangular_variance_closed_form(self):
        for a, b, c in [(1, 2, 10), (0, 5, 10), (-5, 0, 5), (0, 0, 10)]:
            X = Triangular(low=a, mode=b, high=c)
            expected = (a * a + b * b + c * c - a * b - a * c - b * c) / 18
            self.assertAlmostEqual(float(X.var()), expected, places=8)

    def test_Triangular_peak_height_is_two_over_span(self):
        # The triangle's area must be 1, so its height is 2 / (high - low)
        # regardless of where the mode sits.
        for low, mode, high in [(1, 2, 10), (0, 5, 10), (0, 9, 10)]:
            X = Triangular(low=low, mode=mode, high=high)
            self.assertAlmostEqual(float(X.pdf(mode)), 2 / (high - low), places=8)

    def test_Triangular_density_peaks_at_mode(self):
        for low, mode, high in [(1, 2, 10), (0, 9, 10), (-5, 0, 5)]:
            X = Triangular(low=low, mode=mode, high=high)
            xs = np.linspace(low, high, 200001)
            self.assertAlmostEqual(float(xs[np.argmax(X.pdf(xs))]), mode, places=3)

    def test_Triangular_sides_are_linear(self):
        # Halfway up the rising side, the density should be half its peak.
        X = Triangular(low=0, mode=4, high=10)
        peak = 2 / 10
        self.assertAlmostEqual(float(X.pdf(2)), peak * 0.5, places=8)
        # And on the falling side, a third of the way back down from the mode.
        self.assertAlmostEqual(float(X.pdf(8)), peak * (10 - 8) / (10 - 4), places=8)

    def test_Triangular_mode_at_endpoints_allowed(self):
        # Right triangles: sloping only one way.
        X = Triangular(low=0, mode=0, high=10)
        self.assertAlmostEqual(float(X.mean()), 10 / 3, places=8)
        self.assertAlmostEqual(float(X.pdf(0)), 0.2, places=8)
        Y = Triangular(low=0, mode=10, high=10)
        self.assertAlmostEqual(float(Y.mean()), 20 / 3, places=8)
        self.assertAlmostEqual(float(Y.pdf(10)), 0.2, places=8)

    def test_Triangular_more_spread_than_PERT(self):
        # Triangular weights the extremes as heavily as the mode, so on the
        # same three numbers it is the wider of the two.
        low, mode, high = 0, 5, 10
        t = float(Triangular(low=low, mode=mode, high=high).sd())
        p = float(PERT(low=low, mode=mode, high=high).sd())
        self.assertGreater(t, p)

    def test_Triangular_draws_within_bounds(self):
        seed(0)
        sims = RV(Triangular(low=1, mode=2, high=10)).sim(1000)
        self.assertTrue(all(1 <= float(v) <= 10 for v in sims))

    def test_Triangular_draw_is_scalar(self):
        seed(0)
        self.assertIsInstance(Triangular(low=1, mode=2, high=10).draw(), Scalar)

    def test_Triangular_xlim_is_full_support(self):
        self.assertEqual(Triangular(low=1, mode=2, high=10).xlim, (1, 10))

    def test_Triangular_invalid_bounds_raise(self):
        self.assertRaises(Exception, lambda: Triangular(low=5, mode=5, high=5))
        self.assertRaises(Exception, lambda: Triangular(low=10, mode=5, high=1))

    def test_Triangular_mode_outside_bounds_raises(self):
        self.assertRaises(Exception, lambda: Triangular(low=0, mode=-1, high=10))
        self.assertRaises(Exception, lambda: Triangular(low=0, mode=11, high=10))

    def test_Triangular_non_numeric_raises(self):
        self.assertRaises(Exception, lambda: Triangular(low="a", mode=5, high=10))
        self.assertRaises(Exception, lambda: Triangular(low=0, mode="a", high=10))
        self.assertRaises(Exception, lambda: Triangular(low=0, mode=5, high="a"))

    def test_Triangular_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Triangular(1, 2, 10).draw()
        RV(Triangular(1, 2, 10)).sim(100).plot()
        Triangular(1, 2, 10).plot()
        plt.figure()
        Triangular(1, 2, 10).plot(cdf=True)
        plt.close("all")


class TestKumaraswamy(unittest.TestCase):
    """The Kumaraswamy distribution on [0, 1].

    The only distribution in the package with no ``scipy.stats`` backing, so
    the closed forms are checked against the definitions written out here
    rather than against another implementation.
    """

    # --- density, cdf and quantile match their closed forms ---

    def test_Kumaraswamy_pdf_formula(self):
        # pdf(x) = a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)
        a, b = 2.5, 3.0
        X = Kumaraswamy(shape1=a, shape2=b)
        for x in [0.05, 0.25, 0.5, 0.75, 0.95]:
            expected = a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)
            self.assertAlmostEqual(float(X.pdf(x)), expected, places=12)

    def test_Kumaraswamy_cdf_formula(self):
        # cdf(x) = 1 - (1 - x ** a) ** b
        a, b = 2.0, 4.0
        X = Kumaraswamy(shape1=a, shape2=b)
        for x in [0.0, 0.1, 0.5, 0.9, 1.0]:
            self.assertAlmostEqual(float(X.cdf(x)), 1 - (1 - x**a) ** b, places=12)

    def test_Kumaraswamy_quantile_formula(self):
        # quantile(q) = (1 - (1 - q) ** (1 / b)) ** (1 / a)
        a, b = 3.0, 1.5
        X = Kumaraswamy(shape1=a, shape2=b)
        for q in [0.01, 0.25, 0.5, 0.75, 0.99]:
            expected = (1 - (1 - q) ** (1 / b)) ** (1 / a)
            self.assertAlmostEqual(float(X.quantile(q)), expected, places=12)

    def test_Kumaraswamy_quantile_inverts_cdf(self):
        X = Kumaraswamy(shape1=2.5, shape2=3.0)
        for q in [0.05, 0.3, 0.5, 0.8, 0.95]:
            self.assertAlmostEqual(float(X.cdf(X.quantile(q))), q, places=10)

    def test_Kumaraswamy_pdf_integrates_to_one(self):
        # Numerically integrating the density is an independent check that
        # the a * b out front is the right normalizing constant.
        from scipy.integrate import quad

        for a, b in [(0.5, 0.5), (2, 3), (5, 1.2)]:
            X = Kumaraswamy(shape1=a, shape2=b)
            total, _ = quad(lambda x: float(X.pdf(x)), 0, 1)
            self.assertAlmostEqual(total, 1.0, places=6)

    # --- support ---

    def test_Kumaraswamy_support_is_zero_to_one(self):
        X = Kumaraswamy(shape1=2, shape2=3)
        self.assertEqual(float(X.pdf(-0.1)), 0.0)
        self.assertEqual(float(X.pdf(1.1)), 0.0)
        self.assertEqual(float(X.cdf(0)), 0.0)
        self.assertEqual(float(X.cdf(1)), 1.0)
        self.assertEqual(float(X.quantile(0)), 0.0)
        self.assertEqual(float(X.quantile(1)), 1.0)

    def test_Kumaraswamy_xlim_is_full_support(self):
        # Bounded at both ends, so the default window is all of [0, 1] with
        # no probability trimmed, like Beta.
        self.assertEqual(Kumaraswamy(shape1=2, shape2=5).xlim, (0, 1))

    def test_Kumaraswamy_density_infinite_at_edges(self):
        # With a < 1 the density blows up at 0, and with b < 1 at 1. Both
        # are correct, and neither should raise or warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            edges = Kumaraswamy(shape1=0.5, shape2=0.5).pdf(np.array([0.0, 1.0]))
        self.assertTrue(np.isinf(edges).all())

    # --- moments ---

    def test_Kumaraswamy_moments_are_beta_integrals(self):
        # E[X ** n] = b * B(1 + n / a, b), from substituting t = x ** a.
        a, b = 2.0, 3.0
        X = Kumaraswamy(shape1=a, shape2=b)
        first = b * special.beta(1 + 1 / a, b)
        second = b * special.beta(1 + 2 / a, b)
        self.assertAlmostEqual(float(X.mean()), first, places=10)
        self.assertAlmostEqual(float(X.var()), second - first**2, places=10)
        self.assertAlmostEqual(float(X.sd()), math.sqrt(second - first**2), places=10)

    def test_Kumaraswamy_mean_var_known_values(self):
        # a = b = 2: E[X] = 2 * B(3/2, 2) = 8 / 15 and E[X ** 2] = 2 * B(2, 2)
        # = 1 / 3, both worked out by hand.
        X = Kumaraswamy(shape1=2, shape2=2)
        self.assertAlmostEqual(float(X.mean()), 8 / 15, places=12)
        self.assertAlmostEqual(float(X.var()), 1 / 3 - (8 / 15) ** 2, places=12)

    def test_Kumaraswamy_median_formula(self):
        a, b = 2.0, 2.0
        median = (1 - 2 ** (-1 / b)) ** (1 / a)
        self.assertAlmostEqual(
            float(Kumaraswamy(shape1=a, shape2=b).median()), median, places=12
        )

    def test_Kumaraswamy_larger_shape1_shifts_right_larger_shape2_shifts_left(self):
        base = float(Kumaraswamy(shape1=2, shape2=2).mean())
        self.assertGreater(float(Kumaraswamy(shape1=5, shape2=2).mean()), base)
        self.assertLess(float(Kumaraswamy(shape1=2, shape2=5).mean()), base)

    # --- relationships with distributions already in Symbulate ---

    def test_Kumaraswamy_shape2_one_is_Beta(self):
        # b = 1 leaves cdf(x) = x ** a, which is exactly Beta(a, 1).
        a = 3.0
        X, Y = Kumaraswamy(shape1=a, shape2=1), Beta(shape1=a, shape2=1)
        for x in [0.1, 0.4, 0.7, 0.95]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=10)
            self.assertAlmostEqual(float(X.cdf(x)), float(Y.cdf(x)), places=10)
        self.assertAlmostEqual(float(X.mean()), float(Y.mean()), places=10)

    def test_Kumaraswamy_shape1_one_is_Beta(self):
        # a = 1 leaves cdf(x) = 1 - (1 - x) ** b, which is exactly Beta(1, b).
        b = 4.0
        X, Y = Kumaraswamy(shape1=1, shape2=b), Beta(shape1=1, shape2=b)
        for x in [0.1, 0.4, 0.7, 0.95]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=10)
            self.assertAlmostEqual(float(X.cdf(x)), float(Y.cdf(x)), places=10)
        self.assertAlmostEqual(float(X.mean()), float(Y.mean()), places=10)

    def test_Kumaraswamy_one_one_is_Uniform(self):
        X = Kumaraswamy(shape1=1, shape2=1)
        for x in [0.1, 0.5, 0.9]:
            self.assertAlmostEqual(float(X.pdf(x)), 1.0, places=12)
            self.assertAlmostEqual(float(X.cdf(x)), x, places=12)
        self.assertAlmostEqual(float(X.mean()), 0.5, places=12)

    def test_Kumaraswamy_is_not_Beta_in_general(self):
        # Only a = 1 and b = 1 coincide with a Beta. Guard against anyone
        # "simplifying" this into a Beta with the same two parameters.
        X, Y = Kumaraswamy(shape1=2, shape2=3), Beta(shape1=2, shape2=3)
        self.assertNotAlmostEqual(float(X.mean()), float(Y.mean()), places=3)
        self.assertNotAlmostEqual(float(X.pdf(0.5)), float(Y.pdf(0.5)), places=3)

    def test_Kumaraswamy_power_shape1_is_Beta(self):
        # If X is Kumaraswamy(a, b), then X ** a is Beta(1, b): the general
        # link between the two families.
        seed(42)
        a, b = 2.5, 3.0
        X = RV(Kumaraswamy(shape1=a, shape2=b))
        sims = (X**a).sim(Nsim)
        pval = stats.kstest(sims, stats.beta(a=1, b=b).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Kumaraswamy_from_Uniform_inverse_cdf(self):
        # If U is Uniform(0, 1), then (1 - (1 - U) ** (1 / b)) ** (1 / a) is
        # Kumaraswamy(a, b) -- the inverse-cdf construction.
        seed(42)
        a, b = 2.0, 4.0
        U = RV(Uniform(0, 1))
        sims = ((1 - (1 - U) ** (1 / b)) ** (1 / a)).sim(Nsim)
        pval = stats.kstest(sims, Kumaraswamy(shape1=a, shape2=b).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Kumaraswamy_a_one_matches_Beta_sims(self):
        seed(42)
        sims = Kumaraswamy(shape1=1, shape2=3).sim(Nsim)
        pval = stats.kstest(sims, stats.beta(a=1, b=3).cdf).pvalue
        self.assertTrue(pval > 0.01)

    # --- sampling ---

    def test_Kumaraswamy_distributional(self):
        seed(42)
        X = Kumaraswamy(shape1=2.5, shape2=3.0)
        pval = stats.kstest(X.sim(Nsim), X.cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Kumaraswamy_sim_stays_in_support(self):
        seed(42)
        sims = RV(Kumaraswamy(shape1=0.5, shape2=0.5)).sim(Nsim)
        self.assertTrue(all(0 <= sim <= 1 for sim in sims))

    def test_Kumaraswamy_draw_is_scalar_in_support(self):
        seed(0)
        value = Kumaraswamy(shape1=2, shape2=3).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)
        self.assertLessEqual(float(value), 1.0)

    def test_Kumaraswamy_sample_mean_near_theoretical(self):
        seed(42)
        X = Kumaraswamy(shape1=2, shape2=3)
        sample_mean = RV(X).sim(Nsim).mean()
        self.assertAlmostEqual(float(sample_mean), float(X.mean()), places=2)

    # --- reproducibility ---

    def test_Kumaraswamy_same_seed_gives_same_sims(self):
        seed(2024)
        first = list(RV(Kumaraswamy(shape1=2, shape2=3)).sim(500))
        seed(2024)
        second = list(RV(Kumaraswamy(shape1=2, shape2=3)).sim(500))
        self.assertEqual(first, second)

    def test_Kumaraswamy_different_seed_gives_different_sims(self):
        seed(1)
        first = list(RV(Kumaraswamy(shape1=2, shape2=3)).sim(500))
        seed(2)
        second = list(RV(Kumaraswamy(shape1=2, shape2=3)).sim(500))
        self.assertNotEqual(first, second)

    # --- parameter validation ---

    def test_Kumaraswamy_error_shape1_not_positive(self):
        for bad in [0, -2]:
            self.assertRaisesRegex(
                Exception,
                "shape1 must be a positive number",
                lambda v=bad: Kumaraswamy(shape1=v, shape2=3),
            )

    def test_Kumaraswamy_error_shape2_not_positive(self):
        for bad in [0, -2]:
            self.assertRaisesRegex(
                Exception,
                "shape2 must be a positive number",
                lambda v=bad: Kumaraswamy(shape1=3, shape2=v),
            )

    def test_Kumaraswamy_error_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "shape1 must be a positive number",
            lambda: Kumaraswamy(shape1="x", shape2=3),
        )
        self.assertRaisesRegex(
            Exception,
            "shape2 must be a positive number",
            lambda: Kumaraswamy(shape1=3, shape2=None),
        )

    def test_Kumaraswamy_stacks_shape1_and_shape2(self):
        # Both parameters wrong -> both mistakes reported at once.
        with self.assertRaises(Exception) as cm:
            Kumaraswamy(shape1=-1, shape2=0)
        message = str(cm.exception)
        self.assertIn("Invalid parameters:", message)
        self.assertIn("shape1 must be a positive number", message)
        self.assertIn("shape2 must be a positive number", message)

    def test_Kumaraswamy_boundary_parameters_accepted(self):
        # Small shapes and the flat a = b = 1 case are valid, not errors.
        Kumaraswamy(shape1=1, shape2=1)
        Kumaraswamy(shape1=0.1, shape2=0.1)

    def test_Kumaraswamy_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Kumaraswamy(shape1=2, shape2=3).draw()
        RV(Kumaraswamy(shape1=2, shape2=3)).sim(100).plot()
        Kumaraswamy(shape1=2, shape2=3).plot()
        plt.figure()
        Kumaraswamy(shape1=2, shape2=3).plot(cdf=True)
        plt.figure()
        Kumaraswamy(shape1=2, shape2=5).plot()
        Kumaraswamy(shape1=0.5, shape2=0.5).plot()  # infinite density at both edges
        plt.close("all")


class TestStudentT(unittest.TestCase):

    def test_StudentT_df_error(self):
        self.assertRaises(Exception, lambda: StudentT(df=0))

    def test_StudentT_to_Normal(self):
        seed(42)
        X = StudentT(df=Nsim)
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_ChiSquare_to_StudentT(self):
        seed(42)
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
        seed(42)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        StudentT(df=8, noncentrality=1.5).draw()
        RV(StudentT(df=8, noncentrality=1.5)).sim(100).plot()
        StudentT(df=8, noncentrality=1.5).plot()
        plt.figure()
        StudentT(df=8, noncentrality=1.5).plot(cdf=True)
        plt.close("all")


class TestSkewT(unittest.TestCase):

    def test_SkewT_shape1_error(self):
        self.assertRaises(Exception, lambda: SkewT(shape1=0, shape2=2))

    def test_SkewT_shape2_error(self):
        self.assertRaises(Exception, lambda: SkewT(shape1=2, shape2=-1))

    def test_SkewT_scale_error(self):
        self.assertRaises(Exception, lambda: SkewT(shape1=2, shape2=2, scale=0))

    def test_SkewT_loc_error(self):
        self.assertRaises(Exception, lambda: SkewT(shape1=2, shape2=2, loc="a"))

    def test_SkewT_distribution(self):
        seed(42)
        X = RV(SkewT(shape1=5, shape2=2, loc=1, scale=2))
        sims = X.sim(Nsim)
        cdf = stats.jf_skew_t(5, 2, loc=1, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_SkewT_pdf_matches_scipy(self):
        X = SkewT(shape1=5, shape2=2, loc=1, scale=2)
        th = stats.jf_skew_t(5, 2, loc=1, scale=2)
        for x in [-3, -1, 0, 1, 3, 6]:
            self.assertAlmostEqual(float(X.pdf(x)), float(th.pdf(x)), places=12)

    def test_SkewT_symmetric_equals_StudentT(self):
        # shape1 == shape2 == df / 2 is a symmetric Student's t with df d.o.f.
        X = SkewT(shape1=3, shape2=3)
        Y = StudentT(df=6)
        for x in [-2, -0.5, 0, 1, 2.5]:
            self.assertAlmostEqual(float(X.pdf(x)), float(Y.pdf(x)), places=12)

    def test_SkewT_symmetric_median_zero(self):
        X = SkewT(shape1=4, shape2=4)
        self.assertAlmostEqual(float(X.median()), 0.0, places=9)

    def test_SkewT_right_skew_mean_positive(self):
        # shape1 > shape2 puts the longer tail on the right: mean > median.
        X = SkewT(shape1=6, shape2=2)
        self.assertGreater(float(X.mean()), float(X.median()))

    def test_SkewT_left_skew_mean_negative(self):
        # shape1 < shape2 mirrors shape2 < shape1 about zero.
        right = SkewT(shape1=5, shape2=2)
        left = SkewT(shape1=2, shape2=5)
        self.assertAlmostEqual(float(left.mean()), -float(right.mean()), places=9)

    def test_SkewT_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        SkewT(shape1=5, shape2=2).draw()
        RV(SkewT(shape1=5, shape2=2)).sim(100).plot()
        SkewT(shape1=5, shape2=2).plot()
        plt.figure()
        SkewT(shape1=5, shape2=2).plot(cdf=True)
        plt.close("all")


class TestChiSquare(unittest.TestCase):

    def test_ChiSquare_error(self):
        self.assertRaises(Exception, lambda: ChiSquare(df=0.5))

    def test_ChiSquare_to_Gamma(self):
        seed(42)
        X = RV(ChiSquare(df=10))
        sims = X.sim(Nsim)
        cdf = stats.gamma(a=5, scale=1 / 0.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_to_F(self):
        seed(42)
        X, Y = RV(ChiSquare(df=3) * ChiSquare(df=5))
        sims = ((X / 3) / (Y / 5)).sim(Nsim)
        cdf = stats.f(dfn=3, dfd=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ChiSquare_to_Beta(self):
        seed(42)
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
        seed(42)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        RV(ChiSquare(df=4, noncentrality=3)).sim(100).plot()
        ChiSquare(df=4, noncentrality=3).plot()
        plt.figure()
        ChiSquare(df=4, noncentrality=3).plot(cdf=True)
        plt.close("all")


class TestF(unittest.TestCase):

    def test_F_error(self):
        self.assertRaises(Exception, lambda: F(dfN=0, dfD=5))

    def test_inverse_T(self):
        seed(42)
        X = RV(F(dfN=4, dfD=8))
        sims = (1 / X).sim(Nsim)
        cdf = stats.f(dfn=8, dfd=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_StudentT_to_F(self):
        seed(42)
        X = RV(StudentT(df=15))
        sims = (X**2).sim(Nsim)
        cdf = stats.f(dfn=1, dfd=15).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_F_to_Beta(self):
        seed(42)
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
        seed(42)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        RV(F(dfN=5, dfD=10, noncentrality=4)).sim(100).plot()
        F(dfN=5, dfD=10, noncentrality=4).plot()
        plt.figure()
        F(dfN=5, dfD=10, noncentrality=4).plot(cdf=True)
        plt.close("all")


class TestHotelling(unittest.TestCase):

    def test_Hotelling_dim_not_integer(self):
        self.assertRaises(Exception, lambda: Hotelling(dim=2.5, df=10))

    def test_Hotelling_dim_not_positive(self):
        self.assertRaises(Exception, lambda: Hotelling(dim=0, df=10))

    def test_Hotelling_df_too_small(self):
        # df must exceed dim - 1, so dim=3 needs df > 2.
        self.assertRaises(Exception, lambda: Hotelling(dim=3, df=2))

    def test_Hotelling_is_scaled_F(self):
        # T^2(p, nu) = c * F(p, nu - p + 1), c = nu * p / (nu - p + 1).
        dim, df = 3, 10
        c = df * dim / (df - dim + 1)
        X = Hotelling(dim=dim, df=df)
        th = stats.f(dfn=dim, dfd=df - dim + 1, scale=c)
        for x in [0.5, 2.0, 5.0, 12.0]:
            self.assertAlmostEqual(float(X.pdf(x)), float(th.pdf(x)), places=12)

    def test_Hotelling_scale_attribute(self):
        X = Hotelling(dim=3, df=10)
        self.assertAlmostEqual(X.scale, 10 * 3 / (10 - 3 + 1))

    def test_Hotelling_mean_matches_scaled_F(self):
        dim, df = 3, 10
        c = df * dim / (df - dim + 1)
        X = Hotelling(dim=dim, df=df)
        self.assertAlmostEqual(
            float(X.mean()), float(stats.f(dfn=dim, dfd=df - dim + 1, scale=c).mean())
        )

    def test_Hotelling_distribution(self):
        # Simulate T^2 = nu * Z' W^{-1} Z from a standard normal vector Z and
        # an independent Wishart scatter matrix W, and check it against the
        # Hotelling distribution.
        dim, df = 3, 8
        X = Hotelling(dim=dim, df=df)
        rng = np.random.default_rng(7)
        sims = np.empty(3000)
        for i in range(len(sims)):
            z = rng.standard_normal(dim)
            w = stats.wishart(df=df, scale=np.eye(dim)).rvs(random_state=rng)
            sims[i] = df * z @ np.linalg.inv(w) @ z
        pval = stats.kstest(sims, X.cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Hotelling_dim_one_is_studentt_squared(self):
        # With dim = 1 the multiplier is 1 and T^2 = F(1, df) = StudentT(df)^2.
        seed(42)
        X = RV(StudentT(df=9))
        sims = (X**2).sim(Nsim)
        pval = stats.kstest(sims, Hotelling(dim=1, df=9).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Hotelling_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        Hotelling(dim=3, df=10).draw()
        RV(Hotelling(dim=3, df=10)).sim(100).plot()
        Hotelling(dim=3, df=10).plot()
        plt.figure()
        Hotelling(dim=3, df=10).plot(cdf=True)
        plt.close("all")


class TestCauchy(unittest.TestCase):

    def test_Cauchy_mean(self):
        X = Cauchy()
        math.isnan(X.mean())

    def test_Cauchy_to_T(self):
        # Seed 42 is pathological for this heavy-tailed Cauchy KS test
        # (lands in the ~1% false-rejection region); use a robust seed.
        seed(0)
        X = RV(Cauchy())
        sims = X.sim(Nsim)
        cdf = stats.t(df=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Cauchy_inverse(self):
        # Seed 42 is pathological for this heavy-tailed Cauchy KS test
        # (lands in the ~1% false-rejection region); use a robust seed.
        seed(0)
        X = RV(Cauchy())
        sims = (1 / X).sim(Nsim)
        cdf = stats.cauchy(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Cauchy_additive(self):
        seed(42)
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
        seed(42)
        X = LogNormal(mu=10, sigma=5)
        sims = X.sim(Nsim).apply(log)
        cdf = stats.norm(loc=10, scale=5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Normal_to_LogNormal(self):
        seed(42)
        X = RV(Normal(mean=10, sd=5))
        sims = X.apply(exp).sim(Nsim)
        cdf = stats.lognorm(s=5, scale=exp(10)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogNormal_Product(self):
        seed(42)
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

    # --- the degenerate (sigma = 0) point mass supports the full API ---
    # Regression tests: this branch used to define only pdf/cdf/mean/var/sd/
    # median, so `quantile` and `** n` raised a bare AttributeError and `plot`
    # raised a bare TypeError (its scalar-only pdf could not take an array).

    def test_LogNormal_sigma_zero_quantile(self):
        X = LogNormal(mu=0, sigma=0)
        # Every quantile of a point mass is the point itself.
        for q in [0.0, 0.25, 0.5, 0.99, 1.0]:
            self.assertAlmostEqual(float(X.quantile(q)), 1.0)
        np.testing.assert_allclose(
            np.asarray(X.quantile(np.array([0.1, 0.5, 0.9])), dtype=float),
            [1.0, 1.0, 1.0],
        )
        # Outside [0, 1] scipy's ppf gives nan, so this does too.
        self.assertTrue(np.isnan(float(X.quantile(1.5))))

    def test_LogNormal_sigma_zero_power_draws(self):
        X = LogNormal(mu=2, sigma=0)
        values = list((X**5).draw())
        self.assertEqual(len(values), 5)
        for value in values:
            self.assertAlmostEqual(float(value), np.exp(2))

    def test_LogNormal_sigma_zero_pdf_cdf_accept_arrays(self):
        X = LogNormal(mu=0, sigma=0)
        np.testing.assert_allclose(
            np.asarray(X.pdf(np.array([0.5, 1.0, 2.0])), dtype=float), [0.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            np.asarray(X.cdf(np.array([0.5, 1.0, 2.0])), dtype=float), [0.0, 1.0, 1.0]
        )
        # A single number still comes back as a plain float, not an array.
        self.assertIsInstance(X.pdf(1.0), float)
        self.assertIsInstance(X.cdf(1.0), float)

    def test_LogNormal_sigma_zero_plot_does_not_raise(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        with warnings.catch_warnings():
            # A point mass has zero height on a continuous grid, so matplotlib
            # warns about identical y-limits. Rendering a point mass nicely is a
            # separate question; this test only pins that plot() no longer
            # raises a TypeError from a scalar-only pdf.
            warnings.simplefilter("ignore")
            plt.figure()
            try:
                LogNormal(mu=0, sigma=0).plot()
                plt.figure()
                LogNormal(mu=0, sigma=0).plot(cdf=True)
            finally:
                plt.close("all")


class TestPareto(unittest.TestCase):

    def test_Pareto_check_mean(self):
        x = stats.pareto(b=-3)
        math.isnan(x.mean())

    def test_Pareto_to_Exponential(self):
        seed(42)
        X = RV(Pareto(shape=1.5, scale=0.1))
        sims = (log(X / 0.1)).sim(Nsim)
        cdf = stats.expon(scale=1 / 1.5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Pareto_error_shape_nonpositive(self):
        self.assertRaises(Exception, lambda: Pareto(shape=0, scale=1))

    def test_Pareto_error_shape_negative(self):
        self.assertRaises(Exception, lambda: Pareto(shape=-1, scale=1))

    def test_Pareto_error_scale_nonpositive(self):
        self.assertRaises(Exception, lambda: Pareto(shape=2, scale=0))

    def test_Pareto_draw_above_scale(self):
        seed(42)
        X = RV(Pareto(shape=2, scale=3))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim >= 3 for sim in sims))

    def test_Pareto_distributional(self):
        seed(42)
        X = RV(Pareto(shape=2, scale=1))
        sims = X.sim(Nsim)
        cdf = stats.pareto(b=2, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)


class TestBurr(unittest.TestCase):

    def test_Burr_distributional(self):
        seed(42)
        X = RV(Burr(shape1=3, shape2=2, scale=1))
        sims = X.sim(Nsim)
        cdf = stats.burr12(3, 2, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Burr_mean_var_sd(self):
        X = Burr(shape1=3, shape2=2, scale=1)
        th = stats.burr12(3, 2, scale=1)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_Burr_pdf(self):
        X = Burr(shape1=3, shape2=2, scale=2)
        for x in [0.5, 1, 3]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(stats.burr12(3, 2, scale=2).pdf(x))
            )

    def test_Burr_shape2_one_is_loglogistic(self):
        # b = 1 is the log-logistic (Fisk) distribution.
        X = Burr(shape1=2.5, shape2=1, scale=1)
        for x in [0.3, 1, 4]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(stats.fisk(2.5).pdf(x)), places=9
            )

    def test_Burr_shape1_one_shifted_is_pareto(self):
        # a = 1 is a Pareto (Type II / Lomax); shifting by the scale gives a
        # Pareto (Type I) with the same tail exponent.
        seed(42)
        X = RV(Burr(shape1=1, shape2=2, scale=1))
        sims = (X + 1).sim(Nsim)
        cdf = stats.pareto(b=2, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Burr_draw_is_scalar_in_support(self):
        seed(0)
        value = Burr(shape1=3, shape2=2, scale=1).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_Burr_error_shape1_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: Burr(shape1=v, shape2=2, scale=1)
            )

    def test_Burr_error_shape2_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: Burr(shape1=2, shape2=v, scale=1)
            )

    def test_Burr_error_scale_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: Burr(shape1=2, shape2=2, scale=v)
            )

    def test_Burr_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class; a < 1 also
        # exercises the monotone-decreasing high-density x-window.
        Burr(shape1=3, shape2=2, scale=1).draw()
        RV(Burr(shape1=3, shape2=2, scale=1)).sim(100).plot()
        Burr(shape1=3, shape2=2, scale=1).plot()
        Burr(shape1=0.5, shape2=2, scale=1).plot()
        plt.figure()
        Burr(shape1=3, shape2=2, scale=1).plot(cdf=True)
        plt.close("all")


class TestLomax(unittest.TestCase):

    def test_Lomax_distributional(self):
        seed(42)
        X = RV(Lomax(shape=3, scale=1))
        sims = X.sim(Nsim)
        cdf = stats.lomax(3, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Lomax_mean_var_sd(self):
        X = Lomax(shape=3, scale=1)
        th = stats.lomax(3, scale=1)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_Lomax_pdf(self):
        X = Lomax(shape=3, scale=2)
        for x in [0, 1, 3]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(stats.lomax(3, scale=2).pdf(x))
            )

    def test_Lomax_is_burr_shape1_one(self):
        # Lomax(b, scale) is exactly the a = 1 special case of Burr.
        for x in [0.0, 0.5, 2.0]:
            self.assertAlmostEqual(
                float(Lomax(shape=2, scale=1).pdf(x)),
                float(Burr(shape1=1, shape2=2, scale=1).pdf(x)),
                places=9,
            )

    def test_Lomax_shifted_is_pareto(self):
        # Adding the scale to a Lomax gives a Pareto (Type I) with the same
        # tail exponent.
        seed(42)
        X = RV(Lomax(shape=3, scale=1))
        sims = (X + 1).sim(Nsim)
        cdf = stats.pareto(b=3, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Lomax_draw_is_scalar_in_support(self):
        seed(0)
        value = Lomax(shape=3, scale=1).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_Lomax_default_params(self):
        X = Lomax()
        self.assertEqual(X.shape, 1.0)
        self.assertEqual(X.scale, 1.0)

    def test_Lomax_error_shape_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(Exception, lambda v=bad: Lomax(shape=v, scale=1))

    def test_Lomax_error_scale_nonpositive(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(Exception, lambda v=bad: Lomax(shape=2, scale=v))

    def test_Lomax_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Lomax(shape=3, scale=1).draw()
        RV(Lomax(shape=3, scale=1)).sim(100).plot()
        Lomax(shape=3, scale=1).plot()
        plt.figure()
        Lomax(shape=3, scale=1).plot(cdf=True)
        plt.close("all")


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
        seed(42)
        A, B = RV(Normal(mean=0, var=1) * Normal(mean=0, var=1))
        sims = (A**2 + B**2).apply(sqrt).sim(Nsim)
        cdf = stats.rayleigh.cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Rayleigh_to_Chi(self):
        seed(42)
        X = RV(Rayleigh())
        sims = X.sim(Nsim)
        cdf = stats.chi(df=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Rayleigh_mean(self):
        X = Rayleigh()
        self.assertAlmostEqual(float(X.mean()), np.sqrt(np.pi / 2), places=5)

    def test_Rayleigh_distributional(self):
        seed(42)
        X = RV(Rayleigh())
        sims = X.sim(Nsim)
        cdf = stats.rayleigh.cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Rayleigh_default_scale_is_one(self):
        X = Rayleigh()
        self.assertEqual(X.scale, 1.0)

    def test_Rayleigh_distributional_nondefault_scale(self):
        seed(42)
        X = RV(Rayleigh(scale=3))
        sims = X.sim(Nsim)
        cdf = stats.rayleigh(scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Rayleigh_mean_var_across_scales(self):
        for s in [0.5, 1, 2, 5]:
            X = Rayleigh(scale=s)
            th = stats.rayleigh(scale=s)
            self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
            self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)

    def test_Rayleigh_stretch_identity(self):
        # scale * Rayleigh() ~ Rayleigh(scale)
        seed(42)
        X = RV(Rayleigh())
        sims = (3 * X).sim(Nsim)
        cdf = stats.rayleigh(scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Rayleigh_invalid_scale_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Rayleigh(scale=b))

    def test_Rayleigh_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        Rayleigh(2).draw()
        RV(Rayleigh(2)).sim(100).plot()
        Rayleigh(2).plot()
        plt.figure()
        Rayleigh(2).plot(cdf=True)
        plt.close("all")


class TestHalfNormal(unittest.TestCase):

    def test_HalfNormal_distributional(self):
        seed(42)
        X = RV(HalfNormal(scale=2))
        sims = X.sim(Nsim)
        cdf = stats.halfnorm(scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_HalfNormal_is_abs_of_Normal(self):
        # |X| for X ~ Normal(0, scale) should match HalfNormal(scale).
        seed(42)
        X = RV(Normal(mean=0, sd=2))
        sims = X.apply(abs).sim(Nsim)
        cdf = stats.halfnorm(scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_HalfNormal_mean_var_sd(self):
        X = HalfNormal(scale=2)
        th = stats.halfnorm(scale=2)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_HalfNormal_pdf_at_zero(self):
        X = HalfNormal(scale=1)
        self.assertAlmostEqual(float(X.pdf(0)), np.sqrt(2 / np.pi), places=6)

    def test_HalfNormal_draw_is_scalar_in_support(self):
        seed(0)
        value = HalfNormal(scale=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_HalfNormal_default_scale_is_one(self):
        X = HalfNormal()
        self.assertEqual(X.scale, 1.0)

    def test_HalfNormal_invalid_scale_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: HalfNormal(scale=b))

    def test_HalfNormal_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        HalfNormal(2).draw()
        RV(HalfNormal(2)).sim(100).plot()
        HalfNormal(2).plot()
        plt.figure()
        HalfNormal(2).plot(cdf=True)
        plt.close("all")


class TestHalfCauchy(unittest.TestCase):

    def test_HalfCauchy_distributional(self):
        seed(42)
        X = RV(HalfCauchy(scale=2))
        sims = X.sim(Nsim)
        cdf = stats.halfcauchy(scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_HalfCauchy_is_abs_of_Cauchy(self):
        # |X| for X ~ Cauchy(0, scale) should match HalfCauchy(scale).
        seed(42)
        X = RV(Cauchy(loc=0, scale=2))
        sims = X.apply(abs).sim(Nsim)
        cdf = stats.halfcauchy(scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_HalfCauchy_median_is_scale(self):
        for s in [0.5, 1, 2, 5]:
            self.assertAlmostEqual(float(HalfCauchy(scale=s).median()), s, places=6)

    def test_HalfCauchy_pdf_at_zero(self):
        X = HalfCauchy(scale=1)
        self.assertAlmostEqual(float(X.pdf(0)), 2 / np.pi, places=6)

    def test_HalfCauchy_moments_are_infinite(self):
        # The Cauchy tail is heavy enough that the defining integrals
        # diverge; scipy reports this as inf rather than a finite value.
        X = HalfCauchy(scale=1)
        self.assertTrue(np.isinf(float(X.mean())))
        self.assertTrue(np.isinf(float(X.var())))
        self.assertTrue(np.isinf(float(X.sd())))

    def test_HalfCauchy_cdf_matches_scipy(self):
        X = HalfCauchy(scale=2)
        th = stats.halfcauchy(scale=2)
        for x in [0.0, 0.5, 1.0, 2.0, 10.0]:
            self.assertAlmostEqual(float(X.cdf(x)), float(th.cdf(x)), places=8)

    def test_HalfCauchy_xlim_anchored_at_zero_and_covers_probability(self):
        # Support is [0, inf): the fixed lower bound is used as-is, so the
        # window starts exactly at 0 (not ppf(0.001)), and the unbounded upper
        # side is a quantile cut.
        X = HalfCauchy(scale=1)
        low, high = X.xlim
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, float(X.quantile(0.999)), places=8)
        self.assertAlmostEqual(float(X.cdf(high)) - float(X.cdf(low)), 0.999, places=4)

    def test_HalfCauchy_draw_is_scalar_in_support(self):
        seed(0)
        value = HalfCauchy(scale=2).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)

    def test_HalfCauchy_default_scale_is_one(self):
        X = HalfCauchy()
        self.assertEqual(X.scale, 1.0)

    def test_HalfCauchy_invalid_scale_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: HalfCauchy(scale=b))

    def test_HalfCauchy_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        HalfCauchy(2).draw()
        RV(HalfCauchy(2)).sim(100).plot()
        HalfCauchy(2).plot()
        plt.figure()
        HalfCauchy(2).plot(cdf=True)
        X = HalfCauchy(2)
        X.xlim = (0, 10)  # a window set by hand, since plot() takes none
        plt.figure()
        X.plot()
        plt.close("all")

    def test_HalfCauchy_heavier_tailed_than_HalfNormal(self):
        # Both peak at 0, but the Cauchy tail keeps far more probability
        # out past the bulk -- the reason it is the more permissive prior.
        tail_cauchy = 1 - float(HalfCauchy(scale=1).cdf(10))
        tail_normal = 1 - float(HalfNormal(scale=1).cdf(10))
        self.assertGreater(tail_cauchy, tail_normal)


class TestWeibull(unittest.TestCase):

    def test_Weibull_distributional(self):
        seed(42)
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
        seed(0)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Weibull(1.5, 2).draw()
        RV(Weibull(1.5, 2)).sim(100).plot()
        Weibull(1.5, 2).plot()
        plt.figure()
        Weibull(1.5, 2).plot(cdf=True)
        plt.close("all")


class TestLogistic(unittest.TestCase):

    def test_Logistic_distributional(self):
        seed(42)
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
        seed(0)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Logistic(0, 1).draw()
        RV(Logistic(0, 1)).sim(100).plot()
        Logistic(0, 1).plot()
        plt.figure()
        Logistic(0, 1).plot(cdf=True)
        plt.close("all")


class TestGompertz(unittest.TestCase):

    def test_Gompertz_distributional(self):
        seed(42)
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
        seed(0)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Gompertz(1.5, 2).draw()
        RV(Gompertz(1.5, 2)).sim(100).plot()
        Gompertz(1.5, 2).plot()
        plt.figure()
        Gompertz(1.5, 2).plot(cdf=True)
        plt.close("all")


class TestMakeham(unittest.TestCase):

    def test_Makeham_distributional(self):
        seed(42)
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

    def test_Makeham_hazard_matches_documented_formula(self):
        # The force of mortality is (makeham + shape * e**(x/scale)) / scale:
        # the scale divides the WHOLE hazard, so the constant term contributes
        # makeham / scale. The docstring used to divide only the exponential
        # term, which is what this pins down.
        shape, makeham, scale = 1.5, 0.3, 2.0
        X = Makeham(shape=shape, makeham=makeham, scale=scale)
        x = np.array([0.1, 0.5, 1.0, 2.0])
        hazard = np.asarray(X.pdf(x), dtype=float) / (
            1 - np.asarray(X.cdf(x), dtype=float)
        )
        expected = (makeham + shape * np.exp(x / scale)) / scale
        np.testing.assert_allclose(hazard, expected, rtol=1e-9)

    def test_Makeham_hazard_at_zero_is_shape_plus_makeham_over_scale(self):
        # At x = 0 the exponential term is 1, so the hazard is
        # (makeham + shape) / scale.
        for shape, makeham, scale in [
            (1.5, 0.3, 2.0),
            (2.0, 0.0, 1.0),
            (0.5, 1.0, 4.0),
        ]:
            X = Makeham(shape=shape, makeham=makeham, scale=scale)
            self.assertAlmostEqual(float(X.pdf(0)), (makeham + shape) / scale, places=9)

    def test_Makeham_draw_is_scalar_in_support(self):
        seed(0)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Makeham(1.5, 0.3, 2).draw()
        RV(Makeham(1.5, 0.3, 2)).sim(100).plot()
        Makeham(1.5, 0.3, 2).plot()
        plt.figure()
        Makeham(1.5, 0.3, 2).plot(cdf=True)
        plt.close("all")


class TestLaplace(unittest.TestCase):

    def test_Laplace_distributional(self):
        seed(42)
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
        seed(0)
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
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        Laplace(0, 1).draw()
        RV(Laplace(0, 1)).sim(100).plot()
        Laplace(0, 1).plot()
        plt.figure()
        Laplace(0, 1).plot(cdf=True)
        plt.close("all")


class TestDeMoivre(unittest.TestCase):

    def test_DeMoivre_distributional(self):
        seed(42)
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
        seed(0)
        value = DeMoivre(omega=100).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)
        self.assertLessEqual(float(value), 100.0)

    def test_DeMoivre_invalid_omega_raises(self):
        for bad in [-1, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: DeMoivre(omega=b))

    def test_DeMoivre_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        DeMoivre(100).draw()
        RV(DeMoivre(100)).sim(100).plot()
        DeMoivre(100).plot()
        plt.figure()
        DeMoivre(100).plot(cdf=True)
        plt.close("all")


class TestGEV(unittest.TestCase):

    def test_GEV_distributional(self):
        # shape (xi) uses the standard EVT sign, so scipy's c is -shape.
        seed(42)
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
        seed(0)
        value = GEV(loc=0, scale=1, shape=0.1).draw()
        self.assertIsInstance(value, Scalar)

    def test_GEV_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: GEV(scale=b))

    def test_GEV_invalid_loc_shape_raise(self):
        self.assertRaises(Exception, lambda: GEV(loc="a"))
        self.assertRaises(Exception, lambda: GEV(shape="a"))

    def test_GEV_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        GEV(0, 1, 0.2).draw()
        RV(GEV(0, 1, 0.2)).sim(100).plot()
        GEV(0, 1, 0.2).plot()
        plt.figure()
        GEV(0, 1, 0.2).plot(cdf=True)
        plt.close("all")


class TestGumbel(unittest.TestCase):

    def test_Gumbel_distributional(self):
        seed(42)
        X = RV(Gumbel(loc=2, scale=3))
        sims = X.sim(Nsim)
        cdf = stats.gumbel_r(loc=2, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gumbel_mean_var_sd(self):
        X = Gumbel(loc=2, scale=3)
        th = stats.gumbel_r(loc=2, scale=3)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_Gumbel_is_gev_shape_zero(self):
        # Gumbel(loc, scale) is exactly GEV(loc, scale, shape=0).
        X = Gumbel(loc=1, scale=2)
        self.assertIsInstance(X, GEV)
        self.assertEqual(X.shape, 0)
        for x in [-1, 0, 3, 6]:
            self.assertAlmostEqual(
                float(X.pdf(x)), float(GEV(loc=1, scale=2, shape=0).pdf(x)), places=9
            )

    def test_Gumbel_matches_scipy_gumbel_r(self):
        X = Gumbel(loc=0, scale=1)
        for x in [-1, 0, 2, 5]:
            self.assertAlmostEqual(float(X.pdf(x)), float(stats.gumbel_r().pdf(x)))

    def test_Gumbel_neg_log_exponential(self):
        # -log(Exponential(1)) has a standard Gumbel(0, 1) distribution.
        seed(42)
        E = RV(Exponential(rate=1))
        sims = (-log(E)).sim(Nsim)
        cdf = stats.gumbel_r(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Gumbel_default_params(self):
        X = Gumbel()
        self.assertEqual(X.loc, 0)
        self.assertEqual(X.scale, 1)
        self.assertEqual(X.shape, 0)

    def test_Gumbel_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: Gumbel(scale=b))

    def test_Gumbel_invalid_loc_raises(self):
        self.assertRaises(Exception, lambda: Gumbel(loc="a"))

    def test_Gumbel_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all inherited from GEV / the base class.
        Gumbel(0, 1).draw()
        RV(Gumbel(0, 1)).sim(100).plot()
        Gumbel(0, 1).plot()
        plt.figure()
        Gumbel(0, 1).plot(cdf=True)
        plt.close("all")


class TestGPD(unittest.TestCase):

    def test_GPD_distributional(self):
        # shape (xi) maps straight onto scipy's c (same sign, unlike GEV).
        seed(42)
        X = RV(GPD(loc=2, scale=3, shape=0.2))
        sims = X.sim(Nsim)
        cdf = stats.genpareto(c=0.2, loc=2, scale=3).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_GPD_mean_var_sd(self):
        X = GPD(loc=2, scale=3, shape=0.2)
        th = stats.genpareto(c=0.2, loc=2, scale=3)
        self.assertAlmostEqual(float(X.mean()), float(th.mean()), places=6)
        self.assertAlmostEqual(float(X.var()), float(th.var()), places=6)
        self.assertAlmostEqual(float(X.sd()), float(th.std()), places=6)

    def test_GPD_default_is_exponential(self):
        # shape=0 is the exponential distribution: support [loc, inf), mean 1.
        X = GPD()
        self.assertEqual(X.shape, 0)
        self.assertAlmostEqual(float(X.mean()), 1.0, places=6)
        self.assertAlmostEqual(
            float(X.cdf(1.3)), float(stats.expon().cdf(1.3)), places=9
        )

    def test_GPD_shape_zero_matches_exponential_scale(self):
        # GPD(shape=0, loc=0, scale=s) is exactly Exponential(scale=s).
        gpd = GPD(loc=0, scale=2, shape=0)
        expo = Exponential(scale=2)
        for x in [0.0, 0.5, 1.7, 4.0]:
            self.assertAlmostEqual(float(gpd.pdf(x)), float(expo.pdf(x)), places=9)

    def test_GPD_shape_sign_selects_the_support(self):
        # xi >= 0 -> support [loc, inf); xi < 0 -> bounded above at
        # loc - scale/shape.
        heavy = GPD(loc=0, scale=1, shape=0.3)
        self.assertEqual(float(heavy.quantile(0.0)), 0.0)  # lower bound = loc
        self.assertEqual(float(heavy.quantile(1.0)), np.inf)  # heavy right tail
        light = GPD(loc=0, scale=2, shape=0)
        self.assertEqual(float(light.quantile(0.0)), 0.0)
        self.assertEqual(float(light.quantile(1.0)), np.inf)
        bounded = GPD(loc=0, scale=2, shape=-0.5)
        self.assertEqual(float(bounded.quantile(0.0)), 0.0)
        # loc - scale/shape = 0 - 2/(-0.5) = 4
        self.assertAlmostEqual(float(bounded.quantile(1.0)), 4.0, places=6)

    def test_GPD_draw_is_scalar_in_support(self):
        seed(0)
        value = GPD(loc=1, scale=1, shape=0.1).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 1.0)  # never below loc

    def test_GPD_default_scale_is_one(self):
        X = GPD(loc=0, shape=0.2)
        self.assertEqual(X.scale, 1)

    def test_GPD_invalid_scale_raises(self):
        for bad in [-2, 0, "a"]:
            self.assertRaises(Exception, lambda b=bad: GPD(scale=b))

    def test_GPD_invalid_loc_shape_raise(self):
        self.assertRaises(Exception, lambda: GPD(loc="a"))
        self.assertRaises(Exception, lambda: GPD(shape="a"))

    def test_GPD_plots_without_error(self):
        plt.close("all")  # isolate from a prior test's leftover axes
        # draw / RV / sim / plot all wired through the base class.
        GPD(0, 1, 0.2).draw()
        RV(GPD(0, 1, 0.2)).sim(100).plot()
        GPD(0, 1, 0.2).plot()
        plt.figure()
        GPD(0, 1, 0.2).plot(cdf=True)
        plt.close("all")


class MultivariatePlotTestCase(unittest.TestCase):
    """Base class for the multivariate distributions, which are plottable.

    Starts each test on a fresh figure. A plot of two variables fills its
    figure with three panels -- the joint distribution plus each variable's
    own -- and refuses to share it (MARGINAL_OVERLAY_ERROR), so a figure left
    open by an earlier test would make the next one fail rather than quietly
    overlay onto it.
    """

    def setUp(self):
        plt.close("all")


class TestMultivariateNormal(MultivariatePlotTestCase):

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
        seed(42)
        X = MultivariateNormal(mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        draw = X.draw()
        self.assertEqual(len(draw), 3)

    def test_MultivariateNormal_marginal(self):
        seed(42)
        X, _ = RV(MultivariateNormal(mean=[3, 7], cov=[[4, 0], [0, 9]]))
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_MultivariateNormal_plots_joint_density(self):
        # Two variables: one joint distribution, so no arguments needed.
        X = MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
        X.plot()
        self.assertEqual(plt.gcf().get_suptitle(), "Joint Contour Plot")
        self.assertEqual(plt.gca().get_xlabel(), "Variable 1")
        self.assertEqual(plt.gca().get_ylabel(), "Variable 2")
        plt.close("all")

    def test_MultivariateNormal_plot_contour(self):
        X = MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
        X.plot(contour=True)
        self.assertEqual(plt.gcf().get_suptitle(), "Joint Contour Plot")
        plt.close("all")

    def test_MultivariateNormal_plot_3d_defaults_to_the_matrix(self):
        # Above two variables there is no single joint plot, so plot() shows
        # every pair rather than asking which one to pick.
        plt.close("all")
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        X.plot()
        panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
        self.assertEqual(len(panels), 6)
        self.assertEqual(
            plt.gcf()._suptitle.get_text(), "Probability Density Functions"
        )
        plt.close("all")

    def test_MultivariateNormal_plot_two_variables_is_one_joint_plot(self):
        # Naming exactly two variables still draws their joint distribution,
        # not a 2-by-2 matrix -- two variables have one joint plot between
        # them. It gets the standard three panels: the joint distribution,
        # plus each of the two variables on its own.
        plt.close("all")
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        X.plot(variables=(0, 2))
        panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
        self.assertEqual(len(panels), 3)
        # plt.gca() is the joint panel, and it names the pair that was asked
        # for -- the 1st and 3rd variables, not the 1st and 2nd.
        self.assertEqual(plt.gca().get_xlabel(), "Variable 1")
        self.assertEqual(plt.gca().get_ylabel(), "Variable 3")
        plt.close("all")

    def test_MultivariateNormal_plot_the_old_pairs_keyword_explains_itself(self):
        plt.close("all")
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        with self.assertRaises(ValueError) as cm:
            X.plot(pairs=True)
        self.assertIn("no longer needed", str(cm.exception))
        plt.close("all")

    def test_MultivariateNormal_plot_pairs_lower_triangle(self):
        # n diagonal panels plus n(n-1)/2 lower-triangle panels, and no
        # upper triangle: 4 variables -> 4 + 6 = 10 panels. The 6 joint panels
        # each put a colorbar in the empty cell mirroring them, so the figure
        # holds 16 axes in all.
        X = MultivariateNormal(mean=[1, 2, 3, 4], cov=np.eye(4).tolist())
        X.plot()
        panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
        self.assertEqual(len(panels), 10)
        self.assertEqual(len(plt.gcf().axes), 10 + 6)
        plt.close("all")

    def test_MultivariateNormal_plot_pairs_labels_every_row(self):
        # A diagonal panel's y-axis is that variable's own density, not the
        # variable, so it says which -- and says "Marginal" so it can't be
        # read as the joint density its colorbar measures. The rest of the
        # left column names its row's variable. Matches a simulated matrix.
        plt.close("all")
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        X.plot()
        panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
        # Panels are added row by row, so the first one is (0, 0) -- the
        # diagonal panel for variable 1.
        self.assertEqual(panels[0].get_ylabel(), "Marginal Density")
        self.assertEqual(
            sorted(a.get_ylabel() for a in panels if a.get_ylabel()),
            [
                "Marginal Density",
                "Marginal Density",
                "Marginal Density",
                "Variable 2",
                "Variable 3",
            ],
        )
        plt.close("all")

    def test_MultivariateNormal_plot_pairs_colorbars_name_their_pair(self):
        # Each joint panel keeps its own color scale, so each gets its own
        # colorbar in the empty cell mirroring it, named for the pair it
        # explains.
        plt.close("all")
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        X.plot()
        bars = [a for a in plt.gcf().axes if a.get_subplotspec() is None]
        self.assertEqual(
            sorted(a.get_title() for a in bars),
            [
                "Variables 1 & 2",
                "Variables 1 & 3",
                "Variables 2 & 3",
            ],
        )
        # "Joint", so a bar can't be confused with the marginal density on the
        # diagonal panel it sits across from.
        self.assertEqual({a.get_ylabel() for a in bars}, {"Joint Density"})
        plt.close("all")

    def test_MultivariateNormal_plot_pairs_colorbars_avoid_the_panels(self):
        plt.close("all")
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        X.plot()
        fig = plt.gcf()
        fig.canvas.draw()
        panels = [a.get_position() for a in fig.axes if a.get_subplotspec() is not None]
        for bar in [a for a in fig.axes if a.get_subplotspec() is None]:
            for panel in panels:
                self.assertFalse(bar.get_position().overlaps(panel))
        plt.close("all")

    def test_MultivariateNormal_plot_pairs_subset_of_variables(self):
        X = MultivariateNormal(mean=[1, 2, 3, 4], cov=np.eye(4).tolist())
        X.plot(variables=(0, 2, 3))
        # 3 diagonal panels plus the 3 pairs among them, each pair's colorbar
        # in the mirroring cell.
        panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
        self.assertEqual(len(panels), 6)
        self.assertEqual(len(plt.gcf().axes), 6 + 3)
        plt.close("all")

    def test_MultivariateNormal_marginal_1d_is_exact(self):
        # A single variable of a multivariate normal is normal, with that
        # variable's own mean and variance.
        X = MultivariateNormal(mean=[1, 2], cov=[[4, 0.5], [0.5, 9]])
        self.assertAlmostEqual(float(X._marginal_1d(1).mean()), 2)
        self.assertAlmostEqual(float(X._marginal_1d(1).sd()), 3)

    def test_MultivariateNormal_joint_func_is_exact_submatrix(self):
        # The pair's density is the bivariate normal built from those two
        # entries of the mean vector and the matching 2x2 block of cov --
        # not an approximation of it.
        cov = [[4, 0.5, 0.2], [0.5, 1, 0.0], [0.2, 0.0, 2]]
        X = MultivariateNormal(mean=[1, 2, 3], cov=cov)
        pair = stats.multivariate_normal([1, 3], [[4, 0.2], [0.2, 2]])
        points = np.array([[0.5, 2.0], [1.0, 3.0], [3.0, 4.5]])
        computed = X._joint_func(0, 2)(points[:, 0], points[:, 1])
        for got, expected in zip(computed, pair.pdf(points)):
            self.assertAlmostEqual(float(got), float(expected))

    def test_MultivariateNormal_plot_rejects_cdf_and_type(self):
        X = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        self.assertRaises(ValueError, lambda: X.plot(cdf=True))
        self.assertRaises(ValueError, lambda: X.plot(type="hist"))
        plt.close("all")

    def test_MultivariateNormal_plot_bad_variables(self):
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        # No variables at all, out of range, repeated, not a number. (Three or
        # more is no longer an error -- it asks for a matrix of every pair --
        # and neither is one, which asks for that variable's own distribution.)
        self.assertRaises(Exception, lambda: X.plot(variables=()))
        self.assertRaises(Exception, lambda: X.plot(variables=(0, 7)))
        self.assertRaises(Exception, lambda: X.plot(variables=(1, 1)))
        self.assertRaises(Exception, lambda: X.plot(variables=(0, "a")))
        plt.close("all")

    def test_MultivariateNormal_plot_one_variable_is_its_own_distribution(self):
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        for arg in [1, [1], (1,)]:
            plt.close("all")
            X.plot(variables=arg)
            panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
            self.assertEqual(len(panels), 1)
            self.assertEqual(plt.gca().get_xlabel(), "Variable 2")
            self.assertEqual(plt.gca().get_title(), "Probability Density Function")
        plt.close("all")

    def test_MultivariateNormal_plot_pairs_cannot_share_a_figure(self):
        X = MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        Normal(0, 1).plot()
        self.assertRaises(ValueError, lambda: X.plot())
        plt.close("all")

    def test_MultivariateNormal_plot_shade_explains_itself(self):
        # A joint plot has no single curve to fill under, so .shade() gives
        # an explanation rather than an AttributeError.
        X = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        plot = X.plot()
        with self.assertRaises(Exception) as cm:
            plot.shade(lt=0)
        self.assertIn("shade", str(cm.exception))
        plt.close("all")

    def test_MultivariateNormal_is_multivariate_distribution(self):
        X = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        self.assertIsInstance(X, MultivariateDistribution)

    def test_MultivariateNormal_mean_cov_methods(self):
        X = MultivariateNormal(mean=[1, 2], cov=[[2, 0.8], [0.8, 1]])
        self.assertIsInstance(X.mean(), Vector)
        np.testing.assert_allclose(np.array(X.mean()), [1, 2])
        np.testing.assert_allclose(X.cov(), [[2, 0.8], [0.8, 1]])

    def test_MultivariateNormal_var_sd_corr(self):
        cov = [[2, 0.8], [0.8, 1]]
        X = MultivariateNormal(mean=[1, 2], cov=cov)
        np.testing.assert_allclose(np.array(X.var()), np.diag(cov))
        np.testing.assert_allclose(np.array(X.sd()), np.sqrt(np.diag(cov)))
        corr = X.corr()
        np.testing.assert_allclose(np.diag(corr), [1.0, 1.0])
        np.testing.assert_allclose(corr, [[1, 0.8 / np.sqrt(2)], [0.8 / np.sqrt(2), 1]])

    def test_MultivariateNormal_3d_correlated_matches_scipy(self):
        # Exact agreement with scipy on a correlated 3-D case.
        mean = [1, -2, 3]
        cov = [[4, 1, 0.5], [1, 2, -0.3], [0.5, -0.3, 1]]
        X = MultivariateNormal(mean=mean, cov=cov)
        th = stats.multivariate_normal(mean=mean, cov=cov)
        np.testing.assert_allclose(np.array(X.mean()), th.mean)
        np.testing.assert_allclose(X.cov(), th.cov)

    def test_MultivariateNormal_nonsymmetric_cov_friendly_error(self):
        # A non-symmetric matrix has complex eigenvalues; the check must raise
        # a friendly Exception, not a numpy TypeError.
        with self.assertRaises(Exception) as ctx:
            MultivariateNormal(mean=[0, 0], cov=[[0, -1], [1, 0]])
        self.assertNotIsInstance(ctx.exception, TypeError)


class TestMultivariateT(MultivariatePlotTestCase):

    def test_MultivariateT_mean_cov_error(self):
        self.assertRaises(
            Exception, lambda: MultivariateT(mean=[2], cov=[[2, 2], [3, 4]], df=5)
        )

    def test_MultivariateT_cov_square_error(self):
        self.assertRaises(
            Exception,
            lambda: MultivariateT(mean=[2, 4], cov=[[2, 4, 5], [2, 1]], df=5),
        )

    def test_MultivariateT_cov_not_psd(self):
        self.assertRaises(
            Exception,
            lambda: MultivariateT(mean=[0, 0], cov=[[-1, 0], [0, 1]], df=5),
        )

    def test_MultivariateT_df_error(self):
        for bad in [0, -1, "a"]:
            self.assertRaises(
                Exception,
                lambda v=bad: MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=v),
            )

    def test_MultivariateT_draw_shape(self):
        seed(42)
        X = MultivariateT(mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], df=5)
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)

    def test_MultivariateT_pdf_matches_scipy(self):
        X = MultivariateT(mean=[1, 2], cov=[[2, 0.5], [0.5, 1]], df=5)
        th = stats.multivariate_t(loc=[1, 2], shape=[[2, 0.5], [0.5, 1]], df=5)
        for pt in [[0, 0], [1, 2], [2, 3]]:
            self.assertAlmostEqual(float(X.pdf(pt)), float(th.pdf(pt)), places=9)

    def test_MultivariateT_marginal_is_student_t(self):
        # Each marginal of a multivariate t is a univariate t with the same
        # df: location mean_i and scale sqrt(cov_ii).
        seed(42)
        X, _ = RV(MultivariateT(mean=[3, 7], cov=[[4, 0], [0, 9]], df=5))
        sims = X.sim(Nsim)
        cdf = stats.t(df=5, loc=3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_MultivariateT_large_df_approaches_normal(self):
        # As df grows, the multivariate t approaches a MultivariateNormal
        # with covariance equal to the scale matrix.
        cov = [[2, 0.5], [0.5, 1]]
        mvt = MultivariateT(mean=[0, 0], cov=cov, df=100000)
        mvn = MultivariateNormal(mean=[0, 0], cov=cov)
        for pt in [[0, 0], [1, 1], [2, -1]]:
            self.assertAlmostEqual(float(mvt.pdf(pt)), float(mvn.pdf(pt)), places=3)

    def test_MultivariateT_plots_joint_density(self):
        X = MultivariateT(mean=[0, 0], cov=[[1, 0.3], [0.3, 2]], df=4)
        X.plot()
        self.assertEqual(plt.gcf().get_suptitle(), "Joint Contour Plot")
        plt.close("all")

    def test_MultivariateT_plots_when_moments_are_undefined(self):
        # df = 1 has no mean and no covariance, but the density -- and so
        # the plot -- is perfectly well defined.
        X = MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=1)
        X.plot()
        plt.close("all")

    def test_MultivariateT_joint_func_keeps_df(self):
        # The pair's distribution is a bivariate t with the *same* degrees
        # of freedom and the matching 2x2 block of the scale matrix.
        cov = [[1, 0.4, 0.0], [0.4, 2, 0.1], [0.0, 0.1, 1]]
        X = MultivariateT(mean=[0, 1, 2], cov=cov, df=5)
        pair = stats.multivariate_t(loc=[1, 2], shape=[[2, 0.1], [0.1, 1]], df=5)
        points = np.array([[1.0, 2.0], [0.5, 1.0]])
        computed = X._joint_func(1, 2)(points[:, 0], points[:, 1])
        for got, expected in zip(computed, pair.pdf(points)):
            self.assertAlmostEqual(float(got), float(expected))

    def test_MultivariateT_mean_is_location(self):
        X = MultivariateT(mean=[3, 7], cov=[[4, 0], [0, 9]], df=5)
        self.assertIsInstance(X.mean(), Vector)
        np.testing.assert_allclose(np.array(X.mean()), [3, 7])

    def test_MultivariateT_cov_is_scaled_scale_matrix(self):
        # For df > 2 the covariance is df / (df - 2) times the scale matrix.
        scale = [[2, 0.5], [0.5, 1]]
        df = 5
        X = MultivariateT(mean=[0, 0], cov=scale, df=df)
        np.testing.assert_allclose(
            X.cov(), (df / (df - 2)) * np.asarray(scale, dtype=float)
        )
        # sd()/corr() derive from that covariance.
        np.testing.assert_allclose(np.diag(X.corr()), [1.0, 1.0])

    def test_MultivariateT_cov_undefined_for_low_df(self):
        for df in [1, 2]:
            X = MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=df)
            self.assertRaises(Exception, X.cov)

    def test_MultivariateT_mean_undefined_for_low_df(self):
        # The mean exists only for df > 1 (like the univariate Student's t).
        for df in [0.5, 1]:
            X = MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=df)
            self.assertRaises(Exception, X.mean)
        # For df > 1 the mean is the location vector.
        X = MultivariateT(mean=[3, 7], cov=[[1, 0], [0, 1]], df=1.5)
        np.testing.assert_allclose(np.array(X.mean()), [3, 7])

    def test_MultivariateT_is_multivariate_distribution(self):
        X = MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=5)
        self.assertIsInstance(X, MultivariateDistribution)


class TestMultivariateLogNormal(MultivariatePlotTestCase):

    def test_MVLogNormal_validation_inherited(self):
        # Validation is delegated to the underlying MultivariateNormal.
        self.assertRaises(
            Exception,
            lambda: MultivariateLogNormal(mean=[0, 0], cov=[[1, 0, 0], [0, 1, 0]]),
        )
        self.assertRaises(
            Exception,
            lambda: MultivariateLogNormal(mean=[0, 0], cov=[[0, -1], [1, 0]]),
        )

    def test_MVLogNormal_draw_positive(self):
        seed(42)
        X = MultivariateLogNormal(mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)
        self.assertTrue(all(v > 0 for v in draw))

    def test_MVLogNormal_mean_cov_var_closed_form(self):
        mu = np.array([0.0, 0.5, -0.3])
        sig = np.array([[0.4, 0.1, 0.05], [0.1, 0.3, -0.08], [0.05, -0.08, 0.2]])
        X = MultivariateLogNormal(mean=mu.tolist(), cov=sig.tolist())
        d = np.diag(sig)
        exp_mean = np.exp(mu + d / 2)
        exp_cov = np.exp(np.add.outer(mu, mu) + np.add.outer(d, d) / 2) * (
            np.exp(sig) - 1
        )
        np.testing.assert_allclose(np.array(X.mean()), exp_mean)
        np.testing.assert_allclose(X.cov(), exp_cov)
        np.testing.assert_allclose(np.array(X.var()), np.diag(exp_cov))
        np.testing.assert_allclose(np.array(X.sd()), np.sqrt(np.diag(exp_cov)))

    def test_MVLogNormal_mean_matches_simulation(self):
        seed(0)
        X = MultivariateLogNormal(mean=[0.0, 0.5], cov=[[0.4, 0.1], [0.1, 0.3]])
        sims = np.array([list(X.draw()) for _ in range(50000)])
        np.testing.assert_allclose(sims.mean(0), np.array(X.mean()), rtol=0.05)

    def test_MVLogNormal_corr_diagonal_is_one(self):
        X = MultivariateLogNormal(
            mean=[0, 0, 0], cov=[[1, 0.3, 0.1], [0.3, 1, 0.2], [0.1, 0.2, 1]]
        )
        np.testing.assert_allclose(np.diag(X.corr()), [1.0, 1.0, 1.0])

    def test_MVLogNormal_pdf_matches_transform(self):
        mean = [0.0, 0.5]
        cov = [[0.4, 0.1], [0.1, 0.3]]
        X = MultivariateLogNormal(mean=mean, cov=cov)
        normal = stats.multivariate_normal(mean, cov)
        for pt in [[1.0, 1.0], [0.5, 2.0], [2.0, 0.5]]:
            expected = normal.pdf(np.log(pt)) / np.prod(pt)
            self.assertAlmostEqual(float(X.pdf(pt)), float(expected), places=12)

    def test_MVLogNormal_pdf_batch_uses_per_row_jacobian(self):
        # Regression test: the Jacobian divisor used to be np.prod(x) over the
        # whole array, so a 2-D input divided every row by the product of every
        # entry instead of by its own row's product -- silently wrong densities
        # with no error raised. A batch must agree with row-by-row evaluation.
        mean = [0.0, 0.5]
        cov = [[0.4, 0.1], [0.1, 0.3]]
        X = MultivariateLogNormal(mean=mean, cov=cov)
        batch = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 3.0]])
        expected = np.array([float(X.pdf(row)) for row in batch])
        got = np.asarray(X.pdf(batch), dtype=float)
        self.assertEqual(got.shape, (3,))
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_MVLogNormal_pdf_batch_three_dimensional(self):
        # Same check with three components, where the whole-array product and
        # the per-row product differ by more.
        X = MultivariateLogNormal(mean=[0, 0, 0], cov=np.eye(3).tolist())
        batch = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
        expected = np.array([float(X.pdf(row)) for row in batch])
        np.testing.assert_allclose(
            np.asarray(X.pdf(batch), dtype=float), expected, rtol=1e-12
        )

    def test_MVLogNormal_marginal_is_lognormal(self):
        # Each component is a univariate LogNormal with the underlying
        # component's mean and sd.
        seed(42)
        A, B = RV(MultivariateLogNormal(mean=[0.3, 1.0], cov=[[0.25, 0], [0, 0.5]]))
        sims = A.sim(Nsim)
        cdf = stats.lognorm(s=0.5, scale=np.exp(0.3)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_MVLogNormal_log_is_normal(self):
        # Taking logs recovers the underlying multivariate normal marginals.
        seed(42)
        A, B = RV(MultivariateLogNormal(mean=[2, -1], cov=[[1, 0], [0, 4]]))
        sims = A.apply(np.log).sim(Nsim)
        cdf = stats.norm(loc=2, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_MVLogNormal_is_multivariate_distribution(self):
        X = MultivariateLogNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        self.assertIsInstance(X, MultivariateDistribution)

    def test_MVLogNormal_two_variables_is_one_joint_plot(self):
        X = MultivariateLogNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
        self.assertEqual(X._free_dim(), 2)
        X.plot()
        plt.close("all")

    def test_MVLogNormal_marginal_is_lognormal(self):
        # One variable of the underlying normal is a Normal, so one
        # variable here is an ordinary LogNormal.
        X = MultivariateLogNormal(mean=[0.3, -0.2], cov=[[1.0, 0.5], [0.5, 0.8]])
        marginal = X._marginal_1d(1)
        self.assertIsInstance(marginal, LogNormal)
        expected = LogNormal(mu=-0.2, sigma=np.sqrt(0.8))
        for v in [0.25, 1.0, 2.5, 6.0]:
            self.assertAlmostEqual(float(marginal.pdf(v)), float(expected.pdf(v)))

    def test_MVLogNormal_joint_func_matches_transformed_normal(self):
        # The pair's density is the underlying bivariate normal's, read at
        # the logarithms and divided by the x * y Jacobian.
        mean = [0.3, -0.2, 1.0]
        cov = [[1.0, 0.5, 0.2], [0.5, 0.8, 0.1], [0.2, 0.1, 0.6]]
        X = MultivariateLogNormal(mean=mean, cov=cov)
        index = [0, 2]
        pair = stats.multivariate_normal(
            np.asarray(mean)[index], np.asarray(cov)[np.ix_(index, index)]
        )
        func = X._joint_func(0, 2)
        for x in [0.4, 1.0, 3.0]:
            for y in [0.5, 2.0, 7.0]:
                expected = float(pair.pdf([np.log(x), np.log(y)])) / (x * y)
                self.assertAlmostEqual(
                    float(func(np.array([x]), np.array([y]))[0]), expected
                )

    def test_MVLogNormal_joint_func_zero_outside_positive_quadrant(self):
        # A log-normal value is always strictly positive.
        X = MultivariateLogNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
        func = X._joint_func(0, 1)
        xs = np.array([-1.0, 0.0, 2.0, 2.0])
        ys = np.array([2.0, 2.0, -1.0, 0.0])
        for value in func(xs, ys):
            self.assertEqual(float(value), 0.0)

    def test_MVLogNormal_joint_func_integrates_to_one(self):
        X = MultivariateLogNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
        func = X._joint_func(0, 1)
        # A rectangle rule over the bulk of the support; the tail beyond it
        # is negligible, so the total should be very close to 1.
        step = 0.05
        grid = np.arange(step / 2, 40, step)
        xs, ys = np.meshgrid(grid, grid)
        total = func(xs.ravel(), ys.ravel()).sum() * step * step
        self.assertAlmostEqual(float(total), 1.0, places=3)

    def test_MVLogNormal_plot_pairs(self):
        MultivariateLogNormal(
            mean=[0, 0, 0], cov=[[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]
        ).plot()
        plt.close("all")


class TestWishart(unittest.TestCase):

    def test_Wishart_scale_square_error(self):
        self.assertRaises(Exception, lambda: Wishart(df=5, scale=[[2, 4, 5], [2, 1]]))

    def test_Wishart_scale_not_pd(self):
        self.assertRaises(Exception, lambda: Wishart(df=5, scale=[[-1, 0], [0, 1]]))

    def test_Wishart_df_error(self):
        # df must exceed p - 1 (here p = 2, so df must be > 1).
        for bad in [1, 0.5, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: Wishart(df=v, scale=[[1, 0], [0, 1]])
            )

    def test_Wishart_draw_shape(self):
        seed(42)
        X = Wishart(df=5, scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)
        self.assertEqual(len(draw[0]), 3)

    def test_Wishart_draw_symmetric_pd(self):
        seed(42)
        draw = Wishart(df=6, scale=[[2, 0.5], [0.5, 1]]).draw()
        m = np.array([list(row) for row in draw])
        self.assertTrue(np.allclose(m, m.T))
        self.assertTrue(np.all(np.linalg.eigvals(m) > 0))

    def test_Wishart_pdf_matches_scipy(self):
        X = Wishart(df=5, scale=[[2, 0.5], [0.5, 1]])
        th = stats.wishart(df=5, scale=[[2, 0.5], [0.5, 1]])
        for pt in [[[1, 0], [0, 1]], [[3, 0.5], [0.5, 2]]]:
            self.assertAlmostEqual(float(X.pdf(pt)), float(th.pdf(pt)), places=9)

    def test_Wishart_mean_matches_df_times_scale(self):
        # E[W] = df * scale; check the Monte Carlo mean of the (0, 0) entry.
        seed(42)
        df, scale = 8, [[2, 0.5], [0.5, 1]]
        X = Wishart(df=df, scale=scale)
        sims = [X.draw()[0][0] for _ in range(Nsim)]
        self.assertAlmostEqual(np.mean(sims), df * scale[0][0], delta=0.5)

    def test_Wishart_mean_method_returns_df_times_scale(self):
        # Regression test: `mean()` did not exist at all and raised a bare
        # AttributeError, even though the docstring documents the mean as
        # df * scale. It returns the matrix shaped the way `draw()` does.
        df, scale = 5, [[2, 0.5], [0.5, 3]]
        X = Wishart(df=df, scale=scale)
        got = np.array([list(row) for row in X.mean()], dtype=float)
        np.testing.assert_allclose(got, df * np.asarray(scale, dtype=float))

    def test_Wishart_mean_shape_matches_draw(self):
        seed(42)
        X = Wishart(df=6, scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        mean, draw = X.mean(), X.draw()
        self.assertIsInstance(mean, Vector)
        self.assertEqual(len(mean), len(draw))
        self.assertEqual(len(mean[0]), len(draw[0]))

    def test_Wishart_vector_summaries_raise_friendly(self):
        # Regression test: these raised a bare AttributeError. A matrix-valued
        # draw has no per-component vector of variances, so each must explain
        # that and point at `mean()` instead of leaking an AttributeError.
        X = Wishart(df=5, scale=[[1, 0], [0, 1]])
        for name in ["var", "sd", "cov", "corr"]:
            with self.assertRaises(Exception) as caught:
                getattr(X, name)()
            message = str(caught.exception)
            self.assertNotIsInstance(caught.exception, AttributeError)
            self.assertIn(name, message)
            self.assertIn("mean()", message)

    def test_Wishart_diagonal_is_scaled_chisquare(self):
        # With an identity scale, each diagonal entry is chi-square(df).
        seed(42)
        X = Wishart(df=6, scale=[[1, 0], [0, 1]])
        sims = [X.draw()[0][0] for _ in range(Nsim)]
        pval = stats.kstest(sims, stats.chi2(df=6).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Wishart_plot_raises(self):
        X = Wishart(df=5, scale=[[1, 0], [0, 1]])
        self.assertRaises(Exception, X.plot)


class TestInverseWishart(unittest.TestCase):

    def test_InverseWishart_scale_square_error(self):
        self.assertRaises(
            Exception, lambda: InverseWishart(df=5, scale=[[2, 4, 5], [2, 1]])
        )

    def test_InverseWishart_scale_not_pd(self):
        self.assertRaises(
            Exception, lambda: InverseWishart(df=5, scale=[[-1, 0], [0, 1]])
        )

    def test_InverseWishart_df_error(self):
        for bad in [1, 0.5, -1, "a"]:
            self.assertRaises(
                Exception, lambda v=bad: InverseWishart(df=v, scale=[[1, 0], [0, 1]])
            )

    def test_InverseWishart_draw_shape(self):
        seed(42)
        X = InverseWishart(df=5, scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)
        self.assertEqual(len(draw[0]), 3)

    def test_InverseWishart_draw_symmetric_pd(self):
        seed(42)
        draw = InverseWishart(df=6, scale=[[2, 0.5], [0.5, 1]]).draw()
        m = np.array([list(row) for row in draw])
        self.assertTrue(np.allclose(m, m.T))
        self.assertTrue(np.all(np.linalg.eigvals(m) > 0))

    def test_InverseWishart_pdf_matches_scipy(self):
        X = InverseWishart(df=5, scale=[[2, 0.5], [0.5, 1]])
        th = stats.invwishart(df=5, scale=[[2, 0.5], [0.5, 1]])
        for pt in [[[1, 0], [0, 1]], [[3, 0.5], [0.5, 2]]]:
            self.assertAlmostEqual(float(X.pdf(pt)), float(th.pdf(pt)), places=9)

    def test_InverseWishart_mean_matches_formula(self):
        # E[X] = scale / (df - p - 1), which requires df > p + 1.
        seed(42)
        df, scale = 10, [[2, 0.5], [0.5, 1]]
        X = InverseWishart(df=df, scale=scale)
        sims = [X.draw()[0][0] for _ in range(Nsim)]
        expected = scale[0][0] / (df - 2 - 1)
        self.assertAlmostEqual(np.mean(sims), expected, delta=0.05)

    def test_InverseWishart_mean_method_returns_formula(self):
        # Regression test: `mean()` did not exist and raised a bare
        # AttributeError, though the docstring documents scale / (df - p - 1).
        df, scale = 8, [[2, 0.5], [0.5, 3]]
        X = InverseWishart(df=df, scale=scale)
        got = np.array([list(row) for row in X.mean()], dtype=float)
        np.testing.assert_allclose(got, np.asarray(scale, dtype=float) / (df - 2 - 1))

    def test_InverseWishart_mean_undefined_raises_friendly(self):
        # The mean exists only for df > p + 1. Below that it must say so
        # rather than return a nonsense (negative) matrix.
        X = InverseWishart(df=3, scale=[[1, 0], [0, 1]])
        with self.assertRaises(Exception) as caught:
            X.mean()
        self.assertIn("df > p + 1", str(caught.exception))
        # ...and the distribution is still perfectly usable otherwise.
        seed(42)
        self.assertEqual(len(X.draw()), 2)

    def test_InverseWishart_vector_summaries_raise_friendly(self):
        # Regression test: these raised a bare AttributeError.
        X = InverseWishart(df=6, scale=[[1, 0], [0, 1]])
        for name in ["var", "sd", "cov", "corr"]:
            with self.assertRaises(Exception) as caught:
                getattr(X, name)()
            message = str(caught.exception)
            self.assertNotIsInstance(caught.exception, AttributeError)
            self.assertIn(name, message)
            self.assertIn("mean()", message)

    def test_InverseWishart_inverse_is_wishart(self):
        # If X ~ InverseWishart(df, scale), then inv(X) ~ Wishart(df, inv(scale)).
        # Check the (0, 0) entry of the inverse against chi-square with an
        # identity scale (whose inverse is also the identity).
        seed(42)
        X = InverseWishart(df=6, scale=[[1, 0], [0, 1]])
        sims = []
        for _ in range(Nsim):
            m = np.array([list(row) for row in X.draw()])
            sims.append(np.linalg.inv(m)[0, 0])
        pval = stats.kstest(sims, stats.chi2(df=6).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_InverseWishart_plot_raises(self):
        X = InverseWishart(df=5, scale=[[1, 0], [0, 1]])
        self.assertRaises(Exception, X.plot)


class TestBivariateNormal(MultivariatePlotTestCase):

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
        seed(42)
        X, Y = RV(BivariateNormal(mean1=30, mean2=50, sd1=8, sd2=6, corr=-0.4))
        Z = 4 * X - 2 * Y
        Z_mean = 4 * 30 + (-2) * 50
        Z_var = 16 * 64 + 4 * 36 + 2 * (4) * (-2) * (-0.4 * 8 * 6)
        sims = Z.sim(Nsim)
        cdf = stats.norm(loc=Z_mean, scale=sqrt(Z_var)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_BivNormal_condDistr_r(self):
        seed(42)
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
        seed(42)
        X, _ = RV(BivariateNormal(mean1=5, mean2=10, sd1=2, sd2=3, corr=0.6))
        sims = X.sim(Nsim)
        cdf = stats.norm(loc=5, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_BivariateNormal_explicit_cov(self):
        # Exercise the var1/var2/cov keyword paths (instead of sd/corr)
        X = BivariateNormal(mean1=1, mean2=2, var1=4, var2=9, cov=3.0)
        np.testing.assert_allclose(X.cov(), [[4, 3.0], [3.0, 9]])

    def test_BivariateNormal_explicit_cov_not_psd_raises(self):
        # An explicit covariance too large relative to the variances is not
        # positive semi-definite and must be rejected (regression: this used
        # to construct silently and emit a raw numpy RuntimeWarning).
        self.assertRaises(
            Exception,
            lambda: BivariateNormal(mean1=0, mean2=0, var1=1, var2=1, cov=5),
        )


class TestMultinomial(MultivariatePlotTestCase):

    def test_Multinomial_error_n_negative(self):
        self.assertRaises(Exception, lambda: Multinomial(n=-1, p=[0.5, 0.5]))

    def test_Multinomial_error_n_float(self):
        self.assertRaises(Exception, lambda: Multinomial(n=5.5, p=[0.5, 0.5]))

    def test_Multinomial_error_p_sum(self):
        self.assertRaises(Exception, lambda: Multinomial(n=10, p=[0.5, 0.6]))

    def test_Multinomial_error_p_negative(self):
        self.assertRaises(Exception, lambda: Multinomial(n=10, p=[-0.1, 1.1]))

    def test_Multinomial_p_sum_within_float_tolerance_constructs(self):
        # sum([0.4, 0.3, 0.2, 0.1]) == 0.9999999999999999 in floating point,
        # not exactly 1 -- this must not be rejected.
        Multinomial(n=12, p=[0.4, 0.3, 0.2, 0.1])

    def test_Multinomial_p_sum_repeated_fraction_constructs(self):
        # sum([1 / 6] * 6) also lands a few ULPs away from 1.
        Multinomial(n=6, p=[1 / 6] * 6)

    def test_Multinomial_error_p_sum_still_rejected_when_actually_wrong(self):
        self.assertRaises(Exception, lambda: Multinomial(n=5, p=[0.5, 0.6]))

    def test_Multinomial_draw_sums_to_n(self):
        seed(42)
        X = Multinomial(n=20, p=[0.2, 0.5, 0.3])
        draw = X.draw()
        self.assertEqual(sum(draw), 20)

    def test_Multinomial_corr_zero_variance_component_is_nan_not_warning(self):
        # Regression test: a category with probability 0 never varies, so
        # its correlation row/column is undefined (nan). This used to leak
        # a raw "RuntimeWarning: invalid value encountered in divide" with
        # no context; it should now compute quietly, with nan as the
        # documented, intentional result.
        import warnings

        X = Multinomial(n=10, p=[1, 0, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            corr = X.corr()
        self.assertTrue(np.all(np.isnan(corr)))

    def test_Multinomial_marginals_match_Binomial(self):
        seed(42)
        p = [0.3, 0.5, 0.2]
        n = 15
        X0, X1, X2 = RV(Multinomial(n=n, p=p))
        for rv, pi in zip([X0, X1, X2], p):
            sims = rv.sim(Nsim)
            expected_mean = n * pi
            expected_var = n * pi * (1 - pi)
            self.assertAlmostEqual(float(sims.mean()), expected_mean, delta=0.15)
            self.assertAlmostEqual(float(sims.var()), expected_var, delta=0.15)

    def test_Multinomial_two_categories_plot_points_to_Binomial(self):
        # Two counts that must add to n vary in only one direction, so
        # there is no joint plot to draw -- it is a Binomial.
        X = Multinomial(n=10, p=[0.5, 0.5])
        with self.assertRaises(Exception) as cm:
            X.plot()
        self.assertIn("Binomial", str(cm.exception))

    def test_Multinomial_three_categories_plots_joint_pmf(self):
        # Three categories vary in two directions (the third count is
        # whatever is left), so this is the single-joint-plot case.
        X = Multinomial(n=10, p=[0.5, 0.3, 0.2])
        X.plot()
        self.assertEqual(plt.gcf().get_suptitle(), "Joint PMF Plot")
        plt.close("all")

    def test_Multinomial_is_discrete(self):
        # Counts are whole numbers, so the joint plot draws a probability at
        # each possible pair rather than a smooth surface between them.
        self.assertTrue(Multinomial(n=10, p=[0.5, 0.3, 0.2]).discrete)

    def test_Multinomial_joint_func_totals_one_over_the_grid(self):
        # The joint probabilities of two counts cover every possible pair,
        # so they add to 1 -- including the impossible pairs, which get 0.
        X = Multinomial(n=12, p=[0.5, 0.3, 0.2])
        counts = np.arange(13)
        xs, ys = np.meshgrid(counts, counts)
        total = X._joint_func(0, 1)(xs.ravel(), ys.ravel()).sum()
        self.assertAlmostEqual(float(total), 1.0)

    def test_Multinomial_joint_func_matches_pooled_multinomial(self):
        # Pooling the other categories into one gives a three-category
        # multinomial exactly.
        X = Multinomial(n=12, p=[0.5, 0.3, 0.2])
        expected = stats.multinomial(12, [0.5, 0.3, 0.2]).pmf([5, 4, 3])
        got = X._joint_func(0, 1)(np.array([5]), np.array([4]))[0]
        self.assertAlmostEqual(float(got), float(expected))

    def test_Multinomial_joint_func_zero_for_impossible_pairs(self):
        # Two counts can't add up to more than the number of trials.
        X = Multinomial(n=5, p=[0.5, 0.3, 0.2])
        got = X._joint_func(0, 1)(np.array([4]), np.array([4]))[0]
        self.assertEqual(float(got), 0.0)

    def test_Multinomial_marginal_1d_is_Binomial(self):
        X = Multinomial(n=10, p=[0.5, 0.3, 0.2])
        marginal = X._marginal_1d(1)
        self.assertIsInstance(marginal, Binomial)
        self.assertAlmostEqual(float(marginal.mean()), 3.0)

    def test_Multinomial_plot_window_zooms_when_counts_range_too_far(self):
        # A small number of trials shows every count, so the triangular
        # shape of the joint support is visible; a large one would need
        # more cells than are readable, so it falls back to the window
        # holding most of the probability.
        small = Multinomial(n=10, p=[0.5, 0.3, 0.2])
        self.assertEqual(len(small._plot_values(0)), 11)
        large = Multinomial(n=1000, p=[0.5, 0.3, 0.2])
        self.assertLess(len(large._plot_values(0)), 1001)
        large.plot()
        plt.close("all")

    def test_Multinomial_plot_pairs_is_titled_mass_functions(self):
        # Discrete panels are probability masses, not densities.
        plt.close("all")
        Multinomial(n=12, p=[0.3, 0.3, 0.2, 0.2]).plot()
        self.assertEqual(plt.gcf()._suptitle.get_text(), "Probability Mass Functions")
        plt.close("all")

    def test_Multinomial_plot_pairs_colorbars_say_probability(self):
        # A discrete joint panel shows a probability, not a density.
        plt.close("all")
        Multinomial(n=12, p=[0.3, 0.3, 0.2, 0.2]).plot()
        bars = [a for a in plt.gcf().axes if a.get_subplotspec() is None]
        self.assertEqual({a.get_ylabel() for a in bars}, {"Joint Probability"})
        plt.close("all")

    def test_Multinomial_plot_pairs(self):
        # A pairs plot fills the whole figure, so it refuses to draw into one
        # that already has axes. Start from a clean figure so the test does
        # not depend on whether an earlier test left one open -- without this
        # it passes alone and fails in a full run.
        plt.close("all")
        X = Multinomial(n=12, p=[0.4, 0.3, 0.2, 0.1])
        X.plot()
        # 10 panels; the 6 joint ones each carry a colorbar as well.
        panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
        self.assertEqual(len(panels), 10)
        plt.close("all")

    def test_Multinomial_pdf(self):
        X = Multinomial(n=10, p=[0.5, 0.3, 0.2])
        expected = stats.multinomial(10, [0.5, 0.3, 0.2]).pmf([5, 3, 2])
        self.assertAlmostEqual(float(X.pdf([5, 3, 2])), expected)

    def test_Multinomial_mean_cov_var(self):
        n, p = 10, [0.5, 0.3, 0.2]
        X = Multinomial(n=n, p=p)
        pa = np.asarray(p)
        np.testing.assert_allclose(np.array(X.mean()), n * pa)
        np.testing.assert_allclose(X.cov(), n * (np.diag(pa) - np.outer(pa, pa)))
        # Diagonal of the covariance is the per-category binomial variance.
        np.testing.assert_allclose(np.array(X.var()), n * pa * (1 - pa))

    def test_Multinomial_is_multivariate_distribution(self):
        X = Multinomial(n=10, p=[0.5, 0.5])
        self.assertIsInstance(X, MultivariateDistribution)


class TestMultivariateHypergeometric(MultivariatePlotTestCase):

    def test_MVHypergeom_error_m_negative(self):
        self.assertRaises(
            Exception, lambda: MultivariateHypergeometric(m=[10, -1, 6], n=5)
        )

    def test_MVHypergeom_error_m_non_integer(self):
        self.assertRaises(
            Exception, lambda: MultivariateHypergeometric(m=[10.5, 8, 6], n=5)
        )

    def test_MVHypergeom_error_n_too_large(self):
        # n cannot exceed the total number of items, sum(m) = 24.
        self.assertRaises(
            Exception, lambda: MultivariateHypergeometric(m=[10, 8, 6], n=25)
        )

    def test_MVHypergeom_error_n_negative(self):
        self.assertRaises(
            Exception, lambda: MultivariateHypergeometric(m=[10, 8, 6], n=-1)
        )

    def test_MVHypergeom_draw_sums_to_n(self):
        seed(42)
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)
        self.assertEqual(int(sum(draw)), 6)

    def test_MVHypergeom_mean_cov_var_match_scipy(self):
        m, n = [10, 8, 6], 6
        X = MultivariateHypergeometric(m=m, n=n)
        th = stats.multivariate_hypergeom(m, n)
        np.testing.assert_allclose(np.array(X.mean()), th.mean())
        np.testing.assert_allclose(X.cov(), th.cov())
        np.testing.assert_allclose(np.array(X.var()), th.var())
        np.testing.assert_allclose(np.array(X.sd()), np.sqrt(th.var()))

    def test_MVHypergeom_corr_diagonal_is_one(self):
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        np.testing.assert_allclose(np.diag(X.corr()), [1.0, 1.0, 1.0])

    def test_MVHypergeom_pdf_matches_scipy(self):
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        th = stats.multivariate_hypergeom([10, 8, 6], 6)
        for pt in [[2, 2, 2], [6, 0, 0], [1, 2, 3]]:
            self.assertAlmostEqual(float(X.pdf(pt)), float(th.pmf(pt)), places=12)

    def test_MVHypergeom_marginal_is_hypergeometric(self):
        # Lumping all other types together, each count is a univariate
        # hypergeometric: type i vs. the rest. Here type 0 has 10 of 24,
        # drawing 6, so the first count is Hypergeometric(n=6, N0=14, N1=10).
        seed(42)
        A, B, C = RV(MultivariateHypergeometric(m=[10, 8, 6], n=6))
        sims = A.sim(Nsim)
        expected = stats.hypergeom(
            M=24, n=10, N=6
        )  # scipy: M pop, n successes, N draws
        obs, exp = [], []
        for k in range(7):
            e = Nsim * expected.pmf(k)
            if e > 5:
                exp.append(e)
                obs.append(sum(1 for s in sims if s == k))
        pval = stats.chisquare(obs, np.array(exp) * sum(obs) / sum(exp)).pvalue
        self.assertTrue(pval > 0.01)

    def test_MVHypergeom_is_multivariate_distribution(self):
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        self.assertIsInstance(X, MultivariateDistribution)

    def test_MVHypergeom_is_discrete(self):
        # Every draw is a vector of whole counts.
        self.assertTrue(MultivariateHypergeometric(m=[10, 8, 6], n=6).discrete)

    def test_MVHypergeom_three_types_is_one_joint_plot(self):
        # The counts add up to n, so three types vary in only two
        # directions -- the single-joint-plot case, needing no dims.
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        self.assertEqual(X._free_dim(), 2)
        X.plot()
        plt.close("all")

    def test_MVHypergeom_marginal_is_hypergeometric(self):
        # Lumping the other types together leaves an ordinary
        # hypergeometric for one type's count.
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        marginal = X._marginal_1d(1)
        self.assertIsInstance(marginal, Hypergeometric)
        expected = Hypergeometric(n=6, N0=16, N1=8)
        for k in range(7):
            self.assertAlmostEqual(float(marginal.pdf(k)), float(expected.pdf(k)))

    def test_MVHypergeom_joint_func_totals_one_over_the_grid(self):
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        counts = np.arange(7)
        xs, ys = np.meshgrid(counts, counts)
        total = X._joint_func(0, 1)(xs.ravel(), ys.ravel()).sum()
        self.assertAlmostEqual(float(total), 1.0)

    def test_MVHypergeom_joint_func_matches_pooled(self):
        # Pooling the other types into one gives a three-type multivariate
        # hypergeometric exactly.
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        pooled = stats.multivariate_hypergeom([10, 8, 6], 6)
        func = X._joint_func(0, 1)
        for x in range(7):
            for y in range(7 - x):
                self.assertAlmostEqual(
                    float(func(np.array([x]), np.array([y]))[0]),
                    float(pooled.pmf([x, y, 6 - x - y])),
                )

    def test_MVHypergeom_joint_func_zero_for_impossible_pairs(self):
        # Two counts cannot use more than the n items drawn.
        X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
        func = X._joint_func(0, 1)
        self.assertEqual(float(func(np.array([4]), np.array([5]))[0]), 0.0)

    def test_MVHypergeom_plot_pairs(self):
        MultivariateHypergeometric(m=[10, 8, 6, 4], n=6).plot()
        plt.close("all")


class TestDirichlet(MultivariatePlotTestCase):

    def test_Dirichlet_error_alpha_non_positive(self):
        self.assertRaises(Exception, lambda: Dirichlet(alpha=[2, -1, 3]))

    def test_Dirichlet_error_alpha_zero(self):
        self.assertRaises(Exception, lambda: Dirichlet(alpha=[2, 0, 3]))

    def test_Dirichlet_error_alpha_too_short(self):
        self.assertRaises(Exception, lambda: Dirichlet(alpha=[2]))

    def test_Dirichlet_error_alpha_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "alpha must be a list of at least two positive numbers",
            lambda: Dirichlet(alpha=["a", "b"]),
        )

    def test_Dirichlet_mean_var_sd(self):
        alpha = [2, 3, 5]
        X = Dirichlet(alpha=alpha)
        expected = stats.dirichlet(alpha)
        # mean / var / sd each return one entry per category.
        self.assertEqual(len(X.mean()), 3)
        np.testing.assert_allclose(np.array(X.mean()), expected.mean())
        np.testing.assert_allclose(np.array(X.var()), expected.var())
        np.testing.assert_allclose(np.array(X.sd()), np.sqrt(expected.var()))

    def test_Dirichlet_cov_corr(self):
        alpha = [2, 3, 5]
        X = Dirichlet(alpha=alpha)
        expected = stats.dirichlet(alpha)
        np.testing.assert_allclose(X.cov(), expected.cov())
        # var() (base) is the diagonal of cov(); matches scipy's var().
        np.testing.assert_allclose(np.diag(X.cov()), expected.var())
        np.testing.assert_allclose(np.diag(X.corr()), [1.0, 1.0, 1.0])

    def test_Dirichlet_is_multivariate_distribution(self):
        X = Dirichlet(alpha=[2, 3, 5])
        self.assertIsInstance(X, MultivariateDistribution)

    def test_Dirichlet_pdf(self):
        alpha = [2, 3, 5]
        X = Dirichlet(alpha=alpha)
        point = [0.2, 0.3, 0.5]
        expected = stats.dirichlet(alpha).pdf(point)
        self.assertAlmostEqual(float(X.pdf(point)), float(expected))

    def test_Dirichlet_no_cdf(self):
        # A Dirichlet has no natural cdf; it should not advertise one.
        X = Dirichlet(alpha=[2, 3, 5])
        self.assertFalse(hasattr(X, "cdf"))

    def test_Dirichlet_draw_shape_and_sums_to_one(self):
        seed(42)
        X = Dirichlet(alpha=[2, 3, 5])
        draw = X.draw()
        self.assertEqual(len(draw), 3)
        self.assertTrue(np.all(np.array(draw) >= 0))
        self.assertAlmostEqual(float(sum(draw)), 1.0)

    def test_Dirichlet_sim_shape_and_sums_to_one(self):
        seed(42)
        X = Dirichlet(alpha=[2, 3, 5])
        sims = X.sim(100)
        arr = np.array(list(sims))
        self.assertEqual(arr.shape, (100, 3))
        np.testing.assert_allclose(arr.sum(axis=1), np.ones(100))

    def test_Dirichlet_marginals_match_Beta(self):
        # Each proportion X_i is marginally Beta(alpha_i, alpha0 - alpha_i).
        seed(42)
        alpha = [2, 3, 5]
        alpha0 = sum(alpha)
        components = RV(Dirichlet(alpha=alpha))
        for i, a_i in enumerate(alpha):
            sims = components[i].sim(Nsim)
            cdf = stats.beta(a_i, alpha0 - a_i).cdf
            pval = stats.kstest(sims, cdf).pvalue
            self.assertTrue(pval > 0.01)

    def test_Dirichlet_plots_joint_density(self):
        # Three proportions add to 1, so two of them vary freely: this is
        # the single-joint-plot case, drawn over the simplex.
        Dirichlet(alpha=[2, 3, 5]).draw()
        Dirichlet(alpha=[2, 3, 5]).plot()
        self.assertEqual(plt.gcf().get_suptitle(), "Joint Contour Plot")
        plt.close("all")

    def test_Dirichlet_plot_window_is_full_proportion_range(self):
        # Framed on [0, 1] on both axes, so the triangle a pair of
        # proportions lives on stays fully in view.
        X = Dirichlet(alpha=[2, 3, 5])
        X.plot()
        self.assertEqual(plt.gca().get_xlim(), (0.0, 1.0))
        self.assertEqual(plt.gca().get_ylim(), (0.0, 1.0))
        plt.close("all")

    def test_Dirichlet_two_categories_plot_points_to_Beta(self):
        X = Dirichlet(alpha=[2, 3])
        with self.assertRaises(Exception) as cm:
            X.plot()
        self.assertIn("Beta", str(cm.exception))

    def test_Dirichlet_marginal_1d_is_Beta(self):
        alpha = [2, 3, 5]
        X = Dirichlet(alpha=alpha)
        marginal = X._marginal_1d(1)
        self.assertIsInstance(marginal, Beta)
        self.assertAlmostEqual(float(marginal.mean()), 3 / 10)

    def test_Dirichlet_joint_func_integrates_to_one(self):
        # The joint density of two proportions is a real density over the
        # triangle where they sum to at most 1, so it integrates to 1.
        X = Dirichlet(alpha=[2, 3, 5])
        n = 600
        edges = np.linspace(0, 1, n + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        xs, ys = np.meshgrid(centers, centers)
        total = X._joint_func(0, 1)(xs.ravel(), ys.ravel()).sum() * (1.0 / n) ** 2
        self.assertAlmostEqual(float(total), 1.0, places=2)

    def test_Dirichlet_joint_func_zero_outside_the_simplex(self):
        # Two proportions can't add up to more than 1.
        X = Dirichlet(alpha=[2, 3, 5])
        got = X._joint_func(0, 1)(np.array([0.7]), np.array([0.7]))[0]
        self.assertEqual(float(got), 0.0)

    def test_Dirichlet_plots_pairs_with_Beta_marginals_on_the_diagonal(self):
        X = Dirichlet(alpha=[3, 2, 4, 5])
        X.plot()
        # 10 panels; the 6 joint ones each carry a colorbar as well.
        panels = [a for a in plt.gcf().axes if a.get_subplotspec() is not None]
        self.assertEqual(len(panels), 10)
        plt.close("all")

    def test_Dirichlet_plots_with_concentration_below_one(self):
        # Concentrations below 1 push the density to infinity at the edge of
        # the simplex; the surface is still drawn, scaled by its largest
        # finite value.
        Dirichlet(alpha=[0.5, 0.5, 0.5]).plot()
        plt.close("all")


class TestDirichletMultinomial(MultivariatePlotTestCase):

    def test_DirichletMultinomial_error_n_negative(self):
        self.assertRaises(
            Exception, lambda: DirichletMultinomial(n=-1, alpha=[2, 3, 5])
        )

    def test_DirichletMultinomial_error_n_float(self):
        self.assertRaises(
            Exception, lambda: DirichletMultinomial(n=5.5, alpha=[2, 3, 5])
        )

    def test_DirichletMultinomial_error_alpha_too_short(self):
        self.assertRaises(Exception, lambda: DirichletMultinomial(n=10, alpha=[2]))

    def test_DirichletMultinomial_error_alpha_non_positive(self):
        self.assertRaises(
            Exception, lambda: DirichletMultinomial(n=10, alpha=[2, -1, 3])
        )

    def test_DirichletMultinomial_draw_sums_to_n(self):
        seed(42)
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)
        self.assertEqual(int(sum(draw)), 10)

    def test_DirichletMultinomial_mean_cov_var_match_scipy(self):
        n, alpha = 10, [2, 3, 5]
        X = DirichletMultinomial(n=n, alpha=alpha)
        th = stats.dirichlet_multinomial(alpha, n)
        np.testing.assert_allclose(np.array(X.mean()), th.mean())
        np.testing.assert_allclose(X.cov(), th.cov())
        np.testing.assert_allclose(np.array(X.var()), th.var())
        np.testing.assert_allclose(np.array(X.sd()), np.sqrt(th.var()))

    def test_DirichletMultinomial_corr_diagonal_is_one(self):
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        np.testing.assert_allclose(np.diag(X.corr()), [1.0, 1.0, 1.0])

    def test_DirichletMultinomial_pdf_matches_scipy(self):
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        th = stats.dirichlet_multinomial([2, 3, 5], 10)
        for pt in [[3, 3, 4], [10, 0, 0], [1, 4, 5]]:
            self.assertAlmostEqual(float(X.pdf(pt)), float(th.pmf(pt)), places=12)

    def test_DirichletMultinomial_overdispersed_vs_multinomial(self):
        # Same mean as Multinomial(n, alpha/alpha0), but larger variances.
        n, alpha = 10, [2, 3, 5]
        alpha0 = sum(alpha)
        X = DirichletMultinomial(n=n, alpha=alpha)
        p = [a / alpha0 for a in alpha]
        mult_var = [n * pi * (1 - pi) for pi in p]
        np.testing.assert_allclose(
            np.array(X.mean()), [n * pi for pi in p]
        )  # same mean
        self.assertTrue(all(dv > mv for dv, mv in zip(X.var(), mult_var)))

    def test_DirichletMultinomial_marginal_is_beta_binomial(self):
        # Each count is marginally BetaBinomial(n, alpha_i, alpha0 - alpha_i).
        seed(42)
        A, B, C = RV(DirichletMultinomial(n=10, alpha=[2, 3, 5]))
        sims = A.sim(Nsim)
        th = stats.betabinom(n=10, a=2, b=8)  # alpha0 - alpha_0 = 10 - 2 = 8
        obs, exp = [], []
        for k in range(11):
            e = Nsim * th.pmf(k)
            if e > 5:
                exp.append(e)
                obs.append(sum(1 for s in sims if s == k))
        pval = stats.chisquare(obs, np.array(exp) * sum(obs) / sum(exp)).pvalue
        self.assertTrue(pval > 0.01)

    def test_DirichletMultinomial_is_multivariate_distribution(self):
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        self.assertIsInstance(X, MultivariateDistribution)

    def test_DirichletMultinomial_is_discrete(self):
        self.assertTrue(DirichletMultinomial(n=10, alpha=[2, 3, 5]).discrete)

    def test_DirichletMultinomial_three_categories_is_one_joint_plot(self):
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        self.assertEqual(X._free_dim(), 2)
        X.plot()
        plt.close("all")

    def test_DirichletMultinomial_marginal_is_betabinomial(self):
        # One count on its own is the univariate analogue, the
        # beta-binomial, with the other alphas pooled into the second shape.
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        marginal = X._marginal_1d(0)
        self.assertIsInstance(marginal, BetaBinomial)
        expected = BetaBinomial(n=10, shape1=2, shape2=8)
        for k in range(11):
            self.assertAlmostEqual(float(marginal.pdf(k)), float(expected.pdf(k)))

    def test_DirichletMultinomial_joint_func_totals_one_over_the_grid(self):
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        counts = np.arange(11)
        xs, ys = np.meshgrid(counts, counts)
        total = X._joint_func(0, 1)(xs.ravel(), ys.ravel()).sum()
        self.assertAlmostEqual(float(total), 1.0)

    def test_DirichletMultinomial_joint_func_matches_pooled(self):
        # Pooling the other categories adds up their alphas and gives a
        # three-category Dirichlet-multinomial exactly.
        X = DirichletMultinomial(n=8, alpha=[2, 3, 5, 4])
        pooled = stats.dirichlet_multinomial([2.0, 3.0, 9.0], 8)
        func = X._joint_func(0, 1)
        for x in range(9):
            for y in range(9 - x):
                self.assertAlmostEqual(
                    float(func(np.array([x]), np.array([y]))[0]),
                    float(pooled.pmf([x, y, 8 - x - y])),
                )

    def test_DirichletMultinomial_joint_func_zero_for_impossible_pairs(self):
        X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
        func = X._joint_func(0, 1)
        self.assertEqual(float(func(np.array([7]), np.array([6]))[0]), 0.0)

    def test_DirichletMultinomial_plot_pairs(self):
        DirichletMultinomial(n=10, alpha=[2, 3, 5, 4]).plot()
        plt.close("all")


class TestNegativeMultinomial(MultivariatePlotTestCase):

    def test_NegativeMultinomial_error_r_zero(self):
        self.assertRaisesRegex(
            Exception,
            "r must be a positive integer",
            lambda: NegativeMultinomial(r=0, p=[0.3, 0.2]),
        )

    def test_NegativeMultinomial_error_r_float(self):
        self.assertRaises(Exception, lambda: NegativeMultinomial(r=2.5, p=[0.3, 0.2]))

    def test_NegativeMultinomial_error_r_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "r must be a positive integer",
            lambda: NegativeMultinomial(r="x", p=[0.3, 0.2]),
        )

    def test_NegativeMultinomial_error_p_sums_to_one(self):
        # sum(p) == 1 leaves no probability for the stopping category, so the
        # run would never end.
        self.assertRaisesRegex(
            Exception,
            "p must be a list of non-negative numbers that sum to less than 1",
            lambda: NegativeMultinomial(r=3, p=[0.5, 0.5]),
        )

    def test_NegativeMultinomial_error_p_sums_above_one(self):
        self.assertRaises(Exception, lambda: NegativeMultinomial(r=3, p=[0.7, 0.6]))

    def test_NegativeMultinomial_error_p_sum_at_boundary_within_float_tolerance(self):
        # sum([1/3, 1/3, 1/3]) lands a few ULPs away from 1 in floating
        # point, but this is still the invalid "leaves no probability for
        # the stopping category" boundary and must still be rejected.
        self.assertRaises(Exception, lambda: NegativeMultinomial(r=3, p=[1 / 3] * 3))

    def test_NegativeMultinomial_p_sum_safely_below_one_constructs(self):
        NegativeMultinomial(r=3, p=[0.3, 0.2])

    def test_NegativeMultinomial_error_p_negative(self):
        self.assertRaises(Exception, lambda: NegativeMultinomial(r=3, p=[-0.1, 0.2]))

    def test_NegativeMultinomial_error_p_empty(self):
        self.assertRaises(Exception, lambda: NegativeMultinomial(r=3, p=[]))

    def test_NegativeMultinomial_error_p_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "p must be a list of non-negative numbers",
            lambda: NegativeMultinomial(r=3, p=["a", "b"]),
        )

    def test_NegativeMultinomial_p0_is_leftover(self):
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        self.assertAlmostEqual(X.p0, 0.5)

    def test_NegativeMultinomial_draw_shape(self):
        seed(42)
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 2)
        self.assertTrue(all(int(v) >= 0 for v in draw))

    def test_NegativeMultinomial_pdf_sums_to_one(self):
        # The support is unbounded, so check the pmf sums to 1 over a
        # truncation large enough that the remaining tail is negligible.
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        counts = np.arange(120)
        grid = np.stack(np.meshgrid(counts, counts, indexing="ij"), axis=-1)
        self.assertAlmostEqual(float(X.pdf(grid.reshape(-1, 2)).sum()), 1.0, places=10)

    def test_NegativeMultinomial_mean_cov_var_match_brute_force(self):
        # Ground truth summed straight from the pmf, independent of the
        # closed-form formulas being tested.
        r, p = 3, [0.3, 0.2]
        X = NegativeMultinomial(r=r, p=p)
        counts = np.arange(150)
        grid = np.stack(np.meshgrid(counts, counts, indexing="ij"), axis=-1).reshape(
            -1, 2
        )
        probabilities = X.pdf(grid)
        expected_mean = (probabilities[:, None] * grid).sum(axis=0)
        centered = grid - expected_mean
        expected_cov = (
            probabilities[:, None, None] * centered[:, :, None] * centered[:, None, :]
        ).sum(axis=0)
        np.testing.assert_allclose(np.array(X.mean()), expected_mean, atol=1e-8)
        np.testing.assert_allclose(X.cov(), expected_cov, atol=1e-8)
        np.testing.assert_allclose(np.array(X.var()), np.diag(expected_cov), atol=1e-8)
        np.testing.assert_allclose(
            np.array(X.sd()), np.sqrt(np.diag(expected_cov)), atol=1e-8
        )

    def test_NegativeMultinomial_counts_are_positively_correlated(self):
        # The distinguishing feature versus the Multinomial: nothing caps the
        # total, so every off-diagonal covariance is positive.
        X = NegativeMultinomial(r=3, p=[0.3, 0.2, 0.1])
        cov = X.cov()
        off_diagonal = cov[~np.eye(3, dtype=bool)]
        self.assertTrue(np.all(off_diagonal > 0))
        np.testing.assert_allclose(np.diag(X.corr()), [1.0, 1.0, 1.0])

    def test_NegativeMultinomial_pdf_impossible_counts_are_zero(self):
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        self.assertEqual(X.pdf([-1, 0]), 0.0)
        self.assertEqual(X.pdf([1.5, 0]), 0.0)

    def test_NegativeMultinomial_pdf_zero_probability_category(self):
        # A category with probability 0 never occurs, so any positive count
        # for it is impossible while a count of 0 is fine.
        X = NegativeMultinomial(r=2, p=[0.4, 0.0])
        self.assertGreater(X.pdf([1, 0]), 0.0)
        self.assertEqual(X.pdf([1, 1]), 0.0)

    def test_NegativeMultinomial_pdf_accepts_several_points(self):
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        points = [[0, 0], [1, 0], [2, 3]]
        np.testing.assert_allclose(X.pdf(points), [X.pdf(point) for point in points])

    def test_NegativeMultinomial_pdf_error_wrong_length(self):
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        self.assertRaisesRegex(
            Exception, "pdf needs one count per category", lambda: X.pdf([1, 2, 3])
        )

    def test_NegativeMultinomial_one_category_is_Pascal(self):
        # With a single counted category the run is an ordinary wait for the
        # r-th stop, so the count is exactly Pascal(r, p0).
        r, p1 = 4, 0.35
        X = NegativeMultinomial(r=r, p=[p1])
        Y = Pascal(r=r, p=1 - p1)
        for k in range(8):
            self.assertAlmostEqual(X.pdf([k]), float(Y.pmf(k)), places=12)
        self.assertAlmostEqual(float(X.mean()[0]), float(Y.mean()))
        self.assertAlmostEqual(float(X.var()[0]), float(Y.var()))

    def test_NegativeMultinomial_marginal_is_Pascal(self):
        # Ignoring every category but i and the stopping category, count i is
        # Pascal(r, p0 / (p0 + p_i)) -- this also checks draw() reproduces the
        # distribution the pmf describes.
        seed(42)
        r, p = 3, [0.3, 0.2]
        X = NegativeMultinomial(r=r, p=p)
        components = RV(X)
        for i, p_i in enumerate(p):
            sims = components[i].sim(Nsim)
            th = stats.nbinom(r, X.p0 / (X.p0 + p_i))
            obs, exp = [], []
            for k in range(60):
                e = Nsim * th.pmf(k)
                if e > 5:
                    exp.append(e)
                    obs.append(sum(1 for s in sims if s == k))
            pval = stats.chisquare(obs, np.array(exp) * sum(obs) / sum(exp)).pvalue
            self.assertTrue(pval > 0.01)

    def test_NegativeMultinomial_sim_shape(self):
        seed(42)
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        arr = np.array(list(X.sim(100)))
        self.assertEqual(arr.shape, (100, 2))

    def test_NegativeMultinomial_no_cdf(self):
        # Like the other multivariate distributions, it should not advertise
        # a cdf -- there is no natural way to order the vectors it produces.
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        self.assertFalse(hasattr(X, "cdf"))

    def test_NegativeMultinomial_is_multivariate_distribution(self):
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        self.assertIsInstance(X, MultivariateDistribution)

    def test_NegativeMultinomial_is_discrete(self):
        self.assertTrue(NegativeMultinomial(r=3, p=[0.3, 0.2]).discrete)

    def test_NegativeMultinomial_is_not_sum_constrained(self):
        # Unlike the Multinomial, nothing fixes a total here -- the number
        # of draws varies -- so every count varies freely.
        X = NegativeMultinomial(r=3, p=[0.3, 0.2, 0.25])
        self.assertEqual(X._free_dim(), 3)
        self.assertEqual(X._n_components(), 3)

    def test_NegativeMultinomial_two_categories_is_one_joint_plot(self):
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        self.assertEqual(X._free_dim(), 2)
        X.plot()
        plt.close("all")

    def test_NegativeMultinomial_marginal_is_pascal(self):
        # Watching one category and the stopping category, and ignoring the
        # rest, leaves Pascal(r, p0 / (p0 + p_i)).
        X = NegativeMultinomial(r=3, p=[0.3, 0.2])
        marginal = X._marginal_1d(0)
        self.assertIsInstance(marginal, Pascal)
        expected = Pascal(r=3, p=0.5 / 0.8)
        for k in range(15):
            self.assertAlmostEqual(float(marginal.pdf(k)), float(expected.pdf(k)))

    def test_NegativeMultinomial_joint_func_totals_one_over_the_grid(self):
        # The support is unbounded, so this only approaches 1; a wide grid
        # gets close enough to show no probability has gone missing.
        X = NegativeMultinomial(r=3, p=[0.3, 0.2, 0.25])
        counts = np.arange(220)
        xs, ys = np.meshgrid(counts, counts)
        total = X._joint_func(0, 1)(xs.ravel(), ys.ravel()).sum()
        self.assertAlmostEqual(float(total), 1.0, places=6)

    def test_NegativeMultinomial_joint_func_matches_renormalized_pair(self):
        # Ignoring the other categories rescales the three probabilities
        # still in play to add up to 1 -- without that rescaling the answer
        # is wrong, so this pins the rule down.
        X = NegativeMultinomial(r=3, p=[0.3, 0.2, 0.25])
        total = 0.25 + 0.3 + 0.2
        expected = NegativeMultinomial(r=3, p=[0.3 / total, 0.2 / total])
        func = X._joint_func(0, 1)
        for x in range(6):
            for y in range(6):
                self.assertAlmostEqual(
                    float(func(np.array([x]), np.array([y]))[0]),
                    float(expected.pdf([x, y])),
                )

    def test_NegativeMultinomial_joint_func_has_no_impossible_pairs(self):
        # No total is fixed, so even a large pair of counts is possible.
        X = NegativeMultinomial(r=3, p=[0.3, 0.2, 0.25])
        func = X._joint_func(0, 1)
        self.assertGreater(float(func(np.array([30]), np.array([30]))[0]), 0.0)

    def test_NegativeMultinomial_plot_pairs(self):
        NegativeMultinomial(r=3, p=[0.3, 0.2, 0.25]).plot()
        plt.close("all")


# ===========================================================================
# Parameter validation: type guards and helpful error messages
#
# Covers the wrong inputs a student is likely to make: a non-number (e.g. a
# string), a float where a whole number is required, or two mutually
# exclusive parameters. assertRaisesRegex checks the message text, so a
# cryptic low-level TypeError (e.g. "'<' not supported between ...") would
# fail the test rather than pass silently.
# ===========================================================================


class TestDrawReturnsScalar(unittest.TestCase):
    """Every scalar distribution's ``draw()`` returns a ``Scalar``.

    Regression tests: ``NegativeBinomial``, ``Cauchy``, and ``Pareto`` override
    ``draw`` to sample through numpy rather than scipy, and used to return a
    plain ``int``/``float`` while every other distribution returned a
    ``Scalar``. ``Int`` and ``Float`` subclass ``int`` and ``float``, so the
    values stay usable in arithmetic either way -- this pins the type so the
    overrides cannot drift from the base class again.
    """

    def test_overridden_draws_return_scalar(self):
        seed(42)
        for X in [
            NegativeBinomial(r=3, p=0.5),
            Cauchy(loc=0, scale=1),
            Pareto(shape=2, scale=1),
        ]:
            self.assertIsInstance(X.draw(), Scalar)

    def test_base_class_draws_return_scalar(self):
        seed(42)
        for X in [Normal(0, 1), Poisson(3), Binomial(10, 0.5), Gamma(2)]:
            self.assertIsInstance(X.draw(), Scalar)

    def test_overridden_draws_still_behave_as_numbers(self):
        seed(42)
        self.assertIsInstance(NegativeBinomial(r=3, p=0.5).draw(), int)
        self.assertIsInstance(Cauchy(loc=0, scale=1).draw(), float)
        self.assertIsInstance(Pareto(shape=2, scale=1).draw(), float)
        # Arithmetic and comparison keep working on the wrapped values.
        self.assertGreaterEqual(Pareto(shape=2, scale=3).draw() + 0, 3)

    def test_overridden_draws_respect_support(self):
        seed(42)
        for _ in range(200):
            # NegativeBinomial counts total trials, so it starts at r.
            self.assertGreaterEqual(NegativeBinomial(r=3, p=0.5).draw(), 3)
            # Pareto is bounded below by its scale.
            self.assertGreaterEqual(Pareto(shape=2, scale=3).draw(), 3)


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
            Exception, "a must be an integer", lambda: DiscreteUniform(a="x", b=5)
        )

    def test_DiscreteUniform_b_non_numeric(self):
        self.assertRaisesRegex(
            Exception, "b must be an integer", lambda: DiscreteUniform(a=0, b="x")
        )

    def test_DiscreteUniform_a_non_integer(self):
        # A fractional bound used to be accepted and then made every pmf,
        # mean, and quantile nan, with scipy raising nothing of its own.
        self.assertRaisesRegex(
            Exception, "a must be an integer", lambda: DiscreteUniform(a=0.5, b=5)
        )

    def test_DiscreteUniform_b_non_integer(self):
        self.assertRaisesRegex(
            Exception, "b must be an integer", lambda: DiscreteUniform(a=0, b=3.5)
        )

    def test_DiscreteUniform_accepts_whole_valued_floats(self):
        # Only a *fractional* bound is rejected. A whole-valued float works
        # fine in scipy's randint and reaches students easily -- out of a
        # division, or from a NumPy array -- so it must keep working, and the
        # bounds are stored as plain ints either way.
        for a, b in [(1.0, 6.0), (0, 6.0), (0, 10 / 2), (np.float64(1), np.float64(6))]:
            X = DiscreteUniform(a=a, b=b)
            self.assertIsInstance(X.a, int)
            self.assertIsInstance(X.b, int)
            reference = DiscreteUniform(a=int(a), b=int(b))
            self.assertAlmostEqual(float(X.mean()), float(reference.mean()))
            self.assertAlmostEqual(float(X.pmf(int(b))), float(reference.pmf(int(b))))

    def test_DiscreteUniform_whole_valued_float_bounds_compare(self):
        # The b < a check still fires when the bounds arrive as floats.
        self.assertRaisesRegex(
            Exception, "b cannot be less than a", lambda: DiscreteUniform(a=6.0, b=3.0)
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
            Exception,
            "shape1 must be a positive number",
            lambda: Beta(shape1="x", shape2=1),
        )

    def test_Beta_b_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "shape2 must be a positive number",
            lambda: Beta(shape1=1, shape2="x"),
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
            Exception, "shape must be a positive number", lambda: Pareto(shape="x")
        )

    def test_Pareto_scale_non_numeric(self):
        self.assertRaisesRegex(
            Exception,
            "scale must be a positive number",
            lambda: Pareto(shape=2, scale="x"),
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

    def test_Beta_stacks_shape1_and_shape2(self):
        with self.assertRaises(Exception) as cm:
            Beta(shape1=-1, shape2="x")
        message = str(cm.exception)
        self.assertIn("shape1 must be a positive number", message)
        self.assertIn("shape2 must be a positive number", message)

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
    """Default plotting x-limits: the support step.

    One rule, applied to each end of the support independently: a **fixed
    bound is used as-is**, an **unbounded side is cut at a quantile**
    (``_PLOT_TAIL`` / ``1 - _PLOT_TAIL``). This replaced a highest-density
    interval, which was removed: it cost a root-find per distribution to buy
    a window only a few percent narrower on most distributions, and one still
    too wide to read on the heavy-tailed ones it was meant to help.

    That window is then zoomed automatically when the probability fills too
    little of it -- :class:`TestDistributionAutoZoom` covers that second step,
    so every distribution used here is one whose probability does fill its
    window, leaving the support rule visible on its own.

    All checks read the closed-form ``xlim`` (no simulation), so they are
    deterministic.
    """

    TAIL = distributions._PLOT_TAIL

    def _quantile_window(self, d):
        return (float(d.quantile(self.TAIL)), float(d.quantile(1 - self.TAIL)))

    # --- bounded both ends: the full true support, nothing trimmed ---

    def test_binomial_keeps_full_support(self):
        # A small binomial's probability reaches its bounds, so both are kept
        # and the tails stay on screen. (Large n does not -- see
        # TestDistributionAutoZoom.)
        self.assertEqual(Binomial(10, 0.5).xlim, (0, 10))
        self.assertEqual(Binomial(10, 0.3).xlim, (0, 10))

    def test_bounded_continuous_keep_full_support(self):
        self.assertEqual(Beta(2, 5).xlim, (0, 1))
        self.assertEqual(Uniform(2, 7).xlim, (2, 7))
        self.assertEqual(Kumaraswamy(2, 3).xlim, (0, 1))
        self.assertEqual(Triangular(0, 2, 5).xlim, (0, 5))
        self.assertEqual(TruncatedNormal(0, 1, a=-2, b=2).xlim, (-2, 2))

    def test_bounded_discrete_keep_full_support(self):
        self.assertEqual(DiscreteUniform(1, 6).xlim, (1, 6))
        self.assertEqual(Bernoulli(0.3).xlim, (0, 1))
        self.assertEqual(BetaBinomial(10, 2, 3).xlim, (0, 10))
        self.assertEqual(Zipf(2, 10).xlim, (1, 10))

    def test_hypergeometric_window_holds_all_mass(self):
        # Its support isn't a simple (0, n), so assert the window covers all
        # of the probability rather than a specific tuple -- i.e. not trimmed.
        d = Hypergeometric(n=5, N0=10, N1=20)
        lo, hi = d.xlim
        self.assertAlmostEqual(float(d.cdf(hi) - d.cdf(lo - 1)), 1.0, places=9)

    # --- unbounded both ends: a quantile cut at each end ---

    def test_symmetric_unbounded_are_quantile_cut(self):
        for d in [Normal(0, 1), StudentT(5), Cauchy(0, 1), Laplace(0, 1)]:
            self.assertEqual(d.xlim, self._quantile_window(d))

    def test_skewed_unbounded_are_quantile_cut(self):
        # Skew no longer changes the rule -- that was the HDI's job.
        for d in [Gumbel(), GEV(), LogGamma(2), SkewT(2, 3)]:
            self.assertEqual(d.xlim, self._quantile_window(d))

    # --- fixed lower bound only: true bound below, quantile cut above ---

    def test_fixed_lower_bound_used_as_is(self):
        self.assertEqual(Poisson(2).xlim[0], 0)
        self.assertEqual(Exponential(rate=2).xlim[0], 0)
        self.assertEqual(Gamma(2, 3).xlim[0], 0)
        self.assertEqual(Rayleigh().xlim[0], 0)
        self.assertEqual(Geometric(0.3).xlim[0], 1)
        self.assertEqual(NegativeBinomial(4, 0.3).xlim[0], 4)
        self.assertEqual(Pareto(3, 5).xlim[0], 5)  # scale
        self.assertEqual(GPD(loc=2, scale=1, shape=0.3).xlim[0], 2)  # loc

    def test_fixed_lower_bound_upper_edge_is_quantile(self):
        # This is what the HDI removal changed: the upper edge is now exactly
        # quantile(1 - _PLOT_TAIL), with no highest-density trim.
        for d in [
            Poisson(2),
            Exponential(rate=2),
            Gamma(2, 3),
            ChiSquare(4),
            F(5, 7),
            LogNormal(0, 1),
            Weibull(2, 3),
            Geometric(0.3),
            Pareto(3, 2),
        ]:
            self.assertAlmostEqual(
                float(d.xlim[1]), float(d.quantile(1 - self.TAIL)), places=9
            )

    # --- the rule holds for every distribution, read from the true support ---

    def test_rule_matches_true_support_for_every_distribution(self):
        # Data-driven: whatever scipy reports as the support, a finite end must
        # appear verbatim in xlim and an infinite end must be a quantile. Every
        # distribution here fills its own window, so none of them zooms --
        # Binomial(1000, 0.5) is deliberately not in the list, since it does.
        for d in [
            Bernoulli(0.3),
            Binomial(20, 0.5),
            BetaBinomial(10, 2, 3),
            Hypergeometric(5, 10, 7),
            NegativeHypergeometric(3, 5, 4),
            DiscreteUniform(1, 6),
            DeMoivre(6),
            Zipf(2, 10),
            Uniform(1, 4),
            Beta(2, 3),
            Kumaraswamy(2, 3),
            LogUniform(2, 3),
            Bates(5),
            IrwinHall(5),
            PERT(0, 2, 5),
            Triangular(0, 2, 5),
            TruncatedNormal(0, 1, a=-2, b=2),
            BetaNegativeBinomial(3, 2, 4),
            Geometric(0.3),
            NegativeBinomial(4, 0.3),
            Pascal(4, 0.3),
            Poisson(2),
            Exponential(rate=2),
            Gamma(2, 3),
            InverseGamma(3, 2),
            ChiSquare(4),
            F(5, 7),
            LogNormal(0, 1),
            Pareto(3, 2),
            Burr(3, 2),
            Lomax(3),
            Rayleigh(),
            HalfNormal(2),
            HalfCauchy(2),
            Weibull(2, 3),
            Gompertz(2, 3),
            Makeham(2, 3, 1),
            GPD(0.3, 1, 2),
            Normal(0, 1),
            Cauchy(),
            StudentT(5),
            Logistic(),
            Laplace(),
            Gumbel(),
            GEV(),
            LogGamma(2),
            SkewT(2, 3),
            ExponentiallyModifiedGaussian(),
        ]:
            name = type(d).__name__
            support_low, support_high = d._support()
            lo, hi = d.xlim
            if np.isfinite(support_low):
                self.assertEqual(float(lo), support_low, name)
            else:
                self.assertAlmostEqual(float(lo), float(d.quantile(self.TAIL)), 9, name)
            if np.isfinite(support_high):
                self.assertEqual(float(hi), support_high, name)
            else:
                self.assertAlmostEqual(
                    float(hi), float(d.quantile(1 - self.TAIL)), 9, name
                )

    # --- degenerate parameters still give a usable, ordered window ---

    def test_degenerate_parameters_give_ordered_window(self):
        # scipy can return a quantile outside its own support for a degenerate
        # parameter -- geom.ppf(0.999, p=1) is 0.0 although the support starts
        # at 1 -- which would invert the window. It must collapse onto the
        # exact support bound instead.
        # scipy itself warns ("divide by zero in log1p") when asked for a
        # quantile of a degenerate discrete distribution; that noise is not
        # what this test is about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.assertEqual(Geometric(1).xlim, (1, 1))
            for d in [
                Geometric(1),
                Poisson(0),
                Zipf(2, 1),
                Bernoulli(0),
                Bernoulli(1),
                NegativeBinomial(3, 1),
                Pascal(3, 1),
                DiscreteUniform(3, 3),
            ]:
                lo, hi = d.xlim
                self.assertLessEqual(lo, hi, type(d).__name__)

    # --- the HDI machinery is gone and must stay gone ---

    def test_hdi_helpers_removed(self):
        for name in ["_discrete_hdi_xlim", "_continuous_hdi_xlim", "_PLOT_COVERAGE"]:
            self.assertFalse(
                hasattr(distributions, name),
                f"{name} was removed with the HDI window; do not reintroduce it",
            )
        self.assertFalse(hasattr(Normal(0, 1), "_hdi_window"))

    def test_source_mentions_no_hdi(self):
        source = inspect.getsource(distributions)
        self.assertNotIn("hdi", source.lower())


class TestDistributionAutoZoom(unittest.TestCase):
    """The second step of the default window: it zooms itself.

    The window derived from the support (:class:`TestDistributionXlim`) is
    kept only while the probability fills at least ``_ZOOM_FRACTION`` of it.
    Below that the plot is mostly empty axis, so the two-sided quantile
    window (``_zoom_xlim``) is used instead. Nothing is passed to ``plot`` to
    get this -- there is no window argument at all any more, which
    :class:`TestDistributionPlotTakesNoXlim` covers. The overlay and
    rendering checks use a non-interactive backend.
    """

    TAIL = distributions._PLOT_TAIL

    def tearDown(self):
        plt.close("all")

    # --- the zoomed window is exactly the two-sided quantile window ---

    def test_zoom_is_the_quantile_window(self):
        for d in [
            Binomial(1000, 0.5),
            Poisson(20),
            Geometric(0.2),
            Beta(2, 3),
            Uniform(2, 7),
            Gamma(2),
            LogNormal(0, 1),
            Normal(0, 1),
        ]:
            self.assertEqual(
                d._zoom_xlim(),
                (float(d.quantile(self.TAIL)), float(d.quantile(1 - self.TAIL))),
                type(d).__name__,
            )

    # --- a spread-out distribution zooms off its bounds, by itself ---

    def test_wide_binomial_zooms_itself(self):
        # The motivating example: a thousand-trial binomial puts everything in
        # a band a tenth of its support wide, so (0, 1000) is drawn as a spike
        # in an empty axis. It now frames the band with nothing passed in.
        d = Binomial(1000, 0.5)
        self.assertEqual(d.xlim, d._zoom_xlim())
        lo, hi = d.xlim
        self.assertGreater(lo, 400)
        self.assertLess(hi, 600)

    def test_concentrated_continuous_zooms_itself(self):
        # Beta(2, 200) lives in the first few percent of (0, 1).
        d = Beta(2, 200)
        self.assertEqual(d.xlim, d._zoom_xlim())
        self.assertLess(d.xlim[1], 0.1)

    def test_zoom_gives_up_a_fixed_lower_bound_too(self):
        # A half-bounded distribution is judged the same way: Poisson(1000)
        # has a true lower bound of 0 but nothing within 900 of it.
        d = Poisson(1000)
        self.assertEqual(d.xlim, d._zoom_xlim())
        self.assertGreater(d.xlim[0], 800)
        # Poisson(2) does reach its bound, so the bound is kept.
        self.assertEqual(Poisson(2).xlim[0], 0)

    # --- a distribution that fills its window keeps every bound ---

    def test_filled_windows_are_left_alone(self):
        # Their probability reaches the bounds, so the tails stay on screen
        # and a student sees the true endpoints.
        for d, expected in [
            (Binomial(10, 0.5), (0, 10)),
            (DiscreteUniform(1, 6), (1, 6)),
            (Bernoulli(0.3), (0, 1)),
            (Uniform(2, 7), (2, 7)),
            (Beta(2, 5), (0, 1)),
            (Triangular(0, 2, 5), (0, 5)),
            (TruncatedNormal(0, 1, a=-2, b=2), (-2, 2)),
        ]:
            self.assertEqual(d.xlim, expected, type(d).__name__)

    def test_crossover_is_the_zoom_fraction(self):
        # Directly the rule, swept across a family that crosses it: a binomial
        # keeps (0, n) exactly while its probability fills at least
        # _ZOOM_FRACTION of it, and is the quantile window otherwise.
        fraction = distributions._ZOOM_FRACTION
        crossed = set()
        for n in [2, 5, 10, 20, 30, 50, 100, 300, 1000]:
            d = Binomial(n, 0.5)
            zoom_lo, zoom_hi = d._zoom_xlim()
            fills = (zoom_hi - zoom_lo) >= fraction * n
            crossed.add(fills)
            self.assertEqual(d.xlim, (0, n) if fills else (zoom_lo, zoom_hi), f"n={n}")
        # The sweep must actually straddle the crossover, or it proves nothing.
        self.assertEqual(crossed, {True, False})

    # --- an unbounded distribution has nothing to give up ---

    def test_unbounded_are_unchanged(self):
        for d in [Normal(0, 1), Cauchy(0, 1), StudentT(5), Gumbel(), SkewT(2, 3)]:
            self.assertEqual(d.xlim, d._zoom_xlim(), type(d).__name__)

    def test_zoom_never_reaches_outside_the_support(self):
        # The quantile window sits inside the window the support gives, so
        # zooming can only ever tighten the view -- it never invents range.
        for d in [
            Poisson(50),
            Exponential(1),
            Gamma(2),
            Geometric(0.3),
            Beta(2, 5),
            Binomial(1000, 0.5),
        ]:
            name = type(d).__name__
            support_lo, support_hi = d._support()
            zoom_lo, zoom_hi = d._zoom_xlim()
            if np.isfinite(support_lo):
                self.assertGreaterEqual(zoom_lo, support_lo, name)
            if np.isfinite(support_hi):
                self.assertLessEqual(zoom_hi, support_hi, name)

    # --- what the plot actually draws ---

    def test_plot_uses_the_zoomed_window(self):
        # The drawn window is the zoomed one, plus the half step of air a
        # discrete plot puts around a window it chose for itself.
        plt.figure()
        d = Binomial(100, 0.5)
        d.plot()
        low, high = d.xlim
        self.assertEqual(tuple(plt.gca().get_xlim()), (low - 0.5, high + 0.5))
        lo, hi = plt.gca().get_xlim()
        self.assertGreater(lo, 5)  # not the full (0, 100) support
        self.assertLess(hi, 95)

    def test_plot_keeps_a_filled_window_whole(self):
        # All of (0, 10) is on screen, with half a step of air at each end so
        # the dots on 0 and 10 don't sit on the axis spine.
        plt.figure()
        Binomial(10, 0.5).plot()
        self.assertEqual(tuple(plt.gca().get_xlim()), (-0.5, 10.5))

    # --- overlay: the theoretical curve no longer stretches the shared axis ---

    def test_overlay_does_not_stretch_to_full_support(self):
        # 10000 draws of Binomial(100, 0.5) realize only a narrow band, and the
        # theoretical curve's full (0, 100) support would widen the shared axis
        # to the whole range. The automatic zoom keeps it tight.
        plt.figure()
        RV(Binomial(100, 0.5)).sim(10000).plot()
        Binomial(100, 0.5).plot()
        lo, hi = plt.gca().get_xlim()
        self.assertGreater(lo, 5)
        self.assertLess(hi, 95)

    # --- a window set by hand still wins ---

    def test_window_set_by_hand_is_used_as_given(self):
        # Exactly (10, 90): a hand-chosen window skips the half step of
        # padding a self-chosen discrete window gets.
        plt.figure()
        d = Binomial(100, 0.5)
        d.xlim = (10, 90)
        d.plot()
        self.assertEqual(tuple(plt.gca().get_xlim()), (10.0, 90.0))

    # --- every base-plot distribution renders on its own window ---

    def test_default_window_renders_for_every_distribution(self):
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
            name = type(d).__name__
            for window in (d.xlim, d._zoom_xlim()):
                lo, hi = window
                self.assertTrue(np.isfinite(lo) and np.isfinite(hi), name)
                self.assertLessEqual(lo, hi, name)
            plt.figure()
            d.plot()  # must not raise
            plt.close("all")

    # --- a window that collapses to one value still yields a usable axis ---

    def test_single_value_window_expands_and_is_quiet(self):
        # When one outcome carries essentially all the mass the window
        # collapses to a point; the axis must not be set to a singular
        # (equal) range, and no raw matplotlib warning should reach the user.
        plt.figure()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Geometric(0.999).plot()
            lo, hi = plt.gca().get_xlim()
        plt.close("all")
        self.assertLess(lo, hi)  # not singular
        self.assertFalse(
            any("identical" in str(w.message).lower() for w in caught),
            "set_xlim received identical limits",
        )


class TestDistributionPlotTakesNoXlim(unittest.TestCase):
    """``Distribution.plot()`` has no window argument at all.

    ``xlim=`` used to select the window, including the string ``"zoom"`` for
    a tight one. The window chooses itself now (see
    :class:`TestDistributionAutoZoom`), so the argument is gone rather than
    left as a near-no-op, and a stray one gets a pointer to the two ways of
    setting a window by hand instead of an opaque matplotlib error.
    """

    def tearDown(self):
        plt.close("all")

    def test_signature_has_no_xlim(self):
        params = inspect.signature(Normal(0, 1).plot).parameters
        self.assertNotIn("xlim", params)

    def test_xlim_argument_raises_with_a_pointer(self):
        for bad in ["zoom", "trim", "hdi", (10, 90), None]:
            plt.figure()
            with self.assertRaises(ValueError) as cm:
                Binomial(100, 0.5).plot(xlim=bad)
            message = str(cm.exception)
            self.assertIn("does not take an `xlim=` argument", message)
            self.assertIn("X.xlim = ", message)  # how to set one by hand

    def test_setting_xlim_by_hand_still_works(self):
        # The replacement the message points at, both ways round.
        plt.figure()
        X = Poisson(3)
        X.xlim = (0, 4)
        X.plot()
        self.assertEqual(tuple(plt.gca().get_xlim()), (0.0, 4.0))
        plt.close("all")
        plt.figure()
        Poisson(3).plot()
        plt.xlim(0, 4)
        self.assertEqual(tuple(plt.gca().get_xlim()), (0.0, 4.0))


class TestDistributionXlimLaziness(unittest.TestCase):
    """``xlim`` is computed on first read, never in ``__init__``.

    A window is only ever needed for plotting, but ``__init__`` runs far more
    often: ``PoissonProcess`` and ``ContinuousTimeMarkovChain`` build a fresh
    ``Exponential`` on *every draw*. When the window was computed eagerly --
    and, at the time, by highest-density root-finding -- ``.sim(1000)`` on a
    Poisson process took about 26 seconds to produce a plot nobody asked for.
    The HDI is gone, but the laziness is kept on its own merits: no
    simulation should pay for a plotting window.
    """

    def setUp(self):
        """Count window computations, restoring the original in tearDown."""
        self.calls = []
        self._original = distributions.Distribution._compute_xlim

        def counted(inner_self):
            self.calls.append(type(inner_self).__name__)
            return self._original(inner_self)

        distributions.Distribution._compute_xlim = counted

    def tearDown(self):
        distributions.Distribution._compute_xlim = self._original

    # --- construction alone computes nothing ---

    def test_construction_does_not_compute_window(self):
        for make in [
            lambda: Exponential(rate=2),
            lambda: Gamma(2, 3),
            lambda: Poisson(3),
            lambda: Geometric(0.3),
            lambda: Pareto(3, 2),
            lambda: Weibull(2, 3),
            lambda: Binomial(1000, 0.5),
            lambda: Normal(0, 1),
            lambda: Beta(2, 3),
        ]:
            self.calls.clear()
            make()
            self.assertEqual(self.calls, [])

    def test_first_read_computes_and_caches(self):
        d = Exponential(rate=2)
        self.assertEqual(self.calls, [])
        first = d.xlim
        self.assertEqual(self.calls, ["Exponential"])
        # A second read reuses the cached window rather than recomputing.
        self.assertIs(d.xlim, first)
        self.assertEqual(self.calls, ["Exponential"])

    def test_plot_computes_the_window(self):
        plt.figure()
        Exponential(rate=2).plot()
        self.assertEqual(self.calls, ["Exponential"])
        plt.close("all")

    def test_setting_xlim_overrides_the_default(self):
        d = Exponential(rate=2)
        d.xlim = (0, 5)
        self.assertEqual(d.xlim, (0, 5))
        self.assertEqual(self.calls, [])

    def test_xlim_set_before_super_init_survives(self):
        # `Zeta` sets its window *before* calling up, because its
        # support-derived default would ask for the 99.9th percentile of a
        # power law -- a quantile search that effectively never returns. So
        # `Distribution.__init__` must not reset the window, and no default
        # may be computed for it.
        self.assertEqual(Zeta(2.5).xlim, (1, 20))
        self.assertEqual(Zeta(1.1).xlim, (1, 20))
        self.assertEqual(self.calls, [])

        # The same contract, on a throwaway subclass, so the guarantee is
        # tested directly rather than only through Zeta.
        class PreSet(distributions.Distribution):
            def __init__(self):
                self.xlim = (-7, 7)
                super().__init__({"loc": 0, "scale": 1}, stats.norm, False)

        self.assertEqual(PreSet().xlim, (-7, 7))
        self.assertEqual(self.calls, [])

    # --- regression: simulating a process must not compute any window ---

    def test_poisson_process_sim_computes_no_window(self):
        # The reported bug: `Exponential` is rebuilt on every draw, so an
        # eager window meant 1000 computations for one `.sim(1000)`.
        X = RV(PoissonProcessProbabilitySpace(rate=2))
        X[2.3].sim(100)
        self.assertEqual(self.calls, [])

    def test_poisson_process_rv_sim_computes_no_window(self):
        PoissonProcess(rate=2)[2.3].sim(100)
        self.assertEqual(self.calls, [])

    def test_continuous_time_markov_chain_sim_computes_no_window(self):
        generator = [[-2, 1, 1], [1, -2, 1], [1, 1, -2]]
        ContinuousTimeMarkovChain(generator, [1, 0, 0])[1.5].sim(100)
        self.assertEqual(self.calls, [])

    def test_distribution_sim_computes_no_window(self):
        # Plain simulation never needs a plotting window either.
        RV(Exponential(rate=2)).sim(100)
        self.assertEqual(self.calls, [])

    # --- multivariate distributions have no window at all ---

    def test_multivariate_has_no_xlim(self):
        # These set up only a pdf and override `plot` to raise, so `.xlim` is
        # meaningless -- and must stay absent rather than fail confusingly
        # deeper in on a missing `quantile` or scipy object.
        for make in [
            lambda: MultivariateNormal([0, 0], [[1, 0], [0, 1]]),
            lambda: BivariateNormal(),
            lambda: Multinomial(10, [0.3, 0.3, 0.4]),
            lambda: Dirichlet([1, 2, 3]),
        ]:
            d = make()
            self.assertFalse(hasattr(d, "xlim"))
            with self.assertRaises(AttributeError) as cm:
                d.xlim
            self.assertIn("no 'xlim'", str(cm.exception))
            self.assertNotIn("quantile", str(cm.exception))


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
            self.assertEqual(
                d.plot(cdf=True).ax.get_title(), "Cumulative Distribution Function"
            )
            plt.close("all")

    def test_default_pmf_title_for_discrete(self):
        plt.figure()
        self.assertEqual(
            Binomial(10, 0.5).plot().ax.get_title(), "Probability Mass Function"
        )

    def test_default_pdf_title_for_continuous(self):
        plt.figure()
        self.assertEqual(
            Normal(0, 1).plot().ax.get_title(), "Probability Density Function"
        )

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

    # --- the automatically zoomed x-window is reused unchanged for the CDF ---

    def test_cdf_reuses_zoom_window(self):
        # The window is picked before `cdf` is looked at, so both curves are
        # drawn over the same automatically zoomed range.
        plt.figure()
        pdf_win = Binomial(100, 0.5).plot().ax.get_xlim()
        plt.close("all")
        plt.figure()
        cdf_win = Binomial(100, 0.5).plot(cdf=True).ax.get_xlim()
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


class TestDistributionDegenerateParameterGuard(unittest.TestCase):
    """Guard 1: a distribution whose parameters leave nothing to draw gets a
    friendly error naming itself and its window, not a bare NumPy or
    matplotlib internals error.

    Two distinct degenerate shapes, so two distinct messages. The window
    itself can come back non-finite -- ``(nan, nan)`` for zero variance,
    ``(1.0, inf)`` for a tail so heavy the 0.999 quantile never returns --
    or the window can be fine while every evaluated pdf/pmf/cdf value is
    non-finite. Before this guard the first reached matplotlib as "Axis
    limits cannot be NaN or Inf" and the second reached NumPy as "zero-size
    array to reduction operation maximum which has no identity".
    """

    def setUp(self):
        plt.figure()
        # A degenerate distribution warns on its way to the error (identical
        # y-limits, invalid values in scipy); the message is what's tested.
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("ignore")

    def tearDown(self):
        self._warnings.__exit__(None, None, None)
        plt.close("all")

    # The four reproductions from the graphics testing strategy document.
    NAN_WINDOW = [
        ("Normal", lambda: Normal(mean=0, sd=0)),
        ("Uniform", lambda: Uniform(a=5, b=5)),
    ]
    INFINITE_WINDOW = [("Pareto", lambda: Pareto(shape=1e-3, scale=1))]
    NONFINITE_CURVE = [("Gamma", lambda: Gamma(shape=1e-6, rate=1))]

    def test_nan_window_names_the_degenerate_edge(self):
        for name, build in self.NAN_WINDOW:
            for cdf in [False, True]:
                with self.subTest(distribution=name, cdf=cdf):
                    plt.figure()
                    with self.assertRaises(ValueError) as caught:
                        build().plot(cdf=cdf)
                    message = str(caught.exception)
                    self.assertIn(name, message)
                    self.assertIn("not a finite range of values", message)
                    self.assertIn("degenerate edge", message)
                    plt.close("all")

    def test_infinite_window_names_the_heavy_tail(self):
        for name, build in self.INFINITE_WINDOW:
            for cdf in [False, True]:
                with self.subTest(distribution=name, cdf=cdf):
                    plt.figure()
                    with self.assertRaises(ValueError) as caught:
                        build().plot(cdf=cdf)
                    message = str(caught.exception)
                    self.assertIn(name, message)
                    self.assertIn("not a finite range of values", message)
                    self.assertIn("heavy", message)
                    plt.close("all")

    def test_nonfinite_curve_names_the_distribution_and_window(self):
        for name, build in self.NONFINITE_CURVE:
            with self.subTest(distribution=name):
                with self.assertRaises(ValueError) as caught:
                    build().plot()
                message = str(caught.exception)
                self.assertIn(name, message)
                self.assertIn("undefined or infinite everywhere", message)
                # The window is named so a student can see what was searched.
                self.assertIn("plotting window", message)

    def test_no_raw_numpy_or_matplotlib_error_survives(self):
        # The point of the guard: none of the four may reach the student as
        # the underlying library's own wording.
        raw = ["zero-size array", "Axis limits cannot be NaN or Inf"]
        for name, build in (
            self.NAN_WINDOW + self.INFINITE_WINDOW + self.NONFINITE_CURVE
        ):
            with self.subTest(distribution=name):
                plt.figure()
                with self.assertRaises(ValueError) as caught:
                    build().plot()
                for fragment in raw:
                    self.assertNotIn(fragment, str(caught.exception))
                plt.close("all")

    def test_ordinary_distributions_are_unaffected(self):
        # The guard must not fire on anything well-behaved, including the
        # heavy-tailed and single-point-collapse cases that already worked.
        for d in [
            Normal(0, 1),
            Poisson(3),
            Binomial(10, 0.5),
            HalfCauchy(1),
            Geometric(0.99),
            Cauchy(0, 1),
            Beta(0.5, 0.5),
        ]:
            for cdf in [False, True]:
                with self.subTest(distribution=repr(d), cdf=cdf):
                    plt.figure()
                    d.plot(cdf=cdf)  # must not raise
                    plt.close("all")


class TestTheoreticalCDFPDFOverlayGuard(unittest.TestCase):
    """Guard 2: a pdf/pmf and a cdf can't share one axes.

    They use incompatible y-scales (a density can exceed 1, a cdf runs 0 to
    1), so the combined plot has no correct reading -- this is the hard-error
    tier of the overlay policy, alongside MARGINAL_OVERLAY_ERROR. The tag
    lives on the Axes, so separate Axes objects (ax.twinx(), plt.subplots())
    are the escape hatch for anyone who does want both.
    """

    def tearDown(self):
        plt.close("all")

    # --- same kind still overlays, exactly as before ---

    def test_two_pdfs_still_overlay(self):
        plt.figure()
        Normal(0, 1).plot()
        Normal(2, 1).plot()
        self.assertEqual(len(plt.gca().get_lines()), 2)

    def test_two_cdfs_still_overlay(self):
        plt.figure()
        Normal(0, 1).plot(cdf=True)
        Normal(2, 1).plot(cdf=True)
        self.assertEqual(len(plt.gca().get_lines()), 2)

    def test_a_pmf_and_a_pdf_are_the_same_kind(self):
        # Both are tagged "pdf/pmf", so a discrete and a continuous curve
        # overlay without complaint.
        plt.figure()
        Binomial(10, 0.5).plot()
        Normal(5, 2).plot()
        self.assertGreaterEqual(len(plt.gca().get_lines()), 2)

    # --- mismatched kinds raise ---

    def test_pdf_then_cdf_raises(self):
        plt.figure()
        Normal(0, 1).plot()
        with self.assertRaises(ValueError) as caught:
            Normal(0, 1).plot(cdf=True)
        message = str(caught.exception)
        self.assertIn("pdf/pmf", message)
        self.assertIn("cdf", message)
        self.assertIn("different y-axis scales", message)

    def test_cdf_then_pdf_raises(self):
        plt.figure()
        Normal(0, 1).plot(cdf=True)
        with self.assertRaises(ValueError) as caught:
            Normal(0, 1).plot()
        self.assertIn("different y-axis scales", str(caught.exception))

    def test_error_message_is_the_named_constant(self):
        plt.figure()
        Poisson(3).plot()
        with self.assertRaises(ValueError) as caught:
            Poisson(3).plot(cdf=True)
        self.assertEqual(
            str(caught.exception),
            THEORETICAL_CDF_PDF_OVERLAY_ERROR.format(existing="pdf/pmf", new="cdf"),
        )

    def test_refused_overlay_leaves_the_first_plot_intact(self):
        # The check runs before anything is drawn, so a refusal doesn't
        # damage the plot that was already there.
        plt.figure()
        Normal(0, 1).plot()
        before = len(plt.gca().get_lines())
        with self.assertRaises(ValueError):
            Normal(0, 1).plot(cdf=True)
        self.assertEqual(len(plt.gca().get_lines()), before)
        self.assertEqual(plt.gca().get_title(), "Probability Density Function")

    # --- escape hatches ---

    def test_twinx_is_an_escape_hatch(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        Normal(0, 1).plot(ax=ax1)
        Normal(0, 1).plot(ax=ax2, cdf=True)  # must not raise
        self.assertEqual(len(ax1.get_lines()), 1)
        self.assertEqual(len(ax2.get_lines()), 1)

    def test_separate_subplots_are_an_escape_hatch(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        Normal(0, 1).plot(ax=ax1)
        Normal(0, 1).plot(ax=ax2, cdf=True)  # must not raise
        self.assertEqual(ax1.get_title(), "Probability Density Function")
        self.assertEqual(ax2.get_title(), "Cumulative Distribution Function")

    def test_a_fresh_figure_is_unaffected(self):
        plt.figure()
        Normal(0, 1).plot()
        plt.figure()
        Normal(0, 1).plot(cdf=True)  # new axes, new tag
        self.assertEqual(plt.gca().get_title(), "Cumulative Distribution Function")

    # --- the tag itself ---

    def test_axes_records_which_kind_was_drawn(self):
        plt.figure()
        Normal(0, 1).plot()
        self.assertEqual(plt.gca()._symbulate_theoretical_kind, "pdf/pmf")
        plt.figure()
        Normal(0, 1).plot(cdf=True)
        self.assertEqual(plt.gca()._symbulate_theoretical_kind, "cdf")

    def test_an_empty_tagged_axes_does_not_block(self):
        # The guard requires ax.has_data(): a tag left on an axes that was
        # since cleared is not a plot to conflict with.
        plt.figure()
        Normal(0, 1).plot()
        plt.gca().clear()
        Normal(0, 1).plot(cdf=True)  # must not raise
        self.assertEqual(plt.gca().get_title(), "Cumulative Distribution Function")


class TestDistributionPlotKeepsFirstTitle(unittest.TestCase):
    """``Distribution.plot()`` sets a title only when the axes has none.

    Titles used to always overwrite while labels only filled in when unset,
    so an overlay could end up titled by the second curve and labeled by the
    first. Both now keep the first plot's framing.
    """

    def tearDown(self):
        plt.close("all")

    def test_lone_plot_still_gets_its_own_title(self):
        for d, expected in [
            (Normal(0, 1), "Probability Density Function"),
            (Binomial(10, 0.5), "Probability Mass Function"),
        ]:
            with self.subTest(distribution=repr(d)):
                plt.figure()
                self.assertEqual(d.plot().ax.get_title(), expected)
                plt.close("all")

    def test_cdf_lone_plot_still_gets_its_own_title(self):
        plt.figure()
        self.assertEqual(
            Normal(0, 1).plot(cdf=True).ax.get_title(),
            "Cumulative Distribution Function",
        )

    def test_second_curve_keeps_the_first_title(self):
        plt.figure()
        Normal(0, 1).plot()
        Normal(3, 1).plot()
        self.assertEqual(plt.gca().get_title(), "Probability Density Function")

    def test_pdf_over_a_histogram_keeps_the_histograms_framing(self):
        # The case the keep-first rule exists for: title and y-label now
        # agree, both belonging to the plot that was there first.
        plt.figure()
        RV(Normal(0, 1)).sim(200).plot(type="hist", suggest=False)
        title, ylabel = plt.gca().get_title(), plt.gca().get_ylabel()
        Normal(0, 1).plot()
        self.assertEqual(plt.gca().get_title(), title)
        self.assertEqual(plt.gca().get_ylabel(), ylabel)

    def test_a_title_set_by_hand_survives(self):
        plt.figure()
        plt.gca().set_title("My own title")
        Normal(0, 1).plot()
        self.assertEqual(plt.gca().get_title(), "My own title")


class TestDistributionPlotLegend(unittest.TestCase):
    """``Distribution.plot()`` labels its curve and shows a legend once the
    axes holds more than one labeled series -- another theoretical curve,
    or a simulated plot (hist, density, ecdf, ...) it's overlaid on. A
    lone curve stays legend-free, matching every other plot type's rule
    (see ``_refresh_legend`` in ``plot.py``).
    """

    def tearDown(self):
        plt.close("all")

    # --- a lone curve gets no legend ---

    def test_lone_curve_has_no_legend(self):
        plt.figure()
        Normal(0, 1).plot()
        self.assertIsNone(plt.gca().get_legend())

    # --- two theoretical curves overlaid get a legend naming each one ---

    def test_two_theoretical_curves_get_a_legend(self):
        plt.figure()
        Normal(0, 1).plot()
        Normal(2, 1).plot()
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(labels, ["Normal(0, 1)", "Normal(2, 1)"])

    # --- default label names the distribution itself ---

    def test_default_label_is_the_distribution_repr(self):
        plt.figure()
        Binomial(10, 0.5).plot()
        Poisson(3).plot()
        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertEqual(labels, ["Binomial(10, 0.5)", "Poisson(3)"])

    # --- an explicit label= still wins over the default ---

    def test_explicit_label_overrides_default(self):
        plt.figure()
        Normal(0, 1).plot(label="theoretical")
        Normal(2, 1).plot()
        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertEqual(labels, ["theoretical", "Normal(2, 1)"])

    # --- a simulated density overlaid with its theoretical pdf ---

    def test_simulated_density_and_theoretical_pdf_get_a_legend(self):
        plt.figure()
        RV(Normal(0, 1)).sim(500).plot(type="density")
        Normal(0, 1).plot()
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(labels, ["Variable 1", "Normal(0, 1)"])

    # --- a simulated histogram overlaid with its theoretical pmf ---

    def test_simulated_histogram_and_theoretical_pmf_get_a_legend(self):
        plt.figure()
        RV(Binomial(10, 0.5)).sim(500).plot(type="hist")
        Binomial(10, 0.5).plot()
        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertEqual(labels, ["Variable 1", "Binomial(10, 0.5)"])

    # --- an empirical ECDF overlaid with its theoretical cdf ---

    def test_empirical_ecdf_and_theoretical_cdf_get_a_legend(self):
        plt.figure()
        RV(Normal(0, 1)).sim(500).plot(type="ecdf")
        Normal(0, 1).plot(cdf=True)
        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertEqual(labels, ["Variable 1", "Normal(0, 1)"])

    # --- the discrete pmf's dashed connecting line is not a second entry ---

    def test_discrete_connecting_line_is_not_a_second_legend_entry(self):
        plt.figure()
        Binomial(10, 0.5).plot()
        Poisson(3).plot()
        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertEqual(len(labels), 2)


class TestDistributionShade(unittest.TestCase):
    """``DistributionPlot.shade(lt, le, gt, ge)`` -- chained off ``.plot()``.

    ``Distribution.plot()`` returns a ``DistributionPlot``, and ``.shade()``
    fills a tail or interval under the curve it drew (pmf/pdf or cdf).
    There is no standalone ``shade`` on a distribution, so a curve always
    exists first. Bounds read as probability inequalities; the strict
    (``lt``/``gt``) vs. inclusive (``le``/``ge``) choice matters for
    discrete distributions. The shaded region honors the displayed x-window
    (the distribution's own window, zoomed or not, one set by hand, or an
    overlay union), not the distribution's static ``self.xlim``.
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
        # Binomial(100, 0.5) zooms itself, so the axis is far tighter than its
        # (0, 100) support; an open left tail must start at the visible edge,
        # which the fork's support-based version could not do.
        plt.figure()
        p = Binomial(100, 0.5).plot()
        axlo, _ = p.ax.get_xlim()
        self.assertGreater(axlo, 0)  # window is tighter than full support
        p.shade(le=50)
        xs = self._impulse_xs(p.ax)
        self.assertGreaterEqual(min(xs), int(np.floor(axlo)))
        self.assertNotIn(0, xs)  # the full support's lower edge

    def test_open_tail_follows_an_overlay_widened_axis(self):
        # The case where the two genuinely differ now that the window zooms
        # itself: overlaying unions the axes, so the display runs wider than
        # the curve's own zoomed window. The shading must follow the display.
        plt.figure()
        d = Binomial(100, 0.5)
        Binomial(100, 0.1).plot()  # its window sits well to the left of d's
        p = d.plot()
        axlo, _ = p.ax.get_xlim()
        self.assertLess(axlo, d.xlim[0])  # display is wider than self.xlim
        p.shade(le=50)
        self.assertLess(min(self._impulse_xs(p.ax)), d.xlim[0])

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


class TestFastSim(unittest.TestCase):
    """The batched ``.sim(n)`` fast path.

    ``NegativeHypergeometric`` and ``TruncatedNormal`` are backed by scipy
    samplers with a large per-call setup cost that does not shrink when
    fewer samples are asked for -- nhypergeom rebuilds a CDF table and an
    inverse-CDF interpolator every call. Drawing n samples in one batched
    call instead of n single ones is worth ~850x and ~80x respectively.
    These tests pin the scope of that optimization: exactly two
    distributions, only ``.sim()``, and only on an untransformed RV.
    """

    def test_only_two_distributions_override_the_hook(self):
        # The whole design rests on _fast_sim being None everywhere else, so
        # every other distribution keeps its existing one-draw-at-a-time
        # loop. A new override should be a deliberate, reviewed addition.
        overriders = sorted(
            cls.__name__
            for cls in distributions.Distribution.__subclasses__()
            if cls._fast_sim is not distributions.Distribution._fast_sim
        )
        self.assertEqual(overriders, ["NegativeHypergeometric", "TruncatedNormal"])

    def test_base_hook_returns_none(self):
        for dist in [Normal(0, 1), Binomial(10, 0.5), Poisson(3), Uniform(a=0, b=1)]:
            with self.subTest(dist=type(dist).__name__):
                self.assertIsNone(dist._fast_sim(5))

    def test_overrides_return_n_samples(self):
        for dist in [
            NegativeHypergeometric(r=3, N0=10, N1=8),
            TruncatedNormal(mean=0, sd=1, a=-1, b=2),
        ]:
            with self.subTest(dist=type(dist).__name__):
                batch = dist._fast_sim(17)
                self.assertIsNotNone(batch)
                self.assertEqual(len(batch), 17)

    def test_batched_negative_hypergeometric_matches_the_theoretical_pmf(self):
        # The point of the change is speed, not a different distribution, so
        # check the batched output against the exact pmf with a chi-square.
        seed(42)
        r, N0, N1 = 3, 10, 8
        X = NegativeHypergeometric(r=r, N0=N0, N1=N1)
        values = np.array(list(RV(X).sim(Nsim)), dtype=int)
        support = list(range(0, N1 + 1))
        observed = [int((values == k).sum()) for k in support]
        expected = [Nsim * float(X.pmf(k)) for k in support]
        # Lump any sparse bins together so every expected count is big enough
        # for a chi-square -- but only add the lumped bin if it is non-empty,
        # since an all-zero bin makes the statistic nan.
        keep = [i for i, e in enumerate(expected) if e >= 5]
        drop = [i for i in range(len(expected)) if i not in keep]
        obs = [observed[i] for i in keep]
        exp = [expected[i] for i in keep]
        if drop:
            obs.append(sum(observed[i] for i in drop))
            exp.append(sum(expected[i] for i in drop))
        # Rescale so the two sum identically -- dropping nothing leaves them
        # equal already, but lumping can leave a rounding gap.
        exp = [e * sum(obs) / sum(exp) for e in exp]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01, f"chi-square p={pval}")

    def test_batched_truncated_normal_matches_the_theoretical_cdf(self):
        seed(42)
        X = TruncatedNormal(mean=0, sd=1, a=-1, b=2)
        values = list(RV(X).sim(Nsim))
        pval = stats.kstest(values, lambda q: [float(X.cdf(v)) for v in q]).pvalue
        self.assertTrue(pval > 0.01, f"KS p={pval}")
        # And it really is inside the truncation bounds.
        self.assertTrue(all(-1 <= v <= 2 for v in values))

    def test_batched_and_looped_paths_agree_distributionally(self):
        # Same distribution either way -- compared with a two-sample KS test,
        # since the two paths draw different underlying bits (see the notes:
        # scipy's batched rvs does not consume the stream in the same order).
        seed(42)
        X = TruncatedNormal(mean=0, sd=1, a=-1, b=2)
        batched = list(RV(X).sim(4000))
        looped = [X.draw() for _ in range(4000)]
        pval = stats.ks_2samp(batched, looped).pvalue
        self.assertTrue(pval > 0.01, f"two-sample KS p={pval}")

    def test_apply_identity_still_takes_the_slow_path(self):
        # A hand-written `lambda x: x` is a different object than the
        # _IDENTITY sentinel, so `is` correctly fails and the ordinary
        # per-draw loop runs. If this ever passed, .apply() transformations
        # would be silently skipped.
        X = RV(NegativeHypergeometric(r=3, N0=10, N1=8))
        self.assertIs(X.func, _IDENTITY)
        self.assertIsNot(X.apply(lambda x: x).func, _IDENTITY)
        # A real transformation must still be applied to every value.
        seed(42)
        doubled = list(X.apply(lambda x: 2 * x).sim(200))
        self.assertTrue(all(v % 2 == 0 for v in doubled))

    def test_fast_path_does_not_change_pdf_cdf_or_mean(self):
        # Only .sim() is affected; every other method must be untouched.
        nh = NegativeHypergeometric(r=3, N0=10, N1=8)
        ref = stats.nhypergeom(M=18, n=8, r=3)
        self.assertAlmostEqual(float(nh.pmf(2)), float(ref.pmf(2)), places=12)
        self.assertAlmostEqual(float(nh.cdf(4)), float(ref.cdf(4)), places=12)
        self.assertAlmostEqual(float(nh.mean()), float(ref.mean()), places=12)
        tn = TruncatedNormal(mean=0, sd=1, a=-1, b=2)
        ref2 = stats.truncnorm(a=-1.0, b=2.0, loc=0, scale=1)
        self.assertAlmostEqual(float(tn.pdf(0.5)), float(ref2.pdf(0.5)), places=12)
        self.assertAlmostEqual(float(tn.cdf(0.5)), float(ref2.cdf(0.5)), places=12)
        self.assertAlmostEqual(float(tn.mean()), float(ref2.mean()), places=12)

    def test_sim_returns_the_usual_result_types(self):
        # The fast path builds its own Results/RVResults, so check it did not
        # quietly change what .sim() hands back.
        X = NegativeHypergeometric(r=3, N0=10, N1=8)
        space_sims = X.sim(50)
        self.assertIsInstance(space_sims, Results)
        self.assertEqual(len(space_sims), 50)
        rv_sims = RV(X).sim(50)
        self.assertIsInstance(rv_sims, RVResults)
        self.assertEqual(len(rv_sims), 50)
        self.assertTrue(all(isinstance(v, Scalar) for v in rv_sims))

    def test_sim_still_validates_n(self):
        X = NegativeHypergeometric(r=3, N0=10, N1=8)
        for bad in [0, -1, 2.5]:
            with self.subTest(n=bad):
                self.assertRaises(ValueError, lambda b=bad: X.sim(b))
                self.assertRaises(ValueError, lambda b=bad: RV(X).sim(b))

    def test_timing_regression_generous_margin(self):
        # Not a benchmark -- a tripwire. Before batching this took upwards of
        # ten seconds; batched it is ~0.03s. A one-second ceiling catches a
        # future change that reinstates the per-draw loop without being
        # sensitive to how busy the machine is.
        X = NegativeHypergeometric(r=3, N0=10, N1=8)
        start = time.perf_counter()
        RV(X).sim(10000)
        self.assertLess(time.perf_counter() - start, 1.0)
