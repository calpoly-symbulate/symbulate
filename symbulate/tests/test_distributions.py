import math
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


class TestBetaBinomial(unittest.TestCase):

    def test_BetaBinomial_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        BetaBinomial(n=10, shape1=2, shape2=3).draw()
        RV(BetaBinomial(n=10, shape1=2, shape2=3)).sim(100).plot()
        BetaBinomial(n=10, shape1=2, shape2=3).plot()
        BetaBinomial(n=10, shape1=2, shape2=3).plot(cdf=True)
        plt.close("all")


class TestBetaNegativeBinomial(unittest.TestCase):

    def test_BetaNegativeBinomial_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        BetaNegativeBinomial(r=5, shape1=3, shape2=2).draw()
        RV(BetaNegativeBinomial(r=5, shape1=3, shape2=2)).sim(100).plot()
        BetaNegativeBinomial(r=5, shape1=3, shape2=2).plot()
        BetaNegativeBinomial(r=5, shape1=3, shape2=2).plot(cdf=True)
        plt.close("all")


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


class TestNegativeHypergeometric(unittest.TestCase):

    def test_NegativeHypergeometric_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        NegativeHypergeometric(r=3, N0=7, N1=5).draw()
        RV(NegativeHypergeometric(r=3, N0=7, N1=5)).sim(100).plot()
        NegativeHypergeometric(r=3, N0=7, N1=5).plot()
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
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        sims = RV(Zipf(shape=0.7, n=12)).sim(Nsim)
        self.assertTrue(all(1 <= sim <= 12 for sim in sims))

    def test_Zipf_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
        value = Zipf(shape=1.2, n=10).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 1.0)
        self.assertLessEqual(float(value), 10.0)

    # --- reproducibility ---

    def test_Zipf_same_seed_gives_same_sims(self):
        distributions.rng = np.random.default_rng(2024)
        first = list(RV(Zipf(shape=1.2, n=10)).sim(500))
        distributions.rng = np.random.default_rng(2024)
        second = list(RV(Zipf(shape=1.2, n=10)).sim(500))
        self.assertEqual(first, second)

    def test_Zipf_different_seed_gives_different_sims(self):
        distributions.rng = np.random.default_rng(1)
        first = list(RV(Zipf(shape=1.2, n=10)).sim(500))
        distributions.rng = np.random.default_rng(2)
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
        # draw / RV / sim / plot all wired through the base class.
        Zipf(shape=1.2, n=10).draw()
        RV(Zipf(shape=1.2, n=10)).sim(100).plot()
        Zipf(shape=1.2, n=10).plot()
        Zipf(shape=1.2, n=10).plot(cdf=True)
        Zipf(shape=1.2, n=500).plot(xlim="zoom")
        plt.close("all")


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


class TestIrwinHall(unittest.TestCase):

    def test_IrwinHall_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(IrwinHall(n=5))
        sims = X.sim(Nsim)
        cdf = stats.irwinhall(5).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_IrwinHall_general_bounds_distributional(self):
        # Sum of n uniforms on [a, b], not just on [0, 1].
        distributions.rng = np.random.default_rng(42)
        X = RV(IrwinHall(n=4, a=10, b=20))
        sims = X.sim(Nsim)
        cdf = stats.irwinhall(4, loc=4 * 10, scale=20 - 10).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_IrwinHall_matches_sum_of_uniforms(self):
        # The defining property: adding n independent uniforms gives this
        # distribution.
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
        sims = RV(IrwinHall(n=4, a=10, b=20)).sim(1000)
        self.assertTrue(all(40 <= float(v) <= 80 for v in sims))

    def test_IrwinHall_draw_is_scalar(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        IrwinHall(5).draw()
        RV(IrwinHall(5)).sim(100).plot()
        IrwinHall(5).plot()
        IrwinHall(5).plot(cdf=True)
        IrwinHall(30).plot(xlim="zoom")
        plt.close("all")


class TestBates(unittest.TestCase):

    def test_Bates_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        n = 5
        U = RV(Uniform(0, 1) ** n)
        sims = U.apply(lambda u: sum(u) / n).sim(Nsim)
        cdf = stats.irwinhall(n, loc=0, scale=1 / n).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Bates_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        Bates(n=5).draw()
        RV(Bates(n=5)).sim(100).plot()
        Bates(n=5).plot()
        Bates(n=5).plot(cdf=True)
        plt.close("all")


class TestLogUniform(unittest.TestCase):

    def test_LogUniform_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        X = RV(LogUniform(a=1, b=100))
        sims = log(X).sim(Nsim)
        cdf = stats.uniform(0, np.log(100)).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogUniform_exp_of_uniform(self):
        # exp(Uniform(log a, log b)) has a LogUniform(a, b) distribution.
        distributions.rng = np.random.default_rng(42)
        U = RV(Uniform(np.log(1), np.log(100)))
        sims = exp(U).sim(Nsim)
        cdf = stats.loguniform(1, 100).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogUniform_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        LogUniform(a=1, b=100).draw()
        RV(LogUniform(a=1, b=100)).sim(100).plot()
        LogUniform(a=1, b=100).plot()
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


class TestInverseGamma(unittest.TestCase):

    def test_InverseGamma_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        X = RV(Gamma(shape=3, rate=2))
        sims = (1 / X).sim(Nsim)
        cdf = stats.invgamma(3, scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_InverseGamma_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        InverseGamma(shape=3, scale=2).draw()
        RV(InverseGamma(shape=3, scale=2)).sim(100).plot()
        InverseGamma(shape=3, scale=2).plot()
        InverseGamma(shape=3, scale=2).plot(cdf=True)
        plt.close("all")


class TestScaledInverseChiSquare(unittest.TestCase):

    def test_ScaledInverseChiSquare_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        X = RV(ScaledInverseChiSquare(df=6, scale=2))
        sims = (6 * 2 / X).sim(Nsim)
        cdf = stats.chi2(df=6).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_ScaledInverseChiSquare_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all inherited from InverseGamma / base class.
        ScaledInverseChiSquare(df=6, scale=2).draw()
        RV(ScaledInverseChiSquare(df=6, scale=2)).sim(100).plot()
        ScaledInverseChiSquare(df=6, scale=2).plot()
        ScaledInverseChiSquare(df=6, scale=2).plot(cdf=True)
        plt.close("all")


class TestLogGamma(unittest.TestCase):

    def test_LogGamma_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(LogGamma(shape=2))
        sims = X.sim(Nsim)
        cdf = stats.loggamma(c=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogGamma_is_log_of_Gamma(self):
        # The defining property: log(Gamma) has this distribution. Note the
        # direction -- this is the opposite of LogNormal, where the variable's
        # own log is normal.
        distributions.rng = np.random.default_rng(42)
        G = RV(Gamma(shape=2, scale=1))
        sims = G.apply(log).sim(Nsim)
        pval = stats.kstest(sims, stats.loggamma(c=2).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_LogGamma_exp_is_Gamma(self):
        # The same fact read the other way round: exponentiating gives a Gamma.
        distributions.rng = np.random.default_rng(7)
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
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        LogGamma(shape=2).draw()
        RV(LogGamma(shape=2)).sim(100).plot()
        LogGamma(shape=2).plot()
        LogGamma(shape=2).plot(cdf=True)
        plt.close("all")


class TestBeta(unittest.TestCase):

    def test_Beta_error_shape1(self):
        self.assertRaises(Exception, lambda: Beta(shape1=-10, shape2=3))

    def test_Beta_error_shape2(self):
        self.assertRaises(Exception, lambda: Beta(shape1=3, shape2=-10))

    def test_Beta_to_Uniform(self):
        distributions.rng = np.random.default_rng(42)
        X = Beta(shape1=1, shape2=1)
        sims = X.sim(Nsim)
        cdf = stats.uniform(loc=0, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_symmetry(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Beta(shape1=4, shape2=5))
        sims = (1 - X).sim(Nsim)
        cdf = stats.beta(a=5, b=4).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_to_Exponential(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Beta(shape1=0.7, shape2=1))
        sims = (-log(X)).sim(Nsim)
        cdf = stats.expon(scale=1 / 0.7).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_to_F(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Beta(shape1=10 / 2, shape2=12 / 2))
        sims = (12 * X / (10 * (1 - X))).sim(Nsim)
        cdf = stats.f(dfn=10, dfd=12).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Beta_mean_pdf(self):
        X = Beta(shape1=2, shape2=5)
        self.assertAlmostEqual(float(X.mean()), 2 / 7)
        self.assertAlmostEqual(float(X.pdf(0.5)), stats.beta(a=2, b=5).pdf(0.5))


class TestPERT(unittest.TestCase):

    def test_PERT_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
        sims = RV(PERT(low=1, mode=2, high=10)).sim(1000)
        values = [float(v) for v in sims]
        self.assertTrue(all(1 <= v <= 10 for v in values))

    def test_PERT_draw_is_scalar(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        PERT(1, 2, 10).draw()
        RV(PERT(1, 2, 10)).sim(100).plot()
        PERT(1, 2, 10).plot()
        PERT(1, 2, 10).plot(cdf=True)
        plt.close("all")


class TestTriangular(unittest.TestCase):

    def test_Triangular_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
        sims = RV(Triangular(low=1, mode=2, high=10)).sim(1000)
        self.assertTrue(all(1 <= float(v) <= 10 for v in sims))

    def test_Triangular_draw_is_scalar(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        Triangular(1, 2, 10).draw()
        RV(Triangular(1, 2, 10)).sim(100).plot()
        Triangular(1, 2, 10).plot()
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
        distributions.rng = np.random.default_rng(42)
        a, b = 2.5, 3.0
        X = RV(Kumaraswamy(shape1=a, shape2=b))
        sims = (X**a).sim(Nsim)
        pval = stats.kstest(sims, stats.beta(a=1, b=b).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Kumaraswamy_from_Uniform_inverse_cdf(self):
        # If U is Uniform(0, 1), then (1 - (1 - U) ** (1 / b)) ** (1 / a) is
        # Kumaraswamy(a, b) -- the inverse-cdf construction.
        distributions.rng = np.random.default_rng(42)
        a, b = 2.0, 4.0
        U = RV(Uniform(0, 1))
        sims = ((1 - (1 - U) ** (1 / b)) ** (1 / a)).sim(Nsim)
        pval = stats.kstest(sims, Kumaraswamy(shape1=a, shape2=b).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Kumaraswamy_a_one_matches_Beta_sims(self):
        distributions.rng = np.random.default_rng(42)
        sims = Kumaraswamy(shape1=1, shape2=3).sim(Nsim)
        pval = stats.kstest(sims, stats.beta(a=1, b=3).cdf).pvalue
        self.assertTrue(pval > 0.01)

    # --- sampling ---

    def test_Kumaraswamy_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = Kumaraswamy(shape1=2.5, shape2=3.0)
        pval = stats.kstest(X.sim(Nsim), X.cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Kumaraswamy_sim_stays_in_support(self):
        distributions.rng = np.random.default_rng(42)
        sims = RV(Kumaraswamy(shape1=0.5, shape2=0.5)).sim(Nsim)
        self.assertTrue(all(0 <= sim <= 1 for sim in sims))

    def test_Kumaraswamy_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
        value = Kumaraswamy(shape1=2, shape2=3).draw()
        self.assertIsInstance(value, Scalar)
        self.assertGreaterEqual(float(value), 0.0)
        self.assertLessEqual(float(value), 1.0)

    def test_Kumaraswamy_sample_mean_near_theoretical(self):
        distributions.rng = np.random.default_rng(42)
        X = Kumaraswamy(shape1=2, shape2=3)
        sample_mean = RV(X).sim(Nsim).mean()
        self.assertAlmostEqual(float(sample_mean), float(X.mean()), places=2)

    # --- reproducibility ---

    def test_Kumaraswamy_same_seed_gives_same_sims(self):
        distributions.rng = np.random.default_rng(2024)
        first = list(RV(Kumaraswamy(shape1=2, shape2=3)).sim(500))
        distributions.rng = np.random.default_rng(2024)
        second = list(RV(Kumaraswamy(shape1=2, shape2=3)).sim(500))
        self.assertEqual(first, second)

    def test_Kumaraswamy_different_seed_gives_different_sims(self):
        distributions.rng = np.random.default_rng(1)
        first = list(RV(Kumaraswamy(shape1=2, shape2=3)).sim(500))
        distributions.rng = np.random.default_rng(2)
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
        # draw / RV / sim / plot all wired through the base class.
        Kumaraswamy(shape1=2, shape2=3).draw()
        RV(Kumaraswamy(shape1=2, shape2=3)).sim(100).plot()
        Kumaraswamy(shape1=2, shape2=3).plot()
        Kumaraswamy(shape1=2, shape2=3).plot(cdf=True)
        Kumaraswamy(shape1=2, shape2=5).plot(xlim="zoom")
        Kumaraswamy(shape1=0.5, shape2=0.5).plot()  # infinite density at both edges
        plt.close("all")


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
        distributions.rng = np.random.default_rng(42)
        X = RV(Pareto(shape=2, scale=3))
        sims = X.sim(Nsim)
        self.assertTrue(all(sim >= 3 for sim in sims))

    def test_Pareto_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Pareto(shape=2, scale=1))
        sims = X.sim(Nsim)
        cdf = stats.pareto(b=2, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)


class TestBurr(unittest.TestCase):

    def test_Burr_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        X = RV(Burr(shape1=1, shape2=2, scale=1))
        sims = (X + 1).sim(Nsim)
        cdf = stats.pareto(b=2, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Burr_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class; a < 1 also
        # exercises the monotone-decreasing high-density x-window.
        Burr(shape1=3, shape2=2, scale=1).draw()
        RV(Burr(shape1=3, shape2=2, scale=1)).sim(100).plot()
        Burr(shape1=3, shape2=2, scale=1).plot()
        Burr(shape1=0.5, shape2=2, scale=1).plot()
        Burr(shape1=3, shape2=2, scale=1).plot(cdf=True)
        plt.close("all")


class TestLomax(unittest.TestCase):

    def test_Lomax_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        X = RV(Lomax(shape=3, scale=1))
        sims = (X + 1).sim(Nsim)
        cdf = stats.pareto(b=3, scale=1).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_Lomax_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        Lomax(shape=3, scale=1).draw()
        RV(Lomax(shape=3, scale=1)).sim(100).plot()
        Lomax(shape=3, scale=1).plot()
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


class TestHalfNormal(unittest.TestCase):

    def test_HalfNormal_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(HalfNormal(scale=2))
        sims = X.sim(Nsim)
        cdf = stats.halfnorm(scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_HalfNormal_is_abs_of_Normal(self):
        # |X| for X ~ Normal(0, scale) should match HalfNormal(scale).
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        HalfNormal(2).draw()
        RV(HalfNormal(2)).sim(100).plot()
        HalfNormal(2).plot()
        HalfNormal(2).plot(cdf=True)
        plt.close("all")


class TestHalfCauchy(unittest.TestCase):

    def test_HalfCauchy_distributional(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(HalfCauchy(scale=2))
        sims = X.sim(Nsim)
        cdf = stats.halfcauchy(scale=2).cdf
        pval = stats.kstest(sims, cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_HalfCauchy_is_abs_of_Cauchy(self):
        # |X| for X ~ Cauchy(0, scale) should match HalfCauchy(scale).
        distributions.rng = np.random.default_rng(42)
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
        # A monotone-decreasing density peaks at its lower support bound, so
        # the highest-density window starts exactly at 0 (not ppf(0.001)).
        X = HalfCauchy(scale=1)
        low, high = X.xlim
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(float(X.cdf(high)) - float(X.cdf(low)), 0.998, places=4)

    def test_HalfCauchy_draw_is_scalar_in_support(self):
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        HalfCauchy(2).draw()
        RV(HalfCauchy(2)).sim(100).plot()
        HalfCauchy(2).plot()
        HalfCauchy(2).plot(cdf=True)
        HalfCauchy(2).plot(xlim=(0, 10))
        plt.close("all")

    def test_HalfCauchy_heavier_tailed_than_HalfNormal(self):
        # Both peak at 0, but the Cauchy tail keeps far more probability
        # out past the bulk -- the reason it is the more permissive prior.
        tail_cauchy = 1 - float(HalfCauchy(scale=1).cdf(10))
        tail_normal = 1 - float(HalfNormal(scale=1).cdf(10))
        self.assertGreater(tail_cauchy, tail_normal)


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


class TestGumbel(unittest.TestCase):

    def test_Gumbel_distributional(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
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
        # draw / RV / sim / plot all inherited from GEV / the base class.
        Gumbel(0, 1).draw()
        RV(Gumbel(0, 1)).sim(100).plot()
        Gumbel(0, 1).plot()
        Gumbel(0, 1).plot(cdf=True)
        plt.close("all")


class TestGPD(unittest.TestCase):

    def test_GPD_distributional(self):
        # shape (xi) maps straight onto scipy's c (same sign, unlike GEV).
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(0)
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
        # draw / RV / sim / plot all wired through the base class.
        GPD(0, 1, 0.2).draw()
        RV(GPD(0, 1, 0.2)).sim(100).plot()
        GPD(0, 1, 0.2).plot()
        GPD(0, 1, 0.2).plot(cdf=True)
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


class TestMultivariateT(unittest.TestCase):

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
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
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

    def test_MultivariateT_plot_raises(self):
        X = MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=5)
        self.assertRaises(Exception, X.plot)


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
        distributions.rng = np.random.default_rng(42)
        X = Wishart(df=5, scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)
        self.assertEqual(len(draw[0]), 3)

    def test_Wishart_draw_symmetric_pd(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        df, scale = 8, [[2, 0.5], [0.5, 1]]
        X = Wishart(df=df, scale=scale)
        sims = [X.draw()[0][0] for _ in range(Nsim)]
        self.assertAlmostEqual(np.mean(sims), df * scale[0][0], delta=0.5)

    def test_Wishart_diagonal_is_scaled_chisquare(self):
        # With an identity scale, each diagonal entry is chi-square(df).
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        X = InverseWishart(df=5, scale=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        draw = X.draw()
        self.assertIsInstance(draw, Vector)
        self.assertEqual(len(draw), 3)
        self.assertEqual(len(draw[0]), 3)

    def test_InverseWishart_draw_symmetric_pd(self):
        distributions.rng = np.random.default_rng(42)
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
        distributions.rng = np.random.default_rng(42)
        df, scale = 10, [[2, 0.5], [0.5, 1]]
        X = InverseWishart(df=df, scale=scale)
        sims = [X.draw()[0][0] for _ in range(Nsim)]
        expected = scale[0][0] / (df - 2 - 1)
        self.assertAlmostEqual(np.mean(sims), expected, delta=0.05)

    def test_InverseWishart_inverse_is_wishart(self):
        # If X ~ InverseWishart(df, scale), then inv(X) ~ Wishart(df, inv(scale)).
        # Check the (0, 0) entry of the inverse against chi-square with an
        # identity scale (whose inverse is also the identity).
        distributions.rng = np.random.default_rng(42)
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


class TestDirichlet(unittest.TestCase):

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
        distributions.rng = np.random.default_rng(42)
        X = Dirichlet(alpha=[2, 3, 5])
        draw = X.draw()
        self.assertEqual(len(draw), 3)
        self.assertTrue(np.all(np.array(draw) >= 0))
        self.assertAlmostEqual(float(sum(draw)), 1.0)

    def test_Dirichlet_sim_shape_and_sums_to_one(self):
        distributions.rng = np.random.default_rng(42)
        X = Dirichlet(alpha=[2, 3, 5])
        sims = X.sim(100)
        arr = np.array(list(sims))
        self.assertEqual(arr.shape, (100, 3))
        np.testing.assert_allclose(arr.sum(axis=1), np.ones(100))

    def test_Dirichlet_marginals_match_Beta(self):
        # Each proportion X_i is marginally Beta(alpha_i, alpha0 - alpha_i).
        distributions.rng = np.random.default_rng(42)
        alpha = [2, 3, 5]
        alpha0 = sum(alpha)
        components = RV(Dirichlet(alpha=alpha))
        for i, a_i in enumerate(alpha):
            sims = components[i].sim(Nsim)
            cdf = stats.beta(a_i, alpha0 - a_i).cdf
            pval = stats.kstest(sims, cdf).pvalue
            self.assertTrue(pval > 0.01)

    def test_Dirichlet_plots_without_error(self):
        # draw / RV / sim / plot all wired through, marginals overlaid.
        Dirichlet(alpha=[2, 3, 5]).draw()
        Dirichlet(alpha=[2, 3, 5]).plot()
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
            GPD(loc=0, scale=1, shape=0.3),
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
        self.assertEqual(GPD(loc=0, scale=1, shape=0.3).xlim[0], 0)  # loc
        self.assertEqual(GPD(loc=2, scale=1, shape=0.3).xlim[0], 2)  # loc
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
