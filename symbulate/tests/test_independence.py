"""Tests for symbulate.independence — AssumeIndependent.

Covers:
  - TypeError when non-RV arguments are passed (int, float, Distribution)
  - Error message names the bad type
  - ValueError when two RVs share the same probability space
  - ValueError message mentions "different probability spaces"
  - Same-space pair detected even when not the first two args (inner loop coverage)
  - Return value is a tuple of RV instances
  - Correct length for 1, 2, and 3 input RVs
  - All output RVs share the new joint ProbabilitySpace
  - Output ProbabilitySpace differs from each input's original space
  - Marginal distributions are preserved after AssumeIndependent
  - RVs from separate spaces remain independent on the resulting joint space
  - Custom func= on an RV is preserved through AssumeIndependent
  - Loop variable binding is correct (i=i in lambda captures each index independently)

Design note: AssumeIndependent combines RVs that are already on *different* probability
spaces into a single joint space.  It does NOT remove within-space correlation — passing
two RVs from the same space (e.g., both unpacked from the same joint distribution)
correctly raises ValueError.  The independence tests below use RVs from separate spaces,
which is the intended use case.
"""
import unittest
import numpy as np
import scipy.stats as stats

from symbulate import (
    RV, Normal, Exponential, Binomial, BoxModel,
    AssumeIndependent,
)
from symbulate import distributions

Nsim = 10000


# ===========================================================================
# Error handling
# ===========================================================================

class TestAssumeIndependentErrors(unittest.TestCase):
    """TypeError and ValueError raised for invalid inputs."""

    # --- TypeError: non-RV arguments ---
    # The isinstance check runs for each arg in order.  Placing the bad arg
    # first ensures the TypeError branch fires before the inner prob_space
    # loop attempts to access attributes on an unvalidated object.

    def test_non_rv_int_raises_type_error(self):
        X = RV(Normal(0, 1))
        self.assertRaises(TypeError, lambda: AssumeIndependent(5, X))

    def test_non_rv_float_raises_type_error(self):
        X = RV(Normal(0, 1))
        self.assertRaises(TypeError, lambda: AssumeIndependent(3.14, X))

    def test_non_rv_distribution_raises_type_error(self):
        """A Distribution not wrapped in RV must raise TypeError."""
        self.assertRaises(TypeError, lambda: AssumeIndependent(Normal(0, 1)))

    def test_type_error_message_names_int_type(self):
        try:
            AssumeIndependent(42)
            self.fail("Expected TypeError")
        except TypeError as e:
            self.assertIn("int", str(e))

    def test_type_error_message_names_distribution_type(self):
        try:
            AssumeIndependent(Normal(0, 1))
            self.fail("Expected TypeError")
        except TypeError as e:
            self.assertIn("Normal", str(e))

    # --- ValueError: same probability space ---

    def test_same_prob_space_raises_value_error(self):
        """Two RVs from the same joint space (Normal**2 unpacking) are rejected."""
        X, Y = RV(Normal(0, 1) ** 2)
        self.assertRaises(ValueError, lambda: AssumeIndependent(X, Y))

    def test_same_prob_space_non_adjacent_pair_raises_value_error(self):
        """Same-space check covers all pairs, not just adjacent args (inner loop)."""
        X, Y = RV(Normal(0, 1) ** 2)
        Z = RV(Exponential(1))
        # Z is first (valid, different space), but X and Y still share a space
        self.assertRaises(ValueError, lambda: AssumeIndependent(Z, X, Y))

    def test_value_error_message_mentions_different_probability_spaces(self):
        X, Y = RV(Normal(0, 1) ** 2)
        try:
            AssumeIndependent(X, Y)
            self.fail("Expected ValueError")
        except ValueError as e:
            self.assertIn("different probability spaces", str(e))


# ===========================================================================
# Return type and structure
# ===========================================================================

class TestAssumeIndependentReturnType(unittest.TestCase):
    """Return type, length, and structural relationships of the output tuple."""

    def test_returns_tuple(self):
        X = RV(Normal(0, 1))
        Y = RV(Exponential(1))
        self.assertIsInstance(AssumeIndependent(X, Y), tuple)

    def test_single_rv_returns_tuple_of_length_one(self):
        X = RV(Normal(0, 1))
        result = AssumeIndependent(X)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)

    def test_two_rvs_returns_length_two(self):
        X = RV(Normal(0, 1))
        Y = RV(Exponential(1))
        self.assertEqual(len(AssumeIndependent(X, Y)), 2)

    def test_three_rvs_returns_length_three(self):
        X = RV(Normal(0, 1))
        Y = RV(Exponential(1))
        Z = RV(Binomial(n=10, p=0.4))
        self.assertEqual(len(AssumeIndependent(X, Y, Z)), 3)

    def test_output_elements_are_rv_instances(self):
        X = RV(Normal(0, 1))
        Y = RV(Exponential(1))
        X2, Y2 = AssumeIndependent(X, Y)
        self.assertIsInstance(X2, RV)
        self.assertIsInstance(Y2, RV)

    def test_all_output_rvs_share_the_new_joint_prob_space(self):
        """All outputs are on the same new ProbabilitySpace created by AssumeIndependent."""
        X = RV(Normal(0, 1))
        Y = RV(Exponential(1))
        Z = RV(Binomial(n=5, p=0.3))
        X2, Y2, Z2 = AssumeIndependent(X, Y, Z)
        self.assertIs(X2.prob_space, Y2.prob_space)
        self.assertIs(Y2.prob_space, Z2.prob_space)

    def test_output_prob_space_differs_from_each_input(self):
        """AssumeIndependent creates a new joint space, not reusing any input's space."""
        X = RV(Normal(0, 1))
        Y = RV(Exponential(1))
        X2, Y2 = AssumeIndependent(X, Y)
        self.assertIsNot(X2.prob_space, X.prob_space)
        self.assertIsNot(Y2.prob_space, Y.prob_space)


# ===========================================================================
# Marginals preserved
# ===========================================================================

class TestAssumeIndependentMarginals(unittest.TestCase):
    """Each output RV has the same marginal distribution as the corresponding input."""

    def test_first_marginal_normal(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(3, 2))
        Y = RV(Exponential(1))
        X2, _ = AssumeIndependent(X, Y)
        sims = X2.sim(Nsim)
        pval = stats.kstest(sims, stats.norm(loc=3, scale=2).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_second_marginal_exponential(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        Y = RV(Exponential(rate=2))
        _, Y2 = AssumeIndependent(X, Y)
        sims = Y2.sim(Nsim)
        pval = stats.kstest(sims, stats.expon(scale=0.5).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_discrete_marginal_mean_and_variance_preserved(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        Y = RV(Binomial(n=10, p=0.4))
        _, Y2 = AssumeIndependent(X, Y)
        sims = Y2.sim(Nsim)
        self.assertAlmostEqual(float(sims.mean()), 4.0, delta=0.1)
        self.assertAlmostEqual(float(sims.var()), 10 * 0.4 * 0.6, delta=0.1)

    def test_single_rv_marginal_preserved(self):
        """AssumeIndependent with one RV: the output still has the original distribution."""
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(5, 1))
        (X2,) = AssumeIndependent(X)
        sims = X2.sim(Nsim)
        pval = stats.kstest(sims, stats.norm(loc=5, scale=1).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_third_marginal_in_three_rv_call(self):
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        Y = RV(Normal(5, 3))
        Z = RV(Exponential(rate=0.5))
        _, _, Z2 = AssumeIndependent(X, Y, Z)
        sims = Z2.sim(Nsim)
        pval = stats.kstest(sims, stats.expon(scale=2).cdf).pvalue
        self.assertTrue(pval > 0.01)


# ===========================================================================
# Independence on the joint space
# ===========================================================================

class TestAssumeIndependentIndependence(unittest.TestCase):
    """RVs from separate spaces remain independent after being joined.

    Independence is verified via the sum-of-two-N(0,1) test:
    if X2 and Y2 are independent N(0,1), then X2 + Y2 ~ N(0, sqrt(2)).
    """

    def test_two_normal_rvs_from_separate_spaces_are_independent(self):
        """Two separately-created N(0,1) RVs remain independent on the joint space."""
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        Y = RV(Normal(0, 1))
        X2, Y2 = AssumeIndependent(X, Y)
        sims = (X2 + Y2).sim(Nsim)
        pval = stats.kstest(sims, stats.norm(0, np.sqrt(2)).cdf).pvalue
        self.assertTrue(pval > 0.01)

    def test_mixed_distribution_rvs_are_independent(self):
        """Normal and Exponential RVs from separate spaces share no dependence."""
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        Y = RV(Exponential(rate=1))
        X2, Y2 = AssumeIndependent(X, Y)
        # Marginals must be individually correct (necessary condition for independence)
        x_sims = X2.sim(Nsim)
        y_sims = Y2.sim(Nsim)
        pval_x = stats.kstest(x_sims, stats.norm(0, 1).cdf).pvalue
        pval_y = stats.kstest(y_sims, stats.expon(scale=1).cdf).pvalue
        self.assertTrue(pval_x > 0.01)
        self.assertTrue(pval_y > 0.01)

    def test_three_rvs_all_on_independent_joint_space(self):
        """Three separately-created RVs are all independent on the joint space."""
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        Y = RV(Normal(0, 1))
        Z = RV(Normal(0, 1))
        X2, Y2, Z2 = AssumeIndependent(X, Y, Z)
        # X2 + Y2 + Z2 ~ N(0, sqrt(3)) if all three are independent N(0,1)
        sims = (X2 + Y2 + Z2).sim(Nsim)
        pval = stats.kstest(sims, stats.norm(0, np.sqrt(3)).cdf).pvalue
        self.assertTrue(pval > 0.01)


# ===========================================================================
# Custom functions and closure binding
# ===========================================================================

class TestAssumeIndependentCustomFunctions(unittest.TestCase):
    """Custom func= on RV and loop variable binding are preserved correctly."""

    def test_custom_func_on_first_rv_preserved(self):
        """func=sum on the first RV keeps its effect: output values must be in {0..5}."""
        distributions.rng = np.random.default_rng(42)
        P = BoxModel([0, 1], size=5)
        X = RV(P, sum)           # sum of 5 coin flips: support {0,1,2,3,4,5}
        Y = RV(Normal(0, 1))
        X2, _ = AssumeIndependent(X, Y)
        sims = X2.sim(1000)
        self.assertTrue(all(v in {0, 1, 2, 3, 4, 5} for v in sims),
                        "sum() was not applied — func= on first RV was lost")

    def test_custom_func_on_second_rv_preserved(self):
        """func=sum on the second RV keeps its effect: output values must be in {0..5}."""
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        P = BoxModel([0, 1], size=5)
        Y = RV(P, sum)           # sum of 5 coin flips: support {0,1,2,3,4,5}
        _, Y2 = AssumeIndependent(X, Y)
        sims = Y2.sim(1000)
        self.assertTrue(all(v in {0, 1, 2, 3, 4, 5} for v in sims),
                        "sum() was not applied — func= on second RV was lost")

    def test_non_identity_func_distributional(self):
        """X = sum of 5 Bernoulli(0.4) trials after AssumeIndependent ~ Binomial(5, 0.4)."""
        distributions.rng = np.random.default_rng(42)
        P = BoxModel([0, 1], size=5, probs=[0.6, 0.4])
        X = RV(P, sum)
        Y = RV(Normal(0, 1))
        X2, _ = AssumeIndependent(X, Y)
        sims = X2.sim(Nsim)
        exp_list, obs_list = [], []
        counts = {k: 0 for k in range(6)}
        for v in sims:
            counts[int(v)] += 1
        for k in range(6):
            expected = Nsim * stats.binom(n=5, p=0.4).pmf(k)
            if expected > 5:
                exp_list.append(expected)
                obs_list.append(counts[k])
        pval = stats.chisquare(obs_list,
                               np.array(exp_list) * sum(obs_list) / sum(exp_list)).pvalue
        self.assertTrue(pval > 0.01)

    def test_closure_binding_correct_for_three_rvs(self):
        """Each output RV uses its own index i, not the last i in the loop.

        A classic Python closure bug: if the loop variable i is captured by
        reference rather than by value, all three outputs would extract index 2
        (the last value), making all simulated means approximately 20.0.
        This test fails if the ``i=i`` default-argument binding in
        independence.py is broken.
        """
        distributions.rng = np.random.default_rng(42)
        X = RV(Normal(0, 1))
        Y = RV(Normal(10, 1))
        Z = RV(Normal(20, 1))
        X2, Y2, Z2 = AssumeIndependent(X, Y, Z)
        self.assertAlmostEqual(float(X2.sim(Nsim).mean()),  0.0, delta=0.1)
        self.assertAlmostEqual(float(Y2.sim(Nsim).mean()), 10.0, delta=0.1)
        self.assertAlmostEqual(float(Z2.sim(Nsim).mean()), 20.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
