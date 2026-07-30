"""Tests for symbulate.markov_chains.

Covers both the discrete-time and continuous-time Markov chains: input
validation, lazy on-demand state generation (forward generation and
backward cached lookups), state labels, the embedded transition matrix
of a continuous-time chain, sample-path evaluation, interarrival times,
and the marginal/transition behavior of the chains.

Reproducibility: the discrete chain draws through ``markov_chains.rng``
and the continuous chain additionally draws interarrival times through
``distributions.rng``, so the ``seed`` helper reseeds both.
"""

import unittest

import numpy as np
import scipy.stats as stats

from symbulate import *
from symbulate import markov_chains as mc
from symbulate import distributions
from symbulate.markov_chains import (
    MarkovChainResult,
    MarkovChainProbabilitySpace,
    MarkovChain,
    ContinuousTimeMarkovChainResult,
    ContinuousTimeMarkovChainProbabilitySpace,
    ContinuousTimeMarkovChain,
)
from symbulate.probability_space import ProbabilitySpace
from symbulate.result import (
    InfiniteVector,
    ContinuousTimeFunction,
    DiscreteValued,
)

Nsim = 10000

# These tests reseed the module-level generators that the chains draw
# through. Save and restore them around the module so this file is
# hermetic and does not leak rng state to other test files.
_saved_rng = {}


def setUpModule():
    _saved_rng["mc"] = mc.rng
    _saved_rng["dist"] = distributions.rng


def tearDownModule():
    mc.rng = _saved_rng["mc"]
    distributions.rng = _saved_rng["dist"]


def seed(value=42):
    """Reseed both generators the Markov chains draw through."""
    mc.rng = np.random.default_rng(value)
    distributions.rng = np.random.default_rng(value)


# Deterministic 2-state matrices for assertions that don't need a seed.
STAY = [[1.0, 0.0], [0.0, 1.0]]  # identity: never leaves its start state
SWAP = [[0.0, 1.0], [1.0, 0.0]]  # always flips state each step


class TestMarkovChainResultValidation(unittest.TestCase):

    def test_rows_must_sum_to_one(self):
        self.assertRaises(
            Exception, lambda: MarkovChainResult([[0.5, 0.4], [0.3, 0.7]], [0.5, 0.5])
        )

    def test_probabilities_cannot_be_negative(self):
        # Row still sums to 1, but contains a negative entry.
        self.assertRaises(
            Exception, lambda: MarkovChainResult([[-0.5, 1.5], [0.5, 0.5]], [0.5, 0.5])
        )

    def test_matrix_must_be_square(self):
        self.assertRaises(
            Exception,
            lambda: MarkovChainResult([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]], [0.5, 0.5]),
        )

    def test_initial_dist_length_must_match(self):
        self.assertRaises(Exception, lambda: MarkovChainResult(STAY, [1.0]))

    def test_state_labels_length_must_match(self):
        self.assertRaises(
            Exception, lambda: MarkovChainResult(STAY, [1.0, 0.0], state_labels=["A"])
        )

    def test_initial_dist_cannot_be_negative(self):
        self.assertRaises(Exception, lambda: MarkovChainResult(STAY, [-0.5, 1.5]))

    def test_initial_dist_must_sum_to_one(self):
        self.assertRaises(Exception, lambda: MarkovChainResult(STAY, [0.5, 0.3]))


class TestMarkovChainResultBehavior(unittest.TestCase):

    def test_is_infinite_vector_and_discrete_valued(self):
        path = MarkovChainResult(STAY, [1.0, 0.0])
        self.assertIsInstance(path, InfiniteVector)
        self.assertIsInstance(path, DiscreteValued)

    def test_default_state_labels_are_range(self):
        path = MarkovChainResult(STAY, [1.0, 0.0])
        self.assertEqual(list(path.state_labels), [0, 1])

    def test_n_states_set(self):
        path = MarkovChainResult(STAY, [1.0, 0.0])
        self.assertEqual(path.n_states, 2)

    def test_initial_state_respects_initial_dist(self):
        # initial_dist [0, 1] forces the chain to start in state 1.
        path = MarkovChainResult(STAY, [0.0, 1.0])
        self.assertEqual(path[0], 1)

    def test_identity_chain_stays(self):
        path = MarkovChainResult(STAY, [1.0, 0.0])
        self.assertEqual([path[i] for i in range(5)], [0, 0, 0, 0, 0])

    def test_swap_chain_alternates(self):
        path = MarkovChainResult(SWAP, [1.0, 0.0])
        self.assertEqual([path[i] for i in range(5)], [0, 1, 0, 1, 0])

    def test_custom_state_labels_used(self):
        path = MarkovChainResult(SWAP, [1.0, 0.0], state_labels=["A", "B"])
        self.assertEqual([path[i] for i in range(3)], ["A", "B", "A"])

    def test_backward_access_uses_cache(self):
        # Generate forward to index 5, then read an earlier index (else-branch).
        path = MarkovChainResult(SWAP, [1.0, 0.0])
        _ = path[5]
        self.assertEqual(len(path.states), 6)
        self.assertEqual(path[2], 0)

    def test_get_states_returns_self(self):
        path = MarkovChainResult(STAY, [1.0, 0.0])
        self.assertIs(path.get_states(), path)


class TestMarkovChainProbabilitySpace(unittest.TestCase):

    def test_is_probability_space(self):
        space = MarkovChainProbabilitySpace(STAY, [1.0, 0.0])
        self.assertIsInstance(space, ProbabilitySpace)

    def test_draw_returns_result(self):
        space = MarkovChainProbabilitySpace(STAY, [1.0, 0.0])
        self.assertIsInstance(space.draw(), MarkovChainResult)


class TestMarkovChain(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(MarkovChain(STAY, [1.0, 0.0]), RV)

    def test_draw_returns_result(self):
        self.assertIsInstance(MarkovChain(STAY, [1.0, 0.0]).draw(), MarkovChainResult)

    def test_invalid_matrix_raises_at_draw(self):
        # MarkovChain validates lazily: the bad matrix is caught at draw time.
        chain = MarkovChain([[0.5, 0.4], [0.3, 0.7]], [0.5, 0.5])
        self.assertRaises(Exception, chain.draw)

    def test_indexing_returns_state_over_sims(self):
        seed()
        chain = MarkovChain(SWAP, [1.0, 0.0])
        # Deterministic: always 0 at even steps, 1 at odd steps.
        self.assertTrue(all(v == 1 for v in chain[1].sim(100)))
        self.assertTrue(all(v == 0 for v in chain[2].sim(100)))

    def test_initial_distribution_goodness_of_fit(self):
        seed()
        chain = MarkovChain([[0.5, 0.5], [0.5, 0.5]], [0.3, 0.7])
        tab = chain[0].sim(Nsim).tabulate()
        obs = [tab[0], tab[1]]
        exp = [Nsim * 0.3, Nsim * 0.7]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01)

    def test_transition_goodness_of_fit(self):
        seed()
        # Always start in state 0; step-1 state follows row 0 = [0.2, 0.8].
        chain = MarkovChain([[0.2, 0.8], [0.5, 0.5]], [1.0, 0.0])
        tab = chain[1].sim(Nsim).tabulate()
        obs = [tab[0], tab[1]]
        exp = [Nsim * 0.2, Nsim * 0.8]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01)

    def test_reproducible_under_same_seed(self):
        chain = MarkovChain([[0.3, 0.7], [0.6, 0.4]], [0.5, 0.5])
        seed(123)
        first = list(chain[2].sim(50))
        seed(123)
        second = list(chain[2].sim(50))
        self.assertEqual(first, second)


# Generator (Q) matrices for continuous-time tests.
Q2 = [[-1.0, 1.0], [2.0, -2.0]]


class TestContinuousTimeMarkovChainProbabilitySpaceValidation(unittest.TestCase):

    def test_rows_must_sum_to_zero(self):
        self.assertRaises(
            Exception,
            lambda: ContinuousTimeMarkovChainProbabilitySpace(
                [[-1.0, 0.5], [2.0, -2.0]], [1.0, 0.0]
            ),
        )

    def test_diagonal_cannot_be_positive(self):
        self.assertRaises(
            Exception,
            lambda: ContinuousTimeMarkovChainProbabilitySpace(
                [[1.0, -1.0], [2.0, -2.0]], [1.0, 0.0]
            ),
        )

    def test_offdiagonal_cannot_be_negative(self):
        # 3-state generator: row 0 = [0, -1, 1] has a non-positive diagonal
        # but a negative off-diagonal, isolating that branch.
        self.assertRaises(
            Exception,
            lambda: ContinuousTimeMarkovChainProbabilitySpace(
                [[0.0, -1.0, 1.0], [0.5, -1.0, 0.5], [0.5, 0.5, -1.0]], [1.0, 0.0, 0.0]
            ),
        )

    def test_matrix_must_be_square(self):
        self.assertRaises(
            Exception,
            lambda: ContinuousTimeMarkovChainProbabilitySpace(
                [[-1.0, 0.5, 0.5], [1.0, -2.0, 1.0]], [1.0, 0.0]
            ),
        )

    def test_initial_dist_length_must_match(self):
        self.assertRaises(
            Exception,
            lambda: ContinuousTimeMarkovChainProbabilitySpace(Q2, [1.0]),
        )

    def test_state_labels_length_must_match(self):
        self.assertRaises(
            Exception,
            lambda: ContinuousTimeMarkovChainProbabilitySpace(
                Q2, [1.0, 0.0], state_labels=["A"]
            ),
        )

    def test_absorbing_state_raises(self):
        self.assertRaises(
            Exception,
            lambda: ContinuousTimeMarkovChainProbabilitySpace(
                [[0.0, 0.0], [1.0, -1.0]], [0.5, 0.5]
            ),
        )


class TestContinuousTimeMarkovChainProbabilitySpace(unittest.TestCase):

    def test_default_state_labels_are_range(self):
        space = ContinuousTimeMarkovChainProbabilitySpace(Q2, [1.0, 0.0])
        self.assertEqual(list(space.state_labels), [0, 1])

    def test_embedded_transition_matrix(self):
        # Q = [[-1, 1], [2, -2]] -> embedded P = [[0, 1], [1, 0]].
        space = ContinuousTimeMarkovChainProbabilitySpace(Q2, [1.0, 0.0])
        np.testing.assert_allclose(space.transition_matrix, [[0.0, 1.0], [1.0, 0.0]])

    def test_embedded_transition_matrix_three_state(self):
        Q3 = [[-2.0, 1.0, 1.0], [1.0, -1.0, 0.0], [2.0, 2.0, -4.0]]
        space = ContinuousTimeMarkovChainProbabilitySpace(Q3, [1.0, 0.0, 0.0])
        expected = [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
        np.testing.assert_allclose(space.transition_matrix, expected)

    def test_draw_returns_result(self):
        seed()
        space = ContinuousTimeMarkovChainProbabilitySpace(Q2, [1.0, 0.0])
        self.assertIsInstance(space.draw(), ContinuousTimeMarkovChainResult)


class TestContinuousTimeMarkovChain(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(ContinuousTimeMarkovChain(Q2, [1.0, 0.0]), RV)

    def test_draw_returns_continuous_time_function(self):
        seed()
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0]).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)
        self.assertIsInstance(path, DiscreteValued)

    def test_path_starts_in_initial_state(self):
        seed()
        # initial_dist [1, 0] -> chain starts in state 0 -> label 'A'.
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0], state_labels=["A", "B"]).draw()
        self.assertEqual(path(0), "A")

    def test_path_values_are_valid_labels(self):
        seed()
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0], state_labels=["A", "B"]).draw()
        for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            self.assertIn(path(t), ["A", "B"])

    def test_interarrival_times_are_positive(self):
        seed()
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0]).draw()
        for n in range(5):
            self.assertGreater(path.interarrival_times[n], 0)

    def test_reproducible_under_same_seed(self):
        chain = ContinuousTimeMarkovChain(Q2, [1.0, 0.0], state_labels=["A", "B"])
        seed(7)
        first = [chain.draw()(t) for t in [0.5, 1.0, 2.0]]
        seed(7)
        second = [chain.draw()(t) for t in [0.5, 1.0, 2.0]]
        self.assertEqual(first, second)


class TestBirthDeathProcess(unittest.TestCase):

    def test_is_continuous_time_markov_chain(self):
        X = BirthDeathProcess(birth_rates=1, death_rates=1.5, num_states=5)
        self.assertIsInstance(X, ContinuousTimeMarkovChain)
        self.assertIsInstance(X, RV)

    def test_generator_matrix_constant_rates(self):
        # Constant birth 2, death 3, over 4 states. Births on the super-
        # diagonal (0 at the top), deaths on the sub-diagonal (0 at state 0),
        # diagonals make each row sum to 0.
        X = BirthDeathProcess(birth_rates=2, death_rates=3, num_states=4)
        Q = X.prob_space.generator_matrix
        expected = np.array(
            [
                [-2, 2, 0, 0],
                [3, -5, 2, 0],
                [0, 3, -5, 2],
                [0, 0, 3, -3],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(Q, expected)
        # Rows sum to 0.
        np.testing.assert_allclose(Q.sum(axis=1), np.zeros(4), atol=1e-12)

    def test_boundary_rates_zeroed(self):
        # Top state's birth and state 0's death are forced to 0.
        X = BirthDeathProcess(birth_rates=2, death_rates=3, num_states=4)
        self.assertEqual(X.birth_rates[-1], 0.0)
        self.assertEqual(X.death_rates[0], 0.0)

    def test_rate_forms_agree(self):
        # A constant, a list, and a callable specifying the same rates should
        # all build the same generator matrix.
        const = BirthDeathProcess(birth_rates=2, death_rates=3, num_states=4)
        lst = BirthDeathProcess(
            birth_rates=[2, 2, 2, 2], death_rates=[3, 3, 3, 3], num_states=4
        )
        fun = BirthDeathProcess(
            birth_rates=lambda n: 2, death_rates=lambda n: 3, num_states=4
        )
        np.testing.assert_allclose(
            const.prob_space.generator_matrix, lst.prob_space.generator_matrix
        )
        np.testing.assert_allclose(
            const.prob_space.generator_matrix, fun.prob_space.generator_matrix
        )

    def test_state_dependent_death_rate_mms(self):
        # M/M/s with s = 2: death rate min(n, 2) * mu.
        mu = 1.5
        X = BirthDeathProcess(
            birth_rates=1, death_rates=lambda n: min(n, 2) * mu, num_states=5
        )
        np.testing.assert_allclose(
            X.death_rates, [0.0, 1.5, 3.0, 3.0, 3.0]
        )  # state 0 zeroed

    def test_error_num_states_too_small(self):
        self.assertRaises(
            Exception,
            lambda: BirthDeathProcess(birth_rates=1, death_rates=1, num_states=1),
        )

    def test_error_negative_rate(self):
        self.assertRaises(
            Exception,
            lambda: BirthDeathProcess(birth_rates=-1, death_rates=1, num_states=5),
        )

    def test_error_rate_list_wrong_length(self):
        self.assertRaises(
            Exception,
            lambda: BirthDeathProcess(birth_rates=[1, 1], death_rates=1, num_states=5),
        )

    def test_error_initial_out_of_range(self):
        self.assertRaises(
            Exception,
            lambda: BirthDeathProcess(
                birth_rates=1, death_rates=1, num_states=5, initial=5
            ),
        )

    def test_error_pure_birth_has_dead_end(self):
        # All death rates 0 -> the top state has no way out -> rejected.
        self.assertRaises(
            Exception,
            lambda: BirthDeathProcess(birth_rates=1, death_rates=0, num_states=5),
        )

    def test_path_starts_at_initial(self):
        seed()
        X = BirthDeathProcess(birth_rates=1, death_rates=2, num_states=10, initial=3)
        self.assertEqual(X.draw()(0), 3)

    def test_path_values_are_valid_states(self):
        seed()
        X = BirthDeathProcess(birth_rates=1, death_rates=1.5, num_states=20)
        path = X.draw()
        for t in [0.0, 1.0, 5.0, 10.0]:
            v = path(t)
            self.assertIn(v, range(20))

    def test_mm1_stationary_is_geometric(self):
        # An M/M/1 queue (lambda=1, mu=2, rho=0.5) has stationary distribution
        # P(N=n) = (1-rho) rho^n. Simulate the queue length at a large time,
        # started empty, and check it against that geometric.
        seed()
        lam, mu, rho = 1.0, 2.0, 0.5
        X = BirthDeathProcess(birth_rates=lam, death_rates=mu, num_states=40)
        sims = [X.draw()(60.0) for _ in range(4000)]
        obs, exp = [], []
        for n in range(0, 12):
            e = len(sims) * (1 - rho) * rho**n
            if e > 5:
                exp.append(e)
                obs.append(sum(1 for s in sims if s == n))
        pval = stats.chisquare(obs, np.array(exp) * sum(obs) / sum(exp)).pvalue
        self.assertTrue(pval > 0.01)

    def test_reproducible_under_same_seed(self):
        X = BirthDeathProcess(birth_rates=1, death_rates=1.5, num_states=20)
        seed(7)
        first = [X.draw()(t) for t in [0.5, 1.0, 2.0]]
        seed(7)
        second = [X.draw()(t) for t in [0.5, 1.0, 2.0]]
        self.assertEqual(first, second)


class TestMMQueues(unittest.TestCase):
    """The M/M/... queue wrappers over BirthDeathProcess."""

    def test_all_are_birth_death_processes(self):
        for X in [
            MM1(1, 1.5),
            MMs(3, 2, 2),
            MMsK(4, 1, 2, 5),
            MMss(3, 1, 3),
            MMsKN(0.1, 1, 2, 6, 6),
            MMInfinity(5, 1),
        ]:
            self.assertIsInstance(X, BirthDeathProcess)
            self.assertIsInstance(X, RV)

    def test_mm1_rates(self):
        X = MM1(1, 1.5, num_states=5)
        np.testing.assert_allclose(X.birth_rates, [1, 1, 1, 1, 0])
        np.testing.assert_allclose(X.death_rates, [0, 1.5, 1.5, 1.5, 1.5])

    def test_mms_service_rate_is_state_dependent(self):
        # min(n, s) * mu, with s = 2, mu = 2.
        X = MMs(3, 2, servers=2, num_states=6)
        np.testing.assert_allclose(X.death_rates, [0, 2, 4, 4, 4, 4])

    def test_mmsk_is_finite_and_blocks_at_capacity(self):
        X = MMsK(4, 1, servers=2, capacity=5)
        self.assertEqual(X.num_states, 6)  # states 0..5
        self.assertEqual(X.birth_rates[-1], 0.0)  # blocked at capacity

    def test_mmss_is_special_case_of_mmsk(self):
        X = MMss(3, 1, servers=3)
        self.assertIsInstance(X, MMsK)
        self.assertEqual(X.capacity, 3)
        self.assertEqual(X.num_states, 4)

    def test_mmskn_arrival_rate_is_state_dependent(self):
        # (population - n) * lambda, with N = 6, lambda = 0.1; top zeroed.
        X = MMsKN(0.1, 1, servers=2, capacity=6, population=6)
        np.testing.assert_allclose(X.birth_rates, [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    def test_mminfinity_service_rate_is_state_dependent(self):
        # n * mu, with mu = 1.
        X = MMInfinity(5, 1, num_states=6)
        np.testing.assert_allclose(X.death_rates, [0, 1, 2, 3, 4, 5])

    # --- validation ---

    def test_error_nonpositive_rate(self):
        self.assertRaises(Exception, lambda: MM1(0, 1.5))
        self.assertRaises(Exception, lambda: MM1(1, -2))

    def test_error_bad_servers(self):
        self.assertRaises(Exception, lambda: MMs(3, 2, servers=0))
        self.assertRaises(Exception, lambda: MMs(3, 2, servers=2.5))

    def test_error_capacity_exceeds_population(self):
        self.assertRaises(
            Exception,
            lambda: MMsKN(0.1, 1, servers=2, capacity=7, population=5),
        )

    # --- distributional checks against the characteristic formulas ---

    def test_mm1_stationary_is_geometric(self):
        # M/M/1 (lambda=1, mu=2, rho=0.5): P(N=n) = (1-rho) rho^n.
        seed()
        rho = 0.5
        X = MM1(1, 2, num_states=40)
        sims = [X.draw()(40.0) for _ in range(1500)]
        obs, exp = [], []
        for n in range(0, 10):
            e = len(sims) * (1 - rho) * rho**n
            if e > 5:
                exp.append(e)
                obs.append(sum(1 for s in sims if s == n))
        pval = stats.chisquare(obs, np.array(exp) * sum(obs) / sum(exp)).pvalue
        self.assertTrue(pval > 0.01)

    def test_mmss_blocking_matches_erlang_b(self):
        # Erlang loss: long-run P(all servers busy) is the Erlang B formula.
        seed()
        from math import factorial

        lam, mu, s = 3.0, 1.0, 3
        a = lam / mu
        X = MMss(lam, mu, s)
        sims = [X.draw()(50.0) for _ in range(1500)]
        empirical = sum(x == s for x in sims) / len(sims)
        erlang_b = (a**s / factorial(s)) / sum(
            a**k / factorial(k) for k in range(s + 1)
        )
        self.assertAlmostEqual(empirical, erlang_b, delta=0.03)

    def test_mminfinity_stationary_is_poisson(self):
        # M/M/infinity: long-run number in system is Poisson(lambda/mu).
        # Use a chi-square goodness-of-fit (the counts are discrete).
        seed()
        lam, mu = 3.0, 1.0
        X = MMInfinity(lam, mu, num_states=30)
        sims = [X.draw()(30.0) for _ in range(1500)]
        pmf = stats.poisson(lam / mu).pmf
        obs, exp = [], []
        for n in range(0, 11):
            e = len(sims) * pmf(n)
            if e > 5:
                exp.append(e)
                obs.append(sum(1 for s in sims if s == n))
        pval = stats.chisquare(obs, np.array(exp) * sum(obs) / sum(exp)).pvalue
        self.assertTrue(pval > 0.01)


if __name__ == "__main__":
    unittest.main()
