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


if __name__ == "__main__":
    unittest.main()
