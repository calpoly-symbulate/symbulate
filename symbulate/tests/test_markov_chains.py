"""Tests for symbulate.markov_chains.

Covers both the discrete-time and continuous-time Markov chains: input
validation, lazy on-demand state generation (forward generation and
backward cached lookups), state labels, the embedded transition matrix
of a continuous-time chain, sample-path evaluation, interarrival times,
and the marginal/transition behavior of the chains.

The ``*Arrivals`` classes cover reading a continuous-time path event by
event -- ``states()``, ``interarrival_times()``, and ``arrival_times()``,
the same trio a Poisson process path supports -- for the continuous-time
chain, the birth-death wrapper, and the Yule process, including that the
holding times come out Exponential with the right rate.

Reproducibility: the discrete chain draws through ``markov_chains.rng``
and the continuous chain additionally draws interarrival times through
``distributions.rng``, so the ``seed`` helper reseeds both.
"""

import unittest

import numpy as np
import scipy.stats as stats

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt

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
        seed(42)
        chain = MarkovChain(SWAP, [1.0, 0.0])
        # Deterministic: always 0 at even steps, 1 at odd steps.
        self.assertTrue(all(v == 1 for v in chain[1].sim(100)))
        self.assertTrue(all(v == 0 for v in chain[2].sim(100)))

    def test_initial_distribution_goodness_of_fit(self):
        seed(42)
        chain = MarkovChain([[0.5, 0.5], [0.5, 0.5]], [0.3, 0.7])
        tab = chain[0].sim(Nsim).tabulate()
        obs = [tab[0], tab[1]]
        exp = [Nsim * 0.3, Nsim * 0.7]
        pval = stats.chisquare(obs, exp).pvalue
        self.assertTrue(pval > 0.01)

    def test_transition_goodness_of_fit(self):
        seed(42)
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
Q3 = [[-2.0, 1.0, 1.0], [1.0, -1.0, 0.0], [2.0, 2.0, -4.0]]


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
        space = ContinuousTimeMarkovChainProbabilitySpace(Q3, [1.0, 0.0, 0.0])
        expected = [[0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
        np.testing.assert_allclose(space.transition_matrix, expected)

    def test_draw_returns_result(self):
        seed(42)
        space = ContinuousTimeMarkovChainProbabilitySpace(Q2, [1.0, 0.0])
        self.assertIsInstance(space.draw(), ContinuousTimeMarkovChainResult)


class TestContinuousTimeMarkovChain(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(ContinuousTimeMarkovChain(Q2, [1.0, 0.0]), RV)

    def test_draw_returns_continuous_time_function(self):
        seed(42)
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0]).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)
        self.assertIsInstance(path, DiscreteValued)

    def test_path_starts_in_initial_state(self):
        seed(42)
        # initial_dist [1, 0] -> chain starts in state 0 -> label 'A'.
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0], state_labels=["A", "B"]).draw()
        self.assertEqual(path(0), "A")

    def test_path_values_are_valid_labels(self):
        seed(42)
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0], state_labels=["A", "B"]).draw()
        for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
            self.assertIn(path(t), ["A", "B"])

    def test_interarrival_times_are_positive(self):
        seed(42)
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


class TestContinuousTimeMarkovChainArrivals(unittest.TestCase):
    """The states / interarrival-times / arrival-times trio on a drawn path.

    This is the same interface a Poisson process sample path provides, so
    the same ``states()``, ``interarrival_times()``, and ``arrival_times()``
    functions have to work on a continuous-time Markov chain path.
    """

    def test_helpers_accept_a_path(self):
        seed(42)
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0]).draw()
        self.assertIsInstance(states(path), InfiniteVector)
        self.assertIsInstance(interarrival_times(path), InfiniteVector)
        self.assertIsInstance(arrival_times(path), InfiniteVector)

    def test_states_are_labels_not_indices(self):
        # Regression: get_states() used to return the embedded chain's raw
        # state indices (0, 1, ...), which disagreed with what evaluating
        # the path returns. The states must be the labels.
        seed(42)
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0], state_labels=["A", "B"]).draw()
        for n in range(6):
            self.assertIn(states(path)[n], ["A", "B"])

    def test_first_state_matches_path_at_time_zero(self):
        seed(42)
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0], state_labels=["A", "B"]).draw()
        self.assertEqual(states(path)[0], path(0))

    def test_states_agree_with_path_just_after_each_jump(self):
        # The n-th state is the one the chain is in immediately after the
        # (n-1)-th jump, so the two descriptions of the path must line up.
        seed(42)
        path = ContinuousTimeMarkovChain(Q3, [1.0, 0.0, 0.0]).draw()
        jumps = arrival_times(path)
        for n in range(5):
            self.assertEqual(path(jumps[n] + 1e-9), states(path)[n + 1])

    def test_arrival_times_are_the_running_total_of_interarrivals(self):
        seed(42)
        path = ContinuousTimeMarkovChain(Q2, [1.0, 0.0]).draw()
        waits = interarrival_times(path)
        jumps = arrival_times(path)
        for n in range(5):
            self.assertAlmostEqual(jumps[n], sum(waits[i] for i in range(n + 1)))

    def test_arrival_times_are_increasing(self):
        seed(42)
        path = ContinuousTimeMarkovChain(Q3, [1.0, 0.0, 0.0]).draw()
        jumps = arrival_times(path)
        for earlier, later in zip(
            [jumps[n] for n in range(6)], [jumps[n] for n in range(1, 7)]
        ):
            self.assertLess(earlier, later)

    def test_holding_time_is_exponential_with_the_states_rate(self):
        # Q2 leaves state 0 at rate 1, so the first holding time is
        # Exponential(1): mean 1 and variance 1.
        seed(42)
        chain = ContinuousTimeMarkovChain(Q2, [1.0, 0.0])
        wait = chain.apply(interarrival_times)[0].sim(Nsim)
        self.assertAlmostEqual(wait.mean(), 1.0, delta=0.05)
        self.assertAlmostEqual(wait.var(), 1.0, delta=0.15)

    def test_holding_time_scales_with_the_rate_of_leaving(self):
        # Started in state 1, which Q2 leaves at rate 2: mean 1 / 2.
        seed(42)
        chain = ContinuousTimeMarkovChain(Q2, [0.0, 1.0])
        wait = chain.apply(interarrival_times)[0].sim(Nsim)
        self.assertAlmostEqual(wait.mean(), 0.5, delta=0.03)

    def test_arrival_times_are_random_variables_that_simulate(self):
        # The time of the second jump out of state 0 is the sum of two
        # independent Exponential(1) waits (0 -> 1 -> 0), so it has mean
        # 1 + 1 / 2 = 1.5 given Q2's rates of 1 and 2.
        seed(42)
        chain = ContinuousTimeMarkovChain(Q2, [1.0, 0.0])
        second_jump = chain.apply(arrival_times)[1].sim(Nsim)
        self.assertAlmostEqual(second_jump.mean(), 1.5, delta=0.06)


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
        seed(42)
        X = BirthDeathProcess(birth_rates=1, death_rates=2, num_states=10, initial=3)
        self.assertEqual(X.draw()(0), 3)

    def test_path_values_are_valid_states(self):
        seed(42)
        X = BirthDeathProcess(birth_rates=1, death_rates=1.5, num_states=20)
        path = X.draw()
        for t in [0.0, 1.0, 5.0, 10.0]:
            v = path(t)
            self.assertIn(v, range(20))

    def test_mm1_stationary_is_geometric(self):
        # An M/M/1 queue (lambda=1, mu=2, rho=0.5) has stationary distribution
        # P(N=n) = (1-rho) rho^n. Simulate the queue length at a large time,
        # started empty, and check it against that geometric.
        seed(42)
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


class TestBirthDeathProcessArrivals(unittest.TestCase):
    """A birth-death path read event by event, inherited from the CTMC."""

    def test_helpers_accept_a_path(self):
        seed(42)
        path = BirthDeathProcess(birth_rates=1, death_rates=1.5, num_states=20).draw()
        self.assertIsInstance(states(path), InfiniteVector)
        self.assertIsInstance(interarrival_times(path), InfiniteVector)
        self.assertIsInstance(arrival_times(path), InfiniteVector)

    def test_states_are_counts_and_start_at_initial(self):
        seed(42)
        X = BirthDeathProcess(birth_rates=1, death_rates=2, num_states=10, initial=3)
        path = X.draw()
        self.assertEqual(states(path)[0], 3)
        self.assertEqual(states(path)[0], path(0))

    def test_states_step_by_one(self):
        # Only a birth or a death can happen, so consecutive counts differ by 1.
        seed(42)
        path = BirthDeathProcess(
            birth_rates=2, death_rates=2, num_states=30, initial=10
        ).draw()
        visited = [states(path)[n] for n in range(15)]
        for earlier, later in zip(visited, visited[1:]):
            self.assertEqual(abs(later - earlier), 1)

    def test_states_honor_custom_labels(self):
        # Regression: with custom state_labels the states used to come back
        # as bare indices instead of the labels the path itself returns.
        seed(42)
        labels = ["empty", "low", "medium", "high"]
        path = BirthDeathProcess(
            birth_rates=1, death_rates=1.5, num_states=4, state_labels=labels
        ).draw()
        self.assertEqual(states(path)[0], "empty")
        for n in range(6):
            self.assertIn(states(path)[n], labels)

    def test_holding_time_uses_the_combined_birth_and_death_rate(self):
        # From count 4 (interior), a birth at rate 2 or a death at rate 3 ends
        # the stay, so the wait is Exponential(5): mean 1 / 5.
        seed(42)
        X = BirthDeathProcess(birth_rates=2, death_rates=3, num_states=20, initial=4)
        wait = X.apply(interarrival_times)[0].sim(Nsim)
        self.assertAlmostEqual(wait.mean(), 1 / 5, delta=0.01)

    def test_empty_queue_waits_only_for_an_arrival(self):
        # At count 0 nothing can die, so the wait is Exponential(birth rate)
        # alone: mean 1 / 2 for an arrival rate of 2.
        seed(42)
        X = BirthDeathProcess(birth_rates=2, death_rates=3, num_states=20, initial=0)
        wait = X.apply(interarrival_times)[0].sim(Nsim)
        self.assertAlmostEqual(wait.mean(), 0.5, delta=0.02)

    def test_arrival_times_are_the_running_total_of_interarrivals(self):
        seed(42)
        path = BirthDeathProcess(
            birth_rates=1, death_rates=1.5, num_states=20, initial=5
        ).draw()
        waits = interarrival_times(path)
        jumps = arrival_times(path)
        for n in range(5):
            self.assertAlmostEqual(jumps[n], sum(waits[i] for i in range(n + 1)))

    def test_mm1_queue_inherits_the_accessors(self):
        # The M/M/... wrappers are birth-death processes, so they get this too.
        seed(42)
        path = MM1(arrival_rate=1, service_rate=2, num_states=30).draw()
        self.assertEqual(states(path)[0], 0)
        self.assertGreater(interarrival_times(path)[0], 0)
        self.assertGreater(arrival_times(path)[1], arrival_times(path)[0])


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
        seed(42)
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
        seed(42)
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
        seed(42)
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


class TestYuleProcess(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(YuleProcess(birth_rate=0.5), RV)

    def test_draw_returns_continuous_time_function(self):
        seed(42)
        path = YuleProcess(birth_rate=0.5).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)
        self.assertIsInstance(path, DiscreteValued)

    def test_path_starts_at_initial(self):
        seed(42)
        self.assertEqual(YuleProcess(birth_rate=0.5).draw()(0), 1)
        self.assertEqual(YuleProcess(birth_rate=0.5, initial=3).draw()(0), 3)

    def test_path_is_nondecreasing(self):
        # A pure-birth process only ever grows.
        seed(42)
        path = YuleProcess(birth_rate=0.5).draw()
        values = [path(t) for t in [0, 1, 2, 4, 6, 8]]
        self.assertEqual(values, sorted(values))

    def test_interarrival_times_are_positive(self):
        seed(42)
        path = YuleProcess(birth_rate=0.5).draw()
        for k in range(5):
            self.assertGreater(path.interarrival_times[k], 0)

    def test_states_climb_one_at_a_time_from_initial(self):
        # Regression: a Yule path had interarrival and arrival times but no
        # states, so states() raised instead of returning the populations.
        seed(42)
        path = YuleProcess(birth_rate=0.5, initial=2).draw()
        self.assertEqual([states(path)[k] for k in range(5)], [2, 3, 4, 5, 6])
        self.assertEqual(states(path)[0], path(0))

    def test_waits_shrink_as_the_population_grows(self):
        # With n individuals the next birth arrives at rate n * birth_rate, so
        # the wait from population 1 averages 1 / 0.5 = 2 and the next one,
        # from population 2, averages 1 / 1.0 = 1.
        seed(42)
        X = YuleProcess(birth_rate=0.5)
        first = X.apply(interarrival_times)[0].sim(Nsim)
        second = X.apply(interarrival_times)[1].sim(Nsim)
        self.assertAlmostEqual(first.mean(), 2.0, delta=0.1)
        self.assertAlmostEqual(second.mean(), 1.0, delta=0.05)

    def test_arrival_times_are_the_running_total_of_interarrivals(self):
        seed(42)
        path = YuleProcess(birth_rate=0.5).draw()
        waits = interarrival_times(path)
        jumps = arrival_times(path)
        for k in range(5):
            self.assertAlmostEqual(jumps[k], sum(waits[i] for i in range(k + 1)))

    def test_error_nonpositive_birth_rate(self):
        self.assertRaises(Exception, lambda: YuleProcess(birth_rate=0))
        self.assertRaises(Exception, lambda: YuleProcess(birth_rate=-1))

    def test_error_bad_initial(self):
        self.assertRaises(Exception, lambda: YuleProcess(birth_rate=0.5, initial=0))
        self.assertRaises(Exception, lambda: YuleProcess(birth_rate=0.5, initial=1.5))

    def test_marginal_is_geometric(self):
        # Started from 1, N(t) is Geometric with parameter exp(-birth_rate * t).
        seed(42)
        lam, t = 0.5, 2.0
        p = np.exp(-lam * t)
        X = YuleProcess(birth_rate=lam)
        sims = [X.draw()(t) for _ in range(4000)]
        th = stats.geom(p)
        obs, exp = [], []
        for k in range(1, 12):
            e = len(sims) * th.pmf(k)
            if e > 5:
                exp.append(e)
                obs.append(sum(1 for s in sims if s == k))
        pval = stats.chisquare(obs, np.array(exp) * sum(obs) / sum(exp)).pvalue
        self.assertTrue(pval > 0.01)

    def test_mean_grows_exponentially(self):
        # E[N(t)] = initial * exp(birth_rate * t).
        seed(42)
        lam, t, initial = 0.5, 3.0, 2
        X = YuleProcess(birth_rate=lam, initial=initial)
        sims = [X.draw()(t) for _ in range(3000)]
        self.assertAlmostEqual(np.mean(sims), initial * np.exp(lam * t), delta=0.5)

    def test_reproducible_under_same_seed(self):
        X = YuleProcess(birth_rate=0.5)
        seed(7)
        first = [X.draw()(t) for t in [1.0, 3.0, 5.0]]
        seed(7)
        second = [X.draw()(t) for t in [1.0, 3.0, 5.0]]
        self.assertEqual(first, second)


class TestSIR(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(SIR(100, 0.3, 0.1), RV)

    def test_draw_returns_continuous_time_function(self):
        seed(42)
        path = SIR(100, 0.3, 0.1).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_path_starts_at_initial_state(self):
        seed(42)
        path = SIR(100, 0.3, 0.1, initial_infected=5, initial_recovered=2).draw()
        self.assertEqual(list(path(0)), [93, 5, 2])

    def test_population_is_conserved(self):
        seed(42)
        path = SIR(500, 0.3, 0.1, initial_infected=5).draw()
        for t in [0, 1, 5, 20, 100]:
            self.assertEqual(sum(path(t)), 500)

    def test_S_nonincreasing_R_nondecreasing(self):
        seed(42)
        path = SIR(500, 0.3, 0.1, initial_infected=5).draw()
        arr = np.array(path.states)
        self.assertTrue(np.all(np.diff(arr[:, 0]) <= 0))  # S only falls
        self.assertTrue(np.all(np.diff(arr[:, 2]) >= 0))  # R only rises

    def test_epidemic_terminates_with_no_infectives(self):
        seed(42)
        path = SIR(500, 0.3, 0.1, initial_infected=5).draw()
        self.assertEqual(path.states[-1][1], 0)  # final I is 0

    def test_compartments_exposed_as_time_functions(self):
        seed(42)
        path = SIR(100, 0.3, 0.1).draw()
        for c in ["S", "I", "R"]:
            self.assertIsInstance(getattr(path, c), ContinuousTimeFunction)

    def test_compartment_plot_defaults_to_epidemic_end(self):
        seed(42)
        path = SIR(1000, 0.3, 0.1, initial_infected=5).draw()
        plt.figure()
        path.I.plot()
        xs = plt.gca().lines[-1].get_xdata()
        self.assertAlmostEqual(xs[0], 0)
        self.assertAlmostEqual(xs[-1], path.event_times[-1])
        plt.close("all")

    def test_compartment_plot_explicit_tmax_overrides(self):
        seed(42)
        path = SIR(1000, 0.3, 0.1, initial_infected=5).draw()
        plt.figure()
        path.I.plot(tmax=30)
        self.assertAlmostEqual(plt.gca().lines[-1].get_xdata()[-1], 30)
        plt.close("all")

    def test_plot_draws_every_compartment(self):
        # path.plot() overlays all three compartments, each labeled, over the
        # outbreak's time axis -- shorthand for plotting S, I, R individually.
        seed(42)
        path = SIR(1000, 0.3, 0.1, initial_infected=5).draw()
        plt.figure()
        path.plot()
        lines = plt.gca().lines
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [l.get_label() for l in lines],
            ["Susceptible", "Infectious", "Recovered"],
        )
        self.assertAlmostEqual(lines[-1].get_xdata()[-1], path.event_times[-1])
        plt.close("all")

    def test_sim_one_plots_like_draw(self):
        # .sim(1).plot() routes through RVResults' ensemble plotter but should
        # still draw the three labeled compartments, like .draw().plot().
        seed(42)
        plt.figure()
        SIR(1000, 0.3, 0.1, initial_infected=5).sim(1).plot()
        lines = plt.gca().lines
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [l.get_label() for l in lines],
            ["Susceptible", "Infectious", "Recovered"],
        )
        plt.close("all")

    def test_sim_many_shares_one_legend_and_color_per_compartment(self):
        # Overlaying k outbreaks draws 3*k lines but adds each compartment to
        # the legend only once, and holds each compartment to a single color.
        seed(42)
        plt.figure()
        SIR(1000, 0.3, 0.1, initial_infected=5).sim(5).plot()
        lines = plt.gca().lines
        self.assertEqual(len(lines), 15)
        legend = [l.get_label() for l in lines if not l.get_label().startswith("_")]
        self.assertEqual(legend, ["Susceptible", "Infectious", "Recovered"])
        # every 3rd line is the same compartment; each must be one color
        for offset in range(3):
            colors = {lines[offset + 3 * k].get_color() for k in range(5)}
            self.assertEqual(len(colors), 1)
        plt.close("all")

    def test_supercritical_outbreak_larger_than_subcritical(self):
        # R0 = infection_rate / recovery_rate. Above 1 a large outbreak is
        # likely; below 1 the epidemic dies out quickly. Checks the rates are
        # wired correctly (infection vs recovery).
        seed(42)
        big = np.mean(
            [
                SIR(1000, 0.3, 0.1, initial_infected=5).draw().states[-1][2]
                for _ in range(150)
            ]
        )
        small = np.mean(
            [
                SIR(1000, 0.05, 0.1, initial_infected=5).draw().states[-1][2]
                for _ in range(150)
            ]
        )
        self.assertGreater(big, 500)
        self.assertLess(small, 100)

    def test_validation(self):
        self.assertRaises(Exception, lambda: SIR(0, 0.3, 0.1))  # population
        self.assertRaises(Exception, lambda: SIR(100, 0, 0.1))  # infection_rate
        self.assertRaises(Exception, lambda: SIR(100, 0.3, -1))  # recovery_rate
        self.assertRaises(Exception, lambda: SIR(100, 0.3, 0.1, initial_infected=0))
        self.assertRaises(
            Exception,
            lambda: SIR(10, 0.3, 0.1, initial_infected=8, initial_recovered=5),
        )

    def test_reproducible_under_same_seed(self):
        X = SIR(500, 0.3, 0.1, initial_infected=5)
        seed(7)
        first = [list(X.draw()(t)) for t in [1.0, 5.0, 20.0]]
        seed(7)
        second = [list(X.draw()(t)) for t in [1.0, 5.0, 20.0]]
        self.assertEqual(first, second)


class TestSEIR(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(SEIR(100, 0.4, 0.2, 0.1), RV)

    def test_path_starts_at_initial_state(self):
        seed(42)
        path = SEIR(100, 0.4, 0.2, 0.1, initial_infected=5, initial_exposed=3).draw()
        self.assertEqual(list(path(0)), [92, 3, 5, 0])  # order S, E, I, R

    def test_population_is_conserved(self):
        seed(42)
        path = SEIR(500, 0.4, 0.2, 0.1, initial_infected=5).draw()
        for t in [0, 1, 5, 20, 100]:
            self.assertEqual(sum(path(t)), 500)

    def test_epidemic_terminates_with_no_exposed_or_infectives(self):
        seed(42)
        final = SEIR(500, 0.4, 0.2, 0.1, initial_infected=5).draw().states[-1]
        self.assertEqual(final[1], 0)  # E
        self.assertEqual(final[2], 0)  # I

    def test_compartments_exposed_as_time_functions(self):
        seed(42)
        path = SEIR(100, 0.4, 0.2, 0.1).draw()
        for c in ["S", "E", "I", "R"]:
            self.assertIsInstance(getattr(path, c), ContinuousTimeFunction)

    def test_plot_draws_every_compartment(self):
        # path.plot() overlays all four compartments (E included), each labeled.
        seed(42)
        path = SEIR(1000, 0.5, 0.2, 0.1, initial_infected=5).draw()
        plt.figure()
        path.plot()
        lines = plt.gca().lines
        self.assertEqual(len(lines), 4)
        self.assertEqual(
            [l.get_label() for l in lines],
            ["Susceptible", "Exposed", "Infectious", "Recovered"],
        )
        plt.close("all")

    def test_validation(self):
        self.assertRaises(Exception, lambda: SEIR(100, 0.4, 0, 0.1))  # incubation
        self.assertRaises(
            Exception, lambda: SEIR(100, 0.4, 0.2, 0.1, initial_exposed=-1)
        )


if __name__ == "__main__":
    unittest.main()
