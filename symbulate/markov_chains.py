import numpy as np

from .distributions import Exponential
from .math import inf
from .probability_space import ProbabilitySpace
from .random_variables import RV
from .result import (
    InfiniteVector, ContinuousTimeFunction, DiscreteValued
)

EPS = 1e-15


class MarkovChainResult(InfiniteVector, DiscreteValued):
    """One simulated sample path of a discrete-time Markov chain.

    Stores the sequence of states visited by the chain. New states are
    generated on demand as you index further into the path.

    Attributes
    ----------
    transition_matrix : numpy.ndarray
        The matrix of transition probabilities between states.
    initial_dist : array-like
        The starting probabilities for each state.
    state_labels : range or list
        The names (labels) of each state.
    n_states : int
        The total number of states in the chain.
    states : list of int
        The sequence of state indices visited so far.
    """

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Create one simulated sample path of a discrete-time Markov chain.

        Parameters
        ----------
        transition_matrix : array-like of shape (n, n)
            A square matrix where entry ``[i][j]`` is the probability of
            moving from state ``i`` to state ``j``. Each row must sum to 1.
        initial_dist : array-like of length n
            The probability of starting in each state.
        state_labels : array-like of length n, optional
            Names to use for each state. Defaults to ``0, 1, ..., n-1``.

        Raises
        ------
        Exception
            If any row of ``transition_matrix`` does not sum to 1, any
            entry is negative, the matrix is not square, or the lengths
            of ``initial_dist`` or ``state_labels`` do not match the
            number of states.
        """
        # Check transition matrix
        for row in transition_matrix:
            if abs(sum(row) - 1) > EPS:
                raise Exception("Rows of a transition matrix must sum to 1.")
            for p in row:
                if p < 0:
                    raise Exception("Probabilities cannot be negative.")
        # Check that dimensions agree
        self.transition_matrix = np.asarray(transition_matrix)
        m, n = self.transition_matrix.shape
        if m != n:
            raise Exception("Transition matrix must be square.")
        if len(initial_dist) != n:
            raise Exception("Initial distribution must be a vector whose "
                            "length matches the dimensions of the "
                            "transition matrix.")
        self.initial_dist = initial_dist
        # Process state labels
        if state_labels is not None:
            if len(state_labels) != n:
                raise Exception("There must be as many state labels as "
                                "there are states.")
            self.state_labels = state_labels
        else:
            self.state_labels = range(n)
        self.n_states = n

        # Generate initial state.
        # (self.states stores the indexes of the states, while
        #  self.values stores the labels of the states.)
        state = np.random.choice(range(n), p=self.initial_dist)
        self.states = [state]

        def _func(n):
            m = len(self.states)
            # If nth state not generated yet, generate it.
            if n >= m:
                state = self.states[m - 1]
                for _ in range(m, n + 1):
                    state = np.random.choice(
                        range(self.n_states),
                        p=self.transition_matrix[state, :]
                    )
                    self.states.append(state)
            else:
                state = self.states[n]
            return self.state_labels[state]

        super().__init__(_func)

    def get_states(self):
        """Return the sequence of states visited by this sample path.

        Returns
        -------
        MarkovChainResult
            This object itself, which can be indexed to get the state
            at any time step.
        """
        return self


class MarkovChainProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a discrete-time Markov chain.

    Each draw from this space produces one simulated sample path
    of the Markov chain.

    Attributes
    ----------
    transition_matrix : numpy.ndarray
        The matrix of transition probabilities between states.
    initial_dist : array-like
        The starting probabilities for each state.
    state_labels : range or list
        The names (labels) of each state.
    """

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Initialize probability space for a (discrete-time) Markov chain.

        Args:
          transition_matrix: n x n transition matrix
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        def _draw():
            return MarkovChainResult(transition_matrix,
                                     initial_dist,
                                     state_labels)

        super().__init__(_draw)


class MarkovChain(RV):
    """A discrete-time Markov chain, treated as a random variable.

    Each time you simulate this object, you get a random sample path
    — a sequence of states evolving according to the transition matrix.

    Attributes
    ----------
    prob_space : MarkovChainProbabilitySpace
        The underlying probability space used to generate sample paths.
    """

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Initialize a (discrete-time) Markov chain.

        Args:
          transition_matrix: n x n transition matrix
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        prob_space = MarkovChainProbabilitySpace(
            transition_matrix,
            initial_dist,
            state_labels)
        super().__init__(prob_space)


class ContinuousTimeMarkovChainResult(ContinuousTimeFunction,
                                      DiscreteValued):
    """One simulated sample path of a continuous-time Markov chain.

    Stores the sequence of states and the times spent in each state.
    You can evaluate it at any time ``t`` to find which state the chain
    is in at that moment.

    Attributes
    ----------
    states : MarkovChainResult
        The sequence of states visited (as state indices).
    rates : numpy.ndarray
        The rate of leaving each state (the negative of the diagonal
        of the generator matrix).
    times : InfiniteVector
        Unscaled interarrival times drawn from an Exponential(1) distribution.
    state_labels : range or list
        The names (labels) of each state.
    interarrival_times : InfiniteVector
        The actual time spent in each state, scaled by the holding rate.
    """

    def __init__(self, states, rates,
                 unscaled_interarrival_times,
                 state_labels):
        """Create one simulated sample path of a continuous-time Markov chain.

        Parameters
        ----------
        states : MarkovChainResult
            The sequence of states (by index) that the chain visits.
        rates : array-like
            The rate of leaving each state.
        unscaled_interarrival_times : InfiniteVector
            Exponential(1) random variables used to compute the actual
            time spent in each state.
        state_labels : range or list
            The names (labels) of each state.
        """
        self.states = states
        self.rates = rates
        self.times = unscaled_interarrival_times
        self.state_labels = state_labels

        # Define an InfiniteVector of the interarrival times.
        def interarrival_times(n):
            """Return the actual time spent in the n-th state.

            Parameters
            ----------
            n : int
                The index of the state visit (0-indexed).

            Returns
            -------
            float
                The time spent in state ``n``, equal to the unscaled
                Exponential(1) draw divided by the holding rate of that state.
            """
            for i in range(n + 1):
                state = self.states[i]
                interarrival_time = self.times[i] / self.rates[state]
            return interarrival_time
        self.interarrival_times = InfiniteVector(interarrival_times)

        def _func(t):
            total_time = 0
            n = 0
            while True:
                state = self.states[n]
                total_time += self.times[n] / self.rates[state]
                if total_time > t:
                    return self.state_labels[state]
                n += 1
        super().__init__(_func)


class ContinuousTimeMarkovChainProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a continuous-time Markov chain.

    Each draw from this space produces one simulated sample path of the
    continuous-time Markov chain, including the state sequence and the
    random times spent in each state.

    Attributes
    ----------
    generator_matrix : numpy.ndarray
        The generator (or Q) matrix of the chain. Off-diagonal entries
        are transition rates; diagonal entries are negative holding rates.
    initial_dist : array-like
        The starting probabilities for each state.
    state_labels : range or list
        The names (labels) of each state.
    n_states : int
        The total number of states.
    transition_matrix : numpy.ndarray
        The embedded discrete-time transition matrix derived from the
        generator matrix.
    """

    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        """Initialize a probability space for a continuous-time Markov chain.

        Args:
          generator_matrix: n x n generator matrix whose rows sum to 0
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        # Check generator matrix
        for i, row in enumerate(generator_matrix):
            if abs(sum(row)) > EPS:
                raise Exception("Rows of a generator matrix must sum to 0.")
            for j, q in enumerate(row):
                if j == i:
                    if q > 0:
                        raise Exception("Diagonal elements of a generator matrix " +
                                        "cannot be positive.")
                else:
                    if q < 0:
                        raise Exception("Off-diagonal elements of a generator matrix " +
                                        "cannot be negative.")
        # Check that dimensions agree
        self.generator_matrix = np.array(generator_matrix)
        m, n = self.generator_matrix.shape
        if m != n:
            raise Exception("Transition matrix must be square.")
        if len(initial_dist) != n:
            raise Exception("Initial distribution must be a vector whose "
                            "length matches the dimensions of the "
                            "transition matrix.")
        self.initial_dist = initial_dist
        # Process state labels
        if state_labels is not None:
            if len(state_labels) != n:
                raise Exception("There must be as many state labels as "
                                "there are states.")
            self.state_labels = state_labels
        else:
            self.state_labels = range(n)
        self.n_states = n

        # determine transition matrix
        transition_matrix = []
        for i, row in enumerate(self.generator_matrix):
            rate = -row[i]
            transition_matrix.append(
                [p / rate if j != i else 0 for j, p in enumerate(row)]
            )
        self.transition_matrix = np.array(transition_matrix)

        # A continuous-time Markov chain is specified by the
        # sequence of states and the unscaled interarrival times.
        def _draw():
            states = MarkovChain(self.transition_matrix,
                                 self.initial_dist).draw()
            rates = -np.diag(self.generator_matrix)
            unscaled_interarrival_times = (Exponential(1) ** inf).draw()
            return ContinuousTimeMarkovChainResult(
                states,
                rates,
                unscaled_interarrival_times,
                self.state_labels)

        super().__init__(_draw)


class ContinuousTimeMarkovChain(RV):
    """A continuous-time Markov chain, treated as a random variable.

    Each time you simulate this object, you get a random sample path
    — a function of continuous time that jumps between states at
    random moments according to the generator matrix.

    Attributes
    ----------
    prob_space : ContinuousTimeMarkovChainProbabilitySpace
        The underlying probability space used to generate sample paths.
    """

    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        """Initialize a continuous-time Markov chain.

        Args:
          generator_matrix: n x n generator matrix whose rows sum to 0
          initial_dist: length n vector of the initial distribution
          state_labels: length n vector of the labels of each state
                        (defaults to 0, 1, ..., n-1)
        """

        prob_space = ContinuousTimeMarkovChainProbabilitySpace(
            generator_matrix,
            initial_dist,
            state_labels)
        super().__init__(prob_space)
