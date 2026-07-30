import numbers

import numpy as np

from .distributions import Exponential
from .math import inf
from .probability_space import ProbabilitySpace
from .random_variables import RV
from .result import InfiniteVector, ContinuousTimeFunction, DiscreteValued

EPS = 1e-15
rng = np.random.default_rng()


def _as_rate_array(spec, num_states, name):
    """Turn a birth- or death-rate specification into a per-state array.

    Accepts three convenient forms and returns a NumPy array with one rate
    per state:

    - a single number -- the same rate in every state,
    - a callable ``rate(n)`` -- evaluated at each state index ``n``,
    - a list or array -- one rate per state (must have ``num_states`` entries).

    Parameters
    ----------
    spec : number, callable, or array-like
        The rate specification, in one of the forms above.
    num_states : int
        The number of states in the process.
    name : str
        The parameter name (``"birth_rates"`` or ``"death_rates"``), used in
        error messages.

    Returns
    -------
    numpy.ndarray
        The rate in each state, as a length-``num_states`` array of floats.

    Raises
    ------
    Exception
        If a list is the wrong length, or any rate is negative or not a
        finite number.
    """
    if callable(spec):
        values = [spec(n) for n in range(num_states)]
    elif np.ndim(spec) == 0:
        values = [spec] * num_states
    else:
        values = list(spec)
        if len(values) != num_states:
            raise Exception(
                "%s must have one rate per state: expected %d values, got %d. "
                "Pass a single number for a constant rate, a function of the "
                "state, or a list with one entry per state."
                % (name, num_states, len(values))
            )
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise Exception("%s must all be non-negative numbers." % name)
    return arr


class MarkovChainResult(InfiniteVector, DiscreteValued):
    """One simulated sample path of a discrete-time Markov chain.

    Stores the sequence of states visited by the chain. New states are
    generated on demand as you index further into the path.

    Parameters
    ----------
    transition_matrix : array-like of shape (n, n)
        A square matrix where entry ``[i][j]`` is the probability of
        moving from state ``i`` to state ``j``. Each row must sum to 1.
    initial_dist : array-like of length n
        The probability of starting in each state.
    state_labels : array-like of length n, optional
        Names to use for each state. Defaults to ``0, 1, ..., n-1``.

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

    Examples
    --------
    >>> P = [[0.9, 0.1], [0.3, 0.7]]
    >>> path = MarkovChainResult(P, [0.5, 0.5])
    >>> path[0], path[1], path[2]  # doctest: +SKIP
    (0, 0, 1)
    """

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Create one simulated sample path of a discrete-time Markov chain.

        Raises
        ------
        Exception
            If any row of ``transition_matrix`` does not sum to 1, any
            entry is negative, the matrix is not square, the lengths
            of ``initial_dist`` or ``state_labels`` do not match the
            number of states, or ``initial_dist`` is not a valid
            probability distribution (negative entries or does not sum to 1).
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
            raise Exception(
                "Initial distribution must be a vector whose "
                "length matches the dimensions of the "
                "transition matrix."
            )
        if any(p < 0 for p in initial_dist):
            raise Exception("Initial distribution cannot have negative probabilities.")
        if abs(sum(initial_dist) - 1) > EPS:
            raise Exception("Initial distribution must sum to 1.")
        self.initial_dist = initial_dist
        # Process state labels
        if state_labels is not None:
            if len(state_labels) != n:
                raise Exception(
                    "There must be as many state labels as " "there are states."
                )
            self.state_labels = state_labels
        else:
            self.state_labels = range(n)
        self.n_states = n

        # Generate initial state.
        # (self.states stores the indexes of the states, while
        #  self.values stores the labels of the states.)
        state = rng.choice(n, p=self.initial_dist)
        self.states = [state]

        def _func(n):
            m = len(self.states)
            # If nth state not generated yet, generate it.
            if n >= m:
                state = self.states[m - 1]
                for _ in range(m, n + 1):
                    state = rng.choice(
                        self.n_states, p=self.transition_matrix[state, :]
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

        Examples
        --------
        >>> P = [[0.9, 0.1], [0.3, 0.7]]
        >>> path = MarkovChainResult(P, [1.0, 0.0])
        >>> states = path.get_states()
        >>> states[0], states[1]  # doctest: +SKIP
        (0, 0)
        """
        return self


class MarkovChainProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a discrete-time Markov chain.

    Each draw from this space produces one simulated sample path
    of the Markov chain.

    Parameters
    ----------
    transition_matrix : array-like of shape (n, n)
        A square matrix where entry ``[i][j]`` is the probability of
        moving from state ``i`` to state ``j``. Each row must sum to 1.
    initial_dist : array-like of length n
        The probability of starting in each state.
    state_labels : array-like of length n, optional
        Names to use for each state. Defaults to ``0, 1, ..., n-1``.

    Attributes
    ----------
    transition_matrix : numpy.ndarray
        The matrix of transition probabilities between states.
    initial_dist : array-like
        The starting probabilities for each state.
    state_labels : range or list
        The names (labels) of each state.

    Examples
    --------
    >>> P = [[0.9, 0.1], [0.3, 0.7]]
    >>> prob_space = MarkovChainProbabilitySpace(P, [0.5, 0.5], state_labels=['A', 'B'])
    """

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Initialize probability space for a (discrete-time) Markov chain."""

        def _draw():
            return MarkovChainResult(transition_matrix, initial_dist, state_labels)

        super().__init__(_draw)


class MarkovChain(RV):
    """A discrete-time Markov chain, treated as a random variable.

    Each time you simulate this object, you get a random sample path
    — a sequence of states evolving according to the transition matrix.

    Parameters
    ----------
    transition_matrix : array-like of shape (n, n)
        A square matrix where entry ``[i][j]`` is the probability of
        moving from state ``i`` to state ``j``. Each row must sum to 1.
    initial_dist : array-like of length n
        The probability of starting in each state.
    state_labels : array-like of length n, optional
        Names to use for each state. Defaults to ``0, 1, ..., n-1``.

    Attributes
    ----------
    prob_space : MarkovChainProbabilitySpace
        The underlying probability space used to generate sample paths.

    Examples
    --------
    >>> P = [[0.9, 0.1], [0.3, 0.7]]
    >>> chain = MarkovChain(P, [0.5, 0.5])
    >>> path = chain.draw()
    >>> path[0], path[1], path[2]  # doctest: +SKIP
    (0, 0, 1)
    """

    def __init__(self, transition_matrix, initial_dist, state_labels=None):
        """Initialize a (discrete-time) Markov chain."""

        prob_space = MarkovChainProbabilitySpace(
            transition_matrix, initial_dist, state_labels
        )
        super().__init__(prob_space)


class ContinuousTimeMarkovChainResult(ContinuousTimeFunction, DiscreteValued):
    """One simulated sample path of a continuous-time Markov chain.

    Stores the sequence of states and the times spent in each state.
    You can evaluate it at any time ``t`` to find which state the chain
    is in at that moment.

    Parameters
    ----------
    states : MarkovChainResult
        The sequence of states visited (as state indices).
    rates : array-like
        The rate of leaving each state (the negative diagonal of the
        generator matrix).
    unscaled_interarrival_times : InfiniteVector
        Exponential(1) random variables used to compute the actual
        time spent in each state.
    state_labels : range or list
        The names or labels of each state.

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

    Examples
    --------
    >>> Q = [[-1, 1], [2, -2]]
    >>> chain = ContinuousTimeMarkovChain(Q, [1.0, 0.0])
    >>> path = chain.draw()
    >>> path(2.5)  # doctest: +SKIP
    'A'
    """

    def __init__(self, states, rates, unscaled_interarrival_times, state_labels):
        """Create one simulated sample path of a continuous-time Markov chain."""
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

    Parameters
    ----------
    generator_matrix : array-like of shape (n, n)
        The generator (or Q) matrix of the chain. Off-diagonal entries
        are transition rates (non-negative); diagonal entries are the
        negative holding rates. Each row must sum to 0.
    initial_dist : array-like of length n
        The probability of starting in each state.
    state_labels : array-like of length n, optional
        Names to use for each state. Defaults to ``0, 1, ..., n-1``.

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

    Examples
    --------
    >>> Q = [[-1, 1], [2, -2]]
    >>> space = ContinuousTimeMarkovChainProbabilitySpace(Q, [1.0, 0.0], state_labels=['A', 'B'])
    >>> path = space.draw()
    >>> path(1.0)  # doctest: +SKIP
    'A'
    """

    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        """Initialize a probability space for a continuous-time Markov chain.

        Raises
        ------
        Exception
            If any row of ``generator_matrix`` does not sum to 0, any
            off-diagonal entry is negative, any diagonal entry is positive,
            the matrix is not square, the lengths of ``initial_dist`` or
            ``state_labels`` do not match the number of states, or any
            state is absorbing (rate = 0).
        """

        # Check generator matrix
        for i, row in enumerate(generator_matrix):
            if abs(sum(row)) > EPS:
                raise Exception("Rows of a generator matrix must sum to 0.")
            for j, q in enumerate(row):
                if j == i:
                    if q > 0:
                        raise Exception(
                            "Diagonal elements of a generator matrix "
                            + "cannot be positive."
                        )
                else:
                    if q < 0:
                        raise Exception(
                            "Off-diagonal elements of a generator matrix "
                            + "cannot be negative."
                        )
        # Check that dimensions agree
        self.generator_matrix = np.array(generator_matrix)
        m, n = self.generator_matrix.shape
        if m != n:
            raise Exception("Generator matrix must be square.")
        if len(initial_dist) != n:
            raise Exception(
                "Initial distribution must be a vector whose "
                "length matches the dimensions of the "
                "transition matrix."
            )
        self.initial_dist = initial_dist
        # Process state labels
        if state_labels is not None:
            if len(state_labels) != n:
                raise Exception(
                    "There must be as many state labels as " "there are states."
                )
            self.state_labels = state_labels
        else:
            self.state_labels = range(n)
        self.n_states = n

        # determine transition matrix
        transition_matrix = []
        for i, row in enumerate(self.generator_matrix):
            rate = -row[i]
            if rate == 0:
                raise Exception(
                    f"State {i} is absorbing (rate = 0). "
                    "Continuous-time Markov chains with absorbing states "
                    "are not supported."
                )
            transition_matrix.append(
                [p / rate if j != i else 0 for j, p in enumerate(row)]
            )
        self.transition_matrix = np.array(transition_matrix)

        # Build the unscaled interarrival-time space once, not once per draw:
        # it never varies, and each `.draw()` on it already yields a fresh,
        # independent sequence.
        unscaled_interarrivals = Exponential(1) ** inf

        # A continuous-time Markov chain is specified by the
        # sequence of states and the unscaled interarrival times.
        def _draw():
            states = MarkovChain(self.transition_matrix, self.initial_dist).draw()
            rates = -np.diag(self.generator_matrix)
            return ContinuousTimeMarkovChainResult(
                states, rates, unscaled_interarrivals.draw(), self.state_labels
            )

        super().__init__(_draw)


class ContinuousTimeMarkovChain(RV):
    """A continuous-time Markov chain, treated as a random variable.

    Each time you simulate this object, you get a random sample path
    — a function of continuous time that jumps between states at
    random moments according to the generator matrix.

    Parameters
    ----------
    generator_matrix : array-like of shape (n, n)
        The generator (or Q) matrix of the chain. Off-diagonal entries
        are transition rates (non-negative); diagonal entries are the
        negative holding rates. Each row must sum to 0.
    initial_dist : array-like of length n
        The probability of starting in each state.
    state_labels : array-like of length n, optional
        Names to use for each state. Defaults to ``0, 1, ..., n-1``.

    Attributes
    ----------
    prob_space : ContinuousTimeMarkovChainProbabilitySpace
        The underlying probability space used to generate sample paths.

    Examples
    --------
    >>> Q = [[-1, 1], [2, -2]]
    >>> chain = ContinuousTimeMarkovChain(Q, [1.0, 0.0], state_labels=['A', 'B'])
    >>> path = chain.draw()
    >>> path(2.5)  # doctest: +SKIP
    'A'
    """

    def __init__(self, generator_matrix, initial_dist, state_labels=None):
        """Initialize a continuous-time Markov chain."""

        prob_space = ContinuousTimeMarkovChainProbabilitySpace(
            generator_matrix, initial_dist, state_labels
        )
        super().__init__(prob_space)


class BirthDeathProcess(ContinuousTimeMarkovChain):
    """A birth-death process, treated as a random variable.

    A birth-death process is a continuous-time Markov chain on the states
    ``0, 1, 2, ..., num_states - 1`` that can only move up by one ("a birth")
    or down by one ("a death") at a time. From state ``n`` it moves to
    ``n + 1`` at the birth rate and to ``n - 1`` at the death rate. Each draw
    is a sample path -- a function of continuous time giving the population
    (or queue length, count, ...) at each moment.

    This is a convenience wrapper that builds the generator matrix for you
    from the birth and death rates and hands it to
    :class:`ContinuousTimeMarkovChain`; it adds no new simulation logic. A
    great many classic models are just birth-death processes with particular
    rates, so they are all reachable through this one class -- see the
    examples below for the ``M/M/1``, ``M/M/s``, finite-capacity, Erlang-loss,
    and machine-repair queues.

    Parameters
    ----------
    birth_rates : number, callable, or array-like
        The rate of moving up one state ("a birth"). May be a single number
        (the same rate in every state), a function ``birth_rate(n)`` of the
        state, or a list with one entry per state.
    death_rates : number, callable, or array-like
        The rate of moving down one state ("a death"), in the same three
        forms as ``birth_rates``.
    num_states : int
        The number of states, so the state space is
        ``0, 1, ..., num_states - 1``. Must be at least 2. For a process with
        no natural upper limit (such as an ``M/M/1`` queue) choose a
        ``num_states`` large enough that the top state is almost never
        reached -- this truncates the (infinite) state space so it can be
        simulated.
    initial : int, optional
        The starting state. Must be between 0 and ``num_states - 1``.
        Default is 0.
    state_labels : array-like of length num_states, optional
        Names to use for each state. Defaults to the state numbers
        ``0, 1, ..., num_states - 1``, so a path returns the count directly.

    Attributes
    ----------
    birth_rates : numpy.ndarray
        The birth rate in each state (with the top state's birth rate set
        to 0, since there is no higher state).
    death_rates : numpy.ndarray
        The death rate in each state (with state 0's death rate set to 0,
        since there is no lower state).
    num_states : int
        The number of states.
    prob_space : ContinuousTimeMarkovChainProbabilitySpace
        The underlying probability space used to generate sample paths.

    Notes
    -----
    The birth rate of the top state and the death rate of state 0 are
    automatically set to 0, because there is no state to move to -- so, for a
    finite-capacity queue, simply capping ``num_states`` gives the correct
    "blocking" behavior with no extra work.

    Every state must have some way out (a positive birth or death rate);
    otherwise the process could get stuck there forever, which a
    continuous-time Markov chain does not support. This means a pure-birth
    process (all death rates 0) is not expressible this way, since its top
    state would be a dead end.

    Examples
    --------
    >>> from symbulate import *
    >>> # M/M/1 queue: arrivals at rate 1, service at rate 1.5, truncated at 40.
    >>> queue = BirthDeathProcess(birth_rates=1, death_rates=1.5, num_states=41)
    >>> path = queue.draw()
    >>> path(5.0)  # doctest: +SKIP
    2
    >>> # M/M/s queue with s = 3 servers: service rate is min(n, s) * mu.
    >>> mu = 1.5
    >>> mms = BirthDeathProcess(
    ...     birth_rates=1, death_rates=lambda n: min(n, 3) * mu, num_states=41
    ... )
    """

    def __init__(
        self, birth_rates, death_rates, num_states, initial=0, state_labels=None
    ):
        """Initialize a birth-death process.

        Raises
        ------
        Exception
            If ``num_states`` is not an integer of at least 2; if a rate is
            negative or the wrong length; if ``initial`` is not a state index
            between 0 and ``num_states - 1``; or if some state has no way out
            (both its birth and death rates are 0).
        """
        if not isinstance(num_states, numbers.Integral) or num_states < 2:
            raise Exception(
                "num_states must be a whole number that is at least 2 (the "
                "states are 0, 1, ..., num_states - 1)."
            )

        births = _as_rate_array(birth_rates, num_states, "birth_rates")
        deaths = _as_rate_array(death_rates, num_states, "death_rates")

        # There is no state above the top one or below 0, so those moves are
        # impossible regardless of what the user supplied.
        births[num_states - 1] = 0.0
        deaths[0] = 0.0

        # A continuous-time Markov chain cannot have a state with no way out.
        stuck = np.where(births + deaths == 0)[0]
        if len(stuck) > 0:
            raise Exception(
                "State %d has no way out: its birth and death rates are both "
                "0, so the process would stay there forever. Give it a "
                "positive birth or death rate. (A pure-birth or pure-death "
                "process cannot be built this way.)" % int(stuck[0])
            )

        if not isinstance(initial, numbers.Integral) or not (0 <= initial < num_states):
            raise Exception(
                "initial must be a starting state between 0 and %d "
                "(num_states - 1)." % (num_states - 1)
            )

        # Assemble the generator matrix: births on the super-diagonal, deaths
        # on the sub-diagonal, and each row's diagonal set so the row sums to 0.
        generator_matrix = np.zeros((num_states, num_states))
        for n in range(num_states):
            if n + 1 < num_states:
                generator_matrix[n, n + 1] = births[n]
            if n - 1 >= 0:
                generator_matrix[n, n - 1] = deaths[n]
            generator_matrix[n, n] = -(births[n] + deaths[n])

        initial_dist = [0.0] * num_states
        initial_dist[initial] = 1.0

        self.birth_rates = births
        self.death_rates = deaths
        self.num_states = num_states

        super().__init__(generator_matrix, initial_dist, state_labels)
