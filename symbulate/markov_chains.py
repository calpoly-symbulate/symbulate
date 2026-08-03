import numbers

import numpy as np

from .distributions import Exponential
from .index_sets import TimeInterval
from .math import inf
from .probability_space import ProbabilitySpace
from .random_variables import RV
from .result import (
    InfiniteVector,
    ContinuousTimeFunction,
    DiscreteValued,
    Vector,
    _BoundedTimeFunction,
)

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

    Three sequences describe the same path in a different way, and each is
    generated lazily, on demand:

    - ``states`` -- the states visited, in order, ignoring how long the
      chain stayed in each one,
    - ``interarrival_times`` -- how long the chain stayed in each of those
      states before jumping (also called the *holding* times),
    - ``get_arrival_times()`` -- the running total of those, i.e. the clock
      time of each jump.

    They line up index by index: the chain sits in ``states[n]`` for
    ``interarrival_times[n]`` units of time, and jumps out of it at
    ``get_arrival_times()[n]``. This is the same interface a
    :class:`~symbulate.poisson_process.PoissonProcess` sample path provides,
    and the ``states()``, ``interarrival_times()``, and ``arrival_times()``
    functions work on either one.

    Parameters
    ----------
    state_indices : MarkovChainResult
        The sequence of states visited, as state indices.
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
    states : InfiniteVector
        The sequence of states visited, as state *labels* -- the same values
        the path itself returns, so ``path(0)`` and ``states[0]`` agree.
    state_indices : MarkovChainResult
        The same sequence as state indices (``0, 1, ..., n_states - 1``),
        which is what the rates and labels are indexed by.
    rates : numpy.ndarray
        The rate of leaving each state (the negative of the diagonal
        of the generator matrix).
    times : InfiniteVector
        Unscaled interarrival times drawn from an Exponential(1) distribution.
    state_labels : range or list
        The names (labels) of each state.
    interarrival_times : InfiniteVector
        The actual time spent in each state visited, scaled by that state's
        holding rate. The time spent in a state whose rate of leaving is
        ``r`` is Exponential with rate ``r``, so it averages ``1 / r``.

    Examples
    --------
    >>> Q = [[-1, 1], [2, -2]]
    >>> chain = ContinuousTimeMarkovChain(Q, [1.0, 0.0], state_labels=['A', 'B'])
    >>> path = chain.draw()
    >>> path(2.5)  # doctest: +SKIP
    'A'
    >>> path.states[0]
    'A'
    >>> path.interarrival_times[0]  # doctest: +SKIP
    0.508
    >>> path.get_arrival_times()[0]  # doctest: +SKIP
    0.508
    """

    def __init__(self, state_indices, rates, unscaled_interarrival_times, state_labels):
        """Create one simulated sample path of a continuous-time Markov chain."""
        self.state_indices = state_indices
        self.rates = rates
        self.times = unscaled_interarrival_times
        self.state_labels = state_labels

        # The states as the user sees them: the labels, matching what
        # evaluating the path returns.
        self.states = InfiniteVector(lambda n: self.state_labels[self.state_indices[n]])

        # The time spent in the n-th state visited. Rescaling an
        # Exponential(1) draw by that state's rate of leaving gives an
        # Exponential(rate) holding time.
        self.interarrival_times = InfiniteVector(
            lambda n: self.times[n] / self.rates[self.state_indices[n]]
        )

        def _func(t):
            total_time = 0
            n = 0
            while True:
                total_time += self.interarrival_times[n]
                if total_time > t:
                    return self.states[n]
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
            state_indices = MarkovChain(
                self.transition_matrix, self.initial_dist
            ).draw()
            rates = -np.diag(self.generator_matrix)
            return ContinuousTimeMarkovChainResult(
                state_indices, rates, unscaled_interarrivals.draw(), self.state_labels
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

    Notes
    -----
    A sample path can also be described by *when* it jumps rather than by
    its value at each time. Three functions read that description off a
    drawn path, exactly as they do for a Poisson process:

    - ``states(path)`` -- the states visited, in order,
    - ``interarrival_times(path)`` -- how long the chain stayed in each of
      them (the holding times),
    - ``arrival_times(path)`` -- the clock time of each jump.

    The time the chain spends in a state is Exponential with that state's
    rate of leaving (the negative diagonal entry of the generator matrix),
    so from state ``i`` it averages ``-1 / generator_matrix[i][i]``. Because
    each of these is a random sequence, ``.apply()`` turns it into a random
    variable that can be simulated -- ``chain.apply(arrival_times)[0]`` is
    the (random) time of the first jump.

    Examples
    --------
    >>> from symbulate import *
    >>> Q = [[-1, 1], [2, -2]]
    >>> chain = ContinuousTimeMarkovChain(Q, [1.0, 0.0], state_labels=['A', 'B'])
    >>> path = chain.draw()
    >>> path(2.5)  # doctest: +SKIP
    'A'

    The states visited, the time spent in each, and the times of the jumps.

    >>> states(path)[0]
    'A'
    >>> interarrival_times(path)[0]  # doctest: +SKIP
    0.508
    >>> arrival_times(path)[0]  # doctest: +SKIP
    0.508

    The time of the first jump is a random variable, so it can be simulated.

    >>> chain.apply(arrival_times)[0].sim(1000).mean()  # doctest: +SKIP
    1.002
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

    Being a continuous-time Markov chain, a birth-death path can also be read
    off event by event with ``states``, ``interarrival_times``, and
    ``arrival_times`` -- the counts it passed through, how long it stayed at
    each of them, and the clock time of each birth or death. Nothing but a
    birth or a death can happen in state ``n``, and either one ends the stay,
    so the time spent at count ``n`` is Exponential with rate
    ``birth_rates[n] + death_rates[n]`` -- averaging
    ``1 / (birth_rates[n] + death_rates[n])``. A busier state is left sooner.

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

    An empty M/M/1 queue can only fill up, so the wait for the first event is
    Exponential(1) -- the first arrival -- and averages 1.

    >>> states(path)[0]
    0
    >>> queue.apply(interarrival_times)[0].sim(1000).mean()  # doctest: +SKIP
    0.988
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


def _require_positive(value, name):
    """Raise a friendly error unless ``value`` is a positive number."""
    if not isinstance(value, numbers.Real) or value <= 0:
        raise Exception("%s must be a positive number." % name)


def _require_positive_integer(value, name):
    """Raise a friendly error unless ``value`` is a positive whole number."""
    if not isinstance(value, numbers.Integral) or value <= 0:
        raise Exception("%s must be a positive integer." % name)


# --------------------------------------------------------------------------
# M/M/... queues
#
# Each of these is a birth-death process with particular arrival ("birth")
# and service ("death") rates, so they are thin wrappers over
# BirthDeathProcess. They are named in Kendall's notation: M/M/s/K/N means
# Markovian (Poisson) arrivals, Markovian (exponential) service, s servers,
# capacity K, and a calling population of N. The state is the number of
# customers in the system (waiting plus in service), and each draw is a
# sample path -- the number in the system as a function of continuous time.
# --------------------------------------------------------------------------


class MM1(BirthDeathProcess):
    """An M/M/1 queue: a single server, unlimited waiting room.

    Customers arrive as a Poisson process at rate ``arrival_rate`` and are
    served one at a time by a single server, each service taking an
    Exponential time with rate ``service_rate``. The state is the number of
    customers in the system. This is the canonical first queueing example.

    As a birth-death process the arrival (birth) rate is a constant
    ``arrival_rate`` in every state and the service (death) rate is a
    constant ``service_rate`` in every state ``n >= 1``.

    Parameters
    ----------
    arrival_rate : float
        The rate ``lambda`` at which customers arrive. Must be positive.
    service_rate : float
        The rate ``mu`` at which the server completes a customer. Must be
        positive. The queue is stable only when ``arrival_rate <
        service_rate``.
    num_states : int, optional
        Truncation of the (unbounded) state space for simulation: the states
        are ``0, 1, ..., num_states - 1``. Choose it large enough that the
        queue essentially never reaches the top. Default 100.

    Attributes
    ----------
    arrival_rate, service_rate : float
        The arrival and service rates.
    servers : int
        The number of servers (always 1 for this model).

    Examples
    --------
    >>> from symbulate import *
    >>> queue = MM1(arrival_rate=1, service_rate=1.5)
    >>> queue.draw()(5.0)  # doctest: +SKIP
    2
    """

    def __init__(self, arrival_rate, service_rate, num_states=100):
        """Initialize an M/M/1 queue."""
        _require_positive(arrival_rate, "arrival_rate")
        _require_positive(service_rate, "service_rate")
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.servers = 1
        super().__init__(
            birth_rates=arrival_rate,
            death_rates=service_rate,
            num_states=num_states,
        )


class MMs(BirthDeathProcess):
    """An M/M/s queue: ``s`` servers, unlimited waiting room.

    Like the :class:`MM1` queue but with ``servers`` identical servers, so up
    to ``servers`` customers are served at once. The service (death) rate is
    therefore state-dependent: with ``n`` customers in the system,
    ``min(n, servers)`` of them are being served, so the total service rate
    is ``min(n, servers) * service_rate``. Standard model for call centers,
    banks, and other multi-server service systems.

    Parameters
    ----------
    arrival_rate : float
        The rate ``lambda`` at which customers arrive. Must be positive.
    service_rate : float
        The rate ``mu`` at which each busy server completes a customer. Must
        be positive. The queue is stable only when
        ``arrival_rate < servers * service_rate``.
    servers : int
        The number of servers ``s``. Must be a positive integer.
    num_states : int, optional
        Truncation of the (unbounded) state space for simulation. Default 100.

    Attributes
    ----------
    arrival_rate, service_rate : float
        The arrival and per-server service rates.
    servers : int
        The number of servers.

    Examples
    --------
    >>> from symbulate import *
    >>> queue = MMs(arrival_rate=3, service_rate=2, servers=2)
    >>> queue.draw()(5.0)  # doctest: +SKIP
    1
    """

    def __init__(self, arrival_rate, service_rate, servers, num_states=100):
        """Initialize an M/M/s queue."""
        _require_positive(arrival_rate, "arrival_rate")
        _require_positive(service_rate, "service_rate")
        _require_positive_integer(servers, "servers")
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.servers = servers
        super().__init__(
            birth_rates=arrival_rate,
            death_rates=lambda n: min(n, servers) * service_rate,
            num_states=num_states,
        )


class MMsK(BirthDeathProcess):
    """An M/M/s/K queue: ``s`` servers and a finite capacity ``K``.

    Like the :class:`MMs` queue, but the system holds at most ``capacity``
    customers (those in service plus those waiting); a customer who arrives
    to a full system is turned away ("blocked"). Because the state space is
    genuinely finite (``0, 1, ..., capacity``), this is an *exact* chain, not
    a truncation -- there is no accuracy caveat. The long-run fraction of
    blocked customers is the standard quantity of interest.

    Parameters
    ----------
    arrival_rate : float
        The rate ``lambda`` at which customers arrive. Must be positive.
    service_rate : float
        The rate ``mu`` at which each busy server completes a customer. Must
        be positive.
    servers : int
        The number of servers ``s``. Must be a positive integer.
    capacity : int
        The maximum number of customers ``K`` allowed in the system. Must be
        a positive integer.

    Attributes
    ----------
    arrival_rate, service_rate : float
        The arrival and per-server service rates.
    servers : int
        The number of servers.
    capacity : int
        The system capacity ``K``.

    Examples
    --------
    >>> from symbulate import *
    >>> queue = MMsK(arrival_rate=4, service_rate=1, servers=2, capacity=5)
    >>> queue.draw()(5.0)  # doctest: +SKIP
    4
    """

    def __init__(self, arrival_rate, service_rate, servers, capacity):
        """Initialize an M/M/s/K queue."""
        _require_positive(arrival_rate, "arrival_rate")
        _require_positive(service_rate, "service_rate")
        _require_positive_integer(servers, "servers")
        _require_positive_integer(capacity, "capacity")
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.servers = servers
        self.capacity = capacity
        # States 0, 1, ..., capacity. The birth rate at the top state is
        # automatically 0 (BirthDeathProcess), which is exactly the blocking.
        super().__init__(
            birth_rates=arrival_rate,
            death_rates=lambda n: min(n, servers) * service_rate,
            num_states=capacity + 1,
        )


class MMss(MMsK):
    """An M/M/s/s queue (Erlang loss system): ``s`` servers, no waiting room.

    The special case of :class:`MMsK` with capacity equal to the number of
    servers, so there is no room to wait: a customer who arrives while all
    ``servers`` servers are busy is lost. This is the classic telecom
    trunking model, and the long-run fraction of lost customers is given by
    the Erlang B formula.

    Parameters
    ----------
    arrival_rate : float
        The rate ``lambda`` at which customers arrive. Must be positive.
    service_rate : float
        The rate ``mu`` at which each busy server completes a customer. Must
        be positive.
    servers : int
        The number of servers ``s`` (which is also the capacity). Must be a
        positive integer.

    Attributes
    ----------
    arrival_rate, service_rate : float
        The arrival and per-server service rates.
    servers : int
        The number of servers (and the capacity).
    capacity : int
        The system capacity, equal to ``servers``.

    Examples
    --------
    >>> from symbulate import *
    >>> loss = MMss(arrival_rate=3, service_rate=1, servers=3)
    >>> loss.draw()(5.0)  # doctest: +SKIP
    2
    """

    def __init__(self, arrival_rate, service_rate, servers):
        """Initialize an M/M/s/s (Erlang loss) queue."""
        super().__init__(arrival_rate, service_rate, servers, capacity=servers)


class MMsKN(BirthDeathProcess):
    """An M/M/s/K/N queue: finite capacity ``K`` and finite population ``N``.

    The finite-population ("machine repair" or "repairman") model. There are
    ``population`` customers (e.g. machines); each one that is *not* currently
    in the system generates an arrival (e.g. breaks down) at rate
    ``arrival_rate``, so with ``n`` customers already in the system the total
    arrival rate is ``(population - n) * arrival_rate`` -- it *slows down* as
    the system fills, because fewer customers remain outside. Up to ``servers``
    customers are served (repaired) at once, so the service rate is
    ``min(n, servers) * service_rate``. The system holds at most ``capacity``.

    Parameters
    ----------
    arrival_rate : float
        The rate ``lambda`` at which each customer *outside* the system
        generates an arrival. Must be positive.
    service_rate : float
        The rate ``mu`` at which each busy server completes a customer. Must
        be positive.
    servers : int
        The number of servers ``s``. Must be a positive integer.
    capacity : int
        The maximum number ``K`` of customers in the system. Must be a
        positive integer no larger than ``population``.
    population : int
        The total number ``N`` of customers in the calling population. Must
        be a positive integer. For the classic repairman problem, set
        ``capacity`` equal to ``population``.

    Attributes
    ----------
    arrival_rate, service_rate : float
        The per-customer arrival rate and per-server service rate.
    servers : int
        The number of servers.
    capacity : int
        The system capacity ``K``.
    population : int
        The calling-population size ``N``.

    Examples
    --------
    >>> from symbulate import *
    >>> repair = MMsKN(
    ...     arrival_rate=0.1, service_rate=1, servers=2, capacity=6, population=6
    ... )
    >>> repair.draw()(5.0)  # doctest: +SKIP
    1
    """

    def __init__(self, arrival_rate, service_rate, servers, capacity, population):
        """Initialize an M/M/s/K/N (machine-repair) queue."""
        _require_positive(arrival_rate, "arrival_rate")
        _require_positive(service_rate, "service_rate")
        _require_positive_integer(servers, "servers")
        _require_positive_integer(capacity, "capacity")
        _require_positive_integer(population, "population")
        if capacity > population:
            raise Exception(
                "capacity (%d) cannot exceed population (%d): the system "
                "cannot hold more customers than exist." % (capacity, population)
            )
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.servers = servers
        self.capacity = capacity
        self.population = population
        super().__init__(
            birth_rates=lambda n: (population - n) * arrival_rate,
            death_rates=lambda n: min(n, servers) * service_rate,
            num_states=capacity + 1,
        )


class MMInfinity(BirthDeathProcess):
    """An M/M/infinity queue: infinitely many servers (no waiting, ever).

    Every customer begins service immediately upon arrival, because there is
    always a free server. With ``n`` customers in the system, all ``n`` are
    in service at once, so the service (death) rate is ``n * service_rate``.
    Standard model for self-service systems and for transient or short-lived
    populations. Its long-run number in the system is Poisson with mean
    ``arrival_rate / service_rate``.

    Parameters
    ----------
    arrival_rate : float
        The rate ``lambda`` at which customers arrive. Must be positive.
    service_rate : float
        The rate ``mu`` at which each customer completes service. Must be
        positive.
    num_states : int, optional
        Truncation of the (unbounded) state space for simulation. Choose it
        comfortably above ``arrival_rate / service_rate`` so the top is
        essentially never reached. Default 100.

    Attributes
    ----------
    arrival_rate, service_rate : float
        The arrival and per-customer service rates.

    Examples
    --------
    >>> from symbulate import *
    >>> queue = MMInfinity(arrival_rate=5, service_rate=1)
    >>> queue.draw()(5.0)  # doctest: +SKIP
    6
    """

    def __init__(self, arrival_rate, service_rate, num_states=100):
        """Initialize an M/M/infinity queue."""
        _require_positive(arrival_rate, "arrival_rate")
        _require_positive(service_rate, "service_rate")
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        super().__init__(
            birth_rates=arrival_rate,
            death_rates=lambda n: n * service_rate,
            num_states=num_states,
        )


class YuleProcessResult(ContinuousTimeFunction, DiscreteValued):
    """One simulated sample path of a Yule (pure-birth) process.

    Stores enough to evaluate the population size at any time: the starting
    population, the per-individual birth rate, and a lazy sequence of
    Exponential(1) draws that set the (rescaled) holding times. New births
    are generated on demand as you evaluate the path at later and later
    times, so the path is unbounded -- there is no truncation.

    Parameters
    ----------
    initial : int
        The starting population size.
    birth_rate : float
        The rate at which each individual gives birth.
    unscaled_holding_times : InfiniteVector
        Exponential(1) random variables; the actual time spent at population
        ``initial + k`` is the ``k``-th of these divided by that population's
        total birth rate.

    Attributes
    ----------
    initial : int
        The starting population size.
    birth_rate : float
        The per-individual birth rate.
    states : InfiniteVector
        The population sizes passed through, in order: ``initial``,
        ``initial + 1``, ``initial + 2``, ... -- since nothing ever dies,
        the population climbs by one at every birth.
    interarrival_times : InfiniteVector
        The actual time between successive births.

    Examples
    --------
    >>> process = YuleProcess(birth_rate=0.5)
    >>> path = process.draw()
    >>> path(3.0)  # doctest: +SKIP
    4
    >>> path.states[0], path.states[1]
    (1, 2)
    """

    def __init__(self, initial, birth_rate, unscaled_holding_times):
        """Create one simulated sample path of a Yule process."""
        self.initial = initial
        self.birth_rate = birth_rate
        self.times = unscaled_holding_times

        # Nothing ever dies, so the population passes through every size from
        # ``initial`` upward, one birth at a time.
        self.states = InfiniteVector(lambda k: self.initial + k)

        # With ``m`` individuals present, each gives birth at ``birth_rate``,
        # so the next birth happens at total rate ``m * birth_rate``. The time
        # spent at population ``initial + k`` is therefore an Exponential(1)
        # draw divided by ``(initial + k) * birth_rate``.
        self.interarrival_times = InfiniteVector(
            lambda k: self.times[k] / ((self.initial + k) * self.birth_rate)
        )

        def _func(t):
            total_time = 0
            n = 0
            while True:
                total_time += self.interarrival_times[n]
                if total_time > t:
                    return self.states[n]
                n += 1

        super().__init__(_func)


class YuleProcessProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a Yule (pure-birth) process.

    Each draw from this space produces one simulated sample path of the
    process.

    Parameters
    ----------
    birth_rate : float
        The rate at which each individual gives birth. Must be positive.
    initial : int, optional
        The starting population size. Must be a positive integer. Default 1.

    Attributes
    ----------
    birth_rate : float
        The per-individual birth rate.
    initial : int
        The starting population size.

    Examples
    --------
    >>> space = YuleProcessProbabilitySpace(birth_rate=0.5)
    >>> path = space.draw()
    >>> path(0.0)
    1
    """

    def __init__(self, birth_rate, initial=1):
        """Initialize the probability space for a Yule process.

        Raises
        ------
        Exception
            If ``birth_rate`` is not a positive number, or ``initial`` is not
            a positive integer.
        """
        if not isinstance(birth_rate, numbers.Real) or birth_rate <= 0:
            raise Exception("birth_rate must be a positive number.")
        if not isinstance(initial, numbers.Integral) or initial < 1:
            raise Exception(
                "initial must be a positive integer: at least one individual "
                "is needed for a birth to happen."
            )
        self.birth_rate = birth_rate
        self.initial = initial

        # Built once; each `.draw()` on it yields a fresh, independent
        # sequence of Exponential(1) holding times.
        unscaled_holding_times = Exponential(1) ** inf

        def _draw():
            return YuleProcessResult(initial, birth_rate, unscaled_holding_times.draw())

        super().__init__(_draw)


class YuleProcess(RV):
    """A Yule process (pure birth), treated as a random variable.

    The Yule process -- also called the Yule-Furry or linear pure-birth
    process -- is a continuous-time model of a growing population. Each of the
    individuals currently alive independently gives birth at rate
    ``birth_rate``, and nothing ever dies, so with ``n`` individuals present
    the next birth arrives at total rate ``n * birth_rate``. It is one of the
    two canonical first examples in a stochastic-processes course (alongside
    the Poisson process) and the pure-birth ancestor of the birth-death
    chains used for queues.

    Each time you simulate this object you get a random sample path -- the
    population size as a (right-continuous, increasing) step function of
    continuous time. The path is generated lazily and is unbounded: because
    the population only ever grows, there is no state-space truncation.

    Parameters
    ----------
    birth_rate : float
        The rate at which each individual gives birth. Must be positive.
    initial : int, optional
        The starting population size. Must be a positive integer. Default 1.

    Attributes
    ----------
    birth_rate : float
        The per-individual birth rate.
    initial : int
        The starting population size.
    prob_space : YuleProcessProbabilitySpace
        The underlying probability space used to generate sample paths.

    Notes
    -----
    Starting from a single individual, the population size at time ``t`` is
    geometrically distributed with parameter ``exp(-birth_rate * t)``, and its
    mean grows exponentially: ``E[N(t)] = initial * exp(birth_rate * t)``.

    A drawn path can also be read off birth by birth with ``states``,
    ``interarrival_times``, and ``arrival_times``, as for a Poisson process
    or any continuous-time Markov chain. Here the wait for the next birth
    shrinks as the population grows: with ``n`` individuals present it is
    Exponential with rate ``n * birth_rate``, averaging
    ``1 / (n * birth_rate)``.

    Examples
    --------
    >>> from symbulate import *
    >>> process = YuleProcess(birth_rate=0.5)
    >>> path = process.draw()
    >>> path(0.0)
    1
    >>> path(5.0)  # doctest: +SKIP
    7

    The population climbs one at a time, and each wait is shorter than the last.

    >>> states(path)[0], states(path)[1]
    (1, 2)
    >>> arrival_times(path)[0]  # doctest: +SKIP
    0.693
    """

    def __init__(self, birth_rate, initial=1):
        """Initialize a Yule process."""
        self.birth_rate = birth_rate
        self.initial = initial
        super().__init__(YuleProcessProbabilitySpace(birth_rate, initial))


# --------------------------------------------------------------------------
# Compartmental epidemic models (SIR / SEIR)
#
# These are continuous-time Markov chains over compartment counts, but the
# state (S, I, R) or (S, E, I, R) is a vector rather than a single number,
# and an epidemic on a finite population always ends (once no one is
# infectious there are no more possible events -- an absorbing state, which
# the finite ContinuousTimeMarkovChain does not allow). So each path is
# simulated to completion with Gillespie's algorithm and stored, rather than
# built from a generator matrix. Each compartment is also exposed as its own
# scalar function of time for plotting the epidemic curves.
# --------------------------------------------------------------------------


class _EpidemicResult(ContinuousTimeFunction):
    """Base class for one simulated sample path of a compartmental model.

    A subclass sets ``self.compartments`` (the labels, e.g. ``["S", "I",
    "R"]``) and implements ``_transition_rates(counts)``, returning the list
    of possible events as ``(rate, change)`` pairs. This class runs Gillespie's
    algorithm to simulate the whole trajectory to completion (which is finite,
    since a finite-population epidemic always ends), stores it, and evaluates
    the state at any time by lookup. Each compartment is exposed as its own
    scalar :class:`ContinuousTimeFunction` -- e.g. ``path.I`` -- for plotting
    and analysis, and calling the path itself returns the whole state vector.

    Attributes
    ----------
    event_times : list of float
        The time of each event (``event_times[0]`` is 0, the start).
    states : list of tuple
        The compartment counts after each event; ``states[0]`` is the initial
        state.
    """

    def __init__(self, initial_counts):
        """Simulate the whole sample path with Gillespie's algorithm."""
        counts = list(initial_counts)
        self.event_times = [0.0]
        self.states = [tuple(counts)]
        t = 0.0
        while True:
            events = self._transition_rates(counts)
            total_rate = sum(rate for rate, _ in events)
            # Once no event is possible the epidemic is over (absorbing state).
            if total_rate <= 0:
                break
            t += rng.exponential(1.0 / total_rate)
            # Choose which event occurs, with probability proportional to rate.
            threshold = rng.random() * total_rate
            cumulative = 0.0
            for rate, change in events:
                cumulative += rate
                if threshold < cumulative:
                    counts = [c + d for c, d in zip(counts, change)]
                    break
            self.event_times.append(t)
            self.states.append(tuple(counts))

        super().__init__(self._state_at)

        # Expose each compartment as its own scalar function of time, so the
        # epidemic curves can be plotted one compartment at a time. Give each a
        # bounded domain that ends when the epidemic does, so plotting a curve
        # defaults its time axis to the length of the outbreak rather than the
        # generic 0 to 10.
        end = self.event_times[-1]
        for i, label in enumerate(self.compartments):
            curve = _BoundedTimeFunction(lambda t, i=i: self(t)[i])
            if end > 0:
                curve.index_set = TimeInterval(0, end)
            setattr(self, label, curve)

    def _state_at(self, t):
        """Return the compartment counts at time ``t`` as a Vector."""
        k = int(np.searchsorted(self.event_times, t, side="right")) - 1
        return Vector(self.states[max(k, 0)])

    def _transition_rates(self, counts):
        """Return a list of ``(rate, change)`` pairs for the current counts.

        ``change`` is a tuple added to the counts when that event fires. A
        subclass must implement this.
        """
        raise NotImplementedError


class SIRResult(_EpidemicResult):
    """One simulated sample path of an SIR epidemic (S, I, R counts)."""

    def __init__(self, initial_counts, infection_rate, recovery_rate, population):
        """Create one SIR sample path."""
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.population = population
        self.compartments = ["S", "I", "R"]
        super().__init__(initial_counts)

    def _transition_rates(self, counts):
        """Infection (S -> I) at rate beta*S*I/N, recovery (I -> R) at gamma*I."""
        S, I, R = counts
        infection = self.infection_rate * S * I / self.population
        recovery = self.recovery_rate * I
        return [(infection, (-1, 1, 0)), (recovery, (0, -1, 1))]


class SIRProbabilitySpace(ProbabilitySpace):
    """The probability space underlying an SIR epidemic model.

    Each draw produces one simulated sample path.

    Parameters
    ----------
    population : int
        The total population size ``N``. Must be a positive integer.
    infection_rate : float
        The rate ``beta``: a susceptible and an infectious individual meet and
        transmit at rate ``beta / N``, so infections happen at total rate
        ``beta * S * I / N``. Must be positive.
    recovery_rate : float
        The rate ``gamma`` at which each infectious individual recovers. Must
        be positive.
    initial_infected : int, optional
        The number infectious at time 0. Must be a positive integer. Default 1.
    initial_recovered : int, optional
        The number already recovered (immune) at time 0. Must be a
        non-negative integer. Default 0.
    """

    def __init__(
        self,
        population,
        infection_rate,
        recovery_rate,
        initial_infected=1,
        initial_recovered=0,
    ):
        """Initialize the probability space for an SIR epidemic."""
        _require_positive_integer(population, "population")
        _require_positive(infection_rate, "infection_rate")
        _require_positive(recovery_rate, "recovery_rate")
        _require_positive_integer(initial_infected, "initial_infected")
        if not isinstance(initial_recovered, numbers.Integral) or initial_recovered < 0:
            raise Exception("initial_recovered must be a non-negative integer.")
        if initial_infected + initial_recovered > population:
            raise Exception(
                "initial_infected + initial_recovered cannot exceed the " "population."
            )
        self.population = population
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered

        initial_counts = [
            population - initial_infected - initial_recovered,
            initial_infected,
            initial_recovered,
        ]

        def _draw():
            return SIRResult(initial_counts, infection_rate, recovery_rate, population)

        super().__init__(_draw)


class SIR(RV):
    """An SIR epidemic model, treated as a random variable.

    The standard stochastic epidemic on a closed population of ``population``
    individuals, each of whom is **S**usceptible, **I**nfectious, or
    **R**ecovered (immune). A susceptible individual becomes infectious
    through contact with an infectious one, and an infectious individual
    eventually recovers; recovered individuals never become susceptible
    again. Two things happen at random:

    - **Infection** ``S -> I`` at total rate ``infection_rate * S * I /
      population`` (more susceptibles and more infectives means faster spread),
    - **Recovery** ``I -> R`` at total rate ``recovery_rate * I``.

    Each draw is a sample path: the counts ``(S, I, R)`` as a function of
    continuous time, simulated until the epidemic ends (no one infectious
    left). Calling the path at a time returns the ``(S, I, R)`` vector; each
    compartment is also available on its own as ``path.S``, ``path.I``, and
    ``path.R`` for plotting the epidemic curves.

    Parameters
    ----------
    population : int
        The total population size ``N``. Must be a positive integer.
    infection_rate : float
        The infection rate ``beta`` (see above). Must be positive. The basic
        reproduction number is ``R0 = infection_rate / recovery_rate``: above 1
        a large outbreak is possible, below 1 the epidemic dies out quickly.
    recovery_rate : float
        The recovery rate ``gamma``. Must be positive.
    initial_infected : int, optional
        The number infectious at time 0. Must be a positive integer. Default 1.
    initial_recovered : int, optional
        The number already immune at time 0. Must be a non-negative integer.
        Default 0.

    Attributes
    ----------
    population, infection_rate, recovery_rate : int or float
        The model parameters.
    initial_infected, initial_recovered : int
        The initial counts.
    prob_space : SIRProbabilitySpace
        The underlying probability space used to generate sample paths.

    Examples
    --------
    >>> from symbulate import *
    >>> epidemic = SIR(population=1000, infection_rate=0.3, recovery_rate=0.1,
    ...                initial_infected=5)
    >>> path = epidemic.draw()
    >>> list(path(0.0))   # (S, I, R) at the start
    [995, 5, 0]
    >>> path.I(20.0)      # number infectious at time 20  # doctest: +SKIP
    287
    """

    def __init__(
        self,
        population,
        infection_rate,
        recovery_rate,
        initial_infected=1,
        initial_recovered=0,
    ):
        """Initialize an SIR epidemic model."""
        self.population = population
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.initial_recovered = initial_recovered
        super().__init__(
            SIRProbabilitySpace(
                population,
                infection_rate,
                recovery_rate,
                initial_infected,
                initial_recovered,
            )
        )


class SEIRResult(_EpidemicResult):
    """One simulated sample path of an SEIR epidemic (S, E, I, R counts)."""

    def __init__(
        self,
        initial_counts,
        infection_rate,
        incubation_rate,
        recovery_rate,
        population,
    ):
        """Create one SEIR sample path."""
        self.infection_rate = infection_rate
        self.incubation_rate = incubation_rate
        self.recovery_rate = recovery_rate
        self.population = population
        self.compartments = ["S", "E", "I", "R"]
        super().__init__(initial_counts)

    def _transition_rates(self, counts):
        """S -> E at beta*S*I/N, E -> I at sigma*E, I -> R at gamma*I."""
        S, E, I, R = counts
        infection = self.infection_rate * S * I / self.population
        incubation = self.incubation_rate * E
        recovery = self.recovery_rate * I
        return [
            (infection, (-1, 1, 0, 0)),
            (incubation, (0, -1, 1, 0)),
            (recovery, (0, 0, -1, 1)),
        ]


class SEIRProbabilitySpace(ProbabilitySpace):
    """The probability space underlying an SEIR epidemic model.

    Each draw produces one simulated sample path.

    Parameters
    ----------
    population : int
        The total population size ``N``. Must be a positive integer.
    infection_rate : float
        The rate ``beta`` at which contact produces exposures ``S -> E`` (total
        rate ``beta * S * I / N``). Must be positive.
    incubation_rate : float
        The rate ``sigma`` at which each exposed individual becomes infectious
        ``E -> I``. Must be positive. Its reciprocal is the mean latent period.
    recovery_rate : float
        The rate ``gamma`` at which each infectious individual recovers
        ``I -> R``. Must be positive.
    initial_infected : int, optional
        The number infectious at time 0. Must be a positive integer. Default 1.
    initial_exposed : int, optional
        The number exposed (infected but not yet infectious) at time 0. Must be
        a non-negative integer. Default 0.
    initial_recovered : int, optional
        The number already immune at time 0. Must be a non-negative integer.
        Default 0.
    """

    def __init__(
        self,
        population,
        infection_rate,
        incubation_rate,
        recovery_rate,
        initial_infected=1,
        initial_exposed=0,
        initial_recovered=0,
    ):
        """Initialize the probability space for an SEIR epidemic."""
        _require_positive_integer(population, "population")
        _require_positive(infection_rate, "infection_rate")
        _require_positive(incubation_rate, "incubation_rate")
        _require_positive(recovery_rate, "recovery_rate")
        _require_positive_integer(initial_infected, "initial_infected")
        for name, value in [
            ("initial_exposed", initial_exposed),
            ("initial_recovered", initial_recovered),
        ]:
            if not isinstance(value, numbers.Integral) or value < 0:
                raise Exception("%s must be a non-negative integer." % name)
        if initial_infected + initial_exposed + initial_recovered > population:
            raise Exception(
                "initial_infected + initial_exposed + initial_recovered cannot "
                "exceed the population."
            )
        self.population = population
        self.infection_rate = infection_rate
        self.incubation_rate = incubation_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.initial_exposed = initial_exposed
        self.initial_recovered = initial_recovered

        initial_counts = [
            population - initial_infected - initial_exposed - initial_recovered,
            initial_exposed,
            initial_infected,
            initial_recovered,
        ]

        def _draw():
            return SEIRResult(
                initial_counts,
                infection_rate,
                incubation_rate,
                recovery_rate,
                population,
            )

        super().__init__(_draw)


class SEIR(RV):
    """An SEIR epidemic model, treated as a random variable.

    Like the :class:`SIR` model, but with an added **E**xposed compartment for
    individuals who have been infected but are not yet infectious (a latent
    period). A susceptible first becomes exposed, then infectious, then
    recovered:

    - **Exposure** ``S -> E`` at total rate ``infection_rate * S * I /
      population`` (only *infectious* individuals expose others),
    - **Onset** ``E -> I`` at total rate ``incubation_rate * E``,
    - **Recovery** ``I -> R`` at total rate ``recovery_rate * I``.

    Each draw is a sample path: the counts ``(S, E, I, R)`` as a function of
    continuous time, simulated until the epidemic ends (no one exposed or
    infectious left). Each compartment is available as ``path.S``, ``path.E``,
    ``path.I``, and ``path.R`` for plotting the epidemic curves.

    Parameters
    ----------
    population : int
        The total population size ``N``. Must be a positive integer.
    infection_rate : float
        The exposure rate ``beta``. Must be positive.
    incubation_rate : float
        The rate ``sigma`` of becoming infectious (``1 / sigma`` is the mean
        latent period). Must be positive.
    recovery_rate : float
        The recovery rate ``gamma``. Must be positive.
    initial_infected : int, optional
        The number infectious at time 0. Must be a positive integer. Default 1.
    initial_exposed : int, optional
        The number exposed at time 0. Must be a non-negative integer. Default 0.
    initial_recovered : int, optional
        The number already immune at time 0. Must be a non-negative integer.
        Default 0.

    Attributes
    ----------
    population, infection_rate, incubation_rate, recovery_rate : int or float
        The model parameters.
    initial_infected, initial_exposed, initial_recovered : int
        The initial counts.
    prob_space : SEIRProbabilitySpace
        The underlying probability space used to generate sample paths.

    Examples
    --------
    >>> from symbulate import *
    >>> epidemic = SEIR(population=1000, infection_rate=0.4, incubation_rate=0.2,
    ...                 recovery_rate=0.1, initial_infected=5)
    >>> path = epidemic.draw()
    >>> list(path(0.0))   # (S, E, I, R) at the start
    [995, 0, 5, 0]
    """

    def __init__(
        self,
        population,
        infection_rate,
        incubation_rate,
        recovery_rate,
        initial_infected=1,
        initial_exposed=0,
        initial_recovered=0,
    ):
        """Initialize an SEIR epidemic model."""
        self.population = population
        self.infection_rate = infection_rate
        self.incubation_rate = incubation_rate
        self.recovery_rate = recovery_rate
        self.initial_infected = initial_infected
        self.initial_exposed = initial_exposed
        self.initial_recovered = initial_recovered
        super().__init__(
            SEIRProbabilitySpace(
                population,
                infection_rate,
                incubation_rate,
                recovery_rate,
                initial_infected,
                initial_exposed,
                initial_recovered,
            )
        )
