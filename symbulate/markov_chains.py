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
