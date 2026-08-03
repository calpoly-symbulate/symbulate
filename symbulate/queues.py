"""Queues whose service times are not exponential (the G/G/1 family).

The Markovian queues -- ``M/M/1``, ``M/M/s``, and their finite-capacity and
finite-population relatives -- live in ``markov_chains.py``, because the
number of customers in such a system is a birth-death continuous-time Markov
chain. Once service times stop being exponential that is no longer true: how
much longer the customer in service will take depends on how long they have
already been there, so the number in the system is not a Markov chain and no
generator matrix describes it.

What is still simple is the *waiting time* of each customer in turn. Lindley's
recursion gives it exactly, one customer at a time, for any interarrival and
service distributions -- which is what this module implements. With several
servers the same idea works on the times at which each of them next comes
free, so ``GGs`` joins ``GG1`` here rather than needing machinery of its own.
"""

import heapq
import numbers

import numpy as np

from .distributions import Distribution, Exponential, MultivariateDistribution
from .math import inf
from .probability_space import ProbabilitySpace
from .random_variables import RV
from .renewal_process import (
    RenewalProcessResult,
    _is_always_zero,
    _smallest_possible_time,
)
from .result import InfiniteVector


def _validate_interarrival_dist(interarrival_dist):
    """Check that a distribution can serve as a queue's time between arrivals.

    The same requirement a renewal process makes of its interarrival times --
    a single nonnegative number per draw, not always 0 -- since the arrival
    stream of a ``G/G/1`` queue *is* a renewal process. The nonnegativity
    bound is read from the underlying scipy distribution's support (via
    ``renewal_process._smallest_possible_time``), so a newly added
    distribution is checked with no per-distribution code.

    Parameters
    ----------
    interarrival_dist : Distribution
        The candidate distribution of time between customer arrivals.

    Raises
    ------
    TypeError
        If ``interarrival_dist`` is not a Symbulate ``Distribution``, or is a
        multivariate one (a whole vector per draw rather than one time).
    ValueError
        If the distribution can produce a negative time, or produces a time
        of 0 on every draw.
    """
    if not isinstance(interarrival_dist, Distribution):
        message = (
            "interarrival_dist must be a Symbulate distribution describing the "
            "time between customer arrivals, such as Exponential(rate=1), "
            "Gamma(shape=2, rate=1), or Uniform(a=0, b=2). You gave "
            f"{type(interarrival_dist).__name__}."
        )
        if isinstance(interarrival_dist, (int, float)):
            message += (
                f" (If you meant customers arriving at rate "
                f"{interarrival_dist}, the times between them are "
                f"Exponential(rate={interarrival_dist}) -- so use "
                f"MG1(arrival_rate={interarrival_dist}, service_dist=...) or "
                f"GG1(Exponential(rate={interarrival_dist}), ...).)"
            )
        raise TypeError(message)

    name = type(interarrival_dist).__name__

    if isinstance(interarrival_dist, MultivariateDistribution):
        raise TypeError(
            "interarrival_dist must be a distribution of single numbers, but "
            f"{name} produces a whole vector of numbers on each draw. A queue "
            "needs one waiting time between one customer and the next. Try a "
            "one-number distribution such as Exponential(rate=1) or "
            "Gamma(shape=2, rate=1)."
        )

    smallest = _smallest_possible_time(interarrival_dist)
    if smallest is not None and smallest < 0:
        raise ValueError(
            f"interarrival_dist must never produce a negative time, but {name} "
            f"can produce values as small as {smallest:g}. It is the time "
            "between one arrival and the next, so a negative value would have "
            "a customer arrive before the customer ahead of them. Try a "
            "distribution that is never negative, such as Exponential(rate=1), "
            "Gamma(shape=2, rate=1), LogNormal(0, 1), or Uniform(a=0, b=2)."
        )

    if _is_always_zero(interarrival_dist):
        raise ValueError(
            f"interarrival_dist must sometimes produce a positive time, but "
            f"{name} produces a time of 0 on every draw, so every customer "
            "would arrive at the same instant and the queue would never empty. "
            "Choose a distribution with an average time between arrivals "
            "greater than 0, such as Exponential(rate=1) or "
            "Gamma(shape=2, rate=1)."
        )


def _validate_service_dist(service_dist):
    """Check that a distribution can serve as a queue's service time.

    Like the interarrival check, but a service time is allowed to be 0 on
    every draw: a server that finishes instantly is a degenerate model, not an
    impossible one, and no customer ever waits in it.

    Parameters
    ----------
    service_dist : Distribution
        The candidate distribution of the time to serve one customer.

    Raises
    ------
    TypeError
        If ``service_dist`` is not a Symbulate ``Distribution``, or is a
        multivariate one.
    ValueError
        If the distribution can produce a negative service time.
    """
    if not isinstance(service_dist, Distribution):
        message = (
            "service_dist must be a Symbulate distribution describing how long "
            "the server takes with one customer, such as Exponential(rate=2), "
            "Gamma(shape=2, rate=4), or Uniform(a=0, b=1). You gave "
            f"{type(service_dist).__name__}."
        )
        if isinstance(service_dist, (int, float)):
            message += (
                f" (If you meant the server completing customers at rate "
                f"{service_dist}, each service takes an "
                f"Exponential(rate={service_dist}) time -- so use "
                f"GM1(interarrival_dist=..., service_rate={service_dist}) or "
                f"GG1(..., Exponential(rate={service_dist})).)"
            )
        raise TypeError(message)

    name = type(service_dist).__name__

    if isinstance(service_dist, MultivariateDistribution):
        raise TypeError(
            "service_dist must be a distribution of single numbers, but "
            f"{name} produces a whole vector of numbers on each draw. A queue "
            "needs one service time per customer. Try a one-number "
            "distribution such as Exponential(rate=2) or "
            "Gamma(shape=2, rate=4)."
        )

    smallest = _smallest_possible_time(service_dist)
    if smallest is not None and smallest < 0:
        raise ValueError(
            f"service_dist must never produce a negative time, but {name} can "
            f"produce values as small as {smallest:g}. It is how long the "
            "server takes with one customer, so it cannot be negative. Try a "
            "distribution that is never negative, such as Exponential(rate=2), "
            "Gamma(shape=2, rate=4), LogNormal(0, 1), or Uniform(a=0, b=1)."
        )


def _validate_rate(rate, name, meaning):
    """Check that a rate given in place of a distribution is a positive number.

    Parameters
    ----------
    rate : float
        The candidate rate.
    name : str
        The parameter name, used in the error message.
    meaning : str
        A phrase completing "It is ...", explaining what the rate counts.

    Raises
    ------
    TypeError
        If ``rate`` is not a number.
    ValueError
        If ``rate`` is not positive.
    """
    if isinstance(rate, bool) or not isinstance(rate, numbers.Real):
        raise TypeError(
            f"{name} must be a number, got {type(rate).__name__}. It is "
            f"{meaning}, for example {name}=2."
        )
    if rate <= 0:
        raise ValueError(
            f"{name} must be a positive number, got {rate}. It is {meaning}, "
            f"so it has to be greater than 0."
        )


def _validate_servers(servers):
    """Check that a queue's server count is a positive whole number.

    Parameters
    ----------
    servers : int
        The candidate number of servers.

    Raises
    ------
    TypeError
        If ``servers`` is not a whole number.
    ValueError
        If ``servers`` is not at least 1.
    """
    if isinstance(servers, bool) or not isinstance(servers, numbers.Integral):
        raise TypeError(
            f"servers must be a whole number, got {type(servers).__name__}. It "
            "is how many customers can be served at once, for example "
            "servers=3. (There is no such thing as half a server -- if you "
            "meant a server that works at some other speed, change "
            "service_dist instead.)"
        )
    if servers < 1:
        raise ValueError(
            f"servers must be at least 1, got {servers}. A queue with no "
            "server would never serve anybody, so the line would grow forever."
        )


def _traffic_intensity(interarrival_dist, service_dist, servers=1):
    """The traffic intensity (utilization) of a queue, or ``None`` if unknown.

    The traffic intensity ``rho`` is the average service time divided by the
    average time between arrivals, shared out over the servers -- the long-run
    fraction of time a server is busy. It decides whether the queue settles
    down: when ``rho < 1`` the waiting times have a steady-state distribution,
    and when ``rho >= 1`` work arrives at least as fast as the servers can
    clear it, so the queue grows without bound.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of time between arrivals.
    service_dist : Distribution
        The distribution of one service time.
    servers : int, optional
        How many customers can be served at once. Default 1.

    Returns
    -------
    float or None
        The traffic intensity, or ``None`` when either average is unavailable
        (some distributions have no mean, and a point mass written as
        ``Uniform(a=2, b=2)`` reports a nan one).
    """
    try:
        mean_interarrival = float(interarrival_dist.mean())
        mean_service = float(service_dist.mean())
    except (AttributeError, TypeError, ValueError):
        return None
    if not (np.isfinite(mean_interarrival) and np.isfinite(mean_service)):
        return None
    if mean_interarrival == 0:
        return None
    return mean_service / (servers * mean_interarrival)


class _QueueResult(InfiniteVector):
    """Shared machinery for one sample path of a single-line queue.

    The path is the sequence of *waiting times*: ``path[n]`` is how long
    customer ``n`` stands in line before their service begins, not counting
    the service itself. This class holds the two i.i.d. input sequences, the
    waiting times worked out so far, and the sequences derived from them --
    everything except the recursion that turns arrivals and services into
    waits, which is what distinguishes one queue from another and is supplied
    by a subclass's ``_wait_at``.

    Parameters
    ----------
    interarrival_times : InfiniteVector or iterable of float
        The times between successive arrivals. Entry ``n`` is the gap between
        customer ``n - 1`` and customer ``n``; entry 0 is the time until
        customer 0 arrives.
    service_times : InfiniteVector or iterable of float
        The service times. Entry ``n`` is how long a server takes with
        customer ``n``.

    Attributes
    ----------
    interarrival_times : InfiniteVector or iterable of float
        The times between successive arrivals.
    service_times : InfiniteVector or iterable of float
        The service times.
    arrival_times : InfiniteVector
        When each customer arrives -- the running total of the interarrival
        times.
    sojourn_times : InfiniteVector
        How long each customer spends in the system altogether: waiting plus
        being served, ``W[n] + S[n]``.
    departure_times : InfiniteVector
        When each customer leaves, ``arrival_times[n] + sojourn_times[n]``.
    waits : list
        The waiting times generated so far.

    See Also
    --------
    GG1Result : One server, via Lindley's recursion.
    GGsResult : Several servers, via their next-free times.
    """

    def __init__(self, interarrival_times, service_times):
        """Set up one sample path from its arrival and service sequences."""
        self.interarrival_times = interarrival_times
        self.service_times = service_times
        # waits[n] is customer n's waiting time. Customers are served only as
        # far into the queue as has actually been asked about, and each wait is
        # kept once worked out -- the same lazily-extending, self-caching
        # pattern RandomWalkResult uses for its positions.
        self.waits = []

        super().__init__(self._wait_at)

        # Derived sequences, each generated on demand from the two input
        # sequences and the waiting times above.
        self.arrival_times = InfiniteVector(
            lambda n: float(sum(self.interarrival_times[i] for i in range(n + 1)))
        )
        self.sojourn_times = InfiniteVector(
            lambda n: float(self[n] + self.service_times[n])
        )
        self.departure_times = InfiniteVector(
            lambda n: float(self.arrival_times[n] + self.sojourn_times[n])
        )

    def _wait_at(self, n):
        """Return customer ``n``'s waiting time, serving that far if needed."""
        raise NotImplementedError(
            "A queue path has to say how a customer's wait is worked out. Use "
            "GG1Result (one server) or GGsResult (several servers)."
        )

    def get_waiting_times(self):
        """Return the waiting times of this path.

        The path *is* its waiting times, so this returns the path itself. It
        exists to be read alongside the other sequences.

        Returns
        -------
        _QueueResult
            This path, whose entry ``n`` is customer ``n``'s waiting time.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 3, [1.5] * 3)
        >>> path.get_waiting_times()[1]
        0.5
        """
        return self

    def get_service_times(self):
        """Return the service times of this path.

        Returns
        -------
        InfiniteVector or iterable of float
            Entry ``n`` is how long a server takes with customer ``n``.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 3, [1.5] * 3)
        >>> path.get_service_times()[0]
        1.5
        """
        return self.service_times

    def get_interarrival_times(self):
        """Return the times between successive arrivals.

        Returns
        -------
        InfiniteVector or iterable of float
            Entry ``n`` is the gap between customer ``n - 1``'s arrival and
            customer ``n``'s.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 3, [1.5] * 3)
        >>> path.get_interarrival_times()[2]
        1.0
        """
        return self.interarrival_times

    def get_arrival_times(self):
        """Return the times at which the customers arrive.

        Returns
        -------
        InfiniteVector
            Entry ``n`` is the clock time at which customer ``n`` arrives.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 3, [1.5] * 3)
        >>> path.get_arrival_times()[2]
        3.0
        """
        return self.arrival_times

    def get_sojourn_times(self):
        """Return how long each customer spends in the system.

        The sojourn time (also called the *time in system*, or the *response
        time*) is the waiting time plus the service time.

        Returns
        -------
        InfiniteVector
            Entry ``n`` is customer ``n``'s waiting time plus service time.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 3, [1.5] * 3)
        >>> path.get_sojourn_times()[1]  # waited 0.5, then served for 1.5
        2.0
        """
        return self.sojourn_times

    def get_departure_times(self):
        """Return the times at which the customers leave.

        Returns
        -------
        InfiniteVector
            Entry ``n`` is the clock time at which customer ``n``'s service
            finishes.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 3, [1.5] * 3)
        >>> path.get_departure_times()[1]  # arrived at 2.0, left 2.0 later
        4.0
        """
        return self.departure_times

    def get_arrival_process(self):
        """Return the arrival stream as a count over continuous time.

        The customers arrive according to a renewal process, so the same
        interarrival times this path was built from also describe a counting
        function: evaluate the returned path at a time ``t`` to get how many
        customers had arrived by then.

        Returns
        -------
        RenewalProcessResult
            The arrivals as a step function of continuous time.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 5, [1.5] * 5)
        >>> arrivals = path.get_arrival_process()
        >>> arrivals(2.5)  # customers 0 and 1 arrived, at times 1.0 and 2.0
        2
        """
        return RenewalProcessResult(self.interarrival_times)


class GG1Result(_QueueResult):
    """One simulated sample path of a single-server queue.

    The path is the sequence of *waiting times*: ``path[n]`` is how long
    customer ``n`` stands in line before their service begins, not counting
    the service itself. Customers are numbered ``0, 1, 2, ...`` in the order
    they arrive, and customer 0 arrives to an empty system, so ``path[0]`` is
    always 0.

    Each waiting time is worked out from the one before it by **Lindley's
    recursion**::

        W[n + 1] = max(W[n] + S[n] - A[n + 1], 0)

    In words: when customer ``n + 1`` shows up, customer ``n`` still has
    ``W[n] + S[n]`` worth of waiting and service to finish, of which
    ``A[n + 1]`` has already elapsed during the gap between the two arrivals.
    Whatever is left over is the new customer's wait -- and if the leftover is
    negative the server went idle in between, so the new customer walks
    straight up and waits 0.

    Waiting times are generated on demand as you index further into the path,
    and cached once generated, so reading ``path[100]`` and then ``path[10]``
    describes one single queue rather than two unrelated ones.

    Parameters
    ----------
    interarrival_times : InfiniteVector or iterable of float
        The times between successive arrivals. Entry ``n`` is the gap between
        customer ``n - 1`` and customer ``n``; entry 0 is the time until
        customer 0 arrives.
    service_times : InfiniteVector or iterable of float
        The service times. Entry ``n`` is how long the server takes with
        customer ``n``.

    Attributes
    ----------
    interarrival_times : InfiniteVector or iterable of float
        The times between successive arrivals.
    service_times : InfiniteVector or iterable of float
        The service times.
    arrival_times : InfiniteVector
        When each customer arrives -- the running total of the interarrival
        times.
    sojourn_times : InfiniteVector
        How long each customer spends in the system altogether: waiting plus
        being served, ``W[n] + S[n]``.
    departure_times : InfiniteVector
        When each customer leaves, ``arrival_times[n] + sojourn_times[n]``.
    waits : list
        The waiting times generated so far, starting with customer 0's wait
        of 0.

    Examples
    --------
    >>> from symbulate import *
    >>> # Customers arrive every 1 unit of time; each service takes 1.5.
    >>> path = GG1Result([1.0] * 5, [1.5] * 5)
    >>> path[0]  # the first customer walks straight up to the server
    0.0
    >>> path[1]  # arrives 1.0 later, but the service takes 1.5
    0.5
    >>> path[3]  # the server falls half a unit further behind each customer
    1.5

    The queue this path came from is usually the more convenient way in.

    >>> queue = GG1(Exponential(rate=1), Exponential(rate=2))
    >>> queue.draw()[10]  # doctest: +SKIP
    0.317

    See Also
    --------
    GG1 : The queue these paths are drawn from.
    RenewalProcessResult : The arrival stream on its own, as a count over time.
    """

    def _wait_at(self, n):
        """Serve up to customer ``n`` by Lindley's recursion, and return
        their wait."""
        while len(self.waits) <= n:
            i = len(self.waits)
            if i == 0:
                # Customer 0 arrives to an empty system.
                wait = 0.0
            else:
                # What is left of customer i-1's wait and service when
                # customer i arrives. A plain float is stored rather than
                # whatever numpy type the draws came back as, so a student
                # reading a stretch of the path sees times, not type names.
                wait = float(
                    max(
                        self.waits[i - 1]
                        + self.service_times[i - 1]
                        - self.interarrival_times[i],
                        0.0,
                    )
                )
            self.waits.append(wait)
        return self.waits[n]


class GG1ProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a single-server queue.

    Each draw from this space produces one simulated sample path of the
    queue -- the waiting times of customer 0, 1, 2, ... in turn.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of the (i.i.d.) time between customer arrivals.
    service_dist : Distribution
        The distribution of the (i.i.d.) time to serve one customer.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.
    service_dist : Distribution
        The service-time distribution.

    Raises
    ------
    TypeError
        If either argument is not a Symbulate ``Distribution`` over single
        numbers.
    ValueError
        If either distribution can produce a negative time, or if
        ``interarrival_dist`` produces a time of 0 on every draw.

    Examples
    --------
    >>> from symbulate import *
    >>> space = GG1ProbabilitySpace(Exponential(rate=1), Exponential(rate=2))
    >>> space.draw()[5]  # doctest: +SKIP
    0.204
    """

    def __init__(self, interarrival_dist, service_dist):
        """Create a probability space for a single-server queue."""
        _validate_interarrival_dist(interarrival_dist)
        _validate_service_dist(service_dist)
        self.interarrival_dist = interarrival_dist
        self.service_dist = service_dist

        # Build the two i.i.d. sequences once, not once per draw: neither
        # distribution ever changes, and each `.draw()` on them already
        # yields a fresh, independent sequence. (Same reasoning as
        # RenewalProcess and PoissonProcess.)
        interarrivals = self.interarrival_dist**inf
        services = self.service_dist**inf

        def draw():
            return GG1Result(interarrivals.draw(), services.draw())

        super().__init__(draw)


class GG1(RV):
    """A G/G/1 queue: one server, any arrival and service distributions.

    Customers arrive one after another, wait their turn in a single line, and
    are served one at a time, first come first served. ``X[n]`` is how long
    customer ``n`` waits in line before their service begins.

    The name is `Kendall's notation <https://en.wikipedia.org/wiki/Kendall%27s_notation>`_
    ``G/G/1``: **G**\\ eneral (any) distribution of the time between arrivals,
    **G**\\ eneral distribution of the service times, and **1** server. It is
    the realistic queueing model -- real service times are rarely exponential
    -- and it is famous for having *no* closed-form answer for the average
    wait, which is exactly why simulating it is the standard way to study it.

    Each waiting time follows from the one before by **Lindley's
    recursion** -- see :class:`GG1Result` for what it says and why.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of the time between one arrival and the next. Must be
        a Symbulate ``Distribution`` over single nonnegative numbers (e.g.
        ``Exponential(rate=1)``, ``Gamma(shape=2, rate=1)``,
        ``Uniform(a=0, b=2)``). The arrivals therefore form a
        :class:`~symbulate.renewal_process.RenewalProcess`.
    service_dist : Distribution
        The distribution of the time the server takes with one customer. Same
        requirements, except that a service time of 0 is allowed.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.
    service_dist : Distribution
        The service-time distribution.
    utilization : float or None
        The traffic intensity ``rho`` -- the average service time divided by
        the average time between arrivals, which is the long-run fraction of
        time the server is busy. ``None`` when either average is unavailable.
    prob_space : GG1ProbabilitySpace
        The underlying probability space used to generate sample paths.

    Raises
    ------
    TypeError
        If either argument is not a Symbulate ``Distribution`` over single
        numbers.
    ValueError
        If either distribution can produce a negative time, or if
        ``interarrival_dist`` produces a time of 0 on every draw.

    Notes
    -----
    Customers are numbered ``0, 1, 2, ...`` in arrival order, so time here is
    counted in *customers*, not in clock time: ``X[10]`` is the eleventh
    customer's wait, whenever they happen to show up. This is the one real
    difference from :class:`~symbulate.markov_chains.MM1`, which tracks the
    number of customers in the system at each moment of continuous time. Both
    describe the same queue when service is exponential; they just report
    different things about it.

    Whether the queue settles down is decided by ``utilization``. When
    ``utilization < 1`` the waiting times have a steady-state distribution,
    and ``X[n]`` for a large ``n`` is a draw from (very nearly) that
    distribution. When ``utilization >= 1`` work arrives at least as fast as
    the server can clear it, so the line grows without bound and ``X[n]``
    keeps getting larger -- worth simulating deliberately, but not a queue
    with a long-run average wait to estimate.

    A drawn path carries the rest of the story with it. Alongside the waiting
    times it reports ``service_times``, ``interarrival_times``,
    ``arrival_times``, ``sojourn_times`` (waiting plus service, i.e. the total
    time in the system), and ``departure_times``. Wrapping one of those in
    ``.apply()`` turns it into a random variable that can be simulated like
    any other -- see the examples.

    Examples
    --------
    >>> from symbulate import *
    >>> # Arrivals on average 1 apart, services on average 0.5 long.
    >>> queue = GG1(Exponential(rate=1), Uniform(a=0, b=1))
    >>> queue.utilization
    0.5
    >>> queue[20].sim(1000).mean()  # the 21st customer's average wait  # doctest: +SKIP
    0.334

    Any pair of nonnegative distributions works, including ones with no
    exponential anywhere in sight.

    >>> lumpy = GG1(Gamma(shape=2, rate=2), LogNormal(-1.5, 0.75))
    >>> lumpy.draw()[30]  # doctest: +SKIP
    0.212

    The total time in the system -- waiting plus being served -- is one
    ``.apply()`` away.

    >>> in_system = queue.apply(lambda path: path.sojourn_times[20])
    >>> in_system.sim(1000).mean()  # doctest: +SKIP
    0.833

    See Also
    --------
    MG1 : Poisson arrivals and general service (the M/G/1 queue).
    GM1 : General arrivals and exponential service (the G/M/1 queue).
    MM1 : The same queue when both are exponential, tracked as a queue length
        over continuous time instead.
    RenewalProcess : The arrival stream on its own.
    """

    def __init__(self, interarrival_dist, service_dist):
        """Create a G/G/1 queue."""
        prob_space = GG1ProbabilitySpace(interarrival_dist, service_dist)
        self.interarrival_dist = prob_space.interarrival_dist
        self.service_dist = prob_space.service_dist
        self.utilization = _traffic_intensity(self.interarrival_dist, self.service_dist)
        super().__init__(prob_space)


class MG1(GG1):
    """An M/G/1 queue: Poisson arrivals, any service distribution.

    The special case of :class:`GG1` in which customers arrive as a Poisson
    process -- so the times between arrivals are Exponential with rate
    ``arrival_rate`` -- while the service times may be anything. This is the
    classic model for the common situation where arrivals are unpredictable
    but services are not exponential at all, and it is the one member of the
    family with a famous formula: the **Pollaczek-Khinchine** formula gives
    the long-run average wait exactly, so a simulation can be checked against
    it.

    Parameters
    ----------
    arrival_rate : float
        The rate ``lambda`` at which customers arrive -- the average number
        arriving per unit of time. Must be positive.
    service_dist : Distribution
        The distribution of the time the server takes with one customer. Must
        be a Symbulate ``Distribution`` over single nonnegative numbers.

    Attributes
    ----------
    arrival_rate : float
        The arrival rate ``lambda``.
    interarrival_dist : Exponential
        The interarrival-time distribution built from ``arrival_rate``.
    service_dist : Distribution
        The service-time distribution.
    utilization : float or None
        The traffic intensity ``rho``, here ``arrival_rate`` times the average
        service time.

    Raises
    ------
    TypeError
        If ``arrival_rate`` is not a number, or ``service_dist`` is not a
        Symbulate ``Distribution`` over single numbers.
    ValueError
        If ``arrival_rate`` is not positive, or ``service_dist`` can produce a
        negative time.

    Notes
    -----
    The Pollaczek-Khinchine formula says the long-run average wait in line is
    ``arrival_rate * E[S^2] / (2 * (1 - utilization))``, where ``S`` is one
    service time. Notice that it depends on ``E[S^2]``, not just on the
    average service time: two servers that take the same time on average
    produce different queues, and the more erratic one produces the longer
    line. That is the standard lesson of this model, and it is easy to watch
    happen by simulating two ``MG1`` queues with equal ``utilization``. A
    server that takes exactly the same time on every customer (the ``D``, for
    deterministic, of ``M/D/1``) is the extreme case: ``E[S^2]`` is then as
    small as it can be for that average, and the line is as short as it can be.

    Examples
    --------
    >>> from symbulate import *
    >>> # Arrivals at rate 1; services averaging 0.5, barely varying.
    >>> steady = MG1(arrival_rate=1, service_dist=Gamma(shape=20, rate=40))
    >>> # The same average service time, but erratic.
    >>> erratic = MG1(arrival_rate=1, service_dist=Exponential(rate=2))
    >>> steady.utilization == erratic.utilization  # the same load either way
    True
    >>> steady[50].sim(1000).mean()  # doctest: +SKIP
    0.267
    >>> erratic[50].sim(1000).mean()  # nearly twice the wait  # doctest: +SKIP
    0.494

    See Also
    --------
    GG1 : The general single-server queue this specializes.
    GM1 : The other half of the relaxation -- general arrivals, exponential
        service.
    """

    def __init__(self, arrival_rate, service_dist):
        """Create an M/G/1 queue."""
        _validate_rate(
            arrival_rate,
            "arrival_rate",
            "the average number of customers arriving per unit of time",
        )
        self.arrival_rate = arrival_rate
        super().__init__(Exponential(rate=arrival_rate), service_dist)


class GM1(GG1):
    """A G/M/1 queue: any arrival distribution, exponential service.

    The special case of :class:`GG1` in which each service takes an
    Exponential time with rate ``service_rate``, while the times between
    arrivals may be anything. It is taught alongside :class:`MG1` as the other
    half of the relaxation of ``M/M/1``: one model keeps Poisson arrivals and
    frees the service times, this one keeps exponential service and frees the
    arrivals.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of the time between one arrival and the next. Must be
        a Symbulate ``Distribution`` over single nonnegative numbers.
    service_rate : float
        The rate ``mu`` at which the server completes customers -- the average
        number it could finish per unit of time if it never went idle. Must be
        positive.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.
    service_rate : float
        The service rate ``mu``.
    service_dist : Exponential
        The service-time distribution built from ``service_rate``.
    utilization : float or None
        The traffic intensity ``rho``, here the average time between arrivals
        divided into ``1 / service_rate``.

    Raises
    ------
    TypeError
        If ``interarrival_dist`` is not a Symbulate ``Distribution`` over
        single numbers, or ``service_rate`` is not a number.
    ValueError
        If ``interarrival_dist`` can produce a negative time or produces a
        time of 0 on every draw, or if ``service_rate`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> # Arrivals almost, but not quite, regular; exponential service.
    >>> queue = GM1(interarrival_dist=Gamma(shape=8, rate=8), service_rate=2)
    >>> round(queue.utilization, 3)
    0.5
    >>> queue[50].sim(1000).mean()  # doctest: +SKIP
    0.221

    Steadier arrivals mean shorter waits at the same load: an M/M/1 queue
    under the same utilization waits longer.

    >>> MM1_wait = GM1(interarrival_dist=Exponential(rate=1), service_rate=2)
    >>> MM1_wait[50].sim(1000).mean()  # doctest: +SKIP
    0.503

    See Also
    --------
    GG1 : The general single-server queue this specializes.
    MG1 : The other half of the relaxation -- Poisson arrivals, general
        service.
    """

    def __init__(self, interarrival_dist, service_rate):
        """Create a G/M/1 queue."""
        _validate_rate(
            service_rate,
            "service_rate",
            "the average number of customers the server could finish per unit "
            "of time",
        )
        self.service_rate = service_rate
        super().__init__(interarrival_dist, Exponential(rate=service_rate))


class GGsResult(_QueueResult):
    """One simulated sample path of a queue with several servers.

    Like :class:`GG1Result`, ``path[n]`` is how long customer ``n`` waits in
    line before their service begins. The difference is what they are waiting
    for: with ``servers`` servers sharing one line, a customer waits only until
    the *first* of them comes free, so nobody waits at all while any server is
    idle.

    That is all the recursion needs to track -- the times at which each server
    next comes free. Customer ``n`` arrives at ``A[n]``, takes the server that
    frees up soonest, and waits::

        W[n] = max(earliest free time - A[n], 0)

    and that server is then busy until ``A[n] + W[n] + S[n]``. It is the
    multi-server generalization of Lindley's recursion (the vector recursion of
    Kiefer and Wolfowitz), and with ``servers=1`` it *is* Lindley's recursion,
    written in clock time instead of in leftovers.

    Waiting times are generated on demand as you index further into the path,
    and cached once generated, so reading ``path[100]`` and then ``path[10]``
    describes one single queue rather than two unrelated ones.

    Parameters
    ----------
    interarrival_times : InfiniteVector or iterable of float
        The times between successive arrivals. Entry ``n`` is the gap between
        customer ``n - 1`` and customer ``n``; entry 0 is the time until
        customer 0 arrives.
    service_times : InfiniteVector or iterable of float
        The service times. Entry ``n`` is how long a server takes with
        customer ``n``.
    servers : int
        How many customers can be served at once.

    Attributes
    ----------
    servers : int
        How many customers can be served at once.

    Notes
    -----
    Customers leave in the order they were *served*, which with more than one
    server need not be the order they arrived: a customer with a short service
    can overtake a slower one being served beside them. So
    ``departure_times`` is not necessarily increasing, unlike in the
    single-server case.

    Examples
    --------
    >>> from symbulate import *
    >>> # A customer every 1 unit of time, each service taking 1.5, 2 servers.
    >>> path = GGsResult([1.0] * 6, [1.5] * 6, servers=2)
    >>> [path[n] for n in range(5)]  # two servers keep up; nobody waits
    [0.0, 0.0, 0.0, 0.0, 0.0]
    >>> # The same queue with one server falls behind (Lindley's recursion).
    >>> [GGsResult([1.0] * 6, [1.5] * 6, servers=1)[n] for n in range(5)]
    [0.0, 0.5, 1.0, 1.5, 2.0]

    See Also
    --------
    GGs : The queue these paths are drawn from.
    GG1Result : The single-server path, via Lindley's recursion.
    """

    def __init__(self, interarrival_times, service_times, servers):
        """Create one simulated sample path of a queue with several servers."""
        self.servers = servers
        # When each server next comes free. They all start free at time 0, and
        # heapq keeps the earliest of them at position 0 -- the only one the
        # recursion ever asks about.
        self._free_times = [0.0] * servers
        # The arrival time of the last customer served so far, kept as a
        # running total so the recursion does not re-add the interarrival times
        # from the start on every customer.
        self._arrival_clock = 0.0
        super().__init__(interarrival_times, service_times)

    def _wait_at(self, n):
        """Serve up to customer ``n``, and return how long they waited."""
        while len(self.waits) <= n:
            i = len(self.waits)
            self._arrival_clock += self.interarrival_times[i]
            # Whatever is left of the wait for the soonest free server. A plain
            # float is stored rather than whatever numpy type the draws came
            # back as, so a student reading a stretch of the path sees times,
            # not type names.
            wait = float(max(self._free_times[0] - self._arrival_clock, 0.0))
            # That server takes this customer, and is busy until they leave.
            heapq.heapreplace(
                self._free_times,
                self._arrival_clock + wait + self.service_times[i],
            )
            self.waits.append(wait)
        return self.waits[n]


class GGsProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a queue with several servers.

    Each draw from this space produces one simulated sample path of the
    queue -- the waiting times of customer 0, 1, 2, ... in turn.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of the (i.i.d.) time between customer arrivals.
    service_dist : Distribution
        The distribution of the (i.i.d.) time to serve one customer.
    servers : int
        How many customers can be served at once. Must be at least 1.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.
    service_dist : Distribution
        The service-time distribution.
    servers : int
        How many customers can be served at once.

    Raises
    ------
    TypeError
        If either distribution is not a Symbulate ``Distribution`` over single
        numbers, or ``servers`` is not a whole number.
    ValueError
        If either distribution can produce a negative time, if
        ``interarrival_dist`` produces a time of 0 on every draw, or if
        ``servers`` is less than 1.

    Examples
    --------
    >>> from symbulate import *
    >>> space = GGsProbabilitySpace(Exponential(rate=1), Exponential(rate=1), 2)
    >>> space.draw()[5]  # doctest: +SKIP
    0.118
    """

    def __init__(self, interarrival_dist, service_dist, servers):
        """Create a probability space for a queue with several servers."""
        _validate_interarrival_dist(interarrival_dist)
        _validate_service_dist(service_dist)
        _validate_servers(servers)
        self.interarrival_dist = interarrival_dist
        self.service_dist = service_dist
        self.servers = servers

        # Build the two i.i.d. sequences once, not once per draw: neither
        # distribution ever changes, and each `.draw()` on them already
        # yields a fresh, independent sequence.
        interarrivals = self.interarrival_dist**inf
        services = self.service_dist**inf

        def draw():
            return GGsResult(interarrivals.draw(), services.draw(), self.servers)

        super().__init__(draw)


class GGs(RV):
    """A G/G/s queue: ``s`` servers sharing one line, any distributions.

    Customers arrive one after another and join a single line, and the customer
    at its head goes to whichever of the ``servers`` servers comes free first.
    ``X[n]`` is how long customer ``n`` waits in line before their service
    begins.

    This is the multi-server generalization of :class:`GG1` -- the bank with
    several tellers and one queue, the call center with several agents -- and
    the realistic version of :class:`~symbulate.markov_chains.MMs`, since it
    does not need the service times to be exponential. Waiting times come from
    the times at which the servers next come free; see :class:`GGsResult`.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of the time between one arrival and the next. Must be
        a Symbulate ``Distribution`` over single nonnegative numbers (e.g.
        ``Exponential(rate=1)``, ``Gamma(shape=2, rate=1)``,
        ``Uniform(a=0, b=2)``).
    service_dist : Distribution
        The distribution of the time one server takes with one customer. Same
        requirements, except that a service time of 0 is allowed. Every server
        is identical: they all serve from this one distribution.
    servers : int
        How many customers can be served at once. Must be at least 1.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.
    service_dist : Distribution
        The service-time distribution.
    servers : int
        How many customers can be served at once.
    utilization : float or None
        The traffic intensity ``rho`` -- the average service time divided by
        the average time between arrivals, shared out over the servers, which
        is the long-run fraction of time a server is busy. ``None`` when either
        average is unavailable.
    prob_space : GGsProbabilitySpace
        The underlying probability space used to generate sample paths.

    Raises
    ------
    TypeError
        If either distribution is not a Symbulate ``Distribution`` over single
        numbers, or ``servers`` is not a whole number.
    ValueError
        If either distribution can produce a negative time, if
        ``interarrival_dist`` produces a time of 0 on every draw, or if
        ``servers`` is less than 1.

    Notes
    -----
    Adding servers divides the load: ``utilization`` is
    ``mean service / (servers * mean interarrival)``, so a queue that is
    unstable with one server can be perfectly calm with two. Stability is
    again ``utilization < 1``.

    Two servers of a given speed are not the same as one server twice as fast,
    and a course usually asks students to compare them. The pair of slow
    servers keeps the *line* shorter, because a customer can be served while
    the other server is stuck on a long job; the single fast server gets each
    customer out of the *system* sooner, because the service itself is half as
    long. Simulating both settles which matters for a given question.

    ``M/G/s`` and ``G/M/s`` are not separate classes -- pass an ``Exponential``
    on whichever side is Markovian, exactly as ``GG1(Exponential(...), ...)``
    is the ``M/G/1`` queue.

    Examples
    --------
    >>> from symbulate import *
    >>> # Three tellers, a customer every 2 minutes, services averaging 5.
    >>> bank = GGs(Exponential(rate=1 / 2), Gamma(shape=2, rate=2 / 5), servers=3)
    >>> round(bank.utilization, 3)
    0.833
    >>> bank[50].sim(1000).mean()  # doctest: +SKIP
    5.42

    A second server can rescue a queue that one server cannot keep up with.

    >>> GGs(Exponential(rate=1), Exponential(rate=0.8), servers=1).utilization
    1.25
    >>> GGs(Exponential(rate=1), Exponential(rate=0.8), servers=2).utilization
    0.625

    See Also
    --------
    GG1 : The single-server queue, whose recursion this generalizes.
    MMs : The same queue when both distributions are exponential, tracked as a
        queue length over continuous time instead.
    """

    def __init__(self, interarrival_dist, service_dist, servers):
        """Create a G/G/s queue."""
        prob_space = GGsProbabilitySpace(interarrival_dist, service_dist, servers)
        self.interarrival_dist = prob_space.interarrival_dist
        self.service_dist = prob_space.service_dist
        self.servers = prob_space.servers
        self.utilization = _traffic_intensity(
            self.interarrival_dist, self.service_dist, self.servers
        )
        super().__init__(prob_space)
