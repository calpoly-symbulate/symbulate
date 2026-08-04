"""Queues whose service times are not exponential (``GG1`` and ``GGs``).

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
from .result import ContinuousTimeFunction, DiscreteValued, InfiniteVector


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


class _QueueResult(ContinuousTimeFunction, DiscreteValued):
    """Shared machinery for one sample path of a single-line queue.

    The path is the **number of customers in the system** as a function of
    continuous time: ``path(t)`` is how many are there at time ``t``, waiting
    plus in service. It steps up by one at each arrival and down by one at each
    departure, exactly like an
    :class:`~symbulate.markov_chains.MM1` path, and starts at 0 because the
    system is empty before the first customer shows up.

    Underneath, the count is worked out from *when* each customer arrives and
    leaves, and departures depend on how long each customer waited. That wait
    is what distinguishes one queue discipline from another, so it is supplied
    by a subclass's ``_wait_at``; everything downstream of it -- the customer
    times, the merged event stream, and the count itself -- lives here.

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
    states : InfiniteVector
        The counts the path passes through, in order, ignoring how long it
        stays at each: ``0, 1, 2, 1, ...``. The same meaning ``states`` has for
        a continuous-time Markov chain.
    interarrival_times : InfiniteVector
        How long the count stays at each of those values before the next event
        -- again as for a continuous-time Markov chain. Note that an *event* is
        an arrival **or** a departure, so these are not the gaps between
        customer arrivals; for those, difference ``customer_arrival_times``.
    waiting_times : InfiniteVector
        How long each customer waits in line before being served, not counting
        their own service.
    service_times : InfiniteVector or iterable of float
        How long a server takes with each customer.
    sojourn_times : InfiniteVector
        How long each customer spends in the system altogether: waiting plus
        being served.
    customer_arrival_times : InfiniteVector
        When each customer arrives.
    customer_departure_times : InfiniteVector
        When each customer leaves.
    waits : list
        The waiting times worked out so far.

    See Also
    --------
    GG1Result : One server, via Lindley's recursion.
    GGsResult : Several servers, via their next-free times.
    """

    def __init__(self, interarrival_times, service_times):
        """Set up one sample path from its arrival and service sequences."""
        self._arrival_gaps = interarrival_times
        self.service_times = service_times
        # waits[n] is customer n's waiting time. Customers are served only as
        # far into the queue as has actually been asked about, and each wait is
        # kept once worked out -- the same lazily-extending, self-caching
        # pattern RandomWalkResult uses for its positions.
        self.waits = []

        # The customer-level view. Each arrival time is the one before it plus
        # the next gap, rather than a fresh sum from customer 0: an
        # InfiniteVector fills its cache in order, so entry n - 1 is always
        # ready by the time entry n is worked out. Re-summing instead makes
        # reading a long stretch of the path cost a multiple of its length --
        # 3.3 seconds out to customer 2000, against 0.007 seconds like this.
        self.customer_arrival_times = InfiniteVector(
            lambda n: float(
                self._arrival_gaps[0]
                if n == 0
                else self.customer_arrival_times[n - 1] + self._arrival_gaps[n]
            )
        )
        self.waiting_times = InfiniteVector(self._wait_at)
        self.sojourn_times = InfiniteVector(
            lambda n: float(self.waiting_times[n] + self.service_times[n])
        )
        self.customer_departure_times = InfiniteVector(
            lambda n: float(self.customer_arrival_times[n] + self.sojourn_times[n])
        )

        # The event-level view, built by merging the two customer streams: the
        # times the count changes, and what it changes to. Both are generated
        # on demand and kept, like everything else here.
        self._event_times = []
        self._counts = []
        # Customers who have arrived but not yet left, by when they will leave;
        # heapq keeps the soonest departure at position 0. With one server that
        # is simply the next customer, but with several a short service can
        # finish before a long one that started earlier.
        self._in_system = []
        self._next_arrival = 0

        self.states = InfiniteVector(self._count_after_event)
        self.interarrival_times = InfiniteVector(self._holding_time_at)

        super().__init__(self._count_at)

    def _wait_at(self, n):
        """Return customer ``n``'s waiting time, serving that far if needed."""
        raise NotImplementedError(
            "A queue path has to say how a customer's wait is worked out. Use "
            "GG1Result (one server) or GGsResult (several servers)."
        )

    def _next_arrival_time(self):
        """Return when the next customer arrives, or ``None`` if none will.

        A path drawn from a queue has an endless stream of arrivals. One built
        by hand from a list of times runs out of them, and after that only the
        customers already in the system are left to leave.
        """
        try:
            return self.customer_arrival_times[self._next_arrival]
        except IndexError:
            return None

    def _extend_events(self, count):
        """Work out events until ``count`` of them are known, or none are left.

        One event is one change in the number in the system. The next one is
        whichever comes first: the next customer arriving, or the soonest
        departure among those already in the system.
        """
        while len(self._event_times) < count:
            arrival = self._next_arrival_time()
            if arrival is None and not self._in_system:
                # No arrivals left and nobody still being served, so the count
                # stays where it is for good.
                return
            if self._in_system and (arrival is None or self._in_system[0] < arrival):
                # A departure comes first, so the count goes down. (A tie goes
                # to the arrival: the count steps up and back down at the same
                # instant, which leaves the value at that instant unchanged.)
                self._event_times.append(heapq.heappop(self._in_system))
                self._counts.append(self._counts[-1] - 1)
            else:
                heapq.heappush(
                    self._in_system,
                    self.customer_departure_times[self._next_arrival],
                )
                self._next_arrival += 1
                self._event_times.append(arrival)
                self._counts.append((self._counts[-1] if self._counts else 0) + 1)

    def _count_at(self, t):
        """Return the number of customers in the system at time ``t``."""
        known = 0
        while True:
            self._extend_events(known + 1)
            if len(self._event_times) <= known or self._event_times[known] > t:
                break
            known += 1
        # The count set by the last event at or before t, or 0 if the first
        # customer has not arrived yet.
        return self._counts[known - 1] if known else 0

    def _count_after_event(self, n):
        """Return the ``n``-th value the count takes, ignoring durations."""
        if n == 0:
            # Every path starts empty, before any event has happened.
            return 0
        self._extend_events(n)
        if not self._counts:
            return 0
        # Asking past the last event of a hand-built path: nothing more happens.
        return self._counts[min(n, len(self._counts)) - 1]

    def _holding_time_at(self, n):
        """Return how long the count stays at its ``n``-th value."""
        self._extend_events(n + 1)
        if len(self._event_times) <= n:
            # Nothing further happens, so the count stays there for good.
            return inf
        if n == 0:
            return float(self._event_times[0])
        return float(self._event_times[n] - self._event_times[n - 1])

    def get_arrival_times(self):
        """Return the times at which the count changes.

        The event times: when the number in the system steps up (an arrival) or
        down (a departure). This is what ``arrival_times(path)`` gives for a
        continuous-time Markov chain too -- the times of the jumps -- so the
        name is the package's, not this module's. For the times at which
        *customers* arrive, read ``customer_arrival_times``.

        Returns
        -------
        InfiniteVector
            Entry ``n`` is the clock time of the ``n``-th change in the count.

        Examples
        --------
        >>> from symbulate import *
        >>> # A customer every 1.0, each served in 1.5: arrival, arrival,
        >>> # departure, ... as the single server falls behind.
        >>> path = GG1Result([1.0] * 5, [1.5] * 5)
        >>> [path.get_arrival_times()[n] for n in range(3)]
        [1.0, 2.0, 2.5]
        """
        return InfiniteVector(self._event_time_at)

    def _event_time_at(self, n):
        """Return the clock time of the ``n``-th change in the count."""
        self._extend_events(n + 1)
        if len(self._event_times) <= n:
            # A hand-built path with no events left: the change never comes.
            return inf
        return float(self._event_times[n])

    def get_number_waiting(self):
        """Return the number of customers *waiting* over continuous time.

        The path itself counts everybody in the system, including the ones
        being served. This counts only those still in line, which is the number
        in the system less however many servers are busy.

        Returns
        -------
        ContinuousTimeFunction
            The number of customers waiting at each time.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GG1Result([1.0] * 5, [1.5] * 5)
        >>> waiting = path.get_number_waiting()
        >>> path(3.2), waiting(3.2)  # one of the two is being served
        (2, 1)
        """
        servers = getattr(self, "servers", 1)
        return ContinuousTimeFunction(lambda t: max(self(t) - servers, 0))

    def get_arrival_process(self):
        """Return the arrival stream as a count over continuous time.

        The customers arrive according to a renewal process, so the gaps this
        path was built from also describe a counting function: evaluate the
        returned path at a time ``t`` to get how many customers had arrived by
        then -- however many of them have since left.

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
        return RenewalProcessResult(self._arrival_gaps)


class GG1Result(_QueueResult):
    """One simulated sample path of a single-server queue.

    ``path(t)`` is the number of customers in the system at time ``t`` -- the
    one being served plus everybody in line. It starts at 0, steps up at each
    arrival, and steps down at each departure.

    Getting the departures right is the whole problem, and for one server
    **Lindley's recursion** solves it: customer ``n``'s waiting time follows
    from the customer before them by ::

        W[n + 1] = max(W[n] + S[n] - A[n + 1], 0)

    In words: when customer ``n + 1`` shows up, customer ``n`` still has
    ``W[n] + S[n]`` worth of waiting and service to finish, of which
    ``A[n + 1]`` has already elapsed during the gap between the two arrivals.
    Whatever is left over is the new customer's wait -- and if the leftover is
    negative the server went idle in between, so the new customer walks
    straight up and waits 0. Customer ``n`` then leaves at
    ``A[0] + ... + A[n] + W[n] + S[n]``, which is what the count needs.

    The path is worked out only as far as it has been asked about, and kept
    once worked out, so evaluating it at ``t = 100`` and then at ``t = 10``
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

    Examples
    --------
    >>> from symbulate import *
    >>> # A customer every 1 unit of time; each service takes 1.5, so the
    >>> # single server falls steadily behind and the line builds up.
    >>> path = GG1Result([1.0] * 8, [1.5] * 8)
    >>> [path(t) for t in [0.5, 1.0, 3.0, 6.0]]
    [0, 1, 2, 3]

    The waiting times the count was built from are still there to read, and
    ``waiting_times[0]`` is always 0 -- customer 0 arrives to an empty system.

    >>> [path.waiting_times[n] for n in range(4)]
    [0.0, 0.5, 1.0, 1.5]
    >>> path.customer_departure_times[0]  # arrived at 1.0, served for 1.5
    2.5

    See Also
    --------
    GG1 : The queue these paths are drawn from.
    GGsResult : The same path with several servers.
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
                        - self._arrival_gaps[i],
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
    >>> space.draw()(5.0)  # doctest: +SKIP
    2
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
    are served one at a time, first come first served. ``X[t]`` is the number of
    customers in the system at time ``t`` -- the one being served plus everybody
    still in line -- exactly what :class:`~symbulate.markov_chains.MM1` reports,
    but for any pair of distributions rather than exponential ones only.

    The name is `Kendall's notation <https://en.wikipedia.org/wiki/Kendall%27s_notation>`_
    ``G/G/1``: **G**\\ eneral (any) distribution of the time between arrivals,
    **G**\\ eneral distribution of the service times, and **1** server. It is
    the realistic queueing model -- real service times are rarely exponential
    -- and it is famous for having *no* closed-form answer for the average
    wait, which is exactly why simulating it is the standard way to study it.

    The count steps up at each arrival and down at each departure, and the
    departures are found from **Lindley's recursion** on the waiting times --
    see :class:`GG1Result` for what it says and why.

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
    Time is clock time, as for every other continuous-time process in the
    package: ``X[10]`` is how many customers are in the system at time 10, and
    a drawn path can be evaluated at any time you like. Because the count only
    ever changes by one, a path also reads event by event with ``states``,
    ``interarrival_times``, and ``arrival_times`` -- the counts it passed
    through, how long it stayed at each, and when each change happened, the same
    three views a continuous-time Markov chain offers. Note that an *event* here
    is an arrival **or** a departure, so ``interarrival_times`` is not the gaps
    between customer arrivals.

    Whether the queue settles down is decided by ``utilization``. When
    ``utilization < 1`` the number in the system has a steady-state
    distribution, and ``X[t]`` for a large ``t`` is a draw from (very nearly)
    that distribution. When ``utilization >= 1`` work arrives at least as fast
    as the server can clear it, so the line grows without bound -- worth
    simulating deliberately, but not a queue with a long-run average to
    estimate.

    A drawn path also carries the customer-level story the count was built
    from: ``waiting_times``, ``service_times``, ``sojourn_times`` (waiting plus
    service, i.e. the total time in the system), ``customer_arrival_times``, and
    ``customer_departure_times``. Wrapping one of those in ``.apply()`` turns it
    into a random variable that can be simulated like any other -- see the
    examples. ``get_number_waiting()`` gives the number still in *line* over
    continuous time, which is the count less the customer in service.

    Examples
    --------
    >>> from symbulate import *
    >>> # Arrivals on average 1 apart, services on average 0.5 long.
    >>> queue = GG1(Exponential(rate=1), Uniform(a=0, b=1))
    >>> queue.utilization
    0.5
    >>> queue[20].sim(1000).mean()  # the average number in the system at t=20  # doctest: +SKIP
    0.98

    Any pair of nonnegative distributions works, including ones with no
    exponential anywhere in sight.

    >>> lumpy = GG1(Gamma(shape=2, rate=2), LogNormal(-1.5, 0.75))
    >>> lumpy.draw()(30.0)  # doctest: +SKIP
    1

    The waiting time of a particular customer is one ``.apply()`` away.

    >>> wait = queue.apply(lambda path: path.waiting_times[20])
    >>> wait.sim(1000).mean()  # doctest: +SKIP
    0.334

    The total time in the system -- waiting plus being served -- likewise.

    >>> in_system = queue.apply(lambda path: path.sojourn_times[20])
    >>> in_system.sim(1000).mean()  # doctest: +SKIP
    0.833

    See Also
    --------
    MG1 : Poisson arrivals and general service (the M/G/1 queue).
    GM1 : General arrivals and exponential service (the G/M/1 queue).
    GGs : The same queue with several servers sharing one line.
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

    The formula is about the *wait*, which a drawn path reports as
    ``waiting_times`` alongside the number in the system the path itself gives.

    Examples
    --------
    >>> from symbulate import *
    >>> # Arrivals at rate 1; services averaging 0.5, barely varying.
    >>> steady = MG1(arrival_rate=1, service_dist=Gamma(shape=20, rate=40))
    >>> # The same average service time, but erratic.
    >>> erratic = MG1(arrival_rate=1, service_dist=Exponential(rate=2))
    >>> steady.utilization == erratic.utilization  # the same load either way
    True
    >>> steady[50].sim(1000).mean()  # average number in the system  # doctest: +SKIP
    0.77
    >>> erratic[50].sim(1000).mean()  # a longer line on the same load  # doctest: +SKIP
    1.01

    The waits behind those numbers are what the formula predicts.

    >>> wait = erratic.apply(lambda path: path.waiting_times[50])
    >>> wait.sim(1000).mean()  # P-K says 1 * 0.5 / (2 * 0.5) = 0.5  # doctest: +SKIP
    0.494

    See Also
    --------
    GG1 : The general single-server queue this specializes.
    GM1 : The other half of the relaxation -- general arrivals, exponential
        service.
    GGs : Several servers sharing one line, for any distributions.
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
    >>> queue[50].sim(1000).mean()  # average number in the system  # doctest: +SKIP
    0.67

    Steadier arrivals mean a shorter line at the same load: an M/M/1 queue
    under the same utilization holds more customers.

    >>> mm1 = GM1(interarrival_dist=Exponential(rate=1), service_rate=2)
    >>> mm1[50].sim(1000).mean()  # doctest: +SKIP
    1.02

    See Also
    --------
    GG1 : The general single-server queue this specializes.
    MG1 : The other half of the relaxation -- Poisson arrivals, general
        service.
    GGs : Several servers sharing one line, for any distributions.
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

    Like :class:`GG1Result`, ``path(t)`` is the number of customers in the
    system at time ``t``. The difference is how long they wait to be served:
    with ``servers`` servers sharing one line, a customer waits only until the
    *first* of them comes free, so nobody waits at all while any server is
    idle.

    That is all the recursion needs to track -- the times at which each server
    next comes free. Customer ``n`` arrives at ``A[n]``, takes the server that
    frees up soonest, and waits::

        W[n] = max(earliest free time - A[n], 0)

    and that server is then busy until ``A[n] + W[n] + S[n]``. It is the
    multi-server generalization of Lindley's recursion (the vector recursion of
    Kiefer and Wolfowitz), and with ``servers=1`` it *is* Lindley's recursion,
    written in clock time instead of in leftovers.

    The path is worked out only as far as it has been asked about, and kept once
    worked out, so evaluating it at ``t = 100`` and then at ``t = 10`` describes
    one single queue rather than two unrelated ones.

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
    ``customer_departure_times`` is not necessarily increasing, unlike in the
    single-server case. The count itself is unaffected -- it steps down at each
    departure whoever it belongs to.

    Examples
    --------
    >>> from symbulate import *
    >>> # A customer every 1 unit of time, each service taking 1.5, 2 servers.
    >>> path = GGsResult([1.0] * 8, [1.5] * 8, servers=2)
    >>> [path.waiting_times[n] for n in range(5)]  # two servers keep up
    [0.0, 0.0, 0.0, 0.0, 0.0]
    >>> # The same queue with one server falls behind (Lindley's recursion).
    >>> one = GGsResult([1.0] * 8, [1.5] * 8, servers=1)
    >>> [one.waiting_times[n] for n in range(5)]
    [0.0, 0.5, 1.0, 1.5, 2.0]

    Two servers therefore hold a shorter line than one at the same moment.

    >>> path(6.0), one(6.0)
    (2, 3)

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
        super().__init__(interarrival_times, service_times)

    def _wait_at(self, n):
        """Serve up to customer ``n``, and return how long they waited."""
        while len(self.waits) <= n:
            i = len(self.waits)
            # This recursion works in clock time, so it reads the arrival times
            # the path already keeps rather than accumulating its own copy --
            # the two could otherwise drift apart.
            arrival = self.customer_arrival_times[i]
            # Whatever is left of the wait for the soonest free server. A plain
            # float is stored rather than whatever numpy type the draws came
            # back as, so a student reading a stretch of the path sees times,
            # not type names.
            wait = float(max(self._free_times[0] - arrival, 0.0))
            # That server takes this customer, and is busy until they leave.
            heapq.heapreplace(self._free_times, arrival + wait + self.service_times[i])
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
    ``X[t]`` is the number of customers in the system at time ``t`` -- those
    being served plus everybody still in line.

    This is the multi-server generalization of :class:`GG1` -- the bank with
    several tellers and one queue, the call center with several agents -- and
    the realistic version of :class:`~symbulate.markov_chains.MMs`, since it
    does not need the service times to be exponential. The departures behind the
    count come from the times at which the servers next come free; see
    :class:`GGsResult`.

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
    long. Simulating both settles which matters for a given question --
    ``X[t]`` for the line, ``path.sojourn_times[n]`` for the visit.

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
    >>> bank[50].sim(1000).mean()  # average number in the bank at t=50  # doctest: +SKIP
    4.31

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
