import numpy as np

from .distributions import Distribution, Exponential, MultivariateDistribution
from .index_sets import Reals
from .math import inf
from .probability_space import ProbabilitySpace
from .result import InfiniteVector, ContinuousTimeFunction, DiscreteValued
from .random_variables import RV
from .random_processes import RandomProcess


def _smallest_possible_time(interarrival_dist):
    """The smallest value an interarrival-time distribution can produce.

    Read from the underlying scipy distribution's support, so a new
    distribution is checked correctly with no per-distribution code.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution to inspect.

    Returns
    -------
    float or None
        The smallest possible value, or ``None`` when it cannot be
        determined (a point-mass distribution such as ``Uniform(a=2, b=2)``
        reports no support at all).
    """
    try:
        low, _ = interarrival_dist._support()
    except AttributeError:
        # No scipy object to ask -- a hand-written point-mass branch, like
        # LogNormal(mu, 0). Its quantile function still reports the single
        # value it can produce.
        try:
            low = float(interarrival_dist.quantile(0))
        except (AttributeError, TypeError, ValueError):
            return None
    return None if np.isnan(low) else low


def _is_always_zero(interarrival_dist):
    """Whether a distribution produces a time of 0 on every single draw.

    Such a distribution would put infinitely many events at time 0, so the
    count of events by any later time would never finish computing.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution to inspect. Assumed to be nonnegative already.

    Returns
    -------
    bool
        ``True`` only when the distribution is certainly a point mass at 0.
    """
    try:
        _, high = interarrival_dist._support()
        if high == 0:
            return True
    except AttributeError:
        pass
    # Some point masses at 0 (e.g. Poisson(0)) report a wider support than
    # they can actually reach, so fall back to the mean: for a nonnegative
    # distribution, a mean of 0 means every draw is 0.
    try:
        return float(interarrival_dist.mean()) == 0
    except (AttributeError, TypeError, ValueError):
        return False


def _validate_interarrival_dist(interarrival_dist):
    """Check that a distribution can serve as a renewal interarrival time.

    A renewal process counts events as time passes, so each interarrival
    time must be a single nonnegative number, and the times must not all be
    0. This is the renewal-process analogue of ``PoissonProcess``'s
    ``rate > 0`` check.

    Parameters
    ----------
    interarrival_dist : Distribution
        The candidate distribution of time between events.

    Raises
    ------
    TypeError
        If ``interarrival_dist`` is not a Symbulate ``Distribution``, or is
        a multivariate one (a whole vector per draw rather than one time).
    ValueError
        If the distribution can produce a negative time, or produces a time
        of 0 on every draw.
    """
    if not isinstance(interarrival_dist, Distribution):
        message = (
            "interarrival_dist must be a Symbulate distribution describing the "
            "time between events, such as Exponential(rate=1), "
            "Gamma(shape=2, rate=1), or Uniform(a=0, b=2). You gave "
            f"{type(interarrival_dist).__name__}."
        )
        if isinstance(interarrival_dist, (int, float)):
            message += (
                f" (If you wanted events arriving at rate {interarrival_dist}, "
                f"use PoissonProcess(rate={interarrival_dist}) or "
                f"RenewalProcess(Exponential(rate={interarrival_dist})).)"
            )
        raise TypeError(message)

    name = type(interarrival_dist).__name__

    if isinstance(interarrival_dist, MultivariateDistribution):
        raise TypeError(
            "interarrival_dist must be a distribution of single numbers, but "
            f"{name} produces a whole vector of numbers on each draw. A "
            "renewal process needs one waiting time per event. Try a "
            "one-number distribution such as Exponential(rate=1) or "
            "Gamma(shape=2, rate=1)."
        )

    smallest = _smallest_possible_time(interarrival_dist)
    if smallest is not None and smallest < 0:
        raise ValueError(
            f"interarrival_dist must never produce a negative time, but {name} "
            f"can produce values as small as {smallest:g}. The time between "
            "two events is a waiting time, so it cannot be negative -- a "
            "negative waiting time would make the count of events go "
            "backwards. Try a distribution that is never negative, such as "
            "Exponential(rate=1), Gamma(shape=2, rate=1), LogNormal(0, 1), or "
            "Uniform(a=0, b=2)."
        )

    if _is_always_zero(interarrival_dist):
        raise ValueError(
            f"interarrival_dist must sometimes produce a positive time, but "
            f"{name} produces a time of 0 on every draw, which would put "
            "infinitely many events at time 0. Choose a distribution with an "
            "average waiting time greater than 0, such as Exponential(rate=1) "
            "or Gamma(shape=2, rate=1)."
        )


class RenewalProcessResult(ContinuousTimeFunction, DiscreteValued):
    """A single realization of a renewal process.

    Identical in structure to ``PoissonProcessResult``: it accumulates
    whatever interarrival times it is given and reports how many events have
    occurred by time ``t``. Nothing here assumes the times are exponential.

    Parameters
    ----------
    interarrival_times : iterable of float
        Sequence of nonnegative interarrival times.

    Attributes
    ----------
    interarrival_times : iterable of float
        The interarrival times of this sample path.

    Examples
    --------
    >>> from symbulate import *
    >>> path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
    >>> path[1.5]  # doctest: +SKIP
    1
    """

    def __init__(self, interarrival_times):
        """Create a renewal process sample path from interarrival times."""
        self.interarrival_times = interarrival_times

        def func(t):
            total_time = 0
            for n, time in enumerate(self.interarrival_times):
                total_time += time
                if t < total_time:
                    return n

        super().__init__(func)

    def get_states(self):
        """Get the sequence of states of the renewal process.

        Returns
        -------
        InfiniteVector
            An infinite vector of the states of the process starting at 0.

        Examples
        --------
        >>> from symbulate import *
        >>> path = RenewalProcess(Gamma(shape=2, rate=1)).draw()
        >>> states = path.get_states()
        >>> states[0]
        0
        >>> states[3]
        3
        """
        return InfiniteVector(lambda n: n)


class RenewalProcessProbabilitySpace(ProbabilitySpace):
    """Probability space for a renewal process.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of the (i.i.d.) time between events. Must be a
        Symbulate ``Distribution`` over single nonnegative numbers (e.g.
        ``Exponential``, ``Gamma``, ``Uniform(a=0, b=2)``).

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.

    Raises
    ------
    TypeError
        If ``interarrival_dist`` is not a Symbulate ``Distribution`` over
        single numbers.
    ValueError
        If ``interarrival_dist`` can produce a negative time, or produces a
        time of 0 on every draw.

    Examples
    --------
    >>> from symbulate import *
    >>> space = RenewalProcessProbabilitySpace(Gamma(shape=2, rate=1))
    >>> path = space.draw()
    >>> path[1.0]  # doctest: +SKIP
    0
    """

    def __init__(self, interarrival_dist):
        """Create a probability space for a renewal process."""
        _validate_interarrival_dist(interarrival_dist)
        self.interarrival_dist = interarrival_dist

        # Build the interarrival-time space once, not once per draw: the
        # distribution never changes, and each `.draw()` on it already yields
        # a fresh, independent sequence. (Same reasoning as PoissonProcess.)
        interarrivals = self.interarrival_dist**inf

        def draw():
            return RenewalProcessResult(interarrivals.draw())

        super().__init__(draw)


class RenewalProcess(RandomProcess, RV):
    """A random renewal process and a random variable.

    A renewal process counts events whose interarrival times are i.i.d.
    draws from any distribution of nonnegative times. A Poisson process is
    the special case where that distribution is exponential, so
    ``RenewalProcess(Exponential(rate=2))`` behaves just like
    ``PoissonProcess(rate=2)``.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of time between events. Must be a Symbulate
        ``Distribution`` over single nonnegative numbers.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.

    Raises
    ------
    TypeError
        If ``interarrival_dist`` is not a Symbulate ``Distribution`` over
        single numbers.
    ValueError
        If ``interarrival_dist`` can produce a negative time, or produces a
        time of 0 on every draw.

    Notes
    -----
    A distribution that is a point mass at a single value, such as
    ``Uniform(a=2, b=2)``, reports no support for Symbulate to check, so it
    is accepted on the assumption that the value is positive.

    Examples
    --------
    >>> from symbulate import *
    >>> N = RenewalProcess(Gamma(shape=2, rate=1))
    >>> N[10].mean()  # doctest: +SKIP
    4.5
    """

    def __init__(self, interarrival_dist):
        """Create a renewal process with the given interarrival distribution."""
        prob_space = RenewalProcessProbabilitySpace(interarrival_dist)
        self.interarrival_dist = prob_space.interarrival_dist
        RandomProcess.__init__(self, prob_space, Reals())
        RV.__init__(self, prob_space)


def _validate_rate(rate):
    """Check that ``rate`` can serve as the rate of a Poisson counting process.

    The same check ``PoissonProcessProbabilitySpace`` makes, worded for a
    process whose events carry a jump size.

    Parameters
    ----------
    rate : float
        The candidate rate: the average number of events per unit time.

    Raises
    ------
    TypeError
        If ``rate`` is not a number.
    ValueError
        If ``rate`` is not positive.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise TypeError(
            "rate must be a positive number: the average number of events per "
            f"unit time. You gave {type(rate).__name__}. For example, "
            "rate=2 for two events per unit time on average."
        )
    if rate == 0:
        raise ValueError(
            "rate must be positive, got 0. A rate of 0 would mean no events "
            "ever occur, so the total would stay at 0 forever. Try a rate "
            "above 0 -- for example, rate=2 for two events per unit time on "
            "average."
        )
    if rate < 0:
        raise ValueError(
            f"rate must be positive, got {rate}. The rate is how frequently "
            "events happen, so it cannot go below 0. If what you wanted is a "
            "total that can go down as well as up, that belongs in jump_dist "
            "instead -- for example, jump_dist=Normal(mean=0, sd=1)."
        )


def _validate_jump_dist(jump_dist):
    """Check that a distribution can serve as a jump size.

    Unlike an interarrival time, a jump size has no sign restriction: a
    compound Poisson process is free to go down as well as up. All that is
    required is one number per event.

    Parameters
    ----------
    jump_dist : Distribution
        The candidate distribution of the jump added at each event.

    Raises
    ------
    TypeError
        If ``jump_dist`` is not a Symbulate ``Distribution``, or is a
        multivariate one (a whole vector per draw rather than one jump).
    """
    if not isinstance(jump_dist, Distribution):
        message = (
            "jump_dist must be a Symbulate distribution describing the size of "
            "the jump at each event, such as Exponential(rate=1), "
            "Gamma(shape=2, rate=1), or Normal(mean=0, sd=1). You gave "
            f"{type(jump_dist).__name__}."
        )
        if isinstance(jump_dist, (int, float)) and not isinstance(jump_dist, bool):
            message += (
                f" (If every jump is exactly {jump_dist}, write that as a "
                f"distribution too: Uniform(a={jump_dist}, b={jump_dist}).)"
            )
        raise TypeError(message)

    if isinstance(jump_dist, MultivariateDistribution):
        raise TypeError(
            "jump_dist must be a distribution of single numbers, but "
            f"{type(jump_dist).__name__} produces a whole vector of numbers on "
            "each draw. A compound Poisson process adds one number to a "
            "running total at each event. Try a one-number distribution such "
            "as Exponential(rate=1) or Normal(mean=0, sd=1)."
        )


class CompoundPoissonProcessResult(ContinuousTimeFunction, DiscreteValued):
    """A single realization of a compound Poisson process.

    A step function that starts at 0 and jumps at each event. The jump
    sizes line up with the interarrival times index by index: the process
    waits ``interarrival_times[n]`` before its ``n``-th jump, and that jump
    adds ``jump_sizes[n]`` to the running total.

    Parameters
    ----------
    interarrival_times : iterable of float
        Sequence of times between events.
    jump_sizes : iterable of float
        Sequence of jump sizes, one per event.

    Attributes
    ----------
    interarrival_times : iterable of float
        The times between events on this sample path.
    jump_sizes : iterable of float
        The jump added at each event on this sample path.
    states : InfiniteVector
        The value the process holds between events: ``states[n]`` is the
        running total during the ``n``-th waiting period, so ``states[0]``
        is 0 (nothing has jumped yet). Reached with ``states(path)`` or
        ``path.get_states()``.

    See Also
    --------
    RenewalProcessResult : Counts the events without the jump sizes.

    Examples
    --------
    >>> from symbulate import *
    >>> path = CompoundPoissonProcess(rate=1, jump_dist=Exponential(rate=1)).draw()
    >>> path[2.5]  # doctest: +SKIP
    1.83

    The pieces the path is built from are all available:

    >>> path.jump_sizes[0]  # doctest: +SKIP
    0.71
    >>> path.get_arrival_times()[0]  # doctest: +SKIP
    0.43
    """

    def __init__(self, interarrival_times, jump_sizes):
        """Create a compound Poisson sample path from times and jump sizes."""
        self.interarrival_times = interarrival_times
        self.jump_sizes = jump_sizes
        # The running total after each event, indexed so that the process
        # sits at states[n] for interarrival_times[n] units of time -- the
        # same index-by-index convention ContinuousTimeMarkovChainResult
        # uses. Nothing has jumped during the first wait, so states[0] = 0.
        self.states = InfiniteVector(
            lambda n: sum(self.jump_sizes[i] for i in range(n))
        )

        def func(t):
            total_time = 0
            total_jumps = 0
            for n, time in enumerate(self.interarrival_times):
                total_time += time
                if t < total_time:
                    return total_jumps
                total_jumps += self.jump_sizes[n]

        super().__init__(func)


class CompoundPoissonProcessProbabilitySpace(ProbabilitySpace):
    """Probability space for a compound Poisson process.

    Parameters
    ----------
    rate : float
        The rate of the underlying Poisson process: the average number of
        events per unit time. Must be positive.
    jump_dist : Distribution
        The distribution of the (i.i.d.) jump added at each event. Must be a
        Symbulate ``Distribution`` over single numbers.

    Attributes
    ----------
    rate : float
        The rate of events.
    jump_dist : Distribution
        The jump-size distribution.

    Raises
    ------
    TypeError
        If ``rate`` is not a number, or ``jump_dist`` is not a Symbulate
        ``Distribution`` over single numbers.
    ValueError
        If ``rate`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> space = CompoundPoissonProcessProbabilitySpace(2, Exponential(rate=1))
    >>> path = space.draw()
    >>> path[1.0]  # doctest: +SKIP
    1.42
    """

    def __init__(self, rate, jump_dist):
        """Create a probability space for a compound Poisson process."""
        _validate_rate(rate)
        _validate_jump_dist(jump_dist)
        self.rate = rate
        self.jump_dist = jump_dist

        # Both sequences are built once, not once per draw: neither the rate
        # nor the jump distribution changes, and each `.draw()` already gives
        # a fresh, independent sequence. (Same reasoning as PoissonProcess.)
        # The event times are exponential, exactly as in PoissonProcess --
        # the compounding is the second sequence, not a different first one.
        interarrivals = Exponential(rate=self.rate) ** inf
        jumps = self.jump_dist**inf

        def draw():
            return CompoundPoissonProcessResult(interarrivals.draw(), jumps.draw())

        super().__init__(draw)


class CompoundPoissonProcess(RandomProcess, RV):
    """A random compound Poisson process and a random variable.

    Events happen as in a ``PoissonProcess``, but each event now carries a
    size drawn from ``jump_dist``, and the process reports the running total
    of those sizes rather than a count of events. Writing ``N(t)`` for the
    number of events by time ``t`` and ``Y1, Y2, ...`` for the jumps, the
    value at time ``t`` is ``Y1 + Y2 + ... + Y_N(t)``.

    This is the standard model for a total that accumulates in lumps at
    random times: claims arriving at an insurer, losses in a portfolio,
    rainfall in storms, deposits and withdrawals in an account.

    Parameters
    ----------
    rate : float
        The rate of the underlying Poisson process: the average number of
        events per unit time. Must be positive.
    jump_dist : Distribution
        The distribution of the jump added at each event. Must be a Symbulate
        ``Distribution`` over single numbers. Jumps may be negative, so the
        total can go down as well as up.

    Attributes
    ----------
    rate : float
        The rate of events.
    jump_dist : Distribution
        The jump-size distribution.

    Raises
    ------
    TypeError
        If ``rate`` is not a number, or ``jump_dist`` is not a Symbulate
        ``Distribution`` over single numbers.
    ValueError
        If ``rate`` is not positive.

    See Also
    --------
    PoissonProcess : Counts the events, with no jump sizes attached.
    RenewalProcess : Counts events whose waiting times need not be
        exponential.

    Notes
    -----
    Two facts worth checking a simulation against. If a jump averages
    ``E[Y]``, then the total by time ``t`` averages ``rate * t * E[Y]`` --
    the average number of events times the average size of one. Its variance
    is ``rate * t * E[Y ** 2]``, which is *not* the number of events times
    the variance of a jump: both the number of events and their sizes vary,
    and the second moment is what combines the two.

    A jump of exactly 1 at every event recovers the count itself, so
    ``CompoundPoissonProcess(rate=2, jump_dist=Uniform(a=1, b=1))`` behaves
    like ``PoissonProcess(rate=2)``.

    Examples
    --------
    Three claims per unit time on average, each claim averaging 500. The
    average total claimed by time 10 is then 3 * 10 * 500 = 15000.

    >>> from symbulate import *
    >>> X = CompoundPoissonProcess(rate=3, jump_dist=Exponential(rate=1 / 500))
    >>> X(10).mean()  # doctest: +SKIP
    15034.2
    """

    def __init__(self, rate, jump_dist):
        """Create a compound Poisson process with the given rate and jumps."""
        prob_space = CompoundPoissonProcessProbabilitySpace(rate, jump_dist)
        self.rate = prob_space.rate
        self.jump_dist = prob_space.jump_dist
        RandomProcess.__init__(self, prob_space, Reals())
        RV.__init__(self, prob_space)
