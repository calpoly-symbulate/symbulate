import numpy as np

from .distributions import Distribution, MultivariateDistribution
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
