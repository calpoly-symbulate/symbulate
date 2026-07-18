from .distributions import Distribution
from .index_sets import Reals
from .math import inf
from .probability_space import ProbabilitySpace
from .result import InfiniteVector, ContinuousTimeFunction, DiscreteValued
from .random_variables import RV
from .random_processes import RandomProcess


class RenewalProcessResult(ContinuousTimeFunction, DiscreteValued):
    """A single realization of a renewal process.

    Identical in structure to PoissonProcessResult: it just accumulates
    whatever interarrival times it is given and reports how many have
    occurred by time t. Nothing here assumes the times are exponential.

    Parameters
    ----------
    interarrival_times : iterable of float
        Sequence of interarrival times (any nonnegative distribution).

    Examples
    --------
    >>> from symbulate import *
    >>> path = RenewalProcess(interarrival_dist=Gamma(shape=2, rate=1)).draw()
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
        """
        return InfiniteVector(lambda n: n)


class RenewalProcessProbabilitySpace(ProbabilitySpace):
    """Probability space for a renewal process.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of the (i.i.d.) time between events. Must be a
        Symbulate ``Distribution`` (e.g. ``Exponential``, ``Gamma``,
        ``Uniform``) supporting nonnegative draws.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.

    Raises
    ------
    TypeError
        If ``interarrival_dist`` is not a Symbulate ``Distribution``.

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
        if not isinstance(interarrival_dist, Distribution):
            raise TypeError(
                "interarrival_dist must be a Symbulate Distribution "
                f"(e.g., Exponential(rate=1), Gamma(shape=2, rate=1)), "
                f"got {type(interarrival_dist).__name__}."
            )
        self.interarrival_dist = interarrival_dist

        def draw():
            interarrival_times = (self.interarrival_dist ** inf).draw()
            return RenewalProcessResult(interarrival_times)

        super().__init__(draw)


class RenewalProcess(RandomProcess, RV):
    """A random renewal process and a random variable.

    A renewal process counts events whose interarrival times are i.i.d.
    draws from an arbitrary (user-specified) distribution. A Poisson
    process is the special case where that distribution is Exponential.

    Parameters
    ----------
    interarrival_dist : Distribution
        The distribution of time between events.

    Attributes
    ----------
    interarrival_dist : Distribution
        The interarrival-time distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> N = RenewalProcess(interarrival_dist=Gamma(shape=2, rate=1))
    >>> N[2].mean()  # doctest: +SKIP
    1.0
    """

    def __init__(self, interarrival_dist):
        """Create a renewal process with the given interarrival distribution."""
        self.interarrival_dist = interarrival_dist
        prob_space = RenewalProcessProbabilitySpace(self.interarrival_dist)
        RandomProcess.__init__(self, prob_space, Reals())
        RV.__init__(self, prob_space)
