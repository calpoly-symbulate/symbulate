from .distributions import Exponential
from .math import inf
from .probability_space import ProbabilitySpace
from .result import InfiniteVector, ContinuousTimeFunction, DiscreteValued
from .random_variables import RV
from .random_processes import RandomProcess


class PoissonProcessResult(ContinuousTimeFunction, DiscreteValued):
    """A single realization of a Poisson process.

    Parameters
    ----------
    interarrival_times : iterable of float
        Sequence of exponential interarrival times.

    Attributes
    ----------
    interarrival_times : list
        A list of interarrival times for the Poisson process.

    Examples
    --------
    >>> from symbulate import *
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> path = PoissonProcess(rate=1).draw()
    >>> path[1.5]
    1
    """

    def __init__(self, interarrival_times):
        """Create a Poisson process sample path from interarrival times."""
        self.interarrival_times = interarrival_times

        def func(t):
            total_time = 0
            for n, time in enumerate(self.interarrival_times):
                total_time += time
                if t < total_time:
                    return n

        super().__init__(func)

    def get_states(self):
        """Get the sequence of states of the Poisson process.

        Returns
        -------
        InfiniteVector
            An infinite vector of the states of the Poisson process starting at 0.

        Examples
        --------
        >>> from symbulate import *
        >>> path = PoissonProcess(rate=1).draw()
        >>> states = path.get_states()
        >>> states[0]
        0
        >>> states[3]
        3
        """
        return InfiniteVector(lambda n: n)


class PoissonProcessProbabilitySpace(ProbabilitySpace):
    """Probability space for a Poisson process.

    Parameters
    ----------
    rate : float
        The rate (λ) of the Poisson process. Must be positive.

    Attributes
    ----------
    rate : float
        The rate parameter of the Poisson process.

    Examples
    --------
    >>> from symbulate import *
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> space = PoissonProcessProbabilitySpace(rate=2)
    >>> path = space.draw()
    >>> path[1.0]
    2
    """

    def __init__(self, rate):
        """Create a probability space for a Poisson process."""
        self.rate = rate

        def draw():
            interarrival_times = (Exponential(rate=self.rate) ** inf).draw()
            return PoissonProcessResult(interarrival_times)

        super().__init__(draw)


class PoissonProcess(RandomProcess, RV):
    """A random Poisson process and a random variable.

    Parameters
    ----------
    rate : float
        The rate (λ) of the Poisson process. Must be positive.

    Attributes
    ----------
    rate : float
        The rate parameter of the Poisson process.

    Examples
    --------
    >>> from symbulate import *
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> N = PoissonProcess(rate=1)
    >>> N[2].mean()  # doctest: +SKIP
    2.0
    """

    def __init__(self, rate):
        """Create a Poisson process with the given rate."""
        self.rate = rate
        prob_space = PoissonProcessProbabilitySpace(self.rate)
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)
