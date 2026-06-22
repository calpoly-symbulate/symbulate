from .index_sets import Naturals
from .random_variables import RV
from .result import TimeFunction, is_scalar


class RandomProcess(RV):
    """Defines a random process.

    A random process defines a random variable X(t)
    for each time t in an collection of times
    (called an index set).

    Parameters
    ----------
    prob_space : ProbabilitySpace
        The underlying probability space for the random process.
    index_set : IndexSet, optional
        The collection of times for the process. Defaults to the
        natural numbers ``0, 1, 2, 3, ...``.
    func : callable, optional
        A function ``(outcome, t)`` returning the value of the process
        at time ``t`` for a given ``outcome``. Defaults to
        ``lambda outcome, t: outcome[t]``, which treats each outcome
        as a time-indexed function.

    Attributes
    ----------
    index_set : IndexSet
        The index set of times for this process.
    rvs : dict
        Random variables assigned at specific times via ``X[t] = rv``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = RandomProcess(Bernoulli(p=0.5) ** inf)
    >>> X[3].sim(5) # random
    Results([0, 1, 0, 1, 1])
    """

    def __init__(
        self, prob_space, index_set=Naturals(), func=lambda outcome, t: outcome[t]
    ):
        """Initialize a RandomProcess."""
        self.index_set = index_set
        # This dict stores random variables at specific times.
        self.rvs = {}

        # Define the function for the RV.
        def _func(outcome):
            def x(t):
                # First, check if the time is in self.rvs.
                if t in self.rvs:
                    return self.rvs[t].func(outcome)
                return func(outcome, t)

            return TimeFunction.from_index_set(self.index_set, x)

        super().__init__(prob_space, _func)

    def __setitem__(self, t, value):
        """Set the random variable or constant value of the process at time ``t``.

        Parameters
        ----------
        t : int or float
            The time index at which to set the value. Must be in the
            process's index set.
        value : RV or scalar
            The random variable or scalar constant to assign at time ``t``.

        Raises
        ------
        KeyError
            If ``t`` is not in the index set of the random process.

        Examples
        --------
        >>> from symbulate import *
        >>> X = RandomProcess(Bernoulli(p=0.5) ** inf)
        >>> X[0] = 1
        >>> X[0].sim(5) # random
        Results([1, 1, 1, 1, 1])
        """
        if t not in self.index_set:
            raise KeyError(
                "Time %s is not in the index set for this " "random process." % str(t)
            )
        # If value is a RV, store it in self.rvs.
        if isinstance(value, RV):
            self.rvs[t] = value
        # If value is a scalar, create and store a constant random variable
        elif is_scalar(value):
            self.rvs[t] = RV(self.prob_space, lambda outcome: value)

    def __getitem__(self, t):
        """Get the random variable representing the process at time ``t``.

        Parameters
        ----------
        t : int or float
            The time index to retrieve.

        Returns
        -------
        RV
            The random variable ``X(t)`` at time ``t``.

        Examples
        --------
        >>> from symbulate import *
        >>> X = RandomProcess(Bernoulli(p=0.5) ** inf)
        >>> X[0].sim(5) # random
        Results([0, 1, 0, 1, 1])
        """
        # First, check if the time is in self.rvs.
        if t in self.rvs:
            return self.rvs[t]
        return super().__getitem__(t)

    def __call__(self, t):
        """Return the random variable representing the process at time ``t``.

        Parameters
        ----------
        t : int or float
            The time index to evaluate.

        Returns
        -------
        RV
            The random variable ``X(t)`` at time ``t``.

        Examples
        --------
        >>> from symbulate import *
        >>> X = RandomProcess(Bernoulli(p=0.5) ** inf)
        >>> X(0).sim(5) # random
        Results([0, 1, 0, 1, 1])
        """
        return RV(self.prob_space, lambda outcome: self.func(outcome)(t))
