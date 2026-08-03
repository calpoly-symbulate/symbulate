from .index_sets import IndexSet, Naturals
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

    Raises
    ------
    TypeError
        If ``index_set`` is not an ``IndexSet`` instance.
    TypeError
        If ``func`` is not callable.

    Examples
    --------
    >>> from symbulate import *
    >>> X = RandomProcess(Bernoulli(p=0.5) ** inf)
    >>> X[3].sim(5)  # doctest: +SKIP
    Results([0, 1, 0, 1, 1])
    """

    def __init__(
        self, prob_space, index_set=Naturals(), func=lambda outcome, t: outcome[t]
    ):
        """Initialize a RandomProcess."""
        if not isinstance(index_set, IndexSet):
            raise TypeError(
                f"index_set must be an IndexSet instance (e.g., Naturals(), Reals()), "
                f"got {type(index_set).__name__}."
            )
        if not callable(func):
            raise TypeError(
                f"func must be callable (e.g., lambda outcome, t: outcome[t]), "
                f"got {type(func).__name__}."
            )
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
        >>> X[0].sim(5)  # doctest: +SKIP
        Results([1, 1, 1, 1, 1])
        """
        if t not in self.index_set:
            raise KeyError(
                f"Time {t!r} is not in the index set for this random process "
                f"({type(self.index_set).__name__})."
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
        >>> X[0].sim(5)  # doctest: +SKIP
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
        >>> X(0).sim(5)  # doctest: +SKIP
        Results([0, 1, 0, 1, 1])
        """
        return RV(self.prob_space, lambda outcome: self.func(outcome)(t))


def _resolve_initial(initial, older, older_name, default=0):
    """Settle a process's starting condition from its current and older name.

    Every process in the package names its starting condition ``initial``.
    Several of them used to spell it something else -- ``initial_value``,
    ``x0``, ``initial_dist`` -- and those spellings are still accepted so
    existing code and notebooks keep working. This picks whichever was
    given. See ``MODEL-DECISIONS.md``, "One Name for a Process's Starting
    Condition".

    Parameters
    ----------
    initial : object or None
        What was passed as ``initial``.
    older : object or None
        What was passed under the process's older name.
    older_name : str
        That older name, used in the error message.
    default : object, optional
        What to use when neither was given. Default is 0.

    Returns
    -------
    object
        The starting condition to use.

    Raises
    ------
    ValueError
        If both names were given, since they mean the same thing and there
        is no sensible way to combine them.

    Examples
    --------
    >>> _resolve_initial(5, None, "initial_value")
    5
    >>> _resolve_initial(None, 7, "initial_value")
    7
    >>> _resolve_initial(None, None, "initial_value")
    0
    """
    if initial is not None and older is not None:
        raise ValueError(
            f"Specify either initial or {older_name}, not both. They mean "
            f"the same thing -- where the process starts -- and initial is "
            f"the name to prefer; {older_name} is kept only so older code "
            f"keeps working."
        )
    if initial is not None:
        return initial
    if older is not None:
        return older
    return default
