import math
import numbers
import operator as op

import numpy as np
import scipy.stats as stats

from .random_variables import RV
from .result import Tuple, TimeFunction, ContinuousTimeFunction, DiscreteValued
from .results import Results

pi = math.pi
e = math.e
inf = float("inf")

floor = math.floor
ceil = math.ceil


def operation_factory(operation):
    """Create a function that applies a scalar operation to Symbulate objects.

    Wraps a scalar operation so it can be applied element-wise to
    `RV`, `Tuple`, `TimeFunction`, and `Results` objects via recursion.

    Parameters
    ----------
    operation : callable
        A scalar function to wrap (e.g. ``math.sqrt``).

    Returns
    -------
    callable
        A function that accepts a scalar, `RV`, `Tuple`, `TimeFunction`,
        or `Results` and applies ``operation`` appropriately.

    Examples
    --------
    >>> import math
    >>> log2 = operation_factory(lambda x: math.log(x, 2))
    >>> log2(8)
    3.0
    """

    def _op_func(x):
        if isinstance(x, (RV, Tuple, TimeFunction)):
            # recursively call op_fun until x is a scalar
            return x.apply(_op_func)
        elif isinstance(x, Results):
            return x.apply(_op_func)
        else:
            return operation(x)

    return _op_func


sqrt = operation_factory(math.sqrt)
exp = operation_factory(math.exp)
sin = operation_factory(math.sin)
cos = operation_factory(math.cos)
tan = operation_factory(math.tan)
factorial = operation_factory(math.factorial)


def log(value, base=e):
    """Compute the logarithm of a value with an optional base.

    Parameters
    ----------
    value : float or RV or Tuple or TimeFunction or Results
        The value to take the logarithm of.
    base : float, optional
        The logarithm base. Defaults to Euler's number ``e``
        (natural logarithm).

    Returns
    -------
    float or RV or Tuple or TimeFunction or Results
        The logarithm of ``value`` in the given ``base``.

    Raises
    ------
    ValueError
        If ``base`` is not a positive number other than 1.

    Examples
    --------
    >>> log(math.e)
    1.0
    >>> X = RV(Normal(0, 1))
    >>> log(X ** 2).draw()
    """
    if not isinstance(base, numbers.Real) or base <= 0 or base == 1:
        raise ValueError("base must be a positive number other than 1.")
    return operation_factory(lambda x: math.log(x, base))(value)


def mean(x):
    """Compute the arithmetic mean of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values to average. Must be non-empty.

    Returns
    -------
    float
        The arithmetic mean of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is a single real number, not iterable, or contains
        non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> mean([1, 2, 3, 4, 5])
    3.0
    """
    if isinstance(x, numbers.Real):
        raise TypeError("Taking the mean with one value is unnecessary.")
    if not hasattr(x, "__iter__"):
        raise TypeError("mean requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("mean requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("mean requires a collection of numeric values.")
    return sum(x) / len(x)


def cumsum(x):
    """Compute the cumulative sum of a sequence.

    Parameters
    ----------
    x : Results or array-like
        The sequence of values to accumulate.

    Returns
    -------
    Results or array-like
        The cumulative sum of ``x``.

    Examples
    --------
    >>> X = RV(Normal(0, 1))
    >>> X.sim(5).cumsum()
    """
    return x.cumsum()


def var(x):
    """Compute the population variance of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the variance. Must be non-empty.

    Returns
    -------
    float
        The population variance of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is not iterable or contains non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> var([2, 4, 4, 4, 5, 5, 7, 9])
    4.0
    """
    if not hasattr(x, "__iter__"):
        raise TypeError("var requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("var requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("var requires a collection of numeric values.")
    mu = mean(x)
    return mean([(i - mu) ** 2 for i in x])


def sd(x):
    """Compute the population standard deviation of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the standard deviation. Must be
        non-empty.

    Returns
    -------
    float
        The population standard deviation of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is not iterable or contains non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> sd([2, 4, 4, 4, 5, 5, 7, 9])
    2.0
    """
    if not hasattr(x, "__iter__"):
        raise TypeError("sd requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("sd requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("sd requires a collection of numeric values.")
    return math.sqrt(var(x))


def median(x):
    """Compute the median of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the median. Must be non-empty.

    Returns
    -------
    float
        The median of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is a single real number, not iterable, or contains
        non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> median([1, 2, 3, 4, 5])
    3.0
    """
    if isinstance(x, numbers.Real):
        raise TypeError("Taking the median of one value is unnecessary.")
    if not hasattr(x, "__iter__"):
        raise TypeError("median requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("median requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("median requires a collection of numeric values.")
    return np.median(x)


def min_max_diff(x):
    """Compute the range (maximum minus minimum) of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the range. Must be non-empty.

    Returns
    -------
    float
        The difference between the maximum and minimum of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is a single real number, not iterable, or contains
        non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> min_max_diff([1, 2, 3, 4, 5])
    4
    """
    if isinstance(x, numbers.Real):
        raise TypeError("Taking the range of one value is unnecessary.")
    if not hasattr(x, "__iter__"):
        raise TypeError("min_max_diff requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("min_max_diff requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("min_max_diff requires a collection of numeric values.")
    return max(x) - min(x)


def med_abs_dev(x):
    """Compute the median absolute deviation (MAD) of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the MAD.

    Returns
    -------
    float
        The median of the absolute deviations from the median of ``x``.

    Examples
    --------
    >>> med_abs_dev([1, 2, 3, 4, 5])
    1.0
    """
    med = median(x)
    return median([abs(i - med) for i in x])


def quantile(q):
    """Return a function that computes the q-th quantile of a collection.

    Parameters
    ----------
    q : float
        The quantile to compute, in the range [0, 1].

    Returns
    -------
    callable
        A function that accepts an iterable of floats and returns
        the ``q``-th quantile.

    Raises
    ------
    ValueError
        If ``q`` is not in the range [0, 1].

    Examples
    --------
    >>> q25 = quantile(0.25)
    >>> q25([1, 2, 3, 4, 5])
    2.0
    """
    if not (0 <= q <= 1):
        raise ValueError(
            f"q must be between 0 and 1, got {q}. "
            "For example, use quantile(0.25) for the 25th percentile."
        )
    return lambda x: np.quantile(x, q)


def is_discrete(x):
    """Determine whether a collection of simulated values appears to be discrete.

    Uses a repeat-count heuristic: counts how often each unique value appears,
    then returns ``True`` if more than 80% of those unique values have a count
    greater than 1. Discrete distributions (e.g. Binomial, Poisson) produce
    repeated values, while continuous distributions rarely repeat exactly.

    Parameters
    ----------
    x : iterable
        The simulated values to inspect. Must be non-empty.

    Returns
    -------
    bool
        ``True`` if more than 80% of distinct values appear more than once.

    Raises
    ------
    TypeError
        If ``x`` is an ``RV`` (use ``X.sim(n)`` first), a single real
        number, or not iterable.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> is_discrete([0, 1, 1, 0, 1])
    True
    >>> is_discrete([0.5, 1.2, 3.7])
    False
    """
    if isinstance(x, RV):
        raise TypeError(
            "is_discrete requires simulated results, not an RV. Use X.sim(n) first."
        )
    if isinstance(x, numbers.Real):
        raise TypeError(
            "is_discrete requires a collection of values, not a single number."
        )
    if not hasattr(x, "__iter__"):
        raise TypeError("is_discrete requires an iterable collection.")
    x = list(x)
    if len(x) == 0:
        raise ValueError("is_discrete requires a non-empty collection.")
    counts = {}
    for v in x:
        counts[v] = counts.get(v, 0) + 1
    return sum(c > 1 for c in counts.values()) > 0.8 * len(counts)


def quartiles(x):
    """Compute the quartiles of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values to summarize. Must be non-empty.

    Returns
    -------
    dict
        A dictionary mapping each quartile probability (0, 0.25, 0.50, 0.75, 1)
        to its value.

    Raises
    ------
    TypeError
        If ``x`` contains non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> quartiles([1, 2, 3, 4, 5])
    {0.0: 1.0, 0.25: 2.0, 0.5: 3.0, 0.75: 4.0, 1.0: 5.0}
    """
    x = list(x)
    if len(x) == 0:
        raise ValueError("quartiles requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("quartiles requires a collection of numeric values.")
    return {q: float(np.quantile(x, q)) for q in [0.00, 0.25, 0.50, 0.75, 1.00]}


def deciles(x):
    """Compute the deciles of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values to summarize. Must be non-empty.

    Returns
    -------
    dict
        A dictionary mapping each decile probability (0 through 1 by 0.1)
        to its value.

    Raises
    ------
    TypeError
        If ``x`` contains non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> deciles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    {0.0: 1.0, 0.1: 1.9, 0.2: 2.8, 0.3: 3.7, 0.4: 4.6, 0.5: 5.5, 0.6: 6.4, 0.7: 7.3, 0.8: 8.2, 0.9: 9.1, 1.0: 10.0}
    """
    x = list(x)
    if len(x) == 0:
        raise ValueError("deciles requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("deciles requires a collection of numeric values.")
    return {round(q / 10, 1): float(np.quantile(x, q / 10)) for q in range(0, 11)}


def iqr(x):
    """Compute the interquartile range (IQR) of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the IQR. Must be non-empty.

    Returns
    -------
    float
        The difference between the 75th and 25th percentiles of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is a single real number, not iterable, or contains
        non-numeric values.
    ValueError
        If ``x`` is empty.

    Examples
    --------
    >>> iqr([1, 2, 3, 4, 5])
    2.0
    """
    if isinstance(x, numbers.Real):
        raise TypeError("Taking the iqr of one value is unnecessary.")
    if not hasattr(x, "__iter__"):
        raise TypeError("iqr requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("iqr requires a non-empty collection.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("iqr requires a collection of numeric values.")
    q75, q25 = np.quantile(x, [0.75, 0.25])
    return q75 - q25


def orderstatistics(n):
    """Return a function that computes the n-th order statistic of a collection.

    Parameters
    ----------
    n : int
        The rank of the order statistic to retrieve (1-indexed, so ``n=1``
        returns the minimum).

    Returns
    -------
    callable
        A function that accepts an iterable of floats and returns the
        ``n``-th smallest value.

    Raises
    ------
    ValueError
        If ``n`` is less than or equal to 0.

    Examples
    --------
    >>> second_min = orderstatistics(2)
    >>> second_min([5, 3, 1, 4, 2])
    2
    """
    if n <= 0:
        raise ValueError("Out of bounds. Lowest order is 1.")
    else:
        return lambda x: np.partition(x, n - 1)[n - 1]


def skewness(x):
    """Compute the skewness of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the skewness. Must contain
        at least 3 values.

    Returns
    -------
    float
        The skewness of ``x``.

    Raises
    ------
    TypeError
        If ``x`` is a single real number, not iterable, or contains
        non-numeric values.
    ValueError
        If ``x`` is empty or contains fewer than 3 values.

    Examples
    --------
    >>> X = RV(Exponential(1))
    >>> X.sim(10000).apply(skewness)
    """
    if isinstance(x, numbers.Real):
        raise TypeError("Finding the skewness of one value is unnecessary.")
    if not hasattr(x, "__iter__"):
        raise TypeError("skewness requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("skewness requires a non-empty collection.")
    if len(x) < 3:
        raise ValueError("skewness requires at least 3 values.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("skewness requires a collection of numeric values.")
    return stats.skew(x)


def kurtosis(x):
    """Compute the excess kurtosis of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the kurtosis. Must contain
        at least 4 values.

    Returns
    -------
    float
        The excess kurtosis of ``x`` (Fisher's definition, normal = 0.0).

    Raises
    ------
    TypeError
        If ``x`` is a single real number, not iterable, or contains
        non-numeric values.
    ValueError
        If ``x`` is empty or contains fewer than 4 values.

    Examples
    --------
    >>> X = RV(Normal(0, 1))
    >>> X.sim(10000).apply(kurtosis)
    """
    if isinstance(x, numbers.Real):
        raise TypeError("Finding the kurtosis of one value is unnecessary.")
    if not hasattr(x, "__iter__"):
        raise TypeError("kurtosis requires an iterable collection.")
    if len(x) == 0:
        raise ValueError("kurtosis requires a non-empty collection.")
    if len(x) < 4:
        raise ValueError("kurtosis requires at least 4 values.")
    if not all(isinstance(v, numbers.Real) for v in x):
        raise TypeError("kurtosis requires a collection of numeric values.")
    return stats.kurtosis(x)


def moment(k):
    """Return a function that computes the k-th central moment of a collection.

    Parameters
    ----------
    k : int
        The order of the central moment to compute.

    Returns
    -------
    callable
        A function that accepts an iterable of floats and returns the
        ``k``-th central moment.

    Raises
    ------
    TypeError
        If ``k`` is not an integer.
    ValueError
        If ``k`` is negative.

    Examples
    --------
    >>> second_moment = moment(2)
    >>> second_moment([1, 2, 3, 4, 5])
    2.0
    """
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if k < 0:
        raise ValueError("k must be a non-negative integer.")
    return lambda x: stats.moment(x, order=k)


def trimmed_mean(alpha):
    """Return a function that computes the trimmed mean of a collection.

    Parameters
    ----------
    alpha : float
        The fraction of observations to trim from each end of the sorted
        data, in the range [0, 0.5).

    Returns
    -------
    callable
        A function that accepts an iterable of floats and returns the
        mean after trimming ``alpha`` from each tail.

    Raises
    ------
    ValueError
        If ``alpha`` is not in the range [0, 0.5).

    Examples
    --------
    >>> tm = trimmed_mean(0.1)
    >>> tm([1, 2, 3, 4, 5])
    3.0
    """
    if not (0 <= alpha < 0.5):
        raise ValueError(
            f"alpha must be in [0, 0.5), got {alpha}. "
            "For example, use trimmed_mean(0.1) to trim 10% from each tail."
        )
    return lambda x: stats.trim_mean(x, alpha)


def comparefun(x, compare, value):
    """Count elements in a collection that satisfy a binary comparison.

    Parameters
    ----------
    x : iterable
        The collection of elements to test.
    compare : callable
        A binary comparison function (e.g. ``operator.eq``) that accepts
        an element of ``x`` and ``value`` and returns a bool.
    value : any
        The value to compare each element of ``x`` against.

    Returns
    -------
    int
        The number of elements in ``x`` for which ``compare(element, value)``
        is ``True``.

    Examples
    --------
    >>> import operator
    >>> comparefun([1, 2, 3, 4, 5], operator.gt, 3)
    2
    """
    return sum(1 for i in x if compare(i, value))


def count(func=lambda x: True):
    """Return a function that counts elements satisfying a predicate.

    Parameters
    ----------
    func : callable, optional
        A predicate function that accepts a single element and returns a bool.
        Defaults to ``lambda x: True``, which counts all elements.

    Returns
    -------
    callable
        A function that accepts an iterable and returns the number of
        elements for which ``func`` returns ``True``.

    Raises
    ------
    TypeError
        If ``func`` is not callable.

    Examples
    --------
    >>> count(lambda x: x > 3)([1, 2, 3, 4, 5])
    2
    """
    if not callable(func):
        raise TypeError(
            "func must be a callable (e.g., a lambda or function), "
            f"but got {type(func).__name__}. "
            "For example, use count(lambda x: x > 0)."
        )

    def _func(x):
        return sum(1 for i in x if func(i))

    return _func


def count_eq(value):
    """Return a function that counts elements equal to a given value.

    Parameters
    ----------
    value : any
        The value to compare elements against.

    Returns
    -------
    callable
        A function that accepts an iterable and returns the count of
        elements equal to ``value``.

    Examples
    --------
    >>> count_eq(3)([1, 2, 3, 3, 5])
    2
    """

    def func(x):
        return comparefun(x, op.eq, value)

    return func


def count_neq(value):
    """Return a function that counts elements not equal to a given value.

    Parameters
    ----------
    value : any
        The value to compare elements against.

    Returns
    -------
    callable
        A function that accepts an iterable and returns the count of
        elements not equal to ``value``.

    Examples
    --------
    >>> count_neq(3)([1, 2, 3, 3, 5])
    3
    """

    def func(x):
        return comparefun(x, op.ne, value)

    return func


def count_lt(value):
    """Return a function that counts elements strictly less than a given value.

    Parameters
    ----------
    value : any
        The threshold to compare elements against.

    Returns
    -------
    callable
        A function that accepts an iterable and returns the count of
        elements strictly less than ``value``.

    Examples
    --------
    >>> count_lt(3)([1, 2, 3, 4, 5])
    2
    """

    def func(x):
        return comparefun(x, op.lt, value)

    return func


def count_gt(value):
    """Return a function that counts elements strictly greater than a given value.

    Parameters
    ----------
    value : any
        The threshold to compare elements against.

    Returns
    -------
    callable
        A function that accepts an iterable and returns the count of
        elements strictly greater than ``value``.

    Examples
    --------
    >>> count_gt(3)([1, 2, 3, 4, 5])
    2
    """

    def func(x):
        return comparefun(x, op.gt, value)

    return func


def count_geq(value):
    """Return a function that counts elements greater than or equal to a given value.

    Parameters
    ----------
    value : any
        The threshold to compare elements against.

    Returns
    -------
    callable
        A function that accepts an iterable and returns the count of
        elements greater than or equal to ``value``.

    Examples
    --------
    >>> count_geq(3)([1, 2, 3, 4, 5])
    3
    """

    def func(x):
        return comparefun(x, op.ge, value)

    return func


def count_leq(value):
    """Return a function that counts elements less than or equal to a given value.

    Parameters
    ----------
    value : any
        The threshold to compare elements against.

    Returns
    -------
    callable
        A function that accepts an iterable and returns the count of
        elements less than or equal to ``value``.

    Examples
    --------
    >>> count_leq(3)([1, 2, 3, 4, 5])
    3
    """

    def func(x):
        return comparefun(x, op.le, value)

    return func


def interarrival_times(continuous_time_function):
    """Return the interarrival times of a continuous-time discrete-state process.

    Parameters
    ----------
    continuous_time_function : ContinuousTimeFunction and DiscreteValued
        A realization of a continuous-time, discrete-state process,
        such as a ContinuousTimeMarkovChain or PoissonProcess result.

    Returns
    -------
    array-like
        The times between each successive state change.

    Raises
    ------
    TypeError
        If the argument is not a continuous-time, discrete-valued function.

    Examples
    --------
    >>> from symbulate import *
    >>> X = RV(PoissonProcessProbabilitySpace(rate=2))
    >>> interarrival_times(X.draw())  # doctest: +SKIP
    (0.12, 0.53, 0.24, 1.06, 0.31, 0.78, ...)

    For a continuous-time Markov chain these are the times spent in each state
    visited -- Exponential with that state's rate of leaving, so they need not
    all have the same distribution.

    >>> Y = ContinuousTimeMarkovChain([[-1, 1], [2, -2]], [1.0, 0.0])
    >>> interarrival_times(Y.draw())  # doctest: +SKIP
    (0.51, 1.47, 0.30, 0.14, 2.28, 0.04, ...)

    Wrapping the function in ``.apply()`` makes it a random variable, so the
    time spent in the first state can be simulated like any other.

    >>> Y.apply(interarrival_times)[0].sim(1000).mean()  # doctest: +SKIP
    0.997
    """
    if not (
        isinstance(continuous_time_function, ContinuousTimeFunction)
        and isinstance(continuous_time_function, DiscreteValued)
    ):
        raise TypeError(
            "Interarrival times are only defined for "
            "continuous-time, discrete-valued functions."
        )
    return continuous_time_function.get_interarrival_times()


def arrival_times(continuous_time_function):
    """Return the arrival times of a continuous-time discrete-state process.

    Parameters
    ----------
    continuous_time_function : ContinuousTimeFunction and DiscreteValued
        A realization of a continuous-time, discrete-state process,
        such as a ContinuousTimeMarkovChain or PoissonProcess result.

    Returns
    -------
    array-like
        The times at which each successive state change occurs.

    Raises
    ------
    TypeError
        If the argument is not a continuous-time, discrete-valued function.

    Examples
    --------
    >>> from symbulate import *
    >>> X = RV(PoissonProcessProbabilitySpace(rate=2))
    >>> arrival_times(X.draw())  # doctest: +SKIP
    (0.12, 0.65, 0.89, 1.95, 2.26, 3.04, ...)

    These are the running total of the interarrival times, so for a
    continuous-time Markov chain they are the times at which the chain jumps.

    >>> Y = ContinuousTimeMarkovChain([[-1, 1], [2, -2]], [1.0, 0.0])
    >>> arrival_times(Y.draw())  # doctest: +SKIP
    (0.51, 1.97, 2.27, 2.41, 4.69, 4.73, ...)
    """
    if not (
        isinstance(continuous_time_function, ContinuousTimeFunction)
        and isinstance(continuous_time_function, DiscreteValued)
    ):
        raise TypeError(
            "Arrival times are only defined for "
            "continuous-time, discrete-valued functions."
        )
    return continuous_time_function.get_arrival_times()


def states(discrete_valued_function):
    """Return the sequence of states of a discrete-valued process realization.

    Parameters
    ----------
    discrete_valued_function : DiscreteValued
        A realization of a discrete-valued process, such as a
        MarkovChain or PoissonProcess result.

    Returns
    -------
    InfiniteVector
        The sequence of values (states) visited by the process.

    Raises
    ------
    TypeError
        If the argument is not a discrete-valued function.

    Examples
    --------
    >>> from symbulate import *
    >>> T = [[0.5, 0.5], [0.5, 0.5]]
    >>> X = RV(MarkovChainProbabilitySpace(transition_matrix=T, initial_dist=[1, 0]))
    >>> states(X.draw())  # doctest: +SKIP
    (0, 1, 1, 0, 1, 0, ...)

    For a continuous-time process these are the states visited in order,
    ignoring how long the process stayed in each one. They come back as the
    state *labels*, so they match what evaluating the path returns.

    >>> Y = ContinuousTimeMarkovChain([[-1, 1], [2, -2]], [1.0, 0.0],
    ...                               state_labels=['A', 'B'])
    >>> path = Y.draw()
    >>> states(path)[0] == path(0)
    True
    """
    if not isinstance(discrete_valued_function, DiscreteValued):
        raise TypeError("States are only defined for discrete-valued " "functions.")
    return discrete_valued_function.get_states()
