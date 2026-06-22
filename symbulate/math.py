import math
import numbers
import operator as op

import numpy as np
import scipy.stats as stats

from .random_variables import RV
from .result import (
    Tuple,
    TimeFunction,
    ContinuousTimeFunction,
    DiscreteValued
)
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

    Examples
    --------
    >>> log(math.e)
    1.0
    >>> X = RV(Normal(0, 1))
    >>> log(X ** 2).draw()
    """
    return operation_factory(lambda x: math.log(x, base))(value)

def mean(x):
    """Compute the arithmetic mean of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values to average. Must contain more than one element.

    Returns
    -------
    float
        The arithmetic mean of ``x``.

    Raises
    ------
    Exception
        If ``x`` is a single real number.

    Examples
    --------
    >>> mean([1, 2, 3, 4, 5])
    3.0
    """
    if isinstance(x, numbers.Real):
        raise Exception("Taking the mean with one value is unnecessary.")
    else:
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
        The values for which to compute the variance.

    Returns
    -------
    float
        The population variance of ``x``.

    Examples
    --------
    >>> var([2, 4, 4, 4, 5, 5, 7, 9])
    4.0
    """
    return mean([(i - mean(x)) ** 2 for i in x])

def sd(x):
    """Compute the population standard deviation of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the standard deviation.

    Returns
    -------
    float
        The population standard deviation of ``x``.

    Examples
    --------
    >>> sd([2, 4, 4, 4, 5, 5, 7, 9])
    2.0
    """
    return math.sqrt(var(x))

def median(x):
    """Compute the median of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the median. Must contain
        more than one element.

    Returns
    -------
    float
        The median of ``x``.

    Raises
    ------
    Exception
        If ``x`` is a single real number.

    Examples
    --------
    >>> median([1, 2, 3, 4, 5])
    3.0
    """
    if isinstance(x, numbers.Real):
        raise Exception("Taking the median of one value is unnecessary.")
    else:
        return np.median(x)

def min_max_diff(x):
    """Compute the range (maximum minus minimum) of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the range. Must contain
        more than one element.

    Returns
    -------
    float
        The difference between the maximum and minimum of ``x``.

    Raises
    ------
    Exception
        If ``x`` is a single real number.

    Examples
    --------
    >>> min_max_diff([1, 2, 3, 4, 5])
    4
    """
    if isinstance(x, numbers.Real):
        raise Exception("Taking the range of one value is unnecessary.")
    else:
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
    return median(list(abs(i-median(x)) for i in x))

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

    Examples
    --------
    >>> q25 = quantile(0.25)
    >>> q25([1, 2, 3, 4, 5])
    2.0
    """
    return lambda x: np.percentile(x, q * 100)

def iqr(x):
    """Compute the interquartile range (IQR) of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the IQR. Must contain
        more than one element.

    Returns
    -------
    float
        The difference between the 75th and 25th percentiles of ``x``.

    Raises
    ------
    Exception
        If ``x`` is a single real number.

    Examples
    --------
    >>> iqr([1, 2, 3, 4, 5])
    2.0
    """
    if isinstance(x, numbers.Real):
        raise Exception("Taking the iqr of one value is unnecessary.")
    else:
        q75, q25 = np.percentile(x, [75, 25])
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
    Exception
        If ``n`` is less than or equal to 0.

    Examples
    --------
    >>> second_min = orderstatistics(2)
    >>> second_min([5, 3, 1, 4, 2])
    2
    """
    if n <= 0:
        raise Exception("Out of bounds. Lowest order is 1.")
    else:
        return lambda x: np.partition(x, n - 1)[n - 1]

def skewness(x):
    """Compute the skewness of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the skewness. Must contain
        more than one element.

    Returns
    -------
    float
        The skewness of ``x``.

    Raises
    ------
    Exception
        If ``x`` is a single real number.

    Examples
    --------
    >>> X = RV(Exponential(1))
    >>> X.sim(10000).apply(skewness)
    """
    if isinstance(x, numbers.Real):
        raise Exception("Finding the skenewss of one value is unnecessary,")
    else:
        return stats.skew(x)

def kurtosis(x):
    """Compute the excess kurtosis of a collection of values.

    Parameters
    ----------
    x : iterable of float
        The values for which to compute the kurtosis. Must contain
        more than one element.

    Returns
    -------
    float
        The excess kurtosis of ``x`` (Fisher's definition, normal = 0.0).

    Raises
    ------
    Exception
        If ``x`` is a single real number.

    Examples
    --------
    >>> X = RV(Normal(0, 1))
    >>> X.sim(10000).apply(kurtosis)
    """
    if isinstance(x, numbers.Real):
        raise Exception("Finding the kurtosis of one value is unnecessary.")
    else:
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

    Examples
    --------
    >>> second_moment = moment(2)
    >>> second_moment([1, 2, 3, 4, 5])
    2.0
    """
    return lambda x: stats.moment(x, k)

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

    Examples
    --------
    >>> tm = trimmed_mean(0.1)
    >>> tm([1, 2, 3, 4, 5])
    3.0
    """
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
    count = 0
    for i in x:
        if compare(i, value):
            count += 1
    return count

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

    Examples
    --------
    >>> count(lambda x: x > 3)([1, 2, 3, 4, 5])
    2
    """
    def _func(x):
        val = 0
        for i in x:
            if func(i):
                val += 1
        return val
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
    >>> interarrival_times(X.draw())
    """
    if not (isinstance(continuous_time_function,
                       ContinuousTimeFunction) and
            isinstance(continuous_time_function,
                       DiscreteValued)):
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
    >>> arrival_times(X.draw())
    """
    if not (isinstance(continuous_time_function,
                       ContinuousTimeFunction) and
            isinstance(continuous_time_function,
                       DiscreteValued)):
        raise TypeError(
            "Interarrival times are only defined for "
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
    >>> states(X.draw())
    """
    if not isinstance(discrete_valued_function, DiscreteValued):
        raise TypeError(
            "States are only defined for discrete-valued "
            "functions."
        )
    return discrete_valued_function.get_states()
