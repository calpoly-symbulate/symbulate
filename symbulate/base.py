import math

import numpy as np
import scipy.stats as stats


class Arithmetic:
    """Mixin providing arithmetic operators (+, -, \\*, /) for subclasses.

    Subclasses must implement the _operation_factory method,
    which specifies how each operation acts on instances of
    that class.
    """

    def __add__(self, other):
        """Return the element-wise sum (e.g., X + Y or X + 3).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Right-hand operand.

        Returns
        -------
        RV or RandomProcess
            Element-wise sum of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> Y = RV(Normal(0, 1))
        >>> (X + Y).draw()
        >>> (X + 3).draw()
        """
        op_func = self._operation_factory(lambda x, y: x + y)
        return op_func(self, other)

    def __radd__(self, other):
        """Return the element-wise sum with scalar on the left (e.g., 3 + X).

        Parameters
        ----------
        other : scalar
            Left-hand operand.

        Returns
        -------
        RV or RandomProcess
            Element-wise sum of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (3 + X).draw()
        """
        return self.__add__(other)

    def __sub__(self, other):
        """Return the element-wise difference (e.g., X - Y or X - 3).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Right-hand operand.

        Returns
        -------
        RV or RandomProcess
            Element-wise difference of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> Y = RV(Normal(0, 1))
        >>> (X - Y).draw()
        >>> (X - 3).draw()
        """
        op_func = self._operation_factory(lambda x, y: x - y)
        return op_func(self, other)

    def __rsub__(self, other):
        """Return the difference with scalar on the left (e.g., 3 - X).

        Parameters
        ----------
        other : scalar
            Left-hand operand.

        Returns
        -------
        RV or RandomProcess
            Element-wise difference of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (3 - X).draw()
        """
        op_func = self._operation_factory(lambda x, y: y - x)
        return op_func(self, other)

    def __neg__(self):
        """Return the element-wise negation (e.g., -X).

        Returns
        -------
        RV or RandomProcess
            Element-wise negation.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (-X).draw()
        """
        return -1 * self

    def __mul__(self, other):
        """Return the element-wise product (e.g., X * Y or X * 2).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Right-hand operand.

        Returns
        -------
        RV or RandomProcess
            Element-wise product of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> Y = RV(Normal(0, 1))
        >>> (X * Y).draw()
        >>> (X * 2).draw()
        """
        op_func = self._operation_factory(lambda x, y: x * y)
        return op_func(self, other)

    def __rmul__(self, other):
        """Return the element-wise product with scalar on the left (e.g., 2 * X).

        Parameters
        ----------
        other : scalar
            Left-hand operand.

        Returns
        -------
        RV or RandomProcess
            Element-wise product of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (2 * X).draw()
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Return the element-wise quotient (e.g., X / Y or X / 2).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Right-hand operand (divisor).

        Returns
        -------
        RV or RandomProcess
            Element-wise quotient of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> Y = RV(Normal(0, 1))
        >>> (X / Y).draw()
        >>> (X / 2).draw()
        """
        op_func = self._operation_factory(lambda x, y: x / y)
        return op_func(self, other)

    def __rtruediv__(self, other):
        """Return the quotient with scalar on the left (e.g., 2 / X).

        Parameters
        ----------
        other : scalar
            Left-hand operand (dividend).

        Returns
        -------
        RV or RandomProcess
            Element-wise quotient of the two operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (2 / X).draw()
        """
        op_func = self._operation_factory(lambda x, y: y / x)
        return op_func(self, other)

    def __pow__(self, other):
        """Return the element-wise power (e.g., X ** 2).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Exponent.

        Returns
        -------
        RV or RandomProcess
            Element-wise power of the operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (X ** 2).draw()
        """
        op_func = self._operation_factory(lambda x, y: x**y)
        return op_func(self, other)

    def __rpow__(self, other):
        """Return the power with scalar base on the left (e.g., 2 ** X).

        Parameters
        ----------
        other : scalar
            Base value.

        Returns
        -------
        RV or RandomProcess
            Element-wise power of the operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (2 ** X).draw()
        """
        op_func = self._operation_factory(lambda x, y: y**x)
        return op_func(self, other)

    def __xor__(self, other):
        """Return the element-wise power using ^ notation (e.g., X ^ 2).

        Alternative to ``**``; calls :meth:`__pow__`.

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Exponent.

        Returns
        -------
        RV or RandomProcess
            Element-wise power of the operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (X ^ 2).draw()
        """
        return self.__pow__(other)

    def __rxor__(self, other):
        """Return the power with scalar base using ^ notation (e.g., 2 ^ X).

        Alternative to ``**``; calls :meth:`__rpow__`.

        Parameters
        ----------
        other : scalar
            Base value.

        Returns
        -------
        RV or RandomProcess
            Element-wise power of the operands.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (2 ^ X).draw()
        """
        return self.__rpow__(other)


class Comparable:
    """Mixin providing comparison operators (<, >, ==) for subclasses.

    Subclasses must implement the _comparison_factory method,
    which specifies how each comparison acts on instances of
    that class.
    """

    def __eq__(self, other):
        """Return an indicator of element-wise equality (e.g., X == 0).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Value to compare against.

        Returns
        -------
        RV or RandomProcess
            Indicator that is 1 where equal, 0 otherwise.

        Examples
        --------
        >>> X = RV(Binomial(10, 0.5))
        >>> (X == 5).draw()
        """
        op_func = self._comparison_factory(lambda x, y: x == y)
        return op_func(self, other)

    def __ne__(self, other):
        """Return an indicator of element-wise inequality (e.g., X != 0).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Value to compare against.

        Returns
        -------
        RV or RandomProcess
            Indicator that is 1 where not equal, 0 otherwise.

        Examples
        --------
        >>> X = RV(Binomial(10, 0.5))
        >>> (X != 5).draw()
        """
        op_func = self._comparison_factory(lambda x, y: x != y)
        return op_func(self, other)

    def __lt__(self, other):
        """Return an indicator of element-wise less than (e.g., X < 0).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Value to compare against.

        Returns
        -------
        RV or RandomProcess
            Indicator that is 1 where less than, 0 otherwise.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (X < 0).draw()
        """
        op_func = self._comparison_factory(lambda x, y: x < y)
        return op_func(self, other)

    def __le__(self, other):
        """Return an indicator of element-wise less than or equal (e.g., X <= 0).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Value to compare against.

        Returns
        -------
        RV or RandomProcess
            Indicator that is 1 where less than or equal, 0 otherwise.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (X <= 0).draw()
        """
        op_func = self._comparison_factory(lambda x, y: x <= y)
        return op_func(self, other)

    def __gt__(self, other):
        """Return an indicator of element-wise greater than (e.g., X > 0).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Value to compare against.

        Returns
        -------
        RV or RandomProcess
            Indicator that is 1 where greater than, 0 otherwise.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (X > 0).draw()
        """
        op_func = self._comparison_factory(lambda x, y: x > y)
        return op_func(self, other)

    def __ge__(self, other):
        """Return an indicator of element-wise greater than or equal (e.g., X >= 0).

        Parameters
        ----------
        other : RV, RandomProcess, or scalar
            Value to compare against.

        Returns
        -------
        RV or RandomProcess
            Indicator that is 1 where greater than or equal, 0 otherwise.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> (X >= 0).draw()
        """
        op_func = self._comparison_factory(lambda x, y: x >= y)
        return op_func(self, other)


class Statistical:
    """Mixin providing statistical methods (mean, var, etc.) for subclasses.

    Subclasses must implement the _statistic_factory and
    _multivariate_statistic_factory methods, which specify how
    (univariate) statistics (e.g., mean and variance), as well as
    multivariate statistics (e.g., covariance and correlation)
    are calculated on the object.
    """

    def sum(self):
        r"""Calculate the sum.

        .. math:: \sum_{i=1}^n x_i

        Returns
        -------
        float
            The sum of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).sum()
        """
        op_func = self._statistic_factory(np.sum)
        return op_func(self)

    def mean(self):
        r"""Calculate the mean (a.k.a. average).

        The mean, or average, is a measure of center.

        .. math:: \mu = \frac{1}{n} \sum_{i=1}^n x_i

        Returns
        -------
        float
            The mean of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).mean()
        """
        op_func = self._statistic_factory(np.mean)
        return op_func(self)

    def quantile(self, q):
        r"""Calculate a specified quantile (percentile).

        The (100q)th quantile is the value x such that

        .. math:: \frac{\#\{ i: x_i \leq x \}}{n} = q

        Parameters
        ----------
        q : float
            A number between 0 and 1 specifying the desired
            quantile or percentile.

        Returns
        -------
        float
            The (100q)th quantile of the values.

        Raises
        ------
        ValueError
            If q is not between 0 and 1.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).quantile(0.25)
        """
        if not (0 <= q <= 1):
            raise ValueError(
                f"q must be between 0 and 1, got {q}. "
                "For example, use quantile(0.25) for the 25th percentile."
            )
        op_func = self._statistic_factory(
            lambda a, axis=None: np.quantile(a, q=q, axis=axis)
        )
        return op_func(self)

    def percentile(self, q):
        r"""Calculate a specified percentile.

        Alias for :meth:`quantile`.

        Parameters
        ----------
        q : float
            A number between 0 and 1 specifying the desired
            quantile or percentile.

        Returns
        -------
        float
            The (100q)th percentile of the values.

        Raises
        ------
        ValueError
            If q is not between 0 and 1.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).percentile(0.25)
        """
        return self.quantile(q)

    def iqr(self):
        r"""Calculate the interquartile range (IQR).

        The IQR is the 75th percentile minus the 25th percentile.

        Returns
        -------
        float
            The interquartile range.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).iqr()
        """
        return self.quantile(0.75) - self.quantile(0.25)

    def median(self):
        r"""Calculate the median.

        The median is the middle number in a *sorted* list.
        It is a measure of center.

        Returns
        -------
        float
            The median of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).median()
        """
        op_func = self._statistic_factory(np.median)
        return op_func(self)

    def std(self):
        r"""Calculate the standard deviation.

        The standard deviation is the square root of the variance.
        It is a measure of spread.

        .. math::

            \sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2}

        Returns
        -------
        float
            The standard deviation of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).std()
        """
        op_func = self._statistic_factory(np.std)
        return op_func(self)

    def sd(self):
        r"""Calculate the standard deviation.

        Alias for :meth:`std`.

        Returns
        -------
        float
            The standard deviation of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).sd()
        """
        return self.std()

    def var(self):
        r"""Calculate the variance.

        The variance is the average squared distance from the mean.
        It is a measure of spread.

        .. math::

            \sigma^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2

        Returns
        -------
        float
            The variance of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).var()
        """
        op_func = self._statistic_factory(np.var)
        return op_func(self)

    def skew(self):
        r"""Calculate the skewness.

        Returns
        -------
        float
            The skewness of the values.

        Examples
        --------
        >>> X = RV(Exponential(1))
        >>> X.sim(10000).skew()
        """
        op_func = self._statistic_factory(stats.skew)
        return op_func(self)

    def skewness(self):
        """Calculate the skewness.

        Alias for :meth:`skew`.

        Returns
        -------
        float
            The skewness of the values.

        Examples
        --------
        >>> X = RV(Exponential(1))
        >>> X.sim(10000).skewness()
        """
        return self.skew()

    def kurtosis(self):
        r"""Calculate the kurtosis.

        Returns
        -------
        float
            The kurtosis of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).kurtosis()
        """
        op_func = self._statistic_factory(stats.kurtosis)
        return op_func(self)

    def max(self):
        r"""Calculate the maximum.

        Returns
        -------
        float
            The maximum of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).max()
        """
        op_func = self._statistic_factory(np.max)
        return op_func(self)

    def min(self):
        r"""Calculate the minimum.

        Returns
        -------
        float
            The minimum of the values.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).min()
        """
        op_func = self._statistic_factory(np.min)
        return op_func(self)

    def min_max_diff(self):
        r"""Calculate the difference between the min and max.

        .. math:: \max - \min

        The min-max diff is also called the range. It is
        a measure of spread.

        Returns
        -------
        float
            The difference between the min and the max.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).min_max_diff()
        """
        return self.max() - self.min()

    def cov(self):
        r"""Calculate the pairwise covariances.

        The covariance is a measure of the relationship between
        two variables. The sign indicates the direction of the
        relationship.

        .. math::

            \sigma_{XY} = \frac{1}{n} \sum_{i=1}^n
            (x_i - \mu_X)(y_i - \mu_Y)

        Returns
        -------
        float or numpy.ndarray
            Pairwise covariances between all dimensions.
            A scalar for 2 dimensions; a matrix for more.

        Examples
        --------
        >>> X, Y = RV(Normal(0, 1)), RV(Normal(0, 1))
        >>> (X & Y).sim(10000).cov()
        """
        op_func = self._multivariate_statistic_factory(
            lambda a: np.cov(a, rowvar=False, ddof=0)
        )
        return op_func(self)

    def corr(self):
        r"""Calculate the pairwise correlations.

        The correlation is the covariance normalized by the
        standard deviations.

        .. math::

            \rho_{XY} = \frac{1}{n} \sum_{i=1}^n
            \frac{x_i - \mu_X}{\sigma_X}
            \frac{y_i - \mu_Y}{\sigma_Y}

        Returns
        -------
        float or numpy.ndarray
            Pairwise correlations between all dimensions.
            A scalar for 2 dimensions; a matrix for more.

        Examples
        --------
        >>> X, Y = RV(Normal(0, 1)), RV(Normal(0, 1))
        >>> (X & Y).sim(10000).corr()
        """
        op_func = self._multivariate_statistic_factory(
            lambda a: np.corrcoef(a, rowvar=False)
        )
        return op_func(self)

    def corrcoef(self):
        r"""Calculate the pairwise correlations.

        Alias for :meth:`corr`.

        Returns
        -------
        float or numpy.ndarray
            Pairwise correlations between all dimensions.
            A scalar for 2 dimensions; a matrix for more.

        Examples
        --------
        >>> X, Y = RV(Normal(0, 1)), RV(Normal(0, 1))
        >>> (X & Y).sim(10000).corrcoef()
        """
        return self.corr()


class Logical:
    """Mixin providing logical operators (and, or, not) for subclasses.

    Subclasses must implement the _logical_factory method, which
    specifies how the logical operator operates on two objects
    of that type.
    """

    def __and__(self, other):
        """Return the logical AND of two events (e.g., A & B).

        Parameters
        ----------
        other : Event
            Right-hand event.

        Returns
        -------
        Event
            A new event that occurs when both events occur.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice(range(1, 7)))
        >>> A = Event(die, lambda x: x > 2)
        >>> B = Event(die, lambda x: x < 5)
        >>> (A & B).draw()  # doctest: +SKIP
        """
        op_func = self._logical_factory(lambda x, y: x and y)
        return op_func(self, other)

    def __or__(self, other):
        """Return the logical OR of two events (e.g., A | B).

        Parameters
        ----------
        other : Event
            Right-hand event.

        Returns
        -------
        Event
            A new event that occurs when either event occurs.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice(range(1, 7)))
        >>> A = Event(die, lambda x: x < 2)
        >>> B = Event(die, lambda x: x > 5)
        >>> (A | B).draw()  # doctest: +SKIP
        """
        op_func = self._logical_factory(lambda x, y: x or y)
        return op_func(self, other)

    def __invert__(self):
        """Return the logical NOT of an event (e.g., ~A).

        Returns
        -------
        Event
            A new event that occurs when the original event does not occur.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice(range(1, 7)))
        >>> A = Event(die, lambda x: x > 3)
        >>> (~A).draw()  # doctest: +SKIP
        """
        op_func = self._logical_factory(lambda x: not x)
        return op_func(self)


class Filterable:
    """Mixin providing filtering and counting methods for subclasses.

    Subclasses must implement the filter method, which specifies how to
    construct a new instance containing only those elements that satisfy
    a given criterion.
    """

    def filter_eq(self, value):
        """Return all elements equal to a given value.

        Parameters
        ----------
        value : any
            The value to filter by.

        Returns
        -------
        Results
            All elements equal to value.

        Examples
        --------
        >>> X = RV(Binomial(10, 0.5))
        >>> X.sim(10000).filter_eq(5)
        """
        return self.filter(lambda x: x == value)

    def filter_neq(self, value):
        """Return all elements not equal to a given value.

        Parameters
        ----------
        value : any
            The value to filter by.

        Returns
        -------
        Results
            All elements not equal to value.

        Examples
        --------
        >>> X = RV(Binomial(10, 0.5))
        >>> X.sim(10000).filter_neq(5)
        """
        return self.filter(lambda x: x != value)

    def filter_lt(self, value):
        """Return all elements less than a given value.

        For elements less than or equal to a value,
        use :meth:`filter_leq`.

        Parameters
        ----------
        value : any
            The value to filter by.

        Returns
        -------
        Results
            All elements less than value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).filter_lt(0)
        """
        return self.filter(lambda x: x < value)

    def filter_leq(self, value):
        """Return all elements less than or equal to a given value.

        For elements strictly less than a value,
        use :meth:`filter_lt`.

        Parameters
        ----------
        value : any
            The value to filter by.

        Returns
        -------
        Results
            All elements less than or equal to value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).filter_leq(0)
        """
        return self.filter(lambda x: x <= value)

    def filter_gt(self, value):
        """Return all elements greater than a given value.

        For elements greater than or equal to a value,
        use :meth:`filter_geq`.

        Parameters
        ----------
        value : any
            The value to filter by.

        Returns
        -------
        Results
            All elements greater than value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).filter_gt(0)
        """
        return self.filter(lambda x: x > value)

    def filter_geq(self, value):
        """Return all elements greater than or equal to a given value.

        For elements strictly greater than a value,
        use :meth:`filter_gt`.

        Parameters
        ----------
        value : any
            The value to filter by.

        Returns
        -------
        Results
            All elements greater than or equal to value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).filter_geq(0)
        """
        return self.filter(lambda x: x >= value)

    def count(self, func=lambda x: True):
        """Count the number of elements satisfying a criterion.

        Parameters
        ----------
        func : callable, optional
            A function that takes an element and returns a bool.
            Only elements for which func returns True are counted.
            Default counts all elements.

        Returns
        -------
        int
            Number of elements for which func returns True.

        Raises
        ------
        TypeError
            If func is not callable.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).count(lambda x: x > 0)
        """
        if not callable(func):
            raise TypeError(
                "func must be a callable (e.g., a lambda or function), "
                f"but got {type(func).__name__}. "
                "For example, use count(lambda x: x > 0)."
            )
        return len(self.filter(func))

    def count_eq(self, value):
        """Count the number of elements equal to a given value.

        Parameters
        ----------
        value : any
            The value to count.

        Returns
        -------
        int
            Number of elements equal to value.

        Examples
        --------
        >>> X = RV(Binomial(10, 0.5))
        >>> X.sim(10000).count_eq(5)
        """
        return len(self.filter_eq(value))

    def count_neq(self, value):
        """Count the number of elements not equal to a given value.

        Parameters
        ----------
        value : any
            The value to exclude.

        Returns
        -------
        int
            Number of elements not equal to value.

        Examples
        --------
        >>> X = RV(Binomial(10, 0.5))
        >>> X.sim(10000).count_neq(5)
        """
        return len(self.filter_neq(value))

    def count_lt(self, value):
        """Count the number of elements less than a given value.

        For the count of elements less than or equal to a value,
        use :meth:`count_leq`.

        Parameters
        ----------
        value : any
            The value to compare against.

        Returns
        -------
        int
            Number of elements less than value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).count_lt(0)
        """
        return len(self.filter_lt(value))

    def count_leq(self, value):
        """Count the number of elements less than or equal to a value.

        For the count of elements strictly less than a value,
        use :meth:`count_lt`.

        Parameters
        ----------
        value : any
            The value to compare against.

        Returns
        -------
        int
            Number of elements less than or equal to value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).count_leq(0)
        """
        return len(self.filter_leq(value))

    def count_gt(self, value):
        """Count the number of elements greater than a given value.

        For the count of elements greater than or equal to a value,
        use :meth:`count_geq`.

        Parameters
        ----------
        value : any
            The value to compare against.

        Returns
        -------
        int
            Number of elements greater than value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).count_gt(0)
        """
        return len(self.filter_gt(value))

    def count_geq(self, value):
        """Count the number of elements greater than or equal to a value.

        For the count of elements strictly greater than a value,
        use :meth:`count_gt`.

        Parameters
        ----------
        value : any
            The value to compare against.

        Returns
        -------
        int
            Number of elements greater than or equal to value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> X.sim(10000).count_geq(0)
        """
        return len(self.filter_geq(value))


class Transformable:
    """Mixin providing transformation methods for subclasses.

    Subclasses must implement the apply method, which specifies how to
    apply a function to the object.
    """

    def __abs__(self):
        """Return the absolute value applied element-wise.

        Called by Python's built-in ``abs()``.

        Returns
        -------
        RV or RandomProcess
            Element-wise absolute value.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> abs(X).draw()
        """
        return self.apply(abs)

    def __round__(self, ndigits=None):
        """Return values rounded to the nearest integer element-wise.

        Called by Python's built-in ``round()``.

        Parameters
        ----------
        ndigits : int, optional
            Number of decimal places. Must be a non-negative integer.
            If None, rounds to the nearest integer.

        Returns
        -------
        RV or RandomProcess
            Element-wise rounded values.

        Raises
        ------
        ValueError
            If ndigits is not a non-negative integer.

        Examples
        --------
        >>> X = RV(Normal(0, 1))
        >>> round(X).draw()
        >>> round(X, 2).draw()
        """
        if ndigits is not None and (not isinstance(ndigits, int) or ndigits < 0):
            raise ValueError(
                f"ndigits must be a non-negative integer, got {ndigits!r}."
            )
        if ndigits is None:
            return self.apply(round)
        return self.apply(lambda x: round(x, ndigits))

    def __floor__(self):
        """Return the floor (round down) of each value element-wise.

        Called by ``math.floor()``.

        Returns
        -------
        RV or RandomProcess
            Element-wise floor values.

        Examples
        --------
        >>> import math
        >>> X = RV(Normal(0, 1))
        >>> math.floor(X).draw()
        """
        return self.apply(math.floor)

    def __ceil__(self):
        """Return the ceiling (round up) of each value element-wise.

        Called by ``math.ceil()``.

        Returns
        -------
        RV or RandomProcess
            Element-wise ceiling values.

        Examples
        --------
        >>> import math
        >>> X = RV(Normal(0, 1))
        >>> math.ceil(X).draw()
        """
        return self.apply(math.ceil)
