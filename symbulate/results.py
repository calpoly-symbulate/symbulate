"""Data structures for storing the results of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""

import difflib
import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import Affine2D

from .base import (
    Arithmetic,
    Statistical,
    Comparable,
    Logical,
    Filterable,
    Transformable,
)
from .plot import (
    configure_axes,
    init_color,
    get_next_color,
    is_discrete,
    count_var,
    compute_density,
    add_colorbar,
    setup_ticks,
    make_tile,
    make_violin,
    make_marginal_impulse,
    make_density2D,
)
from .result import Scalar, Vector, TimeFunction, is_number, is_numeric_vector
from .table import Table

try:
    seaborn_colorblind_closest = difflib.get_close_matches(
        "seaborn-colorblind", plt.style.available, cutoff=0.7
    )
    stylesheet = seaborn_colorblind_closest[0]
except IndexError:
    stylesheet = "ggplot"

plt.style.use(stylesheet)


def _is_hashable(obj):
    """Check whether an object is hashable.

    Parameters
    ----------
    obj : object
        The object to test.

    Returns
    -------
    bool
        True if ``obj`` has a ``__hash__`` attribute,
        False otherwise.
    """
    return hasattr(obj, "__hash__")


def _is_boolean_vector(vector):
    """Check whether every element of an iterable is boolean.

    Parameters
    ----------
    vector : iterable
        The iterable to inspect.

    Returns
    -------
    bool
        True if every element is an instance of ``bool`` or
        ``numpy.bool_``, False otherwise.
    """
    return all(isinstance(x, (bool, np.bool_)) for x in vector)


class Results(Arithmetic, Statistical, Comparable, Logical, Filterable, Transformable):
    """
    Container for the outcomes of a simulation from a probability
    space.

    Stores a list of simulation outcomes and exposes methods for
    filtering, tabulating, and transforming them. Arithmetic,
    comparison, and logical operations are applied element-wise
    across all outcomes via the mixin superclasses.

    Parameters
    ----------
    results : iterable
        Simulation outcomes to store.
    sim_id : float, optional
        Identifier for the simulation run. If None, a timestamp
        is generated automatically.

    Attributes
    ----------
    results : list
        The stored simulation outcomes.
    sim_id : float
        Timestamp identifying the simulation run that produced
        these results.

    Methods
    -------
    apply(func)
        Apply a function to each simulation outcome.
    get(n)
        Return the outcome of the nth simulation.
    tabulate(outcomes=None, normalize=False)
        Count or estimate frequencies of outcomes.
    filter(filt)
        Return outcomes satisfying a criterion.
    plot()
        Plot the simulation results when supported.

    See Also
    --------
    RVResults : Container for outcomes of a random variable.

    Examples
    --------
    Simulate five outcomes from a probability space:

    >>> from symbulate import *
    >>> P = BoxModel(["H", "T"])
    >>> results = P.sim(5)
    >>> len(results)
    5
    """

    def __init__(self, results, sim_id=None):
        self.results = list(results)
        self.sim_id = time.time() if sim_id is None else sim_id

    def apply(self, func):
        """Apply a function to each outcome of a simulation.

        Parameters
        ----------
        func : callable
            A function to apply to each outcome.

        Returns
        -------
        Results
            A Results object of the same length, where each
            outcome is the result of applying ``func`` to each
            outcome from the original Results object.

        See Also
        --------
        filter : Return outcomes satisfying a criterion.
        tabulate : Count or estimate frequencies of outcomes.

        Examples
        --------
        Double every outcome of a die roll simulation:

        >>> from symbulate import *
        >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
        >>> sims = X.sim(5)
        >>> sims.apply(lambda x: x * 2)  #random
        Index  Result
        0      4
        """
        return type(self)([func(result) for result in self.results], self.sim_id)

    def __getitem__(self, n):
        """
        Return selected dimensions or filtered results.

        Parameters
        ----------
        n : int, slice, array-like of int, or Results
            Index, slice, collection of indices, or a boolean
            Results object used as a mask.

        Returns
        -------
        Results
            A Results object containing the selected values or
            filtered outcomes.

        See Also
        --------
        get : Return the outcome of the nth simulation.
        filter : Return outcomes satisfying a criterion.

        Examples
        --------
        Select the first component of each 2-D outcome:

        >>> from symbulate import *
        >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6], size=2))
        >>> sims = X.sim(5)
        >>> sims[0]  #random
        Index  Result
        0      3
        """
        # if n is a Results object, use it as a boolean mask
        if isinstance(n, Results):
            return self.filter(n)
        # if n is a numeric array of values, return a Results
        # object with those dimensions
        elif is_numeric_vector(n):
            return self.apply(lambda result: type(result)(result[i] for i in n))
        # otherwise, return the nth value of every simulation
        return self.apply(lambda result: result[n])

    def __iter__(self):
        """
        Iterate over the simulation outcomes.

        Yields
        ------
        object
            Next simulation outcome.

        Examples
        --------
        >>> from symbulate import *
        >>> P = BoxModel(["H", "T"])
        >>> results = P.sim(3)
        >>> for outcome in results:
        ...     print(outcome)  # doctest: +SKIP
        """
        for result in self.results:
            yield result

    def __len__(self):
        """
        Return the number of simulation outcomes.

        Returns
        -------
        int
            Number of stored outcomes.

        Examples
        --------
        >>> from symbulate import *
        >>> P = BoxModel([1, 2, 3, 4, 5, 6])
        >>> results = P.sim(100)
        >>> len(results)
        100
        """
        return len(self.results)

    def get(self, n):
        """Return the outcome of the nth simulation.

        Parameters
        ----------
        n : int, slice, or array-like of int
            Index or indices of the simulation results to
            retrieve.

        Returns
        -------
        object or Results
            The outcome of the nth simulation. If ``n`` is an
            array-like of integers, returns a Results object
            containing those results.

        Notes
        -----
        Although a Results object can be iterated like a list,
        indexing with ``x[n]`` does not return the nth
        simulation result. Instead, it returns a Results object
        containing the nth *dimension* of every simulation.
        Use ``get(n)`` to retrieve the full outcome of the nth
        simulation.

        See Also
        --------
        Results.__getitem__ :
            Index into each simulation outcome by dimension.

        Examples
        --------
        Retrieve the first simulation outcome:

        >>> from symbulate import *
        >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
        >>> sims = X.sim(5)
        >>> sims.get(0)  #random
        3

        Retrieve a slice of outcomes:

        >>> sims.get(slice(0, 3))  #random
        Index  Result
        0      3
        """

        # if n is a numeric array, return a Results object with those results
        if is_numeric_vector(n):
            return type(self)(self.results[i] for i in n)
        # otherwise, return the nth result (this also works when n is a slice)
        return self.results[n]

    def _get_counts(self):
        """Tally the number of times each outcome appears.

        Unhashable outcomes such as lists are converted to
        tuples when all elements are hashable, or to strings
        otherwise, before being used as dictionary keys.

        Returns
        -------
        dict
            A mapping from each observed outcome to its integer
            count.
        """
        counts = {}
        for result in self.results:
            if _is_hashable(result):
                outcome = result
            elif isinstance(result, list) and all(_is_hashable(x) for x in result):
                outcome = tuple(result)
            else:
                outcome = str(result)
            if outcome in counts:
                counts[outcome] += 1
            else:
                counts[outcome] = 1
        return counts

    def tabulate(self, outcomes=None, normalize=False):
        """Count how many times each outcome appears.

        Parameters
        ----------
        outcomes : list, optional
            Outcomes to include in the table. By default,
            tabulates all outcomes that appear in the Results.
            Use this option to include outcomes that may not
            appear in the Results.
        normalize : bool, default False
            If True, return relative frequencies. If False,
            return counts.

        Returns
        -------
        Table
            A Table mapping each observed outcome to its count
            or relative frequency.

        See Also
        --------
        filter : Return outcomes satisfying a criterion.

        Examples
        --------
        Tabulate outcomes from 100 coin flips:

        >>> from symbulate import *
        >>> P = BoxModel(["H", "T"])
        >>> results = P.sim(100)
        >>> results.tabulate()  #random

        Include an outcome that may not have appeared:

        >>> results.tabulate(outcomes=["H", "T", "E"])  #random

        Show relative frequencies instead of counts:

        >>> results.tabulate(normalize=True)  #random
        """
        return Table(self._get_counts(), outcomes, normalize)

    # The Filterable superclass will use this to define all of the
    # .filter_*() and .count_*() methods.
    def filter(self, filt):
        """Return outcomes satisfying the given criterion.

        Parameters
        ----------
        filt : callable or Results
            Either a function that takes an outcome and returns
            a boolean, or a Results object of booleans the same
            length as this Results object.

        Returns
        -------
        Results
            A Results object containing only the outcomes for
            which ``filt`` returns True (or where ``filt`` is
            True).

        Raises
        ------
        Exception
            If ``filt`` is a Results object from a different
            simulation (different ``sim_id``).
        ValueError
            If ``filt`` is a Results object of a different
            length, or contains non-boolean values.
        TypeError
            If ``filt`` is neither callable nor a boolean
            Results object.

        See Also
        --------
        tabulate : Count or estimate frequencies of outcomes.

        Examples
        --------
        Filter using a function:

        >>> from symbulate import *
        >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
        >>> sims = X.sim(100)
        >>> sims.filter(lambda x: x > 4)  #random

        Filter using a boolean Results object from the same
        simulation:

        >>> sims.filter(sims > 4)  #random
        """
        if isinstance(filt, Results):
            if self.sim_id != filt.sim_id:
                raise Exception(
                    "In order to filter one Results object "
                    "by another, they must come from the "
                    "same simulation."
                )
            if len(filt) != len(self):
                raise ValueError(
                    "Filter must be the same length as the " "Results object."
                )
            if not _is_boolean_vector(filt):
                raise ValueError("Every element in the filter must be a boolean.")
            return type(self)(x for x, cond in zip(self, filt) if cond)
        elif callable(filt):
            return type(self)(x for x in self if filt(x))
        else:
            raise TypeError(
                "A filter must be either a function or a "
                "boolean Results object of the same length."
            )

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):
        """Create an element-wise binary operation on Results.

        Parameters
        ----------
        op : callable
            A binary operator to apply element-wise.

        Returns
        -------
        callable
            A function that applies ``op`` element-wise between
            this Results object and another operand (either
            another Results of the same length from the same
            simulation, or a scalar).
        """

        def _op_func(self, other):
            if isinstance(other, Results):
                if len(self) != len(other):
                    raise Exception("Results objects must be of the " "same length.")
                if self.sim_id != other.sim_id:
                    raise Exception(
                        "Results objects must come from the " "same simulation."
                    )
                return type(self)([op(x, y) for x, y in zip(self, other)], self.sim_id)
            else:
                return self.apply(lambda x: op(x, other))

        return _op_func

    # The Comparison superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    def _comparison_factory(self, op):
        """Create an element-wise comparison operation on Results.

        Delegates directly to ``_operation_factory``.

        Parameters
        ----------
        op : callable
            A binary comparison operator to apply element-wise.

        Returns
        -------
        callable
            A function that applies ``op`` element-wise between
            this Results object and another operand.

        See Also
        --------
        _operation_factory :
            Create an element-wise binary operation.
        """
        return self._operation_factory(op)

    # The Statistical superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    def _statistic_factory(self, _):
        """Raise an error for statistical operations on Results.

        Statistical functions are only available for simulations
        of random variables. This method always raises an
        Exception, directing the user to define an RV first.

        Parameters
        ----------
        _ : object
            Ignored. Accepted for interface compatibility with
            the Statistical mixin.

        Raises
        ------
        Exception
            Always raised; ``Results`` does not support
            statistical operations.

        See Also
        --------
        RVResults._statistic_factory :
            Statistical operations for random variable results.
        """
        raise Exception(
            "Statistical functions are only available "
            "for simulations of random variables. "
            "Define a RV on this probability space "
            "and then try again."
        )

    def _multivariate_statistic_factory(self, op):
        """Raise an error for multivariate statistics on Results.

        Delegates to ``_statistic_factory``, which always raises
        an Exception for base Results objects.

        Parameters
        ----------
        op : callable
            Ignored. Accepted for interface compatibility.

        Raises
        ------
        Exception
            Always raised via ``_statistic_factory``.

        See Also
        --------
        _statistic_factory :
            Error handler for statistical operations.
        RVResults._multivariate_statistic_factory :
            Multivariate statistics for RV results.
        """
        self._statistic_factory(op)

    # The Logical superclass will use this to define the three
    # logical operations: and (&), or (|), not (~).
    def _logical_factory(self, op):
        """Create an element-wise logical operation on Results.

        Parameters
        ----------
        op : callable
            A logical operator (``and``, ``or``, or ``not``)
            to apply element-wise.

        Returns
        -------
        callable
            A function that applies ``op`` element-wise and
            validates that the involved Results objects contain
            only boolean values.

        Raises
        ------
        ValueError
            If ``self`` or ``other`` contains non-boolean
            values.
        TypeError
            If ``other`` is not a Results object (for binary
            logical operations).
        """

        def _op_func(self, other=None):
            # check that the vector only contains booleans
            if not _is_boolean_vector(self):
                raise ValueError(
                    "Logical operations are only defined for "
                    "boolean (True/False) Results objects."
                )
            # other will be None when op is the "not" operator
            if other is None:
                return Results([op(x) for x in self], self.sim_id)
            else:
                if isinstance(other, Results):
                    if self.sim_id != other.sim_id:
                        raise Exception(
                            "Results objects must come " "from the same simulation."
                        )
                    if not _is_boolean_vector(other):
                        raise ValueError(
                            "Logical operations are only defined for "
                            "boolean (True/False) Results objects."
                        )
                else:
                    raise TypeError(
                        "Logical operations are only defined "
                        "between two Results, not between a Result "
                        "and a %s." % type(other).__name__
                    )
                return Results([op(x, y) for x, y in zip(self, other)], self.sim_id)

        return _op_func

    def plot(self):
        """
        Plot the simulation results.

        Raises
        ------
        Exception
            If the results do not correspond to simulations of a
            random variable.
        """
        raise Exception(
            "Only simulations of random variables (RV) "
            "can be plotted, but you simulated from a "
            "probability space. You must first define a RV "
            "on your probability space and simulate it. "
            "Then call .plot() on those simulations."
        )

    def __repr__(self):
        """
        Return a string representation of the results.

        Returns
        -------
        str
            Tabular representation of the stored simulation
            outcomes. Shows all rows for 11 or fewer results;
            truncates to the first 9 and the last row otherwise.
        """

        i_last = len(self) - 1
        max_index_length = len(str(i_last))

        if max_index_length <= 5:
            index_header_space = ""
            index_value_space = " " * 4
        else:
            index_header_space = " " * (max_index_length - 5)
            index_value_space = " " * (max_index_length - 1)

        table_rows = []

        table_rows.append(f"Index{index_header_space} Result")

        for i, result in enumerate(self.results):
            table_rows.append(f"{str(i)}{index_value_space} {str(result)}")

            if len(self) > 9 and i >= 8:
                index_value_space = " " * (5 - len(str(i_last)))

                if len(self) > 11:
                    table_rows.append(
                        f"{'.' * max_index_length}{index_value_space} "
                        f"{'.' * len(str(self.get(i_last)))}"
                    )
                elif len(self) == 11:
                    table_rows.append(
                        f"{str(i_last - 1)}{' ' * (5 - len(str(i_last - 1)))} "
                        f"{str(self.get(i_last - 1))}"
                    )

                table_rows.append(
                    f"{str(i_last)}{index_value_space} {str(self.get(i_last))}"
                )
                break

        return "\n".join(table_rows)

    def _repr_html_(self):
        """
        Return an HTML table representation of the results.

        Returns
        -------
        str
            An HTML string rendering simulation outcomes in a
            two-column table (Index, Result). Displays the first
            nine rows; if there are more, appends a ``...`` row
            followed by the last entry.
        """

        table_template = """
    <table>
      <thead>
        <th width="10%">Index</th>
        <th width="90%">Result</th>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
        """
        row_template = """
        <tr>
          <td>%s</td><td>%s</td>
        </tr>
        """

        def _truncate(result):
            if len(result) > 100:
                return result[:100] + "..."
            return result

        table_body = ""
        for i, result in enumerate(self.results):
            table_body += row_template % (i, _truncate(str(result)))
            # if we've already printed 9 rows, skip to end
            if i >= 8:
                table_body += "<tr><td>...</td><td>...</td></tr>"
                i_last = len(self) - 1
                table_body += row_template % (i_last, _truncate(str(self.get(i_last))))
                break
        return table_template.format(table_body=table_body)


class RVResults(Results):
    """
    Container for simulation outcomes of a random variable.

    Extends ``Results`` with statistical methods and plotting
    functionality for numerical simulation outcomes.

    Parameters
    ----------
    results : iterable
        Simulation outcomes to store.
    sim_id : float, optional
        Identifier for the simulation run. If None, a timestamp
        is generated automatically.

    Attributes
    ----------
    dim : int or None
        Dimension of each simulation outcome, if consistent.
        Set to 1 for scalar outcomes, the vector length for
        vector outcomes, or None when outcomes are of mixed
        or indeterminate dimension.
    index_set : IndexSet or None
        Common index set when outcomes are ``TimeFunction``
        instances. None if outcomes are not time functions or
        if index sets differ across outcomes.

    Methods
    -------
    standardize()
        Standardize the results to have mean 0 and variance 1.
    tabulate(outcomes=None, normalize=False)
        Count or estimate frequencies of outcomes.
    plot(...)
        Visualize the simulation results.

    See Also
    --------
    Results : Container for probability space simulation outcomes.

    Examples
    --------
    Simulate 100 rolls of a fair die:

    >>> from symbulate import *
    >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
    >>> sims = X.sim(100)
    >>> float(sims.mean())  #random
    3.45
    """

    def __init__(self, results, sim_id=None):
        """
        Initialize simulation results for a random variable.

        Parameters
        ----------
        results : iterable
            Simulation outcomes to store.
        sim_id : float, optional
            Identifier for the simulation run. If None, a
            timestamp is generated automatically.
        """
        super().__init__(results, sim_id)
        init_color()
        # get type and dimension of the first result, if it exists
        iterresults = iter(self)
        try:
            first_result = next(iterresults)
        except StopIteration:
            return
        # determine the index set (if each realization is a TimeFunction)
        if isinstance(first_result, TimeFunction):
            self.index_set = first_result.index_set
        else:
            self.index_set = None
        # determine the dimension
        if is_number(first_result):
            self.dim = 1
        elif is_numeric_vector(first_result):
            self.dim = len(first_result)
        else:
            self.dim = None
        # iterate over remaining results, ensure they are consistent with the first
        for result in iterresults:
            if isinstance(result, TimeFunction) and result.index_set != self.index_set:
                self.index_set = None
            if (is_number(result) and self.dim != 1) or (
                is_numeric_vector(result) and self.dim != len(result)
            ):
                self.dim = None

    def _set_array(self):
        """Cache simulation outcomes as a NumPy array.

        Sets ``self.array`` to a NumPy array of the stored
        outcomes when they have consistent numeric dimension.
        No-ops if ``self.array`` is already set or if the
        outcomes are ``TimeFunction`` instances.

        Raises
        ------
        Exception
            If outcomes have inconsistent or non-numeric
            dimension (i.e. ``self.dim`` is None).
        """
        # check if it has already been set
        if hasattr(self, "array"):
            return
        # don't set array for TimeFunctions
        elif self.index_set is not None:
            return
        # otherwise set array
        elif self.dim is not None:
            self.array = np.asarray(self.results)
        else:
            raise Exception(
                "This operation is only possible with results "
                "of consistent dimension."
            )

    # The Statistical superclass will use this to define all of the
    # usual comparison operations (e.g., <, >, ==, !=, etc.).
    def _statistic_factory(self, op):
        """Create a summary statistic function for RV results.

        Dispatches to scalar, vector, or time-function
        computation depending on ``self.dim`` and
        ``self.index_set``.

        Parameters
        ----------
        op : callable
            A NumPy reduction function (e.g. ``np.mean``).

        Returns
        -------
        callable
            A zero-argument function that applies ``op`` to the
            stored outcomes and returns a ``Scalar``, ``Vector``,
            or ``TimeFunction`` as appropriate.

        Raises
        ------
        NotImplementedError
            If the outcomes have non-numeric or inconsistent
            dimension and are not ``TimeFunction`` instances.

        See Also
        --------
        Results._statistic_factory :
            Raises unconditionally for probability-space results.
        """

        def _op_func(self):
            self._set_array()
            if self.dim == 1:
                return Scalar(op(a=self.array))
            elif self.dim is not None:
                return Vector(op(a=self.array, axis=0))
            elif self.index_set is not None:

                def _func(t):
                    return _op_func(self[t])

                return TimeFunction.from_index_set(self.index_set, _func)
            raise NotImplementedError(
                "Statistics can only be calculated for numerical "
                "data of consistent dimension."
            )

        return _op_func

    def _multivariate_statistic_factory(self, op):
        """Create a multivariate summary statistic for RV results.

        Computes the statistic across two or more outcome
        dimensions. For 2-D outcomes, returns the scalar
        cross-term (e.g. covariance). For higher-dimensional
        outcomes, returns the full matrix.

        Parameters
        ----------
        op : callable
            A NumPy multivariate function (e.g. ``np.cov``).

        Returns
        -------
        callable
            A zero-argument function that applies ``op`` to the
            stored outcomes.

        Raises
        ------
        Exception
            If ``self.dim`` is 1; multivariate statistics
            require at least 2 dimensions.
        NotImplementedError
            If the outcomes have non-numeric or inconsistent
            dimension.

        See Also
        --------
        _statistic_factory :
            Univariate summary statistics for RV results.
        """

        def _op_func(self):
            self._set_array()
            if self.dim == 2:
                return op(self.array)[0, 1]
            elif self.dim > 2:
                return op(self.array)
            elif self.dim == 1:
                raise Exception(
                    "This multivariate statistic is only defined when "
                    "when there are at least 2 dimensions."
                )
            raise NotImplementedError(
                "Statistics can only be calculated for numerical "
                "data of consistent dimension."
            )

        return _op_func

    def standardize(self):
        """Standardize the results to have mean 0 and variance 1.

        Subtracts the sample mean and divides by the sample
        standard deviation.

        Returns
        -------
        RVResults
            A new RVResults object where each outcome has been
            shifted by the sample mean and scaled by the sample
            standard deviation.

        Raises
        ------
        Exception
            If the results cannot be standardized (e.g. outcomes
            have non-numeric or inconsistent dimension).

        See Also
        --------
        mean : Compute the sample mean of the results.
        std : Compute the sample standard deviation.

        Examples
        --------
        Standardize 1000 simulated die rolls:

        >>> from symbulate import *
        >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
        >>> sims = X.sim(1000)
        >>> standardized = sims.standardize()
        >>> abs(float(standardized.mean())) < 0.1
        True
        """
        self._set_array()
        if self.dim is not None:
            return (self - self.mean()) / self.std()
        else:
            raise Exception("Could not standardize the given results.")

    def tabulate(self, outcomes=None, normalize=False):
        """Count how many times each outcome appears.

        Parameters
        ----------
        outcomes : list, optional
            Outcomes to include in the table. By default,
            tabulates all outcomes that appear in the Results.
            Use this option to include outcomes that may not
            appear in the Results.
        normalize : bool, default False
            If True, return relative frequencies. If False,
            return counts.

        Returns
        -------
        Table
            A Table mapping each observed value to its count or
            relative frequency, labeled "Value" in the header.

        See Also
        --------
        Results.tabulate :
            Tabulate outcomes from a probability space.
        filter : Return outcomes satisfying a criterion.

        Examples
        --------
        Count outcomes from 50 fair coin flips encoded as 0/1:

        >>> from symbulate import *
        >>> X = RV(BoxModel([0, 1]))
        >>> sims = X.sim(50)
        >>> sims.tabulate()  #random

        Show relative frequencies:

        >>> sims.tabulate(normalize=True)  #random
        """
        return Table(self._get_counts(), outcomes, normalize, "Value")

    def plot(
        self, type=None, alpha=None, normalize=True, jitter=False, bins=None, **kwargs
    ):
        """Plot the simulated random variable results.

        Parameters
        ----------
        type : str, tuple, or list, optional
            Plot type or types to display. Valid values are
            ``"hist"``, ``"bar"``, ``"impulse"``, ``"density"``,
            ``"rug"``, ``"scatter"``, ``"tile"``, ``"violin"``,
            and ``"marginal"``. If None, selects automatically
            based on whether the data appear discrete.
        alpha : float, optional
            Transparency of plotted elements, between 0 and 1.
            Defaults to 0.5 for most plot types.
        normalize : bool, default True
            If True, plot relative frequencies or densities. If
            False, plot raw counts.
        jitter : bool, default False
            If True, add small random noise to discrete values
            to reduce overplotting.
        bins : int, optional
            Number of bins for histograms or tile plots.
            Defaults to 30 for 1-D histograms and 10 for tiles.
        **kwargs
            Additional keyword arguments passed to the
            underlying matplotlib plotting function.

        Raises
        ------
        Exception
            If ``type`` is not a recognized string, tuple, or
            list.

        See Also
        --------
        tabulate : Count or estimate frequencies of outcomes.

        Examples
        --------
        Plot a histogram of 1000 simulated die rolls:

        >>> from symbulate import *
        >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
        >>> X.sim(1000).plot()  # doctest: +SKIP

        Plot an impulse chart for a discrete distribution:

        >>> X.sim(1000).plot(type="impulse")  # doctest: +SKIP

        Plot a scatter plot for a bivariate random variable:

        >>> X2 = RV(BoxModel([1, 2, 3, 4, 5, 6], size=2))
        >>> X2.sim(500).plot(type="scatter")  # doctest: +SKIP
        """
        if type is not None:
            if isinstance(type, str):
                type = (type,)
            elif not isinstance(type, (tuple, list)):
                raise Exception("I don't know how to plot a " + str(type))

        if self.dim == 1:
            # make sure self.array, a Numpy array, has been set
            self._set_array()

            # determine plotting parameters
            counts = self._get_counts()
            discrete = is_discrete(counts.values())
            if type is None:
                type = ("impulse",) if discrete else ("hist",)
            if alpha is None:
                alpha = 0.5
            if bins is None:
                bins = 30
            n = len(self)

            # initialize figure
            fig = plt.gcf()
            ax = plt.gca()
            color = get_next_color(ax)

            if "density" in type:
                if discrete:
                    xs = sorted(list(counts.keys()))
                    probs = [counts[x] / n for x in xs]
                    ax.plot(xs, probs, marker="o", color=color, linestyle="-")
                    if len(type) == 1:
                        plt.ylabel("Relative Frequency")
                else:
                    density = compute_density(self.array)
                    xs = np.linspace(self.array.min(), self.array.max(), 1000)
                    ax.plot(xs, density(xs), linewidth=2, color=color)
                    if len(type) == 1 or (len(type) == 2 and "rug" in type):
                        plt.ylabel("Density")

            if "hist" in type or "bar" in type:
                ax.hist(
                    self.array,
                    bins=bins,
                    density=normalize,
                    color=color,
                    alpha=alpha,
                    **kwargs,
                )
                plt.ylabel("Density" if normalize else "Count")
            elif "impulse" in type:
                xs = list(counts.keys())
                freqs = list(counts.values())
                if normalize:
                    freqs = [freq / n for freq in freqs]
                if jitter:
                    a = 0.02 * (max(xs) - min(xs))
                    xs = [x + np.random.uniform(low=-a, high=a) for x in xs]
                # plot the impulses
                ax.vlines(xs, 0, freqs, color=color, alpha=alpha, **kwargs)
                configure_axes(
                    ax, xs, freqs, ylabel="Relative Frequency" if normalize else "Count"
                )
            if "rug" in type:
                xs = self.array
                if discrete:
                    noise_level = 0.002 * (self.array.max() - self.array.min())
                    xs = xs + np.random.normal(scale=noise_level, size=n)
                ax.plot(xs, [0.001] * n, "|", linewidth=5, color="k")
                if len(type) == 1:
                    setup_ticks([], [], ax.yaxis)
        elif self.dim == 2:
            # make sure self.array, a Numpy array, has been set
            self._set_array()
            x, y = self.array[:, 0], self.array[:, 1]

            x_count = count_var(x)
            y_count = count_var(y)
            x_height = x_count.values()
            y_height = y_count.values()
            discrete_x = is_discrete(x_height)
            discrete_y = is_discrete(y_height)

            if type is None:
                type = ("scatter",)
            if alpha is None:
                alpha = 0.5
            if bins is None:
                bins = 10 if "tile" in type else 30

            if "marginal" in type:
                fig = plt.gcf()
                gs = GridSpec(4, 4)
                ax = fig.add_subplot(gs[1:4, 0:3])
                ax_marg_x = fig.add_subplot(gs[0, 0:3])
                ax_marg_y = fig.add_subplot(gs[1:4, 3])
                color = get_next_color(ax)
                if "density" in type:
                    densityX = compute_density(x)
                    densityY = compute_density(y)
                    x_lines = np.linspace(min(x), max(x), 1000)
                    y_lines = np.linspace(min(y), max(y), 1000)
                    ax_marg_x.plot(
                        x_lines,
                        densityX(x_lines),
                        linewidth=2,
                        color=get_next_color(ax),
                    )
                    ax_marg_y.plot(
                        y_lines,
                        densityY(y_lines),
                        linewidth=2,
                        color=get_next_color(ax),
                        transform=Affine2D().rotate_deg(270) + ax_marg_y.transData,
                    )
                else:
                    if discrete_x:
                        make_marginal_impulse(
                            x_count, get_next_color(ax), ax_marg_x, alpha, "x"
                        )
                    else:
                        ax_marg_x.hist(
                            x,
                            color=get_next_color(ax),
                            density=normalize,
                            alpha=alpha,
                            bins=bins,
                        )
                    if discrete_y:
                        make_marginal_impulse(
                            y_count, get_next_color(ax), ax_marg_y, alpha, "y"
                        )
                    else:
                        ax_marg_y.hist(
                            y,
                            color=get_next_color(ax),
                            density=normalize,
                            alpha=alpha,
                            bins=bins,
                            orientation="horizontal",
                        )
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)
            else:
                fig = plt.gcf()
                ax = plt.gca()
                color = get_next_color(ax)

            nullfmt = NullFormatter()  # removes labels on fig

            if "scatter" in type:
                if jitter:
                    x = x + np.random.normal(
                        loc=0, scale=0.01 * (x.max() - x.min()), size=len(x)
                    )
                    y = y + np.random.normal(
                        loc=0, scale=0.01 * (y.max() - y.min()), size=len(y)
                    )
                ax.scatter(x, y, alpha=alpha, c=color, **kwargs)
            elif "hist" in type:
                histo = ax.hist2d(x, y, bins=bins, cmap="Blues")

                # When normalize=True, use density instead of counts
                if normalize:
                    caxes = add_colorbar(fig, type, histo[3], "Density")
                    # change scale to density instead of counts
                    plt.draw()
                    new_labels = []
                    for label in caxes.get_yticklabels():
                        new_labels.append(int(label.get_text()) / len(x))
                    caxes.set_yticklabels(new_labels)
                else:
                    caxes = add_colorbar(fig, type, histo[3], "Count")
            elif "density" in type:
                den = make_density2D(x, y, ax)
                add_colorbar(fig, type, den, "Density")
            elif "tile" in type:
                hm = make_tile(x, y, bins, discrete_x, discrete_y, ax)
                add_colorbar(fig, type, hm, "Relative Frequency")
            elif "violin" in type:
                if discrete_x and not discrete_y:
                    positions = sorted(list(x_count.keys()))
                    make_violin(self.array, positions, ax, "x", alpha)
                elif not discrete_x and discrete_y:
                    positions = sorted(list(y_count.keys()))
                    make_violin(self.array, positions, ax, "y", alpha)
        else:
            if alpha is None:
                alpha = np.log(2) / np.log(len(self) + 1)
            ax = plt.gca()
            color = get_next_color(ax)
            for result in self.results:
                result.plot(alpha=alpha, color=color, **kwargs)
            plt.xlabel("Index")
