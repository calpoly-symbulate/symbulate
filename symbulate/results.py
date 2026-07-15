"""Data structures for storing the results of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""

import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D

from .base import (
    Arithmetic,
    Statistical,
    Comparable,
    Logical,
    Filterable,
    Transformable,
    _build_mv_filter,
)
from .plot import (
    HIST_DEFAULT_BINS,
    auto_jitter_mode,
    classify_data,
    default_plot_type,
    get_next_color,
    jitter_suggestion_message,
    should_show_jitter_suggestion,
    should_show_suggestion,
    suggestion_message,
    count_var,
    compute_density,
    add_colorbar,
    make_bar,
    make_dotplot,
    make_density,
    make_density2D,
    make_ecdf,
    make_hist,
    make_hist2d,
    make_impulse,
    make_marginal_impulse,
    make_mosaic,
    make_segmented_density,
    make_segmented_hist,
    make_rug,
    make_scatter,
    make_segmented_rug,
    make_tile,
    make_violin,
    make_violinplot,
    make_boxplot,
    make_grouped_boxplot,
    SymbulatePlot,
)
from .result import Scalar, Vector, TimeFunction, is_number, is_numeric_vector
from .table import Table

# The package style (symbulate.mplstyle) is applied when .plot is
# imported above -- see plot.py and DECISIONS.md, "Decision: .mplstyle
# Standards".


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
    return getattr(obj, "__hash__", None) is not None


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


def _is_categorical_1d(results):
    """Check whether results are a 1D collection of categorical (string) values.

    Categorical outcomes -- e.g. ``BoxModel(["H", "T"])`` -- are not numbers,
    so an ``RVResults`` built from them gets ``dim=None`` and skips the
    ``dim == 1`` plotting branch. ``RVResults.plot()`` uses this to route them
    to the 1D categorical plot types (bar / dot plot / impulse) instead of the
    higher-dimensional catch-all, which would fail on non-plottable values.

    Parameters
    ----------
    results : iterable
        The stored simulation outcomes.

    Returns
    -------
    bool
        True if the outcomes are a 1D collection of strings/bytes, False
        otherwise (numeric, higher-dimensional, or non-string objects).
    """
    try:
        arr = np.asarray(list(results))
    except Exception:
        return False
    if arr.ndim != 1:
        return False
    if arr.dtype.kind in ("U", "S"):
        return True
    if arr.dtype.kind == "O":
        return all(isinstance(x, (str, bytes)) for x in arr.tolist())
    return False


def _sim_with_progress(draw_func, n, progress_delay=5.0, bar_width=30):
    """Run n draws, showing a live progress bar on stderr if simulation takes longer than progress_delay seconds.

    Once the bar is showing, it only redraws when the displayed percentage
    actually changes, not on every single draw. Redrawing every iteration
    was measured to add up to ~40x overhead on cheap draws (each redraw is a
    real stderr write + flush); this way, a run makes at most ~100 redraws
    total regardless of n, which is visually identical and costs roughly 2x
    baseline instead.
    """
    start = time.monotonic()
    showed_progress = False
    last_pct = -1
    draws = []

    def _render_bar(done):
        filled = int(bar_width * done / n)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = int(100 * done / n)
        sys.stderr.write(f"\rSimulating... [{bar}] {done}/{n} ({pct}%)")
        sys.stderr.flush()

    for i in range(n):
        draws.append(draw_func())
        if not showed_progress and time.monotonic() - start >= progress_delay:
            showed_progress = True
        if showed_progress:
            pct = int(100 * (i + 1) / n)
            if pct != last_pct:
                _render_bar(i + 1)
                last_pct = pct

    if showed_progress:
        _render_bar(n)
        sys.stderr.write("\n")
        sys.stderr.flush()

    return draws


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
        """
        Initialize simulation results from a probability space.
        """
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

        Raises
        ------
        TypeError
            If ``func`` is not callable.

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
        >>> sims.apply(lambda x: x * 2)  # doctest: +SKIP
        Index  Result
        0      4
        """
        if not callable(func):
            raise TypeError(
                "func must be a callable (e.g., a lambda or function), "
                f"but got {type(func).__name__}. "
                "For example, use apply(lambda x: x * 2)."
            )
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
        >>> sims[0]  # doctest: +SKIP
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
        >>> sims.get(0)  # doctest: +SKIP
        3

        Retrieve a slice of outcomes:

        >>> sims.get(slice(0, 3))  # doctest: +SKIP
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

    def tabulate(
        self, outcomes=None, normalize=False, bin=False, nbins=None, binwidth=None
    ):
        """Count how many times each outcome appears.

        Parameters
        ----------
        outcomes : list, optional
            Outcomes to include in the table. By default,
            tabulates all outcomes that appear in the Results.
            Use this option to include outcomes that may not
            appear in the Results. Ignored when ``bin=True``.
        normalize : bool, default False
            If True, return relative frequencies. If False,
            return counts.
        bin : bool, default False
            If True, group numeric results into equal-width bins
            for a compact summary of continuous values. If False,
            tabulate each distinct outcome exactly (the default;
            appropriate for discrete results).
        nbins : int, optional
            Number of equal-width bins to use when ``bin=True``.
            Defaults to 10. Cannot be used together with ``binwidth``.
        binwidth : float, optional
            Width of each bin when ``bin=True``; the number of bins
            is determined automatically from the data range. Cannot
            be used together with ``nbins``.

        Returns
        -------
        Table
            A Table mapping each observed outcome to its count
            or relative frequency. When ``bin=True``, the keys
            are interval labels of the form ``"[a, b)"`` for all
            bins except the last, which is ``"[a, b]"`` (closed
            on both ends so the maximum value is included).

        Raises
        ------
        ValueError
            If both ``nbins`` and ``binwidth`` are specified.
        UserWarning
            If ``nbins`` or ``binwidth`` are specified when ``bin=False``.

        See Also
        --------
        filter : Return outcomes satisfying a criterion.

        Examples
        --------
        Tabulate outcomes from 100 coin flips:

        >>> from symbulate import *
        >>> P = BoxModel(["H", "T"])
        >>> results = P.sim(100)
        >>> results.tabulate()  # doctest: +SKIP

        Include an outcome that may not have appeared:

        >>> results.tabulate(outcomes=["H", "T", "E"])  # doctest: +SKIP

        Show relative frequencies instead of counts:

        >>> results.tabulate(normalize=True)  # doctest: +SKIP

        Summarize continuous results with equal-width bins:

        >>> RV(Normal(0, 1)).sim(1000).tabulate(bin=True)  # doctest: +SKIP

        Use a custom number of bins:

        >>> RV(Normal(0, 1)).sim(1000).tabulate(bin=True, nbins=20)  # doctest: +SKIP

        Use a custom bin width:

        >>> RV(Normal(0, 1)).sim(1000).tabulate(bin=True, binwidth=0.5)  # doctest: +SKIP
        """
        if nbins is not None and binwidth is not None:
            raise ValueError(
                "Cannot specify both nbins and binwidth. "
                "Use nbins to set the number of bins, or binwidth to set the bin width."
            )
        if not bin and (nbins is not None or binwidth is not None):
            warnings.warn(
                "nbins and binwidth have no effect when bin=False.",
                UserWarning,
                stacklevel=2,
            )
        if bin:
            if binwidth is not None:
                data_range = max(self.results) - min(self.results)
                n = max(1, int(np.ceil(data_range / binwidth)))
            else:
                n = nbins if nbins is not None else 10
            counts, edges = np.histogram(self.results, bins=n)
            hash_map, labels = {}, []
            for i, count in enumerate(counts):
                closing = "]" if i == len(counts) - 1 else ")"
                label = f"[{edges[i]:.4g}, {edges[i + 1]:.4g}{closing}"
                labels.append(label)
                hash_map[label] = int(count)
            return Table(hash_map, labels, normalize, "Bin")
        return Table(self._get_counts(), outcomes, normalize)

    # The Filterable superclass will use this to define all of the
    # .filter_*() and .count_*() methods.
    def filter(self, *args):
        """Return outcomes satisfying the given criterion.

        Univariate
        ----------
        filter(func)
            Keep outcomes for which the callable ``func`` returns True.
        filter(bool_results)
            Keep outcomes where the corresponding element of a same-sim
            boolean Results object is True.

        Multivariate (joint distributions such as ``X & Y & Z``)
        ----------------------------------------------------------
        Pass one argument per component.  All component conditions are
        ANDed together.

        Per-component callables — each receives its own component value:

        >>> sims.filter(lambda x: x > 3, lambda y: y == 1)

        Per-component ``(op, value)`` tuples (no lambda required):

        >>> sims.filter(('>', 3), ('==', 1))

        Use ``None`` to skip a component position:

        >>> sims.filter(('>', 3), None, ('>', 0))

        Supported operators: ``'=='``, ``'!='``, ``'<'``, ``'<='``,
        ``'>'``, ``'>='``.

        Parameters
        ----------
        *args : callable, Results, (str, value) tuple, or None
            A single callable or boolean Results for univariate filtering,
            or multiple per-component conditions for multivariate filtering.

        Returns
        -------
        Results
            A Results object containing only the outcomes satisfying all
            specified conditions.

        Raises
        ------
        Exception
            If a Results filter comes from a different simulation.
        ValueError
            If a Results filter has a different length, non-boolean values,
            or an op-tuple uses an unrecognised operator string.
        TypeError
            If the arguments are neither callable, boolean Results, nor
            valid per-component conditions.

        See Also
        --------
        tabulate : Count or estimate frequencies of outcomes.

        Examples
        --------
        Filter using a function:

        >>> from symbulate import *
        >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
        >>> sims = X.sim(100)
        >>> sims.filter(lambda x: x > 4)  # doctest: +SKIP

        Filter using a boolean Results object from the same
        simulation:

        >>> sims.filter(sims > 4)  # doctest: +SKIP
        """
        if len(args) == 0:
            raise TypeError("filter() requires at least one argument.")
        if len(args) == 1:
            filt = args[0]
            if isinstance(filt, Results):
                if self.sim_id != filt.sim_id:
                    raise Exception(
                        "In order to filter one Results object "
                        "by another, they must come from the "
                        "same simulation."
                    )
                if len(filt) != len(self):
                    raise ValueError(
                        "Filter must be the same length as the Results object."
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
        filt = _build_mv_filter(args)
        return type(self)(x for x in self if filt(x))

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

        Examples
        --------
        Attempting to plot outcomes from a probability space raises
        an exception directing the user to define an RV first:

        >>> from symbulate import *
        >>> P = BoxModel(["H", "T"])
        >>> P.sim(100).plot()  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        Exception: Only simulations of random variables (RV) ...
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

        Examples
        --------
        >>> from symbulate import *
        >>> P = BoxModel(["H", "T"])
        >>> results = P.sim(3)
        >>> print(repr(results))  # doctest: +SKIP
        Index  Result
        0      H
        1      T
        2      H
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
                if len(self) > 9:
                    table_body += "<tr><td>...</td><td>...</td></tr>"
                    i_last = len(self) - 1
                    table_body += row_template % (
                        i_last,
                        _truncate(str(self.get(i_last))),
                    )
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

    See Also
    --------
    Results : Container for probability space simulation outcomes.

    Examples
    --------
    Simulate 100 rolls of a fair die:

    >>> from symbulate import *
    >>> X = RV(BoxModel([1, 2, 3, 4, 5, 6]))
    >>> sims = X.sim(100)
    >>> float(sims.mean())  # doctest: +SKIP
    3.45
    """

    def __init__(self, results, sim_id=None):
        """
        Initialize simulation results for a random variable.
        """
        super().__init__(results, sim_id)
        # get type and dimension of the first result, if it exists
        self.dim = None
        self.index_set = None
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
                return Scalar(op(self.array))
            elif self.dim is not None:
                return Vector(op(self.array, axis=0))
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
            elif self.dim is not None and self.dim > 2:
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

    def tabulate(
        self, outcomes=None, normalize=False, bin=False, nbins=None, binwidth=None
    ):
        """Count how many times each outcome appears.

        Parameters
        ----------
        outcomes : list, optional
            Outcomes to include in the table. By default,
            tabulates all outcomes that appear in the Results.
            Use this option to include outcomes that may not
            appear in the Results. Ignored when ``bin=True``.
        normalize : bool, default False
            If True, return relative frequencies. If False,
            return counts.
        bin : bool, default False
            If True, group numeric results into equal-width bins
            for a compact summary of continuous values. If False,
            tabulate each distinct value exactly (the default;
            appropriate for discrete results).
        nbins : int, optional
            Number of equal-width bins to use when ``bin=True``.
            Defaults to 10. Cannot be used together with ``binwidth``.
        binwidth : float, optional
            Width of each bin when ``bin=True``; the number of bins
            is determined automatically from the data range. Cannot
            be used together with ``nbins``.

        Returns
        -------
        Table
            A Table mapping each observed value to its count or
            relative frequency, labeled "Value" in the header.
            When ``bin=True``, the keys are interval labels of
            the form ``"[a, b)"`` for all bins except the last,
            which is ``"[a, b]"`` (closed on both ends so the
            maximum value is included), under a "Bin" header.

        Raises
        ------
        ValueError
            If both ``nbins`` and ``binwidth`` are specified.
        UserWarning
            If ``nbins`` or ``binwidth`` are specified when ``bin=False``.

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
        >>> sims.tabulate()  # doctest: +SKIP

        Show relative frequencies:

        >>> sims.tabulate(normalize=True)  # doctest: +SKIP

        Summarize continuous results with equal-width bins:

        >>> RV(Normal(0, 1)).sim(1000).tabulate(bin=True)  # doctest: +SKIP

        Use a custom number of bins:

        >>> RV(Normal(0, 1)).sim(1000).tabulate(bin=True, nbins=20)  # doctest: +SKIP

        Use a custom bin width:

        >>> RV(Normal(0, 1)).sim(1000).tabulate(bin=True, binwidth=0.5)  # doctest: +SKIP
        """
        if nbins is not None and binwidth is not None:
            raise ValueError(
                "Cannot specify both nbins and binwidth. "
                "Use nbins to set the number of bins, or binwidth to set the bin width."
            )
        if not bin and (nbins is not None or binwidth is not None):
            warnings.warn(
                "nbins and binwidth have no effect when bin=False.",
                UserWarning,
                stacklevel=2,
            )
        if bin:
            if binwidth is not None:
                data_range = max(self.results) - min(self.results)
                n = max(1, int(np.ceil(data_range / binwidth)))
            else:
                n = nbins if nbins is not None else 10
            counts, edges = np.histogram(self.results, bins=n)
            hash_map, labels = {}, []
            for i, count in enumerate(counts):
                closing = "]" if i == len(counts) - 1 else ")"
                label = f"[{edges[i]:.4g}, {edges[i + 1]:.4g}{closing}"
                labels.append(label)
                hash_map[label] = int(count)
            return Table(hash_map, labels, normalize, "Bin")
        return Table(self._get_counts(), outcomes, normalize, "Value")

    def plot(
        self,
        type=None,
        alpha=None,
        normalize=True,
        jitter=None,
        bins=None,
        suggest=None,
        **kwargs,
    ):
        """Plot the simulated random variable results.

        Parameters
        ----------
        type : str, tuple, or list, optional
            Plot type or types to display. Valid values are
            ``"hist"``, ``"bar"``, ``"impulse"``, ``"density"``,
            ``"ecdf"``, ``"dotplot"``, ``"rug"``, ``"scatter"``,
            ``"tile"``, ``"mosaic"``, ``"violin"``, ``"box"`` (alias
            ``"boxplot"``), and ``"marginal"`` (2D data also accepts
            ``"hist2d"``, ``"density2d"``, ``"segmented_rug"``,
            ``"segmented_density"``, and ``"segmented_hist"``).
            ``"mosaic"`` is only meaningful for two discrete-ish
            variables -- the same configuration ``"tile"`` targets.
            ``"segmented_density"`` and ``"segmented_hist"`` need one
            discrete and one continuous variable and draw one small
            density curve (or histogram) per level of the discrete
            variable, stacked along the discrete axis.
            If None, a default is chosen from the data: whether each
            variable looks discrete (``classify_data``) and whether
            the sample is small select an entry from the
            ``DEFAULT_PLOT_TYPE`` lookup table in ``plot.py``.
            On 2D data, ``"hist"`` draws a binned color mesh,
            ``"density"`` a smooth density surface (pass
            ``contour=True`` for a banded contour plot), and
            ``"rug"`` a segmented rug for one discrete and one
            continuous variable.
        alpha : float, optional
            Transparency of plotted elements, between 0 and 1. Each
            plot type has its own default: histograms (1D, 2D, and
            segmented) 0.65, scatter 0.25, rug 0.5, box plots 0.75,
            segmented density fills 0.4, impulse / density curves /
            dot plots fully opaque, and 0.5 for violin and marginal
            panels.
        normalize : bool, default True
            If True, plot relative frequencies or densities. If
            False, plot raw counts. Dot plots always show counts.
        jitter : bool, str, or None, optional
            How to spread out coincident points in a 2D scatter plot.
            If None (default), the layout is chosen from the data:
            two discrete variables get countable clusters
            (``"spiral"``, or ``"bins"`` for heavy pile-ups) and
            anything else draws exact positions. Explicit options:
            ``"spiral"`` (coincident points form a tight countable
            cluster on their grid crossing), ``"bins"`` (points fill
            a histogram-style box around their value), ``"random"``
            (small random noise; ``True`` is a legacy alias), or
            ``False`` (exact positions). Has no effect on 1D impulse
            and dot plots -- overlays of those already spread apart
            automatically.
        bins : int, optional
            Number of bins for histograms (1D, 2D, and segmented), or
            for a continuous axis of a tile plot. Defaults to 30. Dot
            plots are never binned.
        suggest : bool or None, optional
            Whether to print a note under the plot naming the plot
            being shown and the reasonable alternatives for this
            data. ``None`` (default) prints it only on the first
            ``.plot()`` call of the session; ``True`` prints it on
            every call; ``False`` never prints it.
        **kwargs
            Additional keyword arguments passed to the underlying
            matplotlib plotting function. Notable options:
            ``bandwidth`` (smoothing for ``type="density"`` and
            ``type="segmented_density"``, passed to scipy's
            ``gaussian_kde``), ``contour`` and ``levels`` (2D
            density), ``hex=True`` (hexagonal bins for a 2D
            histogram), ``outliers`` (box plots: ``True``, the
            default, stops the whiskers at 1.5 times the
            interquartile range and draws more extreme points
            individually as outliers; ``False`` extends the whiskers
            to the minimum and maximum values instead), and ``label``
            (legend name for hist, impulse, dot, scatter, segmented
            density, and segmented histogram plots).

        Returns
        -------
        SymbulatePlot
            A wrapper around the matplotlib axes the plot was
            drawn on. Its printed representation is empty, so
            Jupyter shows only the plot.

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

        Plot a dot plot of a small simulation:

        >>> X.sim(30).plot(type="dotplot")  # doctest: +SKIP

        Plot a scatter plot for a bivariate random variable:

        >>> X2 = RV(BoxModel([1, 2, 3, 4, 5, 6], size=2))
        >>> X2.sim(500).plot(type="scatter")  # doctest: +SKIP
        """
        if type is not None:
            if isinstance(type, str):
                type = (type,)
            elif not isinstance(type, (tuple, list)):
                raise Exception(
                    f"Unrecognized plot type {type!r}. "
                    "Valid types are: 'hist', 'bar', 'impulse', 'density', "
                    "'ecdf', 'dotplot', 'rug', 'scatter', 'tile', 'mosaic', "
                    "'violin', 'box' (alias 'boxplot'), 'marginal' (and, "
                    "for 2D data, 'hist2d', 'density2d', 'segmented_rug', "
                    "'segmented_density', 'segmented_hist')."
                )

        # Filled in by the dim == 1 and dim == 2 branches with
        # (shown, default, alternatives) so the suggestion note can be
        # printed after the plot renders. _jitter_note is set when a
        # scatter is drawn on two discrete variables, where the jitter
        # layout matters.
        _suggestion = None
        _jitter_note = None

        if self.dim == 1:
            # make sure self.array, a Numpy array, has been set
            self._set_array()

            # Detect near-constant data caused by floating-point precision
            # (e.g. Y = X * cos(pi/2) where cos(pi/2) ≈ 6e-17 instead of 0).
            # If the entire data range is negligible relative to the values'
            # magnitude, collapse everything to its effective mean for display.
            _plot_array = self.array
            if len(self.array) > 1:
                _data_range = self.array.max() - self.array.min()
                _data_abs_max = abs(self.array).max()
                _NEAR_CONST_TOL = 1e-9
                if _data_range < _NEAR_CONST_TOL * max(_data_abs_max, 1.0):
                    _center = float(np.mean(self.array))
                    if abs(_center) < _NEAR_CONST_TOL:
                        _center = 0.0
                    _plot_array = np.full_like(self.array, _center)

            # determine plotting parameters
            counts = count_var(_plot_array)
            discrete, small_n = classify_data(_plot_array)
            configuration = "1D_discrete" if discrete else "1D_continuous"
            default, alternatives = default_plot_type(configuration, small_n)
            if type is None:
                type = (default,)
            _suggestion = (type[0], default, alternatives)
            # Each 1D plot type defaults its own alpha inside its make_*
            # helper (HIST_ALPHA, DENSITY_ALPHA, RUG_ALPHA, IMPULSE_ALPHA,
            # DOTPLOT_ALPHA), so alpha stays None here unless the user
            # set it explicitly.
            if jitter and ("impulse" in type or "dotplot" in type):
                warnings.warn(
                    "jitter has no effect on impulse and dot plots. "
                    "Overlays of those plot types already spread apart "
                    "automatically. jitter still applies to 2D scatter "
                    "plots.",
                    UserWarning,
                    stacklevel=2,
                )
            if bins is not None and "dotplot" in type:
                warnings.warn(
                    "bins has no effect on dot plots. Every observation is "
                    "drawn at its exact value, so dot plots are never "
                    "binned. Try type='hist' if you want binned data.",
                    UserWarning,
                    stacklevel=2,
                )
            n = len(self)

            # initialize figure
            fig = plt.gcf()
            ax = plt.gca()
            color = get_next_color(ax)

            if "dotplot" in type:
                make_dotplot(
                    _plot_array,
                    ax,
                    color,
                    alpha=alpha,
                    **kwargs,
                )
            if "density" in type:
                if discrete:
                    xs = sorted(list(counts.keys()))
                    probs = [counts[x] / n for x in xs]
                    ax.plot(xs, probs, marker="o", color=color, linestyle="-")
                    if len(type) == 1:
                        plt.ylabel("Relative Frequency")
                else:
                    # bandwidth is popped here so it never leaks into the
                    # hist/impulse branches of a combined type.
                    make_density(
                        _plot_array,
                        ax,
                        color,
                        bandwidth=kwargs.pop("bandwidth", None),
                        alpha=alpha,
                    )
            if "hist" in type:
                make_hist(
                    _plot_array,
                    ax,
                    color,
                    bins=bins,
                    normalize=normalize,
                    alpha=alpha,
                    **kwargs,
                )
            elif "bar" in type:
                make_bar(
                    _plot_array,
                    ax,
                    color,
                    normalize=normalize,
                    alpha=alpha,
                    **kwargs,
                )
            elif "impulse" in type:
                make_impulse(
                    _plot_array,
                    ax,
                    color,
                    normalize=normalize,
                    alpha=alpha,
                    **kwargs,
                )
            elif "box" in type or "boxplot" in type:
                make_boxplot(_plot_array, ax, color, alpha=alpha, **kwargs)
            elif "violin" in type:
                make_violinplot(_plot_array, ax, color, alpha=alpha, **kwargs)
            if "rug" in type:
                make_rug(_plot_array, ax, color, alpha=alpha)
            if "ecdf" in type:
                make_ecdf(
                    _plot_array,
                    ax,
                    color,
                    normalize=normalize,
                    alpha=alpha,
                    **kwargs,
                )
        elif self.dim == 2:
            # make sure self.array, a Numpy array, has been set
            self._set_array()
            x, y = self.array[:, 0], self.array[:, 1]

            # x_count / y_count feed the marginal impulses and violin
            # positions below; discreteness itself comes from
            # classify_data.
            x_count = count_var(x)
            y_count = count_var(y)
            discrete_x, small_n = classify_data(x)
            discrete_y, _ = classify_data(y)

            if discrete_x and discrete_y:
                configuration = "2D_dd"
            elif not discrete_x and not discrete_y:
                configuration = "2D_cc"
            else:
                configuration = "2D_mixed"
            default, alternatives = default_plot_type(configuration, small_n)
            if type is None:
                type = (default,)
            # The lookup-table tokens name the 2D variants explicitly
            # (hist2d, density2d, segmented_rug); the shorter names are
            # what users have always passed on 2D data. The suggestion
            # note uses the explicit token so its display name reads
            # "2D Histogram", not "Histogram".
            _2d_token = {
                "hist": "hist2d",
                "density": "density2d",
                "rug": "segmented_rug",
            }
            _suggestion = (_2d_token.get(type[0], type[0]), default, alternatives)
            # Scatter defaults its own alpha (SCATTER_ALPHA) inside
            # make_scatter, and the mesh types (hist/density/tile) encode
            # magnitude with a colormap instead of transparency. The
            # legacy 0.5 default still applies to the violin and marginal
            # panels, which have no per-type constant yet.
            legacy_alpha = 0.5 if alpha is None else alpha

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
                    marg_bins = bins if bins is not None else HIST_DEFAULT_BINS
                    if discrete_x:
                        make_marginal_impulse(
                            x_count, get_next_color(ax), ax_marg_x, legacy_alpha, "x"
                        )
                    else:
                        ax_marg_x.hist(
                            x,
                            color=get_next_color(ax),
                            density=normalize,
                            alpha=legacy_alpha,
                            bins=marg_bins,
                        )
                    if discrete_y:
                        make_marginal_impulse(
                            y_count, get_next_color(ax), ax_marg_y, legacy_alpha, "y"
                        )
                    else:
                        ax_marg_y.hist(
                            y,
                            color=get_next_color(ax),
                            density=normalize,
                            alpha=legacy_alpha,
                            bins=marg_bins,
                            orientation="horizontal",
                        )
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)
            else:
                fig = plt.gcf()
                ax = plt.gca()
                color = get_next_color(ax)

            # The 'marginal' layout keeps the legacy left-side colorbar
            # (add_colorbar): the mesh helpers' own right-side colorbar
            # would squeeze the y-marginal panel, so they are called with
            # colorbar=False and add_colorbar places it instead.
            if "scatter" in type:
                # jitter=None (the default) resolves from the data: two
                # discrete variables get the countable clustered layout
                # auto_jitter_mode picks; anything else draws exact
                # positions. classify_data makes the call so continuous
                # data is never snapped to integers.
                scatter_jitter = jitter
                if scatter_jitter is None:
                    scatter_jitter = (
                        auto_jitter_mode(x, y) if discrete_x and discrete_y else False
                    )
                make_scatter(
                    x,
                    y,
                    ax,
                    color,
                    alpha=alpha,
                    jitter=scatter_jitter,
                    **kwargs,
                )
                if discrete_x and discrete_y:
                    _jitter_note = (
                        "random" if scatter_jitter is True else scatter_jitter
                    )
            elif "hist" in type or "hist2d" in type:
                if "marginal" in type:
                    histo = make_hist2d(
                        x,
                        y,
                        ax,
                        bins=bins,
                        normalize=normalize,
                        colorbar=False,
                        **kwargs,
                    )
                    mappable = histo[3] if isinstance(histo, tuple) else histo
                    add_colorbar(
                        fig, type, mappable, "Density" if normalize else "Count"
                    )
                else:
                    make_hist2d(x, y, ax, bins=bins, normalize=normalize, **kwargs)
            elif "density" in type or "density2d" in type:
                if "marginal" in type:
                    den = make_density2D(x, y, ax, colorbar=False, **kwargs)
                    add_colorbar(fig, type, den, "Density")
                else:
                    make_density2D(x, y, ax, **kwargs)
            elif "rug" in type or "segmented_rug" in type:
                make_segmented_rug(
                    x,
                    y,
                    ax,
                    color,
                    alpha=alpha,
                    discrete_x=discrete_x,
                    discrete_y=discrete_y,
                )
            elif "segmented_density" in type:
                make_segmented_density(
                    x,
                    y,
                    ax,
                    color,
                    bandwidth=kwargs.pop("bandwidth", None),
                    alpha=alpha,
                    discrete_x=discrete_x,
                    discrete_y=discrete_y,
                    **kwargs,
                )
            elif "segmented_hist" in type:
                make_segmented_hist(
                    x,
                    y,
                    ax,
                    color,
                    bins=bins,
                    normalize=normalize,
                    alpha=alpha,
                    discrete_x=discrete_x,
                    discrete_y=discrete_y,
                    **kwargs,
                )
            elif "tile" in type:
                hm = make_tile(
                    x,
                    y,
                    ax,
                    normalize=normalize,
                    bins=bins,
                    discrete_x=discrete_x,
                    discrete_y=discrete_y,
                    colorbar="marginal" not in type,
                    **kwargs,
                )
                if "marginal" in type:
                    add_colorbar(
                        fig, type, hm, "Relative Frequency" if normalize else "Count"
                    )
            elif "mosaic" in type:
                make_mosaic(x, y, ax, normalize=normalize, **kwargs)
            elif "violin" in type:
                if discrete_x and not discrete_y:
                    positions = sorted(list(x_count.keys()))
                    make_violin(self.array, positions, ax, color, "x", legacy_alpha)
                elif not discrete_x and discrete_y:
                    positions = sorted(list(y_count.keys()))
                    make_violin(self.array, positions, ax, color, "y", legacy_alpha)
                elif discrete_x:
                    raise ValueError(
                        "A violin plot needs one discrete variable and one "
                        "continuous variable, but both of yours look "
                        "discrete. Try a tile plot for two discrete "
                        "variables."
                    )
                else:
                    raise ValueError(
                        "A violin plot needs one discrete variable and one "
                        "continuous variable, but both of yours look "
                        "continuous. Try a scatter plot for two continuous "
                        "variables."
                    )
            elif "box" in type or "boxplot" in type:
                make_grouped_boxplot(
                    x,
                    y,
                    ax,
                    color,
                    alpha=alpha,
                    discrete_x=discrete_x,
                    discrete_y=discrete_y,
                    **kwargs,
                )

            if "marginal" in type:
                # The marginal layout has no room for the center panel's
                # title -- it would collide with the top marginal panel.
                ax.set_title("")
        elif self.index_set is None and _is_categorical_1d(self.results):
            # 1D categorical (string) outcomes. These are not numbers, so
            # they get dim=None and skip the dim == 1 branch, but they are a
            # genuine 1D categorical variable: route them to the
            # "1D_categorical" lookup configuration and the categorical plot
            # types (bar / dot plot / impulse), which all accept strings.
            values = np.asarray(list(self.results))
            discrete, small_n = classify_data(values)
            default, alternatives = default_plot_type("1D_categorical", small_n)
            if type is None:
                type = (default,)
            _suggestion = (type[0], default, alternatives)
            ax = plt.gca()
            color = get_next_color(ax)
            if "bar" in type:
                make_bar(values, ax, color, normalize=normalize, alpha=alpha, **kwargs)
            elif "dotplot" in type:
                make_dotplot(values, ax, color, alpha=alpha, **kwargs)
            elif "impulse" in type:
                make_impulse(
                    values, ax, color, normalize=normalize, alpha=alpha, **kwargs
                )
            else:
                raise ValueError(
                    f"{type[0]!r} can't be used for categorical (text) data. "
                    "Categorical data works with type='bar', type='dotplot', or "
                    "type='impulse'."
                )
        else:
            if alpha is None:
                alpha = np.log(2) / np.log(len(self) + 1)
            ax = plt.gca()
            color = get_next_color(ax)
            # All realizations share one color and read as an ensemble,
            # so make_sample_path's per-path "Path k" legend entries
            # would be meaningless here -- suppress them (matplotlib
            # skips labels that start with an underscore).
            kwargs.setdefault("label", "_nolegend_")
            for result in self.results:
                result.plot(alpha=alpha, color=color, **kwargs)
            plt.xlabel("Index")

        # Print the suggestion notes after the plot has rendered: the
        # plot being shown with its reasonable alternatives, and -- for
        # a scatter of two discrete variables -- the jitter layout in
        # use with the other layouts a student can pass (see
        # should_show_suggestion / should_show_jitter_suggestion for
        # the suggest= policy).
        if _suggestion is not None and should_show_suggestion(suggest):
            print(suggestion_message(*_suggestion))
        if _jitter_note is not None and should_show_jitter_suggestion(suggest):
            print(jitter_suggestion_message(_jitter_note))
        return SymbulatePlot(ax)
