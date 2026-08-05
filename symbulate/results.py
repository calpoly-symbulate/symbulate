"""Data structures for storing the results of a simulation.

This module provides data structures for storing the
results of a simulation, either outcomes from a
probability space or realizations of a random variable /
random process.
"""

import numbers
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

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
    B_1D,
    K_2D,
    TILE_DEFAULT_BINS,
    DISCRETE_INDEX_OFFSET,
    DOTPLOT_MAX_STACK,
    JOINT_PAIRS_MAX_DIM,
    JOINT_PAIRS_OVERLAY_ERROR,
    JOINT_PAIRS_PANEL_SIZE,
    add_pairs_panel_colorbar,
    MARGINAL_OVERLAY_ERROR,
    setup_marginal_axes,
    thin_marginal_frequency_ticks,
    marginal_rug_tick_height,
    auto_jitter_mode,
    classify_values,
    default_plot_type,
    dotplot_tallest_stack,
    get_next_color,
    jitter_suggestion_message,
    should_show_jitter_suggestion,
    should_show_suggestion,
    suggestion_message,
    count_var,
    add_colorbar,
    make_bar,
    make_dotplot,
    make_density,
    make_density2D,
    make_ecdf,
    make_hist,
    make_hist2d,
    make_impulse,
    make_mosaic,
    make_segmented_density,
    make_segmented_hist,
    make_rug,
    make_scatter,
    make_segmented_rug,
    make_tile,
    setup_tile_axis,
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


def _draw_marginal_panel(
    ax_marg,
    main_ax,
    values,
    discrete,
    main_type,
    axis,
    color,
    small_n,
    normalize,
    bins,
    wants_density,
    bandwidth,
    hist_edges,
):
    """Draw one axis's content for RVResults.plot()'s marginal=True layout.

    Renders that axis's own (marginal) distribution, routed through the
    same classify_values-driven choice a standalone 1D variable would get
    (dotplot/impulse for a discrete axis, rug/hist for a continuous one),
    or a density curve when the main panel itself is density/density2d.
    ``axis="y"`` draws sideways (``orientation="horizontal"``) so the
    y-marginal panel's value axis lines up with the main panel's y-axis.

    Values on a discrete axis whose main-panel type packs *this specific
    axis* into integer rank slots instead of real values (violin, box,
    and the segmented_* types always do; tile only does for a
    categorical/non-whole-number/pathological-range axis -- a
    whole-number axis is laid out at real values by tile too, same as
    hist2d/density2d/scatter -- see ``DISCRETE_INDEX_OFFSET`` and how the
    caller resolves ``main_type`` independently per axis for tile) are
    converted to the matching rank codes here, so the marginal's
    dots/stems land under the correct main-panel column or row -- this
    is the fix for the tile+marginal coordinate mismatch confirmed in
    DECISIONS.md. Axis *limits* are synced by the caller afterward, by
    reading the main panel's actual final limits -- not derived here.

    Parameters
    ----------
    ax_marg : matplotlib.axes.Axes
        The marginal panel to draw on (``ax_marg_x`` or ``ax_marg_y``).
    main_ax : matplotlib.axes.Axes
        The joint panel. Only its size is used, to scale a rug's ticks so
        they match the joint panel's (see ``marginal_rug_tick_height``).
    values : numpy.ndarray
        This axis's simulated values (``x`` or ``y``).
    discrete : bool
        Whether this axis is discrete-ish, from ``classify_values``.
    main_type : str or None
        The exact renderer the main panel used for *this axis*
        specifically (e.g. ``"tile"``, ``"hist2d"``, ``"violin"``) --
        not necessarily ``type[0]``, since short names like ``"hist"``
        resolve to different renderers depending on the data
        configuration, and not necessarily the same for both axes of a
        single call, since tile can lay out one axis at real values and
        the other at rank-index cells. ``None`` means this axis is
        real-valued (no index-code conversion needed) even though the
        main panel's overall type might be a member of
        ``DISCRETE_INDEX_OFFSET``.
    axis : {"x", "y"}
        Which marginal panel this is.
    color : color
        Color for this panel's marginal plot, from ``get_next_color(ax)``
        (the main axes' color cycle, matching the rest of this method).
    small_n : bool
        Whether the sample is small, from ``classify_values`` -- picks
        dotplot/rug (small) vs. impulse/hist (large).
    normalize : bool
        Passed through to the impulse/hist renderer.
    bins : int, array-like, or None
        The user's ``bins=`` argument, used when ``hist_edges`` is None.
    wants_density : bool
        If True, draw a density curve regardless of ``discrete``/
        ``small_n`` (matches the main panel being density/density2d).
    bandwidth : float, str, or None
        Passed through to the density curve.
    hist_edges : numpy.ndarray or None
        Exact bin edges to reuse from the main panel's own binning
        (``make_hist2d``'s returned edges, or ``setup_tile_axis``'s, for
        a continuous axis on ``hist2d``/``tile``), so the marginal
        histogram's bars align with the mesh's cells instead of
        independently recomputing (coincidentally similar) bins. None
        for every other main type, where only the axis range needs to
        match -- handled by the caller via the main panel's final limits.
    """
    orientation = "vertical" if axis == "x" else "horizontal"

    # Show gridlines only along this panel's frequency axis (density /
    # relative frequency), never its value axis. The top (x) marginal is
    # drawn vertically, so its frequency runs up the y-axis -> horizontal
    # gridlines; the right (y) marginal is drawn horizontally, so its
    # frequency runs along the x-axis -> vertical gridlines. (The global
    # style is horizontal-only, which is right for the x-marginal but wrong
    # for the y-marginal, so set both explicitly here.)
    freq_axis = "y" if orientation == "vertical" else "x"
    ax_marg.set_axisbelow(True)
    ax_marg.grid(False)
    ax_marg.grid(True, axis=freq_axis)

    if wants_density:
        make_density(
            values, ax_marg, color, bandwidth=bandwidth, orientation=orientation
        )
        thin_marginal_frequency_ticks(ax_marg, orientation)
        ax_marg.set_title("")
        return

    plot_values = values
    offset = DISCRETE_INDEX_OFFSET.get(main_type)
    if discrete and offset is not None:
        plot_values = np.searchsorted(np.unique(values), values) + offset

    # A dot plot always shows counts; the others show counts only with
    # normalize=False. A count axis gets whole-number ticks -- there is no
    # such thing as half a simulated value.
    counts = normalize is False
    if discrete:
        if small_n:
            make_dotplot(plot_values, ax_marg, color, orientation=orientation)
            counts = True
        else:
            make_impulse(
                plot_values,
                ax_marg,
                color,
                normalize=normalize,
                orientation=orientation,
            )
    else:
        if small_n:
            # Sized against the joint panel rather than this strip, so a
            # marginal rug's ticks are the same length as the joint panel's
            # own rug ticks instead of shrinking with the strip.
            make_rug(
                plot_values,
                ax_marg,
                color,
                orientation=orientation,
                tick_height=marginal_rug_tick_height(main_ax, ax_marg, orientation),
            )
        else:
            make_hist(
                plot_values,
                ax_marg,
                color,
                bins=hist_edges if hist_edges is not None else bins,
                normalize=normalize,
                orientation=orientation,
            )
    thin_marginal_frequency_ticks(ax_marg, orientation, integer=counts)
    ax_marg.set_title("")


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

    def _pairs_variable_label(self, index):
        """Return the axis label for variable ``index``.

        Numbered from 1 the way variables are written mathematically, and
        worded to match the same label on a
        :class:`~symbulate.distributions.MultivariateDistribution` pairs
        plot, so a simulated matrix and a theoretical one can be read side
        by side.

        Parameters
        ----------
        index : int
            Index of the variable, counting from 0.

        Returns
        -------
        str
            The label, e.g. ``"Variable 1"`` for variable 0.
        """
        return "Variable %d" % (index + 1)

    def _pairs_resolve_variables(self):
        """Return the variables a pairs matrix will include.

        Every one of them: a subset is selected by indexing the random
        variable instead, e.g. ``X[[0, 2]].sim(1000).plot()``, so the matrix
        itself takes no choice.

        Returns
        -------
        tuple of int
            The variables to include, in order.

        Raises
        ------
        ValueError
            If there are fewer than two variables to plot.
        Exception
            If there are more variables than a readable matrix holds.
        """
        self._set_array()
        if self.dim is None or self.dim < 2:
            raise ValueError(
                "A pairs matrix shows how variables relate to each other, so "
                "it needs at least two of them. These results have "
                + ("only one." if self.dim == 1 else "an inconsistent number.")
                + " Plot them with .plot() instead."
            )
        chosen = tuple(range(self.dim))
        if len(chosen) > JOINT_PAIRS_MAX_DIM:
            raise Exception(
                "A pairs plot of %d variables would need %d panels, too many "
                "to read on one screen. Simulate the variables you want to "
                "look at instead, for example X[[0, 1, 2]].sim(1000).plot()."
                % (len(chosen), len(chosen) * (len(chosen) + 1) // 2)
            )
        return chosen

    def _pairs_column(self, index):
        """Return the simulated values of one variable, as an array."""
        self._set_array()
        return self.array[:, index]

    def _pairs_diagonal_type(self, index):
        """Return the plot type for one variable's own panel.

        Whatever a plot of that variable on its own would show: the same
        1-D default the ``dim == 1`` dispatch picks, from the same
        classification and the same lookup table. So a continuous variable
        gets a histogram (a rug plot on a small simulation), a discrete one
        an impulse plot (a dot plot when small), and a panel of the matrix
        looks like the plot a student would get by simulating that variable
        by itself.

        Parameters
        ----------
        index : int
            Which variable.

        Returns
        -------
        str
            The default 1-D plot type for that variable's data.
        """
        values = self._pairs_column(index)
        discrete, small_n = classify_values(values, n_unique_threshold=B_1D)
        configuration = "1D_discrete" if discrete else "1D_continuous"
        default, _ = default_plot_type(configuration, small_n)
        return default

    def _pairs_joint_configuration(self, x_index, y_index):
        """Return the data configuration of one pair, and its discreteness.

        Each axis is judged independently against the 2-D per-axis budget
        ``K_2D``, exactly as the ``dim == 2`` dispatch does.

        Parameters
        ----------
        x_index, y_index : int
            Which variables go on the x and y axes.

        Returns
        -------
        tuple
            ``(configuration, discrete_x, discrete_y)``, where the
            configuration is one of ``"2D_dd"``, ``"2D_cc"``, or
            ``"2D_mixed"``.
        """
        discrete_x, small_n = classify_values(
            self._pairs_column(x_index), n_unique_threshold=K_2D, large_n_rescue=False
        )
        discrete_y, _ = classify_values(
            self._pairs_column(y_index), n_unique_threshold=K_2D, large_n_rescue=False
        )
        if discrete_x and discrete_y:
            configuration = "2D_dd"
        elif not discrete_x and not discrete_y:
            configuration = "2D_cc"
        else:
            configuration = "2D_mixed"
        return configuration, discrete_x, discrete_y, small_n

    def _pairs_joint_type(self, x_index, y_index):
        """Return the plot type for one pair's panel.

        Whatever a plot of that pair on its own would show: the same 2-D
        default the ``dim == 2`` dispatch picks, from the same per-axis
        classification and the same lookup table. Two continuous variables
        get a 2-D histogram (a scatter on a small simulation), two discrete
        ones a tile plot (a scatter when small), and a mixed pair a tile
        plot with the continuous axis binned (a segmented rug when small).

        Parameters
        ----------
        x_index, y_index : int
            Which variables go on the x and y axes.

        Returns
        -------
        str
            The default 2-D plot type for that pair's data.
        """
        configuration, _, _, small_n = self._pairs_joint_configuration(x_index, y_index)
        default, _ = default_plot_type(configuration, small_n)
        return default

    def _plot_pairs(
        self, alpha=None, normalize=True, bins=None, suggest=None, **kwargs
    ):
        """Draw a matrix of every pair of the chosen variables.

        Builds a ``k`` by ``k`` grid of panels: each variable's own
        distribution down the diagonal, and each pair's joint distribution
        in the lower triangle. The upper triangle is left blank because
        panel ``(i, j)`` and panel ``(j, i)`` show the same relationship
        with the axes swapped, so filling both would draw everything twice.
        Laid out to match a ``MultivariateDistribution`` pairs plot, so a
        simulation can be compared with the distribution it came from.

        Parameters
        ----------
        alpha : float, optional
            Transparency of the panels.
        normalize : bool, default True
            Whether the panels show densities rather than raw counts. The
            same value is used for every panel, so they are comparable.
        bins : int, optional
            Number of equal-width bins a continuous axis is split into.
            One value is used throughout, so a variable is binned the same
            way in every panel it appears in.
        suggest : bool, optional
            Accepted for consistency with ``plot``; the panels never print
            their own suggestion notes.
        **kwargs
            Additional keyword arguments forwarded to the panels.

        Returns
        -------
        SymbulatePlot
            A wrapper around the bottom-left panel, as a representative of
            the layout.

        Raises
        ------
        ValueError
            If the figure already has a plot on it, which the grid can't
            be built into.
        """
        chosen = self._pairs_resolve_variables()
        k = len(chosen)

        fig = plt.gcf()
        # The grid fills the whole figure, so it can neither be added to a
        # figure that already has a plot on it nor accept one later --
        # the same rule the marginal layout and the theoretical pairs plot
        # follow rather than stacking two layouts on top of each other.
        if fig.axes:
            raise ValueError(JOINT_PAIRS_OVERLAY_ERROR)
        # Tag the figure so a joint panel routed back through the 2-D dispatch
        # (a scatter or a segmented rug -- see _draw_pairs_joint) draws just
        # the panel, instead of building the three-panel layout a standalone
        # two-variable plot gets. There is no room for strips inside a panel,
        # and the matrix's diagonal already shows each variable on its own.
        fig._symbulate_pairs = True
        # Size the figure to the grid, so panels stay readable as variables
        # are added instead of each one shrinking inside a single-plot figure.
        fig.set_size_inches(k * JOINT_PAIRS_PANEL_SIZE, k * JOINT_PAIRS_PANEL_SIZE)
        gs = GridSpec(k, k, figure=fig)

        # One bin count for the whole matrix. Every panel in a column bins
        # the same variable over the same data, so an equal-width binning
        # with the same count gives identical edges -- which is what makes
        # the panels in a column directly comparable.
        panel_bins = TILE_DEFAULT_BINS if bins is None else bins

        corner = None
        # (mappable, row, col) per joint panel, so each can be given its own
        # colorbar once the layout has settled (see the end of this method).
        joint_panels = []
        for row in range(k):
            for col in range(row + 1):
                ax = fig.add_subplot(gs[row, col])
                if row == col:
                    # The diagonal is this variable on its own, so it is
                    # exactly the univariate plot -- reuse the whole 1-D
                    # dispatch rather than redraw it. The helpers all draw
                    # on plt.gca(), so making this panel current is what
                    # routes the plot into it.
                    plt.sca(ax)
                    self._pairs_subset((chosen[row],)).plot(
                        type=self._pairs_diagonal_type(chosen[row]),
                        alpha=alpha,
                        normalize=normalize,
                        suggest=False,
                        **kwargs,
                    )
                else:
                    joint_panels.append(
                        (
                            self._draw_pairs_joint(
                                chosen[col],
                                chosen[row],
                                ax,
                                normalize=normalize,
                                bins=panel_bins,
                                alpha=alpha,
                                **kwargs,
                            ),
                            row,
                            col,
                        )
                    )
                # Drop the title each panel drew for itself. On its own a plot
                # is titled with its type ("Density Curve", "Tile Plot"), but
                # in a matrix that repeats the same two or three words down
                # every panel and crowds them; the figure's own "Pairs Plot"
                # says what the layout is. The theoretical pairs plot clears
                # its panels' titles the same way.
                ax.set_title("")
                # Name the variables only along the outside edges, so the
                # inner panels aren't crowded with repeated labels.
                if row == k - 1:
                    ax.set_xlabel(self._pairs_variable_label(chosen[col]))
                else:
                    ax.set_xlabel("")
                    # Every panel in a column covers the same variable over
                    # the same range, so hiding the inner x tick labels
                    # loses nothing -- the bottom panel's still apply. The y
                    # tick labels stay on every panel, since a diagonal
                    # panel's y-axis is a density and genuinely differs
                    # from its neighbors'.
                    ax.set_xticklabels([])
                # The left column names its row's variable, so the labels read
                # down the side in order -- including the top-left panel, which
                # is the only one in its row and would otherwise go unnamed
                # until the bottom of its column. That panel's y-axis is really
                # a density rather than the variable, so the label names the
                # row it heads rather than the axis it sits on; this is the
                # convention seaborn's PairGrid uses too.
                if col == 0:
                    ax.set_ylabel(self._pairs_variable_label(chosen[row]))
                else:
                    ax.set_ylabel("")
                if row == k - 1 and col == 0:
                    corner = ax

        fig.suptitle("Pairs Plot")
        fig.tight_layout()

        # Each joint panel gets its own colorbar, in the empty cell mirroring
        # it across the diagonal. Done after tight_layout, so the cells are
        # where they will finally be.
        quantity = "Density" if normalize else "Count"
        for mappable, row, col in joint_panels:
            if mappable is None:
                continue
            add_pairs_panel_colorbar(
                fig,
                gs[col, row].get_position(fig),
                mappable,
                "%s & %s"
                % (
                    self._pairs_variable_label(chosen[col]),
                    self._pairs_variable_label(chosen[row]),
                ),
                quantity,
            )

        # Leave the bottom-left panel current, so the returned plot and the
        # figure's idea of "the" axes agree.
        if corner is not None:
            plt.sca(corner)
        return SymbulatePlot(corner)

    def _pairs_subset(self, indices):
        """Return results holding just the chosen variables.

        Parameters
        ----------
        indices : tuple of int
            Which variables to keep, in order. One index gives
            single-variable results, two gives a pair.

        Returns
        -------
        RVResults
            The same simulation, restricted to those variables.
        """
        if len(indices) == 1:
            index = indices[0]
            return RVResults([outcome[index] for outcome in self.results])
        return RVResults(
            [tuple(outcome[i] for i in indices) for outcome in self.results]
        )

    def _draw_pairs_joint(self, x_index, y_index, ax, normalize, bins, alpha, **kwargs):
        """Draw one pair's joint panel of a pairs matrix.

        Calls the same drawing helpers the ``dim == 2`` dispatch does, with
        the colorbar off: a matrix of panels each carrying its own colorbar
        would spend more of the figure on scales than on data, and the
        panels share one meaning (density, or count) anyway.

        Parameters
        ----------
        x_index, y_index : int
            Which variables go on the x and y axes.
        ax : matplotlib.axes.Axes
            The panel to draw on.
        normalize : bool
            Whether to show densities rather than raw counts.
        bins : int
            Number of equal-width bins for a continuous axis.
        alpha : float or None
            Transparency of the panel.
        **kwargs
            Additional keyword arguments forwarded to the helper.

        Returns
        -------
        matplotlib.cm.ScalarMappable or None
            The mesh or image drawn, so the caller can give the panel its own
            colorbar in the cell mirroring it -- or ``None`` for a panel that
            encodes nothing in color (a scatter, a segmented rug) and so has
            no scale to explain.
        """
        joint_type = self._pairs_joint_type(x_index, y_index)
        if joint_type not in ("hist2d", "tile"):
            # A scatter or a segmented rug: no color scale, so it can go
            # through the whole 2-D dispatch, which is where the jitter and
            # the segment colors are worked out. The helpers draw on
            # plt.gca(), so making this panel current routes it here.
            plt.sca(ax)
            self._pairs_subset((x_index, y_index)).plot(
                type=joint_type,
                alpha=alpha,
                normalize=normalize,
                suggest=False,
                **kwargs,
            )
            return None

        x = self._pairs_column(x_index)
        y = self._pairs_column(y_index)
        _, discrete_x, discrete_y, _ = self._pairs_joint_configuration(x_index, y_index)
        if joint_type == "hist2d":
            histogram = make_hist2d(
                x,
                y,
                ax,
                bins=bins,
                normalize=normalize,
                colorbar=False,
                **kwargs,
            )
            # Square bins hand back (counts, xedges, yedges, mesh); hexagonal
            # ones hand back the mesh itself.
            return histogram[3] if isinstance(histogram, tuple) else histogram
        else:
            # bins only bins a continuous axis, and make_tile warns if it is
            # handed one when both variables are discrete (every distinct
            # value already gets its own cell), so only pass it when there
            # is a continuous axis for it to apply to.
            if not (discrete_x and discrete_y):
                kwargs["bins"] = bins
            return make_tile(
                x,
                y,
                ax,
                normalize=normalize,
                discrete_x=discrete_x,
                discrete_y=discrete_y,
                colorbar=False,
                **kwargs,
            )

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
            ``"tile"``, ``"mosaic"``, ``"violin"``, and ``"box"``
            (alias ``"boxplot"``).
            ``"mosaic"`` is only meaningful for two discrete-ish
            variables -- the same configuration ``"tile"`` targets.
            If None, a default is chosen from the data: whether each
            variable looks discrete (``classify_values``) and whether
            the sample is small select an entry from the
            ``DEFAULT_PLOT_TYPE`` lookup table in ``plot.py``.
            The short names adapt to the data. On two continuous
            variables, ``"hist"`` draws a binned color mesh and
            ``"density"`` a smooth density surface (pass
            ``contour=False`` for one smooth gradient instead of the
            default contour bands). On mixed data
            (one discrete, one continuous), ``"rug"``, ``"hist"``, and
            ``"density"`` draw the *segmented* rug / histogram / density
            -- one small plot per level of the discrete variable,
            stacked along the discrete axis. The explicit names
            ``"hist2d"``, ``"density2d"``, ``"segmented_rug"``,
            ``"segmented_density"``, and ``"segmented_hist"`` are also
            accepted if you want to force a particular one.
        alpha : float, optional
            Transparency of plotted elements, between 0 and 1. Each
            plot type has its own default: histograms (1D, 2D, and
            segmented) 0.65, scatter 0.25, rug 0.5, box plots 0.75,
            segmented density fills 0.4 (with ``ridge=True``; the
            curves themselves default to fully opaque), impulse /
            density curves / dot plots fully opaque, and 0.5 for
            violin and marginal panels.
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
            ``gaussian_kde``), ``ridge=True`` (fill under each
            segmented density curve for the classic ridgeline look),
            ``contour`` and ``levels`` (2D density), ``hex=True``
            (hexagonal bins for a 2D histogram), ``outliers`` (box
            plots: ``True``, the default, stops the whiskers at 1.5
            times the interquartile range and draws more extreme
            points individually as outliers; ``False`` extends the
            whiskers to the minimum and maximum values instead),
            ``label`` (legend name for hist, impulse, dot, scatter,
            segmented density, and segmented histogram plots), and
            ``equal_width`` (mosaic plots: ``True`` draws every column
            the same width instead of proportional to its marginal
            frequency -- a 100%-stacked bar chart per ``x`` value;
            default ``False`` keeps mosaic's standard proportional
            widths).

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

        Three or more variables are drawn as a matrix of every pair, with
        no argument needed:

        >>> X3 = RV(BoxModel([1, 2, 3, 4, 5, 6], size=3))
        >>> X3.sim(500).plot()  # doctest: +SKIP
        >>> X3[[0, 2]].sim(500).plot()  # just the 1st and 3rd  # doctest: +SKIP
        >>> X3.sim(500).plot(type="path")  # one line per realization  # doctest: +SKIP

        Notes
        -----
        **Two variables** are drawn on three panels: the two of them
        together in the main panel, and each one on its own in a strip
        beside the matching axis -- above for x, to the right for y. The
        strips are that variable's own (marginal) distribution, drawn the
        way a plot of it alone would be, so the joint picture and the two
        one-variable pictures can be read against each other. This is the
        layout for every two-variable plot type, so there is no keyword to
        ask for it (``marginal=True`` used to be that keyword). Because the
        three panels fill the figure, a two-variable plot cannot share a
        figure with another plot.

        With three or more variables there is no single joint plot, so
        ``.plot()`` draws a **matrix of every pair**: each variable's own
        distribution down the diagonal, and each pair's joint distribution
        below it. ``type="path"`` asks instead for one line per realization
        against its index. Because the matrix fills the figure with its own
        panels, it cannot share a figure with another plot.

        Every panel is chosen exactly as a plot of that data on its own
        would be -- the same classification and the same lookup table:

        - **diagonal** -- whatever that variable alone would get: a
          histogram when it looks continuous, an impulse plot when it looks
          discrete, and on a small simulation a rug plot or a dot plot.
        - **below the diagonal** -- whatever that pair alone would get: a
          2-D histogram when both variables look continuous, a tile plot
          when both look discrete or when they are mixed (the continuous
          axis binned), and on a small simulation a scatter plot or a
          segmented rug.

        A panel that encodes something in color -- a tile plot or a 2-D
        histogram -- gets its own colorbar in the empty cell mirroring it
        across the diagonal, named for the pair it explains.

        The upper triangle is left blank, because panel ``(i, j)`` and
        panel ``(j, i)`` show the same relationship with the axes
        swapped. Every panel in a column covers the same variable with
        the same number of bins, so columns are directly comparable.
        """

        if type is not None:
            if isinstance(type, str):
                type = (type,)
            elif not isinstance(type, (tuple, list)):
                raise Exception(
                    f"Unrecognized plot type {type!r}. "
                    "Valid types are: 'hist', 'bar', 'impulse', 'density', "
                    "'ecdf', 'dotplot', 'rug', 'scatter', 'tile', 'mosaic', "
                    "'violin', 'box' (alias 'boxplot') (and, "
                    "for 2D data, 'hist2d', 'density2d', 'segmented_rug', "
                    "'segmented_density', 'segmented_hist')."
                )
            if "marginal" in type:
                raise ValueError(
                    "'marginal' is not a type= value. A plot of two "
                    "variables always shows each variable's own "
                    "distribution in a strip beside the main panel, so "
                    "there is nothing to ask for -- use type= to choose "
                    "the main panel's plot type, e.g. .plot(type='tile')."
                )

        # Overlay policy, hard-error tier: a prior two-variable plot built a
        # three-panel layout on this figure that a second plot can't share.
        # Fail with a student-friendly message instead of silently drawing
        # into one of the marginal strips. (A fresh Jupyter cell gets a fresh
        # figure, so this only fires on a genuine second plot.)
        if getattr(plt.gcf(), "_symbulate_marginal", False):
            raise ValueError(MARGINAL_OVERLAY_ERROR)

        # Filled in by the dim == 1 and dim == 2 branches with
        # (shown, default, alternatives) so the suggestion note can be
        # printed after the plot renders. _jitter_note is set when a
        # scatter is drawn on two discrete variables, where the jitter
        # layout matters.
        _suggestion = None
        _jitter_note = None

        # marginal=True used to be how the strips showing each variable on its
        # own were asked for. Every two-variable plot has them now, so say so
        # rather than letting the stray keyword reach matplotlib.
        if "marginal" in kwargs:
            raise ValueError(
                "marginal= is no longer needed: a plot of two variables "
                "always shows each variable's own distribution in a strip "
                "beside the main panel, so .plot() alone does it. Drop "
                "marginal=True."
            )

        # pairs=True used to be how the matrix was asked for. It is the default
        # now, so say so rather than letting the stray keyword reach matplotlib
        # and come back as an error about a Rectangle.
        if "pairs" in kwargs:
            raise ValueError(
                "pairs= is no longer needed: three or more variables are drawn "
                "as a matrix of every pair by default, so .plot() alone does "
                "it. Drop pairs=True. (For the old plot of one line per "
                "realization against its index, use type='path'.)"
            )

        # Three or more variables have no single joint plot, so the default is
        # the matrix of every pair. type="path" asks for the old behavior: each
        # realization drawn against its index. Gated on a known dimension, so
        # sample paths of a random process -- whose results are time functions
        # rather than tuples of numbers, and whose dim is None -- are not
        # affected and keep falling through to the path branch below.
        if self.dim is not None and self.dim > 2:
            default, alternatives = default_plot_type("nD", False)
            if type is None:
                type = (default,)
            if "pairs" in type:
                # Printed here rather than at the end, because the matrix
                # returns early; the panels themselves stay quiet.
                if should_show_suggestion(suggest):
                    print(suggestion_message(type[0], default, alternatives))
                return self._plot_pairs(
                    alpha=alpha,
                    normalize=normalize,
                    bins=bins,
                    suggest=False,
                    **kwargs,
                )
            if "path" not in type:
                raise ValueError(
                    "%r can't be used for %d variables at once. Three or more "
                    "variables are drawn as a matrix of every pair "
                    "(type='pairs', the default) or as one line per "
                    "realization against its index (type='path'). To draw a "
                    "particular pair on its own, simulate that pair, e.g. "
                    "(X & Z).sim(1000).plot()." % (type[0], self.dim)
                )
            # Fall through to the path branch, with the note it should print.
            _suggestion = (type[0], default, alternatives)
        if "dims" in kwargs:
            raise ValueError(
                "dims= is not a plotting argument for simulated results: "
                "choose the variables when you simulate them instead, by "
                "indexing the random variable -- X[[0, 2]].sim(1000).plot() "
                "for the 1st and 3rd. (A distribution's own .plot() does take "
                "variables=, since there is nothing to simulate.)"
            )

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
            # 1-D uses the 1-D crowding budget B_1D (see classify_values).
            discrete, small_n = classify_values(_plot_array, n_unique_threshold=B_1D)
            configuration = "1D_discrete" if discrete else "1D_continuous"
            default, alternatives = default_plot_type(configuration, small_n)
            # A dot plot stacks one dot per observation, so a single tall
            # stack (e.g. the near-constant Binomial(1, 0.1)) shrinks the
            # dots to specks no matter how few distinct values there are or
            # how small the sample is. When a dot plot would be the automatic
            # default but the tallest stack exceeds DOTPLOT_MAX_STACK, treat
            # the data like large n and use that configuration's large-n
            # default (impulse here) instead. Only the default is redirected;
            # an explicit type='dotplot' is still honored.
            if (
                type is None
                and default == "dotplot"
                and dotplot_tallest_stack(_plot_array) > DOTPLOT_MAX_STACK
            ):
                default, alternatives = default_plot_type(configuration, False)
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

            # x_count / y_count feed the violin positions below;
            # discreteness itself comes from classify_values.
            x_count = count_var(x)
            y_count = count_var(y)
            # Each 2-D axis is judged independently against the per-axis budget
            # K_2D, so one axis over budget bins only that axis (-> a mixed
            # tile) and both over budget bin both (-> a 2-D histogram).
            discrete_x, small_n = classify_values(
                x, n_unique_threshold=K_2D, large_n_rescue=False
            )
            discrete_y, _ = classify_values(
                y, n_unique_threshold=K_2D, large_n_rescue=False
            )

            if discrete_x and discrete_y:
                configuration = "2D_dd"
            elif not discrete_x and not discrete_y:
                configuration = "2D_cc"
            else:
                configuration = "2D_mixed"
            default, alternatives = default_plot_type(configuration, small_n)
            if type is None:
                type = (default,)
            # On continuous x continuous data the short names hist/density
            # mean the 2D mesh variants, so map them to the explicit tokens
            # for the suggestion note's display name ("2D Histogram" rather
            # than "Histogram"). On mixed data the short names are already
            # what the table lists (they resolve to the segmented variants in
            # the dispatch), so no remap is needed there.
            if configuration == "2D_mixed":
                _2d_token = {}
            else:
                _2d_token = {"hist": "hist2d", "density": "density2d"}
            # equal_width draws mosaic's title as "Stacked Plot" instead of
            # "Mosaic Plot" (see make_mosaic) -- remap the suggestion note's
            # display name to match, the same way hist/density remap above.
            if "mosaic" in type and kwargs.get("equal_width"):
                _2d_token["mosaic"] = "mosaic_equal_width"
            _suggestion = (_2d_token.get(type[0], type[0]), default, alternatives)
            # Scatter defaults its own alpha (SCATTER_ALPHA) inside
            # make_scatter, and the mesh types (hist/density/tile) encode
            # magnitude with a colormap instead of transparency. The
            # legacy 0.5 default still applies to the violin panel, which
            # has no per-type constant yet.
            legacy_alpha = 0.5 if alpha is None else alpha

            # Every two-variable plot shows each variable's own distribution
            # in a strip beside the main panel. Two exceptions, neither of
            # them a user choice:
            #
            # - A mosaic plot already shows both -- x's through its column
            #   widths, y's through its own marginal column -- so strips
            #   would draw each of them twice.
            # - A panel of a pairs matrix has no room for them, and the
            #   matrix's own diagonal is already each variable on its own.
            #   (A joint panel that draws a scatter or a segmented rug comes
            #   back through here; see _draw_pairs_joint.)
            marginal = "mosaic" not in type and not getattr(
                plt.gcf(), "_symbulate_pairs", False
            )
            # Peeked (not popped) before the main-panel dispatch below,
            # since some branches (segmented density) pop "bandwidth" out
            # of kwargs for their own use -- the marginal density curve
            # reuses the same bandwidth the user asked for, so it has to
            # be read before that happens.
            _marginal_bandwidth = kwargs.get("bandwidth")
            # Set by whichever main-panel branch below actually renders,
            # to the exact renderer name it used (not just type[0], which
            # can be a short name like "hist" that several different
            # renderers resolve to depending on configuration). Read by
            # the marginal-panel block after the dispatch to align each
            # marginal with the *actual* main-panel coordinate system.
            # Tracked per axis (not one shared value) because tile can
            # resolve differently per axis: a whole-number discrete axis
            # is laid out at real values, but a categorical/non-whole-
            # number/pathological-range one falls back to rank-index
            # cells -- only the latter needs the marginal's index-code
            # conversion, and the tile branch below sets each axis's
            # resolved type independently to capture that.
            _resolved_main_type_x = None
            _resolved_main_type_y = None
            # Set only by the hist2d/tile branches when marginal=True, to
            # the exact bin edges they used, so a marginal histogram can
            # share them instead of independently (if coincidentally)
            # recomputing the same edges.
            _marginal_hist_edges = (None, None)

            if marginal:
                fig = plt.gcf()
                # Builds its own three-panel GridSpec, so it can't be
                # retrofitted onto a figure that already has a plot on it --
                # setup_marginal_axes raises MARGINAL_OVERLAY_ERROR if it is
                # asked to, the same way overlaying *onto* this layout does.
                ax, ax_marg_x, ax_marg_y = setup_marginal_axes(fig)
                # The helpers all draw on plt.gca(), so make the joint panel
                # current -- it is also the panel this call's SymbulatePlot
                # wraps and the one a reader means by "the plot".
                plt.sca(ax)
                color = get_next_color(ax)
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
                # positions. classify_values makes the call so continuous
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
                _resolved_main_type_x = _resolved_main_type_y = "scatter"
            elif "hist" in type or "hist2d" in type:
                # On mixed data the short name "hist" means the segmented
                # histogram; "hist2d" always forces the 2D mesh.
                if configuration == "2D_mixed" and "hist2d" not in type:
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
                    _resolved_main_type_x = _resolved_main_type_y = "segmented_hist"
                elif marginal:
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
                        fig, marginal, mappable, "Density" if normalize else "Count"
                    )
                    _resolved_main_type_x = _resolved_main_type_y = "hist2d"
                    if isinstance(histo, tuple):
                        _marginal_hist_edges = (histo[1], histo[2])
                else:
                    make_hist2d(x, y, ax, bins=bins, normalize=normalize, **kwargs)
            elif "density" in type or "density2d" in type:
                # On mixed data the short name "density" means the segmented
                # density; "density2d" always forces the 2D surface.
                if configuration == "2D_mixed" and "density2d" not in type:
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
                    _resolved_main_type_x = _resolved_main_type_y = "segmented_density"
                elif marginal:
                    den = make_density2D(x, y, ax, colorbar=False, **kwargs)
                    add_colorbar(fig, marginal, den, "Density")
                    _resolved_main_type_x = _resolved_main_type_y = "density2d"
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
                _resolved_main_type_x = _resolved_main_type_y = "segmented_rug"
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
                _resolved_main_type_x = _resolved_main_type_y = "segmented_density"
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
                _resolved_main_type_x = _resolved_main_type_y = "segmented_hist"
            elif "tile" in type:
                hm = make_tile(
                    x,
                    y,
                    ax,
                    normalize=normalize,
                    bins=bins,
                    discrete_x=discrete_x,
                    discrete_y=discrete_y,
                    colorbar=not marginal,
                    **kwargs,
                )
                # tile lays out a whole-number discrete axis at real
                # values (matplotlib's own locator), same as a continuous
                # axis -- only a categorical/non-whole-number/
                # pathological-range discrete axis falls back to
                # compacted rank-index cells (see setup_tile_axis). Each
                # axis is independent, so "tile" is only the resolved
                # type -- for DISCRETE_INDEX_OFFSET purposes -- on the
                # axis(es) that actually fell back to rank-index cells;
                # a real-valued axis gets None (treated like scatter/
                # hist2d/density2d: real values, no index conversion).
                _resolved_main_type_x = "tile" if discrete_x else None
                _resolved_main_type_y = "tile" if discrete_y else None
                if marginal:
                    add_colorbar(
                        fig,
                        marginal,
                        hm,
                        "Relative Frequency" if normalize else "Count",
                    )
                    _tile_bins = bins if bins is not None else TILE_DEFAULT_BINS
                    _, _, _, tile_x_ticks, tile_x_edges = setup_tile_axis(
                        x, discrete_x, _tile_bins
                    )
                    _, _, _, tile_y_ticks, tile_y_edges = setup_tile_axis(
                        y, discrete_y, _tile_bins
                    )
                    if discrete_x and tile_x_ticks is None:
                        _resolved_main_type_x = None
                    if discrete_y and tile_y_ticks is None:
                        _resolved_main_type_y = None
                    _marginal_hist_edges = (tile_x_edges, tile_y_edges)
            elif "mosaic" in type:
                make_mosaic(x, y, ax, normalize=normalize, **kwargs)
                _resolved_main_type_x = _resolved_main_type_y = "mosaic"
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
                _resolved_main_type_x = _resolved_main_type_y = "violin"
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
                _resolved_main_type_x = _resolved_main_type_y = "box"

            if marginal:
                edges_x, edges_y = _marginal_hist_edges
                wants_density = "density" in type or "density2d" in type
                marg_x_color = get_next_color(ax)
                marg_y_color = get_next_color(ax)
                _draw_marginal_panel(
                    ax_marg_x,
                    ax,
                    x,
                    discrete_x,
                    _resolved_main_type_x,
                    "x",
                    marg_x_color,
                    small_n,
                    normalize,
                    bins,
                    wants_density,
                    _marginal_bandwidth,
                    edges_x,
                )
                _draw_marginal_panel(
                    ax_marg_y,
                    ax,
                    y,
                    discrete_y,
                    _resolved_main_type_y,
                    "y",
                    marg_y_color,
                    small_n,
                    normalize,
                    bins,
                    wants_density,
                    _marginal_bandwidth,
                    edges_y,
                )
                # Read the main panel's own actual final limits (after
                # everything above has drawn) rather than re-deriving each
                # plot type's extent formula independently -- this is what
                # correctly aligns a marginal panel with e.g. tile's
                # (-0.5, n_cells - 0.5) index extent or violin/boxplot's
                # matplotlib-assigned category positions, without having to
                # hardcode any of those formulas here. sharex/sharey then
                # keeps them locked together for any later interaction.
                ax_marg_x.set_xlim(ax.get_xlim())
                ax_marg_x.sharex(ax)
                ax_marg_y.set_ylim(ax.get_ylim())
                ax_marg_y.sharey(ax)
                plt.setp(ax_marg_x.get_xticklabels(), visible=False)
                plt.setp(ax_marg_y.get_yticklabels(), visible=False)
                # Drop each marginal's own value-axis label ("Value"): the
                # main panel's X and Y labels already name those axes, and
                # the shared axes make the marginal's copy redundant clutter
                # right next to them. The frequency-axis label (Density /
                # Count) on each marginal is kept.
                ax_marg_x.set_xlabel("")
                ax_marg_y.set_ylabel("")
                # There is no room for the main panel's own title -- it would
                # collide with the strip above it -- but what the plot *is*
                # ("Tile Plot", "2-D Histogram") is worth keeping, so it moves
                # to the figure, above all three panels. Read it back from the
                # panel rather than re-deriving it, so it stays whatever the
                # plot type actually titled itself.
                fig.suptitle(ax.get_title())
                ax.set_title("")
                # Leave the joint panel current. Drawing the strips and the
                # colorbar moved plt.gca() off it (fig.add_axes makes its new
                # axes current), and the joint panel is the one plt.gca()
                # should mean after a two-variable plot.
                plt.sca(ax)
        elif self.index_set is None and _is_categorical_1d(self.results):
            # 1D categorical (string) outcomes. These are not numbers, so
            # they get dim=None and skip the dim == 1 branch, but they are a
            # genuine 1D categorical variable: route them to the
            # "1D_categorical" lookup configuration and the categorical plot
            # types (bar / dot plot / impulse), which all accept strings.
            values = np.asarray(list(self.results))
            discrete, small_n = classify_values(values)
            default, alternatives = default_plot_type("1D_categorical", small_n)
            # Same tall-stack fallback as the numeric 1D branch above: a
            # dot plot of a few categories at large-ish n stacks each into
            # an unreadable column, so redirect the default to the
            # large-n categorical default (a bar chart). Explicit
            # type='dotplot' is still honored.
            if (
                type is None
                and default == "dotplot"
                and dotplot_tallest_stack(values) > DOTPLOT_MAX_STACK
            ):
                default, alternatives = default_plot_type("1D_categorical", False)
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
