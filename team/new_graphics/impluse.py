import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# These helpers live in symbulate/plot.py, which stays through the
# graphics overhaul -- only the impulse code inside results.py is
# being deleted. make_impulse() below is its replacement.
from symbulate.plot import configure_axes, count_var, get_next_color

# Same rng pattern already used in results.py for impulse jitter.
rng = np.random.default_rng()


# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different line width for different plot types," so these
# cannot live in symbulate.mplstyle (see DECISIONS.md, "Decision:
# .mplstyle Standards"). They migrate to the top of symbulate/plot.py.
#
# Stem line width: DECISIONS.md, "Decision: Visual Style Guide" sets
# impulse lines at 1.5 -> 2.2 specifically for visibility (no
# overplotting risk, unlike histograms/scatter/density).
IMPULSE_LINEWIDTH = 2.2
IMPULSE_MARKER = "o"
IMPULSE_MARKER_SIZE = 90
# DECISIONS.md tunes alpha for hist/scatter/density but does not call
# out impulse; the approved prototype image shows fully opaque stems
# and markers, so impulse defaults to alpha=1.0 unless overridden.
IMPULSE_ALPHA = 1.0
IMPULSE_LEGEND_LOC = "upper right"
# When several impulse plots share the axes, adjacent series' stems are
# separated by this fraction of the smallest gap between values, and the
# whole group is re-centered so it straddles the value symmetrically --
# as in the approved prototype, where the two stems at each integer sit
# just left and just right of it. A lone series sits exactly on the
# values.
IMPULSE_SERIES_OFFSET = 0.35

# True-distribution overlay: a smooth curve through the exact pmf
# values, with no markers, so it reads as a reference curve rather
# than another simulated variable -- the same role a density curve
# plays over a histogram. Line width matches the density-curve
# standard (2.0 -> 1.8) for that reason.
TRUE_DIST_LINEWIDTH = 1.8
TRUE_DIST_LINESTYLE = "-"
TRUE_DIST_CURVE_POINTS = 600
TRUE_DIST_LABEL_DEFAULT = "True Distribution"


def _refresh_legend(ax):
    """Show a legend once two or more labeled series share the axes.

    A lone impulse plot (or a lone true-distribution overlay) stays
    legend-free; overlaying a second series -- another simulated
    variable, or a true distribution -- turns the legend on.
    """
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(loc=IMPULSE_LEGEND_LOC)


def make_impulse(
    values, ax, color, normalize=True, alpha=None, jitter=False, label=None, **kwargs
):
    """Draw a 1D impulse (stem) plot of simulated discrete values.

    This is the existing impulse block from ``RVResults.plot()`` in
    ``symbulate/results.py`` (counting via ``count_var``, jitter, and
    axis limits via ``configure_axes`` are unchanged) extended with a
    marker cap on each stem and per-series labeling so the plot matches
    the approved impulse prototype and overlays legibly.

    Impulse plots overlay naturally: a second call on the same axes
    draws on top of the first, and a legend appears automatically in
    the top right once two or more series (impulse plots and/or a true
    distribution overlay) share the axes. Each series is named by
    ``label``, or "Variable 1", "Variable 2", ... in call order when no
    label is given. When several series share the axes, the whole group
    is re-centered so their stems spread symmetrically around each
    value -- a pair straddles it, one just left and one just right,
    separated by ``IMPULSE_SERIES_OFFSET`` -- so no stem hides another
    and every stem still clearly belongs to its value. A lone series
    sits exactly on the values.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers in ``symbulate/plot.py``.

    Parameters
    ----------
    values : array-like
        The simulated values to count, e.g. ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the stems and markers, from ``get_next_color(ax)``.
    normalize : bool, default True
        If True, stem heights are relative frequencies that sum to 1,
        comparable to a pmf. If False, stem heights are raw counts.
    alpha : float, optional
        Marker/stem transparency between 0 and 1. Defaults to the
        package standard for impulse plots (1.0, fully opaque).
    jitter : bool, default False
        If True, add small random noise to discrete x-values to reduce
        overplotting, matching the ``jitter`` option already on
        ``RVResults.plot()``.
    label : str, optional
        Name for this series in the legend. Defaults to "Variable k",
        where k counts the impulse plots drawn on these axes so far.
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    tuple
        The ``(xs, freqs)`` values plotted, so the caller can inspect
        or further style the stems.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.random.default_rng().poisson(4, 1000)
    >>> ax = plt.gca()
    >>> make_impulse(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = IMPULSE_ALPHA

    # Unchanged from the "impulse" in type branch of RVResults.plot().
    n = len(values)
    counts = count_var(values)
    xs = list(counts.keys())
    freqs = list(counts.values())
    if normalize:
        freqs = [freq / n for freq in freqs]
    if jitter:
        a = 0.02 * (max(xs) - min(xs))
        xs = [x + rng.uniform(low=-a, high=a) for x in xs]

    # Track the impulse series drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle) so overlays from separate .plot() calls see them.
    prior_series = getattr(ax, "_impulse_series", [])
    if label is None:
        label = f"Variable {len(prior_series) + 1}"

    stems = ax.vlines(
        xs, 0, freqs, color=color, linewidth=IMPULSE_LINEWIDTH, alpha=alpha
    )
    dots = ax.scatter(
        xs,
        freqs,
        s=IMPULSE_MARKER_SIZE,
        marker=IMPULSE_MARKER,
        color=color,
        alpha=alpha,
        label=label,
        zorder=3,
        **kwargs,
    )
    prior_series.append(
        {
            "stems": stems,
            "dots": dots,
            "xs": np.asarray(xs, dtype=float),
            "freqs": np.asarray(freqs, dtype=float),
        }
    )
    ax._impulse_series = prior_series

    # Re-center the whole group so the series' stems spread symmetrically
    # around each value (a pair straddles it, one just left and one just
    # right), keeping every stem visibly attached to its value while no
    # stem hides another. The gap is measured per series so jittered
    # values can't shrink it.
    if len(prior_series) > 1:
        gaps = []
        for s in prior_series:
            unique_xs = np.unique(s["xs"])
            if len(unique_xs) > 1:
                gaps.append(np.diff(unique_xs).min())
        gap = min(gaps) if gaps else 1.0
        for i, s in enumerate(prior_series):
            offset = (i - (len(prior_series) - 1) / 2) * IMPULSE_SERIES_OFFSET * gap
            shifted_xs = s["xs"] + offset
            s["stems"].set_segments(
                [[(x, 0), (x, f)] for x, f in zip(shifted_xs, s["freqs"])]
            )
            s["dots"].set_offsets(np.column_stack([shifted_xs, s["freqs"]]))

    # Unchanged from RVResults.plot(): reuses the shared axis-limit
    # buffering / label logic every other plot type goes through.
    configure_axes(
        ax,
        xs,
        freqs,
        xlabel="Value",
        ylabel="Relative Frequency" if normalize else "Count",
    )

    fig = ax.get_figure()
    if not getattr(fig, "_impulse_suptitle_set", False):
        fig.suptitle("Impulse Graph", fontsize=16, fontweight="bold")
        fig._impulse_suptitle_set = True
    ax.set_title(
        "Relative Frequency Impulse Plot" if normalize else "Count Impulse Plot"
    )
    _refresh_legend(ax)
    return xs, freqs


def overlay_true_distribution(pmf, ax, xlim=None, color=None, label=None, **kwargs):
    """Overlay a discrete distribution's true pmf on an impulse plot.

    Adapts the discrete branch of ``Distribution.plot()``
    (``symbulate/distributions.py``): integer x-values via
    ``np.arange``, xlim from the existing axes, and
    ``get_next_color(ax)`` when no color is given. The difference is
    styling -- a smooth, marker-free curve through the pmf values
    (cubic spline interpolation, clipped at zero so the tails can't
    dip negative) -- so the true answer reads as a reference curve
    over the simulated stems, the way a density curve reads over a
    histogram.

    Meant to be called after ``make_impulse()`` on the same axes --
    e.g. ``Poisson(5).plot()`` layered on top of
    ``RV(Poisson(5)).sim(1000).plot(type="impulse")``. ``pmf`` accepts
    the ``.pdf`` / ``.pmf`` callable already exposed on distribution
    objects in ``symbulate/distributions.py`` (the two are aliases of
    each other for discrete distributions).

    Parameters
    ----------
    pmf : callable
        Vectorized function mapping an array of integers to
        probabilities, e.g. a ``Distribution`` object's ``.pmf``.
    ax : matplotlib.axes.Axes
        The axes to draw on -- typically the axes an impulse plot was
        already drawn on, so the two layers share a scale.
    xlim : tuple of int, optional
        Inclusive ``(min, max)`` range of integer values to evaluate
        the pmf at. Defaults to the current axes' x-limits, rounded
        outward to the nearest integers.
    color : color, optional
        Color for the markers and connecting line, from
        ``get_next_color(ax)``. Defaults to the next color in the
        cycle if not given, matching the other plot helpers.
    label : str, optional
        Name for this series in the legend. Defaults to
        "True Distribution".
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    tuple
        The ``(xs, ys)`` integer values and exact pmf heights the
        smooth curve passes through.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import poisson
    >>> ax = plt.gca()
    >>> overlay_true_distribution(lambda xs: poisson.pmf(xs, 5), ax)  # doctest: +SKIP
    """
    if xlim is None:
        xlower, xupper = ax.get_xlim()
        xlim = (int(np.floor(xlower)), int(np.ceil(xupper)))
    if color is None:
        color = get_next_color(ax)
    if label is None:
        label = TRUE_DIST_LABEL_DEFAULT

    xs = np.arange(xlim[0], xlim[1] + 1)
    ys = np.asarray(pmf(xs), dtype=float)

    # Smooth curve through the pmf values instead of dot-to-dot
    # segments. A cubic spline gives the roundest curve through the
    # peaks; any small dips below zero it introduces in the tails are
    # clipped away, since probabilities can't be negative.
    if len(xs) >= 2:
        curve_xs = np.linspace(xs[0], xs[-1], TRUE_DIST_CURVE_POINTS)
        spline_degree = min(3, len(xs) - 1)
        curve_ys = make_interp_spline(xs, ys, k=spline_degree)(curve_xs)
        curve_ys = np.clip(curve_ys, 0, None)
    else:
        curve_xs, curve_ys = xs, ys

    ax.plot(
        curve_xs,
        curve_ys,
        linestyle=TRUE_DIST_LINESTYLE,
        linewidth=TRUE_DIST_LINEWIDTH,
        color=color,
        label=label,
        zorder=4,
        **kwargs,
    )
    _refresh_legend(ax)
    return xs, ys


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    # Figure 1: the two-variable overlay prototype -- Variable 1 (sky
    # blue) peaking ~3-4, Variable 2 (orange) peaking ~5-6, the pair of
    # stems automatically straddling each value symmetrically.
    ax = plt.gca()
    make_impulse(list(rng.poisson(3.3, 5000)), ax, get_next_color(ax))
    make_impulse(list(rng.poisson(5.3, 5000)), ax, get_next_color(ax))

    # Figure 2: the true-distribution overlay -- simulated Poisson(5)
    # relative frequencies with the exact pmf drawn on top.
    from scipy.stats import poisson

    plt.figure()
    ax = plt.gca()
    make_impulse(list(rng.poisson(5, 2000)), ax, get_next_color(ax), label="Simulated")
    overlay_true_distribution(
        lambda xs: poisson.pmf(xs, 5), ax, color=get_next_color(ax)
    )

    plt.show()
