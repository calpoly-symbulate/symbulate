"""Draft of the histogram plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_hist()`` below when the user asks for a histogram or the default
plot lookup table selects one (1D continuous-ish data).

Visual target: the approved histogram prototype (Task 1E) -- Okabe-Ito
bars with white edges, density-normalized, light horizontal reference
grid, no top or right spines. The grid, spines, fonts, and figure size
come from ``symbulate/symbulate.mplstyle``; the per-plot-type values
(alpha, edge styling, bin count) are the named constants below, which
migrate to the top of ``symbulate/plot.py`` at integration time.

Run this file directly to render the prototype:

    python team/new_graphics/hist.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
HIST_ALPHA = 0.65
HIST_EDGECOLOR = "white"
HIST_EDGEWIDTH = 0.8
HIST_DEFAULT_BINS = 30
HIST_LEGEND_LOC = "upper right"


def make_hist(
    values, ax, color, bins=None, normalize=True, alpha=None, label=None, **kwargs
):
    """Draw a 1D histogram of simulated values on the given axes.

    Draws in the style of the approved histogram prototype: solid
    bars with thin white edges so adjacent bars stay visually
    distinct. The x-axis is always labeled "Value"; the y-axis label
    and title read "Density" / "Density Histogram" when normalized,
    "Count" / "Count Histogram" otherwise.

    Histograms overlay naturally: a second call on the same axes draws
    on top of the first, and a legend appears automatically in the top
    right once two or more histograms share the axes. Each histogram
    is named by ``label``, or "Variable 1", "Variable 2", ... in call
    order when no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_tile``, ``make_density2D``) are called.

    Parameters
    ----------
    values : array-like
        The simulated values to bin, e.g. ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the bars, from ``get_next_color(ax)``.
    bins : int, optional
        Number of equal-width bins. Defaults to 30.
    normalize : bool, default True
        If True, bar areas sum to 1 so the histogram approximates a
        density and can be compared to a pdf curve. If False, bar
        heights are raw counts.
    alpha : float, optional
        Bar transparency between 0 and 1. Defaults to the package
        standard for histograms (0.65).
    label : str, optional
        Name for this histogram in the legend. Defaults to
        "Variable k", where k counts the histograms drawn on these
        axes so far.
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    tuple
        The ``(counts, bin_edges, patches)`` tuple from ``ax.hist``,
        so the caller can inspect or further style the bars.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from symbulate.plot import get_next_color
    >>> values = np.random.default_rng().normal(10, 2, 1000)
    >>> ax = plt.gca()
    >>> make_hist(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if bins is None:
        bins = HIST_DEFAULT_BINS
    if alpha is None:
        alpha = HIST_ALPHA
    # Count the histograms drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle) so overlays from separate .plot() calls see it.
    n_prior_hists = getattr(ax, "_hist_count", 0)
    if label is None:
        label = f"Variable {n_prior_hists + 1}"
    ax._hist_count = n_prior_hists + 1
    histogram = ax.hist(
        values,
        bins=bins,
        density=normalize,
        color=color,
        alpha=alpha,
        edgecolor=HIST_EDGECOLOR,
        linewidth=HIST_EDGEWIDTH,
        label=label,
        **kwargs,
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if normalize else "Count")
    ax.set_title("Density Histogram" if normalize else "Count Histogram")
    # A legend only helps once there is more than one histogram to
    # tell apart; a lone histogram stays legend-free.
    if ax._hist_count > 1:
        ax.legend(loc=HIST_LEGEND_LOC)
    return histogram


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    values = rng.normal(10, 2, 2000)

    # Figure 1: the single-histogram prototype.
    ax = plt.gca()
    # Okabe-Ito sky blue -- the *first* hue in the symbulate.mplstyle
    # color cycle, so every plot type starts from this same color;
    # once integrated, it will come from get_next_color(ax) instead.
    make_hist(values, ax, "#56B4E9")

    # Figure 2: the overlay prototype -- three count histograms with
    # the automatic "Variable k" legend in the top right.
    plt.figure()
    ax = plt.gca()
    make_hist(rng.normal(9, 2, 1500), ax, "#56B4E9", normalize=False)
    make_hist(rng.normal(17, 3, 1500), ax, "#E69F00", normalize=False)
    make_hist(rng.normal(14, 3, 1500), ax, "#009E73", normalize=False)

    plt.show()
