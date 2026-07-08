"""Draft of the 2D scatter plot type for the graphics overhaul.

Staging code -- not yet wired into the package. This file is the
planned replacement for the scatter block inside ``RVResults.plot()``
(``symbulate/results.py``, the ``"scatter" in type`` branch of the
dim == 2 case), which is slated for deletion. It depends only on
``symbulate/plot.py`` -- which stays -- and carries the existing
branch's logic over unchanged: the jitter noise formula and the core
``ax.scatter`` call are copied from ``results.py`` as-is.

What's new on top of that carried-over logic is what the current
inline code doesn't do: per-series labels with an automatic legend so
overlaid scatters stay tellable-apart, axis labels and a title, and a
filled-circle point style at low alpha, so overlapping points stay
readable while a lone point still reads as solid. (Note: the finalized
Visual Style Guide in DECISIONS.md currently specifies *unfilled*
circles for scatter; filled dots are a pending override -- flag for the
decision log.)

Also new is an orderly-jitter mode for the Discrete x Discrete cell of
the 2D lookup table (``jitter="orderly"``): instead of scattering
coincident points with random noise, the points that share an integer
(x, y) coordinate are laid out in a neat centered sub-grid inside a
unit "bin" cell, and the grid lines are redrawn at the half-integer bin
boundaries so each integer sits centered in its own cell. This lets a
student read the density in each cell by counting, which random jitter
makes impossible. Random jitter (``jitter=True``) and no jitter
(``jitter=False``) are still available; plain scatter still targets the
Continuous x Continuous cell.

Run this file directly to render the prototype:

    python new_graphics/scatter.py
"""

import matplotlib.pyplot as plt
import numpy as np

# This helper lives in symbulate/plot.py, which stays through the
# graphics overhaul -- only the plotting code inside results.py is
# being deleted. make_scatter() below is its replacement.
from symbulate.plot import get_next_color

# Same rng pattern already used in results.py for scatter jitter.
rng = np.random.default_rng()

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
#
# Alpha follows the finalized Visual Style Guide (scatter 0.5 -> 0.25),
# low so overlapping points stay individually readable instead of
# turning into a solid blob. Point style is filled circles -- note this
# overrides the Guide's *unfilled* scatter decision (pending decision
# log update).
SCATTER_ALPHA = 0.25
SCATTER_MARKER_SIZE = 40
SCATTER_LEGEND_LOC = "upper right"
# Orderly-jitter mode: the ordered points fill this fraction of the unit
# bin cell, leaving a margin so no point touches a bin boundary line.
SCATTER_BIN_SPREAD = 0.7


def _refresh_legend(ax):
    """Show a legend once two or more labeled series share the axes.

    A lone scatter stays legend-free; overlaying a second series turns
    the legend on so the two variables can be told apart.
    """
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(loc=SCATTER_LEGEND_LOC)


def _grid_offsets(k, spread):
    """Return ``k`` (dx, dy) offsets on a centered square-ish lattice.

    The offsets fill a region ``spread`` wide and tall, centered on
    (0, 0), so ``k`` coincident points can be spread into an orderly
    block inside a bin cell. A single point sits dead center; larger
    counts fill a ``ceil(sqrt(k))``-column grid row by row (the last
    row may be partly filled, which still reads as a tidy lattice).
    """
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))
    offsets = []
    for m in range(k):
        row, col = divmod(m, ncols)
        dx = 0.0 if ncols == 1 else (col / (ncols - 1) - 0.5) * spread
        dy = 0.0 if nrows == 1 else (row / (nrows - 1) - 0.5) * spread
        offsets.append((dx, dy))
    return offsets


def _orderly_bin_layout(x, y, spread):
    """Lay coincident discrete points out in orderly sub-grids.

    Each point is assigned to the integer bin nearest its (x, y), then
    the points sharing a bin are spread onto a centered lattice inside
    that unit cell (see ``_grid_offsets``). Returns the new float
    coordinates plus the integer bins, so the caller can draw bin grid
    lines over the right range.
    """
    xr = np.round(x).astype(int)
    yr = np.round(y).astype(int)
    new_x = xr.astype(float).copy()
    new_y = yr.astype(float).copy()

    bins = {}
    for i in range(len(xr)):
        bins.setdefault((xr[i], yr[i]), []).append(i)

    for (cx, cy), idxs in bins.items():
        for pos, (dx, dy) in enumerate(_grid_offsets(len(idxs), spread)):
            new_x[idxs[pos]] = cx + dx
            new_y[idxs[pos]] = cy + dy

    return new_x, new_y, xr, yr


def _draw_bin_grid(ax, xr, yr):
    """Draw grid lines at the half-integer bin boundaries.

    Integer values get labeled major ticks centered in each cell, and
    the grid is drawn on unlabeled minor ticks halfway between them, so
    every integer (x, y) sits in the middle of its own bin cell. The
    integer extent is unioned across calls, so an overlaid second series
    extends the same bin grid rather than replacing it.
    """
    xmin, xmax = int(xr.min()), int(xr.max())
    ymin, ymax = int(yr.min()), int(yr.max())
    prev = getattr(ax, "_bin_extent", None)
    if prev is not None:
        xmin, xmax = min(xmin, prev[0]), max(xmax, prev[1])
        ymin, ymax = min(ymin, prev[2]), max(ymax, prev[3])
    ax._bin_extent = (xmin, xmax, ymin, ymax)

    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))
    ax.set_xticks(np.arange(xmin - 0.5, xmax + 1.0, 1.0), minor=True)
    ax.set_yticks(np.arange(ymin - 0.5, ymax + 1.0, 1.0), minor=True)
    ax.grid(False, which="major")
    ax.grid(True, which="minor", axis="both")


def make_scatter(
    x,
    y,
    ax,
    color,
    alpha=None,
    jitter=False,
    label=None,
    xlabel=None,
    ylabel=None,
    **kwargs,
):
    """Draw a 2D scatter plot of paired simulated values.

    This is the existing scatter block from ``RVResults.plot()`` in
    ``symbulate/results.py`` (the jitter noise formula and the core
    scatter call are unchanged) extended with per-series labeling, an
    automatic legend, axis labels, and a filled-circle point style.

    Scatter plots overlay naturally: a second call on the same axes
    draws on top of the first, and a legend appears automatically in
    the top right once two or more series share the axes. Each series
    is named by ``label``, or "Variable 1", "Variable 2", ... in call
    order when no label is given. The low alpha keeps the overlap
    region readable even with filled points.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers in ``symbulate/plot.py``.

    Parameters
    ----------
    x : array-like
        The first coordinate of each simulated pair, e.g. column 0 of
        ``RVResults.array``.
    y : array-like
        The second coordinate of each simulated pair, e.g. column 1 of
        ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the circles, from ``get_next_color(ax)``.
    alpha : float, optional
        Point transparency between 0 and 1. Defaults to the package
        standard for scatter plots (0.25).
    jitter : bool or str, default False
        How to spread out coincident points (discrete data):

        - ``False`` -- draw points at their exact coordinates. Right
          for two continuous variables (no coincident points).
        - ``True`` -- add small random noise to both coordinates,
          matching the ``jitter`` option already on
          ``RVResults.plot()``. Reduces overplotting but scrambles
          density.
        - ``"orderly"`` -- lay the points that share an integer
          (x, y) coordinate out in a neat centered sub-grid inside a
          unit bin cell, and draw the grid lines at the half-integer
          bin boundaries. Right for two discrete variables: a student
          can read each cell's density by counting. Assumes the data
          are integer-valued.
    label : str, optional
        Name for this series in the legend. Defaults to "Variable k",
        where k counts the scatters drawn on these axes so far.
    xlabel : str, optional
        Label for the x-axis. Defaults to "Variable 1".
    ylabel : str, optional
        Label for the y-axis. Defaults to "Variable 2".
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    matplotlib.collections.PathCollection
        The collection from ``ax.scatter``, so the caller can inspect
        or further style the points.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.uniform(0, 7, 200)
    >>> y = 0.8 * x + rng.normal(0, 0.7, 200)
    >>> ax = plt.gca()
    >>> make_scatter(x, y, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = SCATTER_ALPHA
    if jitter not in (False, True, "orderly"):
        raise ValueError(
            f"jitter must be False, True, or 'orderly', not {jitter!r}. "
            "Use jitter='orderly' for two discrete variables, jitter=True "
            "for random noise, or jitter=False (the default) for continuous "
            "variables."
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    bins_xr = bins_yr = None
    if jitter == "orderly":
        x, y, bins_xr, bins_yr = _orderly_bin_layout(x, y, SCATTER_BIN_SPREAD)
    elif jitter:
        # Unchanged from the "scatter" in type branch of RVResults.plot().
        x = x + rng.normal(loc=0, scale=0.01 * (x.max() - x.min()), size=len(x))
        y = y + rng.normal(loc=0, scale=0.01 * (y.max() - y.min()), size=len(y))

    # Count the scatters drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color cycle)
    # so overlays from separate .plot() calls see it.
    n_prior_scatters = getattr(ax, "_scatter_count", 0)
    if label is None:
        label = f"Variable {n_prior_scatters + 1}"
    ax._scatter_count = n_prior_scatters + 1

    points = ax.scatter(
        x,
        y,
        s=SCATTER_MARKER_SIZE,
        color=color,
        alpha=alpha,
        label=label,
        **kwargs,
    )

    if jitter == "orderly":
        _draw_bin_grid(ax, bins_xr, bins_yr)

    ax.set_xlabel("Variable 1" if xlabel is None else xlabel)
    ax.set_ylabel("Variable 2" if ylabel is None else ylabel)
    ax.set_title("2D Scatter Plot")
    _refresh_legend(ax)
    return points


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    # Figure 1: the single-scatter prototype -- two positively
    # correlated continuous variables, sky blue (first cycle color).
    x = rng.uniform(0, 7, 300)
    y = 0.8 * x + rng.normal(0, 0.8, 300)

    ax = plt.gca()
    make_scatter(x, y, ax, get_next_color(ax))

    # Figure 2: the overlay prototype -- a second pair of variables on
    # the same axes picks up orange and turns on the "Variable k"
    # legend automatically.
    x2 = rng.uniform(0, 7, 300)
    y2 = 0.5 * x2 + rng.normal(2.5, 0.8, 300)

    plt.figure()
    ax = plt.gca()
    make_scatter(x, y, ax, get_next_color(ax))
    make_scatter(x2, y2, ax, get_next_color(ax))

    # Figure 3: the Discrete x Discrete prototype -- positively
    # correlated integer data with orderly jitter, so coincident points
    # fill neat sub-grids inside bin cells you can count.
    xd = rng.integers(0, 7, 400)
    yd = np.clip(xd + rng.integers(-1, 2, 400), 0, 6)

    plt.figure()
    ax = plt.gca()
    make_scatter(xd, yd, ax, get_next_color(ax), jitter="orderly")

    plt.show()
