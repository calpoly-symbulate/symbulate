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

Also new are the structured jitter modes for the Discrete x Discrete
cell of the 2D lookup table. ``jitter="spiral"`` (formerly
``"orderly"``): instead of scattering coincident points with random
noise, the points that share an integer (x, y) coordinate form a tight
compass-pattern cluster centered on that value's grid crossing -- the
first point dead center on the crossing, the next four on the grid
lines at the cardinal positions (N, E, S, W), the next four at the
diagonals (NE, SE, NW, SW). Neighboring dots in a cluster touch: the
spacing is the dot's own on-screen diameter, converted to data units
through ``ax.transData`` and recomputed on resize -- the same
touching-dot geometry the dot plot uses -- so a cluster always reads
at natural dot size, never spread apart by the axis scale. Because
the cluster hugs its grid crossing, it reads as one shared value
(never as separate values), and a student can read each value's
density by counting, which random jitter makes impossible.
The compass template only holds 9, so past ~12 coincident points a
spiral pile-up stops being cleanly countable -- ``jitter="bins"``
switches to a histogram-style reading instead: the grid lines move to
the half-integer bin edges so each value gets a visible box with its
axis label centered inside (exactly how a histogram centers a bar
over its bin), and the points fill their box from the bottom-left
corner -- left to right, then up a row -- dots touching, like a tiny
dot histogram in each box. ``jitter="auto"`` picks between them
from the data: "bins" once any single value holds
SCATTER_AUTO_BINS_THRESHOLD or more points, "spiral" otherwise, with a
printed note when it switches. Random jitter (``jitter=True``) and no
jitter (``jitter=False``) are still available; plain scatter still
targets the Continuous x Continuous cell.

Every variant draws both horizontal and vertical grid lines, extending
the style sheet's horizontal reference grid; in spiral and bins modes
the ticks are forced to the integers so the grid lines pass through
the cluster centers.

The scatter plot targets small simulations -- the default plot lookup
table selects it only when n <= SCATTER_MAX_N (40) -- so the demos
below stick to n = 40.

Run this file directly to render the prototype:

    python new_graphics/scatter.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

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
# Spiral and bins jitter modes lay coincident points out with their
# dots touching: the spacing is the dot's own diameter converted to
# data units through ax.transData (recomputed on resize), so clusters
# read at natural dot size whatever the axis scale. This is how much
# of the value's bin (the cell between half-integer boundaries) a
# cluster may occupy before its spacing compresses instead of growing
# -- 0.8 leaves a 0.1 margin inside each bin edge, so a cluster never
# reaches the neighboring value.
SCATTER_BIN_SPREAD = 0.8
# Auto mode: jitter="auto" reads the data and uses "bins" once any
# single (x, y) value holds this many points -- past the 9-dot compass
# template, where a spiral pile-up stops being cleanly countable --
# and "spiral" otherwise.
SCATTER_AUTO_BINS_THRESHOLD = 12

# Default-plot lookup rule (Task 1A): the scatter plot targets small
# simulations; the lookup table selects it only when n <= SCATTER_MAX_N.
SCATTER_MAX_N = 40


def _refresh_legend(ax):
    """Show a legend once two or more labeled series share the axes.

    A lone scatter stays legend-free; overlaying a second series turns
    the legend on so the two variables can be told apart.
    """
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(loc=SCATTER_LEGEND_LOC)


# Structured-jitter fill order: a 3x3 compass template. The first point
# in a cell sits dead center, points 2-5 take the cardinal positions
# (N, E, S, W), and points 6-9 take the remaining diagonal positions
# (NE, SE, NW, SW).
_COMPASS_ORDER = [
    (0, 0),  # center
    (0, 1),  # N
    (1, 0),  # E
    (0, -1),  # S
    (-1, 0),  # W
    (1, 1),  # NE
    (1, -1),  # SE
    (-1, 1),  # NW
    (-1, -1),  # SW
]


def _axes_size_px(ax):
    """Return the rendered (width, height) of the axes in pixels."""
    (x0, y0), (x1, y1) = ax.transAxes.transform([(0.0, 0.0), (1.0, 1.0)])
    return x1 - x0, y1 - y0


def _dot_steps(ax):
    """Return the dot diameter in (x, y) data units.

    This is the spacing at which two neighboring dots exactly touch on
    screen -- the touching-dot geometry the dot plot prototype uses,
    measured through ``ax.transData`` so it tracks the current axis
    limits and figure size.
    """
    diameter_px = np.sqrt(SCATTER_MARKER_SIZE) * ax.figure.dpi / 72.0
    (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    px_per_xunit = max(abs(x1 - x0), 1e-9)
    px_per_yunit = max(abs(y1 - y0), 1e-9)
    return diameter_px / px_per_xunit, diameter_px / px_per_yunit


def _spiral_offsets(k, step_x, step_y):
    """Return ``k`` (dx, dy) offsets clustered on (0, 0), compass-first.

    Coincident points fill the compass template in ``_COMPASS_ORDER``:
    dead center first, then the cardinal positions (N, E, S, W) one
    dot diameter away, then the diagonals (NE, SE, NW, SW). The center
    point sits exactly on the grid intersection and the cardinal
    points lie on the grid lines themselves, so the whole cluster
    visibly belongs to that one (x, y) value. More than 9 coincident
    points no longer fit the template, so they fall back to a centered
    ``ceil(sqrt(k))``-column touching lattice. Either way the spacing
    compresses once the cluster would outgrow SCATTER_BIN_SPREAD, so
    it never reaches the neighboring value.
    """
    if k <= len(_COMPASS_ORDER):
        sx = min(step_x, SCATTER_BIN_SPREAD / 2.0)
        sy = min(step_y, SCATTER_BIN_SPREAD / 2.0)
        return [(dx * sx, dy * sy) for dx, dy in _COMPASS_ORDER[:k]]
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))
    sx = min(step_x, SCATTER_BIN_SPREAD / max(ncols - 1, 1))
    sy = min(step_y, SCATTER_BIN_SPREAD / max(nrows - 1, 1))
    offsets = []
    for m in range(k):
        row, col = divmod(m, ncols)
        offsets.append(((col - (ncols - 1) / 2.0) * sx, (row - (nrows - 1) / 2.0) * sy))
    return offsets


def _bins_offsets(k, step_x, step_y):
    """Return ``k`` (dx, dy) offsets filling the value's box like a
    tiny dot histogram.

    Dots start in the box's bottom-left corner and fill left to right,
    then move up a row, touching -- so a fuller box reads as a bigger
    fill, the way a taller histogram bar reads as a bigger count. Rows
    are kept square-ish (about sqrt(k) dots wide, capped by how many
    touching dots fit across the box) so a pile-up reads as a compact
    countable block rather than a long string of dots; if the rows
    would still outgrow the box vertically, the row spacing compresses
    to keep every dot inside its own value's box.
    """
    half = SCATTER_BIN_SPREAD / 2.0
    sx = min(step_x, SCATTER_BIN_SPREAD)
    max_cols = max(1, int(SCATTER_BIN_SPREAD / sx))
    ncols = min(max_cols, int(np.ceil(np.sqrt(k))))
    nrows = int(np.ceil(k / ncols))
    sy = min(step_y, SCATTER_BIN_SPREAD / nrows)
    offsets = []
    for m in range(k):
        row, col = divmod(m, ncols)
        offsets.append((-half + (col + 0.5) * sx, -half + (row + 0.5) * sy))
    return offsets


def _relayout_clusters(ax):
    """Recompute every cluster's dot positions from the current axes.

    The touching-dot spacing depends on the axis limits and rendered
    figure size, so this runs when the series is first drawn and again
    from the resize/draw hooks whenever the geometry changes (same
    pattern as the dot plot prototype).
    """
    state = getattr(ax, "_scatter_jitter_state", None)
    if state is None or not state["series"]:
        return
    step_x, step_y = _dot_steps(ax)
    for series in state["series"]:
        xi, yi = series["xi"], series["yi"]
        new_x = xi.astype(float).copy()
        new_y = yi.astype(float).copy()
        values = {}
        for i in range(len(xi)):
            values.setdefault((xi[i], yi[i]), []).append(i)
        for (cx, cy), idxs in values.items():
            if series["mode"] == "bins":
                offsets = _bins_offsets(len(idxs), step_x, step_y)
            else:
                offsets = _spiral_offsets(len(idxs), step_x, step_y)
            for pos, (dx, dy) in enumerate(offsets):
                new_x[idxs[pos]] = cx + dx
                new_y[idxs[pos]] = cy + dy
        series["dots"].set_offsets(np.column_stack([new_x, new_y]))
    state["last_geometry"] = _geometry_signature(ax)


def _geometry_signature(ax):
    """The rendered size and view limits the last layout was based on."""
    return (_axes_size_px(ax), ax.get_xlim(), ax.get_ylim())


def _on_canvas_change(ax):
    """Redo the cluster geometry if the axes' rendering has changed."""
    state = getattr(ax, "_scatter_jitter_state", None)
    if state is None or state["relayout_running"] or not state["series"]:
        return
    if state["last_geometry"] == _geometry_signature(ax):
        return
    state["relayout_running"] = True
    try:
        _relayout_clusters(ax)
        ax.figure.canvas.draw_idle()
    finally:
        state["relayout_running"] = False


def _init_jitter_state(ax):
    """Set up per-axes cluster state and the geometry-change hooks."""
    state = {"series": [], "last_geometry": None, "relayout_running": False}
    ax._scatter_jitter_state = state
    ax.figure.canvas.mpl_connect("resize_event", lambda event: _on_canvas_change(ax))
    ax.figure.canvas.mpl_connect("draw_event", lambda event: _on_canvas_change(ax))
    return state


def _max_coincident(x, y):
    """Count the points sharing the most-repeated integer (x, y) value."""
    pairs = np.column_stack([np.round(x).astype(int), np.round(y).astype(int)])
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    return int(counts.max())


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
        - ``"spiral"`` -- points that share an integer (x, y)
          coordinate form a tight compass-pattern cluster on that
          value's grid crossing (center first, then N/E/S/W on the
          grid lines, then the diagonals), neighboring dots touching,
          so the cluster reads as one shared value at natural dot
          size. Right for two discrete variables with modest pile-ups:
          a student can read each value's density by counting. Assumes
          the data are integer-valued.
        - ``"bins"`` -- histogram-style boxes: the grid lines move to
          the half-integer bin edges, so each value gets a visible box
          with its axis label centered inside, the way a histogram
          centers a bar over its bin. Points sharing a value fill
          their box from the bottom-left corner -- left to right, then
          up a row -- dots touching, like a tiny dot histogram, so a
          fuller box means a bigger count even when dozens of points
          share one value.
        - ``"auto"`` -- pick for the data: ``"bins"`` once any single
          value holds ``SCATTER_AUTO_BINS_THRESHOLD`` (12) or more
          points, ``"spiral"`` otherwise. Prints a note when it
          chooses bins.
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
    >>> x = rng.uniform(0, 7, 40)
    >>> y = 0.8 * x + rng.normal(0, 0.7, 40)
    >>> ax = plt.gca()
    >>> make_scatter(x, y, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = SCATTER_ALPHA
    if jitter not in (False, True, "spiral", "bins", "auto"):
        raise ValueError(
            f"jitter must be False, True, 'spiral', 'bins', or 'auto', not "
            f"{jitter!r}. For two discrete variables use jitter='auto' (picks "
            "the best layout for your data), jitter='spiral' (tight countable "
            "clusters), or jitter='bins' (spreads big pile-ups across each "
            "value's bin). Use jitter=True for random noise, or jitter=False "
            "(the default) for continuous variables."
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if jitter == "auto":
        peak = _max_coincident(x, y)
        if peak >= SCATTER_AUTO_BINS_THRESHOLD:
            jitter = "bins"
            print(
                f"The most repeated value appears {peak} times -- too many "
                "for the tight cluster style to stay countable -- so the "
                "points are spread across each value's bin "
                "(jitter='bins'). Pass jitter='spiral' to force tight "
                "clusters instead."
            )
        else:
            jitter = "spiral"

    xi = yi = None
    if jitter in ("spiral", "bins"):
        # Pin the view at least half a unit past the occupied crossings
        # (via the data limits, so later overlays still autoscale):
        # otherwise a lone heavily-repeated value makes autoscale zoom
        # into the cluster itself, which then reads as many separate
        # values instead of one.
        xi = np.round(x).astype(int)
        yi = np.round(y).astype(int)
        ax.update_datalim(
            [(xi.min() - 0.5, yi.min() - 0.5), (xi.max() + 0.5, yi.max() + 0.5)]
        )
        # Draw at the value centers for now; the touching-dot cluster
        # layout needs settled axis limits, so it happens at the end of
        # this call (and again from the resize/draw hooks).
        x, y = xi.astype(float), yi.astype(float)
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

    if jitter in ("spiral", "bins"):
        # Integer major ticks so every value label sits at its integer
        # position (spiral: grid lines through the cluster centers;
        # bins: labels centered in their boxes).
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    if jitter == "bins":
        # Histogram-style boxes: move the grid lines to the half-integer
        # bin edges (minor ticks, marks hidden) so each value gets a
        # visible box with its label centered inside, like a histogram
        # bar over its bin.
        ax.xaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
        ax.tick_params(which="minor", length=0)
        ax.grid(False, axis="both", which="major")
        ax.grid(True, axis="both", which="minor")
    else:
        # Vertical grid lines on top of the style sheet's horizontal
        # ones, so values read off both axes.
        ax.grid(True, axis="both")

    ax.set_xlabel("Variable 1" if xlabel is None else xlabel)
    ax.set_ylabel("Variable 2" if ylabel is None else ylabel)
    ax.set_title("2D Scatter Plot")
    _refresh_legend(ax)

    if jitter in ("spiral", "bins"):
        state = getattr(ax, "_scatter_jitter_state", None)
        if state is None:
            state = _init_jitter_state(ax)
        state["series"].append({"dots": points, "xi": xi, "yi": yi, "mode": jitter})
        # Settle the view limits now so the touching-dot spacing is
        # measured against the geometry that will actually render.
        ax.autoscale_view()
        _relayout_clusters(ax)

    return points


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    # Scatter targets small simulations (n <= SCATTER_MAX_N), so every
    # figure below uses n = 40.

    # Figure 1: the single-scatter prototype -- two positively
    # correlated continuous variables, sky blue (first cycle color).
    x = rng.uniform(0, 7, 40)
    y = 0.8 * x + rng.normal(0, 0.8, 40)

    ax = plt.gca()
    make_scatter(x, y, ax, get_next_color(ax))

    # Figure 2: the overlay prototype -- a second pair of variables on
    # the same axes picks up orange and turns on the "Variable k"
    # legend automatically.
    x2 = rng.uniform(0, 7, 40)
    y2 = 0.5 * x2 + rng.normal(2.5, 0.8, 40)

    plt.figure()
    ax = plt.gca()
    make_scatter(x, y, ax, get_next_color(ax))
    make_scatter(x2, y2, ax, get_next_color(ax))

    # Figure 3: the Discrete x Discrete prototype -- positively
    # correlated integer data with spiral jitter, so coincident
    # points form tight compass clusters (center, then N/E/S/W, then
    # the diagonals) on the grid crossings, countable per value.
    xd = rng.integers(1, 6, 40)
    yd = np.clip(xd + rng.integers(-1, 2, 40), 1, 5)

    plt.figure()
    ax = plt.gca()
    make_scatter(xd, yd, ax, get_next_color(ax), jitter="spiral")

    # Figure 4: a heavy pile-up -- skewed discrete data concentrated
    # enough that one value holds a dozen-plus points (around 18 of the
    # 40 land on (0, 0)). jitter="auto" notices and switches to the
    # bins layout (printing a note), spreading the big cluster across
    # its value's bin so every dot stays separated.
    xh = rng.binomial(2, 0.15, 40)
    yh = rng.binomial(2, 0.2, 40)

    plt.figure()
    ax = plt.gca()
    make_scatter(xh, yh, ax, get_next_color(ax), jitter="auto")

    # Figure 5: the same pile-up with jitter="bins" requested
    # explicitly -- histogram-style boxes with the grid lines on the
    # half-integer bin edges and each value label centered in its box.
    plt.figure()
    ax = plt.gca()
    make_scatter(xh, yh, ax, get_next_color(ax), jitter="bins")

    plt.show()
