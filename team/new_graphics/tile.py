"""Draft of the 2D tile plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_tile()`` below for 2D data with at least one discrete axis
(``type="tile"``), replacing the current ``make_tile`` in
``symbulate/plot.py`` (``matshow``-based, hardcoded ``cmap="Blues"``, no
colorbar, relative-frequency only).

Visual target: a grid of filled cells colored by how often each cell
occurred, in the package's sequential colormap (viridis, from
``symbulate.mplstyle``), with a colorbar on the right labeled "Relative
Frequency" or "Count". A discrete axis gets one labeled cell per distinct
value, equal-sized regardless of the gaps between values; a continuous
axis is split into equal-width bins and labeled like a numeric histogram
axis.

The tile plot handles two discrete variables *and* the mixed case -- one
discrete axis, one continuous axis -- automatically binning whichever
axis is continuous, with the same equal-width binning ``make_hist2d``
uses. (The old ``setup_tile`` had this continuous branch too; an earlier
draft dropped it as a duplicate of ``make_hist2d``, but the mixed
discrete-x-continuous configuration is genuinely neither plot type's job,
so it lives here: tile owns "grid of labeled cells", ``make_hist2d`` owns
the fully-binned continuous-x-continuous mesh.) Continuous-x-continuous
data is still routed to ``hist2d`` / ``density2d`` instead.

Overlay category (see CLAUDE.md, "Overlay Policy"): **readability
warning** -- a second tile plot on the same axes is drawn, but the two
color scales compete, so a student-friendly warning is printed below
the plot (the same category as two 2D histograms).

Run this file directly to render the prototype:

    python team/new_graphics/tile.py
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express per-plot-type values, so these cannot live in
# symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py. The
# colormap itself is NOT set here -- it comes from image.cmap
# (viridis) in symbulate.mplstyle.
TILE_DEFAULT_BINS = 30  # equal-width bins for a continuous axis; matches
# make_hist2d's default so mixed discrete/continuous tiles bin the same way
TILE_CBAR_SIZE = "5%"  # colorbar width, as a fraction of the axes
TILE_CBAR_PAD = 0.1  # gap between the axes and the colorbar, inches
TILE_CBAR_TICKS = 8  # number of evenly spaced colorbar ticks, including
# both endpoints (0 and the peak value); matches density2d's colorbar
TILE_CBAR_DECIMALS = 3  # decimal places for the relative-frequency colorbar
# labels (raw counts are labeled as whole numbers instead)

TILE_OVERLAY_WARNING = (
    "Warning: you drew a second tile plot on the same plot. The two "
    "color scales compete, so the result may be hard to read. Consider "
    "plotting them in separate cells, or using type='scatter' instead."
)


def _setup_tile_axis(values, discrete, bins):
    """Cell indices, count, extent, and ticks for one tile-plot axis.

    A discrete axis gets one cell per distinct value; a continuous axis
    is split into ``bins`` equal-width bins the same way ``make_hist2d``
    does (``np.histogram``-style edges, with the largest value falling in
    the last bin).

    Parameters
    ----------
    values : numpy.ndarray
        The simulated values for this axis.
    discrete : bool
        If True, use one labeled cell per distinct value. If False, bin
        the values into ``bins`` equal-width bins.
    bins : int
        Number of bins to use when ``discrete`` is False.

    Returns
    -------
    tuple
        ``(idx, n_cells, extent, ticks)`` -- the cell index of every
        value, the number of cells along the axis, the ``(low, high)``
        imshow extent for the axis, and either ``(positions, labels)``
        for a discrete axis or ``None`` for a continuous one (whose ticks
        are left to matplotlib's numeric locator).
    """
    if discrete:
        labels = np.unique(values)
        idx = np.searchsorted(labels, values)
        n_cells = len(labels)
        # imshow centers each cell on its integer index, so a cell spans
        # index +/- 0.5.
        extent = (-0.5, n_cells - 0.5)
        ticks = (np.arange(n_cells), labels)
    else:
        low, high = values.min(), values.max()
        if low == high:
            # Degenerate axis (all one value): widen so the bins have
            # nonzero width instead of collapsing.
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, bins + 1)
        # digitize returns 1..len(edges); shift to 0-based bins and clip
        # the maximum value (which lands one past the last bin) back in.
        idx = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
        n_cells = bins
        # Data-unit extent so matplotlib labels the axis like a numeric
        # histogram axis; the equal-width imshow columns line up exactly
        # with the equal-width bins.
        extent = (edges[0], edges[-1])
        ticks = None
    return idx, n_cells, extent, ticks


def make_tile(
    x, y, ax, normalize=True, bins=None, discrete_x=None, discrete_y=None, **kwargs
):
    """Draw a 2D tile plot of simulated (x, y) pairs on the given axes.

    Draws a grid of filled cells colored by how often each cell occurred,
    using the package's sequential colormap (viridis, from
    ``symbulate.mplstyle``), with a colorbar on the right labeled
    "Relative Frequency" (or "Count" when ``normalize=False``). The
    x-axis is labeled "X", the y-axis "Y", and the title reads "Tile
    Plot". The color scale always starts from 0 so a cell that never
    occurred reads as "no data" rather than an arbitrary color. The
    colorbar ticks both endpoints (0 and the peak value) with
    ``TILE_CBAR_TICKS`` (8) evenly spaced ticks; relative-frequency
    labels are rounded to ``TILE_CBAR_DECIMALS`` (3) decimals and raw
    counts to whole numbers. This matches the density2d colorbar.

    Each axis is handled according to whether its variable is discrete or
    continuous, detected automatically from the data (see ``discrete_x`` /
    ``discrete_y``). A discrete axis gets one labeled cell per distinct
    value, equal-sized regardless of the gaps between values. A continuous
    axis is split into ``bins`` equal-width bins -- the same binning
    ``make_hist2d`` uses -- and labeled like a numeric histogram axis.
    This covers both two-discrete-variable data and the mixed case (one
    discrete axis, one continuous axis); continuous-x-continuous data is
    routed to ``hist2d`` / ``density2d`` instead.

    Unlike the 1D plot types, a tile plot encodes magnitude with a
    colormap instead of the categorical color cycle, so no ``color``
    parameter is taken. Overlays are the "readability warning" category
    of the overlay policy: a second call on the same axes still draws,
    but prints a warning that the color scales compete.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_hist2d``, ``make_density2D``) are
    called.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first
        column of ``RVResults.array``. Discrete or continuous.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
        Discrete or continuous.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    normalize : bool, default True
        If True, cell colors show the relative frequency of each cell
        (all cells sum to 1). If False, colors show raw counts.
    bins : int, optional
        Number of equal-width bins for a continuous axis, matching
        ``make_hist2d``. Defaults to ``TILE_DEFAULT_BINS`` (30). Only
        applies when at least one axis is continuous; passing it for two
        discrete variables has no effect and raises a warning.
    discrete_x : bool, optional
        Whether the x-axis is discrete (one cell per value) or continuous
        (binned). If None (default), detected from the data: float values
        are treated as continuous, everything else (int, bool, string) as
        discrete.
    discrete_y : bool, optional
        Same as ``discrete_x`` for the y-axis.
    **kwargs
        Additional keyword arguments passed to ``ax.imshow``.

    Returns
    -------
    matplotlib.image.AxesImage
        The image from ``ax.imshow``, so the caller can inspect the
        cells or attach further styling.

    Examples
    --------
    Two discrete variables:

    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.poisson(3, 1000)
    >>> y = rng.poisson(2, 1000)
    >>> ax = plt.gca()
    >>> make_tile(x, y, ax)  # doctest: +SKIP

    Mixed data -- discrete y, continuous x (x is binned automatically):

    >>> x = rng.normal(0, 1, 1000)
    >>> y = rng.poisson(3, 1000)
    >>> make_tile(x, y, plt.gca())  # doctest: +SKIP
    """
    xs, ys = np.asarray(x), np.asarray(y)
    # Auto-detect which axes are continuous (and so need binning) from
    # their dtype: float is treated as continuous, everything else (int,
    # bool, string/object) as discrete. The caller can override either
    # decision with discrete_x / discrete_y -- once classify_data is wired
    # in (Phase 2), results.py will pass those explicitly.
    if discrete_x is None:
        discrete_x = not np.issubdtype(xs.dtype, np.floating)
    if discrete_y is None:
        discrete_y = not np.issubdtype(ys.dtype, np.floating)

    # bins only bins a continuous axis. With two discrete variables there is
    # nothing to bin -- every distinct value already gets its own cell -- so
    # a bins= argument has no effect. Warn rather than silently ignore it
    # (same pattern as make_density2D's levels= warning). This check is on
    # the explicit argument, so it fires before the default is filled in.
    if bins is not None and discrete_x and discrete_y:
        warnings.warn(
            "bins only applies when one axis is continuous (it sets how many "
            "equal-width bins that axis is split into). Both of these "
            "variables are discrete, so every distinct value already gets its "
            "own cell and bins was ignored.",
            UserWarning,
            stacklevel=2,
        )
    if bins is None:
        bins = TILE_DEFAULT_BINS

    # Count the tile plots drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_tile_count", 0)
    ax._tile_count = n_prior + 1

    # Build each axis independently: a discrete axis gets one labeled cell
    # per distinct value; a continuous axis is binned into equal-width
    # bins exactly like make_hist2d.
    x_idx, nx, x_extent, x_ticks = _setup_tile_axis(xs, discrete_x, bins)
    y_idx, ny, y_extent, y_ticks = _setup_tile_axis(ys, discrete_y, bins)
    intensity = np.zeros((ny, nx))
    np.add.at(intensity, (y_idx, x_idx), 1)
    if normalize:
        intensity /= len(xs)

    # No cmap argument: the sequential colormap comes from image.cmap
    # (viridis) in symbulate.mplstyle. vmin=0 anchors the color scale
    # at zero. origin="lower" puts the smallest values at bottom-left;
    # aspect="auto" lets the cells fill the axes box. The extent puts a
    # binned axis in data units (so matplotlib labels it like a numeric
    # histogram axis) and a discrete axis in cell-index units.
    mesh = ax.imshow(
        intensity,
        origin="lower",
        aspect="auto",
        vmin=0,
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
        **kwargs,
    )
    # A filled mesh covers the whole axes, so the reference grid has
    # nothing to sit on -- turn it off rather than let fragments show
    # at the edges.
    ax.grid(False)
    # A discrete axis gets one tick per cell, labeled with the value; a
    # continuous (binned) axis keeps matplotlib's automatic numeric ticks
    # over its data range, matching the make_hist2d look.
    if x_ticks is not None:
        ax.set_xticks(x_ticks[0])
        ax.set_xticklabels(x_ticks[1])
    if y_ticks is not None:
        ax.set_yticks(y_ticks[0])
        ax.set_yticklabels(y_ticks[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Tile Plot")
    # Colorbar on the right, sized relative to the axes so it tracks
    # figure resizing (the approved replacement for the old hardcoded
    # fig.add_axes colorbar).
    cax = make_axes_locatable(ax).append_axes(
        "right", size=TILE_CBAR_SIZE, pad=TILE_CBAR_PAD
    )
    cbar = ax.get_figure().colorbar(mesh, cax=cax)
    cbar.set_label("Relative Frequency" if normalize else "Count")
    # Match the density2d colorbar: the color scale starts at 0 (vmin=0
    # above), and we tick both ends of the bar with a fixed number of
    # evenly spaced ticks -- matplotlib's default locator otherwise trims
    # short of the endpoints. get_clim() is authoritative once the colorbar
    # exists. Labels are rounded to TILE_CBAR_DECIMALS for the relative-
    # frequency scale, or to whole numbers for raw counts.
    vmin, vmax = mesh.get_clim()
    cbar.set_ticks(np.linspace(vmin, vmax, TILE_CBAR_TICKS))
    decimals = TILE_CBAR_DECIMALS if normalize else 0
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
    )
    if ax._tile_count > 1:
        print(TILE_OVERLAY_WARNING)
    return mesh


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    # Two correlated discrete variables: y tracks x plus noise, so the
    # bright cells fall along a diagonal band.
    x = rng.poisson(3, 4000)
    y = np.clip(x + rng.integers(-2, 3, 4000), 0, None)

    # Figure 1: the relative-frequency prototype.
    ax = plt.gca()
    make_tile(x, y, ax)

    # Figure 2: the same data as raw counts.
    plt.figure()
    ax = plt.gca()
    make_tile(x, y, ax, normalize=False)

    # Figure 3: mixed data -- discrete y (Poisson), continuous x (Normal).
    # The x-axis is binned automatically; the y-axis keeps one labeled row
    # per level.
    plt.figure()
    ax = plt.gca()
    cont_x = rng.normal(0, 1, 4000) + 0.4 * y  # correlate with the levels
    make_tile(cont_x, y, ax)

    # Figure 4: the same mixed data with the axes swapped -- continuous y,
    # discrete x -- to show binning follows whichever axis is continuous.
    plt.figure()
    ax = plt.gca()
    make_tile(y, cont_x, ax)

    plt.show()
