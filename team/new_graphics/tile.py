"""Draft of the 2D tile plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_tile()`` below for 2D discrete-x-discrete data (``type="tile"``),
replacing the current ``make_tile`` in ``symbulate/plot.py`` (``matshow``-
based, hardcoded ``cmap="Blues"``, no colorbar, relative-frequency only).

Visual target: a grid of filled cells -- one cell per distinct (x, y)
pair -- colored by how often that pair occurred, in the package's
sequential colormap (viridis, from ``symbulate.mplstyle``), with a
colorbar on the right labeled "Relative Frequency" or "Count". Because
the plot is categorical (one cell per value, equal-sized regardless of
the gaps between values), each axis is labeled with the distinct values
themselves.

This draft is deliberately discrete-only. The old ``setup_tile`` also
had a continuous branch that binned a numeric axis into equal-width
bins -- but that is exactly what ``make_hist2d`` now does, so carrying
it here would duplicate that plot type. ``classify_data`` routes
continuous 2D data to ``hist2d`` / ``density2d`` instead, leaving the
tile plot to do the one thing a 2D histogram cannot: keep every
discrete value in its own labeled cell.

Overlay category (see CLAUDE.md, "Overlay Policy"): **readability
warning** -- a second tile plot on the same axes is drawn, but the two
color scales compete, so a student-friendly warning is printed below
the plot (the same category as two 2D histograms).

Run this file directly to render the prototype:

    python team/new_graphics/tile.py
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express per-plot-type values, so these cannot live in
# symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py. The
# colormap itself is NOT set here -- it comes from image.cmap
# (viridis) in symbulate.mplstyle.
TILE_CBAR_SIZE = "5%"  # colorbar width, as a fraction of the axes
TILE_CBAR_PAD = 0.1  # gap between the axes and the colorbar, inches

TILE_OVERLAY_WARNING = (
    "Warning: you drew a second tile plot on the same plot. The two "
    "color scales compete, so the result may be hard to read. Consider "
    "plotting them in separate cells, or using type='scatter' instead."
)


def make_tile(x, y, ax, normalize=True, **kwargs):
    """Draw a 2D tile plot of simulated (x, y) pairs on the given axes.

    Draws a grid of filled cells -- one cell per distinct (x, y) pair --
    colored by how often that pair occurred, using the package's
    sequential colormap (viridis, from ``symbulate.mplstyle``), with a
    colorbar on the right labeled "Relative Frequency" (or "Count" when
    ``normalize=False``). The x-axis is labeled "X", the y-axis "Y",
    and the title reads "Tile Plot". The color scale always starts from
    0 so a pair that never occurred reads as "no data" rather than an
    arbitrary color.

    The tile plot is for two discrete variables: each axis gets one
    labeled cell per distinct value, equal-sized regardless of the gaps
    between values, so a student can read the joint distribution as a
    table of shaded counts. Continuous data is routed to ``hist2d`` /
    ``density2d`` instead.

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
        column of ``RVResults.array``. Meant to be discrete.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
        Meant to be discrete.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    normalize : bool, default True
        If True, cell colors show the relative frequency of each pair
        (all cells sum to 1). If False, colors show raw counts.
    **kwargs
        Additional keyword arguments passed to ``ax.imshow``.

    Returns
    -------
    matplotlib.image.AxesImage
        The image from ``ax.imshow``, so the caller can inspect the
        cells or attach further styling.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.poisson(3, 1000)
    >>> y = rng.poisson(2, 1000)
    >>> ax = plt.gca()
    >>> make_tile(x, y, ax)  # doctest: +SKIP
    """
    xs, ys = np.asarray(x), np.asarray(y)
    # Count the tile plots drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_tile_count", 0)
    ax._tile_count = n_prior + 1

    # One cell per distinct value on each axis. searchsorted maps every
    # value to its cell index; np.unique returns the labels sorted, so
    # the indices line up with the cell positions 0, 1, 2, ...
    x_lab = np.unique(xs)
    y_lab = np.unique(ys)
    intensity = np.zeros((len(y_lab), len(x_lab)))
    x_idx = np.searchsorted(x_lab, xs)
    y_idx = np.searchsorted(y_lab, ys)
    np.add.at(intensity, (y_idx, x_idx), 1)
    if normalize:
        intensity /= len(xs)

    # No cmap argument: the sequential colormap comes from image.cmap
    # (viridis) in symbulate.mplstyle. vmin=0 anchors the color scale
    # at zero. origin="lower" puts the smallest values at bottom-left;
    # aspect="auto" lets the cells fill the axes box.
    mesh = ax.imshow(
        intensity, origin="lower", aspect="auto", vmin=0, **kwargs
    )
    # A filled mesh covers the whole axes, so the reference grid has
    # nothing to sit on -- turn it off rather than let fragments show
    # at the edges.
    ax.grid(False)
    # Categorical axes: one tick per cell, labeled with the value.
    ax.set_xticks(range(len(x_lab)))
    ax.set_yticks(range(len(y_lab)))
    ax.set_xticklabels(x_lab)
    ax.set_yticklabels(y_lab)
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

    plt.show()
