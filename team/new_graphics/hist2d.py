"""Draft of the 2D histogram plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_hist2d()`` below when the user asks for a 2D histogram
(``type="hist"`` on 2D data).

Visual target: the approved 2D histogram prototype (Task 1E) -- a
filled bin mesh in the viridis sequential colormap (set globally in
``symbulate/symbulate.mplstyle``, replacing the hardcoded
``cmap="Blues"``), with a colorbar on the right labeled "Density" or
"Count", placed with ``mpl_toolkits.axes_grid1.make_axes_locatable``
(the approved layout helper that replaces the old hardcoded
``fig.add_axes([0, 0.1, 0.05, 0.8])``). Axis labels read "X" and "Y",
and the title reads "2-D Histogram".

Overlay category (see CLAUDE.md, "Overlay Policy"): **readability
warning** -- a second 2D histogram on the same axes is drawn, but the
two color scales compete, so a student-friendly warning is printed
below the plot.

Run this file directly to render the prototype:

    python team/new_graphics/hist2d.py
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
HIST2D_DEFAULT_BINS = 30
HIST2D_CBAR_SIZE = "5%"  # colorbar width, as a fraction of the axes
HIST2D_CBAR_PAD = 0.1  # gap between the axes and the colorbar, inches

HIST2D_OVERLAY_WARNING = (
    "Warning: you drew a second 2-D histogram on the same plot. The two "
    "color scales compete, so the result may be hard to read. Consider "
    "plotting them in separate cells, or using type='scatter' instead."
)


def make_hist2d(x, y, ax, bins=None, normalize=True, hex=False, **kwargs):
    """Draw a 2D histogram of simulated (x, y) pairs on the given axes.

    Draws in the style of the approved 2D histogram prototype: a
    filled bin mesh using the package's sequential colormap (viridis,
    from ``symbulate.mplstyle``), with a colorbar on the right labeled
    "Density" (or "Count" when ``normalize=False``). The x-axis is
    labeled "X", the y-axis "Y", and the title reads "2-D Histogram".
    The color scale always starts from 0 so empty bins read as
    "no data" rather than an arbitrary color.

    With ``hex=True`` the bins are hexagons instead of squares and the
    title reads "Hexbin Plot"; everything else (colormap, colorbar,
    normalization, overlay warning) behaves the same.

    Unlike the 1D plot types, a 2D histogram encodes magnitude with a
    colormap instead of the categorical color cycle, so no ``color``
    parameter is taken. Overlays are the "readability warning"
    category of the overlay policy: a second call on the same axes
    still draws, but prints a warning that the color scales compete.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_tile``, ``make_density2D``) are
    called.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first
        column of ``RVResults.array``.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    bins : int, optional
        Number of equal-width bins along each axis (the number of
        hexagons across the x-axis when ``hex=True``). Defaults to 30.
    normalize : bool, default True
        If True, bin colors show density (cell areas integrate to 1)
        so the plot approximates the joint density. If False, colors
        show raw counts.
    hex : bool, default False
        If True, bin the data into hexagons instead of squares and
        title the plot "Hexbin Plot". (The name shadows the built-in
        ``hex()``, matching how the package's ``type=`` parameter
        shadows ``type()``.)
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    tuple or matplotlib.collections.PolyCollection
        With square bins, the ``(counts, xedges, yedges, mesh)`` tuple
        from ``ax.hist2d``; with ``hex=True``, the hexagon collection
        from ``ax.hexbin``. Either way the caller can inspect the bins
        or attach further styling.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x, y = rng.normal(0, 1, (2, 1000))
    >>> ax = plt.gca()
    >>> make_hist2d(x, y, ax)  # doctest: +SKIP
    """
    if bins is None:
        bins = HIST2D_DEFAULT_BINS
    xs, ys = np.asarray(x), np.asarray(y)
    # Count the 2D histograms drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_hist2d_count", 0)
    ax._hist2d_count = n_prior + 1
    # No cmap argument in either branch: the sequential colormap comes
    # from image.cmap in symbulate.mplstyle. vmin=0 anchors the color
    # scale at zero.
    if hex:
        histogram = ax.hexbin(xs, ys, gridsize=bins, vmin=0, **kwargs)
        # The hexagon tiling has a jagged outline that doesn't fill
        # the square axes box. Painting the axes background with the
        # colormap's zero color makes everything beyond the tiling
        # read as zero density, so the plot edge is the clean square
        # of the axes, matching the 2-D histogram.
        ax.set_facecolor(histogram.get_cmap()(0))
        if normalize:
            # hexbin has no density option, so rescale the counts by
            # hand: density = count / (n * cell area), which makes the
            # hexagon volumes sum to 1 exactly like a density
            # histogram. All hexagons are congruent, so the shoelace
            # formula on one hexagon's vertices (in data units) gives
            # the cell area.
            v = histogram.get_paths()[0].vertices[:6]
            area = 0.5 * abs(
                np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
            )
            density = histogram.get_array() / (len(xs) * area)
            histogram.set_array(density)
            histogram.set_clim(0, float(density.max()))
        mesh = histogram
    else:
        histogram = ax.hist2d(xs, ys, bins=bins, density=normalize, vmin=0, **kwargs)
        mesh = histogram[3]
    # A filled mesh covers the whole axes, so the reference grid has
    # nothing to sit on -- turn it off rather than let fragments show
    # at the edges.
    ax.grid(False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Hexbin Plot" if hex else "2-D Histogram")
    # Colorbar on the right, sized relative to the axes so it tracks
    # figure resizing (the approved replacement for the old hardcoded
    # fig.add_axes colorbar).
    cax = make_axes_locatable(ax).append_axes(
        "right", size=HIST2D_CBAR_SIZE, pad=HIST2D_CBAR_PAD
    )
    cbar = ax.get_figure().colorbar(mesh, cax=cax)
    cbar.set_label("Density" if normalize else "Count")
    if ax._hist2d_count > 1:
        print(HIST2D_OVERLAY_WARNING)
    return histogram


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    # Correlated bivariate normal, matching the prototype's tilted
    # elliptical shape.
    xy = rng.multivariate_normal([0, 0], [[2.0, 1.2], [1.2, 2.0]], size=3000)

    # Figure 1: the square-bin prototype.
    ax = plt.gca()
    make_hist2d(xy[:, 0], xy[:, 1], ax)

    # Figure 2: the same data with hexagon bins.
    plt.figure()
    ax = plt.gca()
    make_hist2d(xy[:, 0], xy[:, 1], ax, hex=True)

    plt.show()
