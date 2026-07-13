"""Draft of the 2D density / contour plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_density2D()`` below for 2D continuous-x-continuous data, replacing
the current ``make_density2D`` in ``symbulate/plot.py`` (``imshow``-based,
raw min/max grid, hardcoded ``cmap="Blues"``, no colorbar helper).

Both modes render the *same* underlying KDE surface as ``levels``
discrete color bands (default ``DENSITY2D_LEVELS = 8``) -- only the white
outline overlay and the title differ. ``contour=True`` does not switch
to a different statistic (e.g. raw bin counts): the colorbar always reads
"Density" since that's what's actually computed, matching the existing
``add_colorbar(fig, type, den, "Density")`` call for this plot type in
``results.py`` (line 1497). If the team wants a genuinely count-based
contour mode instead, that's a separate design decision -- flag it before
integration.

Visual target: a banded "2D Density Plot" (``contour=False``) and a
banded "Contour Plot" (``contour=True``, same bands plus thin white
outlines between them, topographic look). This replaces the earlier
draft's smooth ~100-level density mode: the team decided the density
plot should use a small number of discrete color values so bands can be
matched to the colorbar by eye, starting with a default of 8 (to be
revisited if it doesn't look good). Per DECISIONS.md's 2D density
prototype findings: ``contourf`` (not ``imshow``) is the correct rendering
approach, axis limits are quantile-based rather than raw min/max, the KDE
grid is 300x300 minimum, and low-density values are clipped so they don't
wash out the color scale. The per-plot-type values below are named
constants, which migrate to the top of ``symbulate/plot.py`` at
integration time.

Run this file directly to render the prototype:

    python new_graphics/density2d.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different level count for density vs. contour mode," so these
# cannot live in symbulate.mplstyle (see DECISIONS.md, "Decision:
# .mplstyle Standards"). They migrate to the top of symbulate/plot.py.
DENSITY2D_GRID_POINTS = 300  # DECISIONS.md: 300x300 minimum -- the old
# 100x100 grid in plot.py's make_density2D produces visible granularity
DENSITY2D_QUANTILE_LOW = 0.001
DENSITY2D_QUANTILE_HIGH = 0.999
DENSITY2D_PADDING_FRAC = 0.1  # same rationale as the 1D density case in
# density.py: quantile bounds, not raw min/max, so outlier-heavy data
# doesn't stretch the axes
DENSITY2D_VMIN_FRAC = 0.02  # DECISIONS.md: clip below Z.max() * 0.02 so
# low-density regions don't wash out the color scale
DENSITY2D_LEVELS = 8  # default number of discrete color bands, shared by
# both modes (levels= overrides per call). 8 is the team's first guess --
# revisit after seeing the demos if it doesn't look good.
DENSITY2D_CONTOUR_LINE_COLOR = "white"
DENSITY2D_CONTOUR_LINEWIDTH = 0.3  # DECISIONS.md: thin white lines
# between bands improve readability
DENSITY2D_CONTOUR_LINE_ALPHA = 0.4
DENSITY2D_CBAR_FALLBACK_DECIMALS = 3  # colorbar tick labels are rounded
# dynamically (just enough decimals to tell adjacent band edges apart);
# this is the fallback when that can't be computed


def _density2d_grid(x, y):
    """Quantile-bounded evaluation grid and KDE surface for 2D density.

    Parameters
    ----------
    x, y : array-like
        The two simulated variables.

    Returns
    -------
    tuple
        ``(Xgrid, Ygrid, Z, extent)`` -- the meshgrid, the evaluated
        density surface reshaped to match it, and ``(xmin, xmax, ymin,
        ymax)`` for the axis limits.
    """
    x, y = np.asarray(x), np.asarray(y)
    xlow, xhigh = np.quantile(x, [DENSITY2D_QUANTILE_LOW, DENSITY2D_QUANTILE_HIGH])
    ylow, yhigh = np.quantile(y, [DENSITY2D_QUANTILE_LOW, DENSITY2D_QUANTILE_HIGH])
    xspan, yspan = xhigh - xlow, yhigh - ylow
    xpad = DENSITY2D_PADDING_FRAC * xspan if xspan > 0 else 1.0
    ypad = DENSITY2D_PADDING_FRAC * yspan if yspan > 0 else 1.0
    xmin, xmax = xlow - xpad, xhigh + xpad
    ymin, ymax = ylow - ypad, yhigh + ypad

    kde = gaussian_kde(np.vstack([x, y]))
    Xgrid, Ygrid = np.meshgrid(
        np.linspace(xmin, xmax, DENSITY2D_GRID_POINTS),
        np.linspace(ymin, ymax, DENSITY2D_GRID_POINTS),
    )
    Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
    return Xgrid, Ygrid, Z, (xmin, xmax, ymin, ymax)


def make_density2D(x, y, ax, contour=False, levels=None, **kwargs):
    """Draw a 2D density surface as discrete color bands.

    Both modes plot the *same* KDE-estimated density surface with
    ``ax.contourf``, quantized into ``levels`` discrete color values so
    each band can be matched to the colorbar by eye. The only difference
    between the modes is the rendering:

    - ``contour=False`` (default): the color bands alone, producing the
      "2D Density Plot" look.
    - ``contour=True``: the same bands plus thin white outlines between
      them, producing the topographic "Contour Plot" look.

    The axis limits are quantile-based (0.1st to 99.9th percentile of
    each variable, plus padding), not raw min/max, so outlier-heavy data
    doesn't stretch the plot -- same rationale as the 1D density curve.
    The KDE is evaluated on a 300x300 grid, and values below 2% of the
    peak density are clipped to the bottom of the color scale so
    low-density regions don't wash it out. Colorbar tick labels are
    rounded automatically to just enough decimal places to tell adjacent
    band edges apart (falling back to
    ``DENSITY2D_CBAR_FALLBACK_DECIMALS`` if that can't be computed).

    A second ``make_density2D`` call on the same axes cannot overlay
    naturally -- a filled 2D surface completely obscures whatever was
    drawn before it, and the two color scales compete for the same
    colorbar space. Rather than silently producing a misleading plot,
    this prints a readability warning (matching the existing warning
    category for two 2D tile or two 2D histogram plots) and still draws,
    hiding the first plot underneath.

    Parameters
    ----------
    x, y : array-like
        The two simulated variables.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    contour : bool, default False
        If True, add thin white outlines between the color bands
        (the "Contour Plot" look).
    levels : int, optional
        Number of discrete color values to split the density surface
        into. Defaults to ``DENSITY2D_LEVELS`` (8). Must be a whole
        number of at least 2.
    **kwargs
        Additional keyword arguments passed to ``ax.contourf``.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The object returned by ``ax.contourf``, so the caller can
        inspect or further style the surface.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.normal(0, 1, 2000)
    >>> y = 0.7 * x + rng.normal(0, 1, 2000)
    >>> ax = plt.gca()
    >>> make_density2D(x, y, ax)  # doctest: +SKIP
    """
    if levels is None:
        levels = DENSITY2D_LEVELS
    # bool is an int subclass, so check it explicitly -- levels=True would
    # otherwise slip through as levels=1.
    if (
        isinstance(levels, bool)
        or not isinstance(levels, (int, np.integer))
        or levels < 2
    ):
        raise ValueError(
            "levels must be a whole number of at least 2 -- it sets how "
            f"many discrete colors the density surface is split into. You "
            f"passed levels={levels!r}. Try levels=8 (the default)."
        )

    n_prior = getattr(ax, "_density2d_count", 0)
    if n_prior > 0:
        print(
            "Showing a second 2D density plot on the same axes. The two "
            "color scales compete and this plot now covers up the first "
            "one. Try two separate plots (e.g. subplots) instead."
        )
    ax._density2d_count = n_prior + 1

    Xgrid, Ygrid, Z, (xmin, xmax, ymin, ymax) = _density2d_grid(x, y)

    zmax = Z.max()
    vmin = DENSITY2D_VMIN_FRAC * zmax
    # contourf treats the level values as band *boundaries* (N boundaries
    # -> N - 1 colors), so build levels + 1 edges to get exactly `levels`
    # discrete colors.
    level_edges = np.linspace(vmin, zmax, levels + 1)
    # Clip the surface itself to vmin instead of passing contourf's
    # extend="min": extend draws a triangular arrow cap on the colorbar to
    # flag the clipped range, which reads as an odd, non-rectangular
    # colorbar shape. Clipping the data directly still keeps low-density
    # regions from washing out the color scale, but the colorbar stays a
    # plain rectangle.
    Z_clipped = np.clip(Z, vmin, zmax)

    filled = ax.contourf(
        Xgrid, Ygrid, Z_clipped, levels=level_edges, cmap="viridis", **kwargs
    )
    if contour:
        ax.contour(
            Xgrid,
            Ygrid,
            Z_clipped,
            levels=level_edges,
            colors=DENSITY2D_CONTOUR_LINE_COLOR,
            linewidths=DENSITY2D_CONTOUR_LINEWIDTH,
            alpha=DENSITY2D_CONTOUR_LINE_ALPHA,
        )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Contour Plot" if contour else "2D Density Plot")
    # symbulate.mplstyle's global grid is horizontal-only (axes.grid.axis:
    # y); the approved prototypes call for both horizontal and vertical
    # reference lines, so this overrides it for this plot type -- same
    # per-type override pattern as density.py and rug.py. Note: since the
    # filled surface covers the full axes (axisbelow=True keeps the grid
    # behind the data), the grid won't actually show through the fill --
    # only at the tick marks along the spines.
    ax.grid(True, axis="both")

    # Colorbar via make_axes_locatable, per the approved layout helpers --
    # replaces the hardcoded fig.add_axes([0, 0.1, 0.05, 0.8]) used by
    # add_colorbar() in plot.py (a known bug: see symbulate_graphics_plan.md
    # Phase 3, "Colorbar layout"). Only the first call on a given axes adds
    # one: a second call already prints the readability warning above and
    # covers the first surface, so a second colorbar would just overlap the
    # first at nearly the same position, garbling both sets of tick labels.
    if n_prior == 0:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(filled, cax=cax)
        cbar.set_label("Density")
        # Round the tick labels dynamically: just enough decimal places to
        # tell adjacent band edges apart (resolution = half the band
        # spacing), instead of matplotlib's raw unrounded level values.
        # A formatter (rather than set_ticklabels) keeps matplotlib's
        # automatic tick thinning working at higher `levels` counts.
        spacing = level_edges[1] - level_edges[0]
        if np.isfinite(spacing) and spacing > 0:
            decimals = int(max(0, np.ceil(-np.log10(spacing / 2))))
        else:
            decimals = DENSITY2D_CBAR_FALLBACK_DECIMALS
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
        )

    return filled


if __name__ == "__main__":
    # Reproduce the two approved prototype images with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[1] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    x = rng.normal(0, 1.5, 3000)
    y = 0.7 * x + rng.normal(0, 1.2, 3000)

    # Figure 1: the density prototype (contour=False, default 8 bands).
    ax = plt.gca()
    make_density2D(x, y, ax)

    # Figure 2: the contour prototype (contour=True, default 8 bands).
    plt.figure()
    ax = plt.gca()
    make_density2D(x, y, ax, contour=True)

    # Figure 3: a custom band count, for comparing candidate defaults.
    plt.figure()
    ax = plt.gca()
    make_density2D(x, y, ax, levels=16)

    plt.show()
