"""Draft of the 2D density / contour plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_density2D()`` below for 2D continuous-x-continuous data, replacing
the current ``make_density2D`` in ``symbulate/plot.py`` (``imshow``-based,
raw min/max grid, hardcoded ``cmap="Blues"``, no colorbar helper).

Both the smooth density look and the banded contour look come from the
*same* underlying KDE surface -- only the rendering (fill level count,
white outline overlay) and title differ. ``contour=True`` does not switch
to a different statistic (e.g. raw bin counts): the colorbar always reads
"Density" since that's what's actually computed, matching the existing
``add_colorbar(fig, type, den, "Density")`` call for this plot type in
``results.py`` (line 1497). If the team wants a genuinely count-based
contour mode instead, that's a separate design decision -- flag it before
integration.

Visual target: the two approved reference images -- a smooth "2D Density
Plot" (``contour=False``, ~100 fill levels so band edges disappear) and a
banded "Contour Plot" (``contour=True``, ~12 fill levels with thin white
outlines between bands, topographic look). Per DECISIONS.md's 2D density
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
DENSITY2D_SMOOTH_LEVELS = 100  # many fill levels -> band edges vanish,
# reproduces the smooth "2D Density Plot" reference image
DENSITY2D_CONTOUR_LEVELS = 12  # few fill levels -> visible topographic
# bands, reproduces the "Contour Plot" reference image
DENSITY2D_CONTOUR_LINE_COLOR = "white"
DENSITY2D_CONTOUR_LINEWIDTH = 0.3  # DECISIONS.md: thin white lines
# between bands improve readability
DENSITY2D_CONTOUR_LINE_ALPHA = 0.4


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


def make_density2D(x, y, ax, contour=False, **kwargs):
    """Draw a 2D density surface, as smooth shading or banded contours.

    Both modes plot the *same* KDE-estimated density surface with
    ``ax.contourf`` -- only the rendering changes:

    - ``contour=False`` (default): many fill levels so band edges are
      imperceptible, producing the smooth "2D Density Plot" look.
    - ``contour=True``: few fill levels with thin white outlines between
      them, producing the banded, topographic "Contour Plot" look.

    The axis limits are quantile-based (0.1st to 99.9th percentile of
    each variable, plus padding), not raw min/max, so outlier-heavy data
    doesn't stretch the plot -- same rationale as the 1D density curve.
    The KDE is evaluated on a 300x300 grid, and values below 2% of the
    peak density are clipped to the bottom of the color scale so
    low-density regions don't wash it out.

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
        If True, draw the banded contour-line style instead of the
        smooth density style.
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
    n_levels = DENSITY2D_CONTOUR_LEVELS if contour else DENSITY2D_SMOOTH_LEVELS
    levels = np.linspace(vmin, zmax, n_levels)
    # Clip the surface itself to vmin instead of passing contourf's
    # extend="min": extend draws a triangular arrow cap on the colorbar to
    # flag the clipped range, which reads as an odd, non-rectangular
    # colorbar shape. Clipping the data directly still keeps low-density
    # regions from washing out the color scale, but the colorbar stays a
    # plain rectangle.
    Z_clipped = np.clip(Z, vmin, zmax)

    filled = ax.contourf(Xgrid, Ygrid, Z_clipped, levels=levels, cmap="viridis", **kwargs)
    if contour:
        ax.contour(
            Xgrid,
            Ygrid,
            Z_clipped,
            levels=levels,
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

    # Figure 1: the smooth density prototype (contour=False, default).
    ax = plt.gca()
    make_density2D(x, y, ax)

    # Figure 2: the banded contour prototype (contour=True).
    plt.figure()
    ax = plt.gca()
    make_density2D(x, y, ax, contour=True)

    plt.show()
