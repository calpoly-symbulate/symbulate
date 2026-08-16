"""Draft of the 2D density / contour plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_density2D()`` below for 2D continuous-x-continuous data, replacing
the current ``make_density2D`` in ``symbulate/plot.py`` (``imshow``-based,
raw min/max grid, hardcoded ``cmap="Blues"``, no colorbar helper).

The default (``contour=False``) is a *continuous* density plot: the KDE
surface is rendered as a smooth color gradient with no visible bands. The
``levels`` argument does not apply here and is ignored (with a warning) if
passed. Setting ``contour=True`` switches to a topographic "Contour Plot":
the same KDE surface split into ``levels`` discrete color bands (default
``DENSITY2D_LEVELS = 8``) with thin white outlines between them, so bands
can be matched to the colorbar by eye. ``levels`` only takes effect in
this mode. Both modes render the same underlying statistic -- the colorbar
always reads "Density" since that's what's actually computed, matching the
existing ``add_colorbar(fig, type, den, "Density")`` call for this plot
type in ``results.py`` (line 1497). If the team wants a genuinely
count-based contour mode instead, that's a separate design decision --
flag it before integration.

Visual target: a smooth continuous "2D Density Plot" (``contour=False``,
the default) and a banded "Contour Plot" (``contour=True``, discrete bands
plus thin white outlines between them, topographic look). Per DECISIONS.md's
2D density prototype findings: ``contourf`` (not ``imshow``) is the correct
rendering approach for both modes -- the continuous look is a large fixed
level count (``DENSITY2D_CONTINUOUS_LEVELS``) rather than a switch to
``imshow`` -- axis limits are quantile-based rather than raw min/max, and the
KDE grid is 300x300 minimum. The color scale runs from 0 to the peak density
so the colorbar starts at 0 (DECISIONS.md Phase 4). The per-plot-type values
below are named constants, which migrate to the top of ``symbulate/plot.py``
at integration time.

Run this file directly to render the prototype:

    python new_graphics/density2d.py
"""

import warnings

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
DENSITY2D_CONTINUOUS_LEVELS = 256  # number of contourf bands used for the
# default continuous density plot (contour=False). High enough that the
# bands blend into a smooth gradient, keeping the contourf rendering
# approach DECISIONS.md calls for instead of switching to imshow.
DENSITY2D_LEVELS = 8  # default number of discrete color bands for the
# contour plot (contour=True); levels= overrides per call. 8 is the team's
# first guess -- revisit after seeing the demos if it doesn't look good.
DENSITY2D_CONTOUR_LINE_COLOR = "white"
DENSITY2D_CONTOUR_LINEWIDTH = 0.3  # DECISIONS.md: thin white lines
# between bands improve readability
DENSITY2D_CONTOUR_LINE_ALPHA = 0.4
DENSITY2D_CBAR_DECIMALS = 3  # colorbar tick labels are rounded to this many
# decimal places
DENSITY2D_CBAR_TICKS = 8  # number of evenly spaced colorbar ticks (including
# both endpoints, 0 and the peak density) for the continuous density plot;
# the contour plot instead ticks its discrete band edges


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
    """Draw a 2D density surface from a KDE estimate.

    Both modes plot the *same* KDE-estimated density surface with
    ``ax.contourf``; they differ in how finely it is quantized:

    - ``contour=False`` (default): a *continuous* density plot. The
      surface is drawn with a large fixed number of color bands
      (``DENSITY2D_CONTINUOUS_LEVELS``) so they blend into a smooth
      gradient with no visible banding -- the "2D Density Plot" look. The
      ``levels`` argument does not apply here; passing it warns and has no
      effect.
    - ``contour=True``: a topographic "Contour Plot". The same surface is
      split into ``levels`` discrete color bands with thin white outlines
      between them, so each band can be matched to the colorbar by eye.

    The axis limits are quantile-based (0.1st to 99.9th percentile of
    each variable, plus padding), not raw min/max, so outlier-heavy data
    doesn't stretch the plot -- same rationale as the 1D density curve.
    The KDE is evaluated on a 300x300 grid. The color scale runs from 0 to
    the peak density, so the colorbar starts at 0 and ticks both endpoints
    (0 and the peak); its labels are rounded to ``DENSITY2D_CBAR_DECIMALS``
    (3) decimal places.

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
        If False (default), draw a continuous, smoothly shaded density
        surface. If True, draw a topographic contour plot: discrete
        color bands with thin white outlines between them.
    levels : int, optional
        Number of discrete color bands, used only when ``contour=True``.
        Defaults to ``DENSITY2D_LEVELS`` (8) in that mode. Must be a whole
        number of at least 2. Ignored (with a warning) when
        ``contour=False``, since the continuous density plot has no bands.
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
    if contour:
        if levels is None:
            levels = DENSITY2D_LEVELS
        # bool is an int subclass, so check it explicitly -- levels=True
        # would otherwise slip through as levels=1.
        if (
            isinstance(levels, bool)
            or not isinstance(levels, (int, np.integer))
            or levels < 2
        ):
            raise ValueError(
                "levels must be a whole number of at least 2 -- it sets how "
                f"many discrete color bands the contour plot is split into. "
                f"You passed levels={levels!r}. Try levels=8 (the default)."
            )
    else:
        # Continuous density plot: there are no discrete bands to control,
        # so levels has no meaning here. Warn if the user passed one rather
        # than silently ignoring it, then fall back to the large fixed count
        # that makes the surface look continuous.
        if levels is not None:
            warnings.warn(
                "levels only applies to the contour plot (contour=True), "
                "which splits the density into discrete color bands. The "
                "default 2D density plot is a continuous color surface with "
                "no bands, so levels was ignored. Pass contour=True to use "
                "it.",
                UserWarning,
                stacklevel=2,
            )
        levels = DENSITY2D_CONTINUOUS_LEVELS

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
    # The color scale runs from 0 to the peak density so the colorbar starts
    # at 0 (DECISIONS.md Phase 4: "tile and density plots: ensure color scale
    # always starts from 0"). KDE density is non-negative, so no low-end
    # clipping is needed. contourf treats the level values as band
    # *boundaries* (N boundaries -> N - 1 colors), so build levels + 1 edges
    # to get exactly `levels` discrete colors.
    level_edges = np.linspace(0, zmax, levels + 1)

    filled = ax.contourf(Xgrid, Ygrid, Z, levels=level_edges, cmap="viridis", **kwargs)
    if contour:
        ax.contour(
            Xgrid,
            Ygrid,
            Z,
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
        # Tick both ends of the bar (0 and the peak density), which
        # matplotlib's default locator otherwise trims. In contour mode the
        # discrete band edges are the natural ticks -- and since they span
        # 0 to zmax, the endpoints come for free. In continuous mode there
        # are hundreds of bands, so use a small set of evenly spaced ticks
        # between the same endpoints instead.
        if contour:
            ticks = level_edges
        else:
            ticks = np.linspace(0, zmax, DENSITY2D_CBAR_TICKS)
        cbar.set_ticks(ticks)
        # Round every label to a fixed number of decimals. A formatter
        # (rather than set_ticklabels) keeps matplotlib's automatic tick
        # thinning working when there are many band edges.
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.{DENSITY2D_CBAR_DECIMALS}f}")
        )

    return filled


if __name__ == "__main__":
    # Reproduce the two approved prototype images with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    x = rng.normal(0, 1.5, 3000)
    y = 0.7 * x + rng.normal(0, 1.2, 3000)

    # Figure 1: the continuous density prototype (contour=False, default).
    ax = plt.gca()
    make_density2D(x, y, ax)

    # Figure 2: the contour prototype (contour=True, default 8 bands).
    plt.figure()
    ax = plt.gca()
    make_density2D(x, y, ax, contour=True)

    # Figure 3: a custom band count for the contour plot, for comparing
    # candidate defaults. levels only applies when contour=True.
    plt.figure()
    ax = plt.gca()
    make_density2D(x, y, ax, contour=True, levels=16)

    plt.show()
