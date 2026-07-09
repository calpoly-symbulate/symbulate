"""Draft of the 1D density plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_density()`` below when the user asks for a density curve or the
default plot lookup table selects one (1D continuous-ish data). This
replaces the density branch currently inline in ``results.py`` (around
line 1351), which evaluates the KDE over ``_plot_array.min()`` to
``_plot_array.max()`` -- raw min/max, not quantile-based, so a single
outlier (e.g. an Exponential's right tail) stretches the axis and
squashes the real curve into a sliver.

Visual target: a single smooth density curve, Okabe-Ito colored, no
fill, following symbulate.mplstyle's grid/spine/font conventions. The
per-plot-type values (alpha, line width, KDE grid resolution, x-axis
padding) are the named constants below, which migrate to the top of
``symbulate/plot.py`` at integration time.

Run this file directly to render the prototype:

    python new_graphics/density.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
DENSITY_LINEWIDTH = 1.8  # DECISIONS.md Visual Style Guide: 2.0 -> 1.8
# Full opacity for a standalone curve. DECISIONS.md's alpha=0.15 for
# density curves is specifically for the density-overlaid-on-a-
# histogram composition, which doesn't exist yet -- out of scope here.
DENSITY_ALPHA = 1.0
DENSITY_GRID_POINTS = 1000  # matches the resolution already used in
# results.py's current (unbounded) density branch
DENSITY_LEGEND_LOC = "upper right"

# Reasonable x-axis limits: evaluate the KDE over a quantile-based
# range plus padding, not raw min/max, so outlier-heavy distributions
# (e.g. Exponential) don't stretch the axis and squash the curve into
# a sliver. Same rationale as the 2D density decision in DECISIONS.md
# ("quantile bounds ... not raw min/max, which is sensitive to
# outliers"), applied to 1D.
DENSITY_QUANTILE_LOW = 0.001
DENSITY_QUANTILE_HIGH = 0.999
DENSITY_PADDING_FRAC = 0.1  # extra padding, as a fraction of the
# quantile-bounded span, added to each side so the curve visibly
# tapers to (near) zero instead of being cut off mid-slope


def _density_xrange(values):
    """Quantile-based x-axis bounds, with padding, for one density curve.

    Parameters
    ----------
    values : array-like
        The simulated values the density curve will be drawn from.

    Returns
    -------
    tuple
        ``(xmin, xmax)`` for evaluating and displaying the KDE.
    """
    values = np.asarray(values)
    qlow, qhigh = np.quantile(values, [DENSITY_QUANTILE_LOW, DENSITY_QUANTILE_HIGH])
    span = qhigh - qlow
    padding = DENSITY_PADDING_FRAC * span if span > 0 else 1.0
    return qlow - padding, qhigh + padding


def make_density(
    values, ax, color, bandwidth=None, alpha=None, label=None, **kwargs
):
    """Draw a 1D kernel density curve of simulated values on the given axes.

    Draws in the style of the approved density prototype: a single
    smooth line with no fill. The x-axis is always labeled "Value";
    the y-axis label and title are "Density" / "Density Curve". The
    curve is evaluated and displayed over a quantile-based x-range
    (the 0.1th to 99.9th percentile of ``values``, plus padding)
    rather than the raw min/max, so outlier-heavy data doesn't
    stretch the axis and flatten the visible curve.

    Density curves overlay naturally: a second call on the same axes
    draws on top of the first, and a legend appears automatically in
    the top right once two or more curves share the axes. Each curve
    is named by ``label``, or "Variable 1", "Variable 2", ... in call
    order when no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_hist``, ``make_tile``) are called.

    Parameters
    ----------
    values : array-like
        The simulated values to estimate a density from, e.g.
        ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Line color, from ``get_next_color(ax)``.
    bandwidth : float or str, optional
        Passed through to ``scipy.stats.gaussian_kde`` as
        ``bw_method``. Defaults to scipy's own default (Scott's rule).
    alpha : float, optional
        Line transparency between 0 and 1. Defaults to the package
        standard for a standalone density curve (1.0, fully opaque).
    label : str, optional
        Name for this curve in the legend. Defaults to "Variable k",
        where k counts the density curves drawn on these axes so far.
    **kwargs
        Additional keyword arguments passed to ``ax.plot``.

    Returns
    -------
    list
        The list of ``Line2D`` objects from ``ax.plot``, so the caller
        can inspect or further style the curve.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from symbulate.plot import get_next_color
    >>> values = np.random.default_rng().normal(0, 1, 2000)
    >>> ax = plt.gca()
    >>> make_density(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = DENSITY_ALPHA

    kde = gaussian_kde(values, bw_method=bandwidth)
    xmin, xmax = _density_xrange(values)
    grid = np.linspace(xmin, xmax, DENSITY_GRID_POINTS)
    density = kde(grid)

    # Count the curves drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color and make_hist use) so
    # overlays from separate .plot() calls see it.
    n_prior_curves = getattr(ax, "_density_count", 0)
    if label is None:
        label = f"Variable {n_prior_curves + 1}"
    ax._density_count = n_prior_curves + 1

    line = ax.plot(
        grid,
        density,
        color=color,
        alpha=alpha,
        linewidth=DENSITY_LINEWIDTH,
        label=label,
        **kwargs,
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title("Density Curve")
    ax.set_ylim(bottom=0)
    # symbulate.mplstyle's global grid is horizontal-only (axes.grid.axis:
    # y), but the approved density prototype shows both horizontal and
    # vertical reference lines, so this overrides it for this plot type
    # specifically -- the same per-type override pattern rug.py already
    # uses for its axis='x'-only standalone grid.
    ax.grid(True, axis="both")
    # A legend only helps once there is more than one curve to tell
    # apart; a lone curve stays legend-free.
    if ax._density_count > 1:
        ax.legend(loc=DENSITY_LEGEND_LOC)
    return line


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[1] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    values = rng.normal(0, 1, 2000)

    # Figure 1: the single-curve prototype.
    ax = plt.gca()
    # The prototype uses Okabe-Ito sky blue. Note: this is the *second*
    # hue in the symbulate.mplstyle color cycle (the first is orange
    # E69F00) -- once integrated, the color will come from
    # get_next_color(ax) instead.
    make_density(values, ax, "#56B4E9")

    # Figure 2: the overlay prototype -- three density curves with the
    # automatic "Variable k" legend in the top right.
    plt.figure()
    ax = plt.gca()
    make_density(rng.normal(0, 1, 1500), ax, "#56B4E9")
    make_density(rng.normal(3, 1.5, 1500), ax, "#E69F00")
    make_density(rng.exponential(1, 1500), ax, "#009E73")

    plt.show()
