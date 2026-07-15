"""Draft of the segmented density ("ridgeline") plot type for the
graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_segmented_density()`` below when the user asks for a segmented
density plot (``type='segmented_density'``). In the default plot lookup
table (DECISIONS.md, "Decision: Default Plot Lookup Table") the
segmented density plot is an *alternative*
plot -- never the default -- for the 2D discrete x continuous and
continuous x discrete configurations, at both small and large n, so it
will appear in the suggestion message for those configurations. The
graphics plan (Phase 5) also names density ridge plots as the view for
discrete-time continuous-state processes: one ridge per time point shows
how the process's distribution spreads or drifts over time.

Visual target: one small kernel density curve ("ridge") of the
continuous variable per level of the discrete variable, stacked along
the discrete axis with a gentle overlap, each filled with a translucent
Okabe-Ito color under an opaque outline. All ridges in one call share a
single density scale, so a taller peak really is a taller density --
levels are directly comparable. The per-plot-type values (fill alpha,
line width, peak height, KDE resolution, value-axis padding) are the
named constants below, which migrate to the top of ``symbulate/plot.py``
at integration time.

The orientation follows which variable is discrete, mirroring
``make_segmented_rug`` and the mixed tile plot so all views of the same
data line up: discrete ``y`` gives the classic look (horizontal ridges
stacked vertically), discrete ``x`` flips it (vertical ridges stacked
horizontally).

Run this file directly to render the prototype:

    python team/new_graphics/segmented_density.py
"""

import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
SEGMENTED_DENSITY_FILL_ALPHA = 0.4  # translucent fill; the outline stays opaque
SEGMENTED_DENSITY_LINEWIDTH = 1.8  # ridge outlines are density curves, so match
# DENSITY_LINEWIDTH (merge with it at integration if the team agrees)
SEGMENTED_DENSITY_PEAK_SCALE = 1.6  # tallest peak's height, in units of the
# spacing between neighboring baselines; > 1 gives the gentle overlap
# that makes the stacked ridges read as one connected picture
SEGMENTED_DENSITY_GRID_POINTS = 1000  # matches DENSITY_GRID_POINTS
SEGMENTED_DENSITY_LEGEND_LOC = "upper right"
SEGMENTED_DENSITY_TICK_FRAC = 0.2  # height of the fallback tick marks drawn
# for levels too sparse for a density curve, as a fraction of the
# baseline spacing
SEGMENTED_DENSITY_TICK_LINEWIDTH = 1.0  # matches RUG_LINEWIDTH -- the fallback
# ticks are just a tiny rug on that level's baseline

# Reasonable value-axis limits: evaluate every KDE over a shared
# quantile-based range plus padding, not raw min/max, so outlier-heavy
# distributions don't stretch the axis and squash the ridges. Mirrors
# DENSITY_QUANTILE_LOW / HIGH / PADDING_FRAC in symbulate/plot.py --
# share one set of constants at integration time.
SEGMENTED_DENSITY_QUANTILE_LOW = 0.001
SEGMENTED_DENSITY_QUANTILE_HIGH = 0.999
SEGMENTED_DENSITY_PADDING_FRAC = 0.1


def _segmented_density_value_range(values):
    """Quantile-based value-axis bounds, with padding, shared by all ridges.

    Parameters
    ----------
    values : array-like
        The pooled continuous values across every discrete level, so
        all ridges in one call are evaluated over one common grid.

    Returns
    -------
    tuple
        ``(vmin, vmax)`` for evaluating and displaying the KDEs.
    """
    values = np.asarray(values)
    qlow, qhigh = np.quantile(
        values, [SEGMENTED_DENSITY_QUANTILE_LOW, SEGMENTED_DENSITY_QUANTILE_HIGH]
    )
    span = qhigh - qlow
    padding = SEGMENTED_DENSITY_PADDING_FRAC * span if span > 0 else 1.0
    return qlow - padding, qhigh + padding


def make_segmented_density(
    x,
    y,
    ax,
    color=None,
    bandwidth=None,
    alpha=None,
    label=None,
    discrete_x=None,
    discrete_y=None,
    **kwargs,
):
    """Draw a segmented density plot for mixed discrete/continuous data.

    One small kernel density curve ("ridge") of the continuous variable
    per level of the discrete variable, stacked along the discrete axis.
    Where the segmented rug shows every individual observation and the
    mixed tile plot shows binned counts, the segmented density plot --
    commonly called a ridgeline plot -- shows each
    level's estimated *shape* -- how the continuous variable's
    distribution shifts or spreads from level to level, read at a
    glance. It is also the natural view of a discrete-time
    continuous-state process: one ridge per time point.

    All ridges in one call share a single density scale (the tallest
    peak reaches ``SEGMENTED_DENSITY_PEAK_SCALE`` baseline spacings), so peak
    heights are directly comparable across levels. Each ridge is drawn
    over the same quantile-based value range (the 0.1th to 99.9th
    percentile of the pooled continuous values, plus padding) rather
    than the raw min/max, so outlier-heavy data doesn't stretch the
    axis and flatten the ridges.

    The orientation follows which variable is discrete, mirroring
    ``make_segmented_rug`` and the mixed tile plot so the different
    views of the same data line up:

    - discrete ``y``, continuous ``x``: the classic ridgeline look -- one
      horizontal ridge per y-level, stacked vertically.
    - discrete ``x``, continuous ``y``: flipped -- one vertical ridge
      per x-level, stacked horizontally, each extending to the right of
      its baseline.

    Levels with fewer than two distinct values cannot support a density
    estimate; their observations are drawn as short tick marks on that
    level's baseline instead (a tiny rug), and a note explains which
    levels fell back and why.

    A light reference grid sits behind the ridges on both axes: lines
    along the continuous axis for reading values off the density curves
    (as the 1D density plot draws), and one line per level along the
    discrete axis, extending every baseline across the full plot (as
    the segmented rug draws).

    Segmented density plots overlay naturally: a second call on the same
    axes
    draws a second set of ridges on the same baselines (levels are
    matched by value, and levels new to the axes get new baselines),
    and a legend appears automatically once two or more batches share
    the axes. When no ``color`` is given, each batch
    takes the next hue from the active style sheet's color cycle
    automatically -- sky blue first with symbulate.mplstyle, then
    orange, ... -- so successive calls come out distinct with no
    arguments at all. Each batch is named by ``label``, or
    "Variable 1", "Variable 2", ... in call order when no label is
    given. Each batch is scaled to its own tallest peak, so overlays
    compare shapes, not absolute density values.

    This plot is only for mixed data -- exactly one discrete variable
    and one continuous variable. Two discrete variables should use a
    tile plot, and two continuous variables a scatter plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). Once integrated, ``RVResults.plot()`` will
    advance the color cycle with ``get_next_color(ax)`` exactly once
    per call and pass the result in as ``color``, mirroring how the
    other plot helpers in ``symbulate/plot.py``
    (``make_segmented_rug``, ``make_density``) are called; standalone,
    leaving ``color`` unset reads the same cycle, so both paths give
    the same colors.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first column
        of ``RVResults.array``. Discrete or continuous.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
        Discrete or continuous.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color, optional
        Color for the ridges. Defaults to the next color in the active
        style sheet's cycle (sky blue for the first batch on the axes,
        orange for the second, ...), the same cycle
        ``get_next_color(ax)`` in ``symbulate/plot.py`` reads. The
        outline is drawn fully opaque and the fill translucent
        (``SEGMENTED_DENSITY_FILL_ALPHA``).
    bandwidth : float or str, optional
        Passed through to ``scipy.stats.gaussian_kde`` as ``bw_method``
        for every ridge. Defaults to scipy's own default (Scott's rule).
    alpha : float, optional
        Fill transparency between 0 and 1. Defaults to the package
        standard for segmented density fills
        (``SEGMENTED_DENSITY_FILL_ALPHA``, 0.4).
        The outline always stays opaque so the density shape reads
        clearly even where ridges overlap.
    label : str, optional
        Name for this batch of ridges in the legend. Defaults to
        "Variable k", where k counts the segmented density batches drawn on
        these axes so far.
    discrete_x : bool, optional
        Whether the x-axis is the discrete (grouping) variable. If None
        (default), detected from the data: float values are treated as
        continuous, everything else (int, bool, string) as discrete.
    discrete_y : bool, optional
        Same as ``discrete_x`` for the y-axis.
    **kwargs
        Additional keyword arguments passed to ``fill_between`` /
        ``fill_betweenx`` for the ridges.

    Returns
    -------
    list
        The drawn artists, one per level in baseline order -- a
        ``PolyCollection`` for each ridge, or a ``LineCollection`` for
        each sparse level's fallback ticks -- so the caller can inspect
        or further style them.

    Raises
    ------
    ValueError
        If the two variables are not one discrete and one continuous.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> y = rng.integers(0, 4, 200)            # discrete groups
    >>> x = rng.normal(0, 1, 200) + y          # continuous values
    >>> make_segmented_density(x, y, plt.gca())  # doctest: +SKIP
    """
    if alpha is None:
        alpha = SEGMENTED_DENSITY_FILL_ALPHA
    xs, ys = np.asarray(x), np.asarray(y)
    if discrete_x is None:
        discrete_x = not np.issubdtype(xs.dtype, np.floating)
    if discrete_y is None:
        discrete_y = not np.issubdtype(ys.dtype, np.floating)

    # A segmented density plot needs one discrete variable (the
    # baselines) and one
    # continuous variable (the densities). Anything else is a different
    # plot.
    if discrete_x == discrete_y:
        if discrete_x:
            raise ValueError(
                "A segmented density plot needs one discrete variable and one "
                "continuous variable, but both of yours look discrete. Try a "
                "tile plot or a scatter plot with jitter for two discrete "
                "variables."
            )
        raise ValueError(
            "A segmented density plot needs one discrete variable and one "
            "continuous variable, but both of yours look continuous. Try a "
            "scatter plot for two continuous variables."
        )

    # Resolve the automatic color only after validation, so a call that
    # errors out doesn't advance the cycle.
    if color is None:
        color = _next_segmented_density_color(ax)

    # The discrete variable defines the baselines; the continuous
    # variable is the value axis the densities are estimated along.
    if discrete_y:
        levels = np.unique(ys)
        continuous, groups = xs, ys
    else:
        levels = np.unique(xs)
        continuous, groups = ys, xs

    # Baseline positions live on the axes object so a second .plot()
    # call lands its ridges on the same baselines (matched by level
    # value); levels the axes hasn't seen yet get the next free
    # baseline. Same pattern get_next_color and make_density use for
    # their per-axes state.
    positions = getattr(ax, "_segmented_density_positions", None)
    if positions is None:
        positions = {}
        ax._segmented_density_positions = positions
    for level in levels:
        if level not in positions:
            positions[level] = len(positions)

    # Count the segmented density batches drawn on these axes, for the
    # automatic "Variable k" legend names.
    n_prior = getattr(ax, "_segmented_density_count", 0)
    if label is None:
        label = f"Variable {n_prior + 1}"
    ax._segmented_density_count = n_prior + 1

    # Estimate every level's density over one shared grid so the ridges
    # align, then scale them jointly: the tallest peak in this call
    # reaches SEGMENTED_DENSITY_PEAK_SCALE baseline spacings, and every other
    # peak keeps its true height relative to it.
    vmin, vmax = _segmented_density_value_range(continuous)
    grid = np.linspace(vmin, vmax, SEGMENTED_DENSITY_GRID_POINTS)
    densities = {}
    sparse = []
    for level in levels:
        values = continuous[groups == level]
        if np.unique(values).size < 2:
            sparse.append(level)
        else:
            densities[level] = gaussian_kde(values, bw_method=bandwidth)(grid)
    if densities:
        scale = SEGMENTED_DENSITY_PEAK_SCALE / max(d.max() for d in densities.values())

    # Fill translucent, outline opaque -- one artist per ridge, so the
    # shape stays readable where neighboring ridges overlap.
    facecolor = mcolors.to_rgba(color, alpha)

    # Draw from the highest baseline down so that where ridges overlap,
    # the lower (nearer) ridge sits in front -- the classic ridgeline
    # look.
    artists = {}
    for level in sorted(levels, key=positions.get, reverse=True):
        base = positions[level]
        batch_label = label if not artists else None
        if level in densities:
            heights = densities[level] * scale
            if discrete_y:
                artists[level] = ax.fill_between(
                    grid,
                    base,
                    base + heights,
                    facecolor=facecolor,
                    edgecolor=color,
                    linewidth=SEGMENTED_DENSITY_LINEWIDTH,
                    label=batch_label,
                    **kwargs,
                )
            else:
                artists[level] = ax.fill_betweenx(
                    grid,
                    base,
                    base + heights,
                    facecolor=facecolor,
                    edgecolor=color,
                    linewidth=SEGMENTED_DENSITY_LINEWIDTH,
                    label=batch_label,
                    **kwargs,
                )
        else:
            # Too sparse for a density estimate: a tiny rug on the
            # baseline keeps the level (and its data) visible. Dense
            # ridges get their baseline stroke from the fill outline,
            # so draw one explicitly here to keep the rows uniform.
            values = continuous[groups == level]
            lines = ax.vlines if discrete_y else ax.hlines
            baseline_xy = ([vmin, vmax], [base, base])
            if not discrete_y:
                baseline_xy = baseline_xy[::-1]
            ax.plot(*baseline_xy, color=color, linewidth=SEGMENTED_DENSITY_LINEWIDTH)
            artists[level] = lines(
                values,
                base,
                base + SEGMENTED_DENSITY_TICK_FRAC,
                color=color,
                linewidth=SEGMENTED_DENSITY_TICK_LINEWIDTH,
                label=batch_label,
            )
    if sparse:
        level_list = ", ".join(str(level) for level in sparse)
        print(
            f"Note: level(s) {level_list} have fewer than 2 distinct "
            "values, so no density curve can be estimated for them. Their "
            "observations are drawn as tick marks on the baseline instead. "
            "Simulating more draws will fill those levels in."
        )

    # Label the discrete axis with the level values, one tick per
    # baseline (including baselines from earlier overlaid calls), and
    # leave headroom above the top baseline for its ridge; the
    # continuous axis keeps ordinary numeric ticks. Axis labels match
    # the segmented rug and mixed tile plots' "X"/"Y".
    all_levels = sorted(positions, key=positions.get)
    ticks = [positions[level] for level in all_levels]
    lo = min(ticks) - 0.2
    hi = max(ticks) + SEGMENTED_DENSITY_PEAK_SCALE + 0.1
    if discrete_y:
        ax.set_yticks(ticks)
        ax.set_yticklabels(all_levels)
        ax.set_ylim(lo, hi)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(all_levels)
        ax.set_xlim(lo, hi)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Segmented Density Plot")
    # Reference gridlines on both axes: the ridges are density curves,
    # which show both horizontal and vertical reference lines (the same
    # per-type override make_density uses -- symbulate.mplstyle's
    # global grid is horizontal-only), and the lines along the discrete
    # axis give one line per level, matching the integrated segmented
    # rug's convention (they extend the baselines across the full axes
    # width). axisbelow keeps them behind the ridges; the grid's color
    # and width come from symbulate.mplstyle.
    ax.set_axisbelow(True)
    ax.grid(True, axis="both")
    # A legend only helps once there is more than one batch to tell
    # apart; a lone batch stays legend-free.
    if ax._segmented_density_count > 1:
        ax.legend(loc=SEGMENTED_DENSITY_LEGEND_LOC)
    return [artists[level] for level in all_levels if level in artists]


def _next_segmented_density_color(ax):
    """Return the next automatic ridge color for these axes.

    Mirrors get_next_color() in symbulate/plot.py: walks the active
    style sheet's color cycle (symbulate.mplstyle leads with sky blue,
    then orange, ...), stored per axes so overlays advance it. Reading
    the cycle from the style sheet -- the single source of truth --
    guarantees every plot type starts from the same first color.
    (Re-implemented here, the same way dotplot.py does, so this staging
    module works standalone without importing the symbulate package;
    it collapses into get_next_color at integration time.)
    """
    if not hasattr(ax, "_segmented_density_color_cycle"):
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        ax._segmented_density_color_cycle = itertools.cycle(
            prop_cycle.by_key()["color"]
        )
    return next(ax._segmented_density_color_cycle)


if __name__ == "__main__":
    # Render the prototype with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    groups = rng.integers(0, 5, 400)
    values = rng.normal(0, 1, 400) + groups

    # Figure 1: the classic orientation -- discrete y, continuous x,
    # one horizontal ridge per level stacked vertically. No color
    # argument: the first batch on the axes automatically takes the
    # first hue in the symbulate.mplstyle cycle (Okabe-Ito sky blue).
    ax = plt.gca()
    make_segmented_density(values, groups, ax)

    # Figure 2: the same data with the axes swapped -- discrete x, so
    # the ridges stand vertically and stack horizontally.
    plt.figure()
    ax = plt.gca()
    make_segmented_density(groups, values, ax)

    # Figure 3: a sparse level -- level 9 has a single observation, so
    # it falls back to baseline tick marks and prints the note.
    plt.figure()
    ax = plt.gca()
    make_segmented_density(
        np.append(values, 4.0),
        np.append(groups, 9),
        ax,
    )

    # Figure 4: overlay -- a second no-argument call on the same axes
    # lands on the same baselines and automatically advances to the
    # next Okabe-Ito color (orange), with the "Variable k" legend.
    plt.figure()
    ax = plt.gca()
    make_segmented_density(values, groups, ax)
    make_segmented_density(values + 1.5, groups, ax)

    plt.show()
