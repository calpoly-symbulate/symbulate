"""Draft of the segmented histogram plot type for the graphics
overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_segmented_hist()`` below when the user asks for a segmented
histogram (``type='segmented_hist'``). It is the histogram sibling of
the segmented density plot (team decision, 2026-07-15): the same
one-row-per-level layout for mixed discrete/continuous data, but each
level shows a small histogram of the continuous values instead of a
smooth density curve -- binned and blocky where the density is smooth,
which keeps small-sample lumpiness honest instead of smoothing over it.

Visual target: one small histogram per level of the discrete variable,
rising from that level's baseline, stacked along the discrete axis with
the same gentle overlap as the segmented density plot. Every level
shares one set of bin edges (computed from the pooled continuous
values) so bars line up in columns across levels, and one height scale
(the tallest bar reaches ``SEGMENTED_HIST_PEAK_SCALE`` baseline
spacings), so bar heights are directly comparable across levels. Bars
use the 1D histogram's styling -- translucent fill with thin white
edges -- over an explicit baseline in the ridge color. The
per-plot-type values are the named constants below, which migrate to
the top of ``symbulate/plot.py`` at integration time.

The orientation follows which variable is discrete, mirroring the
segmented rug, segmented density, and mixed tile plots: discrete ``y``
gives horizontal rows of upward bars, discrete ``x`` flips it (vertical
columns of rightward bars).

Run this file directly to render the prototype:

    python team/new_graphics/segmented_hist.py
"""

import itertools

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
SEGMENTED_HIST_ALPHA = 0.65  # matches HIST_ALPHA -- the bars are
# ordinary histogram bars, just small
SEGMENTED_HIST_EDGECOLOR = "white"  # matches HIST_EDGECOLOR
SEGMENTED_HIST_EDGEWIDTH = 0.8  # matches HIST_EDGEWIDTH
SEGMENTED_HIST_DEFAULT_BINS = 30  # matches HIST_DEFAULT_BINS; share
# one constant at integration time
SEGMENTED_HIST_PEAK_SCALE = 1.6  # tallest bar's height, in units of
# the spacing between neighboring baselines -- matches
# SEGMENTED_DENSITY_PEAK_SCALE so the two segmented plots of one
# dataset stack identically
SEGMENTED_HIST_BASELINE_LINEWIDTH = 1.8  # matches
# SEGMENTED_DENSITY_LINEWIDTH; the density plot's baseline comes from
# its fill outline, so this keeps the two plots' baselines identical
SEGMENTED_HIST_LEGEND_LOC = "upper right"
SEGMENTED_HIST_TICK_FRAC = 0.2  # height of the fallback tick marks
# drawn for levels too sparse for a histogram, as a fraction of the
# baseline spacing -- matches SEGMENTED_DENSITY_TICK_FRAC
SEGMENTED_HIST_TICK_LINEWIDTH = 1.0  # matches RUG_LINEWIDTH


def make_segmented_hist(
    x,
    y,
    ax,
    color=None,
    bins=None,
    normalize=True,
    alpha=None,
    label=None,
    discrete_x=None,
    discrete_y=None,
    **kwargs,
):
    """Draw a segmented histogram for mixed discrete/continuous data.

    One small histogram of the continuous variable per level of the
    discrete variable, rising from that level's baseline and stacked
    along the discrete axis -- the histogram sibling of the segmented
    density plot. Where the density plot shows each level's smoothed
    shape, the segmented histogram shows the raw binned shape, so
    small-sample lumpiness stays visible instead of being smoothed
    over.

    All levels share one set of bin edges, computed from the pooled
    continuous values, so the bars line up in columns across levels.
    They also share one height scale (the tallest bar in this call
    reaches ``SEGMENTED_HIST_PEAK_SCALE`` baseline spacings), so bar
    heights are directly comparable across levels. With
    ``normalize=True`` (the default) each level's bars show that
    level's *density* -- every level's histogram has area 1, so levels
    are compared by shape no matter how many observations each has.
    With ``normalize=False`` the bars show raw counts, so a level with
    more observations also gets taller bars.

    The orientation follows which variable is discrete, mirroring the
    segmented rug, segmented density, and mixed tile plots so the
    different views of the same data line up:

    - discrete ``y``, continuous ``x``: one horizontal row of upward
      bars per y-level, stacked vertically.
    - discrete ``x``, continuous ``y``: flipped -- one vertical column
      of rightward bars per x-level, stacked horizontally.

    Levels with fewer than two distinct values are shown as short tick
    marks on that level's baseline instead of a histogram (their single
    bar would dwarf every real histogram under the shared density
    scale), and a note explains which levels fell back and why.

    A light reference grid sits behind the bars on both axes, and every
    level draws a baseline across the full shared bin range, matching
    the segmented density plot.

    Segmented histograms overlay naturally: a second call on the same
    axes draws a second set of histograms on the same baselines (levels
    are matched by value, and levels new to the axes get new
    baselines), and a legend appears automatically once two or more
    batches share the axes. When no ``color`` is given, each batch
    takes the next hue from the active style sheet's color cycle
    automatically -- sky blue first with symbulate.mplstyle, then
    orange, ... Each batch is named by ``label``, or "Variable 1",
    "Variable 2", ... in call order when no label is given. Each batch
    is scaled to its own tallest bar and its own bin edges, so overlays
    compare shapes, not absolute heights.

    This plot is only for mixed data -- exactly one discrete variable
    and one continuous variable. Two discrete variables should use a
    tile plot, and two continuous variables a scatter plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). Once integrated, ``RVResults.plot()`` will
    advance the color cycle with ``get_next_color(ax)`` exactly once
    per call and pass the result in as ``color``; standalone, leaving
    ``color`` unset reads the same cycle, so both paths give the same
    colors.

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
        Color for the bars. Defaults to the next color in the active
        style sheet's cycle (sky blue for the first batch on the axes,
        orange for the second, ...), the same cycle
        ``get_next_color(ax)`` in ``symbulate/plot.py`` reads.
    bins : int or sequence, optional
        Number of equal-width bins (or explicit bin edges) for the
        shared bin grid, passed to ``numpy.histogram_bin_edges`` over
        the pooled continuous values. Defaults to the package standard
        for histograms (``SEGMENTED_HIST_DEFAULT_BINS``, 30).
    normalize : bool, optional
        If True (default), each level's bars show that level's density
        (area 1 per level), comparing levels by shape. If False, the
        bars show raw counts, so more-frequent levels get taller bars.
    alpha : float, optional
        Bar fill transparency between 0 and 1. Defaults to the package
        standard for histograms (``SEGMENTED_HIST_ALPHA``, 0.65).
    label : str, optional
        Name for this batch of histograms in the legend. Defaults to
        "Variable k", where k counts the segmented-histogram batches
        drawn on these axes so far.
    discrete_x : bool, optional
        Whether the x-axis is the discrete (grouping) variable. If None
        (default), detected from the data: float values are treated as
        continuous, everything else (int, bool, string) as discrete.
    discrete_y : bool, optional
        Same as ``discrete_x`` for the y-axis.
    **kwargs
        Additional keyword arguments passed to ``ax.bar`` / ``ax.barh``
        for the bars.

    Returns
    -------
    list
        The drawn artists, one per level in baseline order -- a
        ``BarContainer`` for each histogram, or a ``LineCollection``
        for each sparse level's fallback ticks -- so the caller can
        inspect or further style them.

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
    >>> make_segmented_hist(x, y, plt.gca())  # doctest: +SKIP
    """
    if alpha is None:
        alpha = SEGMENTED_HIST_ALPHA
    if bins is None:
        bins = SEGMENTED_HIST_DEFAULT_BINS
    xs, ys = np.asarray(x), np.asarray(y)
    if discrete_x is None:
        discrete_x = not np.issubdtype(xs.dtype, np.floating)
    if discrete_y is None:
        discrete_y = not np.issubdtype(ys.dtype, np.floating)

    # A segmented histogram needs one discrete variable (the baselines)
    # and one continuous variable (the bars). Anything else is a
    # different plot.
    if discrete_x == discrete_y:
        if discrete_x:
            raise ValueError(
                "A segmented histogram needs one discrete variable and one "
                "continuous variable, but both of yours look discrete. Try a "
                "tile plot or a scatter plot with jitter for two discrete "
                "variables."
            )
        raise ValueError(
            "A segmented histogram needs one discrete variable and one "
            "continuous variable, but both of yours look continuous. Try a "
            "scatter plot for two continuous variables."
        )

    # Resolve the automatic color only after validation, so a call that
    # errors out doesn't advance the cycle.
    if color is None:
        color = _next_segmented_hist_color(ax)

    # The discrete variable defines the baselines; the continuous
    # variable is the value axis the bars are binned along.
    if discrete_y:
        levels = np.unique(ys)
        continuous, groups = xs, ys
    else:
        levels = np.unique(xs)
        continuous, groups = ys, xs

    # Baseline positions live on the axes object so a second .plot()
    # call lands its histograms on the same baselines (matched by level
    # value); levels the axes hasn't seen yet get the next free
    # baseline. Same pattern the segmented density plot uses.
    positions = getattr(ax, "_segmented_hist_positions", None)
    if positions is None:
        positions = {}
        ax._segmented_hist_positions = positions
    for level in levels:
        if level not in positions:
            positions[level] = len(positions)

    # Count the segmented-histogram batches drawn on these axes, for
    # the automatic "Variable k" legend names.
    n_prior = getattr(ax, "_segmented_hist_count", 0)
    if label is None:
        label = f"Variable {n_prior + 1}"
    ax._segmented_hist_count = n_prior + 1

    # One shared bin grid over the pooled continuous values, so every
    # level's bars line up in the same columns; then one shared height
    # scale, so the tallest bar in this call reaches
    # SEGMENTED_HIST_PEAK_SCALE baseline spacings and every other bar
    # keeps its true height relative to it.
    edges = np.histogram_bin_edges(continuous, bins=bins)
    widths = np.diff(edges)
    heights = {}
    sparse = []
    for level in levels:
        values = continuous[groups == level]
        if np.unique(values).size < 2:
            sparse.append(level)
        else:
            heights[level] = np.histogram(values, bins=edges, density=normalize)[0]
    if heights:
        scale = SEGMENTED_HIST_PEAK_SCALE / max(h.max() for h in heights.values())

    # The bars use the 1D histogram's styling -- translucent fill,
    # thin white edges -- so a segmented histogram reads as a stack of
    # ordinary small histograms.
    facecolor = mcolors.to_rgba(color, alpha)

    # Draw from the highest baseline down so that where bars overlap
    # the row above, the lower (nearer) histogram sits in front -- the
    # same order as the segmented density plot.
    artists = {}
    for level in sorted(levels, key=positions.get, reverse=True):
        base = positions[level]
        batch_label = label if not artists else None
        # Unlike the density ridge, whose fill outline strokes its own
        # baseline, bars only cover occupied bins -- so every level
        # draws an explicit baseline across the full shared bin range.
        baseline_xy = ([edges[0], edges[-1]], [base, base])
        if not discrete_y:
            baseline_xy = baseline_xy[::-1]
        ax.plot(
            *baseline_xy,
            color=color,
            linewidth=SEGMENTED_HIST_BASELINE_LINEWIDTH,
        )
        if level in heights:
            bar_sizes = heights[level] * scale
            if discrete_y:
                artists[level] = ax.bar(
                    edges[:-1],
                    bar_sizes,
                    width=widths,
                    bottom=base,
                    align="edge",
                    facecolor=facecolor,
                    edgecolor=SEGMENTED_HIST_EDGECOLOR,
                    linewidth=SEGMENTED_HIST_EDGEWIDTH,
                    label=batch_label,
                    **kwargs,
                )
            else:
                artists[level] = ax.barh(
                    edges[:-1],
                    bar_sizes,
                    height=widths,
                    left=base,
                    align="edge",
                    facecolor=facecolor,
                    edgecolor=SEGMENTED_HIST_EDGECOLOR,
                    linewidth=SEGMENTED_HIST_EDGEWIDTH,
                    label=batch_label,
                    **kwargs,
                )
        else:
            # Too sparse for a histogram under the shared scale: a tiny
            # rug on the baseline keeps the level (and its data)
            # visible, matching the segmented density plot's fallback.
            values = continuous[groups == level]
            lines = ax.vlines if discrete_y else ax.hlines
            artists[level] = lines(
                values,
                base,
                base + SEGMENTED_HIST_TICK_FRAC,
                color=color,
                linewidth=SEGMENTED_HIST_TICK_LINEWIDTH,
                label=batch_label,
            )
    if sparse:
        level_list = ", ".join(str(level) for level in sparse)
        print(
            f"Note: level(s) {level_list} have fewer than 2 distinct "
            "values, so their observations are drawn as tick marks on the "
            "baseline instead of a histogram. Simulating more draws will "
            "fill those levels in."
        )

    # Label the discrete axis with the level values, one tick per
    # baseline (including baselines from earlier overlaid calls), and
    # leave headroom above the top baseline for its bars; the
    # continuous axis keeps ordinary numeric ticks. Axis labels match
    # the other mixed-data plots' "X"/"Y".
    all_levels = sorted(positions, key=positions.get)
    ticks = [positions[level] for level in all_levels]
    lo = min(ticks) - 0.2
    hi = max(ticks) + SEGMENTED_HIST_PEAK_SCALE + 0.1
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
    ax.set_title("Segmented Histogram")
    # Reference gridlines on both axes, matching the segmented density
    # plot: lines along the continuous axis for reading values, one
    # line per level along the discrete axis extending the baselines.
    # axisbelow keeps them behind the bars; the grid's color and width
    # come from symbulate.mplstyle.
    ax.set_axisbelow(True)
    ax.grid(True, axis="both")
    # A legend only helps once there is more than one batch to tell
    # apart; a lone batch stays legend-free.
    if ax._segmented_hist_count > 1:
        ax.legend(loc=SEGMENTED_HIST_LEGEND_LOC)
    return [artists[level] for level in all_levels if level in artists]


def _next_segmented_hist_color(ax):
    """Return the next automatic bar color for these axes.

    Mirrors get_next_color() in symbulate/plot.py: walks the active
    style sheet's color cycle (symbulate.mplstyle leads with sky blue,
    then orange, ...), stored per axes so overlays advance it. Reading
    the cycle from the style sheet -- the single source of truth --
    guarantees every plot type starts from the same first color.
    (Re-implemented here, the same way segmented_density.py does, so
    this staging module works standalone without importing the
    symbulate package; it collapses into get_next_color at integration
    time.)
    """
    if not hasattr(ax, "_segmented_hist_color_cycle"):
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        ax._segmented_hist_color_cycle = itertools.cycle(prop_cycle.by_key()["color"])
    return next(ax._segmented_hist_color_cycle)


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
    # one row of upward bars per level. No color argument: the first
    # batch on the axes automatically takes the first hue in the
    # symbulate.mplstyle cycle (Okabe-Ito sky blue).
    ax = plt.gca()
    make_segmented_hist(values, groups, ax)

    # Figure 2: the same data with the axes swapped -- discrete x, so
    # the bars extend rightward and the columns stack horizontally.
    plt.figure()
    ax = plt.gca()
    make_segmented_hist(groups, values, ax)

    # Figure 3: a sparse level -- level 9 has a single observation, so
    # it falls back to baseline tick marks and prints the note.
    plt.figure()
    ax = plt.gca()
    make_segmented_hist(
        np.append(values, 0.0),
        np.append(groups, 9),
        ax,
    )

    # Figure 4: overlay -- a second no-argument call on the same axes
    # lands on the same baselines and automatically advances to the
    # next Okabe-Ito color (orange), with the "Variable k" legend.
    plt.figure()
    ax = plt.gca()
    make_segmented_hist(values, groups, ax)
    make_segmented_hist(values + 1.5, groups, ax)

    plt.show()
