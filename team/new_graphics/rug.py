"""Draft of the rug plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_rug()`` below when the user asks for a rug plot or the default
plot lookup table selects one (1D data with small n).

Visual target: one light Okabe-Ito tick mark per simulated value along
the bottom of the axes. The ticks are short and consistent in height
regardless of context -- they work equally well as a standalone plot
or accompanying another plot type (e.g. ``type=("hist", "rug")``). A
standalone rug hides the y-axis, the left spine, and the gridlines for
a clean number-line look, since the vertical direction carries no
information; when the rug is overlaid on another plot, it leaves the
axes untouched so the companion plot's styling shows through. Fonts and
figure size come from ``symbulate/symbulate.mplstyle``; the
per-plot-type values (alpha, tick height, line width) are the named
constants below, which migrate to the top of ``symbulate/plot.py`` at
integration time.

This file also provides ``make_segmented_rug()`` for mixed
discrete/continuous data -- the small-n counterpart of the mixed tile
plot. Instead of binning the continuous variable, it draws one rug of
the continuous values per discrete level, with the bands stacked along
whichever axis is discrete (mirroring the tile plot's orientation). See
its docstring for details.

Run this file directly to render the prototype:

    python team/new_graphics/rug.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
RUG_ALPHA = 0.5
RUG_TICK_HEIGHT = 0.04  # fraction of the axes height
RUG_LINEWIDTH = 1.0
RUG_LEGEND_LOC = "upper right"


def make_rug(values, ax, color, alpha=None, label=None, **kwargs):
    """Draw a rug plot of simulated values on the given axes.

    Draws one thin vertical tick per simulated value along the bottom
    of the axes. Tick heights are drawn in axes fractions (``RUG_TICK_HEIGHT``
    of the axes height), not data units, so the ticks keep their size if
    something with a meaningful y-scale (e.g. a histogram) is drawn on
    the same axes later.

    The function automatically detects whether this is a standalone rug plot
    or overlaying another plot type. For a standalone rug, it hides the
    y-axis, the left spine, and the gridlines for a clean number-line look.
    For an overlay, it leaves the axes untouched -- it only adds the ticks
    and lets the companion plot own the y-axis, spines, grid, labels, and
    title.

    Rug plots overlay naturally: a second call on the same axes draws on top
    of the first, and a legend appears automatically in the top right once two
    or more rugs share the axes. Each rug is named by ``label``, or
    "Variable 1", "Variable 2", ... in call order when no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_tile``, ``make_density2D``) are called.

    Parameters
    ----------
    values : array-like
        The simulated values to mark, e.g. ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Color for the tick marks, from ``get_next_color(ax)``.
    alpha : float, optional
        Tick transparency between 0 and 1. Defaults to the package
        standard for rug plots (0.5), which keeps stacked values
        visible as darker ticks.
    label : str, optional
        Name for this rug in the legend. Defaults to "Variable k",
        where k counts the rugs drawn on these axes so far.
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    matplotlib.collections.LineCollection
        The collection of tick marks, so the caller can inspect or
        further style them.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from symbulate.plot import get_next_color
    >>> values = np.random.default_rng().normal(0, 1, 60)
    >>> ax = plt.gca()
    >>> make_rug(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = RUG_ALPHA
    # Count the rugs drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color
    # cycle) so overlays from separate .plot() calls see it.
    n_prior_rugs = getattr(ax, "_rug_count", 0)
    if label is None:
        label = f"Variable {n_prior_rugs + 1}"
    ax._rug_count = n_prior_rugs + 1

    # Check if this is a standalone rug plot (nothing else on the axes yet)
    # or an overlay on another plot type.
    is_standalone = (
        len(ax.patches) == 0 and len(ax.lines) == 0 and len(ax.collections) == 0
    )

    rug = ax.vlines(
        np.asarray(values),
        0,
        RUG_TICK_HEIGHT,
        # Axes-fraction y coordinates: ticks rise from the bottom of
        # the axes regardless of the y data limits.
        transform=ax.get_xaxis_transform(),
        color=color,
        alpha=alpha,
        linewidth=RUG_LINEWIDTH,
        label=label,
        **kwargs,
    )

    if is_standalone:
        # Standalone rug plot: the vertical direction carries no
        # information, so hide the y-axis, the left spine, and the
        # gridlines for a clean number-line look. When the rug is
        # overlaid on another plot, none of this runs -- the companion
        # plot owns the axes styling and the rug inherits it untouched.
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(False)

    ax.set_xlabel("Value")
    # A legend only helps once there is more than one rug to tell
    # apart; a lone rug stays legend-free.
    if ax._rug_count > 1:
        ax.legend(loc=RUG_LEGEND_LOC)
    return rug


def make_segmented_rug(
    x, y, ax, color, alpha=None, discrete_x=None, discrete_y=None, **kwargs
):
    """Draw a segmented rug plot for mixed discrete/continuous data.

    The small-n counterpart of the mixed tile plot: instead of binning the
    continuous variable, every simulated value is drawn as a rug tick, and
    the ticks are grouped into one band per level of the discrete variable.
    A student can see each individual data point while still reading how the
    continuous variable is distributed within each discrete level.

    The orientation follows which variable is discrete, mirroring the mixed
    tile plot so the small-n and large-n views of the same data line up:

    - discrete ``y``, continuous ``x``: one band per y-level stacked
      vertically, with vertical ticks along x.
    - discrete ``x``, continuous ``y``: one band per x-level stacked
      horizontally, with horizontal ticks along y.

    Either way the ticks run perpendicular to the continuous value axis,
    marking each observation's position along it inside its band. The
    discrete axis is labeled with the level values; the continuous axis
    keeps ordinary numeric ticks. The tick marks rise from each level's
    baseline and are the same small size as the 1D ``make_rug`` ticks
    (``RUG_TICK_HEIGHT``).

    This plot is only for mixed data -- exactly one discrete variable and
    one continuous variable. Two discrete variables should use a tile plot,
    and two continuous variables a scatter plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` -- pass the result in as ``color``. This mirrors
    how ``make_rug`` and the other plot helpers are called.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first column of
        ``RVResults.array``. Discrete or continuous.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
        Discrete or continuous.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Color for the tick marks, from ``get_next_color(ax)``.
    alpha : float, optional
        Tick transparency between 0 and 1. Defaults to the package standard
        for rug plots (``RUG_ALPHA``, 0.5), so stacked values read as darker.
    discrete_x : bool, optional
        Whether the x-axis is the discrete (grouping) variable. If None
        (default), detected from the data: float values are treated as
        continuous, everything else (int, bool, string) as discrete.
    discrete_y : bool, optional
        Same as ``discrete_x`` for the y-axis.
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    list of matplotlib.collections.LineCollection
        The tick collections, one per discrete level, so the caller can
        inspect or further style them.

    Raises
    ------
    ValueError
        If the two variables are not one discrete and one continuous.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.normal(0, 1, 60)     # continuous
    >>> y = rng.integers(0, 4, 60)   # discrete groups
    >>> make_segmented_rug(x, y, plt.gca(), "#56B4E9")  # doctest: +SKIP
    """
    if alpha is None:
        alpha = RUG_ALPHA
    xs, ys = np.asarray(x), np.asarray(y)
    if discrete_x is None:
        discrete_x = not np.issubdtype(xs.dtype, np.floating)
    if discrete_y is None:
        discrete_y = not np.issubdtype(ys.dtype, np.floating)

    # A segmented rug needs one discrete variable (the groups) and one
    # continuous variable (the values). Anything else is a different plot.
    if discrete_x == discrete_y:
        if discrete_x:
            raise ValueError(
                "A segmented rug plot needs one discrete variable and one "
                "continuous variable, but both of yours look discrete. Try a "
                "tile plot for two discrete variables."
            )
        raise ValueError(
            "A segmented rug plot needs one discrete variable and one "
            "continuous variable, but both of yours look continuous. Try a "
            "scatter plot for two continuous variables."
        )

    # The discrete variable defines the bands; the continuous variable is the
    # value axis. Vertical ticks when the continuous axis is x, horizontal
    # ticks when it is y -- the ticks always run perpendicular to the value
    # axis, mirroring the mixed tile plot's orientation.
    if discrete_y:
        levels = np.unique(ys)
        continuous, groups = xs, ys
    else:
        levels = np.unique(xs)
        continuous, groups = ys, xs

    # Make the ticks the same visual size as the 1D make_rug ticks
    # (RUG_TICK_HEIGHT, a fraction of the axes). The discrete axis is fixed
    # below to span len(levels) data units, so RUG_TICK_HEIGHT * len(levels)
    # data units is that same fraction of the axes. Ticks rise from each
    # level's baseline, so every band reads as its own small 1D rug.
    tick_len = RUG_TICK_HEIGHT * len(levels)
    ticks = []
    for i, level in enumerate(levels):
        values = continuous[groups == level]
        if discrete_y:
            ticks.append(
                ax.vlines(
                    values,
                    i,
                    i + tick_len,
                    color=color,
                    alpha=alpha,
                    linewidth=RUG_LINEWIDTH,
                    **kwargs,
                )
            )
        else:
            ticks.append(
                ax.hlines(
                    values,
                    i,
                    i + tick_len,
                    color=color,
                    alpha=alpha,
                    linewidth=RUG_LINEWIDTH,
                    **kwargs,
                )
            )

    # Label the discrete axis with the level values (one tick per band) and
    # give it a little padding so the outer bands aren't clipped; the
    # continuous axis keeps matplotlib's numeric ticks. Axis labels match the
    # mixed tile plot's "X"/"Y".
    positions = np.arange(len(levels))
    if discrete_y:
        ax.set_yticks(positions)
        ax.set_yticklabels(levels)
        ax.set_ylim(-0.5, len(levels) - 0.5)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(levels)
        ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Segmented Rug Plot")
    # A rug reads cleanest without a reference grid behind the sparse ticks
    # (the same choice as the standalone 1D rug).
    ax.grid(False)
    return ticks


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    values = rng.normal(0, 1, 60)

    # Figure 1: the single-rug prototype. The short, wide figure
    # matches the prototype image; this is demo-only -- the final
    # .plot() integration keeps the global default figure size.
    plt.figure(figsize=(9, 2))
    ax = plt.gca()
    # Okabe-Ito sky blue -- the *first* hue in the symbulate.mplstyle
    # color cycle, so every plot type starts from this same color;
    # once integrated, it will come from get_next_color(ax) instead.
    make_rug(values, ax, "#56B4E9")

    # Figure 2: two overlaid rugs with the automatic "Variable k"
    # legend in the top right.
    plt.figure(figsize=(9, 2))
    ax = plt.gca()
    make_rug(rng.normal(0, 1, 60), ax, "#56B4E9")
    make_rug(rng.normal(1.5, 0.5, 60), ax, "#E69F00")

    # Figure 3: segmented rug for mixed data -- discrete y (Poisson levels),
    # continuous x (Normal). Bands stack vertically; ticks are vertical.
    plt.figure()
    ax = plt.gca()
    groups = rng.integers(0, 4, 200)
    make_segmented_rug(rng.normal(0, 1, 200) + groups, groups, ax, "#56B4E9")

    # Figure 4: the same data with the axes swapped -- discrete x, continuous
    # y -- so the bands stack horizontally and the ticks are horizontal.
    plt.figure()
    ax = plt.gca()
    make_segmented_rug(groups, rng.normal(0, 1, 200) + groups, ax, "#56B4E9")

    plt.show()
