"""Draft of the grouped violin plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_violin()`` below when the user asks for a violin plot on data
with a categorical grouping variable, extending the current
``make_violin`` in ``symbulate/plot.py`` (single discrete axis, no
subgrouping) with an optional ``hue`` variable.

Visual target: the approved grouped-violin prototype -- one violin per
category, filled with an Okabe-Ito hue and a thin black outline, shaped
by a kernel density estimate of that group's values. When a second
categorical variable (``hue``) is given, the violins for each category
are dodged side by side and colored by hue level instead of by category,
with a legend naming each level. Every violin carries a narrow inner
boxplot (pale ivory box, black median line and whiskers) so a student
can read the median and IQR without losing the overall shape. The grid,
spines, fonts, and figure size come from ``symbulate/symbulate.mplstyle``;
the per-plot-type values (alpha, box proportions, dodge width) are the
named constants below, which migrate to the top of ``symbulate/plot.py``
at integration time.

Overlay category (see CLAUDE.md, "Overlay Policy"): **readability
warning** -- a second violin plot on the same axes is drawn, but the
overlapping shapes are hard to read, so a student-friendly warning is
printed below the plot (the same category as two tile plots or two 2D
histograms).

Run this file directly to render the prototype:

    python team/new_graphics/violin.py
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
VIOLIN_ALPHA = 0.8
VIOLIN_EDGECOLOR = "black"
VIOLIN_EDGEWIDTH = 1.0
VIOLIN_GROUP_WIDTH = 0.8  # total width of all violins in one category slot
VIOLIN_WIDTH_RATIO = 0.9  # fraction of its lane a violin body fills
VIOLIN_BOX_WIDTH_RATIO = 0.35  # fraction of the violin width the inner box fills
VIOLIN_BOX_FACECOLOR = "#FDF6E3"  # pale ivory, reads against any hue color
VIOLIN_LEGEND_LOC = "upper right"

VIOLIN_OVERLAY_WARNING = (
    "Warning: you drew a second violin plot on the same plot. Overlapping "
    "violin shapes are hard to tell apart, so the result may be hard to "
    "read. Consider plotting them in separate cells, or using hue= to show "
    "subgroups side by side instead."
)


def make_violin(categories, values, ax, hue=None, hue_name="Group", alpha=None):
    """Draw a violin plot of simulated values grouped by a category.

    One violin per distinct value of ``categories``, each shaped by a
    kernel density estimate of the values in that group and filled with
    an Okabe-Ito hue. Every violin carries a narrow inner boxplot (pale
    box, black median line and whiskers) so the median and IQR stay
    readable underneath the density shape.

    Passing ``hue`` splits each category's violin further by a second
    categorical variable: the violins for a category are dodged side by
    side, one per hue level, and colored by hue level instead of by
    category. A legend titled ``hue_name`` names each level. Without
    ``hue``, every violin shares a single color (the first hue in the
    Okabe-Ito cycle).

    A second call on the same axes still draws (the "readability warning"
    category of the overlay policy), but two sets of violin shapes are
    hard to tell apart, so a warning prints below the plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_tile``, ``make_density2D``) are called.

    Parameters
    ----------
    categories : array-like
        The categorical (discrete) grouping value for each simulated
        value, e.g. ``["Group A", "Group B", ...]``. One violin is drawn
        per distinct value.
    values : array-like
        The simulated continuous values, same length as ``categories``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    hue : array-like, optional
        A second categorical variable, same length as ``categories``,
        that splits each category's violin into one dodged violin per
        hue level. If None (default), one violin is drawn per category
        with no splitting.
    hue_name : str, default "Group"
        Legend title used when ``hue`` is given. Ignored when ``hue`` is
        None.
    alpha : float, optional
        Violin body transparency between 0 and 1. Defaults to the
        package standard for violins (0.8).

    Returns
    -------
    dict
        Maps each hue level (or ``None`` when ``hue`` is not given) to
        the ``(violins, boxplot)`` pair returned by ``ax.violinplot`` and
        ``ax.boxplot``, so the caller can inspect or further style them.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> categories = rng.choice(["A", "B", "C"], 300)
    >>> values = rng.normal(0, 1, 300)
    >>> make_violin(categories, values, plt.gca())  # doctest: +SKIP
    """
    if alpha is None:
        alpha = VIOLIN_ALPHA
    categories = np.asarray(categories)
    values = np.asarray(values)
    category_levels = np.unique(categories)
    positions = np.arange(len(category_levels))

    hue_levels = [None] if hue is None else list(np.unique(hue))
    hue_array = None if hue is None else np.asarray(hue)
    n_hues = len(hue_levels)
    lane_width = VIOLIN_GROUP_WIDTH / n_hues
    violin_width = lane_width * VIOLIN_WIDTH_RATIO
    box_width = violin_width * VIOLIN_BOX_WIDTH_RATIO

    # Okabe-Ito colors come from the active style sheet's color cycle
    # (symbulate.mplstyle: sky blue first, then orange, ...) -- one per
    # hue level, so a lone violin (no hue) gets the same first color
    # every other plot type starts from.
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = {level: next(color_cycle) for level in hue_levels}

    results = {}
    for i, level in enumerate(hue_levels):
        mask = (
            np.ones(len(categories), dtype=bool) if hue is None else hue_array == level
        )
        # Center the lanes for this category slot around its integer
        # position: n_hues=1 puts the single violin dead center,
        # n_hues=2 puts them symmetrically left/right, and so on.
        offset = (i - (n_hues - 1) / 2) * lane_width
        data, lane_positions = [], []
        for pos, cat in zip(positions, category_levels):
            group_values = values[mask & (categories == cat)]
            if len(group_values) == 0:
                continue
            data.append(group_values)
            lane_positions.append(pos + offset)
        if not data:
            continue

        violins = ax.violinplot(
            data,
            positions=lane_positions,
            widths=violin_width,
            showmedians=False,
            showextrema=False,
        )
        for body in violins["bodies"]:
            body.set_facecolor(colors[level])
            body.set_edgecolor(VIOLIN_EDGECOLOR)
            body.set_linewidth(VIOLIN_EDGEWIDTH)
            body.set_alpha(alpha)

        boxplot = ax.boxplot(
            data,
            positions=lane_positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(
                facecolor=VIOLIN_BOX_FACECOLOR,
                edgecolor=VIOLIN_EDGECOLOR,
                linewidth=VIOLIN_EDGEWIDTH,
            ),
            medianprops=dict(color=VIOLIN_EDGECOLOR, linewidth=VIOLIN_EDGEWIDTH),
            whiskerprops=dict(color=VIOLIN_EDGECOLOR, linewidth=VIOLIN_EDGEWIDTH),
            capprops=dict(color=VIOLIN_EDGECOLOR, linewidth=VIOLIN_EDGEWIDTH),
        )
        # The boxplot marks the median and IQR on top of the violin's
        # density shape, so it needs to sit above the violin body.
        for artists in boxplot.values():
            for artist in artists:
                artist.set_zorder(3)
        results[level] = (violins, boxplot)

    ax.set_xticks(positions)
    ax.set_xticklabels(category_levels)
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    ax.set_title("Violin Plot")

    # A legend only helps once there is more than one hue level to tell
    # apart; a plain (non-hue) violin plot stays legend-free.
    if hue is not None:
        handles = [
            Patch(
                facecolor=colors[level],
                edgecolor=VIOLIN_EDGECOLOR,
                alpha=alpha,
                label=str(level),
            )
            for level in hue_levels
        ]
        ax.legend(handles=handles, title=hue_name, loc=VIOLIN_LEGEND_LOC)

    # Count the violin plots drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the color
    # cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_violin_count", 0)
    ax._violin_count = n_prior + 1
    if ax._violin_count > 1:
        print(VIOLIN_OVERLAY_WARNING)
    return results


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)

    # Figure 1: the grouped-violin prototype -- four categories, each
    # split into "No"/"Yes" subgroups with distinct centers and spreads.
    group_specs = {
        "Group A": {"No": (20, 3), "Yes": (19, 4)},
        "Group B": {"No": (28, 4), "Yes": (29, 4)},
        "Group C": {"No": (35, 3), "Yes": (33, 3)},
        "Group D": {"No": (24, 5), "Yes": (25, 6)},
    }
    n_per_cell = 80
    categories, subgroups, values = [], [], []
    for group, subs in group_specs.items():
        for sub, (mean, sd) in subs.items():
            categories.extend([group] * n_per_cell)
            subgroups.extend([sub] * n_per_cell)
            values.extend(rng.normal(mean, sd, n_per_cell))

    ax = plt.gca()
    make_violin(categories, values, ax, hue=subgroups, hue_name="Subgroup")

    # Figure 2: no hue -- one violin per category, single color.
    plt.figure()
    ax = plt.gca()
    make_violin(categories, values, ax)

    # Figure 3: the overlay prototype -- a second call on the same axes
    # still draws, but prints the readability warning below the plot.
    make_violin(categories, values, plt.gca())

    plt.show()
