"""Draft of the grouped box plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_boxplot()`` below when the user asks for a box plot on data with a
categorical grouping variable, the same "category plus optional hue"
design as the draft ``make_violin()`` in ``team/new_graphics/violin.py``.

Visual target: one box per category, colored with an Okabe-Ito hue and a
thin black outline -- box edges at the first and third quartiles, a black
median line, whiskers extending to the most extreme value within 1.5*IQR
of the box, and individual points beyond that drawn as fliers. When a
second categorical variable (``hue``) is given, the boxes for each
category are dodged side by side and colored by hue level instead of by
category, with a legend naming each level. The grid, spines, fonts, and
figure size come from ``symbulate/symbulate.mplstyle``; the per-plot-type
values (box width, flier styling) are the named constants below, which
migrate to the top of ``symbulate/plot.py`` at integration time.

Overlay category (see CLAUDE.md, "Overlay Policy"): **readability
warning** -- a second box plot on the same axes is drawn, but the
overlapping boxes and whiskers are hard to read, so a student-friendly
warning is printed below the plot (the same category as two violin plots,
two tile plots, or two 2D histograms).

Run this file directly to render the prototype:

    python team/new_graphics/new_graphics_demos/boxplot.py
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express per-plot-type values, so these cannot live in
# symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
BOXPLOT_EDGECOLOR = "black"
BOXPLOT_EDGEWIDTH = 1.2
BOXPLOT_GROUP_WIDTH = 0.8  # total width of all boxes in one category slot
BOXPLOT_WIDTH_RATIO = 0.7  # fraction of its lane a box fills
BOXPLOT_FLIER_MARKER = "o"
BOXPLOT_FLIER_SIZE = 4
BOXPLOT_LEGEND_LOC = "upper right"  # default: inside the axes, like every
# other prototype (rug/hist/violin)
BOXPLOT_LEGEND_OUTSIDE_LOC = "lower right"
BOXPLOT_LEGEND_OUTSIDE_ANCHOR = (1.0, 1.0)  # anchors the legend's
# lower-right corner to the axes' top-right corner, so it sits just above
# the plot instead of inside it -- used only when the default inside
# placement would overlap a tall whisker or flier

BOXPLOT_OVERLAY_WARNING = (
    "Warning: you drew a second box plot on the same plot. Overlapping "
    "boxes and whiskers are hard to tell apart, so the result may be hard "
    "to read. Consider plotting them in separate cells, or using hue= to "
    "show subgroups side by side instead."
)


def _legend_overlaps_data(ax, legend, boxplot_results):
    """Check whether the legend's rendered box overlaps a drawn box, whisker, cap, or flier.

    Forces a draw so the legend and every artist have real pixel
    positions, then compares bounding boxes.
    """
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer)
    for boxplot in boxplot_results.values():
        for artists in boxplot.values():
            for artist in artists:
                if legend_bbox.overlaps(artist.get_window_extent(renderer)):
                    return True
    return False


def make_boxplot(categories, values, ax, hue=None, hue_name="Group"):
    """Draw a box plot of simulated values grouped by a category.

    One box per distinct value of ``categories``, filled with an
    Okabe-Ito hue. Each box spans the first to third quartiles with a
    black median line; whiskers extend to the most extreme value within
    1.5 times the interquartile range, and values beyond that are drawn
    as individual fliers.

    Passing ``hue`` splits each category's box further by a second
    categorical variable: the boxes for a category are dodged side by
    side, one per hue level, and colored by hue level instead of by
    category. A legend titled ``hue_name`` names each level, normally
    inside the axes in the top right; if a tall whisker or flier would
    overlap it there, the legend moves just outside/above the axes
    instead. Without ``hue``, every box shares a single color (the first
    hue in the Okabe-Ito cycle).

    A second call on the same axes still draws (the "readability warning"
    category of the overlay policy), but two sets of boxes are hard to
    tell apart, so a warning prints below the plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_tile``, ``make_density2D``) are called,
    and how the draft ``make_violin`` (``team/new_graphics/violin.py``)
    handles its own ``categories``/``hue`` grouping.

    Parameters
    ----------
    categories : array-like
        The categorical (discrete) grouping value for each simulated
        value, e.g. ``["Group A", "Group B", ...]``. One box is drawn
        per distinct value.
    values : array-like
        The simulated continuous values, same length as ``categories``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    hue : array-like, optional
        A second categorical variable, same length as ``categories``,
        that splits each category's box into one dodged box per hue
        level. If None (default), one box is drawn per category with no
        splitting.
    hue_name : str, default "Group"
        Legend title used when ``hue`` is given. Ignored when ``hue`` is
        None.

    Returns
    -------
    dict
        Maps each hue level (or ``None`` when ``hue`` is not given) to
        the dict returned by ``ax.boxplot``, so the caller can inspect or
        further style the boxes, whiskers, caps, medians, and fliers.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> categories = rng.choice(["A", "B", "C"], 300)
    >>> values = rng.normal(0, 1, 300)
    >>> make_boxplot(categories, values, plt.gca())  # doctest: +SKIP
    """
    categories = np.asarray(categories)
    values = np.asarray(values)
    category_levels = np.unique(categories)
    positions = np.arange(len(category_levels))

    hue_levels = [None] if hue is None else list(np.unique(hue))
    hue_array = None if hue is None else np.asarray(hue)
    n_hues = len(hue_levels)
    lane_width = BOXPLOT_GROUP_WIDTH / n_hues
    box_width = lane_width * BOXPLOT_WIDTH_RATIO

    # Okabe-Ito colors come from the active style sheet's color cycle
    # (symbulate.mplstyle: sky blue first, then orange, ...) -- one per
    # hue level, so a lone box plot (no hue) gets the same first color
    # every other plot type starts from.
    color_cycle = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = {level: next(color_cycle) for level in hue_levels}

    results = {}
    for i, level in enumerate(hue_levels):
        mask = (
            np.ones(len(categories), dtype=bool) if hue is None else hue_array == level
        )
        # Center the lanes for this category slot around its integer
        # position: n_hues=1 puts the single box dead center, n_hues=2
        # puts them symmetrically left/right, and so on.
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

        boxplot = ax.boxplot(
            data,
            positions=lane_positions,
            widths=box_width,
            patch_artist=True,
            boxprops=dict(
                facecolor=colors[level],
                edgecolor=BOXPLOT_EDGECOLOR,
                linewidth=BOXPLOT_EDGEWIDTH,
            ),
            medianprops=dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_EDGEWIDTH),
            whiskerprops=dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_EDGEWIDTH),
            capprops=dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_EDGEWIDTH),
            flierprops=dict(
                marker=BOXPLOT_FLIER_MARKER,
                markersize=BOXPLOT_FLIER_SIZE,
                markerfacecolor=colors[level],
                markeredgecolor=BOXPLOT_EDGECOLOR,
            ),
        )
        results[level] = boxplot

    ax.set_xticks(positions)
    ax.set_xticklabels(category_levels)
    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    ax.set_title("Box Plot")

    # A legend only helps once there is more than one hue level to tell
    # apart; a plain (non-hue) box plot stays legend-free.
    if hue is not None:
        handles = [
            Patch(
                facecolor=colors[level],
                edgecolor=BOXPLOT_EDGECOLOR,
                label=str(level),
            )
            for level in hue_levels
        ]
        # Try the default inside-the-axes position first, matching every
        # other prototype. Only move it outside/above if it would actually
        # overlap a drawn box, whisker, cap, or flier.
        legend = ax.legend(handles=handles, title=hue_name, loc=BOXPLOT_LEGEND_LOC)
        if _legend_overlaps_data(ax, legend, results):
            legend.remove()
            ax.legend(
                handles=handles,
                title=hue_name,
                loc=BOXPLOT_LEGEND_OUTSIDE_LOC,
                bbox_to_anchor=BOXPLOT_LEGEND_OUTSIDE_ANCHOR,
            )

    # Count the box plots drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color cycle),
    # to trigger the overlay readability warning.
    n_prior = getattr(ax, "_boxplot_count", 0)
    ax._boxplot_count = n_prior + 1
    if ax._boxplot_count > 1:
        print(BOXPLOT_OVERLAY_WARNING)
    return results


if __name__ == "__main__":
    # Reproduce the violin-plot prototype's fake data, so the two draft
    # plot types can be compared side by side on the same distributions.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[3] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)

    # Figure 1: the grouped-boxplot prototype -- four categories, each
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
    make_boxplot(categories, values, ax, hue=subgroups, hue_name="Subgroup")

    # Figure 2: no hue -- one box per category, single color.
    plt.figure()
    ax = plt.gca()
    make_boxplot(categories, values, ax)

    # Figure 3: the overlay prototype -- two cohorts that share category
    # labels but have different distributions. With no hue to dodge them
    # apart, each call draws exactly one box per category, so the second
    # cohort's box lands right on top of the first's and the warning fires.
    cohort1_specs = {
        "Group A": (19, 4),
        "Group B": (28, 4),
        "Group C": (34, 3),
        "Group D": (24, 5),
    }
    cohort2_specs = {
        "Group A": (30, 3),
        "Group B": (18, 3),
        "Group C": (26, 4),
        "Group D": (12, 2),
    }

    categories1, values1 = [], []
    for group, (mean, sd) in cohort1_specs.items():
        categories1.extend([group] * (2 * n_per_cell))
        values1.extend(rng.normal(mean, sd, 2 * n_per_cell))

    categories2, values2 = [], []
    for group, (mean, sd) in cohort2_specs.items():
        categories2.extend([group] * (2 * n_per_cell))
        values2.extend(rng.normal(mean, sd, 2 * n_per_cell))

    plt.figure()
    ax = plt.gca()
    make_boxplot(categories1, values1, ax)
    make_boxplot(categories2, values2, ax)  # triggers the warning

    plt.show()
