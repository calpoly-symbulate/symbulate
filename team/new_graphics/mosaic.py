"""Draft of the mosaic plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_mosaic()`` below for 2D discrete-x-discrete data
(``type="mosaic"``), per ``DECISIONS.md``'s Default Plot Lookup Table,
where mosaic is listed as an alternative to ``tile`` for that
configuration. There is no mosaic plot in the package today -- this is
a new plot type, not a replacement of existing code.

Visual target: a grid of stacked rectangles, one per (x, y) pair. Column
widths are proportional to each x category's marginal frequency; within
a column, segment heights are proportional to that column's conditional
frequency of each y category. Rectangle *area* therefore encodes the
joint frequency of that (x, y) pair -- the thing a mosaic plot can show
that a tile plot (which encodes frequency with color instead) cannot: at
a glance, whether y's distribution changes across x (an association) or
stays the same shape in every column (independence).

Where tile and hist2d/density2d use the sequential colormap (viridis) to
encode a *magnitude*, mosaic uses the categorical palette (Okabe-Ito, from
``symbulate.mplstyle``) to encode *which y category* a segment is -- one
color per y value, consistent across every column, with a legend. Small
gaps separate columns (and the segments within each column) so the plot
reads as a mosaic of distinct tiles rather than one solid stacked bar.

Meant for two discrete-ish variables (the same configuration ``tile``
targets); continuous data is not binned here -- every distinct value
becomes its own column or segment, so continuous input should still go
through ``hist2d`` / ``density2d`` instead.

Overlay category (see CLAUDE.md, "Overlay Policy"): **readability
warning** -- like ``tile`` and ``hist2d``, a mosaic plot fills the whole
axes, so a second call cannot share space with the first. It still draws
(covering the first one) and prints a warning, rather than erroring.

Run this file directly to render the prototype:

    python team/new_graphics/mosaic.py
"""

import warnings

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express per-plot-type values, so these cannot live in
# symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py. The
# categorical color cycle itself is NOT set here -- it comes from
# axes.prop_cycle (Okabe-Ito) in symbulate.mplstyle.
MOSAIC_COLUMN_GAP = 0.003  # fixed gap, in [0, 1] axis-fraction units,
# between adjacent columns (one column per distinct x value) -- a real
# numeric gap (background shows through), not a border, so it stays thin
MOSAIC_ROW_GAP = 0.002  # fixed gap between the stacked y-segments within
# one column -- slightly smaller than the column gap, which is the more
# prominent visual break in a mosaic plot
# No edgecolor/linewidth default: the gaps above already separate every
# cell (like tile's seamless imshow grid has none either, just no gap at
# all), so an additional border would double the whitespace between
# cells instead of cleaning it up. A caller can still pass edgecolor= /
# linewidth= through **kwargs to add one.
MOSAIC_LEGEND_LOC = "center left"
MOSAIC_LEGEND_BBOX = (1.02, 0.5)  # legend sits outside the axes, to the
# right -- an in-plot legend (e.g. "upper right") would sit on top of
# real data, since the mosaic fills the entire [0, 1] x [0, 1] canvas
# Okabe-Ito only has 7 hues, so a y with more than 7 categories must
# reuse colors. A repeated color alone would give two legend entries an
# identical swatch with no way to tell them apart in the plot -- so each
# additional pass through the palette also adds a hatch pattern, layered
# on top of the (reused) color. hatch.color is black regardless of the
# patch's own edgecolor (which stays unset -- see above), so the pattern
# stays visible without reintroducing a border around every cell.
MOSAIC_HATCH_PATTERNS = ["", "//", "xx", "..", "oo"]
MOSAIC_MIN_LABEL_HEIGHT = 0.04  # a cell's count/percentage label is only
# drawn if the cell is at least this tall...
MOSAIC_MIN_LABEL_WIDTH = 0.05  # ...and at least this wide (both in
# [0, 1] axis-fraction units), so labels never crowd tiny cells
MOSAIC_LABEL_FONT_SIZE = 9
MOSAIC_LABEL_STROKE_WIDTH = 2  # white outline behind the label text so
# it stays readable against every Okabe-Ito color, including the darker
# blue
MOSAIC_LABEL_DECIMALS = 1  # decimal places for percentage labels (e.g.
# "18.2%"); raw counts (normalize=False) are always whole numbers

MOSAIC_OVERLAY_WARNING = (
    "Warning: you drew a second mosaic plot on the same axes. A mosaic "
    "plot re-partitions the entire plot area for its own data, so the "
    "second call covers up the first one. Try two separate plots (e.g. "
    "subplots) instead."
)


def _mosaic_spans(weights, gap):
    """Left edges and widths for a row of segments with fixed gaps.

    Parameters
    ----------
    weights : array-like of float
        Nonnegative, proportional sizes for each segment (need not sum
        to 1 -- they are rescaled here).
    gap : float
        Fixed gap, in the same [0, 1] units as the returned spans,
        reserved between each pair of adjacent segments.

    Returns
    -------
    tuple of numpy.ndarray
        ``(starts, widths)``. Segment ``i`` spans ``[starts[i],
        starts[i] + widths[i]]``; consecutive segments are separated by
        exactly ``gap``, and the whole row spans exactly ``[0, 1]``. A
        segment with weight 0 gets width 0 (invisible), but still
        occupies its slot -- and its gap -- in the layout.
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    total_gap = gap * max(n - 1, 0)
    available = max(1.0 - total_gap, 0.0)
    total_weight = weights.sum()
    widths = weights / total_weight * available if total_weight > 0 else weights
    starts = np.concatenate([[0.0], np.cumsum(widths + gap)[:-1]])
    return starts, widths


def make_mosaic(
    x,
    y,
    ax,
    normalize=True,
    annotate=True,
    legend=True,
    x_label="X",
    y_label="Y",
    **kwargs,
):
    """Draw a mosaic plot of simulated (x, y) pairs on the given axes.

    Divides the axes into one column per distinct ``x`` value, with
    column widths proportional to how often that value occurred (its
    marginal frequency). Within each column, the column is further
    divided into one segment per distinct ``y`` value, with segment
    heights proportional to ``y``'s *conditional* frequency within that
    column. Every rectangle's area is therefore proportional to the
    joint frequency of that ``(x, y)`` pair -- reading the segment
    heights across columns shows whether ``y``'s distribution changes
    with ``x`` (an association) or stays the same shape in every column
    (independence), which a same-color-scale plot like ``tile`` cannot
    show directly.

    Segments are colored by ``y`` category, one color per distinct
    value from the package's categorical palette (Okabe-Ito, from
    ``symbulate.mplstyle``), consistent across every column, with a
    legend placed outside the axes (an in-plot legend would sit on top
    of real data, since the mosaic fills the whole axes). With more than
    7 distinct ``y`` values, colors repeat -- each repeat also adds a
    hatch pattern, so two categories never look identical in the plot or
    the legend, and a warning explains why. Small gaps separate columns
    and the segments within each column so the plot reads as a mosaic of
    distinct tiles.

    Meant for two discrete-ish variables -- the same configuration
    ``tile`` targets. Continuous data is not binned here: every
    distinct value becomes its own column or segment, so continuous
    input should go through ``hist2d`` / ``density2d`` instead.

    A mosaic plot fills the entire axes, so a second call on the same
    axes cannot overlay naturally -- this is the "readability warning"
    category of the overlay policy (the same category as ``tile`` and
    ``hist2d``): the plot still draws, covering the first one, and
    prints a warning rather than erroring.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). This mirrors how the other 2D plot helpers
    in this project (``make_tile``, ``make_hist2d``) are called.

    Parameters
    ----------
    x : array-like
        Simulated values that determine the columns, e.g. the first
        column of ``RVResults.array``. Discrete or categorical.
    y : array-like
        Simulated values that determine the segments within each
        column, same length as ``x``. Discrete or categorical.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    normalize : bool, default True
        Only affects the in-cell labels (the geometry is always
        proportion-based, since column widths and segment heights must
        sum to 1 by construction). If True, labels show each cell's
        joint relative frequency as a percentage (e.g. "18.2%"). If
        False, labels show the raw count instead.
    annotate : bool, default True
        If True, print each cell's count or percentage (see
        ``normalize``) inside the cell, but only when the cell is large
        enough to hold it legibly (see ``MOSAIC_MIN_LABEL_HEIGHT`` /
        ``MOSAIC_MIN_LABEL_WIDTH``). If False, no in-cell labels.
    legend : bool, default True
        If True, add a legend outside the right edge of the axes
        naming each ``y`` category's color.
    x_label : str, default "X"
        Label for the x-axis.
    y_label : str, default "Y"
        Title for the legend naming the ``y`` categories. (The y-axis
        itself has no numeric meaning shared across columns -- each
        column's segments independently span 0 to 1 -- so it is
        hidden rather than labeled.)
    **kwargs
        Additional keyword arguments passed to every ``ax.bar`` call
        (one per ``y`` category). For example ``linewidth=`` to
        override the default cell border width.

    Returns
    -------
    dict
        Maps each distinct ``y`` value to the ``matplotlib.container.
        BarContainer`` of its segments (one bar per column), so the
        caller can inspect or further style a specific category.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` are not the same length.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.choice(["A", "B", "C"], size=1000, p=[0.5, 0.3, 0.2])
    >>> y = rng.choice(["yes", "no"], size=1000)
    >>> ax = plt.gca()
    >>> make_mosaic(x, y, ax)  # doctest: +SKIP
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError(
            "x and y must be the same length to pair them up, but x has "
            f"{len(x)} values and y has {len(y)}. Check that both come "
            "from the same simulation."
        )
    n = len(x)

    # One cell per distinct (x, y) pair, no binning -- the same
    # dense-joint-count approach make_tile uses for two discrete axes.
    x_labels = np.unique(x)
    y_labels = np.unique(y)
    x_idx = np.searchsorted(x_labels, x)
    y_idx = np.searchsorted(y_labels, y)
    joint = np.zeros((len(x_labels), len(y_labels)))
    np.add.at(joint, (x_idx, y_idx), 1)

    # Columns: width proportional to each x value's marginal count.
    x_counts = joint.sum(axis=1)
    x_starts, x_widths = _mosaic_spans(x_counts, MOSAIC_COLUMN_GAP)
    x_positions = x_starts + x_widths / 2

    # Rows within each column: height proportional to that column's
    # conditional frequency of each y value. Computed independently per
    # column, since each column's own total (not the grand total) is
    # what its segment heights divide.
    row_starts = np.zeros((len(x_labels), len(y_labels)))
    row_heights = np.zeros((len(x_labels), len(y_labels)))
    for i in range(len(x_labels)):
        starts_i, heights_i = _mosaic_spans(joint[i, :], MOSAIC_ROW_GAP)
        row_starts[i, :] = starts_i
        row_heights[i, :] = heights_i

    # One color per y category, read directly from the active style
    # sheet's categorical cycle (Okabe-Ito) rather than advancing the
    # shared per-axes cycle get_next_color(ax) uses elsewhere: every
    # mosaic plot should map "first y category" to the same color, not
    # whatever color happens to be next after unrelated prior plots on
    # this axes.
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # More y categories than palette colors: the color alone cycles back
    # to "first y category"'s color, which would give two legend entries
    # an identical swatch with no way to tell them apart. Each full pass
    # through the palette also layers on a hatch pattern (see
    # MOSAIC_HATCH_PATTERNS), so "repeats the sky-blue of category 1" and
    # "repeats it with a hatch" read as clearly different in both the
    # cells and the legend.
    if len(y_labels) > len(palette):
        warnings.warn(
            f"This mosaic plot has {len(y_labels)} distinct y-values, more "
            f"than the {len(palette)} colors in the categorical palette, so "
            "some categories repeat a color (distinguished with a hatch "
            "pattern instead). A mosaic plot is hard to read with this many "
            "categories -- consider a plot type that doesn't rely on "
            "category color, like a tile plot.",
            UserWarning,
            stacklevel=2,
        )

    # Count the mosaic plots drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the color
    # cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_mosaic_count", 0)
    ax._mosaic_count = n_prior + 1

    bars = {}
    for j, y_label_value in enumerate(y_labels):
        heights = row_heights[:, j]
        bottoms = row_starts[:, j]
        color = palette[j % len(palette)]
        hatch = MOSAIC_HATCH_PATTERNS[(j // len(palette)) % len(MOSAIC_HATCH_PATTERNS)]
        bars[y_label_value] = ax.bar(
            x_positions,
            heights,
            width=x_widths,
            bottom=bottoms,
            color=color,
            hatch=hatch,
            label=str(y_label_value),
            **kwargs,
        )

    if annotate:
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                if (
                    row_heights[i, j] < MOSAIC_MIN_LABEL_HEIGHT
                    or x_widths[i] < MOSAIC_MIN_LABEL_WIDTH
                ):
                    continue
                if normalize:
                    text = f"{100 * joint[i, j] / n:.{MOSAIC_LABEL_DECIMALS}f}%"
                else:
                    text = f"{int(round(joint[i, j]))}"
                ax.text(
                    x_positions[i],
                    row_starts[i, j] + row_heights[i, j] / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=MOSAIC_LABEL_FONT_SIZE,
                    color="black",
                    path_effects=[
                        pe.withStroke(
                            linewidth=MOSAIC_LABEL_STROKE_WIDTH, foreground="white"
                        )
                    ],
                )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(v) for v in x_labels])
    ax.set_xlabel(x_label)
    # The y-axis has no numeric meaning shared across columns -- each
    # column's segments independently span 0 to 1 -- so it is hidden
    # entirely rather than labeled, the same way a standalone rug plot
    # hides its uninformative axis.
    ax.yaxis.set_visible(False)
    ax.set_title("Mosaic Plot")
    # A filled plot covers the whole axes, so the reference grid has
    # nothing to sit on -- turn it off rather than let fragments show
    # at the edges (the same reasoning make_tile / make_hist2d use).
    ax.grid(False)

    if legend:
        ax.legend(
            loc=MOSAIC_LEGEND_LOC,
            bbox_to_anchor=MOSAIC_LEGEND_BBOX,
            title=y_label,
        )

    if ax._mosaic_count > 1:
        print(MOSAIC_OVERLAY_WARNING)

    return bars


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)

    # Figure 1: two discrete numeric variables, unequal category sizes --
    # column widths clearly differ.
    x = rng.choice([1, 2, 3, 4], size=4000, p=[0.5, 0.3, 0.15, 0.05])
    y = rng.poisson(2, 4000)
    ax = plt.gca()
    make_mosaic(x, y, ax, x_label="X", y_label="Y")

    # Figure 2: independence -- y's conditional distribution is the same
    # shape in every column, so the segment heights line up across columns.
    plt.figure()
    x_indep = rng.choice(["A", "B", "C"], size=4000, p=[0.5, 0.3, 0.2])
    y_indep = rng.choice(["yes", "no"], size=4000, p=[0.7, 0.3])
    ax = plt.gca()
    make_mosaic(x_indep, y_indep, ax, x_label="Group", y_label="Outcome")

    # Figure 3: association -- y's conditional distribution changes
    # noticeably across columns (the mosaic's core diagnostic use).
    plt.figure()
    group = rng.choice(["A", "B", "C"], size=4000, p=[0.5, 0.3, 0.2])
    p_yes = np.select([group == "A", group == "B", group == "C"], [0.85, 0.5, 0.15])
    outcome = np.where(rng.uniform(size=4000) < p_yes, "yes", "no")
    ax = plt.gca()
    make_mosaic(group, outcome, ax, x_label="Group", y_label="Outcome")

    # Figure 4: raw counts instead of percentages in the cell labels.
    plt.figure()
    ax = plt.gca()
    make_mosaic(group, outcome, ax, normalize=False, x_label="Group", y_label="Outcome")

    # Figure 5: overlay -- second mosaic call prints the readability
    # warning and covers the first.
    plt.figure()
    ax = plt.gca()
    make_mosaic(x_indep, y_indep, ax)
    make_mosaic(group, outcome, ax)

    plt.show()
