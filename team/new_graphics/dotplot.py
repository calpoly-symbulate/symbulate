"""Draft of the dot plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_dotplot()`` below when the user asks for a dot plot or the
default plot lookup table selects one (1D data with n <= DOTPLOT_MAX_N;
larger samples default to impulse / histogram instead, and data whose
values are all distinct will go to a rug plot -- not drafted yet).

Visual target: the classic stacked dot plot. Every observation is one
dot; dots in the same bin stack on top of one another, the first dot
in each stack sits directly on the number line, and stacked dots touch
with no gap. Integer data gets one stack per integer; continuous data
is cut into equal-width bins first. The y-axis stays honest: it reads
"Count" by default, or "Relative frequency" with ``normalize=True``.

Geometry: dots are drawn with ``ax.scatter`` at
(bin center, (level - 1/2) * unit), where unit is 1 count (or 1/n when
normalized). The y-limits are chosen so one stack unit on screen
equals one dot diameter, measured through ``ax.transData`` -- that
equality is what makes the dots touch. The dot diameter equals the bin
width when the stacks fit, capped at DOTPLOT_MAX_DOT_SIZE points;
taller stacks shrink the dots instead of overflowing the axes. Dot
sizes are recomputed whenever the rendered size of the axes changes
(figure resize, tight_layout).

Colors are chosen automatically -- sky blue for the first batch on an
axes, then the remaining Okabe-Ito hues (no black) for overlays -- so
students never have to pass or memorize colors.

Overlays: a second ``make_dotplot`` call on the same axes re-bins
everything jointly and dodges each batch of values side by side inside
the shared bins. For integer data, light tile-style boundary lines
appear at the half-integers (1.5, 2.5, ...) so every dot between 1.5
and 2.5 clearly belongs to 2.

Spines, fonts, and figure size come from
``symbulate/symbulate.mplstyle``, except where the dot plot
deliberately differs: the reference grid is turned off (gridlines add
clutter under stacked dots) and axis labels are bumped to 12 pt. The
per-plot-type values are the named constants below, which migrate to
the top of ``symbulate/plot.py`` at integration time.

Run this file directly to render the prototypes:

    python team/new_graphics/dotplot.py
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
DOTPLOT_ALPHA = 1.0
DOTPLOT_XLABEL = "Value"
# One generic title that stays accurate for every dot plot this module
# can produce: single or overlaid, counts or relative frequencies,
# integer or binned continuous data.
DOTPLOT_TITLE = "Dot Plot"
DOTPLOT_AXIS_LABEL_SIZE = 12
DOTPLOT_TICK_LABEL_SIZE = 10
DOTPLOT_LEGEND_LOC = "upper right"
DOTPLOT_LEGEND_MARKER_SIZE = 8

# Automatic dot colors: the Okabe-Ito palette (no black), reordered so
# sky blue is the single-batch default and overlays cycle through the
# rest (yellow last -- it is the weakest hue on a white background).
# Note this differs from the symbulate.mplstyle cycle, which starts at
# orange; at integration time the team should pick one order for both.
DOTPLOT_COLOR_CYCLE = [
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
]

# Stacked dots never grow past this diameter, in points, no matter how
# short the stacks are. (The y-axis extends past the tallest stack as
# needed to keep the dots touching at this size.)
DOTPLOT_MAX_DOT_SIZE = 12

# Continuous data is cut into at most this many equal-width bins;
# small samples get proportionally chunkier bins (see _n_bins), since
# 30 bins with only 40 dots would leave nearly every stack 1 dot tall.
DOTPLOT_MAX_BINS = 30

# Integer data spanning more than this many integers is binned like
# continuous data. Stand-in for classify_data's N_UNIQUE_THRESHOLD
# until Phase 2 lands.
DOTPLOT_MAX_INTEGER_BINS = 100

# Default-plot lookup rule (Task 1A): the dot plot is the default for
# 1D data only when n <= DOTPLOT_MAX_N.
DOTPLOT_MAX_N = 40

# Vertical headroom above the tallest stack (multiplier on its height).
DOTPLOT_STACK_HEADROOM = 1.05

# Tile-style bin boundary lines, shown only when two or more dot plots
# of integer data share the axes.
DOTPLOT_BIN_LINE_COLOR = "#b0b0b0"
DOTPLOT_BIN_LINE_WIDTH = 0.8
DOTPLOT_BIN_LINE_ALPHA = 0.6


def make_dotplot(values, ax, color=None, normalize=False, label=None):
    """Draw a stacked dot plot of simulated values on the given axes.

    Every observation is one dot. Dots with the same (binned) value
    stack on top of one another, the first dot in each stack sits
    directly on the number line, and stacked dots touch. Integer data
    gets one stack per integer; continuous data is first cut into
    equal-width bins.

    Dot plots overlay naturally: a second call on the same axes re-bins
    both batches of values together and draws them side by side inside
    each bin, in different colors. With integer data, light vertical
    boundary lines appear at 1.5, 2.5, ... so it stays clear which
    integer each dot belongs to. A legend appears automatically in the
    top right once two or more batches share the axes.

    Colors are chosen automatically: sky blue for the first batch, then
    the other Okabe-Ito hues in turn for overlays -- nobody has to
    memorize the palette. Pass ``color=`` only to override.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working), as with the other plot helpers in
    ``symbulate/plot.py`` (``make_tile``, ``make_density2D``).

    Parameters
    ----------
    values : array-like
        The simulated values to plot, e.g. ``RVResults.array``. Must
        be numeric.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color, optional
        Dot color. If omitted (recommended), the first batch on the
        axes is sky blue and each overlay automatically gets the next
        Okabe-Ito hue.
    normalize : bool, default False
        If False, the y-axis shows counts. If True, it shows relative
        frequencies (each count divided by that batch's number of
        values), so batches of different sizes can be compared.
    label : str, optional
        Name for this batch of values in the legend. Defaults to
        "Variable k", where k counts the dot plots drawn on these axes
        so far.

    Returns
    -------
    matplotlib.collections.PathCollection
        The dots, as returned by ``ax.scatter``, so the caller can
        inspect or further style them.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> values = np.random.default_rng().integers(1, 7, 30)
    >>> make_dotplot(values, plt.gca())  # doctest: +SKIP
    """
    values = _clean_values(values)
    if color is None:
        color = _next_dot_color(ax)
    state = getattr(ax, "_dotplot_state", None)
    if state is None:
        state = _init_state(ax, normalize)
    if normalize != state["normalize"]:
        first = "relative frequencies" if state["normalize"] else "counts"
        print(
            "The dot plot already on these axes shows "
            + first
            + ", so the new dots are shown as "
            + first
            + " too. To switch, redraw all the dot plots in a fresh cell "
            + "with the same normalize= setting."
        )
    if label is None:
        label = "Variable {}".format(len(state["series"]) + 1)
    # Dots start empty; _relayout fills in positions and sizes once the
    # shared bins and axes limits are known.
    dots = ax.scatter([], [], color=color, alpha=DOTPLOT_ALPHA, zorder=2)
    state["series"].append(
        {
            "values": values,
            "n": len(values),
            "dots": dots,
            "label": label,
            "color": color,
        }
    )
    _rebin(state)
    _relayout(ax)
    _decorate(ax, state)
    return dots


def _next_dot_color(ax):
    """Return the next automatic dot color for these axes.

    Same per-axes pattern as get_next_color() in symbulate/plot.py,
    but walks DOTPLOT_COLOR_CYCLE (sky blue first) instead of the
    rcParams cycle.
    """
    if not hasattr(ax, "_dotplot_color_cycle"):
        ax._dotplot_color_cycle = itertools.cycle(DOTPLOT_COLOR_CYCLE)
    return next(ax._dotplot_color_cycle)


def _clean_values(values):
    """Validate the values and return them as a 1D float array."""
    arr = np.asarray(list(values))
    if arr.dtype.kind not in "iufb":
        raise TypeError(
            "A dot plot needs numbers, but these values are not numeric "
            "(for example, text like 'H' or 'T'). Try .tabulate() to "
            "count how often each value occurs instead."
        )
    arr = arr.astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(
            "There are no values to plot. Simulate some values first, "
            "for example X.sim(30).plot()."
        )
    return arr


def _looks_discrete(values):
    """Return True if every integer should get its own stack.

    Stand-in for the future classify_data() (see CLAUDE.md) until
    Phase 2 lands -- the dot plot only needs the discrete-ish half of
    that decision here.
    """
    if not np.all(values == np.round(values)):
        return False
    return np.ptp(values) <= DOTPLOT_MAX_INTEGER_BINS


def _n_bins(n, n_series):
    """Number of equal-width bins for n continuous values.

    With overlaid batches, each bin is split into one lane per batch,
    so the bin count shrinks proportionally to keep the lanes -- and
    therefore the dots -- about as wide as in the single-batch case.
    """
    return int(min(DOTPLOT_MAX_BINS, max(5, np.ceil(2 * np.sqrt(n)) / n_series)))


def _bin_counts(values, centers, bin_width, discrete):
    """Count how many values land in each bin."""
    if discrete:
        return np.array([np.sum(np.round(values) == c) for c in centers], dtype=int)
    edges = np.append(centers - bin_width / 2.0, centers[-1] + bin_width / 2.0)
    counts, _ = np.histogram(values, bins=edges)
    return counts


def _init_state(ax, normalize):
    """Set up per-axes dot plot state, styling, and resize handling."""
    state = {
        "series": [],
        "normalize": normalize,
        "bin_lines": [],
        "last_size_px": None,
        "relayout_running": False,
    }
    ax._dotplot_state = state
    # Gridlines add clutter under stacked dots -- the dot plot is the
    # one plot type that turns the style sheet's reference grid off.
    ax.grid(False)
    ax.set_xlabel(DOTPLOT_XLABEL)
    ax.tick_params(labelsize=DOTPLOT_TICK_LABEL_SIZE)
    # Dot sizes depend on the rendered size of the axes, so redo the
    # geometry whenever something changes it (figure resize,
    # tight_layout).
    ax.figure.canvas.mpl_connect("resize_event", lambda event: _on_canvas_change(ax))
    ax.figure.canvas.mpl_connect("draw_event", lambda event: _on_canvas_change(ax))
    return state


def _rebin(state):
    """Recompute the shared bins and each batch's per-bin counts."""
    batches = [s["values"] for s in state["series"]]
    discrete = all(_looks_discrete(v) for v in batches)
    lo = min(v.min() for v in batches)
    hi = max(v.max() for v in batches)
    if discrete:
        centers = np.arange(round(lo), round(hi) + 1, dtype=float)
        bin_width = 1.0
    elif hi == lo:
        centers = np.array([lo])
        bin_width = 1.0
    else:
        n_total = sum(s["n"] for s in state["series"])
        edges = np.linspace(lo, hi, _n_bins(n_total, len(state["series"])) + 1)
        bin_width = edges[1] - edges[0]
        centers = (edges[:-1] + edges[1:]) / 2.0
    state["discrete"] = discrete
    state["centers"] = centers
    state["bin_width"] = bin_width
    for s in state["series"]:
        s["counts"] = _bin_counts(s["values"], centers, bin_width, discrete)


def _relayout(ax):
    """Position and size every dot from the current bins and axes size."""
    state = getattr(ax, "_dotplot_state", None)
    if state is None or not state["series"]:
        return
    centers = state["centers"]
    bin_width = state["bin_width"]
    n_series = len(state["series"])
    # Each batch gets its own lane inside the shared bin (dodge).
    lane_width = bin_width / n_series

    # x padding: half a bin width of air beyond the outermost bin edges.
    ax.set_xlim(centers[0] - bin_width, centers[-1] + bin_width)

    if state["normalize"]:
        units = [1.0 / s["n"] for s in state["series"]]
    else:
        units = [1.0] * n_series

    # Pixel measurements via the axes transforms (valid before any draw).
    lane_width_px = _x_span_px(ax, lane_width)
    height_px = _axes_size_px(ax)[1]

    # Choose the y range so one stack unit on screen is never taller
    # than one lane is wide, nor than the DOTPLOT_MAX_DOT_SIZE cap.
    # That pixel height becomes the dot diameter, so dots stack
    # touching and never spill into the next lane, short stacks cannot
    # inflate the dots past the cap, and taller stacks shrink the dots
    # instead of overflowing the axes.
    max_dot_px = DOTPLOT_MAX_DOT_SIZE * ax.figure.dpi / 72.0
    tallest = max(s["counts"].max() * u for s, u in zip(state["series"], units))
    y_max = max(
        tallest * DOTPLOT_STACK_HEADROOM,
        max(units) * height_px / lane_width_px,
        max(units) * height_px / max_dot_px,
    )
    ax.set_ylim(0, y_max)

    points_per_px = 72.0 / ax.figure.dpi
    for i, (series, unit) in enumerate(zip(state["series"], units)):
        offset = (i - (n_series - 1) / 2.0) * lane_width
        xs = []
        ys = []
        for center, count in zip(centers, series["counts"]):
            xs.extend(np.full(count, center + offset))
            ys.extend((np.arange(1, count + 1) - 0.5) * unit)
        series["dots"].set_offsets(np.column_stack([xs, ys]))
        diameter_px = _y_span_px(ax, unit)
        # scatter sizes are marker areas in points^2 (diameter squared).
        size = (diameter_px * points_per_px) ** 2
        series["dots"].set_sizes(np.full(len(xs), size))

    if state["discrete"]:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if not state["normalize"]:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _redraw_bin_lines(ax, state)
    state["last_size_px"] = _axes_size_px(ax)


def _redraw_bin_lines(ax, state):
    """Draw tile-style bin boundary lines for overlaid integer data.

    With dodged (side-by-side) batches, dots no longer sit exactly on
    their integer, so boundaries at the half-integers (1.5, 2.5, ...)
    make it clear that every dot between 1.5 and 2.5 belongs to 2.
    A single batch stays boundary-free.
    """
    for line in state["bin_lines"]:
        line.remove()
    state["bin_lines"] = []
    if not (state["discrete"] and len(state["series"]) >= 2):
        return
    half = state["bin_width"] / 2.0
    edges = np.append(state["centers"] - half, state["centers"][-1] + half)
    for edge in edges:
        state["bin_lines"].append(
            ax.axvline(
                edge,
                color=DOTPLOT_BIN_LINE_COLOR,
                linewidth=DOTPLOT_BIN_LINE_WIDTH,
                alpha=DOTPLOT_BIN_LINE_ALPHA,
                zorder=1,
            )
        )


def _decorate(ax, state):
    """Apply the title, labels, fonts, and (for overlays) the legend."""
    ax.set_title(DOTPLOT_TITLE)
    ax.set_ylabel("Relative frequency" if state["normalize"] else "Count")
    ax.xaxis.label.set_size(DOTPLOT_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(DOTPLOT_AXIS_LABEL_SIZE)
    # A legend only helps once there is more than one batch to tell
    # apart. Handles are built by hand so the legend dots keep a
    # readable fixed size instead of the data-driven dot diameter.
    if len(state["series"]) > 1:
        handles = [
            Line2D(
                [],
                [],
                linestyle="",
                marker="o",
                color=s["color"],
                markersize=DOTPLOT_LEGEND_MARKER_SIZE,
                label=s["label"],
            )
            for s in state["series"]
        ]
        ax.legend(handles=handles, loc=DOTPLOT_LEGEND_LOC)


def _axes_size_px(ax):
    """Return the rendered (width, height) of the axes in pixels."""
    (x0, y0), (x1, y1) = ax.transAxes.transform([(0.0, 0.0), (1.0, 1.0)])
    return x1 - x0, y1 - y0


def _x_span_px(ax, dx):
    """Return how many pixels wide dx data units are."""
    (x0, _), (x1, _) = ax.transData.transform([(0.0, 0.0), (dx, 0.0)])
    return x1 - x0


def _y_span_px(ax, dy):
    """Return how many pixels tall dy data units are."""
    (_, y0), (_, y1) = ax.transData.transform([(0.0, 0.0), (0.0, dy)])
    return y1 - y0


def _on_canvas_change(ax):
    """Redo the dot geometry if the axes' rendered size has changed."""
    state = getattr(ax, "_dotplot_state", None)
    if state is None or state["relayout_running"] or not state["series"]:
        return
    size = _axes_size_px(ax)
    last = state["last_size_px"]
    if (
        last is not None
        and abs(size[0] - last[0]) < 1.0
        and abs(size[1] - last[1]) < 1.0
    ):
        return
    state["relayout_running"] = True
    try:
        _relayout(ax)
        ax.figure.canvas.draw_idle()
    finally:
        state["relayout_running"] = False


if __name__ == "__main__":
    # Render the prototype scenarios with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)

    # Colors are never passed below -- the first batch on an axes is
    # sky blue and overlays cycle through the other Okabe-Ito hues.

    # Figure 1: 30 rolls of a fair die -- one stack per face, dots
    # touching, bottom dot sitting on the number line.
    ax = plt.gca()
    make_dotplot(rng.integers(1, 7, 30), ax)

    # Figure 2: 40 draws from Normal(0, 1) -- continuous, binned.
    plt.figure()
    ax = plt.gca()
    make_dotplot(rng.normal(0, 1, 40), ax)

    # Figure 3: overlay of integer data -- dodged lanes plus the
    # tile-style boundary lines at 1.5, 2.5, ... The legend defaults
    # to "Variable 1", "Variable 2".
    plt.figure()
    ax = plt.gca()
    make_dotplot(rng.integers(1, 7, 30), ax)
    make_dotplot(rng.integers(1, 7, 30), ax)

    # Figure 4: overlay of continuous data, with custom labels.
    plt.figure()
    ax = plt.gca()
    make_dotplot(rng.normal(-0.5, 1, 30), ax, label="Group A")
    make_dotplot(rng.normal(0.8, 1, 30), ax, label="Group B")

    # Figure 5: normalize=True -- same picture, relative-frequency axis.
    plt.figure()
    ax = plt.gca()
    make_dotplot(rng.integers(1, 7, 30), ax, normalize=True)

    plt.show()
