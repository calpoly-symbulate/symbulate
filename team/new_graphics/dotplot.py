"""Draft of the dot plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_dotplot()`` below when the user asks for a dot plot or the
default plot lookup table selects one (1D data with n <= DOTPLOT_MAX_N;
larger samples default to impulse / histogram instead, and data whose
values are all distinct will go to a rug plot -- not drafted yet).

Visual target: the classic stacked dot plot. Every observation is one
dot, drawn at its exact value on the x-axis; identical values stack on
top of one another, the first dot in each stack sits directly on the
number line, and stacked dots touch with no gap. Values are never
binned -- the dot plot is for discrete data, and the lookup table
routes continuous data elsewhere. The y-axis stays honest: it reads
"Count" by default, or "Relative frequency" with ``normalize=True``.

Geometry: dots are drawn with ``ax.scatter`` at
(value, (level - 1/2) * unit), where unit is 1 count (or 1/n when
normalized). The y-limits are chosen so one stack unit on screen
equals one dot diameter, measured through ``ax.transData`` -- that
equality is what makes the dots touch. The dot diameter fills the gap
between neighboring stacks when they fit, capped at
DOTPLOT_MAX_DOT_SIZE points; taller stacks shrink the dots instead of
overflowing the axes. Dot sizes are recomputed whenever the rendered
size of the axes changes (figure resize, tight_layout).

Colors are chosen automatically from the style sheet's color cycle
(``symbulate.mplstyle``: sky blue first, then the other Okabe-Ito
hues, no black) -- the same cycle every plot type draws from, so a
dot plot's first batch matches every other plot type's first series
and students never have to pass or memorize colors.

Overlays: a second ``make_dotplot`` call on the same axes re-stacks
everything jointly and dodges each batch of values side by side around
the shared values. Light tile-style boundary lines appear halfway
between neighboring stacks (at 1.5, 2.5, ... for integer data) so
every dot between 1.5 and 2.5 clearly belongs to 2.

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
# can produce: single or overlaid, counts or relative frequencies.
DOTPLOT_TITLE = "Dot Plot"
DOTPLOT_AXIS_LABEL_SIZE = 12
DOTPLOT_TICK_LABEL_SIZE = 10
DOTPLOT_LEGEND_LOC = "upper right"
DOTPLOT_LEGEND_MARKER_SIZE = 8

# Stacked dots never grow past this diameter, in points, no matter how
# short the stacks are. (The y-axis extends past the tallest stack as
# needed to keep the dots touching at this size.)
DOTPLOT_MAX_DOT_SIZE = 12

# Default-plot lookup rule (Task 1A): the dot plot is the default for
# 1D data only when n <= DOTPLOT_MAX_N.
DOTPLOT_MAX_N = 40

# Vertical headroom above the tallest stack (multiplier on its height).
DOTPLOT_STACK_HEADROOM = 1.05

# Tile-style boundary lines halfway between neighboring stacks, shown
# only when two or more dot plots share the axes.
DOTPLOT_BOUNDARY_LINE_COLOR = "#b0b0b0"
DOTPLOT_BOUNDARY_LINE_WIDTH = 0.8
DOTPLOT_BOUNDARY_LINE_ALPHA = 0.6


def make_dotplot(values, ax, color=None, normalize=False, label=None):
    """Draw a stacked dot plot of simulated values on the given axes.

    Every observation is one dot, drawn at its exact value on the
    x-axis. Identical values stack on top of one another, the first
    dot in each stack sits directly on the number line, and stacked
    dots touch. Values are never binned -- the dot plot is meant for
    discrete data.

    Dot plots overlay naturally: a second call on the same axes draws
    each batch side by side around the shared values, in different
    colors, with light vertical boundary lines halfway between
    neighboring stacks (at 1.5, 2.5, ... for integer data) so it stays
    clear which value each dot belongs to. A legend appears
    automatically in the top right once two or more batches share the
    axes.

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
    _restack(state)
    _relayout(ax)
    _decorate(ax, state)
    return dots


def _next_dot_color(ax):
    """Return the next automatic dot color for these axes.

    Mirrors get_next_color() in symbulate/plot.py: walks the active
    style sheet's color cycle (symbulate.mplstyle leads with sky blue,
    then orange, ...), stored per axes so overlays advance it. Reading
    the cycle from the style sheet -- the single source of truth --
    guarantees every plot type starts from the same first color.
    (Re-implemented here rather than imported so this staging module
    does not import the symbulate package, which currently swaps the
    matplotlib style on import -- a Phase 2 cleanup item.)
    """
    if not hasattr(ax, "_dotplot_color_cycle"):
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        ax._dotplot_color_cycle = itertools.cycle(prop_cycle.by_key()["color"])
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


def _restack(state):
    """Recompute the shared stack positions and per-batch counts.

    Every distinct value across all batches gets its own stack, placed
    at that exact value -- no binning. The slot around each stack
    (used for dodging, padding, boundary lines, and the dot-size cap)
    is the smallest gap between neighboring values.
    """
    all_values = np.concatenate([s["values"] for s in state["series"]])
    positions = np.unique(all_values)
    if len(positions) > 1:
        spacing = np.diff(positions).min()
    else:
        spacing = 1.0
    state["positions"] = positions
    state["spacing"] = spacing
    for s in state["series"]:
        s["counts"] = np.array([np.sum(s["values"] == p) for p in positions], dtype=int)


def _init_state(ax, normalize):
    """Set up per-axes dot plot state, styling, and resize handling."""
    state = {
        "series": [],
        "normalize": normalize,
        "boundary_lines": [],
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


def _relayout(ax):
    """Position and size every dot from the current stacks and axes size."""
    state = getattr(ax, "_dotplot_state", None)
    if state is None or not state["series"]:
        return
    positions = state["positions"]
    spacing = state["spacing"]
    n_series = len(state["series"])
    # Each batch gets its own lane inside the slot around each value.
    lane_width = spacing / n_series

    # x padding: one slot of air beyond the outermost stacks.
    ax.set_xlim(positions[0] - spacing, positions[-1] + spacing)

    if state["normalize"]:
        units = [1.0 / s["n"] for s in state["series"]]
    else:
        units = [1.0] * n_series

    # Pixel measurements via the axes transforms (valid before any
    # draw). The 1-px floor guards against two nearly-identical values
    # driving the lane width -- and with it the dot size -- to zero.
    lane_width_px = max(_x_span_px(ax, lane_width), 1.0)
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
        for position, count in zip(positions, series["counts"]):
            xs.extend(np.full(count, position + offset))
            ys.extend((np.arange(1, count + 1) - 0.5) * unit)
        series["dots"].set_offsets(np.column_stack([xs, ys]))
        diameter_px = _y_span_px(ax, unit)
        # scatter sizes are marker areas in points^2 (diameter squared).
        size = (diameter_px * points_per_px) ** 2
        series["dots"].set_sizes(np.full(len(xs), size))

    if np.all(positions == np.round(positions)):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if not state["normalize"]:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _redraw_boundary_lines(ax, state)
    state["last_size_px"] = _axes_size_px(ax)


def _redraw_boundary_lines(ax, state):
    """Draw tile-style boundary lines between stacks for overlays.

    With dodged (side-by-side) batches, dots no longer sit exactly on
    their value, so boundaries halfway between neighboring stacks (at
    1.5, 2.5, ... for integer data) make it clear that every dot
    between 1.5 and 2.5 belongs to 2. A single batch stays
    boundary-free.
    """
    for line in state["boundary_lines"]:
        line.remove()
    state["boundary_lines"] = []
    if len(state["series"]) < 2:
        return
    positions = state["positions"]
    half = state["spacing"] / 2.0
    midpoints = (positions[:-1] + positions[1:]) / 2.0
    edges = np.concatenate([[positions[0] - half], midpoints, [positions[-1] + half]])
    for edge in edges:
        state["boundary_lines"].append(
            ax.axvline(
                edge,
                color=DOTPLOT_BOUNDARY_LINE_COLOR,
                linewidth=DOTPLOT_BOUNDARY_LINE_WIDTH,
                alpha=DOTPLOT_BOUNDARY_LINE_ALPHA,
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

    # Figure 2: overlay -- dodged lanes plus the tile-style boundary
    # lines at 1.5, 2.5, ... The legend defaults to "Variable 1",
    # "Variable 2".
    plt.figure()
    ax = plt.gca()
    make_dotplot(rng.integers(1, 7, 30), ax)
    make_dotplot(rng.integers(1, 7, 30), ax)

    # Figure 3: normalize=True -- same picture, relative-frequency axis.
    plt.figure()
    ax = plt.gca()
    make_dotplot(rng.integers(1, 7, 30), ax, normalize=True)

    plt.show()
