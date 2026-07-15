import itertools
import os
import warnings

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline

# Apply the package style: Okabe-Ito categorical palette (sky blue
# first), viridis sequential colormap, and the shared figure/spine/grid
# defaults. Every value that applies the same way across all plot types
# lives in this file; per-plot-type values are the named constants
# below. See DECISIONS.md, "Decision: Visual Style Guide" and
# "Decision: .mplstyle Standards".
plt.style.use(os.path.join(os.path.dirname(__file__), "symbulate.mplstyle"))

rng = np.random.default_rng()

# Data-classification thresholds for the default plot lookup -- see
# classify_data() and DECISIONS.md, "Decision: Data Classification".
N_UNIQUE_THRESHOLD = 40
N_SMALL_THRESHOLD = 100

figure = plt.figure

xlabel = plt.xlabel
ylabel = plt.ylabel

xlim = plt.xlim
ylim = plt.ylim


def get_next_color(axes):
    if not hasattr(axes, "_color_cycle"):
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        axes._color_cycle = itertools.cycle(prop_cycle.by_key()["color"])
    return next(axes._color_cycle)


# Per-plot-type constants. matplotlib rcParams are global and can't
# express "different line width (or alpha) for different plot types"
# (see DECISIONS.md, "Decision: .mplstyle Standards"), so these live
# here instead of symbulate.mplstyle.

# Impulse plot.
IMPULSE_LINEWIDTH = 2.2
IMPULSE_MARKER = "o"
IMPULSE_MARKER_SIZE = 60
IMPULSE_ALPHA = 1.0
IMPULSE_LEGEND_LOC = "upper right"
# Fraction of the smallest gap between values used to separate adjacent
# series' stems when several impulse plots share an axis, so the group
# straddles each value symmetrically instead of overlapping.
IMPULSE_SERIES_OFFSET = 0.35

# True-distribution overlay: a smooth curve through the exact pmf/pdf
# values with no markers, so it reads as a reference curve (like a
# density curve over a histogram) rather than another simulated series.
TRUE_DIST_LINEWIDTH = 1.8
TRUE_DIST_LINESTYLE = "-"
TRUE_DIST_CURVE_POINTS = 600
TRUE_DIST_LABEL_DEFAULT = "True Distribution"

# Histogram (1D): solid bars with thin white edges so adjacent bars
# stay visually distinct. Alpha per DECISIONS.md Visual Style Guide
# (0.5 -> 0.65 -- bolder fill, histograms don't overplot).
HIST_ALPHA = 0.65
HIST_EDGECOLOR = "white"
HIST_EDGEWIDTH = 0.8
HIST_DEFAULT_BINS = 30
HIST_LEGEND_LOC = "upper right"

# Density curve (1D). Line width per DECISIONS.md Visual Style Guide
# (2.0 -> 1.8). Full opacity for a standalone curve: the Guide's
# alpha=0.15 is specifically for the density-overlaid-on-a-histogram
# composition, which doesn't exist yet.
DENSITY_LINEWIDTH = 1.8
DENSITY_ALPHA = 1.0
DENSITY_GRID_POINTS = 1000
DENSITY_LEGEND_LOC = "upper right"
# Evaluate the KDE over a quantile-based range plus padding, not raw
# min/max, so outlier-heavy distributions (e.g. Exponential) don't
# stretch the axis and squash the curve into a sliver.
DENSITY_QUANTILE_LOW = 0.001
DENSITY_QUANTILE_HIGH = 0.999
DENSITY_PADDING_FRAC = 0.1  # extra padding, as a fraction of the
# quantile-bounded span, added to each side so the curve visibly
# tapers to (near) zero instead of being cut off mid-slope

# Rug plot: one light tick per simulated value along the bottom of the
# axes; stacked values read as darker ticks through the alpha.
RUG_ALPHA = 0.5
RUG_TICK_HEIGHT = 0.04  # fraction of the axes height
RUG_LINEWIDTH = 1.0
RUG_LEGEND_LOC = "upper right"

# ECDF (empirical cumulative distribution function) plot: a
# right-continuous step curve. F(x) is the fraction of observations
# <= x (normalize=True, rising 0 -> 1), or the running count
# (normalize=False, rising 0 -> n). Line weight matches the density
# curve; a rising CDF leaves the upper-left corner empty, so the legend
# sits there.
ECDF_LINEWIDTH = 1.8
ECDF_ALPHA = 1.0
ECDF_LEGEND_LOC = "upper left"

# Segmented density plot (a "ridgeline" plot): one small kernel
# density curve ("ridge") per level of
# the discrete variable, stacked along the discrete axis with a gentle
# overlap. All ridges in one call share a single density scale, so peak
# heights are directly comparable across levels. The KDE range reuses
# the density plot's quantile constants (DENSITY_QUANTILE_LOW / HIGH /
# PADDING_FRAC) via _density_xrange, so outliers can't stretch the
# value axis.
SEGMENTED_DENSITY_FILL_ALPHA = 0.4  # translucent fill; the outline stays opaque
SEGMENTED_DENSITY_LINEWIDTH = 1.8  # ridge outlines are density curves, so this
# matches DENSITY_LINEWIDTH
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

# Dot plot: every observation is one dot at its exact value; identical
# values stack, touching, with the bottom dot on the number line.
DOTPLOT_ALPHA = 1.0
DOTPLOT_XLABEL = "Value"
# One generic title that stays accurate for every dot plot, whether
# single or overlaid.
DOTPLOT_TITLE = "Dot Plot"
DOTPLOT_AXIS_LABEL_SIZE = 12
DOTPLOT_TICK_LABEL_SIZE = 10
DOTPLOT_LEGEND_LOC = "upper right"
DOTPLOT_LEGEND_MARKER_SIZE = 8
# Stacked dots never grow past this diameter, in points, no matter how
# short the stacks are. (The y-axis extends past the tallest stack as
# needed to keep the dots touching at this size.)
DOTPLOT_MAX_DOT_SIZE = 12
# Vertical headroom above the tallest stack (multiplier on its height).
DOTPLOT_STACK_HEADROOM = 1.05
# Tile-style boundary lines halfway between neighboring stacks, shown
# only when two or more dot plots share the axes.
DOTPLOT_BOUNDARY_LINE_COLOR = "#b0b0b0"
DOTPLOT_BOUNDARY_LINE_WIDTH = 0.8
DOTPLOT_BOUNDARY_LINE_ALPHA = 0.6

# Scatter (2D). Alpha per DECISIONS.md Visual Style Guide (0.5 -> 0.25),
# low so overlapping points stay individually readable. Point style is
# filled circles -- note this overrides the Guide's *unfilled* scatter
# decision (pending decision-log update).
SCATTER_ALPHA = 0.25
SCATTER_MARKER_SIZE = 40
SCATTER_LEGEND_LOC = "upper right"
# Spiral and bins jitter modes lay coincident points out with their
# dots touching. This is how much of the value's bin (the cell between
# half-integer boundaries) a cluster may occupy before its spacing
# compresses instead of growing -- 0.8 leaves a 0.1 margin inside each
# bin edge, so a cluster never reaches the neighboring value.
SCATTER_BIN_SPREAD = 0.8
# Auto mode: jitter="auto" uses "bins" once any single (x, y) value
# holds this many points -- past the 9-dot compass template, where a
# spiral pile-up stops being cleanly countable -- and "spiral"
# otherwise.
SCATTER_AUTO_BINS_THRESHOLD = 12

# 2D histogram. The colormap itself is NOT set here -- it comes from
# image.cmap (viridis) in symbulate.mplstyle.
HIST2D_DEFAULT_BINS = 30
HIST2D_CBAR_SIZE = "5%"  # colorbar width, as a fraction of the axes
HIST2D_CBAR_PAD = 0.1  # gap between the axes and the colorbar, inches
HIST2D_CBAR_TICKS = 8  # evenly spaced colorbar ticks, including both
# endpoints (0 and the peak value); matches the density2d colorbar
HIST2D_CBAR_DECIMALS = 3  # decimal places for the density colorbar
# labels (raw counts are labeled as whole numbers instead)
HIST2D_OVERLAY_WARNING = (
    "Warning: you drew a second 2-D histogram on the same plot. The two "
    "color scales compete, so the result may be hard to read. Consider "
    "plotting them in separate cells, or using type='scatter' instead."
)

# 2D density / contour plot.
DENSITY2D_GRID_POINTS = 300  # 300x300 minimum -- a 100x100 grid
# produces visible granularity
DENSITY2D_QUANTILE_LOW = 0.001
DENSITY2D_QUANTILE_HIGH = 0.999
DENSITY2D_PADDING_FRAC = 0.1  # quantile bounds, not raw min/max, so
# outlier-heavy data doesn't stretch the axes (same rationale as the
# 1D density case)
DENSITY2D_CONTINUOUS_LEVELS = 256  # number of contourf bands used for
# the default continuous density plot (contour=False). High enough
# that the bands blend into a smooth gradient.
DENSITY2D_LEVELS = 8  # default number of discrete color bands for the
# contour plot (contour=True); levels= overrides per call
DENSITY2D_CONTOUR_LINE_COLOR = "white"
DENSITY2D_CONTOUR_LINEWIDTH = 0.3  # thin white lines between bands
# improve readability
DENSITY2D_CONTOUR_LINE_ALPHA = 0.4
DENSITY2D_CBAR_DECIMALS = 3
DENSITY2D_CBAR_TICKS = 8  # evenly spaced colorbar ticks (including
# both endpoints, 0 and the peak density) for the continuous density
# plot; the contour plot instead ticks its discrete band edges

# Tile plot. The colormap comes from image.cmap (viridis) in
# symbulate.mplstyle.
TILE_DEFAULT_BINS = 30  # equal-width bins for a continuous axis;
# matches make_hist2d's default so mixed discrete/continuous tiles bin
# the same way
TILE_CBAR_SIZE = "5%"
TILE_CBAR_PAD = 0.1
TILE_CBAR_TICKS = 8
TILE_CBAR_DECIMALS = 3
# Separator lines drawn on the cell boundaries of the discrete axis of a
# mixed-data tile plot (one discrete axis, one continuous), so each
# discrete level reads as its own column/row. White reads clearly on the
# viridis mesh, matching the density2d contour lines.
TILE_GRID_LINE_COLOR = "white"
TILE_GRID_LINE_WIDTH = 1.0
TILE_GRID_LINE_ALPHA = 0.6
TILE_OVERLAY_WARNING = (
    "Warning: you drew a second tile plot on the same plot. The two "
    "color scales compete, so the result may be hard to read. Consider "
    "plotting them in separate cells, or using type='scatter' instead."
)

# Mosaic plot: column widths track each x value's marginal frequency;
# within a column, segment heights track that column's conditional
# frequency of each y value, so area encodes joint frequency. Colored by
# y category (Okabe-Ito) rather than a colormap, so it takes a legend
# instead of a colorbar.
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

# Box plot.
BOXPLOT_ALPHA = 0.75
BOXPLOT_EDGECOLOR = "black"
BOXPLOT_LINEWIDTH = 1.2
BOXPLOT_FLIER_MARKER = "o"
BOXPLOT_FLIER_SIZE = 4

# Violin plot.
VIOLIN_EDGECOLOR = "black"
VIOLIN_EDGEWIDTH = 1
VIOLIN_WIDTH = 0.5  # matches ax.violinplot's own default width
VIOLIN_BOX_WIDTH_RATIO = 0.3  # inner IQR box width, as a fraction of VIOLIN_WIDTH
VIOLIN_BOX_FACECOLOR = "#FDF6E3"  # pale ivory, reads against any hue color
VIOLIN_OVERLAY_WARNING = (
    "Warning: you drew a second violin plot on the same plot. Overlapping "
    "violin shapes are hard to tell apart, so the result may be hard to "
    "read. Consider plotting them in separate cells, or using type='box' "
    "instead."
)


class SymbulatePlot:
    """Wrapper object returned by every ``.plot()`` method.

    Wraps the matplotlib axes a plot was drawn on. Its string
    representation is empty so that Jupyter does not print an object
    address below the plot. Returning this object (instead of None)
    lays the foundation for a future plot composition API (e.g.,
    ``.plot() + vline(x=0)``) and a thin interactivity conversion
    layer, without requiring changes to the plot architecture later.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes the plot was drawn on.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
        The axes the plot was drawn on.

    Examples
    --------
    >>> from symbulate import *
    >>> p = RV(Normal(0, 1)).sim(100).plot()  # doctest: +SKIP
    >>> p.ax  # the underlying matplotlib axes  # doctest: +SKIP
    """

    def __init__(self, ax):
        self.ax = ax

    def __repr__(self):
        """Return an empty string so Jupyter prints nothing."""
        return ""


def configure_axes(axes, xdata, ydata, xlabel=None, ylabel=None):
    # Create 5% buffer on either end of plot so that leftmost and rightmost
    # lines are visible. However, if current axes are already bigger,
    # keep current axes.
    data_range = max(xdata) - min(xdata)
    buff = 0.05 * data_range if data_range > 0 else 1.0
    xmin, xmax = axes.get_xlim()
    xmin = min(xmin, min(xdata) - buff)
    xmax = max(xmax, max(xdata) + buff)
    if xmin == xmax:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    plt.xlim(xmin, xmax)

    _, ymax = axes.get_ylim()
    ymax = max(ymax, 1.05 * max(ydata))
    plt.ylim(0, ymax)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)


def plot(*args, **kwargs):
    """Plot a simulation result, or fall back to matplotlib's plot.

    If the first argument has its own ``.plot()`` method (simulation
    results, distributions, time functions), that method is used.
    Otherwise the arguments are passed straight to
    ``matplotlib.pyplot.plot``.

    Parameters
    ----------
    *args
        The object to plot, or raw x/y data for matplotlib.
    **kwargs
        Additional keyword arguments passed to the underlying
        plotting function.

    Returns
    -------
    SymbulatePlot
        A wrapper around the matplotlib axes the plot was drawn on.
        Its printed representation is empty, so Jupyter shows only
        the plot.
    """
    try:
        return args[0].plot(**kwargs)
    except:
        plt.plot(*args, **kwargs)
        return SymbulatePlot(plt.gca())


def classify_data(
    values, n_unique_threshold=N_UNIQUE_THRESHOLD, n_small_threshold=N_SMALL_THRESHOLD
):
    """Classify simulated values for choosing a default plot type.

    Makes two independent determinations that together drive the default
    plot lookup: whether the variable is *discrete-ish* (few enough distinct
    values to draw one mark or cell per value, rather than binning it as
    continuous) and whether the sample is *small* (few enough observations
    to show every individual point, rather than an aggregated summary). This
    replaces the old ``is_discrete``, which conflated the two.

    For multi-dimensional results, call this once per variable (column); each
    variable gets its own ``discrete_ish`` determination.

    The rules, by dtype:

    - object, string (including numpy string dtypes), or boolean outcomes
      are always discrete-ish (categorical data has no continuous reading);
    - float values that are all distinct are continuous;
    - float values with repeats are a user-defined finite support, so
      discrete-ish only if there are at most ``n_unique_threshold`` of them;
    - integer values are discrete-ish only if there are at most
      ``n_unique_threshold`` distinct ones, so wide-support integer
      distributions (e.g. ``Binomial(10000, 0.5)``) read as continuous.

    Parameters
    ----------
    values : array-like
        The simulated values for one variable, e.g. ``RVResults.array``.
    n_unique_threshold : int, optional
        Largest number of distinct values that still counts as discrete-ish.
        Defaults to the module constant ``N_UNIQUE_THRESHOLD``.
    n_small_threshold : int, optional
        A sample with fewer than this many values counts as small. Defaults
        to the module constant ``N_SMALL_THRESHOLD``.

    Returns
    -------
    tuple of (bool, bool)
        ``(discrete_ish, small_n)``.

    Examples
    --------
    >>> import numpy as np
    >>> classify_data(np.array([0, 1, 2, 1, 3, 2] * 2000))  # narrow int, large n
    (True, False)
    >>> classify_data(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))  # all-distinct float, small n
    (False, True)
    >>> classify_data(np.array(["H", "T", "H", "T"]))        # categorical
    (True, True)
    """
    data = np.asarray(list(values))
    n = len(data)
    n_unique = len(set(data))

    if data.dtype == object or data.dtype == bool or data.dtype.kind in ("U", "S"):
        discrete_ish = True
    elif data.dtype == float and n_unique == n:
        discrete_ish = False
    elif data.dtype == float and n_unique < n:
        discrete_ish = n_unique <= n_unique_threshold
    else:
        discrete_ish = n_unique <= n_unique_threshold

    small_n = n < n_small_threshold
    return discrete_ish, small_n


DEFAULT_PLOT_TYPE = {
    ("1D_categorical", True): {
        "default": "dotplot",
        "alternatives": ["impulse", "bar"],
    },
    ("1D_categorical", False): {
        "default": "impulse",
        "alternatives": ["bar", "dotplot"],
    },
    ("1D_discrete", True): {
        "default": "dotplot",
        "alternatives": ["impulse", "bar", "ecdf"],
    },
    ("1D_discrete", False): {
        "default": "impulse",
        "alternatives": ["bar", "hist", "dotplot", "ecdf"],
    },
    ("1D_continuous", True): {
        "default": "rug",
        "alternatives": ["hist", "density", "box", "ecdf"],
    },
    ("1D_continuous", False): {
        "default": "hist",
        "alternatives": ["density", "rug", "box", "ecdf"],
    },
    ("2D_dd", True): {"default": "scatter", "alternatives": ["tile", "mosaic"]},
    ("2D_dd", False): {"default": "tile", "alternatives": ["scatter", "mosaic"]},
    ("2D_cc", True): {"default": "scatter", "alternatives": ["density2d", "hist2d"]},
    ("2D_cc", False): {"default": "hist2d", "alternatives": ["density2d", "scatter"]},
    ("2D_mixed", True): {
        "default": "segmented_rug",
        "alternatives": ["tile", "box", "violin", "segmented_density"],
    },
    ("2D_mixed", False): {
        "default": "tile",
        "alternatives": ["violin", "box", "segmented_density"],
    },
}


def default_plot_type(configuration, small_n):
    """Look up the default plot type and alternatives for a data configuration.

    The lookup key is ``(configuration, small_n)``. ``configuration`` is
    derived from ``classify_data`` output, the data's dtype, and its
    dimension, and is one of:

    - ``"1D_categorical"`` -- 1D string/object outcomes
    - ``"1D_discrete"`` -- 1D numeric discrete-ish
    - ``"1D_continuous"`` -- 1D continuous-ish
    - ``"2D_dd"`` -- 2D, both axes discrete-ish
    - ``"2D_cc"`` -- 2D, both axes continuous-ish
    - ``"2D_mixed"`` -- 2D, one axis discrete and one continuous (either
      order; the plot type bins/segments whichever axis is continuous)

    Process time points (``X[t].sim(n)``) are ordinary 1D results, so they
    use the ``"1D_discrete"`` / ``"1D_continuous"`` configurations.

    Parameters
    ----------
    configuration : str
        A configuration key (see the list above).
    small_n : bool
        Whether the sample is small, as returned by ``classify_data``.

    Returns
    -------
    tuple of (str, list of str)
        ``(default, alternatives)`` -- the default plot type for this
        configuration and the list of reasonable alternatives.

    Raises
    ------
    KeyError
        If ``(configuration, small_n)`` is not a recognized combination.
    """
    entry = DEFAULT_PLOT_TYPE[(configuration, small_n)]
    return entry["default"], entry["alternatives"]


PLOT_DISPLAY_NAME = {
    "bar": "Bar Chart",
    "impulse": "Impulse Plot",
    "hist": "Histogram",
    "density": "Density Plot",
    "ecdf": "ECDF Plot",
    "rug": "Rug Plot",
    "dotplot": "Dot Plot",
    "scatter": "Scatter Plot",
    "tile": "Tile Plot",
    "mosaic": "Mosaic Plot",
    "hist2d": "2D Histogram",
    "density2d": "2D Density Plot",
    "violin": "Violin Plot",
    "box": "Box Plot",
    "boxplot": "Box Plot",
    "segmented_rug": "Segmented Rug Plot",
    "segmented_density": "Segmented Density Plot",
    "marginal": "Marginal Plot",
}


def suggestion_message(shown, default, alternatives):
    """Build the "Currently Showing / Alternative Plots" suggestion message.

    Parameters
    ----------
    shown : str
        The plot type actually being drawn (the user's ``type`` if given,
        otherwise the default).
    default : str
        The default plot type for this data, from ``default_plot_type``.
    alternatives : list of str
        The reasonable alternatives for this data, from ``default_plot_type``.

    Returns
    -------
    str
        A two-line message. The first line names the plot being shown
        (tagged "(Default)" when it is the default); the second lists the
        other reasonable plots, tagging the default and giving the
        ``type="..."`` syntax for the rest.
    """
    showing = PLOT_DISPLAY_NAME.get(shown, shown)
    if shown == default:
        showing += " (Default)"

    options = []
    for t in [default] + list(alternatives):
        if t != shown and t not in options:
            options.append(t)

    parts = []
    for t in options:
        label = PLOT_DISPLAY_NAME.get(t, t)
        if t == default:
            parts.append(f"{label} (Default)")
        else:
            parts.append(f'{label} (type = "{t}")')

    return f"Currently Showing: {showing}\nAlternative Plots: {', '.join(parts)}"


_suggestion_shown = False


def should_show_suggestion(suggest=None):
    """Decide whether to print the suggestion message on this ``.plot()`` call.

    Parameters
    ----------
    suggest : bool or None, optional
        ``None`` (default): show only on the first ``.plot()`` of the session.
        ``True``: show on every call. ``False``: never show.

    Returns
    -------
    bool
        Whether to print the suggestion message. In the ``None`` and ``True``
        cases this also marks the session as having shown the message, so a
        later ``suggest=None`` call stays quiet.
    """
    global _suggestion_shown
    if suggest is False:
        return False
    if suggest is True:
        _suggestion_shown = True
        return True
    # suggest is None: only the first .plot() of the session.
    if _suggestion_shown:
        return False
    _suggestion_shown = True
    return True


# Scatter jitter options, in the order they appear in the jitter note.
JITTER_OPTION_DESCRIPTIONS = {
    "spiral": 'jitter="spiral" (points sharing a value form a tight, '
    "countable cluster)",
    "bins": 'jitter="bins" (points sharing a value fill that value\'s '
    "box, like a tiny dot histogram)",
    "random": 'jitter="random" (small random noise)',
    False: "jitter=False (exact positions -- repeated points draw on "
    "top of each other)",
}


def jitter_suggestion_message(shown):
    """Build the jitter note printed under a discrete 2D scatter plot.

    Names the jitter layout the scatter plot is using and lists the
    other layouts a student can pass, with the exact ``jitter=...``
    syntax -- the same discovery role the plot-type suggestion message
    plays for ``type=``.

    Parameters
    ----------
    shown : str or bool
        The jitter mode actually in use: ``"spiral"``, ``"bins"``,
        ``"random"``, or ``False``.

    Returns
    -------
    str
        A two-line message: the first line names the layout in use,
        the second lists the other jitter options.
    """
    current = JITTER_OPTION_DESCRIPTIONS[shown]
    others = [
        text for mode, text in JITTER_OPTION_DESCRIPTIONS.items() if mode != shown
    ]
    return f"Currently Using: {current}\nOther Jitter Options: {', '.join(others)}"


_jitter_suggestion_shown = False


def should_show_jitter_suggestion(suggest=None):
    """Decide whether to print the jitter note on this ``.plot()`` call.

    Same policy as ``should_show_suggestion``, tracked with its own
    session flag so the first *scatter* of the session still shows the
    jitter note even if the plot-type message already appeared under an
    earlier plot.

    Parameters
    ----------
    suggest : bool or None, optional
        ``None`` (default): show only on the first jittered scatter of
        the session. ``True``: show on every call. ``False``: never
        show.

    Returns
    -------
    bool
        Whether to print the jitter note.
    """
    global _jitter_suggestion_shown
    if suggest is False:
        return False
    if suggest is True:
        _jitter_suggestion_shown = True
        return True
    if _jitter_suggestion_shown:
        return False
    _jitter_suggestion_shown = True
    return True


def count_var(x):
    counts = {}
    for val in x:
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
    return counts


def compute_density(values):
    density = gaussian_kde(values)
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()
    return density


def setup_ticks(pos, lab, ax):
    ax.set_ticks(pos)
    ax.set_ticklabels(lab)


def add_colorbar(fig, type, mappable, label):
    # create axis for cbar to place on left
    if "marginal" not in type:
        caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
    else:  # adjust height if marginals
        caxes = fig.add_axes([0, 0.1, 0.05, 0.57])
    cbar = plt.colorbar(mappable=mappable, cax=caxes)
    caxes.yaxis.set_ticks_position("left")
    cbar.set_label(label)
    caxes.yaxis.set_label_position("left")
    return caxes


def _setup_tile_axis(values, discrete, bins):
    """Cell indices, count, extent, and ticks for one tile-plot axis.

    A discrete axis gets one cell per distinct value; a continuous axis
    is split into ``bins`` equal-width bins the same way ``make_hist2d``
    does (``np.histogram``-style edges, with the largest value falling
    in the last bin).

    Parameters
    ----------
    values : numpy.ndarray
        The simulated values for this axis.
    discrete : bool
        If True, use one labeled cell per distinct value. If False,
        bin the values into ``bins`` equal-width bins.
    bins : int
        Number of bins to use when ``discrete`` is False.

    Returns
    -------
    tuple
        ``(idx, n_cells, extent, ticks)`` -- the cell index of every
        value, the number of cells along the axis, the ``(low, high)``
        imshow extent for the axis, and either ``(positions, labels)``
        for a discrete axis or ``None`` for a continuous one (whose
        ticks are left to matplotlib's numeric locator).
    """
    if discrete:
        labels = np.unique(values)
        idx = np.searchsorted(labels, values)
        n_cells = len(labels)
        # imshow centers each cell on its integer index, so a cell
        # spans index +/- 0.5.
        extent = (-0.5, n_cells - 0.5)
        ticks = (np.arange(n_cells), labels)
    else:
        low, high = values.min(), values.max()
        if low == high:
            # Degenerate axis (all one value): widen so the bins have
            # nonzero width instead of collapsing.
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, bins + 1)
        # digitize returns 1..len(edges); shift to 0-based bins and
        # clip the maximum value (which lands one past the last bin)
        # back in.
        idx = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
        n_cells = bins
        # Data-unit extent so matplotlib labels the axis like a numeric
        # histogram axis; the equal-width imshow columns line up
        # exactly with the equal-width bins.
        extent = (edges[0], edges[-1])
        ticks = None
    return idx, n_cells, extent, ticks


def make_tile(
    x,
    y,
    ax,
    normalize=True,
    bins=None,
    discrete_x=None,
    discrete_y=None,
    colorbar=True,
    **kwargs,
):
    """Draw a 2D tile plot of simulated (x, y) pairs on the given axes.

    Draws a grid of filled cells colored by how often each cell
    occurred, using the package's sequential colormap (viridis, from
    ``symbulate.mplstyle``), with a colorbar on the right labeled
    "Relative Frequency" (or "Count" when ``normalize=False``). The
    x-axis is labeled "X", the y-axis "Y", and the title reads "Tile
    Plot". The color scale always starts from 0 so a cell that never
    occurred reads as "no data" rather than an arbitrary color. The
    colorbar ticks both endpoints (0 and the peak value) with
    ``TILE_CBAR_TICKS`` (8) evenly spaced ticks; relative-frequency
    labels are rounded to ``TILE_CBAR_DECIMALS`` (3) decimals and raw
    counts to whole numbers. This matches the density2d colorbar.

    Each axis is handled according to whether its variable is discrete
    or continuous (see ``discrete_x`` / ``discrete_y``). A discrete
    axis gets one labeled cell per distinct value, equal-sized
    regardless of the gaps between values. A continuous axis is split
    into ``bins`` equal-width bins -- the same binning ``make_hist2d``
    uses -- and labeled like a numeric histogram axis. This covers
    both two-discrete-variable data and the mixed case (one discrete
    axis, one continuous axis); continuous-x-continuous data is routed
    to ``hist2d`` / ``density2d`` instead. On the mixed case, separator
    lines are drawn on the discrete axis' cell boundaries (vertical when
    x is discrete, horizontal when y is discrete) so each discrete level
    reads as its own column or row.

    Unlike the 1D plot types, a tile plot encodes magnitude with a
    colormap instead of the categorical color cycle, so no ``color``
    parameter is taken. Overlays are the "readability warning"
    category of the overlay policy: a second call on the same axes
    still draws, but prints a warning that the color scales compete.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). This mirrors how the other plot helpers in
    this module (``make_hist2d``, ``make_density2D``) are called.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first
        column of ``RVResults.array``. Discrete or continuous.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
        Discrete or continuous.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    normalize : bool, default True
        If True, cell colors show the relative frequency of each cell
        (all cells sum to 1). If False, colors show raw counts.
    bins : int, optional
        Number of equal-width bins for a continuous axis, matching
        ``make_hist2d``. Defaults to ``TILE_DEFAULT_BINS`` (30). Only
        applies when at least one axis is continuous; passing it for
        two discrete variables has no effect and raises a warning.
    discrete_x : bool, optional
        Whether the x-axis is discrete (one cell per value) or
        continuous (binned). If None (default), determined by
        ``classify_data``, the package-wide discreteness check.
    discrete_y : bool, optional
        Same as ``discrete_x`` for the y-axis.
    colorbar : bool, default True
        If True, add a colorbar to the right of the axes. The
        'marginal' layout in ``RVResults.plot()`` passes False and
        places its own colorbar so the marginal panels aren't
        squeezed.
    **kwargs
        Additional keyword arguments passed to ``ax.imshow``.

    Returns
    -------
    matplotlib.image.AxesImage
        The image from ``ax.imshow``, so the caller can inspect the
        cells or attach further styling.

    Examples
    --------
    Two discrete variables:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.poisson(3, 1000)
    >>> y = rng.poisson(2, 1000)
    >>> ax = plt.gca()
    >>> make_tile(x, y, ax)  # doctest: +SKIP

    Mixed data -- discrete y, continuous x (x is binned automatically):

    >>> x = rng.normal(0, 1, 1000)
    >>> y = rng.poisson(3, 1000)
    >>> make_tile(x, y, plt.gca())  # doctest: +SKIP
    """
    xs, ys = np.asarray(x), np.asarray(y)
    # Auto-detect which axes are discrete with classify_data, the
    # package-wide discreteness check, when the caller doesn't say.
    # RVResults.plot() passes its own classify_data determination in
    # explicitly; this fallback keeps a direct make_tile() call
    # consistent with it (e.g. a wide-support integer axis is binned as
    # continuous rather than given one skinny cell per value).
    if discrete_x is None:
        discrete_x = classify_data(xs)[0]
    if discrete_y is None:
        discrete_y = classify_data(ys)[0]

    # bins only bins a continuous axis. With two discrete variables
    # there is nothing to bin -- every distinct value already gets its
    # own cell -- so a bins= argument has no effect. Warn rather than
    # silently ignore it (same pattern as make_density2D's levels=
    # warning). This check is on the explicit argument, so it fires
    # before the default is filled in.
    if bins is not None and discrete_x and discrete_y:
        warnings.warn(
            "bins only applies when one axis is continuous (it sets how many "
            "equal-width bins that axis is split into). Both of these "
            "variables are discrete, so every distinct value already gets its "
            "own cell and bins was ignored.",
            UserWarning,
            stacklevel=2,
        )
    if bins is None:
        bins = TILE_DEFAULT_BINS

    # Count the tile plots drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_tile_count", 0)
    ax._tile_count = n_prior + 1

    # Build each axis independently: a discrete axis gets one labeled
    # cell per distinct value; a continuous axis is binned into
    # equal-width bins exactly like make_hist2d.
    x_idx, nx, x_extent, x_ticks = _setup_tile_axis(xs, discrete_x, bins)
    y_idx, ny, y_extent, y_ticks = _setup_tile_axis(ys, discrete_y, bins)
    intensity = np.zeros((ny, nx))
    np.add.at(intensity, (y_idx, x_idx), 1)
    if normalize:
        intensity /= len(xs)

    # No cmap argument: the sequential colormap comes from image.cmap
    # (viridis) in symbulate.mplstyle. vmin=0 anchors the color scale
    # at zero. origin="lower" puts the smallest values at bottom-left;
    # aspect="auto" lets the cells fill the axes box. The extent puts a
    # binned axis in data units (so matplotlib labels it like a numeric
    # histogram axis) and a discrete axis in cell-index units.
    mesh = ax.imshow(
        intensity,
        origin="lower",
        aspect="auto",
        vmin=0,
        extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
        **kwargs,
    )
    # A filled mesh covers the whole axes, so the reference grid has
    # nothing to sit on -- turn it off rather than let fragments show
    # at the edges.
    ax.grid(False)
    # A discrete axis gets one tick per cell, labeled with the value; a
    # continuous (binned) axis keeps matplotlib's automatic numeric
    # ticks over its data range, matching the make_hist2d look.
    if x_ticks is not None:
        ax.set_xticks(x_ticks[0])
        ax.set_xticklabels(x_ticks[1])
    if y_ticks is not None:
        ax.set_yticks(y_ticks[0])
        ax.set_yticklabels(y_ticks[1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Tile Plot")
    # On mixed data (exactly one discrete axis), draw separator lines on
    # the discrete axis' cell boundaries -- halfway between neighboring
    # level indices -- so each level reads as its own column or row. The
    # lines span the continuous axis and sit on top of the mesh. Vertical
    # lines when x is the discrete axis, horizontal when y is.
    # Both-discrete and both-continuous tiles are left clean.
    if discrete_x != discrete_y:
        n_levels = nx if discrete_x else ny
        draw_line = ax.axvline if discrete_x else ax.axhline
        for boundary in np.arange(n_levels - 1) + 0.5:
            draw_line(
                boundary,
                color=TILE_GRID_LINE_COLOR,
                linewidth=TILE_GRID_LINE_WIDTH,
                alpha=TILE_GRID_LINE_ALPHA,
            )
    if colorbar:
        # Colorbar on the right, sized relative to the axes so it
        # tracks figure resizing (the approved replacement for the old
        # hardcoded fig.add_axes colorbar).
        cax = make_axes_locatable(ax).append_axes(
            "right", size=TILE_CBAR_SIZE, pad=TILE_CBAR_PAD
        )
        cbar = ax.get_figure().colorbar(mesh, cax=cax)
        cbar.set_label("Relative Frequency" if normalize else "Count")
        # The color scale starts at 0 (vmin=0 above); tick both ends of
        # the bar with a fixed number of evenly spaced ticks --
        # matplotlib's default locator otherwise trims short of the
        # endpoints. get_clim() is authoritative once the colorbar
        # exists. Labels are rounded to TILE_CBAR_DECIMALS for the
        # relative-frequency scale, or to whole numbers for raw counts.
        vmin, vmax = mesh.get_clim()
        cbar.set_ticks(np.linspace(vmin, vmax, TILE_CBAR_TICKS))
        decimals = TILE_CBAR_DECIMALS if normalize else 0
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
        )
        # Adding the colorbar axes makes it current; restore the data
        # axes so a follow-up .plot() call overlays the data, not the
        # colorbar.
        plt.sca(ax)
    if ax._tile_count > 1:
        print(TILE_OVERLAY_WARNING)
    return mesh


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
    in this module (``make_tile``, ``make_hist2d``) are called.

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


def make_violin(data, positions, ax, color, axis, alpha):
    """Draw a violin plot of simulated values grouped by one discrete axis.

    One violin per distinct value in ``positions``, along whichever of
    ``axis="x"``/``"y"`` is discrete -- the other axis holds the
    continuous values. Every violin carries a narrow inner boxplot
    (pale ivory box, black median line and whiskers) so the median and
    IQR stay readable underneath the density shape.

    A second call on the same axes still draws, but two overlapping
    sets of violin shapes are hard to tell apart, so a warning prints
    below the plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` -- pass the result in as ``color``. This
    mirrors how ``RVResults.plot()`` calls the other plot helpers
    (``make_tile``, ``make_segmented_rug``).

    Parameters
    ----------
    data : numpy.ndarray
        The simulated (x, y) pairs, e.g. ``RVResults.array``.
    positions : list
        The distinct values of the discrete axis; one violin is drawn
        per position.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the violins, from ``get_next_color(ax)``.
    axis : {"x", "y"}
        Which axis is discrete. ``"x"`` draws vertical violins along
        x, with y as the continuous value axis; ``"y"`` draws
        horizontal violins along y, with x as the continuous value
        axis.
    alpha : float
        Violin body transparency between 0 and 1.

    Returns
    -------
    tuple
        ``(violins, boxplot)`` -- the dicts of Matplotlib artists
        returned by ``ax.violinplot`` and ``ax.boxplot``, so the
        caller can inspect or further style them.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> data = np.column_stack(
    ...     [np.repeat([0, 1, 2], 50), rng.normal(0, 1, 150)]
    ... )
    >>> ax = plt.gca()
    >>> make_violin(data, [0, 1, 2], ax, get_next_color(ax), "x", 0.5)  # doctest: +SKIP
    """
    i, j = (0, 1) if axis == "x" else (1, 0)
    values = [data[data[:, i] == pos, j].tolist() for pos in positions]
    orientation = "horizontal" if axis == "y" else "vertical"
    violins = ax.violinplot(
        dataset=values,
        widths=VIOLIN_WIDTH,
        showmedians=False,
        showextrema=False,
        orientation=orientation,
    )
    # violinplot (no positions= given above) places bodies at matplotlib's
    # own default 1, 2, ..., len(values) -- sequential slots, independent
    # of what the group values themselves are. The ticks must mark those
    # same sequential slots (not positions + 1, which only coincides with
    # them when the discrete values happen to be 0..n-1, e.g. Binomial's
    # support -- and crashes outright for non-numeric group labels like
    # "H"/"T"). Labeling each slot with its real value is exactly what the
    # inner boxplot below already does via its own explicit positions=.
    setup_ticks(
        list(range(1, len(positions) + 1)),
        positions,
        ax.xaxis if axis == "x" else ax.yaxis,
    )
    for body in violins["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(VIOLIN_EDGECOLOR)
        body.set_linewidth(VIOLIN_EDGEWIDTH)
        body.set_alpha(alpha)

    # The inner boxplot marks the median and IQR on top of each violin's
    # density shape. Its positions match violinplot's own implicit
    # 1..n default (violinplot above is not given an explicit
    # positions=), so the two align. manage_ticks=False so this call
    # doesn't clobber the discrete-axis tick labels setup_ticks() set.
    box_width = VIOLIN_WIDTH * VIOLIN_BOX_WIDTH_RATIO
    boxplot = ax.boxplot(
        values,
        positions=list(range(1, len(values) + 1)),
        widths=box_width,
        orientation=orientation,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        boxprops=dict(
            facecolor=VIOLIN_BOX_FACECOLOR,
            edgecolor=VIOLIN_EDGECOLOR,
            linewidth=VIOLIN_EDGEWIDTH,
        ),
        medianprops=dict(color=VIOLIN_EDGECOLOR, linewidth=VIOLIN_EDGEWIDTH),
        whiskerprops=dict(color=VIOLIN_EDGECOLOR, linewidth=VIOLIN_EDGEWIDTH),
        capprops=dict(color=VIOLIN_EDGECOLOR, linewidth=VIOLIN_EDGEWIDTH),
    )
    # The boxplot needs to sit above the violin body it's drawn on top of.
    for artists in boxplot.values():
        for artist in artists:
            artist.set_zorder(3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Violin Plot")

    # Count the violin plots drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_violin_count", 0)
    ax._violin_count = n_prior + 1
    if ax._violin_count > 1:
        print(VIOLIN_OVERLAY_WARNING)
    return violins, boxplot


def make_marginal_impulse(count, color, ax_marg, alpha, axis):
    key, val = list(count.keys()), list(count.values())
    tot = sum(val)
    val = [i / tot for i in val]
    if axis == "x":
        ax_marg.vlines(key, 0, val, color=color, alpha=alpha)
    elif axis == "y":
        ax_marg.hlines(key, 0, val, color=color, alpha=alpha)


def _refresh_legend(ax, loc=IMPULSE_LEGEND_LOC):
    # A lone series stays legend-free; a second one (another series of
    # the same type, or a true-distribution overlay) turns the legend on.
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(loc=loc)


def make_impulse(values, ax, color, normalize=True, alpha=None, label=None, **kwargs):
    """Draw a 1D impulse (stem) plot of simulated discrete values.

    Each stem is capped with a filled marker so the plot matches the
    approved impulse prototype. Impulse plots overlay naturally: a
    second call on the same axes draws on top of the first, and a
    legend appears automatically in the top right once two or more
    series (impulse plots and/or a true-distribution overlay) share
    the axes. Each series is named by ``label``, or "Variable 1",
    "Variable 2", ... in call order when no label is given. When
    several series share the axes, the whole group is re-centered so
    their stems spread symmetrically around each value -- a pair
    straddles it, one just left and one just right, separated by
    ``IMPULSE_SERIES_OFFSET`` -- so no stem hides another and every
    stem still clearly belongs to its value. A lone series sits
    exactly on the values.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``.

    Parameters
    ----------
    values : array-like
        The simulated values to count, e.g. ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the stems and markers, from ``get_next_color(ax)``.
    normalize : bool, default True
        If True, stem heights are relative frequencies that sum to 1,
        comparable to a pmf. If False, stem heights are raw counts.
    alpha : float, optional
        Marker/stem transparency between 0 and 1. Defaults to the
        package standard for impulse plots (``IMPULSE_ALPHA``, fully
        opaque).
    label : str, optional
        Name for this series in the legend. Defaults to "Variable k",
        where k counts the impulse plots drawn on these axes so far.
    **kwargs
        Additional keyword arguments passed to the markers
        (``matplotlib.axes.Axes.scatter``).

    Returns
    -------
    tuple
        The ``(xs, freqs)`` values plotted, so the caller can inspect
        or further style the stems.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> values = np.random.default_rng().poisson(4, 1000)
    >>> ax = plt.gca()
    >>> make_impulse(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = IMPULSE_ALPHA

    n = len(values)
    counts = count_var(values)
    xs = list(counts.keys())
    freqs = list(counts.values())
    if normalize:
        freqs = [freq / n for freq in freqs]

    # Track the impulse series drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle) so overlays from separate .plot() calls see them.
    prior_series = getattr(ax, "_impulse_series", [])
    if label is None:
        label = f"Variable {len(prior_series) + 1}"

    stems = ax.vlines(
        xs, 0, freqs, color=color, linewidth=IMPULSE_LINEWIDTH, alpha=alpha
    )
    dots = ax.scatter(
        xs,
        freqs,
        s=IMPULSE_MARKER_SIZE,
        marker=IMPULSE_MARKER,
        color=color,
        alpha=alpha,
        label=label,
        zorder=3,
        **kwargs,
    )
    prior_series.append(
        {
            "stems": stems,
            "dots": dots,
            "xs": np.asarray(xs, dtype=float),
            "freqs": np.asarray(freqs, dtype=float),
        }
    )
    ax._impulse_series = prior_series

    # Re-center the whole group so the series' stems spread symmetrically
    # around each value (a pair straddles it, one just left and one just
    # right), keeping every stem visibly attached to its value while no
    # stem hides another.
    if len(prior_series) > 1:
        gaps = []
        for s in prior_series:
            unique_xs = np.unique(s["xs"])
            if len(unique_xs) > 1:
                gaps.append(np.diff(unique_xs).min())
        gap = min(gaps) if gaps else 1.0
        for i, s in enumerate(prior_series):
            offset = (i - (len(prior_series) - 1) / 2) * IMPULSE_SERIES_OFFSET * gap
            shifted_xs = s["xs"] + offset
            s["stems"].set_segments(
                [[(x, 0), (x, f)] for x, f in zip(shifted_xs, s["freqs"])]
            )
            s["dots"].set_offsets(np.column_stack([shifted_xs, s["freqs"]]))

    configure_axes(
        ax,
        xs,
        freqs,
        xlabel="Value",
        ylabel="Relative Frequency" if normalize else "Count",
    )

    ax.set_title(
        "Relative Frequency Impulse Plot" if normalize else "Count Impulse Plot"
    )
    _refresh_legend(ax)
    return xs, freqs


def overlay_true_distribution(pmf, ax, xlim=None, color=None, label=None, **kwargs):
    """Overlay a discrete distribution's true pmf on an impulse plot.

    Draws a smooth, marker-free curve through the exact pmf values
    (cubic spline interpolation, clipped at zero so the tails can't
    dip negative), so the true answer reads as a reference curve over
    the simulated stems, the way a density curve reads over a
    histogram.

    Meant to be called after ``make_impulse()`` on the same axes --
    e.g. ``overlay_true_distribution(Poisson(5).pmf, ax)`` layered on
    top of ``RV(Poisson(5)).sim(1000).plot(type="impulse")``. ``pmf``
    accepts the ``.pdf`` / ``.pmf`` callable already exposed on
    distribution objects in ``symbulate/distributions.py`` (the two
    are aliases of each other for discrete distributions).

    Parameters
    ----------
    pmf : callable
        Vectorized function mapping an array of integers to
        probabilities, e.g. a ``Distribution`` object's ``.pmf``.
    ax : matplotlib.axes.Axes
        The axes to draw on -- typically the axes an impulse plot was
        already drawn on, so the two layers share a scale.
    xlim : tuple of int, optional
        Inclusive ``(min, max)`` range of integer values to evaluate
        the pmf at. Defaults to the current axes' x-limits, rounded
        outward to the nearest integers.
    color : color, optional
        Color for the curve, from ``get_next_color(ax)``. Defaults to
        the next color in the cycle if not given, matching the other
        plot helpers.
    label : str, optional
        Name for this series in the legend. Defaults to
        "True Distribution".
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.plot``.

    Returns
    -------
    tuple
        The ``(xs, ys)`` integer values and exact pmf heights the
        smooth curve passes through.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy.stats import poisson
    >>> ax = plt.gca()
    >>> overlay_true_distribution(lambda xs: poisson.pmf(xs, 5), ax)  # doctest: +SKIP
    """
    if xlim is None:
        xlower, xupper = ax.get_xlim()
        xlim = (int(np.floor(xlower)), int(np.ceil(xupper)))
    if color is None:
        color = get_next_color(ax)
    if label is None:
        label = TRUE_DIST_LABEL_DEFAULT

    xs = np.arange(xlim[0], xlim[1] + 1)
    ys = np.asarray(pmf(xs), dtype=float)

    # Smooth curve through the pmf values instead of dot-to-dot
    # segments. A cubic spline gives the roundest curve through the
    # peaks; any small dips below zero it introduces in the tails are
    # clipped away, since probabilities can't be negative.
    if len(xs) >= 2:
        curve_xs = np.linspace(xs[0], xs[-1], TRUE_DIST_CURVE_POINTS)
        spline_degree = min(3, len(xs) - 1)
        curve_ys = make_interp_spline(xs, ys, k=spline_degree)(curve_xs)
        curve_ys = np.clip(curve_ys, 0, None)
    else:
        curve_xs, curve_ys = xs, ys

    ax.plot(
        curve_xs,
        curve_ys,
        linestyle=TRUE_DIST_LINESTYLE,
        linewidth=TRUE_DIST_LINEWIDTH,
        color=color,
        label=label,
        zorder=4,
        **kwargs,
    )
    _refresh_legend(ax)
    return xs, ys


def make_hist(
    values, ax, color, bins=None, normalize=True, alpha=None, label=None, **kwargs
):
    """Draw a 1D histogram of simulated values on the given axes.

    Draws in the style of the approved histogram prototype: solid
    bars with thin white edges so adjacent bars stay visually
    distinct. The x-axis is always labeled "Value"; the y-axis label
    and title read "Density" / "Density Histogram" when normalized,
    "Count" / "Count Histogram" otherwise.

    Histograms overlay naturally: a second call on the same axes draws
    on top of the first, and a legend appears automatically in the top
    right once two or more histograms share the axes. Each histogram
    is named by ``label``, or "Variable 1", "Variable 2", ... in call
    order when no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers (``make_impulse``, ``make_density``).

    Parameters
    ----------
    values : array-like
        The simulated values to bin, e.g. ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the bars, from ``get_next_color(ax)``.
    bins : int, optional
        Number of equal-width bins. Defaults to 30.
    normalize : bool, default True
        If True, bar areas sum to 1 so the histogram approximates a
        density and can be compared to a pdf curve. If False, bar
        heights are raw counts.
    alpha : float, optional
        Bar transparency between 0 and 1. Defaults to the package
        standard for histograms (``HIST_ALPHA``, 0.65).
    label : str, optional
        Name for this histogram in the legend. Defaults to
        "Variable k", where k counts the histograms drawn on these
        axes so far.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.hist``.

    Returns
    -------
    tuple
        The ``(counts, bin_edges, patches)`` tuple from ``ax.hist``,
        so the caller can inspect or further style the bars.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> values = np.random.default_rng().normal(10, 2, 1000)
    >>> ax = plt.gca()
    >>> make_hist(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if bins is None:
        bins = HIST_DEFAULT_BINS
    if alpha is None:
        alpha = HIST_ALPHA
    # The white bar edges are defaults, not overrides, so a user's own
    # edgecolor= / linewidth= keyword still wins. histtype='step' draws
    # nothing but its outline -- in edgecolor -- so it is left alone: a
    # white outline would be invisible on the white background.
    if kwargs.get("histtype", "bar") != "step":
        kwargs.setdefault("edgecolor", HIST_EDGECOLOR)
        kwargs.setdefault("linewidth", HIST_EDGEWIDTH)
    # Count the histograms drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle) so overlays from separate .plot() calls see it.
    n_prior_hists = getattr(ax, "_hist_count", 0)
    if label is None:
        label = f"Variable {n_prior_hists + 1}"
    ax._hist_count = n_prior_hists + 1
    histogram = ax.hist(
        values,
        bins=bins,
        density=normalize,
        color=color,
        alpha=alpha,
        label=label,
        **kwargs,
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if normalize else "Count")
    ax.set_title("Density Histogram" if normalize else "Count Histogram")
    # A legend only helps once there is more than one histogram to
    # tell apart; a lone histogram stays legend-free.
    if ax._hist_count > 1:
        ax.legend(loc=HIST_LEGEND_LOC)
    return histogram


def make_boxplot(values, ax, color, alpha=None, label=None, **kwargs):
    """Draw a box plot of simulated values on the given axes.

    A single box: edges at the first and third quartiles, a black
    median line, whiskers to the most extreme value within 1.5 times
    the interquartile range, and individual points beyond that drawn
    as fliers. Non-finite values (e.g. NaN) are dropped before
    plotting.

    Box plots overlay naturally: a second call on the same axes adds
    another box at the next position, and both boxes' x-ticks are
    labeled automatically -- "Variable 1", "Variable 2", ... in call
    order, or the given ``label``. Unlike a histogram, no legend is
    needed since each box already carries its own tick label.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers (``make_hist``, ``make_rug``).

    Parameters
    ----------
    values : array-like
        The simulated values to summarize, e.g. ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the box, from ``get_next_color(ax)``.
    alpha : float, optional
        Box transparency between 0 and 1. Defaults to the package
        standard for box plots (``BOXPLOT_ALPHA``, 0.75).
    label : str, optional
        Name for this box, shown as its x-tick label. Defaults to
        "Variable k", where k counts the boxes drawn on these axes so
        far.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.boxplot``.

    Returns
    -------
    dict
        The dict of Matplotlib artists returned by ``ax.boxplot``
        (``boxes``, ``medians``, ``whiskers``, ``caps``, ``fliers``),
        so the caller can inspect or further style them.

    Raises
    ------
    ValueError
        If there are no finite values to plot.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> values = np.random.default_rng().normal(10, 2, 200)
    >>> ax = plt.gca()
    >>> make_boxplot(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = BOXPLOT_ALPHA
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(
            "There are no values to plot. Simulate some values first, "
            "for example X.sim(30).plot(type='box')."
        )
    # The box fill/edges/median/whiskers/caps/fliers are all defaults,
    # not overrides, so a user's own boxprops= / widths= / etc. keyword
    # still wins (the same pattern make_hist uses for edgecolor=).
    kwargs.setdefault(
        "boxprops",
        dict(
            facecolor=color,
            edgecolor=BOXPLOT_EDGECOLOR,
            linewidth=BOXPLOT_LINEWIDTH,
            alpha=alpha,
        ),
    )
    kwargs.setdefault(
        "medianprops", dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_LINEWIDTH)
    )
    kwargs.setdefault(
        "whiskerprops", dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_LINEWIDTH)
    )
    kwargs.setdefault(
        "capprops", dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_LINEWIDTH)
    )
    kwargs.setdefault(
        "flierprops",
        dict(
            marker=BOXPLOT_FLIER_MARKER,
            markersize=BOXPLOT_FLIER_SIZE,
            markerfacecolor=color,
            markeredgecolor=BOXPLOT_EDGECOLOR,
            alpha=alpha,
        ),
    )
    kwargs.setdefault("patch_artist", True)
    kwargs.setdefault("orientation", "vertical")
    # Count the boxes drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color
    # cycle) so overlays from separate .plot() calls see it.
    n_prior_boxes = getattr(ax, "_boxplot_count", 0)
    position = n_prior_boxes + 1
    if label is None:
        label = f"Variable {position}"
    ax._boxplot_count = position
    box = ax.boxplot(values, positions=[position], tick_labels=[label], **kwargs)
    ax.set_ylabel("Value")
    ax.set_title("Box Plot")
    return box


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


def make_density(values, ax, color, bandwidth=None, alpha=None, label=None, **kwargs):
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
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers (``make_hist``, ``make_impulse``).

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
        standard for a standalone density curve (``DENSITY_ALPHA``,
        fully opaque).
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
    >>> import numpy as np
    >>> values = np.random.default_rng().normal(0, 1, 2000)
    >>> ax = plt.gca()
    >>> make_density(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = DENSITY_ALPHA
    # The curve width is a default, not an override, so a user's own
    # linewidth= keyword still wins.
    kwargs.setdefault("linewidth", DENSITY_LINEWIDTH)

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
    # specifically -- the same per-type override pattern the scatter and
    # 2D density plots use.
    ax.grid(True, axis="both")
    # A legend only helps once there is more than one curve to tell
    # apart; a lone curve stays legend-free.
    if ax._density_count > 1:
        ax.legend(loc=DENSITY_LEGEND_LOC)
    return line


def make_rug(values, ax, color, alpha=None, label=None, **kwargs):
    """Draw a rug plot of simulated values on the given axes.

    Draws one thin vertical tick per simulated value along the bottom
    of the axes. Tick heights are drawn in axes fractions
    (``RUG_TICK_HEIGHT`` of the axes height), not data units, so the
    ticks keep their size if something with a meaningful y-scale (e.g.
    a histogram) is drawn on the same axes later.

    The function automatically detects whether this is a standalone
    rug plot or overlaying another plot type. For a standalone rug, it
    hides the y-axis, the left spine, and the gridlines for a clean
    number-line look. For an overlay, it leaves the axes untouched --
    it only adds the ticks and lets the companion plot own the y-axis,
    spines, grid, labels, and title.

    Rug plots overlay naturally: a second call on the same axes draws
    on top of the first, and a legend appears automatically in the top
    right once two or more rugs share the axes. Each rug is named by
    ``label``, or "Variable 1", "Variable 2", ... in call order when
    no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers (``make_hist``, ``make_density``).

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
        standard for rug plots (``RUG_ALPHA``, 0.5), which keeps
        stacked values visible as darker ticks.
    label : str, optional
        Name for this rug in the legend. Defaults to "Variable k",
        where k counts the rugs drawn on these axes so far.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.vlines``.

    Returns
    -------
    matplotlib.collections.LineCollection
        The collection of tick marks, so the caller can inspect or
        further style them.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> values = np.random.default_rng().normal(0, 1, 60)
    >>> ax = plt.gca()
    >>> make_rug(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = RUG_ALPHA
    kwargs.setdefault("linewidth", RUG_LINEWIDTH)
    # Count the rugs drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color
    # cycle) so overlays from separate .plot() calls see it.
    n_prior_rugs = getattr(ax, "_rug_count", 0)
    if label is None:
        label = f"Variable {n_prior_rugs + 1}"
    ax._rug_count = n_prior_rugs + 1

    # Check if this is a standalone rug plot (nothing else on the axes
    # yet) or an overlay on another plot type.
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


def make_ecdf(values, ax, color, normalize=True, alpha=None, label=None, **kwargs):
    """Draw an empirical CDF (ECDF) step plot of simulated values.

    The ECDF is a right-continuous step function: at each value ``x`` it
    reports the fraction of observations less than or equal to ``x``, so
    the curve rises from 0 to 1 as it sweeps left to right (using
    ``where="post"``, it stays flat until the next observed value and
    then jumps). It gives an exact, bin-free view of a distribution --
    every observation is reflected -- and reads well for both discrete
    and continuous data.

    With ``normalize=False`` the curve instead rises 0 -> n as a running
    count. The x-axis is labeled "Value"; the y-axis and title read
    "Cumulative Relative Frequency" / "ECDF Plot" when normalized and
    "Cumulative Count" / "ECDF Plot" otherwise.

    ECDFs overlay naturally: a second call on the same axes draws on top
    of the first, and a legend appears automatically in the top left
    once two or more curves share the axes. Each curve is named by
    ``label``, or "Variable 1", "Variable 2", ... in call order when no
    label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers (``make_hist``, ``make_density``).

    Parameters
    ----------
    values : array-like
        The simulated values to accumulate, e.g. ``RVResults.array``.
        Must be numeric (the ECDF orders them along a number line).
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Line color, from ``get_next_color(ax)``.
    normalize : bool, default True
        If True, the step heights are cumulative relative frequencies
        rising to 1, comparable to a true CDF. If False, they are a
        cumulative count rising to the number of observations.
    alpha : float, optional
        Line transparency between 0 and 1. Defaults to the package
        standard for the ECDF curve (``ECDF_ALPHA``, fully opaque).
    label : str, optional
        Name for this curve in the legend. Defaults to "Variable k",
        where k counts the ECDFs drawn on these axes so far.
    **kwargs
        Additional keyword arguments passed to ``ax.step``.

    Returns
    -------
    list
        The list of ``Line2D`` objects from ``ax.step``, so the caller
        can inspect or further style the curve.

    Raises
    ------
    TypeError
        If ``values`` are not numeric.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> values = np.random.default_rng().normal(0, 1, 500)
    >>> ax = plt.gca()
    >>> make_ecdf(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = ECDF_ALPHA
    # The curve width is a default, not an override, so a user's own
    # linewidth= keyword still wins.
    kwargs.setdefault("linewidth", ECDF_LINEWIDTH)

    try:
        data = np.sort(np.asarray(values, dtype=float))
    except (ValueError, TypeError):
        raise TypeError(
            "An ECDF plot needs numbers, but these values are not numeric "
            "(for example, text like 'H' or 'T'). Try type='bar' to show how "
            "often each category occurs instead."
        )
    n = len(data)
    counts = np.arange(1, n + 1)
    ys = counts / n if normalize else counts

    # Count the ECDFs drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color and make_hist use) so
    # overlays from separate .plot() calls see it.
    n_prior = getattr(ax, "_ecdf_count", 0)
    if label is None:
        label = f"Variable {n_prior + 1}"
    ax._ecdf_count = n_prior + 1

    # where="post": F(x) is the proportion (or count) of observations
    # <= x, so the curve holds its level until the next observed value
    # and then jumps -- the standard right-continuous ECDF.
    line = ax.step(
        data,
        ys,
        where="post",
        color=color,
        alpha=alpha,
        label=label,
        **kwargs,
    )
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative Relative Frequency" if normalize else "Cumulative Count")
    ax.set_title("ECDF Plot")
    ax.set_ylim(bottom=0)
    # symbulate.mplstyle's global grid is horizontal-only (axes.grid.axis:
    # y), but the approved ECDF prototype shows both horizontal and
    # vertical reference lines, so this overrides it for this plot type
    # specifically -- the same per-type override the density plot uses.
    ax.grid(True, axis="both")
    # A legend only helps once there is more than one curve to tell
    # apart; a lone curve stays legend-free.
    if ax._ecdf_count > 1:
        ax.legend(loc=ECDF_LEGEND_LOC)
    return line


def make_segmented_rug(
    x, y, ax, color, alpha=None, discrete_x=None, discrete_y=None, **kwargs
):
    """Draw a segmented rug plot for mixed discrete/continuous data.

    The small-n counterpart of the mixed tile plot: instead of binning
    the continuous variable, every simulated value is drawn as a rug
    tick, and the ticks are grouped into one band per level of the
    discrete variable. A student can see each individual data point
    while still reading how the continuous variable is distributed
    within each discrete level.

    The orientation follows which variable is discrete, mirroring the
    mixed tile plot so the small-n and large-n views of the same data
    line up:

    - discrete ``y``, continuous ``x``: one band per y-level stacked
      vertically, with vertical ticks along x.
    - discrete ``x``, continuous ``y``: one band per x-level stacked
      horizontally, with horizontal ticks along y.

    Either way the ticks run perpendicular to the continuous value
    axis, marking each observation's position along it inside its
    band. The discrete axis is labeled with the level values and gets a
    reference gridline at each level (vertical lines when x is discrete,
    horizontal when y is discrete); the continuous axis keeps ordinary
    numeric ticks and no gridlines. The tick marks rise from each
    level's baseline and are the same small size as the 1D ``make_rug``
    ticks (``RUG_TICK_HEIGHT``).

    This plot is only for mixed data -- exactly one discrete variable
    and one continuous variable. Two discrete variables should use a
    tile plot, and two continuous variables a scatter plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` -- pass the result in as ``color``. This
    mirrors how ``make_rug`` and the other plot helpers are called.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first
        column of ``RVResults.array``. Discrete or continuous.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
        Discrete or continuous.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Color for the tick marks, from ``get_next_color(ax)``.
    alpha : float, optional
        Tick transparency between 0 and 1. Defaults to the package
        standard for rug plots (``RUG_ALPHA``, 0.5), so stacked values
        read as darker.
    discrete_x : bool, optional
        Whether the x-axis is the discrete (grouping) variable. If
        None (default), determined by ``classify_data``, the
        package-wide discreteness check.
    discrete_y : bool, optional
        Same as ``discrete_x`` for the y-axis.
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    list of matplotlib.collections.LineCollection
        The tick collections, one per discrete level, so the caller
        can inspect or further style them.

    Raises
    ------
    ValueError
        If the two variables are not one discrete and one continuous.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.normal(0, 1, 60)     # continuous
    >>> y = rng.integers(0, 4, 60)   # discrete groups
    >>> make_segmented_rug(x, y, plt.gca(), "#56B4E9")  # doctest: +SKIP
    """
    if alpha is None:
        alpha = RUG_ALPHA
    kwargs.setdefault("linewidth", RUG_LINEWIDTH)
    xs, ys = np.asarray(x), np.asarray(y)
    # Fall back to classify_data, the package-wide discreteness check,
    # when the caller doesn't specify. RVResults.plot() passes its own
    # classify_data determination in explicitly; this keeps a direct
    # make_segmented_rug() call consistent with it.
    if discrete_x is None:
        discrete_x = classify_data(xs)[0]
    if discrete_y is None:
        discrete_y = classify_data(ys)[0]

    # A segmented rug needs one discrete variable (the groups) and one
    # continuous variable (the values). Anything else is a different
    # plot.
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

    # The discrete variable defines the bands; the continuous variable
    # is the value axis. Vertical ticks when the continuous axis is x,
    # horizontal ticks when it is y -- the ticks always run
    # perpendicular to the value axis, mirroring the mixed tile plot's
    # orientation.
    if discrete_y:
        levels = np.unique(ys)
        continuous, groups = xs, ys
    else:
        levels = np.unique(xs)
        continuous, groups = ys, xs

    # Make the ticks the same visual size as the 1D make_rug ticks
    # (RUG_TICK_HEIGHT, a fraction of the axes). The discrete axis is
    # fixed below to span len(levels) data units, so
    # RUG_TICK_HEIGHT * len(levels) data units is that same fraction of
    # the axes. Ticks rise from each level's baseline, so every band
    # reads as its own small 1D rug.
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
                    **kwargs,
                )
            )

    # Label the discrete axis with the level values (one tick per band)
    # and give it a little padding so the outer bands aren't clipped;
    # the continuous axis keeps matplotlib's numeric ticks. Axis labels
    # match the mixed tile plot's "X"/"Y".
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
    # Reference gridlines run along the discrete axis only -- one line
    # per level, so the bands read as distinct groups -- while the
    # continuous value axis stays clean like the standalone 1D rug.
    # Vertical lines when x is the discrete axis, horizontal when y is
    # (exactly one is discrete here). axisbelow keeps them behind the
    # rug ticks; the grid's color and width come from symbulate.mplstyle.
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis="x" if discrete_x else "y")
    return ticks


def make_segmented_density(
    x,
    y,
    ax,
    color,
    bandwidth=None,
    alpha=None,
    label=None,
    discrete_x=None,
    discrete_y=None,
    **kwargs,
):
    """Draw a segmented density plot for mixed discrete/continuous data.

    One small kernel density curve ("ridge") of the continuous variable
    per level of the discrete variable, stacked along the discrete
    axis. Where the segmented rug shows every individual observation
    and the mixed tile plot shows binned counts, the segmented density
    plot -- commonly called a ridgeline plot -- shows
    each level's estimated *shape* -- how the continuous variable's
    distribution shifts or spreads from level to level, read at a
    glance. It is also the natural view of a discrete-time
    continuous-state process: one ridge per time point.

    All ridges in one call share a single density scale (the tallest
    peak reaches ``SEGMENTED_DENSITY_PEAK_SCALE`` baseline spacings), so peak
    heights are directly comparable across levels. Each ridge is drawn
    over the same quantile-based value range (``_density_xrange`` of
    the pooled continuous values), so outlier-heavy data doesn't
    stretch the axis and flatten the ridges.

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
    the axes. Each batch is named by ``label``, or
    "Variable 1", "Variable 2", ... in call order when no label is
    given. Each batch is scaled to its own tallest peak, so overlays
    compare shapes, not absolute density values.

    This plot is only for mixed data -- exactly one discrete variable
    and one continuous variable. Two discrete variables should use a
    tile plot, and two continuous variables a scatter plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how the other plot helpers here
    (``make_segmented_rug``, ``make_density``) are called.

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
    color : color
        Color for the ridges, from ``get_next_color(ax)``. The outline
        is drawn fully opaque and the fill translucent
        (``SEGMENTED_DENSITY_FILL_ALPHA``).
    bandwidth : float or str, optional
        Passed through to ``scipy.stats.gaussian_kde`` as ``bw_method``
        for every ridge. Defaults to scipy's own default (Scott's
        rule).
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
        ``fill_betweenx`` for the ridges. The ridge styling
        (``facecolor``, ``edgecolor``, ``linewidth``) is applied with
        ``setdefault``, so explicit keyword arguments win.

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
    >>> ax = plt.gca()
    >>> make_segmented_density(x, y, ax, get_next_color(ax))  # doctest: +SKIP
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
    vmin, vmax = _density_xrange(continuous)
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
    # shape stays readable where neighboring ridges overlap. These are
    # defaults, not overrides, so a user's own facecolor= / edgecolor= /
    # linewidth= keyword still wins (the same pattern make_hist uses).
    kwargs.setdefault("facecolor", mcolors.to_rgba(color, alpha))
    kwargs.setdefault("edgecolor", color)
    kwargs.setdefault("linewidth", SEGMENTED_DENSITY_LINEWIDTH)

    # Draw from the highest baseline down so that where ridges overlap,
    # the lower (nearer) ridge sits in front -- the classic ridgeline
    # look.
    artists = {}
    for level in sorted(levels, key=positions.get, reverse=True):
        base = positions[level]
        batch_label = label if not artists else None
        if level in densities:
            heights = densities[level] * scale
            fill = ax.fill_between if discrete_y else ax.fill_betweenx
            artists[level] = fill(
                grid,
                base,
                base + heights,
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
    # axis give one line per level, matching the segmented rug's
    # convention (they extend the baselines across the full axes
    # width). axisbelow keeps them behind the ridges; the grid's color
    # and width come from symbulate.mplstyle.
    ax.set_axisbelow(True)
    ax.grid(True, axis="both")
    # A legend only helps once there is more than one batch to tell
    # apart; a lone batch stays legend-free.
    if ax._segmented_density_count > 1:
        ax.legend(loc=SEGMENTED_DENSITY_LEGEND_LOC)
    return [artists[level] for level in all_levels if level in artists]


def make_grouped_boxplot(
    x, y, ax, color, alpha=None, discrete_x=None, discrete_y=None, **kwargs
):
    """Draw a box plot for mixed discrete/continuous data.

    One box per distinct value of whichever variable is discrete, using
    the other (continuous) variable as the value axis -- the box-plot
    counterpart of ``make_violin`` for this same data configuration.

    The orientation follows which variable is discrete, mirroring
    ``make_segmented_rug`` and the mixed tile plot so the different
    views of the same data line up:

    - discrete ``y``, continuous ``x``: one horizontal box per y-level.
    - discrete ``x``, continuous ``y``: one vertical box per x-level.

    This plot is only for mixed data -- exactly one discrete variable
    and one continuous variable. Two discrete variables should use a
    tile plot, and two continuous variables a scatter plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` -- pass the result in as ``color``. This
    mirrors how ``make_segmented_rug`` and the other plot helpers are
    called.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first
        column of ``RVResults.array``. Discrete or continuous.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
        Discrete or continuous.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the boxes, from ``get_next_color(ax)``.
    alpha : float, optional
        Box transparency between 0 and 1. Defaults to the package
        standard for box plots (``BOXPLOT_ALPHA``, 0.75).
    discrete_x : bool, optional
        Whether the x-axis is the discrete (grouping) variable. If
        None (default), detected from the data: float values are
        treated as continuous, everything else (int, bool, string) as
        discrete.
    discrete_y : bool, optional
        Same as ``discrete_x`` for the y-axis.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.boxplot``.

    Returns
    -------
    dict
        The dict of Matplotlib artists returned by ``ax.boxplot``
        (``boxes``, ``medians``, ``whiskers``, ``caps``, ``fliers``),
        so the caller can inspect or further style them.

    Raises
    ------
    ValueError
        If the two variables are not one discrete and one continuous.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.integers(0, 4, 200)   # discrete groups
    >>> y = rng.normal(0, 1, 200)     # continuous
    >>> make_grouped_boxplot(x, y, plt.gca(), "#56B4E9")  # doctest: +SKIP
    """
    if alpha is None:
        alpha = BOXPLOT_ALPHA
    xs, ys = np.asarray(x), np.asarray(y)
    if discrete_x is None:
        discrete_x = not np.issubdtype(xs.dtype, np.floating)
    if discrete_y is None:
        discrete_y = not np.issubdtype(ys.dtype, np.floating)

    # A grouped box plot needs one discrete variable (the groups) and
    # one continuous variable (the values). Anything else is a
    # different plot.
    if discrete_x == discrete_y:
        if discrete_x:
            raise ValueError(
                "A box plot needs one discrete variable and one continuous "
                "variable, but both of yours look discrete. Try a tile plot "
                "for two discrete variables."
            )
        raise ValueError(
            "A box plot needs one discrete variable and one continuous "
            "variable, but both of yours look continuous. Try a scatter "
            "plot for two continuous variables."
        )

    if discrete_y:
        levels = np.unique(ys)
        continuous, groups = xs, ys
        orientation = "horizontal"
    else:
        levels = np.unique(xs)
        continuous, groups = ys, xs
        orientation = "vertical"

    continuous = np.asarray(continuous, dtype=float)
    finite = np.isfinite(continuous)
    continuous, groups = continuous[finite], groups[finite]
    data = [continuous[groups == level] for level in levels]
    positions = np.arange(len(levels))

    # The box fill/edges/median/whiskers/caps/fliers are all defaults,
    # not overrides, so a user's own boxprops= / widths= / etc. keyword
    # still wins (the same pattern make_boxplot / make_hist use).
    kwargs.setdefault(
        "boxprops",
        dict(
            facecolor=color,
            edgecolor=BOXPLOT_EDGECOLOR,
            linewidth=BOXPLOT_LINEWIDTH,
            alpha=alpha,
        ),
    )
    kwargs.setdefault(
        "medianprops", dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_LINEWIDTH)
    )
    kwargs.setdefault(
        "whiskerprops", dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_LINEWIDTH)
    )
    kwargs.setdefault(
        "capprops", dict(color=BOXPLOT_EDGECOLOR, linewidth=BOXPLOT_LINEWIDTH)
    )
    kwargs.setdefault(
        "flierprops",
        dict(
            marker=BOXPLOT_FLIER_MARKER,
            markersize=BOXPLOT_FLIER_SIZE,
            markerfacecolor=color,
            markeredgecolor=BOXPLOT_EDGECOLOR,
            alpha=alpha,
        ),
    )
    kwargs.setdefault("patch_artist", True)

    boxes = ax.boxplot(data, positions=positions, orientation=orientation, **kwargs)

    # Label the discrete axis with the level values (one tick per box);
    # the continuous axis keeps matplotlib's numeric ticks. Axis labels
    # match the mixed tile plot's and make_segmented_rug's "X"/"Y".
    if discrete_y:
        ax.set_yticks(positions)
        ax.set_yticklabels(levels)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(levels)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Box Plot")
    return boxes


def _axes_size_px(ax):
    """Return the rendered (width, height) of the axes in pixels."""
    (x0, y0), (x1, y1) = ax.transAxes.transform([(0.0, 0.0), (1.0, 1.0)])
    return x1 - x0, y1 - y0


def _x_span_px(ax, dx):
    """Return how many pixels wide dx data units are."""
    (x0, _), (x1, _) = ax.transData.transform([(0.0, 0.0), (dx, 0.0)])
    return x1 - x0


def _dotplot_clean_values(values):
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


def _dotplot_restack(state):
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


def _dotplot_init_state(ax):
    """Set up per-axes dot plot state, styling, and resize handling."""
    state = {
        "series": [],
        "boundary_lines": [],
        "last_size_px": None,
        "relayout_running": False,
    }
    ax._dotplot_state = state
    # Horizontal reference gridlines help students read a count off
    # the y-axis. Keep them below the dots and drop the vertical
    # lines, which would clutter the stacks.
    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_xlabel(DOTPLOT_XLABEL)
    ax.tick_params(labelsize=DOTPLOT_TICK_LABEL_SIZE)
    # Dot sizes depend on the rendered size of the axes, so redo the
    # geometry whenever something changes it (figure resize,
    # tight_layout).
    ax.figure.canvas.mpl_connect(
        "resize_event", lambda event: _dotplot_on_canvas_change(ax)
    )
    ax.figure.canvas.mpl_connect(
        "draw_event", lambda event: _dotplot_on_canvas_change(ax)
    )
    return state


def _dotplot_relayout(ax):
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

    # Every dot is one count, so a stack unit is always 1.

    # Pixel measurements via the axes transforms (valid before any
    # draw). The 1-px floor guards against two nearly-identical values
    # driving the lane width -- and with it the dot size -- to zero.
    lane_width_px = max(_x_span_px(ax, lane_width), 1.0)
    height_px = _axes_size_px(ax)[1]

    # Choose the y range so one count on screen is never taller than
    # one lane is wide, nor than the DOTPLOT_MAX_DOT_SIZE cap. That
    # pixel height becomes the dot diameter, so dots stack touching and
    # never spill into the next lane, short stacks cannot inflate the
    # dots past the cap, and taller stacks shrink the dots instead of
    # overflowing the axes.
    max_dot_px = DOTPLOT_MAX_DOT_SIZE * ax.figure.dpi / 72.0
    tallest = max(s["counts"].max() for s in state["series"])
    y_max = max(
        tallest * DOTPLOT_STACK_HEADROOM,
        height_px / lane_width_px,
        height_px / max_dot_px,
    )
    ax.set_ylim(0, y_max)

    points_per_px = 72.0 / ax.figure.dpi
    # Dodge overlaid batches by exactly one dot diameter so their
    # stacks sit side by side and touch, rather than by the full lane
    # width -- which would leave a gap between the columns once the
    # dot-size cap shrinks the dots below the lane width. The dot
    # diameter is what the y range above was calibrated to; it never
    # exceeds one lane width, so a dodged group still fits inside its
    # slot without colliding with the neighboring value's stacks.
    diameter_px = height_px / y_max
    x_px_per_data = max(_x_span_px(ax, 1.0), 1e-9)
    dodge_step = diameter_px / x_px_per_data
    # scatter sizes are marker areas in points^2 (diameter squared).
    size = (diameter_px * points_per_px) ** 2
    for i, series in enumerate(state["series"]):
        offset = (i - (n_series - 1) / 2.0) * dodge_step
        xs = []
        ys = []
        for position, count in zip(positions, series["counts"]):
            xs.extend(np.full(count, position + offset))
            ys.extend(np.arange(1, count + 1) - 0.5)
        series["dots"].set_offsets(np.column_stack([xs, ys]))
        series["dots"].set_sizes(np.full(len(xs), size))

    if np.all(positions == np.round(positions)):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # The y-axis is always integer counts.
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    _dotplot_boundary_lines(ax, state)
    state["last_size_px"] = _axes_size_px(ax)


def _dotplot_boundary_lines(ax, state):
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


def _dotplot_decorate(ax, state):
    """Apply the title, labels, fonts, and (for overlays) the legend."""
    ax.set_title(DOTPLOT_TITLE)
    ax.set_ylabel("Count")
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


def _dotplot_on_canvas_change(ax):
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
        _dotplot_relayout(ax)
        ax.figure.canvas.draw_idle()
    finally:
        state["relayout_running"] = False


def make_dotplot(values, ax, color, alpha=None, label=None, **kwargs):
    """Draw a stacked dot plot of simulated values on the given axes.

    Every observation is one dot, drawn at its exact value on the
    x-axis. Identical values stack on top of one another, the first
    dot in each stack sits directly on the number line, and stacked
    dots touch. Values are never binned -- the dot plot is meant for
    discrete data. The y-axis always reads "Count": every dot is one
    observation.

    Dot plots overlay naturally: a second call on the same axes draws
    each batch side by side around the shared values, in different
    colors, dodged by exactly one dot diameter so neighboring stacks
    touch. Light vertical boundary lines appear halfway between
    neighboring stacks (at 1.5, 2.5, ... for integer data) so it stays
    clear which value each dot belongs to. A legend appears
    automatically in the top right once two or more batches share the
    axes.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers (``make_impulse``, ``make_hist``).

    Parameters
    ----------
    values : array-like
        The simulated values to plot, e.g. ``RVResults.array``. Must
        be numeric.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Dot color, from ``get_next_color(ax)``.
    alpha : float, optional
        Dot transparency between 0 and 1. Defaults to the package
        standard for dot plots (``DOTPLOT_ALPHA``, fully opaque).
    label : str, optional
        Name for this batch of values in the legend. Defaults to
        "Variable k", where k counts the dot plots drawn on these axes
        so far.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.scatter``. Dot positions and sizes are
        managed by the touching-stack layout, so ``s=`` / ``sizes=``
        are overridden.

    Returns
    -------
    matplotlib.collections.PathCollection
        The dots, as returned by ``ax.scatter``, so the caller can
        inspect or further style them.

    Raises
    ------
    TypeError
        If the values are not numeric.
    ValueError
        If there are no (finite) values to plot.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> values = np.random.default_rng().integers(1, 7, 30)
    >>> ax = plt.gca()
    >>> make_dotplot(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    values = _dotplot_clean_values(values)
    if alpha is None:
        alpha = DOTPLOT_ALPHA
    state = getattr(ax, "_dotplot_state", None)
    if state is None:
        state = _dotplot_init_state(ax)
    if label is None:
        label = "Variable {}".format(len(state["series"]) + 1)
    # Dots start empty; _dotplot_relayout fills in positions and sizes
    # once the shared bins and axes limits are known.
    dots = ax.scatter([], [], color=color, alpha=alpha, zorder=2, **kwargs)
    state["series"].append(
        {
            "values": values,
            "dots": dots,
            "label": label,
            "color": color,
        }
    )
    _dotplot_restack(state)
    _dotplot_relayout(ax)
    _dotplot_decorate(ax, state)
    return dots


# Structured-jitter fill order: a 3x3 compass template. The first point
# in a cell sits dead center, points 2-5 take the cardinal positions
# (N, E, S, W), and points 6-9 take the remaining diagonal positions
# (NE, SE, NW, SW).
_COMPASS_ORDER = [
    (0, 0),  # center
    (0, 1),  # N
    (1, 0),  # E
    (0, -1),  # S
    (-1, 0),  # W
    (1, 1),  # NE
    (1, -1),  # SE
    (-1, 1),  # NW
    (-1, -1),  # SW
]


def _dot_steps(ax):
    """Return the dot diameter in (x, y) data units.

    This is the spacing at which two neighboring dots exactly touch on
    screen -- the touching-dot geometry the dot plot uses, measured
    through ``ax.transData`` so it tracks the current axis limits and
    figure size.
    """
    diameter_px = np.sqrt(SCATTER_MARKER_SIZE) * ax.figure.dpi / 72.0
    (x0, y0), (x1, y1) = ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    px_per_xunit = max(abs(x1 - x0), 1e-9)
    px_per_yunit = max(abs(y1 - y0), 1e-9)
    return diameter_px / px_per_xunit, diameter_px / px_per_yunit


def _spiral_offsets(k, step_x, step_y):
    """Return ``k`` (dx, dy) offsets clustered on (0, 0), compass-first.

    Coincident points fill the compass template in ``_COMPASS_ORDER``:
    dead center first, then the cardinal positions (N, E, S, W) one
    dot diameter away, then the diagonals (NE, SE, NW, SW). The center
    point sits exactly on the grid intersection and the cardinal
    points lie on the grid lines themselves, so the whole cluster
    visibly belongs to that one (x, y) value. More than 9 coincident
    points no longer fit the template, so they fall back to a centered
    ``ceil(sqrt(k))``-column touching lattice. Either way the spacing
    compresses once the cluster would outgrow SCATTER_BIN_SPREAD, so
    it never reaches the neighboring value.
    """
    if k <= len(_COMPASS_ORDER):
        sx = min(step_x, SCATTER_BIN_SPREAD / 2.0)
        sy = min(step_y, SCATTER_BIN_SPREAD / 2.0)
        return [(dx * sx, dy * sy) for dx, dy in _COMPASS_ORDER[:k]]
    ncols = int(np.ceil(np.sqrt(k)))
    nrows = int(np.ceil(k / ncols))
    sx = min(step_x, SCATTER_BIN_SPREAD / max(ncols - 1, 1))
    sy = min(step_y, SCATTER_BIN_SPREAD / max(nrows - 1, 1))
    offsets = []
    for m in range(k):
        row, col = divmod(m, ncols)
        offsets.append(((col - (ncols - 1) / 2.0) * sx, (row - (nrows - 1) / 2.0) * sy))
    return offsets


def _bins_offsets(k, step_x, step_y):
    """Return ``k`` (dx, dy) offsets filling the value's box like a
    tiny dot histogram.

    Dots start in the box's bottom-left corner and fill left to right,
    then move up a row, touching -- so a fuller box reads as a bigger
    fill, the way a taller histogram bar reads as a bigger count. Rows
    are kept square-ish (about sqrt(k) dots wide, capped by how many
    touching dots fit across the box) so a pile-up reads as a compact
    countable block rather than a long string of dots; if the rows
    would still outgrow the box vertically, the row spacing compresses
    to keep every dot inside its own value's box.
    """
    half = SCATTER_BIN_SPREAD / 2.0
    sx = min(step_x, SCATTER_BIN_SPREAD)
    max_cols = max(1, int(SCATTER_BIN_SPREAD / sx))
    ncols = min(max_cols, int(np.ceil(np.sqrt(k))))
    nrows = int(np.ceil(k / ncols))
    sy = min(step_y, SCATTER_BIN_SPREAD / nrows)
    offsets = []
    for m in range(k):
        row, col = divmod(m, ncols)
        offsets.append((-half + (col + 0.5) * sx, -half + (row + 0.5) * sy))
    return offsets


def _relayout_clusters(ax):
    """Recompute every cluster's dot positions from the current axes.

    The touching-dot spacing depends on the axis limits and rendered
    figure size, so this runs when the series is first drawn and again
    from the resize/draw hooks whenever the geometry changes (same
    pattern as the dot plot).
    """
    state = getattr(ax, "_scatter_jitter_state", None)
    if state is None or not state["series"]:
        return
    step_x, step_y = _dot_steps(ax)
    for series in state["series"]:
        xi, yi = series["xi"], series["yi"]
        new_x = xi.astype(float).copy()
        new_y = yi.astype(float).copy()
        values = {}
        for i in range(len(xi)):
            values.setdefault((xi[i], yi[i]), []).append(i)
        for (cx, cy), idxs in values.items():
            if series["mode"] == "bins":
                offsets = _bins_offsets(len(idxs), step_x, step_y)
            else:
                offsets = _spiral_offsets(len(idxs), step_x, step_y)
            for pos, (dx, dy) in enumerate(offsets):
                new_x[idxs[pos]] = cx + dx
                new_y[idxs[pos]] = cy + dy
        series["dots"].set_offsets(np.column_stack([new_x, new_y]))
    state["last_geometry"] = _geometry_signature(ax)


def _geometry_signature(ax):
    """The rendered size and view limits the last layout was based on."""
    return (_axes_size_px(ax), ax.get_xlim(), ax.get_ylim())


def _scatter_on_canvas_change(ax):
    """Redo the cluster geometry if the axes' rendering has changed."""
    state = getattr(ax, "_scatter_jitter_state", None)
    if state is None or state["relayout_running"] or not state["series"]:
        return
    if state["last_geometry"] == _geometry_signature(ax):
        return
    state["relayout_running"] = True
    try:
        _relayout_clusters(ax)
        ax.figure.canvas.draw_idle()
    finally:
        state["relayout_running"] = False


def _init_jitter_state(ax):
    """Set up per-axes cluster state and the geometry-change hooks."""
    state = {"series": [], "last_geometry": None, "relayout_running": False}
    ax._scatter_jitter_state = state
    ax.figure.canvas.mpl_connect(
        "resize_event", lambda event: _scatter_on_canvas_change(ax)
    )
    ax.figure.canvas.mpl_connect(
        "draw_event", lambda event: _scatter_on_canvas_change(ax)
    )
    return state


def _max_coincident(x, y):
    """Count the points sharing the most-repeated integer (x, y) value."""
    pairs = np.column_stack([np.round(x).astype(int), np.round(y).astype(int)])
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    return int(counts.max())


def auto_jitter_mode(x, y):
    """Pick the clustered-jitter layout for two discrete variables.

    This is the automatic choice ``RVResults.plot()`` makes when the
    user does not set ``jitter`` and both variables are discrete
    (per ``classify_data``): ``"bins"`` once any single (x, y) value
    holds ``SCATTER_AUTO_BINS_THRESHOLD`` (12) or more points -- past
    the 9-dot compass template, where a spiral pile-up stops being
    cleanly countable -- and ``"spiral"`` otherwise.

    Parameters
    ----------
    x, y : array-like
        The two simulated variables. Assumed integer-valued.

    Returns
    -------
    str
        ``"spiral"`` or ``"bins"``.
    """
    if _max_coincident(x, y) >= SCATTER_AUTO_BINS_THRESHOLD:
        return "bins"
    return "spiral"


def make_scatter(
    x,
    y,
    ax,
    color,
    alpha=None,
    jitter=False,
    label=None,
    xlabel=None,
    ylabel=None,
    **kwargs,
):
    """Draw a 2D scatter plot of paired simulated values.

    Scatter plots overlay naturally: a second call on the same axes
    draws on top of the first, and a legend appears automatically in
    the top right once two or more series share the axes. Each series
    is named by ``label``, or "Variable 1", "Variable 2", ... in call
    order when no label is given. The low alpha keeps the overlap
    region readable even with filled points.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers in this module.

    Parameters
    ----------
    x : array-like
        The first coordinate of each simulated pair, e.g. column 0 of
        ``RVResults.array``.
    y : array-like
        The second coordinate of each simulated pair, e.g. column 1 of
        ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the circles, from ``get_next_color(ax)``.
    alpha : float, optional
        Point transparency between 0 and 1. Defaults to the package
        standard for scatter plots (``SCATTER_ALPHA``, 0.25).
    jitter : bool or str, default False
        How to spread out coincident points (discrete data):

        - ``False`` -- draw points at their exact coordinates. Right
          for two continuous variables (no coincident points).
          (``RVResults.plot()`` resolves its own default from the
          data: two discrete variables get the clustered layout
          ``auto_jitter_mode`` picks; anything else gets ``False``.)
        - ``"random"`` -- add small random noise to both coordinates.
          Reduces overplotting but scrambles density. ``True`` is
          accepted as a legacy alias and behaves identically.
        - ``"spiral"`` -- points that share an integer (x, y)
          coordinate form a tight compass-pattern cluster on that
          value's grid crossing (center first, then N/E/S/W on the
          grid lines, then the diagonals), neighboring dots touching,
          so the cluster reads as one shared value at natural dot
          size. Right for two discrete variables with modest pile-ups:
          a student can read each value's density by counting. Assumes
          the data are integer-valued.
        - ``"bins"`` -- histogram-style boxes: the grid lines move to
          the half-integer bin edges, so each value gets a visible box
          with its axis label centered inside, the way a histogram
          centers a bar over its bin. Points sharing a value fill
          their box from the bottom-left corner -- left to right, then
          up a row -- dots touching, like a tiny dot histogram, so a
          fuller box means a bigger count even when dozens of points
          share one value.
    label : str, optional
        Name for this series in the legend. Defaults to "Variable k",
        where k counts the scatters drawn on these axes so far.
    xlabel : str, optional
        Label for the x-axis. Defaults to "Variable 1".
    ylabel : str, optional
        Label for the y-axis. Defaults to "Variable 2".
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.scatter``.

    Returns
    -------
    matplotlib.collections.PathCollection
        The collection from ``ax.scatter``, so the caller can inspect
        or further style the points.

    Raises
    ------
    ValueError
        If ``jitter`` is not one of the recognized values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x = rng.uniform(0, 7, 40)
    >>> y = 0.8 * x + rng.normal(0, 0.7, 40)
    >>> ax = plt.gca()
    >>> make_scatter(x, y, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = SCATTER_ALPHA
    # The marker size is a default, not an override, so a user's own
    # s= keyword still wins.
    kwargs.setdefault("s", SCATTER_MARKER_SIZE)
    if jitter not in (False, True, "random", "spiral", "bins"):
        raise ValueError(
            f"jitter must be False, 'random', 'spiral', or 'bins', not "
            f"{jitter!r}. When you don't set jitter, the best layout is "
            "chosen automatically for two discrete variables. Use "
            "jitter='spiral' for tight countable clusters, jitter='bins' "
            "to spread big pile-ups across each value's bin, "
            "jitter='random' for random noise, or jitter=False for exact "
            "positions."
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xi = yi = None
    if jitter in ("spiral", "bins"):
        # Pin the view at least half a unit past the occupied crossings
        # (via the data limits, so later overlays still autoscale):
        # otherwise a lone heavily-repeated value makes autoscale zoom
        # into the cluster itself, which then reads as many separate
        # values instead of one.
        xi = np.round(x).astype(int)
        yi = np.round(y).astype(int)
        ax.update_datalim(
            [(xi.min() - 0.5, yi.min() - 0.5), (xi.max() + 0.5, yi.max() + 0.5)]
        )
        # Draw at the value centers for now; the touching-dot cluster
        # layout needs settled axis limits, so it happens at the end of
        # this call (and again from the resize/draw hooks).
        x, y = xi.astype(float), yi.astype(float)
    elif jitter:
        x = x + rng.normal(loc=0, scale=0.01 * (x.max() - x.min()), size=len(x))
        y = y + rng.normal(loc=0, scale=0.01 * (y.max() - y.min()), size=len(y))

    # Count the scatters drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color cycle)
    # so overlays from separate .plot() calls see it.
    n_prior_scatters = getattr(ax, "_scatter_count", 0)
    if label is None:
        label = f"Variable {n_prior_scatters + 1}"
    ax._scatter_count = n_prior_scatters + 1

    points = ax.scatter(
        x,
        y,
        color=color,
        alpha=alpha,
        label=label,
        **kwargs,
    )

    if jitter in ("spiral", "bins"):
        # Integer major ticks so every value label sits at its integer
        # position (spiral: grid lines through the cluster centers;
        # bins: labels centered in their boxes).
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    if jitter == "bins":
        # Histogram-style boxes: move the grid lines to the half-integer
        # bin edges (minor ticks, marks hidden) so each value gets a
        # visible box with its label centered inside, like a histogram
        # bar over its bin.
        ax.xaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(1, offset=0.5))
        ax.tick_params(which="minor", length=0)
        ax.grid(False, axis="both", which="major")
        ax.grid(True, axis="both", which="minor")
    else:
        # Vertical grid lines on top of the style sheet's horizontal
        # ones, so values read off both axes.
        ax.grid(True, axis="both")

    ax.set_xlabel("Variable 1" if xlabel is None else xlabel)
    ax.set_ylabel("Variable 2" if ylabel is None else ylabel)
    ax.set_title("2D Scatter Plot")
    _refresh_legend(ax, loc=SCATTER_LEGEND_LOC)

    if jitter in ("spiral", "bins"):
        state = getattr(ax, "_scatter_jitter_state", None)
        if state is None:
            state = _init_jitter_state(ax)
        state["series"].append({"dots": points, "xi": xi, "yi": yi, "mode": jitter})
        # Settle the view limits now so the touching-dot spacing is
        # measured against the geometry that will actually render.
        ax.autoscale_view()
        _relayout_clusters(ax)

    return points


def make_hist2d(
    x, y, ax, bins=None, normalize=True, hex=False, colorbar=True, **kwargs
):
    """Draw a 2D histogram of simulated (x, y) pairs on the given axes.

    Draws in the style of the approved 2D histogram prototype: a
    filled bin mesh using the package's sequential colormap (viridis,
    from ``symbulate.mplstyle``), with a colorbar on the right labeled
    "Density" (or "Count" when ``normalize=False``). The x-axis is
    labeled "X", the y-axis "Y", and the title reads "2-D Histogram".
    The color scale always starts from 0 so empty bins read as
    "no data" rather than an arbitrary color. The colorbar ticks both
    endpoints (0 and the peak value) with ``HIST2D_CBAR_TICKS`` (8)
    evenly spaced ticks; density labels are rounded to
    ``HIST2D_CBAR_DECIMALS`` (3) decimals and raw counts to whole
    numbers. This matches the density2d colorbar.

    With ``hex=True`` the bins are hexagons instead of squares and the
    title reads "Hexbin Plot"; everything else (colormap, colorbar,
    normalization, overlay warning) behaves the same.

    Unlike the 1D plot types, a 2D histogram encodes magnitude with a
    colormap instead of the categorical color cycle, so no ``color``
    parameter is taken. Overlays are the "readability warning"
    category of the overlay policy: a second call on the same axes
    still draws, but prints a warning that the color scales compete.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working). This mirrors how the other plot helpers in
    this module (``make_tile``, ``make_density2D``) are called.

    Parameters
    ----------
    x : array-like
        Simulated values for the horizontal axis, e.g. the first
        column of ``RVResults.array``.
    y : array-like
        Simulated values for the vertical axis, same length as ``x``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    bins : int, optional
        Number of equal-width bins along each axis (the number of
        hexagons across the x-axis when ``hex=True``). Defaults to 30.
    normalize : bool, default True
        If True, bin colors show density (cell areas integrate to 1)
        so the plot approximates the joint density. If False, colors
        show raw counts.
    hex : bool, default False
        If True, bin the data into hexagons instead of squares and
        title the plot "Hexbin Plot". (The name shadows the built-in
        ``hex()``, matching how the package's ``type=`` parameter
        shadows ``type()``.)
    colorbar : bool, default True
        If True, add a colorbar to the right of the axes. The
        'marginal' layout in ``RVResults.plot()`` passes False and
        places its own colorbar so the marginal panels aren't
        squeezed.
    **kwargs
        Additional keyword arguments passed to ``ax.hist2d`` (or
        ``ax.hexbin`` when ``hex=True``).

    Returns
    -------
    tuple or matplotlib.collections.PolyCollection
        With square bins, the ``(counts, xedges, yedges, mesh)`` tuple
        from ``ax.hist2d``; with ``hex=True``, the hexagon collection
        from ``ax.hexbin``. Either way the caller can inspect the bins
        or attach further styling.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> x, y = rng.normal(0, 1, (2, 1000))
    >>> ax = plt.gca()
    >>> make_hist2d(x, y, ax)  # doctest: +SKIP
    """
    if bins is None:
        bins = HIST2D_DEFAULT_BINS
    xs, ys = np.asarray(x), np.asarray(y)
    # Count the 2D histograms drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle), to trigger the overlay readability warning.
    n_prior = getattr(ax, "_hist2d_count", 0)
    ax._hist2d_count = n_prior + 1
    # No cmap argument in either branch: the sequential colormap comes
    # from image.cmap in symbulate.mplstyle. vmin=0 anchors the color
    # scale at zero.
    if hex:
        histogram = ax.hexbin(xs, ys, gridsize=bins, vmin=0, **kwargs)
        # The hexagon tiling has a jagged outline that doesn't fill
        # the square axes box. Painting the axes background with the
        # colormap's zero color makes everything beyond the tiling
        # read as zero density, so the plot edge is the clean square
        # of the axes, matching the 2-D histogram.
        ax.set_facecolor(histogram.get_cmap()(0))
        if normalize:
            # hexbin has no density option, so rescale the counts by
            # hand: density = count / (n * cell area), which makes the
            # hexagon volumes sum to 1 exactly like a density
            # histogram. All hexagons are congruent, so the shoelace
            # formula on one hexagon's vertices (in data units) gives
            # the cell area.
            v = histogram.get_paths()[0].vertices[:6]
            area = 0.5 * abs(
                np.sum(v[:, 0] * np.roll(v[:, 1], -1) - np.roll(v[:, 0], -1) * v[:, 1])
            )
            density = histogram.get_array() / (len(xs) * area)
            histogram.set_array(density)
            histogram.set_clim(0, float(density.max()))
        mesh = histogram
    else:
        histogram = ax.hist2d(xs, ys, bins=bins, density=normalize, vmin=0, **kwargs)
        mesh = histogram[3]
    # A filled mesh covers the whole axes, so the reference grid has
    # nothing to sit on -- turn it off rather than let fragments show
    # at the edges.
    ax.grid(False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Hexbin Plot" if hex else "2-D Histogram")
    if colorbar:
        # Colorbar on the right, sized relative to the axes so it
        # tracks figure resizing (the approved replacement for the old
        # hardcoded fig.add_axes colorbar).
        cax = make_axes_locatable(ax).append_axes(
            "right", size=HIST2D_CBAR_SIZE, pad=HIST2D_CBAR_PAD
        )
        cbar = ax.get_figure().colorbar(mesh, cax=cax)
        cbar.set_label("Density" if normalize else "Count")
        # The color scale starts at 0 (vmin=0 above); tick both ends of
        # the bar with a fixed number of evenly spaced ticks --
        # matplotlib's default locator otherwise trims short of the
        # endpoints. get_clim() is authoritative once the colorbar
        # exists. Labels are rounded to HIST2D_CBAR_DECIMALS for the
        # density scale, or to whole numbers for raw counts.
        vmin, vmax = mesh.get_clim()
        cbar.set_ticks(np.linspace(vmin, vmax, HIST2D_CBAR_TICKS))
        decimals = HIST2D_CBAR_DECIMALS if normalize else 0
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
        )
        # Adding the colorbar axes makes it current; restore the data
        # axes so a follow-up .plot() call overlays the data, not the
        # colorbar.
        plt.sca(ax)
    if ax._hist2d_count > 1:
        print(HIST2D_OVERLAY_WARNING)
    return histogram


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


def make_density2D(x, y, ax, contour=False, levels=None, colorbar=True, **kwargs):
    """Draw a 2D density surface from a KDE estimate.

    Both modes plot the *same* KDE-estimated density surface with
    ``ax.contourf``; they differ in how finely it is quantized:

    - ``contour=False`` (default): a *continuous* density plot. The
      surface is drawn with a large fixed number of color bands
      (``DENSITY2D_CONTINUOUS_LEVELS``) so they blend into a smooth
      gradient with no visible banding -- the "2D Density Plot" look.
      The ``levels`` argument does not apply here; passing it warns
      and has no effect.
    - ``contour=True``: a topographic "Contour Plot". The same surface
      is split into ``levels`` discrete color bands with thin white
      outlines between them, so each band can be matched to the
      colorbar by eye.

    The axis limits are quantile-based (0.1st to 99.9th percentile of
    each variable, plus padding), not raw min/max, so outlier-heavy
    data doesn't stretch the plot -- same rationale as the 1D density
    curve. The KDE is evaluated on a 300x300 grid. The color scale
    runs from 0 to the peak density, so the colorbar starts at 0 and
    ticks both endpoints (0 and the peak); its labels are rounded to
    ``DENSITY2D_CBAR_DECIMALS`` (3) decimal places.

    A second ``make_density2D`` call on the same axes cannot overlay
    naturally -- a filled 2D surface completely obscures whatever was
    drawn before it, and the two color scales compete for the same
    colorbar space. Rather than silently producing a misleading plot,
    this prints a readability warning (matching the existing warning
    category for two 2D tile or two 2D histogram plots) and still
    draws, hiding the first plot underneath.

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
        Number of discrete color bands, used only when
        ``contour=True``. Defaults to ``DENSITY2D_LEVELS`` (8) in that
        mode. Must be a whole number of at least 2. Ignored (with a
        warning) when ``contour=False``, since the continuous density
        plot has no bands.
    colorbar : bool, default True
        If True, add a colorbar to the right of the axes (first call
        on these axes only). The 'marginal' layout in
        ``RVResults.plot()`` passes False and places its own colorbar
        so the marginal panels aren't squeezed.
    **kwargs
        Additional keyword arguments passed to ``ax.contourf``.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The object returned by ``ax.contourf``, so the caller can
        inspect or further style the surface.

    Raises
    ------
    ValueError
        If ``levels`` is not a whole number of at least 2 when
        ``contour=True``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
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
        # Continuous density plot: there are no discrete bands to
        # control, so levels has no meaning here. Warn if the user
        # passed one rather than silently ignoring it, then fall back
        # to the large fixed count that makes the surface look
        # continuous.
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
    # The color scale runs from 0 to the peak density so the colorbar
    # starts at 0. KDE density is non-negative, so no low-end clipping
    # is needed. contourf treats the level values as band *boundaries*
    # (N boundaries -> N - 1 colors), so build levels + 1 edges to get
    # exactly `levels` discrete colors. No cmap argument: the
    # sequential colormap comes from image.cmap in symbulate.mplstyle.
    level_edges = np.linspace(0, zmax, levels + 1)

    filled = ax.contourf(Xgrid, Ygrid, Z, levels=level_edges, **kwargs)
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
    # symbulate.mplstyle's global grid is horizontal-only
    # (axes.grid.axis: y); the approved prototypes call for both
    # horizontal and vertical reference lines, so this overrides it for
    # this plot type -- same per-type override pattern as the 1D
    # density curve. Since the filled surface covers the full axes
    # (axisbelow=True keeps the grid behind the data), the grid won't
    # actually show through the fill -- only at the tick marks along
    # the spines.
    ax.grid(True, axis="both")

    # Colorbar via make_axes_locatable, per the approved layout helpers
    # -- replaces the hardcoded fig.add_axes([0, 0.1, 0.05, 0.8]) used
    # by add_colorbar(). Only the first call on a given axes adds one:
    # a second call already prints the readability warning above and
    # covers the first surface, so a second colorbar would just overlap
    # the first at nearly the same position, garbling both sets of tick
    # labels.
    if colorbar and n_prior == 0:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(filled, cax=cax)
        cbar.set_label("Density")
        # Tick both ends of the bar (0 and the peak density), which
        # matplotlib's default locator otherwise trims. In contour mode
        # the discrete band edges are the natural ticks -- and since
        # they span 0 to zmax, the endpoints come for free. In
        # continuous mode there are hundreds of bands, so use a small
        # set of evenly spaced ticks between the same endpoints
        # instead.
        if contour:
            ticks = level_edges
        else:
            ticks = np.linspace(0, zmax, DENSITY2D_CBAR_TICKS)
        cbar.set_ticks(ticks)
        # Round every label to a fixed number of decimals. A formatter
        # (rather than set_ticklabels) keeps matplotlib's automatic
        # tick thinning working when there are many band edges.
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.{DENSITY2D_CBAR_DECIMALS}f}")
        )
        # Adding the colorbar axes makes it current; restore the data
        # axes so a follow-up .plot() call overlays the data, not the
        # colorbar.
        plt.sca(ax)

    return filled
