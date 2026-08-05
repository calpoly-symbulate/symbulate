import itertools
import os
import warnings

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde

# Apply the package style: Okabe-Ito categorical palette (sky blue
# first), viridis sequential colormap, and the shared figure/spine/grid
# defaults. Every value that applies the same way across all plot types
# lives in this file; per-plot-type values are the named constants
# below. See DECISIONS.md, "Decision: Visual Style Guide" and
# "Decision: .mplstyle Standards".
plt.style.use(os.path.join(os.path.dirname(__file__), "symbulate.mplstyle"))

rng = np.random.default_rng()

# Discreteness budgets for the default plot lookup -- see classify_values()
# and DECISIONS.md, "Decision: Data Classification Thresholds (Budget Model)".
# A numeric variable reads as discrete-ish while its number of distinct values
# stays within a crowding budget: B_1D marks for a 1-D plot, and K_2D per axis
# for a 2-D plot's grid. Each 2-D axis is judged independently, so one axis
# over budget bins only that axis (-> a mixed tile) and both over budget bin
# both (-> a 2-D histogram). Categorical (string) data is always discrete (a
# category can't be binned); all-distinct float data is always continuous.
#
# Both budgets are anchored to the default histogram bin count (30): a discrete
# plot stays discrete exactly while it is no finer than the histogram it would
# otherwise bin into (HIST_DEFAULT_BINS / HIST2D_DEFAULT_BINS / TILE_DEFAULT_BINS,
# all 30). Provisional -- expect to tune after inspecting
# team/discrete_continuous_threshold_tests.ipynb.
B_1D = 30
K_2D = 30
# Small/large-n crossover: a sample with fewer than this many observations
# renders as small-n (show every observation) rather than large-n
# (aggregated summary). Provisional -- expect to tune alongside the budgets
# after inspecting team/discrete_continuous_threshold_tests.ipynb.
N_SMALL_THRESHOLD = 123

# Large-n secondary discreteness rule (see classify_values). A genuinely
# discrete distribution whose support overruns the crowding budget -- e.g.
# Binomial(70, 0.5) or Poisson(15) (~30-33 distinct values at n=10000, a hair
# past B_1D=30), but also Binomial(1000, 0.5) or Poisson(200) (~100-115 distinct
# values) -- should still read as discrete and draw an impulse plot. Impulse is
# the strict default for discrete data; it should give way to a histogram only
# when the support is genuinely too wide to draw one stem per value. At large n
# a discrete distribution's values recur reliably, so the fraction of distinct
# values seen more than once is a good signal -- the same repeat-density idea as
# main's old is_discrete.
#
# The reason is_discrete was replaced is that this signal, applied
# unconditionally, also fires on wide-support rounded-float or genuinely
# continuous data (rounding makes values repeat too). The clean discriminator is
# dtype: Symbulate stores genuinely discrete distributions as integers, while
# the rounded-float case is_discrete misclassified is float-dtype. So the ceiling
# on eligible distinct-value counts is split by dtype, and the clause only ever
# flips continuous -> discrete, never the reverse:
#   - INT_REPEAT_CEILING: for integer data (genuine counts), the generous cap.
#     Textbook discrete distributions stay discrete well past the budget; only a
#     truly huge support (Binomial(10000, 0.5), ~300+ distinct values) exceeds
#     it and bins into a histogram -- "histogram only when necessary."
#   - REPEAT_CEILING_FACTOR: for float data, the strict cap (this many times the
#     budget). A wide-support rounded float spread over dozens-to-hundreds of
#     values stays continuous no matter how much it repeats -- this is the guard
#     that fixes the is_discrete failure mode, left tight on purpose.
#   - REPEAT_MIN_OCCUPANCY: only when there are at least this many observations
#     per distinct value, so repeats are a reliable large-n signal rather than
#     sampling noise. Low enough that a wide-support discrete distribution near
#     its integer ceiling still qualifies at typical n (e.g. Binomial(1000, 0.5)
#     at ~90 samples/value).
#   - REPEAT_FRACTION_THRESHOLD: then discrete-ish only if more than this
#     fraction of the distinct values appear more than once (the is_discrete
#     criterion). Deliberately well below is_discrete's 0.8: a moderate-support
#     discrete distribution always has a handful of low-probability tail values
#     seen exactly once, and across thousands of Poisson(15) samples at
#     n=10000 the repeat-fraction bottomed out near 0.79 -- so a 0.6 bar clears
#     every sample with room to spare and keeps the verdict stable seed to
#     seed. Being lenient here costs nothing on the continuous side: genuinely
#     continuous data is turned away by the ceiling and occupancy bounds above,
#     not by this fraction (any variable that passes those with few enough
#     distinct values already repeats nearly all of them).
#
# All provisional -- expect to tune alongside the budgets after inspecting
# team/discrete_continuous_threshold_tests.ipynb.
INT_REPEAT_CEILING = 200
REPEAT_CEILING_FACTOR = 2
REPEAT_MIN_OCCUPANCY = 20
REPEAT_FRACTION_THRESHOLD = 0.6

# Discrete-axis tick label crowding for 2D plots (tile, segmented rug/
# density/hist/box, and violin) -- a separate crowding budget from B_1D /
# K_2D above. classify_values can let a discrete axis carry up to K_2D (30)
# distinct values, and every one of those functions used to label a
# discrete axis with one tick per distinct value; past a few dozen, the
# labels overlapped into an unreadable smear (confirmed with Poisson(200):
# 29 distinct values, 172-224, all crowded onto one axis). MAX_DISCRETE_TICKS
# caps a real-valued numeric axis's matplotlib locator (see MaxNLocator
# usage in make_tile and DECISIONS.md, "Discrete-Axis Tick Label Thinning
# (2D Plots)") and is also the fallback cap for a compacted rank-index
# axis (categorical data, or data that can't be laid out in real values --
# _thin_discrete_ticks). Labels stay horizontal, matching every other axis
# in the package -- rotation was tried and dropped, since a straight label
# reads the same as any other plot's axis, and the cap already keeps the
# count low enough to fit unrotated.
MAX_DISCRETE_TICKS = 10

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

# Impulse plot: bare stems, no marker caps.
IMPULSE_LINEWIDTH = 2.2
IMPULSE_ALPHA = 1.0
IMPULSE_LEGEND_LOC = "upper right"
# Fraction of the smallest gap between values used to separate adjacent
# series' stems when several impulse plots share an axis, so the group
# straddles each value symmetrically instead of overlapping.
IMPULSE_SERIES_OFFSET = 0.35

# Discrete Distribution.plot() (pmf) rendering: a filled dot at each pmf
# value plus a dashed dot-to-dot connecting line -- dashed so it can't be
# mistaken for a continuous curve (a pmf has no value between integers).
TRUE_DIST_MARKER_SIZE = 40
TRUE_DIST_LINEWIDTH = 1.8
TRUE_DIST_LINESTYLE = "--"

# Histogram (1D): solid bars with thin white edges so adjacent bars
# stay visually distinct. Alpha per DECISIONS.md Visual Style Guide
# (0.5 -> 0.65 -- bolder fill, histograms don't overplot).
HIST_ALPHA = 0.65
HIST_EDGECOLOR = "white"
HIST_EDGEWIDTH = 0.8
HIST_DEFAULT_BINS = 30
HIST_LEGEND_LOC = "upper right"

# Bar chart (1D categorical / discrete): one bar per distinct value, no
# binning -- the categorical counterpart of the histogram. Same solid
# fill with thin white edges so adjacent bars stay visually distinct,
# and the same alpha (bars don't overplot). Values are placed as evenly
# spaced categories, not on a numeric axis, so string labels and gappy
# integer supports both read as one bar per value.
BAR_ALPHA = 0.65
BAR_EDGECOLOR = "white"
BAR_EDGEWIDTH = 0.8
BAR_WIDTH = 0.8  # total width of one category's group of bars, in the
# category spacing (categories sit one unit apart); a lone bar chart uses
# the whole width, and overlays split it evenly between the dodged series
BAR_LEGEND_LOC = "upper right"
# Separator lines drawn between neighboring categories when two or more bar
# charts overlay, so each category's dodged group reads as its own column
# (the same tile-style boundary the dot plot draws between its stacks).
BAR_BOUNDARY_LINE_COLOR = "#b0b0b0"
BAR_BOUNDARY_LINE_WIDTH = 0.8
BAR_BOUNDARY_LINE_ALPHA = 0.6

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

# Shaded probability region under a theoretical curve
# (DistributionPlot.shade()). This is an annotation drawn on top of an
# already-plotted pmf/pdf/cdf to highlight a tail or interval, so it uses
# a fixed neutral grey rather than a color-cycle hue -- it is not a new
# data series and should not read as one. The translucent fill lets the
# underlying curve and any overlaid series remain visible through it.
SHADE_COLOR = "grey"
SHADE_ALPHA = 0.5

# Segmented density plot (a "ridgeline" plot): one small kernel
# density curve ("ridge") per level of
# the discrete variable, stacked along the discrete axis with a gentle
# overlap. All ridges in one call share a single density scale, so peak
# heights are directly comparable across levels. The KDE range reuses
# the density plot's quantile constants (DENSITY_QUANTILE_LOW / HIGH /
# PADDING_FRAC) via _density_xrange, so outliers can't stretch the
# value axis. By default each level is an unfilled curve, like the 1D
# density plot; ridge=True fills under each curve for the classic
# ridgeline look.
SEGMENTED_DENSITY_LINE_ALPHA = 1.0  # matches DENSITY_ALPHA -- the curves
# are ordinary density curves, just small
SEGMENTED_DENSITY_FILL_ALPHA = 0.4  # translucent ridge=True fill; the
# curve on top stays opaque
SEGMENTED_DENSITY_LINEWIDTH = 1.8  # ridge outlines are density curves, so this
# matches DENSITY_LINEWIDTH
# Each level's baseline is a grey line styled like a reference gridline
# -- color, width, and alpha come from the grid.* rcParams in
# symbulate.mplstyle. A blended transform makes it span the full axes and
# touch both edges (like the segmented rug plot's gridlines), and it is
# drawn on top of the ridges (zorder=2) so a ridge=True fill can't wash it
# out. The segmented histogram draws its baselines the same way, so the
# two segmented plots of one dataset read identically.
SEGMENTED_DENSITY_PEAK_SCALE = 0.9  # tallest peak's height, in units of the
# spacing between neighboring baselines; <= 1 keeps every ridge contained
# within its own segment, so no curve spills over into a neighboring level
SEGMENTED_DENSITY_GRID_POINTS = 1000  # matches DENSITY_GRID_POINTS
SEGMENTED_DENSITY_LEGEND_LOC = "upper right"
SEGMENTED_DENSITY_TICK_FRAC = 0.2  # height of the fallback tick marks drawn
# for levels too sparse for a density curve, as a fraction of the
# baseline spacing
SEGMENTED_DENSITY_TICK_LINEWIDTH = 1.0  # matches RUG_LINEWIDTH -- the fallback
# ticks are just a tiny rug on that level's baseline

# Segmented histogram: the histogram sibling of the segmented density
# plot -- one small histogram per level of the discrete variable,
# rising from that level's baseline. Every level shares one set of bin
# edges (from the pooled continuous values, HIST_DEFAULT_BINS by
# default) and one height scale, so bars line up in columns and their
# heights are directly comparable across levels. The bars use the 1D
# histogram's styling; the layout values match the segmented density
# plot so the two segmented plots of one dataset stack identically.
SEGMENTED_HIST_ALPHA = 0.65  # matches HIST_ALPHA -- the bars are
# ordinary histogram bars, just small
SEGMENTED_HIST_EDGECOLOR = "white"  # matches HIST_EDGECOLOR
SEGMENTED_HIST_EDGEWIDTH = 0.8  # matches HIST_EDGEWIDTH
SEGMENTED_HIST_PEAK_SCALE = 0.9  # matches SEGMENTED_DENSITY_PEAK_SCALE
# Each level's baseline is a grey line styled like a reference gridline
# -- color, width, and alpha come from the grid.* rcParams in
# symbulate.mplstyle. A blended transform makes it span the full axes and
# touch both edges (like the segmented rug plot's gridlines), and it is
# drawn on top of the bars (zorder=2) so the baseline stays continuous
# across each histogram instead of vanishing where the bars cover it,
# exactly like the segmented density plot.
SEGMENTED_HIST_LEGEND_LOC = "upper right"
SEGMENTED_HIST_TICK_FRAC = 0.2  # matches SEGMENTED_DENSITY_TICK_FRAC
SEGMENTED_HIST_TICK_LINEWIDTH = 1.0  # matches RUG_LINEWIDTH

# Sample path: one simulated realization of a stochastic process,
# drawn as a plain solid connected line (no per-point markers -- the
# old ".--" dot-dash format turned into a cluttered mess of dots once
# a path ran out to 100+ points). Line width is unchanged from the
# shared rcParams fallback (DECISIONS.md, Visual Style Guide: "sample
# paths stay at 1.5"); a single path has no overplotting to soften, so
# it stays fully opaque.
SAMPLE_PATH_LINEWIDTH = 1.5
SAMPLE_PATH_ALPHA = 1.0
SAMPLE_PATH_LEGEND_LOC = "upper right"

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
# Tallest single stack a dot plot can show before it stops being a good
# automatic default. A dot plot draws one dot per observation and stacks
# identical values, so a single very tall stack shrinks every dot to an
# unreadable speck -- no matter how few distinct values there are or how
# small the sample is (e.g. a rare-event indicator like Binomial(1, 0.1),
# which piles most of its mass on one value). When the tallest stack would
# exceed this many dots, the dispatch in RVResults.plot falls back to the
# configuration's large-n default (impulse for numeric, bar for categorical)
# instead. Deliberately independent of n and of the distinct-value budget
# B_1D -- it measures a third kind of crowding (stack height) those two
# don't. Anchored to the same histogram bin count (30): once one stack alone
# is taller than the whole histogram has bins, an aggregated plot reads
# better. Provisional -- tune alongside the budgets via
# team/discrete_continuous_threshold_tests.ipynb.
DOTPLOT_MAX_STACK = 30
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
# the continuous density plot (contour=False). High enough
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

# Joint plot of a theoretical multivariate distribution (the 2-D case of
# MultivariateDistribution.plot()). The theoretical counterpart of the
# density2d/tile plots above: those estimate a surface from simulated
# values, these evaluate the distribution's own pdf/pmf on a grid. The
# colormap comes from image.cmap (viridis) in symbulate.mplstyle.
JOINT_PDF_GRID_POINTS = 200  # points per axis for the continuous surface.
# Lower than DENSITY2D_GRID_POINTS (300) because every grid point is an
# exact pdf evaluation rather than one KDE lookup, and 200 matches the
# number of points the univariate Distribution.plot() curve uses.
JOINT_PDF_LEVELS = 8  # discrete color bands for contour=True, the default;
# matches
# DENSITY2D_LEVELS so a theoretical contour plot bands like a simulated one
JOINT_PDF_CONTINUOUS_LEVELS = 256  # matches DENSITY2D_CONTINUOUS_LEVELS --
# enough bands that the default surface reads as a smooth gradient
JOINT_CONTOUR_LINE_COLOR = "white"  # matches DENSITY2D_CONTOUR_LINE_COLOR
JOINT_CONTOUR_LINEWIDTH = 0.3  # matches DENSITY2D_CONTOUR_LINEWIDTH
JOINT_CONTOUR_LINE_ALPHA = 0.4  # matches DENSITY2D_CONTOUR_LINE_ALPHA
JOINT_CBAR_SIZE = "5%"  # matches DENSITY2D/TILE/HIST2D colorbar geometry
JOINT_CBAR_PAD = 0.1
JOINT_CBAR_TICKS = 8
JOINT_CBAR_DECIMALS = 3
# A discrete joint pmf is drawn as one cell per possible pair of values, so
# the grid grows with the product of the two windows. Past this many cells
# the cells are too small to read (and the pmf evaluation slow), so the
# plot reports the problem instead of rendering an unreadable mesh.
JOINT_PMF_MAX_CELLS = 40000
JOINT_OVERLAY_WARNING = (
    "Warning: you drew a second joint plot on the same plot. The new surface "
    "covers the first one and the two color scales compete, so the result may "
    "be hard to read. Consider plotting them in separate cells."
)
# A pairs=True matrix builds an n-by-n GridSpec layout, so it can neither be
# drawn onto existing axes nor overlaid afterwards -- same hard-error tier as
# MARGINAL_OVERLAY_ERROR, for the same reason.
JOINT_PAIRS_OVERLAY_ERROR = (
    "You can't combine a pairs=True plot with another plot in the same "
    "figure. The pairs layout fills the figure with its own grid of panels, "
    "which a second plot can't share. Plot it in its own cell."
)
# Above this many variables a pairs matrix has more panels than screen (9
# variables is 36 pair panels), so it asks for a subset instead of
# rendering one unreadably.
JOINT_PAIRS_MAX_DIM = 8
JOINT_PAIRS_PANEL_SIZE = 2.2  # width and height, in inches, of one panel of
# a pairs matrix. The figure is sized to the grid rather than left at the
# single-plot figure.figsize from symbulate.mplstyle, which would shrink
# every panel as the number of variables grows.
# Where a joint panel's colorbar sits inside the otherwise-empty grid cell
# mirroring it, as fractions of that cell: (left, bottom, width, height). A
# slim bar left of center, leaving room on its right for the tick labels and
# the "Density"/"Count" label.
JOINT_PAIRS_COLORBAR_INSET = (0.26, 0.08, 0.10, 0.84)
# The pair naming a colorbar ("Variable 1 & Variable 2") is a long string over
# a narrow bar, so it is set smaller than a panel title (axes.titlesize in
# symbulate.mplstyle) -- it identifies the bar rather than titling a plot, and
# at full title size it crowds its cell. It matches the bar's own
# "Density"/"Count" label (axes.labelsize), so the two halves of one bar's
# caption are lettered alike.
JOINT_PAIRS_COLORBAR_TITLE_SIZE = "medium"

# Tile plot. The colormap comes from image.cmap (viridis) in
# symbulate.mplstyle.
TILE_DEFAULT_BINS = 30  # equal-width bins for a continuous axis;
# matches make_hist2d's default so mixed discrete/continuous tiles bin
# the same way
TILE_MAX_GAP_FILL_RANGE = 2000  # a whole-number discrete axis is laid out
# in real data units, one unit-width cell per possible value across its
# observed range (see _setup_tile_axis) -- past this range, that would
# allocate a pathologically huge, mostly-empty grid (e.g. two masses at 0
# and 10000), so it falls back to compacted rank-index cells instead.
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
MOSAIC_MIN_LABEL_HEIGHT = 0.04  # a cell's count/decimal-proportion label
# is only drawn if the cell is at least this tall...
MOSAIC_MIN_LABEL_WIDTH = 0.05  # ...and at least this wide (both in
# [0, 1] axis-fraction units), so labels never crowd tiny cells
MOSAIC_LABEL_FONT_SIZE = 9
MOSAIC_LABEL_DECIMALS = 2  # decimal places for the in-cell decimal
# proportion labels (e.g. "0.18"), matching the y-axis's own 2-decimal
# tick format; raw counts (normalize=False) are always whole numbers
MOSAIC_LABEL_LUMINANCE_THRESHOLD = 0.5  # perceived luminance (ITU-R
# BT.709 weights, 0=black to 1=white) above which black label text reads
# better than white against that cell's fill color -- e.g. the palette's
# yellow and sky blue land above this line (black text), its dark blue
# and green fall below it (white text)

MOSAIC_MARGINAL_WIDTH_FRAC = 0.035  # width, in [0, 1] axis-fraction
# units, reserved for the extra reference column showing y's marginal
# distribution (see marginal_column= below) -- deliberately skinny,
# since it's a color reference to compare the real columns against, not
# real data to read counts or proportions off of (it never gets in-cell
# labels, regardless of annotate=)
MOSAIC_MARGINAL_GAP = 0.03  # gap between the real columns and the
# marginal reference column -- wider than MOSAIC_COLUMN_GAP so the break
# reads as "this one isn't a real x category" rather than just another
# column
MOSAIC_LEGEND_LABEL_GAP = 0.02  # gap, in [0, 1] axis-fraction units,
# between the marginal column's right edge and its category-name labels
# -- these double as the plot's legend (see make_mosaic's legend=)

MOSAIC_YAXIS_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]  # every column's
# segments -- real or marginal -- independently span 0 to 1 as a
# cumulative share of that column, so unlike the x-axis, one shared
# proportion scale on the left is meaningful across every column

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
VIOLIN_ALPHA = 0.5  # default body transparency for a 1D violin (the 2D
# grouped make_violin is passed its alpha explicitly by RVResults.plot)
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

# A two-variable plot builds a three-panel GridSpec layout -- the joint
# distribution plus each variable's own -- that a later .plot() call cannot
# share (a second plot would draw into whichever panel is current, silently
# corrupting the figure). This is the "hard error" tier of the overlay policy
# -- see DECISIONS.md, "Decision: Overlay Behavior".
MARGINAL_OVERLAY_ERROR = (
    "You can't draw another plot on top of a plot of two variables. It "
    "uses three separate panels -- the two variables together, plus each "
    "one on its own -- and a second plot can't share them. Plot each one "
    "in its own cell."
)
# Geometry of that three-panel layout, as a GridSpec of MARGINAL_GRID by
# MARGINAL_GRID cells: the joint panel takes all but the first row and last
# column, each variable's own distribution takes the strip beside it.
# MARGINAL_GRID_RIGHT stops the grid short of the figure's right edge so a
# colormap-based joint panel's colorbar has room out there (add_colorbar
# puts it there for this layout); the plot types that encode nothing in
# color simply leave that strip empty.
MARGINAL_GRID = 4
MARGINAL_GRID_RIGHT = 0.78
# Where that colorbar goes, as (left, bottom, width, height) fractions of the
# figure: past the y strip, aligned with the joint panel. The gap to the
# figure's right edge has to hold the bar's tick labels *and* its label, and a
# density can run to several digits (a sharply peaked one reaches into the
# hundreds), so it is wider than the labels of any one plot need.
MARGINAL_COLORBAR_RECT = (0.80, 0.11, 0.03, 0.52)
# How many tick labels a strip's frequency axis (Density / Count /
# Probability) may show. A strip is a fraction of the joint panel's size, so
# the tick count matplotlib would choose for a full-size axes crowds into
# itself there -- 0, 2, 4, 6, 8 running together. The value axis is left
# alone: it is shared with the joint panel, which sets the ticks there.
MARGINAL_FREQ_TICKS = 3


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


class DistributionPlot(SymbulatePlot):
    """Plot object returned by ``Distribution.plot()``.

    Extends :class:`SymbulatePlot` with :meth:`shade`, so a region under a
    theoretical curve is filled by chaining off the plot that drew it::

        Normal(0, 1).plot().shade(lt=-1.96)

    Shading is reachable only through a ``.plot()`` call -- there is no
    standalone ``shade`` on a distribution -- so a curve always exists
    before a region is shaded.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes the curve was drawn on.
    dist : Distribution
        The distribution whose curve was drawn; read for its ``pdf``,
        ``cdf``, and ``discrete`` flag.
    plot_type : {"pdf", "cdf"}
        Which function was drawn, so :meth:`shade` fills under the
        matching one.
    """

    def __init__(self, ax, dist, plot_type):
        super().__init__(ax)
        self.dist = dist
        self.plot_type = plot_type

    def shade(self, lt=None, le=None, gt=None, ge=None):
        """Shade a tail or interval region under the plotted curve.

        Fills the region of the distribution that satisfies the given
        inequalities, under the curve this plot drew -- the probability
        mass/density function (``cdf=False``) or the cumulative
        distribution function (``cdf=True``). Chain it onto ``plot``::

            Normal(0, 1).plot().shade(lt=-1.96)

        The bounds read as probability inequalities, mirroring the
        mathematical notation students already use:

        - ``lt`` / ``le`` -- the region ``X < lt`` / ``X <= le`` (an upper
          bound; a left tail when used alone).
        - ``gt`` / ``ge`` -- the region ``X > gt`` / ``X >= ge`` (a lower
          bound; a right tail when used alone).

        Combine a lower and an upper bound for an interval, e.g.
        ``gt=3, le=7`` shades ``3 < X <= 7``. With no bounds, the whole
        visible curve is shaded. For discrete distributions the strict
        (``lt``/``gt``) versus inclusive (``le``/``ge``) choice genuinely
        matters -- ``lt=3`` excludes the mass at ``3`` while ``le=3``
        includes it.

        The shaded region spans the currently displayed x-axis, so it
        honors whatever window :meth:`Distribution.plot` produced -- the
        full default range, an ``xlim="zoom"`` window, an explicit
        ``xlim=(low, high)``, or the union created by overlaying onto
        existing axes.

        Parameters
        ----------
        lt : float, optional
            Shade where ``X < lt`` (strict upper bound). Cannot be combined
            with ``le``.
        le : float, optional
            Shade where ``X <= le`` (inclusive upper bound). Cannot be
            combined with ``lt``.
        gt : float, optional
            Shade where ``X > gt`` (strict lower bound). Cannot be combined
            with ``ge``.
        ge : float, optional
            Shade where ``X >= ge`` (inclusive lower bound). Cannot be
            combined with ``gt``.

        Returns
        -------
        DistributionPlot
            This plot, so further regions can be shaded by chaining.

        Examples
        --------
        >>> from symbulate import *
        >>> Normal(0, 1).plot().shade(lt=-1.96)  # left tail  # doctest: +SKIP
        >>> Poisson(3).plot().shade(ge=5)  # right tail  # doctest: +SKIP
        >>> Binomial(10, 0.5).plot().shade(gt=3, le=7)  # interval  # doctest: +SKIP
        """
        if lt is not None and le is not None:
            raise ValueError(
                "Pass either `lt` or `le`, not both -- they are two ways to "
                "set the same upper bound. Use `lt` for a strict bound "
                "(X < value) or `le` for an inclusive one (X <= value)."
            )
        if gt is not None and ge is not None:
            raise ValueError(
                "Pass either `gt` or `ge`, not both -- they are two ways to "
                "set the same lower bound. Use `gt` for a strict bound "
                "(X > value) or `ge` for an inclusive one (X >= value)."
            )

        ax = self.ax
        dist = self.dist
        # Bound the region by the *displayed* window, so an open tail extends
        # to the visible edge of whatever plot() drew (xlim="zoom", a custom
        # xlim, or an overlay union), not the distribution's static self.xlim.
        axlo, axhi = ax.get_xlim()

        lo, hi = axlo, axhi
        lo_user = hi_user = False
        lo_incl = hi_incl = False
        if lt is not None:
            hi, hi_user, hi_incl = lt, True, False
        elif le is not None:
            hi, hi_user, hi_incl = le, True, True
        if gt is not None:
            lo, lo_user, lo_incl = gt, True, False
        elif ge is not None:
            lo, lo_user, lo_incl = ge, True, True

        if hi <= lo:
            raise ValueError(
                "The lower bound (`gt`/`ge`) must be strictly less than the "
                f"upper bound (`lt`/`le`). You asked to shade between {lo} and "
                f"{hi}, which is empty. Check the order of your bounds."
            )

        # Keep the drawing inside the visible window so the shading lines up
        # with the plotted curve even if a bound sits past the axis edge.
        lo_draw, hi_draw = max(lo, axlo), min(hi, axhi)

        # Evaluate the same function that is on screen: the cdf if a cdf was
        # plotted, otherwise the pmf/pdf.
        plotted_cdf = self.plot_type == "cdf"
        curve = dist.cdf if plotted_cdf else dist.pdf

        if not dist.discrete:
            # Continuous pdf or cdf: a smooth filled region under the curve.
            x_shaded = np.linspace(lo_draw, hi_draw, 200)
            ax.fill_between(
                x_shaded, curve(x_shaded), alpha=SHADE_ALPHA, color=SHADE_COLOR
            )
            # Draw a vertical edge at an inclusive, in-view user bound so the
            # closed end of the interval reads as a hard boundary.
            if hi_incl and axlo <= hi <= axhi:
                ax.vlines(hi, 0, curve(hi), color=SHADE_COLOR, alpha=SHADE_ALPHA)
            if lo_incl and axlo <= lo <= axhi:
                ax.vlines(lo, 0, curve(lo), color=SHADE_COLOR, alpha=SHADE_ALPHA)
        elif not plotted_cdf:
            # Discrete pmf: an impulse at each integer value in the region,
            # matching how plot() draws the mass function itself.
            values = np.arange(int(np.floor(axlo)), int(np.floor(axhi)) + 1)
            values = values[(values >= axlo) & (values <= axhi)]
            if hi_user:
                values = values[values <= hi] if hi_incl else values[values < hi]
            if lo_user:
                values = values[values >= lo] if lo_incl else values[values > lo]
            ax.vlines(values, 0, curve(values), color=SHADE_COLOR, alpha=SHADE_ALPHA)
        else:
            # Discrete cdf: fill under the right-continuous step function,
            # matching plot()'s "steps-post" rendering of a discrete cdf.
            x_shaded = np.linspace(lo_draw, hi_draw, 200)
            ax.fill_between(
                x_shaded,
                dist.cdf(x_shaded),
                step="post",
                alpha=SHADE_ALPHA,
                color=SHADE_COLOR,
            )

        return self


class JointDistributionPlot(SymbulatePlot):
    """Plot object returned by ``MultivariateDistribution.plot()``.

    The joint counterpart of :class:`DistributionPlot`. A joint plot shows
    a surface (or a grid of probabilities) over two variables rather than a
    curve over one, so there is no single curve to fill under: this class
    replaces :meth:`DistributionPlot.shade` with an explanation instead of
    letting the missing method surface as an ``AttributeError``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes the joint plot was drawn on. For a ``pairs=True`` matrix
        this is the bottom-left panel; every panel is reachable through
        ``ax.get_figure().axes``.
    dist : MultivariateDistribution
        The distribution that was plotted.
    dims : tuple of int
        The indices of the variables that were plotted, in axis order --
        ``(x, y)`` for a single joint plot, or every variable included for
        a ``pairs=True`` matrix.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
        The axes the joint plot was drawn on.
    dist : MultivariateDistribution
        The distribution that was plotted.
    dims : tuple of int
        The indices of the variables that were plotted.
    """

    def __init__(self, ax, dist, dims):
        super().__init__(ax)
        self.dist = dist
        self.dims = dims

    def shade(self, *args, **kwargs):
        """Shading is not available for a joint plot.

        Raises
        ------
        Exception
            Always raised. ``.shade()`` fills the region under a single
            curve, which a joint plot of two variables does not have.
        """
        raise Exception(
            "You can't shade a joint plot. Shading fills the region under a "
            "single curve, and this plot shows how two variables vary "
            "together, so there is no one curve to fill under. To shade a "
            "region, plot one variable at a time -- for example, "
            "Normal(0, 1).plot().shade(lt=-1.96)."
        )


def configure_axes(
    axes, xdata, ydata, xlabel=None, ylabel=None, orientation="vertical"
):
    """Set axis limits/labels for a value-vs-frequency plot (make_impulse).

    ``orientation="vertical"`` (default) puts ``xdata`` (the values) on
    the x-axis and ``ydata`` (frequency/count) on the y-axis, starting
    at 0 -- today's behavior, unchanged. ``orientation="horizontal"``
    swaps the two, for drawing sideways in a marginal panel: ``xdata``
    goes on the y-axis, ``ydata`` starts at 0 on the x-axis.

    Uses ``axes.set_xlim``/``set_ylim``/``set_xlabel``/``set_ylabel``
    directly rather than the ``plt.xlim``/``plt.ylabel``-style pyplot
    globals, so this works correctly even when ``axes`` isn't the
    current axes (e.g. a marginal panel drawn after the main panel).
    """
    # Create 5% buffer on either end of plot so that leftmost and rightmost
    # lines are visible. However, if current axes are already bigger,
    # keep current axes.
    data_range = max(xdata) - min(xdata)
    buff = 0.05 * data_range if data_range > 0 else 1.0

    if orientation == "vertical":
        value_lim, freq_lim = axes.get_xlim(), axes.get_ylim()
        set_value_lim, set_freq_lim = axes.set_xlim, axes.set_ylim
        set_value_label, set_freq_label = axes.set_xlabel, axes.set_ylabel
    else:
        value_lim, freq_lim = axes.get_ylim(), axes.get_xlim()
        set_value_lim, set_freq_lim = axes.set_ylim, axes.set_xlim
        set_value_label, set_freq_label = axes.set_ylabel, axes.set_xlabel

    vmin, vmax = value_lim
    vmin = min(vmin, min(xdata) - buff)
    vmax = max(vmax, max(xdata) + buff)
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    set_value_lim(vmin, vmax)

    _, fmax = freq_lim
    fmax = max(fmax, 1.05 * max(ydata))
    set_freq_lim(0, fmax)

    if xlabel is not None:
        set_value_label(xlabel)
    if ylabel is not None:
        set_freq_label(ylabel)


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


def classify_values(
    values,
    n_unique_threshold=B_1D,
    n_small_threshold=N_SMALL_THRESHOLD,
    large_n_rescue=True,
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

    The threshold is a crowding budget (see ``B_1D`` / ``K_2D`` and
    DECISIONS.md): callers pass ``B_1D`` for a 1-D plot and ``K_2D`` per
    axis for a 2-D plot. It defaults to ``B_1D``.

    A large-n secondary rule catches numeric distributions that are
    genuinely discrete but whose support overruns that budget -- e.g.
    ``Binomial(70, 0.5)`` or ``Poisson(15)`` (~30-33 distinct values at
    ``n = 10000``), and also wider ones like ``Binomial(1000, 0.5)`` or
    ``Poisson(200)`` (~100-115 distinct values). Impulse is the strict
    default for discrete data; it gives way to a histogram only when the
    support is genuinely too wide to draw one stem per value. When a sample
    has more than ``n_unique_threshold`` distinct values but still no more
    than its dtype's ceiling, has at least ``REPEAT_MIN_OCCUPANCY``
    observations per distinct value, and repeats more than
    ``REPEAT_FRACTION_THRESHOLD`` of its distinct values, it is treated as
    discrete-ish after all. This is the repeat-density idea of the old
    ``is_discrete``, but bounded so the case ``is_discrete`` misclassified
    stays continuous: the ceiling is generous for integer data
    (``INT_REPEAT_CEILING`` -- genuine counts, which Symbulate stores as
    integers) and strict for float data (``REPEAT_CEILING_FACTOR`` times the
    budget), so a wide-support rounded float stays continuous. The clause
    only ever turns a continuous verdict into a discrete one, never the
    reverse.

    Parameters
    ----------
    values : array-like
        The simulated values for one variable, e.g. ``RVResults.array``.
    n_unique_threshold : int, optional
        Largest number of distinct values that still counts as discrete-ish.
        Defaults to the module constant ``B_1D``; the 2-D dispatch passes
        ``K_2D`` per axis instead.
    n_small_threshold : int, optional
        A sample with fewer than this many values counts as small. Defaults
        to the module constant ``N_SMALL_THRESHOLD``.
    large_n_rescue : bool, optional
        Whether to apply the large-n repeat-density rescue described above
        (default ``True``). The 2-D dispatch passes ``False`` for its
        per-axis calls: there the budget model deliberately bins
        both-axes-over-``K_2D`` into a 2-D histogram rather than a huge,
        sparse tile, so the rescue -- which keeps a 1-D impulse default --
        does not apply.

    Returns
    -------
    tuple of (bool, bool)
        ``(discrete_ish, small_n)``.

    Examples
    --------
    >>> import numpy as np
    >>> classify_values(np.array([0, 1, 2, 1, 3, 2] * 2000))  # narrow int, large n
    (True, False)
    >>> classify_values(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))  # all-distinct float, small n
    (False, True)
    >>> classify_values(np.array(["H", "T", "H", "T"]))        # categorical
    (True, True)
    >>> classify_values(np.repeat(np.arange(40), 200))  # 40 distinct ints past the budget, large n, all repeat
    (True, False)
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

    # Large-n secondary rule: rescue a genuinely discrete distribution whose
    # support overran the budget, so impulse stays the default for discrete data
    # and only a genuinely too-wide support bins into a histogram. Applies only
    # when there are repeats to reason about (n_unique < n excludes all-distinct
    # data), the support stays under its dtype's ceiling, and there are enough
    # samples per distinct value that repeats are a reliable signal. Then it is
    # discrete-ish if a clear majority of its distinct values recur. This can
    # only flip a continuous verdict to discrete, never the reverse.
    #
    # The ceiling is generous for integer data (genuine counts -- e.g. Binomial,
    # Poisson -- which Symbulate stores as integers) and strict for float data,
    # where a wide-support rounded float must stay continuous. That dtype split
    # is exactly what keeps this from reintroducing the is_discrete failure mode.
    #
    # The rescue is a 1-D concern (keep the impulse default): callers pass
    # large_n_rescue=False for 2-D per-axis classification, where the budget
    # model deliberately bins both-axes-over-K_2D into a 2-D histogram rather
    # than drawing a huge, sparse tile (see DECISIONS.md).
    if data.dtype.kind in ("i", "u"):
        repeat_ceiling = INT_REPEAT_CEILING
    else:
        repeat_ceiling = REPEAT_CEILING_FACTOR * n_unique_threshold
    if (
        large_n_rescue
        and not discrete_ish
        and n_unique < n
        and n_unique <= repeat_ceiling
        and n >= REPEAT_MIN_OCCUPANCY * n_unique
    ):
        _, counts = np.unique(data, return_counts=True)
        if np.mean(counts > 1) > REPEAT_FRACTION_THRESHOLD:
            discrete_ish = True

    small_n = n < n_small_threshold
    return discrete_ish, small_n


DEFAULT_PLOT_TYPE = {
    ("1D_categorical", True): {
        "default": "dotplot",
        "alternatives": ["bar", "impulse"],
    },
    ("1D_categorical", False): {
        "default": "bar",
        "alternatives": ["dotplot", "impulse"],
    },
    ("1D_discrete", True): {
        "default": "dotplot",
        "alternatives": ["impulse", "hist", "ecdf"],
    },
    ("1D_discrete", False): {
        "default": "impulse",
        "alternatives": ["hist", "dotplot", "ecdf"],
    },
    ("1D_continuous", True): {
        "default": "rug",
        "alternatives": ["hist", "density", "box", "violin", "ecdf"],
    },
    ("1D_continuous", False): {
        "default": "hist",
        "alternatives": ["rug", "density", "box", "violin", "ecdf"],
    },
    ("2D_dd", True): {"default": "scatter", "alternatives": ["tile", "mosaic"]},
    ("2D_dd", False): {"default": "tile", "alternatives": ["scatter", "mosaic"]},
    ("2D_cc", True): {"default": "scatter", "alternatives": ["density2d", "hist2d"]},
    ("2D_cc", False): {"default": "hist2d", "alternatives": ["density2d", "scatter"]},
    # On mixed data the short names rug / hist / density resolve to the
    # segmented rug / histogram / density (RVResults.plot dispatch), so the
    # table lists the short names students actually type. The explicit
    # segmented_rug / segmented_hist / segmented_density names still work as
    # aliases.
    ("2D_mixed", True): {
        "default": "rug",
        "alternatives": ["tile", "box", "violin", "density", "hist"],
    },
    ("2D_mixed", False): {
        "default": "tile",
        "alternatives": ["rug", "box", "violin", "density", "hist"],
    },
    # Three or more variables have no single joint plot, so the default is the
    # matrix of every pair (each of whose panels is itself chosen from this
    # same table). The alternative is the connected-dot plot of each
    # realization against its index, which is what these results used to get
    # by default. Sample size doesn't change either choice -- the panels of
    # the matrix make that call variable by variable.
    ("nD", True): {"default": "pairs", "alternatives": ["path"]},
    ("nD", False): {"default": "pairs", "alternatives": ["path"]},
}


def default_plot_type(configuration, small_n):
    """Look up the default plot type and alternatives for a data configuration.

    The lookup key is ``(configuration, small_n)``. ``configuration`` is
    derived from ``classify_values`` output, the data's dtype, and its
    dimension, and is one of:

    - ``"1D_categorical"`` -- 1D string/object outcomes
    - ``"1D_discrete"`` -- 1D numeric discrete-ish
    - ``"1D_continuous"`` -- 1D continuous-ish
    - ``"2D_dd"`` -- 2D, both axes discrete-ish
    - ``"2D_cc"`` -- 2D, both axes continuous-ish
    - ``"2D_mixed"`` -- 2D, one axis discrete and one continuous (either
      order; the plot type bins/segments whichever axis is continuous)
    - ``"nD"`` -- three or more variables, which have no single joint plot

    Process time points (``X[t].sim(n)``) are ordinary 1D results, so they
    use the ``"1D_discrete"`` / ``"1D_continuous"`` configurations.

    Parameters
    ----------
    configuration : str
        A configuration key (see the list above).
    small_n : bool
        Whether the sample is small, as returned by ``classify_values``.

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
    "mosaic_equal_width": "Stacked Plot",
    "hist2d": "2D Histogram",
    "density2d": "2D Density Plot",
    "violin": "Violin Plot",
    "box": "Box Plot",
    "boxplot": "Box Plot",
    "segmented_rug": "Segmented Rug Plot",
    "segmented_density": "Segmented Density Plot",
    "segmented_hist": "Segmented Histogram",
    "pairs": "Pairs Plot",
    "path": "Path Plot",
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


def _thin_discrete_ticks(positions, labels, max_ticks):
    """Evenly spaced subset of discrete-axis tick positions and labels.

    A discrete axis labels every distinct value by default, which reads
    fine for a handful of levels but overlaps into an unreadable smear
    past a few dozen (see ``MAX_DISCRETE_TICKS`` at the top of this
    module). Above ``max_ticks``, keep only an evenly
    spaced subset of positions -- always including the first and last, so
    the axis's full range still reads -- instead of forcing every label
    onto the axis regardless of how many there are. This only thins which
    labels are *displayed*; it never changes how many cells/bands/bars are
    drawn for the underlying data (callers pass the full, untouched
    position list for drawing and only the thinned result to
    ``set_xticks``/``set_yticks``).

    Parameters
    ----------
    positions : array-like
        Tick positions (cell indices or slot positions), in axis order.
    labels : array-like
        The value to label each position with -- same length and order
        as ``positions``.
    max_ticks : int
        Largest number of labels to keep. If there are already at most
        this many, nothing is thinned.

    Returns
    -------
    tuple
        ``(positions, labels)``, thinned to at most ``max_ticks`` entries.
    """
    positions = np.asarray(positions)
    labels = np.asarray(labels)
    n = len(positions)
    if n <= max_ticks:
        return positions, labels
    keep = np.unique(np.linspace(0, n - 1, max_ticks).round().astype(int))
    return positions[keep], labels[keep]


def setup_marginal_axes(fig):
    """Build the three panels a two-variable plot is drawn on.

    A plot of two variables shows their joint distribution in a large panel,
    with each variable's own distribution in a strip beside the matching
    axis -- above for the x variable, to the right for the y variable -- so
    the joint picture and the two one-variable pictures can be read against
    each other without switching plots.

    Both the simulated (``RVResults.plot()``) and the theoretical
    (``MultivariateDistribution.plot()``) sides call this, so the two
    layouts cannot drift apart.

    The panels are returned rather than made current: the caller draws the
    joint panel first (the helpers all draw on ``plt.gca()``, so it decides
    which panel that is) and then fills the strips.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to build the layout in. It must be empty -- the layout
        fills it, so it can't be added to a figure that already has a plot
        on it.

    Returns
    -------
    tuple of matplotlib.axes.Axes
        ``(ax, ax_marg_x, ax_marg_y)`` -- the joint panel, the strip above
        it, and the strip to its right.

    Raises
    ------
    ValueError
        If the figure already has a plot on it.
    """
    # Fail the same way overlaying *onto* this layout does, rather than
    # stacking a second GridSpec on top of existing content.
    if fig.axes:
        raise ValueError(MARGINAL_OVERLAY_ERROR)
    # Tag the figure so a later .plot() call hits that same hard-error guard
    # instead of drawing into one of the three panels.
    fig._symbulate_marginal = True
    n = MARGINAL_GRID
    gs = GridSpec(n, n, right=MARGINAL_GRID_RIGHT)
    ax = fig.add_subplot(gs[1:n, 0 : n - 1])
    ax_marg_x = fig.add_subplot(gs[0, 0 : n - 1])
    ax_marg_y = fig.add_subplot(gs[1:n, n - 1])
    return ax, ax_marg_x, ax_marg_y


def thin_marginal_frequency_ticks(marg_ax, orientation, integer=False):
    """Cap how many tick labels a marginal strip's frequency axis shows.

    A strip is a fraction of the joint panel's size, so the number of ticks
    matplotlib picks for a full-size axes runs together there. Only the
    frequency axis (Density / Count / Probability) is thinned -- the value
    axis is shared with the joint panel, which owns the ticks there.

    Both the simulated and the theoretical layouts call this, so their strips
    are thinned the same way.

    Parameters
    ----------
    marg_ax : matplotlib.axes.Axes
        The strip.
    orientation : {"vertical", "horizontal"}
        How the strip was drawn. ``"vertical"`` (the strip above the joint
        panel) has its frequency on the y-axis; ``"horizontal"`` (the strip
        to the right) has it on the x-axis.
    integer : bool, default False
        Whether the frequency is a whole number, i.e. a count. Counts get
        whole-number ticks, since half a simulated value doesn't exist.
    """
    axis = marg_ax.yaxis if orientation == "vertical" else marg_ax.xaxis
    axis.set_major_locator(MaxNLocator(nbins=MARGINAL_FREQ_TICKS, integer=integer))
    # A dot plot rebuilds its own locators every time the axes is resized
    # (_dotplot_relayout, which runs on draw), so setting the locator here is
    # not enough on its own -- record the cap where that rebuild can find it.
    marg_ax._symbulate_freq_ticks = MARGINAL_FREQ_TICKS


def marginal_rug_tick_height(main_ax, marg_ax, orientation):
    """Rug tick height that looks the same in a strip as in the joint panel.

    ``make_rug`` sizes its ticks as a fraction of *its own* axes, so that a
    tick keeps its size when something with a real y-scale is drawn on the
    same axes later. In a marginal strip that backfires: the strip is a
    fraction of the joint panel's size, so the same fraction is a visibly
    shorter tick -- a marginal rug beside a segmented rug looked like a
    smaller kind of rug rather than the same one.

    Scaling by the ratio of the two panels' spans cancels that out, so the
    ticks match whatever the layout's proportions happen to be.

    Parameters
    ----------
    main_ax : matplotlib.axes.Axes
        The joint panel, whose tick size is the one to match.
    marg_ax : matplotlib.axes.Axes
        The strip the rug will be drawn in.
    orientation : {"vertical", "horizontal"}
        Which way the rug runs, so the right dimension is compared: a
        vertical rug's ticks rise, so heights are compared; a horizontal
        one's extend sideways, so widths are.

    Returns
    -------
    float
        A fraction of the strip's own axes, to pass as ``make_rug``'s
        ``tick_height``.
    """
    main, marg = main_ax.get_position(), marg_ax.get_position()
    if orientation == "vertical":
        main_span, marg_span = main.height, marg.height
    else:
        main_span, marg_span = main.width, marg.width
    if marg_span <= 0:
        return RUG_TICK_HEIGHT
    # Capped at the whole strip: a pathologically thin strip would otherwise
    # ask for ticks longer than the panel holding them.
    return min(RUG_TICK_HEIGHT * main_span / marg_span, 1.0)


def add_colorbar(fig, marginal, mappable, label, decimals=None):
    """Place a colorbar for a plot that can't hold its own.

    ``decimals``, when given, rounds the tick labels to that many decimal
    places -- the same treatment ``_joint_colorbar`` gives a joint plot's own
    bar, so a theoretical joint distribution reads the same whether or not it
    is drawn with marginal strips. Left out, matplotlib chooses, which is
    right for the count and relative-frequency scales of simulated data.
    """
    if not marginal:
        # No marginals: colorbar on the far left, label on its left.
        caxes = fig.add_axes([0, 0.1, 0.05, 0.8])
        cbar = plt.colorbar(mappable=mappable, cax=caxes)
        caxes.yaxis.set_ticks_position("left")
        cbar.set_label(label)
        caxes.yaxis.set_label_position("left")
    else:
        # Marginal layout: place the colorbar on the far right, past the
        # y-marginal panel, with its ticks and label on the right (the
        # matplotlib default). On the left it would sit on top of the main
        # panel's y-axis label; the layout leaves right-hand room for it by
        # narrowing the GridSpec (see setup_marginal_axes).
        caxes = fig.add_axes(MARGINAL_COLORBAR_RECT)
        cbar = plt.colorbar(mappable=mappable, cax=caxes)
        cbar.set_label(label)
    if decimals is not None:
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
        )
    return caxes


def add_pairs_panel_colorbar(fig, cell, mappable, pair_label, quantity_label):
    """Put one joint panel's colorbar in the empty cell mirroring it.

    A pairs matrix fills only its lower triangle, because panel ``(i, j)``
    and panel ``(j, i)`` would show the same pair twice. That leaves the
    upper triangle empty and exactly the right shape: each joint panel's
    colorbar goes in the cell across the diagonal from it, so the bar has
    room of its own instead of eating into the panel.

    Every panel keeps its own color scale -- a dense pair and a diffuse one
    are each colored over their own range -- so each bar is labeled with the
    pair it belongs to, and reading a color means reading that pair's bar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure holding the matrix.
    cell : matplotlib.transforms.Bbox
        Where the mirroring grid cell sits in the figure, from
        ``gridspec[row, col].get_position(fig)`` *after* the layout is
        settled.
    mappable : matplotlib.cm.ScalarMappable
        The surface or mesh drawn in the joint panel.
    pair_label : str
        Which two variables the panel shows, e.g. ``"X1 & X2"``.
    quantity_label : str
        What the colors measure: ``"Density"``, ``"Count"``, or
        ``"Probability"``. Counts are whole numbers, so their ticks are
        drawn without decimals.

    Returns
    -------
    matplotlib.axes.Axes
        The colorbar's axes.
    """
    left, bottom, width, height = JOINT_PAIRS_COLORBAR_INSET
    caxes = fig.add_axes(
        (
            cell.x0 + left * cell.width,
            cell.y0 + bottom * cell.height,
            width * cell.width,
            height * cell.height,
        )
    )
    cbar = plt.colorbar(mappable=mappable, cax=caxes)
    cbar.set_label(quantity_label)
    # Name the pair above its bar, so a color in the matrix can be traced to
    # the scale that explains it.
    caxes.set_title(pair_label, fontsize=JOINT_PAIRS_COLORBAR_TITLE_SIZE)
    # A count is a whole number of simulated values, so decimals on its ticks
    # would be noise; a density or a probability needs them.
    decimals = 0 if quantity_label == "Count" else JOINT_CBAR_DECIMALS
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}")
    )
    return caxes


def setup_tile_axis(values, discrete, bins):
    """Cell indices, count, extent, ticks, and edges for one tile-plot axis.

    A continuous axis is split into ``bins`` equal-width bins the same
    way ``make_hist2d`` does (``np.histogram``-style edges, with the
    largest value falling in the last bin) and positioned in real data
    units, so matplotlib's own numeric locator labels it like a numeric
    histogram axis.

    A discrete, whole-number axis (a count-style variable -- Binomial,
    Poisson, a die roll, ...) is positioned in real data units too: one
    unit-width cell per whole number across the observed range, not just
    the values that happened to occur. This means an unobserved-but-
    possible value (e.g. a count that never came up in this many
    simulations) shows as a visibly empty column instead of silently
    vanishing, and -- because the axis is now a genuine number line --
    matplotlib's own numeric locator can pick nice, evenly spaced tick
    values for it (a consistent scale), the same as the continuous case,
    instead of a hand-picked subset of whichever values happened to be
    observed. ``TILE_MAX_GAP_FILL_RANGE`` guards against a pathologically
    wide range (e.g. two masses at 0 and 10000) blowing up the cell grid.

    Categorical data (strings/bools/objects) and non-whole-number discrete
    data (repeated floats) have no natural "possible value in between" to
    fill, so they fall back to compacted rank-index cells instead, with
    tick labels thinned by the caller (see ``_thin_discrete_ticks``).

    Not tile-private: also called from ``RVResults.plot()``'s marginal-panel
    code (``results.py``) to compute the exact same per-axis index codes and
    bin edges tile itself uses, so a marginal panel can align with tile's
    coordinate system -- real values for a continuous or whole-number
    discrete axis, compacted rank-index cells for a categorical/
    non-whole-number/pathological-range discrete one (``ticks is not
    None`` signals the latter case) -- instead of drifting to its own
    scheme. See DECISIONS.md's tile/marginal coordinate-mismatch note.

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
        ``(idx, n_cells, extent, ticks, edges)`` -- the cell index of
        every value, the number of cells along the axis, the ``(low,
        high)`` imshow extent for the axis, either the full, untrimmed
        ``(positions, labels)`` for a compacted rank-index discrete axis
        or ``None`` for a real-valued axis (whole-number discrete or
        continuous) whose ticks are left to matplotlib's numeric locator,
        and either ``None`` for a discrete axis (rank-index or
        whole-number real-valued -- there's no single "bin width" for a
        one-value-per-cell axis) or the full array of bin edges for a
        continuous one (for threading the exact same edges into a
        marginal histogram).
    """
    if discrete:
        labels = np.unique(values)
        # A whole number, regardless of storage dtype -- Symbulate often
        # stores discrete outcomes as float64 (e.g. Binomial's 0.0..5.0),
        # not int.
        is_whole_number = (
            np.issubdtype(values.dtype, np.number)
            and not np.issubdtype(values.dtype, np.complexfloating)
            and np.all(labels == np.round(labels))
        )
        span = int(labels[-1] - labels[0]) + 1 if is_whole_number else None
        if is_whole_number and span <= TILE_MAX_GAP_FILL_RANGE:
            lo = int(labels[0])
            idx = (values - lo).astype(int)
            n_cells = span
            extent = (lo - 0.5, lo + span - 0.5)
            ticks = None  # matplotlib's own numeric locator runs free
        else:
            idx = np.searchsorted(labels, values)
            n_cells = len(labels)
            # imshow centers each cell on its integer index, so a cell
            # spans index +/- 0.5.
            extent = (-0.5, n_cells - 0.5)
            ticks = (np.arange(n_cells), labels)
        edges = None
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
    return idx, n_cells, extent, ticks, edges


# Discrete-axis index-position family for each main plot type that packs
# a discrete axis into integer rank slots instead of real values (see
# setup_tile_axis and DECISIONS.md's tile/marginal coordinate-mismatch
# note). Used only by RVResults.plot()'s marginal-panel rebuild
# (results.py) to compute index codes matching that axis's discrete-axis
# convention for its marginal dot/impulse plot -- main types not listed
# here (scatter, hist2d, density2d) keep the axis's real values, so no
# index code is needed for them.
#
# violin is 1-indexed (make_violin never passes its own positions= to
# ax.violinplot, so matplotlib's implicit 1..n default applies); every
# other type here is 0-indexed, matching setup_tile_axis's
# np.searchsorted(np.unique(values), values) rank (confirmed for
# segmented_rug's np.arange(len(levels)) axis limits, segmented_hist's
# and segmented_density's positions[level] = len(positions) growing
# dict on a fresh axes, and make_grouped_boxplot's
# positions = np.arange(len(levels))).
DISCRETE_INDEX_OFFSET = {
    "tile": 0,
    "segmented_rug": 0,
    "segmented_hist": 0,
    "segmented_density": 0,
    "box": 0,
    "boxplot": 0,
    "violin": 1,
}


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

    A discrete axis whose values are whole numbers (a count-style
    variable -- Binomial, Poisson, a die roll, ...) is laid out on a real
    number line: one unit-width cell per possible whole number across the
    observed range, so a value that never came up in this many
    simulations shows as a visibly empty column instead of silently
    vanishing. Because it's a genuine number line, matplotlib's own
    numeric locator picks nice, evenly spaced tick values for it (a
    consistent scale, capped at ``MAX_DISCRETE_TICKS``), exactly like the
    continuous axis case, rather than a hand-picked subset of whichever
    values happened to occur. Categorical data, non-whole-number discrete
    data, or a pathologically wide range instead falls back to compacted
    cells with a curated, evenly spaced subset of labels so the axis
    stays legible. Each axis is capped independently at
    ``MAX_DISCRETE_TICKS`` -- a consistent numeric scale on each axis is
    prioritized over forcing the exact same tick count on both.

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
        ``classify_values``, the package-wide discreteness check.
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
    # Auto-detect which axes are discrete with classify_values, the
    # package-wide discreteness check, when the caller doesn't say.
    # RVResults.plot() passes its own classify_values determination in
    # explicitly; this fallback keeps a direct make_tile() call
    # consistent with it (e.g. a wide-support integer axis is binned as
    # continuous rather than given one skinny cell per value).
    if discrete_x is None:
        discrete_x = classify_values(xs, n_unique_threshold=K_2D)[0]
    if discrete_y is None:
        discrete_y = classify_values(ys, n_unique_threshold=K_2D)[0]

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
    x_idx, nx, x_extent, x_ticks, _ = setup_tile_axis(xs, discrete_x, bins)
    y_idx, ny, y_extent, y_ticks, _ = setup_tile_axis(ys, discrete_y, bins)

    # A whole-number discrete axis (x_ticks/y_ticks is None even though
    # discrete_x/discrete_y is True) is laid out in real data units by
    # setup_tile_axis, so it's handled with the continuous axis below --
    # matplotlib's own numeric locator, capped at MAX_DISCRETE_TICKS. A
    # categorical / non-whole-number / pathologically wide-range discrete
    # axis instead falls back to compacted rank-index cells, thinned here.
    if discrete_x and x_ticks is not None:
        x_ticks = _thin_discrete_ticks(x_ticks[0], x_ticks[1], MAX_DISCRETE_TICKS)
    if discrete_y and y_ticks is not None:
        y_ticks = _thin_discrete_ticks(y_ticks[0], y_ticks[1], MAX_DISCRETE_TICKS)
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
    # A real-valued axis (continuous, or whole-number discrete) keeps
    # matplotlib's own numeric locator -- capped at MAX_DISCRETE_TICKS and
    # restricted to whole numbers for a discrete axis, so it picks nice,
    # evenly spaced tick values instead of crowding. A compacted
    # rank-index discrete axis (categorical data, or data that couldn't
    # be laid out in real values) instead gets its own pre-thinned ticks.
    if discrete_x and x_ticks is None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_DISCRETE_TICKS, integer=True))
    elif x_ticks is not None:
        ax.set_xticks(x_ticks[0])
        ax.set_xticklabels(x_ticks[1])
    if discrete_y and y_ticks is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_DISCRETE_TICKS, integer=True))
    elif y_ticks is not None:
        ax.set_yticks(y_ticks[0])
        ax.set_yticklabels(y_ticks[1])
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
    ax.set_title("Tile Plot")
    # On mixed data (exactly one discrete axis), draw separator lines on
    # the discrete axis' cell boundaries -- every integer position between
    # the axis's own extent endpoints, which are always half a cell-width
    # in from the first/last cell's center, so this works whether that
    # axis is compacted rank-index cells or real-valued whole-number
    # cells -- so each level reads as its own column or row. The lines
    # span the continuous axis and sit on top of the mesh. Vertical lines
    # when x is the discrete axis, horizontal when y is. Both-discrete and
    # both-continuous tiles are left clean.
    if discrete_x != discrete_y:
        disc_extent = x_extent if discrete_x else y_extent
        draw_line = ax.axvline if discrete_x else ax.axhline
        for boundary in np.arange(disc_extent[0] + 1, disc_extent[1]):
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


def _mosaic_spans(weights, gap, total=1.0):
    """Left edges and widths for a row of segments with fixed gaps.

    Parameters
    ----------
    weights : array-like of float
        Nonnegative, proportional sizes for each segment (need not sum
        to 1 -- they are rescaled here).
    gap : float
        Fixed gap, in the same [0, 1] units as the returned spans,
        reserved between each pair of adjacent segments.
    total : float, default 1.0
        The span the segments (plus their gaps) fill, starting at 0.
        The default fills the whole axes; a mosaic plot that reserves
        part of the axes for a marginal reference column (see
        ``make_mosaic``'s ``marginal_column=``) passes a smaller value
        here so the real columns fill only the remaining space.

    Returns
    -------
    tuple of numpy.ndarray
        ``(starts, widths)``. Segment ``i`` spans ``[starts[i],
        starts[i] + widths[i]]``; consecutive segments are separated by
        exactly ``gap``, and the whole row spans exactly ``[0, total]``.
        A segment with weight 0 gets width 0 (invisible), but still
        occupies its slot -- and its gap -- in the layout.
    """
    weights = np.asarray(weights, dtype=float)
    n = len(weights)
    total_gap = gap * max(n - 1, 0)
    available = max(total - total_gap, 0.0)
    total_weight = weights.sum()
    widths = weights / total_weight * available if total_weight > 0 else weights
    starts = np.concatenate([[0.0], np.cumsum(widths + gap)[:-1]])
    return starts, widths


def _readable_text_color(bg_color):
    """Pick a label text color that stays readable on ``bg_color``.

    Parameters
    ----------
    bg_color : color
        Any matplotlib color spec (hex string, named color, RGB tuple).

    Returns
    -------
    str
        ``"black"`` for a light background, ``"white"`` for a dark one,
        based on ``bg_color``'s perceived luminance (see
        ``MOSAIC_LABEL_LUMINANCE_THRESHOLD``).
    """
    r, g, b = mcolors.to_rgb(bg_color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > MOSAIC_LABEL_LUMINANCE_THRESHOLD else "white"


def make_mosaic(
    x,
    y,
    ax,
    normalize=True,
    annotate=True,
    legend=True,
    x_label="Variable 1",
    y_label="Variable 2",
    marginal_column=True,
    equal_width=False,
    **kwargs,
):
    """Draw a mosaic plot of simulated (x, y) pairs on the given axes.

    Divides the axes into one column per distinct ``x`` value, with
    column widths proportional to how often that value occurred (its
    marginal frequency) -- or, with ``equal_width=True``, every column
    the same width regardless of marginal frequency (a 100%-stacked bar
    chart per ``x`` value). Within each column, the column is further
    divided into one segment per distinct ``y`` value, with segment
    heights proportional to ``y``'s *conditional* frequency within that
    column. With the default proportional widths, every rectangle's area
    is therefore proportional to the joint frequency of that ``(x, y)``
    pair -- reading the segment heights across columns shows whether
    ``y``'s distribution changes with ``x`` (an association) or stays
    the same shape in every column (independence), which a
    same-color-scale plot like ``tile`` cannot show directly.
    ``equal_width=True`` trades that area-equals-joint-frequency
    property for a clean, evenly-spaced comparison across ``x``
    categories regardless of how often each one occurs -- segment
    heights (the conditional distributions) are unaffected either way.

    Segments are colored by ``y`` category, one color per distinct
    value from the package's categorical palette (Okabe-Ito, from
    ``symbulate.mplstyle``), consistent across every column. With more
    than 7 distinct ``y`` values, colors repeat -- each repeat also adds
    a hatch pattern, so two categories never look identical, and a
    warning explains why. Small gaps separate columns and the segments
    within each column so the plot reads as a mosaic of distinct tiles.
    Each in-cell label switches between black and white text (see
    ``_readable_text_color``) so it stays legible against its own
    cell's color, whichever end of the palette that is.

    With ``marginal_column=True`` (the default), one extra, deliberately
    skinny column is added after the real ``x`` columns, set off by a
    wider gap. Instead of a conditional distribution of ``y`` within one
    ``x`` value, its segments show ``y``'s *marginal* distribution --
    ``y``'s overall shape with ``x`` ignored entirely. Placed next to
    the real columns, it gives a visual baseline: a real column whose
    segments look like the marginal column suggests ``x`` and ``y`` are
    close to independent there, while one that looks different suggests
    an association -- the conditional-vs-marginal comparison a mosaic
    plot is meant to support. It's a color reference only, not real data
    to read counts off of, so it never gets in-cell labels -- instead,
    each of its segments carries its category name just to its right
    (see ``legend`` below), so the marginal column doubles as the
    plot's legend instead of a separate floating legend box. Its
    x-tick label is ``y_label`` itself (e.g. "Variable 2", the default),
    since the column represents ``y``'s own distribution.

    A shared proportion scale (0 to 1) is shown on the left of the
    axes: every column's segments, real or marginal, independently span
    0 to 1 as a cumulative share of that column, so this one scale
    applies the same way to every column.

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
        frequency conditional on that column's own ``x`` value, as a
        decimal proportion (e.g. "0.18") -- the same quantity the
        segment's height encodes, so the printed number always matches
        what's drawn. If False, labels show the raw joint count
        instead.
    annotate : bool, default True
        If True, print each cell's conditional proportion or joint
        count (see ``normalize``) inside the cell, but only when the
        cell is large enough to hold it legibly (see
        ``MOSAIC_MIN_LABEL_HEIGHT`` / ``MOSAIC_MIN_LABEL_WIDTH``). If
        False, no in-cell labels.
    legend : bool, default True
        If True, label each ``y`` category's color. With
        ``marginal_column=True`` (the default), these labels are
        printed directly beside that column's segments, so the
        marginal column doubles as the legend. With
        ``marginal_column=False``, there's no column to hang labels
        off of, so a standard legend is placed outside the right edge
        of the axes instead.
    x_label : str, default "X"
        Label for the x-axis.
    y_label : str, default "Y"
        With ``marginal_column=True`` (the default), the x-tick label
        for the marginal reference column. With
        ``marginal_column=False``, the title of the standard legend
        instead (there's no marginal column to label). Either way, the
        y-axis itself always shows the same 0-to-1 cumulative-share
        scale, independent of ``y_label``.
    marginal_column : bool, default True
        If True, add an extra column (labeled "Marginal" on the x-axis)
        showing ``y``'s overall distribution with ``x`` ignored, so it
        can be compared by eye against each real column's conditional
        distribution. If False, only the real ``x`` columns are drawn.
    equal_width : bool, default False
        If False (default), column widths are proportional to each
        ``x`` value's marginal count, the standard mosaic-plot
        convention. If True, every real column is drawn the same
        width, regardless of marginal frequency -- a 100%-stacked bar
        chart per ``x`` value instead of a mosaic plot. Only the column
        *widths* change; segment heights (conditional frequencies),
        in-cell labels, and the marginal column (if any) are computed
        from the true counts either way, so a labeled proportion is
        never misrepresented by the width scheme.
    **kwargs
        Additional keyword arguments passed to every ``ax.bar`` call
        (one per ``y`` category). For example ``linewidth=`` to
        override the default cell border width.

    Returns
    -------
    dict
        Maps each distinct ``y`` value to the ``matplotlib.container.
        BarContainer`` of its segments (one bar per real ``x`` column,
        plus one more for the marginal column when
        ``marginal_column=True``), so the caller can inspect or further
        style a specific category.

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

    # Columns: width proportional to each x value's marginal count, or
    # (equal_width=True) every column the same width regardless of
    # marginal count -- a separate weights array feeds _mosaic_spans,
    # which is itself agnostic to which scheme the caller wants. x_counts
    # itself is kept around unconditionally: the in-cell labels below
    # always report the true conditional proportion/count, never
    # something distorted by the width scheme. When marginal_column is
    # on, the real columns only fill the space left after reserving a
    # narrower column (plus a wider gap) for the y marginal reference
    # column added below.
    x_counts = joint.sum(axis=1)
    width_weights = np.ones(len(x_labels)) if equal_width else x_counts
    real_width = (
        1.0 - MOSAIC_MARGINAL_WIDTH_FRAC - MOSAIC_MARGINAL_GAP
        if marginal_column
        else 1.0
    )
    x_starts, x_widths = _mosaic_spans(
        width_weights, MOSAIC_COLUMN_GAP, total=real_width
    )
    x_positions = x_starts + x_widths / 2
    x_tick_labels = [str(v) for v in x_labels]

    n_cols = len(x_labels)
    if marginal_column:
        marginal_start = real_width + MOSAIC_MARGINAL_GAP
        x_starts = np.append(x_starts, marginal_start)
        x_widths = np.append(x_widths, MOSAIC_MARGINAL_WIDTH_FRAC)
        x_positions = np.append(
            x_positions, marginal_start + MOSAIC_MARGINAL_WIDTH_FRAC / 2
        )
        x_tick_labels.append(y_label)
        n_cols += 1

    # Rows within each column: height proportional to that column's
    # conditional frequency of each y value. Computed independently per
    # column, since each column's own total (not the grand total) is
    # what its segment heights divide. The marginal column (if any) is
    # laid out the same way, but from y's totals summed over every x
    # value instead of one column's joint counts -- its own "conditional
    # distribution" is just y's marginal distribution.
    row_starts = np.zeros((n_cols, len(y_labels)))
    row_heights = np.zeros((n_cols, len(y_labels)))
    for i in range(len(x_labels)):
        starts_i, heights_i = _mosaic_spans(joint[i, :], MOSAIC_ROW_GAP)
        row_starts[i, :] = starts_i
        row_heights[i, :] = heights_i
    if marginal_column:
        starts_m, heights_m = _mosaic_spans(joint.sum(axis=0), MOSAIC_ROW_GAP)
        row_starts[-1, :] = starts_m
        row_heights[-1, :] = heights_m

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
    text_colors = []
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
        # A hatch pattern only changes readability at the edges of a
        # cell (where the pattern's own lines sit), not its fill -- the
        # label sits at the cell's center, so only the base color
        # matters here.
        text_colors.append(_readable_text_color(color))

    if annotate:
        # Real columns only -- the marginal reference column (if any) is
        # a color-only comparison, not real data to read counts or
        # proportions off of, so it never gets in-cell labels.
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                if (
                    row_heights[i, j] < MOSAIC_MIN_LABEL_HEIGHT
                    or x_widths[i] < MOSAIC_MIN_LABEL_WIDTH
                ):
                    continue
                if normalize:
                    # Conditional on this column's own x value (matches
                    # what the segment's height encodes), not the joint
                    # frequency over the whole dataset. A decimal
                    # proportion, not a percentage, to match the
                    # y-axis's own 0-to-1 scale.
                    text = f"{joint[i, j] / x_counts[i]:.{MOSAIC_LABEL_DECIMALS}f}"
                else:
                    text = f"{int(round(joint[i, j]))}"
                ax.text(
                    x_positions[i],
                    row_starts[i, j] + row_heights[i, j] / 2,
                    text,
                    ha="center",
                    va="center",
                    fontsize=MOSAIC_LABEL_FONT_SIZE,
                    color=text_colors[j],
                )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_label)
    # Every column's segments -- real or marginal -- independently span
    # 0 to 1 as a cumulative share of that column, so (unlike the
    # x-axis) one shared proportion scale on the left is meaningful
    # across every column: a left-hand reading of "0.40" always means
    # "40% of the way up this column's total," no matter which column.
    ax.yaxis.set_visible(True)
    ax.set_yticks(MOSAIC_YAXIS_TICKS)
    ax.set_yticklabels([f"{t:.2f}" for t in MOSAIC_YAXIS_TICKS])
    ax.set_title("Stacked Plot" if equal_width else "Mosaic Plot")
    # A filled plot covers the whole axes, so the reference grid has
    # nothing to sit on -- turn it off rather than let fragments show
    # at the edges (the same reasoning make_tile / make_hist2d use).
    ax.grid(False)

    if legend:
        if marginal_column:
            # The marginal column's segments already give each y
            # category a color swatch -- print its name just to the
            # right of that swatch instead of a separate floating
            # legend box.
            for j, y_label_value in enumerate(y_labels):
                label_y = row_starts[-1, j] + row_heights[-1, j] / 2
                ax.text(
                    x_positions[-1] + x_widths[-1] / 2 + MOSAIC_LEGEND_LABEL_GAP,
                    label_y,
                    str(y_label_value),
                    ha="left",
                    va="center",
                    fontsize=MOSAIC_LABEL_FONT_SIZE,
                )
        else:
            # No marginal column to hang labels off of -- fall back to
            # a standard legend outside the right edge of the axes.
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
    # Every slot still gets its own violin; past MAX_DISCRETE_TICKS, only
    # an evenly spaced subset of the slots is labeled, mirroring the mixed
    # tile plot and segmented rug/box/density.
    slot_positions = list(range(1, len(positions) + 1))
    tick_pos, tick_lab = _thin_discrete_ticks(
        slot_positions, positions, MAX_DISCRETE_TICKS
    )
    setup_ticks(tick_pos, tick_lab, ax.xaxis if axis == "x" else ax.yaxis)
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

    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
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


def _encode_categories(values):
    """Map values to numeric x positions, keeping category labels for ticks.

    Numeric values are returned unchanged (as floats) so the plot keeps its
    real number line. Categorical values (strings/objects) have no numeric
    position, so each distinct value is assigned an evenly spaced integer
    code -- sorted (alphabetical for strings) so the same categories map to
    the same positions across overlaid calls -- and the sorted category
    labels are returned for the tick labels.

    Parameters
    ----------
    values : array-like
        The simulated values for one variable.

    Returns
    -------
    tuple of (numpy.ndarray, list or None)
        ``(codes, categories)``. ``codes`` is a float array of x positions.
        ``categories`` is the sorted list of distinct category labels when
        the data is categorical, or ``None`` when it is numeric (so the
        caller leaves the numeric axis alone).
    """
    arr = np.asarray(list(values))
    if arr.dtype.kind in "iufb":
        return arr.astype(float), None
    try:
        categories = sorted(set(arr.tolist()))
    except TypeError:
        # Mutually incomparable types (a rare mixed-type outcome): keep
        # first-seen order instead of sorting.
        categories = list(dict.fromkeys(arr.tolist()))
    code = {c: i for i, c in enumerate(categories)}
    return np.array([code[v] for v in arr.tolist()], dtype=float), categories


def make_impulse(
    values,
    ax,
    color,
    normalize=True,
    alpha=None,
    label=None,
    orientation="vertical",
    **kwargs,
):
    """Draw a 1D impulse (stem) plot of simulated discrete values.

    Stems are bare ``vlines`` (``hlines`` when horizontal) -- no marker
    caps. Impulse plots overlay naturally: a
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
        Fill color for the stems, from ``get_next_color(ax)``.
    normalize : bool, default True
        If True, stem heights are relative frequencies that sum to 1,
        comparable to a pmf. If False, stem heights are raw counts.
    alpha : float, optional
        Stem transparency between 0 and 1. Defaults to the package
        standard for impulse plots (``IMPULSE_ALPHA``, fully opaque).
    label : str, optional
        Name for this series in the legend. Defaults to "Variable k",
        where k counts the impulse plots drawn on these axes so far.
    orientation : {"vertical", "horizontal"}, default "vertical"
        "vertical" (default) draws stems rising from the x-axis, values
        on the x-axis, frequency/count on the y-axis -- today's
        behavior. "horizontal" draws stems extending from the y-axis
        instead, values on the y-axis, frequency/count on the x-axis --
        for drawing sideways in a 2D plot's y-marginal panel.
    **kwargs
        Additional keyword arguments passed to the stems
        (``matplotlib.axes.Axes.vlines`` / ``hlines``).

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
    vertical = orientation == "vertical"

    # Categorical values have no numeric position, so map them to evenly
    # spaced integer codes and remember the labels for the ticks; numeric
    # values keep their real positions (categories is None).
    codes, categories = _encode_categories(values)
    n = len(codes)
    counts = count_var(codes)
    # Sort the code positions when categorical so the stems line up with
    # the sorted tick labels; keep first-seen order for numeric data.
    xs = sorted(counts.keys()) if categories is not None else list(counts.keys())
    freqs = [counts[x] for x in xs]
    if normalize:
        freqs = [freq / n for freq in freqs]

    # Track the impulse series drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle) so overlays from separate .plot() calls see them.
    prior_series = getattr(ax, "_impulse_series", [])
    if label is None:
        label = f"Variable {len(prior_series) + 1}"

    # Label goes on the stems (a LineCollection), the only mark an
    # impulse series draws now: matplotlib's legend proxy for a
    # LineCollection is a plain line matching color and width, so the
    # legend swatch stays a clean line.
    if vertical:
        stems = ax.vlines(
            xs,
            0,
            freqs,
            color=color,
            linewidth=IMPULSE_LINEWIDTH,
            alpha=alpha,
            label=label,
            **kwargs,
        )
    else:
        stems = ax.hlines(
            xs,
            0,
            freqs,
            color=color,
            linewidth=IMPULSE_LINEWIDTH,
            alpha=alpha,
            label=label,
            **kwargs,
        )
    prior_series.append(
        {
            "stems": stems,
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
            if vertical:
                s["stems"].set_segments(
                    [[(x, 0), (x, f)] for x, f in zip(shifted_xs, s["freqs"])]
                )
            else:
                s["stems"].set_segments(
                    [[(0, x), (f, x)] for x, f in zip(shifted_xs, s["freqs"])]
                )

    configure_axes(
        ax,
        xs,
        freqs,
        xlabel="Value",
        ylabel="Relative Frequency" if normalize else "Count",
        orientation=orientation,
    )

    # Label the integer code positions with the category names, so a
    # categorical impulse plot reads as one stem per category.
    if categories is not None:
        if vertical:
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([str(c) for c in categories])
        else:
            ax.set_yticks(range(len(categories)))
            ax.set_yticklabels([str(c) for c in categories])

    ax.set_title(
        "Relative Frequency Impulse Plot" if normalize else "Count Impulse Plot"
    )
    _refresh_legend(ax)
    return xs, freqs


def make_hist(
    values,
    ax,
    color,
    bins=None,
    normalize=True,
    alpha=None,
    label=None,
    orientation="vertical",
    **kwargs,
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
    bins : int, array-like, or None, optional
        Number of equal-width bins, or a precomputed array of bin
        edges (passed straight through to ``ax.hist``, which accepts
        either). Defaults to ``HIST_DEFAULT_BINS`` (30) equal-width
        bins spanning the full range of ``values``.
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
    orientation : {"vertical", "horizontal"}, default "vertical"
        "vertical" (default) draws bars rising from the x-axis, values
        on the x-axis -- today's behavior. "horizontal" draws bars
        extending from the y-axis instead, values on the y-axis -- for
        drawing sideways in a 2D plot's y-marginal panel.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.hist``.

    Returns
    -------
    tuple
        The ``(counts, bin_edges, patches)`` tuple from ``ax.hist``, so
        the caller can inspect or further style the bars.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> values = np.random.default_rng().normal(10, 2, 1000)
    >>> ax = plt.gca()
    >>> make_hist(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
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
    if bins is None:
        bins = HIST_DEFAULT_BINS
    histogram = ax.hist(
        values,
        bins=bins,
        density=normalize,
        color=color,
        alpha=alpha,
        label=label,
        orientation=orientation,
        **kwargs,
    )
    value_label, freq_label = "Value", "Density" if normalize else "Count"
    if orientation == "vertical":
        ax.set_xlabel(value_label)
        ax.set_ylabel(freq_label)
    else:
        ax.set_ylabel(value_label)
        ax.set_xlabel(freq_label)
    ax.set_title("Density Histogram" if normalize else "Count Histogram")
    # A legend only helps once there is more than one histogram to
    # tell apart; a lone histogram stays legend-free.
    if ax._hist_count > 1:
        ax.legend(loc=HIST_LEGEND_LOC)
    return histogram


def _bar_categories(series):
    """Union of every series' distinct values, in display order.

    Sorted for a clean, stable order (numbers ascending, strings
    alphabetical); values of mutually incomparable types (a rare
    mixed-type outcome) fall back to first-seen order.
    """
    ordered = []
    seen = set()
    for s in series:
        for value in count_var(s["values"]):
            if value not in seen:
                seen.add(value)
                ordered.append(value)
    try:
        return sorted(ordered)
    except TypeError:
        return ordered


def _bar_boundary_lines(ax):
    """Draw category separator lines between the dodged groups.

    With two or more overlaid bar charts, each category holds a group of
    dodged bars, so a light vertical line halfway between neighboring
    categories makes it clear which group belongs to which value -- the
    tile-style boundary the dot plot draws between its stacks. A lone bar
    chart stays boundary-free.
    """
    state = ax._bar_state
    for line in state["boundary_lines"]:
        line.remove()
    state["boundary_lines"] = []
    if len(state["series"]) < 2:
        return
    # Categories sit at integer positions 0, 1, ..., so the boundary
    # between neighbors k and k + 1 is at k + 0.5.
    for k in range(len(state["positions"]) - 1):
        state["boundary_lines"].append(
            ax.axvline(
                k + 0.5,
                color=BAR_BOUNDARY_LINE_COLOR,
                linewidth=BAR_BOUNDARY_LINE_WIDTH,
                alpha=BAR_BOUNDARY_LINE_ALPHA,
                zorder=0,
            )
        )


def _bar_relayout(ax):
    """Redraw every bar series dodged side by side over shared categories.

    Every distinct value across all series becomes one evenly spaced
    category at an integer position. Within each category the series'
    bars are dodged side by side, together spanning ``BAR_WIDTH``, so
    overlaid charts sit next to each other rather than on top of one
    another. Called whenever a new series is added.
    """
    state = getattr(ax, "_bar_state", None)
    if state is None or not state["series"]:
        return
    series = state["series"]
    # Clear the previous bars so they can be redrawn at their new dodged
    # positions (a new series shifts every earlier series' bars).
    for s in series:
        if s["container"] is not None:
            s["container"].remove()
            s["container"] = None

    categories = _bar_categories(series)
    positions = np.arange(len(categories))
    state["positions"] = positions
    n = len(series)
    width = BAR_WIDTH / n

    for i, s in enumerate(series):
        counts = count_var(s["values"])
        heights = [counts.get(c, 0) for c in categories]
        if s["normalize"]:
            total = sum(heights)
            if total:
                heights = [h / total for h in heights]
        # Dodge each series within the category: series i sits at an
        # offset from the category center so the group straddles it
        # symmetrically. A lone series (n == 1) has offset 0.
        offset = (i - (n - 1) / 2) * width
        s["container"] = ax.bar(
            positions + offset,
            heights,
            width=width,
            color=s["color"],
            alpha=s["alpha"],
            label=s["label"],
            **s["kwargs"],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in categories])
    if len(categories) > 0:
        ax.set_xlim(-0.5, len(categories) - 0.5)
    _bar_boundary_lines(ax)


def make_bar(values, ax, color, normalize=True, alpha=None, label=None, **kwargs):
    """Draw a 1D bar chart of simulated values on the given axes.

    One bar per distinct value, sized by how often that value occurred.
    Unlike a histogram, values are never binned and never placed on a
    numeric axis: each distinct value becomes its own evenly spaced
    category, labeled with the value itself. This makes the bar chart
    the right view for categorical outcomes (e.g. ``"H"`` / ``"T"``) and
    for discrete numbers whose exact values matter, where a histogram's
    bins would blur adjacent values together.

    Drawn in the style of the histogram -- solid bars with thin white
    edges so adjacent bars stay visually distinct. The x-axis is always
    labeled "Value"; the y-axis label and title read "Relative
    Frequency" / "Bar Chart" when normalized and "Count" / "Bar Chart"
    otherwise.

    Bar charts overlay as a grouped bar chart: a second call on the same
    axes redraws every series' bars dodged side by side within each
    category (together spanning ``BAR_WIDTH``), rather than on top of one
    another, so no bar hides another. Light vertical separator lines
    appear between neighboring categories, and a legend appears
    automatically in the top right, once two or more bar charts share the
    axes. Each is named by ``label``, or "Variable 1", "Variable 2", ...
    in call order when no label is given. A lone bar chart takes the full
    category width, with no separators and no legend.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers (``make_hist``, ``make_impulse``).

    Parameters
    ----------
    values : array-like
        The simulated values to count, e.g. ``RVResults.array``. May be
        numeric or categorical (strings).
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the bars, from ``get_next_color(ax)``.
    normalize : bool, default True
        If True, bar heights are relative frequencies that sum to 1,
        comparable to a pmf (each overlaid series is normalized on its
        own). If False, bar heights are raw counts.
    alpha : float, optional
        Bar transparency between 0 and 1. Defaults to the package
        standard for bar charts (``BAR_ALPHA``, 0.65).
    label : str, optional
        Name for this bar chart in the legend. Defaults to "Variable k",
        where k counts the bar charts drawn on these axes so far.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.bar``. The bar ``width`` is managed by the
        grouped layout, so a ``width=`` keyword is ignored.

    Returns
    -------
    matplotlib.container.BarContainer
        This series' bars, as returned by ``ax.bar``, so the caller can
        inspect or further style them.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax = plt.gca()
    >>> make_bar(["H", "T", "H", "H", "T"], ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = BAR_ALPHA
    # The white bar edges are defaults, not overrides, so a user's own
    # edgecolor= / linewidth= keyword still wins (the same pattern
    # make_hist uses for edgecolor=). The bar width is owned by the
    # grouped layout -- it splits BAR_WIDTH between the dodged series --
    # so drop any width the caller passed.
    kwargs.setdefault("edgecolor", BAR_EDGECOLOR)
    kwargs.setdefault("linewidth", BAR_EDGEWIDTH)
    kwargs.pop("width", None)

    # Series live on the axes object itself (the same pattern
    # get_next_color uses for the color cycle) so overlays from separate
    # .plot() calls share one grouped layout.
    state = getattr(ax, "_bar_state", None)
    if state is None:
        state = {"series": [], "boundary_lines": [], "positions": np.array([])}
        ax._bar_state = state
    if label is None:
        label = f"Variable {len(state['series']) + 1}"
    state["series"].append(
        {
            "values": values,
            "color": color,
            "label": label,
            "normalize": normalize,
            "alpha": alpha,
            "kwargs": kwargs,
            "container": None,
        }
    )

    _bar_relayout(ax)
    ax.set_xlabel("Value")
    ax.set_ylabel("Relative Frequency" if normalize else "Count")
    ax.set_title("Bar Chart")
    # A legend only helps once there is more than one bar chart to tell
    # apart; a lone bar chart stays legend-free.
    if len(state["series"]) > 1:
        ax.legend(loc=BAR_LEGEND_LOC)
    return state["series"][-1]["container"]


def make_boxplot(values, ax, color, alpha=None, label=None, outliers=True, **kwargs):
    """Draw a box plot of simulated values on the given axes.

    A single box: edges at the first and third quartiles, a black
    median line, whiskers to the most extreme value within 1.5 times
    the interquartile range, and individual points beyond that drawn
    as outliers. Pass ``outliers=False`` to instead extend the
    whiskers all the way to the minimum and maximum values, with no
    individual outlier points. Non-finite values (e.g. NaN) are
    dropped before plotting.

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
    outliers : bool, optional
        If True (default), the whiskers stop at the most extreme
        value within 1.5 times the interquartile range, and any
        points beyond that are drawn individually as outliers. If
        False, the whiskers extend to the minimum and maximum
        values, so every point falls inside the whiskers and no
        individual outlier points are drawn.
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
    # outliers=False stretches the whiskers to the 0th and 100th
    # percentiles (the data's min and max), leaving nothing beyond
    # them to draw as an individual outlier point. A default, not an
    # override, so a user's own whis= keyword still wins.
    if not outliers:
        kwargs.setdefault("whis", (0, 100))
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


def make_violinplot(values, ax, color, alpha=None, label=None, **kwargs):
    """Draw a single violin plot of 1D simulated values on the given axes.

    One violin: a mirrored kernel density of the values, with a narrow
    inner box plot (pale ivory box, black median line and whiskers) marking
    the median and interquartile range. This is the 1D, single-variable
    counterpart of ``make_violin`` (which draws one violin per level of a
    discrete grouping variable for 2D data), the same way ``make_boxplot``
    is the single-box counterpart of ``make_grouped_boxplot``. Non-finite
    values (e.g. NaN) are dropped before plotting.

    Violin plots overlay side by side: a second call on the same axes adds
    another violin at the next position, and both violins' x-ticks are
    labeled automatically -- "Variable 1", "Variable 2", ... in call order,
    or the given ``label``. Placing them side by side (rather than on top of
    one another) keeps overlapping density shapes readable, matching how
    ``make_boxplot`` overlays.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as ``color``.
    This mirrors how ``RVResults.plot()`` calls the other plot helpers
    (``make_boxplot``, ``make_hist``).

    Parameters
    ----------
    values : array-like
        The simulated values to summarize, e.g. ``RVResults.array``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Fill color for the violin body, from ``get_next_color(ax)``.
    alpha : float, optional
        Body transparency between 0 and 1. Defaults to the package
        standard for violins (``VIOLIN_ALPHA``, 0.5).
    label : str, optional
        Name for this violin, shown as its x-tick label. Defaults to
        "Variable k", where k counts the violins drawn on these axes so
        far.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.violinplot``.

    Returns
    -------
    dict
        The dict of Matplotlib artists returned by ``ax.violinplot``, so
        the caller can inspect or further style the body.

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
    >>> make_violinplot(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = VIOLIN_ALPHA
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError(
            "There are no values to plot. Simulate some values first, "
            "for example X.sim(30).plot(type='violin')."
        )
    # Count the violins drawn on these axes, stored on the axes object
    # itself (the same pattern make_boxplot uses) so overlays from
    # separate .plot() calls sit side by side at successive positions.
    position = getattr(ax, "_violinplot_count", 0) + 1
    if label is None:
        label = f"Variable {position}"
    ax._violinplot_count = position

    violins = ax.violinplot(
        dataset=[values],
        positions=[position],
        widths=VIOLIN_WIDTH,
        showmedians=False,
        showextrema=False,
        **kwargs,
    )
    for body in violins["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(VIOLIN_EDGECOLOR)
        body.set_linewidth(VIOLIN_EDGEWIDTH)
        body.set_alpha(alpha)

    # The inner box plot marks the median and IQR on top of the density
    # shape. manage_ticks=False so it doesn't clobber the tick labels set
    # below (the same approach make_violin uses for its inner boxes).
    box_width = VIOLIN_WIDTH * VIOLIN_BOX_WIDTH_RATIO
    boxplot = ax.boxplot(
        values,
        positions=[position],
        widths=box_width,
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
    for artists in boxplot.values():
        for artist in artists:
            artist.set_zorder(3)

    # Label each violin's position on the x-axis, accumulating across
    # overlaid calls so every violin keeps its own tick label.
    labels = getattr(ax, "_violinplot_labels", [])
    labels.append((position, label))
    ax._violinplot_labels = labels
    ax.set_xticks([pos for pos, _ in labels])
    ax.set_xticklabels([lab for _, lab in labels])
    ax.set_ylabel("Value")
    ax.set_title("Violin Plot")
    return violins


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
    values,
    ax,
    color,
    bandwidth=None,
    alpha=None,
    label=None,
    orientation="vertical",
    **kwargs,
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
    orientation : {"vertical", "horizontal"}, default "vertical"
        "vertical" (default) plots density against value on the
        x-axis -- today's behavior. "horizontal" swaps them (value on
        the y-axis) -- for drawing sideways in a 2D plot's y-marginal
        panel.
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

    vertical = orientation == "vertical"
    line = ax.plot(
        grid if vertical else density,
        density if vertical else grid,
        color=color,
        alpha=alpha,
        label=label,
        **kwargs,
    )
    if vertical:
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel("Value")
        ax.set_xlabel("Density")
        ax.set_xlim(left=0)
    ax.set_title("Density Curve")
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


def make_rug(
    values,
    ax,
    color,
    alpha=None,
    label=None,
    orientation="vertical",
    tick_height=None,
    **kwargs,
):
    """Draw a rug plot of simulated values on the given axes.

    Draws one thin vertical tick per simulated value along the bottom
    of the axes. Tick heights are drawn in axes fractions
    (``tick_height``, defaulting to ``RUG_TICK_HEIGHT`` of the axes
    height), not data units, so the ticks keep their size if something
    with a meaningful y-scale (e.g. a histogram) is drawn on the same
    axes later.

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
    orientation : {"vertical", "horizontal"}, default "vertical"
        "vertical" (default) draws ticks rising from the bottom of the
        axes, values on the x-axis -- today's behavior. "horizontal"
        draws ticks extending from the left edge instead, values on
        the y-axis -- for drawing sideways in a 2D plot's y-marginal
        panel.
    tick_height : float, optional
        How long the ticks are, as a fraction of the axes. Defaults to
        ``RUG_TICK_HEIGHT``. Like ``orientation``, this exists for the
        marginal strips of a 2D plot rather than as a style knob: a strip
        is smaller than the joint panel, so the same fraction there is a
        physically shorter tick -- ``marginal_rug_tick_height`` computes
        the value that makes the two match.
    **kwargs
        Additional keyword arguments passed to
        ``matplotlib.axes.Axes.vlines`` (or ``.hlines`` when
        ``orientation="horizontal"``).

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
    vertical = orientation == "vertical"
    # Count the rugs drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color
    # cycle) so overlays from separate .plot() calls see it.
    n_prior_rugs = getattr(ax, "_rug_count", 0)
    if label is None:
        label = f"Variable {n_prior_rugs + 1}"
    ax._rug_count = n_prior_rugs + 1

    if tick_height is None:
        tick_height = RUG_TICK_HEIGHT

    # Check if this is a standalone rug plot (nothing else on the axes
    # yet) or an overlay on another plot type.
    is_standalone = (
        len(ax.patches) == 0 and len(ax.lines) == 0 and len(ax.collections) == 0
    )

    if vertical:
        rug = ax.vlines(
            np.asarray(values),
            0,
            tick_height,
            # Axes-fraction y coordinates: ticks rise from the bottom of
            # the axes regardless of the y data limits.
            transform=ax.get_xaxis_transform(),
            color=color,
            alpha=alpha,
            label=label,
            **kwargs,
        )
    else:
        rug = ax.hlines(
            np.asarray(values),
            0,
            tick_height,
            # Axes-fraction x coordinates: ticks extend from the left
            # edge of the axes regardless of the x data limits.
            transform=ax.get_yaxis_transform(),
            color=color,
            alpha=alpha,
            label=label,
            **kwargs,
        )

    if is_standalone:
        # Standalone rug plot: the direction with no information (y for
        # a vertical rug, x for a horizontal one) gets hidden -- axis,
        # matching spine, and gridlines -- for a clean number-line look.
        # When the rug is overlaid on another plot, none of this runs --
        # the companion plot owns the axes styling and the rug inherits
        # it untouched.
        if vertical:
            ax.yaxis.set_visible(False)
            ax.spines["left"].set_visible(False)
        else:
            ax.xaxis.set_visible(False)
            ax.spines["bottom"].set_visible(False)
        ax.grid(False)

    if vertical:
        ax.set_xlabel("Value")
    else:
        ax.set_ylabel("Value")
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


def make_sample_path(
    times,
    values,
    ax,
    color,
    linewidth=None,
    alpha=None,
    label=None,
    xlabel=None,
    ylabel=None,
    **kwargs,
):
    """Draw one simulated sample path as a plain connected line.

    This replaces the ``plt.plot(ts, ys, ".--", **kwargs)`` calls
    previously duplicated across ``Tuple.plot()``,
    ``InfiniteVector.plot()``, ``DiscreteTimeFunction.plot()``, and
    ``ContinuousTimeFunction.plot()`` in ``result.py``: the dot-dash
    marker is dropped in favor of a plain solid line (readable whether
    the path has 10 points or 1000), and the package's color cycle,
    axis labels, and title are added.

    Sample paths overlay naturally: a second call on the same axes
    draws on top of the first, and a legend appears automatically in
    the top right once two or more paths share the axes -- the common
    case of comparing several realizations of the same process. Each
    path is named by ``label``, or "Path 1", "Path 2", ... in call
    order when no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``.

    Parameters
    ----------
    times : array-like
        The time or index value of each point along the path, e.g.
        ``range(len(self))`` for a ``Tuple`` or ``np.linspace(tmin,
        tmax, 200)`` for a ``ContinuousTimeFunction``.
    values : array-like
        The simulated value at each entry of ``times``, same length as
        ``times``.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    color : color
        Line color, from ``get_next_color(ax)``.
    linewidth : float, optional
        Line width. Defaults to the package standard for sample paths
        (``SAMPLE_PATH_LINEWIDTH``, 1.5 -- unchanged from matplotlib's
        own default).
    alpha : float, optional
        Line transparency between 0 and 1. Defaults to the package
        standard for sample paths (``SAMPLE_PATH_ALPHA``, fully
        opaque).
    label : str, optional
        Name for this path in the legend. Defaults to "Path k", where
        k counts the sample paths drawn on these axes so far.
    xlabel : str, optional
        Label for the x-axis. Defaults to "Time".
    ylabel : str, optional
        Label for the y-axis. Defaults to "Value".
    **kwargs
        Additional keyword arguments passed to ``matplotlib.axes.Axes.plot``.

    Returns
    -------
    matplotlib.lines.Line2D
        The line drawn by ``ax.plot``, so the caller can inspect or
        further style it.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> rng = np.random.default_rng()
    >>> times = np.arange(101)
    >>> values = np.concatenate([[0], np.cumsum(rng.choice([-1, 1], size=100))])
    >>> ax = plt.gca()
    >>> make_sample_path(times, values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if linewidth is None:
        linewidth = SAMPLE_PATH_LINEWIDTH
    if alpha is None:
        alpha = SAMPLE_PATH_ALPHA

    # Count the sample paths drawn on these axes, stored on the axes
    # object itself (the same pattern get_next_color uses for the
    # color cycle) so overlays from separate .plot() calls see it.
    n_prior_paths = getattr(ax, "_sample_path_count", 0)
    if label is None:
        label = f"Path {n_prior_paths + 1}"
    ax._sample_path_count = n_prior_paths + 1

    (line,) = ax.plot(
        times,
        values,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        **kwargs,
    )

    ax.set_xlabel("Time" if xlabel is None else xlabel)
    ax.set_ylabel("Value" if ylabel is None else ylabel)
    ax.set_title("Sample Path")
    _refresh_legend(ax, loc=SAMPLE_PATH_LEGEND_LOC)

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
        None (default), determined by ``classify_values``, the
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
    # Fall back to classify_values, the package-wide discreteness check,
    # when the caller doesn't specify. RVResults.plot() passes its own
    # classify_values determination in explicitly; this keeps a direct
    # make_segmented_rug() call consistent with it.
    if discrete_x is None:
        discrete_x = classify_values(xs, n_unique_threshold=K_2D)[0]
    if discrete_y is None:
        discrete_y = classify_values(ys, n_unique_threshold=K_2D)[0]

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
    # match the mixed tile plot's "X"/"Y". Every level still gets its own
    # band (drawn above using the full, untouched `positions`); past
    # MAX_DISCRETE_TICKS, only an evenly spaced subset of the bands is
    # labeled, mirroring the mixed tile plot.
    positions = np.arange(len(levels))
    if discrete_y:
        tick_pos, tick_lab = _thin_discrete_ticks(positions, levels, MAX_DISCRETE_TICKS)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)
        # Minor ticks at every band, so a gridline can mark each distinct
        # rug even where the (major) label was thinned away above.
        ax.set_yticks(positions, minor=True)
        ax.set_ylim(-0.5, len(levels) - 0.5)
    else:
        tick_pos, tick_lab = _thin_discrete_ticks(positions, levels, MAX_DISCRETE_TICKS)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.set_xticks(positions, minor=True)
        ax.set_xlim(-0.5, len(levels) - 0.5)
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
    ax.set_title("Segmented Rug Plot")
    # Reference gridlines run along the discrete axis only -- one line per
    # level (band), so every distinct rug reads as its own group, even the
    # ones whose tick label was thinned out to avoid crowding. which="both"
    # draws at the labeled (major) bands and the thinned (minor) bands
    # alike; both pick up the grid style from symbulate.mplstyle. The minor
    # tick marks themselves are hidden, so only gridlines are added and the
    # thinned-label look is kept. axisbelow keeps the grid behind the rug
    # ticks; the continuous value axis stays clean like the standalone rug.
    discrete_axis = "x" if discrete_x else "y"
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, which="both", axis=discrete_axis)
    ax.tick_params(axis=discrete_axis, which="minor", length=0)
    return ticks


def make_segmented_density(
    x,
    y,
    ax,
    color,
    bandwidth=None,
    ridge=False,
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

    By default each level is an unfilled curve, like the 1D density
    plot. ``ridge=True`` fills the area under every curve with a
    translucent wash of the same color -- the classic ridgeline look.

    Levels with fewer than two distinct values cannot support a density
    estimate; their observations are drawn as short tick marks on that
    level's baseline instead (a tiny rug), and a note explains which
    levels fell back and why.

    Each level's curve sits on its own baseline: a grey line spanning
    the full axes and touching both edges -- one per level, matching the
    segmented rug plot's gridlines. It is drawn on top of the ridges so
    it stays continuous across every level, even under a ``ridge=True``
    fill. A light reference grid also runs along the continuous axis, for
    reading values off the curves.

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
        Color for the ridges, from ``get_next_color(ax)``.
    bandwidth : float or str, optional
        Passed through to ``scipy.stats.gaussian_kde`` as ``bw_method``
        for every ridge. Defaults to scipy's own default (Scott's
        rule).
    ridge : bool, optional
        If False (default), each level is an unfilled density curve,
        like the 1D density plot. If True, the area under every curve
        is filled with a translucent wash of the same color (the
        classic ridgeline look); the curve on top stays opaque.
    alpha : float, optional
        Transparency between 0 and 1. With ``ridge=False`` it applies
        to the curves and defaults to fully opaque
        (``SEGMENTED_DENSITY_LINE_ALPHA``, 1.0, like the 1D density
        plot); with ``ridge=True`` it applies to the fills and defaults
        to ``SEGMENTED_DENSITY_FILL_ALPHA`` (0.4), while the curves
        stay opaque so the shapes read clearly where ridges overlap.
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
        Additional keyword arguments passed to ``ax.plot`` for the
        density curves. ``linewidth`` is applied with ``setdefault``
        (``SEGMENTED_DENSITY_LINEWIDTH``), so an explicit keyword
        argument wins; the ``ridge=True`` fill styling comes from the
        module constants.

    Returns
    -------
    list
        The drawn artists, one per level in baseline order -- a
        ``Line2D`` for each density curve (or a ``PolyCollection`` for
        each fill when ``ridge=True``), or a ``LineCollection`` for
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
    # alpha means different things in the two modes: the fill's
    # transparency when ridge=True (the curve on top stays opaque), the
    # curve's own transparency when ridge=False (matching the 1D
    # density plot's fully-opaque default).
    if alpha is None:
        alpha = SEGMENTED_DENSITY_FILL_ALPHA if ridge else SEGMENTED_DENSITY_LINE_ALPHA
    line_alpha = SEGMENTED_DENSITY_LINE_ALPHA if ridge else alpha
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

    # The curve line's width is a default, not an override, so a user's
    # own linewidth= keyword still wins (the same pattern make_hist
    # uses).
    kwargs.setdefault("linewidth", SEGMENTED_DENSITY_LINEWIDTH)

    # Draw from the highest baseline down so that where ridges overlap,
    # the lower (nearer) ridge sits in front -- the classic ridgeline
    # look.
    artists = {}
    for level in sorted(levels, key=positions.get, reverse=True):
        base = positions[level]
        batch_label = label if not artists else None
        # Each level's baseline is a grey line styled like a reference
        # gridline (grid.* rcParams from symbulate.mplstyle). The blended
        # transform makes it span the full axes and touch both edges --
        # 0 and 1 are axes fractions along the value axis, base is in data
        # units on the discrete axis -- like the segmented rug plot's
        # gridlines. zorder=2 keeps it on top of the ridges so a
        # ridge=True fill can't wash it out and it stays continuous across
        # every level. The segmented histogram draws its baselines the
        # same way.
        base_line = ax.hlines if discrete_y else ax.vlines
        base_transform = (
            ax.get_yaxis_transform() if discrete_y else ax.get_xaxis_transform()
        )
        base_line(
            base,
            0,
            1,
            transform=base_transform,
            color=plt.rcParams["grid.color"],
            linewidth=plt.rcParams["grid.linewidth"],
            alpha=plt.rcParams["grid.alpha"],
            zorder=2,
        )
        if level in densities:
            heights = densities[level] * scale
            if ridge:
                # The classic ridgeline look: a translucent wash under
                # the curve. No edge stroke -- the opaque curve drawn
                # below outlines the top, and the grey shelf is the
                # base.
                fill = ax.fill_between if discrete_y else ax.fill_betweenx
                fill(
                    grid,
                    base,
                    base + heights,
                    facecolor=mcolors.to_rgba(color, alpha),
                    edgecolor="none",
                )
            curve_xy = (grid, base + heights)
            if not discrete_y:
                curve_xy = curve_xy[::-1]
            (artists[level],) = ax.plot(
                *curve_xy,
                color=color,
                alpha=line_alpha,
                label=batch_label,
                **kwargs,
            )
        else:
            # Too sparse for a density estimate: a tiny rug rising from
            # the level's baseline keeps the level (and its data)
            # visible.
            values = continuous[groups == level]
            lines = ax.vlines if discrete_y else ax.hlines
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
    # Every level still gets its own baseline and ridge (drawn above using
    # the full, untouched `ticks`); past MAX_DISCRETE_TICKS, only an
    # evenly spaced subset of the baselines is labeled, mirroring the
    # mixed tile plot and segmented rug.
    if discrete_y:
        tick_pos, tick_lab = _thin_discrete_ticks(ticks, all_levels, MAX_DISCRETE_TICKS)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)
        ax.set_ylim(lo, hi)
    else:
        tick_pos, tick_lab = _thin_discrete_ticks(ticks, all_levels, MAX_DISCRETE_TICKS)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.set_xlim(lo, hi)
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
    ax.set_title("Segmented Density Plot")
    # A light reference grid along the continuous axis only, for reading
    # values off the curves. The discrete axis needs no grid line: each
    # level's baseline (the full-width grey line drawn on top above)
    # already marks it. axisbelow keeps the value grid behind the ridges;
    # its color and width come from symbulate.mplstyle. Mirrors the
    # segmented histogram exactly.
    ax.set_axisbelow(True)
    # Clear the style sheet's default single-axis grid first, then enable
    # it on the continuous axis only.
    ax.grid(False)
    ax.grid(True, axis="x" if discrete_y else "y")
    # A legend only helps once there is more than one batch to tell
    # apart; a lone batch stays legend-free.
    if ax._segmented_density_count > 1:
        ax.legend(loc=SEGMENTED_DENSITY_LEGEND_LOC)
    return [artists[level] for level in all_levels if level in artists]


def make_segmented_hist(
    x,
    y,
    ax,
    color,
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

    Each level's histogram sits on its own baseline: a grey line
    spanning the full axes and touching both edges -- one per level,
    matching the segmented rug plot's gridlines. It is drawn on top of
    the bars so it stays continuous across each histogram instead of
    vanishing where the bars cover it. A light reference grid also runs
    along the continuous axis for reading values off the bars.

    Segmented histograms overlay naturally: a second call on the same
    axes draws a second set of histograms on the same baselines (levels
    are matched by value, and levels new to the axes get new
    baselines), and a legend appears automatically once two or more
    batches share the axes. Each batch is named by ``label``, or
    "Variable 1", "Variable 2", ... in call order when no label is
    given. Each batch is binned on its own shared edges and scaled to
    its own tallest bar, so overlays compare shapes, not absolute
    heights.

    This plot is only for mixed data -- exactly one discrete variable
    and one continuous variable. Two discrete variables should use a
    tile plot, and two continuous variables a scatter plot.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how the other plot helpers here
    (``make_segmented_density``, ``make_hist``) are called.

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
        Color for the bars, from ``get_next_color(ax)``.
    bins : int or sequence, optional
        Number of equal-width bins (or explicit bin edges) for the
        shared bin grid, passed to ``numpy.histogram_bin_edges`` over
        the pooled continuous values. Defaults to the package standard
        for histograms (``HIST_DEFAULT_BINS``, 30).
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
        for the bars. The bar styling (``facecolor``, ``edgecolor``,
        ``linewidth``) is applied with ``setdefault``, so explicit
        keyword arguments win.

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
    >>> ax = plt.gca()
    >>> make_segmented_hist(x, y, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = SEGMENTED_HIST_ALPHA
    if bins is None:
        bins = HIST_DEFAULT_BINS
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

    # The bars use the 1D histogram's styling -- translucent fill, thin
    # white edges -- so a segmented histogram reads as a stack of
    # ordinary small histograms. These are defaults, not overrides, so
    # a user's own facecolor= / edgecolor= / linewidth= keyword still
    # wins (the same pattern make_hist uses).
    kwargs.setdefault("facecolor", mcolors.to_rgba(color, alpha))
    kwargs.setdefault("edgecolor", SEGMENTED_HIST_EDGECOLOR)
    kwargs.setdefault("linewidth", SEGMENTED_HIST_EDGEWIDTH)

    # Draw from the highest baseline down so that where bars overlap
    # the row above, the lower (nearer) histogram sits in front -- the
    # same order as the segmented density plot.
    artists = {}
    for level in sorted(levels, key=positions.get, reverse=True):
        base = positions[level]
        batch_label = label if not artists else None
        # Each level's baseline is a grey line styled like a reference
        # gridline (grid.* rcParams from symbulate.mplstyle). The blended
        # transform makes it span the full axes and touch both edges --
        # 0 and 1 are axes fractions along the value axis, base is in data
        # units on the discrete axis -- like the segmented rug plot's
        # gridlines. zorder=2 keeps it on top of the bars so the baseline
        # stays continuous across each histogram instead of vanishing
        # where the bars cover it. Mirrors the segmented density plot.
        base_line = ax.hlines if discrete_y else ax.vlines
        base_transform = (
            ax.get_yaxis_transform() if discrete_y else ax.get_xaxis_transform()
        )
        base_line(
            base,
            0,
            1,
            transform=base_transform,
            color=plt.rcParams["grid.color"],
            linewidth=plt.rcParams["grid.linewidth"],
            alpha=plt.rcParams["grid.alpha"],
            zorder=2,
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
    # Every level still gets its own baseline and bars (drawn above using
    # the full, untouched `ticks`); past MAX_DISCRETE_TICKS, only an
    # evenly spaced subset of the baselines is labeled, mirroring the
    # segmented density plot.
    if discrete_y:
        tick_pos, tick_lab = _thin_discrete_ticks(ticks, all_levels, MAX_DISCRETE_TICKS)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)
        ax.set_ylim(lo, hi)
    else:
        tick_pos, tick_lab = _thin_discrete_ticks(ticks, all_levels, MAX_DISCRETE_TICKS)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        ax.set_xlim(lo, hi)
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
    ax.set_title("Segmented Histogram")
    # A light reference grid along the continuous axis only, for reading
    # values off the bars. The discrete axis needs no grid line: each
    # level's baseline (the full-width grey line drawn on top above)
    # already marks it. axisbelow keeps the value grid behind the bars;
    # its color and width come from symbulate.mplstyle.
    ax.set_axisbelow(True)
    # Clear the style sheet's default single-axis grid first, then enable
    # it on the continuous axis only.
    ax.grid(False)
    ax.grid(True, axis="x" if discrete_y else "y")
    # A legend only helps once there is more than one batch to tell
    # apart; a lone batch stays legend-free.
    if ax._segmented_hist_count > 1:
        ax.legend(loc=SEGMENTED_HIST_LEGEND_LOC)
    return [artists[level] for level in all_levels if level in artists]


def make_grouped_boxplot(
    x,
    y,
    ax,
    color,
    alpha=None,
    discrete_x=None,
    discrete_y=None,
    outliers=True,
    **kwargs,
):
    """Draw a box plot for mixed discrete/continuous data.

    One box per distinct value of whichever variable is discrete, using
    the other (continuous) variable as the value axis -- the box-plot
    counterpart of ``make_violin`` for this same data configuration.
    Each box's whiskers stop at the most extreme value within 1.5
    times the interquartile range, with more extreme points drawn
    individually as outliers; pass ``outliers=False`` to instead
    extend the whiskers to each group's minimum and maximum values.

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
    outliers : bool, optional
        If True (default), each box's whiskers stop at the most
        extreme value within 1.5 times the interquartile range, and
        any points beyond that are drawn individually as outliers.
        If False, the whiskers extend to each group's minimum and
        maximum values, so every point falls inside the whiskers and
        no individual outlier points are drawn.
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
    # outliers=False stretches the whiskers to the 0th and 100th
    # percentiles (each group's min and max), leaving nothing beyond
    # them to draw as an individual outlier point. A default, not an
    # override, so a user's own whis= keyword still wins.
    if not outliers:
        kwargs.setdefault("whis", (0, 100))

    boxes = ax.boxplot(data, positions=positions, orientation=orientation, **kwargs)

    # Label the discrete axis with the level values (one tick per box);
    # the continuous axis keeps matplotlib's numeric ticks. Axis labels
    # match the mixed tile plot's and make_segmented_rug's "X"/"Y". Every
    # level still gets its own box (drawn above using the full, untouched
    # `positions`); past MAX_DISCRETE_TICKS, only an evenly spaced subset
    # of the boxes is labeled.
    if discrete_y:
        tick_pos, tick_lab = _thin_discrete_ticks(positions, levels, MAX_DISCRETE_TICKS)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab)
    else:
        tick_pos, tick_lab = _thin_discrete_ticks(positions, levels, MAX_DISCRETE_TICKS)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
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


def _y_span_px(ax, dy):
    """Return how many pixels tall dy data units are."""
    (_, y0), (_, y1) = ax.transData.transform([(0.0, 0.0), (0.0, dy)])
    return y1 - y0


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


def _dotplot_init_state(ax, orientation="vertical"):
    """Set up per-axes dot plot state, styling, and resize handling."""
    state = {
        "series": [],
        "boundary_lines": [],
        "last_size_px": None,
        "relayout_running": False,
        "categories": None,
        "orientation": orientation,
    }
    ax._dotplot_state = state
    # Reference gridlines along the count axis help students read a
    # count off it. Keep them below the dots and drop the lines along
    # the value axis, which would clutter the stacks.
    ax.set_axisbelow(True)
    if orientation == "vertical":
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.set_xlabel(DOTPLOT_XLABEL)
    else:
        ax.grid(True, axis="x")
        ax.grid(False, axis="y")
        ax.set_ylabel(DOTPLOT_XLABEL)
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
    vertical = state["orientation"] == "vertical"
    positions = state["positions"]
    spacing = state["spacing"]
    n_series = len(state["series"])
    # Each batch gets its own lane inside the slot around each value.
    lane_width = spacing / n_series

    # Value-axis padding: one slot of air beyond the outermost stacks.
    value_lim = (positions[0] - spacing, positions[-1] + spacing)
    set_value_lim = ax.set_xlim if vertical else ax.set_ylim
    set_value_lim(*value_lim)

    # Every dot is one count, so a stack unit is always 1.

    # Pixel measurements via the axes transforms (valid before any
    # draw). The 1-px floor guards against two nearly-identical values
    # driving the lane width -- and with it the dot size -- to zero.
    # "cross_px" is the rendered size of the axes along the count axis
    # (the axis the stacks grow along) -- height for a vertical plot,
    # width for a horizontal one.
    span_px = _x_span_px if vertical else _y_span_px
    lane_width_px = max(span_px(ax, lane_width), 1.0)
    cross_px = _axes_size_px(ax)[1 if vertical else 0]

    # Choose the count-axis range so one count on screen is never wider
    # than one lane, nor than the DOTPLOT_MAX_DOT_SIZE cap. That pixel
    # size becomes the dot diameter, so dots stack touching and never
    # spill into the next lane, short stacks cannot inflate the dots
    # past the cap, and taller stacks shrink the dots instead of
    # overflowing the axes.
    max_dot_px = DOTPLOT_MAX_DOT_SIZE * ax.figure.dpi / 72.0
    tallest = max(s["counts"].max() for s in state["series"])
    count_max = max(
        tallest * DOTPLOT_STACK_HEADROOM,
        cross_px / lane_width_px,
        cross_px / max_dot_px,
    )
    set_count_lim = ax.set_ylim if vertical else ax.set_xlim
    set_count_lim(0, count_max)

    points_per_px = 72.0 / ax.figure.dpi
    # Dodge overlaid batches by exactly one dot diameter so their
    # stacks sit side by side and touch, rather than by the full lane
    # width -- which would leave a gap between the columns once the
    # dot-size cap shrinks the dots below the lane width. The dot
    # diameter is what the count range above was calibrated to; it
    # never exceeds one lane width, so a dodged group still fits inside
    # its slot without colliding with the neighboring value's stacks.
    diameter_px = cross_px / count_max
    value_px_per_data = max(span_px(ax, 1.0), 1e-9)
    dodge_step = diameter_px / value_px_per_data
    # scatter sizes are marker areas in points^2 (diameter squared).
    size = (diameter_px * points_per_px) ** 2
    for i, series in enumerate(state["series"]):
        offset = (i - (n_series - 1) / 2.0) * dodge_step
        value_coords = []
        count_coords = []
        for position, count in zip(positions, series["counts"]):
            value_coords.extend(np.full(count, position + offset))
            count_coords.extend(np.arange(1, count + 1) - 0.5)
        offsets = (
            np.column_stack([value_coords, count_coords])
            if vertical
            else np.column_stack([count_coords, value_coords])
        )
        series["dots"].set_offsets(offsets)
        series["dots"].set_sizes(np.full(len(value_coords), size))

    # Categorical data: label each integer code position with its category
    # name. Numeric data: keep the integer locator so whole-number values
    # get whole-number ticks. Both apply to the value axis.
    value_axis = ax.xaxis if vertical else ax.yaxis
    set_value_ticks = ax.set_xticks if vertical else ax.set_yticks
    set_value_ticklabels = ax.set_xticklabels if vertical else ax.set_yticklabels
    categories = state.get("categories")
    if categories is not None:
        set_value_ticks(positions)
        set_value_ticklabels(
            [
                (
                    str(categories[int(round(p))])
                    if 0 <= int(round(p)) < len(categories)
                    else ""
                )
                for p in positions
            ]
        )
    elif np.all(positions == np.round(positions)):
        value_axis.set_major_locator(MaxNLocator(integer=True))
    # The count axis is always integer counts. This runs on every draw, so
    # it would undo a cap set on the tick count afterwards -- a dot plot in a
    # marginal strip is thinned to fit (thin_marginal_frequency_ticks, which
    # records the cap on the axes), so honor that if it is there.
    count_axis = ax.yaxis if vertical else ax.xaxis
    cap = getattr(ax, "_symbulate_freq_ticks", None)
    count_axis.set_major_locator(
        MaxNLocator(integer=True)
        if cap is None
        else MaxNLocator(integer=True, nbins=cap)
    )
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
    draw_line = ax.axvline if state["orientation"] == "vertical" else ax.axhline
    for edge in edges:
        state["boundary_lines"].append(
            draw_line(
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
    if state["orientation"] == "vertical":
        ax.set_ylabel("Count")
    else:
        ax.set_xlabel("Count")
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


def dotplot_tallest_stack(values):
    """Return the height of the tallest stack a dot plot would draw.

    A dot plot places one dot per observation and stacks identical
    values, so the tallest stack is simply the largest number of
    observations sharing a single value. This is what decides whether a
    dot plot stays legible: unlike the sample size or the number of
    distinct values, one very tall stack alone shrinks every dot to an
    unreadable speck. ``RVResults.plot`` compares this against
    ``DOTPLOT_MAX_STACK`` to decide whether a dot plot is a good
    automatic default (see that constant).

    Parameters
    ----------
    values : array-like
        The simulated values a dot plot would be drawn from. Numeric or
        categorical; values are grouped by exact equality, matching how
        ``make_dotplot`` stacks them.

    Returns
    -------
    int
        The number of observations in the most-repeated value, or 0 if
        there are no values.

    Examples
    --------
    >>> import numpy as np
    >>> dotplot_tallest_stack(np.array([0, 0, 0, 1, 2]))
    3
    >>> dotplot_tallest_stack(np.array(["H", "H", "T"]))
    2
    """
    data = np.asarray(list(values))
    if data.size == 0:
        return 0
    _, counts = np.unique(data, return_counts=True)
    return int(counts.max())


def make_dotplot(
    values, ax, color, alpha=None, label=None, orientation="vertical", **kwargs
):
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
        The simulated values to plot, e.g. ``RVResults.array``. May be
        numeric or categorical (strings). Categorical values are placed
        as evenly spaced categories, labeled with the value.
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
    orientation : {"vertical", "horizontal"}, default "vertical"
        "vertical" (default) stacks dots upward from the x-axis, values
        on the x-axis -- today's behavior. "horizontal" stacks dots
        rightward from the y-axis instead, values on the y-axis -- for
        drawing sideways in a 2D plot's y-marginal panel. Fixed by the
        first dot plot drawn on a given axes; later overlaid calls on
        the same axes keep that axes' orientation regardless of what
        they pass.
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
    # Categorical values have no numeric position, so map them to evenly
    # spaced integer codes and remember the labels for the ticks; numeric
    # values keep their real positions (categories is None). The codes are
    # then validated (finite, non-empty) like any numeric dot plot.
    codes, categories = _encode_categories(values)
    values = _dotplot_clean_values(codes)
    if alpha is None:
        alpha = DOTPLOT_ALPHA
    state = getattr(ax, "_dotplot_state", None)
    if state is None:
        state = _dotplot_init_state(ax, orientation)
    if categories is not None:
        state["categories"] = categories
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
    (per ``classify_values``): ``"bins"`` once any single (x, y) value
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
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
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


def make_density2D(x, y, ax, contour=True, levels=None, colorbar=True, **kwargs):
    """Draw a 2D density surface from a KDE estimate.

    Both modes plot the *same* KDE-estimated density surface with
    ``ax.contourf``; they differ in how finely it is quantized:

    - ``contour=True`` (default): a topographic "Contour Plot". The
      surface is split into ``levels`` discrete color bands with thin
      white outlines between them, so each band can be matched to the
      colorbar by eye rather than guessed at from a gradient.
    - ``contour=False``: a *continuous* density plot. The surface is
      drawn with a large fixed number of color bands
      (``DENSITY2D_CONTINUOUS_LEVELS``) so they blend into a smooth
      gradient with no visible banding -- the "2D Density Plot" look.
      The ``levels`` argument does not apply there; passing it warns
      and has no effect.

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
    ax.set_xlabel("Variable 1")
    ax.set_ylabel("Variable 2")
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


def _count_joint_plot(ax):
    """Record a joint plot on ``ax`` and report whether it is the first.

    A filled surface completely hides whatever was drawn under it, and two
    color scales can't share one colorbar, so a second joint plot on the
    same axes prints the readability warning and skips its colorbar --
    which would otherwise land on top of the first one and garble both sets
    of labels. This is the "warn but still draw" tier of the overlay policy,
    the same one two tile plots or two 2-D histograms fall into. The count
    lives on the axes object, the way ``get_next_color`` keeps the color
    cycle there.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes being drawn on.

    Returns
    -------
    bool
        True if this is the first joint plot on these axes.
    """
    n_prior = getattr(ax, "_joint_plot_count", 0)
    ax._joint_plot_count = n_prior + 1
    if n_prior:
        print(JOINT_OVERLAY_WARNING)
    return n_prior == 0


def _joint_colorbar(mappable, ax, label, ticks):
    """Attach a colorbar for a joint theoretical plot.

    Shared by :func:`make_joint_pdf` and :func:`make_joint_pmf` so both
    place and label their color scale identically. Uses
    ``make_axes_locatable`` (an approved layout helper) so the bar tracks
    the axes it belongs to, and restores the data axes afterwards so a
    later call draws on the plot rather than the colorbar.

    Parameters
    ----------
    mappable : matplotlib.cm.ScalarMappable
        The surface or mesh the color scale describes.
    ax : matplotlib.axes.Axes
        The axes the joint plot was drawn on.
    label : str
        Label for the colorbar ("Density" or "Probability").
    ticks : array-like
        Tick positions, running from 0 to the peak value so both ends of
        the scale are labeled.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=JOINT_CBAR_SIZE, pad=JOINT_CBAR_PAD)
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.set_label(label)
    cbar.set_ticks(ticks)
    # A formatter rather than fixed labels, so matplotlib keeps thinning
    # ticks automatically when there are many band edges.
    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.{JOINT_CBAR_DECIMALS}f}")
    )
    plt.sca(ax)


def make_joint_pdf(
    func,
    xlim,
    ylim,
    ax,
    contour=True,
    colorbar=True,
    xlabel="Variable 1",
    ylabel="Variable 2",
    title=True,
    **kwargs,
):
    """Draw the joint density surface of a continuous 2-D distribution.

    The theoretical counterpart of :func:`make_density2D`: instead of
    estimating a surface from simulated values with a KDE, this evaluates
    the distribution's own joint pdf on a
    ``JOINT_PDF_GRID_POINTS``-by-``JOINT_PDF_GRID_POINTS`` grid and shades
    it with ``ax.contourf``. It is the two-dimensional extension of the
    curve the univariate ``Distribution.plot()`` draws, and it takes the
    same two forms ``make_density2D`` does:

    - ``contour=True`` (default): a topographic contour plot --
      ``JOINT_PDF_LEVELS`` discrete bands with thin white outlines, so each
      band can be matched to the colorbar by eye.
    - ``contour=False``: a smoothly shaded surface, drawn with
      ``JOINT_PDF_CONTINUOUS_LEVELS`` bands so no banding is visible.

    The color scale runs from 0 to the peak density, so the colorbar starts
    at 0. A density that is unbounded at the edge of its support (a
    ``Dirichlet`` with a concentration below 1, for example) has
    non-finite grid values; those are held at the largest finite density on
    the grid rather than being allowed to flatten the whole surface into a
    single color.

    Parameters
    ----------
    func : callable
        The joint density. Called as ``func(x, y)`` with two equal-length
        flat arrays of coordinates, returning the density at each of those
        points as a flat array.
    xlim, ylim : tuple of float
        The ``(low, high)`` window to evaluate the density over, one per
        axis.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    contour : bool, default False
        If False (default), draw a smoothly shaded surface. If True, draw
        discrete contour bands with white outlines between them.
    colorbar : bool, default True
        If True, add a colorbar to the right of the axes. A panel of a
        pairs matrix passes False, since one small color scale per panel
        would crowd out the panels themselves.
    xlabel, ylabel : str, optional
        Axis labels, naming the two variables being plotted.
    title : bool, default True
        Whether to title the axes. A panel of a pairs matrix passes False,
        since the whole figure carries one title instead.
    **kwargs
        Additional keyword arguments passed to ``ax.contourf``.

    Returns
    -------
    matplotlib.contour.QuadContourSet
        The object returned by ``ax.contourf``, so the caller can inspect
        or further style the surface.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> import numpy as np
    >>> f = stats.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]])
    >>> pdf = lambda x, y: f.pdf(np.column_stack([x, y]))
    >>> make_joint_pdf(pdf, (-3, 3), (-3, 3), plt.gca())  # doctest: +SKIP
    """
    # A second surface on these axes hides the first, so warn and let it
    # keep the one colorbar already there.
    first = _count_joint_plot(ax)
    colorbar = colorbar and first

    Xgrid, Ygrid = np.meshgrid(
        np.linspace(xlim[0], xlim[1], JOINT_PDF_GRID_POINTS),
        np.linspace(ylim[0], ylim[1], JOINT_PDF_GRID_POINTS),
    )
    Z = np.asarray(func(Xgrid.ravel(), Ygrid.ravel()), dtype=float).reshape(Xgrid.shape)

    # A density can be unbounded at the boundary of its support, so scale
    # the colors by the largest *finite* value on the grid and hold the
    # non-finite points there -- the same "ignore non-finite heights when
    # scaling" rule the univariate plot() uses for its y-limit.
    finite = np.isfinite(Z)
    zmax = Z[finite].max() if finite.any() else 1.0
    if zmax <= 0:
        # A grid that never leaves 0 (a window off the support) still needs
        # a well-formed, increasing set of color bands.
        zmax = 1.0
    Z = np.where(finite, np.minimum(Z, zmax), zmax)

    levels = JOINT_PDF_LEVELS if contour else JOINT_PDF_CONTINUOUS_LEVELS
    # contourf reads the level values as band *boundaries* (N boundaries ->
    # N - 1 colors), so build levels + 1 edges to get exactly `levels`
    # colors. No cmap argument: the sequential colormap comes from
    # image.cmap in symbulate.mplstyle.
    level_edges = np.linspace(0, zmax, levels + 1)

    filled = ax.contourf(Xgrid, Ygrid, Z, levels=level_edges, **kwargs)
    if contour:
        ax.contour(
            Xgrid,
            Ygrid,
            Z,
            levels=level_edges,
            colors=JOINT_CONTOUR_LINE_COLOR,
            linewidths=JOINT_CONTOUR_LINEWIDTH,
            alpha=JOINT_CONTOUR_LINE_ALPHA,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title("Joint Contour Plot" if contour else "Joint PDF Plot")
    # symbulate.mplstyle's global grid is horizontal-only (axes.grid.axis:
    # y); like the 1-D and 2-D density plots, this one reads better with
    # both directions, so override it for this plot type.
    ax.grid(True, axis="both")

    if colorbar:
        ticks = level_edges if contour else np.linspace(0, zmax, JOINT_CBAR_TICKS)
        _joint_colorbar(filled, ax, "Density", ticks)

    return filled


def make_joint_pmf(
    func,
    xvalues,
    yvalues,
    ax,
    colorbar=True,
    xlabel="Variable 1",
    ylabel="Variable 2",
    title=True,
    **kwargs,
):
    """Draw the joint probability grid of a discrete 2-D distribution.

    The discrete counterpart of :func:`make_joint_pdf`, and the
    theoretical counterpart of :func:`make_tile`: one cell per possible
    pair of values, colored by the probability of that pair. This is the
    same discrete/continuous split the univariate ``Distribution.plot()``
    already makes -- masses at the values a discrete distribution can
    actually take, rather than a smooth surface through the gaps.

    Cells are drawn as a seamless ``imshow`` mesh centered on the values
    themselves (each cell spans half a unit either side), so the axis reads
    in real units and a pair's probability can be read off the colorbar.
    Impossible pairs simply have probability 0 and take the colormap's
    zero color, so the shape of the joint support is visible.

    Parameters
    ----------
    func : callable
        The joint probability mass function. Called as ``func(x, y)`` with
        two equal-length flat arrays of values, returning the probability
        of each of those pairs as a flat array.
    xvalues, yvalues : array-like of int
        The values each variable can take, in increasing order -- one grid
        column per ``xvalues`` entry and one row per ``yvalues`` entry.
    ax : matplotlib.axes.Axes
        The axes to draw on.
    colorbar : bool, default True
        If True, add a colorbar to the right of the axes. A panel of a
        pairs matrix passes False.
    xlabel, ylabel : str, optional
        Axis labels, naming the two variables being plotted.
    title : bool, default True
        Whether to title the axes. A panel of a pairs matrix passes False.
    **kwargs
        Additional keyword arguments passed to ``ax.imshow``.

    Returns
    -------
    matplotlib.image.AxesImage
        The image from ``ax.imshow``, so the caller can inspect or further
        style the mesh.

    Raises
    ------
    ValueError
        If the grid would have more than ``JOINT_PMF_MAX_CELLS`` cells,
        where the cells are too small to read.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from scipy import stats
    >>> f = stats.multinomial(10, [0.5, 0.3, 0.2])
    >>> pmf = lambda x, y: f.pmf(np.column_stack([x, y, 10 - x - y]))
    >>> make_joint_pmf(pmf, np.arange(11), np.arange(11), plt.gca())  # doctest: +SKIP
    """
    xvalues = np.asarray(xvalues)
    yvalues = np.asarray(yvalues)

    n_cells = len(xvalues) * len(yvalues)

    if n_cells > JOINT_PMF_MAX_CELLS:
        raise ValueError(
            "This joint plot would need %d cells (%d values across by %d "
            "up), which is too many to read -- each cell would be a "
            "fraction of a pixel. This happens when the counts range over "
            "many values, which a large number of trials produces. Try a "
            "smaller number of trials, or plot one variable at a time (for "
            "example, Binomial(n, p).plot() for a single count)."
            % (n_cells, len(xvalues), len(yvalues))
        )

    # A second mesh on these axes hides the first, so warn and let it keep
    # the one colorbar already there. Counted after the budget check above,
    # so a plot that never draws isn't counted as one that did.
    first = _count_joint_plot(ax)
    colorbar = colorbar and first

    Xgrid, Ygrid = np.meshgrid(xvalues, yvalues)
    Z = np.asarray(func(Xgrid.ravel(), Ygrid.ravel()), dtype=float).reshape(Xgrid.shape)

    # imshow centers each cell on its value, so the extent runs half a unit
    # past the outermost values on every side -- the cells then tile the
    # axes seamlessly and every tick lands on a cell center. aspect="auto"
    # lets the grid fill the axes instead of forcing square cells, matching
    # make_tile. No cmap argument: viridis comes from image.cmap in
    # symbulate.mplstyle, so probability 0 reads as its dark end.
    kwargs.setdefault("interpolation", "nearest")
    mesh = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=(
            xvalues[0] - 0.5,
            xvalues[-1] + 0.5,
            yvalues[0] - 0.5,
            yvalues[-1] + 0.5,
        ),
        vmin=0,
        **kwargs,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title("Joint PMF Plot")
    # Counts are whole numbers, so only whole-number ticks make sense; cap
    # how many appear so the labels can't crowd into each other.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=MAX_DISCRETE_TICKS, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=MAX_DISCRETE_TICKS, integer=True))
    # A filled mesh covers the axes, so a grid would only show at the
    # spines; turn it off entirely rather than leave stray ticks of it,
    # matching make_tile.
    ax.grid(False)

    if colorbar:
        zmax = Z.max() if Z.size else 1.0
        if zmax <= 0:
            zmax = 1.0
        _joint_colorbar(mesh, ax, "Probability", np.linspace(0, zmax, JOINT_CBAR_TICKS))

    return mesh
