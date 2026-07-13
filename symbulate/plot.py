import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import itertools
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline
from cycler import cycler

figure = plt.figure

xlabel = plt.xlabel
ylabel = plt.ylabel

xlim = plt.xlim
ylim = plt.ylim

color_index = 0
color_cycle = [c["color"] for c in plt.rcParams["axes.prop_cycle"]]


def init_color():
    hex_list = [colors.rgb2hex(rgb) for rgb in cm["tab10"].colors]
    plt.rcParams["axes.prop_cycle"] = cycler("color", hex_list)


def get_next_color(axes):
    if not hasattr(axes, "_color_cycle"):
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        axes._color_cycle = itertools.cycle(prop_cycle.by_key()["color"])
    return next(axes._color_cycle)


# Per-plot-type constants for the impulse plot. matplotlib rcParams are
# global and can't express "different line width for different plot
# types" (see DECISIONS.md, "Decision: .mplstyle Standards"), so these
# live here instead of symbulate.mplstyle.
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


def is_discrete(heights):
    return sum([(i > 1) for i in heights]) > 0.8 * len(heights)


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


def setup_tile(v, bins, discrete):
    if not discrete:
        v_lab = np.linspace(min(v), max(v), bins + 1)
        v_pos = np.arange(0, len(v_lab)) - 0.5
        v_vect = np.digitize(v, v_lab, right=True) - 1
    else:
        v_lab = np.unique(v)  # returns sorted array
        v_pos = range(len(v_lab))
        v_map = dict(zip(v_lab, v_pos))
        v_vect = np.vectorize(v_map.get)(v)
    return v_vect, v_lab, v_pos


def make_tile(x, y, bins, discrete_x, discrete_y, ax):
    x_vect, x_lab, x_pos = setup_tile(x, bins, discrete_x)
    y_vect, y_lab, y_pos = setup_tile(y, bins, discrete_y)
    nums = len(x_vect)
    counts = count_var(list(zip(y_vect, x_vect)))
    y_shape = len(y_lab) if discrete_y else len(y_lab) - 1
    x_shape = len(x_lab) if discrete_x else len(x_lab) - 1
    intensity = np.zeros(shape=(y_shape, x_shape))

    for key, val in counts.items():
        intensity[key] = val / nums
    if not discrete_x:
        x_lab = np.around(x_lab, decimals=1)
    if not discrete_y:
        y_lab = np.around(y_lab, decimals=1)
    hm = ax.matshow(intensity, cmap="Blues", origin="lower", aspect="auto", vmin=0)
    ax.xaxis.set_ticks_position("bottom")
    setup_ticks(x_pos, x_lab, ax.xaxis)
    setup_ticks(y_pos, y_lab, ax.yaxis)
    return hm


def make_violin(data, positions, ax, axis, alpha):
    values = []
    i, j = (0, 1) if axis == "x" else (1, 0)
    values = [data[data[:, i] == pos, j].tolist() for pos in positions]
    orientation = "horizontal" if axis == "y" else "vertical"
    violins = ax.violinplot(dataset=values, showmedians=True, orientation=orientation)
    setup_ticks(
        np.array(positions) + 1, positions, ax.xaxis if axis == "x" else ax.yaxis
    )
    for part in violins["bodies"]:
        part.set_edgecolor("black")
        part.set_alpha(alpha)
    for component in ("cbars", "cmins", "cmaxes", "cmedians"):
        vp = violins[component]
        vp.set_edgecolor("black")
        vp.set_linewidth(1)


def make_marginal_impulse(count, color, ax_marg, alpha, axis):
    key, val = list(count.keys()), list(count.values())
    tot = sum(val)
    val = [i / tot for i in val]
    if axis == "x":
        ax_marg.vlines(key, 0, val, color=color, alpha=alpha)
    elif axis == "y":
        ax_marg.hlines(key, 0, val, color=color, alpha=alpha)


def _refresh_legend(ax):
    # A lone series stays legend-free; a second one (another impulse
    # plot, or a true-distribution overlay) turns the legend on.
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(loc=IMPULSE_LEGEND_LOC)


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


def make_density2D(x, y, ax):
    res = np.vstack([x, y])
    density = gaussian_kde(res)
    xmax, xmin = max(x), min(x)
    ymax, ymin = max(y), min(y)
    Xgrid, Ygrid = np.meshgrid(
        np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100)
    )
    Z = density.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
    den = ax.imshow(
        Z.reshape(Xgrid.shape),
        origin="lower",
        cmap="Blues",
        aspect="auto",
        extent=[xmin, xmax, ymin, ymax],
    )
    return den
