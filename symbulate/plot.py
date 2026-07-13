import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import itertools
from scipy.stats import gaussian_kde
from cycler import cycler

N_UNIQUE_THRESHOLD = 40
N_SMALL_THRESHOLD = 100

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
        "alternatives": ["impulse", "bar"],
    },
    ("1D_discrete", False): {
        "default": "impulse",
        "alternatives": ["bar", "hist", "dotplot"],
    },
    ("1D_continuous", True): {"default": "rug", "alternatives": ["dotplot", "hist"]},
    ("1D_continuous", False): {"default": "hist", "alternatives": ["density", "rug"]},
    ("2D_dd", True): {"default": "scatter", "alternatives": ["tile"]},
    ("2D_dd", False): {"default": "tile", "alternatives": ["scatter"]},
    ("2D_cc", True): {"default": "scatter", "alternatives": ["density2d", "hist2d"]},
    ("2D_cc", False): {"default": "hist2d", "alternatives": ["density2d", "scatter"]},
    ("2D_mixed", True): {
        "default": "segmented_rug",
        "alternatives": ["scatter", "tile"],
    },
    ("2D_mixed", False): {"default": "tile", "alternatives": ["violin", "scatter"]},
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
    "rug": "Rug Plot",
    "dotplot": "Dot Plot",
    "scatter": "Scatter Plot",
    "tile": "Tile Plot",
    "hist2d": "2D Histogram",
    "density2d": "2D Density Plot",
    "violin": "Violin Plot",
    "segmented_rug": "Segmented Rug Plot",
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
