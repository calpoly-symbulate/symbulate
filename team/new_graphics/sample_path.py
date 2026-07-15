"""Draft of the sample path plot type for the graphics overhaul.

**Integrated:** this draft has landed in the package.
``make_sample_path()`` now lives in ``symbulate/plot.py`` (with the
``SAMPLE_PATH_*`` constants) and the ``.plot()`` methods below call it.
This file is kept as the approved prototype reference; see
``new_graphics_demos/sample_path_demo.ipynb`` for the integrated API.

This file was the planned replacement for the ad hoc
``plt.plot(ts, ys, ".--", **kwargs)`` calls inside the ``.plot()``
methods of ``Tuple``, ``InfiniteVector``, ``DiscreteTimeFunction``,
and ``ContinuousTimeFunction`` (``symbulate/result.py``) -- one
simulated realization of a stochastic process, drawn over its index
or time range. It depends only on
``symbulate/plot.py`` -- which stays -- and each caller still computes
its own ``(times, values)`` pair exactly as it does today
(``range(len(self))`` for a ``Tuple``, ``index_set[n]`` for a
``DiscreteTimeFunction``, ``np.linspace(tmin, tmax, 200)`` for a
``ContinuousTimeFunction``, etc.); only the drawing and styling move
into ``make_sample_path()`` below.

What's new on top of that carried-over ``(times, values)`` logic is
what the current inline calls don't do: the package's color cycle
(``get_next_color(ax)``) instead of leaving color to matplotlib's own
default, axis labels ("Time" / "Value") and a title ("Sample Path"),
and per-series labeling with an automatic legend so overlaid paths --
several realizations of the same process compared side by side, a very
common simulation pattern -- stay tellable apart.

Also new: the marker is dropped. The existing ``".--"`` format string
draws a dot at every plotted point connected by a dashed line --
readable at the module's own ``tmin=0, tmax=10`` doctest examples, but
a cluttered mess of dots once a path runs out to 100+ points (the
common case for a sample path, unlike the small-n scatter/dot-plot
types). ``make_sample_path()`` always draws a plain solid connected
line -- ``ax.plot(times, values, ...)`` with no ``marker=`` -- so a
path stays readable at any length, matching the approved prototype.

Overlay category (see CLAUDE.md, "Overlay Policy"): **natural overlay**
-- a second sample path on the same axes just draws on top of the
first, no message, the same as impulse/histogram/density/most types.

Run this file directly to render the prototype:

    python team/new_graphics/sample_path.py
"""

import matplotlib.pyplot as plt
import numpy as np

# This helper lives in symbulate/plot.py, which stays through the
# graphics overhaul -- only the plotting code inside result.py is
# being replaced. make_sample_path() below is its replacement.
from symbulate.plot import get_next_color

rng = np.random.default_rng()

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different line width for different plot types" (see
# DECISIONS.md, "Decision: .mplstyle Standards"), so these cannot live
# in symbulate.mplstyle. They migrate to the top of symbulate/plot.py.
#
# Line width is unchanged from the shared rcParams fallback (DECISIONS.md,
# Visual Style Guide: "sample paths stay at 1.5"). Named here anyway, like
# every other plot type's constants, so a future change has one place to
# happen instead of being buried inside an ax.plot() call.
SAMPLE_PATH_LINEWIDTH = 1.5
# No alpha change is called for in DECISIONS.md (only line width is
# discussed for sample paths) -- a single path has no overplotting to
# soften, so it stays fully opaque.
SAMPLE_PATH_ALPHA = 1.0
SAMPLE_PATH_LEGEND_LOC = "upper right"


def _refresh_legend(ax):
    """Show a legend once two or more labeled paths share the axes.

    A lone sample path stays legend-free; overlaying a second path --
    e.g. two realizations of the same process compared side by side --
    turns the legend on so the two can be told apart.
    """
    handles, _ = ax.get_legend_handles_labels()
    if len(handles) > 1:
        ax.legend(loc=SAMPLE_PATH_LEGEND_LOC)


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

    This replaces the ``plt.plot(ts, ys, ".--", **kwargs)`` call
    currently duplicated across ``Tuple.plot()``,
    ``InfiniteVector.plot()``, ``DiscreteTimeFunction.plot()``, and
    ``ContinuousTimeFunction.plot()`` in ``symbulate/result.py``: the
    dot-dash marker is dropped in favor of a plain solid line (readable
    whether the path has 10 points or 1000), and the package's color
    cycle, axis labels, and title are added.

    Sample paths overlay naturally: a second call on the same axes
    draws on top of the first, and a legend appears automatically in
    the top right once two or more paths share the axes -- the common
    case of comparing several realizations of the same process. Each
    path is named by ``label``, or "Path 1", "Path 2", ... in call
    order when no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how ``RVResults.plot()`` calls the other
    plot helpers in ``symbulate/plot.py``.

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
    _refresh_legend(ax)

    return line


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    # Figure 1: the single-path prototype -- a simple symmetric random
    # walk (steps of +/-1), run out to n = 100 steps so the plain
    # connected line (no per-point markers) has to carry the reading on
    # its own, matching the approved prototype.
    n = 100
    steps = rng.choice([-1, 1], size=n)
    values = np.concatenate([[0], np.cumsum(steps)])
    times = np.arange(n + 1)

    ax = plt.gca()
    make_sample_path(times, values, ax, get_next_color(ax))

    # Figure 2: the overlay prototype -- a second realization of the
    # same random walk on the same axes picks up orange and turns on
    # the "Path k" legend automatically, the common "compare a few
    # realizations" use case.
    steps2 = rng.choice([-1, 1], size=n)
    values2 = np.concatenate([[0], np.cumsum(steps2)])

    plt.figure()
    ax = plt.gca()
    make_sample_path(times, values, ax, get_next_color(ax))
    make_sample_path(times, values2, ax, get_next_color(ax))

    # Figure 3: a continuous-time path (e.g. Brownian motion) -- smooth
    # increments over a fine time grid rather than unit steps, still a
    # single make_sample_path() call.
    t_cont = np.linspace(0, 10, 200)
    dt = t_cont[1] - t_cont[0]
    increments = rng.normal(0, np.sqrt(dt), len(t_cont) - 1)
    values_cont = np.concatenate([[0], np.cumsum(increments)])

    plt.figure()
    ax = plt.gca()
    make_sample_path(t_cont, values_cont, ax, get_next_color(ax))

    plt.show()
