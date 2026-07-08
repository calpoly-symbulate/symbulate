"""Draft of the rug plot type for the graphics overhaul.

Staging code -- not yet wired into the package. The goal is that
``RVResults.plot()`` (in ``results.py``) will eventually call
``make_rug()`` below when the user asks for a rug plot or the default
plot lookup table selects one (1D data with small n).

Visual target: the approved rug plot prototypes (Task 1E). Standalone
rug -- one light Okabe-Ito tick mark per simulated value along the
bottom of the axes, no y-axis ticks or left spine (the vertical
direction carries no information), faint vertical reference
gridlines, and only the bottom axis line kept. Combined with another
plot type (e.g. ``type=("density", "rug")``) -- short ticks hugging
the x-axis under the companion plot, which keeps its own y-axis,
spines, grid, and title. The grid,
spines, fonts, and figure size come from
``symbulate/symbulate.mplstyle``; the per-plot-type values (alpha,
tick height, line width) are the named constants below, which migrate
to the top of ``symbulate/plot.py`` at integration time.

Run this file directly to render the prototype:

    python team/new_graphics/rug.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Per-plot-type constants. matplotlib rcParams are global and cannot
# express "different alpha for different plot types," so these cannot
# live in symbulate.mplstyle (see DECISIONS.md, "Decision: .mplstyle
# Standards"). They migrate to the top of symbulate/plot.py.
RUG_ALPHA = 0.5
RUG_TICK_HEIGHT = 0.4  # fraction of the axes height, standalone rug
RUG_COMBINED_TICK_HEIGHT = 0.04  # short ticks under a density/hist companion
RUG_LINEWIDTH = 1.0
RUG_LEGEND_LOC = "upper right"


def make_rug(values, ax, color, alpha=None, label=None, standalone=True, **kwargs):
    """Draw a rug plot of simulated values on the given axes.

    Draws in the style of the approved rug plot prototypes: one thin
    vertical tick per simulated value along the bottom of the axes.
    The x-axis is always labeled "Value".

    A rug can stand alone or accompany another plot type, and the two
    looks differ (each matching its own approved prototype):

    - **Standalone** (``standalone=True``, the default): tall ticks
      (``RUG_TICK_HEIGHT`` of the axes height) and the title
      "Rug Plot". The y-axis carries no information, so its ticks and
      the left spine are hidden, and faint vertical gridlines
      (instead of the usual horizontal ones) give a reference for
      where values fall.
    - **Combined** (``standalone=False``, e.g. for
      ``type=("density", "rug")``): short ticks
      (``RUG_COMBINED_TICK_HEIGHT``) hugging the x-axis, and the
      y-axis, spines, grid, and title are left alone -- they belong
      to the companion plot type drawn on the same axes.

    Rug plots overlay naturally: a second call on the same axes draws
    on top of the first, and a legend appears automatically in the top
    right once two or more rugs share the axes. Each rug is named by
    ``label``, or "Variable 1", "Variable 2", ... in call order when
    no label is given.

    The caller is responsible for getting the axes (``plt.gca()``, so
    overlays keep working) and for advancing the color cycle with
    ``get_next_color(ax)`` exactly once -- pass the result in as
    ``color``. This mirrors how the other plot helpers in
    ``symbulate/plot.py`` (``make_tile``, ``make_density2D``) are called.

    Tick heights are drawn in axes fractions (``RUG_TICK_HEIGHT`` of
    the axes height), not data units, so the ticks keep their size if
    something with a meaningful y-scale (e.g. a density curve) is
    drawn on the same axes later.

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
        standard for rug plots (0.5), which keeps stacked values
        visible as darker ticks.
    label : str, optional
        Name for this rug in the legend. Defaults to "Variable k",
        where k counts the rugs drawn on these axes so far.
    standalone : bool, default True
        If True, the rug is the whole plot: tall ticks, hidden
        y-axis and left spine, vertical gridlines, "Rug Plot" title.
        If False, the rug accompanies another plot type on the same
        axes: short ticks along the x-axis, and everything else on
        the axes is left untouched.
    **kwargs
        Additional keyword arguments passed to ``matplotlib``.

    Returns
    -------
    matplotlib.collections.LineCollection
        The collection of tick marks, so the caller can inspect or
        further style them.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from symbulate.plot import get_next_color
    >>> values = np.random.default_rng().normal(0, 1, 60)
    >>> ax = plt.gca()
    >>> make_rug(values, ax, get_next_color(ax))  # doctest: +SKIP
    """
    if alpha is None:
        alpha = RUG_ALPHA
    # Count the rugs drawn on these axes, stored on the axes object
    # itself (the same pattern get_next_color uses for the color
    # cycle) so overlays from separate .plot() calls see it.
    n_prior_rugs = getattr(ax, "_rug_count", 0)
    if label is None:
        label = f"Variable {n_prior_rugs + 1}"
    ax._rug_count = n_prior_rugs + 1
    tick_height = RUG_TICK_HEIGHT if standalone else RUG_COMBINED_TICK_HEIGHT
    rug = ax.vlines(
        np.asarray(values),
        0,
        tick_height,
        # Axes-fraction y coordinates: ticks rise from the bottom of
        # the axes regardless of the y data limits.
        transform=ax.get_xaxis_transform(),
        color=color,
        alpha=alpha,
        linewidth=RUG_LINEWIDTH,
        label=label,
        **kwargs,
    )
    if standalone:
        # No y information in a standalone rug: hide the ticks and
        # the left spine, and switch the reference grid to vertical
        # lines (the style file's horizontal grid has no y ticks left
        # to attach to). Grid color/alpha/width still come from
        # symbulate.mplstyle. When the rug accompanies another plot
        # type, that type owns the y-axis, grid, and title, so none
        # of this applies.
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x")
        ax.set_title("Rug Plot")
    ax.set_xlabel("Value")
    # A legend only helps once there is more than one rug to tell
    # apart; a lone rug stays legend-free.
    if ax._rug_count > 1:
        ax.legend(loc=RUG_LEGEND_LOC)
    return rug


if __name__ == "__main__":
    # Reproduce the approved prototype image with fake data.
    from pathlib import Path

    style_path = (
        Path(__file__).resolve().parents[2] / "symbulate" / "symbulate.mplstyle"
    )
    plt.style.use(str(style_path))

    rng = np.random.default_rng(7)
    values = rng.normal(0, 1, 60)

    # Figure 1: the single-rug prototype. The short, wide figure
    # matches the prototype image; this is demo-only -- the final
    # .plot() integration keeps the global default figure size.
    plt.figure(figsize=(9, 2))
    ax = plt.gca()
    # The prototype uses Okabe-Ito sky blue. Note: this is the *second*
    # hue in the symbulate.mplstyle color cycle (the first is orange
    # E69F00) -- once integrated, the color will come from
    # get_next_color(ax) instead.
    make_rug(values, ax, "#56B4E9")

    # Figure 2: two overlaid rugs with the automatic "Variable k"
    # legend in the top right.
    plt.figure(figsize=(9, 2))
    ax = plt.gca()
    make_rug(rng.normal(0, 1, 60), ax, "#56B4E9")
    make_rug(rng.normal(1.5, 0.5, 60), ax, "#E69F00")

    plt.show()
