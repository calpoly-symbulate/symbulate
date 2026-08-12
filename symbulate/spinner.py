"""Probability spinner dials for distributions.

A spinner is a picture of a distribution's quantile function. The dial is
fixed and the needle turns, so a quarter turn always points at the 25th
percentile, half a turn at the median, and three quarters at the 75th --
which is the whole reason the design is worth drawing.

Reached from a distribution::

    Binomial(10, 0.5).spin()        # draw a value, show where it landed
    Binomial(10, 0.5).spinner()     # the dial alone, no spin

There are three dials. A discrete distribution gets the only one that suits
it -- a slice per outcome, sized by that outcome's probability. A continuous
one can be read two ways, and ``style=`` picks which:

    Normal(0, 1).spinner(style="increments")   # round steps of value
    Normal(0, 1).spinner(style="area")         # equally likely slices

Both draw the *same* map from bearing to value. What differs is only the
ruler printed on the rim, which is why either one samples correctly.

Notes
-----
Everything here rests on one invariant. A spin picks a bearing uniformly on
``[0, 360)``, and the value drawn is ``quantile(bearing / 360)`` -- see
:func:`spin_value`, which is the single place that map is written down.
Bearings are measured **clockwise from 12 o'clock**, which is not
matplotlib's convention for anything, so both conversions live in one place
too: :func:`pt` for a point and :func:`wedge_angles` for an arc.
"""

import math
import textwrap

import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge

# Importing plot applies symbulate.mplstyle (the Okabe-Ito palette, the
# viridis colormap, and the shared figure defaults), so SPINNER_PALETTE
# below can be read back out of the resulting rcParams rather than retyped
# -- the dial and every other Symbulate plot then cannot drift apart.
from .plot import SymbulatePlot
from .probability_space import rng
from .result import Scalar

# ---------------------------------------------------------------------------
# Geometry of the dial, in data units where the rim has radius 1 and the
# centre is the origin. The prototype's numbers are SVG pixels against a
# radius of 185; these are those numbers divided by 185.
# ---------------------------------------------------------------------------

SPINNER_RADIUS = 1.0  # the rim
SPINNER_AXIS_LIMIT = 1.42  # axes limits, leaving room for outside labels
SPINNER_FIGSIZE = (5.5, 5.5)  # a square dial; the mplstyle's 6.4x4.8 is not

# Radii, as fractions of SPINNER_RADIUS.
SPINNER_PROB_LABEL_RADIUS = 0.64  # percentage, written inside its own wedge
SPINNER_LABEL_PAD = 0.05  # gap between the rim and an outside label
SPINNER_LEADER_INNER = 1.00  # short leader tick from the rim...
SPINNER_LEADER_OUTER = 1.04  # ...outward, pointing at the value
SPINNER_NEEDLE_LENGTH = 0.94  # blade, centre to tip
SPINNER_NEEDLE_TAIL = 0.20  # counterweight, opposite the blade
SPINNER_NEEDLE_HALF_WIDTH = 0.016  # blade half-width at the pivot
SPINNER_PIVOT_RADIUS = 0.040
SPINNER_PIVOT_INNER_RADIUS = 0.014
SPINNER_BOUND_LABEL_RADIUS = 0.87  # the equal-area seam pair, *inside* the rim
SPINNER_SEAM_INNER = 0.924  # the dashed wrap marker at 12 o'clock...
SPINNER_SEAM_OUTER = 1.00  # ...running in from the rim

# Crowding budgets, in degrees of arc.
SPINNER_MIN_LABEL_SWEEP = 7.0  # a wedge narrower than this carries no
# percentage -- there is nothing to write it on
SPINNER_MIN_LABEL_GAP = 5.0  # fallback spacing between two value labels on
# the rim, used when the renderer cannot be
# measured. Radial text blocks only its glyph
# *height* tangentially, which is why this is so
# much smaller than tangential text would need.

# Line and fill weights.
SPINNER_WEDGE_EDGECOLOR = "white"
SPINNER_WEDGE_EDGEWIDTH = 1.4
SPINNER_RIM_COLOR = "#4f5c6c"
SPINNER_RIM_WIDTH = 2.0
SPINNER_LEADER_COLOR = "#8592a2"
SPINNER_LEADER_WIDTH = 1.3
SPINNER_HATCH_COLOR = "#0b0f16"

# The continuous dial's face. A pale family tint, not the categorical
# palette: on an increments dial the band *widths* carry the information, and
# colouring them would only decorate. If colour is ever wanted here it has to
# be a viridis ramp -- a continuous quantity is not a set of categories.
SPINNER_FACE_TINT = "#dcefe9"
SPINNER_BAND_EDGECOLOR = "#4f5c6c"  # hairline between two bands
SPINNER_BAND_EDGEWIDTH = 0.9
SPINNER_BAND_EDGEALPHA = 0.5

# The seam at 12 o'clock, where the largest values wrap back to the smallest.
SPINNER_SEAM_COLOR = "#c0392b"
SPINNER_SEAM_WIDTH = 2.2
SPINNER_SEAM_DASHES = (3.0, 2.5)

# The needle: a dark neutral, so it reads against every fill the ramp makes.
SPINNER_NEEDLE_COLOR = "#0f1720"

# Label sizes. `font.size` (10) comes from symbulate.mplstyle; these are the
# two relative steps matplotlib already names, so the dial follows the style
# file if its font size is ever changed.
SPINNER_VALUE_LABEL_SIZE = "medium"
SPINNER_PROB_LABEL_SIZE = "small"
SPINNER_VALUE_LABEL_COLOR = "#0f1720"

# A thin light halo, so a label survives the hatching underneath it.
SPINNER_LABEL_HALO_WIDTH = 2.2
SPINNER_LABEL_HALO_LIGHT = "white"
SPINNER_LABEL_HALO_DARK = "#0f1720"
SPINNER_LABEL_HALO_ALPHA = 0.7

# z-order, so the needle is never buried under a wedge.
SPINNER_WEDGE_Z = 1
SPINNER_RIM_Z = 3
SPINNER_LABEL_Z = 4
SPINNER_NEEDLE_Z = 5


# ---------------------------------------------------------------------------
# The continuous dials.
#
# Both draw the same map -- bearing to value is always quantile(bearing/360).
# What differs is the ruler printed on the rim:
#
#   "increments"  equal steps in VALUE, so the arc between two ticks is the
#                 probability of landing in that range: wide where the
#                 density is high, pinched in the tails.
#   "area"        equal steps in ANGLE, so every slice is exactly a 1-in-n
#                 chance, and the values round the rim bunch up where the
#                 distribution is dense.
# ---------------------------------------------------------------------------

SPINNER_STYLES = ("increments", "area")
SPINNER_DEFAULT_STYLE = "increments"

# How many sections each style wants, and the range it will accept. The two
# differ because they are counting different things: a handful of round bands
# reads well, while equal-area slices are a clock face and can take many more.
SPINNER_SECTIONS_DEFAULT = {"increments": 8, "area": 12}
SPINNER_SECTIONS_RANGE = {"increments": (3, 20), "area": (4, 60)}

# The 60 in that table is the one hard limit: past it two interval labels sit
# closer together than a label is tall, and an equal-area dial never drops a
# label (dropping one would merge two slices that are equally likely, which is
# the whole thing the dial exists to show).
SPINNER_AREA_MAX_SECTIONS = SPINNER_SECTIONS_RANGE["area"][1]

# How far either side of 12 o'clock the seam pair sits, as a fraction of one
# slice, and the floor/ceiling on that in degrees. Far enough apart to read as
# two numbers, close enough to read as one seam.
SPINNER_SEAM_LABEL_FRACTION = 0.35
SPINNER_SEAM_LABEL_MIN_OFFSET = 2.6
SPINNER_SEAM_LABEL_MAX_OFFSET = 7.0

# A label this close to straight up or straight down is written horizontally
# instead of radially. Radial text there comes out rotated a full 90 degrees,
# which leaves the infinity signs on an unbounded equal-area dial reading
# sideways -- visible in the prototype's screenshots, and avoidable here.
SPINNER_HORIZONTAL_LABEL_TOLERANCE = 12.0

# The ladder of round increment lengths, and how many bands a length has to
# give to be worth offering. 2.5 is in the ladder deliberately, so quarter
# steps (0.25, 2.5, 25) are offered rather than only the 1/2/5 decades.
SPINNER_INCREMENT_LADDER = (1.0, 2.0, 2.5, 5.0)
SPINNER_INCREMENT_MIN_BANDS = 3
SPINNER_INCREMENT_MAX_BANDS = 40

# A backstop on an absurdly small `increment`: without it a tiny step over a
# wide window would build tens of thousands of candidate ticks, drop nearly
# all of them, and draw a band per survivor.
SPINNER_MAX_INCREMENT_TICKS = 400

# The caption under an equal-area dial. Its point is the lesson: the slices
# are equally likely but the intervals are not equally long.
SPINNER_CAPTION_SIZE = "small"
SPINNER_CAPTION_COLOR = "#4f5c6c"
SPINNER_CAPTION_WIDTH = 78  # characters per line before wrapping
SPINNER_CAPTION_Y = 0.028  # in figure coordinates, up from the bottom
SPINNER_CAPTION_MARGIN = 0.10  # figure bottom margin reserved for it


# ---------------------------------------------------------------------------
# Colour. Seven hues cannot cover a support of twenty-one outcomes, so each
# lap of the palette is shifted in lightness and given a hatch.
# ---------------------------------------------------------------------------

# The Okabe-Ito categorical cycle, sky blue leading -- read from the
# rcParams symbulate.mplstyle just installed, never retyped.
SPINNER_PALETTE = tuple(plt.rcParams["axes.prop_cycle"].by_key()["color"])

# How far each lap is pushed toward white or black. Lap 0 is the palette
# itself. The magnitudes are deliberately not monotone: they are ordered so
# that consecutive laps of the *same* hue stay far apart in lightness.
LAP_MAG = [0, 0.32, 0.62, 0.80, 0.46, 0.16]

# matplotlib hatch strings, one per lap. Repeating a symbol makes it denser.
# Lap 2 is backslashes -- three of them, which Python source spells as six.
HATCH = [None, "///", "\\\\\\", "+++", "...", "////"]

# Which side of the lightness range a hue already sits on. Below this, the
# ramp lightens it; above, it darkens it. A fixed direction is not good
# enough: yellow is already light, so lightening it separates by almost
# nothing.
SPINNER_LAP_LUMINANCE_PIVOT = 0.40

# The dark ink is pure black, not an off-black. With an off-black there is a
# narrow band of mid lightness where neither white nor dark text reaches
# 4.5:1 -- and the shaded laps land in it. Pure black closes the gap; the
# worst case over every fill the ramp can produce becomes about 4.58:1, at
# the crossover where the two inks are equally good.
SPINNER_INK_DARK = "black"
SPINNER_INK_LIGHT = "white"


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


def spin_value(dist, bearing):
    """The value a needle resting at ``bearing`` has landed on.

    This is the contract the whole dial is built from::

        X = quantile(bearing / 360)

    so a quarter turn is the 25th percentile, half a turn the median, and
    three quarters the 75th. It is written down exactly once, here, and
    every other part of the module defers to it.

    Parameters
    ----------
    dist : Distribution
        The distribution being spun.
    bearing : float
        Degrees clockwise from 12 o'clock, in ``[0, 360]``.

    Returns
    -------
    float
        The value drawn. For a discrete distribution this is the outcome
        whose wedge contains ``bearing``.

    Examples
    --------
    >>> from symbulate import Normal
    >>> from symbulate.spinner import spin_value
    >>> spin_value(Normal(0, 1), 180.0) == Normal(0, 1).quantile(0.5)
    True
    """
    return dist.quantile(bearing / 360.0)


def bearing_of(dist, value):
    """The bearing a needle must rest at to point at ``value``.

    The inverse of :func:`spin_value`, used when a caller names the value
    instead of spinning for one. For a *discrete* distribution the cdf is a
    step function, so any bearing inside the outcome's wedge would do; the
    dial points at the middle of that wedge instead, which is what
    :func:`discrete_needle_bearing` works out.

    Parameters
    ----------
    dist : Distribution
        The distribution being spun.
    value : float
        The value to point at.

    Returns
    -------
    float
        Degrees clockwise from 12 o'clock.
    """
    return 360.0 * float(dist.cdf(value))


def pt(theta_deg, r, cx=0.0, cy=0.0):
    """A point at bearing ``theta_deg``, radius ``r``, in data coordinates.

    Bearings are measured **clockwise from 12 o'clock**, so ``0`` is the top,
    ``90`` the right, ``180`` the bottom and ``270`` the left.

    The sign here is the one thing in this module that is easy to get
    silently wrong. The prototype this was ported from is SVG, where y
    increases *downward*, and its helper subtracts the cosine term.
    matplotlib's y increases upward, so it adds. With the wrong sign the
    dial mirrors: the picture still looks entirely plausible, but three
    quarters of a turn lands on the 25th percentile.

    Parameters
    ----------
    theta_deg : float
        Bearing in degrees, clockwise from 12 o'clock.
    r : float
        Distance from the centre.
    cx, cy : float, optional
        The centre of the dial. Defaults to the origin.

    Returns
    -------
    tuple of float
        The ``(x, y)`` point.
    """
    a = math.radians(theta_deg)
    return cx + r * math.sin(a), cy + r * math.cos(a)


def wedge_angles(a0, a1):
    """Convert a bearing arc into the angles ``matplotlib.patches.Wedge`` wants.

    A ``Wedge`` measures counter-clockwise from the positive x-axis; a
    bearing runs clockwise from the positive y-axis. The conversion is the
    same reflection :func:`pt` applies, written for an arc.

    Parameters
    ----------
    a0, a1 : float
        Start and end bearings, in degrees clockwise from 12 o'clock, with
        ``a0 < a1``.

    Returns
    -------
    tuple of float
        ``(theta1, theta2)`` for ``Wedge``, with ``theta1 < theta2``.
    """
    return 90.0 - a1, 90.0 - a0


def label_rotation(bearing):
    """The rotation of a radial label written along ``bearing``.

    A label has to run **along its own spoke, outward**. matplotlib measures a
    text's rotation counter-clockwise from the positive x-axis, so text
    rotated by ``r`` reads in the direction ``(cos r, sin r)``; the outward
    direction at a bearing is ``(sin b, cos b)``. Setting the two equal gives
    ``r = 90 - b``.

    The sign there is the whole function, and getting it backwards
    (``b - 90``) is wrong everywhere except at 3 and 9 o'clock, where the two
    happen to agree. On the diagonals it lays each label *along* the rim, and
    at 12 and 6 o'clock it points the label straight **across the dial** --
    which is what a label colliding with the wheel instead of sitting on its
    tick looks like.

    The result is then folded into ``[-90, 90]`` so no label is ever upside
    down. The cost is that the reading direction reverses across the vertical,
    which is the standard and unavoidable trade for radial text on a full
    circle: the direction can be consistent or the text can always be upright,
    not both. A folded label reads inward, so its *other* end is the one
    anchored to the rim -- hence the second return value.

    Parameters
    ----------
    bearing : float
        Degrees clockwise from 12 o'clock.

    Returns
    -------
    tuple of (float, bool)
        The rotation in degrees, and whether it was folded (i.e. whether the
        label reads inward rather than outward, which decides which end of
        the string is anchored to the rim).
    """
    rotation = (90.0 - bearing + 180.0) % 360.0 - 180.0
    flipped = abs(rotation) > 90.0
    if flipped:
        rotation += -180.0 if rotation > 0.0 else 180.0
    return rotation, flipped


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def relative_luminance(color):
    """WCAG relative luminance of a color, from 0 (black) to 1 (white).

    Parameters
    ----------
    color : color
        Any matplotlib color spec.

    Returns
    -------
    float
        The relative luminance.

    Notes
    -----
    This is the gamma-corrected WCAG definition, not the plain weighted sum
    :func:`symbulate.plot._readable_text_color` uses. The two answer
    different questions: that one picks a label color for a mosaic tile,
    this one has to support a *measured* contrast ratio, and a contrast
    ratio is only defined against WCAG luminance.
    """
    channels = []
    for c in mcolors.to_rgb(color):
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _mix(color, target, fraction):
    """Move ``color`` ``fraction`` of the way toward ``target``."""
    a = mcolors.to_rgb(color)
    b = mcolors.to_rgb(target)
    return mcolors.to_hex(tuple(x + (y - x) * fraction for x, y in zip(a, b)))


def slice_fill(i):
    """The fill for the ``i``-th slice of a dial.

    Walks the categorical palette, and on each lap past the first shifts the
    hue in lightness -- **away from whichever end it already sits at**, so a
    dark hue lightens and a light one darkens. Ramping in one fixed
    direction is not good enough: yellow is already light, so lightening it
    separates by almost nothing.

    Together with :func:`slice_hatch` this yields 42 distinct fills, seven
    hues over six laps.

    Parameters
    ----------
    i : int
        The slice's index, from 0.

    Returns
    -------
    str
        A hex color.
    """
    base = SPINNER_PALETTE[i % len(SPINNER_PALETTE)]
    lap = min(i // len(SPINNER_PALETTE), len(LAP_MAG) - 1)
    if not LAP_MAG[lap]:
        return base
    toward = (
        "white" if relative_luminance(base) < SPINNER_LAP_LUMINANCE_PIVOT else "black"
    )
    return _mix(base, toward, LAP_MAG[lap])


def slice_hatch(i):
    """The hatch for the ``i``-th slice, or ``None`` on the first lap.

    Lightness does the real work of separating a repeated hue -- it is the
    one channel every form of color vision can read -- and the hatch is the
    belt-and-braces.

    Parameters
    ----------
    i : int
        The slice's index, from 0.

    Returns
    -------
    str or None
        A matplotlib hatch string, or ``None`` for an unhatched slice.
    """
    lap = min(i // len(SPINNER_PALETTE), len(HATCH) - 1)
    return HATCH[lap]


def ink_on(fill):
    """Whichever of white or black has more measured contrast on ``fill``.

    Parameters
    ----------
    fill : color
        The slice's fill color.

    Returns
    -------
    str
        ``"white"`` or ``"black"``.
    """
    luminance = relative_luminance(fill)
    light_ratio = 1.05 / (luminance + 0.05)
    dark_ratio = (luminance + 0.05) / 0.05
    return SPINNER_INK_LIGHT if light_ratio > dark_ratio else SPINNER_INK_DARK


# ---------------------------------------------------------------------------
# Slices
# ---------------------------------------------------------------------------


class Slice:
    """One wedge of a dial.

    Attributes
    ----------
    start, end : float
        The bearings the wedge spans, clockwise from 12 o'clock.
    value : float or None
        The outcome the wedge stands for, if it stands for one.
    probability : float
        The chance of landing in it.
    index : int
        Position in the dial, used to pick the fill and hatch.
    """

    def __init__(self, start, end, value, probability, index):
        self.start = start
        self.end = end
        self.value = value
        self.probability = probability
        self.index = index

    @property
    def sweep(self):
        """float : how wide the wedge is, in degrees."""
        return self.end - self.start

    @property
    def midpoint(self):
        """float : the bearing halfway across the wedge."""
        return 0.5 * (self.start + self.end)

    def __repr__(self):
        return (
            f"Slice(value={self.value!r}, "
            f"start={self.start:.3f}, end={self.end:.3f}, "
            f"probability={self.probability:.4f})"
        )


def discrete_slices(dist):
    """One slice per outcome, each sized by its probability.

    The boundaries come from the distribution's own cdf, so the dial is an
    exact drawing of the quantile function: the wedge a bearing falls in is
    the value :func:`spin_value` gives for that bearing.

    The support is trimmed by **Symbulate's own plotting window**
    (:attr:`Distribution.xlim`) rather than a second rule invented here, so
    a spinner covers the same outcomes ``Poisson(3).plot()`` draws. Whatever
    probability that leaves in the tails is absorbed into the first and last
    wedges, so the slices still tile the full circle.

    Parameters
    ----------
    dist : Distribution
        A univariate discrete distribution.

    Returns
    -------
    list of Slice
        The wedges, in bearing order from 12 o'clock.
    """
    low, high = dist.xlim
    values = np.arange(int(math.floor(low)), int(math.ceil(high)) + 1)
    probabilities = np.clip(np.asarray(dist.pmf(values), dtype=float), 0.0, None)

    # Drop outcomes carrying no probability at all -- a gap inside the
    # window (a lattice distribution, say) would otherwise be drawn as a
    # zero-width wedge and take a color out of the cycle for nothing.
    keep = probabilities > 0
    values, probabilities = values[keep], probabilities[keep]

    # Boundaries from the true cdf, so the wedges and the quantile contract
    # agree exactly. The ends are stretched to 0 and 360 to absorb whatever
    # was trimmed (under 0.1% by construction).
    upper = np.asarray(dist.cdf(values), dtype=float)
    edges = np.concatenate([[0.0], 360.0 * upper[:-1], [360.0]])

    return [
        Slice(
            start=float(edges[i]),
            end=float(edges[i + 1]),
            value=values[i].item(),
            probability=float(probabilities[i]),
            index=i,
        )
        for i in range(len(values))
    ]


def discrete_needle_bearing(dist, value):
    """Where the needle rests when asked to point at a discrete outcome.

    A discrete cdf is a step function, so ``360 * cdf(value)`` lands on the
    *edge* of the outcome's wedge, between two slices. The middle of the
    wedge is the honest place to point instead.

    Parameters
    ----------
    dist : Distribution
        A univariate discrete distribution.
    value : float
        The outcome to point at.

    Returns
    -------
    float
        The bearing, in degrees clockwise from 12 o'clock.

    Raises
    ------
    ValueError
        If ``value`` is not one of the outcomes the dial draws.
    """
    for sl in discrete_slices(dist):
        if sl.value == value:
            return sl.midpoint
    raise ValueError(
        f"{value!r} is not one of the outcomes on this spinner. "
        f"{type(dist).__name__}'s dial covers "
        f"{[sl.value for sl in discrete_slices(dist)][:8]}... -- pass one of "
        "those as `value=`, or call .spin() to draw one at random."
    )


# ---------------------------------------------------------------------------
# The continuous ruler
# ---------------------------------------------------------------------------


def nice_num(raw):
    """Round ``raw`` up to a tidy 1 / 2 / 5 times a power of ten.

    Parameters
    ----------
    raw : float
        A positive rough magnitude.

    Returns
    -------
    float
        The tidy value at or above it.
    """
    magnitude = 10.0 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized < 1.5:
        multiple = 1.0
    elif normalized < 3.5:
        multiple = 2.0
    elif normalized < 7.5:
        multiple = 5.0
    else:
        multiple = 10.0
    return multiple * magnitude


def nice_step(span, target_bands):
    """A tidy increment length giving roughly ``target_bands`` bands.

    Parameters
    ----------
    span : float
        The width of the window being ruled.
    target_bands : int
        Roughly how many bands are wanted.

    Returns
    -------
    float
        The increment length.
    """
    return nice_num(span / max(SPINNER_INCREMENT_MIN_BANDS, target_bands))


def increment_options(span):
    """Round increment lengths worth offering across a window of ``span``.

    Every ``multiple * 10 ** k`` for the multiples in
    ``SPINNER_INCREMENT_LADDER`` that lays between
    ``SPINNER_INCREMENT_MIN_BANDS`` and ``SPINNER_INCREMENT_MAX_BANDS`` bands
    across the window. The ladder includes **2.5**, so quarter steps (0.25,
    2.5, 25) are offered rather than only the 1 / 2 / 5 decades.

    Parameters
    ----------
    span : float
        The width of the window being ruled.

    Returns
    -------
    list of float
        The lengths, ascending.
    """
    options = []
    for k in range(-6, 7):
        for multiple in SPINNER_INCREMENT_LADDER:
            step = multiple * 10.0**k
            bands = span / step
            if SPINNER_INCREMENT_MIN_BANDS <= bands <= SPINNER_INCREMENT_MAX_BANDS:
                options.append(step)
    return sorted(options)


def display_range(dist):
    """The window the *ruler* covers, for a continuous distribution.

    The dial always covers the full 0-360 degrees, i.e. all the probability;
    this is only how much of the value axis gets printed round the rim.

    Symbulate distributions already frame themselves --
    :attr:`Distribution.xlim` uses the exact support where there is one and a
    quantile cut where there is not, and zooms itself when the probability
    fills too little of the result. Reusing it means a spinner and a
    ``.plot()`` of the same distribution show the same stretch of the axis.

    Parameters
    ----------
    dist : Distribution
        A univariate continuous distribution.

    Returns
    -------
    tuple of float
        The ``(low, high)`` window.
    """
    low, high = dist.xlim
    return float(low), float(high)


def _resolve_style(dist, style):
    """Settle which dial to draw, before anything is built.

    Resolving up front is deliberate: ``sections`` means a different thing in
    each style and is bounded differently, so a dial built before the style
    was settled can end up captioned with the other style's numbers.

    Parameters
    ----------
    dist : Distribution
        The distribution being spun.
    style : str or None
        What the caller asked for.

    Returns
    -------
    str or None
        The style to draw, or ``None`` for a discrete distribution.

    Raises
    ------
    ValueError
        For an unrecognized style, or any style on a discrete distribution.
    """
    if getattr(dist, "discrete", False):
        if style is not None:
            raise ValueError(
                f"A discrete distribution has only one spinner dial -- one "
                f"slice per outcome, sized by its probability -- so it takes "
                f"no style. Drop the style= argument, or use it on a "
                f"continuous distribution, where it chooses between "
                f"{' and '.join(repr(s) for s in SPINNER_STYLES)}."
            )
        return None
    if style is None:
        return SPINNER_DEFAULT_STYLE
    if style not in SPINNER_STYLES:
        raise ValueError(
            f"{style!r} is not a spinner style. Use "
            f"style='increments' (the default -- equal steps in value, so "
            f"each slice's size is the chance of landing in it) or "
            f"style='area' (equal slices, so every one is the same chance)."
        )
    return style


def _resolve_sections(style, sections, increment=None):
    """Settle how many sections the dial has, before anything is built.

    Parameters
    ----------
    style : str
        The resolved style.
    sections : int or None
        What the caller asked for.
    increment : float or None
        An explicit increment length, which sets the band count itself.

    Returns
    -------
    int
        The section count.

    Raises
    ------
    ValueError
        If ``sections`` is outside the range this style accepts, or if both
        ``sections`` and ``increment`` were given.
    """
    if increment is not None:
        if sections is not None:
            raise ValueError(
                "Give the spinner either `sections=` (how many bands you "
                "want, and their length is worked out for you) or "
                "`increment=` (the length itself, and the count follows from "
                "it) -- not both, since each one determines the other."
            )
        return SPINNER_SECTIONS_DEFAULT[style]
    if sections is None:
        return SPINNER_SECTIONS_DEFAULT[style]
    low, high = SPINNER_SECTIONS_RANGE[style]
    if not _is_whole(sections) or not low <= sections <= high:
        reason = (
            f" Past {SPINNER_AREA_MAX_SECTIONS} slices, two of the values "
            "round the rim sit closer together than a label is tall, and an "
            "equal-area dial never drops one -- every slice is the same "
            "chance, so every boundary has to say where it is."
            if style == "area"
            else " More than about 20 round bands leaves the tails too "
            "pinched to read; try style='area' for a finer dial."
        )
        raise ValueError(
            f"sections={sections!r} is not a section count style={style!r} "
            f"can draw. Use a whole number from {low} to {high}.{reason}"
        )
    return int(sections)


def _is_whole(value):
    """Whether ``value`` is a number with no fractional part."""
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def area_slices(dist, sections):
    """``sections`` slices of equal area -- each exactly a 1-in-n chance.

    Equally spaced in *angle*, so the boundary values (each a quantile) bunch
    together where the distribution is dense. That uneven spacing is the
    lesson the dial exists to teach: equal chance, unequal interval length.

    Parameters
    ----------
    dist : Distribution
        A univariate continuous distribution.
    sections : int
        How many slices.

    Returns
    -------
    tuple of (list of Slice, list of float)
        The slices, and the boundary value at the start of each -- so
        ``boundaries[i]`` is ``quantile(i / sections)``.
    """
    step = 360.0 / sections
    low, high = display_range(dist)
    width = high - low

    boundaries = []
    slices = []
    for i in range(sections):
        # quantile(0.5) and friends land on floating-point dust rather than
        # exactly 0, which prints as -1.6e-05. Snap the dust away.
        value = float(dist.quantile(i / sections))
        if abs(value) < 1e-4 * width:
            value = 0.0
        boundaries.append(value)
        slices.append(
            Slice(
                start=i * step,
                end=(i + 1) * step,
                value=None,  # a slice is an interval here, not one outcome
                probability=1.0 / sections,
                index=i,
            )
        )
    return slices, boundaries


def area_bound_labels(dist):
    """The two values that meet at the seam, for an equal-area dial.

    Going clockwise the dial starts at the smallest value and comes back
    round to the largest, so 12 o'clock carries *both* ends.

    Whether an end is a real bound is settled **exactly**, by asking scipy
    for it: a finite ``quantile(0)`` or ``quantile(1)`` is a genuine support
    bound (Uniform's 0 and 1, Exponential's 0, Pareto's scale), an infinite
    one is a tail and is written as an infinity sign. The prototype guesses
    this from how close the window is to the quantile; there is no need to
    guess here.

    Parameters
    ----------
    dist : Distribution
        A univariate continuous distribution.

    Returns
    -------
    tuple of str
        ``(smallest, largest)``, formatted for the dial.
    """
    lowest = float(dist.quantile(0.0))
    highest = float(dist.quantile(1.0))
    return (
        _format_continuous(lowest) if np.isfinite(lowest) else "−∞",
        _format_continuous(highest) if np.isfinite(highest) else "∞",
    )


def area_caption(sections):
    """The sentence written under an equal-area dial."""
    return (
        f"The spinner lands in each interval with probability "
        f"{_percent(1.0 / sections)} (1 in {sections}). Notice the intervals "
        f"do not all have the same length. "
    )


class Tick:
    """One labelled mark on an increments dial's rim.

    Attributes
    ----------
    bearing : float
        Where it sits, in degrees clockwise from 12 o'clock.
    value : float
        The round value it stands for.
    """

    def __init__(self, bearing, value):
        self.bearing = bearing
        self.value = value

    def __repr__(self):
        return f"Tick(value={self.value!r}, bearing={self.bearing:.3f})"


def increment_slices(dist, sections=None, increment=None):
    """A ruler of round values, and the bands between them.

    Equal steps in **value**, so the arc between two ticks is the probability
    of landing in that range -- wide where the density is high, pinched in the
    tails.

    The order of operations here is load-bearing, and is the third item in the
    specification's list of things that will bite you. Label placement is
    settled *first*, ticks with nowhere to put their value are dropped
    outright, and only then are the bands built from the survivors -- so every
    boundary the picture quotes is a tick the reader can actually see.

    Parameters
    ----------
    dist : Distribution
        A univariate continuous distribution.
    sections : int, optional
        Roughly how many bands are wanted; the length is made tidy.
    increment : float, optional
        The length itself, instead of a count.

    Returns
    -------
    tuple of (list of Slice, list of Tick, float)
        The bands, the surviving ticks, and the increment length used.

    Raises
    ------
    ValueError
        If ``increment`` is not positive, or is so small that the window
        would need an unreasonable number of ticks.
    """
    low, high = display_range(dist)
    span = high - low

    if increment is None:
        step = nice_step(span, sections)
    else:
        if not isinstance(increment, (int, float)) or increment <= 0:
            raise ValueError(
                f"increment={increment!r} is not a length. It is the size of "
                f"one band in the units of the distribution, so it has to be "
                f"a positive number -- for this distribution the tidy choices "
                f"are {[round(s, 10) for s in increment_options(span)]}."
            )
        step = float(increment)
        if span / step > SPINNER_MAX_INCREMENT_TICKS:
            raise ValueError(
                f"increment={increment!r} is too small for this "
                f"distribution: it would rule the window ({low:.4g} to "
                f"{high:.4g}) into more than {SPINNER_MAX_INCREMENT_TICKS} "
                f"bands, far more than a dial can label. The tidy choices "
                f"here are {[round(s, 10) for s in increment_options(span)]}."
            )

    # Candidate ticks: every multiple of `step` in the window, plus the ones
    # just outside it, so a band always closes at a round value.
    candidates = []
    last_bearing = None
    for k in range(math.floor(low / step), math.ceil(high / step) + 1):
        value = k * step
        bearing = 360.0 * float(dist.cdf(value))
        if not 0.0 <= bearing <= 360.0:
            continue
        if last_bearing is not None and abs(bearing - last_bearing) < 1e-9:
            continue  # the same point on the dial
        last_bearing = bearing
        # Snap floating-point dust so a tick at the origin prints as "0".
        candidates.append(Tick(bearing, 0.0 if abs(value) < step * 1e-6 else value))

    # A tick's label competes for room with its neighbours, and the ticks
    # bunch up in the tails where the bands are thin. Priority is the wider of
    # the two bands a tick separates, so the crowded tail ticks give way and
    # the ruler stays readable through the body of the distribution.
    bearings = [tick.bearing for tick in candidates]
    edges = [0.0] + bearings + [360.0]
    priority = [
        max(bearings[i] - edges[i], edges[i + 2] - bearings[i])
        for i in range(len(bearings))
    ]
    ticks = [
        tick
        for tick, has_room in zip(candidates, _label_bearings(bearings, priority))
        if has_room
    ]

    # Bands from the survivors: an open low tail, one per gap, an open high
    # tail. Together they are exactly 360 degrees, so their probabilities sum
    # to 1 whatever was dropped.
    band_edges = [0.0] + [tick.bearing for tick in ticks] + [360.0]
    slices = []
    for i in range(len(band_edges) - 1):
        start, end = band_edges[i], band_edges[i + 1]
        if end - start <= 1e-9:
            continue
        slices.append(
            Slice(
                start=start,
                end=end,
                value=None,
                probability=(end - start) / 360.0,
                index=len(slices),
            )
        )
    return slices, ticks, step


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _dial_axes(ax=None):
    """Get and frame the axes a dial is drawn on.

    The dial fills its whole figure and supplies its own scale, so the axes
    furniture is turned off -- including the y-grid ``symbulate.mplstyle``
    turns on for ordinary plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Draw here instead of building a figure.

    Returns
    -------
    matplotlib.axes.Axes
        The framed axes.

    Raises
    ------
    ValueError
        If the current figure already has a plot on it. A dial re-partitions
        the whole canvas, so a second one would completely cover the first;
        this is the same rule the pairs matrix and the marginal layout
        follow.
    """
    if ax is None:
        fig = plt.gcf()
        if fig.axes:
            raise ValueError(
                "A spinner fills its whole figure, so it cannot be drawn on "
                "top of another plot. Start a new figure first -- call "
                "figure() before the spinner -- or pass an axes of your own "
                "with .spinner(ax=...)."
            )
        fig.set_size_inches(*SPINNER_FIGSIZE)
        ax = fig.add_subplot(111)

    ax.set_aspect("equal")
    ax.set_xlim(-SPINNER_AXIS_LIMIT, SPINNER_AXIS_LIMIT)
    ax.set_ylim(-SPINNER_AXIS_LIMIT, SPINNER_AXIS_LIMIT)
    ax.grid(False)
    ax.set_axis_off()
    return ax


def _near_vertical(bearing):
    """Whether ``bearing`` points close enough to straight up or down.

    A radial label there comes out rotated a full 90 degrees. See
    ``SPINNER_HORIZONTAL_LABEL_TOLERANCE``.
    """
    from_vertical = min(bearing % 180.0, 180.0 - bearing % 180.0)
    return from_vertical <= SPINNER_HORIZONTAL_LABEL_TOLERANCE


def _halo_effects(halo):
    """A thin outline behind a label's glyphs, so a hatch cannot swallow it."""
    if halo is None:
        return None
    return [
        path_effects.withStroke(
            linewidth=SPINNER_LABEL_HALO_WIDTH,
            foreground=halo,
            alpha=SPINNER_LABEL_HALO_ALPHA,
        )
    ]


def _horizontal_text(ax, bearing, text, radius, color, size, halo, zorder):
    """Write ``text`` upright at ``bearing``, pushed clear of the dial.

    Used at the very top and bottom, where radial text would read sideways.
    Anchored by whichever edge faces the centre, so it grows away from the
    dial exactly as a radial label does.
    """
    x, y = pt(bearing, radius)
    return ax.text(
        x,
        y,
        text,
        rotation=0.0,
        ha="center",
        va="bottom" if math.cos(math.radians(bearing)) > 0 else "top",
        color=color,
        fontsize=size,
        zorder=zorder,
        path_effects=_halo_effects(halo),
    )


def _radial_text(
    ax, bearing, text, radius, color, size, halo, zorder, horizontal=False
):
    """Write ``text`` along its own spoke, running outward from ``radius``.

    Anchoring, rather than measuring. The label is rotated and then pinned by
    whichever *end* of the string sits nearest the centre
    (``rotation_mode="anchor"``), so it grows outward on its own. That is
    exactly equivalent to centring it at ``radius + half the text width``,
    without having to ask the renderer how wide it came out -- and it stays
    right when the figure is resized or saved at a different dpi.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The dial's axes.
    bearing : float
        Degrees clockwise from 12 o'clock.
    text : str
        What to write.
    radius : float
        Where the near end of the string is pinned.
    color : color
        The ink.
    size : str or float
        Font size.
    halo : color or None
        A thin outline drawn behind the glyphs, so they survive a hatch.
    zorder : float
        Drawing order.
    horizontal : bool, optional
        Write the label upright rather than radially when it sits close to
        straight up or straight down. Radial text there is rotated a full 90
        degrees, which leaves an infinity sign reading sideways.

    Returns
    -------
    matplotlib.text.Text
        The label.
    """
    if horizontal and _near_vertical(bearing):
        return _horizontal_text(ax, bearing, text, radius, color, size, halo, zorder)
    rotation, flipped = label_rotation(bearing)
    x, y = pt(bearing, radius)
    return ax.text(
        x,
        y,
        text,
        rotation=rotation,
        rotation_mode="anchor",
        # A folded label reads inward, so it is its *right* edge that sits
        # nearest the centre; an unfolded one grows from its left.
        ha="right" if flipped else "left",
        va="center",
        color=color,
        fontsize=size,
        zorder=zorder,
        path_effects=_halo_effects(halo),
    )


def _centred_radial_text(ax, bearing, text, radius, color, size, halo, zorder):
    """Write ``text`` along its spoke, centred on ``radius``.

    Used inside a wedge, where the label is placed at a radius rather than
    hung off the rim.
    """
    rotation, _ = label_rotation(bearing)
    x, y = pt(bearing, radius)
    return ax.text(
        x,
        y,
        text,
        rotation=rotation,
        rotation_mode="anchor",
        ha="center",
        va="center",
        color=color,
        fontsize=size,
        zorder=zorder,
        path_effects=_halo_effects(halo),
    )


def _label_bearings(bearings, priority=None, min_gap=SPINNER_MIN_LABEL_GAP):
    """Which of ``bearings`` have room for a label on the rim.

    A radial label blocks only its glyph *height* tangentially, not its
    width, which is why radial text beats tangential here and why crowding
    largely disappears -- but a very fine dial can still run two labels into
    each other.

    Two things decide the outcome:

    The check is **cyclic**. 0 and 360 are the same point on the dial, so a
    bounded distribution's first and last labels would otherwise be placed
    on top of one another -- which is exactly what happens in the prototype,
    whose scan never compares the last candidate against the first.

    Candidates are considered **in order of ``priority``, not of bearing**.
    Sweeping round the dial and keeping the first of every crowded group
    hands the label to whichever wedge happens to come first, which is not
    the one that deserves it: on ``Binomial(10, 0.5)`` the outcome ``0``
    carries 0.1% of the probability and a third of a degree of arc, yet it
    would take the label and push out ``9``, ten times likelier and drawn on
    ten times the arc. Passing each candidate's sweep as its priority gives
    the label to the wedge with the room for it.

    Parameters
    ----------
    bearings : sequence of float
        Candidate bearings.
    priority : sequence of float, optional
        How much each candidate deserves its label -- its wedge's sweep, in
        practice. Higher wins. ``None`` falls back to bearing order.
    min_gap : float, optional
        The smallest acceptable spacing, in degrees.

    Returns
    -------
    list of bool
        One flag per candidate, in the order the candidates were given.
    """
    n = len(bearings)
    if priority is None:
        order = range(n)
    else:
        # Descending priority, ties broken by bearing so the result does not
        # depend on how the candidates happened to be listed.
        order = sorted(range(n), key=lambda i: (-priority[i], bearings[i]))

    placed = []
    keep = [False] * n
    for i in order:
        bearing = bearings[i]
        separation = (
            min(abs(bearing - other), 360.0 - abs(bearing - other)) for other in placed
        )
        if all(gap >= min_gap for gap in separation):
            keep[i] = True
            placed.append(bearing)
    return keep


def _percent(probability):
    """Format a probability the way the dial writes it inside a wedge."""
    percent = 100.0 * probability
    if percent >= 99.95:
        return "100%"
    if percent >= 10:
        return f"{percent:.1f}".rstrip("0").rstrip(".") + "%"
    if percent >= 1:
        return f"{percent:.1f}%"
    if percent >= 0.1:
        return f"{percent:.2f}%"
    return "<0.1%"


def _format_value(value):
    """Format a discrete outcome for its label on the rim."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.4g}"


def _format_continuous(value):
    """Format a value on a continuous dial's rim.

    Two decimals through the ordinary range, thinning out as the magnitude
    grows so a label never runs longer than it has to. A value snapped to
    zero prints as a bare ``0`` rather than ``0.00``, which is how the
    prototype's dials read.
    """
    magnitude = abs(value)
    if magnitude < 1e-9:
        return "0"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 0.01:
        return f"{value:.2f}"
    return f"{value:.1e}"


def draw_needle(ax, bearing):
    """Draw the needle resting at ``bearing``.

    A slender tapered blade with a short counterweight opposite, pivoting on
    the centre of the dial. The dial itself never moves -- that is what keeps
    the 75th percentile permanently at three quarters of a turn -- so this is
    the only part of the picture that depends on the value drawn.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The dial's axes.
    bearing : float
        Degrees clockwise from 12 o'clock.
    """
    tip = pt(bearing, SPINNER_NEEDLE_LENGTH)
    tail = pt(bearing + 180.0, SPINNER_NEEDLE_TAIL)
    left = pt(bearing - 90.0, SPINNER_NEEDLE_HALF_WIDTH)
    right = pt(bearing + 90.0, SPINNER_NEEDLE_HALF_WIDTH)

    ax.add_patch(
        Polygon(
            [tail, left, tip, right],
            closed=True,
            facecolor=SPINNER_NEEDLE_COLOR,
            edgecolor="none",
            zorder=SPINNER_NEEDLE_Z,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            SPINNER_PIVOT_RADIUS,
            facecolor=SPINNER_NEEDLE_COLOR,
            edgecolor="white",
            linewidth=1.6,
            zorder=SPINNER_NEEDLE_Z + 1,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            SPINNER_PIVOT_INNER_RADIUS,
            facecolor="white",
            edgecolor="none",
            zorder=SPINNER_NEEDLE_Z + 2,
        )
    )


def draw_discrete_dial(dist, ax):
    """Draw the "sized by probability" dial: one wedge per outcome.

    Parameters
    ----------
    dist : Distribution
        A univariate discrete distribution.
    ax : matplotlib.axes.Axes
        Framed axes, from :func:`_dial_axes`.

    Returns
    -------
    list of Slice
        The wedges drawn.
    """
    slices = discrete_slices(dist)

    for sl in slices:
        theta1, theta2 = wedge_angles(sl.start, sl.end)
        fill = slice_fill(sl.index)
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                SPINNER_RADIUS,
                theta1,
                theta2,
                facecolor=fill,
                edgecolor=SPINNER_WEDGE_EDGECOLOR,
                linewidth=SPINNER_WEDGE_EDGEWIDTH,
                zorder=SPINNER_WEDGE_Z,
            )
        )
        hatch = slice_hatch(sl.index)
        if hatch is not None:
            # matplotlib draws a hatch in the patch's *edge* color, and these
            # wedges are edged in white to separate them -- so the hatch goes
            # on as a second, edgeless overlay to keep its own dark color.
            ax.add_patch(
                Wedge(
                    (0.0, 0.0),
                    SPINNER_RADIUS,
                    theta1,
                    theta2,
                    facecolor="none",
                    edgecolor=SPINNER_HATCH_COLOR,
                    linewidth=0.0,
                    hatch=hatch,
                    zorder=SPINNER_WEDGE_Z + 1,
                )
            )

    # The rim, over the wedges so their white edges stop cleanly on it.
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            SPINNER_RADIUS,
            facecolor="none",
            edgecolor=SPINNER_RIM_COLOR,
            linewidth=SPINNER_RIM_WIDTH,
            zorder=SPINNER_RIM_Z,
        )
    )

    # Value labels, just outside the rim, each on its own leader tick. A
    # crowded group hands its one label to the widest wedge in it, not to
    # whichever comes first going clockwise -- see `_label_bearings`.
    midpoints = [sl.midpoint for sl in slices]
    sweeps = [sl.sweep for sl in slices]
    for sl, has_room in zip(slices, _label_bearings(midpoints, priority=sweeps)):
        if not has_room:
            continue  # skip the label, keep the wedge
        inner = pt(sl.midpoint, SPINNER_LEADER_INNER)
        outer = pt(sl.midpoint, SPINNER_LEADER_OUTER)
        ax.plot(
            [inner[0], outer[0]],
            [inner[1], outer[1]],
            color=SPINNER_LEADER_COLOR,
            linewidth=SPINNER_LEADER_WIDTH,
            zorder=SPINNER_LABEL_Z,
        )
        _radial_text(
            ax,
            sl.midpoint,
            _format_value(sl.value),
            radius=SPINNER_RADIUS + SPINNER_LABEL_PAD,
            color=SPINNER_VALUE_LABEL_COLOR,
            size=SPINNER_VALUE_LABEL_SIZE,
            halo=SPINNER_LABEL_HALO_LIGHT,
            zorder=SPINNER_LABEL_Z,
        )

    # Probability labels, inside the wedges that are wide enough to hold one.
    for sl in slices:
        if sl.sweep < SPINNER_MIN_LABEL_SWEEP:
            continue
        ink = ink_on(slice_fill(sl.index))
        _centred_radial_text(
            ax,
            sl.midpoint,
            _percent(sl.probability),
            radius=SPINNER_PROB_LABEL_RADIUS,
            color=ink,
            size=SPINNER_PROB_LABEL_SIZE,
            halo=(
                SPINNER_LABEL_HALO_DARK
                if ink == SPINNER_INK_LIGHT
                else SPINNER_LABEL_HALO_LIGHT
            ),
            zorder=SPINNER_LABEL_Z,
        )

    return slices


def _draw_rim(ax, face=None):
    """The dial's face and its rim."""
    if face is not None:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                SPINNER_RADIUS,
                facecolor=face,
                edgecolor="none",
                zorder=SPINNER_WEDGE_Z - 1,
            )
        )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            SPINNER_RADIUS,
            facecolor="none",
            edgecolor=SPINNER_RIM_COLOR,
            linewidth=SPINNER_RIM_WIDTH,
            zorder=SPINNER_RIM_Z,
        )
    )


def _draw_seam(ax):
    """The dashed marker at 12 o'clock, where the dial wraps.

    Going clockwise a continuous dial runs from the smallest value to the
    largest, so this is the one place on the rim where the values jump.
    """
    inner = pt(0.0, SPINNER_SEAM_INNER)
    outer = pt(0.0, SPINNER_SEAM_OUTER)
    ax.plot(
        [inner[0], outer[0]],
        [inner[1], outer[1]],
        color=SPINNER_SEAM_COLOR,
        linewidth=SPINNER_SEAM_WIDTH,
        dashes=SPINNER_SEAM_DASHES,
        zorder=SPINNER_RIM_Z + 1,
    )


def _draw_band_percentages(ax, slices):
    """Write each band's chance inside it, where there is room for it."""
    for sl in slices:
        if sl.sweep < SPINNER_MIN_LABEL_SWEEP:
            continue
        _centred_radial_text(
            ax,
            sl.midpoint,
            _percent(sl.probability),
            radius=SPINNER_PROB_LABEL_RADIUS,
            color=SPINNER_INK_DARK,
            size=SPINNER_PROB_LABEL_SIZE,
            halo=SPINNER_LABEL_HALO_LIGHT,
            zorder=SPINNER_LABEL_Z,
        )


def draw_increments_dial(dist, ax, sections=None, increment=None):
    """Draw the equal-increments dial: a ruler of round values.

    Parameters
    ----------
    dist : Distribution
        A univariate continuous distribution.
    ax : matplotlib.axes.Axes
        Framed axes, from :func:`_dial_axes`.
    sections : int, optional
        Roughly how many bands.
    increment : float, optional
        The band length, instead of a count.

    Returns
    -------
    tuple of (list of Slice, list of Tick, float)
        The bands, the ticks, and the increment used.
    """
    slices, ticks, step = increment_slices(dist, sections=sections, increment=increment)

    # The bands are drawn, because their arcs are unequal and that inequality
    # *is* the information -- the wide ones are where the density is high.
    # They are left uncoloured on purpose: the widths already say it.
    for sl in slices:
        theta1, theta2 = wedge_angles(sl.start, sl.end)
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                SPINNER_RADIUS,
                theta1,
                theta2,
                facecolor=SPINNER_FACE_TINT,
                edgecolor=SPINNER_BAND_EDGECOLOR,
                linewidth=SPINNER_BAND_EDGEWIDTH,
                alpha=SPINNER_BAND_EDGEALPHA,
                zorder=SPINNER_WEDGE_Z,
            )
        )

    _draw_rim(ax, face=SPINNER_FACE_TINT)
    _draw_seam(ax)

    # Every drawn tick carries its value -- the ones that could not were
    # dropped in `increment_slices`, before the bands were built.
    for tick in ticks:
        inner = pt(tick.bearing, SPINNER_LEADER_INNER)
        outer = pt(tick.bearing, SPINNER_LEADER_OUTER)
        ax.plot(
            [inner[0], outer[0]],
            [inner[1], outer[1]],
            color=SPINNER_LEADER_COLOR,
            linewidth=SPINNER_LEADER_WIDTH,
            zorder=SPINNER_LABEL_Z,
        )
        _radial_text(
            ax,
            tick.bearing,
            _format_continuous(tick.value),
            radius=SPINNER_RADIUS + SPINNER_LABEL_PAD,
            color=SPINNER_VALUE_LABEL_COLOR,
            size=SPINNER_VALUE_LABEL_SIZE,
            halo=SPINNER_LABEL_HALO_LIGHT,
            zorder=SPINNER_LABEL_Z,
        )

    _draw_band_percentages(ax, slices)
    return slices, ticks, step


def draw_area_dial(dist, ax, sections):
    """Draw the equal-area dial: ``sections`` slices of the same chance.

    No tick marks and no slice dividers. The numbers round the rim are the
    scale, and their uneven spacing is the lesson -- equal chance, unequal
    interval length. A mark at every boundary would invite reading the dial as
    an evenly graduated ruler, which is the one thing it is not.

    Parameters
    ----------
    dist : Distribution
        A univariate continuous distribution.
    ax : matplotlib.axes.Axes
        Framed axes, from :func:`_dial_axes`.
    sections : int
        How many equally likely slices.

    Returns
    -------
    tuple of (list of Slice, list of float)
        The slices and their boundary values.
    """
    slices, boundaries = area_slices(dist, sections)

    _draw_rim(ax, face=SPINNER_FACE_TINT)
    _draw_seam(ax)

    # Interval boundaries, every one labelled. `sections` is capped at 60 so
    # this never has to fight for room, which is why nothing is dropped here.
    #
    # These stay radial even at the top and bottom, where the text comes out
    # rotated a full 90 degrees. The upright exception below is safe only
    # because the seam pair has its own ring to itself: at 60 slices these are
    # 6 degrees apart, so three of them land within the upright tolerance of
    # straight down -- and upright text is far wider than it is tall, so all
    # three would collide. Radial text is what makes a fine dial fit at all.
    for i, value in enumerate(boundaries):
        if i == 0:
            continue  # the seam carries this one, as a pair, below
        _radial_text(
            ax,
            slices[i].start,
            _format_continuous(value),
            radius=SPINNER_RADIUS + SPINNER_LABEL_PAD,
            color=SPINNER_VALUE_LABEL_COLOR,
            size=SPINNER_VALUE_LABEL_SIZE,
            halo=SPINNER_LABEL_HALO_LIGHT,
            zorder=SPINNER_LABEL_Z,
        )

    # The seam pair: smallest value just clockwise of 12 o'clock, largest just
    # anticlockwise. They sit *inside* the rim, where the interval labels never
    # go, so however many slices there are the two groups cannot collide.
    smallest, largest = area_bound_labels(dist)
    offset = min(
        SPINNER_SEAM_LABEL_MAX_OFFSET,
        max(
            SPINNER_SEAM_LABEL_MIN_OFFSET,
            (360.0 / sections) * SPINNER_SEAM_LABEL_FRACTION,
        ),
    )
    for bearing, text in ((offset, smallest), (360.0 - offset, largest)):
        _radial_text(
            ax,
            bearing,
            text,
            radius=SPINNER_BOUND_LABEL_RADIUS,
            color=SPINNER_VALUE_LABEL_COLOR,
            size=SPINNER_VALUE_LABEL_SIZE,
            halo=SPINNER_LABEL_HALO_LIGHT,
            zorder=SPINNER_LABEL_Z,
            horizontal=True,
        )

    # No percentage inside the slices, unlike the other two dials. Every slice
    # here carries the *same* chance, so writing it out once per slice would
    # print one number twelve times and crowd out the values that differ. The
    # caption says it once instead.
    return slices, boundaries


def _draw_caption(ax, caption, owns_figure):
    """Write ``caption`` under the dial, if this figure is ours to write on.

    A caller who passed their own axes is placing the dial in a layout of
    their own, so the figure's margins are not ours to move; the sentence is
    still on the returned plot as ``.caption``.
    """
    if caption is None or not owns_figure:
        return
    figure = ax.get_figure()
    figure.subplots_adjust(bottom=SPINNER_CAPTION_MARGIN)
    figure.text(
        0.5,
        SPINNER_CAPTION_Y,
        "\n".join(textwrap.wrap(caption, SPINNER_CAPTION_WIDTH)),
        ha="center",
        va="bottom",
        fontsize=SPINNER_CAPTION_SIZE,
        color=SPINNER_CAPTION_COLOR,
    )


class SpinnerPlot(SymbulatePlot):
    """The plot object a spinner returns.

    Extends :class:`~symbulate.plot.SymbulatePlot` with what the spin
    actually produced, so a caller can check the geometry without reading it
    back off the picture.

    Attributes
    ----------
    ax : matplotlib.axes.Axes
        The axes the dial was drawn on.
    bearing : float
        Where the needle rests, in degrees clockwise from 12 o'clock.
    value : float or None
        The value the needle points at, or ``None`` for a dial drawn with no
        needle position asked for.
    slices : list of Slice
        The wedges drawn.
    style : str or None
        Which dial was drawn -- ``"increments"`` or ``"area"`` for a
        continuous distribution, ``None`` for a discrete one, which has only
        the one dial.
    sections : int or None
        How many bands or equal-area slices were asked for.
    increment : float or None
        The band length an increments dial actually used, tidied.
    ticks : list of Tick
        The labelled marks on an increments dial's rim; empty otherwise.
    boundaries : list of float
        The interval boundary values on an equal-area dial; empty otherwise.
    caption : str or None
        The sentence written under the dial, if it has one.
    """

    def __init__(
        self,
        ax,
        bearing,
        value,
        slices,
        style=None,
        sections=None,
        increment=None,
        ticks=(),
        boundaries=(),
        caption=None,
    ):
        super().__init__(ax)
        self.bearing = bearing
        self.value = value
        self.slices = slices
        self.style = style
        self.sections = sections
        self.increment = increment
        self.ticks = list(ticks)
        self.boundaries = list(boundaries)
        self.caption = caption


def make_spinner(
    dist,
    value=None,
    seed=None,
    ax=None,
    style=None,
    sections=None,
    increment=None,
):
    """Spin a dial for ``dist``, or draw it pointing at a value you choose.

    Parameters
    ----------
    dist : Distribution
        A univariate distribution.
    value : float, optional
        Rest the needle on this value instead of spinning for one.
    seed : int, optional
        Makes a spin reproducible. Cannot be combined with ``value``, which
        leaves nothing random to seed.
    ax : matplotlib.axes.Axes, optional
        Draw here instead of building a figure.
    style : {'increments', 'area'}, optional
        Which continuous dial to draw. Refused for a discrete distribution,
        which has only one.
    sections : int, optional
        How many bands (``'increments'``) or equally likely slices
        (``'area'``).
    increment : float, optional
        A band length for ``'increments'``, instead of a count.

    Returns
    -------
    SpinnerPlot
        The dial, carrying the ``value`` the needle landed on.
    """
    _reject_multivariate(dist)

    # Everything is settled before a single patch is drawn. Two reasons: a
    # `value` that is not on the dial, or a `sections` the style cannot take,
    # is then refused while the figure is still clean; and `sections` means a
    # different thing per style, so a dial built before the style was resolved
    # can end up captioned with the other style's numbers.
    style = _resolve_style(dist, style)
    discrete = style is None
    if discrete and (sections is not None or increment is not None):
        raise ValueError(
            "A discrete spinner has one slice per outcome, sized by its "
            "probability, so there is no section count or increment to "
            "choose. Both belong to the continuous dials."
        )
    if not discrete:
        sections = _resolve_sections(style, sections, increment)
        if style == "area" and increment is not None:
            raise ValueError(
                "`increment=` is a band length in the units of the "
                "distribution, which only style='increments' has -- an "
                "equal-area dial is spaced evenly in probability instead, so "
                "it takes sections= (how many equally likely slices)."
            )

    if value is None:
        # Spin: a bearing uniform on [0, 360), then the contract. A `seed`
        # gets a generator of its own rather than reseeding the shared one, so
        # a reproducible figure does not disturb anyone else's simulation.
        generator = rng if seed is None else np.random.default_rng(seed)
        bearing = float(generator.uniform(0.0, 360.0))
        value = _as_scalar(dist, spin_value(dist, bearing))
    else:
        if seed is not None:
            raise ValueError(
                "`seed=` makes a random spin reproducible, but `value=` "
                "already says exactly where the needle goes, so there is "
                "nothing left to seed. Pass one or the other."
            )
        # A discrete cdf steps, so pointing at an outcome means the middle of
        # its wedge; a continuous one inverts straight through the contract.
        bearing = (
            discrete_needle_bearing(dist, value)
            if discrete
            else bearing_of(dist, value)
        )

    owns_figure = ax is None
    ax = _dial_axes(ax)
    ticks, boundaries, caption = (), (), None
    if discrete:
        slices = draw_discrete_dial(dist, ax)
    elif style == "area":
        slices, boundaries = draw_area_dial(dist, ax, sections)
        caption = area_caption(sections)
    else:
        slices, ticks, increment = draw_increments_dial(
            dist, ax, sections=sections, increment=increment
        )

    draw_needle(ax, bearing)
    _draw_caption(ax, caption, owns_figure)
    return SpinnerPlot(
        ax,
        bearing,
        value,
        slices,
        style=style,
        sections=sections,
        increment=increment,
        ticks=ticks,
        boundaries=boundaries,
        caption=caption,
    )


def _as_scalar(dist, value):
    """Wrap a spun value the way ``Distribution.draw()`` wraps one.

    A spin is a draw, so it should come back looking like one:
    ``Binomial(10, 0.5).draw()`` gives ``6``, not ``np.float64(6.0)``, and a
    spin of the same distribution should read the same. scipy's ``ppf``
    returns a float whatever the support, so a discrete outcome is put back on
    the whole numbers first.

    Parameters
    ----------
    dist : Distribution
        The distribution spun, read for its ``discrete`` flag.
    value : float
        The raw quantile.

    Returns
    -------
    Int or Float
        The value, wrapped.
    """
    number = float(value)
    if not np.isfinite(number):
        # A spin landing in an unbounded tail's very edge can come back
        # infinite; there is no Int or Float for that, so hand it back as is.
        return number
    if getattr(dist, "discrete", False) and number.is_integer():
        return Scalar(int(number))
    return Scalar(number)


def _reject_multivariate(dist):
    """Refuse a distribution that has no single quantile function to draw.

    Raises
    ------
    ValueError
        For a multivariate distribution.
    """
    if not hasattr(dist, "quantile") or getattr(dist, "_scipy", None) is None:
        raise ValueError(
            f"A {type(dist).__name__} has no spinner. It is a distribution "
            "over vectors or matrices, so there is no single value for a "
            "needle to land on. Spin one component at a time instead."
        )
