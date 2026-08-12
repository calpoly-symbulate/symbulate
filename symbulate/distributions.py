import math
import numbers
import warnings
import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq, minimize_scalar
from scipy.special import beta as beta_function
from scipy.special import gammaln, xlogy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .probability_space import ProbabilitySpace
from matplotlib.ticker import MaxNLocator

from .plot import (
    get_next_color,
    set_plot_title,
    THEORETICAL_CDF_PDF_OVERLAY_ERROR,
    DistributionPlot,
    JointDistributionPlot,
    make_joint_pdf,
    make_joint_pmf,
    ECDF_LINEWIDTH,
    JOINT_CBAR_DECIMALS,
    JOINT_PAIRS_MAX_DIM,
    PAIRS_MAX_DISCRETE_TICKS,
    setup_marginal_axes,
    pairs_colorbar_pair_label,
    pairs_joint_label,
    pairs_marginal_label,
    advance_pairs_diagonal_color,
    align_pairs_columns,
    thin_marginal_frequency_ticks,
    JOINT_PMF_MAX_CELLS,
    JOINT_PAIRS_OVERLAY_ERROR,
    JOINT_PAIRS_PANEL_SIZE,
    add_colorbar,
    add_pairs_panel_colorbar,
    SHADE_COLOR,
    SHADE_ALPHA,
    TRUE_DIST_MARKER_SIZE,
    TRUE_DIST_LINEWIDTH,
    TRUE_DIST_LINESTYLE,
    _refresh_legend,
)
from .result import Scalar, Vector, InfiniteVector

# The shared generator, owned by probability_space.py -- this module used to
# create its own, which made seed() unable to reach it. Do not reintroduce a
# local `rng = np.random.default_rng()` here.
from .probability_space import rng


LEGEND_PARAMETER_DECIMALS = 3


def _format_legend_parameter(value):
    """Return one distribution parameter formatted for a legend label.

    This is deliberately display-only: callers continue to retain and use
    the original parameter value for all distribution calculations.
    """
    if isinstance(value, numbers.Real):
        formatted = f"{float(value):.{LEGEND_PARAMETER_DECIMALS}f}"
        formatted = formatted.rstrip("0").rstrip(".")
        return "0" if formatted == "-0" else formatted
    return str(value)


def _validate(*checks):
    """Collect all failed parameter checks and report them together.

    Each argument is a ``(failed, message)`` pair, where ``failed`` is a
    boolean that is ``True`` when the parameter is invalid. Every message
    whose check failed is gathered and raised in a single ``Exception``,
    so a student who mistypes more than one parameter sees all of their
    mistakes at once instead of fixing them one error at a time.

    A single failure raises that message on its own (identical to the old
    per-parameter checks); two or more are listed under a short header.

    Parameters
    ----------
    *checks : tuple of (bool, str)
        One ``(failed, message)`` pair per parameter check.

    Raises
    ------
    Exception
        If any check failed. With one failure the message is raised as-is;
        with several, they are listed under an ``"Invalid parameters:"``
        header, one indented bullet per failed check.

    Notes
    -----
    Each ``failed`` expression should test the parameter's type before its
    value (e.g. ``not isinstance(p, numbers.Real) or p < 0``) so the
    short-circuit avoids a cryptic ``TypeError`` when the value is, say, a
    string. The expressions are evaluated by the caller before being passed
    in, so any cross-parameter comparison (``a > b``) must itself be guarded
    by an ``isinstance`` check.
    """
    errors = [message for failed, message in checks if failed]
    if not errors:
        return
    if len(errors) == 1:
        raise Exception(errors[0])
    raise Exception("Invalid parameters:\n" + "\n".join("  - " + e for e in errors))


def _is_whole_number(value):
    """Whether ``value`` is a whole number, however it is spelled.

    True for an integer and for a float that happens to be whole (``6.0``),
    false for a fractional number or a non-number. Used where a parameter
    counts something and so has to land on an integer, but may reasonably
    reach us as a float -- out of a division, or from a NumPy array.

    This is the same test :class:`~symbulate.index_sets.Naturals` uses for
    membership, kept consistent so ``6.0`` is accepted in both places.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        ``True`` if ``value`` is a real number with no fractional part.
    """
    if isinstance(value, numbers.Integral):
        return True
    return isinstance(value, numbers.Real) and float(value).is_integer()


# Tail probability trimmed from each unbounded side of a default plotting
# window: the window runs from ``quantile(_PLOT_TAIL)`` to
# ``quantile(1 - _PLOT_TAIL)`` wherever the support has no fixed bound. This is
# the 0.1% cut Symbulate has always used, kept unchanged so removing the
# highest-density window restores the original framing exactly.
_PLOT_TAIL = 0.001

# How much of a fixed-bound window the probability must fill for that bound to
# be kept. A fixed bound is honored while the quantile window covers at least
# this fraction of it -- so ``Binomial(10, 0.5)`` keeps all of ``(0, 10)`` (its
# probability fills 80% of that) while ``Binomial(1000, 0.5)`` zooms to
# ``(451, 549)`` (9.8%), instead of drawing a spike in the middle of an empty
# axis. Half is the crossover: at 0.5 a plot is showing as much empty axis as
# distribution, which is the point where the bounds stop being informative.
_ZOOM_FRACTION = 0.5


class Distribution(ProbabilitySpace):
    """Base class for all probability distributions in Symbulate.

    Provides common methods shared by all distributions, including
    drawing samples, computing summary statistics, and plotting.
    You typically won't use this class directly — use one of the
    specific distribution classes instead (e.g., ``Normal``, ``Binomial``).

    Parameters
    ----------
    params : dict
        Named parameter values for the underlying scipy distribution.
    scipy : scipy.stats distribution
        The scipy distribution object used for computation.
    discrete : bool, optional
        ``True`` for discrete distributions, ``False`` for continuous.
        Default is ``True``.

    Attributes
    ----------
    params : dict
        Named parameter values passed to the underlying scipy distribution.
    discrete : bool
        ``True`` for discrete distributions, ``False`` for continuous.
    pdf : callable
        Probability density (or mass) function.
    cdf : callable
        Cumulative distribution function.
    quantile : callable
        Inverse CDF (percent-point function).
    mean : callable
        Returns the expected value of the distribution.
    var : callable
        Returns the variance of the distribution.
    sd : callable
        Returns the standard deviation of the distribution.
    median : callable
        Returns the median of the distribution.
    xlim : tuple of float
        Default x-axis range used when plotting, derived from the true
        support -- see the note below. Computed on first read and cached.

    Notes
    -----
    The default window is one rule applied to each end of the support
    independently: **a fixed bound is used as-is; an unbounded side is cut at
    a quantile** (``_PLOT_TAIL`` / ``1 - _PLOT_TAIL``). So:

    - bounded both ends (``Binomial``, ``Uniform``) -> the full support, e.g.
      ``Binomial(10, 0.5)`` gives ``(0, 10)`` and ``Binomial(1000, 0.5)``
      gives ``(0, 1000)``;
    - fixed lower bound only (``Poisson``, ``Exponential``) -> ``(lower,
      quantile(0.999))``, and symmetrically for a fixed upper bound only;
    - unbounded both ends (``Normal``) -> ``(quantile(0.001),
      quantile(0.999))``.

    **A fixed bound is never given up automatically.** It is exact, so it
    stays, however little of the window the probability fills --
    ``Binomial(1000, 0.5)`` shows all of ``(0, 1000)`` even though everything
    happens between about 451 and 549. The tighter framing is available on
    request: ``plot(xlim="zoom")`` for one plot, or assign
    ``X.xlim = (low, high)`` to change the distribution's own window, which is
    then used exactly as given.

    (``_fills_window`` and ``_ZOOM_FRACTION`` implement the "does the
    probability fill this window" test. They are **not** applied here; they
    frame the panels of a multivariate plot, which has no window argument of
    its own -- see :meth:`MultivariateDistribution._plot_window`.)

    The bounds come from ``scipy``'s own ``support()``, so a subclass gets the
    right window with no per-distribution code. A subclass that must preset a
    window instead (``Zeta``, whose quantile search would never return; the
    degenerate ``LogNormal`` branch, which never calls this ``__init__`` and so
    has no ``scipy`` object to ask) assigns ``self._xlim`` directly rather than
    going through the setter -- that window is still one the distribution chose
    for itself, so it keeps the discrete padding :meth:`plot` describes.

    ``xlim`` is a lazy property, computed on the first read rather than in
    ``__init__``. It is only ever needed for plotting, and ``__init__`` runs
    far more often: ``PoissonProcess`` and ``ContinuousTimeMarkovChain`` build
    a fresh distribution on *every draw*, so eager computation charged every
    simulated value for a window nobody asked to see.
    """

    # Defaults so the lazy `xlim` property below works even for a subclass
    # that skips `Distribution.__init__` (e.g. degenerate `LogNormal`, and the
    # multivariate distributions, which have no window at all).
    _xlim = None  # the window, once computed or set outright
    _scipy = None  # the scipy distribution the support is read from
    # Whether a discrete plot gets one whole-number slot of air at each end
    # (see `plot`). A window the distribution worked out for itself does; one
    # assigned through the `xlim` setter is used exactly as given, since
    # someone chose those numbers. A subclass that presets its own default
    # assigns `_xlim` directly, which leaves this True.
    _xlim_padded = True

    def __init__(self, params, scipy, discrete=True):
        """Initialize the base Distribution."""
        self.params = params

        self.discrete = discrete

        if discrete:
            self.pmf = lambda x: scipy.pmf(x, **self.params)
            self.pdf = self.pmf  # add pdf as an alias for pmf
        else:
            self.pdf = lambda x: scipy.pdf(x, **self.params)

        self.cdf = lambda x: scipy.cdf(x, **self.params)
        self.quantile = lambda x: scipy.ppf(x, **self.params)

        self.median = lambda: scipy.median(**self.params)
        self.mean = lambda: scipy.mean(**self.params)
        self.var = lambda: scipy.var(**self.params)
        self.sd = lambda: scipy.std(**self.params)
        self.sim_func = scipy.rvs
        self._scipy = scipy

        # The window is deliberately left alone here. `_xlim` already
        # defaults to None on the class, so `_compute_xlim` derives the window
        # from the support on the first read of `.xlim` -- and a subclass that
        # set its own `xlim` *before* calling up to here (as `Zeta` does, to
        # avoid a quantile lookup that never finishes) keeps that window
        # instead of having it reset. Either ordering works.

    @property
    def xlim(self):
        """tuple of float : the default x-axis range used when plotting.

        Computed the first time it is read and cached from then on, so a
        distribution that is built but never plotted -- one per draw inside a
        random process, say -- never pays for the window. See the class
        ``Notes``.
        """
        if self._xlim is None:
            self._xlim = self._compute_xlim()
        return self._xlim

    @xlim.setter
    def xlim(self, value):
        """Set an exact plotting window, replacing whatever was deferred.

        This is how a window is chosen by hand, since :meth:`plot` takes no
        window argument. It is used *exactly* as given: a discrete plot skips
        the whole-number slot of padding a self-chosen window gets, because
        these are someone's own numbers.
        """
        self._xlim = value
        self._xlim_padded = False

    def _fast_sim(self, n):
        """Optional fast path for ``.sim(n)``: draw all ``n`` samples at once.

        Returns ``None`` -- the default, and the answer for every
        distribution unless it overrides this. A ``None`` here means
        ``.sim(n)`` runs its ordinary one-draw-at-a-time loop, exactly as
        it always has.

        Override it only when the underlying scipy sampler carries a large
        per-call setup cost that does *not* grow with the number of samples
        requested, so that paying it once per ``.sim(n)`` instead of ``n``
        times is a real saving. As of this change that is
        :class:`NegativeHypergeometric` and :class:`TruncatedNormal`; see
        ``team/final-tasks/audit/fast_sim_notes.md`` for how those two were
        identified and why the list stops there. Adding another is a
        one-line override here, with no change at either call site.

        Deliberately takes no ``random_state`` argument. An override draws
        from this module's own ``rng`` -- the same generator ``draw()``
        uses -- rather than accepting one from whoever called ``.sim()``,
        which lives in a different module. Passing a generator across that
        boundary would make it possible to advance a different stream than
        ``draw()`` does, and the two would silently disagree.

        Parameters
        ----------
        n : int
            How many samples ``.sim()`` was asked for.

        Returns
        -------
        array-like or None
            ``n`` samples drawn in one batched call, or ``None`` to mean
            "no batched path -- use the ordinary loop."
        """
        return None

    def _support(self):
        """The distribution's true support bounds, either of which may be infinite.

        Returns
        -------
        tuple of float
            ``(low, high)`` as reported by the underlying scipy
            distribution's ``support()``. An unbounded side is ``-inf`` or
            ``inf``.
        """
        low, high = self._scipy.support(**self.params)
        return float(low), float(high)

    def _compute_xlim(self):
        """Derive the default plotting window from the true support.

        Each end is handled independently: **a fixed bound is used as-is, and
        an unbounded side is cut at a quantile.** That is the whole rule --
        the window is not narrowed further, so ``Binomial(1000, 0.5)`` frames
        all of ``(0, 1000)``. Pass ``xlim="zoom"`` to :meth:`plot` for the
        tighter window that holds most of the probability.

        Returns
        -------
        tuple of float
            The ``(low, high)`` default window.

        Raises
        ------
        AttributeError
            For a multivariate distribution, which has no single x-axis to
            frame and so no window to compute.
        """
        # The multivariate distributions (MultivariateNormal, Dirichlet,
        # Wishart, ...) set up only a pdf -- never a quantile function -- and
        # their `plot` raises on its own. Raising AttributeError here keeps
        # `.xlim` behaving exactly as it did when it was a plain attribute
        # those classes simply never set: `hasattr(d, "xlim")` stays False.
        if not hasattr(self, "quantile") or self._scipy is None:
            raise AttributeError(
                f"A {type(self).__name__} has no 'xlim'. It is a distribution "
                "over vectors or matrices, so there is no single x-axis to "
                "frame. (Plotting is not available for it either.)"
            )
        low, high = self._support()
        bounded_low, bounded_high = np.isfinite(low), np.isfinite(high)
        # The quantile window fills in whichever side has no fixed bound. A
        # fixed bound is kept exactly as it is: this window is deliberately
        # *not* narrowed when the probability fills little of it, because
        # narrowing it automatically would take `Binomial(1000, 0.5)` away
        # from its true (0, 1000) range without anyone asking. `xlim="zoom"`
        # on `plot` is how that tighter framing is requested.
        if not bounded_low or not bounded_high:
            zoom_low, zoom_high = self._zoom_xlim()
            if not bounded_low:
                low = zoom_low
            if not bounded_high:
                high = zoom_high
        # For a degenerate parameter value scipy can return a quantile outside
        # its own support -- `geom.ppf(0.999, p=1)` is 0.0 though the support
        # starts at 1 -- which would invert the window. The support bound is
        # the exact one, so collapse onto it rather than trust the quantile.
        if high < low:
            return (low, low) if bounded_low else (high, high)
        return (low, high)

    def _fills_window(self, low, high, zoom_low, zoom_high):
        """Whether the probability fills enough of ``(low, high)`` to keep it.

        A fixed bound earns its place only while the region the probability
        actually occupies is a decent share of the window that bound gives.
        Below ``_ZOOM_FRACTION`` of it, the plot is mostly empty axis and the
        tighter quantile window reads better.

        **This is not applied to a univariate plot's default window.** A
        one-dimensional plot keeps its fixed bounds and zooms only when asked
        (``plot(xlim="zoom")``). The test is used by
        :meth:`MultivariateDistribution._plot_window`, where a panel of a
        joint plot has far less room and an unreadable panel can't be fixed
        by passing a window -- there is no per-panel ``xlim`` argument.

        Parameters
        ----------
        low, high : float
            The window derived from the support, either end of which may be a
            quantile already if the support is unbounded there.
        zoom_low, zoom_high : float
            The quantile window, i.e. ``_zoom_xlim()``.

        Returns
        -------
        bool
            ``True`` to keep ``(low, high)``, ``False`` to zoom.
        """
        width = high - low
        zoom_width = zoom_high - zoom_low
        # A window with no width at all (one outcome carries all the
        # probability) has nothing to zoom into, and neither does a quantile
        # window that is somehow the wider of the two. Keep what the support
        # gave, which is the exact bound.
        if not np.isfinite(width) or width <= 0 or zoom_width >= width:
            return True
        return zoom_width >= _ZOOM_FRACTION * width

    def _zoom_xlim(self):
        """The zoomed window: both ends cut at a quantile.

        Frames the region where the probability actually lives by ignoring
        fixed bounds -- treating the distribution as if it had none. For an
        already-unbounded distribution this is the whole story; for a bounded
        one it zooms in, so ``Binomial(1000, 0.5)`` frames roughly
        ``(451, 549)`` instead of its full ``(0, 1000)`` support. It is used
        automatically by :meth:`_compute_xlim` whenever the probability fills
        too little of the support-derived window, so a student never has to
        ask for it or work out the endpoints.

        Returns
        -------
        tuple of float
            The ``(low, high)`` quantile window.
        """
        return (float(self.quantile(_PLOT_TAIL)), float(self.quantile(1 - _PLOT_TAIL)))

    def draw(self):
        """Draw a single random sample from the distribution.

        Returns
        -------
        Scalar
            One random value drawn from the distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> Normal(0, 1).draw()  # doctest: +SKIP
        0.4967141530112327
        """
        return Scalar(self.sim_func(**self.params, random_state=rng))

    def spinner(
        self,
        value=None,
        seed=None,
        ax=None,
        style=None,
        sections=None,
        increment=None,
        **kwargs,
    ):
        """Spin a probability spinner for this distribution, and draw it.

        A spinner is a picture of this distribution's quantile function. The
        dial is fixed and the needle turns, so a quarter turn always points
        at the 25th percentile, half a turn at the median, and three quarters
        at the 75th. **One spin is one draw** from the distribution:

        .. code-block:: text

            bearing ~ Uniform(0, 360)     measured clockwise from 12 o'clock
            value   = quantile(bearing / 360)

        Calling it spins: the needle lands somewhere at random, and the value
        it landed on comes back on the returned plot as ``.value``. Pass
        ``value=`` instead to point the needle at a value you choose, which is
        what to use for a figure that has to look the same every time.

        Which dial is drawn depends on the distribution. A **discrete** one
        gets a slice per outcome, sized by that outcome's probability -- there
        is only the one dial, so it takes no ``style``. A **continuous** one
        can be read two ways:

        - ``style="increments"`` (the default) rules the rim in round steps of
          equal *value*, so each slice's size is the chance of landing in that
          range -- wide where the density is high, pinched in the tails.
        - ``style="area"`` gives every slice the same *area*, so each one is
          exactly a 1-in-``sections`` chance and the values round the rim bunch
          up where the distribution is dense.

        Parameters
        ----------
        value : optional
            Rest the needle on this value instead of spinning for one. For a
            discrete distribution the needle sits in the *middle* of that
            outcome's slice, since a discrete cdf steps and would otherwise
            land on the boundary between two slices.
        seed : int, optional
            Makes a spin reproducible. This uses a generator of its own, so a
            seeded spin does not disturb the shared stream
            :func:`~symbulate.seed` controls. Cannot be combined with
            ``value``, which leaves nothing random to seed.
        ax : matplotlib.axes.Axes, optional
            Draw the dial here instead of building a figure for it.
        style : {'increments', 'area'}, optional
            Which continuous dial to draw. Refused for a discrete
            distribution, which has only one.
        sections : int, optional
            How many bands (``'increments'``, 3 to 20, default 8) or equally
            likely slices (``'area'``, 4 to 60, default 12).
        increment : float, optional
            The band length itself, in the units of the distribution, instead
            of a count. ``'increments'`` only, and not with ``sections``.

        Returns
        -------
        SpinnerPlot
            The dial. Its ``.value`` is what the needle landed on; it also
            carries the ``.bearing``, the ``.slices`` drawn, and the
            ``.caption`` if the dial has one.

        Examples
        --------
        >>> from symbulate import *
        >>> Binomial(10, 0.5).spinner()  # doctest: +SKIP
        >>> Binomial(10, 0.5).spinner().value  # doctest: +SKIP
        6
        >>> Binomial(10, 0.5).spinner(value=7)  # doctest: +SKIP
        >>> Normal(0, 1).spinner(style="area", sections=12)  # doctest: +SKIP
        >>> Exponential(1).spinner(increment=0.5)  # doctest: +SKIP
        """
        # `mode=` named the earlier spinner's two layouts. The equal-sections
        # layout is gone (a discrete dial has one slice per outcome, always),
        # and the choice that remains is between the two *continuous* dials
        # -- so the keyword is renamed rather than kept as an alias, the same
        # way `dims=`/`pairs=`/`equal_width=` name their replacements instead
        # of reaching matplotlib as an opaque error.
        if "mode" in kwargs:
            raise TypeError(
                "`spinner()` no longer takes a `mode=` argument. A discrete "
                "distribution now has a single dial -- one slice per outcome, "
                "sized by its probability -- so there is nothing to choose. "
                "For a continuous distribution, the two layouts are "
                "style='increments' and style='area'."
            )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"`spinner()` got an unexpected argument: {unexpected}. It "
                "takes value=, seed=, ax=, style=, sections=, and increment=."
            )

        from .spinner import make_spinner

        return make_spinner(
            self,
            value=value,
            seed=seed,
            ax=ax,
            style=style,
            sections=sections,
            increment=increment,
        )

    # Override the inherited __pow__ function to take advantage
    # of vectorized simulations.
    def __pow__(self, exponent):
        """Draw multiple independent samples from the distribution.

        Parameters
        ----------
        exponent : int or float
            Number of samples to draw. Pass ``float('inf')`` to create
            an infinite sequence of draws generated lazily on demand.

        Returns
        -------
        ProbabilitySpace
            A probability space whose draws produce ``exponent`` samples
            at a time.

        Examples
        --------
        >>> from symbulate import *
        >>> (Normal(0, 1) ** 3).draw()  # doctest: +SKIP
        [-0.234, 1.724, 0.313]
        """
        if exponent == float("inf"):

            def draw():
                def _func(_):
                    return self.sim_func(**self.params, random_state=rng)

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(
                    self.sim_func(**self.params, size=exponent, random_state=rng)
                )

        return ProbabilitySpace(draw)

    def plot(self, xlim=None, alpha=None, ax=None, cdf=False, **kwargs):
        """Plot the probability function or the cumulative distribution function.

        By default (``cdf=False``), plots the probability density function
        (continuous distributions, a smooth curve) or probability mass
        function (discrete distributions, a smooth curve through the
        masses). With ``cdf=True``, plots the cumulative distribution
        function ``P(X <= x)`` instead: a smooth curve for continuous
        distributions, and a right-continuous step function (no markers)
        for discrete ones.

        Unlike a plot of simulated *data* -- which can be drawn many ways
        (dots, rug, impulse, histogram, density, ecdf, ...) and so takes a
        ``type=`` argument -- a theoretical distribution has only these two
        curves to show, so the choice is the single boolean ``cdf``.

        The plot is titled by what it shows: "Cumulative Distribution
        Function" for ``cdf=True``, and for the default view "Probability
        Density Function" (continuous) or "Probability Mass Function"
        (discrete) -- only when the axes doesn't already have a title, so
        overlaying onto an existing plot keeps that plot's title. Overlaying
        a pdf/pmf and a cdf on the same axes raises instead of drawing an
        unreadable plot -- see ``ax=`` below for comparing both side by side.

        **The x-axis range.** Left alone, the distribution's own
        :attr:`xlim` is used: a fixed bound is shown in full and an unbounded
        side is cut at a quantile. So ``Binomial(1000, 0.5)`` shows all of
        ``(0, 1000)``, ``Poisson(3)`` and ``Exponential`` start at their true
        lower bound 0, and ``Normal`` is cut at both ends. Pass
        ``xlim="zoom"`` for the tighter window holding most of the
        probability (``Binomial(1000, 0.5)`` becomes roughly
        ``(451, 549)``), or ``xlim=(low, high)`` for an exact range.

        For a discrete distribution, a window the distribution chose for
        itself gets one whole-number slot of padding on each end, so the
        boundary value isn't drawn right on the axis spine. A window given by
        hand -- through ``xlim=`` here or by assigning ``X.xlim = (low,
        high)`` -- is used exactly as given, with no padding added.

        Parameters
        ----------
        xlim : tuple of float or str, optional
            The x-axis range. Leave it out for the distribution's own default
            window; pass ``"zoom"`` to frame the region holding most of the
            probability; or pass ``(low, high)`` for exactly that range.
        cdf : bool, default False
            Which function to plot. ``False`` (the default) draws the
            probability density/mass function; ``True`` draws the
            cumulative distribution function ``P(X <= x)``. (For discrete
            distributions the default draws the probability mass function.)
        alpha : float, optional
            Transparency of the plot, from 0 (invisible) to 1 (opaque).
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. Creates or uses the current axes if
            not provided.
        **kwargs
            Additional keyword arguments forwarded to matplotlib.

        Returns
        -------
        DistributionPlot
            A wrapper around the matplotlib axes the plot was drawn on.
            Its printed representation is empty, so Jupyter shows only the
            plot. Chain ``.shade(...)`` onto it to fill a region under the
            curve.

        Examples
        --------
        >>> from symbulate import *
        >>> Normal(0, 1).plot()  # doctest: +SKIP
        >>> Binomial(1000, 0.5).plot()  # the full (0, 1000)  # doctest: +SKIP
        >>> Binomial(1000, 0.5).plot("zoom")  # about (451, 549)  # doctest: +SKIP
        >>> Binomial(1000, 0.5).plot(xlim=(400, 600))  # exact  # doctest: +SKIP
        >>> Poisson(3).plot(cdf=True)  # step function  # doctest: +SKIP
        >>> Normal(0, 1).plot(cdf=True)  # smooth S-curve  # doctest: +SKIP
        """
        # A theoretical distribution has only two curves (pdf/pmf vs. cdf),
        # so it takes a boolean `cdf=`, not the `type=` that selects among the
        # many ways of drawing simulated data. Catch the old `type=` spelling
        # (and stray text) so it gives a clear pointer instead of an opaque
        # matplotlib error after slipping through **kwargs.
        if "type" in kwargs:
            raise ValueError(
                "`Distribution.plot()` does not take a `type=` argument. To "
                "plot the cumulative distribution function, use `cdf=True`; "
                "the default (`cdf=False`) plots the pdf/pmf. (`type=` selects "
                "among the many ways of drawing simulated data; a theoretical "
                "distribution has only these two curves to show.)"
            )
        # Resolve the x-axis range:
        #   None        -> the distribution's default window (a fixed bound
        #                  used as-is, a quantile cut where unbounded);
        #   "zoom"      -> a quantile cut at *both* ends, i.e. the same
        #                  framing an unbounded distribution already gets,
        #                  which zooms in on a bounded default;
        #   (low, high) -> those exact limits, used as given.
        # A window given here is someone's own choice, so it skips the
        # discrete half-step padding below -- the same rule the `xlim` setter
        # follows (see `_xlim_padded`).
        padded = self._xlim_padded
        if xlim is None:
            xlim = self.xlim
        elif isinstance(xlim, str):
            if xlim != "zoom":
                raise ValueError(
                    'The only text value `xlim` accepts is "zoom" (frame the '
                    "region holding most of the probability). You passed "
                    f"xlim={xlim!r}. Otherwise pass xlim=(low, high) for an "
                    "exact range, or leave it out for the default range."
                )
            xlim = self._zoom_xlim()
            padded = False
        else:
            padded = False

        # The window itself can come back non-finite, which is a different
        # failure from the all-non-finite curve guarded below: here there is
        # no range of values to evaluate at all. Two distinct causes, so two
        # distinct messages -- an infinite bound means the 0.999 quantile
        # never came back (Pareto(shape=1e-3) frames `(1.0, inf)`, which
        # reaches matplotlib as a bare "Axis limits cannot be NaN or Inf"),
        # while a nan bound means the quantiles themselves are undefined
        # (Normal(sd=0), Uniform(a=5, b=5)). Checking here also keeps a nan
        # bound away from the `int(xlim[0])` on the discrete branch below,
        # which would raise its own uninformative conversion error.
        if not all(np.isfinite(bound) for bound in xlim):
            if any(np.isnan(bound) for bound in xlim):
                cause = (
                    "This usually means a parameter is at a degenerate edge "
                    "-- zero variance, or bounds that collapse to a single "
                    "point -- which leaves the distribution with no range of "
                    "values to draw."
                )
            else:
                cause = (
                    "This usually means a tail so heavy that no window holds "
                    "most of the probability -- a shape parameter very close "
                    "to 0, say."
                )
            raise ValueError(
                f"{type(self).__name__}'s parameters give the plotting window "
                f"{tuple(xlim)}, which is not a finite range of values. "
                f"{cause} This is rarely a bug in your code -- check the "
                f"distribution's parameters, or choose the window yourself "
                f"with plot(xlim=(low, high))."
            )

        # get the x and y values. The x-window is chosen the same way for
        # both plot types (it only picks x-values); `cdf` decides which
        # function is evaluated there.
        if self.discrete:
            # xs is drawn from the *true*, unpadded window -- the view
            # padding added near the end of this method (see
            # `_symbulate_true_xlim` below) only ever widens the axes, and
            # must never change which values are actually evaluated/drawn.
            xs = np.arange(int(xlim[0]), int(xlim[1]) + 1)
        else:
            xs = np.linspace(xlim[0], xlim[1], 200)
        ys = self.cdf(xs) if cdf else self.pdf(xs)

        # A degenerate parameter (zero variance, a shape parameter at an
        # extreme edge, bounds that collapse to a point, ...) can make every
        # evaluated value non-finite, which would otherwise reach a bare
        # NumPy "zero-size array" crash below with no context for a student.
        # Name the likely cause and point at the fix instead.
        finite = np.isfinite(ys)
        if not np.any(finite):
            quantity = "cdf" if cdf else "pmf" if self.discrete else "pdf"
            raise ValueError(
                f"{type(self).__name__}'s parameters make the {quantity} "
                f"undefined or infinite everywhere in the plotting window "
                f"{tuple(xlim)}. This usually means a parameter is at a "
                f"degenerate edge -- zero variance, a shape parameter at 0, "
                f"or bounds that collapse to a single point -- rather than a "
                f"bug in your code. Check the distribution's parameters."
            )

        # determine limits for y-axes based on y values. Anchor the baseline
        # at exactly 0 so the curve sits right on the x-axis: a pdf/pmf height
        # is never negative and only reads correctly against a zero baseline,
        # and a CDF likewise runs from 0 upward. Padding below 0 would float
        # the curve off the axis and misrepresent it.
        ymax = ys[finite].max()
        ylim = 0, 1.05 * ymax

        # get the current axis (creating one if the figure has none) unless
        # an axis was specified
        if ax is None:
            ax = plt.gca()

        # A pdf/pmf and a cdf use different y-axis scales (one can exceed 1,
        # the other runs from 0 to 1), so overlaying both on one axes -- e.g.
        # Normal(0,1).plot(); Normal(0,1).plot(cdf=True) -- produces a plot
        # that can't be read correctly no matter which curve's title/label
        # wins. Tag the axes with which kind was drawn (the same pattern
        # RVResults.plot() uses for _symbulate_marginal) and refuse a second,
        # mismatched call rather than silently drawing an unreadable overlay.
        # Different Axes objects (e.g. ax1, ax2 = ax1.twinx()) are unaffected,
        # since each has its own tag -- that's the escape hatch for anyone who
        # deliberately wants both curves side by side.
        new_kind = "cdf" if cdf else "pdf/pmf"
        existing_kind = getattr(ax, "_symbulate_theoretical_kind", None)
        if existing_kind is not None and existing_kind != new_kind and ax.has_data():
            raise ValueError(
                THEORETICAL_CDF_PDF_OVERLAY_ERROR.format(
                    existing=existing_kind, new=new_kind
                )
            )

        # If the axes already has a plot on it, widen the window to the union
        # of both, so overlaying a second curve doesn't crop the first. Uses
        # the *true* (unpadded) window from any prior symbulate distribution
        # plot on this axes -- stored below as `_symbulate_true_xlim` --
        # rather than the axes' current view limits: for a discrete
        # distribution those view limits already include the one-slot
        # cosmetic padding added below, and reading them back here would add
        # another slot on every subsequent overlay, growing without bound.
        # Falls back to the axes' actual limits when nothing has tagged them
        # yet (e.g. overlaying onto a simulated plot's axes). An axes with
        # nothing drawn on it yet has placeholder limits of (0, 1), which are
        # not a plot to make room for -- unioning with those would stretch
        # the window and squash the curve into part of the panel (as happens
        # when a caller passes in a fresh, empty axes, e.g. one panel of a
        # pairs matrix).
        if ax.has_data():
            xlower, xupper = getattr(ax, "_symbulate_true_xlim", ax.get_xlim())
            xlim = min(xlim[0], xlower), max(xlim[1], xupper)
            ylower, yupper = ax.get_ylim()
            ylim = min(ylim[0], ylower), max(ylim[1], yupper)

        # set the axis limits
        if xlim[0] == xlim[1]:
            # A window can collapse onto a single value when one outcome
            # carries essentially all the probability (e.g. Geometric(0.99),
            # or a bounded distribution that zoomed onto one value, or an
            # explicit xlim of (a, a)). Give the lone point room so the axis
            # stays well-formed instead of singular -- a full unit for a
            # discrete distribution (there's no second value here to measure
            # a "slot" from, so this matches the one-slot convention below),
            # half a unit otherwise.
            pad = 1.0 if self.discrete else 0.5
            xlim = (xlim[0] - pad, xlim[1] + pad)
        # Recorded before the discrete padding below is added, so a later
        # overlay's union (above) compares against the real data window
        # rather than this cosmetic padding.
        ax._symbulate_true_xlim = xlim
        if self.discrete and padded:
            # A discrete pmf/cdf is drawn at whole-number x-values one unit
            # apart (see `xs` above), so without padding the outermost dot or
            # step sits exactly on the left/right spine -- flush against it
            # with no visual room to tell it isn't cut off. Add one slot of
            # air beyond each end, the same "one slot" convention
            # `configure_axes` uses for impulse/dot plots of simulated data
            # (see plot.py), so a discrete theoretical plot frames the same
            # way a discrete simulated one does. This only widens the
            # *view*: xs (and so the drawn values) are unaffected. Skipped
            # whenever the window came from someone rather than from the
            # distribution -- `plot(xlim=...)`, `plot("zoom")`, or an assigned
            # `X.xlim = (low, high)` (which sets `_xlim_padded = False`) --
            # since those are their numbers, used exactly as given.
            ax.set_xlim(xlim[0] - 1, xlim[1] + 1)
        else:
            ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # get next color in cycle
        color = get_next_color(ax)

        # Default label names the distribution itself, e.g. "Binomial(10,
        # 0.5)", so a legend can tell two theoretical curves apart, or tell
        # a theoretical curve apart from a simulated one it's overlaid on.
        # setdefault, not an override, so a user's own label= still wins.
        # This only reaches the curve/scatter draw calls below (the ones
        # that forward **kwargs) -- the discrete pmf's dashed connecting
        # line draws separately, without **kwargs, so it never gets a
        # second, duplicate legend entry for the same curve.
        kwargs.setdefault(
            "label",
            f"{type(self).__name__}("
            f"{', '.join(_format_legend_parameter(v) for v in self.params.values())}"
            f")",
        )

        if cdf:
            # Match make_ecdf's step-function styling so a theoretical CDF
            # reads as the same kind of curve as its empirical counterpart
            # (type="ecdf"), which students naturally overlay to compare.
            # setdefault, not an override, so a user's own linewidth wins.
            kwargs.setdefault("linewidth", ECDF_LINEWIDTH)
            if self.discrete:
                # A discrete CDF is a right-continuous step function: flat
                # between consecutive integers, jumping at each value. No
                # markers -- the steps show where the mass lands, exactly
                # like make_ecdf's where="post".
                ax.plot(
                    xs, ys, drawstyle="steps-post", color=color, alpha=alpha, **kwargs
                )
            else:
                # A continuous CDF is a smooth S-curve.
                ax.plot(xs, ys, color=color, alpha=alpha, **kwargs)
        else:
            # pdf/pmf: a smooth curve for continuous distributions. For a
            # discrete distribution, a filled dot at each pmf value plus a
            # dashed dot-to-dot connecting line -- dashed so the connector
            # can't be mistaken for a continuous curve (a pmf has no value
            # between integers). Both are styled from the named TRUE_DIST_*
            # constants instead of a hardcoded marker size / default line.
            if self.discrete:
                ax.scatter(
                    xs, ys, s=TRUE_DIST_MARKER_SIZE, color=color, alpha=alpha, **kwargs
                )
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    alpha=alpha,
                    linestyle=TRUE_DIST_LINESTYLE,
                    linewidth=TRUE_DIST_LINEWIDTH,
                )
            else:
                ax.plot(xs, ys, color=color, alpha=alpha, **kwargs)

        # Title the plot by what it shows: the cumulative distribution
        # function, or -- for the default view -- the probability density
        # function (continuous) or probability mass function (discrete).
        # set_plot_title, not ax.set_title: only set it when the axes
        # doesn't already have a title, matching the same "keep the first
        # plot's framing" rule the labels below already followed -- title
        # and label used to disagree (title always overwritten, label only
        # filled if unset), which is exactly how a pdf-then-cdf overlay
        # used to end up titled "Cumulative Distribution Function" while
        # still y-labeled "Density". The overlay guard above blocks that
        # specific combination outright now, but this keeps title and label
        # in agreement for every other overlay too (e.g. a pdf curve added
        # on top of a histogram keeps that histogram's title). The helper
        # is the one shared definition of that rule -- every make_* title
        # in plot.py goes through it too.
        if cdf:
            set_plot_title(ax, "Cumulative Distribution Function")
        elif self.discrete:
            set_plot_title(ax, "Probability Mass Function")
        else:
            set_plot_title(ax, "Probability Density Function")

        # Label the axes for context: the x-axis shows the possible values of
        # the variable, and the y-axis names what its height means for this
        # plot type. "Value" and "Density" match the value plots in plot.py,
        # so a theoretical curve reads the same way as its simulated companion.
        # Only fill labels that aren't already set, so overlaying a curve onto
        # an existing simulated plot keeps that plot's labels (e.g. a
        # count-scale histogram's "Count") instead of clobbering them.
        if not ax.get_xlabel():
            ax.set_xlabel("Value")
        if not ax.get_ylabel():
            if cdf:
                ax.set_ylabel("Cumulative Probability")
            elif self.discrete:
                ax.set_ylabel("Probability")
            else:
                ax.set_ylabel("Density")

        # symbulate.mplstyle's global grid is horizontal-only (axes.grid.axis:
        # y), but these plots read better with both horizontal and vertical
        # reference lines, matching the ECDF plot -- so override it here for
        # this plot type specifically.
        ax.grid(True, axis="both")

        # A legend only helps once there is more than one labeled curve on
        # the axes -- another theoretical curve, or a simulated plot (hist,
        # density, rug, ecdf, ...) it's overlaid on. A lone curve stays
        # legend-free. "upper left" matches make_ecdf's own default, since a
        # rising CDF has more room there than "upper right".
        _refresh_legend(ax, loc="upper left" if cdf else "upper right")

        # Record what this call drew, for the overlay guard above to check
        # on the next call.
        ax._symbulate_theoretical_kind = new_kind

        return DistributionPlot(ax, self, "cdf" if cdf else "pdf")


## Discrete Distributions


class Bernoulli(Distribution):
    """Probability space for a Bernoulli distribution.

    Models a single trial with two outcomes: success (1) with probability
    ``p``, or failure (0) with probability ``1 - p``.

    Parameters
    ----------
    p : float
        Probability of success (1), between 0 and 1.

    Attributes
    ----------
    p : float
        Probability of success (1), between 0 and 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Bernoulli(p=0.5)
    >>> float(X.mean())
    0.5
    >>> float(X.sd())
    0.5
    >>> float(X.pmf(1))
    0.5
    >>> X.draw()  # doctest: +SKIP
    1
    """

    def __init__(self, p):
        """Initialize a Bernoulli distribution.

        Raises
        ------
        Exception
            If ``p`` is not a number between 0 and 1.
        """
        _validate(
            (
                not isinstance(p, numbers.Real) or not 0 <= p <= 1,
                "p must be a number between 0 and 1",
            ),
        )
        self.p = p

        params = {"p": p}
        super().__init__(params, stats.bernoulli, True)


class Binomial(Distribution):
    """Probability space for a binomial distribution.

    Models the number of successes in ``n`` independent trials, each
    with probability ``p`` of success.

    Parameters
    ----------
    n : int
        Number of trials. Must be a non-negative integer.
    p : float
        Probability of success on each trial, between 0 and 1.

    Attributes
    ----------
    n : int
        Number of trials.
    p : float
        Probability of success on each trial, between 0 and 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Binomial(n=4, p=0.5)
    >>> float(X.mean())
    2.0
    >>> float(X.sd())
    1.0
    >>> round(float(X.pmf(2)), 4)
    0.375
    >>> X.draw()  # doctest: +SKIP
    2
    """

    def __init__(self, n, p):
        """Initialize a binomial distribution.

        Raises
        ------
        Exception
            If ``n`` is not a non-negative integer, or ``p`` is not a
            number between 0 and 1.
        """

        _validate(
            (
                not isinstance(n, numbers.Integral) or n < 0,
                "n must be a non-negative integer",
            ),
            (
                not isinstance(p, numbers.Real) or not 0 <= p <= 1,
                "p must be a number between 0 and 1",
            ),
        )
        self.n = n
        self.p = p

        params = {"n": n, "p": p}
        super().__init__(params, stats.binom, True)


class BetaBinomial(Distribution):
    """Probability space for a beta-binomial distribution.

    Models the number of successes in ``n`` independent trials when the
    success probability is not fixed but is itself random, drawn once from
    a ``Beta`` distribution (with shape parameters ``shape1`` and
    ``shape2``) and then shared by all ``n`` trials. This extra,
    trial-to-trial uncertainty in the probability makes the counts more
    spread out (overdispersed) than a plain ``Binomial`` -- which is why the
    beta-binomial is the standard model for overdispersed binomial/count
    data. Because the beta is the conjugate prior for the binomial, it is
    also the beta-binomial that describes the number of successes before the
    probability is observed.

    Parameters
    ----------
    n : int
        Number of trials. Must be a non-negative integer.
    shape1 : float
        First shape parameter (often written α) of the underlying beta
        distribution. Must be positive.
    shape2 : float
        Second shape parameter (often written β) of the underlying beta
        distribution. Must be positive.

    Attributes
    ----------
    n : int
        Number of trials.
    shape1 : float
        First shape parameter (α) of the underlying beta distribution.
    shape2 : float
        Second shape parameter (β) of the underlying beta distribution.

    Notes
    -----
    The mean is ``n * shape1 / (shape1 + shape2)`` -- the same as a
    ``Binomial`` whose success probability equals the beta mean
    ``shape1 / (shape1 + shape2)``. As ``shape1`` and ``shape2`` grow with
    that ratio held fixed the beta concentrates on a single probability and
    the beta-binomial approaches that ``Binomial``; with
    ``shape1 = shape2 = 1`` the probability is uniform and every count from
    0 to ``n`` is equally likely.

    Examples
    --------
    >>> from symbulate import *
    >>> X = BetaBinomial(n=10, shape1=2, shape2=3)
    >>> float(X.mean())
    4.0
    >>> round(float(X.sd()), 4)
    2.4495
    >>> Y = BetaBinomial(n=10, shape1=1, shape2=1)  # uniform on 0..10
    >>> round(float(Y.pmf(0)), 4)
    0.0909
    >>> X.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, n, shape1, shape2):
        """Initialize a beta-binomial distribution.

        Raises
        ------
        Exception
            If ``n`` is not a non-negative integer, or ``shape1`` or
            ``shape2`` is not a positive number.
        """

        _validate(
            (
                not isinstance(n, numbers.Integral) or n < 0,
                "n must be a non-negative integer",
            ),
            (
                not isinstance(shape1, numbers.Real) or shape1 <= 0,
                "shape1 must be a positive number",
            ),
            (
                not isinstance(shape2, numbers.Real) or shape2 <= 0,
                "shape2 must be a positive number",
            ),
        )
        self.n = n
        self.shape1 = shape1
        self.shape2 = shape2

        params = {"n": n, "a": shape1, "b": shape2}
        super().__init__(params, stats.betabinom, True)


class BetaNegativeBinomial(Distribution):
    """Probability space for a beta-negative binomial distribution.

    Models the number of failures before the ``r``-th success (like the
    :class:`Pascal` distribution) when the success probability is not fixed
    but is itself random, drawn once from a ``Beta`` distribution (with
    shape parameters ``shape1`` and ``shape2``) and shared by all trials.
    That extra, trial-to-trial uncertainty in the probability makes the
    counts more spread out (overdispersed) than a plain
    ``Pascal``/``NegativeBinomial`` -- so this is the overdispersed
    negative-binomial counterpart of the :class:`BetaBinomial`. It is also
    known as the beta-Pascal distribution.

    Parameters
    ----------
    r : int
        Target number of successes. Must be a positive integer.
    shape1 : float
        First shape parameter (often written α) of the underlying beta
        distribution. Must be positive.
    shape2 : float
        Second shape parameter (often written β) of the underlying beta
        distribution. Must be positive.

    Attributes
    ----------
    r : int
        Target number of successes.
    shape1 : float
        First shape parameter (α) of the underlying beta distribution.
    shape2 : float
        Second shape parameter (β) of the underlying beta distribution.

    Notes
    -----
    The counts are the number of failures, so the support is
    ``0, 1, 2, ...``. The mean is finite only when ``shape1 > 1`` (and
    equals ``r * shape2 / (shape1 - 1)``). As ``shape1`` and ``shape2`` grow
    with the ratio ``shape1 / (shape1 + shape2)`` held fixed, the beta
    concentrates on a single probability ``p`` and the beta-negative
    binomial approaches a ``Pascal(r, p)`` distribution; with ``r = 1`` it
    is a beta-geometric distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = BetaNegativeBinomial(r=5, shape1=3, shape2=2)
    >>> float(X.mean())
    5.0
    >>> round(float(X.pmf(0)), 4)
    0.1667
    >>> X.draw()  # doctest: +SKIP
    7
    """

    def __init__(self, r, shape1, shape2):
        """Initialize a beta-negative binomial distribution.

        Raises
        ------
        Exception
            If ``r`` is not a positive integer, or ``shape1`` or ``shape2``
            is not a positive number.
        """

        _validate(
            (
                not isinstance(r, numbers.Integral) or r <= 0,
                "r must be a positive integer",
            ),
            (
                not isinstance(shape1, numbers.Real) or shape1 <= 0,
                "shape1 must be a positive number",
            ),
            (
                not isinstance(shape2, numbers.Real) or shape2 <= 0,
                "shape2 must be a positive number",
            ),
        )
        self.r = r
        self.shape1 = shape1
        self.shape2 = shape2

        # scipy's betanbinom takes n as the target number of successes and
        # a, b as the beta shape parameters; it counts failures, so the
        # support is 0, 1, 2, ... like Pascal.
        params = {"n": r, "a": shape1, "b": shape2}
        super().__init__(params, stats.betanbinom, True)


class Hypergeometric(Distribution):
    """Probability space for a hypergeometric distribution.

    Models the number of successes (1s) when drawing ``n`` items
    without replacement from a collection containing ``N0`` zeros
    and ``N1`` ones.

    Parameters
    ----------
    n : int
        Number of draws (without replacement). Must be positive.
    N0 : int
        Number of 0s (failures) in the collection.
    N1 : int
        Number of 1s (successes) in the collection.

    Attributes
    ----------
    n : int
        Number of draws (without replacement).
    N0 : int
        Number of 0s (failures) in the collection.
    N1 : int
        Number of 1s (successes) in the collection.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Hypergeometric(n=2, N0=3, N1=3)
    >>> float(X.mean())
    1.0
    >>> round(float(X.pmf(1)), 4)
    0.6
    >>> X.draw()  # doctest: +SKIP
    1
    """

    def __init__(self, n, N0, N1):
        """Initialize a hypergeometric distribution.

        Raises
        ------
        Exception
            If ``n`` is not a positive integer; if ``N0`` or ``N1`` is not
            a non-negative integer; or if ``N0 + N1`` is less than ``n``.
        """

        _validate(
            (
                not isinstance(n, numbers.Integral) or n <= 0,
                "n must be a positive integer",
            ),
            (
                not isinstance(N0, numbers.Integral) or N0 < 0,
                "N0 must be a non-negative integer",
            ),
            (
                not isinstance(N1, numbers.Integral) or N1 < 0,
                "N1 must be a non-negative integer",
            ),
            (
                isinstance(n, numbers.Integral)
                and isinstance(N0, numbers.Integral)
                and isinstance(N1, numbers.Integral)
                and N0 + N1 < n,
                "N0 + N1 cannot be less than the sample size n",
            ),
        )
        self.n = n
        self.N0 = N0
        self.N1 = N1

        params = {"M": N0 + N1, "n": N1, "N": n}

        super().__init__(params, stats.hypergeom, True)


class NegativeHypergeometric(Distribution):
    """Probability space for a negative hypergeometric distribution.

    The without-replacement counterpart of the :class:`Pascal`
    distribution, and the "keep drawing until" version of
    :class:`Hypergeometric`. Draw items one at a time and *without*
    replacement from a collection of ``N0`` zeros and ``N1`` ones, stopping
    as soon as ``r`` zeros have come out. This is the distribution of how
    many ones were drawn along the way.

    Where a :class:`Hypergeometric` fixes the number of draws and asks how
    many ones turn up, this fixes the number of *zeros* to collect and asks
    the same question -- so the number of draws is what varies.

    Parameters
    ----------
    r : int
        Number of zeros to draw before stopping. Must be a positive
        integer, and cannot exceed ``N0`` -- you cannot wait for more zeros
        than the collection holds.
    N0 : int
        Number of 0s (failures) in the collection. Must be a positive
        integer.
    N1 : int
        Number of 1s (successes) in the collection. Must be a non-negative
        integer.

    Attributes
    ----------
    r : int
        Number of zeros to draw before stopping.
    N0 : int
        Number of 0s (failures) in the collection.
    N1 : int
        Number of 1s (successes) in the collection.

    Notes
    -----
    Note what ``r`` counts here. In :class:`NegativeBinomial` and
    :class:`Pascal` the stopping count ``r`` is a number of *successes*; in
    this distribution it is a number of *zeros* (failures), and the ones
    are what gets counted up. The mean is ``r * N1 / (N0 + 1)``, and the
    possible values run from 0 (no ones drawn) up to ``N1`` (every one
    drawn before the last needed zero).

    Examples
    --------
    >>> from symbulate import *
    >>> X = NegativeHypergeometric(r=3, N0=7, N1=5)
    >>> float(X.mean())
    1.875
    >>> round(float(X.pmf(2)), 4)
    0.2652
    >>> X.draw()  # doctest: +SKIP
    2
    """

    def __init__(self, r, N0, N1):
        """Initialize a negative hypergeometric distribution.

        Raises
        ------
        Exception
            If ``r`` or ``N0`` is not a positive integer; if ``N1`` is not a
            non-negative integer; or if ``r`` is greater than ``N0``.
        """
        _validate(
            (
                not isinstance(r, numbers.Integral) or r <= 0,
                "r must be a positive integer",
            ),
            (
                not isinstance(N0, numbers.Integral) or N0 <= 0,
                "N0 must be a positive integer",
            ),
            (
                not isinstance(N1, numbers.Integral) or N1 < 0,
                "N1 must be a non-negative integer",
            ),
            # Waiting for more zeros than the collection holds can never
            # happen. scipy returns nan rather than raising, so this is
            # caught here to give a clear message instead.
            (
                isinstance(r, numbers.Integral)
                and isinstance(N0, numbers.Integral)
                and 0 < N0 < r,
                "r cannot be greater than N0, the number of 0s available to draw",
            ),
        )
        self.r = r
        self.N0 = N0
        self.N1 = N1

        # scipy's nhypergeom describes the same urn as M balls of which n are
        # the type being counted, drawing until r of the *other* type appear.
        # So M is the whole collection, its n is our N1, and its r is ours.
        params = {"M": N0 + N1, "n": N1, "r": r}
        super().__init__(params, stats.nhypergeom, True)

    def _fast_sim(self, n):
        """Draw all ``n`` samples in one vectorized call.

        A plain ``draw()`` goes through
        ``self.sim_func(**self.params, random_state=rng)``, and scipy's
        ``nhypergeom_gen._rvs`` rebuilds an entire CDF table and an
        inverse-CDF interpolator from scratch on *every* call, however few
        samples are asked for. Paying that once per ``.sim(n)`` rather than
        ``n`` times is the whole saving: roughly 1.1 ms per sample looped
        against about a microsecond per sample batched (~850x).

        See :meth:`Distribution._fast_sim` for why no generator is passed
        in.
        """
        return self.sim_func(**self.params, size=n, random_state=rng)


class Geometric(Distribution):
    """Probability space for a geometric distribution.

    Models the number of trials (including the success) until the
    first success, where each trial has probability ``p`` of success.

    Parameters
    ----------
    p : float
        Probability of success on each trial, strictly between 0 and 1.

    Attributes
    ----------
    p : float
        Probability of success on each trial, strictly between 0 and 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Geometric(p=0.5)
    >>> float(X.mean())
    2.0
    >>> float(X.pmf(1))
    0.5
    >>> X.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, p):
        """Initialize a geometric distribution.

        Raises
        ------
        Exception
            If ``p`` is not a number between 0 and 1.
        """

        _validate(
            (
                not isinstance(p, numbers.Real) or not 0 < p <= 1,
                "p must be a number between 0 and 1",
            ),
        )
        self.p = p

        params = {"p": p}
        super().__init__(params, stats.geom, True)


class NegativeBinomial(Distribution):
    """Probability space for a negative binomial distribution.

    Models the total number of trials (including the ``r`` successes)
    until the ``r``-th success, where each trial has probability ``p``
    of success.

    Parameters
    ----------
    r : int
        Target number of successes. Must be a positive integer.
    p : float
        Probability of success on each trial, between 0 and 1.

    Attributes
    ----------
    r : int
        Target number of successes.
    p : float
        Probability of success on each trial.

    Examples
    --------
    >>> from symbulate import *
    >>> X = NegativeBinomial(r=3, p=0.5)
    >>> float(X.mean())
    6.0
    >>> X.draw()  # doctest: +SKIP
    7
    """

    def __init__(self, r, p):
        """Initialize a negative binomial distribution.

        Raises
        ------
        Exception
            If ``r`` is not a positive integer, or ``p`` is not a number
            between 0 and 1.
        """

        _validate(
            (
                not isinstance(r, numbers.Integral) or r <= 0,
                "r must be a positive integer",
            ),
            (
                not isinstance(p, numbers.Real) or not 0 < p <= 1,
                "p must be a number between 0 and 1",
            ),
        )
        self.r = r
        self.p = p

        params = {"n": r, "p": p, "loc": r}
        super().__init__(params, stats.nbinom, True)

    def draw(self):
        """Draw a single random sample from the negative binomial distribution.

        Returns
        -------
        Scalar
            The total number of trials (including the ``r`` successes)
            until the ``r``-th success.

        Examples
        --------
        >>> from symbulate import *
        >>> NegativeBinomial(r=3, p=0.5).draw()  # doctest: +SKIP
        6
        """

        # Numpy's negative binomial returns numbers in [0, inf),
        # but we want numbers in [r, inf).
        # Wrapped in Scalar so this override returns the same type as every
        # other `draw` (the base class returns `Scalar(...)`); `Int` subclasses
        # `int`, so nothing that treated this as a plain int is affected.
        return Scalar(self.r + rng.negative_binomial(n=self.r, p=self.p))


class Pascal(Distribution):
    """Probability space for a Pascal distribution.

    Models the number of failures before the ``r``-th success, where
    each trial has probability ``p`` of success. Unlike the negative
    binomial, the ``r`` successes themselves are not counted.

    Parameters
    ----------
    r : int
        Target number of successes. Must be a positive integer.
    p : float
        Probability of success on each trial, between 0 and 1.

    Attributes
    ----------
    r : int
        Target number of successes.
    p : float
        Probability of success on each trial.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Pascal(r=1, p=0.5)
    >>> float(X.mean())
    1.0
    >>> round(float(X.pmf(0)), 4)
    0.5
    >>> X.draw()  # doctest: +SKIP
    0
    """

    def __init__(self, r, p):
        """Initialize a Pascal distribution.

        Raises
        ------
        Exception
            If ``r`` is not a positive integer, or ``p`` is not a number
            between 0 and 1.
        """

        _validate(
            (
                not isinstance(r, numbers.Integral) or r <= 0,
                "r must be a positive integer",
            ),
            (
                not isinstance(p, numbers.Real) or not 0 < p <= 1,
                "p must be a number between 0 and 1",
            ),
        )
        self.r = r
        self.p = p

        params = {"n": r, "p": p}
        super().__init__(params, stats.nbinom, True)


class Poisson(Distribution):
    """Probability space for a Poisson distribution.

    Models the number of events occurring in a fixed interval of time
    or space, when events happen at a constant average rate ``lam``
    and independently of each other.

    Parameters
    ----------
    lam : float
        Average number of events per interval (λ). Must be a non-negative
        number; ``lam=0`` gives a point mass at 0.

    Attributes
    ----------
    lam : float
        Average number of events per interval (the rate parameter,
        often written as λ). Must be non-negative.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Poisson(lam=4)
    >>> float(X.mean())
    4.0
    >>> float(X.sd())
    2.0
    >>> X.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, lam):
        """Initialize a Poisson distribution.

        Raises
        ------
        Exception
            If ``lam`` is not a non-negative number.
        """

        _validate(
            (
                not isinstance(lam, numbers.Real) or lam < 0,
                "lam must be a non-negative number",
            ),
        )
        self.lam = lam

        params = {"mu": lam}
        super().__init__(params, stats.poisson, True)


class DiscreteUniform(Distribution):
    """Probability space for a discrete uniform distribution.

    Every integer from ``a`` to ``b`` (inclusive) is equally likely.
    This can model, for example, rolling a fair die.

    Parameters
    ----------
    a : int, optional
        Smallest possible value. Default is 0.
    b : int, optional
        Largest possible value (inclusive). Default is 1.

    Attributes
    ----------
    a : int
        Smallest possible value.
    b : int
        Internal upper bound (stored as ``b + 1`` for scipy compatibility).

    Examples
    --------
    >>> from symbulate import *
    >>> X = DiscreteUniform(a=1, b=6)
    >>> float(X.mean())
    3.5
    >>> float(X.pmf(3))
    0.16666666666666666
    >>> X.draw()  # doctest: +SKIP
    4
    """

    def __init__(self, a=0, b=1):
        """Initialize a discrete uniform distribution.

        Raises
        ------
        Exception
            If ``a`` or ``b`` is not an integer, or if ``b`` is less than
            ``a``.
        """
        # The bounds must be whole numbers, not just numbers: this
        # distribution puts equal probability on the *integers* from a to b,
        # and scipy's randint returns nan for every pmf, mean, and quantile
        # when handed a fractional bound -- with no error of its own -- so a
        # fractional bound has to be caught here or it fails silently later.
        #
        # A whole-valued float (6.0, or anything arriving from numpy or a
        # division) is accepted, matching how `Naturals` tests membership in
        # index_sets.py: only a genuinely fractional bound is rejected.
        whole_a, whole_b = _is_whole_number(a), _is_whole_number(b)
        _validate(
            (not whole_a, "a must be an integer"),
            (not whole_b, "b must be an integer"),
            (whole_a and whole_b and a > b, "b cannot be less than a"),
        )

        # Stored as plain ints so the attributes match their documented type
        # however the bounds were supplied.
        self.a = int(a)
        self.b = int(b) + 1

        params = {"low": self.a, "high": self.b}

        super().__init__(params, stats.randint, True)


class Zipf(Distribution):
    """Probability space for a Zipf distribution.

    Ranks a fixed collection of ``n`` items from most common (rank 1) to
    least common (rank ``n``), and gives the item of rank ``k`` a
    probability proportional to ``1 / k ** shape``. This is Zipf's law: a
    few items take most of the probability and a long tail of items each
    take very little. It is the standard model for word frequencies in a
    text, city sizes in a country, or page visits on a website -- any
    setting where the items can be ranked and the number of them is finite
    and known.

    Parameters
    ----------
    shape : float
        Exponent of the power law. Must be non-negative. A larger ``shape``
        piles more probability onto the first few ranks and shortens the
        tail; ``shape = 0`` makes every rank equally likely.
    n : int
        Number of items being ranked, which is also the largest possible
        value. Must be a positive integer.

    Attributes
    ----------
    shape : float
        Exponent of the power law.
    n : int
        Number of items being ranked.

    Notes
    -----
    The support is **finite**: the integers ``1, 2, ..., n``, and nothing
    outside that range. The probability of rank ``k`` is

    ``pmf(k) = (1 / k ** shape) / H(n, shape)``,

    where ``H(n, shape) = 1 / 1 ** shape + ... + 1 / n ** shape`` is the
    normalizing constant that makes the probabilities add up to 1.

    This wraps ``scipy.stats.zipfian``: the package's ``shape`` is scipy's
    ``a``, and ``n`` is scipy's ``n``. Note that scipy's other power-law
    name, ``scipy.stats.zipf``, is a **different** distribution -- the zeta
    distribution, supported on ``1, 2, 3, ...`` with no upper limit -- so
    it is deliberately not used here. That infinite-support version exists
    only when ``shape > 1`` (otherwise the infinite sum does not converge),
    while the finite-support Zipf is defined for every ``shape >= 0``,
    because a sum of finitely many terms always converges.

    Two special cases are worth knowing. ``shape = 0`` is exactly
    ``DiscreteUniform(1, n)``: every rank equally likely. And with
    ``shape > 1`` held fixed, letting ``n`` grow approaches the zeta
    distribution, since the ranks past ``n`` carry vanishingly little
    probability.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Zipf(shape=1, n=5)
    >>> round(float(X.pmf(1)), 4)  # 1 / (1 + 1/2 + 1/3 + 1/4 + 1/5)
    0.438
    >>> round(float(X.mean()), 4)
    2.1898
    >>> float(X.cdf(5))  # all of the probability is at or below n
    1.0
    >>> float(X.pmf(6))  # nothing above n
    0.0
    >>> X.draw()  # doctest: +SKIP
    1
    """

    def __init__(self, shape, n):
        """Initialize a Zipf distribution.

        Raises
        ------
        Exception
            If ``shape`` is not a non-negative number, or ``n`` is not a
            positive integer.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape < 0,
                "shape must be a non-negative number",
            ),
            (
                not isinstance(n, numbers.Integral) or n < 1,
                "n must be a positive integer",
            ),
        )
        self.shape = shape
        self.n = n

        # scipy.stats.zipfian is the finite-support Zipf; the package's shape
        # is scipy's `a` and n is scipy's `n`. scipy.stats.zipf is NOT this
        # distribution -- it is the infinite-support zeta distribution -- so
        # it must not be used here.
        params = {"a": shape, "n": n}
        super().__init__(params, stats.zipfian, True)


class Zeta(Distribution):
    """Probability space for a zeta distribution.

    A discrete power-law distribution on the whole numbers
    ``1, 2, 3, ...``, with probabilities proportional to ``1 / k ** shape``.
    It is the unbounded counterpart of :class:`Zipf`: where a ``Zipf`` ranks
    a fixed list of ``n`` items, a ``Zeta`` lets the list run on forever.
    Used to model power-law counts such as word frequencies, city sizes, or
    citation counts, where a few outcomes are enormously more common than
    the rest.

    The name comes from the Riemann zeta function, which supplies the
    constant that makes the probabilities add to 1:
    ``P(X = k) = 1 / (k ** shape * zeta(shape))``.

    Parameters
    ----------
    shape : float
        Power-law exponent. Must be **greater than 1** -- at 1 or below the
        probabilities cannot be normalized. Larger values concentrate the
        distribution on the smallest whole numbers.

    Attributes
    ----------
    shape : float
        Power-law exponent.

    Notes
    -----
    This distribution has a genuinely heavy tail, and how many moments
    exist depends on ``shape``: the mean is finite only when
    ``shape > 2`` (where it equals ``zeta(shape - 1) / zeta(shape)``), and
    the variance only when ``shape > 3``. Outside those ranges ``mean()``
    and ``var()`` return ``inf`` rather than a number.

    That tail also means no sensible plotting window can hold most of the
    probability when ``shape`` is close to 1 -- at ``shape = 1.1`` even the
    first hundred values carry only about 40% of it. So :attr:`xlim`
    deliberately shows just the first 20 values, which is where the
    power-law shape is visible, rather than stretching out along a tail
    that never really ends. Pass ``xlim=(1, high)`` to :meth:`plot` to look
    further out.

    **A naming warning about scipy.** ``scipy.stats.zipf`` is this
    distribution, the zeta -- *not* the finite Zipf, which scipy calls
    ``scipy.stats.zipfian`` and this package exposes as :class:`Zipf`.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Zeta(shape=2.5)
    >>> round(float(X.pmf(1)), 4)
    0.7454
    >>> round(float(X.mean()), 4)
    1.9474
    >>> float(Zeta(shape=1.5).mean())  # mean needs shape > 2
    inf
    >>> X.draw()  # doctest: +SKIP
    1
    """

    def __init__(self, shape):
        """Initialize a zeta distribution.

        Raises
        ------
        Exception
            If ``shape`` is not a number greater than 1.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 1,
                "shape must be a number greater than 1",
            ),
        )
        self.shape = shape

        # A fixed head-of-the-distribution window, on purpose: this is the one
        # distribution whose support-derived default would be unusable rather
        # than merely unhelpful. The support is [1, inf), so the rule would cut
        # the open end at a quantile -- and the 99.9th percentile of a power law
        # with an exponent near 1 is so far out that scipy's quantile search
        # effectively never returns. The pmf decreases from k = 1, so the first
        # 20 values are always where the shape shows.
        #
        # Setting it here, before calling up, is safe: `Distribution.__init__`
        # deliberately does not reset the window, so either ordering works.
        # `_xlim` rather than the `xlim` setter, because this is still a
        # window the distribution chose for itself -- it keeps the
        # whole-number slot of discrete padding that the setter's
        # hand-chosen numbers give up.
        self._xlim = (1, 20)

        # scipy's `zipf` is the zeta distribution (its `a` is our shape);
        # scipy's `zipfian` is the finite Zipf that this package's Zipf uses.
        params = {"a": shape}
        super().__init__(params, stats.zipf, True)


class _benford_gen(stats.rv_discrete):
    """scipy generator for Benford's law (the leading-digit law).

    Supported on the leading digits ``1, ..., base - 1``, with
    ``pmf(d) = log_base(1 + 1/d)``. Written with ``log1p`` and a change of
    base rather than ``log(1 + 1/d) / log(base)`` spelled out, so the small
    ``1/d`` at the top of the range keeps its precision. scipy has no
    Benford's law built in, so it is defined here; every other case in this
    module wraps an existing scipy distribution directly.
    """

    def _argcheck(self, base):
        return (base == np.floor(base)) & (base >= 2)

    def _get_support(self, base):
        return 1, base - 1

    def _pmf(self, d, base):
        return np.log1p(1.0 / d) / np.log(base)


_benford = _benford_gen(name="benford")


class Benford(Distribution):
    """Probability space for Benford's law, the leading-digit law.

    A discrete distribution on the possible leading digits
    ``1, 2, ..., base - 1``, giving the digit ``d`` probability
    ``log_base(1 + 1/d)``. In base 10 that makes a leading 1 about six
    times as likely as a leading 9 -- roughly 30.1% against 4.6% -- which
    is nothing like the 1-in-9 a person would guess.

    The law describes the first digit of many real collections of numbers:
    street addresses, populations, physical constants, invoice amounts,
    file sizes. It tends to appear when the values span several orders of
    magnitude and are not pinned to a particular scale. That is also what
    makes it a fraud-detection tool -- figures invented by a person usually
    do not follow it, so an audited ledger whose leading digits are close
    to uniform is worth a second look.

    Parameters
    ----------
    base : int, optional
        The number base the digits are written in. Must be an integer
        that is at least 2. Default is 10, the everyday case.

    Attributes
    ----------
    base : int
        The number base the digits are written in.

    Notes
    -----
    The probabilities come from assuming the *logarithm* of the underlying
    quantity is spread evenly: the digit ``d`` leads exactly when the value
    falls in ``[d, d + 1)`` after scaling, and that interval takes up
    ``log_base(d + 1) - log_base(d) = log_base(1 + 1/d)`` of each cycle of
    the log scale. They add to 1 by telescoping, whatever the base.

    ``base = 2`` is a legitimate but degenerate edge case: the only
    possible leading digit in binary is 1, so the distribution is a point
    mass there.

    scipy has no Benford's law, so this one is defined in this module (see
    ``_benford_gen``) rather than wrapping an existing scipy distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Benford()
    >>> round(float(X.pmf(1)), 4)  # a leading 1, about 30% of the time
    0.301
    >>> round(float(X.pmf(9)), 4)  # a leading 9, about 4.6%
    0.0458
    >>> round(float(X.mean()), 4)
    3.4402
    >>> float(Benford(base=2).pmf(1))  # in binary every number leads with 1
    1.0
    >>> X.draw()  # doctest: +SKIP
    1
    """

    def __init__(self, base=10):
        """Initialize a Benford distribution.

        Raises
        ------
        Exception
            If ``base`` is not an integer greater than or equal to 2.
        """
        # A whole-valued float (10.0, or anything arriving from numpy or a
        # division) is accepted, as elsewhere in this module; only a
        # genuinely fractional base is rejected. The range check is guarded
        # by the type check so a non-number short-circuits before the
        # comparison, per _validate's documented convention.
        whole_base = _is_whole_number(base)
        _validate(
            (not whole_base, "base must be an integer"),
            (whole_base and base < 2, "base must be an integer greater than 1"),
        )
        self.base = int(base)

        params = {"base": self.base}
        super().__init__(params, _benford, True)


## Continuous Distributions


class Uniform(Distribution):
    """Probability space for a continuous uniform distribution.

    Every value between ``a`` and ``b`` is equally likely.

    Parameters
    ----------
    a : float, optional
        Lower bound of the distribution. Default is 0.0.
    b : float, optional
        Upper bound of the distribution. Default is 1.0.

    Attributes
    ----------
    a : float
        Lower bound of the distribution.
    b : float
        Upper bound of the distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Uniform(a=0, b=1)
    >>> float(X.mean())
    0.5
    >>> float(X.pdf(0.5))
    1.0
    >>> X.draw()  # doctest: +SKIP
    0.374
    """

    def __init__(self, a=0.0, b=1.0):
        """Initialize a uniform distribution.

        Raises
        ------
        Exception
            If ``a`` or ``b`` is not a number, or if ``b`` is less than ``a``.
        """
        _validate(
            (not isinstance(a, numbers.Real), "a must be a number"),
            (not isinstance(b, numbers.Real), "b must be a number"),
            (
                isinstance(a, numbers.Real) and isinstance(b, numbers.Real) and a > b,
                "b cannot be less than a",
            ),
        )

        self.a = a
        self.b = b

        params = {"loc": a, "scale": b - a}

        super().__init__(params, stats.uniform, False)


class IrwinHall(Distribution):
    """Probability space for an Irwin-Hall distribution.

    The distribution of the **sum** of ``n`` independent uniform values.
    With the default bounds that is ``Uniform(0, 1) + Uniform(0, 1) + ...``
    added up ``n`` times, so the possible totals run from 0 to ``n``.

    It is the standard way to watch the Central Limit Theorem happen with
    the simplest ingredient there is. One uniform is flat, two make a
    triangle, three already look rounded, and by a dozen the density is
    almost indistinguishable from a normal curve -- even though no single
    piece of the sum resembles one. ``IrwinHall(n=12)`` is the classic
    demonstration case: it has mean 6 and variance exactly 1, so
    subtracting 6 gives a well-known cheap stand-in for a standard normal
    value.

    It is the companion of :class:`Bates`, which is the *mean* of ``n``
    uniforms rather than their sum.

    Parameters
    ----------
    n : int
        How many uniform values are added together. Must be a positive
        integer.
    a : float, optional
        Lower bound of each uniform value being added. Default is 0.0.
    b : float, optional
        Upper bound of each uniform value being added. Default is 1.0.
        Cannot be less than ``a``.

    Attributes
    ----------
    n : int
        How many uniform values are added together.
    a : float
        Lower bound of each uniform value being added.
    b : float
        Upper bound of each uniform value being added.

    Notes
    -----
    The support is the interval ``[n * a, n * b]``: the smallest possible
    total happens when every uniform value lands at its low end, and the
    largest when they all land at the high end. The mean and variance are
    just ``n`` times those of a single uniform,

    ``mean = n * (a + b) / 2``  and  ``var = n * (b - a) ** 2 / 12``,

    since the pieces are independent. Note that the spread grows like the
    square root of ``n`` while the range grows like ``n``, which is why the
    density bunches up in the middle of its support as ``n`` increases.

    Two small cases are distributions in their own right: ``n = 1`` is
    exactly ``Uniform(a, b)``, and ``n = 2`` is exactly
    ``Triangular(2 * a, a + b, 2 * b)``. Dividing the sum by ``n`` gives the
    **average** of the ``n`` uniform values, which is :class:`Bates`:
    ``Bates(n, a, b)`` is the distribution of ``RV(IrwinHall(n, a, b)) / n``.

    This wraps ``scipy.stats.irwinhall``, whose ``scale`` is the width of
    each individual uniform rather than of the whole sum, so the bounds are
    passed as ``loc = n * a`` and ``scale = b - a``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = IrwinHall(n=12)
    >>> float(X.mean())
    6.0
    >>> float(X.var())
    1.0
    >>> round(float(X.pdf(6)), 4)  # nearly Normal(6, 1)'s 0.3989
    0.3939
    >>> float(IrwinHall(n=2).pdf(1))  # the peak of a triangle on [0, 2]
    1.0
    >>> X.draw()  # doctest: +SKIP
    5.87
    """

    def __init__(self, n, a=0.0, b=1.0):
        """Initialize an Irwin-Hall distribution.

        Raises
        ------
        Exception
            If ``n`` is not a positive integer, if ``a`` or ``b`` is not a
            number, or if ``b`` is less than ``a``.
        """
        _validate(
            (
                not isinstance(n, numbers.Integral) or n < 1,
                "n must be a positive integer",
            ),
            (not isinstance(a, numbers.Real), "a must be a number"),
            (not isinstance(b, numbers.Real), "b must be a number"),
            # Only meaningful once both bounds are numbers, so this
            # comparison is guarded by its own type checks.
            (
                isinstance(a, numbers.Real) and isinstance(b, numbers.Real) and a > b,
                "b cannot be less than a",
            ),
        )
        self.n = n
        self.a = a
        self.b = b

        # scipy's irwinhall scales each uniform being summed, not the sum
        # itself: it draws n copies of Uniform(0, scale) and shifts the total
        # by loc. So the width `b - a` is the scale, and the whole sum starts
        # at `n * a` (all n pieces at their low end), not at `a`.
        params = {"n": n, "loc": n * a, "scale": b - a}
        super().__init__(params, stats.irwinhall, False)


class Bates(Distribution):
    """Probability space for a Bates distribution.

    The distribution of the average of ``n`` independent ``Uniform(a, b)``
    random variables. It is a classic teaching example of the central limit
    theorem: with ``n = 1`` it is just the uniform distribution, and as
    ``n`` grows the average's distribution becomes an increasingly tight,
    bell-shaped curve centered at the uniform mean. It is the companion of
    the Irwin-Hall distribution, which is the *sum* of ``n`` uniforms rather
    than their mean.

    Parameters
    ----------
    n : int
        Number of uniform random variables being averaged. Must be a
        positive integer.
    a : float, optional
        Lower bound of each uniform. Default is 0.0.
    b : float, optional
        Upper bound of each uniform. Default is 1.0.

    Attributes
    ----------
    n : int
        Number of uniform random variables being averaged.
    a : float
        Lower bound of each uniform.
    b : float
        Upper bound of each uniform.

    Notes
    -----
    The support is ``[a, b]``. The mean is the uniform mean ``(a + b) / 2``,
    and the variance is ``(b - a)**2 / (12 * n)`` -- the single-uniform
    variance shrunk by a factor of ``n``. Special cases: ``n = 1`` is the
    ``Uniform(a, b)`` distribution, and ``n = 2`` is the (symmetric)
    triangular distribution on ``[a, b]``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Bates(n=5)
    >>> float(X.mean())
    0.5
    >>> round(float(X.var()), 6)
    0.016667
    >>> X.draw()  # doctest: +SKIP
    0.48
    """

    def __init__(self, n, a=0.0, b=1.0):
        """Initialize a Bates distribution.

        Raises
        ------
        Exception
            If ``n`` is not a positive integer, ``a`` or ``b`` is not a
            number, or ``b`` is less than ``a``.
        """
        _validate(
            (
                not isinstance(n, numbers.Integral) or n <= 0,
                "n must be a positive integer",
            ),
            (not isinstance(a, numbers.Real), "a must be a number"),
            (not isinstance(b, numbers.Real), "b must be a number"),
            (
                isinstance(a, numbers.Real) and isinstance(b, numbers.Real) and a > b,
                "b cannot be less than a",
            ),
        )
        self.n = n
        self.a = a
        self.b = b

        # scipy has no Bates, but the mean of n uniforms is the Irwin-Hall
        # sum divided by n: scaling scipy's irwinhall by (b - a) / n turns the
        # sum of n Uniform(0, 1) variables into the mean of n Uniform(a, b)
        # variables, with support [a, b].
        params = {"n": n, "loc": a, "scale": (b - a) / n}
        super().__init__(params, stats.irwinhall, False)


class LogUniform(Distribution):
    """Probability space for a log-uniform (reciprocal) distribution.

    A continuous distribution on ``[a, b]`` that is uniform on a
    logarithmic scale: ``log(X)`` is uniformly distributed between
    ``log(a)`` and ``log(b)``. Equivalently, its density is proportional to
    ``1 / x``, which is why it is also called the reciprocal distribution
    (see :class:`Reciprocal`). It is the standard choice for sampling a
    machine-learning hyperparameter that ranges over several orders of
    magnitude (a learning rate from 1e-5 to 1e-1, say), because it spreads
    the draws evenly across the powers of ten rather than piling them up
    near the top of the range.

    Parameters
    ----------
    a : float
        Lower bound of the support. Must be positive.
    b : float
        Upper bound of the support. Must be greater than ``a``.

    Attributes
    ----------
    a : float
        Lower bound of the support.
    b : float
        Upper bound of the support.

    Notes
    -----
    The density is ``1 / (x * log(b / a))`` on ``[a, b]``. The mean is
    ``(b - a) / log(b / a)`` and the median is the geometric mean
    ``sqrt(a * b)``. If ``X`` has this distribution, then ``log(X)`` is
    ``Uniform(log(a), log(b))`` -- the defining relationship.

    Examples
    --------
    >>> from symbulate import *
    >>> X = LogUniform(a=1, b=100)
    >>> round(float(X.median()), 4)
    10.0
    >>> round(float(X.mean()), 4)
    21.4976
    >>> round(float(X.pdf(1)), 4)
    0.2171
    >>> X.draw()  # doctest: +SKIP
    3.7
    """

    def __init__(self, a, b):
        """Initialize a log-uniform distribution.

        Raises
        ------
        Exception
            If ``a`` or ``b`` is not a positive number, or ``b`` is not
            greater than ``a``.
        """
        _validate(
            (not isinstance(a, numbers.Real) or a <= 0, "a must be a positive number"),
            (not isinstance(b, numbers.Real) or b <= 0, "b must be a positive number"),
            (
                isinstance(a, numbers.Real)
                and isinstance(b, numbers.Real)
                and a > 0
                and b > 0
                and a >= b,
                "b must be greater than a",
            ),
        )
        self.a = a
        self.b = b

        # scipy's loguniform (a.k.a. reciprocal) takes the lower and upper
        # bounds directly as a and b.
        params = {"a": a, "b": b}
        super().__init__(params, stats.loguniform, False)


class Reciprocal(LogUniform):
    """Probability space for a reciprocal distribution.

    An alternative name for the :class:`LogUniform` distribution: its
    density is proportional to ``1 / x`` (the reciprocal) on ``[a, b]``, so
    ``Reciprocal(a, b)`` is identical to ``LogUniform(a, b)``.

    Parameters
    ----------
    a : float
        Lower bound of the support. Must be positive.
    b : float
        Upper bound of the support. Must be greater than ``a``.

    Examples
    --------
    >>> from symbulate import *
    >>> round(float(Reciprocal(a=1, b=100).median()), 4)
    10.0
    """


class Normal(Distribution):
    """Probability space for a normal (Gaussian) distribution.

    The classic bell-shaped distribution, described by its mean and
    standard deviation. You can specify either ``sd`` or ``var``,
    but not both.

    Parameters
    ----------
    mean : float, optional
        Mean of the distribution. Default is 0.0.
    sd : float, optional
        Standard deviation. Must be a non-negative number. Mutually
        exclusive with ``var``. Default is 1.0 (used when neither ``sd``
        nor ``var`` is given).
    var : float, optional
        Variance. Must be a non-negative number. Mutually exclusive
        with ``sd``.

    Attributes
    ----------
    scale : float
        The standard deviation used internally, regardless of whether
        ``sd`` or ``var`` was passed in.

    Notes
    -----
    ``quantile(0)`` and ``quantile(1)`` return ``-inf`` and ``+inf``;
    quantile arguments outside ``[0, 1]`` return ``nan`` (standard scipy
    behavior).

    Examples
    --------
    >>> from symbulate import *
    >>> X = Normal(mean=0, sd=1)
    >>> float(X.mean())
    0.0
    >>> float(X.sd())
    1.0
    >>> float(X.pdf(0))
    0.3989422804014327
    >>> X.draw()  # doctest: +SKIP
    -0.234
    """

    def __init__(self, mean=0.0, sd=None, var=None):
        """Initialize a normal distribution.

        Raises
        ------
        ValueError
            If both ``sd`` and ``var`` are specified with inconsistent values.
        Exception
            If ``mean`` is not a number, or the supplied ``sd`` or ``var``
            is not a non-negative number.

        Warns
        -----
        UserWarning
            If both ``sd`` and ``var`` are specified but are consistent
            (i.e. ``sd == sqrt(var)``). Use only one.
        """

        if sd is not None and var is not None:
            if not (
                isinstance(sd, numbers.Real)
                and isinstance(var, numbers.Real)
                and math.isclose(sd**2, var)
            ):
                raise ValueError("Specify sd or var, but not both.")
            warnings.warn("Both sd and var were provided. Use only one.", UserWarning)
            var = None

        if sd is None and var is None:
            sd = 1.0

        if var is None:
            _validate(
                (not isinstance(mean, numbers.Real), "mean must be a number"),
                (
                    not isinstance(sd, numbers.Real) or sd < 0,
                    "sd must be a non-negative number",
                ),
            )
            self.scale = sd
        else:
            _validate(
                (not isinstance(mean, numbers.Real), "mean must be a number"),
                (
                    not isinstance(var, numbers.Real) or var < 0,
                    "var must be a non-negative number",
                ),
            )
            self.scale = np.sqrt(var)

        params = {"loc": mean, "scale": self.scale}
        super().__init__(params, stats.norm, False)


class TruncatedNormal(Distribution):
    """Probability space for a truncated normal distribution.

    A normal (Gaussian) distribution restricted to an interval ``[a, b]``:
    values that would fall outside the interval are discarded and the
    remaining bell curve is rescaled so it again integrates to one. This is
    the continuous counterpart of truncating a distribution to a range, and
    it is the standard way to model a quantity that is normally distributed
    "in principle" but physically confined to a range -- a measurement that
    cannot go below zero, a rating capped at a maximum, and so on.

    Either bound may be infinite, giving a one-sided truncation: leave ``b``
    at its default for a lower bound only (e.g. ``TruncatedNormal(a=0)``),
    or leave ``a`` at its default for an upper bound only. As with
    :class:`Normal`, you can specify either ``sd`` or ``var``, but not both.

    Parameters
    ----------
    mean : float, optional
        Mean of the *underlying* (untruncated) normal. Default is 0.0.
        Note this is not the mean of the truncated distribution, which is
        pulled toward the center of ``[a, b]``.
    sd : float, optional
        Standard deviation of the underlying normal. Must be a positive
        number. Mutually exclusive with ``var``. Default is 1.0 (used when
        neither ``sd`` nor ``var`` is given).
    var : float, optional
        Variance of the underlying normal. Must be a positive number.
        Mutually exclusive with ``sd``.
    a : float, optional
        Lower truncation bound, in the same units as the data. Default is
        ``-inf`` (no lower bound).
    b : float, optional
        Upper truncation bound, in the same units as the data. Must be
        greater than ``a``. Default is ``+inf`` (no upper bound).

    Attributes
    ----------
    a : float
        Lower truncation bound.
    b : float
        Upper truncation bound.
    scale : float
        The standard deviation of the underlying normal, regardless of
        whether ``sd`` or ``var`` was passed in.

    Notes
    -----
    ``scipy.stats.truncnorm`` takes its bounds in *standardized* units, as
    multiples of the standard deviation away from the mean. This class lets
    you give ``a`` and ``b`` in the natural units of the data and converts
    them internally with ``(a - mean) / sd`` and ``(b - mean) / sd``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = TruncatedNormal(mean=0, sd=1, a=-2, b=2)
    >>> float(X.pdf(3))
    0.0
    >>> bool(-2 <= X.draw() <= 2)
    True

    See Also
    --------
    Normal : The untruncated normal distribution.
    HalfNormal : A standard normal truncated below at zero.
    """

    def __init__(self, mean=0.0, sd=None, var=None, a=-np.inf, b=np.inf):
        """Initialize a truncated normal distribution.

        Raises
        ------
        ValueError
            If both ``sd`` and ``var`` are specified with inconsistent values.
        Exception
            If ``mean``, ``a``, or ``b`` is not a number; if the supplied
            ``sd`` or ``var`` is not a positive number; or if ``b`` is not
            greater than ``a``.

        Warns
        -----
        UserWarning
            If both ``sd`` and ``var`` are specified but are consistent
            (i.e. ``sd == sqrt(var)``). Use only one.
        """

        if sd is not None and var is not None:
            if not (
                isinstance(sd, numbers.Real)
                and isinstance(var, numbers.Real)
                and math.isclose(sd**2, var)
            ):
                raise ValueError("Specify sd or var, but not both.")
            warnings.warn("Both sd and var were provided. Use only one.", UserWarning)
            var = None

        if sd is None and var is None:
            sd = 1.0

        if var is None:
            _validate(
                (not isinstance(mean, numbers.Real), "mean must be a number"),
                (
                    not isinstance(sd, numbers.Real) or sd <= 0,
                    "sd must be a positive number",
                ),
            )
            self.scale = sd
        else:
            _validate(
                (not isinstance(mean, numbers.Real), "mean must be a number"),
                (
                    not isinstance(var, numbers.Real) or var <= 0,
                    "var must be a positive number",
                ),
            )
            self.scale = np.sqrt(var)

        _validate(
            (not isinstance(a, numbers.Real), "a must be a number"),
            (not isinstance(b, numbers.Real), "b must be a number"),
            (
                isinstance(a, numbers.Real) and isinstance(b, numbers.Real) and a >= b,
                "b must be greater than a",
            ),
        )
        self.a = a
        self.b = b

        # scipy.stats.truncnorm takes its bounds in standardized units (number
        # of standard deviations from the mean), so convert the data-unit
        # bounds here. Infinite bounds pass through unchanged.
        params = {
            "a": (a - mean) / self.scale,
            "b": (b - mean) / self.scale,
            "loc": mean,
            "scale": self.scale,
        }
        super().__init__(params, stats.truncnorm, False)

        # Show the hard truncation edge wherever one exists, and fall back to
        # a far quantile on any side left unbounded.
        lower = a if np.isfinite(a) else self.quantile(0.001)
        upper = b if np.isfinite(b) else self.quantile(0.999)

    def _fast_sim(self, n):
        """Draw all ``n`` samples in one vectorized call.

        The same idea as :meth:`NegativeHypergeometric._fast_sim`, less
        extreme: scipy's ``truncnorm._rvs`` inverts a uniform through
        ``_ppf``, which is real per-call work rather than a table rebuild,
        but still work that batching amortizes -- about 80x in testing.

        See :meth:`Distribution._fast_sim` for why no generator is passed
        in.
        """
        return self.sim_func(**self.params, size=n, random_state=rng)


class SkewNormal(Distribution):
    """Probability space for a skew-normal distribution.

    A bell curve that has been tilted to one side. It is the
    :class:`Normal` distribution with one extra parameter, ``shape``,
    which leans the curve left or right: the peak slides toward one side
    and the tail on the other side stretches out. Setting ``shape = 0``
    tilts it not at all and gives back an ordinary
    ``Normal(loc, scale)``. It is used wherever data is bell-shaped but
    visibly lopsided -- test scores, growth measurements, financial
    returns -- and as a model that lets the skewness be estimated instead
    of assumed away.

    Parameters
    ----------
    loc : float, optional
        Location parameter (the center the curve is tilted around).
        Default is 0. This is not the mean of the distribution unless
        ``shape`` is 0 -- see Notes.
    scale : float, optional
        Scale parameter (controls the spread). Must be a positive number.
        Default is 1. This is not the standard deviation of the
        distribution unless ``shape`` is 0 -- see Notes.
    shape : float, optional
        Shape parameter (usually written alpha), which sets the direction
        and amount of the tilt. Any number is allowed. Default is 0,
        which gives the normal distribution. Positive values stretch the
        right tail (skewed right), negative values the left tail (skewed
        left).

    Attributes
    ----------
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    shape : float
        Shape parameter (alpha).

    Notes
    -----
    Tilting the curve moves its center of mass, so ``loc`` and ``scale``
    are *not* the mean and standard deviation once ``shape`` is nonzero.
    Writing ``delta = shape / sqrt(1 + shape ** 2)`` for the tilt measured
    on a 0-to-1 scale,

    ``X.mean() == loc + scale * delta * sqrt(2 / pi)``  and
    ``X.var() == scale ** 2 * (1 - 2 * delta ** 2 / pi)``.

    So tilting to the right pulls the mean above ``loc`` and, at the same
    time, makes the distribution *less* spread out than ``scale`` suggests.
    The distribution covers every real number for any ``shape``: the tilt
    thins one tail but never cuts it off.

    Reflecting the shape reflects the distribution. With ``loc = 0``,
    ``shape`` and ``-shape`` give mirror images of each other, so a
    left-skewed curve is just a right-skewed one flipped about its center.

    The amount of skewness this family can produce is limited: as
    ``shape`` grows the skewness rises toward about 0.995 and no further.
    A more lopsided data set than that needs a different distribution
    (:class:`LogNormal` or :class:`Gamma`, say), not a bigger ``shape``.

    Both of its endpoints are familiar distributions. At ``shape = 0``
    it is exactly ``Normal(loc, scale)``. As ``shape`` grows without
    bound the left half is squeezed away entirely and what is left is a
    half-normal: ``loc + scale * HalfNormal(scale=1)``, that is,
    ``loc + scale * abs(Normal(0, 1))``. Between those extremes it is
    built from the same two pieces mixed together --
    ``loc + scale * (delta * abs(Z) + sqrt(1 - delta ** 2) * W)`` for
    independent standard normals ``Z`` and ``W`` -- which is a handy way
    to see where the one-sided tail comes from.

    This wraps ``scipy.stats.skewnorm``, whose shape parameter ``a`` is
    the same alpha, so it is passed straight through as ``a = shape``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = SkewNormal(loc=0, scale=1, shape=4)
    >>> round(float(X.mean()), 4)  # tilted right, so above loc = 0
    0.7741
    >>> round(float(X.var()), 4)  # and narrower than scale = 1 suggests
    0.4008
    >>> round(float(X.pdf(1)), 4)
    0.4839
    >>> Y = SkewNormal(loc=0, scale=1, shape=0)  # no tilt: a Normal(0, 1)
    >>> round(float(Y.pdf(1)), 4)
    0.242
    >>> X.draw()  # doctest: +SKIP
    0.63
    """

    def __init__(self, loc=0, scale=1, shape=0):
        """Initialize a skew-normal distribution.

        Raises
        ------
        Exception
            If ``loc`` or ``shape`` is not a number, or ``scale`` is not a
            positive number.
        """
        _validate(
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
            (
                not isinstance(shape, numbers.Real),
                "shape must be a number. Use shape = 0 for no tilt (a normal "
                "distribution), a positive shape to stretch the right tail, "
                "or a negative shape to stretch the left tail.",
            ),
        )
        self.loc = loc
        self.scale = scale
        self.shape = shape

        # scipy's skewnorm uses the same shape parameter alpha, under the name a.
        params = {"a": shape, "loc": loc, "scale": scale}
        super().__init__(params, stats.skewnorm, False)
        # Unbounded on both sides, like Laplace and Gumbel, so both ends of
        # the default window are quantile cuts. Even at a large shape the thin
        # tail is only ever thin, never absent, so the window stays reasonable.


class Exponential(Distribution):
    """Probability space for an exponential distribution.

    Models the waiting time between events that occur at a constant
    average rate. Specify either ``rate`` (λ) or ``scale`` (1/λ),
    but not both.

    Parameters
    ----------
    rate : float, optional
        Rate parameter λ. Must be positive. Default is 1.0.
        Mutually exclusive with ``scale``.
    scale : float, optional
        Scale parameter 1/λ. Must be a positive number. Mutually
        exclusive with ``rate``.

    Attributes
    ----------
    rate : float or None
        The rate parameter λ. ``None`` if ``scale`` was specified instead.
    scale : float or None
        The scale parameter 1/λ. ``None`` if ``rate`` was specified instead.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Exponential(rate=1)
    >>> float(X.mean())
    1.0
    >>> float(X.sd())
    1.0
    >>> float(X.pdf(1))
    0.36787944117144233
    >>> X.draw()  # doctest: +SKIP
    0.423
    """

    def __init__(self, rate=None, scale=None):
        """Initialize an exponential distribution.

        Raises
        ------
        Exception
            If both ``rate`` and ``scale`` are specified with inconsistent
            values, or the supplied ``rate`` or ``scale`` is not a positive
            number.

        Warns
        -----
        UserWarning
            If both ``rate`` and ``scale`` are specified but are consistent
            (i.e. ``scale == 1/rate``). Use only one.
        """

        if rate is not None and scale is not None:
            if not (
                isinstance(rate, numbers.Real)
                and isinstance(scale, numbers.Real)
                and math.isclose(scale, 1.0 / rate)
            ):
                raise Exception("Specify either rate or scale, not both.")
            warnings.warn(
                "Both rate and scale were provided. Use only one.", UserWarning
            )
            scale = None

        if rate is None and scale is None:
            rate = 1.0

        if scale is None:
            _validate(
                (
                    not isinstance(rate, numbers.Real) or rate <= 0,
                    "rate must be a positive number",
                ),
            )
            self.rate = rate
            self.scale = None
            params = {"scale": 1.0 / rate}
        else:
            _validate(
                (
                    not isinstance(scale, numbers.Real) or scale <= 0,
                    "scale must be a positive number",
                ),
            )
            self.rate = None
            self.scale = scale
            params = {"scale": scale}

        super().__init__(params, stats.expon, False)


class ExponentiallyModifiedGaussian(Distribution):
    """Probability space for an exponentially modified Gaussian distribution.

    The distribution of a normal value **plus** an independent exponential
    value: ``Normal(mean, sd) + Exponential(rate)``. The result looks like
    a bell curve that has been smeared out to the right -- a rounded peak
    on the left, and a long exponential tail on the right. It is also
    called the ex-Gaussian distribution, and it is the standard model for
    a quantity built from a symmetric part plus a one-sided delay: human
    reaction times (a steady motor response plus a variable decision
    time), or the shape of a chromatography peak.

    The two pieces control the two halves of the shape. ``sd`` sets how
    wide the rounded left side is, and ``rate`` sets how long the right
    tail is: a large ``rate`` means short exponential delays and a nearly
    normal shape, while a small ``rate`` means long delays and a strongly
    skewed one.

    Parameters
    ----------
    mean : float, optional
        Mean of the normal part being added. Default is 0.0. This is not
        the mean of the whole distribution -- see Notes.
    sd : float, optional
        Standard deviation of the normal part being added. Must be a
        positive number. Default is 1.0.
    rate : float, optional
        Rate parameter λ of the exponential part being added. Must be a
        positive number. Default is 1.0. The average delay it contributes
        is ``1 / rate``.

    Attributes
    ----------
    loc : float
        Mean of the normal part (the ``mean`` argument). Stored under this
        name because ``mean`` is the method that reports the mean of the
        whole distribution.
    scale : float
        Standard deviation of the normal part (the ``sd`` argument).
        Stored under this name because ``sd`` is the method that reports
        the standard deviation of the whole distribution.
    rate : float
        Rate parameter λ of the exponential part.

    Notes
    -----
    Adding the exponential part shifts the center to the right and widens
    the spread, so the summary numbers are *not* the ``mean`` and ``sd``
    that were passed in:

    ``X.mean() == mean + 1 / rate``  and  ``X.var() == sd ** 2 + 1 / rate ** 2``.

    Both follow from adding independent pieces: means add, and variances
    add. So ``ExponentiallyModifiedGaussian(mean=0, sd=1, rate=1)`` has
    mean 1, not 0. The distribution is always skewed to the right, and it
    covers every real number, since the normal part can reach arbitrarily
    far in either direction.

    The two familiar distributions it is built from are its limiting
    cases. As ``rate`` grows the exponential delay shrinks to nothing and
    the shape approaches ``Normal(mean, sd)``; as ``sd`` shrinks toward 0
    the normal part becomes a constant and the shape approaches an
    ``Exponential(rate)`` shifted right by ``mean``. Neither endpoint is
    allowed here (both ``sd`` and ``rate`` must be positive) -- use
    :class:`Normal` or :class:`Exponential` directly for those.

    This wraps ``scipy.stats.exponnorm``, whose shape parameter ``K`` is
    the exponential's average delay measured in standard deviations of the
    normal part, so it is passed as ``K = 1 / (rate * sd)``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = ExponentiallyModifiedGaussian(mean=0, sd=1, rate=1)
    >>> float(X.mean())  # 0 + 1 / 1, not the mean=0 that was passed in
    1.0
    >>> float(X.var())  # 1 ** 2 + 1 / 1 ** 2
    2.0
    >>> round(float(X.pdf(0)), 4)
    0.2616
    >>> Y = ExponentiallyModifiedGaussian(mean=0, sd=1, rate=100)
    >>> round(float(Y.pdf(0)), 4)  # a big rate is nearly Normal(0, 1)'s 0.3989
    0.3989
    >>> X.draw()  # doctest: +SKIP
    1.42
    """

    def __init__(self, mean=0.0, sd=1.0, rate=1.0):
        """Initialize an exponentially modified Gaussian distribution.

        Raises
        ------
        Exception
            If ``mean`` is not a number, or ``sd`` or ``rate`` is not a
            positive number.
        """
        _validate(
            (not isinstance(mean, numbers.Real), "mean must be a number"),
            (
                not isinstance(sd, numbers.Real) or sd <= 0,
                "sd must be a positive number. For sd = 0 there is no normal "
                "part left to add, so use Exponential(rate) instead.",
            ),
            (
                not isinstance(rate, numbers.Real) or rate <= 0,
                "rate must be a positive number. For no exponential part at "
                "all, use Normal(mean, sd) instead.",
            ),
        )
        # Not stored as self.mean / self.sd: those names belong to the methods
        # that report the mean and sd of the whole distribution, which include
        # the exponential part and so differ from these two arguments.
        self.loc = mean
        self.scale = sd
        self.rate = rate

        # scipy's exponnorm measures the exponential's average delay (1 / rate)
        # in units of the normal part's standard deviation, so its shape
        # parameter is K = (1 / rate) / sd.
        params = {"K": 1.0 / (rate * sd), "loc": mean, "scale": sd}
        super().__init__(params, stats.exponnorm, False)
        # Unbounded on both sides, like Laplace and Gumbel, so both ends of
        # the default window are quantile cuts, and the automatic zoom has
        # nothing further to give up. A small rate makes the right tail long
        # enough that the window looks lopsided -- pass xlim=(low, high) to
        # plot() to frame the bulk of the probability.


class Gamma(Distribution):
    """Probability space for a gamma distribution.

    A flexible continuous distribution often used to model waiting times.
    Specify either ``rate`` or ``scale``, but not both. The gamma
    generalizes the exponential (``shape=1``) distribution.

    Parameters
    ----------
    shape : float
        Shape parameter α. Must be positive.
    rate : float, optional
        Rate parameter λ. Default is 1.0. Mutually exclusive with ``scale``.
    scale : float, optional
        Scale parameter 1/λ. Must be a positive number. Mutually
        exclusive with ``rate``.

    Attributes
    ----------
    shape : float
        Shape parameter α. Controls the number of "phases."
    rate : float or None
        Rate parameter λ. ``None`` if ``scale`` was specified instead.
    scale : float or None
        Scale parameter 1/λ. ``None`` if ``rate`` was specified instead.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Gamma(shape=2, rate=1)
    >>> float(X.mean())
    2.0
    >>> float(X.sd())
    1.4142135623730951
    >>> X.draw()  # doctest: +SKIP
    1.52
    """

    def __init__(self, shape, rate=None, scale=None):
        """Initialize a gamma distribution.

        Raises
        ------
        Exception
            If ``shape`` is not a positive number; if both ``rate`` and
            ``scale`` are specified with inconsistent values; or if the
            supplied ``rate`` or ``scale`` is not a positive number.

        Warns
        -----
        UserWarning
            If both ``rate`` and ``scale`` are specified but are consistent
            (i.e. ``scale == 1/rate``). Use only one.
        """

        if rate is not None and scale is not None:
            if not (
                isinstance(rate, numbers.Real)
                and isinstance(scale, numbers.Real)
                and math.isclose(scale, 1.0 / rate)
            ):
                raise Exception("Specify either rate or scale, not both.")
            warnings.warn(
                "Both rate and scale were provided. Use only one.", UserWarning
            )
            scale = None

        if rate is None and scale is None:
            rate = 1.0

        if scale is None:
            _validate(
                (
                    not isinstance(shape, numbers.Real) or shape <= 0,
                    "shape must be a positive number",
                ),
                (
                    not isinstance(rate, numbers.Real) or rate <= 0,
                    "rate must be a positive number",
                ),
            )
            self.shape = shape
            self.rate = rate
            self.scale = None
            params = {"a": shape, "scale": 1.0 / rate}
        else:
            _validate(
                (
                    not isinstance(shape, numbers.Real) or shape <= 0,
                    "shape must be a positive number",
                ),
                (
                    not isinstance(scale, numbers.Real) or scale <= 0,
                    "scale must be a positive number",
                ),
            )
            self.shape = shape
            self.rate = None
            self.scale = scale
            params = {"a": shape, "scale": scale}

        super().__init__(params, stats.gamma, False)


class InverseGamma(Distribution):
    """Probability space for an inverse gamma distribution.

    A continuous distribution on (0, infinity): if ``X`` has a gamma
    distribution, then ``1 / X`` has an inverse gamma distribution. It is
    the standard conjugate prior for the variance parameter of a normal
    distribution in Bayesian statistics.

    Parameters
    ----------
    shape : float
        Shape parameter (often written alpha). Must be positive.
    scale : float, optional
        Scale parameter (often written beta). Must be positive. Default is
        1. It equals the rate of the underlying gamma distribution (see
        Notes).

    Attributes
    ----------
    shape : float
        Shape parameter (alpha).
    scale : float
        Scale parameter (beta).

    Notes
    -----
    The mean is finite only when ``shape > 1`` (and equals
    ``scale / (shape - 1)``); the variance only when ``shape > 2``. The
    reciprocal relationship is exact: ``1 / Gamma(shape, rate=scale)`` has
    an ``InverseGamma(shape, scale)`` distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = InverseGamma(shape=3, scale=2)
    >>> float(X.mean())
    1.0
    >>> float(X.sd())
    1.0
    >>> round(float(X.pdf(1)), 4)
    0.5413
    >>> X.draw()  # doctest: +SKIP
    0.75
    """

    def __init__(self, shape, scale=1.0):
        """Initialize an inverse gamma distribution.

        Raises
        ------
        Exception
            If ``shape`` or ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape = shape
        self.scale = scale

        # scipy's invgamma takes a as the shape and scale the scale.
        params = {"a": shape, "scale": scale}
        super().__init__(params, stats.invgamma, False)


class ScaledInverseChiSquare(InverseGamma):
    """Probability space for a scaled inverse chi-square distribution.

    A continuous distribution on (0, infinity), written
    ``Scaled-Inv-chi^2(df, scale)`` in Bayesian textbooks (e.g. Gelman et
    al.). It is a standard conjugate prior for the variance of a normal
    distribution, given in a degrees-of-freedom-and-scale parameterization.
    It is exactly an :class:`InverseGamma` under a change of variables --

        ``ScaledInverseChiSquare(df, scale)``
        ``== InverseGamma(shape=df / 2, scale=df * scale / 2)``

    -- so this class is a convenience name for that inverse gamma, using the
    ``(df, scale)`` parameterization Bayesian work is usually written in.

    Parameters
    ----------
    df : float
        Degrees of freedom (often written nu). Must be positive.
    scale : float, optional
        Scale parameter (often written s-squared or tau-squared) -- a
        variance-like scale, not a standard deviation. Must be positive.
        Default is 1.

    Attributes
    ----------
    df : float
        Degrees of freedom.
    scale : float
        Scale parameter.

    Notes
    -----
    The mean is ``df * scale / (df - 2)``, finite only when ``df > 2``; the
    variance is finite only when ``df > 4``. The name comes from the
    defining relationship: if ``X`` has this distribution, then
    ``df * scale / X`` has a ``ChiSquare(df)`` distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = ScaledInverseChiSquare(df=6, scale=2)
    >>> float(X.mean())
    3.0
    >>> float(X.sd())
    3.0
    >>> round(float(X.pdf(2)), 4)
    0.3361
    >>> X.draw()  # doctest: +SKIP
    2.14
    """

    def __init__(self, df, scale=1.0):
        """Initialize a scaled inverse chi-square distribution.

        Raises
        ------
        Exception
            If ``df`` or ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(df, numbers.Real) or df <= 0,
                "df must be a positive number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        # Scaled-inv-chi^2(df, scale) is InverseGamma(df / 2, df * scale / 2);
        # delegate to InverseGamma so all of its scipy wiring, plotting, and
        # x-window are reused unchanged.
        super().__init__(shape=df / 2, scale=df * scale / 2)
        # Re-expose the distribution's own (df, scale) parameterization
        # (this overrides InverseGamma's shape/scale attributes).
        self.df = df
        self.scale = scale


class LogGamma(Distribution):
    """Probability space for a log-gamma distribution.

    The distribution of the **logarithm of** a gamma random variable: if
    ``G`` has a :class:`Gamma` distribution, then ``log(G)`` has this
    log-gamma distribution. Taking the log compresses the gamma's long
    right tail and stretches its left side, so the result is a
    left-skewed distribution defined over *all* real numbers -- negative
    values are perfectly ordinary. It appears in extreme value theory and,
    in actuarial work, as a model on the log scale of a loss.

    Parameters
    ----------
    shape : float
        Shape parameter of the underlying gamma distribution. Must be
        positive. Larger values make the distribution tighter and less
        skewed.
    loc : float, optional
        Location parameter, shifting the whole distribution. Default is 0.
    scale : float, optional
        Scale parameter, stretching the distribution. Must be positive.
        Default is 1. Note this scales the log-gamma variable itself, so it
        is *not* the scale of the underlying gamma distribution.

    Attributes
    ----------
    shape : float
        Shape parameter of the underlying gamma distribution.
    loc : float
        Location parameter.
    scale : float
        Scale parameter.

    Notes
    -----
    **Read the name carefully -- it points the opposite way from**
    :class:`LogNormal`. A log-normal variable is one *whose own logarithm*
    is normal, so a ``LogNormal`` is always positive. A log-gamma variable,
    by the convention used here, *is itself a logarithm* of a gamma
    variable, so it ranges over all real numbers and is frequently
    negative. Put another way: ``log(LogNormal)`` is normal, whereas
    ``exp(LogGamma)`` is gamma.

    With ``loc = 0`` and ``scale = 1`` the mean and variance have closed
    forms in terms of the digamma and trigamma functions:
    ``mean = digamma(shape)`` and ``var = polygamma(1, shape)``.

    Setting ``shape = 1`` gives the smallest-extreme-value distribution,
    which is the mirror image of this package's :class:`Gumbel` (that one
    models maxima, this reflection models minima).

    Examples
    --------
    >>> from symbulate import *
    >>> X = LogGamma(shape=2)
    >>> round(float(X.mean()), 4)
    0.4228
    >>> round(float(X.sd()), 4)
    0.8031
    >>> round(float(X.pdf(0)), 4)
    0.3679
    >>> X.draw()  # doctest: +SKIP
    0.31
    """

    def __init__(self, shape, loc=0, scale=1):
        """Initialize a log-gamma distribution.

        Raises
        ------
        Exception
            If ``shape`` is not a positive number, ``loc`` is not a number,
            or ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape = shape
        self.loc = loc
        self.scale = scale

        # scipy's loggamma takes c as the shape of the underlying gamma.
        params = {"c": shape, "loc": loc, "scale": scale}
        # Unbounded in both directions, so both ends of the default window
        # are quantile cuts -- the same treatment GEV gets.
        super().__init__(params, stats.loggamma, False)


class InverseGaussian(Distribution):
    """Probability space for an inverse Gaussian (Wald) distribution.

    A right-skewed distribution over the positive numbers. It arises as a
    *first passage time*: if a Brownian motion drifts steadily upward, the
    time it first reaches a fixed level has this distribution. That makes it
    a natural model for a duration or a waiting time -- how long a repair
    takes, how long a customer stays -- where most values cluster near a
    typical length but a few run much longer.

    Parameters
    ----------
    mean : float, optional
        Expected value of the distribution. Must be positive. Default is 1.0.
    shape : float, optional
        Shape parameter (often written λ). Must be positive. Larger values
        concentrate the distribution around ``mean`` and make it less skewed.
        Default is 1.0.

    Attributes
    ----------
    mean_param : float
        Expected value of the distribution, as passed in.
    shape : float
        Shape parameter (λ).

    Notes
    -----
    **Despite the name, it is unrelated to the reciprocal of a normal random
    variable.** "Inverse" refers to a relationship between two functions
    describing Brownian motion, not to dividing anything by anything. It is
    also called the Wald distribution.

    The variance is ``mean ** 3 / shape``, so spread grows quickly with the
    mean and shrinks as the shape grows. As ``shape`` goes to infinity with
    ``mean`` held fixed, the distribution approaches
    ``Normal(mean, mean ** 3 / shape)`` -- that is, a normal distribution
    with the same mean and variance, the skewness washing out as the
    variance shrinks.

    The expected value is stored as ``mean_param`` rather than ``mean``,
    because :class:`Distribution` gives every distribution a ``mean()``
    *method* returning the expected value. Storing the parameter under its
    own name would overwrite that method, so ``X.mean()`` calls the method
    and ``X.mean_param`` reads the number that was passed in. They agree:
    ``X.mean() == X.mean_param``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = InverseGaussian(mean=2, shape=3)
    >>> round(float(X.mean()), 4)
    2.0
    >>> round(float(X.var()), 4)
    2.6667
    >>> round(float(X.pdf(2)), 4)
    0.2443
    >>> X.draw()  # doctest: +SKIP
    1.37
    """

    def __init__(self, mean=1.0, shape=1.0):
        """Initialize an inverse Gaussian distribution.

        Raises
        ------
        Exception
            If ``mean`` or ``shape`` is not a positive number.
        """
        _validate(
            (
                not isinstance(mean, numbers.Real) or mean <= 0,
                "mean must be a positive number",
            ),
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
        )
        # Stored as `mean_param`, not `mean`: `Distribution.__init__` assigns
        # `self.mean` the callable that returns the expected value, so a
        # `self.mean` here would be silently replaced by that method.
        self.mean_param = mean
        self.shape = shape

        # scipy's invgauss is parameterized by `mu` and `scale`, with
        # mean = mu * scale and variance = mu ** 3 * scale ** 2. Setting
        # scale = shape and mu = mean / shape gives back the textbook
        # (mean, shape) convention used here:
        #     mean:      (mean / shape) * shape           = mean
        #     variance:  (mean / shape) ** 3 * shape ** 2 = mean ** 3 / shape
        params = {"mu": mean / shape, "scale": shape}
        # Support is (0, inf), so the default window is (0, quantile(0.999)) --
        # the fixed lower bound kept as-is, the unbounded upper end cut at a
        # quantile. No per-distribution window code is needed.
        super().__init__(params, stats.invgauss, False)


class Beta(Distribution):
    """Probability space for a beta distribution.

    A continuous distribution defined on ``[xmin, xmax]`` -- by default
    ``[0, 1]`` -- often used to model probabilities or proportions. The
    shape changes with parameters ``shape1`` and ``shape2``.

    Parameters
    ----------
    shape1 : float
        First shape parameter (often written α). Must be positive.
    shape2 : float
        Second shape parameter (often written β). Must be positive.
    xmin : float, optional
        Smallest possible value. Default is 0.0.
    xmax : float, optional
        Largest possible value. Must be greater than ``xmin``. Default
        is 1.0.

    Attributes
    ----------
    shape1 : float
        First shape parameter (α). Must be positive.
    shape2 : float
        Second shape parameter (β). Must be positive.
    xmin : float
        Smallest possible value.
    xmax : float
        Largest possible value.

    Notes
    -----
    The two shape parameters set the *shape* of the density and the two
    bounds set *where it sits*, so the two choices are independent. A beta
    on ``[xmin, xmax]`` is the standard one on ``[0, 1]`` stretched by the
    width and shifted to the new start: if ``S`` has a
    ``Beta(shape1, shape2)`` distribution, then

        ``xmin + (xmax - xmin) * S``

    has a ``Beta(shape1, shape2, xmin, xmax)`` distribution. Every summary
    follows from that same stretch-and-shift -- the mean, for instance, is
    ``xmin + (xmax - xmin) * shape1 / (shape1 + shape2)``.

    The default ``xmin=0``, ``xmax=1`` leaves the standard beta unchanged.
    :class:`PERT` is built the same way, stretching a beta onto
    ``[low, high]`` with shapes chosen to put the peak at a given mode.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Beta(shape1=1, shape2=1)
    >>> float(X.mean())
    0.5
    >>> round(float(X.pdf(0.5)), 4)
    1.0
    >>> float(Beta(1, 1, xmin=2, xmax=4).mean())
    3.0
    >>> X.draw()  # doctest: +SKIP
    0.632
    """

    def __init__(self, shape1, shape2, xmin=0.0, xmax=1.0):
        """Initialize a beta distribution.

        Raises
        ------
        Exception
            If ``shape1`` or ``shape2`` is not a positive number, if
            ``xmin`` or ``xmax`` is not a number, or if ``xmax`` is not
            greater than ``xmin``.
        """
        # The bounds are compared to each other below, so that check is
        # guarded by this type check -- otherwise a non-numeric bound would
        # raise a cryptic TypeError from the comparison instead of reporting
        # the friendly message. Same guard PERT uses on its three bounds.
        numeric = isinstance(xmin, numbers.Real) and isinstance(xmax, numbers.Real)
        _validate(
            (
                not isinstance(shape1, numbers.Real) or shape1 <= 0,
                "shape1 must be a positive number",
            ),
            (
                not isinstance(shape2, numbers.Real) or shape2 <= 0,
                "shape2 must be a positive number",
            ),
            (not isinstance(xmin, numbers.Real), "xmin must be a number"),
            (not isinstance(xmax, numbers.Real), "xmax must be a number"),
            (numeric and xmax <= xmin, "xmax must be greater than xmin"),
        )
        self.shape1 = shape1
        self.shape2 = shape2
        self.xmin = xmin
        self.xmax = xmax

        # Stretch the standard beta from [0, 1] onto [xmin, xmax] with
        # scipy's own loc/scale, the same mechanism PERT uses. The defaults
        # give loc=0, scale=1, which is scipy's no-op, so the standard beta
        # is unchanged.
        params = {"a": shape1, "b": shape2, "loc": xmin, "scale": xmax - xmin}
        super().__init__(params, stats.beta, False)


class PERT(Distribution):
    """Probability space for a PERT (beta-PERT) distribution.

    A continuous distribution on ``[low, high]`` built from the three
    numbers an expert can usually estimate: the best case, the most likely
    case, and the worst case. It is the standard distribution of the
    Program Evaluation and Review Technique, used in project management and
    risk analysis to model something like how long a task will take.

    Its appeal is the mean, ``(low + weight * mode + high) / (weight + 2)``
    -- with the default ``weight`` of 4 this is the familiar
    ``(low + 4 * mode + high) / 6``. The most likely value counts heavily,
    but a far-away worst case still pulls the average toward it. So for a
    task estimated at best 1 day, likely 2, worst 10, the mean is about
    3.17 days rather than the 2 you might plan around.

    Underneath, a PERT is a beta distribution stretched onto
    ``[low, high]``, with shape parameters chosen so that its peak sits
    exactly at ``mode``:

    - ``alpha = 1 + weight * (mode - low) / (high - low)``
    - ``beta = 1 + weight * (high - mode) / (high - low)``

    Parameters
    ----------
    low : float
        Smallest possible value -- the best case. Must be less than
        ``high``.
    mode : float
        Most likely value, where the density peaks. Must be between ``low``
        and ``high`` (either end is allowed).
    high : float
        Largest possible value -- the worst case. Must be greater than
        ``low``.
    weight : float, optional
        How much weight the most likely value carries, relative to the two
        extremes. Must be positive. Default is 4, which gives the standard
        PERT distribution; larger values concentrate the distribution more
        tightly around ``mode``.

    Attributes
    ----------
    low : float
        Smallest possible value.
    mode : float
        Most likely value.
    high : float
        Largest possible value.
    weight : float
        Weight carried by the most likely value.
    alpha : float
        First shape parameter of the underlying beta distribution.
    beta : float
        Second shape parameter of the underlying beta distribution.

    Notes
    -----
    A ``mode`` at either endpoint is allowed and needs no special handling:
    ``mode == low`` gives ``alpha = 1`` (a density that only decreases) and
    ``mode == high`` gives ``beta = 1`` (one that only increases).

    Examples
    --------
    >>> from symbulate import *
    >>> X = PERT(low=1, mode=2, high=10)
    >>> round(float(X.mean()), 4)
    3.1667
    >>> round(float(X.median()), 4)
    2.8978
    >>> X.draw()  # doctest: +SKIP
    2.87
    """

    def __init__(self, low, mode, high, weight=4.0):
        """Initialize a PERT distribution.

        Raises
        ------
        Exception
            If ``low``, ``mode``, or ``high`` is not a number; if ``high``
            is not greater than ``low``; if ``mode`` does not lie between
            ``low`` and ``high``; or if ``weight`` is not a positive number.
        """
        # The two cross-parameter checks below compare the three bounds, so
        # they are guarded by this type check -- otherwise a non-numeric
        # argument would raise a cryptic TypeError from the comparison
        # instead of reporting the friendly message.
        numeric = (
            isinstance(low, numbers.Real)
            and isinstance(mode, numbers.Real)
            and isinstance(high, numbers.Real)
        )
        _validate(
            (not isinstance(low, numbers.Real), "low must be a number"),
            (not isinstance(mode, numbers.Real), "mode must be a number"),
            (not isinstance(high, numbers.Real), "high must be a number"),
            (
                not isinstance(weight, numbers.Real) or weight <= 0,
                "weight must be a positive number",
            ),
            (numeric and low >= high, "high must be greater than low"),
            # Only meaningful once the bounds themselves make sense.
            (
                numeric and low < high and not low <= mode <= high,
                "mode must be between low and high",
            ),
        )
        self.low = low
        self.mode = mode
        self.high = high
        self.weight = weight

        # Translate the three-point estimate into the beta shape parameters
        # that put the peak at `mode`, then stretch the standard beta from
        # [0, 1] onto [low, high] with scipy's loc/scale.
        span = high - low
        self.alpha = 1 + weight * (mode - low) / span
        self.beta = 1 + weight * (high - mode) / span

        params = {"a": self.alpha, "b": self.beta, "loc": low, "scale": span}
        super().__init__(params, stats.beta, False)


class Triangular(Distribution):
    """Probability space for a triangular distribution.

    A continuous distribution on ``[low, high]`` whose density is literally
    a triangle: it rises in a straight line from 0 at ``low`` up to a peak
    at ``mode``, then falls in a straight line back to 0 at ``high``. The
    peak height is ``2 / (high - low)``, whatever the mode, since the area
    must be 1.

    Like :class:`PERT` it is built from a three-point estimate -- the best
    case, the most likely case, and the worst case -- which makes it a
    common first choice in simulation and risk modeling when those three
    numbers are all anyone can supply. It is the simpler and blunter of the
    two: its mean is ``(low + mode + high) / 3``, giving the two extremes
    the same weight as the most likely value, whereas a PERT weights the
    mode four times as heavily. A triangular distribution is therefore
    more spread out than the PERT on the same three numbers, and has a
    sharp corner at the mode rather than a smooth peak.

    Parameters
    ----------
    low : float
        Smallest possible value -- the best case. Must be less than
        ``high``.
    mode : float
        Most likely value, where the triangle peaks. Must be between
        ``low`` and ``high`` (either end is allowed).
    high : float
        Largest possible value -- the worst case. Must be greater than
        ``low``.

    Attributes
    ----------
    low : float
        Smallest possible value.
    mode : float
        Most likely value.
    high : float
        Largest possible value.

    Notes
    -----
    A ``mode`` at either endpoint is allowed and gives a right triangle:
    ``mode == low`` slopes only downward, and ``mode == high`` only upward.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Triangular(low=1, mode=2, high=10)
    >>> round(float(X.mean()), 4)
    4.3333
    >>> float(X.median())
    4.0
    >>> round(float(X.pdf(2)), 4)
    0.2222
    >>> X.draw()  # doctest: +SKIP
    3.61
    """

    def __init__(self, low, mode, high):
        """Initialize a triangular distribution.

        Raises
        ------
        Exception
            If ``low``, ``mode``, or ``high`` is not a number; if ``high``
            is not greater than ``low``; or if ``mode`` does not lie
            between ``low`` and ``high``.
        """
        # As in PERT, the two cross-parameter checks compare the bounds, so
        # they are guarded by this type check to keep a non-numeric argument
        # from raising a cryptic TypeError from the comparison instead.
        numeric = (
            isinstance(low, numbers.Real)
            and isinstance(mode, numbers.Real)
            and isinstance(high, numbers.Real)
        )
        _validate(
            (not isinstance(low, numbers.Real), "low must be a number"),
            (not isinstance(mode, numbers.Real), "mode must be a number"),
            (not isinstance(high, numbers.Real), "high must be a number"),
            (numeric and low >= high, "high must be greater than low"),
            # Only meaningful once the bounds themselves make sense.
            (
                numeric and low < high and not low <= mode <= high,
                "mode must be between low and high",
            ),
        )
        self.low = low
        self.mode = mode
        self.high = high

        # scipy's triang takes the mode as a *fraction* of the way along the
        # support (its `c`), not as a value on the data's own scale, so the
        # mode has to be converted before being handed over.
        span = high - low
        params = {"c": (mode - low) / span, "loc": low, "scale": span}
        super().__init__(params, stats.triang, False)


class _kumaraswamy_gen(stats.rv_continuous):
    """The Kumaraswamy distribution written in scipy's generator form.

    Every other distribution in this file wraps one that ``scipy.stats``
    already ships. The Kumaraswamy is the exception -- scipy has no
    ``kumaraswamy`` -- so it is supplied here, written to scipy's own
    ``rv_continuous`` interface. That way :class:`Kumaraswamy` can hand this
    object to ``Distribution.__init__`` exactly as :class:`Beta` hands over
    ``stats.beta``, and it inherits ``draw``, ``sim``, ``plot`` and the
    summary statistics with no special-casing anywhere else.

    Private, and named in lowercase to match the scipy generator classes it
    imitates (``beta_gen``, ``weibull_min_gen``). Students use
    :class:`Kumaraswamy`, never this.

    The two shape parameters ``a`` and ``b`` are positive and the support is
    ``[0, 1]``. Every quantity below has a closed form, so none of it is
    computed numerically:

    - density:  ``a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)``
    - cdf:      ``1 - (1 - x ** a) ** b``
    - quantile: ``(1 - (1 - q) ** (1 / b)) ** (1 / a)``
    - moments:  ``E[X ** n] = b * B(1 + n / a, b)``

    Supplying the quantile function is what makes sampling exact: scipy's
    default ``rvs`` feeds uniform draws through ``_ppf``, which is the
    textbook inverse-cdf construction for this distribution.
    """

    def _pdf(self, x, a, b):
        """Density at ``x``."""
        # The density is genuinely infinite at 0 when a < 1, and at 1 when
        # b < 1; numpy reaches those through a divide-by-zero in the power.
        # The infinity is the correct answer, so silence the warning without
        # touching the value -- plot() already filters non-finite heights.
        with np.errstate(divide="ignore"):
            return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)

    def _cdf(self, x, a, b):
        """Cumulative probability ``P(X <= x)``."""
        return 1 - (1 - x**a) ** b

    def _sf(self, x, a, b):
        """Survival function ``P(X > x)``, exact instead of ``1 - cdf``."""
        return (1 - x**a) ** b

    def _ppf(self, q, a, b):
        """Quantile function: the inverse of :meth:`_cdf`."""
        return (1 - (1 - q) ** (1 / b)) ** (1 / a)

    def _isf(self, q, a, b):
        """Inverse survival function, exact instead of ``ppf(1 - q)``."""
        return (1 - q ** (1 / b)) ** (1 / a)

    def _munp(self, n, a, b):
        """``n``-th raw moment ``E[X ** n] = b * B(1 + n / a, b)``."""
        # Substituting t = x ** a in the defining integral turns it into a
        # beta integral, so the moments are exact rather than quadrature.
        return b * beta_function(1 + n / a, b)


# Support [0, 1], like scipy's own `beta = beta_gen(a=0.0, b=1.0, ...)`. The
# `a`/`b` here are rv_continuous's support bounds, not the shape parameters
# of the same name -- scipy uses both spellings, exactly as it does for beta.
_kumaraswamy = _kumaraswamy_gen(a=0.0, b=1.0, name="kumaraswamy", shapes="a, b")


class Kumaraswamy(Distribution):
    """Probability space for a Kumaraswamy distribution.

    A continuous distribution on [0, 1], used -- like the ``Beta`` -- to
    model a proportion, a percentage, or a probability. It was introduced
    to describe hydrological quantities that are bounded at both ends, such
    as the fraction of a reservoir that is full or the share of a region's
    yearly rainfall that falls in one month.

    Its shape is controlled by two positive parameters. Roughly, ``shape1``
    controls the behavior near 0 and ``shape2`` the behavior near 1: raising
    ``shape1`` pushes the values toward 1, raising ``shape2`` pushes them
    toward 0, and ``shape1 = shape2 = 1`` leaves every value in [0, 1]
    equally likely.

    The Kumaraswamy takes almost the same range of shapes as the ``Beta``
    -- bell-shaped, U-shaped, J-shaped, or flat -- but its cdf and quantile
    function are simple formulas rather than integrals that must be
    evaluated numerically. That is its practical appeal: values are easy to
    simulate by hand, and probabilities are easy to compute with a
    calculator.

    Parameters
    ----------
    shape1 : float
        First shape parameter. Must be positive. Larger values shift the
        distribution toward 1.
    shape2 : float
        Second shape parameter. Must be positive. Larger values shift the
        distribution toward 0.

    Attributes
    ----------
    shape1 : float
        First shape parameter.
    shape2 : float
        Second shape parameter.

    Notes
    -----
    The cumulative distribution function is
    ``cdf(x) = 1 - (1 - x ** shape1) ** shape2`` on [0, 1], and inverting it
    gives the quantile function
    ``quantile(q) = (1 - (1 - q) ** (1 / shape2)) ** (1 / shape1)``. So if
    ``U`` is ``Uniform(0, 1)``, then
    ``(1 - (1 - U) ** (1 / shape2)) ** (1 / shape1)`` is
    ``Kumaraswamy(shape1, shape2)`` -- the inverse-cdf construction, which is
    how this distribution is sampled.

    Two special cases are exactly beta distributions, and only these two:

    - ``shape2 = 1`` gives ``Beta(shape1, 1)``, because both have cdf
      ``x ** shape1``.
    - ``shape1 = 1`` gives ``Beta(1, shape2)``, because both have cdf
      ``1 - (1 - x) ** shape2``.

    In particular ``Kumaraswamy(1, 1)`` is ``Uniform(0, 1)``. For every
    other pair the two families are close but not equal: a ``Kumaraswamy``
    can be matched to a ``Beta`` in mean and variance and still differ in
    its tails. The general link runs the other way -- if ``X`` is
    ``Kumaraswamy(shape1, shape2)``, then ``X ** shape1`` is
    ``Beta(1, shape2)``.

    The moments are exact:
    ``E[X ** n] = shape2 * B(1 + n / shape1, shape2)``, where ``B`` is the
    beta function.

    Unlike the other distributions in this package, this one is not in
    ``scipy.stats``; its density, cdf, quantile function and moments are
    implemented directly from the formulas above.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Kumaraswamy(shape1=2, shape2=2)
    >>> round(float(X.mean()), 4)  # shape2 * B(1 + 1 / shape1, shape2)
    0.5333
    >>> round(float(X.median()), 4)  # (1 - 2 ** (-1 / shape2)) ** (1 / shape1)
    0.5412
    >>> round(float(X.cdf(0.5)), 4)  # 1 - (1 - 0.5 ** 2) ** 2
    0.4375
    >>> float(Kumaraswamy(shape1=1, shape2=1).pdf(0.3))  # this is Uniform(0, 1)
    1.0
    >>> X.draw()  # doctest: +SKIP
    0.6118
    """

    def __init__(self, shape1, shape2):
        """Initialize a Kumaraswamy distribution.

        Raises
        ------
        Exception
            If ``shape1`` or ``shape2`` is not a positive number.
        """
        _validate(
            (
                not isinstance(shape1, numbers.Real) or shape1 <= 0,
                "shape1 must be a positive number",
            ),
            (
                not isinstance(shape2, numbers.Real) or shape2 <= 0,
                "shape2 must be a positive number",
            ),
        )
        self.shape1 = shape1
        self.shape2 = shape2

        # The two shape parameters pass straight through to the generator
        # defined just above, whose scipy shape names are `a` and `b`.
        params = {"a": shape1, "b": shape2}
        super().__init__(params, _kumaraswamy, False)


class StudentT(Distribution):
    """Probability space for Student's t-distribution.

    A bell-shaped distribution similar to the normal but with heavier
    tails. Commonly used in statistical inference when the sample size
    is small. As degrees of freedom increase, it approaches the normal
    distribution.

    With a nonzero ``noncentrality``, this is the **noncentral**
    t-distribution: the (asymmetric) distribution of the t-statistic when
    the true effect is not zero. It is the distribution needed for power
    and effect-size analysis. ``noncentrality = 0`` (the default) is the
    ordinary central t-distribution.

    Parameters
    ----------
    df : int or float
        Degrees of freedom. Must be positive.
    noncentrality : float, optional
        Noncentrality parameter (often written delta). Default is 0, which
        gives the ordinary central t-distribution. A nonzero value shifts
        the distribution and makes it asymmetric.

    Attributes
    ----------
    df : int or float
        Degrees of freedom. Must be positive.
    noncentrality : float
        Noncentrality parameter.

    Notes
    -----
    For the central case (``noncentrality = 0``) the mean is undefined for
    ``df = 1`` (the Cauchy case); ``mean()``, ``var()``, and ``sd()``
    return ``nan`` there.

    Examples
    --------
    >>> from symbulate import *
    >>> X = StudentT(df=10)
    >>> float(X.mean())
    0.0
    >>> Y = StudentT(df=10, noncentrality=2)  # noncentral t, mean > 0
    >>> round(float(Y.mean()), 4)
    2.1674
    >>> X.draw()  # doctest: +SKIP
    0.312
    """

    def __init__(self, df, noncentrality=0):
        """Initialize a Student's t-distribution.

        Raises
        ------
        Exception
            If ``df`` is not a positive number, or ``noncentrality`` is not
            a number.
        """
        _validate(
            (
                not isinstance(df, numbers.Real) or df <= 0,
                "df must be a positive number",
            ),
            (
                not isinstance(noncentrality, numbers.Real),
                "noncentrality must be a number",
            ),
        )
        self.df = df
        self.noncentrality = noncentrality

        # noncentrality = 0 is the ordinary central t: use scipy.stats.t so
        # behavior is bit-for-bit unchanged. A nonzero value is the noncentral
        # t (scipy.stats.nct), which reduces to the central t at nc = 0.
        if noncentrality == 0:
            params = {"df": df}
            super().__init__(params, stats.t, False)
            if df == 1:
                # Central t with df = 1 is Cauchy: undefined moments.
                self.mean = lambda: float("nan")
                self.sd = lambda: float("nan")
                self.var = lambda: float("nan")
        else:
            params = {"df": df, "nc": noncentrality}
            super().__init__(params, stats.nct, False)


class SkewT(Distribution):
    """Probability space for a skew-t distribution.

    A heavy-tailed, asymmetric bell curve: it takes Student's t-distribution
    and lets the two sides of the curve have different weights, so one tail
    can be longer than the other. This makes it the t-distribution's skewed
    cousin, just as the skew-normal is the normal's -- and it is a popular
    model in finance and robust statistics, where returns and errors are
    both fat-tailed and lopsided.

    This is the Jones and Faddy skew-t, controlled by two positive shape
    parameters. When they are equal the distribution is symmetric and equals
    an ordinary Student's t-distribution with ``shape1 + shape2`` degrees of
    freedom. When ``shape1 > shape2`` the longer tail is on the right (right,
    or positive, skew); when ``shape1 < shape2`` it is on the left. Larger
    values of both make the tails lighter, approaching a normal curve.

    Parameters
    ----------
    shape1 : float
        First shape parameter. Must be positive. Controls the weight of the
        right side; making it larger than ``shape2`` skews the distribution
        to the right.
    shape2 : float
        Second shape parameter. Must be positive. Controls the weight of the
        left side; making it larger than ``shape1`` skews the distribution
        to the left.
    loc : float, optional
        Location (shift) parameter. Default is 0.
    scale : float, optional
        Scale parameter. Must be positive. Default is 1.

    Attributes
    ----------
    shape1 : float
        First shape parameter.
    shape2 : float
        Second shape parameter.
    loc : float
        Location (shift) parameter.
    scale : float
        Scale parameter.

    Notes
    -----
    With ``shape1 == shape2 == df / 2`` the distribution is the symmetric
    ``StudentT(df)`` (before shifting and scaling), so a symmetric skew-t
    with ``df`` degrees of freedom uses ``shape1 = shape2 = df / 2``. The
    tails behave like a t-distribution, so low shape values give an
    undefined mean or variance just as a low-``df`` t does.

    Examples
    --------
    >>> from symbulate import *
    >>> X = SkewT(shape1=5, shape2=2)
    >>> round(float(X.median()), 4)
    1.413
    >>> round(float(X.mean()), 4)
    1.7046
    >>> Y = SkewT(shape1=3, shape2=3)  # symmetric: a Student t with df = 6
    >>> round(float(Y.pdf(0)), 4)
    0.3827
    >>> X.draw()  # doctest: +SKIP
    0.87
    """

    def __init__(self, shape1, shape2, loc=0.0, scale=1.0):
        """Initialize a skew-t distribution.

        Raises
        ------
        Exception
            If ``shape1``, ``shape2``, or ``scale`` is not a positive
            number, or if ``loc`` is not a number.
        """
        _validate(
            (
                not isinstance(shape1, numbers.Real) or shape1 <= 0,
                "shape1 must be a positive number",
            ),
            (
                not isinstance(shape2, numbers.Real) or shape2 <= 0,
                "shape2 must be a positive number",
            ),
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape1 = shape1
        self.shape2 = shape2
        self.loc = loc
        self.scale = scale

        # scipy's jf_skew_t takes the two Jones-Faddy shapes as a and b; our
        # shape1 maps to a (right-side weight) and shape2 to b (left-side
        # weight), so shape1 > shape2 gives right skew.
        params = {"a": shape1, "b": shape2, "loc": loc, "scale": scale}
        super().__init__(params, stats.jf_skew_t, False)


class ChiSquare(Distribution):
    """Probability space for a chi-square distribution.

    Arises as the sum of squares of independent standard normal random
    variables. Commonly used in hypothesis testing and confidence
    intervals for variance.

    With a nonzero ``noncentrality``, this is the **noncentral**
    chi-square distribution: the sum of squares of normals with nonzero
    means. It is the distribution the test statistic follows when the null
    hypothesis is false, so it is needed for power analysis.
    ``noncentrality = 0`` (the default) is the ordinary chi-square.

    Parameters
    ----------
    df : int
        Degrees of freedom. Must be a positive integer.
    noncentrality : float, optional
        Noncentrality parameter (often written lambda). Must be
        nonnegative. Default is 0, which gives the ordinary (central)
        chi-square distribution.

    Attributes
    ----------
    df : int
        Degrees of freedom. Must be a positive integer.
    noncentrality : float
        Noncentrality parameter.

    Examples
    --------
    >>> from symbulate import *
    >>> X = ChiSquare(df=4)
    >>> float(X.mean())
    4.0
    >>> float(X.sd())
    2.8284271247461903
    >>> Y = ChiSquare(df=4, noncentrality=3)  # noncentral: mean = df + nc
    >>> float(Y.mean())
    7.0
    >>> X.draw()  # doctest: +SKIP
    3.14
    """

    def __init__(self, df, noncentrality=0):
        """Initialize a chi-square distribution.

        Raises
        ------
        Exception
            If ``df`` is not a positive integer, or ``noncentrality`` is
            not a nonnegative number.
        """
        _validate(
            (
                not isinstance(df, numbers.Integral) or df <= 0,
                "df must be a positive integer",
            ),
            (
                not isinstance(noncentrality, numbers.Real) or noncentrality < 0,
                "noncentrality must be a nonnegative number",
            ),
        )
        self.df = df
        self.noncentrality = noncentrality

        # noncentrality = 0 is the ordinary central chi-square: use
        # scipy.stats.chi2 so behavior is bit-for-bit unchanged. A positive
        # value is the noncentral chi-square (scipy.stats.ncx2), which reduces
        # to the central chi-square at nc = 0.
        if noncentrality == 0:
            params = {"df": df}
            super().__init__(params, stats.chi2, False)
        else:
            params = {"df": df, "nc": noncentrality}
            super().__init__(params, stats.ncx2, False)


class F(Distribution):
    """Probability space for an F-distribution.

    Arises as the ratio of two chi-square random variables divided by
    their degrees of freedom. Commonly used in analysis of variance
    (ANOVA) to compare group variances.

    With a nonzero ``noncentrality``, this is the **noncentral**
    F-distribution: the distribution the ANOVA F-statistic follows when
    the group means genuinely differ, so it is needed for ANOVA power
    analysis. ``noncentrality = 0`` (the default) is the ordinary
    (central) F-distribution.

    Parameters
    ----------
    dfN : int or float
        Degrees of freedom for the numerator. Must be positive.
    dfD : int or float
        Degrees of freedom for the denominator. Must be positive.
    noncentrality : float, optional
        Noncentrality parameter (often written lambda), carried by the
        numerator. Must be nonnegative. Default is 0, which gives the
        ordinary (central) F-distribution.

    Attributes
    ----------
    dfN : int or float
        Degrees of freedom for the numerator.
    dfD : int or float
        Degrees of freedom for the denominator.
    noncentrality : float
        Noncentrality parameter.

    Examples
    --------
    >>> from symbulate import *
    >>> X = F(dfN=5, dfD=10)
    >>> float(X.mean())
    1.25
    >>> Y = F(dfN=5, dfD=10, noncentrality=4)  # noncentral: larger mean
    >>> round(float(Y.mean()), 4)
    2.25
    >>> X.draw()  # doctest: +SKIP
    0.85
    """

    def __init__(self, dfN, dfD, noncentrality=0):
        """Initialize an F-distribution.

        Raises
        ------
        Exception
            If ``dfN`` or ``dfD`` is not a positive number, or
            ``noncentrality`` is not a nonnegative number.
        """

        _validate(
            (
                not isinstance(dfN, numbers.Real) or dfN <= 0,
                "dfN must be a positive number",
            ),
            (
                not isinstance(dfD, numbers.Real) or dfD <= 0,
                "dfD must be a positive number",
            ),
            (
                not isinstance(noncentrality, numbers.Real) or noncentrality < 0,
                "noncentrality must be a nonnegative number",
            ),
        )
        self.dfN = dfN
        self.dfD = dfD
        self.noncentrality = noncentrality

        # noncentrality = 0 is the ordinary central F: use scipy.stats.f so
        # behavior is bit-for-bit unchanged. A positive value is the noncentral
        # F (scipy.stats.ncf), which reduces to the central F at nc = 0.
        if noncentrality == 0:
            params = {"dfn": dfN, "dfd": dfD}
            super().__init__(params, stats.f, False)
        else:
            params = {"dfn": dfN, "dfd": dfD, "nc": noncentrality}
            super().__init__(params, stats.ncf, False)


class Hotelling(Distribution):
    """Probability space for Hotelling's T-squared distribution.

    The multivariate generalization of the squared t-statistic. When you
    test whether the mean vector of ``dim`` correlated measurements differs
    from a hypothesized value, the T-squared statistic collapses that whole
    vector into a single number measuring how far the sample mean sits from
    the hypothesis. This is the distribution that number follows when the
    hypothesis is true, and it is the workhorse of one- and two-sample mean
    tests in multivariate statistics courses -- the multivariate cousin of
    :class:`MultivariateT`.

    Although it summarizes multivariate data, the distribution itself is
    **univariate**: each draw is a single non-negative number, not a vector.
    It is exactly a rescaled :class:`F` distribution,

    ``Hotelling(dim, df) == (df * dim / (df - dim + 1)) * F(dim, df - dim + 1)``,

    so it is drawn and plotted like any other one-dimensional continuous
    distribution.

    Parameters
    ----------
    dim : int
        Number of variables (the dimension ``p`` of the data). Must be a
        positive integer.
    df : int or float
        Degrees of freedom (often the sample size minus one). Must be
        greater than ``dim - 1`` so the equivalent F-distribution is
        defined.

    Attributes
    ----------
    dim : int
        Number of variables.
    df : int or float
        Degrees of freedom.
    scale : float
        The multiplier ``df * dim / (df - dim + 1)`` applied to the
        underlying F-distribution.

    Notes
    -----
    With ``dim = 1`` the multiplier is 1 and the distribution reduces to
    ``F(1, df)``, which is the square of a ``StudentT(df)``: this is the
    familiar fact that Hotelling's T-squared generalizes the squared
    one-sample t-statistic.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Hotelling(dim=3, df=10)
    >>> float(X.mean())
    5.0
    >>> round(float(X.median()), 4)
    3.2251
    >>> X.draw()  # doctest: +SKIP
    2.71

    See Also
    --------
    F : The distribution Hotelling's T-squared rescales.
    MultivariateT : The vector-valued multivariate cousin.
    """

    def __init__(self, dim, df):
        """Initialize a Hotelling's T-squared distribution.

        Raises
        ------
        Exception
            If ``dim`` is not a positive integer, or ``df`` is not a number
            greater than ``dim - 1``.
        """
        _validate(
            (
                not isinstance(dim, numbers.Integral) or dim < 1,
                "dim must be a positive integer",
            ),
            (
                not isinstance(df, numbers.Real)
                or (isinstance(dim, numbers.Integral) and df <= dim - 1),
                "df must be a number greater than dim - 1",
            ),
        )
        self.dim = dim
        self.df = df

        # Hotelling's T-squared is a rescaled F: T^2 = c * F(dim, df - dim + 1)
        # with c = df * dim / (df - dim + 1). scipy's f takes that multiplier
        # as its scale, so all methods come from the base class.
        self.scale = df * dim / (df - dim + 1)
        params = {"dfn": dim, "dfd": df - dim + 1, "scale": self.scale}
        super().__init__(params, stats.f, False)


class Cauchy(Distribution):
    """Probability space for a Cauchy distribution.

    A heavy-tailed symmetric distribution centered at ``loc``. The Cauchy
    distribution has no finite mean or variance — it is a classic example
    where the law of large numbers does not apply.

    Parameters
    ----------
    loc : float, optional
        Location parameter (center of the distribution). Default is 0.
    scale : float, optional
        Scale parameter (controls the spread). Must be a positive number.
        Default is 1.

    Attributes
    ----------
    loc : float
        Location parameter (center of the distribution). Default is 0.
    scale : float
        Scale parameter (controls the spread). Default is 1.

    Notes
    -----
    The Cauchy distribution has no finite moments: ``mean()`` and ``var()``
    are undefined (return ``nan``).

    Examples
    --------
    >>> from symbulate import *
    >>> X = Cauchy(loc=0, scale=1)
    >>> float(X.pdf(0))
    0.3183098861837907
    >>> X.draw()  # doctest: +SKIP
    -2.31
    """

    def __init__(self, loc=0, scale=1):
        """Initialize a Cauchy distribution.

        Raises
        ------
        Exception
            If ``loc`` is not a number, or ``scale`` is not a positive number.
        """
        _validate(
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.loc = loc
        self.scale = scale

        params = {"loc": loc, "scale": scale}

        super().__init__(params, stats.cauchy, False)

    def draw(self):
        """Draw a single random sample from the Cauchy distribution.

        Returns
        -------
        Scalar
            One random value drawn from the Cauchy distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> Cauchy().draw()  # doctest: +SKIP
        1.48
        """
        # Wrapped in Scalar to match the base class and every other `draw`;
        # `Float` subclasses `float`, so this stays usable anywhere the plain
        # float was.
        return Scalar(self.loc + (self.scale * rng.standard_cauchy()))


class LogNormal(Distribution):
    """Probability space for a log-normal distribution.

    If ``Y`` has a log-normal distribution with parameters ``mu`` and
    ``sigma``, then ``log(Y)`` follows a normal distribution with mean
    ``mu`` and standard deviation ``sigma``. Often used to model positive,
    right-skewed quantities such as income or stock prices.

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal distribution. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying normal distribution. Must be
        a non-negative number (``sigma=0`` gives a point mass at
        ``exp(mu)``). Default is 1.0.

    Attributes
    ----------
    norm_mean : float
        Mean of the underlying normal distribution (μ).
    norm_sd : float
        Standard deviation of the underlying normal distribution (σ).
    s : float
        Shape parameter passed to scipy (equal to ``sigma``).

    Examples
    --------
    >>> from symbulate import *
    >>> X = LogNormal(mu=0, sigma=1)
    >>> float(X.mean())
    1.6487212707001282
    >>> X.draw()  # doctest: +SKIP
    0.94
    """

    def __init__(self, mu=0.0, sigma=1.0):
        """Initialize a log-normal distribution.

        Raises
        ------
        Exception
            If ``mu`` is not a number, or ``sigma`` is not a non-negative
            number.
        """

        _validate(
            (not isinstance(mu, numbers.Real), "mu must be a number"),
            (
                not isinstance(sigma, numbers.Real) or sigma < 0,
                "sigma must be a non-negative number",
            ),
        )

        self.norm_mean = mu

        if sigma == 0:
            # A point mass at exp(mu): there is no scipy object to delegate to,
            # so every method this branch needs is supplied by hand. They are
            # written to accept an array as well as a single number, because
            # `plot` evaluates them on a grid of x-values -- passing an array
            # to a scalar-only `pdf` used to raise a bare TypeError.
            _value = np.exp(mu)

            def _scalar_or_array(values, x):
                """Return a plain float for a single x, an array for an array."""
                return float(values) if np.ndim(x) == 0 else values

            self.s = 0
            self.norm_sd = 0
            self.discrete = False
            self.params = {"s": 0, "scale": _value}
            self.pdf = lambda x: _scalar_or_array(
                np.where(np.asarray(x, dtype=float) == _value, 1.0, 0.0), x
            )
            self.cdf = lambda x: _scalar_or_array(
                np.where(np.asarray(x, dtype=float) < _value, 0.0, 1.0), x
            )
            # Every quantile of a point mass is the point itself; a probability
            # outside [0, 1] gives nan, matching scipy's `ppf`.
            self.quantile = lambda q: _scalar_or_array(
                np.where(
                    (np.asarray(q, dtype=float) >= 0)
                    & (np.asarray(q, dtype=float) <= 1),
                    _value,
                    np.nan,
                ),
                q,
            )
            self.mean = lambda: _value
            self.var = lambda: 0.0
            self.sd = lambda: 0.0
            self.median = lambda: _value

            # `__pow__` draws through `sim_func`, so supply one with the same
            # calling convention scipy's `rvs` has (the `**self.params` keywords
            # plus `size` and `random_state`). Without it, `LogNormal(mu, 0) ** n`
            # failed with a bare AttributeError.
            def _degenerate_rvs(s=None, scale=None, size=None, random_state=None):
                if size is None:
                    return _value
                return np.full(size, _value)

            self.sim_func = _degenerate_rvs
            # `_xlim`, not the setter: this branch never reaches
            # `Distribution.__init__`, so it is standing in for the computed
            # default rather than being a window someone chose by hand.
            self._xlim = (0, _value + 1)
            ProbabilitySpace.__init__(self, lambda: Scalar(_value))
            return
        else:
            self.s = sigma
            self.norm_sd = sigma

        params = {"s": self.s, "scale": np.exp(mu)}
        super().__init__(params, stats.lognorm, False)


class Pareto(Distribution):
    """Probability space for a Pareto distribution.

    A heavy-tailed distribution often used to model phenomena where a
    small fraction of items account for a large share of the effect
    (the "80/20 rule"). All values are at least ``scale``.

    Parameters
    ----------
    shape : float, optional
        Shape parameter (tail index). Must be positive. Default is 1.0.
    scale : float, optional
        Minimum possible value (lower bound of the support).
        Must be positive. Default is 1.0.

    Attributes
    ----------
    shape : float
        Shape parameter (tail index). Larger values give thinner tails.
    scale : float
        Minimum possible value (lower bound of the support).

    Notes
    -----
    The mean is finite only when ``shape > 1``; for ``shape <= 1`` it
    diverges to ``inf``. The variance is finite only when ``shape > 2``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Pareto(shape=2, scale=1)
    >>> float(X.mean())
    2.0
    >>> X.draw()  # doctest: +SKIP
    1.34
    """

    def __init__(self, shape=1.0, scale=1.0):
        """Initialize a Pareto distribution.

        Raises
        ------
        Exception
            If ``shape`` or ``scale`` is not a positive number.
        """

        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape = shape
        self.scale = scale

        params = {"b": self.shape, "scale": self.scale}
        super().__init__(params, stats.pareto, False)

    def draw(self):
        """Draw a single random sample from the Pareto distribution.

        Returns
        -------
        Scalar
            One random value drawn from the Pareto distribution.
            Always greater than or equal to ``scale``.

        Examples
        --------
        >>> from symbulate import *
        >>> Pareto(shape=2, scale=1).draw()  # doctest: +SKIP
        1.42
        """

        # Numpy's Pareto is Lomax distribution, or Type II Pareto
        # but we want the more standard parametrization.
        # Wrapped in Scalar to match the base class and every other `draw`;
        # `Float` subclasses `float`, so this stays usable anywhere the plain
        # float was.
        return Scalar(self.scale * (1 + rng.pareto(self.shape)))


class Burr(Distribution):
    """Probability space for a Burr (Burr Type XII) distribution.

    A flexible, heavy-tailed continuous distribution on [0, infinity). In
    actuarial science it is a standard model for claim severity -- the size
    of an insurance loss -- in Klugman's *Loss Models*, where its two shape
    parameters let it fit both the body and the heavy right tail of loss
    data. Its two special cases explain the two shapes: with ``shape2 = 1``
    it is the log-logistic (Fisk) distribution, and with ``shape1 = 1`` it
    is a Pareto (Type II / Lomax) distribution, so a ``Burr`` generalizes
    both.

    Parameters
    ----------
    shape1 : float
        First shape parameter. Must be positive. Controls the shape of the
        body near the origin.
    shape2 : float
        Second shape parameter. Must be positive. The right tail decays like
        a power law with index ``shape1 * shape2``, so a smaller ``shape2``
        gives a heavier tail.
    scale : float, optional
        Scale parameter. Must be positive. Default is 1.

    Attributes
    ----------
    shape1 : float
        First shape parameter (the body shape).
    shape2 : float
        Second shape parameter (the tail exponent).
    scale : float
        Scale parameter.

    Notes
    -----
    The mean is finite only when ``shape1 * shape2 > 1``; more generally the
    ``k``-th moment is finite only when ``shape1 * shape2 > k``, so a
    heavy-tailed ``Burr`` (small ``shape1 * shape2``) can have an infinite
    mean or variance.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Burr(shape1=3, shape2=2)
    >>> round(float(X.median()), 4)
    0.7454
    >>> round(float(X.mean()), 4)
    0.8061
    >>> round(float(X.pdf(1)), 4)
    0.75
    >>> X.draw()  # doctest: +SKIP
    0.62
    """

    def __init__(self, shape1, shape2, scale=1.0):
        """Initialize a Burr (Type XII) distribution.

        Raises
        ------
        Exception
            If ``shape1``, ``shape2``, or ``scale`` is not a positive
            number.
        """
        _validate(
            (
                not isinstance(shape1, numbers.Real) or shape1 <= 0,
                "shape1 must be a positive number",
            ),
            (
                not isinstance(shape2, numbers.Real) or shape2 <= 0,
                "shape2 must be a positive number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape1 = shape1
        self.shape2 = shape2
        self.scale = scale

        # scipy's burr12 takes c and d as the two shapes and scale the scale;
        # our shape1 maps to scipy's c (body shape) and shape2 to scipy's d
        # (tail exponent). loc stays at its 0 default so support starts at 0.
        params = {"c": shape1, "d": shape2, "scale": scale}
        super().__init__(params, stats.burr12, False)


class Lomax(Distribution):
    """Probability space for a Lomax (Pareto Type II) distribution.

    A heavy-tailed continuous distribution on [0, infinity). It is the
    Pareto distribution shifted to start at 0 instead of at its scale, which
    is why it is also called the Pareto Type II distribution. Like the
    ``Pareto``, it is a common model for claim sizes and other quantities
    with a heavy right tail, and it is exactly the ``shape1 = 1`` special
    case of the ``Burr`` distribution.

    Parameters
    ----------
    shape : float, optional
        Shape parameter (the tail index). Must be positive. The right tail
        decays like a power law with index ``shape``, so a smaller ``shape``
        gives a heavier tail. Default is 1.
    scale : float, optional
        Scale parameter. Must be positive. Default is 1.

    Attributes
    ----------
    shape : float
        Shape parameter (the tail index).
    scale : float
        Scale parameter.

    Notes
    -----
    The mean is finite only when ``shape > 1``, and the variance only when
    ``shape > 2``. Adding the scale to a ``Lomax(shape, scale)`` gives a
    ``Pareto(shape, scale)`` (Type I): the Lomax is the same power law with
    its support shifted from ``[scale, inf)`` down to ``[0, inf)``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Lomax(shape=3, scale=1)
    >>> float(X.mean())
    0.5
    >>> round(float(X.sd()), 4)
    0.866
    >>> float(X.pdf(0))
    3.0
    >>> X.draw()  # doctest: +SKIP
    0.41
    """

    def __init__(self, shape=1.0, scale=1.0):
        """Initialize a Lomax (Pareto Type II) distribution.

        Raises
        ------
        Exception
            If ``shape`` or ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape = shape
        self.scale = scale

        # scipy's lomax takes c as the shape and scale the scale; loc stays at
        # its 0 default so the support starts at 0 (the Pareto Type II form).
        params = {"c": shape, "scale": scale}
        super().__init__(params, stats.lomax, False)


class Rayleigh(Distribution):
    """Probability space for a Rayleigh distribution.

    Arises when computing the magnitude of a two-dimensional vector
    whose components are independent, identically distributed normal
    random variables. Often used in signal processing.

    Parameters
    ----------
    scale : float, optional
        Scale parameter of the distribution. Must be positive.
        Default is 1.0.

    Attributes
    ----------
    scale : float
        Scale parameter of the distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Rayleigh()
    >>> round(float(X.mean()), 4)
    1.2533
    >>> X.draw()  # doctest: +SKIP
    0.88
    >>> round(float(Rayleigh(scale=2).mean()), 4)
    2.5066
    """

    def __init__(self, scale=1.0):
        """Initialize a Rayleigh distribution.

        Raises
        ------
        Exception
            If ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.scale = scale

        params = {"scale": scale}
        super().__init__(params, stats.rayleigh, False)


class HalfNormal(Distribution):
    """Probability space for a half-normal distribution.

    The distribution of ``|X|`` where ``X ~ Normal(0, scale)`` -- folding a
    mean-zero normal distribution onto the non-negative half-line. A
    common weakly-informative prior for a standard-deviation or other
    scale parameter in Bayesian modeling (the default choice in Stan and
    PyMC). Fixed at a fold point of 0; a distribution built the same way
    around a nonzero center is a *Folded Normal*, a separate distribution.

    Parameters
    ----------
    scale : float, optional
        Standard deviation of the underlying (unfolded) normal
        distribution. Must be positive. Default is 1.0.

    Attributes
    ----------
    scale : float
        Standard deviation of the underlying normal distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = HalfNormal(scale=1)
    >>> round(float(X.mean()), 4)
    0.7979
    >>> round(float(X.pdf(0)), 4)
    0.7979
    >>> X.draw()  # doctest: +SKIP
    0.42
    """

    def __init__(self, scale=1.0):
        """Initialize a half-normal distribution.

        Raises
        ------
        Exception
            If ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.scale = scale

        params = {"scale": scale}
        super().__init__(params, stats.halfnorm, False)


class HalfCauchy(Distribution):
    """Probability space for a half-Cauchy distribution.

    The distribution of ``|X|`` where ``X ~ Cauchy(0, scale)`` -- folding a
    Cauchy distribution centered at 0 onto the non-negative half-line. Like
    :class:`HalfNormal` it is used as a prior for a standard-deviation or
    other scale parameter in Bayesian modeling, but its much heavier tail
    makes it the more permissive of the two: it keeps meaningful
    probability far from 0, so it is the usual recommendation for the
    group-level standard deviation in a hierarchical model. Fixed at a fold
    point of 0; a distribution built the same way around a nonzero center
    is a *folded Cauchy*, a separate distribution.

    Parameters
    ----------
    scale : float, optional
        Scale parameter of the underlying (unfolded) Cauchy distribution.
        Controls the spread, and equals the median. Must be positive.
        Default is 1.0.

    Attributes
    ----------
    scale : float
        Scale parameter of the underlying Cauchy distribution.

    Notes
    -----
    The half-Cauchy inherits the Cauchy's heavy tail, so it has no finite
    moments: ``mean()``, ``var()``, and ``sd()`` all return ``inf`` (the
    defining integrals diverge). Use ``median()``, which is exactly
    ``scale``, to describe its center instead.

    That same heavy tail makes the default plotting window wide -- covering
    most of the probability genuinely requires reaching far out along the
    tail -- so the density can look like a spike at 0. Pass an explicit
    ``xlim=(0, high)`` to :meth:`plot` to inspect the bulk of the
    distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = HalfCauchy(scale=1)
    >>> round(float(X.median()), 4)
    1.0
    >>> round(float(X.pdf(0)), 4)
    0.6366
    >>> float(X.mean())
    inf
    >>> X.draw()  # doctest: +SKIP
    0.42
    """

    def __init__(self, scale=1.0):
        """Initialize a half-Cauchy distribution.

        Raises
        ------
        Exception
            If ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.scale = scale

        params = {"scale": scale}
        super().__init__(params, stats.halfcauchy, False)


class Weibull(Distribution):
    """Probability space for a Weibull distribution.

    A continuous distribution on [0, infinity) widely used to model
    lifetimes and time to failure. The shape parameter controls how the
    failure rate changes over time: shape < 1 means it decreases, shape
    = 1 means it is constant (and the Weibull reduces to the exponential),
    and shape > 1 means it increases.

    Parameters
    ----------
    shape : float
        Shape parameter k. Must be positive.
    scale : float, optional
        Scale parameter lambda. Must be positive. Default is 1.0.

    Attributes
    ----------
    shape : float
        Shape parameter k.
    scale : float
        Scale parameter lambda.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Weibull(shape=1.5, scale=2)
    >>> round(float(X.mean()), 4)
    1.8055
    >>> X.draw()  # doctest: +SKIP
    2.02
    """

    def __init__(self, shape, scale=1.0):
        """Initialize a Weibull distribution.

        Raises
        ------
        Exception
            If ``shape`` or ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape = shape
        self.scale = scale
        # scipy's Weibull is weibull_min, with c the shape and scale the
        # scale; loc stays at its 0 default so the support starts at 0.
        params = {"c": shape, "scale": scale}
        super().__init__(params, stats.weibull_min, False)


class Logistic(Distribution):
    """Probability space for a logistic distribution.

    A continuous distribution on all real numbers, symmetric and
    bell-shaped like the normal but with heavier tails. Its cumulative
    distribution function is the logistic (S-shaped) function
    ``1 / (1 + exp(-(x - loc) / scale))`` -- the same curve used as the
    link function in logistic regression, which is why this distribution
    underlies that model.

    Parameters
    ----------
    loc : float, optional
        Location parameter (the center, which is also the mean and
        median). Default is 0.
    scale : float, optional
        Scale parameter (controls the spread). Must be a positive number.
        Default is 1.

    Attributes
    ----------
    loc : float
        Location parameter (the center of the distribution).
    scale : float
        Scale parameter (controls the spread).

    Examples
    --------
    >>> from symbulate import *
    >>> X = Logistic(loc=0, scale=1)
    >>> float(X.mean())
    0.0
    >>> float(X.cdf(0))
    0.5
    >>> float(X.pdf(0))
    0.25
    >>> X.draw()  # doctest: +SKIP
    -0.83
    """

    def __init__(self, loc=0, scale=1):
        """Initialize a logistic distribution.

        Raises
        ------
        Exception
            If ``loc`` is not a number, or ``scale`` is not a positive
            number.
        """
        _validate(
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.loc = loc
        self.scale = scale

        params = {"loc": loc, "scale": scale}
        # Symmetric and unbounded on both sides, so the default window is a
        # quantile cut at each end, already centered.
        super().__init__(params, stats.logistic, False)


class Gompertz(Distribution):
    """Probability space for a Gompertz distribution.

    A continuous distribution on [0, infinity) whose hazard rate (in
    actuarial terms, the force of mortality) rises exponentially -- the
    Gompertz mortality law. It is a foundational model in actuarial
    science for how the risk of death climbs with age.

    Parameters
    ----------
    shape : float
        Shape parameter. Must be positive. Larger values shift the
        distribution toward smaller values (death comes sooner).
    scale : float, optional
        Scale parameter. Must be positive. Default is 1.0.

    Attributes
    ----------
    shape : float
        Shape parameter.
    scale : float
        Scale parameter.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Gompertz(shape=1.5, scale=2)
    >>> round(float(X.mean()), 4)
    0.8965
    >>> float(X.pdf(0))
    0.75
    >>> X.draw()  # doctest: +SKIP
    0.83
    """

    def __init__(self, shape, scale=1.0):
        """Initialize a Gompertz distribution.

        Raises
        ------
        Exception
            If ``shape`` or ``scale`` is not a positive number.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape = shape
        self.scale = scale
        # scipy's gompertz takes c as the shape and scale the scale; loc
        # stays at its 0 default so the support starts at 0.
        params = {"c": shape, "scale": scale}
        super().__init__(params, stats.gompertz, False)


class _makeham_gen(stats.rv_continuous):
    """scipy generator for the Gompertz-Makeham mortality law.

    Standardized (``scale`` applied by the framework) with Gompertz shape
    ``c`` and constant (Makeham) hazard term ``lam`` on ``[0, inf)``. The
    hazard is ``lam + c * e**x`` -- a Gompertz hazard that rises
    exponentially, plus a flat age-independent term -- so the cumulative
    hazard is ``lam * x + c * (e**x - 1)`` and the survival function is
    ``exp(-(lam * x + c * (e**x - 1)))``. scipy has no Makeham built in,
    so it is defined here; every other case in this module wraps an
    existing scipy distribution directly.
    """

    def _argcheck(self, c, lam):
        return (c > 0) & (lam >= 0)

    def _get_support(self, c, lam):
        return 0.0, np.inf

    def _pdf(self, x, c, lam):
        # pdf = hazard * survival.
        return (lam + c * np.exp(x)) * np.exp(-(lam * x + c * np.expm1(x)))

    def _cdf(self, x, c, lam):
        return -np.expm1(-(lam * x + c * np.expm1(x)))

    def _sf(self, x, c, lam):
        return np.exp(-(lam * x + c * np.expm1(x)))

    def _rvs(self, c, lam, size=None, random_state=None):
        # Competing risks: the Makeham lifetime is the earlier of a
        # Gompertz lifetime (accelerating hazard c * e**x) and an
        # exponential one (constant hazard lam). Independent risks whose
        # hazards add and whose survival functions multiply -- exactly the
        # Makeham law -- so the minimum is an exact, vectorized sampler.
        g = stats.gompertz.rvs(c, size=size, random_state=random_state)
        if np.all(lam == 0):
            return g
        e = random_state.exponential(scale=1.0 / lam, size=size)
        return np.minimum(g, e)


_makeham = _makeham_gen(name="makeham", a=0.0)


class Makeham(Distribution):
    """Probability space for a Makeham (Gompertz-Makeham) distribution.

    A continuous distribution on [0, infinity) used in actuarial science
    to model human mortality. It is the Gompertz law plus a constant,
    age-independent hazard term (the "accident" term): the force of
    mortality is ``(makeham + shape * e**(x / scale)) / scale``. Setting
    ``makeham = 0`` recovers the plain :class:`Gompertz` distribution.

    Note where the ``scale`` divides: it divides the *whole* hazard, so the
    constant term contributes ``makeham / scale``, not ``makeham``. The two
    parameters are therefore on the same footing -- both are measured
    before the age axis is stretched -- but it does mean that at a
    ``scale`` other than 1, ``makeham`` is not itself the constant hazard.

    Parameters
    ----------
    shape : float
        Gompertz shape parameter. Must be positive. Controls how fast the
        force of mortality accelerates with age.
    makeham : float
        Constant (age-independent) hazard term -- the accident term. Must
        be non-negative. ``0`` reduces the distribution to a Gompertz.
    scale : float, optional
        Scale parameter. Must be positive. Stretches the age axis (larger
        values mean longer lifetimes), as in :class:`Gompertz`. Default is
        1.0.

    Attributes
    ----------
    shape : float
        Gompertz shape parameter.
    makeham : float
        Constant (age-independent) hazard term.
    scale : float
        Scale parameter.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Makeham(shape=1.5, makeham=0.3, scale=2)
    >>> round(float(X.mean()), 4)
    0.8099
    >>> X.draw()  # doctest: +SKIP
    0.71
    """

    def __init__(self, shape, makeham, scale=1.0):
        """Initialize a Makeham distribution.

        Raises
        ------
        Exception
            If ``shape`` or ``scale`` is not a positive number, or
            ``makeham`` is not a non-negative number.
        """
        _validate(
            (
                not isinstance(shape, numbers.Real) or shape <= 0,
                "shape must be a positive number",
            ),
            (
                not isinstance(makeham, numbers.Real) or makeham < 0,
                "makeham must be a non-negative number",
            ),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.shape = shape
        self.makeham = makeham
        self.scale = scale
        # The custom generator's shapes are c (Gompertz shape) and lam (the
        # constant term); scale is applied by the framework as for Gompertz.
        params = {"c": shape, "lam": makeham, "scale": scale}
        super().__init__(params, _makeham, False)


class Laplace(Distribution):
    """Probability space for a Laplace (double exponential) distribution.

    A continuous, symmetric distribution on all real numbers, centered at
    ``loc``. It looks like two exponential distributions placed back to
    back: a sharp peak at the center and heavier tails than the normal.
    Widely used as a robust error model and as the prior behind
    L1/LASSO-style Bayesian regularization.

    Parameters
    ----------
    loc : float, optional
        Location parameter (the center, which is also the mean and
        median). Default is 0.
    scale : float, optional
        Scale parameter (controls the spread). Must be a positive number.
        Default is 1.

    Attributes
    ----------
    loc : float
        Location parameter (the center of the distribution).
    scale : float
        Scale parameter (controls the spread).

    Examples
    --------
    >>> from symbulate import *
    >>> X = Laplace(loc=0, scale=1)
    >>> float(X.mean())
    0.0
    >>> float(X.cdf(0))
    0.5
    >>> float(X.pdf(0))
    0.5
    >>> X.draw()  # doctest: +SKIP
    -0.37
    """

    def __init__(self, loc=0, scale=1):
        """Initialize a Laplace distribution.

        Raises
        ------
        Exception
            If ``loc`` is not a number, or ``scale`` is not a positive
            number.
        """
        _validate(
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.loc = loc
        self.scale = scale

        params = {"loc": loc, "scale": scale}
        # Symmetric and unbounded on both sides, so the default window is a
        # quantile cut at each end, already centered.
        super().__init__(params, stats.laplace, False)


class DeMoivre(Distribution):
    """Probability space for a De Moivre survival distribution.

    The simplest survival law in actuarial science: age at death is
    uniformly distributed between 0 and a limiting age ``omega``, so
    every remaining year of life is equally likely and no one survives
    past ``omega``. It is a reparameterized ``Uniform(0, omega)``, named
    for its use as the earliest mortality model.

    Parameters
    ----------
    omega : float
        Limiting age -- the oldest attainable age, beyond which survival
        is impossible. Must be positive.

    Attributes
    ----------
    omega : float
        Limiting age.

    Examples
    --------
    >>> from symbulate import *
    >>> X = DeMoivre(omega=100)
    >>> float(X.mean())
    50.0
    >>> float(X.pdf(40))
    0.01
    >>> X.draw()  # doctest: +SKIP
    37.4
    """

    def __init__(self, omega):
        """Initialize a De Moivre survival distribution.

        Raises
        ------
        Exception
            If ``omega`` is not a positive number.
        """
        _validate(
            (
                not isinstance(omega, numbers.Real) or omega <= 0,
                "omega must be a positive number",
            ),
        )
        self.omega = omega
        # De Moivre's law is Uniform(0, omega): loc 0, scale omega. Bounded
        # on both ends, so xlim shows the full support (no probability trim).
        params = {"loc": 0, "scale": omega}
        super().__init__(params, stats.uniform, False)


class GEV(Distribution):
    """Probability space for a Generalized Extreme Value (GEV) distribution.

    A continuous distribution for the maximum of many observations. A
    single ``shape`` parameter (the extreme-value index, usually written
    xi) unifies the three classical extreme-value laws in one family:

    - ``shape == 0`` -- the **Gumbel** distribution (light-tailed, support
      all real numbers);
    - ``shape > 0`` -- the **Frechet** distribution (heavy right tail,
      bounded below at ``loc - scale / shape``);
    - ``shape < 0`` -- the **reverse-Weibull** distribution (short tail,
      bounded above at ``loc - scale / shape``).

    Parameters
    ----------
    loc : float, optional
        Location parameter (the center). Default is 0.
    scale : float, optional
        Scale parameter (controls the spread). Must be positive.
        Default is 1.
    shape : float, optional
        Shape parameter (the extreme-value index xi). Default is 0, which
        gives the Gumbel distribution.

    Attributes
    ----------
    loc : float
        Location parameter.
    scale : float
        Scale parameter.
    shape : float
        Shape parameter (extreme-value index xi).

    Examples
    --------
    >>> from symbulate import *
    >>> X = GEV(loc=0, scale=1, shape=0)  # shape=0 is the Gumbel distribution
    >>> round(float(X.mean()), 4)
    0.5772
    >>> X.draw()  # doctest: +SKIP
    1.23
    """

    def __init__(self, loc=0, scale=1, shape=0):
        """Initialize a Generalized Extreme Value distribution.

        Raises
        ------
        Exception
            If ``loc`` or ``shape`` is not a number, or ``scale`` is not a
            positive number.
        """
        _validate(
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
            (not isinstance(shape, numbers.Real), "shape must be a number"),
        )
        self.loc = loc
        self.scale = scale
        self.shape = shape
        # scipy's genextreme uses c = -shape: its sign convention is the
        # opposite of the standard extreme-value index xi. So xi = 0 (Gumbel),
        # xi > 0 (Frechet, heavy right tail), xi < 0 (reverse-Weibull, bounded
        # above) map to c = 0, c < 0, c > 0 respectively.
        params = {"c": -shape, "loc": loc, "scale": scale}
        super().__init__(params, stats.genextreme, False)


class Gumbel(GEV):
    """Probability space for a Gumbel (Extreme Value Type I) distribution.

    A continuous distribution on all real numbers for the maximum of many
    observations. It is exactly the ``shape = 0`` case of the
    :class:`GEV` distribution, so this class is a convenience name for
    ``GEV(loc, scale, shape=0)`` -- the Gumbel is still taught and used by
    name even where the general GEV is available. The distribution is
    right-skewed, with a light left tail and a heavier right tail.

    Parameters
    ----------
    loc : float, optional
        Location parameter (the mode). Default is 0.
    scale : float, optional
        Scale parameter (controls the spread). Must be positive.
        Default is 1.

    Attributes
    ----------
    loc : float
        Location parameter (the mode).
    scale : float
        Scale parameter (controls the spread).
    shape : float
        Extreme-value index, fixed at 0 (inherited from :class:`GEV`).

    Notes
    -----
    This is the Gumbel distribution for maxima. The mean is
    ``loc + euler_gamma * scale``, where ``euler_gamma`` is the
    Euler-Mascheroni constant (about 0.5772).

    Examples
    --------
    >>> from symbulate import *
    >>> X = Gumbel(loc=0, scale=1)
    >>> round(float(X.mean()), 4)
    0.5772
    >>> round(float(X.median()), 4)
    0.3665
    >>> round(float(X.pdf(0)), 4)
    0.3679
    >>> X.draw()  # doctest: +SKIP
    0.37
    """

    def __init__(self, loc=0, scale=1):
        """Initialize a Gumbel distribution.

        Raises
        ------
        Exception
            If ``loc`` is not a number, or ``scale`` is not a positive
            number.
        """
        # The Gumbel is the shape=0 GEV; delegate to GEV so all of its
        # validation, scipy wiring, and plotting are reused unchanged.
        super().__init__(loc=loc, scale=scale, shape=0)


class GPD(Distribution):
    """Probability space for a Generalized Pareto Distribution (GPD).

    A continuous distribution for the size of values above a threshold --
    the amount by which a large observation exceeds a high cutoff. A
    single ``shape`` parameter (usually written xi) sets the tail
    behaviour:

    - ``shape == 0`` -- the **exponential** distribution (light tail,
      support ``[loc, inf)``);
    - ``shape > 0`` -- a heavy right tail (support ``[loc, inf)``);
    - ``shape < 0`` -- a short tail bounded above at ``loc - scale / shape``.

    Parameters
    ----------
    loc : float, optional
        Location parameter (the threshold, i.e. the lower bound of the
        support). Default is 0.
    scale : float, optional
        Scale parameter (controls the spread). Must be positive.
        Default is 1.
    shape : float, optional
        Shape parameter (the tail index xi). Default is 0, which gives the
        exponential distribution.

    Attributes
    ----------
    loc : float
        Location parameter (threshold).
    scale : float
        Scale parameter.
    shape : float
        Shape parameter (tail index xi).

    Notes
    -----
    This maps directly onto ``scipy.stats.genpareto``: the package's
    ``shape`` is scipy's ``c`` with the *same* sign (unlike :class:`GEV`,
    whose ``shape`` is the negative of scipy's ``c``), while ``loc`` and
    ``scale`` pass through unchanged.

    The mean is finite only when ``shape < 1``; for ``shape >= 1`` it
    diverges to ``inf``. The variance is finite only when ``shape < 1/2``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = GPD(loc=0, scale=1, shape=0)  # shape=0 is the exponential
    >>> float(X.mean())
    1.0
    >>> X.draw()  # doctest: +SKIP
    0.42
    """

    def __init__(self, loc=0, scale=1, shape=0):
        """Initialize a Generalized Pareto Distribution.

        Raises
        ------
        Exception
            If ``loc`` or ``shape`` is not a number, or ``scale`` is not a
            positive number.
        """
        _validate(
            (not isinstance(loc, numbers.Real), "loc must be a number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
            (not isinstance(shape, numbers.Real), "shape must be a number"),
        )
        self.loc = loc
        self.scale = scale
        self.shape = shape
        # scipy's genpareto shape c matches the standard tail index xi
        # directly (same sign), so shape passes straight through -- unlike
        # GEV, whose shape is the negative of scipy's c.
        params = {"c": shape, "loc": loc, "scale": scale}
        super().__init__(params, stats.genpareto, False)


## Multivariate Distributions


class MultivariateDistribution(Distribution):
    """Base class for multivariate (vector-valued) distributions.

    Multivariate distributions in Symbulate produce a whole vector on each
    draw rather than a single number, so they need a different shape of API
    than the scalar :class:`Distribution`. This class collects the machinery
    that is the same for every one of them, so each specific distribution
    only has to supply the parts that are genuinely its own.

    A subclass must define:

    - ``mean()`` -- the mean vector, returned as a :class:`Vector`.
    - ``cov()`` -- the covariance matrix, returned as a NumPy 2-D array.
    - ``draw()`` -- one random vector.

    and, in its ``__init__``, set ``self.discrete`` and ``self.pdf`` (there is
    no call to the scalar ``Distribution.__init__``).

    This class then derives, once for all subclasses:

    - ``var()`` -- the variances (the diagonal of ``cov()``), as a ``Vector``.
    - ``sd()`` -- the standard deviations (square roots of ``var()``), as a
      ``Vector``.
    - ``corr()`` -- the correlation matrix, as a NumPy 2-D array.
    - ``__pow__`` -- drawing several independent vectors at once.
    - ``plot()`` -- the joint distribution of two of the variables, or a
      matrix of every pair. To take part in this a subclass also supplies
      ``_marginal_1d(i)`` (the distribution of one variable on its own) and
      ``_joint_func(i, j)`` (the joint pdf/pmf of a pair); a subclass that
      defines neither still gets a plot that explains why it can't be
      drawn yet, rather than a broken one.

    Notes
    -----
    Vector-valued summaries (``mean``, ``var``, ``sd``) return a ``Vector``;
    matrix-valued ones (``cov``, ``corr``) return a NumPy 2-D array so they
    can be used directly in further linear-algebra computations.
    """

    def var(self):
        """Return the variance of each component as a Vector.

        Returns
        -------
        Vector
            The diagonal of the covariance matrix -- the variance of each
            component of the distribution.
        """
        return Vector(np.diag(np.asarray(self.cov(), dtype=float)))

    def sd(self):
        """Return the standard deviation of each component as a Vector.

        Returns
        -------
        Vector
            The square root of each component's variance.
        """
        return Vector(np.sqrt(np.diag(np.asarray(self.cov(), dtype=float))))

    def corr(self):
        """Return the correlation matrix.

        Returns
        -------
        numpy.ndarray
            The correlation matrix, obtained by scaling the covariance matrix
            by the outer product of the component standard deviations. Its
            diagonal entries are all 1, except that a row/column for a
            component with zero variance (e.g. ``Multinomial(n=10, p=[1, 0,
            0])`` -- a category with probability 0 never varies) is entirely
            ``nan``: correlation with a component that never varies from its
            mean is undefined, not zero, so ``nan`` is the honest answer
            rather than a number that looks precise but isn't.
        """
        cov = np.asarray(self.cov(), dtype=float)
        sd = np.sqrt(np.diag(cov))
        # A zero-variance component makes the corresponding outer-product
        # entries 0, so this division would otherwise raise a raw
        # "RuntimeWarning: invalid value encountered in divide" with no
        # context -- the result (nan) is correct and now documented above,
        # so the warning is suppressed rather than the calculation changed.
        with np.errstate(invalid="ignore", divide="ignore"):
            return cov / np.outer(sd, sd)

    # Whether the components must add up to a fixed total -- True for the
    # families whose draws are a breakdown of a whole (``Multinomial``
    # counts adding to n, ``Dirichlet`` proportions adding to 1). One
    # component of such a draw is determined by the others, so a
    # k-component distribution varies freely in only k - 1 directions:
    # three categories is the "two variables, one joint plot" case, the
    # same way two variables is for an unconstrained family. See
    # ``_free_dim``.
    _sum_constrained = False

    def _n_components(self):
        """Return how many components each draw has.

        Returns
        -------
        int
            The length of one draw from the distribution -- the number of
            variables it describes.
        """
        return len(self.mean())

    def _free_dim(self):
        """Return how many of the components vary freely.

        For most distributions this is just the number of components. For
        a sum-constrained family (see ``_sum_constrained``) the last
        component is fixed by the others, so a three-category
        ``Multinomial`` or ``Dirichlet`` varies in two directions, not
        three -- which is what makes it the natural single-joint-plot case.

        Returns
        -------
        int
            The number of freely varying components.
        """
        n = self._n_components()
        return n - 1 if self._sum_constrained else n

    def _variable_label(self, i):
        """Return the axis label for component ``i``.

        Spelled out rather than abbreviated, matching the default labels
        ``make_joint_pdf`` and ``make_joint_pmf`` already use, and readable
        by a student who has not been told that ``X`` means a variable.

        Parameters
        ----------
        i : int
            Index of the component, counting from 0.

        Returns
        -------
        str
            The label, numbered from 1 the way the variables are written
            mathematically -- component 0 is ``"Variable 1"``.
        """
        return "Variable %d" % (i + 1)

    def _marginal_1d(self, i):
        """Return the distribution of component ``i`` on its own.

        Each of these distributions has a *closed-form* marginal in a
        family Symbulate already has, so the distribution of one component
        is an exact one-dimensional distribution rather than something
        estimated by simulation: one component of a ``MultivariateNormal``
        is ``Normal``, one count of a ``Multinomial`` is ``Binomial``, one
        proportion of a ``Dirichlet`` is ``Beta``. :meth:`plot` uses these
        both to frame its axes and to draw the diagonal of a pairs matrix.

        Parameters
        ----------
        i : int
            Index of the component, counting from 0.

        Returns
        -------
        Distribution
            The one-dimensional distribution of that component.

        Raises
        ------
        Exception
            If the subclass has not defined its marginals, so plotting is
            not available for it yet.
        """
        raise Exception(
            "Plotting is not available for this distribution yet, because "
            "the distribution of a single one of its variables has not been "
            "worked out in the code. To visualize it, simulate values and "
            "plot one variable (or a pair of variables) at a time."
        )

    def _joint_func(self, i, j):
        """Return the joint pdf/pmf of components ``i`` and ``j``.

        The two-variable counterpart of :meth:`_marginal_1d`, and likewise
        exact rather than simulated: for a ``MultivariateNormal`` or
        ``MultivariateT`` the pair's distribution comes from pulling out
        those two entries of the mean vector and the corresponding 2x2
        block of the covariance matrix; for a ``Multinomial`` or
        ``Dirichlet`` it comes from pooling every other category into a
        single "everything else" category, which gives back a
        three-category distribution of the same family.

        Parameters
        ----------
        i, j : int
            Indices of the two components, counting from 0.

        Returns
        -------
        callable
            A function of two equal-length flat arrays of coordinates,
            returning the joint density (or probability) at each of those
            points as a flat array.

        Raises
        ------
        Exception
            If the subclass has not defined its pairwise distributions, so
            plotting is not available for it yet.
        """
        raise Exception(
            "Plotting is not available for this distribution yet, because "
            "the joint distribution of a pair of its variables has not been "
            "worked out in the code. To visualize it, simulate values and "
            "plot one variable (or a pair of variables) at a time."
        )

    def _plot_window(self, i):
        """Return the plotting window for component ``i``.

        Starts from the window the component's own one-dimensional
        distribution would use, then **zooms it when the probability fills
        too little of it** (:meth:`Distribution._fills_window`).

        That zoom is the one thing a joint panel does not inherit from
        :meth:`Distribution.plot`, and it is deliberate. A one-dimensional
        plot keeps its fixed bounds by default and zooms only when asked, via
        ``plot(xlim="zoom")``. A panel of a joint plot has no such escape
        hatch -- there is no per-panel ``xlim`` argument, and each panel is a
        fraction of the figure -- so a window the probability barely occupies
        leaves a panel that is mostly empty axis with no way to fix it.

        Parameters
        ----------
        i : int
            Index of the component, counting from 0.

        Returns
        -------
        tuple of float
            The ``(low, high)`` range to plot that component over.
        """
        marginal = self._marginal_1d(i)
        low, high = marginal.xlim
        zoom_low, zoom_high = marginal._zoom_xlim()
        if marginal._fills_window(low, high, zoom_low, zoom_high):
            return (low, high)
        return (zoom_low, zoom_high)

    def _marginal_framed(self, i):
        """Return component ``i``'s own distribution, framed like this plot.

        A marginal strip and a diagonal panel are both drawn by the univariate
        :meth:`Distribution.plot`, which picks its own window -- but here the
        window has to be the one the joint panels use, so the two line up.
        ``plot`` takes no window argument (it frames itself), so the window is
        pinned on the distribution instead. ``_marginal_1d`` builds a fresh
        distribution on every call, so pinning it affects nothing else.

        The ``xlim`` setter is the right way to pin it: a panel must land on
        exactly the joint panels' window, with none of the discrete padding a
        self-chosen window gets, or the column would stop lining up.

        Parameters
        ----------
        i : int
            Index of the component, counting from 0.

        Returns
        -------
        Distribution
            That component's one-dimensional distribution, with its ``xlim``
            set to :meth:`_plot_window`.
        """
        marginal = self._marginal_1d(i)
        marginal.xlim = self._plot_window(i)
        return marginal

    def _plot_values(self, i):
        """Return the values of a discrete component ``i`` to draw cells for.

        The whole numbers inside :meth:`_plot_window`, matching how the
        univariate :meth:`Distribution.plot` picks the values of a discrete
        distribution to draw masses at.

        Parameters
        ----------
        i : int
            Index of the component, counting from 0.

        Returns
        -------
        numpy.ndarray
            The values that component can take, in increasing order.
        """
        low, high = self._plot_window(i)
        return np.arange(int(low), int(high) + 1)

    def _resolve_variables(self, variables):
        """Check a ``variables`` argument and return the variables to plot.

        Also decides *what* to draw, since the two questions are the same
        one: exactly two variables have a single joint distribution between
        them, and three or more do not, so they are drawn as a matrix of
        every pair.

        Parameters
        ----------
        variables : sequence of int or None
            The requested variables. ``None`` falls back to the default:
            the two freely varying variables when there are only two, and
            every variable otherwise.

        Returns
        -------
        tuple
            ``(variables, pairs)`` -- the validated indices, and whether they
            are to be drawn as a matrix rather than one joint plot.

        Raises
        ------
        Exception
            If the distribution has too few freely varying variables to
            plot; if ``variables`` was left out where there is no single natural
            default; or if it is not the right number of distinct, in-range
            whole numbers.
        """
        n = self._n_components()
        free = self._free_dim()

        # A single number means that one variable, the same as a one-element
        # list: variables=2 and variables=[2] both ask for the 2nd variable's
        # own distribution. Normalized here so everything below sees a
        # sequence, and so the range and whole-number checks apply to it too.
        if isinstance(variables, numbers.Integral) and not isinstance(variables, bool):
            variables = (variables,)

        # Naming one variable asks for that variable's own distribution, which
        # exists no matter how the rest of them are constrained -- so this is
        # checked before the "needs two directions" gate below. Plotting one
        # marginal of a two-category Multinomial is a reasonable thing to ask
        # for, even though its *joint* plot is refused.
        one_variable = (
            isinstance(variables, (tuple, list, np.ndarray)) and len(variables) == 1
        )
        if free < 2 and not one_variable:
            raise Exception(
                "A joint plot needs two variables, and this distribution "
                "varies in only one direction, so there is nothing to plot "
                "against anything else. Plot it as the one-dimensional "
                "distribution it is -- a one-variable MultivariateNormal is "
                "a Normal, a two-category "
                "Multinomial(n, [p, 1 - p]) is a Binomial(n, p), and a "
                "two-category Dirichlet([a, b]) is a Beta(a, b)."
            )

        if variables is None:
            if free == 2:
                # Exactly two freely varying variables, so there is only one
                # joint distribution to show and no choice to make -- the
                # same way a univariate plot() never asks which variable.
                variables = (0, 1)
            else:
                # Three or more variables have no single joint plot, so show
                # every pair at once rather than asking which one to pick.
                variables = tuple(range(n))

        # Checked before converting, so a non-sequence that isn't a single
        # variable number reports this instead of a cryptic TypeError from the
        # conversion.
        if not isinstance(variables, (tuple, list, np.ndarray)):
            raise Exception(
                "variables must be a variable number or a list of them -- for "
                "example variables=2 for the 3rd variable on its own, "
                "variables=[0, 2] for the 1st and 3rd together, or "
                "variables=[0, 1, 3] for the 1st, 2nd and 4th. You passed "
                "variables=%r." % (variables,)
            )
        variables = tuple(variables)

        # One variable is its own distribution, two have one joint
        # distribution between them, and more become a matrix of every pair.
        pairs = len(variables) > 2
        if len(variables) < 1:
            raise Exception(
                "variables chooses which variables to plot, so it needs at "
                "least one -- name one for that variable's own distribution, "
                "for example variables=2; two for their joint distribution, "
                "for example variables=(0, 2); or more for a matrix of every "
                "pair among them, for example variables=(0, 1, 3). You passed "
                "variables=%r." % (variables,)
            )

        for d in variables:
            # bool is a subclass of int, so exclude it explicitly: variables=(True,
            # False) would otherwise silently mean variables=(1, 0).
            if isinstance(d, bool) or not isinstance(d, numbers.Integral):
                raise Exception(
                    "Every entry of variables must be a whole number naming a "
                    "variable, counting from 0. You passed variables=%r." % (variables,)
                )
            if not 0 <= d < n:
                raise Exception(
                    "This distribution has %d variables, numbered 0 to %d, "
                    "so variables=%r asks for a variable it doesn't have."
                    % (n, n - 1, variables)
                )
        if len(set(variables)) != len(variables):
            raise Exception(
                "Every entry of variables must be different -- a variable "
                "plotted against itself has nothing to show. You passed "
                "variables=%r." % (variables,)
            )

        return tuple(int(d) for d in variables), pairs

    def _plot_joint(
        self,
        i,
        j,
        ax,
        contour,
        colorbar=True,
        title=True,
        alpha=None,
        max_discrete_ticks=None,
        **kwargs,
    ):
        """Draw the joint distribution of components ``i`` and ``j`` on one axes.

        Chooses between the two joint plot types the same way the univariate
        :meth:`Distribution.plot` chooses between a curve and a set of
        masses: a continuous distribution gets a shaded density surface, a
        discrete one gets a grid of probabilities at the values it can
        actually take.

        Parameters
        ----------
        i, j : int
            Indices of the components on the x- and y-axis, counting from 0.
        ax : matplotlib.axes.Axes
            The axes to draw on.
        contour : bool
            Whether to draw the continuous surface as discrete contour
            bands. Has no effect on a discrete distribution, whose cells
            are already discrete.
        colorbar : bool, default True
            Whether to add a colorbar. A panel of a pairs matrix passes
            False.
        title : bool, default True
            Whether to title the axes. A panel of a pairs matrix passes
            False, since the figure carries one title instead.
        alpha : float, optional
            Transparency of the surface, from 0 (invisible) to 1 (opaque).
        max_discrete_ticks : int, optional
            Cap on how many ticks a discrete axis shows, forwarded to
            ``make_joint_pmf``. ``None`` (default) leaves that function's
            own default in place; a panel of a pairs matrix passes
            ``PAIRS_MAX_DISCRETE_TICKS``, since its panel is a fraction of
            a full-size plot's width. Has no effect on a continuous
            distribution's density surface.
        **kwargs
            Additional keyword arguments forwarded to matplotlib.

        Returns
        -------
        matplotlib.cm.ScalarMappable
            The surface or mesh that was drawn.
        """
        func = self._joint_func(i, j)
        xlabel = self._variable_label(i)
        ylabel = self._variable_label(j)
        if self.discrete:
            if max_discrete_ticks is not None:
                kwargs["max_discrete_ticks"] = max_discrete_ticks
            return make_joint_pmf(
                func,
                self._plot_values(i),
                self._plot_values(j),
                ax,
                colorbar=colorbar,
                xlabel=xlabel,
                ylabel=ylabel,
                title=title,
                alpha=alpha,
                **kwargs,
            )
        return make_joint_pdf(
            func,
            self._plot_window(i),
            self._plot_window(j),
            ax,
            contour=contour,
            colorbar=colorbar,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            alpha=alpha,
            **kwargs,
        )

    def _plot_marginal_panel(self, i, ax, orientation, alpha=None):
        """Draw one variable's own distribution in a strip beside a joint plot.

        The strip is that variable's marginal pdf or pmf -- the exact
        closed-form one from :meth:`_marginal_1d`, not a slice of the joint
        surface -- drawn by the univariate :meth:`Distribution.plot` so it
        looks the same as plotting that variable on its own.

        The strip to the *right* of a joint panel has to run sideways, with
        the variable on the y-axis, so it lines up with the joint panel's y
        variable. Rather than reimplement the univariate plot in the other
        direction, this draws it the usual way and then transposes what was
        drawn -- so the two directions cannot drift apart in styling.

        Parameters
        ----------
        i : int
            Index of the component, counting from 0.
        ax : matplotlib.axes.Axes
            The strip to draw on.
        orientation : {"vertical", "horizontal"}
            ``"vertical"`` puts the variable on the x-axis, for the strip
            above a joint panel; ``"horizontal"`` puts it on the y-axis, for
            the strip to the right of one.
        alpha : float, optional
            Transparency of the curve.
        """
        # Framed on the same window the joint panel uses, so the strip covers
        # the variable over exactly the range the joint panel shows it over.
        self._marginal_framed(i).plot(ax=ax, alpha=alpha)
        # In this layout the same figure also contains the joint distribution,
        # so make clear that this strip is the marginal counterpart.
        marginal_quantity = pairs_marginal_label(ax.get_ylabel())
        # Each strip is one variable's distribution, not a plot in its own
        # right -- the figure's title names what the whole thing is.
        ax.set_title("")
        # A strip is a fraction of the joint panel's size, so the tick count
        # a full-size axes would take crowds into itself here.
        thin_marginal_frequency_ticks(ax, orientation)
        if orientation == "vertical":
            # The joint panel below already names this variable on its x-axis,
            # and the two axes are shared, so the strip's own copy of the label
            # would be clutter right next to it. The density/probability label
            # stays: it is what the strip's height means.
            ax.set_xlabel("")
            ax.set_ylabel(marginal_quantity)
            return

        # Turn the plot on its side: the values move to the y-axis and the
        # density/probability to the x-axis. A continuous marginal is one
        # curve (a Line2D); a discrete one is a dot per value (a
        # PathCollection from scatter) plus a dashed connector.
        for line in ax.lines:
            xdata, ydata = line.get_data()
            line.set_data(ydata, xdata)
        for collection in ax.collections:
            offsets = np.asarray(collection.get_offsets())
            collection.set_offsets(offsets[:, ::-1])
        value_lim, freq_lim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(*freq_lim)
        ax.set_ylim(*value_lim)
        # The labels transpose with the data: what named the height now names
        # the width. The value label is dropped for the same reason as above.
        ax.set_xlabel(marginal_quantity)
        ax.set_ylabel("")

    def _plot_pairs(self, variables, contour, alpha=None, **kwargs):
        """Draw a matrix of every pair of the chosen variables.

        Builds an ``n`` by ``n`` grid of panels: each variable's own
        distribution down the diagonal, and each pair's joint distribution
        in the lower triangle. The upper triangle is left blank because
        panel ``(i, j)`` and panel ``(j, i)`` show the same relationship
        with the axes swapped, so filling both would draw everything twice.

        Parameters
        ----------
        variables : tuple of int
            The variables to include, already validated.
        contour : bool
            Whether the joint panels draw contour bands.
        alpha : float, optional
            Transparency of the panels.
        **kwargs
            Additional keyword arguments forwarded to matplotlib.

        Returns
        -------
        matplotlib.axes.Axes
            The bottom-left panel, as a representative of the layout.

        Raises
        ------
        ValueError
            If the figure already has a plot on it, which the grid can't
            be built into.
        Exception
            If more variables were asked for than a readable matrix holds.
        """
        if len(variables) > JOINT_PAIRS_MAX_DIM:
            raise Exception(
                "A pairs plot of %d variables would need %d panels, too many "
                "to read on one screen. Choose which variables to include, "
                "for example .plot(pairs=True, variables=(0, 1, 2))."
                % (len(variables), len(variables) * (len(variables) + 1) // 2)
            )

        fig = plt.gcf()
        # The grid fills the whole figure, so it can neither be added to a
        # figure that already has a plot on it nor accept one later. Fail
        # the same way the marginal layout does (MARGINAL_OVERLAY_ERROR)
        # rather than stacking a second layout on top of existing content.
        if fig.axes:
            raise ValueError(JOINT_PAIRS_OVERLAY_ERROR)
        k = len(variables)
        # Size the figure to the grid, so panels stay readable as variables
        # are added instead of each one shrinking inside a single-plot figure.
        fig.set_size_inches(k * JOINT_PAIRS_PANEL_SIZE, k * JOINT_PAIRS_PANEL_SIZE)
        gs = GridSpec(k, k, figure=fig)

        corner = None
        # (mappable, row, col) per joint panel, so each can be given its own
        # colorbar once the layout has settled (see the end of this method).
        joint_panels = []
        # Every panel by cell, so the columns can be put on one x scale once
        # they have all drawn (see align_pairs_columns).
        cells = {}
        for row in range(k):
            for col in range(row + 1):
                ax = fig.add_subplot(gs[row, col])
                cells[(row, col)] = ax
                # What this panel's frequency axis measures, for a diagonal
                # panel (see the label section below).
                marginal_quantity = ""
                if row == col:
                    # The diagonal is this variable on its own, so it is
                    # exactly the univariate plot -- reuse it rather than
                    # redraw it, which also advances each panel's own color
                    # cycle (so every diagonal takes the same first color).
                    # Framed on the same window the joint panels in this
                    # column use, so every panel in a column really does
                    # cover the variable over the same range -- which is
                    # what makes hiding the inner x tick labels below safe.
                    advance_pairs_diagonal_color(ax, row)
                    self._marginal_framed(variables[row]).plot(ax=ax, alpha=alpha)
                    ax.set_title("")
                    # Whatever the univariate plot called its own frequency
                    # axis -- "Density" for a continuous family, "Probability"
                    # for a discrete one. Read off the panel rather than
                    # re-derived, so the two can't disagree.
                    marginal_quantity = ax.get_ylabel()
                    # A discrete marginal's value axis otherwise keeps
                    # matplotlib's default locator, sized for a full-size
                    # plot -- too many ticks for this panel's fraction of
                    # the width. Not re-laid-out later (unlike a simulated
                    # dot plot), so setting it directly here is enough.
                    if self.discrete:
                        ax.xaxis.set_major_locator(
                            MaxNLocator(nbins=PAIRS_MAX_DISCRETE_TICKS, integer=True)
                        )
                else:
                    joint_panels.append(
                        (
                            self._plot_joint(
                                variables[col],
                                variables[row],
                                ax,
                                contour,
                                colorbar=False,
                                title=False,
                                alpha=alpha,
                                max_discrete_ticks=PAIRS_MAX_DISCRETE_TICKS,
                                **kwargs,
                            ),
                            row,
                            col,
                        )
                    )
                # Name the variables only along the outside edges, so the
                # inner panels aren't crowded with repeated labels. Set them
                # here rather than leave whatever each panel drew: a
                # diagonal panel came from the univariate plot(), which
                # labels its x-axis "Value" since on its own it has no
                # variable number to use.
                if row == k - 1:
                    ax.set_xlabel(self._variable_label(variables[col]))
                else:
                    ax.set_xlabel("")
                    # Every panel in a column covers the same variable over
                    # the same window, so hiding the inner x tick labels
                    # loses nothing -- the bottom panel's still apply. The y
                    # tick labels stay on every panel, since a diagonal
                    # panel's y-axis is a density and genuinely differs from
                    # its neighbors'.
                    ax.set_xticklabels([])
                # A diagonal panel's y-axis is not the variable at all -- it is
                # that variable's own density or probability -- so it says so,
                # and says which of the matrix's two kinds of distribution it
                # is showing. The variable itself is still named at the bottom
                # of that column, since a diagonal panel sits above one.
                # Everything else in the left column names its row's variable,
                # so the labels read down the side in order; the inner panels
                # stay unlabeled rather than repeat them. A simulated pairs
                # matrix labels itself the same way.
                if marginal_quantity:
                    ax.set_ylabel(pairs_marginal_label(marginal_quantity))
                elif col == 0:
                    ax.set_ylabel(self._variable_label(variables[row]))
                else:
                    ax.set_ylabel("")
                if row == k - 1 and col == 0:
                    corner = ax

        # Name what the panels actually are. Every panel of a theoretical
        # matrix is an exact distribution -- a density for a continuous
        # family, a set of probabilities for a discrete one -- so the title
        # says which, rather than naming the layout.
        # A column all shows one variable, so its panels have to line up: a
        # mass on the diagonal directly above the joint pmf cell for that
        # value. Each panel type frames its own axes differently, so this is
        # not free -- and the simulated matrix uses the same helper.
        align_pairs_columns(cells)

        fig.suptitle(
            "Probability Mass Functions"
            if self.discrete
            else "Probability Density Functions"
        )
        fig.tight_layout()

        # Each joint panel gets its own colorbar, in the empty cell mirroring
        # it across the diagonal -- the same treatment a simulated pairs
        # matrix gets, so the two can be read side by side. Done after
        # tight_layout, so the cells are where they will finally be.
        quantity = pairs_joint_label("Probability" if self.discrete else "Density")
        for mappable, row, col in joint_panels:
            if mappable is None:
                continue
            add_pairs_panel_colorbar(
                fig,
                gs[col, row].get_position(fig),
                mappable,
                pairs_colorbar_pair_label(variables[col], variables[row]),
                quantity,
            )
        return corner

    def plot(self, variables=None, contour=True, alpha=None, ax=None, **kwargs):
        """Plot the joint distribution of two of the variables.

        A multivariate distribution describes several variables at once, so
        it has no single curve to draw the way a one-dimensional
        distribution does. What it does have is a joint distribution for
        any two of its variables, and that is what this plots: a shaded
        density surface (continuous distributions) or a grid of
        probabilities (discrete distributions) showing how likely each
        combination of the two values is.

        What gets plotted depends on how many variables there are:

        - **Two variables** (including a three-category ``Multinomial`` or
          ``Dirichlet``, whose third category is fixed by the other two):
          there is only one joint distribution, so it is plotted with no
          arguments needed.
        - **Three or more**: there is no single "the plot" any more, so
          this draws a **matrix of panels** -- each variable's own
          distribution down the diagonal, each pair's joint distribution
          below it -- rather than silently picking one pair out of several.
        - **One particular pair**: name the two variables you want with
          ``variables``, e.g. ``variables=(0, 2)`` for the 1st and 3rd. Naming three
          or more instead draws the matrix of just those.
        - **One variable on its own**: name just it, e.g. ``variables=2``,
          for that variable's own (marginal) distribution -- the same exact
          curve the diagonal of the matrix shows.

        Each panel shows an *exact* distribution, not an approximation:
        every one of these families has a closed-form distribution for one
        variable (``Normal``, ``Binomial``, ``Beta``) and for a pair of
        them, so nothing here is estimated by simulation.

        Parameters
        ----------
        variables : int or sequence of int, optional
            Which variables to plot, numbered from 0. Name **one** for that
            variable's own distribution, e.g. ``variables=2`` (or
            ``variables=[2]``) for the 3rd variable's marginal pdf/pmf;
            **two** for their joint distribution, e.g. ``variables=(0, 2)``
            for the 1st and 3rd; **three or more** for a matrix of every pair
            among them. Left out, a distribution of two freely varying
            variables draws its one joint plot, and one of three or more draws
            the matrix of all of them. The panels are labeled by the numbers
            asked for, so ``variables=[0, 2, 4]`` labels them "Variable 1",
            "Variable 3" and "Variable 5".
        contour : bool, default True
            Whether to draw a continuous density surface as discrete
            contour bands with outlines between them, so a band can be
            matched to the colorbar by eye. Pass ``contour=False`` for one
            smooth gradient instead. Has no effect on a discrete
            distribution, whose grid of probabilities is already made of
            discrete cells.
        alpha : float, optional
            Transparency of the plot, from 0 (invisible) to 1 (opaque).
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. Creates or uses the current axes if not
            provided. Ignored for a pairs matrix, which builds its own
            panels.
        **kwargs
            Additional keyword arguments forwarded to matplotlib.

        Returns
        -------
        JointDistributionPlot
            A wrapper around the axes the plot was drawn on. Its printed
            representation is empty, so Jupyter shows only the plot.

        Raises
        ------
        Exception
            If the distribution varies in only one direction and more than one
            variable was asked for (plot it as the one-dimensional
            distribution it is); if ``variables`` does not name at least one
            distinct, in-range variable; or if a matrix would have more panels
            than fit.
        ValueError
            If a pairs matrix is drawn on a figure that already has a plot
            on it, or if the removed ``pairs=`` keyword is passed.

        Examples
        --------
        >>> from symbulate import *
        >>> MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]).plot()  # doctest: +SKIP
        >>> X = MultivariateNormal(mean=[1, 2, 3, 4], cov=np.eye(4))  # doctest: +SKIP
        >>> X.plot(variables=(0, 2))  # the 1st and 3rd variables  # doctest: +SKIP
        >>> X.plot()  # every pair at once, for three or more variables  # doctest: +SKIP
        >>> X.plot(variables=2)  # just the 3rd variable's own distribution  # doctest: +SKIP
        >>> Multinomial(n=10, p=[0.5, 0.3, 0.2]).plot()  # doctest: +SKIP
        """
        # A joint plot shows a whole surface rather than one curve, so
        # neither the `type=` that selects among ways of drawing simulated
        # data nor the `cdf=` that switches a univariate curve applies here.
        # Catch both so they give a pointer instead of slipping through
        # **kwargs into an opaque matplotlib error.
        if "type" in kwargs:
            raise ValueError(
                "`plot()` does not take a `type=` argument. To see every "
                "pair of variables instead of one, name them with "
                "variables=; to draw the density surface as one smooth "
                "gradient instead of contour bands, use contour=False. "
                "(`type=` selects among the many ways of drawing simulated "
                "data.)"
            )
        if "cdf" in kwargs:
            raise ValueError(
                "`cdf=True` is not available for a multivariate "
                "distribution's plot. A cumulative distribution function "
                "needs an order on the values, and there is no natural way "
                "to order vectors. Plot one variable at a time for a CDF -- "
                "for example, Normal(0, 1).plot(cdf=True)."
            )

        # pairs=True used to be how the matrix was asked for. It is the
        # default for three or more variables now, so say so rather than
        # letting the stray keyword reach matplotlib.
        if "dims" in kwargs:
            raise ValueError(
                "dims= has been renamed variables= -- for example "
                ".plot(variables=[0, 2]) for the 1st and 3rd variables, or "
                ".plot(variables=[0, 1, 3]) for a matrix of the 1st, 2nd and "
                "4th."
            )

        # A theoretical two-variable plot always shows each variable's own
        # distribution beside the joint one, so there is nothing to switch on.
        # (Simulated results *do* take marginal=, since a plot of data has
        # reasons to want the bare joint panel -- overlaying two of them, for
        # one. An exact distribution has no such need.)
        if "marginal" in kwargs:
            raise ValueError(
                "marginal= is not needed here: a distribution's plot of two "
                "variables always shows each variable's own distribution in a "
                "strip beside the main panel, so .plot() alone does it. Drop "
                "marginal=True. (Simulated results still take marginal= -- it "
                "is only a distribution's own plot that always shows them.)"
            )

        if "pairs" in kwargs:
            raise ValueError(
                "pairs= is no longer needed: a distribution of three or more "
                "variables is drawn as a matrix of every pair by default, so "
                ".plot() alone does it. Drop pairs=True. (Name two variables, "
                "e.g. .plot(variables=(0, 2)), for one joint plot instead.)"
            )

        variables, pairs = self._resolve_variables(variables)

        # One variable named: draw that variable's own distribution, which is
        # the exact closed-form marginal -- the same curve the diagonal of a
        # pairs matrix shows, and the same one plotting that marginal directly
        # would give. No joint surface, so `contour` has nothing to apply to.
        # This *is* a one-dimensional plot, so `xlim=` rides through **kwargs
        # into it and works exactly as it does there -- which is why the check
        # below sits after this branch rather than before it.
        if len(variables) == 1:
            (only,) = variables
            plot = self._marginal_1d(only).plot(alpha=alpha, ax=ax, **kwargs)
            # Name the variable rather than leaving the univariate plot's
            # generic "Value": the caller asked for a numbered variable of this
            # distribution, so say which one it is.
            plot.ax.set_xlabel(self._variable_label(only))
            return plot

        # Past here more than one panel is drawn -- a joint panel plus a strip
        # per variable, or a whole matrix of pairs -- so a single range does
        # not say which panel it belongs to, and the panels have to stay on a
        # shared scale to line up. Each is framed by `_plot_window` instead.
        # Caught here so it says that, rather than reaching make_joint_pdf as
        # a "multiple values for argument 'xlim'" TypeError.
        if "xlim" in kwargs:
            raise ValueError(
                "plot() of several variables at once does not take an xlim= "
                "argument: it draws more than one panel, so a single range "
                "doesn't say which one to apply to, and the panels have to "
                "share a scale to line up. Each panel frames itself. To set a "
                "range, plot one variable on its own, where xlim= works as "
                "usual -- .plot(variables=0, xlim=(low, high))."
            )

        if pairs:
            ax = self._plot_pairs(variables, contour, alpha=alpha, **kwargs)
            return JointDistributionPlot(ax, self, variables)

        # An axes was handed in, so draw the joint distribution on it and
        # nothing else: one axes has no room for the strips, and a caller
        # who named the axes is placing this plot inside a layout of their
        # own. Advance the color cycle once per plot() call, as every plot
        # type does -- a joint plot colors by a colormap rather than the
        # cycle, but skipping this would leave a curve drawn onto the same
        # axes afterwards reusing a color already on the plot.
        if ax is not None:
            get_next_color(ax)
            self._plot_joint(
                variables[0], variables[1], ax, contour, alpha=alpha, **kwargs
            )
            return JointDistributionPlot(ax, self, variables)

        # Two variables are shown on three panels: their joint distribution,
        # plus each one's own distribution in a strip beside the matching
        # axis. Built by the same helper the simulated side uses, so the two
        # layouts stay identical.
        fig = plt.gcf()
        ax, ax_marg_x, ax_marg_y = setup_marginal_axes(fig)
        plt.sca(ax)
        get_next_color(ax)
        # The joint helpers put their own colorbar immediately right of the
        # axes, which here would land on top of the y strip -- so suppress it
        # and let add_colorbar place it past the strip instead, exactly as a
        # simulated two-variable plot does.
        mappable = self._plot_joint(
            variables[0],
            variables[1],
            ax,
            contour,
            colorbar=False,
            alpha=alpha,
            **kwargs,
        )
        quantity = pairs_joint_label("Probability" if self.discrete else "Density")
        if mappable is not None:
            add_colorbar(fig, True, mappable, quantity, decimals=JOINT_CBAR_DECIMALS)

        self._plot_marginal_panel(variables[0], ax_marg_x, "vertical", alpha=alpha)
        self._plot_marginal_panel(variables[1], ax_marg_y, "horizontal", alpha=alpha)
        # Lock each strip to the joint panel's own final limits rather than
        # re-deriving them, so the strip lines up with whatever extent the
        # joint plot actually drew (a pmf mesh runs half a unit past its
        # outermost values). sharex/sharey then keeps them together for any
        # later interaction.
        ax_marg_x.set_xlim(ax.get_xlim())
        ax_marg_x.sharex(ax)
        ax_marg_y.set_ylim(ax.get_ylim())
        ax_marg_y.sharey(ax)
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        # There is no room for the joint panel's own title -- it would collide
        # with the strip above it -- but what the plot is ("Joint Probability
        # Density Function", "Joint Probability Mass Function") is worth
        # keeping, so it moves to the figure,
        # above all three panels. Read back from the panel rather than
        # re-derived, so it stays whatever the joint plot titled itself.
        fig.suptitle(ax.get_title())
        ax.set_title("")
        # Leave the joint panel current: drawing the strips and the colorbar
        # moved plt.gca() off it, and it is the panel plt.gca() should mean.
        plt.sca(ax)
        return JointDistributionPlot(ax, self, variables)

    def __pow__(self, exponent):
        """Draw several independent vectors from the distribution.

        Parameters
        ----------
        exponent : int or float
            Number of independent vectors to draw. Pass ``float('inf')`` to
            create an infinite sequence generated lazily on demand.

        Returns
        -------
        ProbabilitySpace
            A probability space whose draws produce ``exponent`` vectors at
            a time.
        """
        if exponent == float("inf"):

            def draw():
                def _func(n):
                    return self.draw()

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)


class MultivariateNormal(MultivariateDistribution):
    """Probability space for a multivariate normal distribution.

    Generalizes the normal distribution to multiple dimensions. Each
    draw produces a vector of correlated normal values, described by
    a mean vector and a covariance matrix.

    Parameters
    ----------
    mean : array-like of length n
        The mean vector of the distribution.
    cov : array-like of shape (n, n)
        The covariance matrix. Must be symmetric and positive semi-definite.

    Methods
    -------
    mean()
        The mean vector, as a :class:`Vector`.
    cov()
        The covariance matrix, as a NumPy 2-D array.
    var(), sd()
        The per-component variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.

    Examples
    --------
    >>> from symbulate import *
    >>> X = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    >>> list(X.mean())
    [0, 0]
    >>> X.draw()  # doctest: +SKIP
    (1.76, 0.40)
    """

    def __init__(self, mean, cov):
        """Initialize a multivariate normal distribution.

        Raises
        ------
        Exception
            If the mean vector is empty; if the mean and covariance matrix
            have incompatible sizes; if the covariance matrix is not square;
            or if it is not symmetric positive semi-definite.
        """
        if len(mean) < 1:
            raise Exception(
                "The mean vector cannot be empty. Give one mean value per "
                "variable, for example mean=[0, 0] for two variables."
            )

        if len(cov) != len(mean):
            raise Exception(
                "The mean vector and covariance matrix do not match: mean "
                "has %d entries but cov has %d rows. Give one mean per "
                "variable and a cov with that many rows and columns."
                % (len(mean), len(cov))
            )

        if not all(len(row) == len(mean) for row in cov):
            raise Exception(
                "The covariance matrix must be square, with one row and one "
                "column per variable (%d by %d here). Check that every row "
                "of cov has %d entries." % (len(mean), len(mean), len(mean))
            )

        cov_array = np.asarray(cov, dtype=float)
        if not np.allclose(cov_array, cov_array.T):
            raise Exception(
                "The covariance matrix must be symmetric: the entry in row i, "
                "column j must equal the entry in row j, column i. Check that "
                "the off-diagonal entries match."
            )

        # A symmetric matrix has real eigenvalues, so use eigvalsh (which
        # assumes symmetry) rather than eigvals, which returns complex values
        # for a non-symmetric matrix and makes the check unreliable. Allow a
        # tiny negative tolerance so floating-point round-off in an otherwise
        # valid matrix is not rejected.
        if np.min(np.linalg.eigvalsh(cov_array)) < -1e-8:
            raise Exception(
                "The covariance matrix must be positive semi-definite, since "
                "it describes variances and correlations and cannot imply a "
                "negative variance. An off-diagonal (covariance) entry that "
                "is too large relative to the diagonal (variance) entries is "
                "the usual cause."
            )

        self._mean = mean
        self._cov = cov
        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(self._mean, self._cov).pdf(x)

    def mean(self):
        """Return the mean vector.

        Returns
        -------
        Vector
            The mean of each component.
        """
        return Vector(self._mean)

    def cov(self):
        """Return the covariance matrix.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        return np.asarray(self._cov, dtype=float)

    def _marginal_1d(self, i):
        """Return the distribution of variable ``i`` on its own.

        One variable of a multivariate normal is itself normal, with that
        variable's own mean and the square root of its own variance (entry
        ``i, i`` of the covariance matrix) as its standard deviation.

        Parameters
        ----------
        i : int
            Index of the variable, counting from 0.

        Returns
        -------
        Normal
            The distribution of that one variable.
        """
        mean = np.asarray(self._mean, dtype=float)
        cov = np.asarray(self._cov, dtype=float)
        return Normal(mean=mean[i], sd=np.sqrt(cov[i, i]))

    def _joint_func(self, i, j):
        """Return the joint density of variables ``i`` and ``j``.

        Two variables of a multivariate normal are themselves a bivariate
        normal, with those two entries of the mean vector and the
        corresponding 2x2 block of the covariance matrix -- an exact
        marginal distribution, so no simulation or numerical integration is
        involved.

        Parameters
        ----------
        i, j : int
            Indices of the two variables, counting from 0.

        Returns
        -------
        callable
            The joint density, as a function of two equal-length flat
            arrays of coordinates.
        """
        mean = np.asarray(self._mean, dtype=float)
        cov = np.asarray(self._cov, dtype=float)
        index = [i, j]
        pair = stats.multivariate_normal(mean[index], cov[np.ix_(index, index)])
        return lambda x, y: pair.pdf(np.column_stack([np.ravel(x), np.ravel(y)]))

    def draw(self):
        """Draw a single random sample from the multivariate normal distribution.

        Returns
        -------
        Vector
            A random vector drawn from the multivariate normal distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]]).draw()  # doctest: +SKIP
        (1.76, 0.40)
        """
        return Vector(rng.multivariate_normal(self._mean, self._cov))


class BivariateNormal(MultivariateNormal):
    """Probability space for a bivariate normal distribution.

    A special case of the multivariate normal with exactly two variables
    (X and Y). You can specify the relationship between X and Y using
    either correlation (``corr``) or covariance (``cov``).

    Parameters
    ----------
    mean1 : float, optional
        Mean of the first variable. Default is 0.0.
    mean2 : float, optional
        Mean of the second variable. Default is 0.0.
    sd1 : float, optional
        Standard deviation of the first variable. Default is 1.0.
    sd2 : float, optional
        Standard deviation of the second variable. Default is 1.0.
    corr : float, optional
        Correlation between the two variables, between -1 and 1.
        Default is 0.0.
    var1 : float, optional
        Variance of the first variable. Overrides ``sd1`` if provided.
    var2 : float, optional
        Variance of the second variable. Overrides ``sd2`` if provided.
    cov : float, optional
        Covariance between the two variables. Overrides ``corr`` if provided.

    Methods
    -------
    mean(), cov(), var(), sd(), corr()
        Inherited from :class:`MultivariateNormal`. ``mean()`` returns the
        mean vector ``[mean1, mean2]`` and ``cov()`` the 2x2 covariance
        matrix; the others are derived from it.

    Examples
    --------
    >>> from symbulate import *
    >>> X = BivariateNormal(mean1=0, mean2=0, sd1=1, sd2=1, corr=0.5)
    >>> X.draw()  # doctest: +SKIP
    (1.76, 1.20)
    """

    def __init__(
        self,
        mean1=0.0,
        mean2=0.0,
        sd1=1.0,
        sd2=1.0,
        corr=0.0,
        var1=None,
        var2=None,
        cov=None,
    ):
        """Initialize a bivariate normal distribution.

        Raises
        ------
        Exception
            If ``mean1``, ``mean2``, or ``cov`` is not a number; if ``corr``
            is not a number between -1 and 1; if ``sd1``, ``sd2``, ``var1``,
            or ``var2`` is not a non-negative number; or if the resulting
            covariance matrix is not positive semi-definite (an explicit
            ``cov`` too large relative to ``var1`` and ``var2``).
        """

        _validate(
            (not isinstance(mean1, numbers.Real), "mean1 must be a number"),
            (not isinstance(mean2, numbers.Real), "mean2 must be a number"),
            (
                not isinstance(corr, numbers.Real) or not -1 <= corr <= 1,
                "corr must be a number between -1 and 1",
            ),
            (
                not isinstance(sd1, numbers.Real) or sd1 < 0,
                "sd1 must be a non-negative number",
            ),
            (
                not isinstance(sd2, numbers.Real) or sd2 < 0,
                "sd2 must be a non-negative number",
            ),
            (
                var1 is not None and (not isinstance(var1, numbers.Real) or var1 < 0),
                "var1 must be a non-negative number",
            ),
            (
                var2 is not None and (not isinstance(var2, numbers.Real) or var2 < 0),
                "var2 must be a non-negative number",
            ),
            (
                cov is not None and not isinstance(cov, numbers.Real),
                "cov must be a number",
            ),
        )

        # var1/var2 default to sd**2, and cov defaults to corr*sqrt(var1*var2);
        # every value used below has been validated above.
        if var1 is None:
            var1 = sd1**2
        if var2 is None:
            var2 = sd2**2
        if cov is None:
            cov = corr * np.sqrt(var1 * var2)

        # Route through MultivariateNormal so the assembled covariance matrix
        # gets the same positive semi-definite check and the shared
        # mean()/cov()/var()/sd()/corr()/draw() machinery. This also validates
        # an explicitly supplied cov that is too large relative to var1, var2.
        super().__init__([mean1, mean2], [[var1, cov], [cov, var2]])


class MultivariateT(MultivariateDistribution):
    """Probability space for a multivariate t (Student) distribution.

    Generalizes Student's t-distribution to multiple dimensions -- the
    heavy-tailed counterpart of the :class:`MultivariateNormal`. Each draw
    produces a vector of correlated values described by a location vector, a
    scale matrix, and the degrees of freedom that set the tail weight. As
    the degrees of freedom grow, the distribution approaches a multivariate
    normal with covariance equal to the scale matrix.

    Parameters
    ----------
    mean : array-like of length n
        The location (center) vector of the distribution.
    cov : array-like of shape (n, n)
        The scale matrix (also called the shape matrix). Must be symmetric
        and positive semi-definite. It plays the role ``cov`` does for the
        multivariate normal, but it is not the covariance: for ``df > 2``
        the actual covariance is ``df / (df - 2)`` times this matrix.
    df : float
        Degrees of freedom. Must be positive. Smaller values give heavier
        tails; as ``df`` grows the distribution approaches a
        ``MultivariateNormal(mean, cov)``.

    Methods
    -------
    mean()
        The mean vector (equal to the location), as a :class:`Vector`. Exists
        only for ``df > 1``; raises otherwise.
    cov()
        The covariance matrix, as a NumPy 2-D array. This is ``df / (df - 2)``
        times the scale matrix and exists only for ``df > 2``; raises otherwise.
    var(), sd()
        The per-component variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.

    Examples
    --------
    >>> from symbulate import *
    >>> X = MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=5)
    >>> X.draw()  # doctest: +SKIP
    (0.42, -1.13)
    """

    def __init__(self, mean, cov, df):
        """Initialize a multivariate t distribution.

        Raises
        ------
        Exception
            If the mean vector is empty; if the dimensions of ``mean`` and
            ``cov`` are incompatible; if the scale matrix is not square or
            not symmetric positive semi-definite; or if ``df`` is not a
            positive number.
        """
        if len(mean) < 1:
            raise Exception(
                "The mean (location) vector cannot be empty. Give one value "
                "per variable, for example mean=[0, 0] for two variables."
            )

        if len(cov) != len(mean):
            raise Exception(
                "The mean vector and scale matrix do not match: mean has %d "
                "entries but cov has %d rows. Give one location value per "
                "variable and a scale matrix with that many rows and columns."
                % (len(mean), len(cov))
            )

        if not all(len(row) == len(mean) for row in cov):
            raise Exception(
                "The scale matrix must be square, with one row and one column "
                "per variable (%d by %d here). Check that every row of cov "
                "has %d entries." % (len(mean), len(mean), len(mean))
            )

        cov_array = np.asarray(cov, dtype=float)
        if not np.allclose(cov_array, cov_array.T):
            raise Exception(
                "The scale matrix must be symmetric: the entry in row i, "
                "column j must equal the entry in row j, column i. Check that "
                "the off-diagonal entries match."
            )

        # A symmetric matrix has real eigenvalues, so use eigvalsh rather than
        # eigvals (which returns complex values for a non-symmetric matrix).
        # Allow a tiny negative tolerance for floating-point round-off.
        if np.min(np.linalg.eigvalsh(cov_array)) < -1e-8:
            raise Exception(
                "The scale matrix must be positive semi-definite (it plays "
                "the role a covariance matrix does). An off-diagonal entry "
                "that is too large relative to the diagonal entries is the "
                "usual cause."
            )

        if not isinstance(df, numbers.Real) or df <= 0:
            raise Exception("df must be a positive number")

        self._mean = mean
        self._cov = cov
        self._df = df
        self.discrete = False
        self.pdf = lambda x: stats.multivariate_t(
            loc=self._mean, shape=self._cov, df=self._df
        ).pdf(x)

    def mean(self):
        """Return the mean vector.

        The mean equals the location vector, but it exists only for
        ``df > 1`` -- just as the mean of a univariate Student's t is
        undefined at one degree of freedom.

        Returns
        -------
        Vector
            The mean (which equals the location vector).

        Raises
        ------
        Exception
            If ``df <= 1``, where the mean is undefined.
        """
        if self._df <= 1:
            raise Exception(
                "The mean of a multivariate t is undefined for df <= 1; "
                "here df = %s. It exists only for df > 1, where it equals "
                "the location vector." % (self._df,)
            )
        return Vector(self._mean)

    def _n_components(self):
        """Return how many components each draw has.

        Overridden because the base class counts the entries of ``mean()``,
        and this distribution's mean does not exist for ``df <= 1``. The
        location vector always does, and has exactly one entry per
        component, so the count is read from it instead -- letting a
        heavy-tailed ``df = 1`` distribution still be plotted, whose density
        is perfectly well defined even where its moments are not.

        Returns
        -------
        int
            The number of variables the distribution describes.
        """
        return len(self._mean)

    def cov(self):
        """Return the covariance matrix.

        The scale matrix passed to the constructor is not the covariance: the
        covariance is ``df / (df - 2)`` times the scale matrix, and it exists
        only when ``df > 2``.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.

        Raises
        ------
        Exception
            If ``df <= 2``, where the covariance is undefined (infinite).
        """
        if self._df <= 2:
            raise Exception(
                "The covariance of a multivariate t is undefined (infinite) "
                "for df <= 2; here df = %s. It exists only for df > 2, where "
                "it equals df / (df - 2) times the scale matrix." % (self._df,)
            )
        return (self._df / (self._df - 2)) * np.asarray(self._cov, dtype=float)

    def _marginal_1d(self, i):
        """Return the distribution of variable ``i`` on its own.

        One variable of a multivariate t is a t-distribution with the same
        degrees of freedom, centered at that variable's location and scaled
        by the square root of entry ``i, i`` of the scale matrix. Symbulate's
        :class:`StudentT` is always centered at 0 with scale 1, so this
        shifted and scaled version is built on scipy's ``t`` directly,
        through the same wiring every one-dimensional distribution uses.

        Parameters
        ----------
        i : int
            Index of the variable, counting from 0.

        Returns
        -------
        Distribution
            The distribution of that one variable: a t-distribution with
            ``df`` degrees of freedom, shifted and scaled.
        """
        mean = np.asarray(self._mean, dtype=float)
        cov = np.asarray(self._cov, dtype=float)
        params = {
            "df": self._df,
            "loc": mean[i],
            "scale": np.sqrt(cov[i, i]),
        }
        return Distribution(params, stats.t, False)

    def _joint_func(self, i, j):
        """Return the joint density of variables ``i`` and ``j``.

        Two variables of a multivariate t are themselves a bivariate t with
        the *same* degrees of freedom, taking those two entries of the
        location vector and the corresponding 2x2 block of the scale matrix
        -- an exact marginal distribution, like the multivariate normal's.

        Parameters
        ----------
        i, j : int
            Indices of the two variables, counting from 0.

        Returns
        -------
        callable
            The joint density, as a function of two equal-length flat
            arrays of coordinates.
        """
        mean = np.asarray(self._mean, dtype=float)
        cov = np.asarray(self._cov, dtype=float)
        index = [i, j]
        pair = stats.multivariate_t(
            loc=mean[index], shape=cov[np.ix_(index, index)], df=self._df
        )
        return lambda x, y: pair.pdf(np.column_stack([np.ravel(x), np.ravel(y)]))

    def draw(self):
        """Draw a single random sample from the multivariate t distribution.

        Returns
        -------
        Vector
            A random vector drawn from the multivariate t distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> MultivariateT(mean=[0, 0], cov=[[1, 0], [0, 1]], df=5).draw()  # doctest: +SKIP
        (0.42, -1.13)
        """
        return Vector(
            stats.multivariate_t(loc=self._mean, shape=self._cov, df=self._df).rvs(
                random_state=rng
            )
        )


class Wishart(Distribution):
    """Probability space for a Wishart distribution.

    A distribution over symmetric, positive-definite matrices. The Wishart
    generalizes the chi-square distribution to matrices: if you draw ``df``
    independent vectors from a mean-zero :class:`MultivariateNormal` with
    covariance ``scale`` and add up their outer products, the resulting
    ``p x p`` matrix is Wishart-distributed. Because of this, it is the
    standard model for a random covariance (or scatter) matrix, and it is
    the conjugate prior for the *precision* matrix (inverse covariance) of
    a multivariate normal.

    Each draw is a ``p x p`` matrix, returned as a vector of its rows.

    Parameters
    ----------
    df : float
        Degrees of freedom. Must be greater than ``p - 1``, where ``p`` is
        the dimension of ``scale``. With integer ``df`` this is the number
        of normal vectors whose outer products are summed.
    scale : array-like of shape (p, p)
        The scale matrix. Must be symmetric and positive definite. It sets
        the covariance of the underlying normal vectors; the mean of the
        distribution is ``df * scale``.

    Attributes
    ----------
    df : float
        Degrees of freedom.
    scale : array-like of shape (p, p)
        The scale matrix.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Wishart(df=5, scale=[[1, 0], [0, 1]])
    >>> X.draw()  # doctest: +SKIP
    ((3.96, 0.25), (0.25, 3.63))

    See Also
    --------
    InverseWishart : The distribution of the matrix inverse of a Wishart draw.
    """

    def __init__(self, df, scale):
        """Initialize a Wishart distribution.

        Raises
        ------
        Exception
            If ``scale`` is empty, not square, or not symmetric positive
            definite; or if ``df`` is not a number greater than ``p - 1``.
        """
        if len(scale) < 1:
            raise Exception("Scale matrix cannot be empty")

        p = len(scale)
        if not all(len(row) == p for row in scale):
            raise Exception("Scale matrix is not square")

        if not (
            np.all(np.linalg.eigvals(scale) > 0)
            and np.allclose(scale, np.transpose(scale))
        ):
            raise Exception("Scale matrix is not symmetric and positive definite")
        self.scale = scale

        if not isinstance(df, numbers.Real) or df <= p - 1:
            raise Exception(
                "df must be a number greater than one less than the dimension "
                "of the scale matrix (df > %d for a %d x %d scale matrix)"
                % (p - 1, p, p)
            )
        self.df = df

        self.discrete = False
        self.pdf = lambda x: stats.wishart(df=self.df, scale=self.scale).pdf(x)

    def mean(self):
        """Return the mean matrix, ``df * scale``.

        Returns
        -------
        Vector
            The ``p x p`` mean matrix, stored as a vector of its rows -- the
            same shape :meth:`draw` returns, so a draw and the mean can be
            compared entry by entry.

        Examples
        --------
        >>> from symbulate import *
        >>> X = Wishart(df=5, scale=[[1, 0], [0, 1]])
        >>> [[float(v) for v in row] for row in X.mean()]
        [[5.0, 0.0], [0.0, 5.0]]
        """
        matrix = self.df * np.asarray(self.scale, dtype=float)
        return Vector(Vector(row) for row in matrix)

    def _no_vector_summary(self, name):
        """Explain why a vector-shaped summary does not apply to a matrix draw."""
        raise Exception(
            "`%s()` is not available for the Wishart distribution, because "
            "each draw is a whole %d x %d matrix rather than a vector, so "
            "there is no single list of per-component values to report. Use "
            "`.mean()` for the mean matrix (df * scale). To summarize the "
            "spread, simulate matrices and summarize the entries you care "
            "about, e.g. RV(X).sim(1000)." % (name, len(self.scale), len(self.scale))
        )

    def var(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("var")

    def sd(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("sd")

    def cov(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("cov")

    def corr(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("corr")

    def plot(self):
        """Plot is not supported for matrix-valued distributions.

        Raises
        ------
        Exception
            Always raised — plotting is not available for the Wishart
            distribution.
        """
        raise Exception(
            "Plotting is not currently available for the Wishart distribution."
        )

    def draw(self):
        """Draw a single random matrix from the Wishart distribution.

        Returns
        -------
        Vector
            A ``p x p`` symmetric positive-definite matrix, stored as a
            vector of its rows.

        Examples
        --------
        >>> from symbulate import *
        >>> Wishart(df=5, scale=[[1, 0], [0, 1]]).draw()  # doctest: +SKIP
        ((3.96, 0.25), (0.25, 3.63))
        """
        matrix = stats.wishart(df=self.df, scale=self.scale).rvs(random_state=rng)
        return Vector(Vector(row) for row in np.atleast_2d(matrix))

    def __pow__(self, exponent):
        """Draw multiple independent matrices from the Wishart distribution.

        Parameters
        ----------
        exponent : int or float
            Number of matrices to draw. Pass ``float('inf')`` to create an
            infinite sequence of draws generated lazily on demand.

        Returns
        -------
        ProbabilitySpace
            A probability space whose draws produce ``exponent`` matrices
            at a time.

        Examples
        --------
        >>> from symbulate import *
        >>> (Wishart(df=5, scale=[[1, 0], [0, 1]]) ** 3).draw()  # doctest: +SKIP
        [((3.96, 0.25), (0.25, 3.63)), ...]
        """
        if exponent == float("inf"):

            def draw():
                def _func(n):
                    return self.draw()

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)


class InverseWishart(Distribution):
    """Probability space for an inverse-Wishart distribution.

    A distribution over symmetric, positive-definite matrices, obtained by
    inverting a :class:`Wishart` draw: if ``W`` is Wishart with the inverse
    of ``scale``, then ``W`` inverted is inverse-Wishart with this ``scale``.
    It is the standard conjugate prior for the *covariance* matrix of a
    multivariate normal, which makes it a staple of Bayesian multivariate
    models.

    Each draw is a ``p x p`` matrix, returned as a vector of its rows.

    Parameters
    ----------
    df : float
        Degrees of freedom. Must be greater than ``p - 1``, where ``p`` is
        the dimension of ``scale``. The mean exists only when
        ``df > p + 1``, in which case it equals ``scale / (df - p - 1)``.
    scale : array-like of shape (p, p)
        The scale matrix. Must be symmetric and positive definite.

    Attributes
    ----------
    df : float
        Degrees of freedom.
    scale : array-like of shape (p, p)
        The scale matrix.

    Examples
    --------
    >>> from symbulate import *
    >>> X = InverseWishart(df=5, scale=[[1, 0], [0, 1]])
    >>> X.draw()  # doctest: +SKIP
    ((0.31, -0.02), (-0.02, 0.28))

    See Also
    --------
    Wishart : The distribution of the matrix inverse of an inverse-Wishart draw.
    """

    def __init__(self, df, scale):
        """Initialize an inverse-Wishart distribution.

        Raises
        ------
        Exception
            If ``scale`` is empty, not square, or not symmetric positive
            definite; or if ``df`` is not a number greater than ``p - 1``.
        """
        if len(scale) < 1:
            raise Exception("Scale matrix cannot be empty")

        p = len(scale)
        if not all(len(row) == p for row in scale):
            raise Exception("Scale matrix is not square")

        if not (
            np.all(np.linalg.eigvals(scale) > 0)
            and np.allclose(scale, np.transpose(scale))
        ):
            raise Exception("Scale matrix is not symmetric and positive definite")
        self.scale = scale

        if not isinstance(df, numbers.Real) or df <= p - 1:
            raise Exception(
                "df must be a number greater than one less than the dimension "
                "of the scale matrix (df > %d for a %d x %d scale matrix)"
                % (p - 1, p, p)
            )
        self.df = df

        self.discrete = False
        self.pdf = lambda x: stats.invwishart(df=self.df, scale=self.scale).pdf(x)

    def mean(self):
        """Return the mean matrix, ``scale / (df - p - 1)``.

        Returns
        -------
        Vector
            The ``p x p`` mean matrix, stored as a vector of its rows -- the
            same shape :meth:`draw` returns.

        Raises
        ------
        Exception
            If ``df <= p + 1``, where the mean does not exist. (The
            distribution itself is still perfectly well defined there, and
            can still be drawn from -- only its mean is undefined.)

        Examples
        --------
        >>> from symbulate import *
        >>> X = InverseWishart(df=5, scale=[[1, 0], [0, 1]])
        >>> [[float(v) for v in row] for row in X.mean()]
        [[0.5, 0.0], [0.0, 0.5]]
        """
        p = len(self.scale)
        if self.df <= p + 1:
            raise Exception(
                "The mean of an inverse-Wishart exists only for df > p + 1, "
                "where p is the dimension of the scale matrix; here df = %s "
                "and p = %d, so df must be greater than %d. The distribution "
                "can still be drawn from -- only its mean is undefined."
                % (self.df, p, p + 1)
            )
        matrix = np.asarray(self.scale, dtype=float) / (self.df - p - 1)
        return Vector(Vector(row) for row in matrix)

    def _no_vector_summary(self, name):
        """Explain why a vector-shaped summary does not apply to a matrix draw."""
        raise Exception(
            "`%s()` is not available for the inverse-Wishart distribution, "
            "because each draw is a whole %d x %d matrix rather than a vector, "
            "so there is no single list of per-component values to report. Use "
            "`.mean()` for the mean matrix (scale / (df - p - 1)). To summarize "
            "the spread, simulate matrices and summarize the entries you care "
            "about, e.g. RV(X).sim(1000)." % (name, len(self.scale), len(self.scale))
        )

    def var(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("var")

    def sd(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("sd")

    def cov(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("cov")

    def corr(self):
        """Not available for a matrix-valued distribution. See :meth:`mean`."""
        self._no_vector_summary("corr")

    def plot(self):
        """Plot is not supported for matrix-valued distributions.

        Raises
        ------
        Exception
            Always raised — plotting is not available for the
            inverse-Wishart distribution.
        """
        raise Exception(
            "Plotting is not currently available for "
            "the inverse-Wishart distribution."
        )

    def draw(self):
        """Draw a single random matrix from the inverse-Wishart distribution.

        Returns
        -------
        Vector
            A ``p x p`` symmetric positive-definite matrix, stored as a
            vector of its rows.

        Examples
        --------
        >>> from symbulate import *
        >>> InverseWishart(df=5, scale=[[1, 0], [0, 1]]).draw()  # doctest: +SKIP
        ((0.31, -0.02), (-0.02, 0.28))
        """
        matrix = stats.invwishart(df=self.df, scale=self.scale).rvs(random_state=rng)
        return Vector(Vector(row) for row in np.atleast_2d(matrix))

    def __pow__(self, exponent):
        """Draw multiple independent matrices from the inverse-Wishart distribution.

        Parameters
        ----------
        exponent : int or float
            Number of matrices to draw. Pass ``float('inf')`` to create an
            infinite sequence of draws generated lazily on demand.

        Returns
        -------
        ProbabilitySpace
            A probability space whose draws produce ``exponent`` matrices
            at a time.

        Examples
        --------
        >>> from symbulate import *
        >>> (InverseWishart(df=5, scale=[[1, 0], [0, 1]]) ** 3).draw()  # doctest: +SKIP
        [((0.31, -0.02), (-0.02, 0.28)), ...]
        """
        if exponent == float("inf"):

            def draw():
                def _func(n):
                    return self.draw()

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)


class Multinomial(MultivariateDistribution):
    """Probability space for a multinomial distribution.

    Generalizes the binomial distribution to more than two outcomes.
    Models the counts of each outcome across ``n`` independent trials,
    where each trial lands in one of several categories with fixed
    probabilities.

    Parameters
    ----------
    n : int
        Number of trials. Must be a non-negative integer.
    p : array-like of float
        Probability of each outcome. Must be non-negative and sum to 1.

    Attributes
    ----------
    n : int
        Number of trials.
    p : array-like of float
        Probability of each outcome. Must be non-negative and sum to 1.

    Methods
    -------
    mean()
        The mean count vector ``n * p``, as a :class:`Vector`.
    cov()
        The covariance matrix, as a NumPy 2-D array.
    var(), sd()
        The per-category variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Multinomial(n=10, p=[0.5, 0.3, 0.2])
    >>> [float(m) for m in X.mean()]
    [5.0, 3.0, 2.0]
    >>> X.draw()  # doctest: +SKIP
    (5, 3, 2)
    """

    # The counts always add up to n, so the last one is whatever is left
    # over: a three-category multinomial varies in two directions, which
    # makes it the natural single-joint-plot case (two categories is just a
    # binomial). See MultivariateDistribution._free_dim.
    _sum_constrained = True

    def __init__(self, n, p):
        """Initialize a multinomial distribution.

        Raises
        ------
        Exception
            If ``n`` is not a non-negative integer, or the elements of ``p``
            are not non-negative numbers summing to 1.
        """
        # ``p`` is array-like, so guard the sum/min so a non-numeric or empty
        # ``p`` reports the helpful message instead of a cryptic error and can
        # be stacked alongside an invalid ``n``.
        try:
            bad_p = not (math.isclose(sum(p), 1, abs_tol=1e-9) and min(p) >= 0)
        except (TypeError, ValueError):
            bad_p = True

        _validate(
            (
                not isinstance(n, numbers.Integral) or n < 0,
                "n must be a non-negative integer",
            ),
            (bad_p, "Elements of p must be non-negative and sum to 1."),
        )
        self.n = n
        self.p = p

        # A multinomial draw is a vector of counts, so the distribution is
        # discrete: plot() draws a probability at each possible pair of
        # counts rather than a smooth surface between them.
        self.discrete = True
        self.pmf = lambda x: stats.multinomial(n, p).pmf(x)
        self.pdf = self.pmf  # pdf as an alias for pmf, as in Distribution

    def mean(self):
        """Return the mean count vector.

        Returns
        -------
        Vector
            The expected count of each category, ``n * p``.
        """
        return Vector(stats.multinomial(self.n, self.p).mean())

    def cov(self):
        """Return the covariance matrix.

        The counts are negatively correlated because they must sum to ``n``:
        the diagonal entries are ``n * p_i * (1 - p_i)`` and the off-diagonal
        entries are ``-n * p_i * p_j``.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        return np.asarray(stats.multinomial(self.n, self.p).cov(), dtype=float)

    def _marginal_1d(self, i):
        """Return the distribution of category ``i``'s count on its own.

        Ignoring which of the other categories a trial landed in, each
        trial either lands in category ``i`` or it doesn't -- so that one
        count is a binomial with the same number of trials and that
        category's own probability.

        Parameters
        ----------
        i : int
            Index of the category, counting from 0.

        Returns
        -------
        Binomial
            The distribution of that category's count.
        """
        return Binomial(n=self.n, p=float(np.asarray(self.p, dtype=float)[i]))

    def _plot_window(self, i):
        """Return the plotting window for category ``i``'s count.

        A count can be anything from 0 to ``n``, and showing that whole
        range is what makes the triangular shape of the joint support
        visible (two counts can't add up to more than ``n``). But a joint
        plot draws one cell per pair of counts, so the full range only fits
        while ``n`` is small: past that, this falls back to the window
        holding most of the probability -- the same zoomed window a
        one-dimensional plot uses when its bounds stop being informative --
        so a distribution with many trials still plots, showing the region
        the counts actually land in.

        Parameters
        ----------
        i : int
            Index of the category, counting from 0.

        Returns
        -------
        tuple of float
            The ``(low, high)`` range of counts to draw cells for.
        """
        marginal = self._marginal_1d(i)
        low, high = marginal.xlim
        n_values = int(high) - int(low) + 1
        if n_values * n_values <= JOINT_PMF_MAX_CELLS:
            return (low, high)
        # `_zoom_xlim` is the tight, most-of-the-probability window -- the
        # replacement for the highest-density helper this line was originally
        # written against, in a branch that predated that helper's removal.
        return marginal._zoom_xlim()

    def _joint_func(self, i, j):
        """Return the joint probability function of counts ``i`` and ``j``.

        Pooling every other category into a single "everything else"
        category gives back a three-category multinomial exactly -- the
        aggregation property of the multinomial -- so the joint
        distribution of two counts is that smaller multinomial's, read off
        at ``(x, y, n - x - y)``. Pairs that would need more than ``n``
        trials between them are impossible and get probability 0, which is
        what makes the triangular shape of the joint support visible.

        Parameters
        ----------
        i, j : int
            Indices of the two categories, counting from 0.

        Returns
        -------
        callable
            The joint probability function, as a function of two
            equal-length flat arrays of counts.
        """
        p = np.asarray(self.p, dtype=float)
        # The pooled category's probability is whatever is left over. Clamp
        # at 0 so floating-point round-off in the subtraction can't make it
        # a tiny negative number, which is not a valid probability.
        rest = max(0.0, 1.0 - p[i] - p[j])
        pair = stats.multinomial(self.n, [p[i], p[j], rest])
        n = self.n

        def func(x, y):
            x = np.ravel(x)
            y = np.ravel(y)
            pooled = n - x - y
            out = np.zeros(len(x), dtype=float)
            # Only ask scipy about pairs that leave a non-negative count for
            # the pooled category; the rest are impossible, so they keep
            # probability 0.
            possible = pooled >= 0
            if np.any(possible):
                out[possible] = pair.pmf(
                    np.column_stack([x[possible], y[possible], pooled[possible]])
                )
            return out

        return func

    def draw(self):
        """Draw a single random sample from the multinomial distribution.

        Returns
        -------
        Vector
            A vector of counts, one per category, summing to ``n``.

        Examples
        --------
        >>> from symbulate import *
        >>> Multinomial(n=10, p=[0.5, 0.3, 0.2]).draw()  # doctest: +SKIP
        (6, 2, 2)
        """

        return Vector(rng.multinomial(self.n, self.p))


class MultivariateHypergeometric(MultivariateDistribution):
    """Probability space for a multivariate hypergeometric distribution.

    Generalizes the hypergeometric distribution to more than two types. A
    collection contains ``m[i]`` items of type ``i``; you draw ``n`` items
    all at once, without replacement. Each draw is a vector of counts, one
    per type, that sums to ``n``. It is the without-replacement counterpart
    of the ``Multinomial`` -- drawing colored balls from an urn and *not*
    putting them back -- so the counts are negatively correlated.

    Parameters
    ----------
    m : array-like of int
        The number of items of each type in the collection. Must be
        non-negative integers, not all zero.
    n : int
        The number of items drawn (without replacement). Must be a
        non-negative integer no larger than the total ``sum(m)``.

    Attributes
    ----------
    m : array-like of int
        The number of items of each type in the collection.
    n : int
        The number of items drawn.

    Methods
    -------
    mean()
        The mean count vector ``n * m / sum(m)``, as a :class:`Vector`.
    cov()
        The covariance matrix, as a NumPy 2-D array.
    var(), sd()
        The per-type variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.

    Examples
    --------
    >>> from symbulate import *
    >>> X = MultivariateHypergeometric(m=[10, 8, 6], n=6)
    >>> [float(v) for v in X.mean()]
    [2.5, 2.0, 1.5]
    >>> X.draw()  # doctest: +SKIP
    (3, 2, 1)
    """

    # Every draw takes exactly n items, so the counts always add up to n and
    # the last one is whatever is left over: a three-type distribution varies
    # in two directions, which makes it the natural single-joint-plot case
    # (two types is just a Hypergeometric). See
    # MultivariateDistribution._free_dim.
    _sum_constrained = True

    def __init__(self, m, n):
        """Initialize a multivariate hypergeometric distribution.

        Raises
        ------
        Exception
            If ``m`` is not a list of non-negative integers (not all zero),
            or ``n`` is not a non-negative integer no larger than ``sum(m)``.
        """
        # ``m`` is array-like, so guard the conversion and checks the same way
        # ``Multinomial`` guards ``p``: a non-numeric, empty, non-integer, or
        # negative ``m`` reports the helpful message instead of a cryptic error.
        try:
            m_arr = np.asarray(m)
            bad_m = (
                m_arr.ndim != 1
                or len(m_arr) < 1
                or not np.issubdtype(m_arr.dtype, np.integer)
                or np.any(m_arr < 0)
                or m_arr.sum() == 0
            )
        except (TypeError, ValueError):
            bad_m = True
        total = int(m_arr.sum()) if not bad_m else 0

        _validate(
            (
                bad_m,
                "m must be a list of non-negative integers giving the number "
                "of each type in the collection (and not all zero).",
            ),
            (
                not isinstance(n, numbers.Integral)
                or n < 0
                or (not bad_m and n > total),
                "n must be a non-negative integer no larger than the total "
                "number of items sum(m).",
            ),
        )
        self.m = m
        self.n = n

        self.discrete = True
        self.pdf = lambda x: stats.multivariate_hypergeom(self.m, self.n).pmf(x)

    def mean(self):
        """Return the mean count vector.

        Returns
        -------
        Vector
            The expected count of each type, ``n * m / sum(m)``.
        """
        return Vector(stats.multivariate_hypergeom(self.m, self.n).mean())

    def cov(self):
        """Return the covariance matrix.

        The counts are negatively correlated because they must sum to ``n``:
        drawing more of one type leaves fewer draws for the others.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        return np.asarray(
            stats.multivariate_hypergeom(self.m, self.n).cov(), dtype=float
        )

    def _marginal_1d(self, i):
        """Return the distribution of type ``i``'s count on its own.

        Ignoring which of the other types an item happened to be, every
        item drawn either is of type ``i`` or is not -- so that one count
        is an ordinary hypergeometric, drawing the same ``n`` items from a
        collection split into the ``m[i]`` items of that type and all the
        rest lumped together.

        Parameters
        ----------
        i : int
            Index of the type, counting from 0.

        Returns
        -------
        Hypergeometric
            The distribution of that type's count.
        """
        m = np.asarray(self.m, dtype=int)
        return Hypergeometric(n=self.n, N0=int(m.sum() - m[i]), N1=int(m[i]))

    def _joint_func(self, i, j):
        """Return the joint probability function of counts ``i`` and ``j``.

        Pooling every other type into a single "everything else" type gives
        back a three-type multivariate hypergeometric exactly -- the same
        aggregation property the ``Multinomial`` has -- so the joint
        distribution of two counts is that smaller distribution's, read off
        at ``(x, y, n - x - y)``. Pairs that would need more than ``n``
        items between them are impossible and get probability 0, which is
        what makes the triangular shape of the joint support visible.

        Parameters
        ----------
        i, j : int
            Indices of the two types, counting from 0.

        Returns
        -------
        callable
            The joint probability function, as a function of two
            equal-length flat arrays of counts.
        """
        m = np.asarray(self.m, dtype=int)
        rest = int(m.sum() - m[i] - m[j])
        pair = stats.multivariate_hypergeom([int(m[i]), int(m[j]), rest], self.n)
        n = self.n

        def func(x, y):
            x = np.ravel(x)
            y = np.ravel(y)
            pooled = n - x - y
            out = np.zeros(len(x), dtype=float)
            # Only ask scipy about pairs that leave a non-negative count for
            # the pooled type; the rest are impossible, so they keep
            # probability 0.
            possible = pooled >= 0
            if np.any(possible):
                out[possible] = pair.pmf(
                    np.column_stack([x[possible], y[possible], pooled[possible]])
                )
            return out

        return func

    def draw(self):
        """Draw a single random sample from the multivariate hypergeometric distribution.

        Returns
        -------
        Vector
            A vector of counts, one per type, summing to ``n``.

        Examples
        --------
        >>> from symbulate import *
        >>> MultivariateHypergeometric(m=[10, 8, 6], n=6).draw()  # doctest: +SKIP
        (3, 2, 1)
        """
        # scipy's rvs returns a (1, k) array for a single draw, so take row 0.
        sample = stats.multivariate_hypergeom(self.m, self.n).rvs(random_state=rng)
        return Vector(np.atleast_2d(sample)[0])


class NegativeMultinomial(MultivariateDistribution):
    """Probability space for a negative multinomial distribution.

    Generalizes the negative binomial distribution to more than two
    outcomes, the same way the ``Multinomial`` generalizes the binomial.
    Repeat an experiment with several possible outcomes until one
    particular outcome -- the *stopping* category -- has happened ``r``
    times, then count how many times each of the other categories
    happened along the way. Each draw is the vector of those counts.

    Where a ``Multinomial`` fixes the number of trials and lets the counts
    vary, a ``NegativeMultinomial`` fixes the number of times the stopping
    category occurs and lets the number of trials (and so all the other
    counts) vary. That difference flips the sign of the relationship
    between the counts: multinomial counts are negatively correlated,
    because they must add up to a fixed total, while negative multinomial
    counts are *positively* correlated, because a run that takes many
    trials to finish tends to produce more of every category at once.

    Parameters
    ----------
    r : int
        How many times the stopping category has to occur before counting
        stops. Must be a positive integer.
    p : array-like of float
        Probability of each counted category, one entry per category. Must
        be non-negative and sum to strictly less than 1; the leftover
        probability ``1 - sum(p)`` belongs to the stopping category.

    Attributes
    ----------
    r : int
        How many times the stopping category has to occur.
    p : list of float
        Probability of each counted category.
    p0 : float
        Probability of the stopping category, ``1 - sum(p)``.

    Methods
    -------
    mean()
        The mean count vector ``r * p / p0``, as a :class:`Vector`.
    cov()
        The covariance matrix, as a NumPy 2-D array.
    var(), sd()
        The per-category variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.
    pdf(x)
        The probability of a specific vector of counts.

    Notes
    -----
    Because the run stops on the stopping category rather than after a
    fixed number of trials, the counts have no upper bound -- the support
    is infinite, unlike the ``Multinomial``'s. Like the other multivariate
    distributions, there is no ``cdf``: there is no natural way to order
    the vectors it produces.

    Each count ``X_i`` is marginally ``Pascal(r, p0 / (p0 + p_i))`` -- look
    at category ``i`` and the stopping category only, ignore every other
    trial, and what is left is an ordinary wait for the ``r``-th stop.

    Examples
    --------
    >>> from symbulate import *
    >>> X = NegativeMultinomial(r=3, p=[0.3, 0.2])
    >>> [round(float(m), 2) for m in X.mean()]
    [1.8, 1.2]
    >>> X.draw()  # doctest: +SKIP
    (2, 0)

    See Also
    --------
    Multinomial : The fixed-number-of-trials counterpart.
    Pascal : The two-outcome case, and the marginal of each count.
    """

    def __init__(self, r, p):
        """Initialize a negative multinomial distribution.

        Raises
        ------
        Exception
            If ``r`` is not a positive integer, or the elements of ``p`` are
            not non-negative numbers summing to strictly less than 1.
        """
        # ``p`` is array-like, so guard the conversion and checks the same way
        # ``Multinomial`` guards its own ``p``: a non-numeric, empty, or
        # out-of-range ``p`` reports the helpful message instead of a cryptic
        # low-level error, and can be reported alongside an invalid ``r``.
        try:
            p_arr = np.asarray(p, dtype=float)
            p_sum = p_arr.sum()
            bad_p = (
                p_arr.ndim != 1
                or len(p_arr) < 1
                or not np.all(np.isfinite(p_arr))
                or np.any(p_arr < 0)
                or p_sum > 1
                or math.isclose(p_sum, 1, abs_tol=1e-9)
            )
        except (TypeError, ValueError):
            bad_p = True

        _validate(
            (
                not isinstance(r, numbers.Integral) or r <= 0,
                "r must be a positive integer: the number of times the "
                "stopping category has to occur before counting stops.",
            ),
            (
                bad_p,
                "p must be a list of non-negative numbers that sum to less "
                "than 1, one for each category you are counting. The leftover "
                "probability 1 - sum(p) belongs to the stopping category, so "
                "the entries of p cannot add up to 1 or more.",
            ),
        )
        self.r = r
        self.p = list(p)
        self.p0 = float(1 - p_arr.sum())

        # A draw is a vector of counts, so the distribution is discrete:
        # plot() draws a probability at each possible pair of counts rather
        # than a smooth surface between them. ``pdf`` is a method below
        # rather than a lambda, because there is no scipy negative
        # multinomial to delegate to.
        self.discrete = True

    def pdf(self, x):
        """Return the probability of a specific vector of counts.

        Parameters
        ----------
        x : array-like of int
            One count per category, in the same order as ``p``. Unlike the
            ``Multinomial``, the counts do not have to add up to anything in
            particular -- any vector of non-negative whole numbers is
            possible. A 2-D array is also accepted, one vector of counts per
            row.

        Returns
        -------
        float or numpy.ndarray
            The probability of those counts, or 0 for a vector that cannot
            occur (a negative count, or a count that is not a whole number).
            A 2-D input gives one probability per row.

        Raises
        ------
        Exception
            If ``x`` does not have one entry per category.

        Examples
        --------
        >>> from symbulate import *
        >>> round(NegativeMultinomial(r=3, p=[0.3, 0.2]).pdf([0, 0]), 4)
        0.125
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 0 or x_arr.shape[-1] != len(self.p):
            raise Exception(
                "pdf needs one count per category. This distribution has %d "
                "categories, so pass a list of %d whole numbers, for example "
                "pdf(%s)." % (len(self.p), len(self.p), [0] * len(self.p))
            )

        p_arr = np.asarray(self.p, dtype=float)
        # Only whole, non-negative counts are possible; everything else has
        # probability 0. Substitute a 0 for any impossible entry before taking
        # logs so gammaln is never handed a negative number or a nan, then mask
        # those rows back out at the end.
        possible = np.all((x_arr >= 0) & (x_arr == np.floor(x_arr)), axis=-1)
        x_safe = np.where(x_arr >= 0, x_arr, 0.0)

        # Worked on the log scale: the count of trials can be large enough that
        # the gamma functions and the products of probabilities overflow (or
        # round to 0) if computed directly. xlogy handles a category with
        # probability 0 correctly: it contributes nothing when its count is 0,
        # and drives the whole probability to 0 when its count is positive.
        total = x_safe.sum(axis=-1)
        log_pdf = (
            gammaln(self.r + total)
            - gammaln(self.r)
            - gammaln(x_safe + 1).sum(axis=-1)
            + self.r * np.log(self.p0)
            + xlogy(x_safe, p_arr).sum(axis=-1)
        )

        result = np.where(possible, np.exp(log_pdf), 0.0)
        return float(result) if result.ndim == 0 else result

    def mean(self):
        """Return the mean count vector.

        Each category is expected to occur ``r * p_i / p0`` times: the
        stopping category occurs ``r`` times, and category ``i`` occurs
        ``p_i / p0`` times as often as the stopping category does.

        Returns
        -------
        Vector
            The expected count of each category.
        """
        return Vector(self.r * np.asarray(self.p, dtype=float) / self.p0)

    def cov(self):
        """Return the covariance matrix.

        The counts are *positively* correlated, the opposite of the
        ``Multinomial``: nothing caps the total, so a run that happens to
        take many trials to reach the ``r``-th stop inflates every count at
        once. The diagonal entries are ``r * p_i * (p_i + p0) / p0 ** 2``
        and the off-diagonal entries are ``r * p_i * p_j / p0 ** 2``.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        p_arr = np.asarray(self.p, dtype=float)
        return (self.r / self.p0**2) * (
            np.outer(p_arr, p_arr) + np.diag(p_arr * self.p0)
        )

    def _marginal_1d(self, i):
        """Return the distribution of category ``i``'s count on its own.

        Watch category ``i`` and the stopping category, and ignore every
        other draw. What is left is an ordinary wait for the ``r``-th stop,
        so that one count is a ``Pascal`` -- the number of category-``i``
        outcomes seen before the ``r``-th stopping outcome. Ignoring the
        other categories rescales the two probabilities still in play to
        add up to one, which is where ``p0 / (p0 + p[i])`` comes from.

        Parameters
        ----------
        i : int
            Index of the category, counting from 0.

        Returns
        -------
        Pascal
            The distribution of that category's count.
        """
        p = np.asarray(self.p, dtype=float)
        return Pascal(r=self.r, p=self.p0 / (self.p0 + float(p[i])))

    def _joint_func(self, i, j):
        """Return the joint probability function of counts ``i`` and ``j``.

        The same argument as :meth:`_marginal_1d`, one category wider:
        ignoring every category except ``i``, ``j``, and the stopping one
        leaves a two-category negative multinomial with the same ``r``,
        whose probabilities are the three still in play rescaled to add up
        to one. Unlike the ``Multinomial``, there is no upper limit on the
        counts -- the number of draws is what varies here -- so no pair is
        impossible and the support fills the whole quarter-plane.

        Parameters
        ----------
        i, j : int
            Indices of the two categories, counting from 0.

        Returns
        -------
        callable
            The joint probability function, as a function of two
            equal-length flat arrays of counts.
        """
        p = np.asarray(self.p, dtype=float)
        # Rescale over the three probabilities still in play: the two
        # categories being plotted and the stopping category.
        total = self.p0 + float(p[i]) + float(p[j])
        pair = NegativeMultinomial(
            r=self.r, p=[float(p[i]) / total, float(p[j]) / total]
        )

        def func(x, y):
            return np.asarray(
                pair.pdf(np.column_stack([np.ravel(x), np.ravel(y)])), dtype=float
            )

        return func

    def draw(self):
        """Draw a single random sample from the negative multinomial distribution.

        Returns
        -------
        Vector
            A vector of counts, one per category.

        Examples
        --------
        >>> from symbulate import *
        >>> NegativeMultinomial(r=3, p=[0.3, 0.2]).draw()  # doctest: +SKIP
        (2, 0)
        """
        # Drawn as a Poisson-gamma mixture, which reproduces the negative
        # multinomial exactly (and is the same trick that builds an ordinary
        # negative binomial from a Poisson with a gamma-distributed rate, just
        # with one Poisson per category). Simulating the trials one at a time
        # would give the same answer but take a run's worth of draws instead of
        # a fixed handful.
        rate = rng.gamma(shape=self.r, scale=1.0)
        p_arr = np.asarray(self.p, dtype=float)
        return Vector(rng.poisson(rate * p_arr / self.p0))


class Dirichlet(MultivariateDistribution):
    """Probability space for a Dirichlet distribution.

    The multivariate generalization of the beta distribution. Each draw is
    a vector of non-negative numbers that sum to 1, so a Dirichlet is a
    natural model for a random set of proportions -- for example, the mix
    of probabilities across several categories. It is the conjugate prior
    for the ``Multinomial`` distribution, which it pairs with the same way
    the ``Beta`` pairs with the ``Binomial``.

    The single parameter is a vector of concentration parameters
    ``alpha``. Larger values pull draws toward the center of the simplex
    (all proportions roughly equal); values below 1 push mass toward the
    corners (one proportion near 1, the rest near 0). The relative sizes of
    the entries set the average proportions: ``mean = alpha / sum(alpha)``.

    Parameters
    ----------
    alpha : array-like of float
        The concentration parameters, one per category. Must contain at
        least two values, and every value must be strictly positive.

    Attributes
    ----------
    alpha : list of float
        The concentration parameters, one per category.
    alpha0 : float
        The sum of the concentration parameters. Its size controls how
        tightly draws concentrate around the mean.

    Methods
    -------
    mean()
        The mean proportion vector ``alpha / alpha0``, as a :class:`Vector`.
    cov()
        The covariance matrix, as a NumPy 2-D array.
    var(), sd()
        The per-proportion variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.

    Notes
    -----
    A Dirichlet has no simple cumulative distribution function -- there is
    no natural way to order the vectors it produces -- so unlike the
    one-dimensional distributions it provides no ``cdf`` method. Each
    individual proportion ``X_i`` does, however, follow a
    ``Beta(alpha_i, alpha0 - alpha_i)`` distribution, which is what
    ``plot(pairs=True)`` shows down its diagonal.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Dirichlet(alpha=[2, 3, 5])
    >>> [round(float(m), 2) for m in X.mean()]
    [0.2, 0.3, 0.5]
    >>> draw = X.draw()
    >>> round(float(sum(draw)), 10)
    1.0
    >>> X.draw()  # doctest: +SKIP
    (0.19, 0.42, 0.39)
    """

    # The proportions always add up to 1, so the last one is whatever is
    # left over: a three-category Dirichlet varies in two directions, which
    # makes it the natural single-joint-plot case (two categories is just a
    # beta distribution). See MultivariateDistribution._free_dim.
    _sum_constrained = True

    def __init__(self, alpha):
        """Initialize a Dirichlet distribution.

        Raises
        ------
        Exception
            If ``alpha`` is not a list of at least two strictly positive
            numbers.
        """
        # ``alpha`` is array-like, so guard the conversion and checks the
        # same way ``Multinomial`` guards ``p``: a non-numeric, empty, or
        # too-short ``alpha`` reports the helpful message instead of a
        # cryptic low-level error.
        try:
            alpha_arr = np.asarray(alpha, dtype=float)
            bad_alpha = (
                alpha_arr.ndim != 1
                or len(alpha_arr) < 2
                or not np.all(np.isfinite(alpha_arr))
                or np.any(alpha_arr <= 0)
            )
        except (TypeError, ValueError):
            bad_alpha = True

        _validate(
            (
                bad_alpha,
                "alpha must be a list of at least two positive numbers "
                "(the concentration parameters).",
            ),
        )
        self.alpha = list(alpha)
        self.alpha0 = float(alpha_arr.sum())

        # Continuous over the probability simplex. Like the other
        # multivariate distributions (MultivariateNormal, Multinomial), we
        # set up the distribution directly instead of calling
        # super().__init__, whose scalar scipy wiring (a 1-D cdf/quantile,
        # scalar mean/var, an x-limit window) does not apply to a vector of
        # proportions.
        self.discrete = False
        # scipy is the source of truth for the pdf; mean() and cov() are
        # defined as methods below, and var()/sd()/corr() are derived from
        # cov() by the MultivariateDistribution base class.
        self.pdf = lambda x: stats.dirichlet(self.alpha).pdf(x)

    def mean(self):
        """Return the mean proportion vector.

        Returns
        -------
        Vector
            The expected proportion of each category, ``alpha / alpha0``.
        """
        return Vector(stats.dirichlet(self.alpha).mean())

    def cov(self):
        """Return the covariance matrix.

        The proportions are negatively correlated because they must sum to 1.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        return np.asarray(stats.dirichlet(self.alpha).cov(), dtype=float)

    def _marginal_1d(self, i):
        """Return the distribution of proportion ``i`` on its own.

        Pooling every other category together leaves just "this category
        versus the rest", so a single proportion of a Dirichlet is a
        ``Beta(alpha_i, alpha0 - alpha_i)`` on ``[0, 1]`` -- the same way a
        Dirichlet with two categories *is* a beta distribution.

        Parameters
        ----------
        i : int
            Index of the category, counting from 0.

        Returns
        -------
        Beta
            The distribution of that one proportion.
        """
        # alpha0 - alpha_i sums the other (strictly positive) concentrations,
        # so both Beta parameters are positive.
        a_i = float(self.alpha[i])
        return Beta(shape1=a_i, shape2=self.alpha0 - a_i)

    def _plot_window(self, i):
        """Return the plotting window for proportion ``i``.

        Every proportion lives on ``[0, 1]``, and a pair of them lives on
        the triangle where they also sum to at most 1. Framing each axis on
        the full ``[0, 1]`` -- rather than on the window that proportion's
        own beta distribution would use -- keeps that whole triangle in view,
        so the shape of the support a Dirichlet lives on stays visible.

        Parameters
        ----------
        i : int
            Index of the category, counting from 0.

        Returns
        -------
        tuple of float
            The range ``(0, 1)``.
        """
        return (0.0, 1.0)

    def _joint_func(self, i, j):
        """Return the joint density of proportions ``i`` and ``j``.

        Pooling every other category into a single "everything else"
        category gives back a three-category Dirichlet exactly -- the
        aggregation property of the Dirichlet -- so the joint density of two
        proportions is that smaller Dirichlet's, read off at
        ``(x, y, 1 - x - y)``. Pairs summing past 1 are outside the simplex
        and get density 0, which is what draws the triangular support.

        Parameters
        ----------
        i, j : int
            Indices of the two categories, counting from 0.

        Returns
        -------
        callable
            The joint density, as a function of two equal-length flat
            arrays of proportions.
        """
        a = np.asarray(self.alpha, dtype=float)
        pooled_alpha = self.alpha0 - a[i] - a[j]
        pair = stats.dirichlet([a[i], a[j], pooled_alpha])

        def func(x, y):
            x = np.ravel(x)
            y = np.ravel(y)
            pooled = 1.0 - x - y
            out = np.zeros(len(x), dtype=float)
            # scipy's dirichlet rejects points off the open simplex rather
            # than returning 0 for them, so only ask about interior points;
            # everything else is outside the support and keeps density 0.
            inside = (x > 0) & (y > 0) & (pooled > 0)
            if np.any(inside):
                out[inside] = pair.pdf(
                    np.vstack([x[inside], y[inside], pooled[inside]])
                )
            return out

        return func

    def draw(self):
        """Draw a single random sample from the Dirichlet distribution.

        Returns
        -------
        Vector
            A vector of non-negative proportions, one per category, that
            sums to 1.

        Examples
        --------
        >>> from symbulate import *
        >>> Dirichlet(alpha=[2, 3, 5]).draw()  # doctest: +SKIP
        (0.19, 0.42, 0.39)
        """
        return Vector(rng.dirichlet(self.alpha))


class DirichletMultinomial(MultivariateDistribution):
    """Probability space for a Dirichlet-multinomial distribution.

    The compound of a :class:`Dirichlet` and a :class:`Multinomial`: draw a
    probability vector from a ``Dirichlet(alpha)``, then draw counts from a
    ``Multinomial(n, that vector)``. Equivalently, it is the multinomial with
    its probability vector integrated out against a Dirichlet prior. Each draw
    is a vector of counts, one per category, that sums to ``n``.

    Because the category probabilities are themselves random, the counts are
    **overdispersed** relative to an ordinary multinomial -- more spread out
    for the same mean -- which makes this the standard model for
    overdispersed count data in Bayesian text and topic modeling. It is the
    multivariate analogue of the :class:`BetaBinomial`, exactly as the
    ``Dirichlet`` is of the ``Beta``.

    Parameters
    ----------
    n : int
        Number of trials. Must be a non-negative integer.
    alpha : array-like of float
        The Dirichlet concentration parameters, one per category. Must
        contain at least two strictly positive values. Their relative sizes
        set the average category probabilities; their total sets how much the
        probabilities (and hence the counts) vary.

    Attributes
    ----------
    n : int
        Number of trials.
    alpha : list of float
        The Dirichlet concentration parameters.

    Methods
    -------
    mean()
        The mean count vector, as a :class:`Vector`.
    cov()
        The covariance matrix, as a NumPy 2-D array.
    var(), sd()
        The per-category variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.

    Examples
    --------
    >>> from symbulate import *
    >>> X = DirichletMultinomial(n=10, alpha=[2, 3, 5])
    >>> [float(v) for v in X.mean()]
    [2.0, 3.0, 5.0]
    >>> X.draw()  # doctest: +SKIP
    (1, 4, 5)

    See Also
    --------
    Multinomial : The distribution this compounds, with a fixed probability vector.
    Dirichlet : The prior placed on the probability vector.
    BetaBinomial : The univariate analogue.
    """

    # The counts always add up to n, so the last one is whatever is left
    # over: a three-category distribution varies in two directions, which
    # makes it the natural single-joint-plot case (two categories is just a
    # BetaBinomial). See MultivariateDistribution._free_dim.
    _sum_constrained = True

    def __init__(self, n, alpha):
        """Initialize a Dirichlet-multinomial distribution.

        Raises
        ------
        Exception
            If ``n`` is not a non-negative integer, or ``alpha`` is not a
            list of at least two strictly positive numbers.
        """
        # ``alpha`` is array-like, so guard the conversion and checks the same
        # way ``Dirichlet`` does: a non-numeric, empty, too-short, or
        # non-positive ``alpha`` reports the helpful message.
        try:
            alpha_arr = np.asarray(alpha, dtype=float)
            bad_alpha = (
                alpha_arr.ndim != 1
                or len(alpha_arr) < 2
                or not np.all(np.isfinite(alpha_arr))
                or np.any(alpha_arr <= 0)
            )
        except (TypeError, ValueError):
            bad_alpha = True

        _validate(
            (
                not isinstance(n, numbers.Integral) or n < 0,
                "n must be a non-negative integer",
            ),
            (
                bad_alpha,
                "alpha must be a list of at least two positive numbers "
                "(the Dirichlet concentration parameters).",
            ),
        )
        self.n = n
        self.alpha = list(alpha)

        # A draw is a vector of counts, so the distribution is discrete:
        # plot() draws a probability at each possible pair of counts rather
        # than a smooth surface between them.
        self.discrete = True
        self.pdf = lambda x: stats.dirichlet_multinomial(self.alpha, self.n).pmf(x)

    def mean(self):
        """Return the mean count vector.

        Returns
        -------
        Vector
            The expected count of each category, ``n * alpha / sum(alpha)``.
        """
        return Vector(stats.dirichlet_multinomial(self.alpha, self.n).mean())

    def cov(self):
        """Return the covariance matrix.

        The counts are negatively correlated (they must sum to ``n``) and
        overdispersed relative to a multinomial: every variance is inflated by
        the factor ``(n + alpha0) / (1 + alpha0)``, where ``alpha0`` is the
        sum of the concentration parameters.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        return np.asarray(
            stats.dirichlet_multinomial(self.alpha, self.n).cov(), dtype=float
        )

    def _marginal_1d(self, i):
        """Return the distribution of category ``i``'s count on its own.

        Lumping every other category together turns the Dirichlet prior
        into a two-part ``Beta`` and the counts into a two-outcome draw, so
        one count on its own is a beta-binomial -- the univariate analogue
        of this distribution, exactly as one count of a ``Multinomial`` is
        the ``Binomial``.

        Parameters
        ----------
        i : int
            Index of the category, counting from 0.

        Returns
        -------
        BetaBinomial
            The distribution of that category's count.
        """
        alpha = np.asarray(self.alpha, dtype=float)
        return BetaBinomial(
            n=self.n, shape1=float(alpha[i]), shape2=float(alpha.sum() - alpha[i])
        )

    def _joint_func(self, i, j):
        """Return the joint probability function of counts ``i`` and ``j``.

        Pooling every other category into a single "everything else"
        category adds up their concentration parameters and gives back a
        three-category Dirichlet-multinomial exactly -- the same
        aggregation property the ``Multinomial`` and ``Dirichlet`` have --
        so the joint distribution of two counts is that smaller
        distribution's, read off at ``(x, y, n - x - y)``. Pairs adding up
        to more than ``n`` are impossible and get probability 0, which is
        what makes the triangular shape of the joint support visible.

        Parameters
        ----------
        i, j : int
            Indices of the two categories, counting from 0.

        Returns
        -------
        callable
            The joint probability function, as a function of two
            equal-length flat arrays of counts.
        """
        alpha = np.asarray(self.alpha, dtype=float)
        rest = float(alpha.sum() - alpha[i] - alpha[j])
        pair = stats.dirichlet_multinomial(
            [float(alpha[i]), float(alpha[j]), rest], self.n
        )
        n = self.n

        def func(x, y):
            x = np.ravel(x)
            y = np.ravel(y)
            pooled = n - x - y
            out = np.zeros(len(x), dtype=float)
            # Only ask scipy about pairs that leave a non-negative count for
            # the pooled category; the rest are impossible, so they keep
            # probability 0.
            possible = pooled >= 0
            if np.any(possible):
                out[possible] = pair.pmf(
                    np.column_stack([x[possible], y[possible], pooled[possible]])
                )
            return out

        return func

    def draw(self):
        """Draw a single random sample from the Dirichlet-multinomial distribution.

        Draws a probability vector from the ``Dirichlet(alpha)`` prior, then
        draws counts from a ``Multinomial(n, ...)`` with that vector.

        Returns
        -------
        Vector
            A vector of counts, one per category, summing to ``n``.

        Examples
        --------
        >>> from symbulate import *
        >>> DirichletMultinomial(n=10, alpha=[2, 3, 5]).draw()  # doctest: +SKIP
        (1, 4, 5)
        """
        probabilities = rng.dirichlet(self.alpha)
        return Vector(rng.multinomial(self.n, probabilities))


class MultivariateLogNormal(MultivariateDistribution):
    """Probability space for a multivariate log-normal distribution.

    The multivariate generalization of the log-normal: the elementwise
    exponential of a :class:`MultivariateNormal`. If ``Y`` is multivariate
    normal with mean vector ``mean`` and covariance ``cov``, then
    ``X = exp(Y)`` (component by component) is multivariate log-normal. Each
    component is positive, and each is marginally a univariate ``LogNormal``.
    It models several positive, correlated quantities at once -- for example
    correlated prices, incomes, or biological measurements.

    The parameters ``mean`` and ``cov`` describe the *underlying normal*, not
    the log-normal itself. The mean and covariance of the log-normal are
    larger and follow a standard correction (see :meth:`mean` and
    :meth:`cov`), because exponentiating stretches the upper tail.

    Parameters
    ----------
    mean : array-like of length n
        The mean vector of the underlying normal (the mean of ``log(X)``).
    cov : array-like of shape (n, n)
        The covariance matrix of the underlying normal (the covariance of
        ``log(X)``). Must be symmetric and positive semi-definite.

    Methods
    -------
    mean()
        The mean vector of the log-normal, as a :class:`Vector`.
    cov()
        The covariance matrix of the log-normal, as a NumPy 2-D array.
    var(), sd()
        The per-component variances and standard deviations, as ``Vector``\\ s.
    corr()
        The correlation matrix, as a NumPy 2-D array.

    Examples
    --------
    >>> from symbulate import *
    >>> X = MultivariateLogNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    >>> X.draw()  # doctest: +SKIP
    (0.72, 3.41)

    See Also
    --------
    MultivariateNormal : The distribution whose exponential this is.
    LogNormal : The univariate log-normal that each component follows.
    """

    def __init__(self, mean, cov):
        """Initialize a multivariate log-normal distribution.

        Raises
        ------
        Exception
            If the mean vector is empty; if the mean and covariance matrix
            have incompatible sizes; if the covariance matrix is not square;
            or if it is not symmetric positive semi-definite. (These are the
            same requirements as the underlying multivariate normal.)
        """
        # The log-normal is exp() of this normal; composing it reuses all of
        # the underlying normal's validation (shape, symmetry, PSD) and its
        # sampling, so nothing is duplicated here.
        self._normal = MultivariateNormal(mean, cov)
        self.discrete = False
        # The density on the positive orthant is the normal density of log(x),
        # divided by the Jacobian product of the components. The product runs
        # over the last axis only -- the components of one vector -- so a 2-D
        # input (one vector of values per row) divides each row by its own
        # Jacobian instead of by the product of the whole array.
        self.pdf = lambda x: self._normal.pdf(np.log(x)) / np.prod(
            np.asarray(x, dtype=float), axis=-1
        )

    def mean(self):
        """Return the mean vector of the log-normal.

        This is not the exponential of the underlying mean: each component is
        ``exp(mean_i + cov_ii / 2)``, larger than ``exp(mean_i)`` because
        exponentiating stretches the upper tail.

        Returns
        -------
        Vector
            The mean of each component.
        """
        mu = np.asarray(self._normal.mean(), dtype=float)
        variances = np.diag(self._normal.cov())
        return Vector(np.exp(mu + variances / 2))

    def cov(self):
        """Return the covariance matrix of the log-normal.

        Each entry is
        ``exp(mean_i + mean_j + (cov_ii + cov_jj) / 2) * (exp(cov_ij) - 1)``,
        the standard transform of the underlying normal's covariance.

        Returns
        -------
        numpy.ndarray
            The covariance matrix.
        """
        mu = np.asarray(self._normal.mean(), dtype=float)
        sigma = self._normal.cov()
        variances = np.diag(sigma)
        return np.exp(np.add.outer(mu, mu) + np.add.outer(variances, variances) / 2) * (
            np.exp(sigma) - 1
        )

    def _marginal_1d(self, i):
        """Return the distribution of variable ``i`` on its own.

        Taking the logarithm of a multivariate log-normal gives back a
        multivariate normal, one variable of which is an ordinary
        ``Normal`` -- so one variable of this distribution is an ordinary
        ``LogNormal``, built from that variable's own entry of the
        underlying mean vector and covariance matrix.

        Parameters
        ----------
        i : int
            Index of the variable, counting from 0.

        Returns
        -------
        LogNormal
            The distribution of that one variable.
        """
        mu = np.asarray(self._normal.mean(), dtype=float)
        cov = np.asarray(self._normal.cov(), dtype=float)
        return LogNormal(mu=float(mu[i]), sigma=float(np.sqrt(cov[i, i])))

    def _joint_func(self, i, j):
        """Return the joint density of variables ``i`` and ``j``.

        Two variables of the underlying normal are themselves a bivariate
        normal -- those two entries of its mean vector and the matching 2x2
        block of its covariance matrix -- so the pair here is a bivariate
        log-normal. Its density is that bivariate normal's, evaluated at
        the logarithms and divided by ``x * y``, the Jacobian of the
        exponential change of variables.

        The density is 0 anywhere outside the positive quarter-plane, since
        a log-normal value can never be zero or negative.

        Parameters
        ----------
        i, j : int
            Indices of the two variables, counting from 0.

        Returns
        -------
        callable
            The joint density, as a function of two equal-length flat
            arrays of coordinates.
        """
        mu = np.asarray(self._normal.mean(), dtype=float)
        cov = np.asarray(self._normal.cov(), dtype=float)
        index = [i, j]
        pair = stats.multivariate_normal(mu[index], cov[np.ix_(index, index)])

        def func(x, y):
            x = np.ravel(x)
            y = np.ravel(y)
            out = np.zeros(len(x), dtype=float)
            # Only take logarithms where both coordinates are strictly
            # positive; everywhere else is outside the support and stays 0.
            inside = (x > 0) & (y > 0)
            if np.any(inside):
                xs = x[inside]
                ys = y[inside]
                out[inside] = pair.pdf(np.column_stack([np.log(xs), np.log(ys)])) / (
                    xs * ys
                )
            return out

        return func

    def draw(self):
        """Draw a single random sample from the multivariate log-normal distribution.

        Returns
        -------
        Vector
            A random vector of positive values -- the elementwise exponential
            of a draw from the underlying multivariate normal.

        Examples
        --------
        >>> from symbulate import *
        >>> MultivariateLogNormal(mean=[0, 0], cov=[[1, 0], [0, 1]]).draw()  # doctest: +SKIP
        (0.72, 3.41)
        """
        return Vector(np.exp(np.asarray(self._normal.draw(), dtype=float)))
