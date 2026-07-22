import math
import numbers
import warnings
import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt

from .probability_space import ProbabilitySpace
from .plot import (
    get_next_color,
    DistributionPlot,
    ECDF_LINEWIDTH,
    SHADE_COLOR,
    SHADE_ALPHA,
    TRUE_DIST_MARKER_SIZE,
    TRUE_DIST_LINEWIDTH,
    TRUE_DIST_LINESTYLE,
)
from .result import Scalar, Vector, InfiniteVector

rng = np.random.default_rng()


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


# Share of the total probability the default plotting window should cover.
# Matches the central 99.8% the old equal-tailed ppf(0.001)/ppf(0.999) default
# spanned, so switching a distribution to a highest-density window keeps the
# same amount of mass on screen -- only its placement changes.
_PLOT_COVERAGE = 0.998


def _discrete_hdi_xlim(dist, low, coverage=_PLOT_COVERAGE):
    """Highest-density plotting x-limits for an unbounded discrete distribution.

    Returns the smallest and largest of the fewest integer values whose
    probabilities, taken largest first, together reach ``coverage`` of the
    total mass -- an exact highest-density interval. For a right-skewed
    distribution (e.g. a low-rate ``Poisson``) this keeps the dense low
    values and trims the long, thin upper tail, unlike the equal-tailed
    ``ppf(0.001)``/``ppf(0.999)`` window it replaces.

    Parameters
    ----------
    dist : Distribution
        The distribution to measure, read through its ``pmf`` and
        ``quantile``.
    low : int
        The smallest value in the distribution's support (0 for ``Poisson``
        and ``Pascal``, 1 for ``Geometric``, ``r`` for ``NegativeBinomial``).
    coverage : float, optional
        Target share of the total probability the interval must cover.
        Defaults to ``_PLOT_COVERAGE`` (the central 99.8% the old
        equal-tailed default spanned).

    Returns
    -------
    tuple of (int, int)
        ``(min, max)`` of the selected highest-probability values.
    """
    # Enumerate from the support minimum out to a far upper quantile -- past
    # where any high-probability value could sit -- so the accumulation below
    # sees every value that could belong to the interval.
    high = max(int(dist.quantile(1 - 1e-6)), low)
    values = np.arange(low, high + 1)
    probs = dist.pmf(values)
    # Take values in order of decreasing probability, stopping as soon as the
    # accumulated mass first reaches the target coverage.
    order = np.argsort(probs)[::-1]
    accumulated = np.cumsum(probs[order])
    count = min(int(np.searchsorted(accumulated, coverage)) + 1, len(values))
    chosen = values[order[:count]]
    return (int(chosen.min()), int(chosen.max()))


def _continuous_hdi_xlim(dist, low, coverage=_PLOT_COVERAGE):
    """Highest-density plotting x-limits for a skewed, unbounded continuous distribution.

    Finds the horizontal density level ``h`` for which the region
    ``{x : pdf(x) >= h}`` -- a single interval ``[a, b]`` for these unimodal
    densities -- encloses ``coverage`` of the total probability, and returns
    that interval. The endpoints ``a`` and ``b`` are the two equal-density
    points found with :func:`scipy.optimize.brentq`; ``h`` is located by
    bisection, since raising it shrinks the interval and lowers its coverage.
    For a right-skewed density this trims the long upper tail and lifts the
    lower edge off the near-zero-density region, unlike the equal-tailed
    ``ppf(0.001)``/``ppf(0.999)`` window it replaces.

    Handles both interior-mode densities (e.g. ``LogNormal``) and
    monotone-decreasing ones (e.g. ``Pareto``, or ``Gamma`` with shape below
    1): when the density at the lower support bound already exceeds ``h`` --
    including when it is infinite there -- that bound is used as ``a``
    directly, since the density is maximal there and no left equal-density
    point exists. Using the true support bound (not a near-boundary quantile)
    also keeps ``a`` at a clean value and lets the plot's own ``isfinite``
    filter handle a density singularity at the bound.

    Parameters
    ----------
    dist : Distribution
        The distribution to measure, read through its ``pdf``, ``cdf`` and
        ``quantile``.
    low : float
        The lower bound of the distribution's support (0 for ``Gamma``,
        ``ChiSquare``, ``F`` and ``LogNormal``; ``scale`` for ``Pareto``).
    coverage : float, optional
        Target share of the total probability the interval must cover.
        Defaults to ``_PLOT_COVERAGE``.

    Returns
    -------
    tuple of (float, float)
        The equal-density interval ``(a, b)`` enclosing ``coverage``.
    """
    pdf, cdf, quantile = dist.pdf, dist.cdf, dist.quantile
    # Mode-search bounds sit just inside each tail, so the density is never
    # sampled at a boundary where it may be infinite; the right bound also
    # serves as the practical upper edge for a heavy tail.
    search_low = quantile(1e-9)
    right = quantile(1 - 1e-9)
    # The mode is where the density peaks; for a monotone-decreasing density
    # this lands at the left search bound.
    mode = minimize_scalar(
        lambda x: -pdf(x), bounds=(search_low, right), method="bounded"
    ).x
    peak = pdf(mode)
    # Bisect on the density threshold h in (0, peak). At each h the interval is
    # [a, b], the equal-density points on either side of the mode -- clamped to
    # the support's lower bound / practical upper edge when the density there
    # already exceeds h (as on either side of a monotone-decreasing density).
    # Raising h shrinks [a, b] and lowers cdf(b) - cdf(a), so coverage is
    # monotone in h and the bisection converges.
    h_low, h_high = 0.0, peak
    a, b = low, right
    for _ in range(80):
        h = 0.5 * (h_low + h_high)
        a = low if pdf(low) >= h else brentq(lambda x: pdf(x) - h, low, mode)
        b = right if pdf(right) >= h else brentq(lambda x: pdf(x) - h, mode, right)
        if cdf(b) - cdf(a) > coverage:
            h_low = h
        else:
            h_high = h
    return (float(a), float(b))


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
        Default x-axis range used when plotting.
    """

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

        self.xlim = (scipy.ppf(0.001, **self.params), scipy.ppf(0.999, **self.params))

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

    def spinner(self, mode="proportional"):
        """Display a static probability spinner wheel for this distribution.

        Parameters
        ----------
        mode : {'proportional', 'equal'}, optional
            'proportional' sizes each slice by its probability (default).
            'equal' gives every slice the same arc; for discrete distributions
            likelier values appear on more sections.

        Examples
        --------
        >>> from symbulate import *
        >>> Normal(0, 1).spinner()
        >>> Binomial(10, 0.3).spinner(mode='equal')
        >>> Poisson(4).spinner()
        """
        from .spinner import show_spinner

        show_spinner(self, mode=mode)

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

    def _hdi_window(self, coverage=_PLOT_COVERAGE):
        """Compute the highest-density x-range covering a share of the probability.

        Returns the tightest range of x-values that together hold
        ``coverage`` of the probability, reusing the same helpers that set
        the default window for unbounded distributions
        (:func:`_discrete_hdi_xlim` and :func:`_continuous_hdi_xlim`). Used
        by :meth:`plot` for ``xlim="zoom"``, so a curve can be framed on
        the region where the probability actually lives -- even for a
        bounded distribution whose default window spans its full support.

        Parameters
        ----------
        coverage : float, optional
            Share of the total probability to enclose, strictly between 0
            and 1. Defaults to ``_PLOT_COVERAGE``.

        Returns
        -------
        tuple of float
            The ``(low, high)`` x-range enclosing ``coverage`` of the
            probability.
        """
        # The support's lower bound. For a discrete distribution scipy's
        # quantile(0) sits one step below it -- a safe (never too high)
        # start for the enumeration; for a distribution unbounded below it
        # is -inf, so fall back to a far-left finite quantile.
        low = self.quantile(0.0)
        if not np.isfinite(low):
            low = self.quantile(1e-9)
        if self.discrete:
            return _discrete_hdi_xlim(self, int(np.floor(low)), coverage)
        return _continuous_hdi_xlim(self, low, coverage)

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

        The plot is titled by what it shows: "CDF Plot" for ``cdf=True``,
        and for the default view "PDF Plot" (continuous) or "PMF Plot"
        (discrete).

        Parameters
        ----------
        xlim : tuple of float or "zoom", optional
            x-axis range. ``None`` (default) uses the distribution's own
            window: the full support when both ends are bounded (e.g.
            ``Binomial``, ``Uniform``), and a window holding most of the
            probability where a side is unbounded (e.g. ``Poisson``,
            ``Normal``). A ``(min, max)`` tuple sets an exact range.
            ``"zoom"`` frames the plot on the tightest window holding most
            of the probability -- the same high-density window, applied
            even to a bounded distribution whose default shows full
            support. Handy for lining a theoretical curve up against
            simulated data, which occupies only the high-probability part
            of the support.
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
        >>> Binomial(100, 0.5).plot(xlim="zoom")  # tight window, not (0, 100)  # doctest: +SKIP
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
        #   None        -> the distribution's default window (full support
        #                  when bounded, a probability cut where unbounded);
        #   "zoom"      -> the tightest window holding _PLOT_COVERAGE of the
        #                  probability, zooming in even on a bounded default;
        #   (low, high) -> those exact limits, used as given.
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
            xlim = self._hdi_window()

        # get the x and y values. The x-window is chosen the same way for
        # both plot types (it only picks x-values); `cdf` decides which
        # function is evaluated there.
        if self.discrete:
            xs = np.arange(int(xlim[0]), int(xlim[1]) + 1)
        else:
            xs = np.linspace(xlim[0], xlim[1], 200)
        ys = self.cdf(xs) if cdf else self.pdf(xs)

        # determine limits for y-axes based on y values. Anchor the baseline
        # at exactly 0 so the curve sits right on the x-axis: a pdf/pmf height
        # is never negative and only reads correctly against a zero baseline,
        # and a CDF likewise runs from 0 upward. Padding below 0 would float
        # the curve off the axis and misrepresent it.
        ymax = ys[np.isfinite(ys)].max()
        ylim = 0, 1.05 * ymax

        # get the current axis if they exist and no axis is specified
        fig = plt.gcf()
        if ax is None and fig.axes:
            ax = plt.gca()

        # if axis already exists, set it to the union of existing and current axis
        if ax is not None:
            xlower, xupper = ax.get_xlim()
            xlim = min(xlim[0], xlower), max(xlim[1], xupper)
            ylower, yupper = ax.get_ylim()
            ylim = min(ylim[0], ylower), max(ylim[1], yupper)
        else:
            ax = plt.gca()  # creates new axis

        # set the axis limits
        if xlim[0] == xlim[1]:
            # A window can collapse onto a single value when one outcome
            # carries essentially all the probability (e.g. Geometric(0.99),
            # or a bounded distribution under xlim="zoom"). Give the lone point
            # room so the axis stays well-formed instead of singular.
            xlim = (xlim[0] - 0.5, xlim[1] + 0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # get next color in cycle
        color = get_next_color(ax)

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
        if cdf:
            ax.set_title("CDF Plot")
        elif self.discrete:
            ax.set_title("PMF Plot")
        else:
            ax.set_title("PDF Plot")

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
        self.xlim = (
            0,
            1,
        )  # Bernoulli distributions are not defined for x < 0 and x > 1


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
        self.xlim = (0, n)  # Binomial distributions are not defined for x < 0 and x > n


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
        self.xlim = (
            0,
            n,
        )  # Hypergeometric distributions are not defined for x < 0 and x > n


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
        # Highest-density window over the support [1, inf); trims the long
        # upper tail the old equal-tailed ppf window left on screen.
        self.xlim = _discrete_hdi_xlim(self, 1)


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
        # Highest-density window over the support [r, inf); trims the long
        # upper tail the old equal-tailed ppf window left on screen.
        self.xlim = _discrete_hdi_xlim(self, r)

    def draw(self):
        """Draw a single random sample from the negative binomial distribution.

        Returns
        -------
        int
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
        return self.r + rng.negative_binomial(n=self.r, p=self.p)


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
        # Highest-density window over the support [0, inf); trims the long
        # upper tail the old equal-tailed ppf window left on screen.
        self.xlim = _discrete_hdi_xlim(self, 0)


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
        # Highest-density window over the support [0, inf); trims the long
        # upper tail the old equal-tailed ppf window left on screen.
        self.xlim = _discrete_hdi_xlim(self, 0)


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
        self.b = b + 1

        params = {"low": self.a, "high": self.b}

        super().__init__(params, stats.randint, True)
        self.xlim = (a, b)  # Uniform distributions are not defined for x < a and x > b


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
        self.xlim = (a, b)  # Uniform distributions are not defined for x < a and x > b


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
        # Highest-density window over the support [0, inf); trims the long
        # upper tail and covers the same probability as the other one-sided
        # distributions, instead of the wider equal-tailed default.
        self.xlim = _continuous_hdi_xlim(self, 0)


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
        # Highest-density window: trims the long right tail and, for shapes
        # above 1, lifts the left edge off the near-zero-density region.
        self.xlim = _continuous_hdi_xlim(self, 0)


class Beta(Distribution):
    """Probability space for a beta distribution.

    A continuous distribution defined on [0, 1], often used to model
    probabilities or proportions. The shape changes with parameters
    ``a`` and ``b``.

    Parameters
    ----------
    a : float
        First shape parameter (α). Must be positive.
    b : float
        Second shape parameter (β). Must be positive.

    Attributes
    ----------
    a : float
        First shape parameter (α). Must be positive.
    b : float
        Second shape parameter (β). Must be positive.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Beta(a=1, b=1)
    >>> float(X.mean())
    0.5
    >>> round(float(X.pdf(0.5)), 4)
    1.0
    >>> X.draw()  # doctest: +SKIP
    0.632
    """

    def __init__(self, a, b):
        """Initialize a beta distribution.

        Raises
        ------
        Exception
            If ``a`` or ``b`` is not a positive number.
        """

        _validate(
            (not isinstance(a, numbers.Real) or a <= 0, "a must be a positive number"),
            (not isinstance(b, numbers.Real) or b <= 0, "b must be a positive number"),
        )
        self.a = a
        self.b = b

        params = {"a": a, "b": b}
        super().__init__(params, stats.beta, False)
        self.xlim = (0, 1)  # Beta distributions are not defined for x < 0 and x > 1


class StudentT(Distribution):
    """Probability space for Student's t-distribution.

    A bell-shaped distribution similar to the normal but with heavier
    tails. Commonly used in statistical inference when the sample size
    is small. As degrees of freedom increase, it approaches the normal
    distribution.

    Parameters
    ----------
    df : int or float
        Degrees of freedom. Must be positive.

    Attributes
    ----------
    df : int or float
        Degrees of freedom. Must be positive.

    Notes
    -----
    The mean is undefined for ``df = 1`` (the Cauchy case); ``mean()``,
    ``var()``, and ``sd()`` return ``nan`` there.

    Examples
    --------
    >>> from symbulate import *
    >>> X = StudentT(df=10)
    >>> float(X.mean())
    0.0
    >>> X.draw()  # doctest: +SKIP
    0.312
    """

    def __init__(self, df):
        """Initialize a Student's t-distribution.

        Raises
        ------
        Exception
            If ``df`` is not a positive number.
        """
        _validate(
            (
                not isinstance(df, numbers.Real) or df <= 0,
                "df must be a positive number",
            ),
        )
        self.df = df

        params = {"df": df}
        super().__init__(params, stats.t, False)
        if df == 1:
            self.mean = lambda: float("nan")
            self.sd = lambda: float("nan")
            self.var = lambda: float("nan")


class ChiSquare(Distribution):
    """Probability space for a chi-square distribution.

    Arises as the sum of squares of independent standard normal random
    variables. Commonly used in hypothesis testing and confidence
    intervals for variance.

    Parameters
    ----------
    df : int
        Degrees of freedom. Must be a positive integer.

    Attributes
    ----------
    df : int
        Degrees of freedom. Must be a positive integer.

    Examples
    --------
    >>> from symbulate import *
    >>> X = ChiSquare(df=4)
    >>> float(X.mean())
    4.0
    >>> float(X.sd())
    2.8284271247461903
    >>> X.draw()  # doctest: +SKIP
    3.14
    """

    def __init__(self, df):
        """Initialize a chi-square distribution.

        Raises
        ------
        Exception
            If ``df`` is not a positive integer.
        """
        _validate(
            (
                not isinstance(df, numbers.Integral) or df <= 0,
                "df must be a positive integer",
            ),
        )
        self.df = df

        params = {"df": df}
        super().__init__(params, stats.chi2, False)
        # Highest-density window: trims the long right tail and, for df above
        # 2, lifts the left edge off the near-zero-density region.
        self.xlim = _continuous_hdi_xlim(self, 0)


class F(Distribution):
    """Probability space for an F-distribution.

    Arises as the ratio of two chi-square random variables divided by
    their degrees of freedom. Commonly used in analysis of variance
    (ANOVA) to compare group variances.

    Parameters
    ----------
    dfN : int or float
        Degrees of freedom for the numerator. Must be positive.
    dfD : int or float
        Degrees of freedom for the denominator. Must be positive.

    Attributes
    ----------
    dfN : int or float
        Degrees of freedom for the numerator.
    dfD : int or float
        Degrees of freedom for the denominator.

    Examples
    --------
    >>> from symbulate import *
    >>> X = F(dfN=5, dfD=10)
    >>> float(X.mean())
    1.25
    >>> X.draw()  # doctest: +SKIP
    0.85
    """

    def __init__(self, dfN, dfD):
        """Initialize an F-distribution.

        Raises
        ------
        Exception
            If ``dfN`` or ``dfD`` is not a positive number.
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
        )
        self.dfN = dfN
        self.dfD = dfD

        params = {"dfn": dfN, "dfd": dfD}
        super().__init__(params, stats.f, False)
        # Highest-density window: trims the long right tail and, for numerator
        # df above 2, lifts the left edge off the near-zero-density region.
        self.xlim = _continuous_hdi_xlim(self, 0)


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
        float
            One random value drawn from the Cauchy distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> Cauchy().draw()  # doctest: +SKIP
        1.48
        """
        return self.loc + (self.scale * rng.standard_cauchy())


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
            _value = np.exp(mu)
            self.s = 0
            self.norm_sd = 0
            self.discrete = False
            self.params = {"s": 0, "scale": _value}
            self.pdf = lambda x: float(x == _value)
            self.cdf = lambda x: 0.0 if x < _value else 1.0
            self.mean = lambda: _value
            self.var = lambda: 0.0
            self.sd = lambda: 0.0
            self.median = lambda: _value
            self.xlim = (0, _value + 1)
            ProbabilitySpace.__init__(self, lambda: Scalar(_value))
            return
        else:
            self.s = sigma
            self.norm_sd = sigma

        params = {"s": self.s, "scale": np.exp(mu)}
        super().__init__(params, stats.lognorm, False)
        # Highest-density window over the interior-mode density: trims the long
        # right tail and lifts the left edge off the near-zero-density region.
        self.xlim = _continuous_hdi_xlim(self, 0)


class Pareto(Distribution):
    """Probability space for a Pareto distribution.

    A heavy-tailed distribution often used to model phenomena where a
    small fraction of items account for a large share of the effect
    (the "80/20 rule"). All values are at least ``scale``.

    Parameters
    ----------
    b : float, optional
        Shape parameter (tail index). Must be positive. Default is 1.0.
    scale : float, optional
        Minimum possible value (lower bound of the support).
        Must be positive. Default is 1.0.

    Attributes
    ----------
    b : float
        Shape parameter (tail index). Larger values give thinner tails.
    scale : float
        Minimum possible value (lower bound of the support).

    Notes
    -----
    The mean is finite only when ``b > 1``; for ``b <= 1`` it diverges to
    ``inf``. The variance is finite only when ``b > 2``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Pareto(b=2, scale=1)
    >>> float(X.mean())
    2.0
    >>> X.draw()  # doctest: +SKIP
    1.34
    """

    def __init__(self, b=1.0, scale=1.0):
        """Initialize a Pareto distribution.

        Raises
        ------
        Exception
            If ``b`` or ``scale`` is not a positive number.
        """

        _validate(
            (not isinstance(b, numbers.Real) or b <= 0, "b must be a positive number"),
            (
                not isinstance(scale, numbers.Real) or scale <= 0,
                "scale must be a positive number",
            ),
        )
        self.b = b
        self.scale = scale

        params = {"b": self.b, "scale": self.scale}
        super().__init__(params, stats.pareto, False)
        # Highest-density window over the monotone-decreasing density: keeps
        # the peak at the lower bound (scale) and trims the long right tail.
        self.xlim = _continuous_hdi_xlim(self, scale)

    def draw(self):
        """Draw a single random sample from the Pareto distribution.

        Returns
        -------
        float
            One random value drawn from the Pareto distribution.
            Always greater than or equal to ``scale``.

        Examples
        --------
        >>> from symbulate import *
        >>> Pareto(b=2, scale=1).draw()  # doctest: +SKIP
        1.42
        """

        # Numpy's Pareto is Lomax distribution, or Type II Pareto
        # but we want the more standard parametrization
        return self.scale * (1 + rng.pareto(self.b))


class Rayleigh(Distribution):
    """Probability space for a Rayleigh distribution.

    Arises when computing the magnitude of a two-dimensional vector
    whose components are independent, identically distributed normal
    random variables. Often used in signal processing.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Rayleigh()
    >>> round(float(X.mean()), 4)
    1.2533
    >>> X.draw()  # doctest: +SKIP
    0.88
    """

    def __init__(self):
        """Initialize a Rayleigh distribution."""
        params = {}
        super().__init__(params, stats.rayleigh, False)
        # Highest-density window over the support [0, inf); keeps the lower
        # edge at the true bound rather than the equal-tailed 0.1st-percentile
        # start the base default would use.
        self.xlim = _continuous_hdi_xlim(self, 0)


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
        # Highest-density window: trims the long right tail, like Gamma.
        self.xlim = _continuous_hdi_xlim(self, 0)


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
        # Symmetric and unbounded on both sides, so the base equal-tailed
        # ppf(.001, .999) window is already centered and appropriate -- no
        # HDI trim (that is for one-sided skew like Gamma/Weibull).
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
        # Highest-density window: trims the right tail, like Weibull/Gamma.
        self.xlim = _continuous_hdi_xlim(self, 0)


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
        # Symmetric and unbounded on both sides, so the base equal-tailed
        # ppf(.001, .999) window is already centered and appropriate -- no
        # HDI trim (that is for one-sided skew like Gamma/Weibull).
        super().__init__(params, stats.laplace, False)


## Multivariate Distributions


class MultivariateNormal(Distribution):
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

    Attributes
    ----------
    mean : array-like of length n
        The mean vector of the distribution.
    cov : array-like of shape (n, n)
        The covariance matrix. Must be symmetric and positive semi-definite.

    Examples
    --------
    >>> from symbulate import *
    >>> X = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    >>> X.draw()  # doctest: +SKIP
    (1.76, 0.40)
    """

    def __init__(self, mean, cov):
        """Initialize a multivariate normal distribution."""
        if len(mean) != len(cov):
            raise Exception(
                "The dimension of the mean vector"
                + " is not compatible with the dimensions"
                + " of the covariance matrix."
            )

        if len(mean) >= 1:
            self.mean = mean
        else:
            raise Exception("Mean vector and Cov matrix cannot be empty")

        if len(cov) >= 1:
            if all(len(row) == len(mean) for row in cov):
                if np.all(np.linalg.eigvals(cov) >= 0) and np.allclose(
                    cov, np.transpose(cov)
                ):
                    self.cov = cov
                else:
                    raise Exception(
                        "Cov matrix is not symmetric and positive semi-definite"
                    )
            else:
                raise Exception("Cov matrix is not square")
        else:
            raise Exception("Dimension of cov matrix cannot be less than 1")

        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(mean, cov).pdf(x)

    def plot(self):
        """Plot is not supported for multivariate distributions.

        Raises
        ------
        Exception
            Always raised — plotting is not available for the
            multivariate normal distribution.
        """
        raise Exception(
            "Plotting is not currently available for "
            "the multivariate normal distribution."
        )

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

        return Vector(rng.multivariate_normal(self.mean, self.cov))

    def __pow__(self, exponent):
        """Draw multiple independent samples from the multivariate normal distribution.

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
        >>> (MultivariateNormal([0, 0], [[1, 0], [0, 1]]) ** 3).draw()  # doctest: +SKIP
        [(-0.23, 1.72), (0.31, -0.72), (0.88, -0.21)]
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

    Attributes
    ----------
    mean : list of float
        The mean vector ``[mean1, mean2]``.
    cov : list of list of float
        The 2×2 covariance matrix.

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
            is not a number between -1 and 1; or if ``sd1``, ``sd2``,
            ``var1``, or ``var2`` is not a non-negative number.
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

        self.mean = [mean1, mean2]

        # var1/var2 default to sd**2, and cov defaults to corr*sqrt(var1*var2);
        # every value used below has been validated above.
        if var1 is None:
            var1 = sd1**2
        if var2 is None:
            var2 = sd2**2
        if cov is None:
            cov = corr * np.sqrt(var1 * var2)
        self.cov = [[var1, cov], [cov, var2]]
        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(self.mean, self.cov).pdf(x)


class Multinomial(Distribution):
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

    Examples
    --------
    >>> from symbulate import *
    >>> X = Multinomial(n=10, p=[0.5, 0.3, 0.2])
    >>> X.draw()  # doctest: +SKIP
    (5, 3, 2)
    """

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
            bad_p = not (sum(p) == 1 and min(p) >= 0)
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

        self.discrete = False
        self.pdf = lambda x: stats.multinomial(n, p).pmf(x)

    def plot(self):
        """Plot is not supported for multivariate distributions.

        Raises
        ------
        Exception
            Always raised — plotting is not available for the
            multinomial distribution.
        """
        raise Exception(
            "Plotting is not currently available for " "the Multinomial distribution."
        )

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

    def __pow__(self, exponent):
        """Draw multiple independent samples from the multinomial distribution.

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
        >>> (Multinomial(10, [0.5, 0.3, 0.2]) ** 3).draw()  # doctest: +SKIP
        [(5, 3, 2), (4, 4, 2), (6, 2, 2)]
        """
        if exponent == float("inf"):

            def draw():
                def _func(_):
                    return self.draw()

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)
