import bisect
import warnings

import numpy as np
from scipy.integrate import quad

from .distributions import Distribution, Exponential, MultivariateDistribution
from .index_sets import Reals
from .math import inf
from .probability_space import ProbabilitySpace

# The same support-reading helper the renewal process uses to reject a negative
# interarrival time. An intensity is a different quantity, but the question is
# identical -- what is the smallest value this distribution can produce? -- so
# the check is reused rather than written a second time.
from .renewal_process import _smallest_possible_time as _smallest_possible_value
from .result import InfiniteVector, ContinuousTimeFunction, DiscreteValued
from .random_variables import RV
from .random_processes import RandomProcess

# Numerical settings for integrating a time-varying rate function. A rate
# function is supplied by the user, so it can be anything -- smooth, jumpy, or
# not integrable at all -- and the integration has to be honest about which of
# those it got. `_QUAD_TOL` is the largest error, relative to the value itself,
# that an integral is trusted at; `_QUAD_SUBDIVISIONS` are the numbers of
# pieces the interval is cut into, in order, when a single integration is not
# accurate enough on its own (a rate that changes very rapidly needs the
# integration pinned down piece by piece).
_QUAD_LIMIT = 200
_QUAD_TOL = 1e-3
_QUAD_SUBDIVISIONS = (32, 512, 4096)

# Settings for adding a *randomly drawn* intensity up over time (the Cox
# process). `_INTENSITY_STEP` is how far apart the grid of times is that a
# continuously varying intensity path -- a diffusion, say -- is read at; it is
# the default for `CoxProcess(step=)` and is the one accuracy knob there.
# Intensities that hold one value at a time (a Markov chain, a count) are added
# up exactly instead and ignore it. The two `_MAX_` settings are backstops, so
# that asking about a far-off time reports a readable error rather than
# appearing to hang: one caps the grid points read along a path, the other the
# jumps followed along a piecewise-constant one.
_INTENSITY_STEP = 0.01
_MAX_INTENSITY_GRID_POINTS = 1_000_000
_MAX_INTENSITY_JUMPS = 1_000_000


class PoissonProcessResult(ContinuousTimeFunction, DiscreteValued):
    """A single realization of a Poisson process.

    Parameters
    ----------
    interarrival_times : iterable of float
        Sequence of exponential interarrival times.

    Attributes
    ----------
    interarrival_times : list
        A list of interarrival times for the Poisson process.

    Examples
    --------
    >>> from symbulate import *
    >>> path = PoissonProcess(rate=1).draw()
    >>> path[1.5]  # doctest: +SKIP
    1
    """

    def __init__(self, interarrival_times):
        """Create a Poisson process sample path from interarrival times."""
        self.interarrival_times = interarrival_times

        def func(t):
            total_time = 0
            for n, time in enumerate(self.interarrival_times):
                total_time += time
                if t < total_time:
                    return n

        super().__init__(func)

    def get_states(self):
        """Get the sequence of states of the Poisson process.

        Returns
        -------
        InfiniteVector
            An infinite vector of the states of the Poisson process starting at 0.

        Examples
        --------
        >>> from symbulate import *
        >>> path = PoissonProcess(rate=1).draw()
        >>> states = path.get_states()
        >>> states[0]
        0
        >>> states[3]
        3
        """
        return InfiniteVector(lambda n: n)


class PoissonProcessProbabilitySpace(ProbabilitySpace):
    """Probability space for a Poisson process.

    Parameters
    ----------
    rate : float
        The rate (λ) of the Poisson process. Must be positive.

    Attributes
    ----------
    rate : float
        The rate parameter of the Poisson process.

    Raises
    ------
    TypeError
        If ``rate`` is not a number.
    ValueError
        If ``rate`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> space = PoissonProcessProbabilitySpace(rate=2)
    >>> path = space.draw()
    >>> path[1.0]  # doctest: +SKIP
    2
    """

    def __init__(self, rate):
        """Create a probability space for a Poisson process."""
        if not isinstance(rate, (int, float)):
            raise TypeError(
                f"rate must be a positive number, got {type(rate).__name__}."
            )
        if rate <= 0:
            raise ValueError(
                f"rate must be positive, got {rate}. "
                "The rate controls how often events occur per unit time."
            )
        self.rate = rate

        # Build the interarrival-time space once, not once per draw: the rate
        # never changes, and each `.draw()` on it already yields a fresh,
        # independent sequence.
        interarrivals = Exponential(rate=self.rate) ** inf

        def draw():
            return PoissonProcessResult(interarrivals.draw())

        super().__init__(draw)


class PoissonProcess(RandomProcess, RV):
    """A random Poisson process and a random variable.

    Parameters
    ----------
    rate : float
        The rate (λ) of the Poisson process. Must be positive.

    Attributes
    ----------
    rate : float
        The rate parameter of the Poisson process.

    Raises
    ------
    TypeError
        If ``rate`` is not a number.
    ValueError
        If ``rate`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> N = PoissonProcess(rate=1)
    >>> N[2].mean()  # doctest: +SKIP
    2.0
    """

    def __init__(self, rate):
        """Create a Poisson process with the given rate."""
        self.rate = rate
        prob_space = PoissonProcessProbabilitySpace(self.rate)
        RandomProcess.__init__(self, prob_space, Reals())
        RV.__init__(self, prob_space)


def _rate_value(rate_func, t):
    """Evaluate a rate function at one time, checking the answer makes sense.

    Every evaluation is checked -- during validation and during integration
    alike -- so a rate function that only misbehaves in one stretch of time is
    caught there rather than silently producing a nonsense process.

    Parameters
    ----------
    rate_func : callable
        A function ``rate(t)`` giving the rate of events at time ``t``.
    t : float
        The time to evaluate at.

    Returns
    -------
    float
        The rate at time ``t``.

    Raises
    ------
    TypeError
        If ``rate_func`` cannot be called with a single time, or does not
        return a number.
    ValueError
        If the rate at ``t`` is negative, infinite, or undefined.
    """
    try:
        value = rate_func(t)
    except TypeError as error:
        raise TypeError(
            "rate must be a function of one argument, the time t -- for "
            "example, rate=lambda t: 2 + t for a rate that grows over time. "
            f"Calling your function with t={t} did not work ({error})."
        ) from None

    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "rate(t) must give back a number: the rate of events at time t. "
            f"At t={t} your function gave back {type(value).__name__}. For "
            "example, rate=lambda t: 2 + t gives the number 2 at time 0."
        )

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(
            f"rate(t) must be a finite number, but at t={t} your rate "
            f"function gives {value}. A rate is a number of events per unit "
            "time, so it cannot be infinite or undefined. Check the formula "
            f"for division by zero or overflow near t={t}."
        )
    if value < 0:
        raise ValueError(
            f"rate(t) can never be negative, but at t={t} your rate function "
            f"gives {value:g}. The rate is how frequently events happen, so 0 "
            "is the smallest it can be (no events at all during that stretch "
            "of time). If your formula is meant to fade out rather than turn "
            "negative, wrap it so it flattens at 0 -- for example, "
            "rate=lambda t: max(0, 5 - t) instead of rate=lambda t: 5 - t."
        )
    return value


def _validate_rate(rate):
    """Turn the user's ``rate`` into a checked rate function.

    A plain number is accepted and becomes a constant rate, which makes a
    non-homogeneous Poisson process reduce to an ordinary one -- useful for
    comparing the two.

    Parameters
    ----------
    rate : callable or float
        A function ``rate(t)`` giving the rate of events at time ``t``, or a
        positive number for a rate that never changes.

    Returns
    -------
    callable
        A function of one time that checks its own answer on every call.

    Raises
    ------
    TypeError
        If ``rate`` is neither a number nor a function of one time, or if the
        function does not give back a number.
    ValueError
        If ``rate`` is a number that is not positive, or the function's value
        at time 0 is negative, infinite, or undefined.
    """
    if isinstance(rate, bool):
        raise TypeError(
            "rate must be a function of time or a positive number, not "
            "True/False. For example, rate=lambda t: 2 + t or rate=2."
        )

    if isinstance(rate, (int, float, np.integer, np.floating)):
        if rate <= 0:
            reason = (
                "a rate of 0 would mean no events ever occur"
                if rate == 0
                else "the rate is how frequently events happen, so it cannot "
                "be negative"
            )
            raise ValueError(
                f"A rate given as a single number must be positive, got {rate}: "
                f"{reason}. To have events stop and start again, give a "
                "function of time instead -- for example, "
                "rate=lambda t: 0 if t < 5 else 2 for events that only begin "
                "at time 5."
            )
        constant_rate = float(rate)
        return lambda t: constant_rate

    if not callable(rate):
        raise TypeError(
            "rate must be a function of time or a positive number, got "
            f"{type(rate).__name__}. For example, "
            "rate=lambda t: 2 + t for a rate that grows over time, "
            "rate=lambda t: 5 if t < 12 else 1 for a rate that drops at time "
            "12, or rate=2 for a rate that never changes."
        )

    # Check the function once, at time 0, so an unusable one is reported when
    # the process is built rather than on the first draw. Later times are
    # checked as they come up: a rate function is only required to make sense
    # over the stretch of time that is actually asked about, and some are
    # written from a table that runs out.
    _rate_value(rate, 0.0)
    return lambda t: _rate_value(rate, t)


class _CumulativeRate:
    """The expected number of events by time ``t``, for a time-varying rate.

    Writing the rate of events at time ``t`` as ``rate(t)``, this is the
    integral of the rate from 0 to ``t``. It is the quantity the whole
    non-homogeneous Poisson process is built out of: events are generated in
    "expected count" units, where they arrive at a steady rate of 1, and this
    converts back and forth between those units and clock time.

    Values are computed as they are asked for and remembered, so a rate
    function is never integrated over the same stretch of time twice --
    including across the many sample paths of a single ``.sim()`` call, which
    all share one of these.

    Parameters
    ----------
    rate_func : callable
        A checked rate function, as returned by ``_validate_rate``.

    Attributes
    ----------
    rate_func : callable
        The rate function being integrated.
    """

    def __init__(self, rate_func):
        """Create a cumulative rate function that starts out knowing only 0."""
        self.rate_func = rate_func
        # Times whose cumulative rate is already known, in increasing order,
        # alongside the values themselves. A new time is filled in by
        # integrating from the nearest earlier known time, so the work done for
        # one time is never repeated for a later one.
        self._times = [0.0]
        self._values = [0.0]

    def __call__(self, t):
        """The expected number of events by time ``t``.

        Parameters
        ----------
        t : float
            A time. Times at or before 0 give 0.

        Returns
        -------
        float
            The integral of the rate function from 0 to ``t``.

        Raises
        ------
        ValueError
            If the rate function cannot be integrated reliably over the time
            asked about (see ``_integrate``), or is negative, infinite, or
            undefined anywhere the integration looks.
        """
        t = float(t)
        if t <= 0:
            return 0.0

        position = bisect.bisect_left(self._times, t)
        if position < len(self._times) and self._times[position] == t:
            return self._values[position]

        earlier_time = self._times[position - 1]
        value = self._values[position - 1] + self._integrate(earlier_time, t)
        self._times.insert(position, t)
        self._values.insert(position, value)
        return value

    def _integrate(self, a, b):
        """Integrate the rate function from ``a`` to ``b``, or refuse to guess.

        Numerical integration can return a confidently wrong answer for a rate
        function that is not integrable -- ``rate=lambda t: 1 / t`` near 0 is
        the classic case. So the estimated error is checked against the value,
        and an integral that is not trustworthy is retried over smaller and
        smaller pieces before giving up. Giving up raises, rather than
        returning a number nobody should rely on.

        Parameters
        ----------
        a, b : float
            The stretch of time to integrate over, with ``a < b``.

        Returns
        -------
        float
            The integral of the rate function from ``a`` to ``b``.

        Raises
        ------
        ValueError
            If no trustworthy value could be computed.
        """
        value = self._quad(a, b)
        if value is not None:
            return value

        for pieces in _QUAD_SUBDIVISIONS:
            edges = np.linspace(a, b, pieces + 1)
            total = 0.0
            for low, high in zip(edges, edges[1:]):
                piece = self._quad(low, high)
                if piece is None:
                    total = None
                    break
                total += piece
            if total is not None:
                return total

        raise ValueError(
            "Symbulate could not work out the expected number of events "
            f"between time {a:g} and time {b:g}, because the rate function "
            "changes too sharply there to add up reliably. This usually means "
            "the rate grows without bound somewhere in that stretch of time "
            "(rate=lambda t: 1 / t does this at t=0). Check that your rate "
            "function stays finite over the times you are asking about."
        )

    def _quad(self, a, b):
        """Integrate from ``a`` to ``b``, or return ``None`` if not trustworthy.

        Parameters
        ----------
        a, b : float
            The stretch of time to integrate over, with ``a < b``.

        Returns
        -------
        float or None
            The integral, or ``None`` when the estimated error is too large
            relative to the value, or the value is impossible (negative or not
            finite) for the integral of a nonnegative rate.
        """
        with warnings.catch_warnings():
            # scipy's own warning here ("maximum number of subdivisions...")
            # describes an implementation detail of the integration, which is
            # not something to hand a student. The checks below decide what to
            # do about it, and the error message above explains it in terms of
            # the rate function.
            warnings.simplefilter("ignore")
            value, error_estimate = quad(self.rate_func, a, b, limit=_QUAD_LIMIT)
        if not np.isfinite(value) or value < 0:
            return None
        if error_estimate > _QUAD_TOL * max(1.0, value):
            return None
        return value


class NonHomogeneousPoissonProcessResult(ContinuousTimeFunction, DiscreteValued):
    """A single realization of a non-homogeneous Poisson process.

    Parameters
    ----------
    standard_interarrival_times : iterable of float
        Interarrival times of a rate-1 Poisson process, measured in expected
        count rather than in clock time.
    cumulative_rate : callable
        The expected number of events by time ``t``.

    Attributes
    ----------
    standard_interarrival_times : iterable of float
        The rate-1 interarrival times this path was built from.
    cumulative_rate : callable
        The expected number of events by time ``t``.

    Notes
    -----
    The counting is the same walk along a cumulative sum that
    ``PoissonProcessResult`` does, with one substitution: instead of comparing
    the running total against the clock time ``t``, it compares against the
    expected number of events by time ``t``. Events arrive at a steady rate of
    1 on that scale no matter how the real rate varies, which is what makes a
    varying rate simulable at all. Formally, the ``k``-th event happens at the
    time when the cumulative rate reaches the ``k``-th rate-1 arrival, so
    counting events before time ``t`` is the same as counting rate-1 arrivals
    before the cumulative rate at ``t``.

    Examples
    --------
    >>> from symbulate import *
    >>> path = NonHomogeneousPoissonProcess(rate=lambda t: 2 * t).draw()
    >>> path[1.5]  # doctest: +SKIP
    2
    """

    def __init__(self, standard_interarrival_times, cumulative_rate):
        """Create a sample path from rate-1 arrivals and a cumulative rate."""
        self.standard_interarrival_times = standard_interarrival_times
        self.cumulative_rate = cumulative_rate

        def func(t):
            expected_count = self.cumulative_rate(t)
            total = 0
            for n, time in enumerate(self.standard_interarrival_times):
                total += time
                if expected_count < total:
                    return n

        super().__init__(func)

    def get_states(self):
        """Get the sequence of states of the process.

        Returns
        -------
        InfiniteVector
            An infinite vector of the states of the process starting at 0.

        Examples
        --------
        >>> from symbulate import *
        >>> path = NonHomogeneousPoissonProcess(rate=lambda t: 2 * t).draw()
        >>> states = path.get_states()
        >>> states[0]
        0
        >>> states[3]
        3
        """
        return InfiniteVector(lambda n: n)


class NonHomogeneousPoissonProcessProbabilitySpace(ProbabilitySpace):
    """Probability space for a non-homogeneous Poisson process.

    Parameters
    ----------
    rate : callable or float
        A function ``rate(t)`` giving the rate of events at time ``t``, or a
        positive number for a rate that never changes. The rate can never be
        negative.

    Attributes
    ----------
    rate : callable or float
        The rate exactly as it was given.
    cumulative_rate : callable
        ``cumulative_rate(t)`` is the expected number of events by time ``t``.

    Raises
    ------
    TypeError
        If ``rate`` is neither a number nor a function of one time, or the
        function does not give back a number.
    ValueError
        If ``rate`` is a number that is not positive, or the rate is negative,
        infinite, or undefined at time 0.

    Examples
    --------
    >>> from symbulate import *
    >>> space = NonHomogeneousPoissonProcessProbabilitySpace(lambda t: 2 * t)
    >>> path = space.draw()
    >>> path[1.0]  # doctest: +SKIP
    1
    """

    def __init__(self, rate):
        """Create a probability space for a non-homogeneous Poisson process."""
        self.rate = rate
        self.cumulative_rate = _CumulativeRate(_validate_rate(rate))

        # Rate-1 arrivals, built once rather than once per draw (each `.draw()`
        # on this already gives a fresh, independent sequence). The varying
        # rate is applied by `cumulative_rate`, not here.
        standard_interarrivals = Exponential(rate=1) ** inf

        def draw():
            return NonHomogeneousPoissonProcessResult(
                standard_interarrivals.draw(), self.cumulative_rate
            )

        super().__init__(draw)


class NonHomogeneousPoissonProcess(RandomProcess, RV):
    """A Poisson process whose rate changes over time.

    In an ordinary Poisson process events happen at the same rate forever. In
    a non-homogeneous one the rate is a function of time, so events can be
    frequent during one stretch and rare during another -- calls to a help
    line over a day, machine failures as equipment ages, customers arriving
    over a lunch rush.

    Parameters
    ----------
    rate : callable or float
        A function ``rate(t)`` giving the rate of events at time ``t``, or a
        positive number for a rate that never changes (which gives an ordinary
        Poisson process, handy for comparison). The rate can never be
        negative.

    Attributes
    ----------
    rate : callable or float
        The rate exactly as it was given.
    cumulative_rate : callable
        ``cumulative_rate(t)`` is the expected number of events by time ``t``:
        the rate function added up from 0 to ``t``. This is the theoretical
        mean to compare a simulation against, and ``N(t)`` has a
        ``Poisson(cumulative_rate(t))`` distribution.

    Raises
    ------
    TypeError
        If ``rate`` is neither a number nor a function of one time, or the
        function does not give back a number.
    ValueError
        If ``rate`` is a number that is not positive, or the rate is negative,
        infinite, or undefined at time 0.

    Notes
    -----
    Sample paths are generated by the time-change method: events are drawn at
    a steady rate of 1 on the "expected count" scale and read back onto the
    clock through the cumulative rate. This is exact for any nonnegative rate
    function and asks the user for nothing beyond the rate itself. The one
    numerical step is adding the rate function up over time, which is done as
    described in ``_CumulativeRate``; a rate function that cannot be added up
    reliably raises rather than returning a number that looks fine and is not.

    A rate function whose spikes are far narrower than the stretch of time
    being integrated (a rate of 1 with a spike 0.001 wide somewhere in the
    first 100 units of time, say) can be stepped over by the integration and
    its events missed. Rates that vary on a scale anywhere near the times
    being asked about -- daily cycles, gradual wear, a shift change -- are
    unaffected.

    Examples
    --------
    >>> from symbulate import *
    >>> N = NonHomogeneousPoissonProcess(rate=lambda t: 2 * t)
    >>> N.cumulative_rate(3)  # expected number of events by time 3
    9.0
    >>> N(3).mean()  # doctest: +SKIP
    9.0
    """

    def __init__(self, rate):
        """Create a Poisson process with the given time-varying rate."""
        prob_space = NonHomogeneousPoissonProcessProbabilitySpace(rate)
        self.rate = rate
        self.cumulative_rate = prob_space.cumulative_rate
        RandomProcess.__init__(self, prob_space, Reals())
        RV.__init__(self, prob_space)


def _intensity_value(value, where):
    """Check one value of a drawn intensity, and return it as a plain number.

    Every value read off an intensity goes through here, so an intensity that
    only misbehaves in one stretch of time -- or in one state of a Markov chain
    -- is reported at the place it does, rather than silently producing a
    nonsense process.

    Parameters
    ----------
    value : object
        The value read off the intensity. A number, if all is well.
    where : str
        Where the value came from, phrased to begin a sentence: ``"At time
        2.5,"``. Used in the error messages below.

    Returns
    -------
    float
        The value.

    Raises
    ------
    TypeError
        If the value is not a number.
    ValueError
        If the value is negative, infinite, or undefined.
    """
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"{where} the intensity is {value!r}, which is not a number. An "
            "intensity is a rate -- how frequently events happen at that "
            "moment -- so every value of it has to be a number. If the "
            "intensity is a continuous-time Markov chain, label its states "
            "with the rates themselves, as in "
            "ContinuousTimeMarkovChain(Q, initial, state_labels=[1, 5]), "
            "rather than with names like 'low' and 'high'."
        )

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(
            f"{where} the intensity is {value}, but an intensity has to be a "
            "finite number: a number of events per unit time cannot be "
            "infinite or undefined."
        )
    if value < 0:
        raise ValueError(
            f"{where} the intensity is {value:g}, but an intensity can never "
            "be negative -- it is how frequently events happen, so 0 is the "
            "smallest it can be (no events at all just then). A process that "
            "goes negative, such as BrownianMotion, cannot be an intensity as "
            "it stands. Use one that stays nonnegative -- CIR(), "
            "GeometricBrownianMotion(), or a ContinuousTimeMarkovChain whose "
            "states are nonnegative rates -- or square it, exponentiate it, or "
            "otherwise send it through a function that cannot go below 0."
        )
    return value


class _StepCumulativeRate:
    """The integral over time of an intensity that holds one value at a time.

    An intensity path that reports its states and how long it stays in each
    one -- a continuous-time Markov chain, a Poisson or renewal count -- is
    constant between its jumps, so the integral over time is a sum of value
    times duration with nothing whatever to approximate. This computes exactly
    that, following the path one jump further only when a later time is asked
    about.

    Parameters
    ----------
    intensity_path : DiscreteValued
        One drawn intensity path, which must provide ``get_states()`` and
        ``get_interarrival_times()``.
    """

    def __init__(self, intensity_path):
        """Create the integral of one piecewise-constant intensity path."""
        self.intensity_path = intensity_path
        self.states = intensity_path.get_states()
        self.holding_times = intensity_path.get_interarrival_times()
        # The three lists line up: the intensity is `_intensities[k]` over the
        # stretch of time from `_times[k]` to `_times[k + 1]`, and `_values[k]`
        # is the integral up to `_times[k]`. All three are filled in together,
        # one jump at a time, as later times are asked about.
        self._times = [0.0]
        self._values = [0.0]
        self._intensities = []

    def __call__(self, t):
        """The integral of the intensity from 0 to ``t``.

        Parameters
        ----------
        t : float
            A time. Times at or before 0 give 0.

        Returns
        -------
        float
            The integral of the intensity path from 0 to ``t``.

        Raises
        ------
        TypeError
            If a state of the intensity is not a number.
        ValueError
            If a state is negative, infinite, or undefined, or if the path
            jumps more times before ``t`` than can reasonably be followed.
        """
        t = float(t)
        if t <= 0:
            return 0.0

        self._follow_past(t)
        # The last jump at or before `t`. `_follow_past` guarantees there is a
        # later one, so this index is always one the intensity is known at.
        position = bisect.bisect_right(self._times, t) - 1
        return self._values[position] + self._intensities[position] * (
            t - self._times[position]
        )

    def _follow_past(self, t):
        """Follow the intensity path until it jumps at a time after ``t``.

        Parameters
        ----------
        t : float
            The time being asked about.

        Raises
        ------
        ValueError
            If the path jumps more than ``_MAX_INTENSITY_JUMPS`` times before
            ``t``.
        """
        while self._times[-1] <= t:
            n = len(self._intensities)
            if n >= _MAX_INTENSITY_JUMPS:
                raise ValueError(
                    f"The intensity jumps more than {_MAX_INTENSITY_JUMPS:,} "
                    f"times before time {t:g}, which is too many to follow. "
                    "The intensity is changing far more often than the times "
                    "being asked about, so either ask about earlier times or "
                    "give an intensity that changes more slowly."
                )
            intensity = _intensity_value(
                self.states[n], f"In the intensity's state number {n},"
            )
            holding_time = float(self.holding_times[n])
            self._intensities.append(intensity)
            self._times.append(self._times[n] + holding_time)
            # An intensity that never leaves its state again (a Markov chain
            # that has been absorbed) holds it for an infinite time, and 0 rate
            # for an infinite time is 0 events, not an undefined number.
            if np.isinf(holding_time):
                self._values.append(self._values[n] if intensity == 0 else inf)
            else:
                self._values.append(self._values[n] + intensity * holding_time)


class _PathCumulativeRate:
    """The integral over time of a drawn intensity path, added up on a grid.

    An intensity path that changes continuously -- a CIR process, say -- has no
    formula for its integral, and it cannot be handed to a general-purpose
    numerical integrator either: such a path is *generated as it is looked at*,
    so an integrator probing it at times of its own choosing would be filling
    in a different path with every probe, and the error estimate it reports
    would mean nothing.

    So the path is read at a fixed grid of times, ``step`` apart, starting at
    0, and held at each grid value until the next -- the left-hand sum a
    student would draw by hand. Reading it always at the *same* grid, whichever
    times are asked about and in whatever order, is what makes this one
    well-defined increasing function of ``t``, which in turn is what keeps the
    count of events it feeds from ever going backwards.

    Parameters
    ----------
    intensity_path : callable
        One drawn intensity path, which can be evaluated at a time.
    step : float
        How far apart the grid of times is. Smaller is more accurate and
        slower.
    """

    def __init__(self, intensity_path, step):
        """Create the integral of one continuously varying intensity path."""
        self.intensity_path = intensity_path
        self.step = step
        # `_intensities[k]` is the intensity at grid time `k * step`, and
        # `_values[k]` is the integral up to it. Filled in from the start of
        # the grid outward, and remembered, so a path is never read twice at
        # the same time and each grid point is paid for once.
        self._intensities = []
        self._values = [0.0]

    def __call__(self, t):
        """The integral of the intensity from 0 to ``t``.

        Parameters
        ----------
        t : float
            A time. Times at or before 0 give 0.

        Returns
        -------
        float
            The grid sum of the intensity path from 0 to ``t``.

        Raises
        ------
        TypeError
            If the intensity path gives back something other than a number.
        ValueError
            If the intensity is negative, infinite, or undefined anywhere on
            the grid, or if ``t`` is so far off that the grid would not fit.
        """
        t = float(t)
        if t <= 0:
            return 0.0

        last_grid_point = int(t // self.step)
        if last_grid_point + 1 > _MAX_INTENSITY_GRID_POINTS:
            raise ValueError(
                f"Reading the intensity every {self.step:g} units of time up "
                f"to time {t:g} would take more than "
                f"{_MAX_INTENSITY_GRID_POINTS:,} readings, which is too many. "
                "Either ask about earlier times, or pass a larger step -- "
                f"CoxProcess(intensity, step={100 * self.step:g}), say -- "
                "which reads the intensity less often, less accurately, and "
                "much faster."
            )

        self._read_grid_to(last_grid_point)
        return self._values[last_grid_point] + self._intensities[last_grid_point] * (
            t - last_grid_point * self.step
        )

    def _read_grid_to(self, last_grid_point):
        """Read the intensity path out to a grid point, remembering the sum.

        Parameters
        ----------
        last_grid_point : int
            The index of the furthest grid point needed, so the grid time
            ``last_grid_point * step``.
        """
        while len(self._intensities) <= last_grid_point:
            k = len(self._intensities)
            time = k * self.step
            intensity = _intensity_value(
                self.intensity_path(time), f"At time {time:g},"
            )
            self._intensities.append(intensity)
            self._values.append(self._values[k] + intensity * self.step)


def _cumulative_rate_for(drawn_intensity, step):
    """Choose how to add one drawn intensity up over time.

    Three kinds of drawn intensity, each added up the way that suits it -- and
    exactly, in the first two cases:

    - **A single number**, drawn once for the whole path. Then the intensity
      never changes and the integral is just the rate times the time.
    - **A path that holds one value at a time** -- a continuous-time Markov
      chain, a Poisson or renewal count. Then the integral is a sum of value
      times duration (see ``_StepCumulativeRate``).
    - **Any other path**, which is read on a grid and approximated (see
      ``_PathCumulativeRate``).

    Parameters
    ----------
    drawn_intensity : object
        One realization of the intensity: a number or a sample path.
    step : float
        The grid width to use, if the third case applies.

    Returns
    -------
    callable
        ``cumulative_rate(t)``, the integral of this intensity from 0 to ``t``.

    Raises
    ------
    TypeError
        If the drawn intensity is neither a number nor something that can be
        evaluated at a time.
    """
    if isinstance(drawn_intensity, (int, float, np.integer, np.floating)):
        intensity = _intensity_value(drawn_intensity, "For this whole path,")
        return lambda t: intensity * max(float(t), 0.0)

    if isinstance(drawn_intensity, DiscreteValued) and hasattr(
        drawn_intensity, "interarrival_times"
    ):
        return _StepCumulativeRate(drawn_intensity)

    if callable(drawn_intensity):
        return _PathCumulativeRate(drawn_intensity, step)

    raise TypeError(
        f"Drawing from the intensity gave a {type(drawn_intensity).__name__}, "
        "which is neither a number nor a sample path that can be evaluated at "
        "a time, so there is no way to read a rate of events off it. The "
        "intensity should be a distribution, for one random rate that holds "
        "for the whole path (as in CoxProcess(Gamma(shape=2, rate=1))), or a "
        "random process, for a rate that varies over time (as in "
        "CoxProcess(CIR()))."
    )


def _validate_intensity_dist(intensity):
    """Check that a distribution can serve as a single random intensity.

    Parameters
    ----------
    intensity : Distribution
        The candidate distribution of the intensity.

    Raises
    ------
    TypeError
        If the distribution is multivariate, so gives a whole vector on each
        draw rather than one rate.
    ValueError
        If the distribution can produce a negative rate.
    """
    name = type(intensity).__name__

    if isinstance(intensity, MultivariateDistribution):
        raise TypeError(
            f"intensity must give a single rate on each draw, but {name} gives "
            "a whole vector. If you meant several rates at once, build a "
            "separate Cox process for each."
        )

    smallest = _smallest_possible_value(intensity)
    if smallest is not None and smallest < 0:
        raise ValueError(
            f"intensity must never be negative, but {name} can produce values "
            f"as small as {smallest:g}. The intensity is how frequently events "
            "happen, so 0 is the smallest it can be. Try a distribution that "
            "is never negative, such as Gamma(shape=2, rate=1), "
            "Exponential(rate=1), or LogNormal(0, 1)."
        )


def _validate_intensity(intensity):
    """Work out how a Cox process's intensity is to be drawn, and check it.

    Parameters
    ----------
    intensity : Distribution, RV, RandomProcess, callable, or float
        The intensity as the user gave it. See :class:`CoxProcess`.

    Returns
    -------
    draw_intensity : callable
        A function of no arguments giving one realization of the intensity.
    cumulative_rate : callable or None
        The integral over time to use for *every* path, when the intensity is
        not random at all -- there is only one rate to add up, so the work is
        shared, exactly as :class:`NonHomogeneousPoissonProcess` shares it.
        ``None`` when the intensity is random, since each path then has its
        own.

    Raises
    ------
    TypeError
        If ``intensity`` is none of the accepted kinds.
    ValueError
        If ``intensity`` can produce a negative rate.
    """
    if isinstance(intensity, Distribution):
        # One random rate, held for the whole path: the mixed Poisson process.
        _validate_intensity_dist(intensity)
        return intensity.draw, None

    if isinstance(intensity, RV):
        # A whole random intensity, drawn once per path. This covers random
        # processes and continuous-time Markov chains, and also a random
        # variable that simply gives a number.
        return intensity.draw, None

    if isinstance(intensity, bool) or not (
        callable(intensity)
        or isinstance(intensity, (int, float, np.integer, np.floating))
    ):
        raise TypeError(
            "intensity must be one of: a distribution, for one random rate "
            "held for the whole path (as in CoxProcess(Gamma(shape=2, "
            "rate=1))); a random process, for a rate that varies over time (as "
            "in CoxProcess(CIR())); or a function of time or a positive "
            "number, for a rate that is not random at all. You gave "
            f"{intensity!r}."
        )

    # A rate that is not random at all, which makes this an ordinary
    # non-homogeneous Poisson process -- worth allowing so that the two can be
    # compared side by side. Checked and added up by exactly the machinery that
    # class uses.
    return (lambda: intensity), _CumulativeRate(_validate_rate(intensity))


def _validate_step(step):
    """Check the grid width used to add a varying intensity path up over time.

    Parameters
    ----------
    step : float
        The candidate grid width.

    Returns
    -------
    float
        The grid width.

    Raises
    ------
    TypeError
        If ``step`` is not a number.
    ValueError
        If ``step`` is not positive.
    """
    if isinstance(step, bool) or not isinstance(
        step, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            "step must be a positive number: how far apart the times are at "
            f"which a varying intensity is read. You gave "
            f"{type(step).__name__}."
        )
    if step <= 0:
        raise ValueError(
            f"step must be positive, got {step}. It is the gap between the "
            "times at which a varying intensity is read, so a gap of 0 or less "
            "would never move forward. Smaller is more accurate and slower; "
            f"the default is {_INTENSITY_STEP}."
        )
    return float(step)


class CoxProcessResult(NonHomogeneousPoissonProcessResult):
    """A single realization of a Cox process.

    Conditional on its drawn intensity, a Cox process *is* a non-homogeneous
    Poisson process, so this counts events exactly the way
    :class:`NonHomogeneousPoissonProcessResult` does and adds one thing: the
    intensity that was drawn for this particular path, kept so it can be
    looked at and plotted alongside the counts.

    Parameters
    ----------
    standard_interarrival_times : iterable of float
        Interarrival times of a rate-1 Poisson process, measured in expected
        count rather than in clock time.
    cumulative_rate : callable
        The expected number of events by time ``t`` *given this intensity*.
    intensity : float or callable
        The intensity drawn for this path.

    Attributes
    ----------
    intensity : float or callable
        The intensity realized for this path: a number, when the intensity is
        a single random rate, and the drawn sample path, when it is a random
        process.
    standard_interarrival_times : iterable of float
        The rate-1 interarrival times this path was built from.
    cumulative_rate : callable
        The expected number of events by time ``t``, given this path's
        intensity.

    Examples
    --------
    >>> from symbulate import *
    >>> path = CoxProcess(intensity=Gamma(shape=2, rate=1)).draw()
    >>> path.intensity  # doctest: +SKIP
    1.7
    >>> path[1.5]  # doctest: +SKIP
    3
    """

    def __init__(self, standard_interarrival_times, cumulative_rate, intensity):
        """Create a sample path from rate-1 arrivals and a drawn intensity."""
        self.intensity = intensity
        super().__init__(standard_interarrival_times, cumulative_rate)


class CoxProcessProbabilitySpace(ProbabilitySpace):
    """Probability space for a Cox process.

    Parameters
    ----------
    intensity : Distribution, RV, RandomProcess, callable, or float
        How the rate of events is chosen. See :class:`CoxProcess`.
    step : float, optional
        How far apart the times are at which a continuously varying intensity
        path is read. Default is ``0.01``.

    Attributes
    ----------
    intensity : object
        The intensity exactly as it was given.
    step : float
        The grid width in use.

    Raises
    ------
    TypeError
        If ``intensity`` is not one of the kinds listed above.
    ValueError
        If ``intensity`` can be negative, or ``step`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> space = CoxProcessProbabilitySpace(Gamma(shape=2, rate=1))
    >>> path = space.draw()
    >>> path[1.0]  # doctest: +SKIP
    2
    """

    def __init__(self, intensity, step=_INTENSITY_STEP):
        """Create a probability space for a Cox process."""
        self.intensity = intensity
        self.step = _validate_step(step)
        draw_intensity, deterministic_cumulative_rate = _validate_intensity(intensity)

        # Rate-1 arrivals, built once rather than once per draw (each `.draw()`
        # on this already gives a fresh, independent sequence). The intensity
        # is applied by the cumulative rate, not here.
        standard_interarrivals = Exponential(rate=1) ** inf

        def draw():
            drawn_intensity = draw_intensity()
            if deterministic_cumulative_rate is None:
                cumulative_rate = _cumulative_rate_for(drawn_intensity, self.step)
            else:
                cumulative_rate = deterministic_cumulative_rate
            return CoxProcessResult(
                standard_interarrivals.draw(), cumulative_rate, drawn_intensity
            )

        super().__init__(draw)


class CoxProcess(RandomProcess, RV):
    """A Poisson process whose rate is itself random.

    Also called a doubly stochastic Poisson process. In a non-homogeneous
    Poisson process the rate varies over time, but it varies the same way every
    time you simulate. In a Cox process the rate is drawn at random first, and
    only then are events generated at that rate -- so two simulated paths can
    be busy and quiet for entirely different reasons. This is the standard way
    to model counts that vary more than a Poisson process allows: claims from
    drivers who are not equally risky, calls on days that are not equally busy,
    defaults in an economy that switches between calm and stressed.

    Several familiar models are this one class with different intensities:

    - ``CoxProcess(Gamma(shape=2, rate=1))`` -- a **mixed Poisson process**:
      one random rate, held for the whole path. With a gamma intensity the
      count at any time has a negative binomial distribution, the classic
      overdispersed alternative to the Poisson.
    - ``CoxProcess(ContinuousTimeMarkovChain(Q, initial, state_labels=[1, 5]))``
      -- a **Markov-modulated Poisson process**: the rate switches between
      regimes, here between 1 and 5, at random times.
    - ``CoxProcess(CIR())`` -- a rate that wanders continuously, never going
      negative.

    Parameters
    ----------
    intensity : Distribution, RV, RandomProcess, callable, or float
        The rate of events, in any of these forms:

        - a **distribution** -- one rate is drawn from it per path and held for
          the whole path;
        - a **random process** (or any random variable whose draws are sample
          paths, including a :class:`~symbulate.markov_chains.ContinuousTimeMarkovChain`)
          -- one path is drawn from it per path, and its value at time ``t`` is
          the rate at time ``t``;
        - a **function of time**, or a positive **number** -- a rate that is
          not random at all, which gives an ordinary non-homogeneous (or
          ordinary) Poisson process, handy for comparison.

        However it is given, the rate can never be negative.
    step : float, optional
        How far apart the times are at which a continuously varying intensity
        path is read. Default is ``0.01``. Smaller is more accurate and slower.
        Ignored for the intensities that are added up exactly -- a single
        random rate, or a rate that holds one value at a time -- which is most
        of them.

    Attributes
    ----------
    intensity : object
        The intensity exactly as it was given.
    step : float
        The grid width in use.

    Raises
    ------
    TypeError
        If ``intensity`` is not one of the kinds listed above, or gives
        something other than a number as a rate.
    ValueError
        If ``intensity`` can be negative, or ``step`` is not positive.

    Notes
    -----
    Each sample path is generated in two stages: draw one whole intensity,
    then, holding it fixed, generate events at that rate. The second stage is
    exactly a non-homogeneous Poisson process, done by the time-change method
    :class:`NonHomogeneousPoissonProcess` uses (see ``MODEL-DECISIONS.md``,
    "Non-Homogeneous Poisson Process -- Time-Change, Not Thinning"), so given
    the intensity nothing is approximated.

    **How exact this is** comes down to one thing: adding the drawn intensity
    up over time.

    - A single random rate, and a rate that holds one value at a time (a
      continuous-time Markov chain, a Poisson or renewal count), are added up
      **exactly** -- rate times time, and a sum of rate times duration. The
      mixed Poisson and Markov-modulated Poisson processes are therefore exact.
    - Any other intensity path is read every ``step`` units of time and held at
      each reading, which **approximates** the integral. A path that changes
      continuously, such as :class:`~symbulate.diffusion_process.CIR`, falls
      here.

    The drawn intensity is kept on each path as ``.intensity``, which is worth
    plotting: it is what distinguishes this process from a Poisson process with
    the same average rate.

    The mean count is ``E[N(t)] = E[cumulative_rate(t)]``, the average over
    intensities of the integral of the rate. The variance is larger than the
    mean -- by exactly the variance of that integral -- which is the
    overdispersion the model exists to produce.

    Examples
    --------
    >>> from symbulate import *
    >>> N = CoxProcess(intensity=Gamma(shape=2, rate=1))
    >>> N[5].mean()  # doctest: +SKIP
    10.0
    >>> path = N.draw()
    >>> path.intensity  # the rate this path was generated at  # doctest: +SKIP
    1.83
    >>> path.cumulative_rate(5)  # expected count by time 5, at that rate  # doctest: +SKIP
    9.16

    See Also
    --------
    NonHomogeneousPoissonProcess : A rate that varies over time, but not at
        random.
    PoissonProcess : A rate that never changes.
    """

    def __init__(self, intensity, step=_INTENSITY_STEP):
        """Create a Poisson process with the given random intensity."""
        prob_space = CoxProcessProbabilitySpace(intensity, step)
        self.intensity = intensity
        self.step = prob_space.step
        RandomProcess.__init__(self, prob_space, Reals())
        RV.__init__(self, prob_space)
