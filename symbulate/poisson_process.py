import bisect
import warnings

import numpy as np
from scipy.integrate import quad

from .distributions import Exponential
from .index_sets import Reals
from .math import inf
from .probability_space import ProbabilitySpace
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
