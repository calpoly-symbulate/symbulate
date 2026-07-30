import numbers

import numpy as np

from .index_sets import DiscreteTimeSequence, Reals, TimeInterval
from .probability_space import ProbabilitySpace
from .result import (
    DiscreteTimeFunction,
    ContinuousTimeFunction,
)
from .random_variables import RV
from .random_processes import RandomProcess

MACHINE_EPS = 1e-12
rng = np.random.default_rng()


def get_gaussian_process_result(mean_func, cov_func, index_set=Reals()):
    """Create one simulated sample path of a Gaussian process.

    Builds a result object that lazily generates values along the path
    using conditional distributions — new time points are simulated
    consistently with all previously observed values.

    Parameters
    ----------
    mean_func : callable
        A function ``f(t)`` returning the expected value of the process
        at time ``t``.
    cov_func : callable
        A function ``k(s, t)`` returning the covariance between the
        process at times ``s`` and ``t``.
    index_set : DiscreteTimeSequence or Reals, optional
        The set of times over which the process is defined. Defaults
        to all real numbers.

    Returns
    -------
    GaussianProcessResult
        A sample path that can be evaluated at any time in the index set.

    Raises
    ------
    TypeError
        If ``index_set`` is not a ``Reals`` or ``DiscreteTimeSequence`` instance.

    Examples
    --------
    >>> from symbulate import *
    >>> path = get_gaussian_process_result(
    ...     mean_func=lambda t: 0,
    ...     cov_func=lambda s, t: min(s, t)
    ... )
    >>> path(1.0)  # doctest: +SKIP
    0.43
    """

    # Determine whether the process is discrete-time or continous-time
    if isinstance(index_set, DiscreteTimeSequence):
        base_class = DiscreteTimeFunction
    elif isinstance(index_set, Reals):
        base_class = ContinuousTimeFunction
    else:
        raise TypeError(
            f"index_set must be Reals or DiscreteTimeSequence, "
            f"got {type(index_set).__name__}."
        )

    class GaussianProcessResult(base_class):
        """One simulated sample path of a Gaussian process.

        Evaluating this object at a time ``t`` returns the value of the
        path at that moment. New values are generated on demand and are
        always consistent with previously observed values.

        Attributes
        ----------
        mean : numpy.ndarray
            Mean vector over all times observed so far.
        cov : numpy.ndarray
            Covariance matrix over all times observed so far.
        observed : dict
            Maps each previously evaluated time to its simulated value.
        index_set : DiscreteTimeSequence or Reals
            The set of times over which this path is defined.

        Examples
        --------
        >>> from symbulate import *
        >>> X = GaussianProcess(
        ...     mean_func=lambda t: 0,
        ...     cov_func=lambda s, t: min(s, t)
        ... )
        >>> path = X.draw()
        >>> path(1.0)   # doctest: +SKIP
        0.84
        >>> # Re-evaluating the same time always returns the cached value
        >>> path(1.0) == path(1.0)  # doctest: +SKIP
        True
        """

        def __init__(self, mean_func, cov_func):
            """Create one simulated sample path of a Gaussian process."""

            self.mean = np.empty(shape=0)
            self.cov = np.empty(shape=(0, 0))
            self.observed = {}

            def _vfunc(ts):
                # This function assumes that t is an array of times.
                ts = list(ts)

                # Get current times
                times = list(self.observed.keys())

                # If this is a discrete process, t will be an index.
                # Convert it to a time.
                if isinstance(index_set, DiscreteTimeSequence):
                    ts = [t / index_set.fs for t in ts]

                # Check that every t is in the index set
                for t in ts:
                    if t not in index_set:
                        raise KeyError(
                            f"Gaussian process is not defined at time {t}. "
                            f"Time must be in the index set ({index_set!r})."
                        )

                # Create an object to store the results
                n = len(ts)
                values = np.empty(shape=n)
                values[:] = np.nan

                # Handle times that have already been calculated,
                # as well as times where the variance is 0
                i_delete = []
                for i, t in enumerate(ts):
                    if cov_func(t, t) == 0:
                        values[i] = mean_func(t)
                        i_delete.append(i)
                    elif t in self.observed:
                        values[i] = self.observed[t]
                        i_delete.append(i)
                ts = [t for i, t in enumerate(ts) if i not in i_delete]
                if not ts:
                    return values

                # Simulate values for the remaining times
                mean2 = np.array([mean_func(t) for t in ts])
                cov11 = self.cov + MACHINE_EPS * np.eye(len(times))
                cov12 = np.empty(shape=(len(times), len(ts)))
                for i, s in enumerate(times):
                    for j, t in enumerate(ts):
                        cov12[i, j] = cov_func(s, t)
                cov22 = np.empty(shape=(len(ts), len(ts)))
                for i, s in enumerate(ts):
                    for j, t in enumerate(ts):
                        cov22[i, j] = cov_func(s, t)

                cond_mean = mean2 + (
                    cov12.T
                    @ np.linalg.solve(cov11, list(self.observed.values()) - self.mean)
                )
                cond_var = cov22 - (cov12.T @ np.linalg.solve(cov11, cov12))

                # update mean vector and covariance matrix
                self.mean = np.concatenate([self.mean, mean2])
                self.cov = np.block([[cov11, cov12], [cov12.T, cov22]])

                # simulate normal with given mean and variance
                new_values = rng.multivariate_normal(cond_mean, cond_var)

                # store the new values
                for t, v in zip(ts, new_values):
                    self.observed[t] = v
                values[np.isnan(values)] = new_values

                return values

            self.vfunc = _vfunc

            def _func(t):
                return _vfunc([t])[0]

            super().__init__(func=_func)
            self.index_set = index_set

    return GaussianProcessResult(mean_func, cov_func)


class GaussianProcessProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a Gaussian process.

    Each draw from this space produces one simulated sample path of the
    Gaussian process. Paths are generated lazily — values at new times
    are computed on demand, consistent with all previously observed values.

    Parameters
    ----------
    mean_func : callable
        A function ``f(t)`` giving the expected value of the process at
        time ``t``.
    cov_func : callable
        A function ``k(s, t)`` giving the covariance between the process
        at times ``s`` and ``t``.
    index_set : DiscreteTimeSequence or Reals, optional
        The set of times over which the process is defined. Defaults to
        all real numbers.

    Attributes
    ----------
    mean_func : callable
        A function giving the expected value of the process at each time.
    cov_func : callable
        A function giving the covariance between the process at two times.
    index_set : DiscreteTimeSequence or Reals
        The set of times over which the process is defined.

    Raises
    ------
    TypeError
        If ``mean_func`` or ``cov_func`` is not callable.
    TypeError
        If ``index_set`` is not a ``Reals`` or ``DiscreteTimeSequence`` instance.

    Examples
    --------
    >>> from symbulate import *
    >>> P = GaussianProcessProbabilitySpace(
    ...     mean_func=lambda t: 0,
    ...     cov_func=lambda s, t: min(s, t)
    ... )
    >>> path = P.draw()  # doctest: +SKIP
    >>> path(1.0)        # doctest: +SKIP
    0.43
    >>> path(2.0)        # doctest: +SKIP
    1.12
    """

    def __init__(self, mean_func, cov_func, index_set=Reals()):
        """Create a probability space for a Gaussian process."""
        if not callable(mean_func):
            raise TypeError(
                f"mean_func must be callable (e.g., lambda t: 0), "
                f"got {type(mean_func).__name__}."
            )
        if not callable(cov_func):
            raise TypeError(
                f"cov_func must be callable (e.g., lambda s, t: min(s, t)), "
                f"got {type(cov_func).__name__}."
            )
        if not isinstance(index_set, (Reals, DiscreteTimeSequence)):
            raise TypeError(
                f"index_set must be Reals or DiscreteTimeSequence, "
                f"got {type(index_set).__name__}."
            )

        def draw():
            return get_gaussian_process_result(mean_func, cov_func, index_set)

        super().__init__(draw)


class GaussianProcess(RandomProcess, RV):
    """A Gaussian process, modeled as a random variable over sample paths.

    A Gaussian process is a collection of random variables — one for each
    time ``t`` — such that any finite set of them follows a multivariate
    normal distribution. It is fully described by its mean function and
    covariance (kernel) function.

    Parameters
    ----------
    mean_func : callable
        A function ``f(t)`` giving the expected value of the process at
        time ``t``.
    cov_func : callable
        A function ``k(s, t)`` giving the covariance between the process
        at times ``s`` and ``t``.
    index_set : DiscreteTimeSequence or Reals, optional
        The set of times over which the process is defined. Defaults to
        all real numbers.

    Attributes
    ----------
    prob_space : GaussianProcessProbabilitySpace
        The underlying probability space used to generate sample paths.

    Examples
    --------
    >>> from symbulate import *
    >>> X = GaussianProcess(
    ...     mean_func=lambda t: 0,
    ...     cov_func=lambda s, t: min(s, t)
    ... )
    >>> # Draw a single sample path and evaluate at several times
    >>> path = X.draw()                      # doctest: +SKIP
    >>> path(0.5), path(1.0), path(2.0)      # doctest: +SKIP
    (0.23, 0.84, 0.92)
    >>> # Simulate the process at t=1 across many draws
    >>> X[1.0].sim(1000).mean()              # doctest: +SKIP
    0.01
    """

    def __init__(self, mean_func, cov_func, index_set=Reals()):
        """Create a Gaussian process."""

        prob_space = GaussianProcessProbabilitySpace(mean_func, cov_func, index_set)
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)


# Define convenience class for Brownian motion
class BrownianMotionProbabilitySpace(GaussianProcessProbabilitySpace):
    """The probability space underlying a Brownian motion process.

    Each draw from this space produces one simulated sample path of
    Brownian motion. Standard Brownian motion (``drift=0``, ``scale=1``)
    starts at 0 and has independent, normally distributed increments.

    Parameters
    ----------
    drift : float, optional
        The drift parameter μ. Controls the average rate of change per
        unit time. Default is 0 (no drift).
    scale : float, optional
        The scale parameter σ. Controls the volatility (spread) of the
        process. Default is 1.

    Attributes
    ----------
    drift : float
        The drift parameter μ. Defaults to 0.
    scale : float
        The scale parameter σ. Defaults to 1.

    Raises
    ------
    TypeError
        If ``drift`` or ``scale`` is not a number.
    ValueError
        If ``scale`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> P = BrownianMotionProbabilitySpace(drift=0, scale=1)
    >>> path = P.draw()  # doctest: +SKIP
    >>> path(1.0)        # doctest: +SKIP
    -0.32
    >>> # Standard Brownian motion has variance equal to t
    >>> path(4.0)        # doctest: +SKIP
    0.87
    """

    def __init__(self, drift=0, scale=1):
        """Create a probability space for Brownian motion."""
        if not isinstance(drift, (int, float)):
            raise TypeError(f"drift must be a number, got {type(drift).__name__}.")
        if not isinstance(scale, (int, float)):
            raise TypeError(f"scale must be a number, got {type(scale).__name__}.")
        if scale <= 0:
            raise ValueError(
                f"scale must be positive, got {scale}. "
                "A scale of 0 would give a constant process with no randomness."
            )
        super().__init__(
            mean_func=lambda t: drift * t, cov_func=lambda s, t: (scale**2) * min(s, t)
        )


class BrownianMotion(RandomProcess, RV):
    """Brownian motion, modeled as a random variable over sample paths.

    Standard Brownian motion (also called a Wiener process) starts at 0,
    has continuous paths, and has independent normally distributed increments.
    The optional ``drift`` and ``scale`` parameters shift and scale the process.

    Parameters
    ----------
    drift : float, optional
        The drift parameter μ. Controls the average rate of change per
        unit time. Default is 0 (no drift).
    scale : float, optional
        The scale parameter σ. Controls the volatility (spread) of the
        process. Default is 1.

    Attributes
    ----------
    prob_space : BrownianMotionProbabilitySpace
        The underlying probability space used to generate sample paths.

    Examples
    --------
    >>> from symbulate import *
    >>> B = BrownianMotion()
    >>> path = B.draw()          # doctest: +SKIP
    >>> path(1.0)                # doctest: +SKIP
    -0.32
    >>> # Brownian motion with positive drift shifts the mean upward
    >>> B2 = BrownianMotion(drift=0.5, scale=1)
    >>> path2 = B2.draw()        # doctest: +SKIP
    >>> path2(2.0)               # doctest: +SKIP
    1.43
    >>> # Simulate B(1) many times — mean should be close to the drift
    >>> B[1.0].sim(1000).mean()  # doctest: +SKIP
    0.0
    """

    def __init__(self, drift=0, scale=1):
        """Create a Brownian motion process."""
        prob_space = BrownianMotionProbabilitySpace(drift=drift, scale=scale)
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)


# Sentinel for initial_value, asking the path to start from the process's own
# long-run (stationary) distribution instead of from a fixed number.
STATIONARY = "stationary"


def _ornstein_uhlenbeck_funcs(reversion_rate, mean, scale, initial_value):
    """Build the mean and covariance functions of an Ornstein-Uhlenbeck process.

    Both parameterizations are exact closed forms, so an Ornstein-Uhlenbeck
    process needs no simulation code of its own -- it is a Gaussian process,
    and these two functions are all that distinguishes it from Brownian
    motion.

    Parameters
    ----------
    reversion_rate : float
        How strongly the process is pulled back toward ``mean``.
    mean : float
        The level the process is pulled toward.
    scale : float
        The volatility of the random shocks.
    initial_value : float or str
        A number to start every path from, or ``"stationary"`` to start from
        the process's long-run distribution.

    Returns
    -------
    tuple of callable
        The pair ``(mean_func, cov_func)`` to hand to a Gaussian process.

    Notes
    -----
    ``scale ** 2 / (2 * reversion_rate)`` is the long-run variance: the
    variance the process settles down to once the pull toward ``mean`` and the
    random shocks balance out. It appears in both parameterizations.

    Started from ``"stationary"``, the process is *stationary* -- its
    covariance depends only on how far apart two times are, ``abs(s - t)``,
    not on where they sit on the clock. Starting from a fixed number instead
    adds the ``exp(-reversion_rate * (s + t))`` correction, which fades as
    time passes: that is the transient, the stretch of path where the process
    is still travelling from where it started toward ``mean``.
    """
    long_run_var = scale**2 / (2 * reversion_rate)

    if initial_value == STATIONARY:

        def mean_func(t):
            return mean

        def cov_func(s, t):
            return long_run_var * np.exp(-reversion_rate * abs(s - t))

    else:

        def mean_func(t):
            # Travels exponentially from initial_value toward mean.
            return mean + (initial_value - mean) * np.exp(-reversion_rate * t)

        def cov_func(s, t):
            # At s = t = 0 this is exactly 0, so every path starts at
            # initial_value with no randomness -- the same way Brownian
            # motion starts at 0.
            return long_run_var * (
                np.exp(-reversion_rate * abs(s - t)) - np.exp(-reversion_rate * (s + t))
            )

    return mean_func, cov_func


def _validate_ornstein_uhlenbeck(reversion_rate, mean, scale, initial_value):
    """Check the parameters of an Ornstein-Uhlenbeck process.

    Raises
    ------
    TypeError
        If ``reversion_rate``, ``mean``, or ``scale`` is not a number, or if
        ``initial_value`` is neither a number nor ``"stationary"``.
    ValueError
        If ``reversion_rate`` or ``scale`` is not positive.
    """
    if not isinstance(reversion_rate, numbers.Real):
        raise TypeError(
            f"reversion_rate must be a number, got "
            f"{type(reversion_rate).__name__}. It says how strongly the "
            f"process is pulled back toward mean, for example "
            f"reversion_rate=1."
        )
    if not isinstance(mean, numbers.Real):
        raise TypeError(
            f"mean must be a number, got {type(mean).__name__}. It is the "
            f"level the process is pulled toward, for example mean=0."
        )
    if not isinstance(scale, numbers.Real):
        raise TypeError(
            f"scale must be a number, got {type(scale).__name__}. It is the "
            f"volatility of the random shocks, for example scale=1."
        )
    if not (isinstance(initial_value, numbers.Real) or initial_value == STATIONARY):
        raise TypeError(
            f'initial_value must be a number or the word "stationary", got '
            f"{type(initial_value).__name__}. Give a number to start every "
            f'path there, or initial_value="stationary" to start from the '
            f"process's long-run distribution."
        )
    if reversion_rate <= 0:
        raise ValueError(
            f"reversion_rate must be positive, got {reversion_rate}. With no "
            f"pull back toward mean the process would not revert at all -- "
            f"that process is BrownianMotion(drift=0, scale=scale) instead."
        )
    if scale <= 0:
        raise ValueError(
            f"scale must be positive, got {scale}. A scale of 0 would give a "
            "curve with no randomness, sliding straight from initial_value to "
            "mean."
        )


# Define convenience class for the Ornstein-Uhlenbeck process
class OrnsteinUhlenbeckProbabilitySpace(GaussianProcessProbabilitySpace):
    """The probability space underlying an Ornstein-Uhlenbeck process.

    Each draw from this space produces one simulated sample path of the
    Ornstein-Uhlenbeck process. Paths are generated lazily, exactly as for
    any other Gaussian process.

    Parameters
    ----------
    reversion_rate : float, optional
        How strongly the process is pulled back toward ``mean``. Must be
        positive. Larger values pull harder, so the path stays closer to
        ``mean``. Default is 1.
    mean : float, optional
        The level the process is pulled toward. Default is 0.
    scale : float, optional
        The volatility of the random shocks. Must be positive. Default is 1.
    initial_value : float or str, optional
        Where every path starts. Give a number, or ``"stationary"`` to start
        from the process's long-run distribution. Default is 0.

    Attributes
    ----------
    reversion_rate : float
        How strongly the process is pulled back toward ``mean``.
    mean : float
        The level the process is pulled toward.
    scale : float
        The volatility of the random shocks.
    initial_value : float or str
        Where every path starts.

    Raises
    ------
    TypeError
        If ``reversion_rate``, ``mean``, or ``scale`` is not a number, or if
        ``initial_value`` is neither a number nor ``"stationary"``.
    ValueError
        If ``reversion_rate`` or ``scale`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> P = OrnsteinUhlenbeckProbabilitySpace(reversion_rate=1, mean=0, scale=1)
    >>> float(P.draw()(0.0))
    0.0
    >>> path = P.draw()  # doctest: +SKIP
    >>> path(1.0)        # doctest: +SKIP
    -0.29
    """

    def __init__(self, reversion_rate=1, mean=0, scale=1, initial_value=0):
        """Create a probability space for an Ornstein-Uhlenbeck process."""
        _validate_ornstein_uhlenbeck(reversion_rate, mean, scale, initial_value)

        self.reversion_rate = reversion_rate
        self.mean = mean
        self.scale = scale
        self.initial_value = initial_value

        mean_func, cov_func = _ornstein_uhlenbeck_funcs(
            reversion_rate, mean, scale, initial_value
        )
        super().__init__(mean_func=mean_func, cov_func=cov_func)


class OrnsteinUhlenbeck(RandomProcess, RV):
    """An Ornstein-Uhlenbeck process, a random variable over sample paths.

    Brownian motion wanders off and never comes back. An Ornstein-Uhlenbeck
    process is Brownian motion on a leash: the further it strays from
    ``mean``, the harder it is pulled back, so instead of drifting away
    forever it settles into wandering around one level. That makes it the
    standard model for a quantity that fluctuates but does not run away -- an
    interest rate (where it is known as the Vasicek model), a price relative
    to its long-run average, or the velocity of a particle being slowed by
    friction, which is the setting it was invented for.

    It is often written as
    ``dX = reversion_rate * (mean - X) dt + scale * dW``: the first term is
    the pull back toward ``mean``, strongest when ``X`` is furthest away, and
    the second is the random jostling that keeps it moving. It is still a
    Gaussian process, so Symbulate simulates it exactly, with the same
    machinery as :class:`BrownianMotion` and a different covariance function.

    Parameters
    ----------
    reversion_rate : float, optional
        How strongly the process is pulled back toward ``mean``. Must be
        positive. Larger values pull harder, so the path stays closer to
        ``mean``. Default is 1.
    mean : float, optional
        The level the process is pulled toward. Default is 0.
    scale : float, optional
        The volatility of the random shocks. Must be positive. Default is 1.
    initial_value : float or str, optional
        Where every path starts. Give a number, and the path begins there
        exactly and travels toward ``mean`` -- the *transient*. Pass
        ``"stationary"`` to start from the long-run distribution instead, so
        there is no transient and the process looks the same at every time.
        Default is 0.

    Attributes
    ----------
    prob_space : OrnsteinUhlenbeckProbabilitySpace
        The underlying probability space used to generate sample paths.

    Notes
    -----
    Two facts describe the long-run behavior. However far a path starts from
    ``mean``, the distance still left to travel is multiplied by
    ``exp(-reversion_rate * t)`` as time passes, so the mean approaches
    ``mean``. Meanwhile the variance climbs to
    ``scale ** 2 / (2 * reversion_rate)``, the point where the inward pull and
    the random shocks balance. Together those give the long-run distribution
    ``Normal(mean, sd=sqrt(scale ** 2 / (2 * reversion_rate)))``, which is
    what ``initial_value="stationary"`` starts from.

    The covariance between two times falls off like
    ``exp(-reversion_rate * abs(s - t))``: nearby times are strongly related
    and distant ones are nearly independent. So a large ``reversion_rate``
    gives a jagged path with a short memory, and a small one gives a smooth,
    slowly-wandering path.

    Examples
    --------
    >>> from symbulate import *
    >>> X = OrnsteinUhlenbeck(reversion_rate=1, mean=0, scale=1)
    >>> # Every path starts exactly at initial_value
    >>> float(X.draw()(0.0))
    0.0
    >>> path = X.draw()                    # doctest: +SKIP
    >>> path(0.5), path(1.0), path(5.0)    # doctest: +SKIP
    (-0.41, -0.76, 0.33)
    >>> # Starting far from mean, the process is pulled toward it
    >>> Y = OrnsteinUhlenbeck(reversion_rate=1, mean=0, scale=1, initial_value=10)
    >>> Y[3.0].sim(1000).mean()            # doctest: +SKIP
    0.51
    >>> # Started from its long-run distribution, there is no transient
    >>> Z = OrnsteinUhlenbeck(initial_value="stationary")
    >>> Z[0.0].sim(1000).var()             # doctest: +SKIP
    0.49

    See Also
    --------
    BrownianMotion : The same machinery without the pull back toward a mean.
    GaussianProcess : The general process both are built on.
    """

    def __init__(self, reversion_rate=1, mean=0, scale=1, initial_value=0):
        """Create an Ornstein-Uhlenbeck process."""
        prob_space = OrnsteinUhlenbeckProbabilitySpace(
            reversion_rate=reversion_rate,
            mean=mean,
            scale=scale,
            initial_value=initial_value,
        )
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)


def _validate_brownian_bridge(end_time, initial_value, final_value, scale):
    """Check the parameters of a Brownian bridge.

    Raises
    ------
    TypeError
        If any parameter is not a number.
    ValueError
        If ``end_time`` or ``scale`` is not positive.
    """
    for name, value, example in [
        ("end_time", end_time, "end_time=1"),
        ("initial_value", initial_value, "initial_value=0"),
        ("final_value", final_value, "final_value=0"),
        ("scale", scale, "scale=1"),
    ]:
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f"{name} must be a number, got {type(value).__name__}. "
                f"For example, {example}."
            )
    if end_time <= 0:
        raise ValueError(
            f"end_time must be positive, got {end_time}. It is when the "
            f"bridge finishes, and it starts at time 0, so it needs a "
            f"stretch of time to cross."
        )
    if scale <= 0:
        raise ValueError(
            f"scale must be positive, got {scale}. A scale of 0 would give a "
            "straight line from initial_value to final_value with no "
            "randomness."
        )


# Define convenience class for the Brownian bridge
class BrownianBridgeProbabilitySpace(GaussianProcessProbabilitySpace):
    """The probability space underlying a Brownian bridge.

    Each draw from this space produces one simulated sample path of the
    Brownian bridge. Paths are generated lazily, exactly as for any other
    Gaussian process.

    Parameters
    ----------
    end_time : float, optional
        When the bridge finishes. Must be positive. Default is 1.
    initial_value : float, optional
        The value the bridge starts at, at time 0. Default is 0.
    final_value : float, optional
        The value the bridge is tied down to at ``end_time``. Default is 0.
    scale : float, optional
        How much the path wanders between the two ends. Must be positive.
        Default is 1.

    Attributes
    ----------
    end_time : float
        When the bridge finishes.
    initial_value : float
        The value the bridge starts at.
    final_value : float
        The value the bridge ends at.
    scale : float
        How much the path wanders between the two ends.

    Raises
    ------
    TypeError
        If any parameter is not a number.
    ValueError
        If ``end_time`` or ``scale`` is not positive.

    Examples
    --------
    >>> from symbulate import *
    >>> P = BrownianBridgeProbabilitySpace(end_time=1)
    >>> path = P.draw()
    >>> float(path(0)), float(path(1))
    (0.0, 0.0)
    """

    def __init__(self, end_time=1, initial_value=0, final_value=0, scale=1):
        """Create a probability space for a Brownian bridge."""
        _validate_brownian_bridge(end_time, initial_value, final_value, scale)

        self.end_time = end_time
        self.initial_value = initial_value
        self.final_value = final_value
        self.scale = scale

        def mean_func(t):
            # A straight line from initial_value to final_value.
            return initial_value + (final_value - initial_value) * t / end_time

        def cov_func(s, t):
            # Brownian motion's min(s, t), minus the part that pinning down
            # the far end removes. This is 0 whenever s or t is 0 or
            # end_time, which is what nails the path to both ends.
            return scale**2 * (min(s, t) - s * t / end_time)

        # The bridge exists only between its two ends. Outside that stretch
        # this covariance would be negative, which is not a variance at all,
        # so the index set turns an out-of-range time into a clear error
        # instead of a meaningless number.
        super().__init__(
            mean_func=mean_func,
            cov_func=cov_func,
            index_set=TimeInterval(0, end_time),
        )


class BrownianBridge(RandomProcess, RV):
    """A Brownian bridge, a random variable over sample paths.

    A Brownian bridge is Brownian motion that already knows where it has to
    end up. Ordinary Brownian motion starts at 0 and wanders off wherever it
    likes. A bridge is tied down at *both* ends -- it starts at
    ``initial_value`` and must arrive at ``final_value`` at time
    ``end_time`` -- so it wanders in between but always lands on target.

    Because both ends are fixed, the path has nowhere to wander at the very
    start or the very finish, and the most freedom in the middle. Its spread
    at time ``t`` is ``scale ** 2 * t * (end_time - t) / end_time``, which is
    0 at each end and largest halfway across.

    It shows up whenever the end of a random path is already known: the
    standard example in a statistics course is the Kolmogorov-Smirnov
    statistic, which measures the gap between a sample's ECDF and the true
    CDF, and behaves like a Brownian bridge because that gap is pinned at 0
    at both ends by construction.

    Parameters
    ----------
    end_time : float, optional
        When the bridge finishes. Must be positive. The process is only
        defined between time 0 and this time. Default is 1.
    initial_value : float, optional
        The value the bridge starts at, at time 0. Default is 0.
    final_value : float, optional
        The value the bridge is tied down to at ``end_time``. Default is 0.
    scale : float, optional
        How much the path wanders between the two ends. Must be positive.
        Default is 1.

    Attributes
    ----------
    prob_space : BrownianBridgeProbabilitySpace
        The underlying probability space used to generate sample paths.

    Notes
    -----
    Asking for a time before 0 or after ``end_time`` raises a ``KeyError``,
    because the bridge simply does not exist there. This is unlike
    :class:`BrownianMotion`, which runs forever.

    A bridge is exactly what Brownian motion looks like once you are told
    where it ended: subtracting the straight line ``t / end_time * B(end_time)``
    from a Brownian motion ``B`` gives a standard Brownian bridge.

    Examples
    --------
    >>> from symbulate import *
    >>> X = BrownianBridge(end_time=1)
    >>> path = X.draw()
    >>> # Both ends are exact, every time
    >>> float(path(0)), float(path(1))
    (0.0, 0.0)
    >>> # Tie the far end down somewhere else
    >>> Y = BrownianBridge(end_time=4, initial_value=2, final_value=10)
    >>> path = Y.draw()
    >>> float(path(0)), float(path(4))
    (2.0, 10.0)
    >>> # The spread is widest halfway across
    >>> X[0.5].sim(1000).var()   # doctest: +SKIP
    0.24

    See Also
    --------
    BrownianMotion : The same process without the far end tied down.
    TimeInterval : The index set that limits the bridge to its own stretch of time.
    """

    def __init__(self, end_time=1, initial_value=0, final_value=0, scale=1):
        """Create a Brownian bridge."""
        prob_space = BrownianBridgeProbabilitySpace(
            end_time=end_time,
            initial_value=initial_value,
            final_value=final_value,
            scale=scale,
        )
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)
