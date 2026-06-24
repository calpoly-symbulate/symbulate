import numpy as np

from .index_sets import DiscreteTimeSequence, Reals
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
                            f"Time must be in the index set ({type(index_set).__name__})."
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
            raise TypeError(
                f"drift must be a number, got {type(drift).__name__}."
            )
        if not isinstance(scale, (int, float)):
            raise TypeError(
                f"scale must be a number, got {type(scale).__name__}."
            )
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
