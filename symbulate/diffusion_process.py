import bisect
import numbers

import numpy as np
import scipy.stats as stats

from .index_sets import Reals
from .probability_space import ProbabilitySpace
from .result import ContinuousTimeFunction
from .random_variables import RV
from .random_processes import RandomProcess

rng = np.random.default_rng()


def get_diffusion_process_result(drift, diffusion, x0=0, tol=1e-3):
    """Create one simulated sample path of an Ito diffusion.

    The path solves (in the Euler-Maruyama sense)::

        dX_t = drift(X_t, t) dt + diffusion(X_t, t) dW_t,   X_0 = x0

    Values are generated lazily: the path is only ever resolved at the
    times someone actually asks for. Whenever a new time falls between
    two times that have already been simulated, a *bridge* value is
    drawn that is consistent with both of its neighbors -- so zooming
    in on any portion of an already-drawn path fills in detail without
    ever contradicting the coarser picture you started with. This is
    the same lazy/conditional idea ``GaussianProcessResult`` uses for
    Brownian motion, generalized to diffusions whose finite-dimensional
    distributions are not Gaussian and have no closed form.

    Parameters
    ----------
    drift : callable
        A function ``mu(x, t)`` giving the drift of the process.
    diffusion : callable
        A function ``sigma(x, t)`` giving the (non-negative) diffusion
        coefficient of the process.
    x0 : float, optional
        The starting value ``X_0``. Default is 0.
    tol : float, optional
        The finest time resolution the path is allowed to be refined
        to. Smaller values give a more accurate approximation of the
        true diffusion at the cost of more computation. Default 1e-3.

    Returns
    -------
    DiffusionProcessResult
        A sample path that can be evaluated at any nonnegative time.

    Examples
    --------
    >>> from symbulate import *
    >>> path = get_diffusion_process_result(
    ...     drift=lambda x, t: 0,
    ...     diffusion=lambda x, t: 1,
    ... )
    >>> path(1.0)  # doctest: +SKIP
    0.43
    """

    class DiffusionProcessResult(ContinuousTimeFunction):
        """One simulated sample path of a diffusion process.

        Evaluating this object at a time ``t`` returns the value of the
        path at that moment. Times are cached the first time they are
        computed, so re-evaluating the same time always gives the same
        value, and evaluating a new time in between two cached times
        produces a value consistent with both (a "bridge" value) rather
        than an independent new draw.

        Attributes
        ----------
        times : list
            Cached times observed so far, kept in sorted order.
        values : list
            Simulated values corresponding to ``times``.
        """

        def __init__(self, drift, diffusion, x0):
            """Create one simulated sample path of a diffusion process."""
            self.times = [0.0]
            self.values = [float(x0)]

            def _func(t):
                return self._evaluate(t)

            super().__init__(func=_func)
            self.index_set = Reals()

        def _evaluate(self, t):
            if t < 0:
                raise ValueError(
                    "DiffusionProcess is only defined for t >= 0 "
                    f"(the process starts at X(0)={self.values[0]}), got t={t}."
                )

            i = bisect.bisect_left(self.times, t)

            # Already have this exact time cached.
            if i < len(self.times) and self.times[i] == t:
                return self.values[i]

            # t is beyond every time simulated so far: extend forward.
            if i == len(self.times):
                return self._simulate_forward(t)

            # t falls strictly between two already-observed times:
            # fill it in with a bridge value.
            return self._bridge_sample(i - 1, i, t)

        def _simulate_forward(self, t):
            """Extend the path forward (Euler-Maruyama) up to time t."""
            t_cur = self.times[-1]
            x_cur = self.values[-1]
            n_steps = max(1, int(np.ceil((t - t_cur) / tol)))
            dt = (t - t_cur) / n_steps
            sqrt_dt = np.sqrt(dt)
            for _ in range(n_steps):
                dw = rng.normal(0, sqrt_dt)
                x_cur = x_cur + drift(x_cur, t_cur) * dt + diffusion(x_cur, t_cur) * dw
                t_cur += dt
                self.times.append(t_cur)
                self.values.append(x_cur)
            # Guard against floating-point drift in the last step.
            self.times[-1] = t
            return self.values[-1]

        def _bridge_sample(self, i0, i1, t):
            """Recursively fill in a value at time t between two cached
            points times[i0] and times[i1], via repeated midpoint
            bisection. This is exact when drift=0 and diffusion is a
            constant (i.e., for Brownian motion), and is a first-order
            (Euler-Maruyama-consistent) local approximation otherwise,
            which converges as the bisection depth increases.
            """
            t0, t1 = self.times[i0], self.times[i1]
            x0_, x1_ = self.values[i0], self.values[i1]

            if t1 - t0 <= tol:
                return self._draw_bridge_point(t0, x0_, t1, x1_, t)

            tm = (t0 + t1) / 2
            xm = self._draw_bridge_point(t0, x0_, t1, x1_, tm)
            self.times.insert(i1, tm)
            self.values.insert(i1, xm)

            if t == tm:
                return xm
            elif t < tm:
                return self._bridge_sample(i0, i1, t)
            else:
                return self._bridge_sample(i1, i1 + 1, t)

        def _draw_bridge_point(self, t0, x0_, t1, x1_, t):
            """Sample X(t) given X(t0)=x0_ and X(t1)=x1_, using a local
            Gaussian (Brownian-bridge-type) approximation with the
            drift and diffusion coefficients evaluated at (x0_, t0).
            """
            dt = t1 - t0
            dt0 = t - t0
            dt1 = t1 - t
            mu = drift(x0_, t0)
            sigma = diffusion(x0_, t0)

            # Bridge mean: linear interpolation between the endpoints,
            # adjusted so the average drift over [t0, t1] matches mu.
            mean = x0_ + (dt0 / dt) * (x1_ - x0_ - mu * dt) + mu * dt0
            var = max(sigma**2 * dt0 * dt1 / dt, 0.0)

            return rng.normal(mean, np.sqrt(var)) if var > 0 else mean

    return DiffusionProcessResult(drift, diffusion, x0)


class DiffusionProcessProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a diffusion process.

    Each draw from this space produces one simulated sample path of
    the Ito diffusion ``dX_t = drift(X_t, t) dt + diffusion(X_t, t) dW_t``,
    ``X_0 = x0``. Paths are generated lazily -- values at new times are
    computed on demand, consistent with all previously observed values.

    Parameters
    ----------
    drift : callable
        A function ``mu(x, t)`` giving the drift of the process.
    diffusion : callable
        A function ``sigma(x, t)`` giving the diffusion coefficient of
        the process.
    x0 : float, optional
        The starting value ``X_0``. Default is 0.
    tol : float, optional
        The finest time resolution the path is allowed to be refined
        to. Default 1e-3.

    Raises
    ------
    TypeError
        If ``drift`` or ``diffusion`` is not callable.

    Examples
    --------
    >>> from symbulate import *
    >>> P = DiffusionProcessProbabilitySpace(
    ...     drift=lambda x, t: 0.1 * x,
    ...     diffusion=lambda x, t: 0.2 * x,
    ...     x0=1,
    ... )
    >>> path = P.draw()  # doctest: +SKIP
    >>> path(1.0)         # doctest: +SKIP
    1.14
    """

    def __init__(self, drift, diffusion, x0=0, tol=1e-3):
        """Create a probability space for a diffusion process."""
        if not callable(drift):
            raise TypeError(
                f"drift must be callable (e.g., lambda x, t: 0), "
                f"got {type(drift).__name__}."
            )
        if not callable(diffusion):
            raise TypeError(
                f"diffusion must be callable (e.g., lambda x, t: 1), "
                f"got {type(diffusion).__name__}."
            )
        if not isinstance(x0, (int, float)):
            raise TypeError(f"x0 must be a number, got {type(x0).__name__}.")
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError(f"tol must be a positive number, got {tol!r}.")

        def draw():
            return get_diffusion_process_result(drift, diffusion, x0, tol)

        super().__init__(draw)


class DiffusionProcess(RandomProcess, RV):
    """A general Ito diffusion process, modeled as a random variable
    over sample paths.

    A diffusion process solves the stochastic differential equation

    .. math::

        dX_t = \\mu(X_t, t)\\, dt + \\sigma(X_t, t)\\, dW_t,
        \\qquad X_0 = x_0,

    where :math:`W_t` is a standard Brownian motion. Unlike
    ``GaussianProcess``/``BrownianMotion``, a diffusion process is
    generally *not* Gaussian, so there is no closed-form joint
    distribution to condition on exactly. Instead, sample paths are
    generated lazily via a bridge-consistent Euler-Maruyama scheme:
    querying a new time between two already-simulated times draws a
    value consistent with both of them, refining the mesh as finely as
    needed. As a result, one draw's path can be "zoomed in" on
    repeatedly -- exactly like ``BrownianMotion`` -- and the finer
    detail will always agree with the coarser picture already drawn.

    Parameters
    ----------
    drift : callable
        A function ``mu(x, t)`` giving the drift of the process.
    diffusion : callable
        A function ``sigma(x, t)`` giving the diffusion coefficient of
        the process. Should be non-negative.
    x0 : float, optional
        The starting value ``X_0``. Default is 0.
    tol : float, optional
        The finest time resolution the path is allowed to be refined
        to. Smaller values trade accuracy for speed. Default 1e-3.

    Attributes
    ----------
    prob_space : DiffusionProcessProbabilitySpace
        The underlying probability space used to generate sample paths.

    Examples
    --------
    >>> from symbulate import *
    >>> # Ornstein-Uhlenbeck process: dX = -X dt + dW
    >>> X = DiffusionProcess(
    ...     drift=lambda x, t: -x,
    ...     diffusion=lambda x, t: 1,
    ...     x0=0,
    ... )
    >>> path = X.draw()             # doctest: +SKIP
    >>> path(1.0), path(2.0)        # doctest: +SKIP
    (-0.32, 0.11)
    >>> # Zoom in: values on [0, 0.1] are consistent with the path
    >>> # already drawn on [0, 10].
    >>> path(0.05)                  # doctest: +SKIP
    -0.02

    >>> # Geometric Brownian motion: dX = mu*X dt + sigma*X dW
    >>> mu, sigma = 0.05, 0.2
    >>> S = DiffusionProcess(
    ...     drift=lambda x, t: mu * x,
    ...     diffusion=lambda x, t: sigma * x,
    ...     x0=100,
    ... )
    >>> path = S.draw()             # doctest: +SKIP
    >>> path(1.0)                   # doctest: +SKIP
    104.3
    """

    def __init__(self, drift, diffusion, x0=0, tol=1e-3):
        """Create a diffusion process."""
        prob_space = DiffusionProcessProbabilitySpace(drift, diffusion, x0, tol)
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)


def _validate_cir(reversion_rate, mean, scale, initial_value):
    """Check the parameters of a CIR process.

    Raises
    ------
    TypeError
        If any parameter is not a number.
    ValueError
        If ``reversion_rate``, ``mean``, or ``scale`` is not positive, or
        ``initial_value`` is negative.
    """
    for name, value, example in [
        ("reversion_rate", reversion_rate, "reversion_rate=1"),
        ("mean", mean, "mean=0.05"),
        ("scale", scale, "scale=0.1"),
    ]:
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f"{name} must be a number, got {type(value).__name__}. "
                f"For example, {example}."
            )
        if value <= 0:
            raise ValueError(
                f"{name} must be positive, got {value}. A CIR process lives "
                f"above 0 and is pulled toward a positive level, so all three "
                f"of reversion_rate, mean, and scale have to be positive."
            )
    if initial_value is not None:
        if not isinstance(initial_value, numbers.Real):
            raise TypeError(
                f"initial_value must be a number or None, got "
                f"{type(initial_value).__name__}. Leave it out to start at "
                f"mean."
            )
        if initial_value < 0:
            raise ValueError(
                f"initial_value must be at least 0, got {initial_value}. A "
                f"CIR process never goes below 0, so it cannot start there."
            )


def get_cir_result(reversion_rate, mean, scale, initial_value):
    """Create one simulated sample path of a CIR process.

    Unlike :func:`get_diffusion_process_result`, this does not take small
    steps. The CIR process has a known transition law -- given its value now,
    its value at any later time is a **scaled noncentral chi-square** -- so
    each new time is drawn in one shot, exactly, however far ahead it is.
    (That is the same distribution ``ChiSquare(df, noncentrality=...)``
    wraps; it is used through scipy here so that seeding this module's
    ``rng`` controls the whole path.)

    Times are cached, so re-asking a time gives the same value back. Asking
    for a time that falls *between* two times already computed is the one
    approximate case -- see :class:`CIR`.

    Parameters
    ----------
    reversion_rate : float
        How strongly the process is pulled back toward ``mean``.
    mean : float
        The positive level it is pulled toward.
    scale : float
        The volatility multiplier. The size of the random jolts is
        ``scale * sqrt(value)``, which shrinks to nothing as the value
        approaches 0.
    initial_value : float or None
        Where the path starts. ``None`` starts it at ``mean``.

    Returns
    -------
    CIRResult
        A sample path that can be evaluated at any time ``t >= 0``.

    Examples
    --------
    >>> from symbulate import *
    >>> path = get_cir_result(1.0, 0.05, 0.1, 0.03)
    >>> float(path(0))
    0.03
    >>> path(1.0)  # doctest: +SKIP
    0.047
    """
    start = float(mean if initial_value is None else initial_value)

    # Degrees of freedom of the transition law. This is also what decides
    # whether the process can ever touch 0: at df >= 2 -- equivalently
    # 2 * reversion_rate * mean >= scale ** 2, the Feller condition -- it
    # never does.
    degrees_of_freedom = 4.0 * reversion_rate * mean / scale**2

    class CIRResult(ContinuousTimeFunction):
        """One simulated sample path of a CIR process.

        Attributes
        ----------
        times : list
            Cached times observed so far, kept in sorted order.
        values : list
            Simulated values corresponding to ``times``.
        degrees_of_freedom : float
            Degrees of freedom of the transition law, ``4 * reversion_rate *
            mean / scale ** 2``. At 2 or above the path never reaches 0.
        """

        def __init__(self):
            """Create one simulated sample path of a CIR process."""
            self.times = [0.0]
            self.values = [start]
            self.degrees_of_freedom = degrees_of_freedom

            super().__init__(func=self._evaluate)
            self.index_set = Reals()

        def _evaluate(self, t):
            if t < 0:
                raise ValueError(
                    "CIR is only defined for t >= 0 (the path starts at "
                    f"{self.values[0]} at time 0), got t={t}."
                )

            i = bisect.bisect_left(self.times, t)
            if i < len(self.times) and self.times[i] == t:
                return self.values[i]

            if i == len(self.times):
                return self._step_forward(t)

            return self._bridge(i - 1, i, t)

        def _step_forward(self, t):
            """Draw the value at ``t`` from the exact transition law."""
            t_last = self.times[-1]
            x_last = self.values[-1]
            value = _cir_transition(
                x_last, t - t_last, reversion_rate, mean, scale, degrees_of_freedom
            )
            self.times.append(t)
            self.values.append(value)
            return value

        def _bridge(self, i0, i1, t):
            """Fill in a value between two already-known times.

            This is the one place the path is not exact. There is no simple
            formula for a CIR process pinned at both ends, so this uses the
            ordinary Brownian-bridge formula with the jolt size frozen at the
            left-hand value, then keeps the result at or above 0. The error
            shrinks the closer the two known times are.
            """
            t0, t1 = self.times[i0], self.times[i1]
            x0, x1 = self.values[i0], self.values[i1]

            spread = t1 - t0
            ahead, behind = t - t0, t1 - t
            drift = reversion_rate * (mean - x0)
            jolt = scale * np.sqrt(max(x0, 0.0))

            center = x0 + (ahead / spread) * (x1 - x0 - drift * spread) + drift * ahead
            variance = max(jolt**2 * ahead * behind / spread, 0.0)
            value = (
                float(rng.normal(center, np.sqrt(variance))) if variance > 0 else center
            )
            value = max(value, 0.0)

            self.times.insert(i1, t)
            self.values.insert(i1, value)
            return value

    return CIRResult()


def _cir_transition(value, elapsed, reversion_rate, mean, scale, degrees_of_freedom):
    """Draw the CIR value ``elapsed`` later, given it is ``value`` now.

    The transition law is exact: the value ahead is ``spread`` times a
    noncentral chi-square, where the noncentrality carries the current value
    forward. Its mean works out to ``value * decay + mean * (1 - decay)``,
    which is the same pull toward ``mean`` an Ornstein-Uhlenbeck process has.

    Parameters
    ----------
    value : float
        The value now.
    elapsed : float
        How far ahead to jump. Must be positive.
    reversion_rate, mean, scale : float
        The process parameters.
    degrees_of_freedom : float
        ``4 * reversion_rate * mean / scale ** 2``, passed in so it is not
        recomputed on every step.

    Returns
    -------
    float
        The value at the later time, never negative.
    """
    decay = np.exp(-reversion_rate * elapsed)
    spread = scale**2 * (1.0 - decay) / (4.0 * reversion_rate)
    if spread <= 0:
        return float(value)

    noncentrality = value * decay / spread
    draw = stats.ncx2.rvs(df=degrees_of_freedom, nc=noncentrality, random_state=rng)
    return float(max(spread * draw, 0.0))


class CIRProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a CIR process.

    Each draw produces one simulated sample path, generated lazily from the
    process's exact transition law (see :func:`get_cir_result`).

    Parameters
    ----------
    reversion_rate : float, optional
        How strongly the process is pulled back toward ``mean``. Must be
        positive. Default is 1.
    mean : float, optional
        The positive level it is pulled toward. Default is 1.
    scale : float, optional
        The volatility multiplier. Must be positive. Default is 0.5.
    initial_value : float, optional
        Where the path starts. Leave it out to start at ``mean``.

    Attributes
    ----------
    reversion_rate, mean, scale : float
        The process parameters.
    initial_value : float
        Where paths start -- ``mean`` if none was given.

    Raises
    ------
    TypeError
        If any parameter is not a number.
    ValueError
        If ``reversion_rate``, ``mean``, or ``scale`` is not positive, or
        ``initial_value`` is negative.

    Examples
    --------
    >>> from symbulate import *
    >>> P = CIRProbabilitySpace(reversion_rate=1, mean=0.05, scale=0.1)
    >>> float(P.draw()(0))
    0.05
    """

    def __init__(self, reversion_rate=1, mean=1, scale=0.5, initial_value=None):
        """Create a probability space for a CIR process."""
        _validate_cir(reversion_rate, mean, scale, initial_value)

        self.reversion_rate = reversion_rate
        self.mean = mean
        self.scale = scale
        self.initial_value = float(mean if initial_value is None else initial_value)

        def draw():
            return get_cir_result(reversion_rate, mean, scale, initial_value)

        super().__init__(draw)


class CIR(RandomProcess, RV):
    """The Cox-Ingersoll-Ross process, a random variable over sample paths.

    A mean-reverting process that **stays positive**. It behaves like an
    :class:`OrnsteinUhlenbeck` process -- pulled back toward ``mean``, harder
    the further away it strays -- with one change: the size of the random
    jolts is ``scale * sqrt(value)`` rather than a constant. As the value
    approaches 0 the jolts shrink to nothing while the upward pull remains, so
    the process is turned back before it can go negative.

    That is why it is the standard model for an interest rate, which moves
    around a long-run level but cannot sensibly go below 0.

    Unlike :class:`DiffusionProcess`, this is simulated **exactly**. The CIR
    process has a known transition law -- given its value now, its value at
    any later time is a scaled noncentral chi-square -- so no small steps and
    no accumulated error are involved.

    Parameters
    ----------
    reversion_rate : float, optional
        How strongly the process is pulled back toward ``mean``. Must be
        positive. Larger values hold it closer. Default is 1.
    mean : float, optional
        The level it is pulled toward. Must be positive. Default is 1.
    scale : float, optional
        The volatility multiplier -- the jolt size is ``scale *
        sqrt(value)``. Must be positive. Default is 0.5.
    initial_value : float, optional
        Where every path starts. Leave it out to start at ``mean``, which is
        usually what you want. Must be at least 0.

    Attributes
    ----------
    prob_space : CIRProbabilitySpace
        The underlying probability space used to generate sample paths.

    Notes
    -----
    **Whether it can touch 0** is decided by the *Feller condition*,
    ``2 * reversion_rate * mean >= scale ** 2``. When it holds the path never
    reaches 0 at all. When it fails -- a large ``scale`` relative to the pull
    -- the path can touch 0, though it is immediately pushed back up and
    never goes below.

    **Left alone it settles** into a ``Gamma`` distribution with shape
    ``2 * reversion_rate * mean / scale ** 2`` and rate
    ``2 * reversion_rate / scale ** 2``, whose mean is ``mean`` and whose
    variance is ``mean * scale ** 2 / (2 * reversion_rate)``.

    **One approximate corner.** Asking for a time that falls *between* two
    times already computed on the same path uses a bridge approximation,
    because a CIR process pinned at both ends has no simple formula. Asking
    times in increasing order -- which is what plotting and ``X[t]`` do -- is
    exact.

    Examples
    --------
    >>> from symbulate import *
    >>> X = CIR(reversion_rate=1, mean=0.05, scale=0.1)
    >>> # Paths start at mean unless told otherwise
    >>> float(X.draw()(0))
    0.05
    >>> X[5.0].sim(1000).mean()      # doctest: +SKIP
    0.0503
    >>> # Started low, it climbs toward mean
    >>> Y = CIR(reversion_rate=1, mean=0.05, scale=0.1, initial_value=0.01)
    >>> Y[3.0].sim(1000).mean()      # doctest: +SKIP
    0.048

    See Also
    --------
    OrnsteinUhlenbeck : The same mean reversion, but able to go negative.
    DiffusionProcess : The general, approximate way to write any SDE.
    """

    def __init__(self, reversion_rate=1, mean=1, scale=0.5, initial_value=None):
        """Create a CIR process."""
        prob_space = CIRProbabilitySpace(
            reversion_rate=reversion_rate,
            mean=mean,
            scale=scale,
            initial_value=initial_value,
        )
        RandomProcess.__init__(self, prob_space)
        RV.__init__(self, prob_space)
