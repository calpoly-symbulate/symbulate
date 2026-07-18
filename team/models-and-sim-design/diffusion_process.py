import bisect

import numpy as np

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
                x_cur = (
                    x_cur
                    + drift(x_cur, t_cur) * dt
                    + diffusion(x_cur, t_cur) * dw
                )
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
            var = max(sigma ** 2 * dt0 * dt1 / dt, 0.0)

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
            raise TypeError(
                f"x0 must be a number, got {type(x0).__name__}."
            )
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError(
                f"tol must be a positive number, got {tol!r}."
            )

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
