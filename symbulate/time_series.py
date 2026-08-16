import math
import numbers

import numpy as np

from .distributions import (
    Distribution,
    MultivariateDistribution,
    MultivariateNormal,
    Normal,
)
from .probability_space import ProbabilitySpace
from .random_variables import RV

# The same lazily-cached i.i.d. sequence RandomWalk uses for its steps. A
# time-series process needs exactly the same thing for its shocks, so the
# helper is shared rather than duplicated; it is imported under a name that
# reads correctly here. (Worth moving somewhere neutral once the rest of the
# time-series family lands and a shared home is obvious.)
from .random_processes import _resolve_initial
from .random_walk import _step_source as _iid_source
from .result import InfiniteVector


def _validate_ma(coefs, noise_dist, mean):
    """Check the parameters of a moving-average process.

    Raises
    ------
    TypeError
        If ``coefs`` is not a sequence of numbers, ``noise_dist`` is neither
        a ``Distribution`` nor an ``RV``, or ``mean`` is not a number.
    """
    try:
        bad_coefs = any(not isinstance(c, numbers.Real) for c in coefs) or isinstance(
            coefs, (str, bytes)
        )
    except TypeError:
        bad_coefs = True
    if bad_coefs:
        raise TypeError(
            "coefs must be a list of numbers -- one weight per past shock, "
            "for example coefs=[0.8, 0.5] for an MA(2). Use coefs=[] for no "
            "echo at all, which is just independent noise."
        )
    if not isinstance(noise_dist, (Distribution, RV)):
        raise TypeError(
            f"noise_dist must be a Symbulate Distribution (e.g. Normal(0, 1)) "
            f"or a random variable built from one, got "
            f"{type(noise_dist).__name__}. It is the distribution of one "
            f"shock."
        )
    if not isinstance(mean, numbers.Real):
        raise TypeError(
            f"mean must be a number, got {type(mean).__name__}. It is the "
            f"level the process varies around, for example mean=0."
        )


class MAResult(InfiniteVector):
    """One simulated sample path of a moving-average process.

    Each value is a fixed weighted sum of the most recent shocks, so unlike
    most processes here nothing has to be built up step by step: ``X[n]``
    reads straight off the shock sequence, which caches itself.

    Parameters
    ----------
    shocks : InfiniteVector
        The i.i.d. shocks. The first ``len(coefs)`` entries are the
        *pre-sample* shocks -- the ones that happened before time 0 -- so
        entry ``n + len(coefs)`` is the shock at time ``n``. See Notes.
    coefs : list of float
        The weight on each past shock, most recent first.
    mean : float, optional
        The level the process varies around. Default is 0.

    Attributes
    ----------
    shocks : InfiniteVector
        The shock sequence, including the pre-sample shocks at the front.
    coefs : list of float
        The weight on each past shock.
    mean : float
        The level the process varies around.

    Notes
    -----
    A moving-average process is defined for *all* times, with no beginning,
    which is what makes every one of its values have the same distribution.
    Simulating it has to start somewhere, though, and ``X[0]`` needs the
    ``q`` shocks from *before* time 0. Those are supplied at the front of
    ``shocks``, so the process is already running when the first value is
    read -- exactly as the definition says -- rather than starting from
    silence and taking ``q`` steps to reach full size.

    Examples
    --------
    >>> from symbulate import *
    >>> path = MA(coefs=[0.5], noise_dist=Bernoulli(1)).draw()
    >>> float(path[0])  # every shock is 1, so every value is 1 + 0.5
    1.5
    """

    def __init__(self, shocks, coefs, mean=0):
        """Create one simulated sample path of a moving-average process."""
        self.shocks = shocks
        self.coefs = list(coefs)
        self.mean = mean

        q = len(self.coefs)
        # The weight on the current shock is always 1; the supplied
        # coefficients weight the shocks before it.
        weights = [1.0] + self.coefs

        def _func(n):
            # shocks[k] holds the shock at time k - q, so the shock at time n
            # is shocks[n + q] and the one i steps earlier is shocks[n + q - i].
            # Every index is non-negative, so there is no special case for the
            # first few values.
            return self.mean + sum(
                weights[i] * self.shocks[n + q - i] for i in range(q + 1)
            )

        super().__init__(_func)

    def get_shocks(self):
        """Return the shock sequence driving this path.

        Returns
        -------
        InfiniteVector
            The shocks, with the pre-sample ones at the front: entry
            ``n + len(coefs)`` is the shock at time ``n``.
        """
        return self.shocks


class MAProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a moving-average process.

    Each draw from this space produces one simulated sample path.

    Parameters
    ----------
    coefs : list of float
        The weight on each past shock, most recent first. Its length is the
        order ``q``.
    noise_dist : Distribution or RV, optional
        The distribution of one shock. Default is ``Normal(0, 1)``.
    mean : float, optional
        The level the process varies around. Default is 0.

    Attributes
    ----------
    coefs : list of float
        The weight on each past shock.
    noise_dist : Distribution or RV
        The distribution of one shock.
    mean : float
        The level the process varies around.

    Raises
    ------
    TypeError
        If ``coefs`` is not a sequence of numbers, ``noise_dist`` is neither
        a ``Distribution`` nor an ``RV``, or ``mean`` is not a number.

    Examples
    --------
    >>> from symbulate import *
    >>> space = MAProbabilitySpace(coefs=[0.5], noise_dist=Bernoulli(1))
    >>> float(space.draw()[0])
    1.5
    """

    def __init__(self, coefs, noise_dist=None, mean=0):
        """Create a probability space for a moving-average process."""
        if noise_dist is None:
            noise_dist = Normal(0, 1)
        _validate_ma(coefs, noise_dist, mean)

        self.coefs = list(coefs)
        self.noise_dist = noise_dist
        self.mean = mean

        def draw():
            return MAResult(_iid_source(self.noise_dist), self.coefs, self.mean)

        super().__init__(draw)


class MA(RV):
    """A moving-average process: an echo of the most recent random shocks.

    At each time a fresh random *shock* arrives -- think of it as the news
    of the day. In a moving-average process the value now is that shock
    plus a fading echo of the shocks just before it::

        X[n] = mean + shock[n] + coefs[0] * shock[n-1] + coefs[1] * shock[n-2] + ...

    The echo lasts exactly ``q = len(coefs)`` steps and then stops dead. So
    two values more than ``q`` apart share no shocks at all and are
    completely uncorrelated -- a sharp cutoff, not a gradual fade. That
    cutoff is the whole signature of a moving-average process, and it is
    what makes it worth telling apart from an autoregressive one, whose
    memory instead decays forever without ever quite reaching zero.

    Despite the name, this is **not** the everyday "average of the last few
    values": it averages the recent *shocks*, not the recent values, and it
    never looks at its own past.

    Parameters
    ----------
    coefs : list of float
        The weight on each past shock, most recent first. The number of them
        is the order ``q``. ``coefs=[]`` gives independent noise with no echo.
    noise_dist : Distribution or RV, optional
        The distribution of one shock. Default is ``Normal(0, 1)``.
    mean : float, optional
        The level the process varies around. Default is 0. This really is
        the mean of every value *only* when ``noise_dist`` has mean 0 (the
        default, ``Normal(0, 1)``) -- the shocks then average out exactly.
        A ``noise_dist`` with nonzero mean does not average out, so it
        shifts the process's actual mean to
        ``mean + noise_dist.mean() * (1 + sum(coefs))`` instead of
        ``mean`` itself.

    Attributes
    ----------
    prob_space : MAProbabilitySpace
        The underlying probability space used to generate sample paths.
    coefs : list of float
        The weight on each past shock.
    noise_dist : Distribution or RV
        The distribution of one shock.
    mean : float
        The level the process varies around.

    Notes
    -----
    Time is indexed by the whole numbers ``0, 1, 2, ...``.

    Every value has the same distribution, including the very first one.
    Writing ``s`` for the standard deviation of one shock and ``t`` for
    ``coefs``, the mean is ``mean`` and the variance is
    ``s ** 2 * (1 + t[0] ** 2 + t[1] ** 2 + ...)`` at *every* time. This
    holds from ``X[0]`` because the shocks that happened before time 0 are
    simulated too -- see :class:`MAResult`.

    The correlation between ``X[n]`` and ``X[n + k]`` does not depend on
    ``n``, and is exactly 0 once ``k`` exceeds ``q``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = MA(coefs=[0.8, 0.5])  # an MA(2)
    >>> len(X.coefs)
    2
    >>> X[10].sim(1000).var()  # close to 1 + 0.8**2 + 0.5**2 = 1.89  # doctest: +SKIP
    1.87
    >>> Y = MA(coefs=[0.5], noise_dist=Bernoulli(1))  # every shock is 1
    >>> float(Y.draw()[0])
    1.5
    >>> Z = MA(coefs=[])  # no echo: just independent noise
    >>> len(Z.coefs)
    0

    See Also
    --------
    RandomWalk : A process that instead accumulates its steps forever.
    """

    def __init__(self, coefs, noise_dist=None, mean=0):
        """Create a moving-average process."""
        prob_space = MAProbabilitySpace(coefs=coefs, noise_dist=noise_dist, mean=mean)
        self.coefs = prob_space.coefs
        self.noise_dist = prob_space.noise_dist
        self.mean = prob_space.mean
        super().__init__(prob_space)


# Sentinel asking a process to start from its own long-run (stationary)
# distribution rather than a fixed value, matching gaussian_process.STATIONARY.
STATIONARY = "stationary"


def _is_stationary(ar_coefs):
    """Whether an autoregressive process settles down instead of exploding.

    True when every root of the characteristic polynomial lies outside the
    unit circle -- equivalently, when the companion matrix's eigenvalues all
    lie strictly inside it, which is what is checked here. For an AR(1) this
    is just ``abs(phi) < 1``.

    Parameters
    ----------
    ar_coefs : list of float
        The autoregressive coefficients.

    Returns
    -------
    bool
        ``True`` if the process is stationary.
    """
    p = len(ar_coefs)
    if p == 0:
        return True
    companion = np.zeros((p, p))
    companion[0, :] = ar_coefs
    if p > 1:
        companion[1:, :-1] = np.eye(p - 1)
    return bool(np.max(np.abs(np.linalg.eigvals(companion))) < 1)


def _stationary_covariance(ar_coefs, noise_var):
    """The covariance of ``p`` consecutive values of a stationary AR(p).

    Solves the Yule-Walker equations for the autocovariances
    ``gamma(0), ..., gamma(p)`` and returns the ``p x p`` matrix whose
    ``(a, b)`` entry is ``gamma(abs(a - b))`` -- the joint covariance of
    ``X[n-1], ..., X[n-p]``, which is what a stationary start has to be
    drawn from.

    The equations are ``gamma(k) = sum(phi_i * gamma(k - i))`` for
    ``k >= 1``, together with
    ``gamma(0) = sum(phi_i * gamma(i)) + noise_var``, using
    ``gamma(-k) = gamma(k)``. That is ``p + 1`` linear equations in
    ``p + 1`` unknowns.

    Parameters
    ----------
    ar_coefs : list of float
        The autoregressive coefficients. Must be stationary.
    noise_var : float
        The variance of one shock.

    Returns
    -------
    numpy.ndarray
        The ``p x p`` covariance matrix.
    """
    p = len(ar_coefs)
    a = np.zeros((p + 1, p + 1))
    b = np.zeros(p + 1)
    b[0] = noise_var
    a[0, 0] = 1.0
    for i, phi in enumerate(ar_coefs, start=1):
        a[0, i] -= phi
    for k in range(1, p + 1):
        a[k, k] += 1.0
        for i, phi in enumerate(ar_coefs, start=1):
            a[k, abs(k - i)] -= phi
    gamma = np.linalg.solve(a, b)
    return np.array([[gamma[abs(r - c)] for c in range(p)] for r in range(p)])


def _validate_arma(ar_coefs, ma_coefs, noise_dist, mean, initial):
    """Check the parameters of an autoregressive moving-average process.

    Raises
    ------
    TypeError
        If either coefficient list is not a sequence of numbers,
        ``noise_dist`` is neither a ``Distribution`` nor an ``RV``, ``mean``
        is not a number, or ``initial`` is not one of the accepted forms.
    ValueError
        If ``initial`` is a sequence of the wrong length, or is
        ``"stationary"`` where no stationary distribution exists or can be
        computed.
    """
    for name, coefs in (("ar_coefs", ar_coefs), ("ma_coefs", ma_coefs)):
        try:
            bad = isinstance(coefs, (str, bytes)) or any(
                not isinstance(c, numbers.Real) for c in coefs
            )
        except TypeError:
            bad = True
        if bad:
            raise TypeError(
                f"{name} must be a list of numbers, for example "
                f"{name}=[0.5, 0.2]. Use [] for none of that kind of term."
            )
    if not isinstance(noise_dist, (Distribution, RV)):
        raise TypeError(
            f"noise_dist must be a Symbulate Distribution (e.g. Normal(0, 1)) "
            f"or a random variable built from one, got "
            f"{type(noise_dist).__name__}. It is the distribution of one shock."
        )
    if not isinstance(mean, numbers.Real):
        raise TypeError(
            f"mean must be a number, got {type(mean).__name__}. It is the "
            f"level the process varies around, for example mean=0."
        )

    p = len(ar_coefs)
    if isinstance(initial, str) and initial == STATIONARY:
        if p == 0:
            return
        if len(ma_coefs) > 0:
            raise ValueError(
                'initial="stationary" is only available for a pure '
                "autoregressive process (no ma_coefs). With moving-average "
                "terms the long-run distribution has no simple closed form "
                "here, so give a number or a distribution instead."
            )
        if not isinstance(noise_dist, Normal):
            raise ValueError(
                'initial="stationary" needs normal shocks, because that is '
                "the only case whose long-run distribution has a closed "
                f"form; noise_dist is {type(noise_dist).__name__}. Give a "
                "number or a distribution to start from instead."
            )
        if not _is_stationary(ar_coefs):
            raise ValueError(
                f'initial="stationary" is impossible for ar_coefs='
                f"{list(ar_coefs)}: this process does not settle down, so it "
                "has no long-run distribution to start from. An explosive "
                "process is fine to simulate -- just give it a starting "
                "value, for example initial=0."
            )
        return

    if isinstance(initial, (Distribution, RV)):
        return
    if isinstance(initial, numbers.Real):
        return
    try:
        length = len(initial)
        numeric = all(isinstance(v, numbers.Real) for v in initial)
    except TypeError:
        raise TypeError(
            f"initial must be a number, a list of {p} numbers, a "
            f'distribution, or the word "stationary", got '
            f"{type(initial).__name__}."
        )
    if not numeric:
        raise TypeError("initial must contain only numbers.")
    if length != p:
        raise ValueError(
            f"initial has {length} values but this process needs {p} "
            f"(one per autoregressive coefficient). Give {p} starting "
            f"values, or a single number to use for all of them."
        )


def _draw_initial_values(initial, ar_coefs, noise_dist, mean):
    """Produce the ``p`` values the recursion starts from, for one path.

    Called once per sample path, so a random ``initial`` gives a fresh start
    each time.

    Parameters
    ----------
    initial : float, sequence, Distribution, RV, or str
        The starting condition, already validated.
    ar_coefs : list of float
        The autoregressive coefficients; their number is ``p``.
    noise_dist : Distribution or RV
        The distribution of one shock, used for the stationary calculation.
    mean : float
        The level the process varies around.

    Returns
    -------
    list of float
        ``p`` values, oldest first: ``X[-p], ..., X[-1]``.
    """
    p = len(ar_coefs)
    if p == 0:
        return []
    if isinstance(initial, str) and initial == STATIONARY:
        cov = _stationary_covariance(ar_coefs, float(noise_dist.var()))
        return list(MultivariateNormal([mean] * p, cov.tolist()).draw())
    if isinstance(initial, MultivariateDistribution):
        return list(initial.draw())
    if isinstance(initial, (Distribution, RV)):
        # One draw per slot. Exactly right for p = 1; for p > 1 these are
        # independent, which is not the stationary joint distribution --
        # see the note in ARMA's docstring.
        return [initial.draw() for _ in range(p)]
    if isinstance(initial, numbers.Real):
        return [initial] * p
    return list(initial)


class ARMAResult(InfiniteVector):
    """One simulated sample path of an autoregressive moving-average process.

    Values are generated on demand and cached, so reading ``path[500]`` and
    then ``path[10]`` describes one single path rather than two unrelated
    ones -- the same lazily-extending pattern ``MarkovChainResult`` uses.

    Parameters
    ----------
    shocks : InfiniteVector
        The i.i.d. shocks. The first ``len(ma_coefs)`` entries are the
        shocks from before time 0, so entry ``n + len(ma_coefs)`` is the
        shock at time ``n``, exactly as in :class:`MAResult`.
    presample : list of float
        The values before time 0, oldest first: ``X[-p], ..., X[-1]``.
    ar_coefs : list of float
        The weight on each previous *value*.
    ma_coefs : list of float
        The weight on each previous *shock*.
    mean : float, optional
        The level the process varies around. Default is 0.

    Attributes
    ----------
    shocks : InfiniteVector
        The shock sequence, pre-sample shocks at the front.
    presample : list of float
        The values before time 0.
    ar_coefs, ma_coefs : list of float
        The two sets of coefficients.
    mean : float
        The level the process varies around.
    generated : list
        The values generated so far.

    Examples
    --------
    >>> from symbulate import *
    >>> path = AR(coefs=[0.5], noise_dist=Bernoulli(1), mean=0, initial=0).draw()
    >>> float(path[0])  # 0.5 * 0 + 1
    1.0
    >>> float(path[1])  # 0.5 * 1 + 1
    1.5
    """

    def __init__(self, shocks, presample, ar_coefs, ma_coefs, mean=0):
        """Create one simulated sample path of an ARMA process."""
        self.shocks = shocks
        self.presample = list(presample)
        self.ar_coefs = list(ar_coefs)
        self.ma_coefs = list(ma_coefs)
        self.mean = mean
        # NOT `self.values`: InfiniteTuple already uses that name for its own
        # cache, and writing to it here would interleave two sets of appends.
        self.generated = []

        p = len(self.ar_coefs)
        q = len(self.ma_coefs)

        def value_at(index):
            # Times before 0 come from the pre-sample block, which holds
            # X[-p] ... X[-1] in that order.
            if index >= 0:
                return self.generated[index]
            return self.presample[p + index]

        def _func(n):
            m = len(self.generated)
            if n >= m:
                for k in range(m, n + 1):
                    # Deviations from the mean are what the recursion acts
                    # on, so `mean` really is the process's mean rather than
                    # an intercept that has to be converted.
                    total = self.mean
                    for i, phi in enumerate(self.ar_coefs, start=1):
                        total += phi * (value_at(k - i) - self.mean)
                    # shocks[j] holds the shock at time j - q, so the shock
                    # at time k is shocks[k + q]; every index is >= 0.
                    total += self.shocks[k + q]
                    for j, theta in enumerate(self.ma_coefs, start=1):
                        total += theta * self.shocks[k + q - j]
                    self.generated.append(total)
            return self.generated[n]

        super().__init__(_func)

    def get_shocks(self):
        """Return the shock sequence driving this path.

        Returns
        -------
        InfiniteVector
            The shocks, with the pre-sample ones at the front.
        """
        return self.shocks


class ARMAProbabilitySpace(ProbabilitySpace):
    """The probability space underlying an ARMA process.

    Each draw produces one simulated sample path. When ``initial`` is random
    -- a distribution, or ``"stationary"`` -- it is drawn afresh for every
    path.

    Parameters
    ----------
    ar_coefs : list of float
        The weight on each previous value. Its length is the order ``p``.
    ma_coefs : list of float, optional
        The weight on each previous shock. Its length is the order ``q``.
        Default is none.
    noise_dist : Distribution or RV, optional
        The distribution of one shock. Default is ``Normal(0, 1)``.
    mean : float, optional
        The level the process varies around. Default is 0.
    initial : float, sequence, Distribution, or str, optional
        Where the ``p`` values before time 0 come from. Default is 0.

    Attributes
    ----------
    ar_coefs, ma_coefs : list of float
        The two sets of coefficients.
    noise_dist : Distribution or RV
        The distribution of one shock.
    mean : float
        The level the process varies around.
    initial : float, sequence, Distribution, or str
        The starting condition.

    Examples
    --------
    >>> from symbulate import *
    >>> space = ARMAProbabilitySpace(ar_coefs=[0.5], noise_dist=Bernoulli(1))
    >>> float(space.draw()[0])
    1.0
    """

    def __init__(self, ar_coefs, ma_coefs=None, noise_dist=None, mean=0, initial=0):
        """Create a probability space for an ARMA process."""
        if ma_coefs is None:
            ma_coefs = []
        if noise_dist is None:
            noise_dist = Normal(0, 1)
        _validate_arma(ar_coefs, ma_coefs, noise_dist, mean, initial)

        self.ar_coefs = list(ar_coefs)
        self.ma_coefs = list(ma_coefs)
        self.noise_dist = noise_dist
        self.mean = mean
        self.initial = initial

        def draw():
            presample = _draw_initial_values(
                self.initial, self.ar_coefs, self.noise_dist, self.mean
            )
            return ARMAResult(
                _iid_source(self.noise_dist),
                presample,
                self.ar_coefs,
                self.ma_coefs,
                self.mean,
            )

        super().__init__(draw)


class ARMA(RV):
    """An ARMA process: today depends on recent values *and* recent shocks.

    Two different kinds of memory, combined::

        X[n] = mean + ar_coefs[0] * (X[n-1] - mean) + ...
                    + shock[n] + ma_coefs[0] * shock[n-1] + ...

    The **autoregressive** part feeds the process's own past back into it,
    so a shock echoes forever, fading geometrically -- that is the
    :class:`AR` half. The **moving-average** part adds a fixed window of
    recent shocks, whose influence stops dead after ``q`` steps -- that is
    the :class:`MA` half. Real series often want a little of both, which
    takes fewer parameters than forcing either one alone to fit.

    Parameters
    ----------
    ar_coefs : list of float
        The weight on each previous value, most recent first. Its length is
        the order ``p``. Use ``[]`` for a pure moving-average process.
    ma_coefs : list of float, optional
        The weight on each previous shock, most recent first. Its length is
        the order ``q``. Default is none, giving a pure autoregressive
        process.
    noise_dist : Distribution or RV, optional
        The distribution of one shock. Default is ``Normal(0, 1)``.
    mean : float, optional
        The level the process varies around. Default is 0. For a
        stationary process this really is the mean of every value *only*
        when ``noise_dist`` has mean 0 (the default, ``Normal(0, 1)``).
        A ``noise_dist`` with nonzero mean shifts the process's actual
        mean to
        ``mean + noise_dist.mean() * (1 + sum(ma_coefs)) / (1 - sum(ar_coefs))``
        instead of ``mean`` itself, since each shock's drift echoes
        forward through the autoregressive part exactly like a deviation
        from ``mean`` does.
    initial : float, sequence, Distribution, or str, optional
        Where the ``p`` values before time 0 come from. Default is 0. See
        Notes.

    Attributes
    ----------
    prob_space : ARMAProbabilitySpace
        The underlying probability space used to generate sample paths.
    ar_coefs, ma_coefs : list of float
        The two sets of coefficients.
    noise_dist : Distribution or RV
        The distribution of one shock.
    mean : float
        The level the process varies around.
    initial : float, sequence, Distribution, or str
        The starting condition.

    Notes
    -----
    Time is indexed by the whole numbers ``0, 1, 2, ...``.

    ``initial`` accepts any of:

    - **a number** -- all ``p`` starting values are that number (the
      default, 0);
    - **a list of p numbers** -- used as ``X[-p], ..., X[-1]``;
    - **a distribution** -- each starting value drawn from it. This is
      exactly right when ``p = 1``; for ``p > 1`` the draws are
      *independent*, which is **not** the stationary joint distribution,
      because consecutive values of an AR are correlated. Pass a
      multivariate distribution, or ``"stationary"``, to get that right;
    - **a multivariate distribution** of dimension ``p`` -- the starting
      values drawn jointly, correlations included;
    - **"stationary"** -- drawn from the process's own long-run
      distribution, so there is no transient at all. Available for a pure
      autoregressive process with normal shocks and stationary
      coefficients; anything else raises, since no closed form exists.

    Starting from a fixed number leaves a *transient*: the stretch of path
    where the process is still travelling from where it started toward its
    long-run behaviour. That is often the point -- it is how the pull of an
    autoregressive process is seen -- which is why it is the default, the
    same choice :class:`~symbulate.gaussian_process.OrnsteinUhlenbeck`
    makes.

    Unlike the values, the shocks before time 0 are always simulated, so the
    moving-average half never has a startup transient of its own.

    Explosive coefficients (an AR that does not settle down) are allowed and
    simulate fine; only ``initial="stationary"`` rejects them, since such a
    process has no long-run distribution to start from.

    Examples
    --------
    >>> from symbulate import *
    >>> X = ARMA(ar_coefs=[0.5], ma_coefs=[0.4])
    >>> len(X.ar_coefs), len(X.ma_coefs)
    (1, 1)
    >>> Y = ARMA(ar_coefs=[0.5], noise_dist=Bernoulli(1), initial=0)
    >>> path = Y.draw()
    >>> float(path[0]), float(path[1])
    (1.0, 1.5)

    See Also
    --------
    AR : The pure autoregressive case.
    MA : The pure moving-average case.
    """

    def __init__(self, ar_coefs, ma_coefs=None, noise_dist=None, mean=0, initial=0):
        """Create an ARMA process."""
        prob_space = ARMAProbabilitySpace(
            ar_coefs=ar_coefs,
            ma_coefs=ma_coefs,
            noise_dist=noise_dist,
            mean=mean,
            initial=initial,
        )
        self.ar_coefs = prob_space.ar_coefs
        self.ma_coefs = prob_space.ma_coefs
        self.noise_dist = prob_space.noise_dist
        self.mean = prob_space.mean
        self.initial = prob_space.initial
        super().__init__(prob_space)


class AR(ARMA):
    """An autoregressive process: today is a fraction of recent days, plus news.

    The pure autoregressive case of :class:`ARMA`::

        X[n] = mean + coefs[0] * (X[n-1] - mean) + ... + shock[n]

    Because the process feeds its own past back into itself, a single shock
    never fully disappears: it echoes forever, shrinking geometrically. That
    is the opposite of an :class:`MA`, whose memory stops dead after a fixed
    number of steps, and it is why the two are usually taught as a pair --
    they leave opposite fingerprints in the correlation between values a few
    steps apart.

    ``AR(coefs=[phi])`` is the discrete-time counterpart of
    :class:`~symbulate.gaussian_process.OrnsteinUhlenbeck`, and at
    ``phi = 1`` it becomes a :class:`~symbulate.random_walk.RandomWalk`.

    Parameters
    ----------
    coefs : list of float
        The weight on each previous value, most recent first. Its length is
        the order ``p``.
    noise_dist : Distribution or RV, optional
        The distribution of one shock. Default is ``Normal(0, 1)``.
    mean : float, optional
        The level the process varies around. Default is 0.
    initial : float, sequence, Distribution, or str, optional
        Where the ``p`` values before time 0 come from. Default is 0. See
        :class:`ARMA` for the accepted forms.

    Attributes
    ----------
    coefs : list of float
        The weight on each previous value (the same list as ``ar_coefs``).

    Notes
    -----
    For a stationary AR(1) with coefficient ``phi`` and shock standard
    deviation ``s``, every value has mean ``mean`` (when ``noise_dist``
    has mean 0 -- see :class:`ARMA`'s ``mean`` parameter for the
    correction when it does not) and variance ``s ** 2 / (1 - phi ** 2)``,
    and the correlation between values ``k`` steps apart is ``phi ** k``
    -- decaying, but never reaching zero.

    Examples
    --------
    >>> from symbulate import *
    >>> X = AR(coefs=[0.7])
    >>> len(X.coefs)
    1
    >>> Y = AR(coefs=[0.5], noise_dist=Bernoulli(1), initial=0)
    >>> float(Y.draw()[0])
    1.0

    See Also
    --------
    ARMA : The general case, which also takes moving-average terms.
    MA : The pure moving-average counterpart.
    OrnsteinUhlenbeck : The continuous-time version of an AR(1).
    """

    def __init__(self, coefs, noise_dist=None, mean=0, initial=0):
        """Create an autoregressive process."""
        super().__init__(
            ar_coefs=coefs,
            ma_coefs=None,
            noise_dist=noise_dist,
            mean=mean,
            initial=initial,
        )
        self.coefs = self.ar_coefs


class ARProbabilitySpace(ARMAProbabilitySpace):
    """The probability space underlying an autoregressive process.

    The pure-autoregressive specialization of
    :class:`ARMAProbabilitySpace`, in the same way :class:`AR` specializes
    :class:`ARMA`: it takes ``coefs`` rather than ``ar_coefs``, and no
    moving-average terms. Each draw produces one simulated sample path.

    Every process in this package has a probability space of its own, so
    that ``RV(ARProbabilitySpace(...))`` works the same way as it does for
    every other process rather than sending an ``AR`` user to the general
    ARMA space.

    Parameters
    ----------
    coefs : list of float
        The weight on each previous value, most recent first. Its length
        is the order ``p``.
    noise_dist : Distribution or RV, optional
        The distribution of one shock. Default is ``Normal(0, 1)``.
    mean : float, optional
        The level the process varies around. Default is 0.
    initial : float, sequence, Distribution, or str, optional
        Where the ``p`` values before time 0 come from. Default is 0. See
        :class:`ARMA` for the accepted forms.

    Attributes
    ----------
    coefs : list of float
        The weight on each previous value (the same list as ``ar_coefs``).

    Examples
    --------
    >>> from symbulate import *
    >>> space = ARProbabilitySpace(coefs=[0.5], noise_dist=Bernoulli(1))
    >>> float(space.draw()[0])
    1.0

    See Also
    --------
    AR : The process itself.
    ARMAProbabilitySpace : The general case, which also takes ``ma_coefs``.
    """

    def __init__(self, coefs, noise_dist=None, mean=0, initial=0):
        """Create the probability space for an autoregressive process."""
        super().__init__(
            ar_coefs=coefs,
            ma_coefs=None,
            noise_dist=noise_dist,
            mean=mean,
            initial=initial,
        )
        self.coefs = self.ar_coefs


def _garch_is_stationary(arch_coefs, garch_coefs, noise_var):
    """Whether a GARCH process has a finite long-run variance.

    True when the persistence -- the ARCH coefficients scaled by the
    variance of one standardized shock, plus the GARCH coefficients --
    is less than 1. At 1 or above the variance keeps growing instead of
    settling, so there is no unconditional variance to speak of.

    The scaling matters because the ARCH terms weight *squared* shocks:
    a shock with variance other than 1 changes how much persistence
    those coefficients actually carry, even though the coefficients
    themselves didn't change. See ``_unconditional_variance`` for the
    matching long-run-variance formula.

    Parameters
    ----------
    arch_coefs, garch_coefs : list of float
        The two sets of coefficients.
    noise_var : float
        The variance of one draw from ``noise_dist`` (the standardized
        shock). The textbook GARCH convention is a shock with variance
        1, but Symbulate's ``noise_dist`` can be any distribution, so
        this is not assumed here -- pass ``float(noise_dist.var())``.

    Returns
    -------
    bool
        ``True`` if the process settles down.
    """
    return sum(arch_coefs) * noise_var + sum(garch_coefs) < 1


def _unconditional_variance(omega, arch_coefs, garch_coefs, noise_var):
    """The long-run variance ``omega / (1 - persistence)``.

    Parameters
    ----------
    omega : float
        The constant term in the variance recursion.
    arch_coefs, garch_coefs : list of float
        The two sets of coefficients. Must give a persistence below 1
        (see ``_garch_is_stationary``).
    noise_var : float
        The variance of one draw from ``noise_dist``. See
        ``_garch_is_stationary`` for why this has to be passed in rather
        than assumed to be 1.

    Returns
    -------
    float
        The variance the process settles at.
    """
    return omega / (1 - sum(arch_coefs) * noise_var - sum(garch_coefs))


def _validate_garch(omega, arch_coefs, garch_coefs, noise_dist, initial):
    """Check the parameters of a GARCH process.

    Raises
    ------
    TypeError
        If a coefficient list is not a sequence of numbers, ``omega`` or
        ``initial`` is not a number, or ``noise_dist`` is neither a
        ``Distribution`` nor an ``RV``.
    ValueError
        If ``omega`` or ``initial`` is not positive, any coefficient is
        negative, or ``initial="stationary"`` is asked of a process that
        does not settle down.
    """
    for name, coefs in (("arch_coefs", arch_coefs), ("garch_coefs", garch_coefs)):
        try:
            bad = isinstance(coefs, (str, bytes)) or any(
                not isinstance(c, numbers.Real) for c in coefs
            )
        except TypeError:
            bad = True
        if bad:
            raise TypeError(
                f"{name} must be a list of numbers, for example {name}=[0.1]. "
                f"Use [] for none of that kind of term."
            )
        if any(c < 0 for c in coefs):
            raise ValueError(
                f"{name} cannot contain negative numbers: they weight "
                f"squared quantities in the variance recursion, so a "
                f"negative one could make the variance negative."
            )
    if not isinstance(omega, numbers.Real):
        raise TypeError(
            f"omega must be a number, got {type(omega).__name__}. It is the "
            f"constant term in the variance, for example omega=0.2."
        )
    if omega <= 0:
        raise ValueError(
            f"omega must be positive, got {omega}. With omega at 0 the "
            f"variance can collapse to 0 and the process dies out."
        )
    if not isinstance(noise_dist, (Distribution, RV)):
        raise TypeError(
            f"noise_dist must be a Symbulate Distribution (e.g. Normal(0, 1)) "
            f"or a random variable built from one, got "
            f"{type(noise_dist).__name__}. It is the standardized shock that "
            f"gets scaled by the current volatility."
        )
    if initial is None:
        return
    if isinstance(initial, str) and initial == STATIONARY:
        if not _garch_is_stationary(arch_coefs, garch_coefs, float(noise_dist.var())):
            raise ValueError(
                'initial="stationary" is impossible with these coefficients '
                "and this noise_dist: the persistence -- arch_coefs scaled "
                "by the shock's variance, plus garch_coefs -- is 1 or more, "
                "so the variance never settles and there is no long-run "
                "value to start from. Give a positive starting variance "
                "instead, for example initial=1."
            )
        return
    if not isinstance(initial, numbers.Real):
        raise TypeError(
            f"initial must be a positive number (the starting variance) or "
            f'the word "stationary", got {type(initial).__name__}.'
        )
    if initial <= 0:
        raise ValueError(
            f"initial must be positive, got {initial}. It is the variance the "
            f"process starts from, and a variance cannot be zero or negative."
        )


class GARCHResult(InfiniteVector):
    """One simulated sample path of a GARCH process.

    Two sequences are built side by side and cached: the variance at each
    time, and the value itself. Reading ``path[500]`` and then ``path[10]``
    describes one single path, the same lazily-extending pattern the other
    processes here use.

    Parameters
    ----------
    shocks : InfiniteVector
        The i.i.d. standardized shocks, one per time step.
    omega : float
        The constant term in the variance recursion.
    arch_coefs : list of float
        The weight on each previous *squared value*.
    garch_coefs : list of float
        The weight on each previous *variance*.
    initial : float
        The variance the recursion starts from, used for every time before
        0 for both the variance and the squared value.

    Attributes
    ----------
    shocks : InfiniteVector
        The standardized shocks.
    omega : float
        The constant term.
    arch_coefs, garch_coefs : list of float
        The two sets of coefficients.
    initial : float
        The starting variance.
    generated : list
        The values generated so far.
    variances : list
        The variance at each of those times.

    Examples
    --------
    >>> from symbulate import *
    >>> path = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85],
    ...              noise_dist=Bernoulli(1), initial=4).draw()
    >>> float(path[0])  # variance stays at 4, and the shock is always 1
    2.0
    """

    def __init__(self, shocks, omega, arch_coefs, garch_coefs, initial):
        """Create one simulated sample path of a GARCH process."""
        self.shocks = shocks
        self.omega = omega
        self.arch_coefs = list(arch_coefs)
        self.garch_coefs = list(garch_coefs)
        self.initial = initial
        # Not `self.values`: InfiniteTuple already uses that name for its own
        # cache, and writing to it here would interleave two sets of appends.
        self.generated = []
        self.variances = []

        def squared_at(index):
            # Before time 0 the process is treated as having been sitting at
            # the starting variance, so a squared value equals it too.
            if index >= 0:
                return self.generated[index] ** 2
            return self.initial

        def variance_at(index):
            if index >= 0:
                return self.variances[index]
            return self.initial

        def _func(n):
            m = len(self.generated)
            if n >= m:
                for k in range(m, n + 1):
                    var = self.omega
                    for i, a in enumerate(self.arch_coefs, start=1):
                        var += a * squared_at(k - i)
                    for j, b in enumerate(self.garch_coefs, start=1):
                        var += b * variance_at(k - j)
                    self.variances.append(var)
                    self.generated.append(math.sqrt(var) * self.shocks[k])
            return self.generated[n]

        super().__init__(_func)

    def get_variances(self):
        """Return the variance at each time step of this path.

        The variance is what a GARCH model is really about -- the value
        itself is just a standardized shock scaled by it -- so it is worth
        being able to look at directly.

        Returns
        -------
        list of float
            The variances at the times generated so far.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85],
        ...              noise_dist=Bernoulli(1), initial=4).draw()
        >>> path[3]  # generate a few values first
        2.0
        >>> [round(v, 4) for v in path.get_variances()]
        [4.0, 4.0, 4.0, 4.0]
        """
        return self.variances


class GARCHProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a GARCH process.

    Each draw produces one simulated sample path.

    Parameters
    ----------
    omega : float
        The constant term in the variance recursion. Must be positive.
    arch_coefs : list of float
        The weight on each previous squared value. Must be non-negative.
    garch_coefs : list of float, optional
        The weight on each previous variance. Must be non-negative. Default
        is none, which gives an ARCH process.
    noise_dist : Distribution or RV, optional
        The standardized shock. Default is ``Normal(0, 1)``.
    initial : float or str, optional
        The variance the process starts from. Default is the long-run
        variance where one exists, so the process is already settled.

    Attributes
    ----------
    omega : float
        The constant term.
    arch_coefs, garch_coefs : list of float
        The two sets of coefficients.
    noise_dist : Distribution or RV
        The standardized shock.
    initial : float
        The starting variance, resolved to a number.

    Examples
    --------
    >>> from symbulate import *
    >>> space = GARCHProbabilitySpace(omega=0.2, arch_coefs=[0.1],
    ...                               garch_coefs=[0.85])
    >>> round(space.initial, 6)  # the long-run variance, 0.2 / (1 - 0.95)
    4.0
    """

    def __init__(
        self, omega, arch_coefs, garch_coefs=None, noise_dist=None, initial=None
    ):
        """Create a probability space for a GARCH process."""
        if garch_coefs is None:
            garch_coefs = []
        if noise_dist is None:
            noise_dist = Normal(0, 1)
        _validate_garch(omega, arch_coefs, garch_coefs, noise_dist, initial)

        self.omega = omega
        self.arch_coefs = list(arch_coefs)
        self.garch_coefs = list(garch_coefs)
        self.noise_dist = noise_dist

        noise_var = float(self.noise_dist.var())
        settles = _garch_is_stationary(self.arch_coefs, self.garch_coefs, noise_var)
        if initial is None or (
            isinstance(initial, str) and initial == STATIONARY
        ):
            # Starting at the long-run variance means there is no warm-up:
            # the variance is already where it belongs at time 0. When the
            # process does not settle there is no such value, so fall back to
            # omega and let the warm-up show.
            self.initial = (
                _unconditional_variance(
                    omega, self.arch_coefs, self.garch_coefs, noise_var
                )
                if settles
                else omega
            )
        else:
            self.initial = initial

        def draw():
            return GARCHResult(
                _iid_source(self.noise_dist),
                self.omega,
                self.arch_coefs,
                self.garch_coefs,
                self.initial,
            )

        super().__init__(draw)


class GARCH(RV):
    """A GARCH process: the size of the swings is itself random and clusters.

    Most processes here have a fixed amount of randomness. A GARCH lets the
    *volatility* change over time, and makes it depend on what just
    happened::

        variance[n] = omega + arch_coefs[0] * X[n-1] ** 2 + ...
                            + garch_coefs[0] * variance[n-1] + ...
        X[n]        = sqrt(variance[n]) * shock[n]

    A big move makes the next variance bigger, which makes another big move
    more likely. That is **volatility clustering** -- calm stretches and
    turbulent stretches, rather than a steady hum -- and it is the single
    most recognisable feature of financial returns, which is what this model
    was invented for.

    The values themselves are uncorrelated, like plain noise; it is their
    *sizes* that are correlated. That is what makes a GARCH different from
    an :class:`AR`, where the values are correlated directly.

    Parameters
    ----------
    omega : float
        The constant term in the variance. Must be positive.
    arch_coefs : list of float
        The weight on each previous squared value -- how strongly a big move
        raises the next variance. Must be non-negative.
    garch_coefs : list of float, optional
        The weight on each previous variance -- how much the volatility
        level persists. Must be non-negative. Default is none, which gives
        an :class:`ARCH` process.
    noise_dist : Distribution or RV, optional
        The standardized shock that gets scaled by the volatility. Default
        is ``Normal(0, 1)``.
    initial : float or str, optional
        The variance the process starts from. By default this is the
        process's own long-run variance, so there is no warm-up. Give a
        number to start somewhere else and watch it settle.

    Attributes
    ----------
    prob_space : GARCHProbabilitySpace
        The underlying probability space used to generate sample paths.
    omega : float
        The constant term.
    arch_coefs, garch_coefs : list of float
        The two sets of coefficients.
    noise_dist : Distribution or RV
        The standardized shock.
    initial : float
        The starting variance, resolved to a number.

    Notes
    -----
    The classic model is GARCH(1,1) -- one coefficient of each kind:
    ``GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85])``. Longer lists
    give higher orders.

    The process settles down when its *persistence* -- ``arch_coefs``
    scaled by the variance of one shock from ``noise_dist``, plus
    ``garch_coefs`` -- sums to less than 1, and its long-run variance is
    then ``omega / (1 - persistence)``. With the default ``Normal(0, 1)``
    shock (variance 1) persistence is just the coefficients' own sum, which
    is the formula most textbooks show; a ``noise_dist`` with a different
    variance changes how much persistence the same coefficients carry, and
    both formulas above account for that. Since the long-run variance is
    where ``initial`` starts by default, every value has the same variance
    from time 0 onward. Coefficients (and shock variance) giving a
    persistence of 1 or more are allowed and simulate fine -- the variance
    simply grows instead of settling -- but ``initial="stationary"`` rejects
    them, since there is no long-run value to start from.

    Examples
    --------
    >>> from symbulate import *
    >>> X = GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85])
    >>> round(X.initial, 6)  # the long-run variance, 0.2 / (1 - 0.95)
    4.0
    >>> X[20].sim(1000).var()  # close to 4  # doctest: +SKIP
    3.96
    >>> X[20].sim(1000).mean()  # values are centered at 0  # doctest: +SKIP
    -0.02

    See Also
    --------
    ARCH : The special case with no variance-persistence terms.
    AR : A process whose *values*, rather than their sizes, are correlated.
    """

    def __init__(
        self, omega, arch_coefs, garch_coefs=None, noise_dist=None, initial=None
    ):
        """Create a GARCH process."""
        prob_space = GARCHProbabilitySpace(
            omega=omega,
            arch_coefs=arch_coefs,
            garch_coefs=garch_coefs,
            noise_dist=noise_dist,
            initial=initial,
        )
        self.omega = prob_space.omega
        self.arch_coefs = prob_space.arch_coefs
        self.garch_coefs = prob_space.garch_coefs
        self.noise_dist = prob_space.noise_dist
        self.initial = prob_space.initial
        super().__init__(prob_space)


class ARCH(GARCH):
    """An ARCH process: today's variance depends on recent squared values.

    The original volatility-clustering model, and the special case of
    :class:`GARCH` with no variance-persistence terms::

        variance[n] = omega + coefs[0] * X[n-1] ** 2 + ...
        X[n]        = sqrt(variance[n]) * shock[n]

    A big move raises the variance for the next few steps, and then its
    influence stops -- the memory reaches back exactly ``len(coefs)`` steps.
    Adding ``garch_coefs`` to carry the variance level forward as well is
    what turns it into a GARCH, and is why a GARCH(1,1) usually fits real
    data with far fewer parameters than a long ARCH.

    Parameters
    ----------
    omega : float
        The constant term in the variance. Must be positive.
    coefs : list of float
        The weight on each previous squared value. Must be non-negative.
    noise_dist : Distribution or RV, optional
        The standardized shock. Default is ``Normal(0, 1)``.
    initial : float or str, optional
        The variance the process starts from. Defaults to the long-run
        variance, so there is no warm-up.

    Attributes
    ----------
    coefs : list of float
        The weight on each previous squared value (the same list as
        ``arch_coefs``).

    Examples
    --------
    >>> from symbulate import *
    >>> X = ARCH(omega=0.5, coefs=[0.5])
    >>> X.initial  # long-run variance 0.5 / (1 - 0.5)
    1.0
    >>> X.garch_coefs
    []

    See Also
    --------
    GARCH : The general case, which also carries the variance level forward.
    """

    def __init__(self, omega, coefs, noise_dist=None, initial=None):
        """Create an ARCH process."""
        super().__init__(
            omega=omega,
            arch_coefs=coefs,
            garch_coefs=None,
            noise_dist=noise_dist,
            initial=initial,
        )
        self.coefs = self.arch_coefs


class ARCHProbabilitySpace(GARCHProbabilitySpace):
    """The probability space underlying an ARCH process.

    The no-persistence-terms specialization of
    :class:`GARCHProbabilitySpace`, in the same way :class:`ARCH`
    specializes :class:`GARCH`: it takes ``coefs`` rather than
    ``arch_coefs``, and no ``garch_coefs``. Each draw produces one
    simulated sample path.

    Every process in this package has a probability space of its own, so
    that ``RV(ARCHProbabilitySpace(...))`` works the same way as it does
    for every other process rather than sending an ``ARCH`` user to the
    general GARCH space.

    Parameters
    ----------
    omega : float
        The constant floor under the variance. Must be positive.
    coefs : list of float
        The weight on each recent squared value, most recent first. Its
        length is the order ``q``.
    noise_dist : Distribution or RV, optional
        The distribution of one standardized shock. Default is
        ``Normal(0, 1)``.
    initial : float, sequence, Distribution, or str, optional
        Where the values before time 0 come from. See :class:`GARCH` for
        the accepted forms.

    Attributes
    ----------
    coefs : list of float
        The weight on each recent squared value (the same list as
        ``arch_coefs``).

    Examples
    --------
    >>> from symbulate import *
    >>> space = ARCHProbabilitySpace(omega=0.2, coefs=[0.1])
    >>> len(space.coefs)
    1

    See Also
    --------
    ARCH : The process itself.
    GARCHProbabilitySpace : The general case, which also takes
        ``garch_coefs``.
    """

    def __init__(self, omega, coefs, noise_dist=None, initial=None):
        """Create the probability space for an ARCH process."""
        super().__init__(
            omega=omega,
            arch_coefs=coefs,
            garch_coefs=None,
            noise_dist=noise_dist,
            initial=initial,
        )
        self.coefs = self.arch_coefs
