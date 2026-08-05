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
        The level the process varies around. Default is 0. Because the
        shocks average out, this really is the mean of every value.

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
    if initial == STATIONARY:
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
    if initial == STATIONARY:
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
        The level the process varies around. Default is 0. For a stationary
        process this really is the mean of every value.
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
    deviation ``s``, every value has mean ``mean`` and variance
    ``s ** 2 / (1 - phi ** 2)``, and the correlation between values ``k``
    steps apart is ``phi ** k`` -- decaying, but never reaching zero.

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
