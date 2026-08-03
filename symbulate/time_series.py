import numbers

from .distributions import Distribution, Normal
from .probability_space import ProbabilitySpace
from .random_variables import RV

# The same lazily-cached i.i.d. sequence RandomWalk uses for its steps. A
# time-series process needs exactly the same thing for its shocks, so the
# helper is shared rather than duplicated; it is imported under a name that
# reads correctly here. (Worth moving somewhere neutral once the rest of the
# time-series family lands and a shared home is obvious.)
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
