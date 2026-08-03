import numbers

from .distributions import Bernoulli, Distribution
from .math import inf
from .probability_space import ProbabilitySpace
from .random_variables import RV
from .result import InfiniteVector


def _step_source(step_dist):
    """Return a lazily-generated, cached sequence of i.i.d. steps.

    Two kinds of object can supply the steps, and they are replicated in
    different ways:

    - a :class:`~symbulate.distributions.Distribution` uses the package's
      usual ``dist ** inf`` idiom (the same one ``PoissonProcess`` and
      ``RenewalProcess`` use to build their interarrival times);
    - an :class:`~symbulate.random_variables.RV` cannot use ``** inf``,
      because for a random variable ``**`` already means raising the value
      to a power. Calling ``draw()`` once per position gives the same
      i.i.d. sequence, and wrapping it in an ``InfiniteVector`` caches each
      value so the path never changes once read.

    Accepting both matters for the simplest walk of all: a step of ``+1``
    or ``-1`` has to be built by arithmetic (``RV(Bernoulli(p)) * 2 - 1``),
    and arithmetic on a distribution produces an ``RV``, not a
    ``Distribution``.

    Parameters
    ----------
    step_dist : Distribution or RV
        The distribution of a single step.

    Returns
    -------
    InfiniteVector
        The steps ``step[0], step[1], step[2], ...``, generated on demand.
    """
    if isinstance(step_dist, Distribution):
        return (step_dist**inf).draw()
    return InfiniteVector(lambda n: step_dist.draw())


def _validate_random_walk(step_dist, p, initial_value):
    """Check the parameters of a random walk.

    Raises
    ------
    TypeError
        If ``step_dist`` is neither a ``Distribution`` nor an ``RV``, or if
        ``p`` or ``initial_value`` is not a number.
    ValueError
        If neither ``step_dist`` nor ``p`` was given, if both were, or if
        ``p`` is not between 0 and 1.
    """
    if step_dist is None and p is None:
        raise ValueError(
            "A random walk needs to know what one step looks like. Give "
            "either p (for the simple walk that steps +1 with probability p "
            "and -1 otherwise), for example RandomWalk(p=0.5), or step_dist "
            "(for any other step), for example "
            "RandomWalk(step_dist=Normal(0, 1))."
        )
    if step_dist is not None and p is not None:
        raise ValueError(
            "Specify either p or step_dist, not both. p=... is a shorthand "
            "for the simple +1/-1 walk; step_dist=... covers every other "
            "kind of step, so only one of them describes the walk you want."
        )
    if p is not None:
        if not isinstance(p, numbers.Real):
            raise TypeError(
                f"p must be a number, got {type(p).__name__}. It is the "
                f"probability that a step is +1 rather than -1, for example "
                f"p=0.5 for the fair walk."
            )
        if not 0 <= p <= 1:
            raise ValueError(
                f"p must be between 0 and 1, got {p}. It is the probability "
                f"that a step is +1 rather than -1."
            )
    if step_dist is not None and not isinstance(step_dist, (Distribution, RV)):
        raise TypeError(
            f"step_dist must be a Symbulate Distribution (e.g. Normal(0, 1), "
            f"Poisson(2)) or a random variable built from one (e.g. "
            f"RV(Bernoulli(0.5)) * 2 - 1), got {type(step_dist).__name__}."
        )
    if not isinstance(initial_value, numbers.Real):
        raise TypeError(
            f"initial_value must be a number, got "
            f"{type(initial_value).__name__}. It is where every path starts, "
            f"for example initial_value=0."
        )


class RandomWalkResult(InfiniteVector):
    """One simulated sample path of a random walk.

    Stores the position at each time step. Positions are generated on
    demand as you index further into the path, and cached once generated,
    so reading ``path[100]`` and then ``path[10]`` describes one single
    walk rather than two unrelated ones.

    Parameters
    ----------
    steps : InfiniteVector
        The i.i.d. steps to accumulate. ``steps[n]`` is the step taken
        between time ``n`` and time ``n + 1``.
    initial_value : float, optional
        The position at time 0. Default is 0.

    Attributes
    ----------
    steps : InfiniteVector
        The steps being accumulated.
    initial_value : float
        The position at time 0.
    positions : list
        The positions generated so far, starting with ``initial_value``.

    Examples
    --------
    >>> from symbulate import *
    >>> path = RandomWalk(p=0.5).draw()
    >>> float(path[0])  # every path starts at initial_value
    0.0
    >>> abs(float(path[1]) - float(path[0]))  # each step moves by exactly 1
    1.0
    """

    def __init__(self, steps, initial_value=0):
        """Create one simulated sample path of a random walk."""
        self.steps = steps
        self.initial_value = initial_value
        # positions[n] is the position at time n. Time 0 is known with no
        # randomness, so the path always starts out one entry long.
        self.positions = [initial_value]

        def _func(n):
            # Extend the walk only as far as has actually been asked for,
            # picking up from the last position already generated -- the same
            # lazily-extending, self-caching pattern MarkovChainResult uses
            # for its states.
            m = len(self.positions)
            if n >= m:
                position = self.positions[m - 1]
                for i in range(m, n + 1):
                    # The step between time i-1 and time i.
                    position = position + self.steps[i - 1]
                    self.positions.append(position)
            return self.positions[n]

        super().__init__(_func)

    def get_steps(self):
        """Return the individual steps this path took.

        Returns
        -------
        InfiniteVector
            The steps, where entry ``n`` is the step taken between time
            ``n`` and time ``n + 1``.

        Examples
        --------
        >>> from symbulate import *
        >>> path = RandomWalk(p=0.5).draw()
        >>> abs(float(path.get_steps()[0]))
        1.0
        """
        return self.steps


class RandomWalkProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a random walk.

    Each draw from this space produces one simulated sample path of the
    walk.

    Parameters
    ----------
    step_dist : Distribution or RV, optional
        The distribution of a single step. Mutually exclusive with ``p``.
    p : float, optional
        Probability that a step is ``+1`` rather than ``-1``, for the
        simple walk. Must be between 0 and 1. Mutually exclusive with
        ``step_dist``.
    initial_value : float, optional
        The position at time 0. Default is 0.

    Attributes
    ----------
    step_dist : Distribution or RV
        The distribution of a single step, including the ``+1``/``-1``
        random variable built for you when ``p`` was given.
    p : float or None
        The value of ``p``, or ``None`` if ``step_dist`` was given instead.
    initial_value : float
        The position at time 0.

    Raises
    ------
    TypeError
        If ``step_dist`` is neither a ``Distribution`` nor an ``RV``, or if
        ``p`` or ``initial_value`` is not a number.
    ValueError
        If neither ``step_dist`` nor ``p`` was given, if both were, or if
        ``p`` is not between 0 and 1.

    Examples
    --------
    >>> from symbulate import *
    >>> space = RandomWalkProbabilitySpace(p=0.5)
    >>> float(space.draw()[0])
    0.0
    """

    def __init__(self, step_dist=None, p=None, initial_value=0):
        """Create a probability space for a random walk."""
        _validate_random_walk(step_dist, p, initial_value)

        self.p = p
        self.initial_value = initial_value
        if p is not None:
            # The simple walk steps +1 with probability p and -1 otherwise.
            # Bernoulli gives 0/1, so 2 * B - 1 rescales it to -1/+1. That
            # arithmetic produces an RV, which is exactly why _step_source
            # accepts one.
            self.step_dist = RV(Bernoulli(p)) * 2 - 1
        else:
            self.step_dist = step_dist

        def draw():
            return RandomWalkResult(_step_source(self.step_dist), self.initial_value)

        super().__init__(draw)


class RandomWalk(RV):
    """A random walk: the running total of independent random steps.

    The walk starts at ``initial_value`` and adds one independent step at
    each time, so ``X[n]`` is where it stands after ``n`` steps. It is the
    first process in most introductory courses: the simple ``+1``/``-1``
    walk is the gambler's-ruin model, and because ``X[n]`` is a sum of
    independent pieces it is also the most direct picture of the central
    limit theorem -- the spread grows like the square root of ``n``, and
    the distribution of ``X[n]`` approaches a bell curve.

    Describe one step in either of two ways. Give ``p`` for the simple
    walk, which steps ``+1`` with probability ``p`` and ``-1`` otherwise;
    or give ``step_dist`` for any other kind of step.

    Parameters
    ----------
    step_dist : Distribution or RV, optional
        The distribution of a single step -- any Symbulate distribution
        (e.g. ``Normal(0, 1)``, ``Poisson(2)``), or a random variable built
        from one (e.g. ``RV(Bernoulli(0.5)) * 2 - 1``). Mutually exclusive
        with ``p``.
    p : float, optional
        Probability that a step is ``+1`` rather than ``-1``, for the
        simple walk. Must be between 0 and 1. Mutually exclusive with
        ``step_dist``.
    initial_value : float, optional
        Where every path starts. Default is 0.

    Attributes
    ----------
    prob_space : RandomWalkProbabilitySpace
        The underlying probability space used to generate sample paths.
    step_dist : Distribution or RV
        The distribution of a single step.
    p : float or None
        The value of ``p``, or ``None`` if ``step_dist`` was given instead.
    initial_value : float
        Where every path starts.

    Notes
    -----
    Time is indexed by the whole numbers ``0, 1, 2, ...``, so ``X[0]`` is
    the starting point and the first step lands at ``X[1]``.

    For the simple walk, ``X[n]`` has mean ``initial_value + n * (2p - 1)``
    and, when ``p = 0.5``, variance ``n``. That growing variance -- spread
    proportional to the square root of ``n`` -- is the fact the walk is
    usually introduced to demonstrate.

    Examples
    --------
    >>> from symbulate import *
    >>> X = RandomWalk(p=0.5)
    >>> float(X.draw()[0])  # every path starts at 0
    0.0
    >>> X[10].sim(1000).mean()  # doctest: +SKIP
    0.04
    >>> X[10].sim(1000).var()  # close to 10, the variance of a fair walk  # doctest: +SKIP
    9.87
    >>> Y = RandomWalk(step_dist=Normal(0, 1))  # steps need not be +/-1
    >>> Y.draw()[3]  # doctest: +SKIP
    -0.42

    See Also
    --------
    BrownianMotion : The continuous-time limit of a random walk.
    MarkovChain : Another discrete-time process built the same way.
    """

    def __init__(self, step_dist=None, p=None, initial_value=0):
        """Create a random walk."""
        prob_space = RandomWalkProbabilitySpace(
            step_dist=step_dist, p=p, initial_value=initial_value
        )
        self.step_dist = prob_space.step_dist
        self.p = prob_space.p
        self.initial_value = prob_space.initial_value
        super().__init__(prob_space)
