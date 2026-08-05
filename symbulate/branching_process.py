import numbers

import numpy as np

from .distributions import Distribution, MultivariateDistribution
from .probability_space import ProbabilitySpace
from .random_processes import _resolve_initial
from .random_variables import RV
from .result import DiscreteValued, InfiniteVector


def _smallest_possible(offspring_dist):
    """The smallest value the offspring distribution can produce.

    Read from scipy's own ``support()`` through ``Distribution._support``,
    the same way ``RenewalProcess`` checks its interarrival distribution, so
    a newly added distribution is validated with no per-distribution code.

    Parameters
    ----------
    offspring_dist : Distribution
        The distribution to inspect.

    Returns
    -------
    float or None
        The smallest possible value, or ``None`` when it cannot be
        determined.
    """
    try:
        low, _ = offspring_dist._support()
    except AttributeError:
        try:
            low = float(offspring_dist.quantile(0))
        except (AttributeError, TypeError, ValueError):
            return None
    return None if np.isnan(low) else low


def _validate_galton_watson(offspring_dist, initial):
    """Check the parameters of a Galton-Watson process.

    Raises
    ------
    TypeError
        If ``offspring_dist`` is not a Symbulate ``Distribution``, or
        ``initial`` is not a number.
    ValueError
        If ``offspring_dist`` can produce a negative or non-whole number of
        children, or ``initial`` is not a non-negative whole number.
    """
    if not isinstance(offspring_dist, Distribution) or isinstance(
        offspring_dist, MultivariateDistribution
    ):
        raise TypeError(
            f"offspring_dist must be a Symbulate Distribution giving the "
            f"number of children one individual has -- for example "
            f"Poisson(1.5) or Binomial(2, 0.6) -- got "
            f"{type(offspring_dist).__name__}."
        )
    if not offspring_dist.discrete:
        raise ValueError(
            "offspring_dist must be a discrete distribution: it counts "
            "children, so it has to produce whole numbers. Poisson, "
            "Binomial, and Geometric all work; Normal and Exponential do "
            "not."
        )
    low = _smallest_possible(offspring_dist)
    if low is not None and low < 0:
        raise ValueError(
            f"offspring_dist can produce {low}, but a number of children "
            f"cannot be negative. Use a distribution supported on 0, 1, 2, "
            f"... such as Poisson(1.5) or Binomial(2, 0.6)."
        )
    if not isinstance(initial, numbers.Real):
        raise TypeError(
            f"initial must be a number, got {type(initial).__name__}. It is "
            f"how many individuals the process starts with, for example "
            f"initial=1."
        )
    if initial < 0 or initial != int(initial):
        raise ValueError(
            f"initial must be a non-negative whole number, got {initial}. It "
            f"is a count of individuals."
        )


class GaltonWatsonResult(InfiniteVector, DiscreteValued):
    """One simulated family tree of a Galton-Watson process.

    Entry ``n`` is the size of generation ``n``. Generations are produced on
    demand and cached, the same lazily-extending pattern the other processes
    here use.

    Parameters
    ----------
    offspring_dist : Distribution
        How many children one individual has.
    initial : int
        The size of generation 0.

    Attributes
    ----------
    offspring_dist : Distribution
        The offspring distribution.
    initial : int
        The size of generation 0.
    sizes : list of int
        The generation sizes produced so far.

    Examples
    --------
    >>> from symbulate import *
    >>> path = GaltonWatson(offspring_dist=Binomial(2, 1), initial=1).draw()
    >>> [int(path[n]) for n in range(4)]  # everyone has exactly 2 children
    [1, 2, 4, 8]
    """

    def __init__(self, offspring_dist, initial):
        """Create one simulated family tree of a Galton-Watson process."""
        self.offspring_dist = offspring_dist
        self.initial = int(initial)
        # Not `self.values`: InfiniteTuple already uses that name for its own
        # cache, and writing to it here would interleave two sets of appends.
        self.sizes = [self.initial]

        def _func(n):
            m = len(self.sizes)
            if n >= m:
                for _ in range(m, n + 1):
                    parents = self.sizes[-1]
                    if parents == 0:
                        # Extinction is absorbing: once nobody is left there
                        # is nobody to have children, ever. Short-circuiting
                        # keeps reading far into an extinct path cheap.
                        self.sizes.append(0)
                        continue
                    children = (self.offspring_dist**parents).draw()
                    self.sizes.append(int(sum(children)))
            return self.sizes[n]

        super().__init__(_func)

    def get_states(self):
        """Return the sequence of generation sizes.

        Returns
        -------
        GaltonWatsonResult
            This object itself, which can be indexed to get the size of any
            generation.
        """
        return self

    def is_extinct(self, by_generation):
        """Whether the family has died out by a given generation.

        Parameters
        ----------
        by_generation : int
            The generation to check.

        Returns
        -------
        bool
            ``True`` if generation ``by_generation`` is empty.

        Examples
        --------
        >>> from symbulate import *
        >>> path = GaltonWatson(offspring_dist=Binomial(1, 0), initial=1).draw()
        >>> path.is_extinct(1)  # nobody has any children
        True
        """
        return int(self[by_generation]) == 0


class GaltonWatsonProbabilitySpace(ProbabilitySpace):
    """The probability space underlying a Galton-Watson process.

    Each draw produces one simulated family tree.

    Parameters
    ----------
    offspring_dist : Distribution
        How many children one individual has. Must be a discrete
        distribution on the non-negative whole numbers.
    initial : int, optional
        The size of generation 0. Default is 1.

    Attributes
    ----------
    offspring_dist : Distribution
        The offspring distribution.
    initial : int
        The size of generation 0.

    Examples
    --------
    >>> from symbulate import *
    >>> space = GaltonWatsonProbabilitySpace(offspring_dist=Poisson(1.5))
    >>> int(space.draw()[0])
    1
    """

    def __init__(self, offspring_dist, initial=1):
        """Create a probability space for a Galton-Watson process."""
        _validate_galton_watson(offspring_dist, initial)
        self.offspring_dist = offspring_dist
        self.initial = int(initial)

        def draw():
            return GaltonWatsonResult(self.offspring_dist, self.initial)

        super().__init__(draw)


class GaltonWatson(RV):
    """A Galton-Watson process: a family tree that may die out or explode.

    Start with ``initial`` individuals. Each one independently has a random
    number of children, drawn from ``offspring_dist``; those children are
    generation 1. Repeat. ``X[n]`` is the size of generation ``n``::

        X[n + 1] = sum of X[n] independent draws from offspring_dist

    The question the process was invented for -- by Galton and Watson, who
    were asking why aristocratic surnames disappear -- is whether the line
    survives. The answer turns on the **mean number of children** ``m``:

    - ``m < 1`` (*subcritical*): extinction is certain.
    - ``m = 1`` (*critical*): extinction is still certain, but it can take a
      very long time.
    - ``m > 1`` (*supercritical*): the family may die out, but with positive
      probability it grows without bound.

    That sharp threshold at ``m = 1`` is the point of the model, and it also
    shows up in epidemics, where ``m`` is the reproduction number and the
    same cutoff separates an outbreak that fizzles from one that spreads.

    Parameters
    ----------
    offspring_dist : Distribution
        How many children one individual has. Must be a discrete
        distribution on the non-negative whole numbers, such as
        ``Poisson(1.5)`` or ``Binomial(2, 0.6)``.
    initial : int, optional
        The size of generation 0. Default is 1.

    Attributes
    ----------
    prob_space : GaltonWatsonProbabilitySpace
        The underlying probability space used to generate family trees.
    offspring_dist : Distribution
        The offspring distribution.
    initial : int
        The size of generation 0.

    Notes
    -----
    Generation 0 is ``initial``, so the first random step lands at ``X[1]``.

    The expected size of generation ``n`` is ``initial * m ** n``, where
    ``m`` is the mean of ``offspring_dist``. That is the cleanest way to see
    the threshold: below 1 the expectation shrinks geometrically, above 1 it
    grows.

    **Extinction is absorbing.** Once a generation is empty every later one
    is too, and the simulation short-circuits rather than drawing from an
    empty population.

    **A supercritical process really does explode.** Generation ``n`` takes
    about ``m ** n`` draws to produce, so reading far into a fast-growing
    tree is genuinely expensive. That is the process behaving correctly, not
    a performance bug, but it is worth knowing before asking for ``X[100]``
    of a process with ``m = 2``.

    Examples
    --------
    >>> from symbulate import *
    >>> X = GaltonWatson(offspring_dist=Poisson(1.5))
    >>> int(X.draw()[0])  # generation 0 is the starting population
    1
    >>> X[3].sim(1000).mean()  # close to 1.5 ** 3 = 3.375  # doctest: +SKIP
    3.41
    >>> Y = GaltonWatson(offspring_dist=Binomial(2, 1))  # everyone has 2 children
    >>> [int(Y.draw()[n]) for n in range(4)]
    [1, 2, 4, 8]

    See Also
    --------
    RandomWalk : Another process built by accumulating random steps.
    """

    def __init__(self, offspring_dist, initial=1):
        """Create a Galton-Watson branching process."""
        prob_space = GaltonWatsonProbabilitySpace(
            offspring_dist=offspring_dist, initial=initial
        )
        self.offspring_dist = prob_space.offspring_dist
        self.initial = prob_space.initial
        super().__init__(prob_space)
