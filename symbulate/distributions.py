import numbers
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from .probability_space import ProbabilitySpace
from .plot import get_next_color
from .result import Scalar, Vector, InfiniteVector

rng = np.random.default_rng()


class Distribution(ProbabilitySpace):
    """Base class for all probability distributions in Symbulate.

    Provides common methods shared by all distributions, including
    drawing samples, computing summary statistics, and plotting.
    You typically won't use this class directly — use one of the
    specific distribution classes instead (e.g., ``Normal``, ``Binomial``).

    Parameters
    ----------
    params : dict
        Named parameter values for the underlying scipy distribution.
    scipy : scipy.stats distribution
        The scipy distribution object used for computation.
    discrete : bool, optional
        ``True`` for discrete distributions, ``False`` for continuous.
        Default is ``True``.

    Attributes
    ----------
    params : dict
        Named parameter values passed to the underlying scipy distribution.
    discrete : bool
        ``True`` for discrete distributions, ``False`` for continuous.
    pdf : callable
        Probability density (or mass) function.
    cdf : callable
        Cumulative distribution function.
    quantile : callable
        Inverse CDF (percent-point function).
    mean : callable
        Returns the expected value of the distribution.
    var : callable
        Returns the variance of the distribution.
    sd : callable
        Returns the standard deviation of the distribution.
    median : callable
        Returns the median of the distribution.
    xlim : tuple of float
        Default x-axis range used when plotting.
    """

    def __init__(self, params, scipy, discrete=True):
        """Initialize the base Distribution."""
        self.params = params

        self.discrete = discrete

        if discrete:
            self.pmf = lambda x: scipy.pmf(x, **self.params)
            self.pdf = self.pmf  # add pdf as an alias for pmf
        else:
            self.pdf = lambda x: scipy.pdf(x, **self.params)

        self.cdf = lambda x: scipy.cdf(x, **self.params)
        self.quantile = lambda x: scipy.ppf(x, **self.params)

        self.median = lambda: scipy.median(**self.params)
        self.mean = lambda: scipy.mean(**self.params)
        self.var = lambda: scipy.var(**self.params)
        self.sd = lambda: scipy.std(**self.params)
        self.sim_func = scipy.rvs

        self.xlim = (scipy.ppf(0.001, **self.params), scipy.ppf(0.999, **self.params))

    def draw(self):
        """Draw a single random sample from the distribution.

        Returns
        -------
        Scalar
            One random value drawn from the distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> Normal(0, 1).draw()  # doctest: +SKIP
        0.4967141530112327
        """
        return Scalar(self.sim_func(**self.params, random_state=rng))

    # Override the inherited __pow__ function to take advantage
    # of vectorized simulations.
    def __pow__(self, exponent):
        """Draw multiple independent samples from the distribution.

        Parameters
        ----------
        exponent : int or float
            Number of samples to draw. Pass ``float('inf')`` to create
            an infinite sequence of draws generated lazily on demand.

        Returns
        -------
        ProbabilitySpace
            A probability space whose draws produce ``exponent`` samples
            at a time.

        Examples
        --------
        >>> from symbulate import *
        >>> (Normal(0, 1) ** 3).draw()  # doctest: +SKIP
        [-0.234, 1.724, 0.313]
        """
        if exponent == float("inf"):

            def draw():
                def _func(_):
                    return self.sim_func(**self.params, random_state=rng)

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(
                    self.sim_func(**self.params, size=exponent, random_state=rng)
                )

        return ProbabilitySpace(draw)

    def plot(self, xlim=None, alpha=None, ax=None, **kwargs):
        """Plot the probability density or mass function.

        For discrete distributions, dots are drawn at each integer value.
        For continuous distributions, a smooth curve is drawn.

        Parameters
        ----------
        xlim : tuple of float, optional
            x-axis range as ``(min, max)``. Uses distribution defaults
            if not provided.
        alpha : float, optional
            Transparency of the plot, from 0 (invisible) to 1 (opaque).
        ax : matplotlib.axes.Axes, optional
            The axes to draw on. Creates or uses the current axes if
            not provided.
        **kwargs
            Additional keyword arguments forwarded to matplotlib.

        Examples
        --------
        >>> from symbulate import *
        >>> Normal(0, 1).plot()
        """
        # use distribution defaults for xlim if none set
        if xlim is None:
            xlim = self.xlim

        # get the x and y values
        if self.discrete:
            xs = np.arange(int(xlim[0]), int(xlim[1]) + 1)
        else:
            xs = np.linspace(xlim[0], xlim[1], 200)
        ys = self.pdf(xs)

        # determine limits for y-axes based on y values
        ymin, ymax = ys[np.isfinite(ys)].min(), ys[np.isfinite(ys)].max()
        ylim = min(0, ymin - 0.05 * (ymax - ymin)), 1.05 * ymax

        # get the current axis if they exist and no axis is specified
        fig = plt.gcf()
        if ax is None and fig.axes:
            ax = plt.gca()

        # if axis already exists, set it to the union of existing and current axis
        if ax is not None:
            xlower, xupper = ax.get_xlim()
            xlim = min(xlim[0], xlower), max(xlim[1], xupper)
            ylower, yupper = ax.get_ylim()
            ylim = min(ylim[0], ylower), max(ylim[1], yupper)
        else:
            ax = plt.gca()  # creates new axis

        # set the axis limits
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        # get next color in cycle
        color = get_next_color(ax)

        # plot points for discrete distributions
        if self.discrete:
            ax.scatter(xs, ys, s=40, color=color, alpha=alpha, **kwargs)

        # plot curve
        ax.plot(xs, ys, color=color, alpha=alpha, **kwargs)

        # adjust the axes, base x-axis at 0
        ax.spines["bottom"].set_position("zero")


## Discrete Distributions


class Bernoulli(Distribution):
    """Probability space for a Bernoulli distribution.

    Models a single trial with two outcomes: success (1) with probability
    ``p``, or failure (0) with probability ``1 - p``.

    Parameters
    ----------
    p : float
        Probability of success (1), between 0 and 1.

    Attributes
    ----------
    p : float
        Probability of success (1), between 0 and 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Bernoulli(p=0.5)
    >>> float(X.mean())
    0.5
    >>> float(X.sd())
    0.5
    >>> float(X.pmf(1))
    0.5
    >>> X.draw()  # doctest: +SKIP
    1
    """

    def __init__(self, p):
        """Initialize a Bernoulli distribution."""
        if 0 <= p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {"p": p}
        super().__init__(params, stats.bernoulli, True)
        self.xlim = (
            0,
            1,
        )  # Bernoulli distributions are not defined for x < 0 and x > 1


class Binomial(Distribution):
    """Probability space for a binomial distribution.

    Models the number of successes in ``n`` independent trials, each
    with probability ``p`` of success.

    Parameters
    ----------
    n : int
        Number of trials. Must be a non-negative integer.
    p : float
        Probability of success on each trial, between 0 and 1.

    Attributes
    ----------
    n : int
        Number of trials.
    p : float
        Probability of success on each trial, between 0 and 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Binomial(n=4, p=0.5)
    >>> float(X.mean())
    2.0
    >>> float(X.sd())
    1.0
    >>> round(float(X.pmf(2)), 4)
    0.375
    >>> X.draw()  # doctest: +SKIP
    2
    """

    def __init__(self, n, p):
        """Initialize a binomial distribution."""

        if n >= 0 and isinstance(n, numbers.Integral):
            self.n = n
        else:
            raise Exception("n must be a non-negative integer")

        if 0 <= p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {"n": n, "p": p}
        super().__init__(params, stats.binom, True)
        self.xlim = (0, n)  # Binomial distributions are not defined for x < 0 and x > n


class Hypergeometric(Distribution):
    """Probability space for a hypergeometric distribution.

    Models the number of successes (1s) when drawing ``n`` items
    without replacement from a collection containing ``N0`` zeros
    and ``N1`` ones.

    Parameters
    ----------
    n : int
        Number of draws (without replacement). Must be positive.
    N0 : int
        Number of 0s (failures) in the collection.
    N1 : int
        Number of 1s (successes) in the collection.

    Attributes
    ----------
    n : int
        Number of draws (without replacement).
    N0 : int
        Number of 0s (failures) in the collection.
    N1 : int
        Number of 1s (successes) in the collection.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Hypergeometric(n=2, N0=3, N1=3)
    >>> float(X.mean())
    1.0
    >>> round(float(X.pmf(1)), 4)
    0.6
    >>> X.draw()  # doctest: +SKIP
    1
    """

    def __init__(self, n, N0, N1):
        """Initialize a hypergeometric distribution."""

        if n > 0 and isinstance(n, numbers.Integral):
            self.n = n
        else:
            raise Exception("n must be a positive integer")

        if N0 >= 0 and isinstance(N0, numbers.Integral):
            self.N0 = N0
        else:
            raise Exception("N0 must be a non-negative integer")

        if N1 >= 0 and isinstance(N1, numbers.Integral):
            self.N1 = N1
        else:
            raise Exception("N1 must be a non-negative integer")

        params = {"M": N0 + N1, "n": N1, "N": n}

        if N0 + N1 < n:
            raise Exception("N0 + N1 cannot be less than the sample size n")

        super().__init__(params, stats.hypergeom, True)
        self.xlim = (
            0,
            n,
        )  # Hypergeometric distributions are not defined for x < 0 and x > n


class Geometric(Distribution):
    """Probability space for a geometric distribution.

    Models the number of trials (including the success) until the
    first success, where each trial has probability ``p`` of success.

    Parameters
    ----------
    p : float
        Probability of success on each trial, strictly between 0 and 1.

    Attributes
    ----------
    p : float
        Probability of success on each trial, strictly between 0 and 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Geometric(p=0.5)
    >>> float(X.mean())
    2.0
    >>> float(X.pmf(1))
    0.5
    >>> X.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, p):
        """Initialize a geometric distribution."""

        if 0 < p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {"p": p}
        super().__init__(params, stats.geom, True)
        self.xlim = (
            1,
            self.xlim[1],
        )  # Geometric distributions are not defined for x < 1


class NegativeBinomial(Distribution):
    """Probability space for a negative binomial distribution.

    Models the total number of trials (including the ``r`` successes)
    until the ``r``-th success, where each trial has probability ``p``
    of success.

    Parameters
    ----------
    r : int
        Target number of successes. Must be a positive integer.
    p : float
        Probability of success on each trial, between 0 and 1.

    Attributes
    ----------
    r : int
        Target number of successes.
    p : float
        Probability of success on each trial.

    Examples
    --------
    >>> from symbulate import *
    >>> X = NegativeBinomial(r=3, p=0.5)
    >>> float(X.mean())
    6.0
    >>> X.draw()  # doctest: +SKIP
    7
    """

    def __init__(self, r, p):
        """Initialize a negative binomial distribution."""

        if 0 < r and isinstance(r, numbers.Integral):
            self.r = r
        else:
            raise Exception("r must be a positive integer")

        if 0 < p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {"n": r, "p": p, "loc": r}
        super().__init__(params, stats.nbinom, True)
        self.xlim = (
            r,
            self.xlim[1],
        )  # Negative Binomial distributions are not defined for x < r

    def draw(self):
        """Draw a single random sample from the negative binomial distribution.

        Returns
        -------
        int
            The total number of trials (including the ``r`` successes)
            until the ``r``-th success.

        Examples
        --------
        >>> from symbulate import *
        >>> NegativeBinomial(r=3, p=0.5).draw()  # doctest: +SKIP
        6
        """

        # Numpy's negative binomial returns numbers in [0, inf),
        # but we want numbers in [r, inf).
        return self.r + rng.negative_binomial(n=self.r, p=self.p)


class Pascal(Distribution):
    """Probability space for a Pascal distribution.

    Models the number of failures before the ``r``-th success, where
    each trial has probability ``p`` of success. Unlike the negative
    binomial, the ``r`` successes themselves are not counted.

    Parameters
    ----------
    r : int
        Target number of successes. Must be a positive integer.
    p : float
        Probability of success on each trial, between 0 and 1.

    Attributes
    ----------
    r : int
        Target number of successes.
    p : float
        Probability of success on each trial.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Pascal(r=1, p=0.5)
    >>> float(X.mean())
    1.0
    >>> round(float(X.pmf(0)), 4)
    0.5
    >>> X.draw()  # doctest: +SKIP
    0
    """

    def __init__(self, r, p):
        """Initialize a Pascal distribution."""

        if 0 < r and isinstance(r, numbers.Integral):
            self.r = r
        else:
            raise Exception("r must be a positive integer")

        if 0 < p <= 1:
            self.p = p
        else:
            raise Exception("p must be between 0 and 1")

        params = {"n": r, "p": p}
        super().__init__(params, stats.nbinom, True)
        self.xlim = (0, self.xlim[1])  # Pascal distributions are not defined for x < 0


class Poisson(Distribution):
    """Probability space for a Poisson distribution.

    Models the number of events occurring in a fixed interval of time
    or space, when events happen at a constant average rate ``lam``
    and independently of each other.

    Parameters
    ----------
    lam : float
        Average number of events per interval (λ). Must be positive.

    Attributes
    ----------
    lam : float
        Average number of events per interval (the rate parameter,
        often written as λ). Must be positive.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Poisson(lam=4)
    >>> float(X.mean())
    4.0
    >>> float(X.sd())
    2.0
    >>> X.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, lam):
        """Initialize a Poisson distribution."""

        if 0 <= lam:
            self.lam = lam
        else:
            raise Exception("Lambda (lam) must be greater than 0")

        params = {"mu": lam}
        super().__init__(params, stats.poisson, True)
        self.xlim = (0, self.xlim[1])  # Poisson distributions are not defined for x < 0


class DiscreteUniform(Distribution):
    """Probability space for a discrete uniform distribution.

    Every integer from ``a`` to ``b`` (inclusive) is equally likely.
    This can model, for example, rolling a fair die.

    Parameters
    ----------
    a : int, optional
        Smallest possible value. Default is 0.
    b : int, optional
        Largest possible value (inclusive). Default is 1.

    Attributes
    ----------
    a : int
        Smallest possible value.
    b : int
        Internal upper bound (stored as ``b + 1`` for scipy compatibility).

    Examples
    --------
    >>> from symbulate import *
    >>> X = DiscreteUniform(a=1, b=6)
    >>> float(X.mean())
    3.5
    >>> float(X.pmf(3))
    0.16666666666666666
    >>> X.draw()  # doctest: +SKIP
    4
    """

    def __init__(self, a=0, b=1):
        """Initialize a discrete uniform distribution."""
        self.a = a
        self.b = b + 1

        params = {"low": self.a, "high": self.b}

        if a > b:
            raise Exception("b cannot be less than a")

        super().__init__(params, stats.randint, True)
        self.xlim = (a, b)  # Uniform distributions are not defined for x < a and x > b


## Continuous Distributions


class Uniform(Distribution):
    """Probability space for a continuous uniform distribution.

    Every value between ``a`` and ``b`` is equally likely.

    Parameters
    ----------
    a : float, optional
        Lower bound of the distribution. Default is 0.0.
    b : float, optional
        Upper bound of the distribution. Default is 1.0.

    Attributes
    ----------
    a : float
        Lower bound of the distribution.
    b : float
        Upper bound of the distribution.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Uniform(a=0, b=1)
    >>> float(X.mean())
    0.5
    >>> float(X.pdf(0.5))
    1.0
    >>> X.draw()  # doctest: +SKIP
    0.374
    """

    def __init__(self, a=0.0, b=1.0):
        """Initialize a uniform distribution."""
        self.a = a
        self.b = b

        params = {"loc": a, "scale": b - a}

        if a > b:
            raise Exception("b cannot be less than a")

        super().__init__(params, stats.uniform, False)
        self.xlim = (a, b)  # Uniform distributions are not defined for x < a and x > b


class Normal(Distribution):
    """Probability space for a normal (Gaussian) distribution.

    The classic bell-shaped distribution, described by its mean and
    standard deviation. You can specify either ``sd`` or ``var``,
    but not both.

    Parameters
    ----------
    mean : float, optional
        Mean of the distribution. Default is 0.0.
    sd : float, optional
        Standard deviation. Must be positive. Default is 1.0.
    var : float, optional
        Variance. If provided, overrides ``sd``. Must be positive.

    Attributes
    ----------
    scale : float
        The standard deviation used internally, regardless of whether
        ``sd`` or ``var`` was passed in.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Normal(mean=0, sd=1)
    >>> float(X.mean())
    0.0
    >>> float(X.sd())
    1.0
    >>> float(X.pdf(0))
    0.3989422804014327
    >>> X.draw()  # doctest: +SKIP
    -0.234
    """

    def __init__(self, mean=0.0, sd=1.0, var=None):
        """Initialize a normal distribution."""

        if var is None:
            if sd >= 0:
                self.scale = sd
            else:
                raise Exception("sd cannot be less than 0")

        else:
            if var >= 0:
                self.scale = np.sqrt(var)
            else:
                raise Exception("var cannot be less than 0")

        params = {"loc": mean, "scale": self.scale}
        super().__init__(params, stats.norm, False)


class Exponential(Distribution):
    """Probability space for an exponential distribution.

    Models the waiting time between events that occur at a constant
    average rate. Specify either ``rate`` (λ) or ``scale`` (1/λ),
    but not both.

    Parameters
    ----------
    rate : float, optional
        Rate parameter λ. Must be positive. Default is 1.0.
        Mutually exclusive with ``scale``.
    scale : float, optional
        Scale parameter 1/λ. If provided, overrides ``rate``.
        Must be positive.

    Attributes
    ----------
    rate : float or None
        The rate parameter λ. ``None`` if ``scale`` was specified instead.
    scale : float or None
        The scale parameter 1/λ. ``None`` if ``rate`` was specified instead.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Exponential(rate=1)
    >>> float(X.mean())
    1.0
    >>> float(X.sd())
    1.0
    >>> float(X.pdf(1))
    0.36787944117144233
    >>> X.draw()  # doctest: +SKIP
    0.423
    """

    def __init__(self, rate=1.0, scale=None):
        """Initialize an exponential distribution."""

        if scale is None:
            if rate > 0:
                self.rate = rate
                self.scale = scale
            else:
                raise Exception("rate must be positive")
        else:
            if scale > 0:
                self.scale = scale
            else:
                raise Exception("scale must be positive")

        params = {"scale": 1.0 / rate if scale is None else scale}
        super().__init__(params, stats.expon, False)
        self.xlim = (
            0,
            self.xlim[1],
        )  # Exponential distributions are not defined for x < 0


class Gamma(Distribution):
    """Probability space for a gamma distribution.

    A flexible continuous distribution often used to model waiting times.
    Specify either ``rate`` or ``scale``, but not both. The gamma
    generalizes the exponential (``shape=1``) distribution.

    Parameters
    ----------
    shape : float
        Shape parameter α. Must be positive.
    rate : float, optional
        Rate parameter λ. Default is 1.0. Mutually exclusive with ``scale``.
    scale : float, optional
        Scale parameter 1/λ. If provided, overrides ``rate``.
        Must be positive.

    Attributes
    ----------
    shape : float
        Shape parameter α. Controls the number of "phases."
    rate : float or None
        Rate parameter λ. ``None`` if ``scale`` was specified instead.
    scale : float or None
        Scale parameter 1/λ. ``None`` if ``rate`` was specified instead.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Gamma(shape=2, rate=1)
    >>> float(X.mean())
    2.0
    >>> float(X.sd())
    1.4142135623730951
    >>> X.draw()  # doctest: +SKIP
    1.52
    """

    def __init__(self, shape, rate=1.0, scale=None):
        """Initialize a gamma distribution."""

        if 0 < shape:
            self.shape = shape
        else:
            raise Exception("shape parameter must be positive")

        if scale is None:
            if rate > 0:
                self.rate = rate
                self.scale = scale
            else:
                raise Exception("rate must be positive")
        else:
            if scale > 0:
                self.scale = scale
            else:
                raise Exception("scale must be positive")

        params = {"a": shape, "scale": 1.0 / rate if scale is None else scale}
        super().__init__(params, stats.gamma, False)
        self.xlim = (0, self.xlim[1])  # Gamma distributions are not defined for x < 0


class Beta(Distribution):
    """Probability space for a beta distribution.

    A continuous distribution defined on [0, 1], often used to model
    probabilities or proportions. The shape changes with parameters
    ``a`` and ``b``.

    Parameters
    ----------
    a : float
        First shape parameter (α). Must be positive.
    b : float
        Second shape parameter (β). Must be positive.

    Attributes
    ----------
    a : float
        First shape parameter (α). Must be positive.
    b : float
        Second shape parameter (β). Must be positive.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Beta(a=1, b=1)
    >>> float(X.mean())
    0.5
    >>> round(float(X.pdf(0.5)), 4)
    1.0
    >>> X.draw()  # doctest: +SKIP
    0.632
    """

    def __init__(self, a, b):
        """Initialize a beta distribution."""

        if 0 < a:
            self.a = a
        else:
            raise Exception("a must be positive")

        if 0 < b:
            self.b = b
        else:
            raise Exception("b must be positive")

        params = {"a": a, "b": b}
        super().__init__(params, stats.beta, False)
        self.xlim = (0, 1)  # Beta distributions are not defined for x < 0 and x > 1


class StudentT(Distribution):
    """Probability space for Student's t-distribution.

    A bell-shaped distribution similar to the normal but with heavier
    tails. Commonly used in statistical inference when the sample size
    is small. As degrees of freedom increase, it approaches the normal
    distribution.

    Parameters
    ----------
    df : int or float
        Degrees of freedom. Must be positive.

    Attributes
    ----------
    df : int or float
        Degrees of freedom. Must be positive.

    Examples
    --------
    >>> from symbulate import *
    >>> X = StudentT(df=10)
    >>> float(X.mean())
    0.0
    >>> X.draw()  # doctest: +SKIP
    0.312
    """

    def __init__(self, df):
        """Initialize a Student's t-distribution."""
        if df > 0:
            self.df = df
        else:
            raise Exception("df must be greater than 0")

        params = {"df": df}
        super().__init__(params, stats.t, False)
        if df == 1:
            self.mean = lambda: float("nan")
            self.sd = lambda: float("nan")
            self.var = lambda: float("nan")


class ChiSquare(Distribution):
    """Probability space for a chi-square distribution.

    Arises as the sum of squares of independent standard normal random
    variables. Commonly used in hypothesis testing and confidence
    intervals for variance.

    Parameters
    ----------
    df : int
        Degrees of freedom. Must be a positive integer.

    Attributes
    ----------
    df : int
        Degrees of freedom. Must be a positive integer.

    Examples
    --------
    >>> from symbulate import *
    >>> X = ChiSquare(df=4)
    >>> float(X.mean())
    4.0
    >>> float(X.sd())
    2.8284271247461903
    >>> X.draw()  # doctest: +SKIP
    3.14
    """

    def __init__(self, df):
        """Initialize a chi-square distribution."""
        if df > 0 and isinstance(df, numbers.Integral):
            self.df = df
        else:
            raise Exception("df must be a positive integer")

        params = {"df": df}
        super().__init__(params, stats.chi2, False)
        self.xlim = (
            0,
            self.xlim[1],
        )  # Chi-Square distributions are not defined for x < 0


class F(Distribution):
    """Probability space for an F-distribution.

    Arises as the ratio of two chi-square random variables divided by
    their degrees of freedom. Commonly used in analysis of variance
    (ANOVA) to compare group variances.

    Parameters
    ----------
    dfN : int or float
        Degrees of freedom for the numerator. Must be positive.
    dfD : int or float
        Degrees of freedom for the denominator. Must be positive.

    Attributes
    ----------
    dfN : int or float
        Degrees of freedom for the numerator.
    dfD : int or float
        Degrees of freedom for the denominator.

    Examples
    --------
    >>> from symbulate import *
    >>> X = F(dfN=5, dfD=10)
    >>> float(X.mean())
    1.25
    >>> X.draw()  # doctest: +SKIP
    0.85
    """

    def __init__(self, dfN, dfD):
        """Initialize an F-distribution."""

        if dfN > 0:
            self.dfN = dfN
        else:
            raise Exception("dfN must be greater than 0")

        if dfD > 0:
            self.dfD = dfD
        else:
            raise Exception("dfD must be greater than 0")

        params = {"dfn": dfN, "dfd": dfD}
        super().__init__(params, stats.f, False)
        self.xlim = (0, self.xlim[1])  # F distributions are not defined for x < 0


class Cauchy(Distribution):
    """Probability space for a Cauchy distribution.

    A heavy-tailed symmetric distribution centered at ``loc``. The Cauchy
    distribution has no finite mean or variance — it is a classic example
    where the law of large numbers does not apply.

    Parameters
    ----------
    loc : float, optional
        Location parameter (center of the distribution). Default is 0.
    scale : float, optional
        Scale parameter (controls the spread). Default is 1.

    Attributes
    ----------
    loc : float
        Location parameter (center of the distribution). Default is 0.
    scale : float
        Scale parameter (controls the spread). Default is 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Cauchy(loc=0, scale=1)
    >>> float(X.pdf(0))
    0.3183098861837907
    >>> X.draw()  # doctest: +SKIP
    -2.31
    """

    def __init__(self, loc=0, scale=1):
        """Initialize a Cauchy distribution."""
        self.loc = loc
        self.scale = scale

        params = {"loc": loc, "scale": scale}

        super().__init__(params, stats.cauchy, False)

    def draw(self):
        """Draw a single random sample from the Cauchy distribution.

        Returns
        -------
        float
            One random value drawn from the Cauchy distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> Cauchy().draw()  # doctest: +SKIP
        1.48
        """
        return self.loc + (self.scale * rng.standard_cauchy())


class LogNormal(Distribution):
    """Probability space for a log-normal distribution.

    If ``Y`` has a log-normal distribution with parameters ``mu`` and
    ``sigma``, then ``log(Y)`` follows a normal distribution with mean
    ``mu`` and standard deviation ``sigma``. Often used to model positive,
    right-skewed quantities such as income or stock prices.

    Parameters
    ----------
    mu : float, optional
        Mean of the underlying normal distribution. Default is 0.0.
    sigma : float, optional
        Standard deviation of the underlying normal distribution.
        Must be positive. Default is 1.0.

    Attributes
    ----------
    norm_mean : float
        Mean of the underlying normal distribution (μ).
    norm_sd : float
        Standard deviation of the underlying normal distribution (σ).
    s : float
        Shape parameter passed to scipy (equal to ``sigma``).

    Examples
    --------
    >>> from symbulate import *
    >>> X = LogNormal(mu=0, sigma=1)
    >>> float(X.mean())
    1.6487212707001282
    >>> X.draw()  # doctest: +SKIP
    0.94
    """

    def __init__(self, mu=0.0, sigma=1.0):
        """Initialize a log-normal distribution."""

        self.norm_mean = mu

        if sigma == 0:
            _value = np.exp(mu)
            self.s = 0
            self.norm_sd = 0
            self.discrete = False
            self.params = {"s": 0, "scale": _value}
            self.pdf = lambda x: float(x == _value)
            self.cdf = lambda x: 0.0 if x < _value else 1.0
            self.mean = lambda: _value
            self.var = lambda: 0.0
            self.sd = lambda: 0.0
            self.median = lambda: _value
            self.xlim = (0, _value + 1)
            ProbabilitySpace.__init__(self, lambda: Scalar(_value))
            return
        elif sigma > 0:
            self.s = sigma
            self.norm_sd = sigma
        else:
            raise Exception("sigma must be greater than 0")

        params = {"s": self.s, "scale": np.exp(mu)}
        super().__init__(params, stats.lognorm, False)
        self.xlim = (
            0,
            self.xlim[1],
        )  # Log-Normal distributions are not defined for x < 0


class Pareto(Distribution):
    """Probability space for a Pareto distribution.

    A heavy-tailed distribution often used to model phenomena where a
    small fraction of items account for a large share of the effect
    (the "80/20 rule"). All values are at least ``scale``.

    Parameters
    ----------
    b : float, optional
        Shape parameter (tail index). Must be positive. Default is 1.0.
    scale : float, optional
        Minimum possible value (lower bound of the support).
        Must be positive. Default is 1.0.

    Attributes
    ----------
    b : float
        Shape parameter (tail index). Larger values give thinner tails.
    scale : float
        Minimum possible value (lower bound of the support).

    Examples
    --------
    >>> from symbulate import *
    >>> X = Pareto(b=2, scale=1)
    >>> float(X.mean())
    2.0
    >>> X.draw()  # doctest: +SKIP
    1.34
    """

    def __init__(self, b=1.0, scale=1.0):
        """Initialize a Pareto distribution."""

        if b > 0:
            self.b = b
        else:
            raise Exception("b must be greater than 0")

        if scale > 0:
            self.scale = scale
        else:
            raise Exception("scale must be greater than 0")

        params = {"b": self.b, "scale": self.scale}
        super().__init__(params, stats.pareto, False)
        self.xlim = (
            scale,
            self.xlim[1],
        )  # Pareto distributions are not defined for x < scale

    def draw(self):
        """Draw a single random sample from the Pareto distribution.

        Returns
        -------
        float
            One random value drawn from the Pareto distribution.
            Always greater than or equal to ``scale``.

        Examples
        --------
        >>> from symbulate import *
        >>> Pareto(b=2, scale=1).draw()  # doctest: +SKIP
        1.42
        """

        # Numpy's Pareto is Lomax distribution, or Type II Pareto
        # but we want the more standard parametrization
        return self.scale * (1 + rng.pareto(self.b))


class Rayleigh(Distribution):
    """Probability space for a Rayleigh distribution.

    Arises when computing the magnitude of a two-dimensional vector
    whose components are independent, identically distributed normal
    random variables. Often used in signal processing.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Rayleigh()
    >>> round(float(X.mean()), 4)
    1.2533
    >>> X.draw()  # doctest: +SKIP
    0.88
    """

    def __init__(self):
        """Initialize a Rayleigh distribution."""
        params = {}
        super().__init__(params, stats.rayleigh, False)


## Multivariate Distributions


class MultivariateNormal(Distribution):
    """Probability space for a multivariate normal distribution.

    Generalizes the normal distribution to multiple dimensions. Each
    draw produces a vector of correlated normal values, described by
    a mean vector and a covariance matrix.

    Parameters
    ----------
    mean : array-like of length n
        The mean vector of the distribution.
    cov : array-like of shape (n, n)
        The covariance matrix. Must be symmetric and positive semi-definite.

    Attributes
    ----------
    mean : array-like of length n
        The mean vector of the distribution.
    cov : array-like of shape (n, n)
        The covariance matrix. Must be symmetric and positive semi-definite.

    Examples
    --------
    >>> from symbulate import *
    >>> X = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    >>> X.draw()  # doctest: +SKIP
    (1.76, 0.40)
    """

    def __init__(self, mean, cov):
        """Initialize a multivariate normal distribution."""
        if len(mean) != len(cov):
            raise Exception(
                "The dimension of the mean vector"
                + " is not compatible with the dimensions"
                + " of the covariance matrix."
            )

        if len(mean) >= 1:
            self.mean = mean
        else:
            raise Exception("Mean vector and Cov matrix cannot be empty")

        if len(cov) >= 1:
            if all(len(row) == len(mean) for row in cov):
                if np.all(np.linalg.eigvals(cov) >= 0) and np.allclose(
                    cov, np.transpose(cov)
                ):
                    self.cov = cov
                else:
                    raise Exception(
                        "Cov matrix is not symmetric and positive semi-definite"
                    )
            else:
                raise Exception("Cov matrix is not square")
        else:
            raise Exception("Dimension of cov matrix cannot be less than 1")

        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(mean, cov).pdf(x)

    def plot(self):
        """Plot is not supported for multivariate distributions.

        Raises
        ------
        Exception
            Always raised — plotting is not available for the
            multivariate normal distribution.
        """
        raise Exception(
            "Plotting is not currently available for "
            "the multivariate normal distribution."
        )

    def draw(self):
        """Draw a single random sample from the multivariate normal distribution.

        Returns
        -------
        Vector
            A random vector drawn from the multivariate normal distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]]).draw()  # doctest: +SKIP
        (1.76, 0.40)
        """

        return Vector(rng.multivariate_normal(self.mean, self.cov))

    def __pow__(self, exponent):
        """Draw multiple independent samples from the multivariate normal distribution.

        Parameters
        ----------
        exponent : int or float
            Number of samples to draw. Pass ``float('inf')`` to create
            an infinite sequence of draws generated lazily on demand.

        Returns
        -------
        ProbabilitySpace
            A probability space whose draws produce ``exponent`` samples
            at a time.

        Examples
        --------
        >>> from symbulate import *
        >>> (MultivariateNormal([0, 0], [[1, 0], [0, 1]]) ** 3).draw()  # doctest: +SKIP
        [(-0.23, 1.72), (0.31, -0.72), (0.88, -0.21)]
        """
        if exponent == float("inf"):

            def draw():
                def _func(n):
                    return self.draw()

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)


class BivariateNormal(MultivariateNormal):
    """Probability space for a bivariate normal distribution.

    A special case of the multivariate normal with exactly two variables
    (X and Y). You can specify the relationship between X and Y using
    either correlation (``corr``) or covariance (``cov``).

    Parameters
    ----------
    mean1 : float, optional
        Mean of the first variable. Default is 0.0.
    mean2 : float, optional
        Mean of the second variable. Default is 0.0.
    sd1 : float, optional
        Standard deviation of the first variable. Default is 1.0.
    sd2 : float, optional
        Standard deviation of the second variable. Default is 1.0.
    corr : float, optional
        Correlation between the two variables, between -1 and 1.
        Default is 0.0.
    var1 : float, optional
        Variance of the first variable. Overrides ``sd1`` if provided.
    var2 : float, optional
        Variance of the second variable. Overrides ``sd2`` if provided.
    cov : float, optional
        Covariance between the two variables. Overrides ``corr`` if provided.

    Attributes
    ----------
    mean : list of float
        The mean vector ``[mean1, mean2]``.
    cov : list of list of float
        The 2×2 covariance matrix.

    Examples
    --------
    >>> from symbulate import *
    >>> X = BivariateNormal(mean1=0, mean2=0, sd1=1, sd2=1, corr=0.5)
    >>> X.draw()  # doctest: +SKIP
    (1.76, 1.20)
    """

    def __init__(
        self,
        mean1=0.0,
        mean2=0.0,
        sd1=1.0,
        sd2=1.0,
        corr=0.0,
        var1=None,
        var2=None,
        cov=None,
    ):
        """Initialize a bivariate normal distribution."""

        if not -1 <= corr <= 1:
            raise Exception("Correlation must be " "between -1 and 1.")

        self.mean = [mean1, mean2]

        if sd1 < 0:
            raise Exception("sd1 cannot be less than 0")
        if sd2 < 0:
            raise Exception("sd2 cannot be less than 0")

        if var1 is None:
            var1 = sd1**2
        if var2 is None:
            var2 = sd2**2
        if var1 < 0 or var2 < 0:
            raise Exception("var1 and var2 cannot be negative")
        if cov is None:
            cov = corr * np.sqrt(var1 * var2)
        self.cov = [[var1, cov], [cov, var2]]
        self.discrete = False
        self.pdf = lambda x: stats.multivariate_normal(self.mean, self.cov).pdf(x)


class Multinomial(Distribution):
    """Probability space for a multinomial distribution.

    Generalizes the binomial distribution to more than two outcomes.
    Models the counts of each outcome across ``n`` independent trials,
    where each trial lands in one of several categories with fixed
    probabilities.

    Parameters
    ----------
    n : int
        Number of trials. Must be a non-negative integer.
    p : array-like of float
        Probability of each outcome. Must be non-negative and sum to 1.

    Attributes
    ----------
    n : int
        Number of trials.
    p : array-like of float
        Probability of each outcome. Must be non-negative and sum to 1.

    Examples
    --------
    >>> from symbulate import *
    >>> X = Multinomial(n=10, p=[0.5, 0.3, 0.2])
    >>> X.draw()  # doctest: +SKIP
    (5, 3, 2)
    """

    def __init__(self, n, p):
        """Initialize a multinomial distribution."""
        if n >= 0 and isinstance(n, numbers.Integral):
            self.n = n
        else:
            raise Exception("n must be a non-negative integer")

        if sum(p) == 1 and min(p) >= 0:
            self.p = p
        else:
            raise Exception("Elements of p must be non-negative" + " and sum to 1.")

        self.discrete = False
        self.pdf = lambda x: stats.multinomial(n, p).pmf(x)

    def plot(self):
        """Plot is not supported for multivariate distributions.

        Raises
        ------
        Exception
            Always raised — plotting is not available for the
            multinomial distribution.
        """
        raise Exception(
            "Plotting is not currently available for " "the Multinomial distribution."
        )

    def draw(self):
        """Draw a single random sample from the multinomial distribution.

        Returns
        -------
        Vector
            A vector of counts, one per category, summing to ``n``.

        Examples
        --------
        >>> from symbulate import *
        >>> Multinomial(n=10, p=[0.5, 0.3, 0.2]).draw()  # doctest: +SKIP
        (6, 2, 2)
        """

        return Vector(rng.multinomial(self.n, self.p))

    def __pow__(self, exponent):
        """Draw multiple independent samples from the multinomial distribution.

        Parameters
        ----------
        exponent : int or float
            Number of samples to draw. Pass ``float('inf')`` to create
            an infinite sequence of draws generated lazily on demand.

        Returns
        -------
        ProbabilitySpace
            A probability space whose draws produce ``exponent`` samples
            at a time.

        Examples
        --------
        >>> from symbulate import *
        >>> (Multinomial(10, [0.5, 0.3, 0.2]) ** 3).draw()  # doctest: +SKIP
        [(5, 3, 2), (4, 4, 2), (6, 2, 2)]
        """
        if exponent == float("inf"):

            def draw():
                def _func(_):
                    return self.draw()

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)
