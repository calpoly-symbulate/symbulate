import warnings

from .base import Arithmetic, Transformable, Comparable
from .probability_space import Event, ProbabilitySpace
from .result import Vector, join, is_scalar, is_numeric_vector, TimeFunction, Scalar
from .results import RVResults, _sim_with_progress

# The identity function RV uses when no `func` is given, named so that
# sim() can recognize it with `is`. Python evaluates a default argument
# once, so this is the very same object every bare RV(P) already received
# implicitly -- naming it changes nothing except that it can now be
# identified. A hand-written `X.apply(lambda x: x)` builds a *different*
# lambda, so it correctly fails the `is` check and keeps the slow path.
_IDENTITY = lambda x: x  # noqa: E731 -- named on purpose, see comment above


class RV(Arithmetic, Transformable, Comparable):
    """Defines a random variable.

    A random variable is a function which maps an outcome of
    a probability space to a number. Simulating a random
    variable is a two-step process: first, a draw is taken
    from the underlying probability space; then, the function
    is applied to that draw to obtain the realized value of
    the random variable.

    Parameters
    ----------
    prob_space : ProbabilitySpace
        The underlying probability space of the random variable.
    func : callable, optional
        A function that maps draws from the probability space to numbers.
        Defaults to the identity function.

    Attributes
    ----------
    prob_space : ProbabilitySpace
        The underlying probability space of the random variable.
    func : callable
        A function that maps draws from the probability space to numbers.

    Examples
    --------
    >>> from symbulate import *
    >>> P = BoxModel([0, 1], size=5)
    >>> X = RV(P, sum)
    >>> X.draw()  # doctest: +SKIP
    2
    >>> Y = RV(Normal(0, 1))
    >>> Y.draw()  # doctest: +SKIP
    -0.9
    """

    def __init__(self, prob_space, func=_IDENTITY):
        """Create a random variable."""
        if not callable(func):
            raise TypeError(
                "func must be a callable (e.g., a lambda or function), "
                f"but got {type(func).__name__}."
            )
        self.prob_space = prob_space
        self.func = func

    def draw(self):
        """Draw a single realization of the random variable.

        Returns
        -------
        scalar or array-like
            A single realized value of the random variable.

        Examples
        --------
        Use ``seed()`` to make a draw reproducible. ``np.random.seed()`` does
        **not** work here -- Symbulate draws from a NumPy ``Generator``, which
        the legacy global seed does not control.

        >>> from symbulate import *
        >>> seed(0)
        >>> X = RV(Normal(0, 1))
        >>> X.draw()  # doctest: +SKIP
        0.1257302210933933

        The exact value above depends on NumPy's and SciPy's internals, so it
        is not asserted here; what is guaranteed is that reseeding with the
        same value reproduces the same draw.

        >>> seed(0)
        >>> first = X.draw()
        >>> seed(0)
        >>> first == X.draw()
        True
        """
        return self.func(self.prob_space.draw())

    def sim(self, n):
        """Simulate n draws from the random variable.

        Parameters
        ----------
        n : int
            Number of draws to simulate.

        Returns
        -------
        RVResults
            A list-like object containing the simulation results.

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.

        Examples
        --------
        >>> from symbulate import *
        >>> seed(0)
        >>> X = RV(Normal(0, 1))
        >>> X.sim(3)  # doctest: +SKIP
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        # Batched fast path, but only for a bare random variable sitting
        # directly on a distribution that offers one. `func is _IDENTITY`
        # means nothing has been composed on top: the moment .apply(),
        # conditioning, or arithmetic replaces func, the check fails and the
        # ordinary per-draw loop runs, since those have to see each outcome
        # one at a time. Distribution._fast_sim returns None for all but two
        # distributions, so almost everything falls through regardless.
        if self.func is _IDENTITY and hasattr(self.prob_space, "_fast_sim"):
            batch = self.prob_space._fast_sim(n)
            if batch is not None:
                return RVResults([Scalar(v) for v in batch])
        return RVResults(_sim_with_progress(self.draw, n))

    def __call__(self, outcome):
        """Apply the RV's function directly to an outcome.

        Parameters
        ----------
        outcome : any
            An outcome passed directly to the RV's defining function,
            without checking if it belongs to the underlying probability space.

        Returns
        -------
        scalar or tuple
            The value of the RV's function applied to the outcome.

        Examples
        --------
        >>> from symbulate import *
        >>> X = RV(Normal(0, 1))
        >>> X(0.5)  # doctest: +SKIP
        0.5
        """
        warnings.warn(
            "Calling an RV as a function simply applies the "
            "function that defines the RV to the input, regardless of "
            "whether that input is a possible outcome in the underlying "
            "probability space.",
            UserWarning,
            stacklevel=2,
        )
        return self.func(outcome)

    def check_same_prob_space(self, other):
        """Check that this RV and another object share the same probability space.

        Parameters
        ----------
        other : any
            The object to compare probability spaces with. If ``other`` has
            no ``prob_space`` attribute, the check is skipped.

        Raises
        ------
        Exception
            If the two objects are defined on different probability spaces.
        """
        if hasattr(other, "prob_space"):
            self.prob_space.check_same(other.prob_space)

    def apply(self, func):
        """Transform a random variable by a function.

        Parameters
        ----------
        func : callable
            A function to apply to the random variable.

        Returns
        -------
        RV
            A new random variable with the transformation applied.

        Examples
        --------
        >>> from symbulate import *
        >>> X = RV(Exponential(rate=1))
        >>> Y = X.apply(log)
        """

        def _func(outcome):
            return func(self.func(outcome))

        return RV(self.prob_space, _func)

    # This allows us to unpack a random vector,
    # e.g., X, Y = RV(BoxModel([0, 1], size=2))
    def __iter__(self):
        """Iterate over the components of a random vector.

        Allows unpacking a random vector into individual scalar RVs.

        Yields
        ------
        RV
            Each component of the random vector as a scalar RV.

        Raises
        ------
        Exception
            If the random variable does not have multiple components.

        Examples
        --------
        >>> from symbulate import *
        >>> X, Y = RV(BoxModel([0, 1], size=2))
        """
        test = self.draw()
        if hasattr(test, "__iter__"):
            for i in range(len(test)):
                yield self[i]
        else:
            raise Exception(
                "To unpack a random vector, the RV needs to "
                "have multiple components."
            )

    def __getitem__(self, n):
        """Index into a random vector to select a component or sub-vector.

        Parameters
        ----------
        n : int, list of int, slice, or RV
            Index or indices to select. An ``RV`` uses its realization as
            a random index; a list or slice returns a random vector.

        Returns
        -------
        RV
            A new random variable representing the indexed component(s).

        Examples
        --------
        >>> from symbulate import *
        >>> X = RV(BoxModel([0, 1], size=5))
        >>> X[0]        # first component
        >>> X[0:3]      # first three components as a vector
        >>> X[[0, 2]]   # components at indices 0 and 2
        """
        # if n is an RV, return a new random variable
        if isinstance(n, RV):
            return RV(self.prob_space, lambda x: self.func(x)[n.func(x)])
        # if the indices are a list, return a random vector
        elif is_numeric_vector(n):
            return self.apply(lambda x: Vector(x[i] for i in n))
        # if the indices are a slice, return a random vector
        elif isinstance(n, slice):
            # x is the RV's own realized value (e.g. a tuple/Vector from a
            # fixed-size draw), which already knows its own length and
            # already supports native slicing -- so slice it directly
            # rather than rebuilding a range() from n.start/stop/step.
            # The old range()-based version broke on any open-ended slice
            # (X[1:], with n.stop is None) with a raw
            # "TypeError: 'NoneType' object cannot be interpreted as an
            # integer", since range() requires an actual integer stop.
            return self.apply(lambda x: Vector(x[n]))
        # otherwise, return the nth value
        return self.apply(lambda x: x[n])

    def _operation_factory(self, op):
        """Create an arithmetic operation between this RV and another value.

        Used by the ``Arithmetic`` superclass to define operators such as
        ``+``, ``-``, ``*``, ``/``, ``**``, etc.

        Parameters
        ----------
        op : callable
            A binary function representing the arithmetic operation.

        Returns
        -------
        callable
            A method that applies ``op`` between this RV and another RV
            or scalar, returning a new RV.
        """

        def _op_func(self, other):
            # operations between this RV and another RV
            if isinstance(other, RV):
                self.check_same_prob_space(other)

                def _func(outcome):
                    return op(self.func(outcome), other.func(outcome))

                return RV(self.prob_space, _func)
            # operations between this RV and a scalar
            return self.apply(lambda x: op(x, other))

        return _op_func

    def _comparison_factory(self, op):
        """Create a comparison operation between this RV and another value.

        Used by the ``Comparable`` superclass to define comparison operators
        such as ``<``, ``>``, ``==``, ``!=``, etc. Comparisons on an RV
        return an ``Event`` rather than a boolean.

        Parameters
        ----------
        op : callable
            A binary function representing the comparison operation.

        Returns
        -------
        callable
            A method that applies ``op`` between this RV and a scalar or
            another RV, returning an ``Event``.
        """

        def _op_func(self, other):
            if is_scalar(other):
                return Event(self.prob_space, lambda x: op(self.func(x), other))
            elif isinstance(other, RV):
                self.check_same_prob_space(other)
                return Event(self.prob_space, lambda x: op(self.func(x), other.func(x)))
            raise NotImplementedError(
                "Comparisons are only defined between two RVs or "
                "between an RV and a scalar."
            )

        return _op_func

    # Define a joint distribution of two random variables: e.g., X & Y
    def __and__(self, other):
        """Form the joint distribution of two random variables using ``&``.

        Parameters
        ----------
        other : RV, scalar, or TimeFunction
            The other random variable, constant, or time function to join with.

        Returns
        -------
        RV
            A new random variable whose realizations are vectors joining both components.

        Raises
        ------
        Exception
            If other is not an RV, scalar, or TimeFunction.

        Examples
        --------
        >>> from symbulate import *
        >>> X, Y = RV(Normal(0, 1) ** 2)
        >>> Z = X & Y
        """
        self.check_same_prob_space(other)
        if isinstance(other, RV):

            def _func(outcome):
                return join(self.func(outcome), other.func(outcome))

        elif isinstance(other, Event):

            def _func(outcome):
                return join(self.func(outcome), other.func(outcome))

        elif is_scalar(other):

            def _func(outcome):
                return join(self.func(outcome), other)

        elif isinstance(other, TimeFunction):

            def _func(outcome):
                return join(self.func(outcome), other)

        else:
            raise Exception("Joint distributions are only defined for RVs and events.")
        return RV(self.prob_space, _func)

    def __rand__(self, other):
        """Support scalar & RV or TimeFunction & RV by forming a joint distribution.

        Called when a scalar or TimeFunction appears on the left side of ``&``
        (e.g., ``3 & X`` or ``t & X``).

        Parameters
        ----------
        other : scalar or TimeFunction
            The constant or time function to join on the left side.

        Returns
        -------
        RV
            A new random variable whose realizations are vectors with ``other`` as the first component.

        Examples
        --------
        >>> from symbulate import *
        >>> X = RV(Normal(0, 1))
        >>> Z = 3 & X
        """
        self.check_same_prob_space(other)
        if is_scalar(other) or isinstance(other, TimeFunction):

            def _func(outcome):
                return join(other, self.func(outcome))

        elif isinstance(other, ProbabilitySpace):
            other.check_same(self.prob_space)

            def _func(outcome):
                return join(outcome, self.func(outcome))

        else:
            raise Exception("Joint distributions are only defined for RVs and scalars.")
        return RV(self.prob_space, _func)

    # Define conditional distribution of random variable.
    # e.g., X | (X > 3)
    def __or__(self, condition_event):
        """Condition the random variable on an event using ``|``.

        Parameters
        ----------
        condition_event : Event
            The event to condition on.

        Returns
        -------
        RVConditional
            A new conditional random variable.

        Raises
        ------
        NotImplementedError
            If condition_event is not an Event.

        Examples
        --------
        >>> from symbulate import *
        >>> X = RV(Normal(0, 1))
        >>> (X | (X > 0)).draw()  # doctest: +SKIP
        0.7
        """
        # Check that the random variable and event are
        # defined on the same probability space.
        self.check_same_prob_space(condition_event)
        if isinstance(condition_event, Event):
            return RVConditional(self, condition_event)
        else:
            raise NotImplementedError(
                "The right side of | must be an Event (e.g., X > 0), "
                f"but got {type(condition_event).__name__}. "
                "For example, use (X | (X > 0)) to condition on X being positive."
            )


class RVConditional(RV):
    """Defines a random variable conditional on an event.

    RVConditionals are typically produced when you condition a
    RV on an Event object.

    Parameters
    ----------
    random_variable : RV
        The random variable whose conditional distribution is desired.
    condition_event : Event
        The event to condition on.

    Attributes
    ----------
    condition_event : Event
        The event to condition on.

    Examples
    --------
    >>> from symbulate import *
    >>> X, Y = RV(Binomial(10, 0.4) ** 2)
    >>> (X | (X + Y == 5)).draw()  # doctest: +SKIP
    3
    """

    def __init__(self, random_variable, condition_event):
        """Create a conditional random variable."""
        self.condition_event = condition_event
        super().__init__(random_variable.prob_space, random_variable.func)

    def draw(self):
        """Draw a single value from the conditional distribution.

        Returns
        -------
        scalar or array-like
            A single realized value from the conditional distribution.

        Examples
        --------
        >>> from symbulate import *
        >>> X, Y = RV(Binomial(10, 0.4) ** 2)
        >>> (X | (X + Y == 5)).draw()  # doctest: +SKIP
        3
        """
        while True:
            outcome = self.prob_space.draw()
            if self.condition_event.func(outcome):
                return self.func(outcome)
