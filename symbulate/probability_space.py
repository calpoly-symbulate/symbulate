import numpy as np

from .base import Logical
from .result import Vector, InfiniteVector, join
from .results import Results


class ProbabilitySpace:
    """Defines a probability space.

    A probability space encapsulates a random experiment by wrapping
    a ``draw`` function that produces one outcome each time it is called.

    Parameters
    ----------
    draw : callable
        A function that takes no arguments and returns one outcome
        from the probability space.

    Attributes
    ----------
    draw : callable
        A function that takes no arguments and returns one outcome
        from the probability space.

    Examples
    --------
    >>> from symbulate import *
    >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
    >>> die.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, draw):
        """Initialize the probability space."""
        self.draw = draw

    def sim(self, n):
        """Simulate n draws from probability space.

        Parameters
        ----------
        n : int
            How many draws to make.

        Returns
        -------
        Results
            A list-like object containing the simulation results.

        Examples
        --------
        >>> from symbulate import *
        >>> coin = ProbabilitySpace(lambda: np.random.choice(["H", "T"]))
        >>> coin.sim(10)  # doctest: +SKIP
        Results(['H', 'T', 'H', 'H', 'T', 'T', 'H', 'T', 'H', 'T'])
        """
        return Results(self.draw() for _ in range(n))

    def check_same(self, other):
        """Check that two probability spaces are the same object.

        Parameters
        ----------
        other : ProbabilitySpace
            The other probability space to check against.

        Raises
        ------
        ValueError
            If ``self`` and ``other`` are not the same probability space object.
        """
        if self != other:
            raise ValueError("Events must be defined on same probability space.")

    def apply(self, func):
        """Define a new probability space.

        Parameters
        ----------
        func : callable
            A function to apply to each outcome drawn from this space.

        Returns
        -------
        ProbabilitySpace
            A new probability space where each outcome is ``func`` applied
            to one draw from the current space.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> squared_die = die.apply(lambda x: x ** 2)
        >>> squared_die.draw()  # doctest: +SKIP
        16
        """

        def draw():
            return func(self.draw())

        return ProbabilitySpace(draw)

    def __mul__(self, other):
        """Create a joint probability space from two independent probability spaces.

        Parameters
        ----------
        other : ProbabilitySpace
            The other probability space to combine with.

        Returns
        -------
        ProbabilitySpace
            A new probability space whose draws produce one outcome from
            ``self`` and one from ``other``, joined into a single tuple.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> coin = ProbabilitySpace(lambda: np.random.choice(["H", "T"]))
        >>> joint_space = die * coin
        >>> joint_space.draw()  # doctest: +SKIP
        (3, 'H')
        """

        def draw():
            return join(self.draw(), other.draw())

        return ProbabilitySpace(draw)

    def __pow__(self, exponent):
        """Create a new probability space from the current one by taking multiple independent draws.

        Parameters
        ----------
        exponent : int or float
            Number of independent draws to take. Pass ``float('inf')``
            to create an infinite sequence of draws generated lazily
            on demand.

        Returns
        -------
        ProbabilitySpace
            A new probability space whose draws produce a Vector of
            ``exponent`` outcomes from the original space.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> three_dice = die ** 3
        >>> three_dice.draw()  # doctest: +SKIP
        Vector([2, 5, 3])
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


class Event(Logical):
    """Defines an event in a probability space.

    An event is a predicate over outcomes: given one draw from a
    probability space, it returns a bool indicating whether that
    outcome satisfies the event.

    Parameters
    ----------
    prob_space : ProbabilitySpace
        The probability space the event is defined on.
    func : callable
        A function that maps one outcome to a bool — ``True`` if the
        event occurred, ``False`` otherwise.

    Attributes
    ----------
    prob_space : ProbabilitySpace
        The probability space that the event is defined on.
    func : callable
        A function that maps outcomes to a bool for whether
        the event occurred.

    Examples
    --------
    >>> from symbulate import *
    >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
    >>> even_event = Event(die, lambda x: x % 2 == 0)
    >>> even_event.draw()  # doctest: +SKIP
    True
    """

    def __init__(self, prob_space, func):
        """Initialize the event."""
        self.prob_space = prob_space
        self.func = func

    def check_same_prob_space(self, other):
        """Check that the event is defined on the same probability space as another event.

        Parameters
        ----------
        other : Event
            The other event to check against.

        Raises
        ------
        ValueError
            If ``self`` and ``other`` are not defined on the same
            probability space.
        """
        self.prob_space.check_same(other.prob_space)

    # The Logical superclass will use this to define the three
    # logical operations: and (&), or (|), not (~).
    def _logical_factory(self, op):
        """Create a logical operation method comparing two events.

        Parameters
        ----------
        op : callable
            The logical operator (e.g., ``and``, ``or``, ``not``) to
            apply to the two events.

        Returns
        -------
        callable
            A bound method that applies ``op`` to ``self`` and another
            event and returns a new combined ``Event``.
        """

        def _op_func(self, other=None):
            # other will be None when op is the "not" operator
            if other is None:
                return Event(self.prob_space, lambda outcome: op(self.func(outcome)))
            else:
                if isinstance(other, Event):
                    self.check_same_prob_space(other)
                else:
                    raise TypeError(
                        "Logical operations are only defined "
                        "between two Events, not between an Event "
                        f"and a {type(other).__name__}."
                    )
                return Event(
                    self.prob_space,
                    lambda outcome: op(self.func(outcome), other.func(outcome)),
                )

        return _op_func

    # This prevents users from writing expressions like 2 < X < 5,
    # which evaluate to ((2 < X) and (X < 5)). This unfortunately
    # is not well-defined in Python and cannot be overloaded.
    def __bool__(self):
        """Prevent casting an Event to a boolean.

        Raises
        ------
        TypeError
            Always raised to prevent chained comparisons like ``2 < X < 5``.
            Use ``(2 < X) & (X < 5)`` instead.
        """
        raise TypeError(
            "Cannot cast an Event to a boolean. "
            "You may be getting this error if you "
            "wrote an expression like (2 < X < 5). "
            "Try ((2 < X) & (X < 5)) instead."
        )

    def draw(self):
        """Draw one outcome from the probability space and check if the event occurred.

        Returns
        -------
        bool
            ``True`` if the event occurred for the drawn outcome,
            ``False`` otherwise.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> event = Event(die, lambda x: x > 3)
        >>> event.draw()  # doctest: +SKIP
        False
        """
        return self.func(self.prob_space.draw())

    def sim(self, n):
        """Simulate n draws from the probability space and check if the event occurred for each draw.

        Parameters
        ----------
        n : int
            Number of draws to make.

        Returns
        -------
        Results
            A Results object containing a bool for each draw — ``True``
            if the event occurred, ``False`` otherwise.

        Examples
        --------
        >>> from symbulate import *
        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> even_event = Event(die, lambda x: x % 2 == 0)
        >>> even_event.sim(10000).mean()  # doctest: +SKIP
        0.498
        """
        return Results(self.draw() for _ in range(n))


class BoxModel(ProbabilitySpace):
    """Defines a probability space from a box model.

    A box model represents drawing tickets from a collection, with options
    for replacement, custom probabilities, and ordering rules.

    Parameters
    ----------
    box : list or dict
        The collection of tickets to sample from. Specify as a list
        of objects or as a dict mapping objects to their counts.
    size : int or float, optional
        Number of tickets to draw per outcome. Default is ``None``
        (draw 1 ticket and return a scalar). Pass ``float('inf')``
        for an infinite lazy sequence.
    replace : bool, optional
        If ``True``, sample with replacement. Default is ``True``.
    probs : list of float, optional
        Sampling probability for each ticket. All tickets are equally
        likely by default. Ignored when ``box`` is a dict.
    order_matters : bool, optional
        If ``True``, different orderings of the same tickets are counted
        as different outcomes. Default is ``True``.

    Attributes
    ----------
    box : list
        The flat list of tickets (after expanding any dict counts).
    size : int or float or None
        Number of tickets drawn per outcome.
    replace : bool
        Whether tickets are drawn with replacement.
    probs : list of float or None
        Per-ticket sampling probabilities, or ``None`` for uniform.
    order_matters : bool
        Whether ticket order affects the outcome.

    Examples
    --------
    >>> from symbulate import *
    >>> box = ['red', 'blue', 'green']
    >>> box_model = BoxModel(box)
    >>> box_model.draw()  # doctest: +SKIP
    'blue'
    """

    def __init__(self, box, size=None, replace=True, probs=None, order_matters=True):
        """Initialize the box model probability space.

        Raises
        ------
        TypeError
            If the ``box`` is not specified as either a list or a dict.
        ValueError
            If the ``size`` exceeds the number of tickets in the box when sampling without replacement.
        """
        if isinstance(box, list):
            self.box = box
            self.probs = probs
        elif isinstance(box, dict):
            self.box = []
            for ticket, count in box.items():
                self.box.extend([ticket] * count)
            self.probs = None
        else:
            raise TypeError("Box must be specified either as a list or a dict.")
        self.size = None if size == 1 else size
        self.replace = replace
        self.order_matters = order_matters
        self.output_type = Vector
        self.infinite_output_type = InfiniteVector

        # If drawing without replacement, check that the number
        # of draws does not exceed the number of tickets in the box.
        if not self.replace and self.size is not None and self.size > len(self.box):
            raise ValueError(
                "Cannot draw more tickets (without replacement) "
                "than there are tickets in the box."
            )

    def draw(self):
        """Draw one outcome from the box model probability space.

        Returns
        -------
        object or Vector or InfiniteVector
            A single ticket if ``size`` is ``None``, a ``Vector`` of
            tickets if ``size`` is a finite integer greater than 1, or
            an ``InfiniteVector`` of tickets if ``size`` is
            ``float('inf')``.

        Examples
        --------
        >>> from symbulate import *
        >>> box = ['red', 'blue', 'green']
        >>> box_model = BoxModel(box, size=3, replace=True)
        >>> box_model.draw()  # doctest: +SKIP
        Vector(['blue', 'red', 'green'])
        """

        def draw_inds(size):
            return np.random.choice(
                len(self.box), size=size, replace=self.replace, p=self.probs
            )

        if self.size is None:
            return self.box[draw_inds(None)]
        elif self.size == float("inf"):

            def _func(_):
                return self.box[draw_inds(None)]

            return self.infinite_output_type(_func)
        else:
            draws = [self.box[i] for i in draw_inds(self.size)]
            if not self.order_matters:
                draws.sort(key=str)
            return self.output_type(draws)


class DeckOfCards(BoxModel):
    """Defines the probability space for drawing from a standard 52-card deck.

    Each card is a tuple of ``(rank, suit)``, where rank is an integer
    2–10 or one of ``'J'``, ``'Q'``, ``'K'``, ``'A'``, and suit is one
    of ``'Diamonds'``, ``'Hearts'``, ``'Clubs'``, ``'Spades'``.

    Parameters
    ----------
    size : int or float, optional
        Number of cards to draw per outcome. Default is ``None``
        (draw 1 card and return a scalar).
    replace : bool, optional
        If ``True``, draw with replacement. Default is ``False``.
    order_matters : bool, optional
        If ``True``, different orderings of the same cards are counted
        as different outcomes. Default is ``True``.

    Attributes
    ----------
    size : int or float or None
        Number of cards drawn per outcome.
    replace : bool
        Whether cards are drawn with replacement.
    order_matters : bool
        Whether card order affects the outcome.

    Examples
    --------
    >>> from symbulate import *
    >>> deck = DeckOfCards()
    >>> deck.draw()  # doctest: +SKIP
    ('A', 'Spades')
    """

    def __init__(self, size=None, replace=False, order_matters=True):
        """Initialize the deck of cards probability space."""
        box = []
        for rank in list(range(2, 11)) + ["J", "Q", "K", "A"]:
            for suit in ["Diamonds", "Hearts", "Clubs", "Spades"]:
                box.append((rank, suit))
        super().__init__(box, size, replace, probs=None, order_matters=order_matters)
