import numpy as np

from .base import Logical
from .result import Vector, InfiniteVector, join
from .results import Results


class ProbabilitySpace:
    """Defines a probability space.

    Parameters
    ----------
    draw : callable
        A function explaining how to draw one outcome from the probability space.

    Examples
    --------
    Define a probability space for rolling a die:

    >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
    >>> die.draw()
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
        Simulate 10 coin flips:

        >>> coin = ProbabilitySpace(lambda: np.random.choice(["H", "T"]))
        >>> coin.sim(10)
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
        Exception
            If ``self`` and ``other`` are not the same probability space object.
        """
        if self != other:
            raise Exception("Events must be defined on same probability space.")

    def apply(self, func):
        """Define a new probability space.

        Parameters
        ----------
        func : function
            A function to apply to each realization.

        Returns
        -------
        ProbabilitySpace
            A new ProbabilitySpace where each realization is func applied to a realization from the current probability space.

        Examples
        --------
        Square each outcome from a die roll:

        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> squared_die = die.apply(lambda x: x**2)
        >>> squared_die.draw()
        """

        def draw():
            return func(self.draw())

        return ProbabilitySpace(draw)

    def __mul__(self, other):
        """Create a joint probability space from two independent probability spaces.

        Parameters
        ----------
        other : ProbabilitySpace
            The other probability space to join with.

        Returns
        -------
        ProbabilitySpace
            A new probability space where each draw is a pair of outcomes from one draw from ``self`` and one draw from ``other``.

        Examples
        --------
        Model rolling a die and flipping a coin as a joint probability space:

        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> coin = ProbabilitySpace(lambda: np.random.choice(["H", "T"]))
        >>> joint_space = die * coin
        >>> joint_space.draw()
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
            The number of independent draws to take.

        Returns
        -------
        ProbabilitySpace
            A new probability space where each draw is a vector of ``exponent`` outcomes from the original probability space.

        Examples
        --------
        Model rolling 3 dice at once:

        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> three_dice = die ** 3
        >>> three_dice.draw()
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

    An event is a function that takes an outcome from a probability space and returns a boolean indicating whether that event was observed or not.

    Parameters
    ----------
    prob_space : ProbabilitySpace
        The probability space that the event is defined on.
    func : function
        A function that defines the event and maps outcomes to a boolean for whether the event occurred or not.

    Examples
    --------
    Define an event for rolling an even number on a die:

    >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
    >>> even_event = Event(die, lambda x: x % 2 == 0)
    >>> even_event.draw()
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
        Exception
            If the ``self`` and ``other`` are not defined on the same probability space.
        """
        self.prob_space.check_same(other.prob_space)

    # The Logical superclass will use this to define the three
    # logical operations: and (&), or (|), not (~).
    def _logical_factory(self, op):
        """Create a logical operation method comparing two events.

        Parameters
        ----------
        op : callable
            The logical operator to apply to the two events.

        Returns
        -------
        callable
            A bound method that applies ``op`` to two events and returns a new combined event.
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
                        "and a %s." % type(other).__name__
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
        Exception
            If an attempt is made to cast an Event to a boolean.
        """
        raise Exception(
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
            Whether the event occurred ``True`` or not ``False`` for the drawn outcome.

        Examples
        --------
        Check if a single dice roll is greater than 3:

        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> event = Event(die, lambda x: x > 3)
        >>> event.draw()
        True
        """
        return self.func(self.prob_space.draw())

    def sim(self, n):
        """Simulate n draws from the probability space and check if the event occurred for each draw.

        Parameters
        ----------
        n : int
            How many draws to make.

        Returns
        -------
        Results
            A Results object containing a list of boolean results for each draw.

        Examples
        --------
        Estimate the probability of that a dice roll is even:

        >>> die = ProbabilitySpace(lambda: np.random.choice([1, 2, 3, 4, 5, 6]))
        >>> even_event = Event(die, lambda x: x % 2 == 0
        >>> even_event.sim(10).mean()
        0.5
        """
        return Results(self.draw() for _ in range(n))


class BoxModel(ProbabilitySpace):
    """Defines a probability space from a box model.

    Parameters
    ----------
    box : list or dict
        The box to sample from.
        The box can be specified either directly as a list
        of objects or indirectly as a dict of objects and
        their counts.
    size : int, optional
        How many draws to make.
    replace : bool, optional
        Sample with replacement if ``True`` or without if ``False``.
    probs : list, optional
        Probabilities of sampling each ticket
        (by default, all tickets are equally likely). Note
        that this is ignored if box is specified as a dict.
    order_matters : bool, optional
        Count different orderings of the same tickets as different outcomes if ``True`` or the same outcome if ``False``.

    Examples
    --------
    Draw a ticket from a box of colored balls:

    >>> box = ['red', 'blue', 'green']
    >>> box_model = BoxModel(box)
    >>> box_model.draw()
    'blue'
    """

    def __init__(self, box, size=None, replace=True, probs=None, order_matters=True):
        """Initialize the box model probability space.

        Raises
        ------
        Exception
            If the ``box`` is not specified as either a list or a dict.
        Exception
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
            raise Exception("Box must be specified either as a list or a dict.")
        self.size = None if size == 1 else size
        self.replace = replace
        self.order_matters = order_matters
        self.output_type = Vector
        self.infinite_output_type = InfiniteVector

        # If drawing without replacement, check that the number
        # of draws does not exceed the number of tickets in the box.
        if not self.replace and self.size > len(self.box):
            raise Exception(
                "Cannot draw more tickets (without replacement) "
                "than there are tickets in the box."
            )

    def draw(self):
        """Draw one outcome from the box model probability space.

        A function that takes no arguments and returns a value(s) from the
            "box" argument of the BoxModel.

        Returns
        -------
        object or Vector or InfiniteVector
            A single value from the box if size is 1, a Vector of values from the box if size is a finite integer greater than 1, or an InfiniteVector of values from the box if size is infinity.

        Examples
        --------
        Draw 3 tickets from a box:

        >>> box = ['red', 'blue', 'green']
        >>> box_model = BoxModel(box, size=3, replace=True)
        >>> box_model.draw()
        Vector(['blue', 'red', 'green'])
        """

        def draw_inds(size):
            return np.random.choice(len(self.box), size, self.replace, self.probs)

        if self.size is None:
            return self.box[draw_inds(None)]
        elif self.size == float("inf"):

            def _func(_):
                return self.box[draw_inds(None)]

            return self.infinite_output_type(_func)
        else:
            draws = [self.box[i] for i in draw_inds(self.size)]
            if not self.order_matters:
                draws.sort()
            return self.output_type(draws)


class DeckOfCards(BoxModel):
    """Defines the probability space for drawing from a deck of cards.

    Parameters
    ----------
    size : int, optional
        How many draws to make. Defaults to None.
    replace : bool, optional
        Draw with replacement if ``True`` or without replacement if ``False``. Defaults to ``False``.
    order_matters : bool, optional
        If ``True``, count different orderings of the same cards as different outcomes. If ``False``, count different orderings of the same cards as the same outcome. Defaults to ``True``.

    Examples
    --------
    Draw a single card from a deck:

    >>> deck = DeckOfCards()
    >>> deck.draw()
    ('A', 'Spades')
    """

    def __init__(self, size=None, replace=False, order_matters=True):
        """Initialize the deck of cards probability space."""
        box = []
        for rank in list(range(2, 11)) + ["J", "Q", "K", "A"]:
            for suit in ["Diamonds", "Hearts", "Clubs", "Spades"]:
                box.append((rank, suit))
        super().__init__(box, size, replace, probs=None, order_matters=order_matters)
