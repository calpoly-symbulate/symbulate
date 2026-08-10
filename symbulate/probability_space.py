from collections import Counter
import numbers

import numpy as np

rng = np.random.default_rng()

from .base import Logical
from .result import Vector, InfiniteVector, join
from .results import Results, _sim_with_progress


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
    >>> rng = np.random.default_rng()
    >>> die = ProbabilitySpace(lambda: rng.choice([1, 2, 3, 4, 5, 6]))
    >>> die.draw()  # doctest: +SKIP
    3
    """

    def __init__(self, draw):
        """Initialize the probability space."""
        if not callable(draw):
            raise TypeError(
                "draw must be a callable (e.g., a lambda or function) "
                f"that returns one outcome, but got {type(draw).__name__}."
            )
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

        Raises
        ------
        ValueError
            If ``n`` is not a positive integer.

        Examples
        --------
        >>> from symbulate import *
        >>> rng = np.random.default_rng()
        >>> coin = ProbabilitySpace(lambda: rng.choice(["H", "T"]))
        >>> coin.sim(10)  # doctest: +SKIP
        Results(['H', 'T', 'H', 'H', 'T', 'T', 'H', 'T', 'H', 'T'])
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"n must be a positive integer, got {n!r}.")
        return Results(_sim_with_progress(self.draw, n))

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
        >>> rng = np.random.default_rng()
        >>> die = ProbabilitySpace(lambda: rng.choice([1, 2, 3, 4, 5, 6]))
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
        >>> rng = np.random.default_rng()
        >>> die = ProbabilitySpace(lambda: rng.choice([1, 2, 3, 4, 5, 6]))
        >>> coin = ProbabilitySpace(lambda: rng.choice(["H", "T"]))
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

        Raises
        ------
        ValueError
            If ``exponent`` is not a positive integer or ``float('inf')``.

        Examples
        --------
        >>> from symbulate import *
        >>> rng = np.random.default_rng()
        >>> die = ProbabilitySpace(lambda: rng.choice([1, 2, 3, 4, 5, 6]))
        >>> three_dice = die ** 3
        >>> three_dice.draw()  # doctest: +SKIP
        Vector([2, 5, 3])
        """
        if exponent != float("inf") and (not isinstance(exponent, int) or exponent < 1):
            raise ValueError(
                f"Exponent must be a positive integer or float('inf'), got {exponent!r}."
            )
        if exponent == float("inf"):

            def draw():
                def _func(_):
                    return self.draw()

                return InfiniteVector(_func)

        else:

            def draw():
                return Vector(self.draw() for _ in range(exponent))

        return ProbabilitySpace(draw)

    def __rshift__(self, cond_space_func):
        """Create a hierarchical probability space from a prior and a conditional function.

        Parameters
        ----------
        cond_space_func : callable
            A function that takes one outcome drawn from ``self`` (the
            prior) and returns a ``ProbabilitySpace`` (the conditional,
            or "child," space to draw from given that outcome).

        Returns
        -------
        ProbabilitySpace
            A new probability space whose draws produce one outcome from
            ``self`` and one from the conditional space it determines,
            joined into a single tuple.

        Examples
        --------
        >>> from symbulate import *
        >>> P = Beta(1, 2) >> (lambda x: Binomial(10, x))
        >>> P.draw()  # doctest: +SKIP
        (0.34, 3)
        """
        return Hierarchical(self, cond_space_func)


class HierarchicalProbabilitySpace(ProbabilitySpace):
    """Defines a probability space built from a prior and a conditional space.

    A hierarchical probability space draws an outcome from a prior
    probability space, uses that outcome to determine a second
    ("conditional," or "child") probability space, and draws from that
    child space. The two outcomes are joined into a single tuple. This is
    the mechanism behind the ``>>`` operator.

    Parameters
    ----------
    prior_space : ProbabilitySpace
        The probability space to draw the prior outcome from.
    cond_space_func : callable
        A function that takes one outcome drawn from ``prior_space`` and
        returns a ``ProbabilitySpace`` to draw the conditional outcome
        from.

    Attributes
    ----------
    prior_space : ProbabilitySpace
        The probability space the prior outcome is drawn from.
    cond_space_func : callable
        The function mapping a prior outcome to a conditional
        probability space.

    Raises
    ------
    TypeError
        If ``prior_space`` is not a ``ProbabilitySpace``, if
        ``cond_space_func`` is not callable, or if calling
        ``cond_space_func`` on a prior outcome does not return a
        ``ProbabilitySpace``.

    Examples
    --------
    >>> from symbulate import *
    >>> P = HierarchicalProbabilitySpace(Beta(1, 2), lambda x: Binomial(10, x))
    >>> P.draw()  # doctest: +SKIP
    (0.34, 3)
    """

    def __init__(self, prior_space, cond_space_func):
        """Initialize the hierarchical probability space."""
        if not isinstance(prior_space, ProbabilitySpace):
            raise TypeError(
                "The left-hand side of '>>' must be a ProbabilitySpace "
                f"(e.g., a distribution), but got {type(prior_space).__name__}."
            )
        if not callable(cond_space_func):
            raise TypeError(
                "The right-hand side of '>>' must be a callable (e.g., a "
                "lambda or function) that takes the prior's value and "
                f"returns a ProbabilitySpace, but got {type(cond_space_func).__name__}."
            )
        self.prior_space = prior_space
        self.cond_space_func = cond_space_func

        def draw():
            prior_value = prior_space.draw()
            cond_space = cond_space_func(prior_value)
            if not isinstance(cond_space, ProbabilitySpace):
                raise TypeError(
                    "The function passed to '>>' must return a "
                    "ProbabilitySpace, but got "
                    f"{type(cond_space).__name__}. Did you forget to wrap "
                    "the return value in a distribution, e.g. "
                    "'lambda x: Binomial(10, x)'?"
                )
            return join(prior_value, cond_space.draw())

        super().__init__(draw)


def Hierarchical(prior_space, cond_space_func):
    """Create a hierarchical probability space from a prior and a conditional function.

    A convenience alias for ``HierarchicalProbabilitySpace``, and the
    function that ``prior_space >> cond_space_func`` calls.

    Parameters
    ----------
    prior_space : ProbabilitySpace
        The probability space to draw the prior outcome from.
    cond_space_func : callable
        A function that takes one outcome drawn from ``prior_space`` and
        returns a ``ProbabilitySpace`` to draw the conditional outcome
        from.

    Returns
    -------
    ProbabilitySpace
        A new probability space whose draws produce one outcome from
        ``prior_space`` and one from the conditional space it determines,
        joined into a single tuple.

    Examples
    --------
    >>> from symbulate import *
    >>> P = Hierarchical(Beta(1, 2), lambda x: Binomial(10, x))
    >>> P.draw()  # doctest: +SKIP
    (0.34, 3)
    """
    return HierarchicalProbabilitySpace(prior_space, cond_space_func)


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
    >>> rng = np.random.default_rng()
    >>> die = ProbabilitySpace(lambda: rng.choice([1, 2, 3, 4, 5, 6]))
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
        >>> rng = np.random.default_rng()
        >>> die = ProbabilitySpace(lambda: rng.choice([1, 2, 3, 4, 5, 6]))
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
        >>> rng = np.random.default_rng()
        >>> die = ProbabilitySpace(lambda: rng.choice([1, 2, 3, 4, 5, 6]))
        >>> even_event = Event(die, lambda x: x % 2 == 0)
        >>> even_event.sim(10000).mean()  # doctest: +SKIP
        0.498
        """
        return Results(_sim_with_progress(self.draw, n))


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
            If ``box`` is empty.
        ValueError
            If ``size`` is negative.
        ValueError
            If ``probs`` is provided but its length does not match the number
            of tickets in the box.
        ValueError
            If the ``size`` exceeds the number of tickets in the box when
            sampling without replacement.
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
        if len(self.box) == 0:
            raise ValueError(
                "box is empty -- there is nothing to draw from. Give "
                "BoxModel a non-empty list of tickets, e.g. "
                "BoxModel([1, 2, 3])."
            )
        # size=None means "draw 1 ticket"; float('inf') is a legitimate,
        # deliberately unbounded lazy sequence. Only an actual negative
        # number is invalid -- this used to pass silently and only fail
        # later, inside .draw(), with a raw NumPy message ("negative
        # dimensions are not allowed") that never mentions size.
        if (
            size is not None
            and isinstance(size, numbers.Real)
            and size != float("inf")
            and size < 0
        ):
            raise ValueError(
                f"size must be a non-negative number of tickets to draw, "
                f"got size={size!r}."
            )
        if probs is not None and len(probs) != len(self.box):
            raise ValueError(
                f"probs must have the same length as box, "
                f"but got len(probs)={len(probs)} and len(box)={len(self.box)}."
            )
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
            return rng.choice(
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


# ----------------------------------------------------------------------------
# Poker hands
#
# Helpers for classifying a 5-card poker hand drawn from a DeckOfCards. These
# are plain functions (not methods) so they slot into Symbulate's usual flow:
# wrap a deck in an RV and apply one of them, e.g.
#
#     deck = DeckOfCards(size=5)
#     RV(deck).apply(is_full_house).sim(10000).mean()        # P(full house)
#     RV(deck).apply(classify_hand).sim(10000).tabulate()    # all categories
# ----------------------------------------------------------------------------

# Numeric value of each card rank, used to detect straights.
_RANK_VALUES = {rank: rank for rank in range(2, 11)}
_RANK_VALUES.update({"J": 11, "Q": 12, "K": 13, "A": 14})

# Poker hand categories, ordered from best to worst. ``classify_hand``
# returns exactly one of these strings.
POKER_HANDS = (
    "royal flush",
    "straight flush",
    "four of a kind",
    "full house",
    "flush",
    "straight",
    "three of a kind",
    "two pair",
    "pair",
    "high card",
)


def _rank_value(rank):
    """Map a single card rank to its numeric value (2 through 14).

    Parameters
    ----------
    rank : int or str
        A card rank: an integer 2-10, or one of ``'J'``, ``'Q'``,
        ``'K'``, ``'A'``.

    Returns
    -------
    int
        The numeric value of the rank, where ``'J'`` is 11, ``'Q'`` is
        12, ``'K'`` is 13, and ``'A'`` is 14.

    Raises
    ------
    ValueError
        If ``rank`` is not a recognized card rank.
    """
    try:
        return _RANK_VALUES[rank]
    except (KeyError, TypeError):
        raise ValueError(
            f"{rank!r} is not a valid card rank. Ranks must be the integers "
            "2 through 10 or one of 'J', 'Q', 'K', 'A', as produced by "
            "DeckOfCards."
        )


def classify_hand(hand):
    """Classify a 5-card poker hand into its single best category.

    Parameters
    ----------
    hand : iterable of (rank, suit)
        A poker hand of exactly 5 cards, such as one draw from
        ``DeckOfCards(size=5)``. Each card is a ``(rank, suit)`` tuple.

    Returns
    -------
    str
        One of the categories in ``POKER_HANDS``. The categories are
        mutually exclusive and the *best* one is returned, so a hand
        with three of one rank and two of another is ``'full house'``,
        never ``'pair'`` or ``'three of a kind'``.

    Raises
    ------
    ValueError
        If the hand does not contain exactly 5 cards, or a card has an
        unrecognized rank.

    Examples
    --------
    >>> classify_hand([(10, 'Hearts'), ('J', 'Hearts'), ('Q', 'Hearts'),
    ...                 ('K', 'Hearts'), ('A', 'Hearts')])
    'royal flush'
    >>> classify_hand([(3, 'Clubs'), (3, 'Hearts'), (3, 'Spades'),
    ...                 (8, 'Clubs'), (8, 'Diamonds')])
    'full house'
    >>> classify_hand([('A', 'Clubs'), (2, 'Hearts'), (3, 'Spades'),
    ...                 (4, 'Clubs'), (5, 'Diamonds')])
    'straight'
    """
    cards = list(hand)
    if len(cards) != 5:
        raise ValueError(
            "A poker hand must consist of exactly 5 cards, but got "
            f"{len(cards)}. Draw a hand with DeckOfCards(size=5)."
        )
    values = sorted(_rank_value(card[0]) for card in cards)
    suits = [card[1] for card in cards]

    is_flush = len(set(suits)) == 1

    distinct = sorted(set(values))
    is_straight = len(distinct) == 5 and distinct[-1] - distinct[0] == 4
    # The Ace can also play low in the "wheel" straight A-2-3-4-5.
    is_straight = is_straight or set(values) == {14, 2, 3, 4, 5}

    # Rank-count pattern in descending order, e.g. [3, 2] for a full
    # house, [2, 2, 1] for two pair, [2, 1, 1, 1] for a single pair.
    count_pattern = sorted(Counter(values).values(), reverse=True)

    if is_straight and is_flush:
        # A 10-J-Q-K-A straight flush is the special case of a royal flush.
        if set(values) == {10, 11, 12, 13, 14}:
            return "royal flush"
        return "straight flush"
    if count_pattern[0] == 4:
        return "four of a kind"
    if count_pattern == [3, 2]:
        return "full house"
    if is_flush:
        return "flush"
    if is_straight:
        return "straight"
    if count_pattern[0] == 3:
        return "three of a kind"
    if count_pattern == [2, 2, 1]:
        return "two pair"
    if count_pattern == [2, 1, 1, 1]:
        return "pair"
    return "high card"


def is_royal_flush(hand):
    """Return True if the hand is a royal flush (10-J-Q-K-A, one suit)."""
    return classify_hand(hand) == "royal flush"


def is_straight_flush(hand):
    """Return True if the hand is a straight flush (but not a royal flush)."""
    return classify_hand(hand) == "straight flush"


def is_four_of_a_kind(hand):
    """Return True if the hand is four of a kind."""
    return classify_hand(hand) == "four of a kind"


def is_full_house(hand):
    """Return True if the hand is a full house (three of a kind plus a pair)."""
    return classify_hand(hand) == "full house"


def is_flush(hand):
    """Return True if the hand is a flush (but not a straight or royal flush)."""
    return classify_hand(hand) == "flush"


def is_straight(hand):
    """Return True if the hand is a straight (but not a straight flush)."""
    return classify_hand(hand) == "straight"


def is_three_of_a_kind(hand):
    """Return True if the hand's best category is exactly three of a kind."""
    return classify_hand(hand) == "three of a kind"


def is_two_pair(hand):
    """Return True if the hand's best category is exactly two pair."""
    return classify_hand(hand) == "two pair"


def is_pair(hand):
    """Return True if the hand's best category is exactly one pair.

    Because categories are mutually exclusive, a full house or two pair
    returns ``False`` here (they are not *just* a pair).
    """
    return classify_hand(hand) == "pair"


def is_high_card(hand):
    """Return True if the hand is high card (none of the other categories)."""
    return classify_hand(hand) == "high card"
