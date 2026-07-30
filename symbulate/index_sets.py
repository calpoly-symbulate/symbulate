import numbers


class IndexSet:
    """Base class for index sets defining the time domain of a process.

    An index set specifies which values of "time" are valid for
    indexing into a stochastic process. Subclasses define specific
    sets such as the reals, naturals, or integers.
    """

    def __init__(self):
        """Initialize an IndexSet."""
        return

    def __getitem__(self, t):
        """Return t if it is a valid index, otherwise raise a KeyError.

        Parameters
        ----------
        t : numeric
            The time value to look up.

        Returns
        -------
        numeric
            The time value t, if it belongs to the index set.

        Raises
        ------
        KeyError
            If t is not in the index set, with a message naming the set type.

        Examples
        --------
        >>> Reals()[1.5]
        1.5
        >>> Naturals()[3]
        3
        """
        if t in self:
            return t
        else:
            raise KeyError(f"{type(self).__name__} does not contain {t!r}.")

    def __contains__(self, value):
        """Check whether a value belongs to the index set.

        Base implementation always returns False;
        subclasses override this.

        Parameters
        ----------
        value : any
            The value to check.

        Returns
        -------
        bool
            True if value is in the index set, False otherwise.

        Examples
        --------
        >>> 1.5 in Reals()
        True
        >>> 3 in Naturals()
        True
        """
        return False

    def __eq__(self, other):
        """Check whether two index sets are of the same type.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if other is the same type as self.

        Examples
        --------
        >>> Reals() == Reals()
        True
        >>> Reals() == Naturals()
        False
        """
        return type(other) is type(self)

    def __repr__(self):
        """Return the name of the index set.

        Used in the error message a process raises when it is asked for a
        time outside its index set, so subclasses that carry parameters
        (bounds, a sampling frequency) should override this to show them.

        Returns
        -------
        str
            The name of the index set.

        Examples
        --------
        >>> repr(Reals())
        'Reals'
        """
        return type(self).__name__


class Reals(IndexSet):
    """The index set of all finite real numbers.

    Contains all values x such that -inf < x < inf.
    """

    def __init__(self):
        """Initialize a Reals index set."""
        return

    def __contains__(self, value):
        """Check whether a value is a finite real number.

        Parameters
        ----------
        value : any
            The value to check.

        Returns
        -------
        bool
            True if -inf < value < inf, False otherwise.

        Examples
        --------
        >>> 1.5 in Reals()
        True
        >>> float("inf") in Reals()
        False
        """
        try:
            return -float("inf") < value < float("inf")
        except Exception:
            return False


class TimeInterval(Reals):
    """The index set of all times between ``start`` and ``end``, inclusive.

    Use this for a process that only exists over a limited stretch of time,
    rather than forever. A :class:`BrownianBridge` is the main example: it is
    pinned down at both ends, so it is only defined between them, and asking
    for a time outside that stretch is a mistake worth catching rather than
    answering with a meaningless number.

    Like :class:`Reals`, this is continuous time -- every time in the range
    counts, not just whole numbers.

    Parameters
    ----------
    start : numeric
        The first time in the interval.
    end : numeric
        The last time in the interval. Must be greater than ``start``.

    Attributes
    ----------
    start : numeric
        The first time in the interval.
    end : numeric
        The last time in the interval.

    Raises
    ------
    TypeError
        If ``start`` or ``end`` is not a number.
    ValueError
        If ``end`` is not greater than ``start``.

    Examples
    --------
    >>> times = TimeInterval(0, 1)
    >>> 0.5 in times
    True
    >>> 1 in times
    True
    >>> 1.5 in times
    False
    >>> repr(times)
    'TimeInterval(0, 1)'
    """

    def __init__(self, start, end):
        """Initialize a TimeInterval running from start to end."""
        if not isinstance(start, numbers.Real):
            raise TypeError(
                f"start must be a number, got {type(start).__name__}. It is "
                f"the first time in the interval, for example start=0."
            )
        if not isinstance(end, numbers.Real):
            raise TypeError(
                f"end must be a number, got {type(end).__name__}. It is the "
                f"last time in the interval, for example end=1."
            )
        if not end > start:
            raise ValueError(
                f"end must be greater than start, got start={start} and "
                f"end={end}. The interval needs to cover a stretch of time, "
                f"so end cannot come before start (or land on it)."
            )
        self.start = start
        self.end = end

    def __contains__(self, value):
        """Check whether a value lies between start and end, inclusive.

        Parameters
        ----------
        value : any
            The value to check.

        Returns
        -------
        bool
            True if ``start <= value <= end``, False otherwise.

        Examples
        --------
        >>> 0.25 in TimeInterval(0, 1)
        True
        >>> -0.25 in TimeInterval(0, 1)
        False
        """
        try:
            return self.start <= value <= self.end
        except Exception:
            return False

    def __eq__(self, other):
        """Check whether two TimeInterval objects cover the same times.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            True if other is a TimeInterval with the same start and end.

        Examples
        --------
        >>> TimeInterval(0, 1) == TimeInterval(0, 1)
        True
        >>> TimeInterval(0, 1) == TimeInterval(0, 2)
        False
        """
        return (
            isinstance(other, TimeInterval)
            and self.start == other.start
            and self.end == other.end
        )

    def __repr__(self):
        """Return the index set with its bounds, e.g. ``TimeInterval(0, 1)``.

        Returns
        -------
        str
            The name of the index set and the two times it runs between.

        Examples
        --------
        >>> repr(TimeInterval(0, 2))
        'TimeInterval(0, 2)'
        """
        return f"{type(self).__name__}({self.start}, {self.end})"


class Naturals(IndexSet):
    """The index set of natural numbers (non-negative integers).

    Contains all values x such that x >= 0 and x is an integer
    (0, 1, 2, ...).
    """

    def __init__(self):
        """Initialize a Naturals index set."""
        return

    def __contains__(self, value):
        """Check whether a value is a non-negative integer.

        Parameters
        ----------
        value : any
            The value to check.

        Returns
        -------
        bool
            True if value >= 0 and value is an integer,
            False otherwise.

        Examples
        --------
        >>> 3 in Naturals()
        True
        >>> -1 in Naturals()
        False
        >>> 1.5 in Naturals()
        False
        """
        try:
            return value >= 0 and (
                isinstance(value, numbers.Integral) or value.is_integer()
            )
        except Exception:
            return False


class DiscreteTimeSequence(IndexSet):
    """An index set of discrete time values at a given sampling frequency.

    Contains all values of the form n / fs for non-negative integers n.

    Parameters
    ----------
    fs : numeric
        The sampling frequency. Index values are multiples of 1 / fs.

    Attributes
    ----------
    fs : numeric
        The sampling frequency.

    Raises
    ------
    TypeError
        If ``fs`` is not a number.
    ValueError
        If ``fs`` is not positive.
    """

    def __init__(self, fs):
        """Initialize a DiscreteTimeSequence with a given sampling frequency."""
        if not isinstance(fs, (int, float)):
            raise TypeError(f"fs must be a positive number, got {type(fs).__name__}.")
        if fs <= 0:
            raise ValueError(
                f"fs must be positive, got {fs}. "
                "fs is the sampling frequency (time steps per unit)."
            )
        self.fs = fs

    def __getitem__(self, n):
        """Return the time value corresponding to integer index n.

        Parameters
        ----------
        n : int
            The integer index.

        Returns
        -------
        float
            The time value n / fs.

        Examples
        --------
        >>> ts = DiscreteTimeSequence(4)
        >>> ts[2]
        0.5
        >>> ts[8]
        2.0
        """
        return n / self.fs

    def __contains__(self, value):
        """Check whether a value falls on a grid point of the sequence.

        Parameters
        ----------
        value : numeric
            The time value to check.

        Returns
        -------
        bool
            True if value * fs is an integer, False otherwise.

        Examples
        --------
        >>> ts = DiscreteTimeSequence(4)
        >>> 0.25 in ts
        True
        >>> 0.1 in ts
        False
        """
        return float(value * self.fs).is_integer()

    def __eq__(self, index):
        """Check whether two DiscreteTimeSequence objects are equal.

        Parameters
        ----------
        index : object
            The object to compare with.

        Returns
        -------
        bool
            True if index is a DiscreteTimeSequence with the
            same sampling frequency.

        Examples
        --------
        >>> DiscreteTimeSequence(4) == DiscreteTimeSequence(4)
        True
        >>> DiscreteTimeSequence(4) == DiscreteTimeSequence(8)
        False
        """
        return isinstance(index, DiscreteTimeSequence) and (self.fs == index.fs)


class Integers(DiscreteTimeSequence):
    """The index set of non-negative integers (0, 1, 2, ...).

    A special case of DiscreteTimeSequence with sampling frequency
    fs=1.
    """

    def __init__(self):
        """Initialize an Integers index set with sampling frequency 1."""
        super().__init__(1)
