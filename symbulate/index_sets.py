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
            raise KeyError(
                f"{type(self).__name__} does not contain {t!r}."
            )

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
            raise TypeError(
                f"fs must be a positive number, got {type(fs).__name__}."
            )
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
