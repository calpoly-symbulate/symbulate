import numbers


class IndexSet(object):
    """Base class for index sets defining the time domain of a process.

    An index set specifies which values of "time" are valid for
    indexing into a stochastic process. Subclasses define specific
    sets such as the reals, naturals, or integers.
    """

    def __init__(self):
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
            If t is not in the index set.
        """
        if t in self:
            return t
        else:
            raise KeyError("Time %.2f not in index set." % t)

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
        """
        return type(other) == type(self)


class Reals(IndexSet):
    """The index set of all finite real numbers.

    Contains all values x such that -inf < x < inf.
    """

    def __init__(self):
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
        """
        try:
            return -float("inf") < value < float("inf")
        except:
            return False


class Naturals(IndexSet):
    """The index set of natural numbers (non-negative integers).

    Contains all values x such that x >= 0 and x is an integer
    (0, 1, 2, ...).
    """

    def __init__(self):
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
        """
        try:
            return value >= 0 and (
                isinstance(value, numbers.Integral) or value.is_integer()
            )
        except:
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
    """

    def __init__(self, fs):
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
        """
        return isinstance(index, DiscreteTimeSequence) and (self.fs == index.fs)


class Integers(DiscreteTimeSequence):
    """The index set of non-negative integers (0, 1, 2, ...).

    A special case of DiscreteTimeSequence with sampling frequency
    fs=1.
    """

    def __init__(self):
        self.fs = 1
