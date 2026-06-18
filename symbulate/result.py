import numbers
import numpy as np
import matplotlib.pyplot as plt

import symbulate
from .base import Arithmetic, Transformable, Statistical, Filterable
from .index_sets import DiscreteTimeSequence, Reals, Naturals


class Scalar(numbers.Number):
    """
    Numeric scalar value.

    A ``Scalar`` creates either an ``Int`` or ``Float`` object based on
    the type of the input value.

    Methods
    -------
    __new__(value, *args, **kwargs)
        Create an integer or floating-point scalar.
    """

    def __new__(cls, value, *args, **kwargs):
        """
        Create a scalar from a numeric value.

        Parameters
        ----------
        value : int or float
            Value used to create the scalar.

        Returns
        -------
        Int or Float
            Scalar object matching the input type.

        Raises
        ------
        Exception
            If the scalar type is not understood.
        """
        if isinstance(value, numbers.Integral):
            return Int(value)
        elif isinstance(value, (float, np.floating)):
            return Float(value)
        else:
            raise Exception("Scalar type not understood.")


class Int(int, Scalar):
    """
    Integer scalar value.

    An ``Int`` is a scalar represented as an integer.
    """

    def __new__(cls, value, *args, **kwargs):
        """
        Create a scalar from a numeric value.

        Parameters
        ----------
        value : int or float
            Value used to create the scalar.

        Returns
        -------
        Int or Float
            Scalar object matching the input type.

        Raises
        ------
        Exception
            If the scalar type is not understood.
        """
        return super(Int, cls).__new__(cls, value)


class Float(float, Scalar):
    """
    Floating-point scalar value.

    A ``Float`` is a scalar represented as a floating-point number.
    """

    def __new__(cls, value, *args, **kwargs):
        """
        Create a scalar from a numeric value.

        Parameters
        ----------
        value : int or float
            Value used to create the scalar.

        Returns
        -------
        Int or Float
            Scalar object matching the input type.

        Raises
        ------
        Exception
            If the scalar type is not understood.
        """
        return super(Float, cls).__new__(cls, value)


class Tuple(Arithmetic, Transformable, Statistical, Filterable):
    """A collapsible data structure."""

    def __init__(self, values):
        """
        Initialize a tuple from finite iterable data.

        Parameters
        ----------
        values : scalar or iterable
            Values to store in the tuple.

        Raises
        ------
        Exception
            If ``values`` is not finite iterable data.
        """
        if is_scalar(values):
            self.values = (values,)
        elif hasattr(values, "__len__") or hasattr(values, "__next__"):
            self.values = tuple(values)
        else:
            raise Exception("Tuples can only be created from " "finite iterable data.")

    def __getitem__(self, n):
        """
        Return value or values at the specified index.

        Parameters
        ----------
        n : int or numeric vector
            Index or collection of indices.

        Returns
        -------
        object or Tuple
            Retrieved value or values.
        """
        # if n is a numeric array, return a Tuple of those values
        if is_numeric_vector(n):
            return type(self)(self.values[i] for i in n)
        # otherwise, return the value at n
        return self.values[n]

    def __len__(self):
        """
        Return the number of values in the tuple.


        Returns
        -------
        int
            Number of values.
        """
        return len(self.values)

    def __iter__(self):
        """
        Iterate over the values in the tuple.

        Yields
        ------
        object
            Next value in the tuple.
        """
        for value in self.values:
            yield value

    def __hash__(self):
        """
        Return the hash of the tuple.

        Returns
        -------
        int
            Hash value.
        """
        return hash(tuple(self.values))

    # Define comparison operators to handle sorting.
    def __eq__(self, other):
        """
        Determine whether two tuples are equal.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            True if the tuples are equal, otherwise False.
        """
        if not hasattr(other, "__len__"):
            return False
        if len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))

    def __lt__(self, other):
        """
        Determine whether this tuple is less than another.

        Parameters
        ----------
        other : Tuple
            Tuple to compare against.

        Returns
        -------
        bool
            True if this tuple is less than ``other``.
        """
        return tuple(self.values) < tuple(other.values)

    def apply(self, func):
        """Apply function to every element of a Tuple.

        Args:
          func: function to apply to the Tuple

        Example:
          x = Tuple([1, 2, 3])
          y = x.apply(log)

        Note: For most standard functions, you can apply the function to
          the Tuple directly. For example, in the example above,
          y = log(x) would have been equivalent and more readable.

        User defined functions can also be applied.

        Example:
          def log_squared(n):
            return log(n) ** 2
          y = x.apply(log_squared)
        """
        return type(self)(func(e) for e in self)

    # The Filterable superclass will use this to define all of the
    # .filter_*() and .count_*() methods.
    def filter(self, filt):
        """Get only the elements that satisfy the given criterion.

        Args:
          filt: A function that takes in an element and returns
            a boolean.

        Returns:
          Tuple: Another Tuple containing only those elements e
          where filt(e) is True.
        """
        return type(self)(e for e in self if filt(e))

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):

        def _op_func(self, other):
            if is_number(other):
                return type(self)(op(value, other) for value in self)
            elif is_vector(other):
                # check that other is the same length as the Tuple
                if len(self) != len(other):
                    raise Exception(
                        "Arithmetic operations between a %s and a %s "
                        "are only valid if they are the same length. "
                        "You attempted to combine a %s of length %d "
                        "with a %s of length %d."
                        % (
                            type(self).__name__,
                            type(other).__name__,
                            type(self).__name__,
                            len(self),
                            type(other).__name__,
                            len(other),
                        )
                    )
                # return a new Tuple/Vector of the same length
                return type(self)(op(a, b) for a, b in zip(self, other))
            else:
                return NotImplemented

        return _op_func

    # The Statistical superclass will use this to define all of the
    # usual statistical functions (e.g., mean, var, etc.)
    def _statistic_factory(self, op):
        def _op_func(self):
            return op(self.values)

        return _op_func

    def cumsum(self):
        """
        Compute the cumulative sum of the vector.

        Returns
        -------
        InfiniteVector
        Infinite vector containing cumulative sums.
        """
        return type(self)(np.cumsum(self.values))

    def plot(self, **kwargs):
        """
        Plot values from the vector over a specified range.

        Parameters
        ----------
        tmin : int, default=0
            Starting index.
        tmax : int, default=10
            Ending index (exclusive).
        **kwargs
            Additional keyword arguments passed to ``matplotlib.pyplot.plot``.
        """
        plt.plot(range(len(self)), self.values, ".--", **kwargs)

    def __str__(self):
        """
        Return a string representation of the infinite tuple.

        Returns
        -------
        str
           String containing the first few values of the tuple.
        """
        if len(self) <= 6:
            return "(" + ", ".join(str(x) for x in self) + ")"
        else:
            first_few = ", ".join(str(x) for x in self[:5])
            last = str(self[-1])
            return "(" + first_few + ", ..., " + last + ")"

    def __repr__(self):
        """
        Return the string representation of the infinite tuple.


        Returns
        -------
        str
        String representation of the infinite tuple.
        """
        return self.__str__()


class Vector(Tuple):
    """A data structure like a Tuple, except it does not collapse."""

    pass


class TimeFunction(Arithmetic):
    """
    Function indexed by a time set.

    A ``TimeFunction`` represents a function whose values are indexed by
    a time set, such as discrete time, continuous time, or natural numbers.

    Methods
    -------
    from_index_set(index_set, func=None)
        Create the appropriate time function based on the given index set.
    check_same_index_set(other)
        Check whether another object can be combined with this time function.

    Raises
    ------
    Exception
        If operations are attempted between time functions with different
        index sets.
    Exception
        If this time function is combined with an unsupported object type.
    """

    @classmethod
    def from_index_set(cls, index_set, func=None):
        """
        Create a time function from an index set.

        Parameters
        ----------
        index_set : DiscreteTimeSequence, Reals, or Naturals
        The index set used to determine the type of time function to create.
        func : callable, optional
        Function used to generate values for the time function. Defaults to None.

        Returns
        -------
        DiscreteTimeFunction, ContinuousTimeFunction, or InfiniteVector
        A time function object matching the given index set.
        """
        if isinstance(index_set, DiscreteTimeSequence):
            return DiscreteTimeFunction(func, index_set=index_set)
        elif isinstance(index_set, Reals):
            return ContinuousTimeFunction(func)
        elif isinstance(index_set, Naturals):
            return InfiniteVector(func)

    def check_same_index_set(self, other):
        """
        Verify that another object is compatible with this time function.

        Parameters
        ----------
        other : object
        The object to compare against. May be a number, random variable,
        or another TimeFunction.

        Raises
        ------
        Exception
        If ``other`` is a TimeFunction with a different index set.

        Exception
        If ``other`` is not a number, random variable, or TimeFunction.
        """
        if isinstance(other, (numbers.Number, symbulate.RV)):
            return
        elif isinstance(other, TimeFunction):
            if self.index_set != other.index_set:
                raise Exception(
                    "Operations can only be performed on "
                    "TimeFunctions with the same index set."
                )
        else:
            raise Exception(
                "Cannot combine %s with %s."
                % (type(self).__name__, type(other).__name__)
            )


class InfiniteTuple(TimeFunction):
    """
    Lazy representation of an infinite tuple.

    An ``InfiniteTuple`` stores values as needed. Values are generated
    from a function whose input is a natural number index.

    Attributes
    ----------
    func : callable
        Function that returns the value at index ``n``.
    index_set : Naturals
        Index set for the infinite tuple.
    values : list
        Cached values that have already been computed.

    Methods
    -------
    apply(func)
        Apply a function to each element of the infinite tuple.
    """

    def __init__(self, func=lambda n: n):
        """Initializes a (lazy) data structure for an infinite vector.

        Args:
          func: A function of n that returns the value in position n.
                n is assumed to be a natural number (integer >= 0).
                This function can be defined at initialization time,
                or later. By default, it is not set at initialization.
        """
        if func is not None:
            self.func = func
        self.index_set = Naturals()
        self.values = []

    def __getitem__(self, n):
        """
        Return the value or slice at the specified index.

        Parameters
        ----------
        n : int or slice
        Index or slice to retrieve.

        Returns
        -------
        object or InfiniteTuple
        Retrieved value or slice.
        """
        m = len(self.values)
        # Add necessary elements to self.values
        n0 = None
        # handle the case where n is a slice
        if isinstance(n, slice):
            if n.stop is None:
                if n.start is None:
                    return self
                else:
                    return type(self)(lambda i: self[i + n.start])
            if n.stop >= m:
                n0 = n.stop
        elif isinstance(n, numbers.Integral) and n >= m:
            n0 = n
        if n0 is not None:
            for i in range(m, n0 + 1):
                self.values.append(self.func(i))
        # Return the corresponding value(s)
        return self.values[n]

    def __call__(self, n):
        """
        Return the value at the specified index.

        Parameters
        ----------
        n : int
            Index of the value to retrieve.

        Returns
        -------
        object
            Value at the specified index.
        """
        return self[n]

    def __str__(self):
        """
        Return a string representation of the infinite tuple.

        Returns
        -------
        str
            String containing the first few values of the tuple.
        """
        first_few = [str(self[i]) for i in range(6)]
        return "(" + ", ".join(first_few) + ", ...)"

    def __repr__(self):
        """
        Return the string representation of the infinite tuple.

        Returns
        -------
        str
        String representation of the infinite tuple.
        """
        return self.__str__()

    def apply(self, func):
        """Apply function to every element of an InfiniteTuple.

        Args:
          func: function to apply to the InfiniteTuple

        Example:
          x = InfiniteTuple(lambda n: n)
          y = x.apply(log)

        Note: For most standard functions, you can apply the function to
          the InfiniteTuple directly. For example, in the example above,
          y = log(x) would have been equivalent and more readable.

        User defined functions can also be applied.

        Example:
          def log_squared(n):
            return log(n) ** 2
          y = x.apply(log_squared)
        """
        return type(self)(lambda n: func(self[n]))

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):
        """
        Create an arithmetic operation for infinite tuples.

        Parameters
        ----------
        op : callable
            Binary operation to apply element-wise.
        Returns
        -------
        callable
            Function implementing the specified operation.
        """

        def _op_func(self, other):
            self.check_same_index_set(other)
            if is_number(other):
                return type(self)(lambda n: op(self[n], other))
            elif isinstance(other, InfiniteTuple):
                return type(self)(lambda n: op(self[n], other[n]))
            else:
                return NotImplemented

        return _op_func


class InfiniteVector(InfiniteTuple):
    """
    Infinite tuple indexed by the natural numbers.

    An ``InfiniteVector`` extends ``InfiniteTuple`` with operations
    specific to sequences of numeric values.

    Methods
    -------
    cumsum()
        Return the cumulative sum of the vector.
    plot(tmin=0, tmax=10, **kwargs)
        Plot a range of values from the vector.
    """

    def cumsum(self):
        """
        Compute the cumulative sum of the vector.

        Returns
        -------
        InfiniteVector
        Infinite vector containing cumulative sums.
        """

        def _func(n):
            return sum(self[i] for i in range(n + 1))

        return InfiniteVector(_func)

    def plot(self, tmin=0, tmax=10, **kwargs):
        """
        Plot values from the vector over a specified range.

        Parameters
        ----------
        tmin : int, default=0
            Starting index.
        tmax : int, default=10
            Ending index (exclusive).
        **kwargs
            Additional keyword arguments passed to ``matplotlib.pyplot.plot``.
        """
        xs = range(tmin, tmax)
        ys = [self[t] for t in range(tmin, tmax)]
        plt.plot(xs, ys, ".--", **kwargs)


class DiscreteTimeFunction(TimeFunction):
    """
    Function indexed by a discrete time sequence.

    A ``DiscreteTimeFunction`` represents values sampled at evenly spaced
    times according to a sampling rate.

    Attributes
    ----------
    func : callable
        Function used to generate values.
    index_set : DiscreteTimeSequence
        Set of valid discrete time values.
    array_pos : list
        Cached values for nonnegative indices.
    array_neg : list
        Cached values for negative indices.

    Methods
    -------
    apply(func)
        Compose a function with this discrete-time function.
    plot(tmin=0, tmax=10, **kwargs)
        Plot values over a specified time range.
    """

    def __init__(self, func=None, fs=1, index_set=None):
        """Initializes a data structure for a discrete-time function.

        Args:
          func: A function of n that returns the value at time n / fs.
            n is assumed to be any integer (postive or negative).
            By default, it is set to the identity function f[n] = n / fs.
          fs (int): The sampling rate of the function, in Hertz (samples
            per second).
          index_set (IndexSet): The index set of the discrete-time function
            (fs is ignored if this is specified.)
        """
        if func is not None:
            self.func = func
        else:
            self.func = lambda n: n / fs
        if index_set is None:
            self.index_set = DiscreteTimeSequence(fs)
        else:
            self.index_set = index_set
        self.array_pos = []  # stores values for t >= 0
        self.array_neg = []  # stores values for t < 0

    def _get_value_at_index(self, n):
        """
        Return the value at the specified sample index.

        Parameters
        ----------
        n : int
            Sample index.

        Returns
        -------
        object
            Value at the specified index.

        Raises
        ------
        KeyError
            If ``n`` is not an integer.
        """
        if not isinstance(n, numbers.Integral):
            raise KeyError(
                "For a DiscreteTimeFunction f, f[n] returns the "
                "the nth time sample, so n must be an integer. "
                "If you want the value at time t, try f(t) instead."
            )

        if n >= 0:
            m = len(self.array_pos)
            if n >= m:
                for i in range(m, n + 1):
                    self.array_pos.append(self.func(i))
            return self.array_pos[n]
        else:
            m = len(self.array_neg)
            if -n > m:
                for i in range(-m - 1, n - 1, -1):
                    self.array_neg.append(self.func(i))
            return self.array_neg[-n - 1]

    def _get_value_at_time(self, t):
        """
        Return the value at the specified time.

        Parameters
        ----------
        t : float
            Time at which to evaluate the function.

        Returns
        -------
        object
            Value at the specified time.

        Raises
        ------
        KeyError
            If ``t`` is not in the index set.
        """
        fs = self.index_set.fs
        if not t in self.index_set:
            raise KeyError(
                (
                    "No value at time %.2f for a function with "
                    "a sampling rate of %d Hz."
                )
                % (t, fs)
            )
        return self._get_value_at_index(int(t * fs))

    def __getitem__(self, n):
        """
        Return value or values at the specified sample index.

        Parameters
        ----------
        n : int, vector, or slice
            Sample index, vector of indices, or slice to retrieve.

        Returns
        -------
        object or Vector
            Value at the specified index, or vector of values.

        Raises
        ------
        TypeError
            If ``n`` is not a valid index type.
        """
        if is_number(n):
            return self._get_value_at_index(n)
        elif is_numeric_vector(n):
            return Vector(self._get_value_at_index(e) for e in n)
        elif isinstance(n, slice):
            return Vector(
                self._get_value_at_index(e) for e in range(n.start, n.stop, n.step or 1)
            )
        else:
            raise TypeError(
                "Cannot evaluate DiscreteTimeFunction at "
                "index %s (type %s)." % (n, type(n).__name__)
            )

    def __call__(self, t):
        """
        Evaluate the function at the specified time.

        Parameters
        ----------
        t : float, vector, or DiscreteTimeFunction
            Time value, vector of time values, or discrete

        """
        if is_number(t):
            return self._get_value_at_time(t)
        elif is_numeric_vector(t):
            return Vector(self._get_value_at_time(e) for e in t)
        elif isinstance(t, DiscreteTimeFunction):
            self.check_same_index_set(t)
            return DiscreteTimeFunction(
                func=lambda n: self(t[n]), index_set=self.index_set
            )
        else:
            raise TypeError(
                "Cannot evaluate DiscreteTimeFunction at "
                "time %s (type %s)." % (t, type(t).__name__)
            )

    def apply(self, func):
        """Compose function with the TimeFunction.

        Args:
          func: function to compose with the TimeFunction

        Example:
          f = DiscreteTimeFunction(lambda t: t, fs=1)
          g = f.apply(log)

        Note: For most standard functions, you can apply the function to
          the TimeFunction directly. For example, in the example above,
          g = log(f) would have been equivalent and more readable.

        User-defined functions can also be applied.

        Example:
          def log_squared(f):
            return log(f) ** 2
          g = f.apply(log_squared)
        """
        return DiscreteTimeFunction(lambda n: func(self[n]), index_set=self.index_set)

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):
        """
        Create an element-wise arithmetic operation.

        Parameters
        ----------
        op : callable
            Binary operation to apply.

        Returns
        -------
        callable
            Generated operation function.
        """

        def _op_func(self, other):
            self.check_same_index_set(other)
            if is_number(other):
                return DiscreteTimeFunction(
                    lambda n: op(self[n], other), index_set=self.index_set
                )
            elif isinstance(other, DiscreteTimeFunction):
                return DiscreteTimeFunction(
                    lambda n: op(self[n], other[n]), index_set=self.index_set
                )
            else:
                return NotImplemented

        return _op_func

    def __str__(self):
        """
        Return a string representation of the infinite tuple.

        Returns
        -------
        str
            String containing the first few values of the tuple.
        """
        first_few = ", ".join(str(self[n]) for n in range(-2, 3))
        return "(..., " + first_few + ", ...)"

    def __repr__(self):
        """
        Return the string representation of the infinite tuple.


        Returns
        -------
        str
        String representation of the infinite tuple.
        """

        return self.__str__()

    def plot(self, tmin=0, tmax=10, **kwargs):
        """
        Plot values over a specified time range.

        Parameters
        ----------
        tmin : int, optional
            Starting time, by default 0.
        tmax : int, optional
            Ending time, by default 10.
        **kwargs
            Additional keyword arguments passed to
            ``matplotlib.pyplot.plot``.
        """
        nmin = int(np.floor(tmin * self.index_set.fs))
        nmax = int(np.ceil(tmax * self.index_set.fs))
        ts = [self.index_set[n] for n in range(nmin, nmax)]
        ys = [self[n] for n in range(nmin, nmax)]
        plt.plot(ts, ys, ".--", **kwargs)


class ContinuousTimeFunction(TimeFunction):
    """
    Function defined over continuous time.

    A ``ContinuousTimeFunction`` maps real-valued times to values.

    Methods
    -------
    apply(func)
        Compose a function with this continuous-time function.
    """

    def __init__(self, func=lambda t: t):
        """Initializes a data structure for a discrete-time function.

        Args:
          func: A function of n that returns the value in position n.
                n is assumed to be any integer (postive or negative).
                This function can be defined at initialization time,
                or later. By default, it is not set at initialization.
        """
        self.index_set = Reals()
        if func is not None:
            self.func = func

    def __call__(self, t):
        """
        Evaluate the function at the specified time.

        Parameters
        ----------
        t : float, vector, or ContinuousTimeFunction
            Time value, vector of time values, or continuous-time function.

        Returns
        -------
        object, Vector, or ContinuousTimeFunction
            Function value, vector of values, or composed function.

        Raises
        ------
        TypeError
            If ``t`` is not a supported input type.
        """

        if is_number(t):
            return self.func(t)
        elif is_numeric_vector(t):
            try:
                # Use vectorized function if it exists
                return Vector(self.vfunc(t))
            except:
                return Vector(self.func(e) for e in t)
        elif isinstance(t, ContinuousTimeFunction):
            return ContinuousTimeFunction(func=lambda s: self(t(s)))
        else:
            raise TypeError(
                "Cannot evaluate ContinuousTimeFunction at "
                "time %s (type %s)." % (t, type(t).__name__)
            )

    def __getitem__(self, t):
        """
        Return the value at the specified time.

        Parameters
        ----------
        t : float
        Time at which to evaluate the function.

        Returns
        -------
        object
        Value at the specified time.
        """
        return self(t)

    def apply(self, func):
        """Compose function with the TimeFunction.

        Args:
          func: function to compose with the TimeFunction


        Example:
          f = ContinuousTimeFunction(lambda t: t)
          g = f.apply(log)

        Note: For most standard functions, you can apply the function to
          the TimeFunction directly. For example, in the example above,
          g = log(f) would have been equivalent and more readable.

        User-defined functions can also be applied.

        Example:
          def log_squared(f):
            return log(f) ** 2
          g = f.apply(log_squared)
        """
        return ContinuousTimeFunction(lambda t: func(self(t)))

    # The Arithmetic superclass will use this to define all of the
    # usual arithmetic operations (e.g., +, -, *, /, **, ^, etc.).
    def _operation_factory(self, op):
        """
        Create an element-wise arithmetic operation.

        Parameters
        ----------
        op : callable
            Binary operation to apply.

        Returns
        -------
        callable
            Generated operation function.
        """

        def _op_func(self, other):
            self.check_same_index_set(other)
            if is_number(other):
                return ContinuousTimeFunction(lambda t: op(self(t), other))
            elif isinstance(other, ContinuousTimeFunction):
                return ContinuousTimeFunction(lambda t: op(self(t), other(t)))
            else:
                return NotImplemented

        return _op_func

    def __str__(self):
        """
        Return a string representation of the infinite tuple.

        Returns
        -------
        str
           String containing the first few values of the tuple.
        """
        return "[continuous-time function]"

    def __repr__(self):
        """
        Return the string representation of the infinite tuple.


        Returns
        -------
        str
        String representation of the infinite tuple.
        """
        return self.__str__()

    def plot(self, tmin=0, tmax=10, **kwargs):
        """
        Plot values over a specified time range.

        Parameters
        ----------
        tmin : int, optional
            Starting time, by default 0.
        tmax : int, optional
            Ending time, by default 10.
        **kwargs
            Additional keyword arguments passed to
            ``matplotlib.pyplot.plot``.
        """
        ts = np.linspace(tmin, tmax, 200)
        ys = [self(t) for t in ts]
        plt.plot(ts, ys, "-", **kwargs)


class DiscreteValued:
    """
    Object with discrete-valued states and arrival information.

    ...

    Methods
    -------
    get_states()
        Return the states associated with the function.
    get_interarrival_times()
        Return the interarrival times associated with the function.
    get_arrival_times()
        Return the arrival times computed from the cumulative sum
        of the interarrival times.
    """

    def get_states(self):
        """
        Return the states of the function.

        Returns
        -------
        object
            States associated with the function.

        Raises
        ------
        NameError
            If states are not defined.
        """
        if not hasattr(self, "states"):
            raise NameError("States not defined for " "function.")
        return self.states

    def get_interarrival_times(self):
        """
        Return the interarrival times of the function.

        Returns
        -------
        object
            Interarrival times associated with the function.

        Raises
        ------
        NameError
            If interarrival times are not defined.
        """
        if not hasattr(self, "interarrival_times"):
            raise NameError("Interarrival times not " "defined for function.")
        return self.interarrival_times

    def get_arrival_times(self):
        """
        Return the arrival times of the function.

        Returns
        -------
        object
            Arrival times obtained from the cumulative sum of the
            interarrival times.

        Raises
        ------
        NameError
            If interarrival times are not defined.
        """
        if not hasattr(self, "interarrival_times"):
            raise NameError("Interarrival times not " "defined for function.")
        return self.interarrival_times.cumsum()


def join(result1, result2):
    """Joins two result objects into a single result object.

    Args:
      result1: The first result.
      result2: The second result.
    """

    a = tuple(result1.values) if type(result1) == Tuple else (result1,)
    b = tuple(result2.values) if type(result2) == Tuple else (result2,)

    return Tuple(a + b)


def concat(*args):
    """Concatenates scalars and vectors into one data structure.

    Args:
      *args: Any number of scalar or vector objects. The last
          argument can be an InfiniteTuple.

    Returns:
      A Vector or an InfiniteTuple, depending on whether the
      last argument is an InfiniteTuple.
    """
    values = []
    for i, arg in enumerate(args):
        if is_scalar(arg):
            values.append(arg)
        elif is_vector(arg):
            values.extend(arg)
        elif isinstance(arg, InfiniteTuple):
            # check that InfiniteTuple is the last arg
            if i == len(args) - 1:
                # define concatenated InfiniteTuple
                def _func(n):
                    if n < len(values):
                        return values[n]
                    else:
                        return arg[n - len(values)]

                return type(arg)(_func)

            raise Exception("InfiniteTuple must be the last " "argument to concat().")
        else:
            raise TypeError(
                "Every argument to concat() must be either "
                "a scalar, a vector, or an InfiniteTuple."
            )

    return Vector(values)


def is_scalar(x):
    """
    Determine whether an object is a scalar value.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
        True if ``x`` is a number or string, otherwise False.
    """
    return isinstance(x, (numbers.Number, str))


def is_vector(x):
    """
    Determine whether an object is vector-like.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
        True if ``x`` has a length, otherwise False.
    """
    return hasattr(x, "__len__")


def is_number(x):
    """
    Determine whether an object is numeric.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
        True if ``x`` is a number, otherwise False.
    """
    return isinstance(x, numbers.Number)


def is_numeric_vector(x):
    """
    Determine whether an object is a vector of numbers.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    bool
        True if ``x`` has a length and all elements are numeric,
        otherwise False.
    """
    return hasattr(x, "__len__") and all(is_number(i) for i in x)
