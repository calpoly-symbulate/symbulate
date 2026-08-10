"""Tests for symbulate.result.

Covers:
- Scalar, Int, Float creation and type dispatch
- Tuple: init, indexing, iteration, hashing, comparison, arithmetic,
  apply, filter, cumsum, __str__/__repr__
- Vector: inherits Tuple, preserves type through operations
- InfiniteTuple: lazy evaluation, caching, slicing, apply, arithmetic
- InfiniteVector: cumsum
- DiscreteTimeFunction: index vs time access, slicing, apply, arithmetic,
  error cases, __str__
- ContinuousTimeFunction: call, getitem, apply, arithmetic, composition,
  __str__
- DiscreteValued: get_states, get_interarrival_times, get_arrival_times
- join: scalars and Tuples
- concat: scalars, vectors, InfiniteTuple tail, error cases
- is_scalar, is_vector, is_number, is_numeric_vector
- plot() on Tuple, Vector, InfiniteVector, DiscreteTimeFunction, and
  ContinuousTimeFunction returns a SymbulatePlot wrapper
"""

import unittest

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt

from symbulate.plot import SymbulatePlot
from symbulate.result import (
    ContinuousTimeFunction,
    DiscreteTimeFunction,
    DiscreteValued,
    Float,
    InfiniteTuple,
    InfiniteVector,
    Int,
    Scalar,
    Tuple,
    Vector,
    concat,
    is_number,
    is_numeric_vector,
    is_scalar,
    is_vector,
    join,
)

# ---------------------------------------------------------------------------
# Scalar / Int / Float
# ---------------------------------------------------------------------------


class TestScalar(unittest.TestCase):

    def test_int_dispatch(self):
        s = Scalar(3)
        self.assertIsInstance(s, Int)
        self.assertEqual(s, 3)

    def test_float_dispatch(self):
        s = Scalar(3.14)
        self.assertIsInstance(s, Float)
        self.assertAlmostEqual(s, 3.14)

    def test_numpy_int_dispatch(self):
        s = Scalar(np.int64(7))
        self.assertIsInstance(s, Int)
        self.assertEqual(s, 7)

    def test_numpy_float_dispatch(self):
        s = Scalar(np.float64(2.71))
        self.assertIsInstance(s, Float)
        self.assertAlmostEqual(s, 2.71)

    def test_bad_type_raises(self):
        with self.assertRaises(Exception):
            Scalar("hello")

    def test_int_arithmetic(self):
        self.assertEqual(Int(4) + Int(5), 9)
        self.assertEqual(Int(10) - Int(3), 7)

    def test_float_arithmetic(self):
        self.assertAlmostEqual(Float(1.5) + Float(2.5), 4.0)


# ---------------------------------------------------------------------------
# Tuple
# ---------------------------------------------------------------------------


class TestTuple(unittest.TestCase):

    def test_init_from_list(self):
        t = Tuple([1, 2, 3])
        self.assertEqual(len(t), 3)

    def test_init_from_scalar(self):
        t = Tuple(5)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0], 5)

    def test_init_from_generator(self):
        t = Tuple(x for x in range(4))
        self.assertEqual(tuple(t), (0, 1, 2, 3))

    def test_init_bad_type_raises(self):
        with self.assertRaises(Exception):
            Tuple(object())

    def test_getitem_int(self):
        t = Tuple([10, 20, 30])
        self.assertEqual(t[1], 20)

    def test_getitem_numeric_vector(self):
        t = Tuple([10, 20, 30])
        sub = t[[0, 2]]
        self.assertIsInstance(sub, Tuple)
        self.assertEqual(tuple(sub), (10, 30))

    def test_len(self):
        self.assertEqual(len(Tuple([1, 2, 3, 4])), 4)

    def test_iter(self):
        t = Tuple([5, 6, 7])
        self.assertEqual(list(t), [5, 6, 7])

    def test_hash(self):
        t = Tuple([1, 2, 3])
        self.assertIsInstance(hash(t), int)

    def test_eq_same(self):
        self.assertEqual(Tuple([1, 2]), Tuple([1, 2]))

    def test_eq_different_length(self):
        self.assertNotEqual(Tuple([1, 2]), Tuple([1, 2, 3]))

    def test_eq_non_sequence(self):
        self.assertNotEqual(Tuple([1, 2]), 42)

    def test_lt(self):
        self.assertLess(Tuple([1, 2]), Tuple([1, 3]))

    def test_lt_returns_not_implemented_for_non_tuple(self):
        result = Tuple([1, 2]).__lt__(5)
        self.assertEqual(result, NotImplemented)

    def test_apply(self):
        t = Tuple([1, 4, 9])
        result = t.apply(lambda x: x**0.5)
        self.assertAlmostEqual(result[2], 3.0)

    def test_filter(self):
        t = Tuple([1, 2, 3, 4, 5])
        filtered = t.filter(lambda x: x % 2 == 0)
        self.assertEqual(tuple(filtered), (2, 4))

    def test_cumsum(self):
        t = Tuple([1, 2, 3, 4])
        cs = t.cumsum()
        self.assertEqual(list(cs), [1, 3, 6, 10])

    def test_add_scalar(self):
        t = Tuple([1, 2, 3])
        result = t + 10
        self.assertEqual(list(result), [11, 12, 13])

    def test_add_tuple(self):
        t1 = Tuple([1, 2, 3])
        t2 = Tuple([4, 5, 6])
        result = t1 + t2
        self.assertEqual(list(result), [5, 7, 9])

    def test_add_mismatched_length_raises(self):
        with self.assertRaises(Exception):
            Tuple([1, 2]) + Tuple([1, 2, 3])

    def test_mul_scalar(self):
        t = Tuple([2, 3, 4])
        result = t * 3
        self.assertEqual(list(result), [6, 9, 12])

    def test_str_short(self):
        t = Tuple([1, 2, 3])
        self.assertEqual(str(t), "(1, 2, 3)")

    def test_str_long(self):
        t = Tuple(range(10))
        s = str(t)
        self.assertIn("...", s)
        self.assertTrue(s.startswith("(0, 1, 2, 3, 4, ..."))

    def test_repr(self):
        t = Tuple([1, 2])
        self.assertEqual(repr(t), str(t))


# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------


class TestVector(unittest.TestCase):

    def test_is_tuple_subclass(self):
        v = Vector([1, 2, 3])
        self.assertIsInstance(v, Tuple)

    def test_arithmetic_returns_vector(self):
        v = Vector([1, 2, 3])
        result = v + 1
        self.assertIsInstance(result, Vector)

    def test_apply_returns_vector(self):
        v = Vector([1, 4, 9])
        result = v.apply(lambda x: x * 2)
        self.assertIsInstance(result, Vector)


# ---------------------------------------------------------------------------
# InfiniteTuple
# ---------------------------------------------------------------------------


class TestInfiniteTuple(unittest.TestCase):

    def test_default_identity(self):
        iv = InfiniteTuple()
        self.assertEqual(iv[5], 5)

    def test_custom_func(self):
        iv = InfiniteTuple(lambda n: n**2)
        self.assertEqual(iv[3], 9)

    def test_caching(self):
        calls = []

        def f(n):
            calls.append(n)
            return n

        iv = InfiniteTuple(f)
        iv[3]
        iv[3]
        self.assertEqual(calls.count(3), 1)

    def test_call_alias(self):
        iv = InfiniteTuple(lambda n: n + 10)
        self.assertEqual(iv(4), iv[4])

    def test_slice_with_stop(self):
        iv = InfiniteTuple(lambda n: n * 2)
        sl = iv[0:4]
        self.assertEqual(sl, [0, 2, 4, 6])

    def test_slice_no_stop_returns_shifted(self):
        iv = InfiniteTuple(lambda n: n)
        shifted = iv[2:]
        self.assertEqual(shifted[0], 2)
        self.assertEqual(shifted[3], 5)

    def test_slice_no_stop_honors_step(self):
        # Regression test: an open-ended slice with a start used to
        # silently drop the step entirely (iv[2::2] behaved like iv[2:]).
        iv = InfiniteTuple(lambda n: n)
        stepped = iv[2::2]
        self.assertEqual([stepped[i] for i in range(5)], [2, 4, 6, 8, 10])

    def test_slice_no_start_honors_step(self):
        # Same bug, sibling branch: with no start (iv[::2]) the old code
        # returned self unchanged, silently ignoring the step too.
        iv = InfiniteTuple(lambda n: n)
        stepped = iv[::2]
        self.assertEqual([stepped[i] for i in range(5)], [0, 2, 4, 6, 8])

    def test_full_slice_still_returns_self(self):
        iv = InfiniteTuple(lambda n: n)
        self.assertIs(iv[:], iv)

    def test_str_contains_ellipsis(self):
        iv = InfiniteTuple(lambda n: n)
        self.assertIn("...", str(iv))

    def test_repr(self):
        iv = InfiniteTuple(lambda n: n)
        self.assertEqual(repr(iv), str(iv))

    def test_apply(self):
        iv = InfiniteTuple(lambda n: n + 1)
        doubled = iv.apply(lambda x: x * 2)
        self.assertEqual(doubled[3], 8)

    def test_add_scalar(self):
        iv = InfiniteTuple(lambda n: n)
        result = iv + 5
        self.assertEqual(result[2], 7)

    def test_add_infinite_tuple(self):
        iv1 = InfiniteTuple(lambda n: n)
        iv2 = InfiniteTuple(lambda n: n * 2)
        result = iv1 + iv2
        self.assertEqual(result[3], 9)


# ---------------------------------------------------------------------------
# InfiniteVector
# ---------------------------------------------------------------------------


class TestInfiniteVector(unittest.TestCase):

    def test_cumsum_ones(self):
        iv = InfiniteVector(lambda _: 1)
        cs = iv.cumsum()
        self.assertEqual(cs[0], 1)
        self.assertEqual(cs[4], 5)

    def test_cumsum_natural_numbers(self):
        iv = InfiniteVector(lambda n: n)
        cs = iv.cumsum()
        self.assertEqual(cs[3], 0 + 1 + 2 + 3)


# ---------------------------------------------------------------------------
# DiscreteTimeFunction
# ---------------------------------------------------------------------------


class TestDiscreteTimeFunction(unittest.TestCase):

    def test_init_default(self):
        f = DiscreteTimeFunction()
        self.assertEqual(f[0], 0.0)

    def test_init_custom_func(self):
        f = DiscreteTimeFunction(lambda n: n**2)
        self.assertEqual(f[3], 9)

    def test_positive_index(self):
        f = DiscreteTimeFunction(lambda n: n * 3)
        self.assertEqual(f[2], 6)

    def test_negative_index(self):
        f = DiscreteTimeFunction(lambda n: n * 3)
        self.assertEqual(f[-1], -3)

    def test_non_integer_index_raises(self):
        f = DiscreteTimeFunction(lambda n: n)
        with self.assertRaises(KeyError):
            f[1.5]

    def test_call_at_valid_time(self):
        f = DiscreteTimeFunction(lambda n: n**2, fs=2)
        self.assertEqual(f(0.0), 0)
        self.assertEqual(f(0.5), 1)

    def test_call_at_invalid_time_raises(self):
        f = DiscreteTimeFunction(lambda n: n, fs=1)
        with self.assertRaises(KeyError):
            f(0.3)

    def test_call_vector_of_times(self):
        f = DiscreteTimeFunction(lambda n: n, fs=1)
        result = f([0.0, 1.0, 2.0])
        self.assertIsInstance(result, Vector)
        self.assertEqual(list(result), [0, 1, 2])

    def test_getitem_slice(self):
        f = DiscreteTimeFunction(lambda n: n * 3, fs=1)
        result = f[0:4]
        self.assertIsInstance(result, Vector)
        self.assertEqual(list(result), [0, 3, 6, 9])

    def test_getitem_numeric_vector(self):
        f = DiscreteTimeFunction(lambda n: n, fs=1)
        result = f[[0, 1, 2]]
        self.assertIsInstance(result, Vector)
        self.assertEqual(list(result), [0, 1, 2])

    def test_getitem_open_ended_slice_raises_clear_value_error(self):
        # Regression test: f[2:] used to crash with a raw
        # "TypeError: 'NoneType' object cannot be interpreted as an
        # integer" from range(n.start or 0, n.stop, ...) since there is
        # no length to infer a stop from.
        f = DiscreteTimeFunction(lambda n: n * 3, fs=1)
        with self.assertRaisesRegex(ValueError, "without a stop"):
            f[2:]

    def test_getitem_bad_type_raises(self):
        f = DiscreteTimeFunction(lambda n: n)
        with self.assertRaises(TypeError):
            f["bad"]

    def test_apply(self):
        f = DiscreteTimeFunction(lambda n: n + 1, fs=1)
        g = f.apply(lambda x: x**2)
        self.assertEqual(g[0], 1)
        self.assertEqual(g[3], 16)

    def test_add_scalar(self):
        f = DiscreteTimeFunction(lambda n: n, fs=1)
        g = f + 10
        self.assertEqual(g[2], 12)

    def test_add_dtf(self):
        f = DiscreteTimeFunction(lambda n: n, fs=1)
        g = DiscreteTimeFunction(lambda n: n * 2, fs=1)
        h = f + g
        self.assertEqual(h[3], 9)

    def test_str(self):
        f = DiscreteTimeFunction(lambda n: n, fs=1)
        s = str(f)
        self.assertIn("...", s)

    def test_repr(self):
        f = DiscreteTimeFunction(lambda n: n)
        self.assertEqual(repr(f), str(f))

    def test_caching_positive(self):
        calls = []

        def f(n):
            calls.append(n)
            return n

        dtf = DiscreteTimeFunction(f, fs=1)
        dtf[5]
        dtf[5]
        self.assertEqual(calls.count(5), 1)

    def test_caching_negative(self):
        calls = []

        def f(n):
            calls.append(n)
            return n

        dtf = DiscreteTimeFunction(f, fs=1)
        dtf[-3]
        dtf[-3]
        self.assertEqual(calls.count(-3), 1)


# ---------------------------------------------------------------------------
# ContinuousTimeFunction
# ---------------------------------------------------------------------------


class TestContinuousTimeFunction(unittest.TestCase):

    def test_init_default_identity(self):
        f = ContinuousTimeFunction()
        self.assertEqual(f(3.0), 3.0)

    def test_call_scalar(self):
        f = ContinuousTimeFunction(lambda t: t**2)
        self.assertAlmostEqual(f(3.0), 9.0)

    def test_getitem_alias(self):
        f = ContinuousTimeFunction(lambda t: t * 2)
        self.assertAlmostEqual(f[5.0], f(5.0))

    def test_call_vector(self):
        f = ContinuousTimeFunction(lambda t: t**2)
        result = f([1.0, 2.0, 3.0])
        self.assertIsInstance(result, Vector)
        self.assertEqual(list(result), [1.0, 4.0, 9.0])

    def test_call_ctf_composition(self):
        f = ContinuousTimeFunction(lambda t: t**2)
        g = ContinuousTimeFunction(lambda t: t + 1)
        h = f(g)
        self.assertIsInstance(h, ContinuousTimeFunction)
        self.assertAlmostEqual(h(2.0), (2.0 + 1) ** 2)

    def test_call_bad_type_raises(self):
        f = ContinuousTimeFunction(lambda t: t)
        with self.assertRaises(TypeError):
            f("bad")

    def test_apply(self):
        f = ContinuousTimeFunction(lambda t: t)
        g = f.apply(lambda x: x**2)
        self.assertAlmostEqual(g(4.0), 16.0)

    def test_add_scalar(self):
        f = ContinuousTimeFunction(lambda t: t)
        g = f + 5
        self.assertAlmostEqual(g(3.0), 8.0)

    def test_add_ctf(self):
        f = ContinuousTimeFunction(lambda t: t)
        g = ContinuousTimeFunction(lambda t: t * 2)
        h = f + g
        self.assertAlmostEqual(h(4.0), 12.0)

    def test_str(self):
        f = ContinuousTimeFunction(lambda t: t)
        self.assertEqual(str(f), "[continuous-time function]")

    def test_repr(self):
        f = ContinuousTimeFunction(lambda t: t)
        self.assertEqual(repr(f), str(f))


# ---------------------------------------------------------------------------
# DiscreteValued
# ---------------------------------------------------------------------------


class TestDiscreteValued(unittest.TestCase):

    def _make_obj_with_states(self):
        obj = DiscreteValued()
        obj.states = InfiniteVector(lambda n: n % 2)
        return obj

    def _make_obj_with_interarrivals(self):
        obj = DiscreteValued()
        obj.interarrival_times = InfiniteVector(lambda _: 1)
        return obj

    def test_get_states(self):
        obj = self._make_obj_with_states()
        states = obj.get_states()
        self.assertEqual(states[0], 0)

    def test_get_states_raises_if_missing(self):
        obj = DiscreteValued()
        with self.assertRaises(AttributeError):
            obj.get_states()

    def test_get_interarrival_times(self):
        obj = self._make_obj_with_interarrivals()
        times = obj.get_interarrival_times()
        self.assertEqual(times[0], 1)

    def test_get_interarrival_times_raises_if_missing(self):
        obj = DiscreteValued()
        with self.assertRaises(AttributeError):
            obj.get_interarrival_times()

    def test_get_arrival_times(self):
        obj = self._make_obj_with_interarrivals()
        arrivals = obj.get_arrival_times()
        self.assertEqual(arrivals[0], 1)
        self.assertEqual(arrivals[3], 4)

    def test_get_arrival_times_raises_if_missing(self):
        obj = DiscreteValued()
        with self.assertRaises(AttributeError):
            obj.get_arrival_times()


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


class TestJoin(unittest.TestCase):

    def test_join_two_scalars(self):
        result = join(Scalar(1), Scalar(2))
        self.assertIsInstance(result, Tuple)
        self.assertEqual(tuple(result), (1, 2))

    def test_join_two_tuples(self):
        t1 = Tuple([1, 2])
        t2 = Tuple([3, 4])
        result = join(t1, t2)
        self.assertEqual(tuple(result), (1, 2, 3, 4))

    def test_join_scalar_and_tuple(self):
        result = join(Scalar(0), Tuple([1, 2]))
        self.assertEqual(tuple(result), (0, 1, 2))

    def test_join_tuple_and_scalar(self):
        result = join(Tuple([1, 2]), Scalar(3))
        self.assertEqual(tuple(result), (1, 2, 3))


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------


class TestConcat(unittest.TestCase):

    def test_concat_scalars(self):
        result = concat(1, 2, 3)
        self.assertIsInstance(result, Vector)
        self.assertEqual(list(result), [1, 2, 3])

    def test_concat_vectors(self):
        v = Vector([4, 5])
        result = concat(1, v, 3)
        self.assertEqual(list(result), [1, 4, 5, 3])

    def test_concat_with_infinite_tuple_tail(self):
        iv = InfiniteVector(lambda n: n)
        result = concat(10, 20, iv)
        self.assertEqual(result[0], 10)
        self.assertEqual(result[1], 20)
        self.assertEqual(result[2], 0)
        self.assertEqual(result[3], 1)

    def test_concat_infinite_not_last_raises(self):
        iv = InfiniteTuple(lambda n: n)
        with self.assertRaises(Exception):
            concat(iv, 5)

    def test_concat_bad_type_raises(self):
        with self.assertRaises(TypeError):
            concat(object())

    def test_concat_empty(self):
        result = concat()
        self.assertIsInstance(result, Vector)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestIsScalar(unittest.TestCase):

    def test_int(self):
        self.assertTrue(is_scalar(3))

    def test_float(self):
        self.assertTrue(is_scalar(3.14))

    def test_string(self):
        self.assertTrue(is_scalar("hello"))

    def test_list(self):
        self.assertFalse(is_scalar([1, 2, 3]))

    def test_tuple_type(self):
        self.assertFalse(is_scalar((1, 2)))


class TestIsVector(unittest.TestCase):

    def test_list(self):
        self.assertTrue(is_vector([1, 2, 3]))

    def test_symbulate_tuple(self):
        self.assertTrue(is_vector(Tuple([1, 2])))

    def test_int(self):
        self.assertFalse(is_vector(5))

    def test_string(self):
        self.assertTrue(is_vector("abc"))


class TestIsNumber(unittest.TestCase):

    def test_int(self):
        self.assertTrue(is_number(3))

    def test_float(self):
        self.assertTrue(is_number(3.14))

    def test_string(self):
        self.assertFalse(is_number("3"))

    def test_list(self):
        self.assertFalse(is_number([1]))

    def test_numpy_float(self):
        self.assertTrue(is_number(np.float64(1.5)))


class TestIsNumericVector(unittest.TestCase):

    def test_all_numbers(self):
        self.assertTrue(is_numeric_vector([1, 2, 3]))

    def test_mixed(self):
        self.assertFalse(is_numeric_vector([1, "a", 3]))

    def test_scalar(self):
        self.assertFalse(is_numeric_vector(5))

    def test_empty_list(self):
        self.assertTrue(is_numeric_vector([]))

    def test_numpy_array(self):
        self.assertTrue(is_numeric_vector(np.array([1.0, 2.0])))


# ---------------------------------------------------------------------------
# Error message content
# ---------------------------------------------------------------------------


class TestScalarErrorMessages(unittest.TestCase):

    def test_bad_type_message_includes_type_name(self):
        with self.assertRaisesRegex(Exception, "str"):
            Scalar("hello")

    def test_bad_type_message_includes_value(self):
        with self.assertRaisesRegex(Exception, "hello"):
            Scalar("hello")


class TestTimeFunctionIndexSetErrorMessages(unittest.TestCase):

    def test_mismatched_index_set_message_includes_both_set_types(self):
        dtf = DiscreteTimeFunction(lambda n: n)
        ctf = ContinuousTimeFunction(lambda t: t)
        with self.assertRaisesRegex(Exception, "DiscreteTimeSequence"):
            dtf + ctf

    def test_mismatched_index_set_message_includes_other_set_type(self):
        dtf = DiscreteTimeFunction(lambda n: n)
        ctf = ContinuousTimeFunction(lambda t: t)
        with self.assertRaisesRegex(Exception, "Reals"):
            dtf + ctf


class TestDiscreteTimeFunctionErrorMessages(unittest.TestCase):

    def test_non_integer_index_message_suggests_call_syntax(self):
        f = DiscreteTimeFunction(lambda n: n)
        with self.assertRaisesRegex(KeyError, r"f\(t\)"):
            f[1.5]

    def test_getitem_bad_type_message_lists_valid_types(self):
        f = DiscreteTimeFunction(lambda n: n)
        with self.assertRaisesRegex(TypeError, "int, numeric vector, or slice"):
            f["bad"]

    def test_getitem_bad_type_message_includes_actual_type(self):
        f = DiscreteTimeFunction(lambda n: n)
        with self.assertRaisesRegex(TypeError, "str"):
            f["bad"]

    def test_call_bad_type_message_lists_valid_types(self):
        f = DiscreteTimeFunction(lambda n: n)
        with self.assertRaisesRegex(
            TypeError, "number, numeric vector, or DiscreteTimeFunction"
        ):
            f("bad")

    def test_call_bad_type_message_includes_actual_type(self):
        f = DiscreteTimeFunction(lambda n: n)
        with self.assertRaisesRegex(TypeError, "str"):
            f("bad")


class TestContinuousTimeFunctionErrorMessages(unittest.TestCase):

    def test_call_bad_type_message_lists_valid_types(self):
        f = ContinuousTimeFunction(lambda t: t)
        with self.assertRaisesRegex(
            TypeError, "number, numeric vector, or ContinuousTimeFunction"
        ):
            f("bad")

    def test_call_bad_type_message_includes_actual_type(self):
        f = ContinuousTimeFunction(lambda t: t)
        with self.assertRaisesRegex(TypeError, "str"):
            f("bad")


class TestDiscreteValuedErrorMessages(unittest.TestCase):

    def test_get_states_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            DiscreteValued().get_states()

    def test_get_states_message(self):
        with self.assertRaisesRegex(AttributeError, "States not defined"):
            DiscreteValued().get_states()

    def test_get_interarrival_times_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            DiscreteValued().get_interarrival_times()

    def test_get_interarrival_times_message(self):
        with self.assertRaisesRegex(AttributeError, "Interarrival times not defined"):
            DiscreteValued().get_interarrival_times()

    def test_get_arrival_times_raises_attribute_error(self):
        with self.assertRaises(AttributeError):
            DiscreteValued().get_arrival_times()

    def test_get_arrival_times_message(self):
        with self.assertRaisesRegex(AttributeError, "Interarrival times not defined"):
            DiscreteValued().get_arrival_times()


class TestConcatErrorMessages(unittest.TestCase):

    def test_bad_type_message_includes_argument_index_zero(self):
        with self.assertRaisesRegex(TypeError, "Argument 0"):
            concat(object())

    def test_bad_type_message_includes_nonzero_argument_index(self):
        with self.assertRaisesRegex(TypeError, "Argument 2"):
            concat(1, 2, object())

    def test_bad_type_message_includes_type_name(self):
        with self.assertRaisesRegex(TypeError, "object"):
            concat(object())


# ---------------------------------------------------------------------------
# plot() methods return SymbulatePlot
# ---------------------------------------------------------------------------


class TestResultPlotsReturnWrapper(unittest.TestCase):
    """Every result-type plot() method returns a SymbulatePlot wrapper.

    The wrapper's repr is empty so Jupyter prints nothing below the
    plot, and it exposes the matplotlib axes as .ax.
    """

    def tearDown(self):
        plt.close("all")

    def test_tuple_plot_returns_wrapper(self):
        p = Tuple([1, 4, 2, 8, 5]).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_vector_plot_returns_wrapper(self):
        p = Vector([1.0, 2.0, 3.0]).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_infinite_vector_plot_returns_wrapper(self):
        iv = InfiniteVector(lambda n: n**2)
        p = iv.plot(tmin=0, tmax=5)
        self.assertIsInstance(p, SymbulatePlot)

    def test_discrete_time_function_plot_returns_wrapper(self):
        f = DiscreteTimeFunction(lambda n: n, fs=1)
        p = f.plot(tmin=0, tmax=5)
        self.assertIsInstance(p, SymbulatePlot)

    def test_continuous_time_function_plot_returns_wrapper(self):
        f = ContinuousTimeFunction(lambda t: np.sin(t))
        p = f.plot(tmin=0, tmax=6)
        self.assertIsInstance(p, SymbulatePlot)

    def test_wrapper_repr_is_empty(self):
        p = Tuple([1, 2, 3]).plot()
        self.assertEqual(repr(p), "")

    def test_wrapper_exposes_axes(self):
        p = Tuple([1, 2, 3]).plot()
        self.assertIs(p.ax, plt.gca())


# ---------------------------------------------------------------------------
# plot() methods draw sample paths (make_sample_path)
# ---------------------------------------------------------------------------


class TestResultSamplePathPlots(unittest.TestCase):
    """Result-type plot() methods draw sample paths via make_sample_path.

    The old ".--" dot-dash format is replaced by a plain solid line;
    index-based results label the x-axis "Index", time-based ones
    "Time"; overlaid paths get distinct colors and an automatic
    "Path 1", "Path 2", ... legend.
    """

    def tearDown(self):
        plt.close("all")

    def test_tuple_plot_is_solid_line_without_markers(self):
        Tuple([1, 4, 2, 8, 5]).plot()
        (line,) = plt.gca().lines
        self.assertEqual(line.get_linestyle(), "-")
        self.assertEqual(line.get_marker(), "None")

    def test_tuple_plot_labels_index_axis(self):
        Tuple([1, 4, 2, 8, 5]).plot()
        ax = plt.gca()
        self.assertEqual(ax.get_xlabel(), "Index")
        self.assertEqual(ax.get_ylabel(), "Value")
        self.assertEqual(ax.get_title(), "Sample Path")

    def test_infinite_vector_plot_labels_index_axis(self):
        InfiniteVector(lambda n: n**2).plot(tmin=0, tmax=5)
        self.assertEqual(plt.gca().get_xlabel(), "Index")

    def test_discrete_time_function_plot_labels_time_axis(self):
        DiscreteTimeFunction(lambda n: n, fs=1).plot(tmin=0, tmax=5)
        self.assertEqual(plt.gca().get_xlabel(), "Time")

    def test_continuous_time_function_plot_labels_time_axis(self):
        ContinuousTimeFunction(lambda t: np.sin(t)).plot(tmin=0, tmax=6)
        self.assertEqual(plt.gca().get_xlabel(), "Time")

    def test_overlaid_tuples_get_distinct_colors_and_legend(self):
        Tuple([1, 4, 2, 8, 5]).plot()
        Tuple([2, 2, 6, 3, 9]).plot()
        ax = plt.gca()
        self.assertEqual(len({line.get_color() for line in ax.lines}), 2)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertEqual(labels, ["Path 1", "Path 2"])

    def test_lone_path_has_no_legend(self):
        Tuple([1, 4, 2, 8, 5]).plot()
        self.assertIsNone(plt.gca().get_legend())

    def test_explicit_color_kwarg_is_honored(self):
        Tuple([1, 4, 2, 8, 5]).plot(color="black")
        (line,) = plt.gca().lines
        self.assertEqual(line.get_color(), "black")

    def test_label_kwarg_names_the_path(self):
        Tuple([1, 4, 2, 8, 5]).plot(label="First walk")
        Tuple([2, 2, 6, 3, 9]).plot()
        labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
        self.assertEqual(labels, ["First walk", "Path 2"])


if __name__ == "__main__":
    unittest.main()
