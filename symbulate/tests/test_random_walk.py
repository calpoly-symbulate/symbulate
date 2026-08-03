"""Tests for symbulate.random_walk.

Covers the sample-path result object (starting point, step sizes, lazy
caching), the probability space, the RandomWalk process itself, and the
two theoretical facts the walk is usually taught with: the mean grows like
``n * (2p - 1)`` and, for the fair walk, the variance grows like ``n``.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_poisson_process.py.
"""

import unittest

import numpy as np

from symbulate import *
from symbulate import distributions
from symbulate.random_walk import (
    RandomWalk,
    RandomWalkResult,
    RandomWalkProbabilitySpace,
)
from symbulate.result import InfiniteVector

Nsim = 10000


def seed(value=42):
    """Reseed the generator that distribution draws route through."""
    distributions.rng = np.random.default_rng(value)


class TestRandomWalkResult(unittest.TestCase):

    def test_is_infinite_vector(self):
        seed()
        self.assertIsInstance(RandomWalk(p=0.5).draw(), InfiniteVector)

    def test_starts_at_initial(self):
        seed()
        self.assertEqual(float(RandomWalk(p=0.5).draw()[0]), 0.0)
        self.assertEqual(float(RandomWalk(p=0.5, initial=7).draw()[0]), 7.0)

    def test_simple_walk_steps_are_plus_or_minus_one(self):
        seed()
        path = RandomWalk(p=0.5).draw()
        for n in range(50):
            self.assertEqual(abs(float(path[n + 1]) - float(path[n])), 1.0)

    def test_simple_walk_visits_integers(self):
        seed()
        path = RandomWalk(p=0.5).draw()
        for n in range(20):
            self.assertEqual(float(path[n]), int(path[n]))

    def test_path_is_cached_and_stable(self):
        # Reading the same position twice must give the same answer -- a path
        # is one walk, not a fresh one on every lookup.
        seed()
        path = RandomWalk(p=0.5).draw()
        first = [float(path[n]) for n in range(15)]
        again = [float(path[n]) for n in range(15)]
        self.assertEqual(first, again)

    def test_reading_far_ahead_keeps_earlier_values(self):
        # Jumping ahead extends the same path; the earlier positions must not
        # be regenerated behind our back.
        seed()
        path = RandomWalk(p=0.5).draw()
        early = [float(path[n]) for n in range(10)]
        path[500]
        self.assertEqual([float(path[n]) for n in range(10)], early)

    def test_position_is_initial_plus_cumulative_steps(self):
        # The defining property: X[n] is the running total of the steps.
        seed()
        path = RandomWalk(p=0.5, initial=3).draw()
        steps = path.get_steps()
        for n in range(1, 25):
            expected = 3 + sum(float(steps[i]) for i in range(n))
            self.assertAlmostEqual(float(path[n]), expected)

    def test_get_steps_returns_the_steps(self):
        seed()
        path = RandomWalk(p=0.5).draw()
        self.assertIsInstance(path.get_steps(), InfiniteVector)
        self.assertEqual(abs(float(path.get_steps()[0])), 1.0)

    def test_result_constructed_directly_from_steps(self):
        # The result object accumulates whatever steps it is handed, with no
        # assumption about where they came from.
        steps = InfiniteVector(lambda n: 2)
        path = RandomWalkResult(steps, initial=1)
        self.assertEqual([float(path[n]) for n in range(4)], [1.0, 3.0, 5.0, 7.0])


class TestRandomWalkProbabilitySpace(unittest.TestCase):

    def test_stores_parameters(self):
        space = RandomWalkProbabilitySpace(p=0.3, initial=2)
        self.assertEqual(space.p, 0.3)
        self.assertEqual(space.initial, 2)

    def test_p_builds_a_step_random_variable(self):
        # p is shorthand: the space turns it into a +/-1 random variable.
        seed()
        space = RandomWalkProbabilitySpace(p=0.5)
        self.assertIsInstance(space.step_dist, RV)
        self.assertIn(abs(float(space.step_dist.draw())), [1.0])

    def test_step_dist_stored_as_given(self):
        dist = Normal(0, 1)
        self.assertIs(RandomWalkProbabilitySpace(step_dist=dist).step_dist, dist)

    def test_draw_returns_result(self):
        seed()
        self.assertIsInstance(
            RandomWalkProbabilitySpace(p=0.5).draw(), RandomWalkResult
        )


class TestRandomWalk(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(RandomWalk(p=0.5), RV)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(RandomWalk(p=0.5)[3], RV)

    def test_stores_parameters(self):
        X = RandomWalk(p=0.4, initial=5)
        self.assertEqual(X.p, 0.4)
        self.assertEqual(X.initial, 5)
        self.assertIsNone(RandomWalk(step_dist=Normal(0, 1)).p)

    def test_reproducible_under_same_seed(self):
        seed(7)
        first = [float(v) for v in RandomWalk(p=0.5).draw()[:20]]
        seed(7)
        second = [float(v) for v in RandomWalk(p=0.5).draw()[:20]]
        self.assertEqual(first, second)

    def test_sim_gives_independent_paths(self):
        # Two paths from the same process should not be identical.
        seed()
        X = RandomWalk(p=0.5)
        a = [float(X.draw()[n]) for n in range(30)]
        b = [float(X.draw()[n]) for n in range(30)]
        self.assertNotEqual(a, b)

    def test_accepts_a_distribution_step(self):
        seed()
        path = RandomWalk(step_dist=Poisson(2)).draw()
        # Poisson steps are nonnegative, so the walk never decreases.
        for n in range(20):
            self.assertGreaterEqual(float(path[n + 1]), float(path[n]))

    def test_accepts_an_rv_step(self):
        # Arithmetic on a distribution produces an RV, which must also work
        # as a step -- this is the only way to build a +/-1 step by hand.
        seed()
        path = RandomWalk(step_dist=RV(Bernoulli(0.5)) * 2 - 1).draw()
        for n in range(20):
            self.assertEqual(abs(float(path[n + 1]) - float(path[n])), 1.0)


class TestRandomWalkInitialNaming(unittest.TestCase):
    """`initial` is the name; `initial_value` still works.

    Every process in the package is standardising on `initial` for its
    starting condition (MODEL-DECISIONS.md). The older spelling is kept as
    an accepted alias so existing code and notebooks do not break.
    """

    def test_initial_value_alias_still_accepted(self):
        seed()
        self.assertEqual(float(RandomWalk(p=0.5, initial_value=7).draw()[0]), 7.0)

    def test_both_names_agree(self):
        self.assertEqual(
            RandomWalk(p=0.5, initial=4).initial,
            RandomWalk(p=0.5, initial_value=4).initial,
        )

    def test_initial_value_attribute_still_readable(self):
        X = RandomWalk(p=0.5, initial=6)
        self.assertEqual(X.initial_value, 6)
        self.assertEqual(X.prob_space.initial_value, 6)

    def test_defaults_to_zero_when_neither_given(self):
        self.assertEqual(RandomWalk(p=0.5).initial, 0)

    def test_giving_both_names_raises_value_error(self):
        self.assertRaisesRegex(
            ValueError,
            "not both",
            lambda: RandomWalk(p=0.5, initial=1, initial_value=2),
        )


class TestRandomWalkTheory(unittest.TestCase):
    """The two facts the random walk is normally introduced to demonstrate."""

    def test_fair_walk_has_mean_zero(self):
        seed()
        values = RandomWalk(p=0.5)[25].sim(Nsim)
        self.assertAlmostEqual(float(values.mean()), 0.0, delta=0.4)

    def test_fair_walk_variance_grows_like_n(self):
        # Var(X[n]) = n for the fair +/-1 walk. This is the square-root-of-n
        # spread that makes the walk the standard CLT picture.
        seed()
        for n in [10, 40]:
            values = RandomWalk(p=0.5)[n].sim(Nsim)
            self.assertAlmostEqual(float(values.var()) / n, 1.0, delta=0.1)

    def test_biased_walk_mean_is_n_times_drift(self):
        # E[X[n]] = n * (2p - 1).
        seed()
        p, n = 0.7, 20
        values = RandomWalk(p=p)[n].sim(Nsim)
        self.assertAlmostEqual(float(values.mean()), n * (2 * p - 1), delta=0.4)

    def test_initial_shifts_the_mean(self):
        seed()
        values = RandomWalk(p=0.5, initial=10)[16].sim(Nsim)
        self.assertAlmostEqual(float(values.mean()), 10.0, delta=0.4)

    def test_normal_steps_give_variance_n(self):
        # A walk with Normal(0, 1) steps has Var(X[n]) = n as well.
        seed()
        values = RandomWalk(step_dist=Normal(0, 1))[25].sim(Nsim)
        self.assertAlmostEqual(float(values.var()) / 25, 1.0, delta=0.1)

    def test_gamblers_ruin_probability(self):
        # Starting at 10 between absorbing barriers at 0 and 20, a fair walk
        # reaches 20 first half the time.
        seed()

        def reached_top(path, low=0, high=20):
            n = 0
            while low < path[n] < high:
                n += 1
            return 1 if path[n] >= high else 0

        wins = RandomWalk(p=0.5, initial=10).apply(reached_top).sim(2000)
        self.assertAlmostEqual(float(wins.mean()), 0.5, delta=0.05)


class TestRandomWalkErrors(unittest.TestCase):

    def test_no_step_description_raises_value_error(self):
        self.assertRaisesRegex(ValueError, "either p", lambda: RandomWalk())

    def test_both_p_and_step_dist_raises_value_error(self):
        self.assertRaisesRegex(
            ValueError,
            "not both",
            lambda: RandomWalk(step_dist=Normal(0, 1), p=0.5),
        )

    def test_non_numeric_p_raises_type_error(self):
        self.assertRaisesRegex(
            TypeError, "p must be a number", lambda: RandomWalk(p="x")
        )

    def test_p_out_of_range_raises_value_error(self):
        for bad in [-0.1, 1.5]:
            self.assertRaisesRegex(
                ValueError, "between 0 and 1", lambda v=bad: RandomWalk(p=v)
            )

    def test_bad_step_dist_raises_type_error(self):
        # None is deliberately absent: it means "not given", which is the
        # ValueError case covered by test_no_step_description_raises_value_error.
        for bad in [5, "normal", [0, 1]]:
            self.assertRaisesRegex(
                TypeError, "step_dist must be", lambda v=bad: RandomWalk(step_dist=v)
            )

    def test_non_numeric_initial_raises_type_error(self):
        self.assertRaisesRegex(
            TypeError,
            "initial must be a number",
            lambda: RandomWalk(p=0.5, initial="x"),
        )


if __name__ == "__main__":
    unittest.main()
