"""Tests for symbulate.branching_process.

Covers the Galton-Watson family tree: the sample-path result object, the
probability space, and the facts the process is taught for -- the expected
generation size is ``m ** n``, and whether the line survives turns sharply
on whether the mean number of children ``m`` is below, at, or above 1.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_time_series.py.
"""

import unittest

import numpy as np

from symbulate import *
from symbulate import distributions
from symbulate.branching_process import (
    GaltonWatson,
    GaltonWatsonResult,
    GaltonWatsonProbabilitySpace,
)
from symbulate.result import DiscreteValued, InfiniteVector

Nsim = 5000


class TestGaltonWatsonResult(unittest.TestCase):

    def test_is_infinite_vector(self):
        seed(42)
        self.assertIsInstance(
            GaltonWatson(offspring_dist=Poisson(1.5)).draw(), InfiniteVector
        )

    def test_is_discrete_valued(self):
        seed(42)
        self.assertIsInstance(
            GaltonWatson(offspring_dist=Poisson(1.5)).draw(), DiscreteValued
        )

    def test_generation_zero_is_the_initial_population(self):
        seed(42)
        self.assertEqual(int(GaltonWatson(offspring_dist=Poisson(1.5)).draw()[0]), 1)
        self.assertEqual(
            int(GaltonWatson(offspring_dist=Poisson(1.5), initial=7).draw()[0]), 7
        )

    def test_deterministic_doubling(self):
        # Everyone has exactly two children, so the tree doubles each step.
        seed(42)
        path = GaltonWatson(offspring_dist=Binomial(2, 1)).draw()
        self.assertEqual([int(path[n]) for n in range(6)], [1, 2, 4, 8, 16, 32])

    def test_sizes_are_non_negative_integers(self):
        seed(42)
        path = GaltonWatson(offspring_dist=Poisson(1.2)).draw()
        for n in range(12):
            size = int(path[n])
            self.assertGreaterEqual(size, 0)
            self.assertEqual(float(path[n]), size)

    def test_extinction_is_absorbing(self):
        # Nobody has children, so generation 1 is empty and stays empty --
        # and reading far ahead must be instant, not a long loop.
        seed(42)
        path = GaltonWatson(offspring_dist=Binomial(1, 0)).draw()
        self.assertEqual(int(path[1]), 0)
        self.assertEqual(int(path[100000]), 0)

    def test_path_is_cached_and_stable(self):
        seed(42)
        path = GaltonWatson(offspring_dist=Poisson(1.1)).draw()
        first = [int(path[n]) for n in range(10)]
        self.assertEqual(first, [int(path[n]) for n in range(10)])

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed(42)
        path = GaltonWatson(offspring_dist=Poisson(0.9)).draw()
        early = [int(path[n]) for n in range(6)]
        path[40]
        self.assertEqual([int(path[n]) for n in range(6)], early)

    def test_does_not_clobber_the_infinite_vector_cache(self):
        # The internal list must not be called `values`, which InfiniteTuple
        # already uses for its own cache.
        seed(42)
        path = GaltonWatson(offspring_dist=Binomial(2, 1)).draw()
        self.assertEqual([int(path[n]) for n in range(5)], [1, 2, 4, 8, 16])

    def test_get_states_and_is_extinct(self):
        seed(42)
        path = GaltonWatson(offspring_dist=Binomial(1, 0)).draw()
        self.assertIs(path.get_states(), path)
        self.assertTrue(path.is_extinct(1))
        alive = GaltonWatson(offspring_dist=Binomial(2, 1)).draw()
        self.assertFalse(alive.is_extinct(3))


class TestGaltonWatsonProbabilitySpace(unittest.TestCase):

    def test_stores_parameters(self):
        space = GaltonWatsonProbabilitySpace(offspring_dist=Poisson(1.5), initial=4)
        self.assertEqual(space.initial, 4)
        self.assertIsInstance(space.offspring_dist, Poisson)

    def test_default_initial_is_one(self):
        self.assertEqual(
            GaltonWatsonProbabilitySpace(offspring_dist=Poisson(1.5)).initial, 1
        )

    def test_draw_returns_result(self):
        seed(42)
        self.assertIsInstance(
            GaltonWatsonProbabilitySpace(offspring_dist=Poisson(1.5)).draw(),
            GaltonWatsonResult,
        )


class TestGaltonWatsonConstruction(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(GaltonWatson(offspring_dist=Poisson(1.5)), RV)

    def test_getitem_returns_rv(self):
        self.assertIsInstance(GaltonWatson(offspring_dist=Poisson(1.5))[3], RV)

    def test_reproducible_under_same_seed(self):
        seed(9)
        first = [int(v) for v in GaltonWatson(offspring_dist=Poisson(1.3)).draw()[:10]]
        seed(9)
        self.assertEqual(
            first,
            [int(v) for v in GaltonWatson(offspring_dist=Poisson(1.3)).draw()[:10]],
        )


class TestGaltonWatsonTheory(unittest.TestCase):
    """The threshold at a mean of one child, which the process exists to show."""

    def test_expected_size_is_m_to_the_n(self):
        seed(42)
        for m in [0.8, 1.0, 1.5]:
            X = GaltonWatson(offspring_dist=Poisson(m))
            for n in [1, 2, 3]:
                self.assertAlmostEqual(
                    float(X[n].sim(Nsim).mean()), m**n, delta=0.25 * max(1, m**n)
                )

    def test_initial_population_scales_the_mean(self):
        seed(42)
        m, start = 1.2, 10
        X = GaltonWatson(offspring_dist=Poisson(m), initial=start)
        self.assertAlmostEqual(float(X[3].sim(Nsim).mean()), start * m**3, delta=2.0)

    def test_subcritical_dies_out(self):
        # m < 1: extinction is certain, and fast.
        seed(42)
        X = GaltonWatson(offspring_dist=Poisson(0.8))
        extinct = X.apply(lambda p: 1 if int(p[15]) == 0 else 0).sim(2000)
        self.assertGreater(float(extinct.mean()), 0.95)

    def test_supercritical_extinction_probability(self):
        # m = 1.5 with Poisson offspring: the extinction probability solves
        # s = exp(m * (s - 1)), giving about 0.417.
        seed(42)
        X = GaltonWatson(offspring_dist=Poisson(1.5))
        extinct = X.apply(lambda p: 1 if int(p[15]) == 0 else 0).sim(2000)
        self.assertAlmostEqual(float(extinct.mean()), 0.417, delta=0.05)

    def test_certain_offspring_never_dies(self):
        seed(42)
        X = GaltonWatson(offspring_dist=Binomial(2, 1))
        extinct = X.apply(lambda p: 1 if int(p[10]) == 0 else 0).sim(200)
        self.assertEqual(float(extinct.mean()), 0.0)

    def test_zero_initial_population_stays_empty(self):
        seed(42)
        path = GaltonWatson(offspring_dist=Poisson(2.0), initial=0).draw()
        self.assertEqual([int(path[n]) for n in range(5)], [0, 0, 0, 0, 0])


class TestGaltonWatsonErrors(unittest.TestCase):

    def test_non_distribution_offspring_raises_type_error(self):
        for bad in [5, "poisson", [1, 2]]:
            self.assertRaisesRegex(
                TypeError,
                "offspring_dist must be",
                lambda v=bad: GaltonWatson(offspring_dist=v),
            )

    def test_continuous_offspring_raises_value_error(self):
        # A number of children has to be a whole number.
        for bad in [Normal(0, 1), Exponential(1), Uniform(0, 3)]:
            self.assertRaisesRegex(
                ValueError,
                "must be a discrete distribution",
                lambda v=bad: GaltonWatson(offspring_dist=v),
            )

    def test_negative_support_offspring_raises_value_error(self):
        # DiscreteUniform(-1, 2) can produce -1 children.
        self.assertRaisesRegex(
            ValueError,
            "cannot be negative",
            lambda: GaltonWatson(offspring_dist=DiscreteUniform(-1, 2)),
        )

    def test_bad_initial_raises(self):
        self.assertRaisesRegex(
            TypeError,
            "initial must be a number",
            lambda: GaltonWatson(offspring_dist=Poisson(1.5), initial="x"),
        )
        for bad in [-1, 2.5]:
            self.assertRaisesRegex(
                ValueError,
                "non-negative whole number",
                lambda v=bad: GaltonWatson(offspring_dist=Poisson(1.5), initial=v),
            )


if __name__ == "__main__":
    unittest.main()
