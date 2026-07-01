"""Tests for symbulate/spinner.py — probability spinner wheel visualization."""

import unittest
import matplotlib.pyplot as plt
from collections import Counter
from unittest.mock import patch

from symbulate import (
    Bernoulli, Binomial, Geometric, Poisson, DiscreteUniform,
    NegativeBinomial, Hypergeometric,
    Normal, Exponential, Gamma, Beta, Uniform, StudentT, ChiSquare,
    BivariateNormal, MultivariateNormal,
)
from symbulate.spinner import _build_slices, _label_color_map, show_spinner


TOL = 1e-6


def _probs_sum_to_one(slices):
    return abs(sum(s['prob'] for s in slices) - 1.0) < TOL


# ─── discrete proportional ────────────────────────────────────────────────────

class TestDiscreteProportional(unittest.TestCase):

    def test_binomial_probs_sum_to_one(self):
        slices = _build_slices(Binomial(10, 0.3), 'proportional')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_binomial_probs_match_pmf(self):
        dist = Binomial(10, 0.3)
        slices = _build_slices(dist, 'proportional')
        for s in slices:
            if s['label'].startswith('≥'):
                continue
            v = int(s['label'])
            expected = float(dist.pmf(v))
            self.assertAlmostEqual(s['prob'], expected, places=4,
                msg=f"Slice for {v}: got {s['prob']:.6f}, expected {expected:.6f}")

    def test_bernoulli_has_two_slices(self):
        slices = _build_slices(Bernoulli(0.4), 'proportional')
        labels = {s['label'] for s in slices}
        self.assertIn('0', labels)
        self.assertIn('1', labels)

    def test_bernoulli_probabilities_correct(self):
        p = 0.4
        dist = Bernoulli(p)
        slices = _build_slices(dist, 'proportional')
        prob_map = {s['label']: s['prob'] for s in slices}
        self.assertAlmostEqual(prob_map['1'], p, places=6)
        self.assertAlmostEqual(prob_map['0'], 1 - p, places=6)

    def test_discrete_uniform_all_slices_equal(self):
        slices = _build_slices(DiscreteUniform(1, 6), 'proportional')
        self.assertEqual(len(slices), 6)
        for s in slices:
            self.assertAlmostEqual(s['prob'], 1 / 6, places=6)

    def test_poisson_probs_sum_to_one(self):
        slices = _build_slices(Poisson(4), 'proportional')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_poisson_mode_is_largest_slice(self):
        # Mode of Poisson(4) is 3 or 4
        slices = _build_slices(Poisson(4), 'proportional')
        most_likely = max(slices, key=lambda s: s['prob'])
        self.assertIn(most_likely['label'], ('3', '4'))

    def test_geometric_probs_sum_to_one(self):
        slices = _build_slices(Geometric(p=0.4), 'proportional')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_geometric_probs_decrease(self):
        # Geometric is strictly decreasing: P(1) > P(2) > P(3)...
        slices = _build_slices(Geometric(p=0.4), 'proportional')
        non_tail = [s for s in slices if not s['label'].startswith('≥')]
        probs = [s['prob'] for s in non_tail]
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

    def test_no_zero_prob_slices(self):
        slices = _build_slices(Binomial(10, 0.3), 'proportional')
        for s in slices:
            self.assertGreater(s['prob'], 0)

    def test_binomial_support_within_bounds(self):
        slices = _build_slices(Binomial(10, 0.3), 'proportional')
        for s in slices:
            if not s['label'].startswith('≥'):
                v = int(s['label'])
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 10)


# ─── discrete equal ───────────────────────────────────────────────────────────

class TestDiscreteEqual(unittest.TestCase):

    def test_all_slices_equal_prob(self):
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        n = len(slices)
        for s in slices:
            self.assertAlmostEqual(s['prob'], 1.0 / n, places=9)

    def test_probs_sum_to_one(self):
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_labels_are_valid_binomial_values(self):
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        for s in slices:
            v = int(s['label'])
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 10)

    def test_most_probable_value_appears_most(self):
        # Mode of Binomial(10, 0.3) is 3; adjacent values 2 and 4 may tie
        # depending on quantile discretisation with 20 sections
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        counts = Counter(s['label'] for s in slices)
        most_common_label = counts.most_common(1)[0][0]
        self.assertIn(most_common_label, ('2', '3', '4'))

    def test_repeated_labels_exist_for_skewed_dist(self):
        # Most values repeat in equal mode for Binomial
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        labels = [s['label'] for s in slices]
        self.assertGreater(len(labels), len(set(labels)))

    def test_fair_die_all_six_values_appear(self):
        # DiscreteUniform(1,6) — all 6 values should appear in equal mode
        slices = _build_slices(DiscreteUniform(1, 6), 'equal')
        labels = {s['label'] for s in slices}
        for v in range(1, 7):
            self.assertIn(str(v), labels)


# ─── continuous proportional ──────────────────────────────────────────────────

class TestContinuousProportional(unittest.TestCase):

    def test_normal_probs_sum_to_one(self):
        slices = _build_slices(Normal(0, 1), 'proportional')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_normal_center_bins_largest(self):
        # Normal(0,1) peaks at 0 — middle slices should exceed edge slices
        slices = _build_slices(Normal(0, 1), 'proportional')
        probs = [s['prob'] for s in slices]
        n = len(probs)
        self.assertGreater(probs[n // 2], probs[0])
        self.assertGreater(probs[n // 2], probs[-1])

    def test_exponential_probs_sum_to_one(self):
        slices = _build_slices(Exponential(rate=1), 'proportional')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_exponential_first_bin_largest(self):
        # Exponential is monotone decreasing — first bin has most mass
        slices = _build_slices(Exponential(rate=1), 'proportional')
        probs = [s['prob'] for s in slices]
        self.assertGreater(probs[0], probs[-1])

    def test_beta_probs_sum_to_one(self):
        slices = _build_slices(Beta(2, 5), 'proportional')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_beta_left_skewed_mass(self):
        # Beta(2, 5) is right-skewed (mode < 0.5) — first half has more mass
        slices = _build_slices(Beta(2, 5), 'proportional')
        n = len(slices)
        first_half = sum(s['prob'] for s in slices[:n // 2])
        second_half = sum(s['prob'] for s in slices[n // 2:])
        self.assertGreater(first_half, second_half)

    def test_uniform_probs_are_equal(self):
        # Uniform(0, 1): equal-width bins should all have equal mass
        slices = _build_slices(Uniform(0, 1), 'proportional')
        probs = [s['prob'] for s in slices]
        self.assertAlmostEqual(min(probs), max(probs), places=4)

    def test_gamma_probs_sum_to_one(self):
        slices = _build_slices(Gamma(shape=2, rate=1), 'proportional')
        self.assertTrue(_probs_sum_to_one(slices))


# ─── continuous equal ─────────────────────────────────────────────────────────

class TestContinuousEqual(unittest.TestCase):

    def test_all_slices_equal_prob(self):
        slices = _build_slices(Normal(0, 1), 'equal')
        n = len(slices)
        for s in slices:
            self.assertAlmostEqual(s['prob'], 1.0 / n, places=9)

    def test_probs_sum_to_one(self):
        slices = _build_slices(Normal(0, 1), 'equal')
        self.assertTrue(_probs_sum_to_one(slices))

    def test_labels_are_ordered(self):
        # Equal quantile sections go left to right across the distribution
        slices = _build_slices(Normal(0, 1), 'equal')
        vals = [float(s['label']) for s in slices]
        self.assertEqual(vals, sorted(vals))

    def test_normal_spans_negative_and_positive(self):
        slices = _build_slices(Normal(0, 1), 'equal')
        lo = float(slices[0]['label'])
        hi = float(slices[-1]['label'])
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_exponential_labels_all_positive(self):
        slices = _build_slices(Exponential(rate=1), 'equal')
        for s in slices:
            self.assertGreater(float(s['label']), 0)


# ─── label-color consistency ──────────────────────────────────────────────────

class TestLabelColors(unittest.TestCase):

    def test_same_label_gets_same_color(self):
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        color_map = _label_color_map(slices)
        # Every slice's label must appear in the map exactly once
        for s in slices:
            self.assertIn(s['label'], color_map)

    def test_number_of_colors_equals_unique_labels(self):
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        color_map = _label_color_map(slices)
        unique_labels = {s['label'] for s in slices}
        self.assertEqual(len(color_map), len(unique_labels))

    def test_distinct_labels_get_distinct_colors(self):
        # For proportional discrete uniform, every label is unique
        slices = _build_slices(DiscreteUniform(1, 6), 'proportional')
        color_map = _label_color_map(slices)
        colors = list(color_map.values())
        self.assertEqual(len(colors), len(set(colors)))

    def test_repeated_label_maps_to_one_color(self):
        # In equal mode some labels repeat — they must all share one color
        slices = _build_slices(Binomial(10, 0.3), 'equal')
        color_map = _label_color_map(slices)
        repeated = [l for l, c in Counter(s['label'] for s in slices).items() if c > 1]
        for label in repeated:
            # Only one entry per label in the map
            self.assertIn(label, color_map)


# ─── multivariate distributions raise errors ──────────────────────────────────

class TestMultivariateError(unittest.TestCase):

    def test_bivariate_normal_raises_value_error(self):
        dist = BivariateNormal(mean1=0, mean2=0, sd1=1, sd2=1, cov=0)
        with self.assertRaises(ValueError):
            show_spinner(dist)

    def test_multivariate_normal_raises_value_error(self):
        dist = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            show_spinner(dist)

    def test_error_message_mentions_multivariate(self):
        dist = BivariateNormal(mean1=0, mean2=0, sd1=1, sd2=1, cov=0)
        try:
            show_spinner(dist)
            self.fail("Expected ValueError")
        except ValueError as e:
            self.assertIn('multivariate', str(e).lower())


# ─── smoke tests: .spinner() runs without error ───────────────────────────────

DISCRETE_DISTS = [
    Bernoulli(0.5),
    Binomial(10, 0.3),
    Geometric(p=0.4),
    Poisson(4),
    DiscreteUniform(1, 6),
    NegativeBinomial(3, 0.5),
    Hypergeometric(n=5, N0=10, N1=10),
]

CONTINUOUS_DISTS = [
    Normal(0, 1),
    Exponential(rate=1),
    Gamma(shape=2, rate=1),
    Beta(2, 5),
    Uniform(0, 1),
    StudentT(df=5),
    ChiSquare(df=4),
]


class TestSpinnerSmokeDiscrete(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def _run(self, dist, mode):
        with patch('matplotlib.pyplot.show'):
            show_spinner(dist, mode=mode)

    def test_bernoulli_proportional(self):
        self._run(Bernoulli(0.5), 'proportional')

    def test_bernoulli_equal(self):
        self._run(Bernoulli(0.5), 'equal')

    def test_binomial_proportional(self):
        self._run(Binomial(10, 0.3), 'proportional')

    def test_binomial_equal(self):
        self._run(Binomial(10, 0.3), 'equal')

    def test_geometric_proportional(self):
        self._run(Geometric(p=0.4), 'proportional')

    def test_geometric_equal(self):
        self._run(Geometric(p=0.4), 'equal')

    def test_poisson_proportional(self):
        self._run(Poisson(4), 'proportional')

    def test_poisson_equal(self):
        self._run(Poisson(4), 'equal')

    def test_discrete_uniform_proportional(self):
        self._run(DiscreteUniform(1, 6), 'proportional')

    def test_discrete_uniform_equal(self):
        self._run(DiscreteUniform(1, 6), 'equal')


class TestSpinnerSmokeContinuous(unittest.TestCase):

    def tearDown(self):
        plt.close('all')

    def _run(self, dist, mode):
        with patch('matplotlib.pyplot.show'):
            show_spinner(dist, mode=mode)

    def test_normal_proportional(self):
        self._run(Normal(0, 1), 'proportional')

    def test_normal_equal(self):
        self._run(Normal(0, 1), 'equal')

    def test_exponential_proportional(self):
        self._run(Exponential(rate=1), 'proportional')

    def test_exponential_equal(self):
        self._run(Exponential(rate=1), 'equal')

    def test_beta_proportional(self):
        self._run(Beta(2, 5), 'proportional')

    def test_beta_equal(self):
        self._run(Beta(2, 5), 'equal')

    def test_gamma_proportional(self):
        self._run(Gamma(shape=2, rate=1), 'proportional')

    def test_uniform_proportional(self):
        self._run(Uniform(0, 1), 'proportional')

    def test_student_t_proportional(self):
        self._run(StudentT(df=5), 'proportional')

    def test_chisquare_proportional(self):
        self._run(ChiSquare(df=4), 'proportional')


class TestSpinnerMethod(unittest.TestCase):
    """Verify that Distribution.spinner() calls through correctly."""

    def tearDown(self):
        plt.close('all')

    def test_method_default_mode(self):
        with patch('matplotlib.pyplot.show'):
            Normal(0, 1).spinner()

    def test_method_equal_mode(self):
        with patch('matplotlib.pyplot.show'):
            Binomial(10, 0.3).spinner(mode='equal')

    def test_method_proportional_mode(self):
        with patch('matplotlib.pyplot.show'):
            Poisson(4).spinner(mode='proportional')


if __name__ == '__main__':
    unittest.main()
