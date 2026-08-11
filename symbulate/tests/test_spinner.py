"""Tests for symbulate/spinner.py -- the probability spinner dial.

Covers, in the order the port's build plan introduces them:

  - the geometry contract: a bearing is measured clockwise from 12 o'clock,
    and the value drawn is ``quantile(bearing / 360)``
  - the SVG-to-matplotlib coordinate flip (a mirrored dial is the failure
    this pins down)
  - the discrete "sized by probability" dial: wedges, value labels,
    probability labels
  - the color ramp: 42 distinct (fill, hatch) pairs, and label contrast
  - the two continuous dials: equal area (every slice the same chance) and
    equal increments (a ruler of round values, bands sized by chance)
  - the nice-number ladder behind ``sections=`` and ``increment=``

Tests numbered below refer to the acceptance tests in the port
specification, section 8; all eight are covered.

Tests seed Symbulate's shared generator with ``seed()`` -- not
``np.random.seed()``, which does not reach it -- matching the convention in
test_distributions.py.
"""

import math
import unittest

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from symbulate import (
    Bernoulli,
    Binomial,
    DiscreteUniform,
    Exponential,
    Geometric,
    Normal,
    Poisson,
    Uniform,
    seed,
)
from symbulate.probability_space import rng
from symbulate.result import Float, Int
from symbulate.spinner import (
    HATCH,
    LAP_MAG,
    SPINNER_AXIS_LIMIT,
    SPINNER_FACE_TINT,
    SPINNER_LEADER_COLOR,
    SPINNER_MIN_LABEL_SWEEP,
    SPINNER_PALETTE,
    SPINNER_RADIUS,
    SPINNER_SEAM_COLOR,
    SPINNER_SECTIONS_RANGE,
    _format_continuous,
    label_rotation,
    area_bound_labels,
    area_slices,
    bearing_of,
    discrete_slices,
    display_range,
    increment_options,
    increment_slices,
    ink_on,
    nice_num,
    nice_step,
    pt,
    relative_luminance,
    slice_fill,
    slice_hatch,
    spin_value,
    wedge_angles,
)


def contrast_ratio(fg, bg):
    """WCAG contrast ratio between two colors, from 1:1 to 21:1."""
    light, dark = sorted((relative_luminance(fg), relative_luminance(bg)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


# ---------------------------------------------------------------------------
# Test 1 -- the invariant. Written before any drawing code existed, because it
# is what pins the geometry down: bearing clockwise from 12 o'clock, and
# X = quantile(bearing / 360).
# ---------------------------------------------------------------------------


class TestGeometryContract(unittest.TestCase):
    """The one invariant: a spin picks a bearing, the value is its quantile."""

    def test_quarter_half_three_quarter_turns(self):
        """1/4, 1/2, 3/4 of a turn are the 25th, 50th, 75th percentiles."""
        for dist in [
            Normal(0, 1),
            Normal(-3, 2.5),
            Exponential(1),
            Uniform(0, 1),
            Binomial(10, 0.4),
            Poisson(4),
            Geometric(0.3),
        ]:
            with self.subTest(dist=type(dist).__name__):
                # Exact equality, not approximate -- spin_value must be the
                # quantile function itself, with no grid or interpolation
                # anywhere in between.
                self.assertEqual(spin_value(dist, 90.0), dist.quantile(0.25))
                self.assertEqual(spin_value(dist, 180.0), dist.quantile(0.50))
                self.assertEqual(spin_value(dist, 270.0), dist.quantile(0.75))

    def test_full_sweep_matches_quantile_everywhere(self):
        """Not just the three landmarks -- every bearing is its quantile."""
        dist = Normal(2, 3)
        for bearing in np.linspace(0.0, 360.0, 721):
            self.assertEqual(spin_value(dist, bearing), dist.quantile(bearing / 360.0))

    def test_bearing_of_inverts_spin_value(self):
        """Test 8 (continuous half): the needle goes back where it came from."""
        dist = Normal(0, 1)
        for bearing in [1.0, 45.0, 90.0, 180.0, 270.0, 359.0]:
            value = spin_value(dist, bearing)
            self.assertAlmostEqual(bearing_of(dist, value), bearing, places=9)


class TestPointHelper(unittest.TestCase):
    """The coordinate helper -- where a mirrored dial would come from.

    The prototype is SVG, where y increases downward; matplotlib's y
    increases upward. Getting the sign wrong still draws a plausible
    picture, but 3/4 of a turn lands on the 25th percentile.
    """

    def test_cardinal_bearings(self):
        """0 -> top, 90 -> right, 180 -> bottom, 270 -> left."""
        for bearing, expected in [
            (0.0, (0.0, 1.0)),
            (90.0, (1.0, 0.0)),
            (180.0, (0.0, -1.0)),
            (270.0, (-1.0, 0.0)),
            (360.0, (0.0, 1.0)),
        ]:
            with self.subTest(bearing=bearing):
                x, y = pt(bearing, 1.0)
                self.assertAlmostEqual(x, expected[0], places=12)
                self.assertAlmostEqual(y, expected[1], places=12)

    def test_bearings_run_clockwise(self):
        """A small positive bearing moves right and down from the top."""
        x, y = pt(10.0, 1.0)
        self.assertGreater(x, 0.0)  # to the right of 12 o'clock
        self.assertLess(y, 1.0)  # and below the very top
        # This is the assertion that fails if the y-flip is missed: with an
        # unflipped helper, bearing 10 would come out *above* the top.

    def test_radius_and_centre_are_honored(self):
        x, y = pt(90.0, 2.0, cx=1.0, cy=-1.0)
        self.assertAlmostEqual(x, 3.0, places=12)
        self.assertAlmostEqual(y, -1.0, places=12)

    def test_wedge_angles_span_the_bearing_arc(self):
        """A Wedge is counter-clockwise from +x; a bearing is clockwise from +y."""
        theta1, theta2 = wedge_angles(0.0, 90.0)
        # bearings 0..90 is the upper-right quadrant: 0..90 clockwise from the
        # top is 90..0 counter-clockwise from the positive x-axis.
        self.assertAlmostEqual(theta1, 0.0, places=12)
        self.assertAlmostEqual(theta2, 90.0, places=12)
        # and the arc always runs theta1 -> theta2 counter-clockwise
        for a0, a1 in [(0, 30), (30, 200), (200, 360), (170, 190)]:
            t1, t2 = wedge_angles(float(a0), float(a1))
            self.assertLess(t1, t2)
            self.assertAlmostEqual(t2 - t1, a1 - a0, places=12)


class TestLabelRotation(unittest.TestCase):
    """A label runs along its own spoke, outward, and is never upside down.

    Regression tests for a rotation sign error. ``label_rotation`` returned
    ``bearing - 90`` where outward radial text needs ``90 - bearing``. The two
    agree at 3 and 9 o'clock, which is why the dial looked almost right: on the
    diagonals the labels ran *along* the rim, and at 12 and 6 o'clock they ran
    straight across the wheel, colliding with it instead of sitting on their
    ticks.
    """

    def _reading_direction(self, bearing):
        """The unit vector the label's glyphs advance along."""
        rotation, flipped = label_rotation(bearing)
        # A folded label reads inward and is anchored by its far end, so the
        # string still grows away from the centre.
        sign = -1.0 if flipped else 1.0
        return np.array(
            [
                sign * math.cos(math.radians(rotation)),
                sign * math.sin(math.radians(rotation)),
            ]
        )

    def _outward(self, bearing):
        """The unit vector pointing away from the centre at ``bearing``."""
        return np.array(
            [math.sin(math.radians(bearing)), math.cos(math.radians(bearing))]
        )

    def test_labels_run_outward_all_the_way_round(self):
        """The one that fails on the old sign, everywhere but 3 and 9 o'clock."""
        for bearing in np.arange(0.0, 360.0, 2.5):
            with self.subTest(bearing=bearing):
                dot = float(self._reading_direction(bearing) @ self._outward(bearing))
                self.assertAlmostEqual(dot, 1.0, places=9)

    def test_the_cardinal_bearings_specifically(self):
        """Spelled out, because these are the ones that read wrong on a dial."""
        for bearing, expected in [
            (0.0, (0.0, 1.0)),  # top -> label runs up
            (90.0, (1.0, 0.0)),  # right -> runs right
            (180.0, (0.0, -1.0)),  # bottom -> runs down
            (270.0, (-1.0, 0.0)),  # left -> runs left
        ]:
            with self.subTest(bearing=bearing):
                direction = self._reading_direction(bearing)
                self.assertAlmostEqual(direction[0], expected[0], places=9)
                self.assertAlmostEqual(direction[1], expected[1], places=9)

    def test_no_label_is_upside_down(self):
        for bearing in np.arange(0.0, 360.0, 2.5):
            rotation, _ = label_rotation(bearing)
            self.assertLessEqual(abs(rotation), 90.0 + 1e-9)

    def test_the_reading_direction_reverses_across_the_vertical(self):
        """Upright everywhere is bought at this price, and it is the right one."""
        self.assertFalse(label_rotation(45.0)[1])
        self.assertTrue(label_rotation(315.0)[1])


# ---------------------------------------------------------------------------
# Test 2 -- fair sampling. A spin is a draw from the distribution.
# ---------------------------------------------------------------------------


class TestFairSampling(unittest.TestCase):
    """Spinning really is sampling from the distribution.

    These are the only tests here that touch Symbulate's shared generator, so
    they put its state back exactly as they found it. Several tests elsewhere
    in the suite simulate without seeding first -- they seed
    ``np.random.seed``, which does not reach this generator at all -- so a
    test file that leaves the shared stream advanced can change what a later
    file simulates and break it from a distance.
    """

    def setUp(self):
        self._rng_state = rng.bit_generator.state

    def tearDown(self):
        rng.bit_generator.state = self._rng_state

    def test_uniform_bearings_reproduce_the_distribution(self):
        """100k bearings through the contract recover mean and sd."""
        bearings = np.random.default_rng(1).uniform(0.0, 360.0, 100000)

        normal = np.array([spin_value(Normal(0, 1), b) for b in bearings])
        self.assertAlmostEqual(normal.mean(), 0.0, delta=0.02)
        self.assertAlmostEqual(normal.std(), 1.0, delta=0.02)

        exponential = np.array([spin_value(Exponential(1), b) for b in bearings])
        self.assertAlmostEqual(exponential.mean(), 1.0, delta=0.02)

    def test_discrete_empirical_pmf_matches_theory(self):
        """Binomial(10, 0.4): empirical pmf within 0.006 of the true pmf."""
        bearings = np.random.default_rng(2).uniform(0.0, 360.0, 100000)
        dist = Binomial(10, 0.4)
        drawn = np.array([spin_value(dist, b) for b in bearings])
        for k in range(11):
            empirical = np.mean(drawn == k)
            self.assertLess(abs(empirical - float(dist.pmf(k))), 0.006)

    def test_calling_spinner_spins(self):
        """.spinner() draws a value and reports it on the plot."""
        plt.close("all")
        plt.figure()
        plot = Binomial(10, 0.5).spinner(seed=7)
        self.assertIn(int(plot.value), range(11))
        self.assertTrue(plt.gcf().axes)
        plt.close("all")

    def test_a_spin_is_reproducible_under_a_seed(self):
        plt.close("all")
        plt.figure()
        first = Poisson(4).spinner(seed=123).value
        plt.close("all")
        plt.figure()
        second = Poisson(4).spinner(seed=123).value
        plt.close("all")
        self.assertEqual(first, second)

    def test_unseeded_spins_differ(self):
        """Without a seed the needle really does land somewhere new."""
        values = []
        for _ in range(12):
            plt.close("all")
            plt.figure()
            values.append(Normal(0, 1).spinner().value)
        plt.close("all")
        self.assertGreater(len(set(values)), 1)

    def test_a_seeded_spin_does_not_disturb_the_shared_stream(self):
        """A seeded spin uses its own generator, leaving seed() alone."""
        seed(99)
        expected = Normal(0, 1).draw()
        seed(99)
        plt.close("all")
        plt.figure()
        Binomial(10, 0.5).spinner(seed=5)
        plt.close("all")
        self.assertEqual(Normal(0, 1).draw(), expected)

    def test_a_spun_value_looks_like_a_drawn_one(self):
        """A spin is a draw, so it comes back wrapped the same way.

        Without this, scipy's float-returning ``ppf`` makes a discrete spin
        read as ``np.float64(6.0)`` where ``.draw()`` gives ``6``.
        """
        plt.close("all")
        plt.figure()
        discrete = Binomial(10, 0.5).spinner(seed=2).value
        self.assertIsInstance(discrete, Int)
        self.assertEqual(repr(discrete), str(int(discrete)))
        plt.close("all")
        plt.figure()
        continuous = Normal(0, 1).spinner(seed=2).value
        self.assertIsInstance(continuous, Float)
        plt.close("all")

    def test_seed_and_value_together_are_refused(self):
        plt.close("all")
        plt.figure()
        with self.assertRaises(ValueError) as caught:
            Binomial(10, 0.5).spinner(value=3, seed=1)
        self.assertIn("nothing left to seed", str(caught.exception))
        plt.close("all")


# ---------------------------------------------------------------------------
# The discrete dial -- sized by probability, one slice per outcome.
# ---------------------------------------------------------------------------


class TestDiscreteSlices(unittest.TestCase):
    """Slice geometry: arcs are the probabilities, and they agree with ppf."""

    def test_slices_tile_the_whole_circle(self):
        for dist in [Binomial(10, 0.5), Poisson(3), Bernoulli(0.3), Geometric(0.4)]:
            with self.subTest(dist=type(dist).__name__):
                slices = discrete_slices(dist)
                self.assertAlmostEqual(slices[0].start, 0.0, places=9)
                self.assertAlmostEqual(slices[-1].end, 360.0, places=9)
                for earlier, later in zip(slices, slices[1:]):
                    self.assertAlmostEqual(earlier.end, later.start, places=12)

    def test_arc_is_the_probability(self):
        """A bounded distribution's arcs are exactly 360 * pmf."""
        dist = Binomial(10, 0.5)
        for sl in discrete_slices(dist):
            self.assertAlmostEqual(
                sl.end - sl.start, 360.0 * float(dist.pmf(sl.value)), places=9
            )

    def test_probabilities_sum_to_one(self):
        for dist in [Binomial(6, 0.25), Poisson(5), DiscreteUniform(1, 6)]:
            with self.subTest(dist=type(dist).__name__):
                slices = discrete_slices(dist)
                self.assertAlmostEqual(
                    sum(sl.probability for sl in slices), 1.0, delta=2e-3
                )

    def test_wedges_agree_with_the_quantile_contract(self):
        """The slice a bearing lands in carries the value ppf gives it.

        This is the consistency that makes the picture honest: the dial is a
        drawing of the quantile function, so reading the wedge and applying
        the contract must never disagree.
        """
        for dist in [Binomial(10, 0.4), DiscreteUniform(1, 6), Poisson(3)]:
            with self.subTest(dist=type(dist).__name__):
                slices = discrete_slices(dist)
                for sl in slices:
                    # sample inside the wedge, away from its boundaries
                    for frac in (0.05, 0.5, 0.95):
                        bearing = sl.start + frac * (sl.end - sl.start)
                        value = spin_value(dist, bearing)
                        # the trimmed tail slivers are absorbed into the end
                        # wedges, so only interior wedges are compared
                        if sl is slices[0] or sl is slices[-1]:
                            continue
                        self.assertEqual(value, sl.value)

    def test_support_is_trimmed_by_the_packages_own_rule(self):
        """An unbounded discrete support reuses Distribution.xlim, not a new rule."""
        dist = Poisson(3)
        low, high = dist.xlim
        values = [sl.value for sl in discrete_slices(dist)]
        self.assertGreaterEqual(min(values), math.floor(low))
        self.assertLessEqual(max(values), math.ceil(high))

    def test_bernoulli_has_two_slices(self):
        slices = discrete_slices(Bernoulli(0.25))
        self.assertEqual([sl.value for sl in slices], [0, 1])
        self.assertAlmostEqual(slices[0].probability, 0.75, places=9)
        self.assertAlmostEqual(slices[1].probability, 0.25, places=9)


class TestDiscreteDialFigure(unittest.TestCase):
    """What actually lands on the axes."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def _wedges(self, ax):
        return [p for p in ax.patches if isinstance(p, Wedge)]

    def test_one_wedge_per_outcome(self):
        dist = Binomial(10, 0.5)
        ax = dist.spinner().ax
        expected = len(discrete_slices(dist))
        # each outcome contributes its fill wedge; repeated palette laps add
        # a second, hatched overlay wedge on top of the same arc
        fills = [w for w in self._wedges(ax) if w.get_hatch() is None]
        self.assertEqual(len(fills), expected)

    def test_axes_furniture_is_off(self):
        ax = Binomial(5, 0.5).spinner().ax
        self.assertFalse(ax.axison)
        self.assertEqual(ax.get_aspect(), 1.0)
        self.assertAlmostEqual(ax.get_xlim()[1], SPINNER_AXIS_LIMIT, places=6)
        self.assertAlmostEqual(ax.get_ylim()[0], -SPINNER_AXIS_LIMIT, places=6)

    def test_every_outcome_is_labelled_when_there_is_room(self):
        """A six-outcome dial names all six."""
        ax = DiscreteUniform(1, 6).spinner().ax
        labels = {t.get_text() for t in ax.texts}
        for value in range(1, 7):
            self.assertIn(str(value), labels)

    def test_probability_labels_only_on_wide_enough_wedges(self):
        """A wedge narrower than the minimum sweep carries no percentage."""
        dist = Poisson(3)
        ax = dist.spinner().ax
        percent_labels = [t for t in ax.texts if t.get_text().endswith("%")]
        wide = [
            sl
            for sl in discrete_slices(dist)
            if sl.end - sl.start >= SPINNER_MIN_LABEL_SWEEP
        ]
        self.assertEqual(len(percent_labels), len(wide))

    def test_no_rim_label_overlaps_the_wheel(self):
        """Measured on the drawn figure, not derived -- the visible symptom.

        With the rotation sign wrong, a label at 12 or 6 o'clock ran straight
        back across the dial, so its box reached well inside the rim.
        """
        for dist, kwargs in [
            (Binomial(20, 0.5), {"value": 10}),
            (Binomial(10, 0.5), {"value": 6}),
            (DiscreteUniform(1, 6), {"value": 4}),
            (Normal(0, 1), {"sections": 12}),
            (Normal(0, 1), {"style": "area"}),
        ]:
            with self.subTest(dist=type(dist).__name__, **kwargs):
                plt.close("all")
                figure = plt.figure()
                plot = dist.spinner(**kwargs)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                to_data = plot.ax.transData.inverted()
                for text in plot.ax.texts:
                    label = text.get_text()
                    # In-slice percentages belong inside, and the equal-area
                    # seam pair sits inside the rim on its own ring by design.
                    if label.endswith("%") or "∞" in label:
                        continue
                    box = text.get_window_extent(renderer=renderer)
                    corners = to_data.transform(
                        [
                            (box.x0, box.y0),
                            (box.x1, box.y0),
                            (box.x0, box.y1),
                            (box.x1, box.y1),
                        ]
                    )
                    nearest = min(math.hypot(x, y) for x, y in corners)
                    self.assertGreaterEqual(
                        nearest,
                        SPINNER_RADIUS,
                        f"label {label!r} reaches r={nearest:.3f}, inside the rim",
                    )
        plt.close("all")

    def test_labels_are_never_upside_down(self):
        """Radial text is folded into [-90, 90] so nothing reads inverted."""
        ax = Binomial(12, 0.5).spinner().ax
        for text in ax.texts:
            rotation = text.get_rotation()
            folded = (rotation + 180.0) % 360.0 - 180.0
            self.assertLessEqual(abs(folded), 90.0 + 1e-9)

    def test_the_dial_does_not_rotate(self):
        """Two spins of the same distribution lay the wedges in one place."""
        first = [
            (w.theta1, w.theta2)
            for w in self._wedges(Binomial(8, 0.5).spinner(value=2).ax)
        ]
        plt.close("all")
        second = [
            (w.theta1, w.theta2)
            for w in self._wedges(Binomial(8, 0.5).spinner(value=7).ax)
        ]
        self.assertEqual(first, second)

    def test_needle_points_at_the_given_value(self):
        """Test 8 (discrete half): the needle sits mid-slice for an outcome."""
        dist = Binomial(10, 0.5)
        plot = dist.spinner(value=3)
        target = next(sl for sl in discrete_slices(dist) if sl.value == 3)
        midpoint = 0.5 * (target.start + target.end)
        self.assertAlmostEqual(plot.bearing, midpoint, places=9)

    def test_the_needle_lands_inside_the_slice_it_reports(self):
        """A spin's own value and its needle position cannot disagree."""
        dist = Binomial(10, 0.5)
        for attempt in range(20):
            plt.close("all")
            plt.figure()
            plot = dist.spinner(seed=attempt)
            landed = next(sl for sl in plot.slices if sl.start <= plot.bearing < sl.end)
            self.assertEqual(landed.value, plot.value)
        plt.close("all")

    def test_draws_onto_a_supplied_axes(self):
        fig, ax = plt.subplots()
        returned = Binomial(4, 0.5).spinner(ax=ax).ax
        self.assertIs(returned, ax)

    def test_multivariate_distribution_is_refused(self):
        from symbulate import MultivariateNormal

        dist = MultivariateNormal([0, 0], [[1, 0], [0, 1]])
        with self.assertRaises(Exception) as caught:
            dist.spinner()
        self.assertIn("spinner", str(caught.exception).lower())

    def test_removed_mode_keyword_names_its_replacement(self):
        """The old mode= argument raises rather than reaching matplotlib."""
        with self.assertRaises(TypeError) as caught:
            Binomial(10, 0.5).spinner(mode="equal")
        message = str(caught.exception)
        self.assertIn("mode", message)
        self.assertIn("style", message)


# ---------------------------------------------------------------------------
# Tests 6 and 7 -- colour. Seven hues cannot cover a 21-outcome support, so
# each lap of the palette is shifted in lightness and given a hatch.
# ---------------------------------------------------------------------------


class TestSliceColors(unittest.TestCase):
    """Test 6: fills are distinct. Test 7: labels stay readable on them."""

    def test_forty_two_distinct_fills(self):
        pairs = [(slice_fill(i), slice_hatch(i)) for i in range(42)]
        self.assertEqual(len(set(pairs)), 42)

    def test_first_lap_is_the_palette_itself(self):
        """Lap 0 is the mplstyle prop_cycle, untouched."""
        for i, color in enumerate(SPINNER_PALETTE):
            self.assertEqual(slice_fill(i), color)
            self.assertIsNone(slice_hatch(i))

    def test_ramp_runs_away_from_the_hue_s_own_end(self):
        """A light hue darkens and a dark hue lightens -- not one fixed way.

        Ramping in a fixed direction is the bug this pins: yellow is already
        light, so lightening it separates by almost nothing.
        """
        for index, base in enumerate(SPINNER_PALETTE):
            with self.subTest(color=base):
                start = relative_luminance(base)
                second_lap = relative_luminance(slice_fill(index + 7))
                self.assertGreater(abs(second_lap - start), 0.03)

    def test_every_in_slice_label_clears_four_point_five_to_one(self):
        """Test 7: contrast-aware ink, on every fill the ramp can produce."""
        for i in range(42):
            fill = slice_fill(i)
            with self.subTest(fill=fill):
                self.assertGreaterEqual(contrast_ratio(ink_on(fill), fill), 4.5)

    def test_pure_black_is_the_dark_ink(self):
        """An off-black leaves a mid-lightness gap where neither ink reaches 4.5."""
        dark_choices = {
            ink_on(slice_fill(i)) for i in range(42) if ink_on(slice_fill(i)) != "white"
        }
        self.assertEqual(dark_choices, {"black"})

    def test_lap_tables_line_up(self):
        self.assertEqual(len(LAP_MAG), len(HATCH))
        self.assertEqual(LAP_MAG[0], 0)
        self.assertIsNone(HATCH[0])

    def test_luminance_endpoints(self):
        self.assertAlmostEqual(relative_luminance("white"), 1.0, places=6)
        self.assertAlmostEqual(relative_luminance("black"), 0.0, places=6)


# ---------------------------------------------------------------------------
# Tests 3, 4, 5 -- the two continuous dials.
# ---------------------------------------------------------------------------


CONTINUOUS = [Normal(0, 1), Exponential(1), Uniform(0, 1), Normal(-2, 0.5)]


class TestBandsSumToOne(unittest.TestCase):
    """Test 3: whatever is dropped, the bands still cover all the probability."""

    def test_increments_bands_sum_to_one(self):
        low, high = SPINNER_SECTIONS_RANGE["increments"]
        for dist in CONTINUOUS:
            for sections in range(low, high + 1):
                with self.subTest(dist=type(dist).__name__, sections=sections):
                    slices, _, _ = increment_slices(dist, sections=sections)
                    total = sum(sl.probability for sl in slices)
                    self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_area_bands_sum_to_one(self):
        low, high = SPINNER_SECTIONS_RANGE["area"]
        for dist in CONTINUOUS:
            for sections in range(low, high + 1):
                with self.subTest(dist=type(dist).__name__, sections=sections):
                    slices, _ = area_slices(dist, sections)
                    total = sum(sl.probability for sl in slices)
                    self.assertAlmostEqual(total, 1.0, delta=1e-9)

    def test_bands_tile_the_circle_without_gaps(self):
        for sections in (3, 8, 20):
            slices, _, _ = increment_slices(Normal(0, 1), sections=sections)
            self.assertAlmostEqual(slices[0].start, 0.0, places=9)
            self.assertAlmostEqual(slices[-1].end, 360.0, places=9)
            for earlier, later in zip(slices, slices[1:]):
                self.assertAlmostEqual(earlier.end, later.start, places=12)


class TestEqualArea(unittest.TestCase):
    """Test 4: every boundary is labelled, and nothing is ticked."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_every_interval_boundary_carries_a_value(self):
        low, high = SPINNER_SECTIONS_RANGE["area"]
        for sections in range(low, high + 1):
            with self.subTest(sections=sections):
                plt.close("all")
                plt.figure()
                plot = Normal(0, 1).spinner(style="area", sections=sections)
                # One label per interior boundary, plus the two that meet at
                # the seam. Nothing is ever dropped on this dial.
                labels = [t for t in plot.ax.texts]
                self.assertEqual(len(labels), (sections - 1) + 2)

    def test_no_tick_marks_are_drawn(self):
        plt.figure()
        plot = Normal(0, 1).spinner(style="area", sections=12)
        leaders = [
            line
            for line in plot.ax.lines
            if mcolors.to_hex(line.get_color()) == mcolors.to_hex(SPINNER_LEADER_COLOR)
        ]
        self.assertEqual(leaders, [])

    def test_no_slice_dividers_are_drawn(self):
        """The uneven spacing of the numbers is the scale, not a graduation."""
        plt.figure()
        plot = Normal(0, 1).spinner(style="area", sections=12)
        self.assertEqual([p for p in plot.ax.patches if isinstance(p, Wedge)], [])

    def test_slices_are_all_the_same_chance(self):
        slices, _ = area_slices(Normal(0, 1), 12)
        for sl in slices:
            self.assertAlmostEqual(sl.probability, 1.0 / 12, places=12)
            self.assertAlmostEqual(sl.sweep, 30.0, places=12)

    def test_boundaries_are_the_quantiles(self):
        dist = Normal(0, 1)
        _, boundaries = area_slices(dist, 12)
        for i, value in enumerate(boundaries):
            if i == 0:
                continue  # quantile(0) is -inf for an unbounded distribution
            self.assertAlmostEqual(value, float(dist.quantile(i / 12)), places=9)

    def test_intervals_are_short_where_the_density_is_high(self):
        """The lesson the dial exists to teach, asserted."""
        _, boundaries = area_slices(Normal(0, 1), 12)
        widths = [b - a for a, b in zip(boundaries[1:], boundaries[2:])]
        middle = widths[len(widths) // 2]
        self.assertLess(middle, widths[0])
        self.assertLess(middle, widths[-1])

    def test_a_real_bound_is_named_and_a_tail_is_an_infinity_sign(self):
        """Settled by an exact test on quantile(0)/quantile(1), not a guess."""
        # Uniform is bounded at both ends, so both are real numbers.
        self.assertEqual(area_bound_labels(Uniform(0, 1)), ("0", "1.00"))
        # Normal is unbounded at both, so both are tails.
        self.assertEqual(area_bound_labels(Normal(0, 1)), ("−∞", "∞"))
        # Exponential is one of each -- the case a heuristic gets wrong.
        smallest, largest = area_bound_labels(Exponential(1))
        self.assertEqual(smallest, "0")  # a genuine lower bound
        self.assertEqual(largest, "∞")  # a tail

    def test_infinity_signs_are_written_upright(self):
        """Radial text at the very top reads sideways; these two do not."""
        plt.figure()
        plot = Normal(0, 1).spinner(style="area", sections=12)
        for text in plot.ax.texts:
            if text.get_text() in ("∞", "−∞"):
                self.assertEqual(text.get_rotation(), 0.0)

    def test_caption_reports_the_chance_and_the_lesson(self):
        plt.figure()
        plot = Normal(0, 1).spinner(style="area", sections=12)
        self.assertIn("8.3%", plot.caption)
        self.assertIn("1 in 12", plot.caption)
        self.assertIn("not all have the same length", plot.caption)

    def test_sections_above_the_cap_is_refused_with_the_reason(self):
        plt.figure()
        with self.assertRaises(ValueError) as caught:
            Normal(0, 1).spinner(style="area", sections=61)
        message = str(caught.exception)
        self.assertIn("60", message)
        self.assertIn("label", message)


class TestEqualIncrements(unittest.TestCase):
    """Test 5: no bare ticks, and every band boundary is a drawn tick."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_every_band_boundary_is_a_drawn_tick(self):
        low, high = SPINNER_SECTIONS_RANGE["increments"]
        for dist in CONTINUOUS:
            for sections in range(low, high + 1):
                with self.subTest(dist=type(dist).__name__, sections=sections):
                    slices, ticks, _ = increment_slices(dist, sections=sections)
                    drawn = {round(tick.bearing, 9) for tick in ticks}
                    edges = {round(sl.start, 9) for sl in slices}
                    edges |= {round(sl.end, 9) for sl in slices}
                    # 0 and 360 close the two open tails rather than sitting on
                    # a tick -- except on a distribution bounded there, where a
                    # tick lands exactly on the seam and its tail band has no
                    # width at all (Exponential's 0, Uniform's both ends).
                    interior = edges - {0.0, 360.0}
                    self.assertTrue(
                        interior <= drawn,
                        f"band boundaries with no tick: {sorted(interior - drawn)}",
                    )
                    # ...and the converse: no tick is left without a band.
                    self.assertTrue(drawn <= edges)

    def test_every_drawn_tick_carries_a_value(self):
        """A tick with nowhere to put its value is dropped, not left bare."""
        plt.figure()
        plot = Normal(0, 1).spinner(style="increments", sections=12)
        values = {t.get_text() for t in plot.ax.texts if not t.get_text().endswith("%")}
        for tick in plot.ticks:
            self.assertIn(_format_continuous(tick.value), values)

    def test_ticks_are_multiples_of_the_increment(self):
        _, ticks, step = increment_slices(Normal(0, 1), sections=8)
        for tick in ticks:
            self.assertAlmostEqual(
                tick.value / step, round(tick.value / step), places=9
            )

    def test_a_ticks_bearing_is_its_cumulative_probability(self):
        """The arc up to a tick is the chance of landing below its value."""
        dist = Normal(0, 1)
        _, ticks, _ = increment_slices(dist, sections=8)
        for tick in ticks:
            self.assertAlmostEqual(
                tick.bearing, 360.0 * float(dist.cdf(tick.value)), places=9
            )

    def test_arc_between_ticks_is_the_probability_of_that_range(self):
        dist = Normal(0, 1)
        slices, ticks, _ = increment_slices(dist, sections=8)
        for sl, (lower, upper) in zip(slices[1:-1], zip(ticks, ticks[1:])):
            expected = float(dist.cdf(upper.value)) - float(dist.cdf(lower.value))
            self.assertAlmostEqual(sl.probability, expected, places=9)

    def test_bands_are_uncoloured(self):
        """The widths carry the information; colour would only decorate."""
        plt.figure()
        plot = Normal(0, 1).spinner(style="increments", sections=8)
        fills = {
            mcolors.to_hex(w.get_facecolor())
            for w in plot.ax.patches
            if isinstance(w, Wedge)
        }
        self.assertEqual(fills, {mcolors.to_hex(SPINNER_FACE_TINT)})

    def test_percentages_only_on_bands_wide_enough(self):
        plt.figure()
        plot = Normal(0, 1).spinner(style="increments", sections=8)
        percents = [t for t in plot.ax.texts if t.get_text().endswith("%")]
        wide = [sl for sl in plot.slices if sl.sweep >= SPINNER_MIN_LABEL_SWEEP]
        self.assertEqual(len(percents), len(wide))


class TestNiceNumbersAndIncrements(unittest.TestCase):
    """The 1 / 2 / 5 ladder, and the increment= / sections= pairing."""

    def test_nice_num_climbs_the_ladder(self):
        for raw, expected in [
            (1.0, 1.0),
            (1.4, 1.0),
            (1.6, 2.0),
            (3.0, 2.0),
            (4.0, 5.0),
            (7.0, 5.0),
            (8.0, 10.0),
            (0.13, 0.1),
            (230.0, 200.0),
        ]:
            with self.subTest(raw=raw):
                self.assertAlmostEqual(nice_num(raw), expected, places=12)

    def test_nice_step_respects_the_minimum_band_count(self):
        self.assertAlmostEqual(nice_step(6.0, 1), nice_step(6.0, 3), places=12)

    def test_quarter_steps_are_offered(self):
        """2.5 must be in the ladder, or 0.25 is never offered."""
        options = increment_options(1.0)
        self.assertIn(0.25, [round(o, 10) for o in options])
        wide = [round(o, 10) for o in increment_options(10.0)]
        self.assertIn(2.5, wide)

    def test_options_all_give_a_sensible_band_count(self):
        span = 6.18
        for step in increment_options(span):
            bands = span / step
            self.assertGreaterEqual(bands, 3)
            self.assertLessEqual(bands, 40)

    def test_explicit_increment_sets_the_band_length(self):
        _, ticks, step = increment_slices(Normal(0, 1), increment=0.5)
        self.assertAlmostEqual(step, 0.5, places=12)
        for tick in ticks:
            self.assertAlmostEqual(tick.value * 2, round(tick.value * 2), places=9)

    def test_increment_and_sections_together_are_refused(self):
        plt.close("all")
        plt.figure()
        with self.assertRaises(ValueError) as caught:
            Normal(0, 1).spinner(sections=8, increment=0.5)
        self.assertIn("not both", str(caught.exception))
        plt.close("all")

    def test_increment_on_an_equal_area_dial_is_refused(self):
        plt.close("all")
        plt.figure()
        with self.assertRaises(ValueError) as caught:
            Normal(0, 1).spinner(style="area", increment=0.5)
        self.assertIn("sections", str(caught.exception))
        plt.close("all")

    def test_an_absurdly_small_increment_is_refused_with_suggestions(self):
        with self.assertRaises(ValueError) as caught:
            increment_slices(Normal(0, 1), increment=1e-4)
        self.assertIn("too small", str(caught.exception))

    def test_a_non_positive_increment_is_refused(self):
        with self.assertRaises(ValueError):
            increment_slices(Normal(0, 1), increment=0)
        with self.assertRaises(ValueError):
            increment_slices(Normal(0, 1), increment=-1.0)


class TestContinuousDialDispatch(unittest.TestCase):
    """Style and section resolution, settled before anything is drawn."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_increments_is_the_default_style(self):
        plt.figure()
        self.assertEqual(Normal(0, 1).spinner().style, "increments")

    def test_default_sections_per_style(self):
        plt.figure()
        self.assertEqual(Normal(0, 1).spinner().sections, 8)
        plt.close("all")
        plt.figure()
        self.assertEqual(Normal(0, 1).spinner(style="area").sections, 12)

    def test_a_discrete_distribution_refuses_a_style(self):
        plt.figure()
        with self.assertRaises(ValueError) as caught:
            Binomial(10, 0.5).spinner(style="area")
        self.assertIn("one spinner dial", str(caught.exception))

    def test_a_discrete_distribution_refuses_sections(self):
        plt.figure()
        with self.assertRaises(ValueError) as caught:
            Binomial(10, 0.5).spinner(sections=8)
        self.assertIn("one slice per outcome", str(caught.exception))

    def test_an_unknown_style_names_the_two_that_exist(self):
        plt.figure()
        with self.assertRaises(ValueError) as caught:
            Normal(0, 1).spinner(style="equal")
        message = str(caught.exception)
        self.assertIn("increments", message)
        self.assertIn("area", message)

    def test_a_refused_setting_leaves_the_figure_clean(self):
        """Everything is resolved before a patch is drawn."""
        plt.figure()
        with self.assertRaises(ValueError):
            Normal(0, 1).spinner(style="area", sections=999)
        self.assertEqual(plt.gcf().axes, [])

    def test_display_range_reuses_the_distributions_own_window(self):
        for dist in CONTINUOUS:
            with self.subTest(dist=type(dist).__name__):
                self.assertEqual(
                    display_range(dist), tuple(float(v) for v in dist.xlim)
                )

    def test_a_continuous_spin_lands_on_a_finite_value(self):
        plt.figure()
        plot = Normal(0, 1).spinner(seed=3)
        self.assertTrue(np.isfinite(plot.value))
        self.assertTrue(plt.gcf().axes)

    def test_continuous_round_trip(self):
        """Test 8: spinner(value=v) points where a spin landing on v would."""
        dist = Normal(0, 1)
        for value in (-1.5, -0.25, 0.0, 0.8, 2.0):
            plt.close("all")
            plt.figure()
            plot = dist.spinner(value=value)
            self.assertAlmostEqual(spin_value(dist, plot.bearing), value, places=9)

    def test_the_seam_is_marked_on_both_continuous_dials(self):
        for style in ("increments", "area"):
            with self.subTest(style=style):
                plt.close("all")
                plt.figure()
                plot = Normal(0, 1).spinner(style=style)
                seams = [
                    line
                    for line in plot.ax.lines
                    if mcolors.to_hex(line.get_color())
                    == mcolors.to_hex(SPINNER_SEAM_COLOR)
                ]
                self.assertEqual(len(seams), 1)

    def test_the_dial_does_not_rotate_with_the_value(self):
        plt.figure()
        first = [t.get_text() for t in Normal(0, 1).spinner(value=-1.0).ax.texts]
        plt.close("all")
        plt.figure()
        second = [t.get_text() for t in Normal(0, 1).spinner(value=2.0).ax.texts]
        self.assertEqual(first, second)

    def test_caption_is_not_written_onto_a_supplied_axes(self):
        """A caller's own layout is not ours to add margins to."""
        fig, ax = plt.subplots()
        plot = Normal(0, 1).spinner(style="area", ax=ax)
        self.assertIsNotNone(plot.caption)  # still reported...
        self.assertEqual(fig.texts, [])  # ...but not drawn


class TestPaletteComesFromTheStyleFile(unittest.TestCase):
    """The palette is read from symbulate.mplstyle, not retyped."""

    def test_palette_matches_the_prop_cycle(self):
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.assertEqual(list(SPINNER_PALETTE), list(cycle))

    def test_sky_blue_leads(self):
        self.assertEqual(SPINNER_PALETTE[0].upper(), "#56B4E9")


if __name__ == "__main__":
    unittest.main()
