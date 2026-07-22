"""Tests for symbulate plotting functionality — dev branch.

Covers:
  - 1D RVResults: impulse, hist, density, rug, and combinations
  - 2D RVResults: scatter, hist, density, tile, violin, marginal
  - Distribution.plot() for continuous and discrete distributions
  - Stochastic process path plots (Markov chains, Poisson process)
  - Normalization correctness (histogram area, impulse heights)
  - Axis labels, xlim bounds, and figure structure
  - Error handling (plotting from a bare ProbabilitySpace)

Tests seed numpy's global generator (np.random.seed) before simulating,
matching the convention in test_distributions.py, so the simulated data
(and with it the classify_values-driven default plot type) is stable from
run to run.
"""

import contextlib
import importlib
import io
import unittest
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, PathCollection

from symbulate import (
    RV,
    Normal,
    Binomial,
    Bernoulli,
    Geometric,
    Poisson,
    Exponential,
    Gamma,
    Beta,
    Uniform,
    DiscreteUniform,
    MultivariateNormal,
    Multinomial,
    BivariateNormal,
    BoxModel,
    MarkovChain,
    PoissonProcess,
    ProbabilitySpace,
    cos,
    pi,
)
from symbulate import plot as symbulate_plot
from symbulate.plot import (
    SymbulatePlot,
    B_1D,
    K_2D,
    N_SMALL_THRESHOLD,
    DOTPLOT_MAX_STACK,
    classify_values,
    default_plot_type,
    dotplot_tallest_stack,
    get_next_color,
    suggestion_message,
    should_show_suggestion,
    make_bar,
    BAR_ALPHA,
    make_dotplot,
    make_impulse,
    make_violin,
    make_violinplot,
    make_ecdf,
    make_mosaic,
    _mosaic_spans,
    _readable_text_color,
    make_sample_path,
    make_segmented_density,
    make_segmented_hist,
    make_tile,
    make_segmented_rug,
    make_grouped_boxplot,
    _thin_discrete_ticks,
    MAX_DISCRETE_TICKS,
    DEFAULT_PLOT_TYPE,
    PLOT_DISPLAY_NAME,
    SAMPLE_PATH_ALPHA,
    SAMPLE_PATH_LINEWIDTH,
    TILE_DEFAULT_BINS,
    HIST_DEFAULT_BINS,
)
from symbulate.results import RVResults

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def histogram_area(ax):
    """Return the total area (width × height) of all histogram bars on ax."""
    return sum(p.get_width() * p.get_height() for p in ax.patches)


def visible_ticks(ax, axis="x"):
    """Tick locations actually within the axis's current view limits.

    ``ax.get_xticks()``/``get_yticks()`` include candidate locations a
    locator considered just outside the visible range (never drawn), so
    tests that care about what a viewer would actually see filter to this
    instead.
    """
    lim = ax.get_xlim() if axis == "x" else ax.get_ylim()
    ticks = ax.get_xticks() if axis == "x" else ax.get_yticks()
    return [t for t in ticks if lim[0] <= t <= lim[1]]


class PlotTestCase(unittest.TestCase):
    """Base class: closes all open figures after every test."""

    def tearDown(self):
        plt.close("all")


# ===========================================================================
# 1D discrete RVResults
# ===========================================================================


class TestPlot1DDiscrete(PlotTestCase):
    """Plots of 1D discrete RVResults (default type: impulse)."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Binomial(n=10, p=0.4)).sim(500)

    def test_default_type_is_impulse_not_histogram(self):
        """Discrete data should produce vlines (LineCollection), not bars."""
        self.sims.plot()
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0, "Expected vlines collection")
        self.assertEqual(len(ax.patches), 0, "Unexpected histogram bars")

    def test_impulse_ylabel_relative_frequency(self):
        self.sims.plot()
        self.assertEqual(plt.gca().get_ylabel(), "Relative Frequency")

    def test_impulse_normalize_false_ylabel_count(self):
        self.sims.plot(normalize=False)
        self.assertEqual(plt.gca().get_ylabel(), "Count")

    def test_impulse_heights_sum_to_one(self):
        """With normalize=True, the vline heights must sum to exactly 1.0."""
        self.sims.plot(normalize=True)
        segs = plt.gca().collections[0].get_segments()
        heights = [abs(seg[1][1] - seg[0][1]) for seg in segs]
        self.assertAlmostEqual(sum(heights), 1.0, places=10)

    def test_explicit_impulse_type(self):
        self.sims.plot(type="impulse")
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0)
        self.assertEqual(len(ax.patches), 0)

    def test_impulse_with_jitter_warns(self):
        """jitter has no effect on impulse plots; passing it should warn."""
        with self.assertWarns(UserWarning):
            self.sims.plot(jitter=True)

    def test_impulse_with_jitter_stems_unperturbed(self):
        """Despite the warning, stems still land at the exact integer values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.sims.plot(jitter=True)
        segs = plt.gca().collections[0].get_segments()
        xs = [seg[0][0] for seg in segs]
        self.assertTrue(all(float(x).is_integer() for x in xs))

    def test_impulse_has_no_marker_dots(self):
        """Impulse plots draw bare stems only -- no scatter/marker dots."""
        self.sims.plot(type="impulse")
        ax = plt.gca()
        self.assertFalse(any(isinstance(c, PathCollection) for c in ax.collections))

    def test_impulse_legend_label_survives_dot_removal(self):
        """The series label (now on the stems, not a marker) still reaches
        the legend once a second series overlays the first."""
        self.sims.plot(type="impulse", label="First")
        RV(Binomial(n=10, p=0.6)).sim(500).plot(type="impulse", label="Second")
        legend_labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertEqual(legend_labels, ["First", "Second"])


# ===========================================================================
# Dot-plot tall-stack fallback
# ===========================================================================


class TestDotplotTallStackFallback(PlotTestCase):
    """A dot plot whose tallest stack is too tall falls back to a default.

    The dot plot is the small-n default for discrete data, but it stacks
    one dot per observation, so a single tall stack (e.g. a rare-event
    indicator) shrinks every dot to a speck. When the tallest stack
    exceeds ``DOTPLOT_MAX_STACK``, the default is redirected -- to the
    impulse plot for numeric data and the bar chart for categorical --
    while an explicit ``type='dotplot'`` is still honored.
    """

    def test_tallest_stack_helper_counts_most_repeated_value(self):
        self.assertEqual(dotplot_tallest_stack(np.array([0, 0, 0, 1, 2])), 3)
        self.assertEqual(dotplot_tallest_stack(np.array(["H", "H", "T"])), 2)

    def test_tallest_stack_helper_empty_is_zero(self):
        self.assertEqual(dotplot_tallest_stack(np.array([])), 0)

    def test_binomial_indicator_small_n_falls_back_to_impulse(self):
        """RV(Binomial(1, 0.1)).sim(111): n is small (< 123) so the discrete
        default would be a dot plot, but ~90% of the mass lands on one value,
        so the tallest stack blows past DOTPLOT_MAX_STACK and it renders as an
        impulse plot instead."""
        np.random.seed(42)
        sims = RV(Binomial(1, 0.1)).sim(111)
        self.assertLess(len(sims), N_SMALL_THRESHOLD)  # would be small-n
        self.assertGreater(
            dotplot_tallest_stack(np.asarray(sims.results)), DOTPLOT_MAX_STACK
        )
        sims.plot(suggest=False)
        ax = plt.gca()
        self.assertIn("Impulse", ax.get_title())
        self.assertNotEqual(ax.get_title(), "Dot Plot")

    def test_fallback_suggestion_still_offers_dotplot(self):
        """The suggestion note names the impulse default but keeps the dot
        plot as an alternative students can still ask for."""
        np.random.seed(42)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            RV(Binomial(1, 0.1)).sim(111).plot(suggest=True)
        note = buf.getvalue()
        self.assertIn("Impulse Plot (Default)", note)
        self.assertIn('Dot Plot (type = "dotplot")', note)

    def test_explicit_dotplot_is_honored_despite_tall_stack(self):
        """An explicit type='dotplot' request draws a dot plot regardless of
        stack height -- only the automatic default is redirected."""
        np.random.seed(42)
        RV(Binomial(1, 0.1)).sim(111).plot(type="dotplot", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Dot Plot")

    def test_short_stacks_small_n_stays_a_dotplot(self):
        """A genuine small-n discrete sample with short stacks (a fair die)
        keeps the dot plot default."""
        np.random.seed(42)
        sims = RV(BoxModel([1, 2, 3, 4, 5, 6])).sim(60)
        self.assertLessEqual(
            dotplot_tallest_stack(np.asarray(sims.results)), DOTPLOT_MAX_STACK
        )
        sims.plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Dot Plot")

    def test_categorical_tall_stack_falls_back_to_bar(self):
        """Categorical data whose few categories each stack tall redirects to
        the large-n categorical default (a bar chart)."""
        np.random.seed(42)
        sims = RV(BoxModel(["H", "T"])).sim(111)
        self.assertGreater(
            dotplot_tallest_stack(np.asarray(sims.results)), DOTPLOT_MAX_STACK
        )
        sims.plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Bar Chart")


# ===========================================================================
# 1D continuous RVResults
# ===========================================================================


class TestPlot1DContinuous(PlotTestCase):
    """Plots of 1D continuous RVResults (default type: hist)."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(600)

    def test_default_type_is_histogram(self):
        self.sims.plot()
        self.assertGreater(len(plt.gca().patches), 0)

    def test_hist_ylabel_density_when_normalized(self):
        self.sims.plot(normalize=True)
        self.assertEqual(plt.gca().get_ylabel(), "Density")

    def test_hist_ylabel_count_when_not_normalized(self):
        self.sims.plot(normalize=False)
        self.assertEqual(plt.gca().get_ylabel(), "Count")

    def test_hist_normalized_area_integrates_to_one(self):
        """A normalized histogram's bars must integrate to approximately 1."""
        self.sims.plot(normalize=True, bins=30)
        self.assertAlmostEqual(histogram_area(plt.gca()), 1.0, places=5)

    def test_hist_custom_bin_count(self):
        self.sims.plot(bins=50)
        self.assertEqual(len(plt.gca().patches), 50)

    def test_bar_type_produces_bar_chart_not_hist(self):
        """type='bar' is now a distinct bar chart, no longer a hist alias."""
        self.sims.plot(type="bar", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Bar Chart")

    def test_density_produces_exactly_one_line(self):
        self.sims.plot(type="density")
        self.assertEqual(len(plt.gca().lines), 1)

    def test_density_ylabel(self):
        self.sims.plot(type="density")
        self.assertEqual(plt.gca().get_ylabel(), "Density")

    def test_density_line_is_everywhere_non_negative(self):
        self.sims.plot(type="density")
        y = plt.gca().lines[0].get_ydata()
        self.assertTrue(np.all(y >= 0), "Density estimate has negative values")

    def test_density_line_is_everywhere_finite(self):
        self.sims.plot(type="density")
        y = plt.gca().lines[0].get_ydata()
        self.assertTrue(np.all(np.isfinite(y)))

    def test_rug_produces_tick_collection(self):
        """Rug ticks are drawn as vlines, which form a LineCollection."""
        self.sims.plot(type="rug")
        self.assertGreater(len(plt.gca().collections), 0)

    def test_standalone_rug_hides_y_axis(self):
        """A lone rug is a number line: no y-axis, no left spine."""
        self.sims.plot(type="rug")
        ax = plt.gca()
        self.assertFalse(ax.yaxis.get_visible())
        self.assertFalse(ax.spines["left"].get_visible())

    def test_density_plus_rug_produces_line_and_ticks(self):
        self.sims.plot(type=("density", "rug"))
        ax = plt.gca()
        self.assertEqual(len(ax.lines), 1)  # the density curve
        self.assertGreater(len(ax.collections), 0)  # the rug ticks

    def test_rug_overlaid_on_density_keeps_y_axis(self):
        """When the rug accompanies another plot, the axes are untouched."""
        self.sims.plot(type=("density", "rug"))
        self.assertTrue(plt.gca().yaxis.get_visible())

    def test_hist_plus_rug_produces_bars_and_ticks(self):
        self.sims.plot(type=("hist", "rug"))
        ax = plt.gca()
        self.assertGreater(len(ax.patches), 0)
        self.assertGreater(len(ax.collections), 0)

    def test_two_overlaid_plots_accumulate_patches(self):
        """Plotting twice on the same axes should add more bars."""
        self.sims.plot()
        n_first = len(plt.gca().patches)
        RV(Normal(0, 1)).sim(600).plot()
        self.assertGreater(len(plt.gca().patches), n_first)


# ===========================================================================
# Numerical precision: near-constant data (e.g. X * cos(pi/2))
# ===========================================================================


class TestNumericalPrecision(PlotTestCase):
    """Plots of RVs whose simulated values are numerically near-constant.

    The motivating case (from a 2017 bug report):
        X = RV(Normal(0, 1))
        Y = X * cos(pi / 2)
    Y should be 0 with probability 1, but cos(pi/2) ≈ 6.12e-17 in floating
    point, so every simulated value is O(1e-17) rather than exactly 0.
    Without the fix the histogram x-axis displays a 1e-16 scale, which is
    visually indistinguishable from (and misleadingly different from) a point
    mass at 0.

    After the fix, near-constant data (range < 1e-9 × scale) is collapsed to
    its effective mean before plotting and rendered as a single impulse.
    """

    # ------------------------------------------------------------------
    # Core scenario: Y = X * cos(pi/2)
    # ------------------------------------------------------------------

    def test_near_zero_constant_produces_impulse_not_hist(self):
        """Y = X * cos(pi/2) must plot as a single impulse, not a histogram.

        cos(pi/2) ≈ 6e-17, so Y is numerically indistinguishable from 0.
        Impulse plots produce a LineCollection (vlines); histograms produce
        patches (bars).
        """
        X = RV(Normal(0, 1))
        Y = X * cos(pi / 2)
        Y.sim(1000).plot()
        ax = plt.gca()
        self.assertGreater(
            len(ax.collections),
            0,
            "Expected an impulse (vlines / LineCollection) for near-zero data",
        )
        self.assertEqual(
            len(ax.patches),
            0,
            "Histogram bars should not appear for near-zero near-constant data",
        )

    def test_near_zero_constant_impulse_placed_at_zero(self):
        """The single impulse must be at x=0, not at x≈6e-17."""
        X = RV(Normal(0, 1))
        Y = X * cos(pi / 2)
        Y.sim(1000).plot()
        segs = plt.gca().collections[0].get_segments()
        xs = [seg[0][0] for seg in segs]
        self.assertEqual(
            len(xs), 1, "Expected exactly one impulse for a point mass at 0"
        )
        self.assertAlmostEqual(xs[0], 0.0, places=10)

    def test_near_zero_constant_impulse_height_is_one(self):
        """With normalize=True the single impulse must have height 1.0."""
        X = RV(Normal(0, 1))
        Y = X * cos(pi / 2)
        Y.sim(500).plot(normalize=True)
        segs = plt.gca().collections[0].get_segments()
        total_height = sum(abs(seg[1][1] - seg[0][1]) for seg in segs)
        self.assertAlmostEqual(total_height, 1.0, places=10)

    def test_near_zero_xlim_is_not_degenerate(self):
        """x-axis must span a human-readable range, not collapse to ~1e-16."""
        X = RV(Normal(0, 1))
        Y = X * cos(pi / 2)
        Y.sim(500).plot()
        xlim = plt.gca().get_xlim()
        self.assertGreater(
            xlim[1] - xlim[0],
            0.1,
            f"x-axis range {xlim[1] - xlim[0]:.2e} is too small (floating-point artifact?)",
        )

    def test_near_zero_xlim_contains_zero(self):
        """Zero must lie within the plotted x-axis range."""
        X = RV(Normal(0, 1))
        Y = X * cos(pi / 2)
        Y.sim(500).plot()
        xmin, xmax = plt.gca().get_xlim()
        self.assertLessEqual(xmin, 0.0)
        self.assertGreaterEqual(xmax, 0.0)

    # ------------------------------------------------------------------
    # Exactly-zero data (X * 0 gives all values == 0.0 exactly)
    # ------------------------------------------------------------------

    def test_exactly_zero_data_produces_impulse(self):
        """If all simulated values are exactly 0, show a point mass at 0."""
        X = RV(Normal(0, 1))
        Y = X * 0
        Y.sim(500).plot()
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0)
        self.assertEqual(len(ax.patches), 0)

    def test_exactly_zero_impulse_placed_at_zero(self):
        X = RV(Normal(0, 1))
        (X * 0).sim(500).plot()
        segs = plt.gca().collections[0].get_segments()
        xs = [seg[0][0] for seg in segs]
        self.assertEqual(len(xs), 1)
        self.assertAlmostEqual(xs[0], 0.0, places=10)

    # ------------------------------------------------------------------
    # Near-constant data centered away from zero
    # ------------------------------------------------------------------

    def test_near_constant_nonzero_produces_impulse(self):
        """X * 1e-12 + 5 is near-constant around 5; must plot as impulse, not hist."""
        X = RV(Normal(0, 1))
        Y = X * 1e-12 + 5
        Y.sim(500).plot()
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0)
        self.assertEqual(len(ax.patches), 0)

    def test_near_constant_nonzero_impulse_at_center(self):
        """The impulse for X * 1e-12 + 5 should be placed at x≈5."""
        X = RV(Normal(0, 1))
        Y = X * 1e-12 + 5
        Y.sim(500).plot()
        segs = plt.gca().collections[0].get_segments()
        xs = [seg[0][0] for seg in segs]
        self.assertEqual(len(xs), 1)
        self.assertAlmostEqual(xs[0], 5.0, places=5)

    # ------------------------------------------------------------------
    # Regression guards: distributions with genuine spread are unaffected
    # ------------------------------------------------------------------

    def test_normal_distribution_still_plots_as_histogram(self):
        """Normal(0, 1) has large spread; the precision fix must not trigger."""
        RV(Normal(0, 1)).sim(600).plot()
        ax = plt.gca()
        self.assertGreater(
            len(ax.patches),
            0,
            "Normal(0,1) should still produce a histogram after the precision fix",
        )

    def test_discrete_distribution_unaffected(self):
        """Binomial results have genuine spread; precision fix must not touch them."""
        RV(Binomial(n=10, p=0.5)).sim(500).plot()
        ax = plt.gca()
        # Discrete data auto-selects impulse, but via classify_values, not ours
        self.assertGreater(len(ax.collections), 0)
        self.assertEqual(len(ax.patches), 0)

    def test_exponential_distribution_still_plots_as_histogram(self):
        """Exponential(1) has large spread; precision fix must not trigger."""
        RV(Exponential(rate=1)).sim(600).plot()
        self.assertGreater(len(plt.gca().patches), 0)

    def test_small_but_real_uniform_still_plots_as_histogram(self):
        """Uniform(0, 0.001) has spread ~0.001 >> 1e-9; must remain a histogram."""
        RV(Uniform(a=0, b=0.001)).sim(600).plot()
        self.assertGreater(len(plt.gca().patches), 0)


class TestPlot1DOtherDistributions(PlotTestCase):
    """Smoke tests ensuring other continuous/discrete distributions plot cleanly."""

    def setUp(self):
        np.random.seed(42)

    def test_exponential_default(self):
        RV(Exponential(rate=1)).sim(300).plot()
        self.assertGreater(len(plt.gca().patches), 0)

    def test_poisson_default_is_impulse(self):
        """Poisson results should default to an impulse plot.

        classify_values counts distinct values (Poisson(3) has far fewer
        than N_UNIQUE_THRESHOLD), so the discrete determination is
        stable at any n — unlike the old is_discrete(), whose singleton
        budget made this test flaky until n was raised.
        """
        RV(Poisson(lam=3)).sim(5000).plot()
        self.assertGreater(len(plt.gca().collections), 0)
        self.assertEqual(len(plt.gca().patches), 0)

    def test_gamma_density(self):
        RV(Gamma(shape=2, rate=1)).sim(300).plot(type="density")
        self.assertEqual(len(plt.gca().lines), 1)


# ===========================================================================
# 2D RVResults
# ===========================================================================


class TestPlot2DContinuous(PlotTestCase):
    """2D plots of continuous joint distributions."""

    def setUp(self):
        np.random.seed(42)
        X, Y = RV(Normal(0, 1) ** 2)
        self.sims = (X & Y).sim(500)

    def test_default_large_n_is_hist2d(self):
        """Per DEFAULT_PLOT_TYPE, large-n continuous 2D data defaults to
        a 2D histogram (a QuadMesh with a Density colorbar)."""
        self.sims.plot()  # 500 pairs
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0)
        self.assertEqual(len(ax.patches), 0)
        self.assertEqual(plt.gcf().axes[-1].get_ylabel(), "Density")

    def test_default_small_n_is_scatter(self):
        """Per DEFAULT_PLOT_TYPE, small-n continuous 2D data defaults to
        a scatter plot."""
        X, Y = RV(Normal(0, 1) ** 2)
        (X & Y).sim(40).plot()
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "2D Scatter Plot")
        self.assertEqual(len(ax.collections[0].get_offsets()), 40)

    def test_scatter_explicit(self):
        self.sims.plot(type="scatter")
        self.assertGreater(len(plt.gca().collections), 0)

    def test_hist2d_normalize_false(self):
        """2D histogram with normalize=False must produce a QuadMesh."""
        self.sims.plot(type="hist", normalize=False)
        # hist2d produces a QuadMesh, which shows up as a Collection
        self.assertGreater(len(plt.gca().collections), 0)

    def test_density_2d_produces_mesh_or_image(self):
        """2D density uses imshow (AxesImage) or a mesh."""
        self.sims.plot(type="density")
        ax = plt.gca()
        visual_elements = ax.collections + ax.images
        self.assertGreater(len(visual_elements), 0)

    def test_marginal_creates_three_subplot_axes(self):
        """Marginal plot needs a main axes plus two marginal axes (3 total).

        Note: add_colorbar() adds a fourth axes in some plot types, so we
        assert >= 3 rather than == 3.
        """
        self.sims.plot(marginal=True)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)

    def test_marginal_density_creates_at_least_three_axes(self):
        self.sims.plot(type="density", marginal=True)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)

    def test_marginal_true_with_default_type_draws_main_panel(self):
        """marginal=True with no explicit type= must still resolve type
        to the data's default (2D histogram here) and draw it on the
        main panel, rather than leaving the center panel blank."""
        p = self.sims.plot(marginal=True)
        self.assertGreater(len(p.ax.collections), 0)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)

    def test_marginal_main_panel_keeps_xy_labels_no_value_labels(self):
        """The main panel keeps its "X"/"Y" labels, and neither marginal
        panel shows a redundant "Value" axis label next to it."""
        for main_type in ["hist", "tile", "density"]:
            p = self.sims.plot(type=main_type, marginal=True)
            self.assertEqual(p.ax.get_xlabel(), "Variable 1")
            self.assertEqual(p.ax.get_ylabel(), "Variable 2")
            for a in plt.gcf().axes:
                self.assertNotEqual(a.get_xlabel(), "Value")
                self.assertNotEqual(a.get_ylabel(), "Value")
            plt.close("all")

    def test_marginal_gridlines_follow_frequency_axis(self):
        """Each marginal panel shows gridlines only along its frequency
        axis: the top (x) marginal horizontal (y-axis) gridlines, the
        right (y) marginal vertical (x-axis) gridlines."""

        def xvis(a):
            return any(gl.get_visible() for gl in a.get_xgridlines())

        def yvis(a):
            return any(gl.get_visible() for gl in a.get_ygridlines())

        p = self.sims.plot(type="hist", marginal=True)
        # Panels are the non-main, non-colorbar axes (the colorbar caxes is
        # a narrow strip); the top marginal sits highest, the right one
        # sits furthest right.
        panels = [
            a for a in plt.gcf().axes if a is not p.ax and a.get_position().width > 0.1
        ]
        marg_x = max(panels, key=lambda a: a.get_position().y0)
        marg_y = max(panels, key=lambda a: a.get_position().x0)
        self.assertTrue(yvis(marg_x))  # top: horizontal only
        self.assertFalse(xvis(marg_x))
        self.assertTrue(xvis(marg_y))  # right: vertical only
        self.assertFalse(yvis(marg_y))


class TestMarginalOverlayHardError(PlotTestCase):
    """A marginal=True layout is the 'hard error' tier of the overlay
    policy: it builds three panels a second plot can't share."""

    def setUp(self):
        np.random.seed(42)
        X, Y = RV(Normal(0, 1) ** 2)
        self.sims = (X & Y).sim(300)

    def test_second_plot_after_marginal_raises(self):
        self.sims.plot(marginal=True, suggest=False)
        with self.assertRaises(ValueError) as cm:
            self.sims.plot(suggest=False)
        self.assertIn("marginal=True", str(cm.exception))

    def test_second_marginal_after_marginal_raises(self):
        self.sims.plot(marginal=True, suggest=False)
        with self.assertRaises(ValueError):
            self.sims.plot(marginal=True, suggest=False)

    def test_marginal_after_plain_plot_raises(self):
        """A marginal layout can't be retrofitted onto a figure that
        already has a plot on it either."""
        self.sims.plot(suggest=False)
        with self.assertRaises(ValueError) as cm:
            self.sims.plot(marginal=True, suggest=False)
        self.assertIn("marginal=True", str(cm.exception))

    def test_fresh_marginal_still_works(self):
        """The guard must not block a marginal plot on a fresh figure."""
        p = self.sims.plot(marginal=True, suggest=False)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)
        self.assertGreater(len(p.ax.collections), 0)

    def test_marginal_in_new_figure_after_close_works(self):
        """Closing the figure (a fresh Jupyter cell) clears the guard."""
        self.sims.plot(marginal=True, suggest=False)
        plt.close("all")
        self.sims.plot(marginal=True, suggest=False)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)


class TestPlot2DDiscrete(PlotTestCase):
    """2D plots involving discrete dimensions."""

    def setUp(self):
        np.random.seed(42)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        self.sims = (X & Y).sim(500)

    def test_tile_produces_a_mesh(self):
        """Tile plot uses matshow (renders as a QuadMesh or AxesImage)."""
        self.sims.plot(type="tile")
        ax = plt.gca()
        visual_elements = ax.collections + ax.images
        self.assertGreater(len(visual_elements), 0)

    def test_scatter_works_on_discrete_data(self):
        self.sims.plot(type="scatter")
        self.assertGreater(len(plt.gca().collections), 0)


class TestPlot2DViolin(PlotTestCase):
    """Violin plots require one discrete and one continuous dimension."""

    def setUp(self):
        np.random.seed(42)

    def test_violin_discrete_x_continuous_y(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin")
        self.assertGreater(len(plt.gca().collections), 0)

    def test_violin_continuous_x_discrete_y(self):
        X, Y = RV(Normal(0, 1) * Binomial(5, 0.4))
        sims = (X & Y).sim(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin")
        self.assertGreater(len(plt.gca().collections), 0)

    def test_violin_title_and_labels(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Violin Plot")
        self.assertEqual(ax.get_xlabel(), "Variable 1")
        self.assertEqual(ax.get_ylabel(), "Variable 2")

    def test_violin_body_uses_okabe_ito_color_not_default(self):
        """Violin bodies should pick up the color cycle, not mpl's own default."""
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin")
        ax = plt.gca()
        facecolor = tuple(ax.collections[0].get_facecolor()[0])
        # matplotlib's own unstyled violinplot default is a shade of purple
        # ((0.267, 0.005, 0.329, ...) from the 'viridis'-adjacent default);
        # Okabe-Ito sky blue is (0.337, 0.706, 0.914, ...).
        self.assertAlmostEqual(facecolor[0], 0.337, places=2)
        self.assertAlmostEqual(facecolor[1], 0.706, places=2)

    def test_violin_has_inner_boxplot(self):
        """Every violin carries a narrow inner boxplot for median/IQR."""
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin")
        ax = plt.gca()
        self.assertGreater(len(ax.patches), 0)

    def test_violin_discrete_axis_tick_labels_are_the_category_values(self):
        """The discrete axis is labeled with the actual category values,
        not the boxplot overlay's own 1..n position defaults."""
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin")
        ax = plt.gca()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        self.assertEqual(labels, ["0.0", "1.0", "2.0", "3.0", "4.0", "5.0"])

    def test_violin_overlay_prints_warning(self):
        import io
        import contextlib

        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin", suggest=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sims.plot(type="violin", suggest=False)
        self.assertIn("second violin plot", buf.getvalue())

    def test_violin_tick_position_matches_its_own_label(self):
        """Regression test: tick marks must sit at the same x-position as
        the violin body they label, not one unit off. Binomial(5, 0.4)'s
        support (0-5) happens to make the old positions + 1 bug
        invisible, so this uses a die roll (1-6) instead, where the old
        bug placed every tick one unit away from its violin."""
        X, Y = RV(BoxModel([1, 2, 3, 4, 5, 6]) * Normal(0, 1))
        sims = (X & Y).sim(300)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin")
        ax = plt.gca()
        xticks = list(ax.get_xticks())
        labels = [float(t.get_text()) for t in ax.get_xticklabels() if t.get_text()]
        self.assertEqual(xticks, labels)

    def test_violin_categorical_group_labels_do_not_crash(self):
        """Regression test: a categorical (non-numeric) discrete axis used
        to crash with a numpy UFuncTypeError from `positions + 1`."""
        from symbulate.plot import make_violin, get_next_color

        data = np.column_stack(
            [
                np.array(["H"] * 20 + ["T"] * 20, dtype=object),
                np.random.normal(0, 1, 40),
            ]
        )
        ax = plt.gca()
        make_violin(data, ["H", "T"], ax, get_next_color(ax), "x", 0.5)
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        self.assertEqual(labels, ["H", "T"])

    def test_violin_two_discrete_raises_friendly_error(self):
        """Regression test: two discrete variables used to silently draw
        nothing instead of raising, unlike make_grouped_boxplot."""
        X, Y = RV(Binomial(5, 0.4) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="violin")
        self.assertIn("tile", str(cm.exception))

    def test_violin_two_continuous_raises_friendly_error(self):
        """Regression test: two continuous variables used to silently draw
        nothing instead of raising, unlike make_grouped_boxplot."""
        X, Y = RV(Normal(0, 1) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="violin")
        self.assertIn("scatter", str(cm.exception))


# ===========================================================================
# New integrated plot types and options (graphics overhaul, Phase 2)
# ===========================================================================


class TestPlot1DHistStyling(PlotTestCase):
    """Styling and overlay behavior of the integrated make_hist."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(600)

    def test_hist_default_alpha(self):
        """Histogram bars default to HIST_ALPHA (0.65)."""
        self.sims.plot()
        self.assertAlmostEqual(plt.gca().patches[0].get_alpha(), 0.65)

    def test_hist_explicit_alpha_wins(self):
        self.sims.plot(alpha=0.3)
        self.assertAlmostEqual(plt.gca().patches[0].get_alpha(), 0.3)

    def test_hist_xlabel_and_title(self):
        self.sims.plot()
        ax = plt.gca()
        self.assertEqual(ax.get_xlabel(), "Value")
        self.assertEqual(ax.get_title(), "Density Histogram")

    def test_hist_count_title_when_not_normalized(self):
        self.sims.plot(normalize=False)
        self.assertEqual(plt.gca().get_title(), "Count Histogram")

    def test_hist_user_edgecolor_does_not_raise(self):
        """edgecolor= used to flow straight into ax.hist; it still must."""
        self.sims.plot(edgecolor="black")
        self.assertGreater(len(plt.gca().patches), 0)

    def test_step_histogram_outline_is_not_white(self):
        """histtype='step' draws only its outline, so the white-edge
        default must not apply -- it would be invisible on the white
        background."""
        self.sims.plot(histtype="step")
        edge = plt.gca().patches[0].get_edgecolor()
        self.assertNotEqual(tuple(edge[:3]), (1.0, 1.0, 1.0))

    def test_single_hist_has_no_legend(self):
        self.sims.plot()
        self.assertIsNone(plt.gca().get_legend())

    def test_overlaid_hists_get_variable_k_legend(self):
        self.sims.plot()
        RV(Normal(3, 1)).sim(600).plot()
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(labels, ["Variable 1", "Variable 2"])

    def test_hist_label_override(self):
        self.sims.plot()
        RV(Normal(3, 1)).sim(600).plot(label="Second")
        labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
        self.assertIn("Second", labels)


class TestPlot1DBar(PlotTestCase):
    """The integrated make_bar: a no-binning categorical/discrete bar chart."""

    def setUp(self):
        np.random.seed(42)

    def test_bar_categorical_one_bar_per_value(self):
        """One bar per distinct category; normalized heights sum to 1."""
        ax = plt.gca()
        make_bar(
            ["red", "green", "blue", "red", "red", "green"], ax, get_next_color(ax)
        )
        self.assertEqual(len(ax.patches), 3)
        self.assertAlmostEqual(sum(p.get_height() for p in ax.patches), 1.0, places=10)

    def test_bar_categories_sorted(self):
        """Distinct values are placed in sorted (here alphabetical) order."""
        ax = plt.gca()
        make_bar(["c", "a", "b", "a"], ax, get_next_color(ax))
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["a", "b", "c"])

    def test_bar_count_mode_integer_heights(self):
        """normalize=False gives raw integer counts and a Count y-label."""
        ax = plt.gca()
        make_bar(["a", "a", "b"], ax, get_next_color(ax), normalize=False)
        self.assertEqual(ax.get_ylabel(), "Count")
        self.assertEqual(sorted(p.get_height() for p in ax.patches), [1.0, 2.0])

    def test_bar_xlabel_and_title(self):
        ax = plt.gca()
        make_bar(["a", "b"], ax, get_next_color(ax))
        self.assertEqual(ax.get_xlabel(), "Value")
        self.assertEqual(ax.get_title(), "Bar Chart")

    def test_bar_default_alpha(self):
        ax = plt.gca()
        make_bar(["a", "b"], ax, get_next_color(ax))
        self.assertAlmostEqual(ax.patches[0].get_alpha(), BAR_ALPHA)

    def test_bar_explicit_alpha_wins(self):
        ax = plt.gca()
        make_bar(["a", "b"], ax, get_next_color(ax), alpha=0.3)
        self.assertAlmostEqual(ax.patches[0].get_alpha(), 0.3)

    def test_single_bar_has_no_legend(self):
        ax = plt.gca()
        make_bar(["a", "b"], ax, get_next_color(ax))
        self.assertIsNone(ax.get_legend())

    def test_overlaid_bars_get_variable_k_legend(self):
        ax = plt.gca()
        make_bar(["a", "b"], ax, get_next_color(ax))
        make_bar(["a", "c"], ax, get_next_color(ax))
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(labels, ["Variable 1", "Variable 2"])

    def test_overlay_shares_union_of_categories(self):
        """Overlaid series share one sorted axis of every distinct value."""
        ax = plt.gca()
        make_bar(["a", "b"], ax, get_next_color(ax))
        make_bar(["b", "c"], ax, get_next_color(ax))
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["a", "b", "c"])

    def test_overlay_dodges_bars_side_by_side(self):
        """Within a shared category, the two series' bars sit at different x."""
        ax = plt.gca()
        make_bar(["a", "a", "b"], ax, get_next_color(ax))
        make_bar(["a", "b", "b"], ax, get_next_color(ax))
        state = ax._bar_state
        # Category "a" is position 0; each series' bar for it is offset from 0.
        centers = [
            s["container"].patches[0].get_x()
            + s["container"].patches[0].get_width() / 2
            for s in state["series"]
        ]
        self.assertNotAlmostEqual(centers[0], centers[1])
        # The two bars are dodged by exactly one bar width (no overlap).
        widths = [s["container"].patches[0].get_width() for s in state["series"]]
        self.assertAlmostEqual(abs(centers[0] - centers[1]), widths[0])

    def test_overlay_draws_category_boundary_lines(self):
        """Overlaid bar charts get one separator between each pair of levels."""
        ax = plt.gca()
        make_bar(["a", "b", "c"], ax, get_next_color(ax))
        self.assertEqual(len(ax._bar_state["boundary_lines"]), 0)  # lone chart: none
        make_bar(["a", "b", "c"], ax, get_next_color(ax))
        # Three categories -> two interior boundaries, at x = 0.5 and 1.5.
        xs = sorted(line.get_xdata()[0] for line in ax._bar_state["boundary_lines"])
        self.assertEqual(xs, [0.5, 1.5])

    def test_bar_numeric_discrete_via_plot_dispatch(self):
        """type='bar' on a discrete RV routes to make_bar (one bar per face)."""
        RV(BoxModel([1, 2, 3, 4, 5, 6])).sim(600).plot(type="bar", suggest=False)
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Bar Chart")
        self.assertEqual(len(ax.patches), 6)


class TestPlot1DViolin(PlotTestCase):
    """The 1D single violin (make_violinplot), wired into type='violin'."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(400)

    def test_violin_via_plot_renders(self):
        """type='violin' on 1D data draws a violin (not an empty plot)."""
        self.sims.plot(type="violin", suggest=False)
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Violin Plot")
        # A violin body is a PolyCollection; the inner box adds patches/lines.
        self.assertGreater(len(ax.collections), 0)

    def test_violin_default_alpha(self):
        ax = plt.gca()
        make_violinplot(np.random.normal(0, 1, 200), ax, get_next_color(ax))
        self.assertAlmostEqual(ax.collections[0].get_alpha(), 0.5)

    def test_violin_empty_raises(self):
        ax = plt.gca()
        with self.assertRaises(ValueError):
            make_violinplot(np.array([]), ax, get_next_color(ax))

    def test_violin_overlay_side_by_side(self):
        """A second violin sits at the next position with its own tick label."""
        ax = plt.gca()
        make_violinplot(np.random.normal(0, 1, 200), ax, get_next_color(ax))
        make_violinplot(np.random.normal(2, 1, 200), ax, get_next_color(ax))
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["Variable 1", "Variable 2"])


class TestPlotCategorical(PlotTestCase):
    """1D categorical (string) outcomes, routed through .plot()."""

    def setUp(self):
        np.random.seed(42)
        self.colors = RV(BoxModel(["red", "green", "blue"]))

    def test_string_rv_default_small_is_dotplot(self):
        """Small-n categorical default is a dot plot, labeled by category."""
        self.colors.sim(30).plot(suggest=False)
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Dot Plot")
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["blue", "green", "red"])

    def test_string_rv_default_large_is_bar(self):
        """Large-n categorical default is a bar chart."""
        self.colors.sim(2000).plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Bar Chart")

    def test_string_rv_impulse_labels(self):
        """type='impulse' on strings draws stems labeled by category."""
        self.colors.sim(500).plot(type="impulse", suggest=False)
        ax = plt.gca()
        self.assertIn("Impulse", ax.get_title())
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["blue", "green", "red"])

    def test_string_rv_dotplot_labels(self):
        self.colors.sim(500).plot(type="dotplot", suggest=False)
        labels = [t.get_text() for t in plt.gca().get_xticklabels()]
        self.assertEqual(labels, ["blue", "green", "red"])

    def test_string_rv_invalid_type_raises(self):
        """A non-categorical type on string data gives a friendly error."""
        with self.assertRaises(ValueError):
            self.colors.sim(100).plot(type="hist", suggest=False)

    def test_make_dotplot_categorical_direct(self):
        """make_dotplot handles strings directly (one stack per category)."""
        ax = plt.gca()
        make_dotplot(np.array(["b", "a", "b", "c", "a", "b"]), ax, get_next_color(ax))
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["a", "b", "c"])

    def test_make_impulse_categorical_direct(self):
        """make_impulse handles strings directly, labeled by category."""
        ax = plt.gca()
        make_impulse(np.array(["b", "a", "b", "c"]), ax, get_next_color(ax))
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["a", "b", "c"])


class TestPlot1DDensityFeatures(PlotTestCase):
    """The integrated make_density: styling, bandwidth, discrete pmf."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(600)

    def test_density_title_and_xlabel(self):
        self.sims.plot(type="density")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Density Curve")
        self.assertEqual(ax.get_xlabel(), "Value")

    def test_density_default_alpha_opaque(self):
        self.sims.plot(type="density")
        alpha = plt.gca().lines[0].get_alpha()
        self.assertTrue(alpha is None or alpha == 1.0)

    def test_density_bandwidth_changes_curve(self):
        self.sims.plot(type="density")
        default_y = plt.gca().lines[0].get_ydata().copy()
        plt.close("all")
        self.sims.plot(type="density", bandwidth=0.1)
        rough_y = plt.gca().lines[0].get_ydata()
        self.assertFalse(np.allclose(default_y, rough_y))

    def test_density_overlay_gets_legend(self):
        self.sims.plot(type="density")
        RV(Normal(3, 1)).sim(600).plot(type="density")
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(labels, ["Variable 1", "Variable 2"])

    def test_discrete_density_still_dot_line_pmf(self):
        """type='density' on discrete data keeps the dot-line pmf plot."""
        RV(Binomial(n=10, p=0.4)).sim(500).plot(type="density")
        ax = plt.gca()
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(ax.get_ylabel(), "Relative Frequency")


class TestPlot1DEcdf(PlotTestCase):
    """The new type='ecdf' (empirical CDF step plot)."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(200)

    def test_ecdf_draws_one_step_line_with_title(self):
        self.sims.plot(type="ecdf")
        ax = plt.gca()
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(ax.get_title(), "ECDF Plot")

    def test_ecdf_axis_labels_normalized(self):
        self.sims.plot(type="ecdf")
        ax = plt.gca()
        self.assertEqual(ax.get_xlabel(), "Value")
        self.assertEqual(ax.get_ylabel(), "Cumulative Relative Frequency")

    def test_ecdf_ylabel_count_when_not_normalized(self):
        self.sims.plot(type="ecdf", normalize=False)
        self.assertEqual(plt.gca().get_ylabel(), "Cumulative Count")

    def test_ecdf_is_monotone_nondecreasing_to_one(self):
        """A normalized ECDF never decreases and reaches exactly 1.0."""
        self.sims.plot(type="ecdf")
        y = plt.gca().lines[0].get_ydata()
        self.assertTrue(np.all(np.diff(y) >= 0))
        self.assertAlmostEqual(y[-1], 1.0, places=10)

    def test_ecdf_count_tops_out_at_n(self):
        self.sims.plot(type="ecdf", normalize=False)
        y = plt.gca().lines[0].get_ydata()
        self.assertEqual(y[-1], len(self.sims))

    def test_ecdf_overlay_gets_legend(self):
        self.sims.plot(type="ecdf")
        RV(Normal(2, 1)).sim(200).plot(type="ecdf")
        self.assertIsNotNone(plt.gca().get_legend())

    def test_ecdf_non_numeric_raises_friendly_error(self):
        with self.assertRaises(TypeError) as cm:
            make_ecdf(np.array(["H", "T", "H"]), plt.gca(), "#000000")
        self.assertIn("numeric", str(cm.exception))


class TestPlot1DDotplot(PlotTestCase):
    """The new type='dotplot' (stacked dot plot)."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(BoxModel([1, 2, 3, 4, 5, 6])).sim(30)

    def test_dotplot_draws_one_dot_per_observation(self):
        self.sims.plot(type="dotplot")
        offsets = plt.gca().collections[0].get_offsets()
        self.assertEqual(len(offsets), 30)

    def test_dotplot_labels_and_title(self):
        self.sims.plot(type="dotplot")
        ax = plt.gca()
        self.assertEqual(ax.get_xlabel(), "Value")
        self.assertEqual(ax.get_ylabel(), "Count")
        self.assertEqual(ax.get_title(), "Dot Plot")

    def test_dotplot_is_default_for_small_discrete(self):
        """Per DEFAULT_PLOT_TYPE, small-n discrete data defaults to dotplot."""
        self.sims.plot()  # 30 die rolls
        offsets = plt.gca().collections[0].get_offsets()
        self.assertEqual(len(offsets), 30)
        self.assertEqual(plt.gca().get_title(), "Dot Plot")

    def test_impulse_still_default_for_large_discrete(self):
        """Per DEFAULT_PLOT_TYPE, large-n discrete data defaults to impulse."""
        RV(BoxModel([1, 2, 3, 4, 5, 6])).sim(500).plot()
        segs = plt.gca().collections[0].get_segments()
        self.assertGreater(len(segs), 0)

    def test_dotplot_overlay_dodges_and_gets_legend(self):
        self.sims.plot(type="dotplot")
        RV(BoxModel([1, 2, 3, 4, 5, 6])).sim(30).plot(type="dotplot")
        ax = plt.gca()
        self.assertEqual(len([c for c in ax.collections if len(c.get_offsets())]), 2)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        # Boundary lines between stacks appear only for overlays.
        self.assertGreater(len(ax.lines), 0)

    def test_dotplot_jitter_warns(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(type="dotplot", jitter=True)

    def test_dotplot_bins_warns(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(type="dotplot", bins=5)

    def test_dotplot_categorical_now_renders(self):
        """make_dotplot now accepts categorical (string) data: one stack per
        category, labeled with the value (it no longer raises)."""
        from symbulate.plot import make_dotplot

        ax = plt.gca()
        make_dotplot(["H", "T", "H"], ax, "#56B4E9")
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["H", "T"])

    def test_dotplot_returns_wrapper(self):
        p = self.sims.plot(type="dotplot")
        self.assertIsInstance(p, SymbulatePlot)


class TestPlot1DBoxStyling(PlotTestCase):
    """The new type='box' (1D box plot), and its 'boxplot' alias."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(200)

    def test_box_draws_one_patch(self):
        self.sims.plot(type="box")
        self.assertGreater(len(plt.gca().patches), 0)

    def test_box_title_and_ylabel(self):
        self.sims.plot(type="box")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Box Plot")
        self.assertEqual(ax.get_ylabel(), "Value")

    def test_box_default_alpha(self):
        """Box fill defaults to BOXPLOT_ALPHA (0.75)."""
        self.sims.plot(type="box")
        self.assertAlmostEqual(plt.gca().patches[0].get_alpha(), 0.75)

    def test_box_explicit_alpha_wins(self):
        self.sims.plot(type="box", alpha=0.3)
        self.assertAlmostEqual(plt.gca().patches[0].get_alpha(), 0.3)

    def test_box_default_label_is_variable_one(self):
        self.sims.plot(type="box")
        labels = [t.get_text() for t in plt.gca().get_xticklabels()]
        self.assertEqual(labels, ["Variable 1"])

    def test_box_label_override(self):
        self.sims.plot(type="box", label="Control")
        labels = [t.get_text() for t in plt.gca().get_xticklabels()]
        self.assertIn("Control", labels)

    def test_boxplot_alias_behaves_like_box(self):
        self.sims.plot(type="boxplot")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Box Plot")
        self.assertGreater(len(ax.patches), 0)

    def test_two_overlaid_boxes_get_two_patches_and_labels(self):
        """Plotting twice adds a second box at the next position."""
        self.sims.plot(type="box")
        RV(Normal(5, 1)).sim(200).plot(type="box")
        ax = plt.gca()
        self.assertEqual(len(ax.patches), 2)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["Variable 1", "Variable 2"])

    def test_box_plus_rug_produces_patch_and_ticks(self):
        self.sims.plot(type=("box", "rug"))
        ax = plt.gca()
        self.assertGreater(len(ax.patches), 0)
        self.assertGreater(len(ax.collections), 0)

    def test_box_constant_data_does_not_raise(self):
        from symbulate.plot import make_boxplot, get_next_color

        ax = plt.gca()
        make_boxplot(np.array([5.0] * 10), ax, get_next_color(ax))
        self.assertEqual(len(ax.patches), 1)

    def test_box_single_value_does_not_raise(self):
        from symbulate.plot import make_boxplot, get_next_color

        ax = plt.gca()
        make_boxplot(np.array([5.0]), ax, get_next_color(ax))
        self.assertEqual(len(ax.patches), 1)

    def test_box_empty_data_raises_friendly_error(self):
        from symbulate.plot import make_boxplot, get_next_color

        ax = plt.gca()
        with self.assertRaises(ValueError) as cm:
            make_boxplot(np.array([]), ax, get_next_color(ax))
        self.assertIn("no values", str(cm.exception))

    def test_box_all_nan_data_raises_friendly_error(self):
        from symbulate.plot import make_boxplot, get_next_color

        ax = plt.gca()
        with self.assertRaises(ValueError) as cm:
            make_boxplot(np.array([np.nan, np.nan]), ax, get_next_color(ax))
        self.assertIn("no values", str(cm.exception))

    def test_box_outliers_default_draws_outlier_points(self):
        """With outliers=True (default), extreme points become fliers."""
        from symbulate.plot import make_boxplot, get_next_color

        values = np.append(np.random.normal(0, 1, 50), 25.0)
        ax = plt.gca()
        box = make_boxplot(values, ax, get_next_color(ax))
        self.assertIn(25.0, box["fliers"][0].get_ydata())

    def test_box_outliers_false_extends_whiskers_to_extremes(self):
        """outliers=False stretches the whiskers to min/max, no fliers."""
        from symbulate.plot import make_boxplot, get_next_color

        values = np.append(np.random.normal(0, 1, 50), 25.0)
        ax = plt.gca()
        box = make_boxplot(values, ax, get_next_color(ax), outliers=False)
        self.assertEqual(len(box["fliers"][0].get_ydata()), 0)
        whisker_ends = [w.get_ydata()[1] for w in box["whiskers"]]
        self.assertAlmostEqual(max(whisker_ends), 25.0)
        self.assertAlmostEqual(min(whisker_ends), values.min())

    def test_box_outliers_kwarg_flows_through_plot(self):
        """outliers= passed to .plot() reaches the box plot helper."""

        def n_flier_points():
            return sum(
                len(line.get_xdata())
                for line in plt.gca().lines
                if line.get_linestyle() == "None"
            )

        # A planted extreme value is always a flier under the default
        # 1.5-IQR rule, so the assertion does not depend on a random draw
        # happening to contain an outlier (Normal(0, 1) sometimes has
        # none). self.sims uses the package RNG, which np.random.seed does
        # not control, so plotting it directly made this test flaky.
        planted = RVResults(np.append(np.random.normal(0, 1, 200), 25.0))
        planted.plot(type="box")
        self.assertGreater(n_flier_points(), 0)
        plt.close("all")
        planted.plot(type="box", outliers=False)
        self.assertEqual(n_flier_points(), 0)

    def test_box_returns_wrapper(self):
        p = self.sims.plot(type="box")
        self.assertIsInstance(p, SymbulatePlot)


class TestPlot2DScatterFeatures(PlotTestCase):
    """The integrated make_scatter: alpha, jitter modes, legend."""

    def setUp(self):
        np.random.seed(42)
        X, Y = RV(Normal(0, 1) ** 2)
        self.sims = (X & Y).sim(100)
        Xd, Yd = RV(Binomial(5, 0.4) ** 2)
        self.discrete_sims = (Xd & Yd).sim(40)

    def test_scatter_default_alpha(self):
        self.sims.plot(type="scatter")
        self.assertAlmostEqual(plt.gca().collections[0].get_alpha(), 0.25)

    def test_scatter_axis_labels_and_title(self):
        self.sims.plot(type="scatter")
        ax = plt.gca()
        self.assertEqual(ax.get_xlabel(), "Variable 1")
        self.assertEqual(ax.get_ylabel(), "Variable 2")
        self.assertEqual(ax.get_title(), "2D Scatter Plot")

    def test_scatter_jitter_true_still_works(self):
        """jitter=True is the legacy alias for jitter='random'."""
        self.discrete_sims.plot(type="scatter", jitter=True, suggest=False)
        self.assertGreater(len(plt.gca().collections), 0)

    def test_scatter_jitter_random(self):
        self.discrete_sims.plot(type="scatter", jitter="random", suggest=False)
        self.assertGreater(len(plt.gca().collections), 0)

    def test_scatter_jitter_spiral(self):
        self.discrete_sims.plot(type="scatter", jitter="spiral", suggest=False)
        self.assertGreater(len(plt.gca().collections), 0)

    def test_scatter_jitter_bins(self):
        self.discrete_sims.plot(type="scatter", jitter="bins", suggest=False)
        self.assertGreater(len(plt.gca().collections), 0)

    def test_scatter_jitter_auto_no_longer_an_option(self):
        """'auto' was removed: automatic layout is now the default."""
        with self.assertRaises(ValueError):
            self.discrete_sims.plot(type="scatter", jitter="auto", suggest=False)

    def test_scatter_invalid_jitter_raises_friendly_error(self):
        with self.assertRaises(ValueError) as cm:
            self.discrete_sims.plot(type="scatter", jitter="wiggle")
        self.assertIn("spiral", str(cm.exception))

    def test_default_jitter_clusters_discrete_pairs(self):
        """With jitter unset, two discrete variables auto-cluster, so
        every one of the 40 points lands at its own distinct position
        instead of overplotting."""
        self.discrete_sims.plot(type="scatter", suggest=False)
        offsets = np.asarray(plt.gca().collections[0].get_offsets())
        self.assertEqual(len(np.unique(offsets, axis=0)), len(offsets))

    def test_default_no_jitter_for_continuous(self):
        """With jitter unset, continuous data is drawn at its exact
        coordinates -- never snapped or perturbed."""
        self.sims.plot(type="scatter", suggest=False)
        offsets = np.asarray(plt.gca().collections[0].get_offsets())
        self.assertTrue(
            np.allclose(np.sort(offsets, axis=0), np.sort(self.sims.array, axis=0))
        )

    def test_explicit_jitter_false_disables_clustering(self):
        """jitter=False is respected even for discrete data: repeated
        pairs draw on top of each other at exact positions."""
        self.discrete_sims.plot(type="scatter", jitter=False, suggest=False)
        offsets = np.asarray(plt.gca().collections[0].get_offsets())
        self.assertLess(len(np.unique(offsets, axis=0)), len(offsets))

    def test_scatter_overlay_gets_legend(self):
        self.sims.plot(type="scatter")
        X, Y = RV(Normal(2, 1) ** 2)
        (X & Y).sim(100).plot(type="scatter")
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        labels = [t.get_text() for t in legend.get_texts()]
        self.assertEqual(labels, ["Variable 1", "Variable 2"])

    def test_scatter_user_marker_size_does_not_raise(self):
        """s= used to flow straight into ax.scatter; it still must."""
        self.sims.plot(type="scatter", s=100)
        self.assertGreater(len(plt.gca().collections), 0)


class TestPlot2DMeshFeatures(PlotTestCase):
    """The integrated 2D hist / density / tile helpers."""

    def setUp(self):
        np.random.seed(42)
        X, Y = RV(Normal(0, 1) ** 2)
        self.sims = (X & Y).sim(500)
        Xd, Yd = RV(Binomial(5, 0.4) ** 2)
        self.discrete_sims = (Xd & Yd).sim(500)
        Xm, Ym = RV(Binomial(5, 0.4) * Normal(0, 1))
        self.mixed_sims = (Xm & Ym).sim(500)

    def test_hist2d_normalize_true_now_works(self):
        """normalize=True previously crashed in the colorbar code."""
        self.sims.plot(type="hist", normalize=True)
        self.assertGreater(len(plt.gca().collections), 0)

    def test_hist2d_colorbar_label(self):
        self.sims.plot(type="hist")
        cax = plt.gcf().axes[-1]
        self.assertEqual(cax.get_ylabel(), "Density")

    def test_hist2d_gca_restored_after_colorbar(self):
        """After a 2D hist, gca() must be the data axes, not the colorbar."""
        p = self.sims.plot(type="hist")
        self.assertIs(plt.gca(), p.ax)

    def test_hist2d_hex_option(self):
        self.sims.plot(type="hist", hex=True)
        self.assertEqual(plt.gca().get_title(), "Hexbin Plot")

    def test_hist2d_title(self):
        self.sims.plot(type="hist")
        self.assertEqual(plt.gca().get_title(), "2-D Histogram")

    def test_density2d_draws_contour_surface(self):
        self.sims.plot(type="density")
        ax = plt.gca()
        self.assertGreater(len(ax.collections + ax.images), 0)
        self.assertEqual(ax.get_title(), "2D Density Plot")

    def test_density2d_contour_mode(self):
        self.sims.plot(type="density", contour=True)
        self.assertEqual(plt.gca().get_title(), "Contour Plot")

    def test_density2d_levels_without_contour_warns(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(type="density", levels=5)

    def test_density2d_bad_levels_raises_friendly_error(self):
        with self.assertRaises(ValueError) as cm:
            self.sims.plot(type="density", contour=True, levels=1)
        self.assertIn("levels", str(cm.exception))

    def test_tile_normalize_false_colorbar_label(self):
        self.discrete_sims.plot(type="tile", normalize=False)
        cax = plt.gcf().axes[-1]
        self.assertEqual(cax.get_ylabel(), "Count")

    def test_tile_bins_with_two_discrete_warns(self):
        with self.assertWarns(UserWarning):
            self.discrete_sims.plot(type="tile", bins=5)

    def test_tile_mixed_discrete_continuous(self):
        """Tile plots now bin a continuous axis instead of failing."""
        self.mixed_sims.plot(type="tile")
        ax = plt.gca()
        self.assertGreater(len(ax.images), 0)

    def test_tile_mixed_draws_separators_on_discrete_axis(self):
        """A mixed-data tile separates its discrete levels with boundary
        lines on the discrete axis only: vertical when x is discrete,
        horizontal when y is discrete."""
        # mixed_sims is Binomial(5, 0.4) * Normal(0, 1): x is discrete.
        p = self.mixed_sims.plot(type="tile")
        vlines = [ln for ln in p.ax.lines if len(set(ln.get_xdata())) == 1]
        hlines = [ln for ln in p.ax.lines if len(set(ln.get_ydata())) == 1]
        self.assertGreater(len(vlines), 0)
        self.assertEqual(len(hlines), 0)
        # every separator sits on a half-integer cell boundary
        for ln in vlines:
            self.assertAlmostEqual(ln.get_xdata()[0] % 1, 0.5)
        plt.close("all")

        # Swap the order so y is the discrete axis: separators go horizontal.
        Xc, Yd = RV(Normal(0, 1) * Binomial(5, 0.4))
        swapped = (Xc & Yd).sim(500)
        p2 = swapped.plot(type="tile")
        vlines2 = [ln for ln in p2.ax.lines if len(set(ln.get_xdata())) == 1]
        hlines2 = [ln for ln in p2.ax.lines if len(set(ln.get_ydata())) == 1]
        self.assertEqual(len(vlines2), 0)
        self.assertGreater(len(hlines2), 0)

    def test_tile_two_discrete_has_no_separators(self):
        """Separator lines are a mixed-data feature; a both-discrete tile
        stays clean."""
        p = self.discrete_sims.plot(type="tile")
        separators = [
            ln
            for ln in p.ax.lines
            if len(set(ln.get_xdata())) == 1 or len(set(ln.get_ydata())) == 1
        ]
        self.assertEqual(len(separators), 0)

    def test_tile_fallback_uses_classify_values_not_dtype(self):
        """A direct make_tile call with unspecified discreteness classifies
        with classify_values, not the raw dtype: a wide-support integer axis
        (many distinct values) is treated as continuous and binned into
        TILE_DEFAULT_BINS cells, not given one skinny cell per value."""
        rng = np.random.default_rng(0)
        x = rng.integers(0, 200, 2000)  # int dtype, >40 unique values
        y = rng.normal(0, 1, 2000)  # continuous
        mesh = make_tile(x, y, plt.gca())  # discrete_x/discrete_y default None
        # shape is (ny, nx); a binned (continuous) x axis has TILE_DEFAULT_BINS
        # columns, whereas the old dtype rule would have made ~200.
        self.assertEqual(mesh.get_array().shape[1], TILE_DEFAULT_BINS)

    def test_segmented_rug_fallback_uses_classify_values_not_dtype(self):
        """A direct make_segmented_rug call classifies with classify_values:
        a wide-support integer variable counts as continuous, so pairing it
        with another continuous variable raises the friendly two-continuous
        error instead of drawing hundreds of bands (the old dtype rule
        would have called the integer axis discrete and drawn them)."""
        rng = np.random.default_rng(0)
        x = rng.integers(0, 200, 2000)  # continuous under classify_values
        y = rng.normal(0, 1, 2000)  # continuous
        with self.assertRaises(ValueError) as cm:
            make_segmented_rug(x, y, plt.gca(), "#56B4E9")
        self.assertIn("continuous", str(cm.exception))

    def test_segmented_rug_mixed_data(self):
        """type='rug' on 2D mixed data draws one band per discrete level."""
        self.mixed_sims.plot(type="rug")
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0)
        self.assertEqual(ax.get_title(), "Segmented Rug Plot")

    def test_segmented_rug_gridlines_on_discrete_axis_only(self):
        """Gridlines run along the discrete axis only: vertical (x) when x
        is discrete, horizontal (y) when y is discrete."""
        # mixed_sims is Binomial(5, 0.4) * Normal(0, 1): x is discrete.
        p = self.mixed_sims.plot(type="rug")
        self.assertTrue(all(g.get_visible() for g in p.ax.get_xgridlines()))
        self.assertFalse(any(g.get_visible() for g in p.ax.get_ygridlines()))
        plt.close("all")

        # Swap the order so the continuous variable is first: y is discrete.
        Xc, Yd = RV(Normal(0, 1) * Binomial(5, 0.4))
        swapped = (Xc & Yd).sim(500)
        p2 = swapped.plot(type="rug")
        self.assertFalse(any(g.get_visible() for g in p2.ax.get_xgridlines()))
        self.assertTrue(all(g.get_visible() for g in p2.ax.get_ygridlines()))

    def test_segmented_rug_gridline_per_band_when_labels_thinned(self):
        """With more discrete levels than MAX_DISCRETE_TICKS, the labels
        thin out but every distinct rug (band) still gets a gridline: the
        labeled bands via major ticks, the rest via minor ticks, with grid
        drawn on both."""
        # DiscreteUniform(1, 25) has 25 levels: > MAX_DISCRETE_TICKS (10),
        # but <= K_2D (30), so it still classifies as discrete.
        Xd, Yc = RV(DiscreteUniform(1, 25) * Normal(0, 1))
        sims = (Xd & Yc).sim(3000)
        ax = sims.plot(type="rug").ax
        n_levels = len(np.unique(sims.array[:, 0]))

        major = ax.get_xticks()
        minor = ax.get_xticks(minor=True)
        # Labels are thinned onto the major ticks.
        self.assertLessEqual(len(major), MAX_DISCRETE_TICKS)
        # Major + minor ticks together cover every band exactly once.
        self.assertEqual(len(major) + len(minor), n_levels)
        # Gridlines drawn on both major and minor bands of the discrete
        # (x) axis, and none on the continuous (y) axis.
        self.assertTrue(
            all(t.gridline.get_visible() for t in ax.xaxis.get_major_ticks())
        )
        self.assertTrue(
            all(t.gridline.get_visible() for t in ax.xaxis.get_minor_ticks())
        )
        self.assertFalse(
            any(t.gridline.get_visible() for t in ax.yaxis.get_major_ticks())
        )

    def test_segmented_rug_two_discrete_raises_friendly_error(self):
        with self.assertRaises(ValueError) as cm:
            self.discrete_sims.plot(type="rug")
        self.assertIn("tile", str(cm.exception))

    def test_marginal_hist_combo_still_draws(self):
        self.sims.plot(type="hist", marginal=True)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)


class TestPlot2DMosaic(PlotTestCase):
    """The integrated mosaic plot (discrete x discrete, alternative to tile)."""

    def setUp(self):
        np.random.seed(42)
        Xd, Yd = RV(Binomial(5, 0.4) ** 2)
        self.discrete_sims = (Xd & Yd).sim(500)

    def test_mosaic_produces_bars(self):
        self.discrete_sims.plot(type="mosaic")
        self.assertGreater(len(plt.gca().patches), 0)

    def test_mosaic_title(self):
        self.discrete_sims.plot(type="mosaic")
        self.assertEqual(plt.gca().get_title(), "Mosaic Plot")

    def test_mosaic_xlabel_is_variable_1(self):
        self.discrete_sims.plot(type="mosaic")
        self.assertEqual(plt.gca().get_xlabel(), "Variable 1")

    def test_mosaic_yaxis_shows_zero_to_one_ticks(self):
        """Every column's segments span 0 to 1 the same way, so a shared
        0-to-1 proportion scale on the left is meaningful across every
        column."""
        self.discrete_sims.plot(type="mosaic")
        ax = plt.gca()
        self.assertTrue(ax.yaxis.get_visible())
        labels = [t.get_text() for t in ax.get_yticklabels()]
        self.assertEqual(labels, ["0.00", "0.25", "0.50", "0.75", "1.00"])

    def test_mosaic_legend_uses_marginal_column_labels_by_default(self):
        """With marginal_column=True (the default), category names are
        printed beside the marginal column instead of a floating legend."""
        self.discrete_sims.plot(type="mosaic")
        ax = plt.gca()
        self.assertIsNone(ax.get_legend())
        arr = np.asarray(self.discrete_sims.results)
        y_labels = sorted(str(v) for v in np.unique(arr[:, 1]))
        label_texts = sorted(t.get_text() for t in ax.texts if t.get_text() in y_labels)
        self.assertEqual(label_texts, y_labels)

    def test_mosaic_legend_falls_back_without_marginal_column(self):
        """With marginal_column=False, there's no column to hang labels
        off of, so a standard floating legend appears instead."""
        self.discrete_sims.plot(type="mosaic", marginal_column=False)
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(legend.get_title().get_text(), "Variable 2")

    def test_mosaic_legend_false_shows_no_category_labels(self):
        p = self.discrete_sims.plot(type="mosaic", legend=False, annotate=False)
        self.assertEqual(len(p.ax.texts), 0)

    def test_mosaic_columns_sum_to_full_width(self):
        """Column widths (plus gaps) must span the full [0, 1] x-axis."""
        p = self.discrete_sims.plot(type="mosaic")
        self.assertAlmostEqual(p.ax.get_xlim()[0], 0.0)
        self.assertAlmostEqual(p.ax.get_xlim()[1], 1.0)

    def test_mosaic_normalize_false_labels_are_whole_numbers(self):
        """normalize=False switches in-cell labels from decimal
        proportions to whole-number counts."""
        p = self.discrete_sims.plot(type="mosaic", normalize=False)
        texts = [t.get_text() for t in p.ax.texts]
        self.assertGreater(len(texts), 0)
        for text in texts:
            self.assertNotIn(".", text)

    def test_mosaic_normalize_true_labels_are_decimals(self):
        p = self.discrete_sims.plot(type="mosaic")
        texts = [t.get_text() for t in p.ax.texts]
        self.assertGreater(len(texts), 0)
        self.assertTrue(any("." in text for text in texts))

    def test_mosaic_annotate_false_has_no_labels(self):
        """annotate=False alone: no in-cell labels, but the marginal
        column's legend labels (controlled separately by legend=) still
        draw as ax.text(), so isolate with legend=False too."""
        p = self.discrete_sims.plot(type="mosaic", annotate=False, legend=False)
        self.assertEqual(len(p.ax.texts), 0)

    def test_mosaic_annotate_false_still_shows_legend_labels(self):
        """With annotate=False, only the marginal column's category-name
        legend labels remain -- no in-cell proportion/count labels."""
        p = self.discrete_sims.plot(type="mosaic", annotate=False)
        arr = np.asarray(self.discrete_sims.results)
        y_labels = sorted(str(v) for v in np.unique(arr[:, 1]))
        texts = sorted(t.get_text() for t in p.ax.texts)
        self.assertEqual(texts, y_labels)

    def test_mosaic_overlay_prints_warning(self):
        """A second mosaic call on the same axes prints (not warns) a
        readability warning -- the same overlay category as tile/hist2d."""
        ax = plt.gca()
        arr = np.asarray(self.discrete_sims.results)
        x, y = arr[:, 0], arr[:, 1]
        make_mosaic(x, y, ax)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            make_mosaic(x, y, ax)
        self.assertIn("second mosaic plot", buf.getvalue())

    def test_mosaic_length_mismatch_raises_friendly_error(self):
        with self.assertRaises(ValueError) as cm:
            make_mosaic(np.array([1, 2, 3]), np.array([1, 2]), plt.gca())
        self.assertIn("same length", str(cm.exception))

    def test_mosaic_many_categories_warns_and_hatches(self):
        """More than 7 y-categories repeats a palette color; the repeat is
        distinguished with a hatch pattern and a warning explains why."""
        rng = np.random.default_rng(0)
        x = rng.integers(0, 3, 2000)
        y = rng.integers(0, 10, 2000)  # 10 distinct values > 7 palette colors
        with self.assertWarns(UserWarning) as cm:
            bars = make_mosaic(x, y, plt.gca())
        self.assertIn("more than the", str(cm.warning))
        # category 0 and category 7 share a color; only 7's bars are hatched.
        color_0 = bars[0].patches[0].get_facecolor()
        color_7 = bars[7].patches[0].get_facecolor()
        self.assertEqual(color_0, color_7)
        self.assertEqual(bars[0].patches[0].get_hatch(), "")
        self.assertIsNotNone(bars[7].patches[0].get_hatch())

    def test_mosaic_is_listed_as_2d_discrete_alternative(self):
        for small_n in (True, False):
            _, alternatives = default_plot_type("2D_dd", small_n)
            self.assertIn("mosaic", alternatives)

    def test_mosaic_display_name(self):
        self.assertEqual(PLOT_DISPLAY_NAME["mosaic"], "Mosaic Plot")

    def test_mosaic_spans_total_parameter_rescales_span(self):
        """total= lets a row of segments fill less than the whole [0, 1]
        axis -- used to reserve room for the marginal reference column."""
        starts, widths = _mosaic_spans([1, 1], gap=0.0, total=0.5)
        self.assertAlmostEqual(widths.sum(), 0.5)
        self.assertAlmostEqual(starts[0], 0.0)

    def test_mosaic_marginal_column_present_by_default(self):
        """marginal_column defaults to True: an extra column, labeled
        with y_label (default "Variable 2"), appears after the real x
        categories."""
        p = self.discrete_sims.plot(type="mosaic")
        labels = [t.get_text() for t in p.ax.get_xticklabels()]
        self.assertEqual(labels[-1], "Variable 2")

    def test_mosaic_marginal_column_uses_custom_y_label(self):
        p = self.discrete_sims.plot(type="mosaic", y_label="Outcome")
        labels = [t.get_text() for t in p.ax.get_xticklabels()]
        self.assertEqual(labels[-1], "Outcome")

    def test_mosaic_marginal_column_false_omits_it(self):
        """With marginal_column=False, there's no extra column -- exactly
        one x-tick per real x category."""
        arr = np.asarray(self.discrete_sims.results)
        n_x = len(np.unique(arr[:, 0]))
        p = self.discrete_sims.plot(type="mosaic", marginal_column=False)
        labels = [t.get_text() for t in p.ax.get_xticklabels()]
        self.assertEqual(len(labels), n_x)

    def test_mosaic_marginal_column_extends_every_bar_container(self):
        """Every category's BarContainer gets one extra bar (the marginal
        column) on top of one per real x category."""
        arr = np.asarray(self.discrete_sims.results)
        x, y = arr[:, 0], arr[:, 1]
        n_x = len(np.unique(x))
        bars = make_mosaic(x, y, plt.gca())
        for container in bars.values():
            self.assertEqual(len(container.patches), n_x + 1)

    def test_mosaic_marginal_column_is_skinnier_than_real_columns(self):
        """The marginal column is a color reference, not real data -- it
        should read as visibly narrower than a typical real x column
        (individual real columns can still be narrower still, if that x
        value is rare)."""
        arr = np.asarray(self.discrete_sims.results)
        x, y = arr[:, 0], arr[:, 1]
        bars = make_mosaic(x, y, plt.gca())
        any_container = next(iter(bars.values()))
        real_widths = [p.get_width() for p in any_container.patches[:-1]]
        marginal_width = any_container.patches[-1].get_width()
        self.assertLess(marginal_width, np.mean(real_widths))

    def test_mosaic_marginal_column_has_no_in_cell_labels(self):
        """The marginal column is a color reference only -- it never gets
        count/percentage labels, even with annotate=True (the default).
        legend=False on both sides isolates in-cell labels from the
        marginal column's separate legend-label text."""
        arr = np.asarray(self.discrete_sims.results)
        x, y = arr[:, 0], arr[:, 1]
        p = self.discrete_sims.plot(type="mosaic", legend=False)
        n_labels_with_marginal = len(p.ax.texts)
        plt.close("all")
        p2 = self.discrete_sims.plot(type="mosaic", marginal_column=False, legend=False)
        n_labels_without_marginal = len(p2.ax.texts)
        self.assertEqual(n_labels_with_marginal, n_labels_without_marginal)

    def test_mosaic_label_decimal_is_conditional_not_joint(self):
        """In-cell decimal labels show the frequency conditional on each
        column's own x value (matching the segment's height), not the
        joint frequency over the whole dataset."""
        x = np.array([0] * 80 + [1] * 20)
        y = np.array([0] * 60 + [1] * 20 + [0] * 5 + [1] * 15)
        make_mosaic(x, y, plt.gca(), marginal_column=False)
        texts = sorted(t.get_text() for t in plt.gca().texts)
        self.assertEqual(texts, ["0.25", "0.25", "0.75", "0.75"])

    def test_mosaic_marginal_column_matches_y_marginal_frequency(self):
        """The marginal column's segment heights track y's overall relative
        frequency (summed over every x value), not any one column's."""
        arr = np.asarray(self.discrete_sims.results)
        x, y = arr[:, 0], arr[:, 1]
        bars = make_mosaic(x, y, plt.gca())
        y_labels = np.unique(y)
        counts = np.array([(y == val).sum() for val in y_labels], dtype=float)
        expected_fracs = counts / counts.sum()
        marginal_heights = np.array(
            [bars[val].patches[-1].get_height() for val in y_labels]
        )
        actual_fracs = marginal_heights / marginal_heights.sum()
        for expected, actual in zip(expected_fracs, actual_fracs):
            self.assertAlmostEqual(expected, actual, places=6)

    def test_mosaic_column_heights_match_conditional_frequency(self):
        """Each real column's segment heights track y's frequency
        *conditional* on that column's own x value -- not the joint
        frequency over the whole dataset, and not y's marginal shape."""
        arr = np.asarray(self.discrete_sims.results)
        x, y = arr[:, 0], arr[:, 1]
        bars = make_mosaic(x, y, plt.gca())
        x_labels = np.unique(x)
        y_labels = np.unique(y)
        for i, x_val in enumerate(x_labels):
            mask = x == x_val
            counts = np.array(
                [(y[mask] == y_val).sum() for y_val in y_labels], dtype=float
            )
            expected_fracs = counts / counts.sum()
            heights = np.array(
                [bars[y_val].patches[i].get_height() for y_val in y_labels]
            )
            actual_fracs = heights / heights.sum()
            for expected, actual in zip(expected_fracs, actual_fracs):
                self.assertAlmostEqual(expected, actual, places=6)

    def test_readable_text_color_black_on_light_background(self):
        self.assertEqual(_readable_text_color("#F0E442"), "black")

    def test_readable_text_color_white_on_dark_background(self):
        self.assertEqual(_readable_text_color("#0072B2"), "white")

    def test_mosaic_labels_use_contrasting_colors(self):
        """Labels drawn on a dark-palette category use white text; labels on
        a light-palette category use black text -- not one hardcoded color
        for every cell regardless of its background."""
        rng = np.random.default_rng(1)
        x = rng.integers(0, 3, 3000)
        y = rng.integers(0, 7, 3000)  # covers every Okabe-Ito color
        make_mosaic(x, y, plt.gca())
        label_colors = {t.get_color() for t in plt.gca().texts}
        self.assertIn("black", label_colors)
        self.assertIn("white", label_colors)

    # -----------------------------------------------------------------
    # equal_width: 100%-stacked bar chart (equal-width columns)
    # -----------------------------------------------------------------

    def setup_unequal_marginal_sims(self, seed=7):
        """x has a heavily skewed marginal (Bernoulli(0.9)), so the two
        real columns' default proportional widths are visibly unequal --
        the case the prompt asks equal_width to be tested against."""
        np.random.seed(seed)
        Xd, Yd = RV(Bernoulli(p=0.9) * Bernoulli(p=0.5))
        return (Xd & Yd).sim(500)

    def test_mosaic_default_widths_vary_with_unequal_marginal_counts(self):
        """Without equal_width, visibly-unequal marginal counts must
        produce columns of visibly different widths -- the standard
        mosaic behavior, confirmed here so the next test's uniformity
        can be attributed to equal_width and not incidental equal counts
        in this data."""
        sims = self.setup_unequal_marginal_sims()
        p = sims.plot(type="mosaic", marginal_column=False, suggest=False)
        widths = sorted({round(patch.get_width(), 6) for patch in p.ax.patches})
        self.assertGreater(len(widths), 1)

    def test_mosaic_equal_width_produces_uniform_column_widths(self):
        """equal_width=True must give every real column the same width,
        even though the same data's marginal counts are visibly unequal
        (see the previous test)."""
        sims = self.setup_unequal_marginal_sims()
        p = sims.plot(
            type="mosaic", equal_width=True, marginal_column=False, suggest=False
        )
        widths = sorted({round(patch.get_width(), 6) for patch in p.ax.patches})
        self.assertEqual(len(widths), 1)

    def test_mosaic_equal_width_does_not_change_segment_heights(self):
        """equal_width changes column width only -- each column's segment
        heights (conditional frequencies) must match the default
        proportional-width mode exactly."""
        sims = self.setup_unequal_marginal_sims()
        p_default = sims.plot(type="mosaic", marginal_column=False, suggest=False)
        heights_default = sorted(
            round(patch.get_height(), 6) for patch in p_default.ax.patches
        )
        plt.close("all")
        p_equal = sims.plot(
            type="mosaic", equal_width=True, marginal_column=False, suggest=False
        )
        heights_equal = sorted(
            round(patch.get_height(), 6) for patch in p_equal.ax.patches
        )
        self.assertEqual(heights_default, heights_equal)

    def test_mosaic_equal_width_labels_show_true_conditional_proportion(self):
        """In-cell labels must still report the true conditional
        proportion (division by the real per-column count), not something
        distorted by equal-width columns."""
        sims = self.setup_unequal_marginal_sims()
        arr = np.asarray(sims.results)
        x, y = arr[:, 0], arr[:, 1]
        p = sims.plot(type="mosaic", equal_width=True, marginal_column=False)
        x_labels = np.unique(x)
        y_labels = np.unique(y)
        # Every printed decimal label, across every column, must be a valid
        # conditional proportion somewhere in the joint table -- a loose but
        # simple correctness check that doesn't depend on matching each
        # label back to its exact cell position.
        all_possible = set()
        for x_val in x_labels:
            mask = x == x_val
            counts = np.array(
                [(y[mask] == y_val).sum() for y_val in y_labels], dtype=float
            )
            for frac in counts / counts.sum():
                all_possible.add(round(frac, 2))
        printed = {round(float(t.get_text()), 2) for t in p.ax.texts}
        self.assertTrue(printed.issubset(all_possible))

    def test_mosaic_equal_width_still_respects_marginal_column(self):
        """equal_width and marginal_column are independent switches --
        equal_width=True must still draw the marginal reference column
        (skinnier than the equal-width real columns) when
        marginal_column=True (the default)."""
        sims = self.setup_unequal_marginal_sims()
        p = sims.plot(type="mosaic", equal_width=True, suggest=False)
        widths = sorted({round(patch.get_width(), 6) for patch in p.ax.patches})
        # Exactly two distinct widths: the (uniform) real columns, and the
        # narrower marginal column.
        self.assertEqual(len(widths), 2)

    def test_mosaic_equal_width_title_is_stacked_plot(self):
        """equal_width=True changes the title to "Stacked Plot" -- a clear
        visual signal that column widths are equal, not proportional."""
        sims = self.setup_unequal_marginal_sims()
        p = sims.plot(type="mosaic", equal_width=True, suggest=False)
        self.assertEqual(p.ax.get_title(), "Stacked Plot")

    def test_mosaic_default_title_is_unaffected(self):
        """Without equal_width, the title stays "Mosaic Plot"."""
        sims = self.setup_unequal_marginal_sims()
        p = sims.plot(type="mosaic", suggest=False)
        self.assertEqual(p.ax.get_title(), "Mosaic Plot")

    def test_mosaic_equal_width_display_name(self):
        self.assertEqual(PLOT_DISPLAY_NAME["mosaic_equal_width"], "Stacked Plot")

    def test_mosaic_equal_width_is_opt_in_only(self):
        """equal_width must not become a new automatic default for 2D
        discrete data -- mosaic (in either mode) stays explicit-opt-in-only."""
        for small_n in (True, False):
            _, alternatives = default_plot_type("2D_dd", small_n)
            self.assertNotIn("mosaic_equal_width", alternatives)
            self.assertNotIn("stacked_bar", alternatives)


class TestPlot2DBox(PlotTestCase):
    """The new type='box' (grouped box plot) for mixed discrete/continuous data."""

    def setUp(self):
        np.random.seed(42)

    def test_box_discrete_x_continuous_y(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        sims.plot(type="box")
        ax = plt.gca()
        self.assertGreater(len(ax.patches), 0)
        self.assertEqual(ax.get_title(), "Box Plot")

    def test_box_continuous_x_discrete_y(self):
        X, Y = RV(Normal(0, 1) * Binomial(5, 0.4))
        sims = (X & Y).sim(500)
        sims.plot(type="box")
        self.assertGreater(len(plt.gca().patches), 0)

    def test_boxplot_alias_behaves_like_box(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(500).plot(type="boxplot")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Box Plot")
        self.assertGreater(len(ax.patches), 0)

    def test_box_two_discrete_raises_friendly_error(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="box")
        self.assertIn("tile", str(cm.exception))

    def test_box_two_continuous_raises_friendly_error(self):
        X, Y = RV(Normal(0, 1) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="box")
        self.assertIn("scatter", str(cm.exception))

    def test_box_outliers_false_extends_whiskers_to_extremes(self):
        """outliers=False stretches each group's whiskers to min/max."""
        from symbulate.plot import make_grouped_boxplot, get_next_color

        x = np.repeat([0, 1], 50)
        y = np.append(np.random.normal(0, 1, 99), 30.0)
        ax = plt.gca()
        boxes = make_grouped_boxplot(x, y, ax, get_next_color(ax), outliers=False)
        for flier in boxes["fliers"]:
            self.assertEqual(len(flier.get_ydata()), 0)
        whisker_ends = [w.get_ydata()[1] for w in boxes["whiskers"]]
        self.assertAlmostEqual(max(whisker_ends), 30.0)

    def test_box_outliers_kwarg_flows_through_plot(self):
        """outliers= passed to .plot() reaches the grouped box helper."""

        def n_flier_points():
            return sum(
                len(line.get_xdata())
                for line in plt.gca().lines
                if line.get_linestyle() == "None"
            )

        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        sims.plot(type="box")
        self.assertGreater(n_flier_points(), 0)
        plt.close("all")
        sims.plot(type="box", outliers=False)
        self.assertEqual(n_flier_points(), 0)


class TestPlot2DSegmentedDensity(PlotTestCase):
    """The new type='segmented_density' for mixed discrete/continuous data."""

    def setUp(self):
        np.random.seed(42)

    def test_segmented_density_discrete_x_continuous_y(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        sims.plot(type="segmented_density")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Segmented Density Plot")
        # One unfilled Line2D curve per observed level of X, and no
        # fills by default (ridge=False)
        n_levels = len(np.unique(sims.array[:, 0]))
        self.assertEqual(len(ax.lines), n_levels)
        self.assertFalse(any(isinstance(c, PolyCollection) for c in ax.collections))
        # Discrete x -> flipped orientation: baselines on the x-axis
        self.assertEqual(len(ax.get_xticks()), n_levels)

    def test_segmented_density_continuous_x_discrete_y(self):
        X, Y = RV(Normal(0, 1) * Binomial(5, 0.4))
        sims = (X & Y).sim(500)
        sims.plot(type="segmented_density")
        ax = plt.gca()
        # Discrete y -> classic orientation: baselines on the y-axis
        n_levels = len(np.unique(sims.array[:, 1]))
        self.assertEqual(len(ax.lines), n_levels)
        self.assertEqual(len(ax.get_yticks()), n_levels)

    def test_segmented_density_ridge_fills_under_curves(self):
        X, Y = RV(Normal(0, 1) * Binomial(5, 0.4))
        sims = (X & Y).sim(500)
        sims.plot(type="segmented_density", ridge=True)
        ax = plt.gca()
        # ridge=True adds one translucent PolyCollection fill per
        # level, under the same opaque curves
        n_levels = len(np.unique(sims.array[:, 1]))
        fills = [c for c in ax.collections if isinstance(c, PolyCollection)]
        self.assertEqual(len(fills), n_levels)
        self.assertEqual(len(ax.lines), n_levels)

    def test_segmented_density_baselines_full_width_and_on_top(self):
        """Each level's baseline is a grey gridline-styled line spanning the
        full axes (x endpoints 0 and 1 in axes fractions, so it touches both
        edges like the segmented rug plot) and drawn on top of the ridges, so
        it stays continuous even under a ridge=True fill. A value grid runs
        along the continuous axis only."""
        import matplotlib.colors as mcolors

        values = np.random.normal(0, 1, 120)  # continuous x
        groups = np.repeat([0, 1, 2], 40)  # discrete y (levels)
        ax = plt.gca()
        make_segmented_density(values, groups, ax, "#56B4E9")
        grey = mcolors.to_rgba(plt.rcParams["grid.color"])
        baselines = [
            c
            for c in ax.collections
            if len(c.get_color()) and np.allclose(c.get_color()[0][:3], grey[:3])
        ]
        self.assertEqual(len(baselines), 3)
        for base in baselines:
            xs = [pt[0] for seg in base.get_segments() for pt in seg]
            self.assertEqual((min(xs), max(xs)), (0.0, 1.0))  # full width
            self.assertGreater(base.get_zorder(), 1)
        # Value grid along the continuous axis (x), not the discrete (y).
        self.assertTrue(any(gl.get_visible() for gl in ax.xaxis.get_gridlines()))
        self.assertFalse(any(gl.get_visible() for gl in ax.yaxis.get_gridlines()))

    def test_segmented_density_two_discrete_raises_friendly_error(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="segmented_density")
        self.assertIn("tile", str(cm.exception))

    def test_segmented_density_two_continuous_raises_friendly_error(self):
        X, Y = RV(Normal(0, 1) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="segmented_density")
        self.assertIn("scatter", str(cm.exception))

    def test_segmented_density_bandwidth_passes_through(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(500).plot(type="segmented_density", bandwidth=0.2)
        self.assertGreater(len(plt.gca().lines), 0)

    def test_segmented_density_sparse_level_falls_back_to_ticks(self):
        # Level 9 has a single observation: no KDE is possible, so it
        # becomes baseline tick marks (a LineCollection) plus a printed
        # note saying what happened and how to fix it.
        values = np.append(np.random.normal(0, 1, 90), 4.0)
        groups = np.append(np.repeat([0, 1, 2], 30), 9)
        ax = plt.gca()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            artists = make_segmented_density(values, groups, ax, "#56B4E9")
        self.assertIn("tick marks", out.getvalue())
        self.assertIn("9", out.getvalue())
        # 3 dense ridges + 1 tick collection, one artist per level
        self.assertEqual(len(artists), 4)

    def test_segmented_density_overlay_legend_names(self):
        values = np.random.normal(0, 1, 120)
        groups = np.repeat([0, 1, 2], 40)
        ax = plt.gca()
        make_segmented_density(values, groups, ax, "#56B4E9")
        self.assertIsNone(ax.get_legend())  # a lone batch has no legend
        make_segmented_density(values + 1, groups, ax, "#E69F00")
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(legend_texts, ["Variable 1", "Variable 2"])

    def test_density_is_a_mixed_data_alternative(self):
        # The table lists the short name "density"; on mixed data it resolves
        # to the segmented density (the "segmented_density" alias also works).
        self.assertIn("density", DEFAULT_PLOT_TYPE[("2D_mixed", True)]["alternatives"])
        self.assertIn("density", DEFAULT_PLOT_TYPE[("2D_mixed", False)]["alternatives"])
        self.assertEqual(
            PLOT_DISPLAY_NAME["segmented_density"], "Segmented Density Plot"
        )

    def test_density_short_name_is_segmented_on_mixed(self):
        """On mixed data, type='density' produces the segmented density."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(type="density", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Segmented Density Plot")

    def test_density2d_still_forces_surface_on_mixed(self):
        """type='density2d' forces the 2D surface even on mixed data."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(
            type="density2d", suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "2D Density Plot")


class TestPlot2DSegmentedHist(PlotTestCase):
    """The new type='segmented_hist' for mixed discrete/continuous data."""

    def setUp(self):
        np.random.seed(42)

    def test_segmented_hist_discrete_x_continuous_y(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        sims.plot(type="segmented_hist")
        ax = plt.gca()
        self.assertEqual(ax.get_title(), "Segmented Histogram")
        self.assertGreater(len(ax.patches), 0)
        # Discrete x -> flipped orientation: baselines on the x-axis
        n_levels = len(np.unique(sims.array[:, 0]))
        self.assertEqual(len(ax.get_xticks()), n_levels)

    def test_segmented_hist_continuous_x_discrete_y(self):
        X, Y = RV(Normal(0, 1) * Binomial(5, 0.4))
        sims = (X & Y).sim(500)
        sims.plot(type="segmented_hist")
        ax = plt.gca()
        # Discrete y -> classic orientation: baselines on the y-axis
        n_levels = len(np.unique(sims.array[:, 1]))
        self.assertGreater(len(ax.patches), 0)
        self.assertEqual(len(ax.get_yticks()), n_levels)

    def test_segmented_hist_two_discrete_raises_friendly_error(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="segmented_hist")
        self.assertIn("tile", str(cm.exception))

    def test_segmented_hist_two_continuous_raises_friendly_error(self):
        X, Y = RV(Normal(0, 1) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(200).plot(type="segmented_hist")
        self.assertIn("scatter", str(cm.exception))

    def test_segmented_hist_shares_bin_edges_across_levels(self):
        # All levels are binned on one shared grid, so across the whole
        # plot the bars' left edges take at most `bins` distinct values.
        values = np.random.normal(0, 1, 300) + np.repeat([0, 1, 2], 100)
        groups = np.repeat([0, 1, 2], 100)
        ax = plt.gca()
        make_segmented_hist(values, groups, ax, "#56B4E9", bins=10)
        lefts = {round(p.get_x(), 9) for p in ax.patches}
        self.assertLessEqual(len(lefts), 10)

    def test_segmented_hist_bins_and_normalize_pass_through(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(500).plot(type="segmented_hist", bins=12, normalize=False)
        self.assertGreater(len(plt.gca().patches), 0)

    def test_segmented_hist_sparse_level_falls_back_to_ticks(self):
        # Level 9 has a single observation: under the shared scale its
        # bar would dwarf the real histograms, so it becomes baseline
        # tick marks (a LineCollection) plus a printed note.
        values = np.append(np.random.normal(0, 1, 90), 0.0)
        groups = np.append(np.repeat([0, 1, 2], 30), 9)
        ax = plt.gca()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            artists = make_segmented_hist(values, groups, ax, "#56B4E9")
        self.assertIn("tick marks", out.getvalue())
        self.assertIn("9", out.getvalue())
        # 3 histogram rows + 1 tick collection, one artist per level
        self.assertEqual(len(artists), 4)

    def test_segmented_hist_overlay_legend_names(self):
        values = np.random.normal(0, 1, 120)
        groups = np.repeat([0, 1, 2], 40)
        ax = plt.gca()
        make_segmented_hist(values, groups, ax, "#56B4E9")
        self.assertIsNone(ax.get_legend())  # a lone batch has no legend
        make_segmented_hist(values + 1, groups, ax, "#E69F00")
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        self.assertEqual(legend_texts, ["Variable 1", "Variable 2"])

    def test_segmented_hist_baselines_full_width_and_on_top(self):
        """Each level's baseline is a grey gridline-styled line that spans the
        full axes (x endpoints 0 and 1 in axes fractions, so it touches both
        edges like the segmented rug plot) and sits on top of the bars, so it
        stays continuous across each histogram instead of vanishing where the
        bars cover it. A value grid still runs along the continuous axis
        only."""
        import matplotlib.colors as mcolors

        values = np.random.normal(0, 1, 300)  # continuous x
        groups = np.repeat([0, 1, 2], 100)  # discrete y (levels)
        ax = plt.gca()
        make_segmented_hist(values, groups, ax, "#56B4E9")
        grey = mcolors.to_rgba(plt.rcParams["grid.color"])
        baselines = [
            c
            for c in ax.collections
            if len(c.get_color()) and np.allclose(c.get_color()[0][:3], grey[:3])
        ]
        # One baseline per level, styled like the reference grid.
        self.assertEqual(len(baselines), 3)
        for base in baselines:
            self.assertEqual(base.get_linewidth()[0], plt.rcParams["grid.linewidth"])
            self.assertEqual(base.get_alpha(), plt.rcParams["grid.alpha"])
            # Full width: x endpoints are 0 and 1 (axes fractions), and it
            # sits on top of the bars (default patch zorder is 1).
            xs = [pt[0] for seg in base.get_segments() for pt in seg]
            self.assertEqual((min(xs), max(xs)), (0.0, 1.0))
            self.assertGreater(base.get_zorder(), 1)
        # Value grid runs along the continuous axis (x), not the discrete (y).
        self.assertTrue(any(gl.get_visible() for gl in ax.xaxis.get_gridlines()))
        self.assertFalse(any(gl.get_visible() for gl in ax.yaxis.get_gridlines()))

    def test_hist_is_a_mixed_data_alternative(self):
        # The table lists the short name "hist"; on mixed data it resolves to
        # the segmented histogram (the "segmented_hist" alias also works).
        self.assertIn("hist", DEFAULT_PLOT_TYPE[("2D_mixed", True)]["alternatives"])
        self.assertIn("hist", DEFAULT_PLOT_TYPE[("2D_mixed", False)]["alternatives"])
        self.assertEqual(PLOT_DISPLAY_NAME["segmented_hist"], "Segmented Histogram")

    def test_hist_short_name_is_segmented_on_mixed(self):
        """On mixed data, type='hist' produces the segmented histogram."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(type="hist", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Segmented Histogram")

    def test_hist2d_still_forces_mesh_on_mixed(self):
        """type='hist2d' forces the 2D mesh even on mixed data."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(type="hist2d", suggest=False)
        self.assertEqual(plt.gca().get_title(), "2-D Histogram")

    def test_hist_short_name_stays_2d_on_continuous(self):
        """On two continuous variables, type='hist' is still the 2D mesh."""
        RV(Normal(0, 1) * Normal(0, 1)).sim(500).plot(type="hist", suggest=False)
        self.assertEqual(plt.gca().get_title(), "2-D Histogram")


# ===========================================================================
# Marginal panel rebuild: real helpers, classify_values-routed types,
# coordinate alignment with the main panel
# ===========================================================================


class TestMarginalPanelRebuild(PlotTestCase):
    """marginal=True panels now use the redesigned 1D helpers, are routed
    through classify_values like a standalone 1D variable, and are aligned
    to the main panel's actual coordinate system instead of drifting to
    their own scheme."""

    def _marginal_axes(self, p):
        """Return (ax_marg_x, ax_marg_y) from the current figure, in the
        order RVResults.plot() creates them (x, then y, then any
        colorbar axes)."""
        others = [a for a in plt.gcf().axes if a is not p.ax]
        return others[0], others[1]

    def test_tile_marginal_coordinate_mismatch_is_fixed(self):
        """The confirmed regression case: DiscreteUniform(50,60) has
        support starting at 50, so tile's index-position discrete axis
        used to disagree completely with a real-valued marginal axis."""
        np.random.seed(0)
        X, Y = RV(DiscreteUniform(a=50, b=60) * Poisson(lam=5))
        p = (X & Y).sim(2000).plot(type="tile", marginal=True, suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertEqual(p.ax.get_xlim(), ax_marg_x.get_xlim())
        self.assertEqual(p.ax.get_ylim(), ax_marg_y.get_ylim())

    def _assert_aligned_and_populated(self, p):
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertEqual(p.ax.get_xlim(), ax_marg_x.get_xlim())
        self.assertEqual(p.ax.get_ylim(), ax_marg_y.get_ylim())
        for ax_marg in (ax_marg_x, ax_marg_y):
            n_artists = (
                len(ax_marg.patches) + len(ax_marg.collections) + len(ax_marg.lines)
            )
            self.assertGreater(n_artists, 0)
        return ax_marg_x, ax_marg_y

    def test_scatter_continuous_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(50).plot(type="scatter", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_scatter_discrete_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(50).plot(type="scatter", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_tile_discrete_discrete_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(2000).plot(type="tile", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_hist2d_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(type="hist2d", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_density2d_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(type="density2d", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_segmented_rug_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Poisson(lam=3) * Normal(0, 1))
        p = (X & Y).sim(2000).plot(type="rug", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_violin_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Poisson(lam=3) * Normal(0, 1))
        p = (X & Y).sim(2000).plot(type="violin", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_box_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Poisson(lam=3) * Normal(0, 1))
        p = (X & Y).sim(2000).plot(type="box", marginal=True, suggest=False)
        self._assert_aligned_and_populated(p)

    def test_mosaic_marginal_raises_and_points_to_marginal_column(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(500).plot(type="mosaic", marginal=True)
        self.assertIn("marginal_column", str(cm.exception))

    def test_small_n_discrete_marginal_is_dotplot(self):
        """A small-n discrete axis's marginal should be a dot plot,
        matching what that axis would show standalone -- not always
        impulse/hist regardless of sample size."""
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(50).plot(type="scatter", marginal=True, suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertTrue(hasattr(ax_marg_x, "_dotplot_state"))
        self.assertTrue(hasattr(ax_marg_y, "_dotplot_state"))

    def test_large_n_discrete_marginal_is_impulse(self):
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(2000).plot(type="tile", marginal=True, suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertTrue(hasattr(ax_marg_x, "_impulse_series"))
        self.assertTrue(hasattr(ax_marg_y, "_impulse_series"))

    def test_small_n_continuous_marginal_is_rug(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(50).plot(type="scatter", marginal=True, suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertGreater(getattr(ax_marg_x, "_rug_count", 0), 0)
        self.assertGreater(getattr(ax_marg_y, "_rug_count", 0), 0)

    def test_large_n_continuous_marginal_is_hist(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(type="hist2d", marginal=True, suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertGreater(getattr(ax_marg_x, "_hist_count", 0), 0)
        self.assertGreater(getattr(ax_marg_y, "_hist_count", 0), 0)

    def test_density_mode_gives_density_curve_marginals(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(type="density2d", marginal=True, suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertGreater(getattr(ax_marg_x, "_density_count", 0), 0)
        self.assertGreater(getattr(ax_marg_y, "_density_count", 0), 0)

    def test_hist2d_marginal_bar_edges_match_hist2d_edges(self):
        """The marginal histogram must reuse hist2d's own bin edges, not
        an independently (if coincidentally) recomputed set."""
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        sims = (X & Y).sim(2000)
        p = sims.plot(type="hist2d", marginal=True, bins=17, suggest=False)
        ax_marg_x, _ = self._marginal_axes(p)

        plt.close("all")
        from symbulate.plot import make_hist2d

        fig, ax = plt.subplots()
        _, xedges, _, _ = make_hist2d(sims.array[:, 0], sims.array[:, 1], ax, bins=17)
        plt.close(fig)

        marg_edges = sorted(
            {round(p.get_x(), 6) for p in ax_marg_x.patches}
            | {round(p.get_x() + p.get_width(), 6) for p in ax_marg_x.patches}
        )
        expected_edges = sorted({round(e, 6) for e in xedges})
        self.assertEqual(marg_edges, expected_edges)

    def test_tile_continuous_axis_marginal_bar_edges_match_tile_edges(self):
        """The marginal histogram for tile's continuous axis (mixed
        discrete x continuous data) must reuse tile's own bin edges, not
        an independently (if coincidentally) recomputed set."""
        np.random.seed(1)
        X, Y = RV(Poisson(lam=3) * Normal(0, 1))
        sims = (X & Y).sim(2000)
        p = sims.plot(type="tile", marginal=True, bins=17, suggest=False)
        _, ax_marg_y = self._marginal_axes(p)  # y is the continuous axis here

        from symbulate.plot import setup_tile_axis

        _, _, _, _, expected_edges_arr = setup_tile_axis(sims.array[:, 1], False, 17)

        marg_edges = sorted(
            {round(p.get_y(), 6) for p in ax_marg_y.patches}
            | {round(p.get_y() + p.get_height(), 6) for p in ax_marg_y.patches}
        )
        expected_edges = sorted({round(e, 6) for e in expected_edges_arr})
        self.assertEqual(marg_edges, expected_edges)


# ===========================================================================
# Discrete-axis tick label thinning (tile, segmented rug/density/hist/box,
# violin) -- regression coverage for the dense, overlapping tick labels a
# high-cardinality discrete axis used to produce (e.g. Poisson(200): 29
# distinct values, 172-224, all crowded onto one axis).
# ===========================================================================


class TestThinDiscreteTicksHelper(unittest.TestCase):
    """Direct unit tests for the shared _thin_discrete_ticks() helper."""

    def test_passthrough_below_threshold(self):
        positions = np.arange(5)
        labels = np.array([10, 20, 30, 40, 50])
        pos, lab = _thin_discrete_ticks(positions, labels, 10)
        np.testing.assert_array_equal(pos, positions)
        np.testing.assert_array_equal(lab, labels)

    def test_passthrough_at_exact_threshold(self):
        positions = np.arange(10)
        labels = np.arange(10)
        pos, lab = _thin_discrete_ticks(positions, labels, 10)
        self.assertEqual(len(pos), 10)

    def test_thins_above_threshold(self):
        positions = np.arange(29)
        labels = np.arange(200, 229)
        pos, lab = _thin_discrete_ticks(positions, labels, 20)
        self.assertLessEqual(len(pos), 20)
        self.assertEqual(len(pos), len(lab))
        # The edges are always kept so the axis's full range still reads.
        self.assertEqual(pos[0], positions[0])
        self.assertEqual(pos[-1], positions[-1])
        # Every kept label is one of the real distinct values.
        self.assertTrue(np.isin(lab, labels).all())


class TestPlot2DDiscreteTickThinning(PlotTestCase):
    """Every 2D plot helper that labels a discrete axis thins its tick
    labels past MAX_DISCRETE_TICKS, while still drawing every distinct
    value's cell/band/box/violin -- only the *displayed labels* are
    thinned, never the underlying data."""

    def test_tile_dense_discrete_axis_gets_a_consistent_evenly_spaced_scale(self):
        x = np.resize(np.arange(29), 3000)  # 29 distinct values (like Poisson(200))
        y = np.resize(np.arange(25), 3000)  # 25 distinct values
        ax = plt.gca()
        mesh = make_tile(x, y, ax, discrete_x=True, discrete_y=True)
        # Every distinct value still gets its own cell.
        self.assertEqual(mesh.get_array().shape, (25, 29))
        # Both axes are laid out on a real number line, so matplotlib's own
        # locator picks a *constant step* between tick values (0, 3, 6, ...)
        # instead of a hand-picked subset of whichever values happened to
        # occur, and keeps roughly to MAX_DISCRETE_TICKS.
        x_ticks, y_ticks = visible_ticks(ax, "x"), visible_ticks(ax, "y")
        self.assertLessEqual(len(x_ticks), MAX_DISCRETE_TICKS + 2)
        self.assertLessEqual(len(y_ticks), MAX_DISCRETE_TICKS + 2)
        self.assertEqual(len(set(np.diff(x_ticks))), 1)  # one constant step
        self.assertEqual(len(set(np.diff(y_ticks))), 1)
        # Labels stay horizontal, matching every other axis in the package.
        self.assertTrue(all(t.get_rotation() == 0 for t in ax.get_xticklabels()))

    def test_tile_no_thinning_when_both_axes_are_small(self):
        """Two discrete axes with the same low cardinality are unaffected:
        every value is still labeled on both (step of 1), no rotation."""
        x = np.resize(np.arange(5), 500)
        y = np.resize(np.arange(5), 500)
        ax = plt.gca()
        make_tile(x, y, ax, discrete_x=True, discrete_y=True)
        self.assertEqual([int(t) for t in visible_ticks(ax, "x")], [0, 1, 2, 3, 4])
        self.assertEqual([int(t) for t in visible_ticks(ax, "y")], [0, 1, 2, 3, 4])
        self.assertTrue(all(t.get_rotation() == 0 for t in ax.get_xticklabels()))

    def test_tile_mismatched_axes_each_get_their_own_consistent_scale(self):
        """When one axis has far fewer distinct values than the other, each
        axis gets its own evenly spaced, constant-step ticks -- the tick
        *count* can now differ between axes (5 vs several), because a
        consistent numeric scale on each axis takes priority over forcing
        an identical count on both (see DECISIONS.md)."""
        x = np.resize(np.arange(5), 3000)  # 5 distinct values
        y = np.resize(np.arange(30), 3000)  # 30 distinct values
        ax = plt.gca()
        make_tile(x, y, ax, discrete_x=True, discrete_y=True)
        x_ticks, y_ticks = visible_ticks(ax, "x"), visible_ticks(ax, "y")
        self.assertEqual([int(t) for t in x_ticks], [0, 1, 2, 3, 4])
        self.assertEqual(len(set(np.diff(y_ticks))), 1)  # y still has a constant step
        self.assertGreater(len(y_ticks), len(x_ticks))  # counts need not match

    def test_tile_unobserved_value_shows_as_empty_column_not_dropped(self):
        """A whole-number discrete axis fills in any unobserved-but-possible
        value within its range instead of silently skipping it -- here 3
        never occurs, so the axis still has 6 cells (0-5) and that column's
        count is exactly zero, instead of the axis silently shrinking to 5
        cells for the 5 values that did occur."""
        x = np.resize(np.array([0, 1, 2, 4, 5]), 500)  # 3 never occurs
        y = np.resize(np.array([0, 1]), 500)
        ax = plt.gca()
        mesh = make_tile(x, y, ax, discrete_x=True, discrete_y=True, normalize=False)
        self.assertEqual(mesh.get_array().shape[1], 6)  # 0,1,2,3,4,5 -- 3 filled in
        self.assertEqual(mesh.get_array()[:, 3].sum(), 0)  # the unobserved column

    def test_segmented_rug_thins_dense_discrete_ticks_but_draws_every_band(self):
        x = np.random.normal(0, 1, 3000)  # continuous
        y = np.resize(np.arange(29), 3000)  # discrete, 29 levels
        ax = plt.gca()
        color = get_next_color(ax)
        ticks = make_segmented_rug(x, y, ax, color, discrete_x=False, discrete_y=True)
        self.assertEqual(len(ticks), 29)  # every level still gets its own band
        self.assertLessEqual(len(ax.get_yticks()), MAX_DISCRETE_TICKS)

    def test_segmented_density_thins_dense_discrete_ticks_but_draws_every_level(self):
        x = np.random.normal(0, 1, 3000)
        y = np.resize(np.arange(25), 3000)  # discrete, 25 levels
        ax = plt.gca()
        color = get_next_color(ax)
        artists = make_segmented_density(
            x, y, ax, color, discrete_x=False, discrete_y=True
        )
        self.assertEqual(len(artists), 25)  # every level still gets its own ridge
        self.assertLessEqual(len(ax.get_yticks()), MAX_DISCRETE_TICKS)

    def test_segmented_hist_thins_dense_discrete_ticks_but_draws_every_level(self):
        x = np.random.normal(0, 1, 3000)
        y = np.resize(np.arange(25), 3000)  # discrete, 25 levels
        ax = plt.gca()
        color = get_next_color(ax)
        artists = make_segmented_hist(
            x, y, ax, color, discrete_x=False, discrete_y=True
        )
        self.assertEqual(len(artists), 25)  # every level still gets its own histogram
        self.assertLessEqual(len(ax.get_yticks()), MAX_DISCRETE_TICKS)

    def test_grouped_boxplot_thins_dense_discrete_ticks_but_draws_every_box(self):
        x = np.resize(
            np.arange(22), 3000
        )  # discrete, 22 levels (over MAX_DISCRETE_TICKS)
        y = np.random.normal(0, 1, 3000)
        ax = plt.gca()
        color = get_next_color(ax)
        boxes = make_grouped_boxplot(x, y, ax, color, discrete_x=True, discrete_y=False)
        self.assertEqual(len(boxes["boxes"]), 22)  # every level still gets its own box
        self.assertLessEqual(len(ax.get_xticks()), MAX_DISCRETE_TICKS)
        # Labels stay horizontal, matching every other axis in the package.
        self.assertTrue(all(t.get_rotation() == 0 for t in ax.get_xticklabels()))

    def test_violin_thins_dense_discrete_ticks_but_draws_every_violin(self):
        positions = list(
            range(22)
        )  # 22 distinct group values (over MAX_DISCRETE_TICKS)
        groups = np.resize(np.array(positions), 3000)
        values = np.random.normal(0, 1, 3000)
        data = np.column_stack([groups, values])
        ax = plt.gca()
        color = get_next_color(ax)
        violins, _boxplot = make_violin(data, positions, ax, color, "x", 0.5)
        self.assertEqual(len(violins["bodies"]), 22)  # every group still gets a violin
        self.assertLessEqual(len(ax.get_xticks()), MAX_DISCRETE_TICKS)
        # Labels stay horizontal, matching every other axis in the package.
        self.assertTrue(all(t.get_rotation() == 0 for t in ax.get_xticklabels()))


# ===========================================================================
# Default plot lookup wired into RVResults.plot() + suggestion note
# ===========================================================================


class TestDefaultLookupDispatch(PlotTestCase):
    """classify_values + DEFAULT_PLOT_TYPE now drive .plot()'s defaults."""

    def setUp(self):
        np.random.seed(42)

    def test_small_continuous_defaults_to_rug(self):
        RV(Normal(0, 1)).sim(50).plot()
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0)  # rug ticks
        self.assertFalse(ax.yaxis.get_visible())  # standalone rug look

    def test_large_continuous_defaults_to_hist(self):
        RV(Normal(0, 1)).sim(600).plot()
        self.assertGreater(len(plt.gca().patches), 0)

    def test_2d_mixed_large_defaults_to_tile(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(500).plot()
        self.assertEqual(plt.gca().get_title(), "Tile Plot")

    def test_2d_both_axes_under_K_2D_is_tile(self):
        """Two discrete axes both within the per-axis cap -> tile."""
        RV(BoxModel(list(range(5))) * BoxModel(list(range(5)))).sim(3000).plot(
            suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "Tile Plot")

    def test_2d_one_axis_over_K_2D_stays_tile(self):
        """One axis over the per-axis cap bins only that axis -> still a
        (mixed) tile, not yet a 2-D histogram: per-axis independence."""
        RV(BoxModel(list(range(5))) * BoxModel(list(range(40)))).sim(3000).plot(
            suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "Tile Plot")

    def test_2d_both_axes_over_K_2D_is_hist2d(self):
        """Both axes over the per-axis cap now bin to a 2-D histogram -- the
        behavior change from the budget model (both axes were a tile under the
        old flat single-threshold rule)."""
        RV(BoxModel(list(range(40))) * BoxModel(list(range(40)))).sim(3000).plot(
            suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "2-D Histogram")

    def test_2d_mixed_small_defaults_to_segmented_rug(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(60).plot()
        self.assertEqual(plt.gca().get_title(), "Segmented Rug Plot")

    def test_2d_discrete_large_defaults_to_tile(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        (X & Y).sim(500).plot()
        self.assertEqual(plt.gca().get_title(), "Tile Plot")

    def test_2d_discrete_small_defaults_to_scatter(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        (X & Y).sim(40).plot()
        self.assertEqual(plt.gca().get_title(), "2D Scatter Plot")

    def test_2d_explicit_alias_types_work(self):
        """The lookup-table tokens are accepted as explicit type= values."""
        X, Y = RV(Normal(0, 1) ** 2)
        sims = (X & Y).sim(200)
        sims.plot(type="hist2d")
        self.assertEqual(plt.gca().get_title(), "2-D Histogram")
        plt.close("all")
        sims.plot(type="density2d")
        self.assertEqual(plt.gca().get_title(), "2D Density Plot")
        plt.close("all")
        Xm, Ym = RV(Binomial(5, 0.4) * Normal(0, 1))
        (Xm & Ym).sim(200).plot(type="segmented_rug")
        self.assertEqual(plt.gca().get_title(), "Segmented Rug Plot")

    def test_2d_mixed_explicit_box_type_works(self):
        """'box' is a listed alternative for 2D_mixed data and dispatches
        to make_grouped_boxplot when explicitly requested."""
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(200).plot(type="box")
        self.assertEqual(plt.gca().get_title(), "Box Plot")

    def test_every_listed_alternative_is_choosable(self):
        """Every default and alternative in DEFAULT_PLOT_TYPE renders via
        type= on representative data -- no listed option is a silent
        no-op. (1D_categorical is skipped: it is not reachable as a
        numeric RVResults.)"""

        def joint(dist):
            X, Y = RV(dist)
            return X & Y

        reps = {
            "1D_discrete": lambda: RV(Binomial(5, 0.4)).sim(200),
            "1D_continuous": lambda: RV(Normal(0, 1)).sim(200),
            "2D_dd": lambda: joint(Binomial(5, 0.4) ** 2).sim(200),
            "2D_cc": lambda: joint(Normal(0, 1) ** 2).sim(200),
            "2D_mixed": lambda: joint(Binomial(5, 0.4) * Normal(0, 1)).sim(200),
        }
        for (configuration, _small), entry in DEFAULT_PLOT_TYPE.items():
            if configuration not in reps:
                continue
            for t in [entry["default"]] + entry["alternatives"]:
                np.random.seed(0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = reps[configuration]().plot(type=t)
                drawn = (
                    len(p.ax.collections)
                    + len(p.ax.lines)
                    + len(p.ax.patches)
                    + len(p.ax.images)
                )
                self.assertGreater(drawn, 0, f"{configuration} type={t!r} drew nothing")
                plt.close("all")


class TestSuggestionNote(PlotTestCase):
    """The 'Currently Showing / Alternative Plots' note under plots."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(600)

    def _plot_output(self, **kwargs):
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.sims.plot(**kwargs)
        return buf.getvalue()

    def test_suggest_true_prints_note(self):
        out = self._plot_output(suggest=True)
        self.assertIn("Currently Showing: Histogram (Default)", out)
        self.assertIn("Alternative Plots:", out)

    def test_suggest_false_prints_nothing(self):
        out = self._plot_output(suggest=False)
        self.assertEqual(out, "")

    def test_explicit_type_marks_the_default_alternative(self):
        out = self._plot_output(type="density", suggest=True)
        self.assertIn("Currently Showing: Density Plot", out)
        self.assertIn("Histogram (Default)", out)

    def test_suggest_none_shows_once_per_session(self):
        import symbulate.plot as _plotmod_check  # noqa: F401 (module import)
        import importlib

        plotmod = importlib.import_module("symbulate.plot")
        plotmod._suggestion_shown = False
        first = self._plot_output()
        plt.close("all")
        second = self._plot_output()
        self.assertIn("Currently Showing", first)
        self.assertEqual(second, "")

    def test_2d_note_uses_2d_display_names(self):
        import io
        import contextlib

        X, Y = RV(Normal(0, 1) ** 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            (X & Y).sim(500).plot(type="hist", suggest=True)
        self.assertIn("Currently Showing: 2D Histogram (Default)", buf.getvalue())

    def test_mosaic_equal_width_note_says_stacked_plot(self):
        """equal_width=True's suggestion note must match its axes title
        ("Stacked Plot"), not still read "Mosaic Plot"."""
        import io
        import contextlib

        X, Y = RV(Binomial(5, 0.4) ** 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            (X & Y).sim(500).plot(type="mosaic", equal_width=True, suggest=True)
        self.assertIn("Currently Showing: Stacked Plot", buf.getvalue())

    def test_mosaic_without_equal_width_note_still_says_mosaic_plot(self):
        import io
        import contextlib

        X, Y = RV(Binomial(5, 0.4) ** 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            (X & Y).sim(500).plot(type="mosaic", suggest=True)
        self.assertIn("Currently Showing: Mosaic Plot", buf.getvalue())


class TestJitterNote(PlotTestCase):
    """The jitter-options note under a scatter of two discrete variables."""

    def setUp(self):
        import importlib

        np.random.seed(42)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        self.discrete_sims = (X & Y).sim(40)
        Xc, Yc = RV(Normal(0, 1) ** 2)
        self.continuous_sims = (Xc & Yc).sim(40)
        # Reset the once-per-session flag so each test starts fresh.
        self.plotmod = importlib.import_module("symbulate.plot")
        self.plotmod._jitter_suggestion_shown = False

    def _plot_output(self, sims, **kwargs):
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sims.plot(type="scatter", **kwargs)
        return buf.getvalue()

    def test_note_names_current_layout_and_alternatives(self):
        out = self._plot_output(self.discrete_sims, suggest=True)
        self.assertIn('Currently Using: jitter="spiral"', out)
        self.assertIn("Other Jitter Options:", out)
        self.assertIn('jitter="random"', out)
        self.assertIn("jitter=False", out)

    def test_note_reflects_explicit_jitter_choice(self):
        out = self._plot_output(self.discrete_sims, jitter="bins", suggest=True)
        self.assertIn('Currently Using: jitter="bins"', out)
        self.assertIn('jitter="spiral"', out)

    def test_no_note_for_continuous_scatter(self):
        out = self._plot_output(self.continuous_sims, suggest=True)
        self.assertNotIn("Jitter", out)

    def test_suggest_false_silences_note(self):
        out = self._plot_output(self.discrete_sims, suggest=False)
        self.assertEqual(out, "")

    def test_note_shows_once_per_session(self):
        first = self._plot_output(self.discrete_sims)
        plt.close("all")
        second = self._plot_output(self.discrete_sims)
        self.assertIn("Other Jitter Options:", first)
        self.assertNotIn("Other Jitter Options:", second)


# ===========================================================================
# Distribution.plot()
# ===========================================================================


class TestDistributionPlotContinuous(PlotTestCase):
    """Distribution.plot() for continuous named distributions."""

    def test_normal_produces_one_line(self):
        Normal(0, 1).plot()
        self.assertEqual(len(plt.gca().lines), 1)

    def test_normal_pdf_is_non_negative(self):
        Normal(0, 1).plot()
        self.assertTrue(np.all(plt.gca().lines[0].get_ydata() >= 0))

    def test_normal_default_xlim_covers_tails(self):
        """Default xlim should include the 0.1%–99.9% range (≈ ±3.09 for N(0,1))."""
        Normal(0, 1).plot()
        xlim = plt.gca().get_xlim()
        self.assertLessEqual(xlim[0], -3.0)
        self.assertGreaterEqual(xlim[1], 3.0)

    def test_exponential_xlim_left_edge_is_zero(self):
        """Exponential is only defined for x >= 0."""
        Exponential(rate=1).plot()
        self.assertAlmostEqual(plt.gca().get_xlim()[0], 0.0, places=5)

    def test_uniform_xlim_matches_specified_bounds(self):
        Uniform(a=2, b=5).plot()
        xlim = plt.gca().get_xlim()
        self.assertAlmostEqual(xlim[0], 2.0, places=5)
        self.assertAlmostEqual(xlim[1], 5.0, places=5)

    def test_beta_xlim_is_zero_to_one(self):
        Beta(a=2, b=3).plot()
        xlim = plt.gca().get_xlim()
        self.assertAlmostEqual(xlim[0], 0.0, places=5)
        self.assertAlmostEqual(xlim[1], 1.0, places=5)

    def test_custom_xlim_is_respected(self):
        Normal(0, 1).plot(xlim=(-1, 1))
        xlim = plt.gca().get_xlim()
        self.assertAlmostEqual(xlim[0], -1.0, places=5)
        self.assertAlmostEqual(xlim[1], 1.0, places=5)

    def test_two_distributions_overlay_on_one_axes(self):
        Normal(0, 1).plot()
        Normal(2, 1).plot()
        self.assertEqual(len(plt.gca().lines), 2)

    def test_gamma_pdf_is_everywhere_finite(self):
        Gamma(shape=2, rate=1).plot()
        self.assertTrue(np.all(np.isfinite(plt.gca().lines[0].get_ydata())))


class TestDistributionPlotDiscrete(PlotTestCase):
    """Distribution.plot() for discrete named distributions."""

    def test_binomial_pmf_is_dots_plus_dashed_line(self):
        """Discrete distributions draw a filled dot at each pmf value plus a
        dashed connecting line -- one Line2D (dashed) and one scatter
        collection (the dots)."""
        Binomial(n=10, p=0.4).plot()
        ax = plt.gca()
        self.assertEqual(len(ax.lines), 1)
        self.assertEqual(len(ax.collections), 1)
        self.assertIn(ax.lines[-1].get_linestyle(), ("--", "dashed"))

    def test_pmf_line_is_dashed_dot_to_dot(self):
        """The pmf connecting line is a straight dot-to-dot polyline (one
        point per integer value), dashed -- not a fine-grained spline."""
        Poisson(lam=4).plot()
        ax = plt.gca()
        (line,) = ax.get_lines()
        # one point per integer in the support, not a 100+-point spline
        self.assertLess(len(line.get_xdata()), 40)
        self.assertIn(line.get_linestyle(), ("--", "dashed"))

    def test_discrete_plot_does_not_force_spine_to_zero(self):
        """Distribution.plot() no longer pins the bottom spine at y=0, so a
        distribution centered far from zero keeps a normal axis spine."""
        Binomial(n=200, p=0.5).plot()  # centered ~100, far from 0
        self.assertNotEqual(plt.gca().spines["bottom"].get_position(), "zero")

    def test_true_pmf_overlays_impulse_with_dots_and_dashed_line(self):
        """Overlaying a discrete distribution on an impulse plot adds its
        pmf dots (a scatter collection) and a dashed connecting line."""
        from matplotlib.collections import PathCollection

        RV(Poisson(5)).sim(500).plot(type="impulse")
        Poisson(5).plot()
        ax = plt.gca()
        self.assertTrue(any(isinstance(c, PathCollection) for c in ax.collections))
        self.assertIn(ax.lines[-1].get_linestyle(), ("--", "dashed"))

    def test_binomial_xlim_covers_full_support(self):
        """Support of Binomial(10, p) is {0, …, 10}."""
        Binomial(n=10, p=0.5).plot()
        xlim = plt.gca().get_xlim()
        self.assertLessEqual(xlim[0], 0.0)
        self.assertGreaterEqual(xlim[1], 10.0)

    def test_bernoulli_xlim_covers_zero_and_one(self):
        Bernoulli(p=0.4).plot()
        xlim = plt.gca().get_xlim()
        self.assertLessEqual(xlim[0], 0.0)
        self.assertGreaterEqual(xlim[1], 1.0)

    def test_geometric_xlim_left_edge_at_most_one(self):
        """Geometric is defined for x >= 1."""
        Geometric(p=0.3).plot()
        self.assertLessEqual(plt.gca().get_xlim()[0], 1.0)

    def test_pmf_values_are_non_negative(self):
        """The pmf line's y-values (the probabilities) must all be >= 0."""
        Poisson(lam=4).plot()
        (line,) = plt.gca().get_lines()
        ys = np.asarray(line.get_ydata())
        self.assertTrue(np.all(ys >= 0))

    def test_multivariate_normal_plot_raises(self):
        """MultivariateNormal.plot() raises (not yet implemented)."""
        with self.assertRaises(Exception):
            MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]]).plot()

    def test_multinomial_plot_raises(self):
        """Multinomial.plot() raises (not yet implemented)."""
        with self.assertRaises(Exception):
            Multinomial(n=10, p=[0.2, 0.3, 0.5]).plot()


# ===========================================================================
# Stochastic process path plots
# ===========================================================================


class TestProcessPlots(PlotTestCase):
    """Plots of simulated stochastic process paths (dim > 2 / TimeFunctions)."""

    def test_markov_chain_n_sims_equals_n_lines(self):
        X = RV(MarkovChain([[0.5, 0.5], [0.3, 0.7]], [0.5, 0.5]))
        X.sim(5).plot()
        self.assertEqual(len(plt.gca().lines), 5)

    def test_markov_chain_xlabel_is_index(self):
        X = RV(MarkovChain([[0.5, 0.5], [0.3, 0.7]], [0.5, 0.5]))
        X.sim(3).plot()
        self.assertEqual(plt.gca().get_xlabel(), "Index")

    def test_poisson_process_n_sims_equals_n_lines(self):
        X = RV(PoissonProcess(rate=2))
        X.sim(4).plot()
        self.assertEqual(len(plt.gca().lines), 4)

    def test_ensemble_paths_share_one_color(self):
        """Many realizations of one process read as an ensemble: one color."""
        X = RV(MarkovChain([[0.5, 0.5], [0.3, 0.7]], [0.5, 0.5]))
        X.sim(5).plot()
        colors = {line.get_color() for line in plt.gca().lines}
        self.assertEqual(len(colors), 1)

    def test_ensemble_plot_has_no_legend(self):
        """Per-path 'Path k' legend entries are suppressed for ensembles."""
        X = RV(MarkovChain([[0.5, 0.5], [0.3, 0.7]], [0.5, 0.5]))
        X.sim(5).plot()
        self.assertIsNone(plt.gca().get_legend())

    def test_ensemble_paths_are_solid_lines_without_markers(self):
        """The old '.--' dot-dash format is gone: solid line, no marker."""
        X = RV(PoissonProcess(rate=2))
        X.sim(3).plot()
        for line in plt.gca().lines:
            self.assertEqual(line.get_linestyle(), "-")
            self.assertEqual(line.get_marker(), "None")

    def test_two_time_function_plots_get_distinct_colors_and_legend(self):
        """Separate .plot() calls on realizations overlay like the prototype."""
        X = RV(PoissonProcess(rate=2))
        sims = X.sim(2)
        sims.get(0).plot()
        sims.get(1).plot()
        ax = plt.gca()
        self.assertEqual(len({line.get_color() for line in ax.lines}), 2)
        legend = ax.get_legend()
        self.assertIsNotNone(legend)
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertEqual(labels, ["Path 1", "Path 2"])


# ===========================================================================
# make_sample_path
# ===========================================================================


class TestMakeSamplePath(PlotTestCase):
    """The sample path helper: solid line, package defaults, auto legend."""

    def setUp(self):
        np.random.seed(42)
        self.times = np.arange(101)
        self.values = np.concatenate(
            [[0], np.cumsum(np.random.choice([-1, 1], size=100))]
        )

    def draw_path(self, **kwargs):
        ax = plt.gca()
        return make_sample_path(
            self.times, self.values, ax, get_next_color(ax), **kwargs
        )

    def test_draws_solid_line_without_markers(self):
        line = self.draw_path()
        self.assertEqual(line.get_linestyle(), "-")
        self.assertEqual(line.get_marker(), "None")

    def test_default_linewidth_and_alpha_are_package_standards(self):
        line = self.draw_path()
        self.assertEqual(line.get_linewidth(), SAMPLE_PATH_LINEWIDTH)
        self.assertEqual(line.get_alpha(), SAMPLE_PATH_ALPHA)

    def test_linewidth_and_alpha_overrides(self):
        line = self.draw_path(linewidth=2.5, alpha=0.7)
        self.assertEqual(line.get_linewidth(), 2.5)
        self.assertEqual(line.get_alpha(), 0.7)

    def test_default_labels_and_title(self):
        self.draw_path()
        ax = plt.gca()
        self.assertEqual(ax.get_xlabel(), "Time")
        self.assertEqual(ax.get_ylabel(), "Value")
        self.assertEqual(ax.get_title(), "Sample Path")

    def test_xlabel_ylabel_overrides(self):
        self.draw_path(xlabel="t (seconds)", ylabel="Position")
        ax = plt.gca()
        self.assertEqual(ax.get_xlabel(), "t (seconds)")
        self.assertEqual(ax.get_ylabel(), "Position")

    def test_lone_path_has_no_legend(self):
        self.draw_path()
        self.assertIsNone(plt.gca().get_legend())

    def test_second_path_turns_on_path_k_legend(self):
        self.draw_path()
        self.draw_path()
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertEqual(labels, ["Path 1", "Path 2"])

    def test_custom_label_replaces_auto_name(self):
        self.draw_path(label="Fair coin")
        self.draw_path(label="Biased coin")
        labels = [text.get_text() for text in plt.gca().get_legend().get_texts()]
        self.assertEqual(labels, ["Fair coin", "Biased coin"])

    def test_underscore_label_is_kept_out_of_the_legend(self):
        """RVResults.plot suppresses ensemble legends via '_nolegend_'."""
        self.draw_path(label="_nolegend_")
        self.draw_path(label="_nolegend_")
        self.assertIsNone(plt.gca().get_legend())


# ===========================================================================
# Error handling
# ===========================================================================


class TestPlottingErrors(PlotTestCase):
    """Tests that incorrect usage raises clear exceptions."""

    def test_plotting_raw_probability_space_results_raises(self):
        """Results from a ProbabilitySpace (not RVResults) must raise."""
        P = BoxModel([1, 2, 3, 4, 5, 6])
        with self.assertRaises(Exception):
            P.sim(100).plot()

    def test_error_message_guides_user_to_rv(self):
        """The error should mention 'RV' to point the student in the right direction."""
        P = BoxModel([1, 2, 3, 4, 5, 6])
        try:
            P.sim(100).plot()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("RV", str(e))

    def test_non_string_non_tuple_type_raises(self):
        """A numeric type= argument should raise immediately."""
        sims = RV(Normal(0, 1)).sim(100)
        with self.assertRaises(Exception):
            sims.plot(type=99)

    def test_marginal_as_bare_type_string_raises_helpful_error(self):
        """type="marginal" is no longer valid -- marginal is now its own
        keyword argument, not a type= value."""
        X, Y = RV(Normal(0, 1) ** 2)
        sims = (X & Y).sim(100)
        with self.assertRaises(ValueError) as cm:
            sims.plot(type="marginal")
        self.assertIn("marginal=True", str(cm.exception))

    def test_marginal_inside_type_list_raises_helpful_error(self):
        """type=("hist", "marginal") -- the old way of combining a main
        plot type with marginal panels -- must also raise, not silently
        ignore the stray "marginal" token."""
        X, Y = RV(Normal(0, 1) ** 2)
        sims = (X & Y).sim(100)
        with self.assertRaises(ValueError) as cm:
            sims.plot(type=("hist", "marginal"))
        self.assertIn("marginal=True", str(cm.exception))


# ===========================================================================
# SymbulatePlot wrapper object
# ===========================================================================


class TestSymbulatePlotWrapper(PlotTestCase):
    """Every .plot() method returns a SymbulatePlot wrapper (not None).

    The wrapper's repr is empty so Jupyter prints nothing below the
    plot, and it exposes the matplotlib axes as .ax for the future
    composition API.
    """

    def test_repr_is_empty_string(self):
        """Jupyter must print nothing for the returned object."""
        p = RV(Normal(0, 1)).sim(100).plot()
        self.assertEqual(repr(p), "")

    def test_wrapper_exposes_axes(self):
        p = RV(Normal(0, 1)).sim(100).plot()
        self.assertIs(p.ax, plt.gca())

    def test_1d_continuous_plot_returns_wrapper(self):
        p = RV(Normal(0, 1)).sim(100).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_1d_discrete_plot_returns_wrapper(self):
        p = RV(Binomial(n=10, p=0.4)).sim(100).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_2d_scatter_plot_returns_wrapper(self):
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(100).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_2d_marginal_plot_returns_wrapper(self):
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(100).plot(marginal=True)
        self.assertIsInstance(p, SymbulatePlot)

    def test_distribution_plot_returns_wrapper(self):
        p = Normal(0, 1).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_process_path_plot_returns_wrapper(self):
        """Sample-path plots (the dim=None branch) also return the wrapper."""
        X = RV(PoissonProcess(rate=2))
        p = X.sim(3).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_time_function_result_plot_returns_wrapper(self):
        """plot() on an individual simulated TimeFunction returns the wrapper."""
        X = RV(PoissonProcess(rate=2))
        p = X.sim(1).get(0).plot()
        self.assertIsInstance(p, SymbulatePlot)

    def test_module_level_plot_function_returns_wrapper(self):
        """The public symbulate.plot(...) helper returns the wrapper."""
        sims = RV(Normal(0, 1)).sim(100)
        p = symbulate_plot(sims)
        self.assertIsInstance(p, SymbulatePlot)

    def test_module_level_plot_matplotlib_fallback_returns_wrapper(self):
        """symbulate.plot(...) on raw data (matplotlib fallback) returns the wrapper."""
        p = symbulate_plot([1, 2, 3], [4, 5, 6])
        self.assertIsInstance(p, SymbulatePlot)

    def test_distribution_plot_wrapper_wraps_specified_ax(self):
        """When an explicit ax= is passed, the wrapper must expose that axes."""
        _, (ax1, ax2) = plt.subplots(1, 2)
        p = Normal(0, 1).plot(ax=ax1)
        self.assertIs(p.ax, ax1)


# ===========================================================================
# Figure and axes management
# ===========================================================================


class TestAxesManagement(PlotTestCase):
    """Correct axes behavior and figure hygiene."""

    def test_two_sequential_plots_share_one_axes(self):
        """Two .plot() calls in one figure should not create extra axes."""
        sims = RV(Normal(0, 1)).sim(300)
        sims.plot()
        sims.plot()
        self.assertEqual(len(plt.gcf().axes), 1)

    def test_distribution_plot_on_specified_ax(self):
        """Distribution.plot(ax=ax1) must draw on ax1, leave ax2 empty."""
        _, (ax1, ax2) = plt.subplots(1, 2)
        Normal(0, 1).plot(ax=ax1)
        self.assertEqual(len(ax1.lines), 1)
        self.assertEqual(len(ax2.lines), 0)

    def test_close_all_leaves_no_open_figures(self):
        """Sanity check that plt.close('all') genuinely clears figures."""
        RV(Normal(0, 1)).sim(100).plot()
        plt.close("all")
        self.assertEqual(len(plt.get_fignums()), 0)


# ===========================================================================
# classify_values: discrete-ish / small-n classification
# ===========================================================================


class TestClassifyData(unittest.TestCase):
    """The two-boolean classifier that replaces is_discrete."""

    def test_narrow_int_large_n_is_discrete(self):
        discrete_ish, small_n = classify_values(np.array([0, 1, 2, 1, 3] * 400))
        self.assertTrue(discrete_ish)
        self.assertFalse(small_n)

    def test_all_unique_float_is_continuous(self):
        discrete_ish, small_n = classify_values(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertFalse(discrete_ish)
        self.assertTrue(small_n)

    def test_wide_support_int_is_continuous(self):
        """Wide-support integers (e.g. Binomial(10000, 0.5)) read as continuous."""
        discrete_ish, _ = classify_values(np.arange(200))
        self.assertFalse(discrete_ish)

    def test_repeated_float_narrow_support_is_discrete(self):
        """Float data with few repeated values is a user-defined finite support."""
        discrete_ish, small_n = classify_values(np.array([1.0, 1.5, 2.71, 4.0] * 250))
        self.assertTrue(discrete_ish)
        self.assertFalse(small_n)

    def test_string_categorical_is_discrete(self):
        discrete_ish, _ = classify_values(np.array(["H", "T"] * 60))
        self.assertTrue(discrete_ish)

    def test_many_category_strings_still_discrete(self):
        """A wide-support categorical (52 labels) must not read as continuous."""
        discrete_ish, _ = classify_values(np.array([f"c{i % 52}" for i in range(1000)]))
        self.assertTrue(discrete_ish)

    def test_boolean_is_discrete(self):
        discrete_ish, _ = classify_values(np.array([True, False, True, True]))
        self.assertTrue(discrete_ish)

    def test_degenerate_constant_is_discrete(self):
        discrete_ish, _ = classify_values(np.full(500, 5))
        self.assertTrue(discrete_ish)

    def test_small_n_boundary(self):
        """small_n is True below N_SMALL_THRESHOLD and False at/above it."""
        _, small_below = classify_values(np.arange(N_SMALL_THRESHOLD - 1))
        _, small_at = classify_values(np.arange(N_SMALL_THRESHOLD))
        self.assertTrue(small_below)
        self.assertFalse(small_at)

    def test_1d_budget_boundary_at_B_1D(self):
        """Default threshold is the 1-D budget B_1D: a numeric variable flips
        discrete -> continuous just above B_1D distinct values. Tested with
        all-distinct data so the large-n repeat-density rescue (which requires
        repeats) cannot fire and the base budget rule is isolated."""
        self.assertTrue(classify_values(np.arange(B_1D))[0])
        self.assertFalse(classify_values(np.arange(B_1D + 1))[0])

    def test_2d_per_axis_budget_at_K_2D(self):
        """With the per-axis 2-D cap K_2D passed in, an axis flips discrete ->
        continuous just above K_2D distinct values. All-distinct data isolates
        the base budget rule from the large-n repeat-density rescue."""
        self.assertTrue(classify_values(np.arange(K_2D), n_unique_threshold=K_2D)[0])
        self.assertFalse(
            classify_values(np.arange(K_2D + 1), n_unique_threshold=K_2D)[0]
        )

    def test_over_budget_discrete_distribution_large_n_is_discrete(self):
        """Large-n secondary rule: a genuinely discrete distribution whose
        support just overruns the budget still reads as discrete-ish.

        Poisson(15) and Binomial(70, 0.5) land around 30-33 distinct values at
        n=10000 -- right at or a hair past B_1D=30 -- so a bare distinct-count
        budget would call them continuous on the seeds where they top 30. The
        repeat-density clause rescues them, reliably across seeds. Samples are
        generated with numpy (mirroring RV(Poisson(15)).sim(10000) etc.) so the
        seeds are fixed and the test is deterministic."""
        for seed in range(15):
            rng = np.random.default_rng(seed)
            poisson = rng.poisson(15, 10000)
            binomial = rng.binomial(70, 0.5, 10000)
            self.assertTrue(
                classify_values(poisson)[0],
                msg=f"Poisson(15) seed {seed} should classify as discrete-ish",
            )
            self.assertTrue(
                classify_values(binomial)[0],
                msg=f"Binomial(70, 0.5) seed {seed} should classify as discrete-ish",
            )

    def test_rounded_float_continuous_large_n_stays_continuous(self):
        """The large-n rule must not regress the failure mode is_discrete was
        replaced for: a genuinely continuous variable, even rounded so its
        values repeat, spreads over far more than the unique-value ceiling and
        stays continuous -- reliably across seeds. N(0, 1) rounded to two
        decimals has hundreds of distinct values at n=10000."""
        for seed in range(15):
            rng = np.random.default_rng(seed)
            rounded = np.round(rng.normal(0, 1, 10000), 2)
            self.assertFalse(
                classify_values(rounded)[0],
                msg=f"rounded continuous seed {seed} should stay continuous",
            )

    def test_over_budget_uniform_large_n_is_discrete_via_repeat_rule(self):
        """A synthetic just-over-budget integer support (45 distinct values,
        past B_1D) with heavy repeats at large n is caught by the secondary
        rule -- the deterministic core of the Poisson/Binomial case."""
        over_budget = np.repeat(np.arange(45), 300)  # 45 distinct, each x300
        self.assertTrue(classify_values(over_budget)[0])

    def test_repeat_rule_bounded_by_unique_value_ceiling(self):
        """The repeat-density rule is bounded, not unconditional: a variable
        whose every value repeats but whose support (500 distinct values) is
        far wider than the ceiling stays continuous. This is exactly the case
        an unconditional repeat check -- old is_discrete -- got wrong."""
        wide_support = np.repeat(np.arange(500), 100)  # every value repeats x100
        self.assertFalse(classify_values(wide_support)[0])

    def test_wide_support_integer_distribution_stays_discrete(self):
        """Impulse is the strict default for discrete data: a genuinely discrete
        integer distribution stays discrete well past the budget, up to the
        integer ceiling. Binomial(1000, 0.5) and Poisson(200) land around
        100-115 distinct integer values at n=10000 -- far past B_1D=30 -- and
        must still read as discrete, reliably across seeds."""
        for seed in range(15):
            rng = np.random.default_rng(seed)
            self.assertTrue(
                classify_values(rng.binomial(1000, 0.5, 10000))[0],
                msg=f"Binomial(1000, 0.5) seed {seed} should classify as discrete",
            )
            self.assertTrue(
                classify_values(rng.poisson(200, 10000))[0],
                msg=f"Poisson(200) seed {seed} should classify as discrete",
            )

    def test_integer_support_past_ceiling_bins_to_histogram(self):
        """ "Histogram only when necessary": an integer support genuinely too
        wide to draw one stem per value (past INT_REPEAT_CEILING) reads as
        continuous even with heavy repeats. Binomial(10000, 0.5) realizes
        ~300+ distinct values at n=10000."""
        for seed in range(10):
            rng = np.random.default_rng(seed)
            self.assertFalse(
                classify_values(rng.binomial(10000, 0.5, 10000))[0],
                msg=f"Binomial(10000, 0.5) seed {seed} should classify as continuous",
            )

    def test_repeat_rule_ceiling_is_dtype_split(self):
        """The rescue ceiling is split by dtype. A support of 100 distinct
        values with heavy repeats is rescued to discrete when the values are
        integers (genuine counts), but stays continuous when they are floats --
        the guard against is_discrete's rounded-float failure mode, kept tight
        for floats while integers are treated generously."""
        integer_support = np.repeat(np.arange(100), 200)  # 100 distinct ints
        float_support = np.repeat(np.arange(100) + 0.5, 200)  # 100 distinct floats
        self.assertTrue(classify_values(integer_support)[0])
        self.assertFalse(classify_values(float_support)[0])

    def test_large_n_rescue_can_be_disabled(self):
        """The large-n rescue is 1-D only: with large_n_rescue=False (how the
        2-D per-axis dispatch calls it) an over-budget integer support stays
        continuous, so both-axes-over-K_2D still bins to a 2-D histogram
        instead of a huge, sparse tile."""
        over_budget = np.repeat(np.arange(45), 300)  # 45 distinct ints, heavy repeats
        self.assertTrue(classify_values(over_budget)[0])  # rescued in 1-D
        self.assertFalse(
            classify_values(over_budget, large_n_rescue=False)[0]
        )  # not in 2-D


# ===========================================================================
# default_plot_type: the lookup table
# ===========================================================================


class TestDefaultPlotType(unittest.TestCase):
    """Lookup and structural invariants of DEFAULT_PLOT_TYPE."""

    def test_lookup_returns_string_default_and_list_alternatives(self):
        default, alts = default_plot_type("1D_discrete", False)
        self.assertIsInstance(default, str)
        self.assertIsInstance(alts, list)

    def test_default_never_repeated_in_its_alternatives(self):
        for configuration, small_n in DEFAULT_PLOT_TYPE:
            default, alts = default_plot_type(configuration, small_n)
            self.assertNotIn(default, alts)

    def test_every_token_has_a_display_name(self):
        for entry in DEFAULT_PLOT_TYPE.values():
            for token in [entry["default"]] + entry["alternatives"]:
                self.assertIn(token, PLOT_DISPLAY_NAME)

    def test_unknown_configuration_raises_keyerror(self):
        with self.assertRaises(KeyError):
            default_plot_type("nonsense", True)

    def test_ecdf_is_alternative_for_all_1d_numeric_cells(self):
        """ECDF is offered as an alternative for 1D discrete and continuous
        data (both sample-size splits), but not for categorical data,
        which cannot be ordered."""
        for configuration in ("1D_discrete", "1D_continuous"):
            for small_n in (True, False):
                _, alts = default_plot_type(configuration, small_n)
                self.assertIn("ecdf", alts)
        for small_n in (True, False):
            _, alts = default_plot_type("1D_categorical", small_n)
            self.assertNotIn("ecdf", alts)

    def test_violin_is_alternative_for_both_2d_mixed_cells(self):
        for small_n in (True, False):
            _, alts = default_plot_type("2D_mixed", small_n)
            self.assertIn("violin", alts)


# ===========================================================================
# suggestion_message: the "Currently Showing / Alternative Plots" text
# ===========================================================================


class TestSuggestionMessage(unittest.TestCase):
    """Formatting of the suggestion message."""

    def test_default_marked_when_showing_the_default(self):
        msg = suggestion_message("impulse", "impulse", ["hist", "density"])
        self.assertIn("Currently Showing: Impulse Plot (Default)", msg)

    def test_no_default_marker_when_showing_a_non_default(self):
        msg = suggestion_message("hist", "impulse", ["hist", "density"])
        self.assertIn("Currently Showing: Histogram", msg)
        self.assertNotIn("Histogram (Default)", msg)

    def test_default_appears_in_alternatives_when_type_specified(self):
        msg = suggestion_message("hist", "impulse", ["hist", "density"])
        self.assertIn("Impulse Plot (Default)", msg)

    def test_shown_type_removed_from_alternatives(self):
        msg = suggestion_message("hist", "impulse", ["hist", "density"])
        alt_line = msg.splitlines()[1]
        self.assertNotIn('(type = "hist")', alt_line)

    def test_non_default_alternatives_use_type_syntax(self):
        msg = suggestion_message("impulse", "impulse", ["hist", "density"])
        self.assertIn('Histogram (type = "hist")', msg)
        self.assertIn('Density Plot (type = "density")', msg)

    def test_mixed_suggestion_uses_short_type_names(self):
        """On mixed data the lookup table and suggestion note use the short
        type names (rug/hist/density), never the segmented_ aliases."""
        default, alts = default_plot_type("2D_mixed", True)
        msg = suggestion_message(default, default, alts)
        self.assertNotIn("segmented_", msg)
        self.assertIn('Histogram (type = "hist")', msg)
        self.assertIn('Density Plot (type = "density")', msg)


# ===========================================================================
# should_show_suggestion: the tri-state suggest policy
# ===========================================================================


class TestSuggestionPolicy(unittest.TestCase):
    """The suggest=None/True/False triggering policy and session flag."""

    def setUp(self):
        # Reset the module-level once-per-session flag before each test.
        self.plotmod = importlib.import_module("symbulate.plot")
        self.plotmod._suggestion_shown = False

    def test_none_shows_only_once_per_session(self):
        self.assertTrue(should_show_suggestion(None))
        self.assertFalse(should_show_suggestion(None))
        self.assertFalse(should_show_suggestion(None))

    def test_default_argument_matches_none_behavior(self):
        self.assertTrue(should_show_suggestion())
        self.assertFalse(should_show_suggestion())

    def test_true_shows_every_call(self):
        self.assertTrue(should_show_suggestion(True))
        self.assertTrue(should_show_suggestion(True))

    def test_false_never_shows(self):
        self.assertFalse(should_show_suggestion(False))
        self.assertFalse(should_show_suggestion(False))

    def test_true_marks_session_so_later_none_stays_quiet(self):
        self.assertTrue(should_show_suggestion(True))
        self.assertFalse(should_show_suggestion(None))


if __name__ == "__main__":
    unittest.main()
