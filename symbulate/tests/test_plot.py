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
from matplotlib.collections import PolyCollection, PathCollection, LineCollection
from matplotlib.ticker import MaxNLocator

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
    ContinuousTimeMarkovChain,
    BrownianMotion,
    Tuple,
    ProbabilitySpace,
    cos,
    pi,
)
from symbulate import seed
from symbulate import plot as symbulate_plot
from symbulate.plot import (
    SymbulatePlot,
    B_1D,
    K_2D,
    N_SMALL_THRESHOLD,
    DOTPLOT_MAX_STACK,
    JOINT_PAIRS_MAX_DIM,
    JOINT_PAIRS_OVERLAY_ERROR,
    JOINT_PAIRS_PANEL_SIZE,
    MARGINAL_FREQ_TICKS,
    MARGINAL_FREQ_TICK_ROTATION,
    PAIRS_SUPTITLE,
    JOINT_PAIRS_COLORBAR_TITLE_SIZE,
    classify_values,
    default_plot_type,
    dotplot_tallest_stack,
    get_next_color,
    suggestion_message,
    should_show_suggestion,
    make_bar,
    BAR_ALPHA,
    make_dotplot,
    make_hist,
    make_impulse,
    count_var,
    weighted_count_var,
    make_violin,
    make_violinplot,
    make_ecdf,
    make_mosaic,
    make_stackedbar,
    resolve_mosaic_type,
    MOSAIC_SUGGEST_MAX_CATEGORIES,
    MOSAIC_YAXIS_TICKS,
    _mosaic_spans,
    _readable_text_color,
    make_sample_path,
    make_segmented_density,
    make_segmented_hist,
    make_tile,
    make_joint_pdf,
    make_joint_pmf,
    JOINT_OVERLAY_WARNING,
    JOINT_PDF_GRID_POINTS,
    JOINT_PMF_MAX_CELLS,
    make_segmented_rug,
    make_grouped_boxplot,
    _thin_discrete_ticks,
    _discrete_tick_labels,
    MAX_DISCRETE_TICKS,
    DEFAULT_PLOT_TYPE,
    PLOT_DISPLAY_NAME,
    SAMPLE_PATH_ALPHA,
    SAMPLE_PATH_LINEWIDTH,
    TILE_DEFAULT_BINS,
    HIST_DEFAULT_BINS,
    resolve_hist_bins,
)
from symbulate.results import RVResults, _is_categorical_2d

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
        self.assertEqual(plt.gca().get_title(), "Scatterplot")
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
        self.sims.plot(marginal=True, type="density")
        self.assertGreaterEqual(len(plt.gcf().axes), 3)

    def test_marginal_true_with_default_type_draws_main_panel(self):
        """The three-panel layout must still resolve type= to the data's
        default (a 2-D histogram here) and draw it on the main panel,
        rather than leaving the center panel blank."""
        p = self.sims.plot(marginal=True)
        self.assertGreater(len(p.ax.collections), 0)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)

    def test_marginal_main_panel_keeps_xy_labels_no_value_labels(self):
        """The main panel keeps its "X"/"Y" labels, and neither marginal
        panel shows a redundant "Value" axis label next to it."""
        for main_type in ["hist", "tile", "density"]:
            p = self.sims.plot(marginal=True, type=main_type)
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

        p = self.sims.plot(marginal=True, type="hist")
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


class TestMarginalLayoutIsOptIn(PlotTestCase):
    """``marginal=True`` adds a strip beside each axis, and only then.

    A plot of two simulated variables is one panel by default -- overlaying
    two of them is a normal thing to want, and the three-panel layout cannot
    be shared. The *theoretical* side always shows the strips and has no such
    keyword; see TestTheoreticalTwoVariableLayout.
    """

    def _panels(self):
        return [a for a in plt.gcf().axes if a.get_subplotspec() is not None]

    def test_two_variables_are_one_panel_by_default(self):
        X, Y = RV(Normal(0, 1) ** 2)
        Xd, Yd = RV(Binomial(5, 0.4) ** 2)
        Xm, Ym = RV(Binomial(5, 0.4) * Normal(0, 1))
        cases = [
            ((X & Y).sim(500), None, "Joint Histogram"),
            ((X & Y).sim(500), "scatter", "Scatterplot"),
            ((X & Y).sim(40), None, "Scatterplot"),
            ((Xd & Yd).sim(500), None, "Tile Plot"),
            ((Xm & Ym).sim(500), "violin", "Violin Plot"),
        ]
        for sims, kind, title in cases:
            with self.subTest(type=kind):
                plt.close("all")
                plt.figure()
                if kind is None:
                    sims.plot(suggest=False)
                else:
                    sims.plot(type=kind, suggest=False)
                self.assertEqual(len(self._panels()), 1)
                # The title stays on the panel, and there is no figure title.
                self.assertEqual(plt.gca().get_title(), title)
                self.assertEqual(plt.gcf().get_suptitle(), "")

    def test_marginal_true_adds_the_two_strips(self):
        X, Y = RV(Normal(0, 1) ** 2)
        (X & Y).sim(500).plot(marginal=True, suggest=False)
        self.assertEqual(len(self._panels()), 3)

    def test_two_2d_plots_can_still_overlay(self):
        """The reason the layout is opt-in: without it, two plots of two
        variables share one set of axes."""
        X, Y = RV(Normal(0, 1) ** 2)
        (X & Y).sim(200).plot(type="scatter", suggest=False)
        A, B = RV(Normal(2, 1) ** 2)
        (A & B).sim(200).plot(type="scatter", suggest=False)
        self.assertEqual(len(self._panels()), 1)

    def test_the_strips_show_each_variable_on_its_own(self):
        """The top strip covers x's range, the right strip covers y's --
        each locked to the joint panel's own limits."""
        X, Y = RV((Normal(0, 1) * Normal(10, 1)))
        p = (X & Y).sim(500).plot(marginal=True, suggest=False)
        strips = [a for a in self._panels() if a is not p.ax]
        marg_x = max(strips, key=lambda a: a.get_position().y0)
        marg_y = max(strips, key=lambda a: a.get_position().x0)
        self.assertEqual(marg_x.get_xlim(), p.ax.get_xlim())
        self.assertEqual(marg_y.get_ylim(), p.ax.get_ylim())

    def test_a_marginal_rug_matches_the_joint_panels_rug_ticks(self):
        """make_rug sizes ticks as a fraction of its own axes, so a strip's
        would be shorter than the joint panel's just for being smaller."""
        Xm, Ym = RV(Binomial(5, 0.4) * Normal(0, 1))
        p = (Xm & Ym).sim(40).plot(marginal=True, suggest=False)
        fig = plt.gcf()
        fig.canvas.draw()

        def longest_rug_tick(ax):
            best = 0.0
            for c in ax.collections:
                if not isinstance(c, LineCollection):
                    continue
                transform = c.get_transform()
                for seg in c.get_segments():
                    pts = transform.transform(seg)
                    best = max(
                        best, abs(pts[1][1] - pts[0][1]), abs(pts[1][0] - pts[0][0])
                    )
            return best

        # The main panel is a segmented rug; the continuous (y) strip is a
        # rug. Both should draw the same length tick, in real pixels.
        strips = [a for a in self._panels() if a is not p.ax]
        marg_y = max(strips, key=lambda a: a.get_position().x0)
        main = longest_rug_tick(p.ax)
        strip = longest_rug_tick(marg_y)
        self.assertGreater(main, 0)
        self.assertGreater(strip, 0)
        self.assertAlmostEqual(strip, main, delta=1.0)

    def test_a_strips_frequency_axis_is_not_crowded(self):
        """A strip is a fraction of the joint panel's size, so it can't show
        as many ticks -- 0, 2, 4, 6, 8 ran together."""
        cases = [
            (RV(Binomial(5, 0.4) ** 2), 500, {"normalize": False}),  # impulse, counts
            (RV(Binomial(5, 0.4) ** 2), 40, {}),  # dotplots
            (RV(Normal(0, 1) ** 2), 500, {}),  # histograms
            (RV(Binomial(5, 0.4) * Normal(0, 1)), 40, {}),  # dotplot + rug
        ]
        for rvs, n, kwargs in cases:
            with self.subTest(n=n, **kwargs):
                plt.close("all")
                plt.figure()
                A, B = rvs
                p = (A & B).sim(n).plot(marginal=True, suggest=False, **kwargs)
                fig = plt.gcf()
                # Drawn, because a dot plot rebuilds its own locators on every
                # draw and used to undo the cap here.
                fig.canvas.draw()
                for strip in [a for a in self._panels() if a is not p.ax]:
                    vertical = strip.get_position().height < 0.3
                    axis = strip.yaxis if vertical else strip.xaxis
                    labels = [t for t in axis.get_ticklabels() if t.get_text()]
                    self.assertLessEqual(len(labels), MARGINAL_FREQ_TICKS + 2)

    def _top_and_right_strips(self, joint):
        """The strip above the joint panel, and the one to its right.

        Picked by position: the top strip is the highest, the right strip the
        furthest right. Selecting the top one by *lowest* y0 would return the
        right strip instead -- it sits level with the joint panel -- which
        would make an assertion about the top strip quietly vacuous.
        """
        strips = [a for a in self._panels() if a is not joint]
        return (
            max(strips, key=lambda a: a.get_position().y0),
            max(strips, key=lambda a: a.get_position().x0),
        )

    def test_right_strips_frequency_labels_are_rotated_to_avoid_overlap(self):
        """Regression: the right-hand strip is narrow, so its frequency
        axis's decimal tick labels ("0.00", "0.15", "0.30") laid out
        horizontally ran into each other -- rotating them fits the same
        tick count in the strip's width. The strip above the joint panel
        is wide, not narrow, so its labels stay horizontal."""
        A, B = RV(Normal(0, 1) ** 2)
        p = (A & B).sim(500).plot(marginal=True, suggest=False)
        plt.gcf().canvas.draw()
        top, right = self._top_and_right_strips(p.ax)
        rotated = [t.get_rotation() for t in right.get_xticklabels() if t.get_text()]
        self.assertTrue(rotated, "right strip had no frequency tick labels")
        for angle in rotated:
            self.assertEqual(angle, MARGINAL_FREQ_TICK_ROTATION)
        upright = [t.get_rotation() for t in top.get_yticklabels() if t.get_text()]
        self.assertTrue(upright, "top strip had no frequency tick labels")
        for angle in upright:
            self.assertEqual(angle, 0)

    def test_no_bounding_box_overlap_between_strip_frequency_labels(self):
        """The within-strip crowding this rotation fixes, checked directly
        via rendered bounding boxes rather than just the rotation angle."""
        A, B = RV(Normal(0, 1) ** 2)
        p = (A & B).sim(500).plot(marginal=True, suggest=False)
        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        _, right = self._top_and_right_strips(p.ax)
        labels = [t for t in right.get_xticklabels() if t.get_text()]
        self.assertGreater(len(labels), 1)
        boxes = [t.get_window_extent(renderer=renderer) for t in labels]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                self.assertFalse(boxes[i].overlaps(boxes[j]))

    def test_a_count_axis_stays_whole_numbers(self):
        """Half a simulated value doesn't exist, so thinning a count axis
        must not introduce fractional ticks."""
        Xd, Yd = RV(Binomial(5, 0.4) ** 2)
        p = (Xd & Yd).sim(500).plot(marginal=True, normalize=False, suggest=False)
        plt.gcf().canvas.draw()
        for strip in [a for a in self._panels() if a is not p.ax]:
            vertical = strip.get_position().height < 0.3
            axis = strip.yaxis if vertical else strip.xaxis
            self.assertEqual(strip.get_ylabel() if vertical else "Count", "Count")
            for tick in axis.get_ticklocs():
                self.assertEqual(tick, round(tick))

    def test_one_variable_is_untouched(self):
        (Z,) = RV(Normal(0, 1) ** 1)
        Z.sim(500).plot(suggest=False)
        self.assertEqual(len(self._panels()), 1)
        self.assertEqual(plt.gcf().get_suptitle(), "")
        self.assertNotEqual(plt.gca().get_title(), "")

    def test_the_two_sides_are_asymmetric_on_purpose(self):
        """Simulated data opts in; a distribution always shows the strips and
        refuses the keyword. Pinned here so neither half drifts onto the
        other's rule.
        """
        X, Y = RV(Normal(0, 1) ** 2)
        dist = MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])

        plt.close("all")
        plt.figure()
        (X & Y).sim(200).plot(suggest=False)
        self.assertEqual(len(self._panels()), 1, "simulated default gained strips")

        plt.close("all")
        plt.figure()
        dist.plot()
        self.assertEqual(len(self._panels()), 3, "theoretical lost its strips")

        plt.close("all")
        plt.figure()
        with self.assertRaises(ValueError) as cm:
            dist.plot(marginal=True)
        self.assertIn("not needed here", str(cm.exception))


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

    def test_grouped_violin_accepts_outliers_argument(self):
        """make_violin takes outliers= like make_grouped_boxplot does."""
        import inspect

        from symbulate.plot import make_violin

        params = inspect.signature(make_violin).parameters
        self.assertIn("outliers", params)
        self.assertIs(params["outliers"].default, False)

    def test_grouped_violin_outliers_kwarg_flows_through_plot(self):
        def n_flier_points():
            return sum(
                len(line.get_xdata())
                for line in plt.gca().lines
                if line.get_linestyle() == "None"
            )

        X, Y = RV(Binomial(5, 0.4) * Exponential(1))
        sims = (X & Y).sim(600)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            sims.plot(type="violin", suggest=False)
            self.assertEqual(n_flier_points(), 0, "default should be outliers=False")
            plt.close("all")
            sims.plot(type="violin", suggest=False, outliers=True)
            self.assertGreater(n_flier_points(), 0)

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
        self.assertEqual(plt.gca().get_title(), "Violin Plot")
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
        not the boxplot overlay's own 1..n position defaults.

        The categories are stored alongside a continuous variable, so they
        arrive as floats -- they are still labeled as the whole numbers
        they are (see _discrete_tick_labels), not as "0.0", "1.0", ...
        """
        # Every category 0..5 is planted, rather than simulated from
        # Binomial(5, 0.4) and hoped for: that distribution lands on 5 only
        # about 1% of the time, so in 500 draws the top category is missing
        # roughly 1 run in 170 -- and simulating draws from the package RNG,
        # which np.random.seed does not control, so which run you get depends
        # on test order. Same reasoning as the planted flier in the box tests.
        rng = np.random.default_rng(0)
        planted = RVResults(
            [
                (category, value)
                for category in range(6)
                for value in rng.normal(0, 1, 25)
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PendingDeprecationWarning)
            planted.plot(type="violin")
        ax = plt.gca()
        labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        self.assertEqual(labels, ["0", "1", "2", "3", "4", "5"])

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
        self.assertEqual(ax.get_title(), "Histogram")

    def test_hist_count_title_when_not_normalized(self):
        self.sims.plot(normalize=False)
        self.assertEqual(plt.gca().get_title(), "Histogram")

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


class TestPlot1DHistBinWidthAndEqualArea(PlotTestCase):
    """bin_width= and equal_area= on a single-variable type="hist"."""

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(2000)
        self.values = np.asarray(self.sims.results)

    def test_bin_width_produces_expected_bin_count(self):
        data_range = self.values.max() - self.values.min()
        expected = max(1, int(np.ceil(data_range / 0.25)))
        self.sims.plot(type="hist", bin_width=0.25, suggest=False)
        self.assertEqual(len(plt.gca().patches), expected)

    def test_bin_width_bins_are_no_wider_than_requested(self):
        """The ceiling-division scheme (matching Results.tabulate's
        binwidth=) can only shrink bins to fit the exact data range, never
        widen them past what was asked for."""
        self.sims.plot(type="hist", bin_width=0.5, suggest=False)
        widths = [p.get_width() for p in plt.gca().patches]
        self.assertTrue(all(0 < w <= 0.5 + 1e-9 for w in widths))

    def test_bin_width_zero_raises(self):
        with self.assertRaises(ValueError):
            self.sims.plot(type="hist", bin_width=0, suggest=False)

    def test_bin_width_negative_raises(self):
        with self.assertRaises(ValueError):
            self.sims.plot(type="hist", bin_width=-1.0, suggest=False)

    def test_bins_and_bin_width_together_warns_and_bin_width_wins(self):
        data_range = self.values.max() - self.values.min()
        expected = max(1, int(np.ceil(data_range / 0.5)))
        with self.assertWarns(UserWarning):
            self.sims.plot(type="hist", bins=10, bin_width=0.5, suggest=False)
        self.assertEqual(len(plt.gca().patches), expected)

    def test_bin_width_ignored_outside_hist_warns(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(type="density", bin_width=0.5, suggest=False)

    def test_equal_area_produces_requested_bin_count(self):
        self.sims.plot(type="hist", bins=20, equal_area=True, suggest=False)
        self.assertEqual(len(plt.gca().patches), 20)

    def test_equal_area_bins_have_approximately_equal_area(self):
        self.sims.plot(type="hist", bins=20, equal_area=True, suggest=False)
        areas = [p.get_width() * p.get_height() for p in plt.gca().patches]
        target = 1.0 / 20
        for area in areas:
            self.assertAlmostEqual(area, target, delta=target * 0.35)

    def test_equal_area_widths_are_not_all_equal(self):
        """Sanity check that this differs from a plain equal-width
        histogram -- equal-area bins should be narrow where data is
        dense and wide in the tails, not a constant width."""
        self.sims.plot(type="hist", bins=20, equal_area=True, suggest=False)
        widths = {round(p.get_width(), 6) for p in plt.gca().patches}
        self.assertGreater(len(widths), 1)

    def test_equal_area_and_bin_width_together_warns_and_equal_area_wins(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(
                type="hist", bins=20, bin_width=0.5, equal_area=True, suggest=False
            )
        self.assertEqual(len(plt.gca().patches), 20)

    def test_equal_area_ignored_outside_hist_warns(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(type="density", equal_area=True, suggest=False)

    def test_explicit_bin_edges_array_wins_over_bin_width(self):
        edges = [-3, -1, 0, 1, 3]
        with self.assertWarns(UserWarning):
            self.sims.plot(type="hist", bins=edges, bin_width=0.5, suggest=False)
        self.assertEqual(len(plt.gca().patches), len(edges) - 1)

    def test_explicit_bin_edges_array_wins_over_equal_area(self):
        edges = [-3, -1, 0, 1, 3]
        with self.assertWarns(UserWarning):
            self.sims.plot(type="hist", bins=edges, equal_area=True, suggest=False)
        self.assertEqual(len(plt.gca().patches), len(edges) - 1)

    def test_equal_area_repeated_values_collapse_bins_with_warning(self):
        """Heavily repeated values can force requested quantile edges to
        coincide; resolve_hist_bins should drop the duplicate rather than
        hand ax.hist a zero-width bin, and say so."""
        lumpy = np.array([0.0] * 50 + list(np.linspace(1, 2, 50)))
        with self.assertWarns(UserWarning):
            n_actual = len(resolve_hist_bins(lumpy, bins=40, equal_area=True)) - 1
        self.assertLess(n_actual, 40)


class TestPlot2DHistBinWidthAndEqualAreaUnsupported(PlotTestCase):
    """bin_width=/equal_area= are not yet wired up for two-variable plots;
    they must warn and be ignored rather than fail silently or error."""

    def setUp(self):
        np.random.seed(42)
        X, Y = RV(Normal(0, 1) ** 2)
        self.sims = (X & Y).sim(500)

    def test_bin_width_warns_on_hist2d(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(type="hist", bin_width=0.5, suggest=False)

    def test_equal_area_warns_on_hist2d(self):
        with self.assertWarns(UserWarning):
            self.sims.plot(type="hist", equal_area=True, suggest=False)


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

    # ---- outliers=, mirroring make_boxplot ----

    def test_violin_accepts_outliers_argument(self):
        """A violin takes outliers= the same way a box plot does."""
        import inspect

        from symbulate.plot import make_violinplot

        params = inspect.signature(make_violinplot).parameters
        self.assertIn("outliers", params)
        self.assertIs(params["outliers"].default, False)

    def test_violin_outliers_default_is_false_no_flier_points(self):
        from symbulate.plot import make_violinplot, get_next_color

        values = np.append(np.random.normal(0, 1, 200), 25.0)
        ax = plt.gca()
        make_violinplot(values, ax, get_next_color(ax))
        fliers = [
            y
            for line in ax.lines
            if line.get_linestyle() == "None"
            for y in line.get_ydata()
        ]
        self.assertEqual(len(fliers), 0)

    def test_violin_outliers_true_draws_flier_points(self):
        from symbulate.plot import make_violinplot, get_next_color

        values = np.append(np.random.normal(0, 1, 200), 25.0)
        ax = plt.gca()
        make_violinplot(values, ax, get_next_color(ax), outliers=True)
        fliers = [
            y
            for line in ax.lines
            if line.get_linestyle() == "None"
            for y in line.get_ydata()
        ]
        self.assertIn(25.0, fliers)

    def test_violin_outliers_kwarg_flows_through_plot(self):
        """outliers= passed to .plot() reaches the violin helper."""

        def n_flier_points():
            return sum(
                len(line.get_xdata())
                for line in plt.gca().lines
                if line.get_linestyle() == "None"
            )

        planted = RVResults(np.append(np.random.normal(0, 1, 200), 25.0))
        planted.plot(type="violin", suggest=False)
        self.assertEqual(n_flier_points(), 0, "default should be outliers=False")
        plt.close("all")
        planted.plot(type="violin", suggest=False, outliers=True)
        self.assertGreater(n_flier_points(), 0)

    def test_violin_body_shows_every_value_regardless_of_outliers(self):
        """outliers= only changes the inner box; the density body is
        always built from all the data."""
        from symbulate.plot import make_violinplot, get_next_color

        values = np.append(np.random.normal(0, 1, 200), 25.0)
        extents = []
        for flag in (False, True):
            plt.close("all")
            ax = plt.gca()
            make_violinplot(values, ax, get_next_color(ax), outliers=flag)
            body = ax.collections[0].get_paths()[0].vertices
            extents.append(round(body[:, 1].max(), 6))
        self.assertEqual(extents[0], extents[1])

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
        self.assertEqual(ax.get_title(), "Density (Estimated)")
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


class TestWeightedCountVar(unittest.TestCase):
    """weighted_count_var sums weights per value instead of counting."""

    def test_sums_weights_per_value(self):
        self.assertEqual(
            weighted_count_var([0, 1, 0], [1.0, 5.0, 2.0]), {0: 3.0, 1: 5.0}
        )

    def test_unit_weights_match_count_var(self):
        values = [3, 1, 3, 3, 1]
        self.assertEqual(
            weighted_count_var(values, [1] * len(values)), count_var(values)
        )

    def test_empty_input_gives_empty_dict(self):
        self.assertEqual(weighted_count_var([], []), {})


class TestMakeImpulseWeights(PlotTestCase):
    """make_impulse(weights=...): sum weights per value instead of counting.

    This is what turns "how often was each state jumped into" into "what
    fraction of time was spent in each state" for a states impulse plot.
    """

    def test_weights_sum_instead_of_count(self):
        ax = plt.gca()
        xs, freqs = make_impulse(
            [0, 1, 0, 1, 0],
            ax,
            get_next_color(ax),
            weights=[1.0, 2.0, 3.0, 4.0, 5.0],
            normalize=False,
        )
        self.assertEqual(dict(zip(xs, freqs)), {0: 9.0, 1: 6.0})

    def test_normalize_divides_by_total_weight_not_count(self):
        ax = plt.gca()
        xs, freqs = make_impulse(
            [0, 1],
            ax,
            get_next_color(ax),
            weights=[3.0, 1.0],
            normalize=True,
        )
        self.assertEqual(dict(zip(xs, freqs)), {0: 0.75, 1: 0.25})

    def test_weights_work_with_categorical_values(self):
        ax = plt.gca()
        xs, freqs = make_impulse(
            np.array(["a", "b", "a"]),
            ax,
            get_next_color(ax),
            weights=[1.0, 5.0, 2.0],
            normalize=False,
        )
        labels = [t.get_text() for t in ax.get_xticklabels()]
        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(list(freqs), [3.0, 5.0])

    def test_weighting_can_reverse_which_value_is_tallest(self):
        """The whole point: the rarely-visited value can dominate in time."""
        values = [0, 0, 0, 1]
        ax = plt.gca()
        _, unweighted = make_impulse(values, ax, get_next_color(ax), normalize=False)
        plt.figure()
        ax2 = plt.gca()
        _, weighted = make_impulse(
            values,
            ax2,
            get_next_color(ax2),
            weights=[1.0, 1.0, 1.0, 100.0],
            normalize=False,
        )
        self.assertEqual(list(unweighted), [3, 1])
        self.assertEqual(list(weighted), [3.0, 100.0])

    def test_mismatched_weights_length_raises(self):
        ax = plt.gca()
        with self.assertRaises(ValueError) as cm:
            make_impulse([0, 1, 2], ax, get_next_color(ax), weights=[1.0, 2.0])
        message = str(cm.exception)
        self.assertIn("3", message)
        self.assertIn("2", message)

    def test_no_weights_matches_plain_counting(self):
        ax1 = plt.gca()
        xs1, freqs1 = make_impulse([0, 1, 0], ax1, get_next_color(ax1), normalize=False)
        plt.figure()
        ax2 = plt.gca()
        xs2, freqs2 = make_impulse(
            [0, 1, 0], ax2, get_next_color(ax2), weights=[1, 1, 1], normalize=False
        )
        self.assertEqual(list(xs1), list(xs2))
        self.assertEqual(list(freqs1), list(freqs2))


class TestMakeHistWeights(PlotTestCase):
    """make_hist(weights=...) passes weights straight through to ax.hist."""

    def test_weighted_counts_sum_to_total_weight(self):
        ax = plt.gca()
        values = [1, 1, 1, 5, 5]
        weights = [1.0, 1.0, 1.0, 10.0, 10.0]
        counts, edges, patches = make_hist(
            values, ax, get_next_color(ax), normalize=False, weights=weights, bins=2
        )
        self.assertAlmostEqual(sum(counts), sum(weights))

    def test_weighted_bar_heights_are_the_summed_weights(self):
        ax = plt.gca()
        counts, _, _ = make_hist(
            [1, 1, 1, 5, 5],
            ax,
            get_next_color(ax),
            normalize=False,
            weights=[1.0, 1.0, 1.0, 10.0, 10.0],
            bins=2,
        )
        np.testing.assert_allclose(counts, [3.0, 20.0])

    def test_unweighted_matches_plain_histogram(self):
        ax1 = plt.gca()
        values = [1, 2, 2, 3]
        counts1, _, _ = make_hist(
            values, ax1, get_next_color(ax1), normalize=False, bins=3
        )
        plt.figure()
        ax2 = plt.gca()
        counts2, _, _ = make_hist(
            values,
            ax2,
            get_next_color(ax2),
            normalize=False,
            weights=[1, 1, 1, 1],
            bins=3,
        )
        np.testing.assert_allclose(counts1, counts2)


class TestPlotCategorical2D(PlotTestCase):
    """Two dependent categorical (string) variables, routed through .plot().

    A pair of strings is not a numeric vector, so these results have
    dim=None. They used to fall through to the path-plot catch-all and
    draw each pair as a meaningless two-point line titled "Sample Path".
    """

    def setUp(self):
        np.random.seed(42)

        def event_sim():
            a = BoxModel(["a", "not a"], probs=[0.8, 0.2]).draw()
            if a == "a":
                b = BoxModel(["b", "not b"], probs=[0.7, 0.3]).draw()
            else:
                b = BoxModel(["b", "not b"], probs=[0.6, 0.4]).draw()
            return a, b

        X, Y = RV(ProbabilitySpace(event_sim))
        self.sims = (X & Y).sim(500)

    def test_is_categorical_2d_detects_string_pairs(self):
        self.assertTrue(_is_categorical_2d(self.sims.results))

    def test_is_categorical_2d_rejects_numeric_pairs(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        self.assertFalse(_is_categorical_2d((X & Y).sim(50).results))

    def test_is_categorical_2d_rejects_one_dimensional_strings(self):
        colors = RV(BoxModel(["red", "green"])).sim(50)
        self.assertFalse(_is_categorical_2d(colors.results))

    def test_default_is_mosaic_not_sample_path(self):
        """The reported bug: this drew a "Sample Path" against index."""
        self.sims.plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Mosaic Plot")

    def test_default_lookup_is_mosaic(self):
        default, alternatives = default_plot_type("2D_categorical", False)
        self.assertEqual(default, "mosaic")
        self.assertEqual(alternatives, ["stackedbar", "tile"])

    def test_stackedbar_on_two_categories_is_drawn_as_asked(self):
        """event_sim is 2x2, where a mosaic is the default -- but an
        explicit stackedbar request is still honored."""
        self.sims.plot(type="stackedbar", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Stacked Bar Chart")

    def test_many_categories_still_default_to_a_mosaic_but_say_so(self):
        """The default is not swapped for a crowded pair -- the mosaic is
        drawn as the lookup table says, with a note pointing elsewhere."""
        import io
        import contextlib

        def many_sim():
            a = BoxModel([f"g{i}" for i in range(6)]).draw()
            b = BoxModel(["yes", "no"]).draw()
            return a, b

        X, Y = RV(ProbabilitySpace(many_sim))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            (X & Y).sim(600).plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Mosaic Plot")
        self.assertIn("Mosaic plots get messy", buf.getvalue())

    def test_tile_is_available(self):
        self.sims.plot(type="tile", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Tile Plot")

    def test_unsupported_type_names_the_three_that_work(self):
        with self.assertRaises(ValueError) as cm:
            self.sims.plot(type="scatter", suggest=False)
        message = str(cm.exception)
        for token in ("'mosaic'", "'stackedbar'", "'tile'"):
            self.assertIn(token, message)

    def test_marginal_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.sims.plot(marginal=True, suggest=False)
        self.assertIn("Drop marginal=True", str(cm.exception))

    def test_column_widths_track_the_x_marginal(self):
        """'a' has probability 0.8 and 'not a' 0.2, so the first column
        should be roughly four times wider."""
        self.sims.plot(suggest=False)
        widths = sorted({round(b.get_width(), 4) for b in plt.gca().patches})
        self.assertEqual(len(widths), 2)
        self.assertGreater(widths[1] / widths[0], 2.0)

    def test_suggestion_note_offers_the_alternatives(self):
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.sims.plot(suggest=True)
        note = buf.getvalue()
        self.assertIn("Currently Showing: Mosaic Plot", note)
        self.assertIn('type = "stackedbar"', note)

    def test_one_dimensional_categorical_still_works(self):
        """The 2D branch must not swallow the existing 1D categorical
        case, which routes to bar / dotplot / impulse."""
        colors = RV(BoxModel(["red", "green", "blue"])).sim(500)
        colors.plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Bar Chart")


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


class TestDotplotImpulseAxisConsistency(PlotTestCase):
    """A dot plot and an impulse plot of the same discrete data frame it
    the same way: one slot of air past the outermost value, and
    whole-number ticks. The impulse plot used to take matplotlib's
    defaults instead -- 5% of the data range (which collapses to nothing
    when the range is short) and a locator free to tick at 0.5, 2.5, ...
    """

    def value_axis(self, values, type):
        """(xlim, tick labels) for one plot type, on a fresh figure."""
        plt.figure()
        RVResults(list(values)).plot(type, suggest=False)
        ax = plt.gca()
        return ax.get_xlim(), [t.get_text() for t in ax.get_xticklabels()]

    def test_padding_matches_the_dot_plot(self):
        for values in (
            np.resize([0, 1], 400),  # Bernoulli-like: range of 1
            np.resize(np.arange(4), 400),  # a handful of values
            np.resize(np.arange(150, 250), 3000),  # wide range
        ):
            with self.subTest(values=f"{values.min()}..{values.max()}"):
                dot, _ = self.value_axis(values, "dotplot")
                impulse, _ = self.value_axis(values, "impulse")
                self.assertEqual(impulse, dot)

    def test_padding_is_one_slot_not_a_share_of_the_range(self):
        """Two values one apart get a full unit of air on each side, where
        5% of the range would have given 0.05."""
        (lo, hi), _ = self.value_axis(np.resize([0, 1], 400), "impulse")
        self.assertEqual((lo, hi), (-1.0, 2.0))

    def test_ticks_are_whole_numbers_and_match_the_dot_plot(self):
        for values in (
            np.resize([0, 1], 400),
            np.resize(np.arange(4), 400),
            np.resize(np.arange(1, 21), 2000),
        ):
            with self.subTest(values=f"{values.min()}..{values.max()}"):
                _, dot_ticks = self.value_axis(values, "dotplot")
                _, impulse_ticks = self.value_axis(values, "impulse")
                self.assertEqual(impulse_ticks, dot_ticks)
                for label in impulse_ticks:
                    self.assertNotIn(".", label)

    def test_continuous_values_keep_the_proportional_buffer(self):
        """An impulse plot is for discrete data, but it doesn't refuse
        continuous data -- whose smallest gap can be arbitrarily small, so
        that data keeps the 5%-of-range buffer instead."""
        values = np.linspace(0, 10, 200) + 0.5
        (lo, hi), _ = self.value_axis(values, "impulse")
        self.assertAlmostEqual(lo, 0.5 - 0.5)  # 5% of a range of 10
        self.assertAlmostEqual(hi, 10.5 + 0.5)

    def test_overlay_does_not_crop_the_first_series(self):
        """The limits only ever widen, so a narrower second impulse plot
        leaves the first one's range intact."""
        plt.figure()
        RVResults(list(np.resize(np.arange(21), 2000))).plot("impulse", suggest=False)
        wide = plt.gca().get_xlim()
        RVResults(list(np.resize(np.arange(5, 11), 2000))).plot(
            "impulse", suggest=False
        )
        self.assertEqual(plt.gca().get_xlim(), wide)

    def test_overlay_widens_for_a_broader_second_series(self):
        plt.figure()
        RVResults(list(np.resize(np.arange(5, 11), 2000))).plot(
            "impulse", suggest=False
        )
        narrow = plt.gca().get_xlim()
        RVResults(list(np.resize(np.arange(21), 2000))).plot("impulse", suggest=False)
        wide = plt.gca().get_xlim()
        self.assertLess(wide[0], narrow[0])
        self.assertGreater(wide[1], narrow[1])

    def test_categorical_impulse_still_labeled_by_category(self):
        """The integer locator must not displace the category names."""
        plt.figure()
        RVResults(list(np.resize(["H", "T"], 200))).plot("impulse", suggest=False)
        ax = plt.gca()
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["H", "T"])
        self.assertEqual(ax.get_xlim(), (-1.0, 2.0))  # same slot rule


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
        # 1-D data: one panel, so the title stays on it.
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

    def test_box_outliers_default_is_false_no_flier_points(self):
        """The default extends the whiskers to min/max, so nothing is
        left over to draw as an individual outlier point."""
        from symbulate.plot import make_boxplot, get_next_color

        values = np.append(np.random.normal(0, 1, 50), 25.0)
        ax = plt.gca()
        box = make_boxplot(values, ax, get_next_color(ax))
        self.assertEqual(len(box["fliers"][0].get_ydata()), 0)
        whisker_ends = [w.get_ydata()[1] for w in box["whiskers"]]
        self.assertAlmostEqual(max(whisker_ends), 25.0)

    def test_box_outliers_true_draws_outlier_points(self):
        """outliers=True opts back into the classical 1.5*IQR convention."""
        from symbulate.plot import make_boxplot, get_next_color

        values = np.append(np.random.normal(0, 1, 50), 25.0)
        ax = plt.gca()
        box = make_boxplot(values, ax, get_next_color(ax), outliers=True)
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

        # A planted extreme value is always a flier under the 1.5-IQR
        # rule, so the assertion does not depend on a random draw
        # happening to contain an outlier (Normal(0, 1) sometimes has
        # none). self.sims uses the package RNG, which np.random.seed does
        # not control, so plotting it directly made this test flaky.
        planted = RVResults(np.append(np.random.normal(0, 1, 200), 25.0))
        planted.plot(type="box")
        self.assertEqual(n_flier_points(), 0, "default should be outliers=False")
        plt.close("all")
        planted.plot(type="box", outliers=True)
        self.assertGreater(n_flier_points(), 0)

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
        self.assertEqual(plt.gca().get_title(), "Scatterplot")

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
        self.assertEqual(plt.gca().get_title(), "Joint Histogram")

    def test_density2d_draws_contour_surface(self):
        """Banded by default now, so it still draws collections/images."""
        self.sims.plot(type="density")
        ax = plt.gca()
        self.assertGreater(len(ax.collections + ax.images), 0)
        self.assertEqual(plt.gca().get_title(), "Joint Density (Estimated)")

    def test_density2d_smooth_surface_with_contour_off(self):
        """contour=False is how the smooth gradient is asked for now."""
        self.sims.plot(type="density", contour=False)
        self.assertEqual(plt.gca().get_title(), "Joint Density (Estimated)")

    def test_density2d_contour_mode(self):
        self.sims.plot(type="density", contour=True)
        self.assertEqual(plt.gca().get_title(), "Joint Density (Estimated)")

    def test_density2d_levels_without_contour_warns(self):
        """levels only bands a contour plot, so it warns with contour=False."""
        with self.assertWarns(UserWarning):
            self.sims.plot(type="density", levels=5, contour=False)

    def test_density2d_levels_apply_by_default(self):
        """Bands are the default, so levels= now takes effect without a flag."""
        self.sims.plot(type="density", levels=5)
        sets = [
            c for c in plt.gca().collections if getattr(c, "levels", None) is not None
        ]
        self.assertTrue(any(len(s.levels) - 1 == 5 for s in sets))

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
        self.assertEqual(plt.gca().get_title(), "Rug Plot")

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
        self.sims.plot(marginal=True, type="hist")
        self.assertGreaterEqual(len(plt.gcf().axes), 3)


class TestPlot2DMosaic(PlotTestCase):
    """The mosaic plot (discrete x discrete, alternative to tile)."""

    def setUp(self):
        np.random.seed(42)
        # The category count decides which of the two is drawn, so the
        # fixtures have to sit on the right side of the cutoff.
        # Binomial(3, .) has 4 distinct values, right at
        # MOSAIC_SUGGEST_MAX_CATEGORIES, so this stays a mosaic (the test
        # is > , not >=).
        Xd, Yd = RV(Binomial(3, 0.4) ** 2)
        self.discrete_sims = (Xd & Yd).sim(500)
        # Binomial(6, .) has 7 distinct values -- over the cutoff, so this
        # is what actually draws a stacked bar.
        Xc, Yc = RV(Binomial(6, 0.3) ** 2)
        self.crowded_sims = (Xc & Yc).sim(500)

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
        proportion scale on the left is meaningful across every column."""
        self.discrete_sims.plot(type="mosaic")
        ticks = list(plt.gca().get_yticks())
        self.assertEqual(ticks, MOSAIC_YAXIS_TICKS)

    def test_mosaic_columns_sum_to_full_width(self):
        self.discrete_sims.plot(type="mosaic")
        bars = plt.gca().patches
        n_x = len(set(round(b.get_x(), 6) for b in bars))
        widths = sorted(set(round(b.get_x() + b.get_width(), 6) for b in bars))
        self.assertGreater(n_x, 1)
        self.assertAlmostEqual(widths[-1], 1.0, places=6)

    def test_mosaic_length_mismatch_raises_friendly_error(self):
        with self.assertRaises(ValueError) as cm:
            make_mosaic(["a", "b", "c"], ["x", "y"], plt.gca())
        self.assertIn("same length", str(cm.exception))

    def test_mosaic_many_categories_warns_and_hatches(self):
        x = np.array(["a", "b"] * 20)
        y = np.array([f"c{i}" for i in range(8)] * 5)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bars = make_mosaic(x, y, plt.gca())
        self.assertTrue(any("repeat a color" in str(w.message) for w in caught))
        hatches = {b.patches[0].get_hatch() for b in bars.values()}
        self.assertIn("//", hatches)

    def test_mosaic_is_listed_as_2d_discrete_alternative(self):
        _, alternatives = default_plot_type("2D_dd", False)
        self.assertIn("mosaic", alternatives)

    def test_mosaic_display_name(self):
        self.assertEqual(PLOT_DISPLAY_NAME["mosaic"], "Mosaic Plot")

    def test_mosaic_spans_total_parameter_rescales_span(self):
        starts, widths = _mosaic_spans(np.array([1.0, 1.0]), 0.0, total=0.5)
        self.assertAlmostEqual(starts[0], 0.0)
        self.assertAlmostEqual(starts[-1] + widths[-1], 0.5)

    # ---- no in-cell probability labels (requested change) ----

    def test_mosaic_has_no_in_cell_labels(self):
        """In-cell proportion labels were removed: a number in every cell
        crowded out the shapes the plot exists to show, and the segment
        heights already encode the same quantity against the left scale."""
        self.discrete_sims.plot(type="mosaic")
        # The legend lives in its own artist, so the axes itself should
        # carry no free-floating text at all.
        self.assertEqual(len(plt.gca().texts), 0)

    def test_stackedbar_has_no_in_cell_labels(self):
        self.discrete_sims.plot(type="stackedbar")
        self.assertEqual(len(plt.gca().texts), 0)

    def test_mosaic_rejects_removed_annotate_argument(self):
        """annotate= went away with the labels, so it must not be silently
        accepted. It currently surfaces as matplotlib's own unknown-kwarg
        error (an AttributeError from Rectangle.set); the friendlier
        wrapper for that is a separate, package-wide task."""
        with self.assertRaises((TypeError, AttributeError)):
            make_mosaic(["a", "b"], ["x", "y"], plt.gca(), annotate=True)

    # ---- bars span the full 0 to 1 (requested fix) ----

    def test_mosaic_spans_reserve_no_gap_for_zero_weight_segments(self):
        """A category absent from a column used to leave its gap behind as
        a sliver of empty space, so the column stopped short of 0 or 1."""
        for weights in ([3.0, 4.0, 0.0], [0.0, 4.0, 5.0], [0.0, 4.0, 0.0]):
            starts, widths = _mosaic_spans(np.array(weights), 0.002)
            visible = [(s, s + w) for s, w in zip(starts, widths) if w > 0]
            self.assertAlmostEqual(visible[0][0], 0.0, places=9, msg=str(weights))
            self.assertAlmostEqual(visible[-1][1], 1.0, places=9, msg=str(weights))

    def test_mosaic_spans_keep_gaps_between_visible_segments(self):
        starts, widths = _mosaic_spans(np.array([3.0, 4.0, 5.0]), 0.002)
        gaps = [round(starts[i + 1] - (starts[i] + widths[i]), 6) for i in range(2)]
        self.assertEqual(gaps, [0.002, 0.002])

    def test_mosaic_column_missing_a_category_still_fills_axis(self):
        """End-to-end version of the span fix: y='q' never occurs with
        x='b', so that column has a zero-height segment."""
        x = np.array(["a"] * 10 + ["b"] * 10)
        y = np.array(["p", "q"] * 5 + ["p"] * 10)
        make_mosaic(x, y, plt.gca())
        for col_x in sorted(set(round(b.get_x(), 6) for b in plt.gca().patches)):
            col = [
                b
                for b in plt.gca().patches
                if round(b.get_x(), 6) == col_x and b.get_height() > 0
            ]
            bottom = min(b.get_y() for b in col)
            top = max(b.get_y() + b.get_height() for b in col)
            self.assertAlmostEqual(bottom, 0.0, places=6)
            self.assertAlmostEqual(top, 1.0, places=6)

    # ---- overlay is a hard error (requested change) ----

    def test_mosaic_overlay_raises(self):
        self.discrete_sims.plot(type="mosaic")
        with self.assertRaises(ValueError) as cm:
            self.discrete_sims.plot(type="mosaic")
        self.assertIn("Overlaying a mosaic or stacked bar plot", str(cm.exception))

    def test_stackedbar_over_mosaic_raises(self):
        """The two share one axes counter -- mixing them is still an
        overlay of two full-canvas plots."""
        self.discrete_sims.plot(type="mosaic")
        with self.assertRaises(ValueError):
            self.discrete_sims.plot(type="stackedbar")

    def test_mosaic_overlay_refused_before_drawing_anything(self):
        """The refused second plot must leave the first one untouched."""
        self.discrete_sims.plot(type="mosaic")
        n_before = len(plt.gca().patches)
        with self.assertRaises(ValueError):
            self.discrete_sims.plot(type="mosaic")
        self.assertEqual(len(plt.gca().patches), n_before)

    # ---- mosaic vs stackedbar as two plot types (requested change) ----

    def test_stackedbar_title(self):
        self.crowded_sims.plot(type="stackedbar")
        self.assertEqual(plt.gca().get_title(), "Stacked Bar Chart")

    def test_stackedbar_display_name(self):
        self.assertEqual(PLOT_DISPLAY_NAME["stackedbar"], "Stacked Bar Chart")

    def test_stackedbar_produces_uniform_column_widths(self):
        self.crowded_sims.plot(type="stackedbar")
        widths = {round(b.get_width(), 6) for b in plt.gca().patches}
        self.assertEqual(len(widths), 1)

    def test_mosaic_widths_vary_with_unequal_marginal_counts(self):
        x = np.array(["a"] * 30 + ["b"] * 10)
        y = np.array(["p", "q"] * 20)
        make_mosaic(x, y, plt.gca())
        widths = sorted({round(b.get_width(), 6) for b in plt.gca().patches})
        self.assertEqual(len(widths), 2)
        self.assertAlmostEqual(widths[1] / widths[0], 3.0, places=1)

    def test_stackedbar_does_not_change_segment_heights(self):
        """Only the widths differ between the two types; the conditional
        distributions they show are identical."""
        x = np.array(["a"] * 30 + ["b"] * 10)
        y = np.array(["p", "q"] * 20)
        make_mosaic(x, y, plt.gca())
        mosaic_heights = sorted(round(b.get_height(), 6) for b in plt.gca().patches)
        plt.close("all")
        make_stackedbar(x, y, plt.gca())
        stacked_heights = sorted(round(b.get_height(), 6) for b in plt.gca().patches)
        self.assertEqual(mosaic_heights, stacked_heights)

    def test_equal_width_keyword_raises_pointing_at_stackedbar(self):
        with self.assertRaises(ValueError) as cm:
            self.discrete_sims.plot(type="mosaic", equal_width=True)
        self.assertIn("type='stackedbar'", str(cm.exception))

    def test_marginal_column_keyword_raises(self):
        with self.assertRaises(ValueError) as cm:
            self.discrete_sims.plot(type="mosaic", marginal_column=True)
        self.assertIn("marginal_column=", str(cm.exception))

    def test_mosaic_legend_names_the_y_categories(self):
        """With no marginal column to hang labels off, a standard legend
        outside the axes names each y category."""
        self.discrete_sims.plot(type="mosaic")
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(legend.get_title().get_text(), "Variable 2")

    def test_mosaic_legend_false_shows_no_legend(self):
        make_mosaic(["a", "b"], ["x", "y"], plt.gca(), legend=False)
        self.assertIsNone(plt.gca().get_legend())

    # ---- too many categories: a note, never a substitution ----

    SMALL_X = ["a", "b"] * 10
    SMALL_Y = ["p", "q"] * 10
    BIG_X = [f"x{i}" for i in range(6)] * 5
    BIG_Y = ["p", "q"] * 15

    def test_small_mosaic_says_nothing(self):
        self.assertIsNone(resolve_mosaic_type(self.SMALL_X, self.SMALL_Y, "mosaic"))

    def test_crowded_mosaic_gets_a_note(self):
        note = resolve_mosaic_type(self.BIG_X, self.BIG_Y, "mosaic")
        self.assertIsNotNone(note)
        self.assertIn("6x2 here", note)
        self.assertIn('type="stackedbar"', note)
        self.assertIn('type="tile"', note)

    def test_stackedbar_never_gets_a_note(self):
        """Only a mosaic can be too crowded to read; a stacked bar is
        fine at any number of categories, and is never second-guessed."""
        self.assertIsNone(resolve_mosaic_type(self.BIG_X, self.BIG_Y, "stackedbar"))
        self.assertIsNone(resolve_mosaic_type(self.SMALL_X, self.SMALL_Y, "stackedbar"))

    def test_crowding_is_judged_on_either_axis(self):
        """More than MOSAIC_SUGGEST_MAX_CATEGORIES on y alone is enough."""
        x = ["a", "b"] * 15
        y = [f"y{i}" for i in range(6)] * 5
        self.assertIsNotNone(resolve_mosaic_type(x, y, "mosaic"))

    def test_crowding_boundary_is_max_categories(self):
        n = MOSAIC_SUGGEST_MAX_CATEGORIES
        at = [f"x{i}" for i in range(n)] * 4
        over = [f"x{i}" for i in range(n + 1)] * 4
        y_at = ["p", "q"] * (len(at) // 2)
        y_over = ["p", "q"] * (len(over) // 2)
        self.assertIsNone(resolve_mosaic_type(at, y_at, "mosaic"))
        self.assertIsNotNone(resolve_mosaic_type(over, y_over, "mosaic"))

    def test_resolve_ignores_other_plot_types(self):
        self.assertIsNone(resolve_mosaic_type(["a"], ["b"], "tile"))

    def test_crowded_mosaic_through_plot_still_draws_a_mosaic(self):
        """End to end: type='mosaic' on crowded data renders a mosaic,
        because that is what was asked for."""
        X, Y = RV(Binomial(n=6, p=0.3) * Binomial(n=6, p=0.5))
        (X & Y).sim(800).plot(type="mosaic", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Mosaic Plot")

    def test_crowded_stackedbar_through_plot_stays_a_stacked_bar(self):
        self.crowded_sims.plot(type="stackedbar", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Stacked Bar Chart")

    def test_crowded_mosaic_note_suggests_the_alternatives(self):
        """The mosaic is drawn as asked; the note names what else to try."""
        import io
        import contextlib

        X, Y = RV(Binomial(n=6, p=0.3) * Binomial(n=6, p=0.5))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            (X & Y).sim(800).plot(type="mosaic", suggest=True)
        out = buf.getvalue()
        self.assertIn("Mosaic plots get messy", out)
        self.assertIn("Currently Showing: Mosaic Plot", out)


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
        self.assertEqual(plt.gca().get_title(), "Box Plot")

    def test_box_continuous_x_discrete_y(self):
        X, Y = RV(Normal(0, 1) * Binomial(5, 0.4))
        sims = (X & Y).sim(500)
        sims.plot(type="box")
        self.assertGreater(len(plt.gca().patches), 0)

    def test_boxplot_alias_behaves_like_box(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(500).plot(type="boxplot")
        ax = plt.gca()
        self.assertEqual(plt.gca().get_title(), "Box Plot")
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
        self.assertEqual(n_flier_points(), 0, "default should be outliers=False")
        plt.close("all")
        sims.plot(type="box", outliers=True)
        self.assertGreater(n_flier_points(), 0)


class TestPlot2DCombinedTypes(PlotTestCase):
    """A 2D .plot() given several types draws all of them, the way two
    .plot() calls in one cell already did.

    The 2D dispatch used to be one long if/elif chain, so the first
    branch that matched won and every other type the caller asked for
    was dropped without a word -- .plot(["hist", "density"]) drew the
    histogram alone. It is now two tiers, like the 1D branch: one main
    type, plus the density and rug overlays on top.
    """

    def setUp(self):
        super().setUp()
        np.random.seed(42)
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        self.mixed = (X & Y).sim(1000)
        C1, C2 = RV(Normal(0, 1) * Normal(0, 1))
        self.cont = (C1 & C2).sim(1000)

    def counts(self, ax):
        return len(ax.patches), len(ax.lines), len(ax.collections), len(ax.images)

    def one_call(self, results, types, **kwargs):
        plt.figure()
        results.plot(list(types), suggest=False, **kwargs)
        return self.counts(plt.gca())

    def two_calls(self, results, types, **kwargs):
        plt.figure()
        for type in types:
            results.plot(type, suggest=False, **kwargs)
        return self.counts(plt.gca())

    def test_hist_and_density_matches_two_calls(self):
        """The reported case."""
        combo = ("hist", "density")
        self.assertEqual(
            self.one_call(self.mixed, combo), self.two_calls(self.mixed, combo)
        )

    def test_hist_and_density_draws_both_kinds_of_artist(self):
        patches, lines, _, _ = self.one_call(self.mixed, ("hist", "density"))
        self.assertGreater(patches, 0)  # the histogram's bars
        self.assertGreater(lines, 0)  # the density's curves

    def test_order_in_the_list_does_not_change_what_is_drawn(self):
        """['density','hist'] used to draw the histogram alone, because
        'hist' sat above 'density' in the internal chain -- the caller's
        own ordering was never consulted."""
        self.assertEqual(
            self.one_call(self.mixed, ("hist", "density")),
            self.one_call(self.mixed, ("density", "hist")),
        )

    def test_combinations_match_two_calls(self):
        for results, combo in [
            (self.mixed, ("hist", "rug")),
            (self.mixed, ("tile", "density")),
            (self.mixed, ("hist", "density", "rug")),
            (self.cont, ("hist", "density")),
            (self.cont, ("scatter", "density")),
            (self.cont, ("hist2d", "density2d")),
        ]:
            with self.subTest(combo=combo):
                self.assertEqual(
                    self.one_call(results, combo), self.two_calls(results, combo)
                )

    def test_an_overlay_type_alone_is_still_the_whole_plot(self):
        for type, title in [
            ("density", "Conditional Density (Estimated)"),
            ("rug", "Rug Plot"),
            ("segmented_density", "Conditional Density (Estimated)"),
        ]:
            with self.subTest(type=type):
                plt.figure()
                self.mixed.plot(type, suggest=False)
                self.assertEqual(plt.gca().get_title(), title)

    def test_marginal_layout_draws_both_on_the_main_panel(self):
        """marginal=True builds its own panels, so the combined types have
        to land on the main one rather than a marginal strip."""
        plt.figure()
        self.mixed.plot(["hist", "density"], marginal=True, suggest=False)
        main = plt.gcf().axes[0]
        self.assertGreater(len(main.patches), 0)
        self.assertGreater(len(main.lines), 0)

    def test_bandwidth_reaches_the_density_without_reaching_the_histogram(self):
        """bandwidth is the density's alone -- passed straight through to
        a histogram it would be a stray matplotlib kwarg."""
        plt.figure()
        self.mixed.plot(["hist", "density"], bandwidth=0.5, suggest=False)
        ax = plt.gca()
        self.assertGreater(len(ax.patches), 0)
        self.assertGreater(len(ax.lines), 0)

    def test_two_main_types_still_pick_one(self):
        """Two panel-filling types would hide each other, so one wins --
        the same way the 1D branch treats hist vs bar vs impulse."""
        patches, lines, _, _ = self.one_call(self.cont, ("scatter", "hist2d"))
        self.assertEqual((patches, lines), (0, 0))


class TestPlot2DSegmentedDensity(PlotTestCase):
    """The new type='segmented_density' for mixed discrete/continuous data."""

    def setUp(self):
        np.random.seed(42)

    def test_segmented_density_discrete_x_continuous_y(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        sims.plot(type="segmented_density")
        ax = plt.gca()
        self.assertEqual(plt.gca().get_title(), "Conditional Density (Estimated)")
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
            PLOT_DISPLAY_NAME["segmented_density"], "Conditional Density (Estimated)"
        )

    def test_density_short_name_is_segmented_on_mixed(self):
        """On mixed data, type='density' produces the segmented density."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(type="density", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Conditional Density (Estimated)")

    def test_density2d_still_forces_surface_on_mixed(self):
        """type='density2d' forces the 2D surface even on mixed data."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(
            type="density2d", suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "Joint Density (Estimated)")


class TestPlot2DSegmentedHist(PlotTestCase):
    """The new type='segmented_hist' for mixed discrete/continuous data."""

    def setUp(self):
        np.random.seed(42)

    def test_segmented_hist_discrete_x_continuous_y(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        sims = (X & Y).sim(500)
        sims.plot(type="segmented_hist")
        ax = plt.gca()
        self.assertEqual(plt.gca().get_title(), "Conditional Histogram")
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
        self.assertEqual(PLOT_DISPLAY_NAME["segmented_hist"], "Conditional Histogram")

    def test_hist_short_name_is_segmented_on_mixed(self):
        """On mixed data, type='hist' produces the segmented histogram."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(type="hist", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Conditional Histogram")

    def test_hist2d_still_forces_mesh_on_mixed(self):
        """type='hist2d' forces the 2D mesh even on mixed data."""
        RV(Normal(0, 1) * Binomial(3, 0.5)).sim(500).plot(type="hist2d", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Joint Histogram")

    def test_hist_short_name_stays_2d_on_continuous(self):
        """On two continuous variables, type='hist' is still the 2D mesh."""
        RV(Normal(0, 1) * Normal(0, 1)).sim(500).plot(type="hist", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Joint Histogram")


# ===========================================================================
# Marginal panel rebuild: real helpers, classify_values-routed types,
# coordinate alignment with the main panel
# ===========================================================================


class TestMarginalPanelRebuild(PlotTestCase):
    """Marginal strips use the redesigned 1D helpers, are routed
    through classify_values like a standalone 1D variable, and are aligned
    to the main panel's actual coordinate system instead of drifting to
    their own scheme."""

    def _marginal_axes(self, p):
        """Return (ax_marg_x, ax_marg_y) from the current figure, in the
        order RVResults.plot(marginal=True) creates them (x, then y, then any
        colorbar axes)."""
        others = [a for a in plt.gcf().axes if a is not p.ax]
        return others[0], others[1]

    def test_tile_marginal_coordinate_mismatch_is_fixed(self):
        """The confirmed regression case: DiscreteUniform(50,60) has
        support starting at 50, so tile's index-position discrete axis
        used to disagree completely with a real-valued marginal axis."""
        np.random.seed(0)
        X, Y = RV(DiscreteUniform(a=50, b=60) * Poisson(lam=5))
        p = (X & Y).sim(2000).plot(marginal=True, type="tile", suggest=False)
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
        p = (X & Y).sim(50).plot(marginal=True, type="scatter", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_scatter_discrete_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(50).plot(marginal=True, type="scatter", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_tile_discrete_discrete_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(2000).plot(marginal=True, type="tile", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_hist2d_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(marginal=True, type="hist2d", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_density2d_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(marginal=True, type="density2d", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_segmented_rug_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Poisson(lam=3) * Normal(0, 1))
        p = (X & Y).sim(2000).plot(marginal=True, type="rug", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_violin_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Poisson(lam=3) * Normal(0, 1))
        p = (X & Y).sim(2000).plot(marginal=True, type="violin", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_box_marginal_aligned(self):
        np.random.seed(1)
        X, Y = RV(Poisson(lam=3) * Normal(0, 1))
        p = (X & Y).sim(2000).plot(marginal=True, type="box", suggest=False)
        self._assert_aligned_and_populated(p)

    def test_mosaic_marginal_raises_and_says_to_drop_it(self):
        """A mosaic already shows both marginals itself, so the side
        strips would draw the same thing twice."""
        X, Y = RV(Binomial(5, 0.4) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(500).plot(type="mosaic", marginal=True)
        self.assertIn("Drop marginal=True", str(cm.exception))

    def test_stackedbar_marginal_raises_too(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        with self.assertRaises(ValueError) as cm:
            (X & Y).sim(500).plot(type="stackedbar", marginal=True)
        self.assertIn("stackedbar", str(cm.exception))

    def test_small_n_discrete_marginal_is_dotplot(self):
        """A small-n discrete axis's marginal should be a dot plot,
        matching what that axis would show standalone -- not always
        impulse/hist regardless of sample size."""
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(50).plot(marginal=True, type="scatter", suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertTrue(hasattr(ax_marg_x, "_dotplot_state"))
        self.assertTrue(hasattr(ax_marg_y, "_dotplot_state"))

    def test_large_n_discrete_marginal_is_impulse(self):
        np.random.seed(1)
        X, Y = RV(Binomial(5, 0.4) ** 2)
        p = (X & Y).sim(2000).plot(marginal=True, type="tile", suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertTrue(hasattr(ax_marg_x, "_impulse_series"))
        self.assertTrue(hasattr(ax_marg_y, "_impulse_series"))

    def test_small_n_continuous_marginal_is_rug(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(50).plot(marginal=True, type="scatter", suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertGreater(getattr(ax_marg_x, "_rug_count", 0), 0)
        self.assertGreater(getattr(ax_marg_y, "_rug_count", 0), 0)

    def test_large_n_continuous_marginal_is_hist(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(marginal=True, type="hist2d", suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertGreater(getattr(ax_marg_x, "_hist_count", 0), 0)
        self.assertGreater(getattr(ax_marg_y, "_hist_count", 0), 0)

    def test_density_mode_gives_density_curve_marginals(self):
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        p = (X & Y).sim(2000).plot(marginal=True, type="density2d", suggest=False)
        ax_marg_x, ax_marg_y = self._marginal_axes(p)
        self.assertGreater(getattr(ax_marg_x, "_density_count", 0), 0)
        self.assertGreater(getattr(ax_marg_y, "_density_count", 0), 0)

    def test_hist2d_marginal_bar_edges_match_hist2d_edges(self):
        """The marginal histogram must reuse hist2d's own bin edges, not
        an independently (if coincidentally) recomputed set."""
        np.random.seed(1)
        X, Y = RV(Normal(0, 1) ** 2)
        sims = (X & Y).sim(2000)
        p = sims.plot(marginal=True, type="hist2d", bins=17, suggest=False)
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
        p = sims.plot(marginal=True, type="tile", bins=17, suggest=False)
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
        # Every kept label is one of the real distinct values.
        self.assertTrue(np.isin(lab, labels).all())
        # The kept positions step evenly along the axis. (They no longer
        # have to include the first and last level: pinning the ends is
        # what used to bend the step out of shape beside them.)
        self.assertEqual(len(np.unique(np.diff(pos))), 1)


class TestDiscreteTickStepIsConstant(unittest.TestCase):
    """Thinned discrete-axis labels step evenly, and land on round values
    whenever the levels allow it.

    The old rule picked its labels with linspace().round(), whose step
    drifts: 20 levels came out labeled 1, 3, 5, 7, 9, *12*, 14, 16, 18, 20
    -- the gap widening to 3 once in the middle for no reason a reader
    could see.
    """

    def thin(self, labels, max_ticks=10):
        positions = np.arange(len(labels))
        pos, lab = _thin_discrete_ticks(positions, np.asarray(labels), max_ticks)
        return pos, list(lab)

    def test_the_reported_case(self):
        """20 discrete levels, the segmented histogram in the report."""
        pos, lab = self.thin(np.arange(1, 21))
        self.assertEqual(lab, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

    def test_step_is_constant_across_many_shapes(self):
        cases = {
            "consecutive": np.arange(1, 21),
            "poisson-like": np.arange(172, 225),
            "just over the cap": np.arange(11),
            "wide": np.arange(1000),
            "negative through zero": np.arange(-30, 31),
            "fractional levels": np.arange(20) * 0.5,
            "categorical": np.array(list("abcdefghijklmnopqrstuvwxyz")),
            "irregular": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 50]),
        }
        for name, labels in cases.items():
            with self.subTest(levels=name):
                pos, lab = self.thin(labels)
                self.assertEqual(len(np.unique(np.diff(pos))), 1)
                self.assertGreaterEqual(len(pos), 2)
                self.assertLessEqual(len(pos), 10)

    def test_round_values_are_preferred(self):
        """Whole-number levels get the scale matplotlib's own locator
        would pick, so a segmented plot reads like the tile plot of the
        same data (29 levels 0-28 -> 0, 3, 6, ...)."""
        _, lab = self.thin(np.arange(29))
        self.assertEqual(lab, [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])

    def test_irregular_levels_fall_back_to_a_fixed_stride(self):
        """Round values that land on levels 4, 9 and 11 would read as an
        even scale on an axis where they are nothing of the sort, so the
        honest fixed stride wins instead."""
        pos, lab = self.thin(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 50]))
        self.assertEqual(lab, [1, 3, 5, 7, 9, 11])

    def test_categorical_levels_use_a_fixed_stride(self):
        pos, lab = self.thin(np.array(list("abcdefghijklmnopqrstuvwxyz")))
        self.assertEqual(lab, ["a", "d", "g", "j", "m", "p", "s", "v", "y"])

    def test_never_labels_a_level_that_does_not_exist(self):
        """The locator's round values are only ever *matched* to levels --
        a value no level carries is never invented as a label."""
        levels = np.arange(172, 225)
        _, lab = self.thin(levels)
        self.assertTrue(set(lab).issubset(set(levels.tolist())))


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
# Discrete-axis tick label formatting (segmented rug/density/hist/box,
# violin) -- regression coverage for the trailing ".0" a whole-number
# discrete level used to print with, which a mixed 2-D dataset always
# produces (one array holds both the discrete and the continuous
# variable, so the discrete values arrive as floats).
# ===========================================================================


class TestDiscreteTickLabelsHelper(unittest.TestCase):
    """Direct unit tests for the shared _discrete_tick_labels() helper."""

    def test_whole_number_floats_lose_the_trailing_decimal(self):
        self.assertEqual(_discrete_tick_labels([1.0, 2.0, 3.0]), ["1", "2", "3"])

    def test_integers_are_unchanged(self):
        self.assertEqual(_discrete_tick_labels(np.arange(3)), ["0", "1", "2"])

    def test_fractional_levels_keep_their_decimals(self):
        """All-or-nothing per axis: one fractional level means every label
        keeps its decimal, so the axis doesn't read 0.5, 1, 1.5."""
        self.assertEqual(_discrete_tick_labels([0.5, 1.0, 1.5]), ["0.5", "1.0", "1.5"])

    def test_negative_whole_numbers(self):
        self.assertEqual(_discrete_tick_labels([-2.0, 0.0, 2.0]), ["-2", "0", "2"])

    def test_non_numeric_levels_print_as_themselves(self):
        self.assertEqual(_discrete_tick_labels(["heads", "tails"]), ["heads", "tails"])

    def test_non_finite_levels_do_not_crash(self):
        self.assertEqual(
            _discrete_tick_labels([1.0, np.nan]), ["1.0", str(np.float64(np.nan))]
        )


class TestSegmentedTickStepEndToEnd(PlotTestCase):
    """The reported plot itself: a segmented histogram of 20 discrete
    levels used to label 1, 3, 5, 7, 9, 12, 14, 16, 18, 20."""

    def setUp(self):
        super().setUp()
        groups = np.resize(np.arange(1, 21), 6000)
        rng = np.random.default_rng(0)
        self.res = RVResults(list(zip(groups, groups + rng.exponential(1, 6000))))

    def test_every_segmented_type_steps_evenly(self):
        for type in ("hist", "density", "rug", "box", "violin"):
            with self.subTest(type=type):
                plt.figure()
                self.res.plot(type, suggest=False)
                labels = [t.get_text() for t in plt.gca().get_xticklabels()]
                values = [int(label) for label in labels]
                self.assertEqual(len(set(np.diff(values))), 1)
                self.assertEqual(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])


class TestSegmentedDiscreteTickLabelFormatting(PlotTestCase):
    """Every plot helper that labels a discrete axis with the level values
    themselves prints a whole number without a trailing ".0", matching how
    the tile plot's numeric locator labels the same values."""

    def test_segmented_rug(self):
        x = np.random.normal(0, 1, 300)
        y = np.resize(np.arange(4, dtype=float), 300)  # float-typed levels
        ax = plt.gca()
        make_segmented_rug(
            x, y, ax, get_next_color(ax), discrete_x=False, discrete_y=True
        )
        self.assertEqual(
            [t.get_text() for t in ax.get_yticklabels()], ["0", "1", "2", "3"]
        )

    def test_segmented_density(self):
        x = np.random.normal(0, 1, 300)
        y = np.resize(np.arange(4, dtype=float), 300)
        ax = plt.gca()
        make_segmented_density(
            x, y, ax, get_next_color(ax), discrete_x=False, discrete_y=True
        )
        self.assertEqual(
            [t.get_text() for t in ax.get_yticklabels()], ["0", "1", "2", "3"]
        )

    def test_segmented_hist(self):
        x = np.random.normal(0, 1, 300)
        y = np.resize(np.arange(4, dtype=float), 300)
        ax = plt.gca()
        make_segmented_hist(
            x, y, ax, get_next_color(ax), discrete_x=False, discrete_y=True
        )
        self.assertEqual(
            [t.get_text() for t in ax.get_yticklabels()], ["0", "1", "2", "3"]
        )

    def test_grouped_boxplot(self):
        x = np.resize(np.arange(4, dtype=float), 300)
        y = np.random.normal(0, 1, 300)
        ax = plt.gca()
        make_grouped_boxplot(
            x, y, ax, get_next_color(ax), discrete_x=True, discrete_y=False
        )
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()], ["0", "1", "2", "3"]
        )

    def test_violin(self):
        positions = [0.0, 1.0, 2.0, 3.0]
        groups = np.resize(np.array(positions), 300)
        data = np.column_stack([groups, np.random.normal(0, 1, 300)])
        ax = plt.gca()
        make_violin(data, positions, ax, get_next_color(ax), "x", 0.5)
        self.assertEqual(
            [t.get_text() for t in ax.get_xticklabels()], ["0", "1", "2", "3"]
        )

    def test_string_levels_are_still_labeled_by_name(self):
        """The formatter only touches numeric levels -- a categorical axis
        keeps its category names."""
        x = np.random.normal(0, 1, 300)
        y = np.resize(np.array(["a", "b", "c"]), 300)
        ax = plt.gca()
        make_segmented_rug(
            x, y, ax, get_next_color(ax), discrete_x=False, discrete_y=True
        )
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], ["a", "b", "c"])

    def test_mixed_simulation_labels_the_discrete_axis_without_decimals(self):
        """End to end through RVResults.plot(): a discrete variable paired
        with a continuous one is stored as float, and used to be labeled
        1.0, 2.0, ... on the segmented histogram's discrete axis."""
        x = np.resize(np.arange(1, 7), 600)
        y = x + np.random.exponential(1, 600)
        RVResults(list(zip(x, y))).plot("hist")
        self.assertEqual(
            [t.get_text() for t in plt.gca().get_xticklabels()],
            ["1", "2", "3", "4", "5", "6"],
        )


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
        """Dispatches to make_tile, but the title reads "Joint Histogram"
        since the continuous axis has been binned rather than tiled."""
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(500).plot()
        self.assertEqual(plt.gca().get_title(), "Joint Histogram")

    def test_2d_both_axes_under_K_2D_is_tile(self):
        """Two discrete axes both within the per-axis cap -> tile."""
        RV(BoxModel(list(range(5))) * BoxModel(list(range(5)))).sim(3000).plot(
            suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "Tile Plot")

    def test_2d_one_axis_over_K_2D_stays_tile(self):
        """One axis over the per-axis cap bins only that axis -> still
        dispatches to make_tile (a mixed tile, not yet a 2-D histogram:
        per-axis independence), but the title reads "Joint Histogram"
        since one axis is no longer discrete."""
        RV(BoxModel(list(range(5))) * BoxModel(list(range(40)))).sim(3000).plot(
            suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "Joint Histogram")

    def test_2d_both_axes_over_K_2D_is_hist2d(self):
        """Both axes over the per-axis cap now bin to a 2-D histogram -- the
        behavior change from the budget model (both axes were a tile under the
        old flat single-threshold rule)."""
        RV(BoxModel(list(range(40))) * BoxModel(list(range(40)))).sim(3000).plot(
            suggest=False
        )
        self.assertEqual(plt.gca().get_title(), "Joint Histogram")

    def test_2d_mixed_small_defaults_to_segmented_rug(self):
        X, Y = RV(Binomial(5, 0.4) * Normal(0, 1))
        (X & Y).sim(60).plot()
        self.assertEqual(plt.gca().get_title(), "Rug Plot")

    def test_2d_discrete_large_defaults_to_tile(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        (X & Y).sim(500).plot()
        self.assertEqual(plt.gca().get_title(), "Tile Plot")

    def test_2d_discrete_small_defaults_to_scatter(self):
        X, Y = RV(Binomial(5, 0.4) ** 2)
        (X & Y).sim(40).plot()
        self.assertEqual(plt.gca().get_title(), "Scatterplot")

    def test_2d_explicit_alias_types_work(self):
        """The lookup-table tokens are accepted as explicit type= values."""
        X, Y = RV(Normal(0, 1) ** 2)
        sims = (X & Y).sim(200)
        sims.plot(type="hist2d")
        self.assertEqual(plt.gca().get_title(), "Joint Histogram")
        plt.close("all")
        sims.plot(type="density2d")
        self.assertEqual(plt.gca().get_title(), "Joint Density (Estimated)")
        plt.close("all")
        Xm, Ym = RV(Binomial(5, 0.4) * Normal(0, 1))
        (Xm & Ym).sim(200).plot(type="segmented_rug")
        self.assertEqual(plt.gca().get_title(), "Rug Plot")

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


class TestOverlayTitle(PlotTestCase):
    """On an overlay, the first plot names the figure.

    Every plot type used to set its title unconditionally, so the last
    .plot() call won and the figure ended up named after whatever was drawn
    on top -- a plot of simulated values titled "Probability Density
    Function" once the true curve was overlaid.
    """

    def setUp(self):
        np.random.seed(42)
        self.sims = RV(Normal(0, 1)).sim(500)

    def test_true_distribution_overlay_keeps_simulated_title(self):
        """The reported case: simulate, then overlay the true pdf."""
        self.sims.plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Histogram")
        Normal(0, 1).plot()
        self.assertEqual(plt.gca().get_title(), "Histogram")

    def test_simulated_overlay_keeps_theoretical_title(self):
        """The rule is first-wins, not simulated-wins: drawing the true
        curve first means it names the figure."""
        Normal(0, 1).plot()
        self.assertEqual(plt.gca().get_title(), "Probability Density Function")
        self.sims.plot(suggest=False)
        self.assertEqual(plt.gca().get_title(), "Probability Density Function")

    def test_second_simulated_type_keeps_first_title(self):
        self.sims.plot(suggest=False)
        self.sims.plot(type="density", suggest=False)
        self.assertEqual(plt.gca().get_title(), "Histogram")

    def test_discrete_overlay_keeps_first_title(self):
        d = RV(Binomial(10, 0.4)).sim(500)
        d.plot(suggest=False)
        first = plt.gca().get_title()
        d.plot(type="dotplot", suggest=False)
        self.assertEqual(plt.gca().get_title(), first)

    def test_a_plot_on_a_fresh_axes_still_gets_its_own_title(self):
        """First-wins must not mean no-one-wins."""
        self.sims.plot(type="ecdf", suggest=False)
        self.assertEqual(plt.gca().get_title(), "ECDF Plot")

    def test_title_survives_repeated_draws(self):
        """A dot plot re-decorates itself on every render
        (_dotplot_relayout), so the helper has to be idempotent."""
        import io

        RV(Binomial(10, 0.4)).sim(60).plot(type="dotplot", suggest=False)
        ax = plt.gca()
        for _ in range(3):
            plt.gcf().savefig(io.BytesIO(), format="png")
        self.assertEqual(ax.get_title(), "Dot Plot")

    def test_set_plot_title_helper_is_first_wins(self):
        from symbulate.plot import set_plot_title

        ax = plt.gca()
        set_plot_title(ax, "First")
        set_plot_title(ax, "Second")
        self.assertEqual(ax.get_title(), "First")

    def test_explicit_clear_still_works(self):
        """The pairs matrix and the marginal layout clear a panel's title
        with a raw ax.set_title("") -- that must stay a real clear, not
        become a no-op."""
        self.sims.plot(suggest=False)
        ax = plt.gca()
        ax.set_title("")
        self.assertEqual(ax.get_title(), "")

    def test_pairs_panels_have_no_titles(self):
        """Regression guard: the matrix clears each panel's own type title
        and carries one suptitle instead."""
        X = RV(MultivariateNormal([0, 0, 0], np.eye(3).tolist()))
        X.sim(300).plot(suggest=False)
        fig = plt.gcf()
        panels = [a for a in fig.axes if a.get_subplotspec() is not None]
        self.assertGreater(len(panels), 0)
        self.assertTrue(all(a.get_title() == "" for a in panels))


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
        self.assertIn("Currently Showing: Density (Estimated)", out)
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
        self.assertIn("Currently Showing: Joint Histogram (Default)", buf.getvalue())

    def test_stackedbar_note_says_stacked_bar_plot(self):
        """type='stackedbar' replaced equal_width=True, so its suggestion
        note must match its axes title ("Stacked Bar Chart")."""
        import io
        import contextlib

        X, Y = RV(Binomial(5, 0.4) ** 2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            (X & Y).sim(500).plot(type="stackedbar", suggest=True)
        self.assertIn("Currently Showing: Stacked Bar Chart", buf.getvalue())

    def test_mosaic_note_says_mosaic_plot(self):
        import io
        import contextlib

        # Few enough categories that the mosaic is kept as asked for; a
        # crowded one is switched to a stacked bar and says so instead
        # (see TestPlot2DMosaic).
        X, Y = RV(Binomial(3, 0.4) ** 2)
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
        Beta(shape1=2, shape2=3).plot()
        xlim = plt.gca().get_xlim()
        self.assertAlmostEqual(xlim[0], 0.0, places=5)
        self.assertAlmostEqual(xlim[1], 1.0, places=5)

    def test_custom_xlim_is_respected(self):
        # plot() takes no window argument -- a window chosen by hand is set on
        # the distribution instead.
        X = Normal(0, 1)
        X.xlim = (-1, 1)
        X.plot()
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

    def test_multivariate_normal_plots_joint_density(self):
        """MultivariateNormal.plot() draws the joint density of two variables."""
        MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]).plot()
        self.assertEqual(plt.gcf().get_suptitle(), "Joint Probability Density Function")

    def test_multinomial_plots_joint_pmf(self):
        """Multinomial.plot() draws a probability per pair of counts."""
        Multinomial(n=10, p=[0.2, 0.3, 0.5]).plot()
        self.assertEqual(plt.gcf().get_suptitle(), "Joint Probability Mass Function")


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


class TestSamplePathWholeNumberTicks(PlotTestCase):
    """A path stepping through whole numbers gets whole-number ticks.

    An index, or discrete time, has nothing between one step and the next, so
    ticks at 0.5 mark positions the path does not have. A continuous-time path
    is evaluated at real times and must keep its fractional ticks.
    """

    def _forces_integers(self):
        plt.gcf().canvas.draw()
        locator = plt.gca().xaxis.get_major_locator()
        return isinstance(locator, MaxNLocator) and bool(
            getattr(locator, "_integer", False)
        )

    def test_index_paths_get_whole_numbers(self):
        X = RV(BoxModel([1, 2, 3, 4, 5, 6], size=4))
        X.sim(20).plot(type="path", suggest=False)
        self.assertEqual(plt.gca().get_xlabel(), "Index")
        self.assertTrue(self._forces_integers())
        for tick in plt.gca().get_xticks():
            self.assertEqual(tick, round(tick))

    def test_a_tuple_plots_against_whole_number_indices(self):
        Tuple([1, 4, 2, 8, 5]).plot()
        self.assertTrue(self._forces_integers())

    def test_discrete_time_processes_get_whole_numbers(self):
        MarkovChain([[0.5, 0.5], [0.3, 0.7]], [1, 0]).draw().plot(tmin=0, tmax=6)
        self.assertTrue(self._forces_integers())

    def test_an_ensemble_of_paths_names_its_own_axis(self):
        """Each result knows what it is plotted against, so the ensemble must
        not relabel them all: a Brownian motion's clock read "Index"."""
        cases = [
            ("Brownian motion", RV(BrownianMotion()), "Time"),
            ("Poisson process", RV(PoissonProcess(rate=2)), "Time"),
            (
                "continuous-time Markov chain",
                RV(ContinuousTimeMarkovChain([[-1, 1], [1, -1]], [1, 0])),
                "Time",
            ),
            (
                "discrete-time Markov chain",
                RV(MarkovChain([[0.5, 0.5], [0.3, 0.7]], [1, 0])),
                "Index",
            ),
            ("tuple of variables", RV(BoxModel([1, 2, 3], size=4)), "Index"),
        ]
        for name, rv, expected in cases:
            with self.subTest(process=name):
                plt.close("all")
                plt.figure()
                sims = rv.sim(3)
                if expected == "Index":
                    sims.plot(type="path", suggest=False)
                else:
                    sims.plot(suggest=False)
                self.assertEqual(plt.gca().get_xlabel(), expected)

    def test_continuous_time_paths_keep_real_times(self):
        """The one thing this must not break: a continuous path's clock."""
        cases = [
            ("Brownian motion", BrownianMotion()),
            ("Poisson process", PoissonProcess(rate=3)),
            (
                "continuous-time Markov chain",
                ContinuousTimeMarkovChain([[-1, 1], [1, -1]], [1, 0]),
            ),
        ]
        for name, process in cases:
            with self.subTest(process=name):
                plt.close("all")
                plt.figure()
                # A short window, where the correct ticks are fractional.
                process.draw().plot(tmin=0, tmax=1.5)
                self.assertEqual(plt.gca().get_xlabel(), "Time")
                self.assertFalse(
                    self._forces_integers(),
                    "a continuous-time path was forced onto whole-number ticks",
                )
                fractional = [t for t in plt.gca().get_xticks() if t != round(float(t))]
                self.assertGreater(len(fractional), 0)


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

    def test_default_style_is_a_plain_line(self):
        line = self.draw_path()
        self.assertEqual(line.get_drawstyle(), "default")
        self.assertEqual(line.get_linestyle(), "-")
        self.assertEqual(line.get_marker(), "None")

    def test_line_style_is_the_same_as_the_default(self):
        line = self.draw_path(style="line")
        self.assertEqual(line.get_drawstyle(), "default")
        self.assertEqual(line.get_linestyle(), "-")
        self.assertEqual(line.get_marker(), "None")

    def test_dots_style_marks_each_point_with_a_dashed_line(self):
        line = self.draw_path(style="dots")
        self.assertEqual(line.get_linestyle(), "--")
        self.assertEqual(line.get_marker(), ".")

    def test_steps_style_holds_flat_until_the_next_point(self):
        line = self.draw_path(style="steps")
        self.assertEqual(line.get_drawstyle(), "steps-post")
        self.assertEqual(line.get_linestyle(), "-")
        self.assertEqual(line.get_marker(), "None")

    def test_invalid_style_raises_helpful_error(self):
        with self.assertRaises(ValueError) as cm:
            self.draw_path(style="scatter")
        message = str(cm.exception)
        self.assertIn("scatter", message)
        self.assertIn("line", message)
        self.assertIn("dots", message)
        self.assertIn("steps", message)

    def test_style_does_not_change_the_data_or_the_defaults(self):
        """style= only changes how the points are joined."""
        for style in ("line", "dots", "steps"):
            with self.subTest(style=style):
                plt.close("all")
                line = self.draw_path(style=style)
                np.testing.assert_allclose(line.get_xdata(), self.times)
                np.testing.assert_allclose(line.get_ydata(), self.values)
                self.assertEqual(line.get_linewidth(), SAMPLE_PATH_LINEWIDTH)
                self.assertEqual(line.get_alpha(), SAMPLE_PATH_ALPHA)


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
        self.assertIn('Density (Estimated) (type = "density")', msg)

    def test_mixed_suggestion_uses_short_type_names(self):
        """On mixed data the lookup table and suggestion note use the short
        type names (rug/hist/density), never the segmented_ aliases."""
        default, alts = default_plot_type("2D_mixed", True)
        msg = suggestion_message(default, default, alts)
        self.assertNotIn("segmented_", msg)
        self.assertIn('Histogram (type = "hist")', msg)
        self.assertIn('Density (Estimated) (type = "density")', msg)


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


class TestTheoreticalTwoVariableLayout(PlotTestCase):
    """A theoretical plot of two variables gets the same three panels.

    Both sides call ``setup_marginal_axes``, so the simulated and
    theoretical layouts cannot drift apart. The strips here are the exact
    closed-form marginal pdf/pmf, not an estimate.
    """

    def _panels(self):
        return [a for a in plt.gcf().axes if a.get_subplotspec() is not None]

    def _strips(self, joint):
        strips = [a for a in self._panels() if a is not joint]
        return (
            max(strips, key=lambda a: a.get_position().y0),
            max(strips, key=lambda a: a.get_position().x0),
        )

    def test_continuous_and_discrete_both_get_three_panels(self):
        for dist, title in [
            (
                MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]),
                "Joint Probability Density Function",
            ),
            (Multinomial(n=10, p=[0.2, 0.3, 0.5]), "Joint Probability Mass Function"),
        ]:
            with self.subTest(dist=type(dist).__name__):
                plt.close("all")
                plt.figure()
                dist.plot()
                self.assertEqual(len(self._panels()), 3)
                self.assertEqual(plt.gcf().get_suptitle(), title)
                self.assertEqual(plt.gca().get_title(), "")

    def test_the_right_strip_runs_sideways(self):
        """The strip beside the y-axis has the variable on *its* y-axis, so
        it lines up with the joint panel -- the density runs along x."""
        p = MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]).plot()
        marg_x, marg_y = self._strips(p.ax)
        # Top strip: value on x, density on y (the univariate orientation).
        self.assertEqual(marg_x.get_xlim(), p.ax.get_xlim())
        self.assertEqual(marg_x.get_xlabel(), "")
        self.assertEqual(marg_x.get_ylabel(), "Marginal Density")
        # Right strip: transposed.
        self.assertEqual(marg_y.get_ylim(), p.ax.get_ylim())
        self.assertEqual(marg_y.get_xlabel(), "Marginal Density")
        self.assertEqual(marg_y.get_ylabel(), "")
        # Its curve rises from a density of 0, so its x data is nonnegative
        # and its y data spans the variable's window.
        (curve,) = marg_y.get_lines()
        xdata, ydata = curve.get_data()
        self.assertTrue(np.all(np.asarray(xdata) >= 0))
        self.assertAlmostEqual(min(ydata), p.ax.get_ylim()[0], places=6)

    def test_right_strips_frequency_labels_are_rotated_to_avoid_overlap(self):
        """Regression: the right-hand strip is narrow, so its density-axis
        decimal tick labels ran into each other laid out horizontally --
        rotating them fits the same tick count. Same fix as the simulated
        side (setup_marginal_axes/thin_marginal_frequency_ticks are
        shared), checked here too since the theoretical side hits it
        independently through MultivariateDistribution.plot()."""
        p = MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]).plot()
        marg_x, marg_y = self._strips(p.ax)
        plt.gcf().canvas.draw()
        labels = [t for t in marg_y.get_xticklabels() if t.get_text()]
        self.assertGreater(len(labels), 1)
        for label in labels:
            self.assertEqual(label.get_rotation(), MARGINAL_FREQ_TICK_ROTATION)
        # The top strip is wide, so it keeps upright labels.
        for label in marg_x.get_yticklabels():
            if label.get_text():
                self.assertEqual(label.get_rotation(), 0)
        renderer = plt.gcf().canvas.get_renderer()
        boxes = [t.get_window_extent(renderer=renderer) for t in labels]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                self.assertFalse(boxes[i].overlaps(boxes[j]))

    def test_a_discrete_strip_keeps_its_masses(self):
        """A pmf strip is dots plus a dashed connector, transposed the same
        way -- the dots are a PathCollection, not a line."""
        p = Multinomial(n=10, p=[0.2, 0.3, 0.5]).plot()
        _, marg_y = self._strips(p.ax)
        self.assertGreater(len(marg_y.collections), 0)
        offsets = np.asarray(marg_y.collections[0].get_offsets())
        # Transposed: the probabilities are the x coordinate now.
        self.assertTrue(np.all(offsets[:, 0] >= 0))
        self.assertEqual(marg_y.get_xlabel(), "Marginal Probability")

    def test_two_variable_colorbar_names_the_joint_quantity(self):
        for dist, expected in [
            (
                MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]),
                "Joint Density",
            ),
            (Multinomial(n=10, p=[0.2, 0.3, 0.5]), "Joint Probability"),
        ]:
            with self.subTest(dist=type(dist).__name__):
                plt.close("all")
                dist.plot()
                bars = [a for a in plt.gcf().axes if a.get_subplotspec() is None]
                self.assertEqual([a.get_ylabel() for a in bars], [expected])

    def test_the_strips_are_the_exact_marginals(self):
        """A MultivariateNormal's marginal is a Normal, so the top strip is
        that Normal's pdf -- not a sum over the joint surface."""
        dist = MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])
        p = dist.plot()
        marg_x, _ = self._strips(p.ax)
        (curve,) = marg_x.get_lines()
        xdata, ydata = curve.get_data()
        expected = dist._marginal_1d(0).pdf(np.asarray(xdata))
        np.testing.assert_allclose(ydata, expected, rtol=1e-10)

    def test_an_explicit_axes_draws_only_the_joint(self):
        """A caller who supplies ax= is placing the plot in a layout of their
        own, and one axes has no room for the strips."""
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        MultivariateNormal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]]).plot(ax=ax)
        self.assertEqual(len(self._panels()), 1)
        # Its own title stays on it, since nothing sits above it.
        self.assertEqual(ax.get_title(), "Joint Probability Density Function")
        self.assertEqual(fig.get_suptitle(), "")

    def test_a_pairs_matrix_is_unaffected(self):
        """Three or more variables still draw the matrix, whose diagonal
        already shows each variable on its own."""
        MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3)).plot()
        self.assertEqual(len(self._panels()), 6)
        self.assertEqual(plt.gcf().get_suptitle(), "Probability Density Functions")


class TestJointTheoreticalPlots(PlotTestCase):
    """The joint pdf/pmf helpers behind MultivariateDistribution.plot()."""

    def test_make_joint_pdf_evaluates_on_a_grid(self):
        # The surface is the pdf itself, evaluated on a
        # JOINT_PDF_GRID_POINTS square grid over the given window.
        seen = {}

        def pdf(x, y):
            seen["n"] = len(x)
            return np.exp(-(x**2 + y**2) / 2)

        ax = plt.gca()
        make_joint_pdf(pdf, (-3, 3), (-3, 3), ax)
        self.assertEqual(seen["n"], JOINT_PDF_GRID_POINTS**2)
        self.assertEqual(ax.get_xlim(), (-3, 3))
        self.assertEqual(ax.get_ylim(), (-3, 3))
        self.assertEqual(ax.get_title(), "Joint Probability Density Function")

    def test_make_joint_pdf_contour_titles_and_bands(self):
        pdf = lambda x, y: np.exp(-(x**2 + y**2) / 2)
        ax = plt.gca()
        make_joint_pdf(pdf, (-3, 3), (-3, 3), ax, contour=True)
        self.assertEqual(ax.get_title(), "Joint Probability Density Function")

    def test_make_joint_pdf_survives_an_unbounded_density(self):
        # A density that runs to infinity at the edge of its support is
        # scaled by its largest finite value instead of collapsing the
        # whole surface into one color.
        def pdf(x, y):
            with np.errstate(divide="ignore"):
                return 1.0 / np.maximum(x, 0)

        ax = plt.gca()
        make_joint_pdf(pdf, (0, 1), (0, 1), ax)
        self.assertTrue(np.isfinite(ax.collections[0].get_array().max()))

    def test_make_joint_pdf_labels_and_no_colorbar_for_a_panel(self):
        pdf = lambda x, y: np.exp(-(x**2 + y**2) / 2)
        ax = plt.gca()
        n_axes_before = len(plt.gcf().axes)
        make_joint_pdf(
            pdf,
            (-3, 3),
            (-3, 3),
            ax,
            colorbar=False,
            xlabel="X1",
            ylabel="X3",
            title=False,
        )
        # No extra axes: a colorbar would have added one.
        self.assertEqual(len(plt.gcf().axes), n_axes_before)
        self.assertEqual(ax.get_xlabel(), "X1")
        self.assertEqual(ax.get_ylabel(), "X3")
        self.assertEqual(ax.get_title(), "")

    def test_make_joint_pmf_one_cell_per_pair_of_values(self):
        # Cells are centered on the values, so the mesh spans half a unit
        # past the outermost value on every side.
        pmf = lambda x, y: np.full(len(x), 1 / 16)
        ax = plt.gca()
        mesh = make_joint_pmf(pmf, np.arange(4), np.arange(4), ax)
        self.assertEqual(mesh.get_array().shape, (4, 4))
        self.assertEqual(ax.get_xlim(), (-0.5, 3.5))
        self.assertEqual(ax.get_ylim(), (-0.5, 3.5))
        self.assertEqual(ax.get_title(), "Joint Probability Mass Function")

    def test_second_joint_plot_warns_and_keeps_one_colorbar(self):
        # The "warn but still draw" tier of the overlay policy, the same one
        # two tile plots or two 2-D histograms fall into.
        pdf = lambda x, y: np.exp(-(x**2 + y**2) / 2)
        ax = plt.gca()
        make_joint_pdf(pdf, (-3, 3), (-3, 3), ax)
        n_axes = len(plt.gcf().axes)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            make_joint_pdf(pdf, (-3, 3), (-3, 3), ax)
        self.assertIn("second joint plot", printed.getvalue())
        # No second colorbar: it would land on top of the first one.
        self.assertEqual(len(plt.gcf().axes), n_axes)

    def test_make_joint_pmf_refuses_an_unreadable_grid(self):
        pmf = lambda x, y: np.zeros(len(x))
        side = int(np.sqrt(JOINT_PMF_MAX_CELLS)) + 10
        with self.assertRaises(ValueError) as cm:
            make_joint_pmf(pmf, np.arange(side), np.arange(side), plt.gca())
        self.assertIn("too many to read", str(cm.exception))


# ===========================================================================
# Pairs matrix of simulated results (RVResults.plot())
# ===========================================================================


def _continuous_sim(n=400, k=3, seed=1):
    """Simulated results of ``k`` correlated continuous variables."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    return RVResults(
        [
            tuple(float(base[i] * 0.6 + rng.normal() * 0.8) for _ in range(k))
            for i in range(n)
        ]
    )


def _discrete_sim(n=400, k=3, seed=2):
    """Simulated results of ``k`` small-cardinality discrete variables."""
    rng = np.random.default_rng(seed)
    return RVResults(
        [tuple(int(rng.integers(1, 7)) for _ in range(k)) for _ in range(n)]
    )


def _pairs_panels(fig=None):
    """The matrix's panels, without the per-pair colorbars beside them.

    Panels come from the grid, so they have a subplot spec; the colorbars are
    placed with ``add_axes`` in the empty mirroring cells and have none.
    """
    fig = fig if fig is not None else plt.gcf()
    return [ax for ax in fig.axes if ax.get_subplotspec() is not None]


def _pairs_colorbars(fig=None):
    """The matrix's per-pair colorbar axes."""
    fig = fig if fig is not None else plt.gcf()
    return [ax for ax in fig.axes if ax.get_subplotspec() is None]


def _mixed_sim(n=400, seed=3):
    """One discrete variable and two continuous ones."""
    rng = np.random.default_rng(seed)
    return RVResults(
        [
            (int(rng.integers(1, 4)), float(rng.normal()), float(rng.normal()))
            for _ in range(n)
        ]
    )


class TestPairsPanelChoice(PlotTestCase):
    """Which plot each panel of the matrix gets."""

    def test_continuous_diagonal_is_the_1d_default(self):
        """Whatever that variable alone would get: a histogram."""
        sim = _continuous_sim()
        for i in range(3):
            with self.subTest(variable=i):
                self.assertEqual(sim._pairs_diagonal_type(i), "hist")

    def test_discrete_diagonal_is_the_1d_default(self):
        sim = _discrete_sim()
        for i in range(3):
            with self.subTest(variable=i):
                self.assertEqual(sim._pairs_diagonal_type(i), "impulse")

    def test_small_simulation_uses_the_small_sample_defaults(self):
        """A panel shows what that data alone would show, small n included."""
        small_continuous = _continuous_sim(n=40)
        self.assertEqual(small_continuous._pairs_diagonal_type(0), "rug")
        self.assertEqual(small_continuous._pairs_joint_type(0, 1), "scatter")
        small_discrete = _discrete_sim(n=40)
        self.assertEqual(small_discrete._pairs_diagonal_type(0), "dotplot")
        self.assertEqual(small_discrete._pairs_joint_type(0, 1), "scatter")

    def test_two_continuous_variables_get_a_2d_histogram(self):
        self.assertEqual(_continuous_sim()._pairs_joint_type(0, 1), "hist2d")

    def test_two_discrete_variables_get_a_tile(self):
        self.assertEqual(_discrete_sim()._pairs_joint_type(0, 1), "tile")

    def test_a_mixed_pair_gets_a_tile(self):
        """The continuous axis is binned; the discrete one keeps its levels."""
        self.assertEqual(_mixed_sim()._pairs_joint_type(0, 1), "tile")

    def test_panels_that_encode_nothing_in_color_get_no_colorbar(self):
        """A scatter has no color scale to explain."""
        plt.figure()
        _continuous_sim(n=40, k=3).plot()
        self.assertEqual(len(_pairs_colorbars()), 0)
        self.assertEqual(len(_pairs_panels()), 6)


class TestVariablesChoosesWhatIsDrawn(PlotTestCase):
    """``variables=`` picks the variables, and its length picks the plot.

    One variable is that variable's own distribution, two are their joint
    distribution, three or more are a matrix of every pair.
    """

    def setUp(self):
        super().setUp()
        self.dist = MultivariateNormal(mean=[0, 1, 2, 3, 4], cov=np.eye(5))

    def _panels(self):
        return [a for a in plt.gcf().axes if a.get_subplotspec() is not None]

    def test_a_single_number_draws_that_variables_own_distribution(self):
        for arg in [2, [2], (2,), np.array([2])]:
            with self.subTest(variables=arg):
                plt.close("all")
                plt.figure()
                p = self.dist.plot(variables=arg)
                self.assertEqual(len(self._panels()), 1)
                self.assertEqual(p.ax.get_title(), "Probability Density Function")
                # Named for the variable asked for, not a generic "Value".
                self.assertEqual(p.ax.get_xlabel(), "Variable 3")
                self.assertEqual(p.ax.get_ylabel(), "Density")

    def test_a_single_variable_is_the_exact_marginal(self):
        """Not a slice or a sum over the joint surface."""
        p = self.dist.plot(variables=2)
        (curve,) = p.ax.get_lines()
        xs, ys = curve.get_data()
        expected = self.dist._marginal_1d(2).pdf(np.asarray(xs))
        np.testing.assert_allclose(ys, expected, rtol=1e-10)

    def test_a_single_discrete_variable_draws_its_pmf(self):
        p = Multinomial(n=10, p=[0.2, 0.3, 0.5]).plot(variables=1)
        self.assertEqual(p.ax.get_title(), "Probability Mass Function")
        self.assertEqual(p.ax.get_xlabel(), "Variable 2")

    def test_one_variable_works_where_the_joint_plot_is_refused(self):
        """A two-category Multinomial varies in only one direction, so it has
        no joint plot -- but each of its marginals is a Binomial."""
        dist = Multinomial(n=10, p=[0.4, 0.6])
        with self.assertRaises(Exception):
            dist.plot()
        p = dist.plot(variables=0)
        self.assertEqual(p.ax.get_xlabel(), "Variable 1")

    def test_labels_follow_the_numbers_asked_for(self):
        self.dist.plot(variables=[0, 2, 4])
        labels = set()
        for ax in self._panels():
            labels |= {ax.get_xlabel(), ax.get_ylabel()}
        self.assertEqual(
            {l for l in labels if l.startswith("Variable")},
            {"Variable 1", "Variable 3", "Variable 5"},
        )

    def test_two_variables_still_draw_their_joint_distribution(self):
        self.dist.plot(variables=[0, 2])
        self.assertEqual(len(self._panels()), 3)
        self.assertEqual(plt.gca().get_xlabel(), "Variable 1")
        self.assertEqual(plt.gca().get_ylabel(), "Variable 3")

    def test_bad_variables_explain_themselves(self):
        cases = [
            (7, "asks for a variable it doesn't have"),
            ([7], "asks for a variable it doesn't have"),
            ([], "needs at least one"),
            (1.5, "variable number or a list of them"),
            ("x", "variable number or a list of them"),
            (True, "variable number or a list of them"),
            ([1, 1], "must be different"),
        ]
        for arg, expected in cases:
            with self.subTest(variables=arg):
                plt.close("all")
                plt.figure()
                with self.assertRaises(Exception) as cm:
                    self.dist.plot(variables=arg)
                self.assertIn(expected, str(cm.exception))


class TestPairsColumnsShareAScale(PlotTestCase):
    """A column of a pairs matrix shows one variable at one scale.

    A stem or bar on the diagonal has to sit directly above the tile cell or
    mesh column for that same value. Each panel type frames its own axes
    differently, so this takes ``align_pairs_columns`` -- and both matrices
    call it.
    """

    def _cells(self):
        cells = {}
        for ax in plt.gcf().axes:
            spec = ax.get_subplotspec()
            if spec is not None:
                cells[(spec.rowspan.start, spec.colspan.start)] = ax
        return cells

    def _assert_one_scale_per_variable(self):
        cells = self._cells()
        k = max(row for row, _ in cells) + 1
        for variable in range(k):
            down = {
                tuple(ax.get_xlim())
                for (row, col), ax in cells.items()
                if col == variable
            }
            self.assertEqual(
                len(down), 1, f"column {variable} is drawn at {len(down)} scales"
            )
            # The same variable is the y variable of the joint panels along
            # its row, so it must be at that scale there too.
            across = {
                tuple(ax.get_ylim())
                for (row, col), ax in cells.items()
                if row == variable and row != col
            }
            if across:
                self.assertEqual(
                    down,
                    across,
                    f"variable {variable} is drawn at one scale down its "
                    "column and another across its row",
                )

    def test_columns_share_a_scale_for_every_configuration(self):
        cases = [
            ("discrete large n", RV(Binomial(5, 0.4) ** 3), 600),
            ("continuous large n", RV(Normal(0, 1) ** 3), 600),
            ("discrete small n", RV(Binomial(5, 0.4) ** 3), 40),
            ("continuous small n", RV(Normal(0, 1) ** 3), 40),
        ]
        for name, rvs, n in cases:
            with self.subTest(case=name):
                plt.close("all")
                plt.figure()
                A, B, C = rvs
                (A & B & C).sim(n).plot(suggest=False)
                # Drawn first: a dot plot re-frames its own value axis on
                # every draw, and used to undo the alignment here.
                plt.gcf().canvas.draw()
                self._assert_one_scale_per_variable()

    def test_theoretical_columns_share_a_scale(self):
        for dist in [
            MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3)),
            Multinomial(n=10, p=[0.3, 0.3, 0.2, 0.2]),
        ]:
            with self.subTest(dist=type(dist).__name__):
                plt.close("all")
                plt.figure()
                dist.plot()
                plt.gcf().canvas.draw()
                self._assert_one_scale_per_variable()

    def test_bins_applies_to_the_diagonal_too(self):
        """One bin count for the whole matrix -- the diagonal histogram used
        to keep the default 30 while the joint panels honored bins=."""
        X, Y, Z = RV(Normal(0, 1) ** 3)
        (X & Y & Z).sim(600).plot(bins=12, suggest=False)
        for (row, col), ax in self._cells().items():
            if row == col:
                self.assertEqual(len(ax.patches), 12)
            else:
                meshes = [
                    c.get_array()
                    for c in ax.collections
                    if getattr(c, "get_array", None) is not None
                    and c.get_array() is not None
                ]
                self.assertEqual(meshes[0].shape, (12, 12))

    def test_bins_leaves_a_discrete_diagonal_alone(self):
        """A discrete panel is never binned, so bins= must not disturb it."""
        A, B, C = RV(Binomial(5, 0.4) ** 3)
        (A & B & C).sim(600).plot(bins=12, suggest=False)
        for (row, col), ax in self._cells().items():
            if row == col:
                self.assertEqual(len(ax.patches), 0)  # stems, not bars
                self.assertGreater(len(ax.collections), 0)

    def test_the_diagonal_walks_the_palette(self):
        """Each variable's own distribution reads as its own series, rather
        than every diagonal panel taking the first color."""
        import matplotlib.colors as mcolors

        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        A, B, C = RV(Binomial(5, 0.4) ** 3)
        (A & B & C).sim(600).plot(suggest=False)
        seen = []
        for (row, col), ax in sorted(self._cells().items()):
            if row != col:
                continue
            seen.append(mcolors.to_hex(ax.collections[0].get_color()[0]))
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(seen, [mcolors.to_hex(c) for c in palette[:3]])

    def test_the_figure_title_names_what_the_panels_are(self):
        A, B, C = RV(Normal(0, 1) ** 3)
        (A & B & C).sim(200).plot(suggest=False)
        self.assertEqual(PAIRS_SUPTITLE, "Joint and Marginal Distributions")
        self.assertEqual(plt.gcf().get_suptitle(), PAIRS_SUPTITLE)

    def test_a_colorbar_names_its_pair_compactly(self):
        A, B, C = RV(Normal(0, 1) ** 3)
        (A & B & C).sim(200).plot(suggest=False)
        titles = sorted(
            a.get_title()
            for a in plt.gcf().axes
            if a.get_subplotspec() is None and a.get_title()
        )
        self.assertEqual(
            titles, ["Variables 1 & 2", "Variables 1 & 3", "Variables 2 & 3"]
        )


class TestPairsLayout(PlotTestCase):
    """The grid of panels the matrix builds."""

    def test_lower_triangle_only(self):
        """k variables give k(k+1)/2 panels: diagonal plus lower triangle."""
        for k in (3, 4, 5):
            with self.subTest(variables=k):
                plt.close("all")
                plt.figure()
                _continuous_sim(k=k).plot()
                self.assertEqual(len(_pairs_panels()), k * (k + 1) // 2)

    def test_matrix_is_the_default_for_three_or_more_variables(self):
        """No keyword needed: .plot() alone draws the matrix."""
        plt.figure()
        _continuous_sim(k=3).plot()
        self.assertEqual(len(_pairs_panels()), 6)

    def test_two_variables_still_get_a_single_joint_plot(self):
        """The matrix starts at three variables, not two."""
        plt.figure()
        _continuous_sim(k=2).plot(suggest=False)
        self.assertEqual(len(_pairs_panels()), 1)

    def test_a_subset_is_chosen_by_indexing_the_variable(self):
        """No dims= here: X[[0, 2]].sim(n).plot() picks the variables."""
        plt.figure()
        _continuous_sim(k=4)._pairs_subset((0, 2)).plot(suggest=False)
        self.assertEqual(len(_pairs_panels()), 1)

    def test_figure_is_sized_to_the_grid(self):
        plt.figure()
        _continuous_sim(k=3).plot()
        width, height = plt.gcf().get_size_inches()
        self.assertAlmostEqual(width, 3 * JOINT_PAIRS_PANEL_SIZE)
        self.assertAlmostEqual(height, 3 * JOINT_PAIRS_PANEL_SIZE)

    def test_returns_a_symbulate_plot(self):
        plt.figure()
        returned = _continuous_sim().plot()
        self.assertIsInstance(returned, SymbulatePlot)

    def test_only_the_outer_edges_are_labeled(self):
        plt.figure()
        _continuous_sim(k=3).plot()
        axes = _pairs_panels()
        # The bottom row carries x labels; every other panel has none.
        x_labels = [a.get_xlabel() for a in axes]
        self.assertEqual(
            sorted(l for l in x_labels if l),
            ["Variable 1", "Variable 2", "Variable 3"],
        )
        # A diagonal panel's y-axis is that variable's own distribution rather
        # than the variable, so it names what it measures; the rest of the
        # left column names its row's variable.
        y_labels = [a.get_ylabel() for a in axes]
        self.assertEqual(
            sorted(l for l in y_labels if l),
            [
                "Marginal Density",
                "Marginal Density",
                "Marginal Density",
                "Variable 2",
                "Variable 3",
            ],
        )

    def test_diagonal_panels_name_the_marginal_they_show(self):
        """The diagonal is not the joint distribution its neighbors are."""
        plt.figure()
        _continuous_sim(k=3).plot()
        # Panels are added row by row, so the first one is (0, 0).
        self.assertEqual(_pairs_panels()[0].get_ylabel(), "Marginal Density")

    def test_diagonal_marginal_label_follows_the_quantity_shown(self):
        """Counts aren't densities, so a count diagonal doesn't claim to be."""
        plt.figure()
        _continuous_sim(k=3).plot(normalize=False)
        self.assertEqual(_pairs_panels()[0].get_ylabel(), "Marginal Count")

    def test_inner_x_tick_labels_are_hidden(self):
        """Every panel in a column shares the variable, so they'd repeat."""
        plt.figure()
        _continuous_sim(k=3).plot()
        # Tick label text is filled in at draw time, so draw before reading it.
        fig = plt.gcf()
        fig.canvas.draw()
        for ax in _pairs_panels(fig):
            texts = [t.get_text() for t in ax.get_xticklabels()]
            if ax.get_xlabel():
                # A bottom-row panel: its numbers are the ones that apply to
                # the whole column.
                self.assertTrue(any(texts), "bottom row lost its tick labels")
            else:
                self.assertFalse(any(texts), "inner panel repeated its column's")

    def test_panels_in_a_column_cover_the_same_range(self):
        """One variable is binned the same way wherever it appears."""
        plt.figure()
        _continuous_sim(k=3).plot()
        # The two joint panels of the leftmost column both show variable 0
        # on x, so their x-ranges must agree.
        joint = [a for a in plt.gcf().axes if a.collections and not a.lines]
        self.assertEqual(joint[0].get_xlim(), joint[1].get_xlim())

    def test_one_colorbar_per_joint_panel(self):
        """Each pair keeps its own scale, so each gets its own bar."""
        plt.figure()
        _continuous_sim(k=3).plot()
        # 3 joint panels among 3 variables -> 3 colorbars.
        self.assertEqual(len(_pairs_colorbars()), 3)
        self.assertEqual(len(_pairs_panels()), 6)

    def test_colorbars_name_the_pair_they_explain(self):
        plt.figure()
        _continuous_sim(k=3).plot()
        titles = sorted(a.get_title() for a in _pairs_colorbars())
        self.assertEqual(
            titles,
            [
                "Variables 1 & 2",
                "Variables 1 & 3",
                "Variables 2 & 3",
            ],
        )

    def test_pair_label_matches_the_density_label_on_its_bar(self):
        """A bar's caption is lettered alike top and side.

        The pair label identifies the bar rather than titling a plot, so it
        is the size of the bar's own "Joint Density"/"Joint Count" label,
        below a panel title. Both matrices go through
        ``add_pairs_panel_colorbar``, so the
        simulated and theoretical sides are checked together.
        """
        expected = (
            matplotlib.font_manager.font_scalings[JOINT_PAIRS_COLORBAR_TITLE_SIZE]
            * plt.rcParams["font.size"]
        )
        panel_title = (
            matplotlib.font_manager.font_scalings[plt.rcParams["axes.titlesize"]]
            * plt.rcParams["font.size"]
        )
        self.assertLess(expected, panel_title)
        for name, draw in [
            ("simulated", lambda: _continuous_sim(k=3).plot()),
            (
                "theoretical",
                lambda: MultivariateNormal(mean=[0, 0, 0], cov=np.eye(3)).plot(),
            ),
        ]:
            with self.subTest(matrix=name):
                plt.close("all")
                plt.figure()
                draw()
                bars = _pairs_colorbars()
                self.assertEqual({a.title.get_fontsize() for a in bars}, {expected})
                # The "Joint Density" label sits on the bar's y-axis.
                self.assertEqual(
                    {a.yaxis.label.get_fontsize() for a in bars}, {expected}
                )

    def test_colorbars_sit_in_the_empty_mirroring_cells(self):
        """The upper triangle is blank, so the bars go there."""
        plt.figure()
        _continuous_sim(k=3).plot()
        fig = plt.gcf()
        fig.canvas.draw()
        panel_boxes = [a.get_position() for a in _pairs_panels(fig)]
        for bar in _pairs_colorbars(fig):
            box = bar.get_position()
            for panel in panel_boxes:
                self.assertFalse(
                    box.overlaps(panel), "a colorbar landed on top of a panel"
                )

    def test_density_by_default_counts_when_asked(self):
        # "Joint", so a bar can't be confused with the marginal density or
        # count on the diagonal panel it sits across from.
        for normalize, expected in [(True, "Joint Density"), (False, "Joint Count")]:
            with self.subTest(normalize=normalize):
                plt.close("all")
                plt.figure()
                _continuous_sim(k=3).plot(normalize=normalize)
                labels = {a.get_ylabel() for a in _pairs_colorbars()}
                self.assertEqual(labels, {expected})

    def test_count_ticks_have_no_decimals(self):
        """A count is a whole number of simulated values."""
        plt.figure()
        _continuous_sim(k=3).plot(normalize=False)
        fig = plt.gcf()
        fig.canvas.draw()
        for bar in _pairs_colorbars(fig):
            for text in [t.get_text() for t in bar.get_yticklabels()]:
                if text:
                    self.assertNotIn(".", text)

    def test_density_ticks_keep_their_decimals(self):
        plt.figure()
        _continuous_sim(k=3).plot()
        fig = plt.gcf()
        fig.canvas.draw()
        shown = [
            t.get_text() for bar in _pairs_colorbars(fig) for t in bar.get_yticklabels()
        ]
        self.assertTrue(any("." in t for t in shown if t))

    def test_discrete_and_mixed_matrices_get_colorbars_too(self):
        for label, sim in [("discrete", _discrete_sim()), ("mixed", _mixed_sim())]:
            with self.subTest(data=label):
                plt.close("all")
                plt.figure()
                sim.plot()
                self.assertEqual(len(_pairs_colorbars()), 3)

    def test_every_configuration_draws_without_warnings(self):
        for label, sim in [
            ("continuous", _continuous_sim()),
            ("discrete", _discrete_sim()),
            ("mixed", _mixed_sim()),
        ]:
            with self.subTest(data=label):
                plt.close("all")
                plt.figure()
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    sim.plot()

    def test_panels_have_no_titles_of_their_own(self):
        """One title for the figure, not a plot-type title per panel."""
        for label, sim in [
            ("continuous", _continuous_sim()),
            ("discrete", _discrete_sim()),
            ("mixed", _mixed_sim()),
        ]:
            with self.subTest(data=label):
                plt.close("all")
                plt.figure()
                sim.plot()
                fig = plt.gcf()
                self.assertEqual([a.get_title() for a in _pairs_panels(fig)], [""] * 6)
                self.assertEqual(fig._suptitle.get_text(), PAIRS_SUPTITLE)

    def test_panels_do_not_print_suggestion_notes(self):
        """One note per panel would bury the plot in text."""
        plt.figure()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            _continuous_sim().plot()
        self.assertNotIn("Currently Showing", printed.getvalue())


class TestPairsErrors(PlotTestCase):
    """What the matrix refuses to do, and how it says so."""

    def test_cannot_share_a_figure(self):
        plt.figure()
        RVResults([float(v) for v in np.random.default_rng(0).normal(size=50)]).plot(
            suggest=False
        )
        with self.assertRaises(ValueError) as cm:
            _continuous_sim().plot()
        self.assertEqual(str(cm.exception), JOINT_PAIRS_OVERLAY_ERROR)

    def test_too_many_variables_to_read(self):
        plt.figure()
        many = _continuous_sim(n=50, k=JOINT_PAIRS_MAX_DIM + 1)
        with self.assertRaises(Exception) as cm:
            many.plot()
        self.assertIn("too many", str(cm.exception))
        # It points at indexing the variable, which is how a subset is chosen.
        self.assertIn("X[[0, 1, 2]]", str(cm.exception))

    def test_the_old_dims_keyword_explains_itself(self):
        """It was removed: index the random variable instead."""
        plt.figure()
        with self.assertRaises(ValueError) as cm:
            _continuous_sim(k=3).plot(dims=(0, 1))
        message = str(cm.exception)
        self.assertIn("X[[0, 2]]", message)
        self.assertIn("variables=", message)

    def test_the_old_pairs_keyword_explains_itself(self):
        """It was how the matrix used to be asked for; now it is the default."""
        plt.figure()
        with self.assertRaises(ValueError) as cm:
            _continuous_sim(k=3).plot(pairs=True)
        message = str(cm.exception)
        self.assertIn("no longer needed", message)
        self.assertIn("type='path'", message)

    def test_a_one_variable_type_is_refused_for_a_matrix(self):
        plt.figure()
        with self.assertRaises(ValueError) as cm:
            _continuous_sim(k=3).plot(type="hist")
        message = str(cm.exception)
        self.assertIn("3 variables", message)
        self.assertIn("type='path'", message)

    def test_one_variable_cannot_make_a_matrix(self):
        """Reached only by calling the matrix directly -- .plot() on one
        variable draws that variable."""
        one = RVResults([float(v) for v in np.random.default_rng(0).normal(size=50)])
        with self.assertRaises(ValueError) as cm:
            one._plot_pairs()
        self.assertIn("at least two", str(cm.exception))


class TestPairsLeavesOtherPlotsAlone(PlotTestCase):
    """The feature is additive: nothing else about .plot() changes."""

    def test_one_variable_still_plots_by_default(self):
        plt.figure()
        sim = RVResults([float(v) for v in np.random.default_rng(0).normal(size=200)])
        sim.plot(suggest=False)
        self.assertTrue(plt.gcf().axes)

    def test_two_variables_still_plot_by_default(self):
        """Unchanged: a 2-D histogram plus its colorbar."""
        plt.figure()
        _continuous_sim(k=2).plot(suggest=False)
        self.assertTrue(plt.gcf().axes)
        self.assertTrue(plt.gcf().axes[0].collections)

    def test_path_type_still_draws_the_index_plot(self):
        """The old default is kept, as an alternative."""
        np.random.seed(4)
        plt.figure()
        sim = RV(
            MultivariateNormal(mean=[0, 0, 0], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        ).sim(100)
        sim.plot(type="path", suggest=False)
        self.assertEqual(len(plt.gcf().axes), 1)
        self.assertEqual(plt.gca().get_xlabel(), "Index")
        self.assertEqual(len(plt.gca().lines), 100)

    def test_suggestion_note_offers_the_path_plot(self):
        plt.figure()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            _continuous_sim(k=3).plot(suggest=True)
        message = printed.getvalue()
        self.assertIn("Pairs Plot (Default)", message)
        self.assertIn('Path Plot (type = "path")', message)

    def test_explicit_type_still_honored_on_two_variables(self):
        plt.figure()
        _continuous_sim(k=2).plot(type="scatter", suggest=False)
        self.assertTrue(plt.gca().collections)


if __name__ == "__main__":
    unittest.main()
