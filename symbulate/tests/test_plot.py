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
(and with it the classify_data-driven default plot type) is stable from
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
    classify_data,
    default_plot_type,
    suggestion_message,
    should_show_suggestion,
    make_ecdf,
    make_mosaic,
    make_tile,
    make_segmented_rug,
    DEFAULT_PLOT_TYPE,
    PLOT_DISPLAY_NAME,
    TILE_DEFAULT_BINS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def histogram_area(ax):
    """Return the total area (width × height) of all histogram bars on ax."""
    return sum(p.get_width() * p.get_height() for p in ax.patches)


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

    def test_bar_type_is_alias_for_hist(self):
        self.sims.plot(type="bar")
        self.assertGreater(len(plt.gca().patches), 0)

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
        # Discrete data auto-selects impulse, but via classify_data, not ours
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

        classify_data counts distinct values (Poisson(3) has far fewer
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
        self.sims.plot(type="marginal")
        self.assertGreaterEqual(len(plt.gcf().axes), 3)

    def test_marginal_density_creates_at_least_three_axes(self):
        self.sims.plot(type=("marginal", "density"))
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
        self.assertEqual(ax.get_xlabel(), "X")
        self.assertEqual(ax.get_ylabel(), "Y")

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

    def test_dotplot_non_numeric_raises_friendly_error(self):
        from symbulate.plot import make_dotplot

        with self.assertRaises(TypeError) as cm:
            make_dotplot(["H", "T", "H"], plt.gca(), "#56B4E9")
        self.assertIn("tabulate", str(cm.exception))

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

    def test_tile_fallback_uses_classify_data_not_dtype(self):
        """A direct make_tile call with unspecified discreteness classifies
        with classify_data, not the raw dtype: a wide-support integer axis
        (many distinct values) is treated as continuous and binned into
        TILE_DEFAULT_BINS cells, not given one skinny cell per value."""
        rng = np.random.default_rng(0)
        x = rng.integers(0, 200, 2000)  # int dtype, >40 unique values
        y = rng.normal(0, 1, 2000)  # continuous
        mesh = make_tile(x, y, plt.gca())  # discrete_x/discrete_y default None
        # shape is (ny, nx); a binned (continuous) x axis has TILE_DEFAULT_BINS
        # columns, whereas the old dtype rule would have made ~200.
        self.assertEqual(mesh.get_array().shape[1], TILE_DEFAULT_BINS)

    def test_segmented_rug_fallback_uses_classify_data_not_dtype(self):
        """A direct make_segmented_rug call classifies with classify_data:
        a wide-support integer variable counts as continuous, so pairing it
        with another continuous variable raises the friendly two-continuous
        error instead of drawing hundreds of bands (the old dtype rule
        would have called the integer axis discrete and drawn them)."""
        rng = np.random.default_rng(0)
        x = rng.integers(0, 200, 2000)  # continuous under classify_data
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

    def test_segmented_rug_two_discrete_raises_friendly_error(self):
        with self.assertRaises(ValueError) as cm:
            self.discrete_sims.plot(type="rug")
        self.assertIn("tile", str(cm.exception))

    def test_marginal_hist_combo_still_draws(self):
        self.sims.plot(type=("marginal", "hist"))
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

    def test_mosaic_xlabel_is_x(self):
        self.discrete_sims.plot(type="mosaic")
        self.assertEqual(plt.gca().get_xlabel(), "X")

    def test_mosaic_yaxis_is_hidden(self):
        """The y-axis has no meaning shared across columns, so it's hidden."""
        self.discrete_sims.plot(type="mosaic")
        self.assertFalse(plt.gca().yaxis.get_visible())

    def test_mosaic_legend_present_with_y_title(self):
        self.discrete_sims.plot(type="mosaic")
        legend = plt.gca().get_legend()
        self.assertIsNotNone(legend)
        self.assertEqual(legend.get_title().get_text(), "Y")

    def test_mosaic_columns_sum_to_full_width(self):
        """Column widths (plus gaps) must span the full [0, 1] x-axis."""
        p = self.discrete_sims.plot(type="mosaic")
        self.assertAlmostEqual(p.ax.get_xlim()[0], 0.0)
        self.assertAlmostEqual(p.ax.get_xlim()[1], 1.0)

    def test_mosaic_normalize_false_labels_are_whole_numbers(self):
        """normalize=False switches in-cell labels from percentages to counts."""
        p = self.discrete_sims.plot(type="mosaic", normalize=False)
        texts = [t.get_text() for t in p.ax.texts]
        self.assertGreater(len(texts), 0)
        for text in texts:
            self.assertNotIn("%", text)

    def test_mosaic_normalize_true_labels_are_percentages(self):
        p = self.discrete_sims.plot(type="mosaic")
        texts = [t.get_text() for t in p.ax.texts]
        self.assertGreater(len(texts), 0)
        self.assertTrue(any("%" in text for text in texts))

    def test_mosaic_annotate_false_has_no_labels(self):
        p = self.discrete_sims.plot(type="mosaic", annotate=False)
        self.assertEqual(len(p.ax.texts), 0)

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


# ===========================================================================
# Default plot lookup wired into RVResults.plot() + suggestion note
# ===========================================================================


class TestDefaultLookupDispatch(PlotTestCase):
    """classify_data + DEFAULT_PLOT_TYPE now drive .plot()'s defaults."""

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

    def test_binomial_has_both_scatter_and_line(self):
        """Discrete distributions are drawn as scatter points + connecting line."""
        Binomial(n=10, p=0.4).plot()
        ax = plt.gca()
        self.assertGreater(len(ax.lines), 0)
        self.assertGreater(len(ax.collections), 0)

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
        """All scatter point y-values (PMF) must be >= 0."""
        Poisson(lam=4).plot()
        for coll in plt.gca().collections:
            offsets = coll.get_offsets()
            if len(offsets):
                self.assertTrue(np.all(offsets[:, 1] >= 0))

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
        p = (X & Y).sim(100).plot(type="marginal")
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
# classify_data: discrete-ish / small-n classification
# ===========================================================================


class TestClassifyData(unittest.TestCase):
    """The two-boolean classifier that replaces is_discrete."""

    def test_narrow_int_large_n_is_discrete(self):
        discrete_ish, small_n = classify_data(np.array([0, 1, 2, 1, 3] * 400))
        self.assertTrue(discrete_ish)
        self.assertFalse(small_n)

    def test_all_unique_float_is_continuous(self):
        discrete_ish, small_n = classify_data(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        self.assertFalse(discrete_ish)
        self.assertTrue(small_n)

    def test_wide_support_int_is_continuous(self):
        """Wide-support integers (e.g. Binomial(10000, 0.5)) read as continuous."""
        discrete_ish, _ = classify_data(np.arange(200))
        self.assertFalse(discrete_ish)

    def test_repeated_float_narrow_support_is_discrete(self):
        """Float data with few repeated values is a user-defined finite support."""
        discrete_ish, small_n = classify_data(np.array([1.0, 1.5, 2.71, 4.0] * 250))
        self.assertTrue(discrete_ish)
        self.assertFalse(small_n)

    def test_string_categorical_is_discrete(self):
        discrete_ish, _ = classify_data(np.array(["H", "T"] * 60))
        self.assertTrue(discrete_ish)

    def test_many_category_strings_still_discrete(self):
        """A wide-support categorical (52 labels) must not read as continuous."""
        discrete_ish, _ = classify_data(np.array([f"c{i % 52}" for i in range(1000)]))
        self.assertTrue(discrete_ish)

    def test_boolean_is_discrete(self):
        discrete_ish, _ = classify_data(np.array([True, False, True, True]))
        self.assertTrue(discrete_ish)

    def test_degenerate_constant_is_discrete(self):
        discrete_ish, _ = classify_data(np.full(500, 5))
        self.assertTrue(discrete_ish)

    def test_small_n_boundary(self):
        """small_n is True below N_SMALL_THRESHOLD and False at/above it."""
        _, small_below = classify_data(np.arange(99))
        _, small_at = classify_data(np.arange(100))
        self.assertTrue(small_below)
        self.assertFalse(small_at)


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
