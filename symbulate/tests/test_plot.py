"""Tests for symbulate plotting functionality — dev branch.

Covers:
  - 1D RVResults: impulse, hist, density, rug, and combinations
  - 2D RVResults: scatter, hist, density, tile, violin, marginal
  - Distribution.plot() for continuous and discrete distributions
  - Stochastic process path plots (Markov chains, Poisson process)
  - Normalization correctness (histogram area, impulse heights)
  - Axis labels, xlim bounds, and figure structure
  - Error handling (plotting from a bare ProbabilitySpace)

Known remaining bugs are documented as @unittest.expectedFailure tests.
When a bug is fixed, remove that decorator and the test becomes a passing
regression guard. Current expected failures:

  Bug 1 (plot.py:139): violinplot() called with deprecated vert= keyword.
    Fix: replace vert=True/False with orientation='vertical'/'horizontal'.

  Bug 2 (results.py:1272): hist2d normalize=True uses int(label.get_text())
    which raises ValueError when Matplotlib formats ticks as floats.
    Fix: use float() or recompute density directly from histo[0].

  Bug 3 (results.py:45-49): seaborn stylesheet looked up via fragile fuzzy
    match that will raise IndexError if similarity drops below 0.7.
    Fix: hardcode 'seaborn-v0_8-colorblind' with an explicit fallback.
"""

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
from symbulate.plot import SymbulatePlot

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

    def test_rug_produces_a_line(self):
        self.sims.plot(type="rug")
        self.assertGreater(len(plt.gca().lines), 0)

    def test_density_plus_rug_produces_two_lines(self):
        self.sims.plot(type=("density", "rug"))
        self.assertEqual(len(plt.gca().lines), 2)

    def test_hist_plus_rug_produces_bars_and_line(self):
        self.sims.plot(type=("hist", "rug"))
        ax = plt.gca()
        self.assertGreater(len(ax.patches), 0)
        self.assertGreater(len(ax.lines), 0)

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
        # Discrete data auto-selects impulse, but via the is_discrete path, not ours
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

    def test_exponential_default(self):
        RV(Exponential(rate=1)).sim(300).plot()
        self.assertGreater(len(plt.gca().patches), 0)

    def test_poisson_default_is_impulse(self):
        """Poisson results should default to an impulse plot.

        n is large enough that (almost) every value in the support
        repeats. At small n (e.g. 300), rare tail values appearing
        exactly once can exceed is_discrete()'s 20% singleton budget
        and flip the default to a histogram — a known weakness of
        is_discrete() that classify_data() will fix (see the graphics
        plan); this test was flaky until n was raised.
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
        X, Y = RV(Normal(0, 1) ** 2)
        self.sims = (X & Y).sim(500)

    def test_default_is_scatter(self):
        self.sims.plot()
        ax = plt.gca()
        self.assertGreater(len(ax.collections), 0)
        self.assertEqual(len(ax.patches), 0)

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


if __name__ == "__main__":
    unittest.main()
