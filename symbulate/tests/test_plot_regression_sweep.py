"""Layer 1 + Layer 2 of the graphics testing strategy.

See ``team/final-tasks/graphics/symbulate_graphics_testing_strategy.md`` for
the full design writeup. `test_plot.py` asserts *specific, known* properties
of specific calls ("Normal(0,1).plot() draws exactly one line"); it can't
scale to "does .plot() look reasonable across every distribution" and can't
catch a failure mode nobody has written an assertion for yet. This file adds
a second, complementary layer that doesn't require knowing in advance what
could go wrong:

- Layer 1 -- ``axes_is_blank()``: a generic "did anything actually render"
  check that inspects rendered pixels rather than artist counts or
  ``ax.has_data()``, both of which report a plot as non-blank even when it's
  visually empty (``ax.plot([], [])`` has ``has_data() == True``).
- Layer 2 -- a table-driven sweep, over every univariate ``Distribution``
  subclass found via introspection, asserting ``.plot()`` and
  ``.plot(cdf=True)`` never raise, never warn, and never render blank. A
  build-failing test keeps a newly added distribution from silently escaping
  the sweep, the same contract ``test_continuous_time_processes.py`` already
  enforces for continuous-time processes.
"""

import unittest
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt

from symbulate.distributions import (
    Distribution,
    Bernoulli,
    Binomial,
    BetaBinomial,
    BetaNegativeBinomial,
    Hypergeometric,
    NegativeHypergeometric,
    Geometric,
    NegativeBinomial,
    Pascal,
    Poisson,
    DiscreteUniform,
    Zipf,
    Zeta,
    Benford,
    Uniform,
    IrwinHall,
    Bates,
    LogUniform,
    Normal,
    TruncatedNormal,
    SkewNormal,
    Exponential,
    ExponentiallyModifiedGaussian,
    Gamma,
    InverseGamma,
    LogGamma,
    InverseGaussian,
    Beta,
    PERT,
    Triangular,
    Kumaraswamy,
    StudentT,
    SkewT,
    ChiSquare,
    F,
    Hotelling,
    Cauchy,
    LogNormal,
    Pareto,
    Burr,
    Lomax,
    Rayleigh,
    HalfNormal,
    HalfCauchy,
    Weibull,
    Logistic,
    Gompertz,
    Makeham,
    Laplace,
    DeMoivre,
    GEV,
    GPD,
)


# ===========================================================================
# Layer 1: axes_is_blank
# ===========================================================================


def axes_is_blank(ax, shrink=0.0, pixel_tol=8, area_tol=0.0003):
    """Render the figure and check whether anything is visible inside `ax`.

    Looks at rendered pixels rather than artist objects, so it catches a
    blank plot regardless of *why* it's blank (empty arrays, all-NaN data,
    zero-height bars, alpha=0, size=0 markers, ...) without needing a
    separate check per failure mode. Both ``ax.has_data()`` and counting
    artists are unreliable here: ``ax.plot([], [])`` reports
    ``has_data() == True`` despite drawing nothing.

    Unlike the version of this check first proposed (diff the rendered
    region against the figure's flat background color), this compares two
    renders of the *same* axes -- once as-is, once with every data artist
    (``ax.lines``, ``ax.collections``, ``ax.patches``, ``ax.images``)
    temporarily hidden -- and looks at what changed. A flat-background diff
    works against a bare, default-styled matplotlib axes, but Symbulate
    applies ``symbulate.mplstyle`` on import, which turns on horizontal
    gridlines (``axes.grid: True``, ``axes.grid.axis: y``); those gridlines
    are not the figure's background color, so a background-only diff reports
    every gridded-but-otherwise-empty axes as non-blank, and it is *less*
    sensitive than it looks for a sparse real plot, since a single small
    marker can fall below the gridlines' own pixel footprint. Confirmed by
    running the original background-diff version against real Symbulate
    plots with ``symbulate.mplstyle`` active: it reported a literally empty
    axes as non-blank. Diffing "with data" against "same axes, data hidden"
    isolates exactly the ink the plot call added, independent of
    grid/spine/tick styling.

    Because spines, ticks, and gridlines are identical in both renders (they
    are never in ``ax.lines``/``ax.collections``/``ax.patches``/``ax.images``),
    they cancel out of the diff automatically -- so, unlike the
    background-diff version, this one does not need a wide edge margin to
    avoid mistaking them for content. `shrink` defaults to 0 for exactly
    that reason. A nonzero default actively hurts real Symbulate plots: a
    perfectly flat pdf (any `Uniform`-family distribution) sits within
    ``1 - 1/1.05 ≈ 4.8%`` of the y-axis top under the plotting window's own
    ``1.05 * ymax`` padding, and a sharply peaked one (e.g. `HalfCauchy`'s
    pdf, or its cdf's near-vertical rise off x=0) puts nearly all of its
    visible ink within the first percent of an axis. A `shrink` of 0.06 (the
    value first proposed here) clips both entirely and reports them blank;
    this was caught by running the sweep below, not reasoned about in
    advance -- see ``TestAxesIsBlank`` for the specific real cases that
    forced ``shrink`` down to 0 and ``area_tol`` down to 0.0003.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to check.
    shrink : float
        Fraction of the axes' rendered width/height to trim from each edge
        before comparing pixels. Kept as a parameter (e.g. for a caller that
        adds its own chrome directly via ``ax.lines`` and wants it ignored),
        but 0 is correct for anything drawn the normal way -- see above.
    pixel_tol : int
        Per-channel (0-255) intensity difference above which a pixel counts
        as changed between the two renders.
    area_tol : float
        Fraction of the (shrunk) region that must differ for the axes to be
        considered non-blank.

    Returns
    -------
    bool
        True if the axes renders as blank (no visible content beyond
        gridlines/spines/ticks), False otherwise.

    Notes
    -----
    This is tuned against matplotlib's default figure size (6.4in x 4.8in
    at 100 dpi), which is also what ``symbulate.mplstyle`` currently sets
    (``figure.figsize: 6.4, 4.8``, no custom dpi) -- so no recalibration is
    needed today. If ``symbulate.mplstyle``'s figure size or dpi is ever
    changed, re-run the calibration in ``TestAxesIsBlank`` (and the full
    sweep below it) against the new defaults before trusting
    ``pixel_tol``/``area_tol`` again: a smaller figure packs the same marker
    or curve into fewer pixels, which lowers the area fraction a real (but
    small or sharply peaked) plot occupies.
    """
    fig = ax.figure

    def render():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        buf = np.asarray(renderer.buffer_rgba())[..., :3].astype(int)

        bbox = ax.get_window_extent(renderer=renderer)
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
        dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink
        x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy

        h = buf.shape[0]
        row0, row1 = int(h - y1), int(h - y0)
        col0, col1 = int(x0), int(x1)
        return buf[max(row0, 0) : row1, max(col0, 0) : col1, :]

    data_artists = (
        list(ax.lines) + list(ax.collections) + list(ax.patches) + list(ax.images)
    )

    with_content = render()
    if with_content.size == 0:
        return True

    previously_visible = [artist.get_visible() for artist in data_artists]
    for artist in data_artists:
        artist.set_visible(False)
    without_content = render()
    for artist, was_visible in zip(data_artists, previously_visible):
        artist.set_visible(was_visible)
    fig.canvas.draw()

    diff = np.abs(with_content - without_content).max(axis=-1)
    return float((diff > pixel_tol).mean()) < area_tol


# ===========================================================================
# Layer 1 validation: the strategy doc's own table of synthetic cases
# ===========================================================================


class TestAxesIsBlank(unittest.TestCase):
    """Calibrates axes_is_blank against known blank/non-blank renders.

    Run with symbulate's own mplstyle active (imported above), not bare
    matplotlib defaults, since that's the style every real call in the
    Layer 2 sweep below actually renders under.
    """

    def tearDown(self):
        plt.close("all")

    def test_truly_empty_axes_is_blank(self):
        fig, ax = plt.subplots()
        self.assertTrue(axes_is_blank(ax))

    def test_plot_of_empty_arrays_is_blank(self):
        fig, ax = plt.subplots()
        ax.plot([], [])
        self.assertTrue(ax.has_data())  # the unreliable check says "has data"
        self.assertTrue(axes_is_blank(ax))  # but nothing is visible

    def test_all_nan_line_is_blank(self):
        fig, ax = plt.subplots()
        ax.plot([np.nan] * 5, [np.nan] * 5)
        self.assertTrue(axes_is_blank(ax))

    def test_bar_of_empty_arrays_is_blank(self):
        fig, ax = plt.subplots()
        ax.bar([], [])
        self.assertTrue(axes_is_blank(ax))

    def test_hist_of_empty_array_is_blank(self):
        fig, ax = plt.subplots()
        ax.hist([])  # ten zero-height bars
        self.assertTrue(axes_is_blank(ax))

    def test_real_line_plot_is_not_blank(self):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        self.assertFalse(axes_is_blank(ax))

    def test_real_symbulate_pdf_plot_is_not_blank(self):
        Normal(0, 1).plot()
        self.assertFalse(axes_is_blank(plt.gca()))

    def test_sparse_real_scatter_is_not_blank(self):
        # The sparsest real case: a single point at default marker size
        # (rather than an artificially tiny one), the smallest thing
        # Symbulate itself would ever actually draw.
        fig, ax = plt.subplots()
        ax.scatter([0.5], [0.5])
        self.assertFalse(axes_is_blank(ax))

    # The three cases below are what actually forced shrink down to 0 and
    # area_tol down to 0.0003 -- each was a real, initially-undetected false
    # positive found by running this check against the Layer 2 sweep, not
    # anticipated ahead of time. They stay here as regressions so a future
    # recalibration (e.g. after a symbulate.mplstyle figure-size change)
    # can't silently reintroduce them.

    def test_flat_pdf_near_the_padded_ylim_top_is_not_blank(self):
        # DeMoivre is a reparameterized Uniform: its pdf is a single flat
        # line at 1/omega across the whole window. The plotting window pads
        # the y-axis to 1.05 * ymax, so that flat line sits within ~4.8% of
        # the top -- inside a shrink=0.06 margin, entirely invisible to it.
        DeMoivre(omega=10).plot()
        self.assertFalse(axes_is_blank(plt.gca()))

    def test_sharply_peaked_pdf_near_the_origin_is_not_blank(self):
        # HalfCauchy's pdf peaks at x=0 and falls away within the first
        # percent or two of its plotting window; a shrink margin of even
        # 0.01 clips enough of that peak to drop the visible remainder
        # below a 0.0005 area threshold.
        HalfCauchy(scale=1.0).plot()
        self.assertFalse(axes_is_blank(plt.gca()))

    def test_cdf_plateauing_near_its_asymptote_is_not_blank(self):
        # HalfCauchy's cdf rises almost vertically off x=0 and then
        # plateaus near 1 (its heavy tail pushes the window's quantile-based
        # upper x-bound far out); most of the visible curve sits within a
        # percent of the top of the y-axis for nearly the whole window.
        HalfCauchy(scale=1.0).plot(cdf=True)
        self.assertFalse(axes_is_blank(plt.gca()))


# ===========================================================================
# Layer 2: table-driven sweep over every univariate Distribution subclass
# ===========================================================================

# One entry per distribution: minimal kwargs that construct a valid,
# non-degenerate instance. Values are chosen to be unremarkable (no
# boundary/degenerate parameters) -- boundary-fuzzing toward degenerate
# parameters is future work, per the strategy doc's Layer 2 section, not
# what this sweep covers.
REGISTRY = {
    "Bernoulli": (Bernoulli, dict(p=0.4)),
    "Binomial": (Binomial, dict(n=10, p=0.4)),
    "BetaBinomial": (BetaBinomial, dict(n=10, shape1=2, shape2=3)),
    "BetaNegativeBinomial": (BetaNegativeBinomial, dict(r=5, shape1=2, shape2=3)),
    "Hypergeometric": (Hypergeometric, dict(n=5, N0=10, N1=10)),
    "NegativeHypergeometric": (NegativeHypergeometric, dict(r=3, N0=10, N1=10)),
    "Geometric": (Geometric, dict(p=0.3)),
    "NegativeBinomial": (NegativeBinomial, dict(r=5, p=0.4)),
    "Pascal": (Pascal, dict(r=5, p=0.4)),
    "Poisson": (Poisson, dict(lam=3)),
    "DiscreteUniform": (DiscreteUniform, dict(a=0, b=10)),
    "Zipf": (Zipf, dict(shape=2, n=10)),
    "Zeta": (Zeta, dict(shape=3)),
    "Benford": (Benford, dict(base=10)),
    "Uniform": (Uniform, dict(a=0.0, b=1.0)),
    "IrwinHall": (IrwinHall, dict(n=3, a=0.0, b=1.0)),
    "Bates": (Bates, dict(n=3, a=0.0, b=1.0)),
    "LogUniform": (LogUniform, dict(a=1.0, b=10.0)),
    "Normal": (Normal, dict(mean=0.0, sd=1.0)),
    "TruncatedNormal": (TruncatedNormal, dict(mean=0.0, sd=1.0, a=-2, b=2)),
    "SkewNormal": (SkewNormal, dict(loc=0, scale=1, shape=2)),
    "Exponential": (Exponential, dict(rate=1.0)),
    "ExponentiallyModifiedGaussian": (
        ExponentiallyModifiedGaussian,
        dict(mean=0.0, sd=1.0, rate=1.0),
    ),
    "Gamma": (Gamma, dict(shape=2, rate=1.0)),
    "InverseGamma": (InverseGamma, dict(shape=3, scale=1.0)),
    "LogGamma": (LogGamma, dict(shape=2, loc=0, scale=1)),
    "InverseGaussian": (InverseGaussian, dict(mean=1.0, shape=1.0)),
    "Beta": (Beta, dict(shape1=2, shape2=3)),
    "PERT": (PERT, dict(low=0, mode=5, high=10)),
    "Triangular": (Triangular, dict(low=0, mode=5, high=10)),
    "Kumaraswamy": (Kumaraswamy, dict(shape1=2, shape2=3)),
    "StudentT": (StudentT, dict(df=5)),
    "SkewT": (SkewT, dict(shape1=2, shape2=3)),
    "ChiSquare": (ChiSquare, dict(df=5)),
    "F": (F, dict(dfN=5, dfD=10)),
    "Hotelling": (Hotelling, dict(dim=2, df=5)),
    "Cauchy": (Cauchy, dict(loc=0, scale=1)),
    "LogNormal": (LogNormal, dict(mu=0.0, sigma=1.0)),
    "Pareto": (Pareto, dict(shape=2.0, scale=1.0)),
    "Burr": (Burr, dict(shape1=2, shape2=3, scale=1.0)),
    "Lomax": (Lomax, dict(shape=2.0, scale=1.0)),
    "Rayleigh": (Rayleigh, dict(scale=1.0)),
    "HalfNormal": (HalfNormal, dict(scale=1.0)),
    "HalfCauchy": (HalfCauchy, dict(scale=1.0)),
    "Weibull": (Weibull, dict(shape=2, scale=1.0)),
    "Logistic": (Logistic, dict(loc=0, scale=1)),
    "Gompertz": (Gompertz, dict(shape=1, scale=1.0)),
    "Makeham": (Makeham, dict(shape=1, makeham=0.5, scale=1.0)),
    "Laplace": (Laplace, dict(loc=0, scale=1)),
    "DeMoivre": (DeMoivre, dict(omega=10)),
    "GEV": (GEV, dict(loc=0, scale=1, shape=0.2)),
    "GPD": (GPD, dict(loc=0, scale=1, shape=0.2)),
}

# Distributions deliberately excluded from the sweep, each with a reason.
# Every entry here was checked directly (not assumed) before being skipped:
SKIPPED = {
    # Both raise an explicit, intentional "not currently available" error
    # from .plot() itself -- there is no minimal-kwargs instance of either
    # that would exercise the pdf/cdf-rendering path this sweep checks.
    "Wishart": "plot() is not implemented for this distribution "
    "(raises 'Plotting is not currently available for the Wishart "
    "distribution.' by design); nothing here to sweep.",
    "InverseWishart": "plot() is not implemented for this distribution "
    "(raises 'Plotting is not currently available for the inverse-Wishart "
    "distribution.' by design); nothing here to sweep.",
    # An abstract base class, not a concrete distribution: its __init__
    # takes a raw scipy multivariate object and there is no generic minimal
    # instance to construct (Distribution.mean()/plot() dispatch assumes a
    # concrete subclass's own scipy wrapper conventions). Its .plot() also
    # has a different signature/shape (a joint plot of two or more
    # components) than the single-variable pdf/pmf/cdf every REGISTRY entry
    # is checked against here, so it is out of scope for this sweep even
    # setting construction aside. Concrete multivariate distributions that
    # matter for this sweep (Hotelling, whose plot() *is* a plain 1-D pdf of
    # a scalar statistic) are already registered directly above.
    "MultivariateDistribution": "abstract base class with no generic minimal "
    "constructor, and a fundamentally different (joint, multi-variable) "
    "plot() shape than the scalar pdf/pmf/cdf sweep below checks.",
}


class TestRegistryCoverage(unittest.TestCase):
    """Fails the build if a Distribution subclass is in neither table.

    Mirrors the contract test_continuous_time_processes.py already enforces
    for continuous-time processes: a newly added Distribution subclass must
    be explicitly registered or explicitly skipped-with-reason, so it can't
    silently escape this sweep.
    """

    def test_registry_covers_every_univariate_distribution(self):
        all_subclasses = {cls.__name__ for cls in Distribution.__subclasses__()}
        covered = set(REGISTRY) | set(SKIPPED)
        missing = all_subclasses - covered
        self.assertEqual(
            missing,
            set(),
            f"{sorted(missing)} are Distribution subclasses not covered by "
            f"REGISTRY or SKIPPED in test_plot_regression_sweep.py. Add a "
            f"minimal-kwargs entry to REGISTRY, or a reason to SKIPPED.",
        )

    def test_registry_and_skipped_do_not_overlap(self):
        self.assertEqual(set(REGISTRY) & set(SKIPPED), set())


class TestDistributionPlotSweep(unittest.TestCase):
    """Every registered distribution's .plot() and .plot(cdf=True) must not
    raise, warn, or render a blank axes."""

    def tearDown(self):
        plt.close("all")

    def _check(self, name, cdf):
        cls, kwargs = REGISTRY[name]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dist = cls(**kwargs)
            dist.plot(cdf=cdf)
        ax = plt.gca()
        mode = "plot(cdf=True)" if cdf else "plot()"
        self.assertFalse(axes_is_blank(ax), f"{name}.{mode} rendered a blank axes")
        plt.close("all")

    def test_plot(self):
        for name in sorted(REGISTRY):
            with self.subTest(distribution=name):
                self._check(name, cdf=False)

    def test_cdf_plot(self):
        for name in sorted(REGISTRY):
            with self.subTest(distribution=name):
                self._check(name, cdf=True)


if __name__ == "__main__":
    unittest.main()
