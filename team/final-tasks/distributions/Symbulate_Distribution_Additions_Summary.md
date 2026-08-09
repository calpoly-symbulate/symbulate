# Symbulate: Univariate Distribution Additions

*Summary of changes prepared for the calpoly-symbulate/symbulate `dev` branch*

## Overview

This patch set adds two new univariate distributions and revises two existing ones on the `dev` branch, per the request. All changes follow the conventions already established in `symbulate/distributions.py` (scipy-backed where possible, consolidated parameter validation via the existing `_validate` helper, NumPy-style docstrings with worked examples, and a matching pytest suite in `symbulate/tests/test_distributions.py`). Four separate commits were made, one per distribution, so each can be reviewed or merged independently; a combined patch is also provided.

| Distribution | Status | Change | Commit |
|---|---|---|---|
| Benford's law | New | `Benford(base=10)` discrete distribution, pmf = `log_base(1+1/d)` | `de59738` |
| Inverse Gaussian | New | `InverseGaussian(mean, shape)` wrapping `scipy.stats.invgauss` | `727acc4` |
| Beta | Revised | Added `xmin=0`, `xmax=1` parameters (generalized support) | `a4748a5` |
| Rayleigh | Revised | Added `scale=1.0` parameter | `a68d10a` |

## 1. Benford's Law (new)

Adds `Benford(base=10)`, a discrete distribution over leading digits 1, ..., base-1, with probability mass function `log_base(1 + 1/d)`. This is the classic "first-digit" distribution used in forensic accounting and fraud detection: a leading 1 occurs about 30% of the time and a leading 9 less than 5%, in base 10.

scipy has no built-in Benford's law, so it required a small custom `scipy.stats.rv_discrete` generator (`_benford_gen`), following the same pattern the existing Makeham distribution uses for its custom generator. Because the support is small and fixed (9 values for base 10), scipy's generic `rv_discrete` machinery handles the cdf, quantile function, mean, variance, and sampling automatically from the pmf alone — no custom cdf or sampler was needed.

`Benford(base=10)` — optional `base` parameter generalizes the law to leading digits in any numeral base (`base=2` is a degenerate case where the only possible leading digit is 1).

Validated: probabilities sum to 1, pmf matches the closed-form formula exactly, mean is 3.4402 (the known constant for base 10), simulated draws match the theoretical pmf by chi-square test, and invalid bases (non-integer, less than 2) raise a clear error.

## 2. Inverse Gaussian / Wald (new)

Adds `InverseGaussian(mean=1.0, shape=1.0)`, a continuous distribution on (0, ∞) that is the first-passage-time distribution of a Brownian motion with positive drift. It is a direct wrap of `scipy.stats.invgauss`.

Rather than exposing scipy's own standardized shape parameter, the class is parameterized the way most textbooks present it — by mean and shape (often written IG(μ, λ)) — so that `InverseGaussian(mean=2, shape=3).mean()` really does return 2. The translation to scipy's convention is `mu = mean/shape` and `scale = shape`, derived so that `scale * invgauss(mu)` has the requested mean and a variance of `mean³/shape`.

Validated: `mean()` and `var()` match the requested parameters exactly across several (mean, shape) pairs, pdf values match scipy's own `invgauss` under the translated parameters, and the distribution's variance shrinks toward 0 around a fixed mean as shape grows (its normal-approximation limit). Invalid (non-positive) mean or shape raises a clear error.

## 3. Beta: xmin / xmax parameters (revised)

`Beta(shape1, shape2, xmin=0.0, xmax=1.0)` generalizes the existing Beta distribution, previously fixed to support [0, 1], to any interval [xmin, xmax]. This matches exactly what was requested: if X ~ Beta(shape1, shape2) is the standard beta distribution, then `xmin + (xmax - xmin) * X` now has this generalized distribution.

The implementation reuses scipy's own loc/scale mechanism for the beta distribution (`loc=xmin, scale=xmax-xmin`) — the same stretch-and-shift the existing PERT distribution already relies on internally, so no new machinery was introduced. The default values of `xmin=0` and `xmax=1` exactly reproduce the previous behavior, so all existing calls to `Beta(shape1, shape2)` — including the internal one inside Dirichlet's two-category marginal — are unaffected.

Validated: default bounds are (0, 1), a simulated Beta(a, b) stretched by hand (`xmin + (xmax-xmin)*X`) matches `Beta(a, b, xmin, xmax)` by Kolmogorov–Smirnov test, mean and support shift correctly, and invalid bounds (`xmax <= xmin`, or non-numeric xmin/xmax) raise a clear error. The full existing test suite for Beta, PERT, and Dirichlet (which builds Beta instances internally) still passes unchanged.

## 4. Rayleigh: scale parameter (revised)

`Rayleigh(scale=1.0)` generalizes the existing Rayleigh distribution, previously fixed at scale=1, to any positive scale. A Rayleigh distribution arises as the magnitude of a 2D vector whose independent, mean-zero normal components share a standard deviation — `scale` is that standard deviation. With this change, `scale * Rayleigh()` is now exactly `Rayleigh(scale)`.

The implementation simply passes `scale` through to `scipy.stats.rayleigh`, mirroring how the existing HalfNormal and HalfCauchy classes already expose their own scale parameters. The default of `scale=1.0` preserves the previous behavior exactly.

Validated: default scale is 1.0, simulated draws at scale=2 match scipy's `rayleigh(scale=2)` by Kolmogorov–Smirnov test, mean/variance match scipy's closed forms across several scales, and `scale * Rayleigh()` matches `Rayleigh(scale)` directly. An invalid (non-positive) scale raises a clear error.

## Files changed

- `symbulate/distributions.py` — the four class additions/revisions described above
- `symbulate/__init__.py` — exports `Benford` and `InverseGaussian` (flagged here per the project's convention of calling out `__init__.py` changes explicitly)
- `symbulate/tests/test_distributions.py` — new `TestBenford` and `TestInverseGaussian` classes, plus additional test methods appended to the existing `TestBeta` and `TestRayleigh` classes

## Testing

All new and modified tests pass (90 tests across the four changes), along with the full pre-existing `test_distributions.py` suite (955 tests), confirmed both in the working branch and by re-applying the patch files to a fresh clone of `dev` and re-running the suite there. One pre-existing, unrelated failure was observed on `dev` before any of these changes (`TestMultinomial.test_Multinomial_plot_pairs`, a floating-point rounding edge case in an unrelated distribution) — it is not affected by this work.

Code is formatted with Black, matches the project's docstring and error-message conventions (NumPy-style docstrings with worked doctest examples; validation errors that explain what went wrong), and does not add any new dependencies.

## Delivery

Changes are provided as four separate git patches (one per distribution) plus a combined patch, generated from a branch called `feature/univariate-distribution-additions` off the current tip of `dev` (commit `dce6e10`). Each patch applies cleanly with `git am` on a clean `dev` checkout, in this order:

- `0001-feature-add-Benford-s-law-distribution.patch`
- `0002-feature-add-InverseGaussian-Wald-distribution.patch`
- `0003-feature-add-xmin-xmax-parameters-to-Beta-distributio.patch`
- `0004-feature-add-scale-parameter-to-Rayleigh-distribution.patch`

A combined patch (`0000-combined-all-four-distributions.patch`) applies all four at once with `git apply` or `git am`, for convenience if a single merge is preferred over four.

*To apply: from a clean checkout of dev, run `git am <patch files, in order>`, or `git apply 0000-combined-all-four-distributions.patch` for the single-file version, then run `pytest symbulate/tests/` to confirm.*
