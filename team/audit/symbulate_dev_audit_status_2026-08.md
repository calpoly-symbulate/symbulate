# Symbulate Dev Audit — Current Status Review

**Review date:** August 15, 2026  
**Source audit:** `team/symbulate_dev_audit_2026-08.md`  
**Scope:** Current working tree; code and targeted regression-test review. No changes were made as part of this review.

## Summary

Of the 25 findings in the August audit, 15 are fixed, 3 are partially fixed, and 7 remain open. The high-severity simulation and API correctness findings have largely been addressed. Remaining work is mostly table rendering, performance, documentation/metadata drift, and a handful of lower-level validation and plotting issues.

## Finding-by-finding status

### 1. Multinomial rejects valid floating-point probabilities — Fixed

`Multinomial` now uses tolerance-based validation, and `NegativeMultinomial` safely handles the probability-sum boundary. Regression tests cover common floating-point examples.

### 2. `RV.draw()` seeding documentation is false — Fixed

The package has a shared generator and public `seed()` function. The documentation correctly says that `np.random.seed()` does not control Symbulate randomness.

### 3. GARCH/ARCH variance assumes unit-variance noise — Fixed

Stationarity and unconditional-variance calculations now include `noise_dist.var()`. Tests cover both stationary and explosive non-unit-variance cases.

### 4. `InfiniteTuple` drops slice step — Fixed

Open-ended slices preserve both starting offset and step; regression tests cover `iv[2::2]` and `iv[::2]`.

### 5. `RandomProcess.__setitem__` silently ignores invalid values — Fixed

Assignments that are neither an `RV` nor a scalar now raise a clear `TypeError`.

### 6. Plot kwargs are handled inconsistently — Partially fixed

The two silent-drop cases are fixed: `rug` and `segmented_rug` now receive `**kwargs`. Most matplotlib `TypeError`/`AttributeError` responses for bad kwargs are re-raised as actionable Symbulate `TypeError`s.

One gap remains: `density2d` can still accept an unrecognized kwarg without raising, because matplotlib's `contourf` emits a warning rather than an exception. The current regression test explicitly preserves this no-crash behavior, so a typo in this plot type can still be silently ineffective.

### 7. GPL/MIT license mismatch — Fixed

`setup.py` now declares MIT and uses the MIT PyPI classifier, matching `LICENSE.txt`.

### 8. Open-ended slices raise raw errors — Fixed

`RV` delegates to native slicing for open-ended slices. `DiscreteTimeFunction` rejects an unbounded slice with a clear `ValueError` explaining why it cannot be materialized.

### 9. `dims=` validation occurs after partial plotting — Fixed

The invalid `dims=` check now runs before the pairs-matrix branch can create axes or alter the figure.

### 10. `RVResults.standardize()` exposes `ZeroDivisionError` — Fixed

Zero-variability data now raises an explanatory error before division.

### 11. Monte Carlo tests and simulations are slow — Still open

This is not a numerical-correctness defect, but it remains an operational concern. A targeted run covering table, diffusion-process, and time-series tests exceeded 60 seconds and had reached only 43% completion when it timed out. That does not identify a specific broken test, but it supports the audit's finding that the simulation-heavy suite needs profiling and an explicit CI timeout policy.

Suggested next work:

- Profile repeated `InfiniteVector`/`Result.__getitem__` evaluation in heavy Monte Carlo paths.
- Establish a CI timeout and mark intentionally slow tests appropriately.
- Identify whether frequently reused paths can be evaluated or cached more efficiently without changing simulation semantics.

### 12. Multivariate correlation emits raw zero-division warnings — Fixed

Zero-variance components now return documented `nan` correlation rows/columns without a raw NumPy warning.

### 13. `Table` display breaks at exactly 19 outcomes — Still open

`Table.__repr__` and `_repr_html_` still truncate on the nineteenth row even when there are no remaining rows. For a 19-row table, the output contains all rows, then a fake `...` row, then duplicates the last row before printing the total.

The condition in `symbulate/table.py` should distinguish “the nineteenth row has been reached” from “there are additional rows to suppress,” e.g. only perform truncation when `len(keys) > 19`. Add text and HTML regression tests for exactly 19 outcomes.

### 14. `BoxModel` validates empty boxes and negative sizes too late — Fixed

Both conditions now raise useful construction-time errors.

### 15. Sphinx API docs omit public modules — Fixed

The API toctree now lists the previously omitted process and time-series modules.

### 16. Legacy setup.py-only packaging — Fixed

The project now has `pyproject.toml`, Python/dependency floors, and a README-backed long description.

### 17. Package metadata points to stale repository and maintainer — Still open

`setup.py` still lists:

- `url="https://github.com/dlsun/symbulate"`
- `author_email="dsun09@calpoly.edu"`

These do not match the audit's stated current repository, `calpoly-symbulate/symbulate`, or its current-team information. Update the package metadata once the intended public repository and maintainer contact are confirmed.

### 18. Duplicate violin alpha literal — Fixed

`results.py` imports and uses `VIOLIN_ALPHA` rather than duplicating `0.5`.

### 19. `make_mosaic` uses inconsistent label names and incorrect docs — Fixed

The helper now uses `xlabel`/`ylabel`, and its documentation agrees with the actual defaults.

### 20. Large committed `team/` scratch material — Still open

The directory remains large: 204 files totaling approximately 53 MB in the current checkout. This is greater than the audit's approximate 49 MB figure and continues to make the student-facing repository materially larger than necessary.

Suggested next work is to move exploratory notebooks to an internal branch/repository or strip notebook outputs before committing them.

### 21. `test_smoke.py` is nearly vacuous — Still open

The file still only imports `symbulate` and checks for `RV` and `BoxModel`. It does not exercise the claimed end-to-end coverage or catch public-API import regressions in other subsystems.

A meaningful smoke test should import the full public API and perform one small representative operation in each major subsystem.

### 22. AR/ARMA `mean` documentation fails for nonzero-mean noise — Fixed

The AR, MA, and ARMA documentation now explains how nonzero-mean shocks shift the process mean, and tests verify the formulas.

### 23. Documentation contradictions in CLAUDE.md/DECISIONS.md — Partially fixed

The `make_density2D` and `make_joint_pdf` documentation now correctly states that `contour=True` is the default.

However, `CLAUDE.md` still contains two confirmed stale references:

- It repeatedly refers to `classify_data()`, while the implemented function is `classify_values()`.
- It states `N_SMALL_THRESHOLD = 100`; `plot.py` uses `123`, consistent with the later `DECISIONS.md` decision.

The audit's overlay-policy contradiction should also be rechecked during a documentation pass, since mosaic plots remain a documented exception to the general two-dimensional overlay rule.

### 24. Minor validation and plotting issues — Partially fixed

The spinner-related style concern is addressed: it now derives its palette from configured matplotlib colors rather than maintaining a separate hardcoded palette.

The following parts remain open:

- `MultivariateHypergeometric(m=[10.0, 8.0, 6.0], n=6)` still rejects whole-number floats.
- `Integers` and `DiscreteTimeSequence` still accept negative values even though their docstrings define non-negative domains.
- `tabulate(bin=True)` on categorical data still raises a raw NumPy `UFuncTypeError` rather than a clear validation error.
- A 2-D plot with marginals still calls `get_next_color(ax)` three times, contrary to the documented single-call color-cycle policy.

### 25. Cosmetic and release-hygiene issues — Still open

The following audit items remain present:

- The multivariate-statistic error text still says “when when.”
- `TruncatedNormal.__init__` still computes `lower` and `upper` without using them.
- No `CHANGELOG` file or git tag was found to connect version `0.5.5` to a release artifact.

## Additional issue outside the audit

The previously reviewed `Table + Table` arithmetic problem remains open. `Table._operation_factory` applies arithmetic as if the right operand were scalar. Adding two tables therefore produces a table whose values are nested `Table` objects rather than raising an explanatory error or performing a documented key-aligned operation. No regression test currently covers this behavior.

