# Symbulate Audit Fixes — Implementation Guide

Patches for audit findings 1–10, 12, 14, 16, 18, and 22 (15 findings total).
Findings 2 and 3 were already delivered earlier as standalone patches
(`symbulate_random_seed_fix.patch`, `symbulate_garch_variance_fix.patch`) —
they're included in the "all patches" list below for completeness, but live
one level up, in `output/` rather than `output/batch_patches/`.

Every patch was built independently against the same baseline commit
(`dev` at `6e84a1c`), verified to apply cleanly on its own, and — where the
fix changes behavior rather than just docs/metadata — comes with passing
regression tests. None of this has been run through the *full* test suite
yet; that's the planned next step, not done here.

## Is it one PR per patch?

Yes, by default — CLAUDE.md already states this project's policy explicitly:
**"One PR per task."** Each finding below is scoped to be exactly one PR's
worth of change: one bug, one fix, one regression test, one reviewable diff.

**One real exception, found by testing it, not guessing:** findings **6**
and **9** both edit `symbulate/tests/test_results.py`'s import block (both
independently add `import matplotlib; matplotlib.use("Agg"); import
matplotlib.pyplot as plt`, since both needed it and were built independently
against the same pristine baseline). I confirmed this by applying every
patch to one clone in sequence — 13 of 15 applied with zero conflicts, and
finding 9's patch was the one exception, failing with:

```
error: patch failed: symbulate/tests/test_results.py:19
error: symbulate/tests/test_results.py: patch does not apply
```

This is a shallow, mechanical conflict (duplicate import lines), not a
logical one — the two fixes don't touch the same code, just the same import
block. Still, it means: **merge finding 6 before finding 9**, and when
opening finding 9's PR, do it against a branch that already has finding 6's
change (or just let Claude Code regenerate the diff fresh against current
`dev` — see "How to use the prompts" below, this is exactly the scenario
that approach is robust to and blind `git apply` is not).

Three other findings (**10**, **18**) also touch `results.py` /
`test_results.py`, but in different, non-adjacent sections (the
`standardize()` method and the `legacy_alpha` line respectively) — these
applied without conflict in every order I tried. No special handling needed
for those beyond normal PR review.

Everything else is a fully independent PR, mergeable in any order.

## How to use the prompts

For each finding below, the **patch file** is a precise, verified reference
for what the change should look like — open it to see the exact diff. The
**Claude Code prompt** describes the bug and the fix in words rather than
saying "apply this patch." That's deliberate: a prompt that re-derives the
diff against whatever `dev` actually looks like at PR time is robust to
merge order and to any drift since these patches were generated (like the
finding 6/9 conflict above), whereas `git apply` on a stale patch just fails
outright the moment the file has moved on. Feed Claude Code both — the
patch as "here's a reference implementation, match this" context, and the
prompt as the actual instruction.

A minimal way to hand both over in one shot:

```
Read the attached patch file for context on the intended fix, but don't
apply it directly with git apply -- implement the equivalent change against
the current state of dev, adjusting for anything that's moved since the
patch was generated. <prompt text from below>
```

---

## Finding 1 — Multinomial/NegativeMultinomial reject valid probability vectors

**Patch:** `finding01_multinomial_float_sum.patch`
**Severity:** Critical — this is a currently-failing test on `dev`, not a hypothetical.

**Claude Code prompt:**
> In `symbulate/distributions.py`, `Multinomial.__init__` rejects any `p` where `sum(p) == 1` isn't exactly true in floating point — which means `Multinomial(n=12, p=[0.4, 0.3, 0.2, 0.1])` and every fair-die-style equal-weight case gets rejected, since `sum([1/6]*6) != 1.0` in float arithmetic. This is currently failing `test_Multinomial_plot_pairs` in the test suite. Fix it by using `math.isclose(sum(p), 1, abs_tol=1e-9)` instead of exact equality (`math` is already imported in this file). Apply the same fix to `NegativeMultinomial.__init__`'s analogous check a few hundred lines down — it currently uses `p_arr.sum() >= 1` to reject an invalid `p` (valid `p` for this distribution must sum to *strictly less than* 1, since `p0 = 1 - sum(p)` is the probability of the stopping category and must be positive), which has the same floating-point fragility in the opposite direction: change it to treat anything within `abs_tol=1e-9` of 1, or above, as invalid (`p_sum > 1 or math.isclose(p_sum, 1, abs_tol=1e-9)`). Add regression tests for both: `Multinomial(n=12, p=[0.4, 0.3, 0.2, 0.1])` and `Multinomial(n=6, p=[1/6]*6)` should now construct successfully; `Multinomial(n=5, p=[0.5, 0.6])` should still correctly raise; `NegativeMultinomial(r=3, p=[1/3, 1/3, 1/3])` (sum ≈ 1) should still correctly raise, while `NegativeMultinomial(r=3, p=[0.3, 0.2])` should still work.

---

## Finding 4 — `InfiniteTuple.__getitem__` silently drops the step on open-ended slices

**Patch:** `finding04_infinitetuple_step.patch`
**Severity:** High — silently wrong numbers, not a crash.

**Claude Code prompt:**
> In `symbulate/result.py`, `InfiniteTuple.__getitem__`'s slice handling has two related bugs in its open-ended-slice branch (`n.stop is None`): `iv[2::2]` silently ignores the step and behaves like `iv[2:]` (returns every element instead of every other one), and `iv[::2]` (no start either) returns `self` unchanged, dropping the step just as silently. Fix both at once: compute `start = n.start or 0` and `step = n.step or 1`; if `start == 0 and step == 1` return `self` (preserves the fast identity path for a bare `iv[:]`); otherwise return `type(self)(lambda i: self[start + i * step])`. Add regression tests: `iv[2::2]` should give `2, 4, 6, 8, ...`; `iv[::2]` should give `0, 2, 4, 6, ...`; `iv[:]` should still be the identical object (`assertIs`); `iv[2:]` (no step) should be unchanged from current behavior. Note there's a related but separate bug in `RV.__getitem__`/`DiscreteTimeFunction.__getitem__` (see finding 8) — don't conflate the two fixes.

---

## Finding 5 — `RandomProcess.__setitem__` is a silent no-op for bad values

**Patch:** `finding05_randomprocess_setitem.patch`
**Severity:** High — violates CLAUDE.md's own "never silently do nothing" rule.

**Claude Code prompt:**
> In `symbulate/random_processes.py`, `RandomProcess.__setitem__` has two branches — `isinstance(value, RV)` and `elif is_scalar(value)` — and no `else`, so assigning anything else (e.g. `X[0] = [1, 2, 3]`) silently does nothing: no error, `X.rvs` stays empty, the assignment is just dropped. Add an `else: raise TypeError(...)` branch naming what's accepted (an `RV` or a scalar constant) and showing the value and its type that was rejected. This is the *only* `__setitem__` defined anywhere in the codebase, so the fix automatically covers every process subclass (`BrownianMotion`, `PoissonProcess`, `AR`, etc. — none of them override it). There's an existing test, `test_non_scalar_non_rv_is_ignored`, that currently asserts the broken behavior (`X.rvs` stays `{}` with no error) — update it to assert the new `TypeError` is raised instead (rename it to something like `test_non_scalar_non_rv_raises_typeerror`), and add a test confirming the error message names both accepted types.

---

## Finding 6 — Inconsistent handling of bad plot kwargs

**Patch:** `finding06_plot_kwargs.patch`
**Severity:** High — visible the first time a student makes a typo, and it's inconsistent with itself (silent for some plot types, a raw traceback for others).

**Claude Code prompt:**
> In `symbulate/results.py`, `RVResults.plot()`'s big `type=` dispatch has two related problems. First, the concrete gap named in the audit: the two calls to `make_rug` (1D branch, ~line 2298 before this fix) and `make_segmented_rug` (2D branch, ~line 2505) never forward `**kwargs`, unlike every sibling branch (`make_hist`, `make_bar`, `make_impulse`, `make_boxplot`, `make_violinplot`, `make_ecdf`, `make_dotplot` all pass `**kwargs` through) — so `RV(Normal(0,1)).sim(50).plot(type='rug', linewidth=10)` silently ignores `linewidth`. Fix both call sites to forward `**kwargs`. Second, and separately: a bad kwarg on `type='hist'` currently raises a raw `AttributeError: Rectangle.set() got an unexpected keyword argument 'X'` straight from matplotlib — CLAUDE.md's error-message standard says a student should never see a raw exception like this. I verified empirically that this is *not* a single, consistent exception type: `ax.hist`/`ax.vlines`/`ax.scatter`/`ax.hist2d` (anything that forwards leftover kwargs to a matplotlib Artist's `.set()`) raises `AttributeError`, while `ax.boxplot`/`ax.violinplot` (which validate their own signature directly) raise `TypeError` for the identical kind of mistake — so a fix that only catches `TypeError` will miss most of these. Add a helper function `_call_plot_helper(func, *args, **kwargs)` near the top of `results.py` that calls `func(*args, **kwargs)` inside `try/except (TypeError, AttributeError) as e:`, and if `"unexpected keyword argument" in str(e)`, re-raises as `TypeError(f"plot() got a keyword argument that isn't valid for this plot type ({e}). Check the spelling -- some keyword arguments only apply to certain plot types.") from None` — otherwise re-raises the original exception unchanged (so a genuine bug elsewhere in a `make_*` helper doesn't get mislabeled as a kwarg problem). Route every `make_*` call in the 1D branch of `.plot()` through this helper (there are about 9: dotplot, density, hist, bar, impulse, boxplot, violinplot, rug, ecdf) — this is the reference patch's exact scope, already implemented and tested. As a natural follow-up in the same PR (not done in the reference patch, since it's a bigger, higher-risk mechanical sweep across ~18 more call sites in the 2D and categorical branches — scatter, hist2d, density2d, tile, mosaic, segmented_hist, segmented_density, segmented_rug, and the categorical bar/dotplot/impulse trio), apply the exact same `_call_plot_helper` wrapping there too, verifying each one still passes its existing tests as you go. Add regression tests: `type='rug', linewidth=10` should actually apply the linewidth; a bad kwarg on `type='rug'`, `type='hist'`, and `type='box'` should all raise the same friendly `TypeError` (covering both the `AttributeError`-native and `TypeError`-native underlying paths); normal usage of every plot type should be unaffected.

---

## Finding 7 — License mismatch: `setup.py` says GPLv3, `LICENSE.txt` is MIT

**Patch:** `finding07_license_mismatch.patch`
**Severity:** High — legally contradictory, cheap to fix.

**Claude Code prompt:**
> `setup.py` declares `license="GPLv3"` and a matching PyPI classifier (`"License :: OSI Approved :: GNU General Public License v3 (GPLv3)"`), but `LICENSE.txt` is the full text of the MIT License. Confirm `LICENSE.txt`'s content (it should be the standard MIT template, copyright Dennis Sun and Kevin Ross), then change `setup.py`'s `license=` field to `"MIT"` and the classifier to `"License :: OSI Approved :: MIT License"`. No code or test changes needed — this is metadata-only. Don't touch the stale `url=`/`author_email=` fields in the same file (pointing at `dlsun/symbulate` instead of the actual `calpoly-symbulate/symbulate` repo) — that's a separate, unrelated finding (audit finding 17, not in this batch).

---

## Finding 8 — Open-ended slices crash with a raw `TypeError`

**Patch:** `finding08_open_slice_crash.patch`
**Severity:** Medium-High — routine, discoverable-by-accident usage (`X[1:]`), not an edge case.

**Claude Code prompt:**
> Two separate `__getitem__` methods build `range(n.start or 0, n.stop, n.step or 1)` for slice indexing, which crashes with `TypeError: 'NoneType' object cannot be interpreted as an integer` the moment `n.stop` is `None` (i.e. any open-ended slice like `X[1:]`). Fix each differently, since they have different relationships to "infinite": (1) In `symbulate/random_variables.py`, `RV.__getitem__`'s slice branch currently does `self.apply(lambda x: Vector(x[i] for i in range(n.start or 0, n.stop, n.step or 1)))`. Since `x` (the RV's own realized value, e.g. a tuple from a fixed-size `BoxModel` draw) already supports native slicing correctly including open-ended slices, just delegate to it directly: `return self.apply(lambda x: Vector(x[n]))`. This is simpler, fixes the crash, and gives identical results to the old range-based version for closed slices. (2) In `symbulate/result.py`, `DiscreteTimeFunction.__getitem__`'s slice branch has no underlying container to delegate to — each value comes from a function over an index set unbounded in both directions (negative indices are valid too), so there's no length to infer a stop from. There, add an explicit check: if `n.stop is None`, raise `ValueError("Cannot slice a DiscreteTimeFunction without a stop index (e.g. f[2:10]) -- it has no end to slice up to. Index individual values with f(t) or f[n], or give an explicit stop.")` before the `range(...)` call. Add regression tests: `RV(BoxModel([0,1,2,3,4], size=5))[1:].draw()` should return 4 elements matching what native slicing on the same underlying draw would give (compare against the same outcome via `prob_space.draw()`, not two independent random draws); `DiscreteTimeFunction(lambda n: n*3, fs=1)[2:]` should raise the new `ValueError`, while `f[0:4]` (closed slice) should be unaffected.

---

## Finding 9 — `dims=` validated after the pairs-matrix has already partially drawn

**Patch:** `finding09_dims_validation_order.patch`
**Severity:** Medium-High — visible the first time a student makes a typo; **merge after finding 6** (see "Is it one PR per patch?" above — both touch the same test file's import block).

**Claude Code prompt:**
> In `symbulate/results.py`, `RVResults.plot()` checks `if "dims" in kwargs: raise ValueError(...)` *after* the `self.dim is not None and self.dim > 2` branch, but that branch can return early (the `"pairs" in type` case) — building a `GridSpec`, resizing the figure, and adding a subplot — before the `dims=` check ever runs for that path. So `(X**3).sim(50).plot(dims=[0, 1])` raises correctly in the end, but only after leaving a half-built figure behind, from a re-entrant inner call rather than the intended check. Move the `if "dims" in kwargs: raise ValueError(...)` block up to right after the existing `if "pairs" in kwargs: raise ValueError(...)` check, before the `if self.dim is not None and self.dim > 2:` branch begins, and delete the old, now-dead copy of the check further down. Add a regression test: call `.plot(dims=[0, 1])` on a 3-variable `RVResults`, confirm it raises `ValueError` mentioning `dims=`, and confirm `plt.gcf().get_axes()` is empty afterward (proving nothing was drawn before the check ran).

---

## Finding 10 — `RVResults.standardize()` raises a raw `ZeroDivisionError`

**Patch:** `finding10_standardize_zerodivision.patch`
**Severity:** Medium.

**Claude Code prompt:**
> In `symbulate/results.py`, `RVResults.standardize()` computes `(self - self.mean()) / self.std()` with no check that `self.std()` is nonzero — confirmed to raise a raw `ZeroDivisionError` for a constant `RV` (e.g. `BoxModel([5])`) and for a single-draw simulation (`.sim(1)`), both of which have zero sample standard deviation. Compute `std = self.std()` first, then check `if np.any(np.asarray(std, dtype=float) == 0):` and raise `Exception(f"Can't standardize data with no variability -- every value is {self.mean()!r}. Standardizing divides by the standard deviation, which is 0 when every simulated value is the same.")` before doing the division. Add regression tests: `RVResults([5.0] * 20).standardize()` and `RVResults([3.0]).standardize()` should both raise with a message containing "no variability" (not `ZeroDivisionError`); normal non-constant data should standardize unaffected.

---

## Finding 12 — `MultivariateDistribution.corr()` divides by zero silently

**Patch:** `finding12_corr_divide_by_zero.patch`
**Severity:** Medium.

**Claude Code prompt:**
> In `symbulate/distributions.py`, `MultivariateDistribution.corr()` computes `cov / np.outer(sd, sd)`, which produces a raw `RuntimeWarning: invalid value encountered in divide` with no context when any component has zero variance (e.g. `Multinomial(n=10, p=[1, 0, 0])` — the two zero-probability categories never vary). The `nan` result is actually correct (correlation with something that never varies from its mean is undefined, not zero) — the fix is to document that clearly and suppress the raw warning, not to change the calculation. Wrap the division in `with np.errstate(invalid="ignore", divide="ignore"):` and update the docstring to state that a zero-variance component's row/column comes back as `nan`, and why. Add a regression test that wraps the call in `warnings.catch_warnings(); warnings.simplefilter("error")` and confirms `Multinomial(n=10, p=[1, 0, 0]).corr()` computes quietly (no warning raised) and that every entry is `nan`.

---

## Finding 14 — `BoxModel` doesn't validate `box`/`size`

**Patch:** `finding14_boxmodel_validation.patch`
**Severity:** Medium.

**Claude Code prompt:**
> In `symbulate/probability_space.py`, `BoxModel.__init__` accepts an empty `box` or a negative `size` without complaint, and only fails later, inside `.draw()`, with raw NumPy messages ("a must be a positive integer unless no samples are taken" / "negative dimensions are not allowed") that never mention `box` or `size` by name. Add two checks in `__init__`, right after `self.box` is built (for both the list and dict input forms) and before the existing `probs` length check: (1) `if len(self.box) == 0: raise ValueError("box is empty -- there is nothing to draw from. Give BoxModel a non-empty list of tickets, e.g. BoxModel([1, 2, 3]).")`. (2) For `size`: only reject an actual negative number, not `None` (meaning "draw 1") or `float('inf')` (a legitimate unbounded lazy sequence) — `if size is not None and isinstance(size, numbers.Real) and size != float("inf") and size < 0: raise ValueError(f"size must be a non-negative number of tickets to draw, got size={size!r}.")`. You'll need to add `import numbers` at the top of the file. Add regression tests: `BoxModel([])` should raise mentioning "empty"; `BoxModel([1,2,3], size=-2)` should raise mentioning "size"; `BoxModel([1,2,3], size=float('inf'))` should still work fine (this is the case the negative check must not accidentally break).

---

## Finding 16 — No `pyproject.toml`

**Patch:** `finding16_pyproject_toml.patch`
**Severity:** Medium.

**Claude Code prompt:**
> This project has no `pyproject.toml`, relying entirely on `setup.py`'s legacy fallback, which upstream packaging tooling is deprecating. Add a `pyproject.toml` at the repo root with just a `[build-system]` table: `requires = ["setuptools>=61", "wheel"]` and `build-backend = "setuptools.build_meta"`. Separately, update `setup.py` to add: `python_requires=">=3.8"` (a permissive floor, not a pin -- adjust if the team has a different minimum in mind); version floors on the existing three dependencies, `install_requires=["numpy>=1.17", "scipy>=1.4", "matplotlib>=3.2"]` (the numpy floor matters in particular -- `np.random.default_rng` needs numpy >= 1.17, and this package already depends on it throughout); and `long_description`/`long_description_content_type` read from `README.md` at import time (`from pathlib import Path; long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")`), since without this a PyPI listing renders blank below the one-line summary. Don't touch the `license=`/classifier fields in the same file -- that's finding 7, a separate PR. Verify by building in a clean virtualenv: `python -m venv /tmp/test_venv && /tmp/test_venv/bin/pip install .` should succeed, and `/tmp/test_venv/bin/python -c "import symbulate; from symbulate import RV, Normal; RV(Normal(0,1)).sim(5)"` should run without error.

---

## Finding 18 — `results.py` hardcodes a duplicate alpha value

**Patch:** `finding18_violin_alpha_constant.patch`
**Severity:** Medium — pure drift-prevention refactor, no behavior change.

**Claude Code prompt:**
> In `symbulate/results.py`, the line `legacy_alpha = 0.5 if alpha is None else alpha` re-hardcodes the literal `0.5` that `plot.py` already defines as `VIOLIN_ALPHA` — and `plot.py`'s own comment there says it expects `results.py` to source this from the constant, so this is exactly the drift risk CLAUDE.md's styling section warns about (retuning `VIOLIN_ALPHA` later wouldn't touch this code path). Add `VIOLIN_ALPHA` to the existing `from .plot import (...)` block near the top of `results.py`, and change the line to `legacy_alpha = VIOLIN_ALPHA if alpha is None else alpha`. This is a pure refactor with no behavior change (both currently evaluate to `0.5`), so no new test is strictly required, but do run the existing violin-plot tests in `test_results.py` to confirm nothing broke.

---

## Finding 22 — `AR`/`ARMA`'s documented mean formula breaks for nonzero-mean noise

**Patch:** `finding22_ar_arma_mean_docs.patch`
**Severity:** Medium — a documentation fix, deliberately *not* a behavior change. Read the note below before implementing something more ambitious.

**Claude Code prompt:**
> `AR`/`ARMA`'s docstrings claim `mean` "really is the mean of every value" for a stationary process, but this silently breaks once `noise_dist` has nonzero mean: `AR(coefs=[0.5], noise_dist=Normal(mean=2, sd=1), mean=0)` actually simulates to a mean of ~4.0, not the documented 0. The true formula (verified by simulation) is `mean + noise_dist.mean() * (1 + sum(ma_coefs)) / (1 - sum(ar_coefs))` for AR/ARMA, and `mean + noise_dist.mean() * (1 + sum(coefs))` for the pure moving-average case, `MA` (which has the identical false claim -- "because the shocks average out, this really is the mean of every value" -- not called out in the original audit finding but the same bug in the same file, worth fixing in the same PR). **Before implementing a code fix that centers the shocks to make `mean` literally correct in all cases: check `AR`'s and `ARMA`'s own docstring `Examples` sections first.** Several of them (and `MA`'s) deliberately use `noise_dist=Bernoulli(1)` (a constant, mean-1 "shock") specifically to demonstrate the recursion mechanics with predictable arithmetic, e.g. `AR(coefs=[0.5], noise_dist=Bernoulli(1), mean=0, initial=0).draw()` documents `path[0] == 1.0, path[1] == 1.5` in a working doctest -- these depend on the *current*, uncentered behavior, and "fixing" the recursion to always center shocks around `noise_dist.mean()` would silently change these to `0.0` and break the doctests (verify this before and after with `pytest --doctest-modules symbulate/time_series.py`). Given that, the safe, scoped fix here is documentation-only: update the `mean` parameter docs in `MA`, `ARMA`, and `AR`'s docstrings (`AR` has its own copy of the claim in its "Notes" section, not just inherited from `ARMA`) to state the formula above, so the parameter's actual behavior with a nonzero-mean `noise_dist` is documented truthfully instead of asserted incorrectly. If the team decides later that the recursion itself should change to center shocks (making `mean` unconditionally correct), that's a bigger, deliberate design decision that also has to address the `Bernoulli(1)` doctest examples throughout this file -- flag it as a separate follow-up rather than doing it inside this PR. Add regression tests (documentation-pinning, not behavior-pinning) that simulate `AR`, `ARMA`, and `MA` each with a nonzero-mean `noise_dist` and confirm the simulated mean matches the formula above (within simulation tolerance) -- these tests exist to catch anyone who *does* change the recursion later without updating the docs to match, in either direction.
