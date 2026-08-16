# Symbulate `dev` Branch Audit — August 2026

**Repo:** `calpoly-symbulate/symbulate`, branch `dev`
**Commit audited:** `6e84a1c` (2026-08-05)
**Method:** Full test-suite run (3,281 tests, `pytest`) plus a targeted code review split across five areas — core simulation engine, `distributions.py`, plotting/tables, stochastic-process modules, and packaging/tests/docs — with findings verified by executing reproduction code wherever possible rather than just reading. A prior audit from June 2026 (`team/symbulate_audit.md`) is also re-checked for drift.

## Bottom line

The package is in good shape overall — 3,269 of 3,281 tests pass, the core simulation engine and most of `distributions.py`'s ~65 distribution classes check out numerically against theory, and all five issues from the June audit have been fixed. But there are two bugs worth fixing before the next release: a floating-point equality check that breaks the most common use of `Multinomial` (and already fails in your own test suite), and a random-seed contract that the `RV.draw()` docstring misrepresents. Beyond that, the rest is a long tail of smaller correctness gaps, error-message violations, and documentation drift — useful to clear out, none of it urgent.

---

## Test suite results

Running `pytest symbulate/tests/` on a clean checkout: **3,269 passed, 12 failed**, 21m24s wall clock (this sandbox's CPU, so treat the absolute time as illustrative, not a CI benchmark).

Of the 12 failures, only **one is a genuine logic bug** — `TestMultinomial::test_Multinomial_plot_pairs` fails because `Multinomial(n=12, p=[0.4, 0.3, 0.2, 0.1])` raises `Exception: Elements of p must be non-negative and sum to 1.` (see Finding 1 below; `sum([0.4,0.3,0.2,0.1])` is `0.9999999999999999` in floating point, not `1.0`). This is a real, currently-failing test on `dev` — not a hypothetical.

The other 11 failures are all `Failed: Timeout (>20.0s)` — I set a 20-second-per-test cap to keep the full run from hanging (an earlier uncapped run stalled past 5 minutes on one test). Spot-checking one of them in isolation (`MA(coefs=[0.8,0.5])[20].sim(10000)`, ~8s alone) suggests most of these are legitimately CPU-heavy Monte Carlo tests (10,000–20,000 simulated paths, some walking a lazily-recomputed `InfiniteVector` step by step) rather than infinite loops or wrong answers — independent numerical checks of the underlying processes (MM1, MM∞, MM/s/s, MA, hitting times) matched theory when I re-ran them outside the full-suite time pressure. That said, the fact that a fresh `pytest` run needs a bespoke timeout to finish at all is itself worth a look — see Finding 11.

---

## Findings

Ordered most to least severe.

### 1. `Multinomial(p=...)` rejects mathematically valid probability vectors — **Critical**

**File:** `symbulate/distributions.py:6872` (and the same pattern at `NegativeMultinomial`, line ~7341)

The parameter check is exact floating-point equality:
```python
if not (all(pi >= 0 for pi in p) and sum(p) == 1):
    raise Exception("Elements of p must be non-negative and sum to 1.")
```
`sum([1/6]*6)`, `sum([0.4, 0.3, 0.2, 0.1])`, and `sum([1/7]*7)` are all `!= 1.0` in floating point, so the textbook fair-die example and the equal-weights case in general are rejected outright. This is confirmed by your own test suite: `test_Multinomial_plot_pairs` fails on exactly this. There's no workaround a student would find on their own — the error message doesn't hint that it's a floating-point issue.

**Fix:**
```python
import math
if not (all(pi >= 0 for pi in p) and math.isclose(sum(p), 1.0, abs_tol=1e-9)):
    raise Exception("Elements of p must be non-negative and sum to 1.")
```
Apply the same fix to `NegativeMultinomial`'s `p_arr.sum() >= 1` check (line ~7341), which has the same boundary fragility.

### 2. `RV.draw()` docstring misrepresents the seeding contract — **High**

**File:** `symbulate/probability_space.py`, `symbulate/random_variables.py:68,97`

`np.random.seed(...)` does not affect the package's internal `np.random.default_rng()` instances used throughout `distributions.py` and elsewhere — confirmed by reseeding to `0` three times and getting three different `RV(Normal(0,1)).draw()` values. The `RV.draw()` docstring itself states that `np.random.seed(0)` reproduces `1.764052345967664` (the value from NumPy's legacy global API), which is simply false under the current architecture. This is already tracked internally in `team/random_seed_notes.md` with a proposed fix, but it's still live in the code, and it's the kind of thing that erodes trust fast for a teaching tool — a student who follows the documented example to make a demo reproducible will get a different answer every time and have no idea why.

CLAUDE.md documents the intended pattern correctly for `diffusion_process.rng` and `hitting_times.rng` ("seed that, not `np.random.seed`") — the fix here is to either apply the same pattern consistently and correct every stale docstring, or provide one package-level `symbulate.set_seed()` that reseeds every module's `rng`.

**Fix:** Correct the `RV.draw()` docstring's example to show the actual (currently non-reproducible, or module-specific) behavior; consider adding a single `set_seed()` convenience function that iterates over each module's `rng` object.

### 3. GARCH/ARCH stationary-variance formula is silently wrong for non-unit-variance noise — **High**

**File:** `symbulate/time_series.py` (`_garch_is_stationary`/`_unconditional_variance`, ~lines 835, 855, 1015–1026, 1110–1120) — *finding from the stochastic-process audit, verified numerically*

The stationarity check and the reported long-run variance both assume `noise_dist` has variance 1, but nothing validates that. With `noise_dist=Normal(0, 2)` and ARCH+GARCH coefficients summing to 0.95, the library reports "stationary" with a long-run variance of 4.0 — but the true persistence is `0.1*Var(shock) + 0.85 = 1.25 > 1`, i.e. genuinely explosive. Simulated variance at t=300 (3,000 paths) came out to **~6×10²⁸**, not 4.0. A student using a non-default noise distribution gets a confidently wrong diagnostic with no warning.

**Fix:** Generalize both functions to `sum(arch_coefs)*noise_dist.var() + sum(garch_coefs)`, or validate `noise_dist.var() == 1` at construction time with a clear error message.

### 4. `InfiniteTuple.__getitem__` silently drops the step on open-ended slices — **High**

**File:** `symbulate/result.py:711`

```python
if n.stop is None:
    if n.start is None:
        return self
    else:
        return type(self)(lambda i: self[i + n.start])
```
This ignores `n.step` entirely. Confirmed: `iv[2::2]` returns `[2, 3, 4, 5, 6, ...]` instead of `[2, 4, 6, 8, 10, ...]` — a silent wrong answer, not a crash, which is worse: nothing signals that anything went wrong.

**Fix:**
```python
step = n.step or 1
return type(self)(lambda i: self[n.start + i * step])
```

### 5. `RandomProcess.__setitem__` is a silent no-op for non-RV/non-scalar values — **High**

**File:** `symbulate/random_processes.py:108`

Assigning something that isn't an `RV` or a plain scalar to `X[t] = ...` (e.g. a list) does nothing at all — no error, no effect. Confirmed: `X.rvs` stays empty and subsequent draws are unaffected. Per CLAUDE.md's own philosophy ("never expose a raw exception... but also never silently do nothing"), this should raise a clear message naming the accepted types.

**Fix:** Add an `else: raise TypeError(...)` branch naming what's accepted.

### 6. Inconsistent handling of bad plot kwargs — **High**

**File:** `symbulate/results.py:2298, 2505-2513` vs. every other branch in the same dispatch

`make_rug`/`make_segmented_rug` never forward `**kwargs`, unlike every sibling branch (`make_hist`, `make_bar`, `make_impulse`, `make_boxplot`, `make_violinplot`, `make_ecdf`, `make_dotplot`). Confirmed: `RV(Normal(0,1)).sim(50).plot(type='rug', linewidth=10)` silently ignores `linewidth=10`. Meanwhile the exact same kind of mistake on `type='hist'` raises a raw `AttributeError: Rectangle.set() got an unexpected keyword argument 'X'` straight from matplotlib internals — which directly violates CLAUDE.md's "never expose a raw Python exception to a student without context" rule. So depending on plot type, a bad kwarg is either silently swallowed or crashes unhelpfully; neither is right, and they disagree with each other.

**Fix:** Forward `**kwargs` from both rug call sites for consistency, then separately add a friendly kwarg-validation wrapper (or a `try/except TypeError` that re-raises with guidance) around the matplotlib calls so mistyped kwargs get an educational message instead of a raw traceback.

### 7. License mismatch: `setup.py` says GPLv3, `LICENSE.txt` is MIT — **High**

**File:** `setup.py` vs. `LICENSE.txt`

`setup.py` declares `license="GPLv3"` and a matching PyPI classifier, but `LICENSE.txt` is the full text of the MIT License — a permissive license with no copyleft obligations. These are legally contradictory statements about the same package. Anyone deciding whether they can redistribute or modify the code (an instructor bundling it into course materials, for instance) gets a different answer depending on which file they read.

**Fix:** Pick the actual intended license and make both agree — given the `LICENSE.txt` text, that almost certainly means changing the classifier and `license=` field to MIT.

### 8. Open-ended slices crash with a raw `TypeError` — **Medium-High**

**File:** `symbulate/random_variables.py:241`, `symbulate/result.py:1127`

Both `RV.__getitem__` and `DiscreteTimeFunction.__getitem__` build `range(n.start or 0, n.stop, n.step or 1)`, which breaks for any slice with no stop — `X[1:]`, `f[2:]` — with `TypeError: 'NoneType' object cannot be interpreted as an integer`. Confirmed for both. This is routine, discoverable-by-accident usage for a process/sequence type, not an edge case.

**Fix:** Handle the open-ended case explicitly (return an `InfiniteVector`/generator, or raise a clear "can't slice without a stop; use `[a:b]` or index individually" message) rather than falling through to `range()`.

### 9. `dims=` validation runs after the pairs-matrix has already partially drawn — **Medium-High**

**File:** `symbulate/results.py:2134-2168`

The `self.dim > 2` branch returns early (line 2143) before the top-level `if "dims" in kwargs: raise ValueError(...)` check (line 2161) ever runs for that path. Confirmed: `RV(Normal(0,1)**3).sim(50).plot(dims=[0,1])` builds a `GridSpec`, resizes the figure, adds a subplot — *then* raises (from a re-entrant inner call, not the intended check). The figure is left half-built. The theoretical counterpart (`MultivariateDistribution.plot()` in `distributions.py:5756`) gets this right by checking first.

**Fix:** Move the `dims=` (and any general unsupported-kwarg) check above the `self.dim > 2` branch, before any drawing happens.

### 10. `RVResults.standardize()` raises a raw `ZeroDivisionError` on zero-variance data — **Medium**

**File:** `symbulate/results.py:1363`

Confirmed for both a constant `RV` (`BoxModel([5])`) and a single-draw simulation (`.sim(1)`). Violates the stated error-message standard.

**Fix:** Check for zero variance and raise an explanatory message ("Can't standardize data with no variability — every value is X.").

### 11. Several Monte Carlo tests are slow enough to need a manual timeout to finish a suite run — **Medium**

Not a correctness bug per se, but worth flagging: a plain `pytest symbulate/tests/` on this hardware ran for 21+ minutes and one test alone ran past 5 minutes uncapped before I added a per-test timeout. The architecture (each `RV.draw()` builds a fresh lazily-evaluated `InfiniteVector`/closure chain with no cross-draw memoization, and heavier tests simulate 10,000–20,000 paths through it) means simulation cost scales with both sample count and path length in a way that's easy to hit accidentally — e.g. a student increasing `Nsim` on an MA/ARMA or continuous-time-queue example in a notebook could end up waiting much longer than expected with no feedback beyond the progress bar. Worth profiling the hot path (`Result.__getitem__`'s per-element `self.func(i)` calls) if this becomes a recurring complaint, and worth deciding on an explicit CI timeout so a genuinely hung test fails fast instead of blocking a pipeline indefinitely.

### 12. `MultivariateDistribution.corr()` divides by zero silently — **Medium**

**File:** `symbulate/distributions.py:5094` (shared by `Multinomial`, `MultivariateHypergeometric`, `NegativeMultinomial`, `Dirichlet`, `DirichletMultinomial`, and the Normal/T/LogNormal families)

A zero-variance component (`Multinomial(n=10, p=[1,0,0])`) produces a matrix of `nan` plus a raw `RuntimeWarning: invalid value encountered in divide` with no context.

**Fix:** Detect zero-variance components before dividing and either return `nan` with a clear note in the docstring, or raise an explanatory error.

### 13. `Table` repr duplicates the last row and prints a fake truncation marker at exactly 19 outcomes — **Medium**

**File:** `symbulate/table.py:164-169, 195-197`

`if i >= 18:` fires on the 19th row regardless of whether more rows remain. `Table({i: i for i in range(19)})` prints all 19 real rows, then a spurious `... ...` row, then a duplicate of the last row, then `Total`. 18 or fewer, and 20 or more, both render correctly — only exactly 19 is broken. `_repr_html_` has the identical bug.

**Fix:** Only truncate when `len(keys) > 19`, e.g. guard with `len(keys) > 19` up front rather than comparing the loop index alone.

### 14. `BoxModel` doesn't validate `box`/`size` — **Medium**

**File:** `symbulate/probability_space.py:406`

An empty box or negative `size` passes construction silently and only fails later, inside `.draw()`, with raw NumPy messages ("a must be a positive integer...", "negative dimensions are not allowed").

**Fix:** Validate at construction time with a message naming the actual problem (empty box, negative size) and how to fix it.

### 15. Sphinx API docs omit 7 of the 23 real modules — **Medium**

**File:** `docs/source/api/symbulate.rst`

Missing entirely: `branching_process.py`, `diffusion_process.py`, `hitting_times.py`, `queues.py`, `random_walk.py`, `renewal_process.py`, `time_series.py` — all export public classes through `symbulate/__init__.py` (`GaltonWatson`, `DiffusionProcess`, `CIR`, `MertonJumpDiffusion`, `hitting_time`, `upcrossings`, the `GG1`/`MG1`/`GM1`/`GGs` queue family, `RandomWalk`, `RenewalProcess`, `CompoundPoissonProcess`, `MA`/`AR`/`ARMA`/`GARCH`/`ARCH`). A student consulting the published docs for any of these finds nothing.

**Fix:** Regenerate with `sphinx-apidoc -o docs/source/api symbulate -f` (or add the missing `automodule` blocks by hand), and consider checking doc build/coverage in CI so it can't drift again.

### 16. No `pyproject.toml` — legacy `setup.py`-only packaging — **Medium**

`ls pyproject.toml` → not present; there's no `[build-system]` declaration, no `python_requires`, no version floors on `numpy`/`scipy`/`matplotlib`, and no `long_description` wired to `README.md` (a PyPI listing would render blank). I confirmed `pip install .` fails in this sandbox with a Debian-distutils-specific `install_layout` error, but reproduced a clean success in a fresh venv — so that specific failure is environment noise, not a project bug. The underlying gap is real, though: the project depends entirely on pip's legacy `setup.py` fallback, which upstream tooling is actively deprecating.

**Fix:** Add a minimal `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"
```
and add `python_requires`, dependency floors, and a `long_description` read from `README.md`.

### 17. `setup.py` metadata points to a stale repo/maintainer — **Medium**

`url="https://github.com/dlsun/symbulate"` and `author_email="dsun09@calpoly.edu"`, but the actual repo is `calpoly-symbulate/symbulate` — CLAUDE.md's own Git Workflow section explicitly warns PRs must **not** target `dlsun/symbulate`. `CONTRIBUTORS.md` lists a different, current maintainer/team. Anyone using `pip show symbulate` to find the source or a contact is sent to the wrong place.

**Fix:** Update `url=` and `author_email=` to the current repo/maintainer.

### 18. `results.py` hardcodes a duplicate alpha value instead of reusing the existing constant — **Medium**

**File:** `symbulate/results.py:2357` vs. `symbulate/plot.py:548`

`legacy_alpha = 0.5 if alpha is None else alpha` re-hardcodes the literal that `VIOLIN_ALPHA = 0.5` in `plot.py` already defines — and `plot.py`'s own comment says it expects `results.py` to source this from the constant. This is exactly the drift risk CLAUDE.md's styling rules are meant to prevent: retuning `VIOLIN_ALPHA` later won't touch this code path.

**Fix:** `from .plot import VIOLIN_ALPHA` and use it directly.

### 19. `make_mosaic`'s parameters violate the `xlabel`/`ylabel` naming convention, and its docstring is wrong — **Medium**

**File:** `symbulate/plot.py:2164-2165, 2271-2279`

`make_mosaic` uses `x_label`/`y_label` (with underscores) when CLAUDE.md is explicit that this should be `xlabel`/`ylabel` everywhere, with no exceptions, and every other 2-D helper follows that rule. Separately, the docstring claims defaults of `"X"`/`"Y"` when the actual signature defaults are `"Variable 1"`/`"Variable 2"` — a second, independent bug.

**Fix:** Rename the parameters and correct the docstring.

### 20. `team/` directory: ~49MB of scratch notebooks committed to the student-facing branch — **Medium**

154 files, individual notebooks up to 6.5MB, contributing to a 68MB `.git`. This is exploratory/design content (`*_demo.ipynb`, design-discussion notebooks) mixed directly into the branch an instructor would have students clone, roughly tripling the useful clone size.

**Fix:** Move to a separate internal repo/branch, or at minimum strip notebook outputs (`nbstripout`) before committing.

### 21. `test_smoke.py` is nearly vacuous for what it claims to cover — **Low-Medium**

**File:** `symbulate/tests/test_smoke.py` (6 lines)

Asserts only `hasattr(symbulate, "RV")` and `hasattr(symbulate, "BoxModel")`, despite CLAUDE.md describing it as an "end-to-end import and basic usage smoke test." A broken import anywhere else in the package (e.g. in `queues.py`) wouldn't be caught here even though that's its stated job.

**Fix:** Import the full public API and run one trivial `.sim()` per major subsystem.

### 22. `AR`/`ARMA`'s documented mean formula breaks for nonzero-mean noise — **Medium**

**File:** `symbulate/time_series.py:541-566` — *stochastic-process audit finding, verified numerically*

The docstring's claim that "`mean` really is the mean of every value" for a stationary process silently fails once `noise_dist` has nonzero mean. Confirmed: `AR(coefs=[0.5], noise_dist=Normal(mean=2, sd=1), mean=0)` simulates to ~3.997 (matching the correct theoretical `mean + E[shock]/(1-phi) = 4.0`), not the documented 0, with no warning.

**Fix:** Either document the constraint (noise must be mean-zero for the `mean=` parameter to mean what it says) or generalize the mean the same way `_stationary_covariance` already generalizes the variance.

### 23. Various documentation self-contradictions in `CLAUDE.md`/`DECISIONS.md` — **Low**

Spot-checked and confirmed:
- `classify_data()` is referenced repeatedly in CLAUDE.md; the actual function is named `classify_values` (no `classify_data` exists anywhere in the code).
- `N_SMALL_THRESHOLD` is documented as `100` ("unchanged"); the actual value is `123`, and `DECISIONS.md`'s own changelog records the change CLAUDE.md says didn't happen.
- CLAUDE.md's Overlay Policy says "every 2-D plot is now a hard error" for overlay; mosaic plots are an explicit, working exception (two consecutive mosaic `.plot()` calls draw fine with a warning) — CLAUDE.md's own "Two Variables" section correctly documents this exception elsewhere, so it's an internal contradiction, not a code bug.
- `make_density2D`/`make_joint_pdf` docstrings say `contour` defaults to `False`; the actual (and correctly-behaving) default is `True`.

**Fix:** A documentation pass reconciling CLAUDE.md against DECISIONS.md and the current code; low urgency since none of these mislead end users, only contributors reading CLAUDE.md in isolation.

### 24. Minor distribution/error-message issues — **Low**

- `MultivariateHypergeometric` rejects whole-number floats in `m` (e.g. `[10.0, 8.0, 6.0]`), unlike `DiscreteUniform`'s more permissive `_is_whole_number()` pattern used elsewhere in the same file (`distributions.py:7107`).
- `Integers`/`DiscreteTimeSequence` accept negative values despite docstrings claiming "non-negative integers only" (`index_sets.py:382`) — already flagged by the team's own test comments as known/unresolved.
- `tabulate(bin=True)` on categorical data leaks a raw NumPy `UFuncTypeError` (`results.py:1462`).
- `get_next_color(ax)` is called three times per 2-D `.plot()` with marginals (main panel + two strips), not "exactly once" as CLAUDE.md states — currently harmless since every 2-D overlay is a hard error anyway, but a real deviation from the documented rule.
- `spinner.py` calls `plt.subplots()` directly and hardcodes its own 20-color palette rather than Okabe-Ito — both are documented no-nos elsewhere in CLAUDE.md, though `spinner()` is a standalone diagram outside the `.plot()`/overlay system, so this is very likely an intentional, undocumented exception rather than a live bug.

### 25. Cosmetic issues

- Duplicated word "when when" in a multivariate-statistic error message (`results.py:1317`).
- Dead code in `TruncatedNormal.__init__` (`distributions.py:2178`): `lower`/`upper` are computed, including a wasted quantile call, but never used.
- `setup.py` has no `CHANGELOG` and no git tags to verify `version="0.5.5"` actually corresponds to anything releasable.

---

## Already fixed since the June 2026 audit

All five issues flagged in `team/symbulate_audit.md` are resolved on current `dev`:

| Issue | Status |
|---|---|
| Seaborn-colorblind fuzzy-match via `difflib` | Fixed — removed entirely |
| `init_color()` deprecated `cm.get_cmap()` | Fixed — function removed |
| `ax.violinplot(vert=...)` deprecated kwarg | Fixed — uses `orientation=` |
| `hist2d` normalize — `int()` on float string | Fixed — uses `density=True` + `FuncFormatter` |
| Legacy `np.random.*` global API (13 sites) | Fixed — every file now uses its own `np.random.default_rng()` |

The multivariate PDF bug from that audit (`MultivariateNormal.pdf`/`Multinomial.pdf` returning a frozen scipy object instead of a float) is also confirmed fixed — every multivariate distribution class checked correctly calls `.pdf(x)`/`.pmf(x)`.

---

## Suggested priority order

1. Fix the `Multinomial`/`NegativeMultinomial` floating-point `sum(p)==1` check (Finding 1) — it's actively failing your own tests today, and it's a two-line fix.
2. Correct or remove the false `RV.draw()` seeding example (Finding 2) — cheap, and it's actively misleading for a teaching tool.
3. Fix the GARCH/ARCH stationary-variance formula and the `InfiniteTuple` slice-step bug (Findings 3–4) — both produce silently wrong numbers, which is worse than a crash.
4. Sweep the plotting-kwarg inconsistency and the `dims=`-after-partial-draw bug (Findings 6, 9) — both are visible the first time a student makes a typo.
5. Reconcile the MIT/GPLv3 license mismatch (Finding 7) — a five-minute fix with outsized legal-ambiguity impact.
6. Everything else (Findings 8, 10–25) is worth working through, but none of it is blocking.
