# Symbulate `dev` Branch Re-Audit — August 2026

**Repo:** `calpoly-symbulate/symbulate`, branch `dev`
**Commit audited:** `66eb1bb` (2026-08-10)
**Previous audit:** `symbulate_dev_audit_2026-08.md`, commit `6e84a1c` (2026-08-05)
**Method:** Fresh clone, full read-through of every source module split across ten parallel review passes, each cross-checked against `CLAUDE.md`/`DECISIONS.md`/`MODEL-DECISIONS.md`'s specific invariants (not just general code quality), with every suspected bug confirmed by direct reproduction against the installed package. No pytest run in this pass, per your instruction — findings below are from code reading plus small standalone repro scripts, not the test suite.

## Bottom line

A lot of real work landed since the last audit, and it shows: 13 of the 15 concretely-actionable findings from the previous report are fixed, most of them exactly the way we suggested (the float-tolerance probability check, the seeding docstring and new package-level `seed()`, the GARCH variance generalization, the slice-step and open-ended-slice bugs, the `dims=` validation order, `standardize()`'s zero-variance guard, `BoxModel` construction validation, the MIT/GPLv3 license mismatch, packaging modernization). That's a genuinely good track record.

But this pass found a new **Critical** bug that's more serious than anything in the last report: conditioning (`X | event`) on certain distributions silently returns draws that mostly *violate* the condition — 87% of "conditioned" values failed the condition in a direct test. This is a correctness bug in the core inference primitive of a probability teaching tool, and it produces confident, plausible-looking, wrong output with no error at all. There's also a second, unrelated silent-wrong-answer bug in `Table` arithmetic, and a handful of new High-severity issues (a GARCH warm-up-variance formula bug, an unguarded/undocumented `MultivariateNormal`/`Wishart`-family crash, a cross-cutting mixin bug that breaks `.cov()`/`.corr()` on `Tuple`/`Vector`, and others) introduced or exposed by the large amount of new code (plotting, queues, diffusion processes, Markov chains, hitting times) written since the last pass. Most of the new large subsystems — `queues.py`, `poisson_process.py`, `hitting_times.py`, `markov_chains.py`, `gaussian_process.py` — held up very well against their own documented invariants; the new bugs cluster instead in the older, more heavily-reused core (`random_variables.py`, `base.py`, `table.py`, `probability_space.py`) and in the newest plotting/distribution edge cases.

---

## Fixed since the last audit

| # | Finding | Status |
|---|---|---|
| 1 | `Multinomial`/`NegativeMultinomial` exact-float-equality `p`-sum check | **Fixed** — `math.isclose`, with regression tests |
| 2 | `RV.draw()` seeding docstring misrepresented the contract | **Fixed** — docstring corrected, plus a new package-level `seed()` function that reseeds the shared generator (exactly the suggested remedy) |
| 3 | GARCH/ARCH stationary-variance formula ignored `noise_dist.var()` | **Fixed** — generalized to `sum(arch_coefs)*noise_var + sum(garch_coefs)`, with regression tests (see Finding 4 below for a related bug this fix exposed) |
| 4 (old) | `InfiniteTuple.__getitem__` dropped the slice step | **Fixed** — regression tests present (but see new Finding, negative-index slicing, below) |
| 5 (old) | `RandomProcess.__setitem__` silent no-op for bad values | **Fixed** — raises `TypeError` naming accepted types |
| 6 (old) | `make_rug`/`make_segmented_rug` didn't forward `**kwargs` | **Fixed**, both 1D and 2D branches (but see new Finding, other plot-type branches with the same gap, below) |
| 7 (old) | License mismatch (MIT vs GPLv3) | **Fixed** — both `setup.py` and classifiers now say MIT |
| 8 (old) | Open-ended slice crash on `RV`/`DiscreteTimeFunction` | **Fixed** — `RV` delegates to native slicing; `DiscreteTimeFunction` raises a clear `ValueError` |
| 9 (old) | `dims=` validated after partial pairs-matrix draw | **Fixed** — validation now runs before any drawing |
| 10 (old) | `standardize()` raw `ZeroDivisionError` | **Fixed** — explanatory `Exception` |
| 12 (old) | `MultivariateDistribution.corr()` silent divide-by-zero warning | **Fixed** — wrapped in `np.errstate`, documented as intentional `nan` |
| 14 (old) | `BoxModel` didn't validate `box`/`size` | **Fixed for empty box / negative size** — but narrower than the underlying problem; see new Findings 10–12 below, which show the same silent-construction pattern still happening for invalid `probs` and other malformed `size` values |
| 16 (old) | No `pyproject.toml` / packaging modernization | **Fixed** — `pyproject.toml` added, `python_requires`, dependency floors, and `long_description` all present in `setup.py` |
| 18 (old) | Duplicate hardcoded `VIOLIN_ALPHA` | **Fixed** — `results.py` imports the constant from `plot.py` |
| 19 (old) | `make_mosaic` naming/docstring violation | **Fixed** — uses `xlabel`/`ylabel`, docstring corrected |
| 22 (old) | AR/ARMA mean formula wrong for nonzero-mean noise | **Fixed, docs-only, as recommended** — docstrings corrected; the raw recursion is deliberately unchanged (confirmed the `Bernoulli(1)` doctests still pass under it), avoiding the regression risk flagged in the original report. **However, the same underlying assumption is now violated more severely, and undocumented, in `GARCH`** — see new Finding below. |

**Still open, unchanged from the last audit:**

- **Finding 13** (Table 19-row truncation bug) — still present, byte-for-byte.
- **Finding 15** (Sphinx docs missing 7 modules) — still missing exactly the same 7: `branching_process`, `diffusion_process`, `hitting_times`, `queues`, `random_walk`, `renewal_process`, `time_series`.
- **Finding 17** (`setup.py` stale repo/maintainer) — `url=` still points to `dlsun/symbulate`, `author_email=` still the old address.
- **Finding 20** (`team/` scratch-notebook bloat) — still ~51MB / 201 files.
- **Finding 21** (`test_smoke.py` nearly vacuous) — still exactly two `hasattr` assertions.
- **Finding 23** (CLAUDE.md/DECISIONS.md self-contradictions) — `classify_data()` vs. real name `classify_values()`, `N_SMALL_THRESHOLD` (100 documented vs. 123 real), and the `contour` default docstrings (say `False`, actually `True`) are all still unreconciled.
- **Finding 24** (`MultivariateHypergeometric` rejects whole-number floats; `Integers`/`DiscreteTimeSequence` accept negative values despite their docstring; `tabulate(bin=True)` on categorical data; `get_next_color` called 3× per marginal plot) — all four still present exactly as described.
- **Finding 25** (dead code in `TruncatedNormal.__init__`; "when when" typo) — both still present.

---

## New findings

Ordered most to least severe.

### 1. Conditioning on certain distributions silently returns draws that mostly violate the condition — **Critical**

**File:** `symbulate/random_variables.py:130` (the fast-path check in `RV.sim()`), interacting with `RVConditional.__init__` at line 504

`RV.sim()` has a fast path that skips the normal per-draw loop and calls `self.prob_space._fast_sim(n)` directly whenever `self.func is _IDENTITY` — an optimization for distributions (currently `TruncatedNormal` and `NegativeHypergeometric`) that can vectorize their own sampling. The check only tests `func is _IDENTITY`, not the actual class of `self`. `RVConditional.__init__` passes the wrapped RV's `func` through unchanged, so a *conditioned* RV built on top of one of these two distributions still has `func is _IDENTITY` — and the fast path fires for it too, completely bypassing `RVConditional.draw()`'s rejection-sampling loop and the conditioning event itself.

I confirmed this directly: `Y = RV(TruncatedNormal(mean=0, sd=1, a=-2, b=2)) | (X > 1); Y.sim(2000)` returns a sample in which **86.6% of the values fail the condition `X > 1`** — the simulation is silently drawing from the *unconditioned* distribution and calling it conditioned. There is no error, no warning, and the output looks entirely plausible (it's a real sample from a real distribution — just not the one the student asked for). The same failure reproduces with `NegativeHypergeometric`. No existing test calls `.sim()` on an `RVConditional` built from either of these two classes.

**Fix:** Gate the fast path on the exact class (`type(self) is RV`), not just `func is _IDENTITY` — any subclass that overrides `.draw()`/conditioning behavior without changing `func` needs the ordinary per-draw path.

**Suggested test:** `Y = RV(TruncatedNormal(0,1,-2,2)) | (RV(TruncatedNormal(0,1,-2,2)) > 1)` — actually, more precisely, condition an RV on itself (`X = RV(TruncatedNormal(...)); Y = X | (X > 1)`) and assert `all(v > 1 for v in Y.sim(2000))`. Repeat for `NegativeHypergeometric`. This should also become a standing regression test for "any future `_fast_sim`-eligible distribution used inside a conditional."

### 2. `Table` addition (and other arithmetic) silently produces nested nonsense instead of an error — **High**

**File:** `symbulate/table.py:227-234` (`Table._operation_factory`), via the `Arithmetic` mixin in `base.py`

`Table + Table` doesn't raise and doesn't do a sensible merge — it silently nests. `Table._op_func` applies `op(count, other)` per outcome assuming `other` is a plain scalar; when `other` is actually another `Table`, Python's operator protocol falls back to `other.__radd__(count)`, which builds an entirely new `Table` and stores it as the "sum" under the first table's key. I confirmed: `Table({'a':1,'b':2}) + Table({'a':10})` returns `Table({'a': Table({'a': 11}), 'b': Table({'a': 12})})` — a `Table` of `Table`s, silently, with no error. Combining two `.tabulate()` outputs is an entirely natural thing for a student to try, and the result here is confusing garbage with no error message pointing at the mistake.

**Fix:** In `Table`'s arithmetic path, explicitly check whether `other` is a `Table`/`dict` and either raise a clear `TypeError` ("Table arithmetic only supports a scalar right-hand side...") or implement a real key-aligned merge, rather than falling through to generic scalar-op behavior.

**Suggested test:** `Table({'a':1,'b':2}) + Table({'a':10})` should raise an explanatory error (or produce a documented, correct key-aligned merge) — currently uncovered by any test either way.

### 3. `InfiniteTuple`/`InfiniteVector` give nondeterministic, cache-order-dependent answers for negative indices and slices — **High**

**File:** `symbulate/result.py:743`

I reproduced this directly: building a fresh `InfiniteVector(lambda i: i)` and immediately reading `iv[-3:8]` gives one answer; building an identical `InfiniteVector`, forcing it to extend further first (e.g. by reading `iv[15]`), and *then* reading the same `iv[-3:8]` gives a *different* answer. The result of a negative-index/slice read depends on how much of the lazily-cached sequence happens to have been materialized already — an implementation detail with no relationship to what the slice notation is supposed to mean. This is worse than a crash: two students running what looks like the same code, in a notebook where one of them happened to touch the sequence earlier, get different — and both silently wrong — answers.

**Fix:** Either explicitly reject negative indices/slices on an unbounded lazy sequence with a clear "this is an infinite sequence; negative indices aren't well-defined, use a plain (non-infinite) slice or index from the start" error, or (if a bounded materialize-then-index-from-the-end behavior is actually intended) make the result independent of prior cache state by always fully resolving the needed range deterministically before applying the negative offset.

**Suggested test:** two independently constructed but logically-identical `InfiniteVector`s, one accessed only via `iv[-3:8]`, the other forced to extend its cache first via an unrelated positive index, should return the *same* result for `iv[-3:8]` (or both should raise the same clear error).

### 4. `GARCH`'s presample variance uses the wrong formula, contradicting its own docstring — **High**

**File:** `symbulate/time_series.py:1036-1041` (`GARCHResult.squared_at`)

For negative indices (the presample warm-up period), `squared_at` returns `self.initial`, which is `E[sigma**2]` — the long-run *conditional* variance — not `E[X**2] = E[sigma**2] * noise_var`. This is fine when `noise_dist` has unit variance (the two are equal) but wrong otherwise, and it directly contradicts the docstring's claim that starting from the default `initial` means "every value has the same variance from time 0 onward." I confirmed the gap numerically: with `noise_dist=Normal(0,2)` (`noise_var=4`), `Var(X[0])` measures ≈3.42 against a documented/target value of 4.0, only converging to ≈4.0 by around `t=10–20`. A test in the current suite (`test_long_run_variance_matches_closed_form_for_non_unit_variance_noise`) already knows about this — its own comment calls it "a real (separate, pre-existing) warm-up transient" and deliberately checks variance at `t=20`/`t=40` instead of `t=0` to route around it — but it was never fixed in the source or disclosed in the public docstring.

**Fix:** Thread `noise_var` into `GARCHResult` and use `self.initial * noise_var` as the presample stand-in in `squared_at`.

**Suggested test:** assert `Var(X[0].sim(N))` (not just `X[20]`/`X[40]`) is close to the documented long-run variance when `noise_dist` has non-unit variance.

### 5. `GARCH` silently gives wildly wrong values for nonzero-mean noise, with no validation or documentation — **High**

**File:** `symbulate/time_series.py:946-951` (`_validate_garch`), docstring at line ~1253

This is the same underlying gap that Finding 22 (old) fixed for `AR`/`ARMA`/`MA` — but for `GARCH`, it was neither generalized nor documented. `_validate_garch` only checks that `noise_dist` is a `Distribution`/`RV`; it never checks or documents a requirement that `noise_dist.mean() == 0`. The docstring's own doctest unconditionally asserts values are "centered at 0," with no caveat. I confirmed: `GARCH(omega=0.2, arch_coefs=[0.1], garch_coefs=[0.85], noise_dist=Normal(mean=3, sd=1))` gives a simulated mean around **2137** at `t=20` — not remotely close to 0 — with no error, warning, or docstring caveat anywhere pointing at the cause. Given how carefully the analogous AR/ARMA/MA issue was resolved and documented, this looks like the same fix was simply never ported over to `GARCH`.

**Fix:** Either validate `noise_dist.mean() == 0` at construction (raising a clear error naming the assumption) or add the same documented correction AR/ARMA/MA received.

**Suggested test:** `GARCH(..., noise_dist=Normal(mean=3, sd=1))` should raise a clear error, or its docstring/behavior should account for the drift — not silently report values near 0 while actually producing values in the thousands.

### 6. `Statistical.cov()`/`.corr()`/`.corrcoef()` crash on `Tuple`/`Vector` — a cross-cutting mixin bug — **High**

**File:** `symbulate/base.py:764` (`Statistical`), consumed by `symbulate/result.py:193` (`Tuple`)

`Statistical`'s multivariate-statistic methods call `self._multivariate_statistic_factory(...)`, which is defined and works correctly through `RVResults`, but `Tuple`/`Vector` (also documented `Statistical` consumers) never define it. I confirmed: `(X & Y).draw().cov()` raises a raw `AttributeError`. This is exactly the kind of MRO-assumption bug that's easy to miss because the mixin is exercised only through its most common consumer in testing (`test_base.py` tests every mixin via a single class — `RV`, `RVResults`, or `Event` — never `Tuple`/`Vector`/`TimeFunction`) — it works perfectly through one path and is silently (well, loudly, but only there) broken through another.

**Fix:** Either give `Tuple`/`Vector` their own `_multivariate_statistic_factory`, or have `Statistical` raise a clear, Symbulate-authored message when the consuming class doesn't support it, instead of an unexplained `AttributeError`.

**Suggested test:** extend `test_base.py`'s `Statistical` tests to run through at least one non-`RVResults` consumer (e.g. `Tuple`), not just the one class it happens to already pass through.

### 7. Malformed `mean`/`cov`/`scale` on the matrix-shaped multivariate distributions leaks a raw `TypeError` — **High**

**File:** `symbulate/distributions.py:6649-6663, 6947-6961, 7196-7200, 7385-7389` (`MultivariateNormal`, `MultivariateT`, `Wishart`, `InverseWishart`; `MultivariateLogNormal` inherits the problem via delegation)

These four classes validate their vector/matrix parameters with bare `len(...)` calls and no `try/except`. A plausible mistake — a bare scalar instead of a one-element list (`MultivariateNormal(mean=5, cov=[[1]])`), or a ragged/malformed matrix — reaches the raw `TypeError: object of type 'int' has no len()` with zero context. This is a direct, confirmed violation of the Error Message Standard, and it's inconsistent with the *other* family of multivariate distributions in the same file (`Multinomial`, `Dirichlet`, `MultivariateHypergeometric`, `NegativeMultinomial`, `DirichletMultinomial`), which already wrap the equivalent conversion in `try/except (TypeError, ValueError)` and give a friendly message.

**Fix:** Wrap the `len(...)` conversions in each of the four affected `__init__`s in the same `try/except` pattern the sum-constrained families already use.

**Suggested test:** `MultivariateNormal(mean=5, cov=[[1]])` (and the analogous call for `MultivariateT`, `Wishart`, `InverseWishart`, `MultivariateLogNormal`) should raise a Symbulate-authored message, not a raw `TypeError`.

### 8. `.plot(type='density')`/`type='density2d'` crash with a raw scipy error on too-few points — **High**

**File:** `symbulate/plot.py:4181` (`make_density`), `:6750` (`make_density2D`'s grid helper)

Both call `scipy.stats.gaussian_kde` directly with no guard for fewer than 2 (distinct) points, unlike the sibling `make_segmented_density`, which explicitly checks `np.unique(values).size < 2` and falls back to tick marks. I confirmed: `RV(Normal(0,1)).sim(1).plot(type='density')` raises a raw scipy `ValueError` with no Symbulate context; the 2D density helper crashes the same way on collinear/too-few-point data.

**Fix:** Mirror `make_segmented_density`'s existing guard in both `make_density` and `make_density2D`, raising a friendly, Symbulate-authored message ("a density estimate needs at least 2 distinct values; you have 1 — try `type='rug'` or `type='dotplot'` instead") instead of letting the call reach scipy unguarded.

**Suggested test:** `.sim(1).plot(type='density')` and the 2D equivalent should raise a clear Symbulate message, not a raw scipy exception.

### 9. `DiffusionProcess`'s documented "`sqrt` needs a floor" pitfall is neither guarded nor warned against in practice — **High**

**File:** `symbulate/diffusion_process.py` (class docstring ~lines 256-326; `get_diffusion_process_result` ~lines 23-71)

CLAUDE.md explicitly calls this out as a known trap ("A `sqrt` in `diffusion` needs a floor... one `nan` poisons the rest of the path") and says it should be clearly documented if not internally guarded — but the code does neither. I reproduced the actual failure mode with a natural, CIR-shaped SDE written through the general `DiffusionProcess` API (an unguarded `sqrt(x)` diffusion term): **119 of 200 simulated paths came back `nan` by `t=5`.** There's no docstring warning near the `diffusion=` parameter itself, and no test covering it — a student modeling a mean-reverting, always-positive process the "obvious" way (rather than reaching for the exact `CIR` class) gets silent, majority-`nan` output.

**Fix:** At minimum, add a prominent warning in the `DiffusionProcess` docstring right next to the `diffusion=` parameter, with a concrete `max(x, 0)` example; consider whether an internal floor is safe to add by default for `sqrt`-shaped terms.

**Suggested test:** a regression test simulating a CIR-shaped `DiffusionProcess` with a naive unguarded `sqrt` and asserting the current documented behavior (whether that ends up being "guarded" or "clearly explained") — right now there's nothing pinning this pitfall at all, documented or not.

### 10. `BoxModel` doesn't validate that `probs` is an actual probability vector — **Medium-High**

**File:** `symbulate/probability_space.py:645-649`

`BoxModel.__init__` checks only that `probs` has the right *length*, never that it's non-negative or sums to 1. I confirmed: `BoxModel(['a','b','c'], probs=[0.5, 0.5, 0.5])` constructs without error, and only fails inside `.draw()` with a raw NumPy message — `ValueError: Probabilities do not sum to 1. See Notes section of docstring for more information.` — pointing the student at a numpy docstring section they can't see. This is exactly the "silent construction → raw NumPy error in `.draw()`" pattern the recent `box`/`size` fix (old Finding 14) closed, just for a different parameter.

**Fix:** Validate `probs` at construction (non-negative, sums to ~1 via `math.isclose`) and raise a Symbulate-authored error naming the problem.

**Suggested test:** `BoxModel(['a','b','c'], probs=[0.5, 0.5, 0.5])` should raise at construction with a Symbulate message.

### 11. Every multivariate distribution except three hardcoded names crashes raw on `.spinner()` — **Medium-High**

**File:** `symbulate/spinner.py:246, 252-256`

The multivariate guard is a hardcoded set of three class names (`{"BivariateNormal", "MultivariateNormal", "Multinomial"}`) instead of an `isinstance(dist, MultivariateDistribution)` check (the pattern `branching_process.py` correctly uses for the same purpose). Every other multivariate distribution — `MultivariateT`, `Wishart`, `InverseWishart`, `MultivariateHypergeometric`, `NegativeMultinomial`, `Dirichlet`, `DirichletMultinomial`, `MultivariateLogNormal` — slips past the guard. I confirmed: `Dirichlet([1,1,1]).spinner()` raises a raw `AttributeError: 'Dirichlet' object has no attribute 'quantile'`; `MultivariateT(...)` and `Wishart(...)` fail the same way. `test_spinner.py`'s multivariate-error test class only exercises the two names that happen to be in the set, so the gap is untested as well as unfixed.

**Fix:** Replace the name set with `isinstance(dist, MultivariateDistribution)`.

**Suggested test:** `Wishart(...).spinner()`, `Dirichlet(...).spinner()`, and `MultivariateT(...).spinner()` should each raise the same friendly multivariate-rejection message, not a raw `AttributeError`.

### 12. Pairs-matrix joint panels bypass kwarg validation, leaking a raw exception on the package's own default 3+-variable plot — **Medium-High**

**File:** `symbulate/results.py:2273` (`_draw_pairs_joint`)

The pairs matrix — which is the *default* plot for 3+ simulated variables per CLAUDE.md — draws its `tile`/`hist2d` joint panels by calling `make_tile`/`make_hist2d` directly rather than through the `_call_plot_helper` wrapper the rest of the dispatch uses for friendly kwarg-error translation. A bad kwarg on the most common multi-variable plotting path in the package therefore surfaces a raw matplotlib `AttributeError` instead of the explanatory message every other plot type gets.

**Fix:** Route the pairs-matrix joint-panel calls through the same `_call_plot_helper` wrapper (or an equivalent) used elsewhere in the dispatch.

**Suggested test:** `X[[0,1,2]].sim(50).plot(linewidth=10)` (3+ discrete variables, triggering a tile joint panel) should raise Symbulate's friendly kwarg error, not a raw matplotlib one.

### 13. `initial="stationary"` sentinel check crashes confusingly when `initial` is an `RV` — **Medium**

**File:** `symbulate/time_series.py:397` (`_validate_arma`), `:955` (`_validate_garch`)

Both validators write `if initial == STATIONARY:` before checking `initial`'s type. `AR`/`ARMA`/`GARCH` all explicitly support passing an `RV`/`Distribution` as `initial` (`_draw_initial_values` checks for this) — but an `RV`'s `==` returns an `Event`, not a bool, and casting an `Event` to bool unconditionally raises. I confirmed: `AR(coefs=[0.5], initial=RV(Normal(0,1)))` raises `TypeError: Cannot cast an Event to a boolean. You may be getting this error if you wrote an expression like (2 < X < 5)...` — a message about chained comparisons that has nothing to do with the actual mistake, even though passing a plain `Distribution` the same way works fine.

**Fix:** Check `isinstance(initial, str)` before comparing to the `STATIONARY` sentinel.

**Suggested test:** `AR(coefs=[0.5], initial=RV(Normal(0,1)))` should construct successfully, not raise the unrelated Event-boolean message.

### 14. `MA`/`AR`/`ARMA`/`GARCH`/`ARCH` coefficient validators exhaust generator inputs, silently emptying or crashing — **Medium**

**File:** `symbulate/time_series.py:25-45` (`_validate_ma`), `:358-383` (`_validate_arma`), `:904-929` (`_validate_garch`)

Each validator checks `coefs` with a generator expression (`any(... for c in coefs)`), which exhausts a one-shot iterable. The constructors then call `list(coefs)` on the now-empty iterable. For `MA`/`GARCH`/`ARCH` this silently produces an empty coefficient list — a completely different, undocumented model, no warning. For `AR`/`ARMA` it instead raises a bare, unexplained `TypeError: object of type 'generator' has no len()`. I confirmed: `MA(coefs=(c for c in [0.8, 0.5])).coefs == []`.

**Fix:** Materialize `coefs` via `list(coefs)` once, up front, before any validation loop iterates over it.

**Suggested test:** `MA(coefs=(c for c in [0.8, 0.5]))` should behave identically to `MA(coefs=[0.8, 0.5])`, not silently produce `coefs=[]`.

### 15. `Table` and `Event` are missing from the public API — **Medium**

**File:** `symbulate/__init__.py`

Neither `Table` (returned by every `.tabulate()` call) nor `Event` (returned by every comparison on an `RV`, per `Comparable`) is imported into `symbulate/__init__.py`. I confirmed `from symbulate import Table`/`Event` both fail with `NameError` after `from symbulate import *` — even though `Table`'s own module docstring and three docstring examples in `base.py`'s `Logical` mixin show exactly that usage pattern as if it already worked. This is also why Sphinx's autodoc (built from `__init__.py`) has no page for either class.

**Fix:** Add `from .table import Table` and `from .probability_space import Event` to `__init__.py`.

**Suggested test:** extend `test_smoke.py` (already flagged as too thin, old Finding 21) with `assert hasattr(symbulate, "Table")` and `assert hasattr(symbulate, "Event")`.

### 16. `from symbulate import *` leaks internal names into the public namespace — **Medium**

**File:** `symbulate/__init__.py` (`from .math import *`), `symbulate/math.py` (no `__all__`)

`math.py` has no `__all__`, so its own internal imports leak through the wildcard import into the top-level `symbulate` namespace. Confirmed present after `from symbulate import *`: `np`, `stats`, `math` (the stdlib module), `op` (the `operator` module), `numbers`, plus internal classes `RV`, `Tuple`, `TimeFunction`, `ContinuousTimeFunction`, `DiscreteValued`, `Results`. Since every docstring and demo in this codebase teaches `from symbulate import *`, a student who follows that pattern silently gets `np`/`stats`/`math`/`numbers` bound to library internals — which can mask their own subsequent `import numpy as np` typo — and sees several classes that were never meant to be public API.

**Fix:** Add an explicit `__all__` list to `math.py` covering only its intended public free functions/constants.

**Suggested test:** after `from symbulate import *`, assert `not hasattr(symbulate, 'np')` and `not hasattr(symbulate, 'stats')`.

### 17. `BoxModel`'s `probs=`/dict-`box` interaction is backwards — **Medium**

**File:** `symbulate/probability_space.py:617-621, 645-649`

The docstring says `probs` is "ignored when `box` is a dict," but the length check compares the caller's raw `probs` against the box's *expanded* length regardless of whether `box` is a dict — it should compare against `self.probs`, which is `None` for a dict box. Two confirmed symptoms: `BoxModel({'a':2,'b':3}, probs=[0.5,0.5])` raises a `ValueError` about mismatched length even though `probs` was always going to be discarded; and `BoxModel({'a':2,'b':3}, probs=[0.9,0.9,0.05,0.05,0.05])` (length coincidentally matching the expanded box) constructs with **no error or warning at all**, silently discarding the custom weights — the more dangerous of the two, since nothing tells the caller their weighting was ignored.

**Fix:** Skip the `probs`/length check entirely when `box` is a dict, and/or explicitly raise or warn if both a dict `box` and `probs` are supplied together.

**Suggested test:** both the mismatched-length-but-ignored case and the matching-length-but-discarded case should behave predictably and non-silently.

### 18. `BoxModel`'s `size` validation still misses non-numeric/fractional/boolean values — **Medium**

**File:** `symbulate/probability_space.py:635-644`

The recent fix for old Finding 14 only rejects `size` that is a `numbers.Real` and negative. It doesn't check that `size` is otherwise usable. I confirmed each of these constructs silently and only fails inside `.draw()` with a raw NumPy error: `BoxModel([1,2,3], size="two")`, `size=2.5`, `size=False` — the exact "silent-then-raw-message" pattern the recent fix was meant to close, just triggered by a different kind of bad input.

**Fix:** Require `size` to be `None`, `float('inf')`, or a non-negative `numbers.Integral`; raise a Symbulate error naming the bad value otherwise.

**Suggested test:** `BoxModel([1,2,3], size="two")`, `size=2.5`, `size=False` should each raise at construction.

### 19. `Table.__repr__`'s column alignment breaks with mixed-length outcome labels — **Medium**

**File:** `symbulate/table.py:157-169`

Rows compute their own padding basis independently (`len(outcome_column)` vs. the row's own key length) rather than consistently using one shared `max(max_key_length, len(outcome_column))` the way the header and `Total` row already do. With realistic mixed-length category labels (e.g. survey responses), the value column visibly doesn't line up between rows. This is a distinct bug from the 19-row truncation issue (old Finding 13) — it's triggered by label length, not row count.

**Fix:** Compute one `pad_width` up front from the max of all key lengths and the header, and use it uniformly for every row, the header, and the `Total` row.

**Suggested test:** `repr(Table({'Yes': 5, 'IDontKnowThisOne': 3, 'No': 2}))` — assert every row's value column starts at the same character offset.

### 20. `.plot(type='density')` and the 2D grouped-violin dispatch branches don't forward `**kwargs` — **Medium**

**File:** `symbulate/results.py:2716` (1D `density` branch), `:3052` (2D violin branch)

Same class of bug as the already-fixed `rug`/`segmented_rug` gap (old Finding 6), just in two branches that gap's fix didn't reach: `.plot(type='density', linewidth=10)` silently no-ops instead of applying or raising, and the 2D violin branch accepts no kwargs at all. A comment near the fixed `rug` branch documents that exact gap was closed there; it just wasn't propagated to these two siblings.

**Fix:** Forward `**kwargs` from both branches, same as every other plot-type branch in the dispatch.

**Suggested test:** `.plot(type='density', linewidth=10)` and the 2D violin equivalent should either apply the kwarg or raise the friendly kwarg-error message — not silently ignore it.

### 21. Division by zero in `Arithmetic.__truediv__`/`__rtruediv__` leaks a raw `ZeroDivisionError` — **Medium**

**File:** `symbulate/base.py:161`

No guard, no explanatory message — a bare `ZeroDivisionError`. `RVResults.standardize()` elsewhere in the codebase already shows the team knows this exact pattern needs guarding (that was old Finding 10); it just wasn't applied here, in the far more commonly hit division operator itself.

**Fix:** Catch the zero-division case and raise a Symbulate-authored message naming what was divided by zero.

**Suggested test:** `RV(Normal(0,1)) / 0` (and the reflected form, `1 / RV(Uniform(0,1))` evaluated where the draw is exactly 0) should raise a clear message.

### 22. `Comparable`'s docstrings claim the wrong return type — **Medium**

**File:** `symbulate/base.py:300`

All six comparison-operator docstrings (`__lt__`, `__le__`, `__eq__`, etc.) describe the return value as an "RV" with 1/0 values; the actual runtime type is an `Event`, which is a different class with different semantics (in particular, it can't be cast to bool at all — chained comparisons deliberately raise). This is a real drift between documentation and behavior in one of the most-used pieces of the package's overloaded-operator surface.

**Fix:** Correct all six docstrings to describe the actual `Event` return type and its `&`/`|`/`~` composition, rather than a plain RV.

**Suggested test:** none needed beyond a docstring fix, though `assert isinstance(X < 3, Event)` would be a reasonable addition to a doctest.

### 23. Distributions error message recommends the removed `pairs=True` keyword as its own fix — **Medium**

**File:** `symbulate/distributions.py:6199` (and the `Dirichlet` docstring, line 8334)

`MultivariateDistribution._plot_pairs`'s "too many variables" exception tells the user to try `.plot(pairs=True, variables=(0, 1, 2))` — but `pairs=` was deliberately removed elsewhere in the package (per CLAUDE.md, a stray `pairs=True` now raises its own error). I confirmed both halves: triggering the too-many-variables path gives the stale suggestion, and literally following it raises a second, different, contradictory error ("pairs= is no longer needed... Drop pairs=True"). This is a regression left over from the `pairs=`-removal refactor that a test never caught (there's a test for the old-keyword-explains-itself case, but nothing exercising the too-many-variables branch itself).

**Fix:** Change the example to `.plot(variables=(0, 1, 2))` (drop `pairs=True`); fix the same stale example in `Dirichlet`'s docstring.

**Suggested test:** trigger the too-many-free-variables path and assert the raised message does not contain `"pairs=True"`.

### 24. `RVConditional.draw()`'s rejection sampler has no cap and can hang indefinitely — **Low-Medium**

**File:** `symbulate/random_variables.py:521-524`

An unbounded `while True:` rejection loop with no iteration cap, unlike comparable guards elsewhere in the package (e.g. `RenewalProcess`'s interarrival-distribution validation). Conditioning on a vanishingly rare or effectively-impossible event doesn't raise — it just appears to hang.

**Fix:** Add a maximum-rejection-count backstop that raises a clear, explanatory error rather than hanging silently.

**Suggested test:** conditioning on a manifestly near-impossible event (e.g. `X > 1e10` for a standard normal) should raise within a bounded time rather than hang indefinitely.

### 25. Assorted Low-severity issues

- **`symbulate/random_variables.py:279-281`** — indexing/slicing a scalar (non-vector) `RV` (`RV(Normal(0,1))[1:]`) raises a raw `TypeError: 'Float' object is not subscriptable` instead of an explanatory message.
- **`symbulate/math.py:401`** — `is_discrete()` leaks a raw `unhashable type` `TypeError` on nested-list input, undocumented in its `Raises` section.
- **`symbulate/base.py:300`** — `Comparable` overrides `__eq__` without defining `__hash__`, silently making `RV`/`RVResults` unhashable (Python sets `__hash__ = None` implicitly), unlike `Tuple`, which defines both. Plausible, not fully confirmed as a live problem — worth a quick check before fixing.
- **`symbulate/index_sets.py:359-380`** — `DiscreteTimeSequence.__getitem__` (and `Integers()[...]`) performs no integer validation, unlike the equivalent check in `result.py`'s `DiscreteTimeFunction`. `DiscreteTimeSequence(4)[2.5]` returns `0.625` — a value that then fails the set's own `__contains__` check, an internal self-inconsistency.
- **`symbulate/result.py:743`** — numeric-vector indexing on `InfiniteTuple` raises a raw, unexplained `TypeError` (related to, but distinct from, Finding 3 above).
- **`symbulate/results.py:3373`** — `marginal=True` passed outside the 2D-plot branch silently no-ops rather than raising.
- **`symbulate/plot.py:2273, 6583`** — `make_tile`/`make_hist2d` docstrings still say axes are labeled "X"/"Y"; the real, correct labels are "Variable 1"/"Variable 2".
- **`symbulate/distributions.py:1945-1958, 3345-3357`** — `Benford`/`InverseGaussian` docstring examples show `from symbulate import *` followed by direct use, but neither class is actually exported yet (a deliberate, in-progress state per the test file's own comments) — the docstring itself has no caveat, so a student following the example verbatim hits a `NameError`.
- **`symbulate/spinner.py:253-256`** — even after fixing Finding 11 above, the multivariate-rejection message states what went wrong but not how to fix it (e.g. "call `.spinner()` on a single component instead").
- **`symbulate/markov_chains.py`** — validation helpers (`_as_rate_array`, `_require_positive`, etc.) consistently raise bare `Exception` rather than `TypeError`/`ValueError`, unlike `hitting_times.py`'s more precise exception types elsewhere in the same subsystem. Messages are fine; only the exception class is generic.
- **`symbulate/diffusion_process.py`** — `get_diffusion_process_result`'s docstring documents an `initial` parameter the function signature doesn't actually have (only `x0`); calling it with `initial=` raises `TypeError`.
- **`symbulate/tests/test_hitting_times.py`** — `upcrossings` has no test for the single-epidemic-compartment case, even though the exact-mirror case is tested for `hitting_time`. I confirmed the runtime behavior is already correct (raises the expected `NotImplementedError`); this is a coverage gap, not a live bug.

---

## Areas that held up well

Worth calling out explicitly, since a lot of new code was added since the last audit and most of it is in good shape: `queues.py` and `poisson_process.py` (Lindley's recursion, `GGs` multi-server overtaking, the three-tier `CoxProcess` cumulative-rate integration, the documented NHPP/Cox clock-time gap) matched every one of their many specific documented invariants with no violations found. `markov_chains.py` and `hitting_times.py` likewise matched their required-interface contract, Tier A/B/C dispatch rules, and the `RV`-vs-`RandomProcess` dispatch subtlety exactly as CLAUDE.md specifies. `gaussian_process.py`'s exactness and `diffusion_process.py`'s `CIR`/`MertonJumpDiffusion` exact special cases (including the jump compensator formula) were verified correct by direct numerical reproduction. `random_walk.py` and `branching_process.py` had no findings at all. And the distribution-level plotting-window laziness, `variables=`/pairs-matrix contract, and marginal-strip conventions in `distributions.py` were all implemented and tested exactly as documented.

---

## Suggested priority order

1. **Fix the conditioning fast-path bug (Finding 1)** — it's silently returning majority-wrong samples from the package's core conditioning operator, with no error, no warning, and a highly plausible-looking (just wrong) result. This is the most urgent thing in either audit so far.
2. **Fix the `InfiniteVector` negative-index nondeterminism and the `Table` arithmetic bug (Findings 2–3)** — both are silent-wrong-answer bugs with no error message, the failure mode this whole audit process is most worried about.
3. **Fix the two GARCH bugs (Findings 4–5)** — one is a formula bug, one is a missing validation/documentation gap mirroring a fix already done for AR/ARMA/MA; both are now the most likely place in the time-series code for a student to get a confidently wrong answer.
4. **Fix the cross-cutting `Statistical` mixin bug (Finding 6)** and the multivariate-distribution raw-exception issues (Findings 7, 11) — all three are "crashes instead of works" bugs with an easy, well-precedented fix (the pattern already exists elsewhere in the same file/package).
5. **Sweep the `BoxModel` validation gaps (Findings 10, 17, 18)** — same underlying pattern as the already-fixed empty-box/negative-size issue, just triggered by different inputs; a natural single follow-up PR to the recent fix.
6. Everything else (Findings 8–9, 12–25, plus the still-open items from the last audit) is worth working through, but none of it is blocking.
