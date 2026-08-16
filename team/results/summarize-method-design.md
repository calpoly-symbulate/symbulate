# Design sketch: `RVResults.summarize()`

Goal: `X.sim(10000).summarize()` returns a readable summary appropriate to
the shape/type of `X`'s simulated values, covering the four cases below.
This is a design sketch, not implemented code — it maps out what exists
already, what's new, and how the pieces fit together.

## What already exists (reusable as-is)

All from `symbulate/base.py`'s `Statistical` mixin, already available on
`RVResults`:

| Method | Behavior on `dim==1` | Behavior on `dim>1` |
|---|---|---|
| `.mean()` | `Scalar` | `Vector` (one value per component) |
| `.std()` / `.sd()` | `Scalar` | `Vector` |
| `.quantile(q)` | `Scalar`, single `q` in `[0,1]` | `Vector`, single `q` |
| `.corr()` / `.corrcoef()` | raises (needs ≥2 dims) | scalar for `dim==2`, matrix for more |
| `.median()`, `.var()`, `.iqr()`, `.skew()`, `.kurtosis()`, `.min()`, `.max()` | same pattern | same pattern |

From `symbulate/results.py`:

- `.tabulate(normalize=True, bin=..., nbins=..., binwidth=...)` — exact
  counts by default, or equal-width binned counts with `bin=True`.
- `.plot()` — with no arguments, already picks a sensible default type via
  `classify_values()` (`symbulate/plot.py:984`), which independently
  determines discreteness and small-vs-large sample size per axis. This is
  the exact classifier `.summarize()` should reuse for branching, rather
  than inventing a second one.

From the display-formatting patch (`results_display_formatting.patch`):

- `_format_scalar` / `_format_result` / `_format_display_column` — decimal
  rounding and column alignment. `.summarize()`'s own output tables should
  reuse these rather than reinvent formatting a third time.

## What's genuinely new

1. **Multi-quantile in one call.** `.quantile(q)` only accepts a scalar
   `q`; passing a list breaks its `0 <= q <= 1` bounds check. `.summarize()`
   needs deciles (0.1, ..., 0.9) plus the tail quantiles (0.01, 0.05, 0.25,
   0.75, 0.95, 0.99). Cheapest path: bypass `.quantile()` and call
   `np.quantile(self.array, qs)` directly against the already-cached
   array (`self._set_array()`), sidestepping the single-`q` restriction
   entirely rather than looping 13 individual `.quantile()` calls.
   (Note: `symbulate/math.py` has standalone `deciles(x)` / `quartiles(x)`
   functions, but they take a plain iterable, not an `RVResults` — not
   directly reusable as a method call.)

2. **ECDF-at-jumps table (single discrete case).** No existing helper
   computes this, but it's a trivial derivation from `.tabulate(normalize=True)`:
   sort the outcomes, cumulative-sum the relative frequencies. A few lines,
   not a new subsystem.

3. **Two-way (contingency) table (two discrete case).** `.tabulate()` on a
   2D discrete `RVResults` returns a flat `Table` keyed by `(x, y)` tuples
   — a single column of pairs, not a grid. Pivoting that into rows-by-columns
   needs: the two sorted unique-value sets, a matrix built from the existing
   `(x,y)` counts dict, and a new display class with a matrix-style
   `__repr__`/`_repr_html_` (row/column headers, optional margins). This is
   the single biggest net-new piece. See "Open questions" below for the API
   shape (new `tabulate(twoway=True)` keyword vs. a separate
   `.crosstab()` method).

4. **A `Summary` display object.** Something has to hold and print the
   assembled results — mean, sd, quantiles/corr, a table, and (optionally)
   an embedded note about what `.plot()` would show. Proposed: a small
   class (parallel to `Table`) with `__repr__`/`_repr_html_`, built once
   per `.summarize()` call, not reused elsewhere.

## Proposed dispatch

```python
def summarize(self, plot=True):
    self._set_array()
    if self.dim == 1:
        values = self.array
        discrete, _ = classify_values(values, n_unique_threshold=B_1D)
        return self._summarize_1d(discrete, plot)
    elif self.dim == 2:
        x, y = self.array[:, 0], self.array[:, 1]
        discrete_x, _ = classify_values(x, n_unique_threshold=K_2D, large_n_rescue=False)
        discrete_y, _ = classify_values(y, n_unique_threshold=K_2D, large_n_rescue=False)
        if discrete_x and discrete_y:
            return self._summarize_2d_discrete(plot)
        elif not discrete_x and not discrete_y:
            return self._summarize_2d_continuous(plot)
        else:
            raise NotImplementedError("Mixed discrete/continuous pairs not yet covered.")
    else:
        raise ValueError(".summarize() supports 1 or 2 dimensions today; "
                          "for 3+, summarize each variable or pair separately.")
```

(Mirrors the same `classify_values` calls `RVResults.plot()` and
`_pairs_joint_configuration()` already make for the 2D case — see
`symbulate/results.py`'s `_pairs_joint_configuration`.)

### Case 1: single discrete

```
X.sim(10000).summarize()

Mean:  3.512
SD:    1.706

Value  Rel. Freq  Cumulative
1      0.1668     0.1668
2      0.1701     0.3369
3      0.1636     0.5005
4      0.1665     0.6670
5      0.1653     0.8323
6      0.1677     1.0000

[default plot shown / described]
```

Built from: `.mean()`, `.sd()`, `.tabulate(normalize=True)` (plus a
cumulative-sum column derived from it), `.plot()`.

### Case 2: single continuous

```
X.sim(10000).summarize()

Mean:  0.0041
SD:    0.9987

Quantiles:
  0.01    0.05    0.10    0.25    0.50    0.75    0.90    0.95    0.99
-2.331  -1.638  -1.279  -0.668   0.006   0.679   1.288   1.634   2.328

[default plot shown / described]
```

Built from: `.mean()`, `.sd()`, `np.quantile(self.array, qs)` for the 13
requested probabilities (see "genuinely new" #1), `.plot()`.

### Case 3: two discrete

```
X.sim(10000).summarize()

Mean:  [3.48, 2.01]
SD:    [1.70, 0.82]
Corr:  0.134

        y=1     y=2     y=3
x=1   0.0410  0.0532  0.0398
x=2   0.0601  0.1023  0.0587
x=3   0.0589  0.0980  0.0602
...

[default plot shown / described]
```

Built from: `.mean()` / `.sd()` (already return `Vector`s for `dim==2`),
`.corr()` (already scalar for `dim==2`), the new two-way table (see
"genuinely new" #3), `.plot()`.

### Case 4: two continuous

```
X.sim(10000).summarize()

Mean:  [0.02, 100.14]
SD:    [0.99, 4.98]
Corr:  0.312

Marginal quantiles:
             0.01    0.05    0.10    0.25    0.50    0.75    0.90    0.95    0.99
Variable 1  -2.33   -1.64   -1.28   -0.67    0.01    0.67    1.28    1.62    2.31
Variable 2  92.40   91.79   93.75   96.62  100.05  103.35  106.51  108.11  111.63

[default plot shown / described]
```

Built from: `.mean()` / `.sd()` (`Vector`s), `.corr()` (scalar), marginal
quantiles per component via `np.quantile(self.array, qs, axis=0)`, `.plot()`.

## Effort estimate

| Piece | Est. effort | Why |
|---|---|---|
| Dispatch + 1D discrete + 1D continuous | ~2–3 hrs | Pure glue: existing methods, existing classifier, new ECDF-cumsum (trivial) and multi-quantile call (bypasses one restriction). |
| 2D discrete summary (excluding the two-way table itself) | ~1 hr | `.mean()`/`.sd()`/`.corr()` already return the right shapes. |
| 2D continuous summary | ~1 hr | Same reasoning; marginal quantiles via `axis=0`. |
| Two-way/contingency table + display class | ~half day | The one genuinely new subsystem — building the pivoted matrix and a matrix-aware `__repr__`/`_repr_html_` (reusing alignment helpers from the display-formatting patch). |
| `Summary` display object (shared across all 4 cases) | ~2–3 hrs | Mostly formatting/composition once the pieces above exist. |
| Docstrings, doctests, `See Also` cross-refs (matching existing codebase style) | ~2–3 hrs | Every method in this codebase carries full NumPy-style docs; `.summarize()` should match. |
| Tests (unit tests per case + edge cases: empty results, `dim` mismatches, mixed discrete/continuous) | ~half day | `test_results.py` already has clear patterns to extend. |
| **Total** | **~1.5–2.5 days** | Assuming mixed discrete/continuous pairs and `dim > 2` are explicitly out of scope (raise a clear "not yet supported" error) rather than also being designed now. |

## Open questions to settle before implementing

1. **Two-way table API surface.** New keyword on `.tabulate()`
   (`tabulate(twoway=True)`) so it stays discoverable and doesn't silently
   change existing return types for 2D data? Or a dedicated
   `.crosstab()` method, closer to `pandas.crosstab`? The latter also
   raises the question of whether to support `normalize="rows"` /
   `"columns"` / `"all"` the way pandas does, or just raw counts.
2. **Does `.summarize()` call `.plot()` (side-effecting, opens a
   matplotlib figure) or just describe what `.plot()` would show?**
   Calling it directly is more useful interactively but means
   `.summarize()` has a plotting side effect baked into a
   stats-and-tables method, which is a different contract than every
   other method in `Statistical`. Consider a `plot=True` keyword so it's
   optional.
3. **Return type.** A printable `Summary` object (parallel to `Table`,
   with `__repr__`/`_repr_html_`) is more consistent with how the rest of
   the codebase handles compound output, but a plain `dict` might be more
   useful to script against downstream. Possibly both: return a `Summary`
   object that's also a `dict`-like container (the way `Table` subclasses
   `dict`).
4. **Precision.** Should `.summarize()`'s tables share the module-level
   `RESULT_DISPLAY_DECIMALS` constant from the display-formatting patch,
   or take their own (summaries may want more precision than a raw
   simulation dump, e.g. correlation to 3 decimals rather than 4)?
5. **Mixed discrete/continuous pairs and 3+ dimensions.** Not in the
   original four cases. Worth deciding now whether `.summarize()` should
   raise a clear, actionable error for these (recommended for a first
   version) or attempt a reduced summary (e.g. per-variable stats only,
   no joint table).

## Suggested phased rollout

- **Phase 1:** single discrete + single continuous (`dim == 1`). Smallest
  surface, no new display subsystem needed beyond the ECDF-cumsum table.
- **Phase 2:** two discrete + two continuous (`dim == 2`), which requires
  building the two-way table and settling question 1 above first.
- **Phase 3:** polish — shared `Summary` object, HTML rendering, decide on
  and document behavior for mixed/3+-dimension inputs (even if that
  behavior is just a good error message).
