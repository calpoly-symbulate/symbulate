# Formatting `Results`/`RVResults` display output

**File touched:** `symbulate/results.py` (no other files changed)
**Patch:** `results_display_formatting.patch` (unified diff against `dev`)

## Problem

`X.sim(10000)` (and any `Results`/`RVResults`) printed via `__repr__` (plain
text, e.g. terminal/`print()`) or `_repr_html_` (Jupyter) with no numeric
rounding and no column alignment:

```
Index  Result
0      0.8374857293847221
1      -1.204981723098213
...
```

Floats showed full Python precision, nothing lined up, and vector-valued
outcomes (e.g. `(X & Y).sim(n)`) printed as one ragged tuple string per row.

## What changed

Four new module-level helpers in `symbulate/results.py`, plus a rewrite of
`Results.__repr__`, `Results._repr_html_`, and a new shared
`Results._display_rows()` helper. `RVResults` inherits all of this
unchanged (it never overrode `__repr__`/`_repr_html_`).

- **`RESULT_DISPLAY_DECIMALS = 4`** — module constant controlling how many
  decimal places floats are rounded to for display.
- **`_format_scalar(value, decimals)`** — formats one numeric scalar.
  Floats get fixed decimal places (`f"{value:.4f}"`); ints and bools are
  left exactly as-is (they're typically counts or category codes, not
  measured quantities). Fixed decimal places is what makes right-justified
  columns line up on the decimal point: the distance from the decimal
  point to the right edge of the string is constant, so once the column is
  right-justified to a common width, the points align automatically — no
  need to hunt for the decimal point and pad around it.
- **`_format_result(value, decimals)`** — formats one full outcome.
  Delegates to `_format_scalar` for numeric scalars; for numeric-vector
  outcomes (`Tuple`/`Vector`, e.g. `(1.5, 20)`), formats each component and
  rejoins as `"(comp0, comp1, ...)"`; anything else (categorical outcomes,
  `TimeFunction`s) falls back to `str(value)` unchanged.
- **`_format_display_column(values, decimals)`** — formats *and aligns* an
  entire column. When every value is a numeric vector of the same length,
  each component position is aligned independently (component 0 padded to
  its own widest formatted value across all rows, component 1 to its own,
  etc.), so a column of `(X, Y)` pairs reads as two neat sub-columns
  instead of one ragged tuple. The whole column is then right-justified to
  one common width.
- **`Results._display_rows()`** — a new shared helper returning exactly the
  rows `__repr__`/`_repr_html_` will show: all rows when there are ≤ 11,
  otherwise the first 9 plus the last, with an `("...", None)` marker in
  between. Used by both display methods so they can't disagree.

## Behavioral changes / fixes

- **Numeric rounding.** Floats display at 4 decimal places by default;
  ints/bools are untouched.
- **Column alignment.** The Result column in the plain-text `__repr__` is
  right-justified to a common width (and each component of a vector
  outcome is separately right-justified within its own sub-column). The
  Index column is left-justified.
- **HTML alignment.** `_repr_html_`'s Result `<td>` cells get
  `style="text-align: right"` and the same rounding, so the notebook table
  matches the plain-text one. (Space-padding, unlike in the plain-text
  case, would be collapsed by HTML rendering, so alignment there is done
  with CSS instead of literal padding.)
- **Fixed a latent `__repr__`/`_repr_html_` discrepancy.** The old
  `_repr_html_` inserted a `"..."` row whenever `len(self) > 9`, even at
  exactly 10 or 11 results where nothing was actually omitted (and for
  n=11 it silently *dropped* index 9's row instead of showing it). The old
  `__repr__` handled n=10/11 correctly. Both now use the same
  `_display_rows()` logic, so this inconsistency is gone.
- **Ellipsis marker simplified.** The old dot-fill ellipsis (a string of
  dots matching the *length* of the last row's raw string) is now a plain
  `"..."`, consistent with how `Table.__repr__` already denotes truncation
  elsewhere in the codebase.

## Example (before → after)

```python
>>> from symbulate import *
>>> X = RV(Normal(0, 1))
>>> X.sim(6)
```

Before:
```
Index  Result
0      1.0135476234981723
1      -0.502634981723094
2      0.804812309487123
3      -0.0750128374981723
4      -0.9550123748912374
5      1.8740981723094871
```

After:
```
Index Result
0     1.0135
1     -0.5026
2      0.8048
3     -0.0750
4     -0.9550
5      1.8740
```

Two-dimensional outcomes now align per component:

```
Index               Result
0     (  1.5000,   1.5000)
1     (300.1000, 300.1000)
2     ( 22.2500, 300.1000)
```

## Testing performed

- Ran the existing `symbulate/tests/test_results.py` suite (122 tests) —
  all pass unchanged, including the `TestResultsRepr` class that checks
  header content, truncation, and the "no ellipsis at exactly 11" case.
- Ran `pytest --doctest-modules symbulate/results.py` — all doctests in
  the file pass except one pre-existing, unrelated failure in
  `Results.filter`'s docstring (`sims` used without being defined in that
  example — confirmed present on `dev` before this change via `git
  stash`, not introduced by this patch).
- Manually smoke-tested 1D discrete, 1D continuous (small and 10,000-row
  truncated), 2D vector outcomes, and categorical (non-numeric) `Results`
  from a probability space, in both `repr()` and `_repr_html_()`.

## Out of scope / not changed here

- `Table.__repr__`/`_repr_html_` (the display for `.tabulate()` output) —
  untouched. It already does its own (simpler) alignment; a follow-up
  could apply `_format_scalar` there too for numeric outcome keys.
- Decimal precision is a module constant, not a per-call keyword (e.g. no
  `X.sim(10000).__repr__(decimals=2)`); easy to add later if wanted.
- No scientific-notation fallback for very large or very small magnitudes
  — `f"{value:.4f}"` on something like `1e-8` prints `0.0000`, losing the
  value. Not addressed here since typical teaching-oriented simulated
  values (dice, normal, exponential, etc.) don't hit this in practice, but
  worth flagging if `.summarize()` or other work later surfaces
  extreme-magnitude distributions.
