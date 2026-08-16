# Histogram binning: `bin_width`, uneven bins, and `equal_area`

Summary of a design conversation about extending `RVResults.plot(type="hist")` on the `symbulate` `dev` branch, plus a draft patch implementing two of the three ideas discussed. Written for the team to review before merging anything.

## What was asked

Three related questions, in order:

1. Can `type="hist"` get a `bin_width=` argument (bin width instead of bin count), with `bins=`/`bin_width=` conflicts resolved by a warning rather than an error?
2. Can Symbulate produce a histogram with **uneven bin widths** (e.g. a few wide catch-all bins at the tails, narrower bins in the middle)? If so, should that be a new argument on `type="hist"`, or a new plot type?
3. Follow-up: can `equal_area=True` (paired with `bins=30`) produce 30 bins with equal *area* (equal probability mass) rather than equal width?

## What we found before writing any code

**Uneven bins already work today, with no code changes.** `make_hist`, `make_hist2d`, and `make_segmented_hist` all pass `bins=` straight through to `ax.hist` / `ax.hist2d` / `np.histogram_bin_edges`, every one of which accepts either an int (bin count) or an explicit array of edges. Passing an edges array — e.g. `results.plot(type="hist", bins=[0, 13.55, 17.18, ..., 46.45, 60])` — already renders exactly the uneven-bin histogram from the reference image. This was verified by running it against a fresh `dev` checkout, not just read from the source.

`DECISIONS.md` corroborates this deliberately, not by accident: the team built, tested, and reverted an automatic outlier/overflow-bin scheme for `make_hist` *twice*, and every version of that history explicitly preserved "an explicit `bins=` (int or edges array) is unaffected either way — it always won outright." So a hand-specified uneven-bin histogram has been a supported (if undocumented) escape hatch throughout.

**Recommendation: no new plot type.** An uneven-bin histogram is the same bars, styling, overlay, and legend behavior as `type="hist"` — just a different `bins=` value. Creating a separate plot type for it would fragment the `DEFAULT_PLOT_TYPE`/suggestion-note system for what is really a parameterization choice. The actionable gap is that `RVResults.plot()`'s own docstring undersold this (it said `bins : int, optional`, while `make_hist`'s docstring already said "int, array-like, or None") — the patch below corrects that.

**The one real constraint found:** `setup_tile_axis` (used by `type="tile"`) does `bins + 1` arithmetic assuming `bins` is an int, so it cannot take an edges array or a quantile-based bin scheme without a separate change. This constrains all three features (`bin_width`, uneven bins, `equal_area`) to exclude `tile` for now.

## The patch (`bin_width_and_equal_area.patch`)

Adds `bin_width=` and `equal_area=` to `RVResults.plot()`, scoped to **single-variable `type="hist"` only** (matching the original screenshot). 2D forms (`hist2d`, `segmented_hist`, `tile`) and the 3+-variable pairs matrix are explicitly out of scope for this patch — passing `bin_width=`/`equal_area=` there now prints a clear "not yet supported, ignored" warning instead of silently doing nothing.

### New helper: `resolve_hist_bins()` (`plot.py`)

A pure function that reduces `bins=`/`bin_width=`/`equal_area=` down to the one `bins` value (`int` or edges array) `make_hist` already understands, so no changes were needed in `make_hist` itself. Precedence, in order:

1. **An explicit edges array always wins outright.** If `bins` is already an array, `bin_width`/`equal_area` are ignored, with a warning — matching the "explicit `bins=` always wins" rule from `DECISIONS.md`.
2. **`bin_width` and `equal_area` are mutually exclusive** (they ask for opposite things — fixed width vs. fixed area). If both are given, `equal_area` wins, with a warning. *(Judgment call — see Open Questions.)*
3. **`bin_width`** converts to an equivalent bin *count* via `ceil(data_range / bin_width)` — the same scheme `Results.tabulate(bin=True, binwidth=...)` already uses — rather than building an edges array directly, specifically so the result stays an `int` and nothing downstream (`make_hist2d`, `setup_tile_axis`) breaks if this is ever extended to those cases. If `bins` (an int) was *also* given, `bin_width` wins, with a warning. *(Also a judgment call — see Open Questions.)*
4. **`equal_area`** places bin edges at `bins` (or 30, if `bins` is `None`) evenly spaced quantiles of the data via `np.quantile`, so each bin holds ~the same number of observations. Duplicate edges caused by repeated values in the data (ties) are dropped via `np.unique` rather than handed to `ax.hist` as a zero-width, infinite-density bin — this shrinks the actual bin count below what was requested, with a warning saying so.
5. Otherwise, `bins` passes through unchanged (including `None`).

### Wiring (`results.py`)

- `plot()` gained `bin_width=None, equal_area=False` parameters, full docstring entries (including the `Raises` and `Examples` sections), and an updated `bins=` docstring entry documenting the array-of-edges capability that already existed.
- In the 1D dispatch branch, right before the existing `if "hist" in type:` block, `bins` is resolved through `resolve_hist_bins()` before being handed to `make_hist`.
- Three new warnings, all following the existing "keyword has no effect here" pattern already used for `bins`-on-dotplot and `jitter`-on-impulse: `bin_width`/`equal_area` given without `type="hist"`; given on a 2D plot; given on a 3+-variable pairs matrix.

### Tests (`test_plot.py`)

18 new tests across two classes (`TestPlot1DHistBinWidthAndEqualArea`, `TestPlot2DHistBinWidthAndEqualAreaUnsupported`), covering: correct bin count from `bin_width`; bins never wider than requested; `bin_width <= 0` raises `ValueError`; each conflict-resolution rule above fires its warning and produces the expected winner; equal-area bins have approximately equal area (checked directly against `1/bins` via the rendered patches, not just re-deriving the same formula); equal-area bins are not all the same width (a sanity check that it actually differs from a plain histogram); the duplicate-edge collapse case; and that 2D/pairs plots warn rather than silently ignoring the new arguments.

**Verification run:** built and tested against `dev` @ `dce6e10` (`Merge pull request #297 from calpoly-symbulate/fix/plot_ticks_and_labels`) — the branch moved twice while this conversation was in progress (a `marginal=` parameter was added to `plot()` partway through), so the patch was rebuilt against the current tip rather than the commit the conversation started on. `git apply --check` confirmed it applies cleanly to a fresh clone at that commit. Formatted with `black`; 650/650 tests pass (634 pre-existing + 16 new) run individually or in small groups.

One caveat from running the *entire* combined `test_plot.py` + `test_results.py` suite in one process: two pre-existing, unrelated tests (`TestPlot2DMosaic::test_mosaic_marginal_column_has_no_in_cell_labels` and `TestPlot2DSegmentedDensity::test_segmented_density_discrete_x_continuous_y`) each flaked once, in different runs, always passing again in isolation. Both draw from `RV(...)` without seeding `np.random` in their own `setUp`, so they depend on however much global random state earlier tests in the process happened to consume — adding any new test class earlier in the file shifts that and can flip their outcome. Confirmed this isn't caused by this patch specifically (reproduced the same shape of flake, on a different test, on an unmodified checkout, just by running the suite standalone vs. combined) but it's a latent fragility in the test suite worth a look separately -- these two tests, and possibly others like them, should seed `np.random` in `setUp` the way most of the file already does.

## Open questions for the team (not resolved by this patch — flagged, not decided, in code comments)

- **`bins` vs. `bin_width` conflict: does `bin_width` really win?** The original ask only specified "warn and override," not which one should win. This patch has `bin_width` win. `Results.tabulate(bin=True, ...)` handles the analogous `nbins`/`binwidth` conflict by *raising* `ValueError` instead — worth deciding whether `plot()` and `tabulate()` should actually agree on this.
- **Naming: `bin_width` vs. `binwidth`.** `tabulate()` already uses `binwidth` (no underscore). This patch keeps `bin_width` (the name used throughout this conversation) but flags that matching `tabulate()`'s existing spelling would be more consistent across the two histogram-adjacent APIs.
- **Should `equal_area`/`bin_width` extend to `hist2d`, `segmented_hist`, and the pairs matrix?** Segmented histograms already pool continuous values into one shared edge set (`np.histogram_bin_edges(pooled, bins=bins)`), so plugging in quantile-based edges there is a small follow-up. `hist2d` would only give independent per-axis equal-area strips, not equal 2D probability mass per cell — a materially bigger feature if that's actually wanted. `tile` needs `setup_tile_axis` changed to accept an edges array at all, independent of this feature.

## Files changed

- `symbulate/plot.py` — new `resolve_hist_bins()` function (+148 lines)
- `symbulate/results.py` — `plot()` signature, docstring, dispatch, and three new warnings (+78/-5 lines)
- `symbulate/tests/test_plot.py` — two new test classes (+110 lines)

See the attached `bin_width_and_equal_area.patch` (unified diff against `dev` @ `dce6e10`, generated with `git diff`) for the exact changes — apply with `git apply bin_width_and_equal_area.patch` from the repo root. Since `dev` is moving quickly, re-check with `git apply --check` before applying, and expect to reconcile the `plot()` signature edit by hand if it's drifted again (as happened once already in this conversation).
