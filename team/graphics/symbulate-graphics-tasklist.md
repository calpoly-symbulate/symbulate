# Symbulate Graphics (dev branch): Task List

Companion to the full revisions document. Same priority order; each task lists what
to do, why, and a starter prompt for Claude Code.

---

### 1. Rename `classify_data` → `classify_values`
- **What:** Rename the function (and update its ~114 references — mostly docstrings, comments, and tests) in `plot.py`, `results.py`, `test_plot.py`.
- **Why:** It classifies a raw array of values, not a Symbulate results object, so `classify_values` is more accurate. It's internal-only, so safe to rename.
- **Prompt:** *"Rename `classify_data` to `classify_values` throughout the symbulate package, including docstrings, comments, and tests. It's not part of the public API re-exported in `__init__.py`, so this should be a straightforward find-and-replace — but check each hit, since some are prose references (e.g. 'see classify_data()')."*

---

### 2. Fix discreteness threshold instability at large `n`
- **What:** Add a large-`n` secondary check so genuinely discrete distributions with moderate support (e.g. `Poisson(15)`, `Binomial(70, 0.5)`) don't flip between discrete/continuous based on random seed.
- **Why:** `n_unique <= B_1D` (30) is a hard cutoff; both example distributions land right at that boundary, so their default plot type is currently a near coin-flip. Main's old `is_discrete` (repeat-density based) was more robust here, but was replaced because it misclassified small-n discrete data and rounded floats as continuous/discrete respectively — any fix needs to gain large-n robustness without reintroducing those failure modes.
- **Example:** `RV(Poisson(15)).sim(10000).plot()` should reliably give an impulse plot, not sometimes a histogram.
- **Prompt:** *"In `classify_data`/`classify_values` (plot.py), add a large-n secondary rule inspired by main's old `is_discrete` (fraction of distinct values with count > 1), but bound it by its own unique-value ceiling rather than applying it unconditionally — an unconditional repeat-density check would reclassify wide-support rounded-float or genuinely continuous data as discrete, which is the exact failure mode `is_discrete` was replaced for. Only apply this large-n clause when `n` is large enough for repeats to be a reliable signal. Add tests confirming: (a) `Poisson(15)` and `Binomial(70, 0.5)` at n=10000 reliably classify as discrete-ish across multiple seeds, and (b) a rounded-float continuous variable at large n still classifies as continuous, i.e. this doesn't regress the failure mode `is_discrete` was replaced for."*

---

### 3. Don't let 2-D tile crowding change the discrete/continuous verdict
- **What:** Decide explicitly that `K_2D` should not be tightened just to reduce tile clutter — crowding should be fixed by rendering (see #4), not reclassification.
- **Why:** Tightening `K_2D` would call a genuinely discrete variable "continuous" just to make a busier-looking plot smaller — the wrong tradeoff.
- **Prompt:** *(No code change — a design decision. If revisiting later: "Confirm `K_2D` stays at its current value / anchor; do not adjust it as a fix for tile-plot crowding — that should be handled in `make_tile`'s tick rendering instead.")*

---

### 4. Fix discrete-axis tick crowding in 2-D plots (tile, segmented rug/box/violin, and any other plot with a discrete axis)
- **What:** Thin forced tick labels on a discrete axis in any 2-D plot when it has many distinct values (currently every single value gets a forced label, no matter how many). Confirmed in `_setup_tile_axis` (used by tile) and the segmented rug/box/violin tick-labeling code, but the underlying issue is general to any 2-D plot type with at least one discrete axis, not specific to tile.
- **Why:** Any of these functions always builds one tick per distinct value; with 25–30+ values (common right at the `K_2D` boundary) labels overlap and become unreadable. The reference fork already solved this once for tile (`reduce_ticks`, thins to ~20/~15 ticks) — dev's rewrite dropped it, and the same fix is needed everywhere a discrete axis appears, not just there.
- **Prompt:** *"Across the 2-D plot functions in plot.py that label a discrete axis (`_setup_tile_axis` for tile, and the segmented rug/box/violin tick-labeling code), thin the tick labels shown when an axis has more than ~15-20 distinct values, instead of forcing one label per value. Look at the reference fork's `reduce_ticks` function for prior art, but implement it working from the underlying values/positions rather than re-parsing rendered label text, and apply it consistently across all of these functions rather than just tile. Alternatively, consider positioning discrete cells at their real numeric values (like the continuous branch's `extent` already does) instead of compressed index positions, so matplotlib's normal tick locator can thin automatically — this would also help item 8 (marginal plot alignment)."*

---

### 5. Make histogram binning outlier/skew-aware
- **What:** Replace the fixed 30 equal-width bins spanning raw min–max with something that handles heavy-tailed data.
- **Why:** `RV(F(5,4)).sim(10000)` has a max of ~164 but 99% of data below ~16 — the current histogram crushes the real shape into 1-2 bins. Plain Freedman-Diaconis alone isn't enough either (it implies ~935 bins over the full raw range).
- **Prompt:** *"In `make_hist` (plot.py), replace the fixed-bin-count equal-width histogram with one that pairs an IQR/Freedman-Diaconis-informed bin width with a percentile- or IQR-based range clip (plus an overflow/last-bin for outliers, or an optional log-scale mode). Test against `RV(F(5,4)).sim(10000)` and confirm the bulk of the distribution's shape is no longer compressed into 1-2 bins."*

---

### 6. Re-anchor (or explicitly decouple) discreteness thresholds from bin count
- **What:** If #5 makes bin count dynamic, make sure `B_1D`/`K_2D` (currently deliberately equal to `HIST_DEFAULT_BINS`) don't silently break or become nonsensical.
- **Why:** `DECISIONS.md` ties the 30/30/30 values together on purpose ("a discrete plot should stay discrete exactly while no finer than the histogram it would become"). A dynamic, outlier-inflated bin count (e.g. 935) must not be reused directly as the discreteness cutoff.
- **Prompt:** *"After changing histogram bin-count logic (task 5), check every place `B_1D`/`K_2D`/`HIST_DEFAULT_BINS`/`HIST2D_DEFAULT_BINS`/`TILE_DEFAULT_BINS` are used together or assumed equal, per DECISIONS.md's 'classify_data Thresholds (Budget Model)' section. Explicitly decide and document whether the discreteness cutoff still equals the (now dynamic) bin count or is decoupled into its own fixed value."*

---

### 7. Cap dot-plot stack height - POSSIBLY NOT NECESSARY, BUT READ THE COMMENT
- **What:** Add a crowding check so a dot plot doesn't stack too many dots at one value.
- **Why:** `RV(Binomial(1, 0.1)).sim(111)` with a raised small-n threshold produces a ~100-dot stack at x=0 next to an 11-dot stack at x=1 — lopsided and hard to read, because dot plots never bin and have no per-stack cap.
- **Prompt:** *"In `make_dotplot` (plot.py), add a check on the tallest single stack of dots (independent of total n or number of unique values) and fall back to a different default (e.g. impulse) when it's exceeded. Test with `RV(Binomial(1, 0.1)).sim(111)`."*
- **Comment:** I want to increase the 1D threshold to something like 123 instead of 100, regardless of above. I don't think the lopsided-ness of `RV(Binomial(1, 0.1)).sim(99)` is too bad, but I didn't check too many cases where there is a single stack with many dots. Worth doing a few more checks, but this isn't something you should spend a lot of time on. Just increase the small n threshold from 100 to 123 and check a few cases.

---

### 8. Rebuild marginal plots and fix marginal coordinate/bin mismatches across 2-D plot types
- **What:** Marginal panels (`"marginal" in type`) still use old, pre-redesign helpers (`make_marginal_impulse`, `compute_density`, raw `ax.hist` calls) instead of the redesigned `make_impulse`/`make_density`/`make_hist`. Also fix a real, general bug: whenever a 2-D plot's main panel and its marginal panel(s) use different coordinate systems or independently-computed bins, they can visually misalign. Confirmed concretely for tile+marginal impulse (a discrete axis); the same class of problem needs checking across every 2-D+marginal combination — 2D hist + marginal hist, 2D density + marginal density, 2D tile + marginal impulse, 2D scatter + marginal dot, mixed tile/hist + impulse/hist, etc.
- **Why:** Confirmed the marginal code is byte-for-byte identical to the old reference fork — never touched by the redesign. Confirmed the coordinate-mismatch bug directly for tile: `RV(DiscreteUniform(50,60) * Poisson(5)).sim(2000).plot(type=["tile","marginal"])` gives the main panel an x-axis of `(-0.5, 10.5)` (compressed index positions — tile's discrete axis convention) and the marginal panel `(49.5, 60.5)` (real values) — completely different scales, no `sharex`/`sharey` anywhere. This specific failure mode (index-based main axis vs. real-valued marginal axis) applies to any main-panel plot type that uses tile's index-position convention for a discrete axis (tile itself, and mixed discrete×continuous configurations), paired with a marginal panel that plots at real values. For fully continuous 2-D plots (hist2d+hist marginal, density2d+density marginal), the risk is different in kind: both panels already use real-value coordinates, so there's no index-vs-value mismatch, but bin edges between the main panel and its marginal are only guaranteed to match today because `HIST_DEFAULT_BINS` and `HIST2D_DEFAULT_BINS` happen to both equal 30 (not because anything enforces it), and an explicit `bins=` override isn't verified to thread through identically to both. Scatter+marginal dot plot is lower-risk (scatter already uses real values with no binning/compression) but should still be confirmed once dot-plot marginals exist (see task 30/routing marginal type choice through `classify_data`).
- **Prompt:** *"Rebuild the marginal-panel rendering in the 2-D branch of `RVResults.plot()` (results.py) to call `make_impulse`, `make_density`, and `make_hist` instead of the legacy `make_marginal_impulse`/`compute_density`/raw `ax.hist`. Route the marginal panel's plot-type choice through `classify_data`/`default_plot_type` instead of hardcoding impulse-or-hist. Fix the coordinate/bin mismatch between each 2-D main panel and its marginal panel(s), checking every combination: tile+marginal impulse (confirmed index-vs-real-value mismatch — this may depend on resolving task 4 first, if tile switches to real-valued cell positions), hist2d+marginal hist and density2d+marginal density (verify bin edges are guaranteed to match, not just coincidentally equal via shared default constants — especially when `bins=` is explicitly overridden), and scatter+marginal dot (confirm real-value alignment holds once dot-plot marginals are implemented). Link main and marginal axes with `sharex`/`sharey` wherever coordinate systems genuinely match. Test with `RV(DiscreteUniform(50,60) * Poisson(5)).sim(2000).plot(type=['tile','marginal'])` as the primary regression case."*

---

### 9. Change marginal from `type=[...]` to `marginal=True`
- **What:** Make `marginal` a separate keyword argument instead of a string inside `type=`.
- **Why:** It's a modifier (adds side panels to whatever the main plot type is), not a peer of `"tile"`/`"hist"`/`"scatter"` — every other modifier (`bins=`, `jitter=`) is already its own keyword.
- **Prompt:** *"In `RVResults.plot()` (results.py), replace the `'marginal' in type` checks with a separate `marginal=True/False` keyword argument. Update the type validation error message and docstring accordingly. Do this alongside task 8's rebuild, since the same code is being touched."*

---

### 10. Audit and improve `Distribution.plot()` x-axis limits
- **What:** Confirm the bounded-vs-unbounded xlim rule is intentional (it is — full audit found bounded distributions show true full support, unbounded ones use an equal-tailed `ppf(.001,.999)` window) and switch unbounded distributions to a proper highest-density interval, at least for visibly skewed ones.
- **Why:** Equal-tailed windows waste space on skewed unbounded distributions (`F`, `Gamma`, `Pareto`, `LogNormal`, `ChiSquare`) the same way task 5's histograms do. Keep full-range display for bounded distributions — hiding real, always-possible outcomes at a probability cutoff would be misleading.
- **Prompt:** *"In distributions.py, keep the existing full-support xlim for bounded distributions (Bernoulli, Binomial, Hypergeometric, DiscreteUniform, Uniform, Beta) unchanged. For unbounded discrete distributions (Poisson, Geometric, NegativeBinomial, Pascal), replace the equal-tailed ppf(.001,.999) default with an exact highest-density interval (sort pmf descending, accumulate to target coverage). For skewed unbounded continuous distributions (F, Gamma, Pareto, LogNormal, ChiSquare), implement an HDI via root-finding the equal-density points enclosing the target coverage (scipy.optimize.brentq), bisecting on the density threshold. Leave Normal/StudentT/Cauchy on the equal-tailed default since HDI wouldn't change anything for symmetric distributions."*

---

### 11. Add a `prob=` (not `hdi=`) parameter to `Distribution.plot()`
- **What:** Add a tri-state parameter (`None`/`True`/float) to opt into the tight/HDI-style window from task 10.
- **Why:** Solves the overlay-wastes-space problem (`RV(Binomial(100,0.5)).sim(10000)` realizes 29–68 but `Binomial(100,0.5).plot()`'s curve forces the shared axis to 0–100) via an explicit, opt-in parameter rather than silently guessing overlay context. Call it `prob=` not `hdi=` so it still makes sense once CDF plotting (task 12) exists.
- **Prompt:** *"Add a `prob=None` parameter to `Distribution.plot()` (distributions.py): `None` keeps current xlim behavior; `True` uses a default coverage (e.g. 0.99) via the HDI logic from task 10; a float uses that coverage. Follow the existing `suggest=None/True/False` tri-state pattern used elsewhere in the codebase. Test: `RV(Binomial(100, 0.5)).sim(10000).plot(); Binomial(100, 0.5).plot(prob=True)` should no longer stretch the shared axis to the full (0,100) range."*

---

### 12. Add CDF plotting to `Distribution.plot()`
- **What:** Add `type="cdf"` (preferred) or `cdf=True` (simpler fork-style alternative) to plot the exact CDF instead of the pdf/pmf.
- **Why:** The reference fork already has this; matches dev's existing `type="ecdf"` naming for the empirical version. Rendering is easy to get right: step function for discrete, smooth curve for continuous (fork already does this correctly, and it matches dev's own `make_ecdf` step-function styling).
- **Prompt:** *"Add CDF plotting to `Distribution.plot()` in distributions.py: a `type=` argument accepting `'pdf'` (default) or `'cdf'`. For `type='cdf'` and discrete distributions, draw a step function (`drawstyle='steps-post'` or `ax.step`), no markers. For continuous distributions, draw a smooth curve via `self.cdf(xs)`. Reuse the `prob=` x-window logic from task 11 unchanged — it only picks x-values, independent of whether pdf or cdf is evaluated there."*

---

### 13. Add `Distribution.shade()` for tail/interval probability regions
- **What:** Add a method to shade the region under a plotted pdf/pmf/cdf curve corresponding to an inequality, e.g. `Poisson(5).plot(); Poisson(5).shade(lt=3)` shading `P(X < 3)`.
- **Why:** The reference fork already has this (`shade(lt=, le=, gt=, ge=)`), auto-plotting the curve first if it hasn't been plotted yet — a genuine capability gap in dev, not a cosmetic one, confirmed via the function-name diff between the two repos. Note: `DECISIONS.md`'s "Plot Composition API" section already has a finalized (though unimplemented) `shade(from_x, to_x, label=None)` geom for a different, `+`-composable design — reconcile which signature/interface to use before implementing, rather than assuming the fork's `lt=`/`le=`/`gt=`/`ge=` interface is the settled one.
- **Prompt:** *"Port `Distribution.shade(lt=None, le=None, gt=None, ge=None)` from the reference fork (kevindavisross/symbulate, distributions.py) to dev's `Distribution` class. It should shade the appropriate tail/interval region under the most recently plotted curve (pdf, pmf, or — once task 12 ships — cdf), auto-calling `.plot()` first if the distribution hasn't been plotted yet (see the fork's `self.plotted` flag for how it tracks this). Before implementing, check DECISIONS.md's 'Plot Composition API' section, which already specifies a `shade(from_x, to_x, label=None)` geom as part of a separate `+`-composable design — resolve which interface to build (the fork's inequality-based `lt=/le=/gt=/ge=`, or the already-decided `from_x=/to_x=` interval form) before writing code. Confirm it works correctly with dev's `xlim`/`prob=` logic from tasks 10-11, since the fork's version predates that."*

---

### 14. Reconcile `Distribution.plot()` styling with the redesigned values plots
- **What:** Replace hardcoded values (`s=40` for discrete dots, plain solid connecting line) with the named per-plot-type constants used elsewhere (`IMPULSE_MARKER_SIZE`, `IMPULSE_LINEWIDTH`, etc.), and decide on marker fill/line style for the true-distribution overlay case.
- **Why:** There's already unused, better-styled scaffolding for this (`overlay_true_distribution` — smooth spline, no markers, dedicated constants) that was never wired up. Filled-on-filled dots can fully occlude each other when overlaying simulated vs. theoretical data at nearly the same point.
- **Prompt:** *"In distributions.py's `Distribution.plot()`, replace hardcoded marker size/line styling with named constants consistent with plot.py's per-plot-type conventions (IMPULSE_MARKER_SIZE, IMPULSE_LINEWIDTH, etc.). Decide whether to wire up the existing but currently-unused `overlay_true_distribution` function (smooth spline through pmf points, no markers) as the discrete rendering path, or to keep dot-to-dot but switch it to a dashed/dotted line. For marker fill, use filled dots for a standalone call and unfilled/open markers (or no markers) when overlaying onto existing simulated data — tie this to the same `prob=` opt-in parameter or a similar explicit flag rather than silently detecting overlay context. Also check whether `ax.spines['bottom'].set_position('zero')` (unique to this function) should be removed or matched to how value-plots handle spines."*

---

### 15. Decide on color cycling within a single combined `type=[...]` call
- **What:** Decide whether `RV(Normal()).sim(500).plot(["rug","hist","density"])` should draw all three in one color (current behavior) or cycle colors per type.
- **Why:** Current behavior is consistent with color meaning "which batch of data," not "which representation" — there's a legitimate argument for leaving it alone. Changing it would need a new legend to stay legible.
- **Prompt:** *(Mostly a design decision, not a bug.) If cycling is wanted: "In the dim==1 branch of `RVResults.plot()` (results.py), move the `get_next_color(ax)` call inside each `if 'X' in type:` block instead of calling it once at the top, and add legend labels (e.g. 'Rug', 'Hist', 'Density') so the different-colored layers stay distinguishable."*

---

### 16. Triage the remaining open plot-type decisions
- **What:** Walk `DECISIONS.md`'s "Open Decisions" list and resolve/scope what's actually still outstanding — e.g. violin-vs-ridgeline for mixed data, violin overlays at large k, the tile/scatter+marginal overlay-warning discrepancy vs. the reference fork, and whether `curve(distribution)` should take a `Distribution` object or a raw pdf/pmf function.
- **Why:** These are explicitly marked unresolved in the decision log itself, not newly discovered here — but they should be triaged into a concrete plan rather than left open indefinitely.
- **Prompt:** *"Read the 'Open Decisions' section of DECISIONS.md and, for each item related to plot types (violin/ridgeline, overlay-warning behavior on tile/scatter+marginal, curve(distribution) signature), propose a resolution with rationale, and list what code would need to change for each."*

---

### 17. Holistic pass on plot API argument naming
- **What:** Review argument names across `RVResults.plot()`, `Distribution.plot()`, and the 2-D dispatch for clarity and consistency — e.g. does `suggest=` read clearly on its own? Is `"segmented"` the best term? Do the new names from tasks 9/11/12/13 fit well together and with the rest of the API?
- **Why:** Several new parameter names are being introduced by the tasks above, plus `DECISIONS.md` already flags `.customize()` and the `jitter=` mode names as undecided — better to settle naming once, after the new parameters exist, than piecemeal.
- **Prompt:** *"After tasks 8-14 are implemented, review every parameter name across `RVResults.plot()`, `Distribution.plot()`, and the 2D dispatch functions in plot.py/results.py for clarity and consistency (e.g. `suggest=`, `segmented_*` naming, `marginal=`, `prob=`, `type='cdf'`, `shade()`'s signature). Propose renames where a name doesn't read clearly without the docstring, and flag any inconsistent patterns (e.g. two parameters that do similar things with different naming conventions)."*

---

### 18. Update docstrings, error messages, and warnings
- **What:** Sweep docstrings, `ValueError`/`UserWarning` text, and the "Currently Showing / Alternative Plots" suggestion messages for anything the above tasks changed.
- **Why:** Several docstrings reference `classify_data` by name (task 1), the suggestion messages are generated from lookup tables that tasks 2/5/16 may change, and error messages listing valid `type=` values will need updates (e.g. adding `"cdf"` from task 12).
- **Prompt:** *"After completing the above tasks, grep the codebase for docstrings, error messages, and warnings that reference renamed functions/parameters or list valid `type=` values, and update them. Pay particular attention to `suggestion_message`/`jitter_suggestion_message` and the `DEFAULT_PLOT_TYPE` alternatives lists, since these are the most likely to have drifted."*

---

### 19. Build a gallery notebook of the graphics capability
- **What:** A polished, organized notebook (or small set of notebooks) demonstrating every plot type, default-vs-alternative behavior, overlays, marginals, jitter modes, and the new `prob=`/`type='cdf'`/`shade()` features.
- **Why:** Existing notebooks in `team/` are exploratory/dev-facing, not a clean user-facing reference; this is also a good final check that everything above actually works together.
- **Prompt:** *"Create a gallery notebook (or a few, organized by topic) that demonstrates: every 1D and 2D plot type with its default and alternatives; the overlay mechanism (simulated + theoretical, multiple series); marginal plots; jitter modes; and the prob=/type='cdf'/shade() distribution-plot features. Use this as a final check that the changes from the other tasks work together correctly."*

---

*Note: sample-path plotting for random processes and Markov chains (color/dots/jitter/line-style revisions) is tracked separately — see the companion document `random-process-sample-paths.md`. It needs more prototyping before it can be broken into scoped tasks like the ones above.*
