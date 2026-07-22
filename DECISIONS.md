# DECISIONS.md — Symbulate Graphics Overhaul

This is a lightweight decision log, not a specification or implementation guide.

- **Master roadmap:** `team/symbulate_graphics_plan.md`
- **Implementation guidance:** `CLAUDE.md`
- **This file records:** what we decided and why.

Update this file when a decision is finalized. Never remove an entry — mark it superseded instead.

---

## Decision: Graphics Backend

**Status:** Finalized

**Decision**
> Symbulate will remain on matplotlib as its sole plotting backend. Approved layout helpers (already in the environment): `seaborn.JointGrid`, `seaborn.FacetGrid`, `seaborn.pairplot`, `mpl_toolkits.axes_grid1.make_axes_locatable`. No new top-level dependencies without team agreement.

**Rationale**
> The overlay mechanism — two `.plot()` calls in one Jupyter cell producing one combined figure — depends on matplotlib's global figure/axes state (`plt.gcf()` / `plt.gca()`). All modern declarative libraries use object-based rendering that would break this behavior at an architectural level, not just a cosmetic one. The existing bugs are implementation problems with matplotlib solutions. seaborn covers the hard layout cases (marginals, faceting, scatterplot matrix) without adding a new dependency. Migration risk is high relative to team size and summer timeline.

**Alternatives Considered**
> plotnine, Lets-Plot, Altair, Plotly — all rejected because their object-based rendering models are incompatible with the overlay mechanism.

---

## Decision: Data Classification (`classify_data`)

**Status:** Proposed

**Decision**
> Replace `is_discrete()` with `classify_data(values)`, which returns two booleans: `(discrete_ish, small_n)`. These two determinations — previously conflated in `is_discrete()` — feed independently into the default plot lookup table.

**Rationale**
> `is_discrete()` asked whether most values repeat in the sample, which is the wrong question. It misclassified small-n discrete data as continuous (values haven't repeated yet), wide-support discrete distributions as continuous (`Poisson(1000)`), and rounded floats as discrete. Separating "is the variable discrete-ish" from "is n small" is the correct framing and enables a clean lookup table. Logic: object/bool dtype → always discrete-ish; float with all unique values → continuous; float with repeats or int dtype → discrete-ish if `n_unique ≤ N_UNIQUE_THRESHOLD`; `small_n = len(values) < N_SMALL_THRESHOLD`.

**Alternatives Considered**
> Patching `is_discrete()` — rejected because the function asks the wrong question; a patch cannot fix the design.

---

## Decision: `N_UNIQUE_THRESHOLD` Value

**Status:** Superseded — see "classify_data Thresholds (Budget Model)" below.

**Decision**
> _To be finalized after Task 1A visual test cases. Planning document suggests 40 as a starting point, placing the discrete/continuous-ish boundary around `Poisson(30)` at n=10,000 simulations._

**Rationale**
> _Must be confirmed by seeing what looks right visually, not from numerical analysis alone._

**Alternatives Considered**
> _To be documented during Task 1A._

---

## Decision: `N_SMALL_THRESHOLD` Value

**Status:** Superseded — see "classify_data Thresholds (Budget Model)" below.

**Decision**
> _To be finalized after Task 1A visual test cases. Planning document suggests 100 as a starting point._

**Rationale**
> _Must be confirmed by seeing what looks right visually, not from numerical analysis alone._

**Alternatives Considered**
> _To be documented during Task 1A._

---

## Decision: classify_data Thresholds (Budget Model)

**Status:** Implemented (in `plot.py` / `results.py`); threshold *values* are provisional and expected to be tuned after inspecting `team/discrete_continuous_threshold_tests.ipynb`. Supersedes the "classify_data Thresholds (Provisional)" per-configuration-`k` table this decision previously held, and the two superseded decisions above.

**Decision**
> The discreteness cutoff is framed as a **crowding budget**: the number of *labeled discrete marks/cells* a plot can show before it should bin instead. This gives two constants plus the unchanged small-n threshold (all in `plot.py`):
>
> - **`B_1D` — 1-D discreteness budget = 30 (provisional).** A 1-D numeric variable is discrete-ish iff its number of distinct values `k <= B_1D` (else histogram/rug). Categorical/string data is always discrete (a category can't be binned); all-distinct float data is always continuous (`k = NA`).
> - **`K_2D` — 2-D per-axis discreteness cap = 30 (provisional).** In 2-D, **each axis is judged independently** by the same `k <= K_2D` rule. The three outcomes fall out of the per-axis verdicts:
>   - both axes discrete → **tile**
>   - one axis over budget → that axis is binned, the other stays discrete → **mixed tile** (`discrete × continuous`)
>   - both axes over budget → both binned → **2-D histogram**
> - **`N_SMALL_THRESHOLD` — small/large-n crossover = 123 (global).** Raised from 100; decides small-n vs large-n rendering across every configuration. Still provisional. (Raising it widens the small-n band, which is what surfaced the dot-plot tall-stack problem — see "Dot-Plot Tall-Stack Fallback" below.)
>
> **Why 30 — anchored to the default bin count.** Both budgets equal the default histogram bin count (`HIST_DEFAULT_BINS` / `HIST2D_DEFAULT_BINS` / `TILE_DEFAULT_BINS`, all 30). The principle: a discrete plot (impulse stems, tile cells) should stay discrete exactly while it is **no finer than the histogram it would otherwise bin into**. At `k <= 30` a one-mark-per-value plot is equal to or coarser than the 30-bin histogram *and* carries exact value labels, so it strictly wins; at `k > 30` it would be finer than the histogram (more divisions, denser labels), so binning to 30 is a genuine simplification. `k = 30` is exactly that crossover, and the transition is monotonic in resolution (a discrete plot never has more divisions than the histogram it becomes). Note the flip does **not** change per-cell sparsity — at the boundary a `30×30` tile and the `30×30` histogram it becomes have identical cell counts and sample-per-cell, so the flip is "labeled exact values → binned ranges," not a crowding fix; that is why the bin count, not an absolute crowding limit, is the right anchor.
>
> **Why still per-axis (not the product `kx·ky`).** Even with equal values, the 2-D rule is applied *per axis*, which gives the graceful middle state — one busy axis becomes a mixed tile rather than forcing the whole plot to a 2-D histogram.
>
> **How this resolves the old "per-variable vs per-configuration" question.** `classify_data()` keeps its per-variable signature and its dtype gate (all-distinct float → continuous; object/bool/string → discrete). The *count* threshold is passed in **at the dispatch layer** (`results.py`): the 1-D branch passes `B_1D`; each of the two 2-D axis calls passes `K_2D`. This is non-circular — the dtype gate is per-variable, and the budget is applied per call site by the branch that already knows the dimensionality. (`make_tile` / `make_segmented_rug` default their own fallback classification to `K_2D`.)

**Rationale**
> The budget started from the crowding rationale (the thing that gets crowded is the number of distinct marks/cells, and in 2-D that is judged per axis so a busy axis can bin on its own). We first guessed a smaller 2-D cap than the 1-D budget, then found a cleaner anchor: tie both to the resolution of the continuous fallback — the default histogram bin count (30). A discrete plot then stays discrete exactly while it is no finer than the histogram it would become, which happens to make both budgets the same number, `30`, with a reason behind it rather than a guess. Values remain provisional: the original Task 1A sweep varied `n` at a roughly fixed number of distinct values and never separately swept `k`, so the cutoffs still want a dedicated visual check — which is exactly what `team/discrete_continuous_threshold_tests.ipynb` is for (in particular, whether a `30×30` tile is acceptable or `K_2D` should come down).
>
> **Update (outlier-aware histogram binning, below): `B_1D`/`K_2D` are now explicitly decoupled from `HIST_DEFAULT_BINS`'s *typical* value, not just from its name.** `make_hist` no longer always renders `HIST_DEFAULT_BINS` (30) bars — for skewed/heavy-tailed data it now computes a Freedman-Diaconis-informed, Tukey-fence-clipped bin count (`HIST_MIN_AUTO_BINS`-`HIST_MAX_AUTO_BINS`, i.e. 8-60) instead. `B_1D`/`K_2D` were never literally computed *from* `HIST_DEFAULT_BINS` in code (they're independent hardcoded constants that happened to share its value), so no code changed here — but the anchor rationale above ("no finer than the histogram it would become") now refers specifically to `HIST_DEFAULT_BINS`'s narrower remaining role: the flat equal-width fallback for degenerate-spread data (`IQR == 0` or `n < 2`), not "the" bin count a typical continuous histogram renders with. This is intentional, not a loose end: "how many distinct values before a variable reads as discrete" and "how many bins does an already-continuous histogram use" are different questions that only coincidentally shared one constant before outlier-aware binning existed. **Confirmed explicitly (checked every use site, not assumed): `HIST2D_DEFAULT_BINS` (`make_hist2d`'s own `bins=None` fallback) and `TILE_DEFAULT_BINS` (`make_tile`'s) are untouched by this change — both remain flat `= 30` constants, independent of `HIST_DEFAULT_BINS`/`HIST_MIN_AUTO_BINS`/`HIST_MAX_AUTO_BINS` in code, not just in name. Only `make_hist`'s own binning changed; the 2-D mesh helpers' binning is unaffected.**

**Alternatives Considered**
> - **One flat global `N_UNIQUE_THRESHOLD`** (the original plan, shipped as `= 40`) — replaced; it had no principled anchor and applied the same count to 1-D marks and 2-D grids without distinguishing them.
> - **A per-configuration `k` lookup table** (20 for 1-D, 5-per-axis for 2-D d×d, 10 for mixed) — more knobs than needed; the two-constant budget covers the same cases.
> - **A smaller 2-D cap than 1-D** (e.g. `K_2D = 20`, `B_1D = 40`) — considered, on the grounds that 2-D grids crowd faster; dropped once we anchored to the bin count, which makes them equal and monotonic across the tile→histogram flip (a `K_2D > 30` tile would be *finer* than the 30-bin histogram it becomes, which is backwards).
> - **Product rule `kx·ky <= B`, bin both when over** — bounds total cells but has no graceful middle state (jumps straight to a 2-D histogram); the per-axis rule was preferred for the mixed-tile intermediate.

---

## Decision: Large-n Discreteness Rescue — Repeat-Density Rule with a dtype-Split Ceiling

**Status:** Implemented (in `plot.py`'s `classify_values`); constant *values* provisional, same status as `B_1D`/`K_2D` (tune via `team/discrete_continuous_threshold_tests.ipynb`).

**Decision**
> Impulse is a **strict default for 1-D discrete data**: a genuinely discrete distribution should stay discrete (impulse plot) even when its support runs well past the `B_1D` crowding budget, and give way to a histogram **only when the support is genuinely too wide to draw one stem per value**. On top of the base budget (`n_unique <= B_1D`/`K_2D`), `classify_values` applies a **large-n secondary rule** that only ever flips a *continuous* verdict to *discrete*, never the reverse. A sample is rescued to discrete-ish when all of:
> - it has repeats to reason about (`n_unique < n`);
> - `n_unique` is within its **dtype's ceiling** (see below);
> - it has at least `REPEAT_MIN_OCCUPANCY` (= 20, provisional) observations per distinct value, so repeats are a reliable large-n signal, not sampling noise;
> - more than `REPEAT_FRACTION_THRESHOLD` (= 0.6, provisional) of its distinct values recur (the old `is_discrete` repeat-density criterion; 0.6 is deliberately below `is_discrete`'s 0.8, since a moderate-support discrete distribution always has a few once-seen tail values).
>
> **The ceiling is split by dtype — the key refinement.** Symbulate stores genuinely discrete distributions (Binomial, Poisson, Geometric, ...) as **integers**, while the rounded-float case `is_discrete` misclassified is **float**-dtype. So:
> - **integer data** gets a generous cap, `INT_REPEAT_CEILING` (= 200, provisional) — textbook discrete distributions stay discrete far past the budget (`Binomial(1000, 0.5)` ≈ 110 distinct, `Poisson(200)` ≈ 100 distinct → impulse); only a truly huge support (`Binomial(10000, 0.5)`, ~300+ distinct) exceeds it and bins into a histogram.
> - **float data** keeps the strict cap, `REPEAT_CEILING_FACTOR` (= 2) × the budget (so 60 for `B_1D = 30`) — a wide-support rounded float stays continuous no matter how much it repeats. This tight float cap is the guard that keeps the rule from reintroducing the `is_discrete` failure mode.
>
> **The rescue is 1-D only.** `classify_values` takes `large_n_rescue` (default `True`); the 2-D per-axis dispatch passes `large_n_rescue=False`. In 2-D the budget model deliberately bins both-axes-over-`K_2D` into a 2-D histogram (a 40×40 discrete tile is genuinely too crowded), and Task 3 fixes 2-D crowding by rendering, not reclassification — so keeping the impulse-strict rescue out of the 2-D verdict is intentional, and leaves every existing 2-D dispatch verdict unchanged.

**Rationale**
> Task 2 flagged that the bare `n_unique <= B_1D` cutoff makes textbook distributions sitting near 30 distinct values (`Poisson(15)`, `Binomial(70, 0.5)`) flip discrete/continuous by random seed. A repeat-density signal (main's old `is_discrete`) is robust at large n but, applied unconditionally, mis-labels wide-support rounded floats as discrete — the exact reason `is_discrete` was replaced. The earlier fix bounded the rescue with a single ceiling (`REPEAT_CEILING_FACTOR × budget` = 60) and a strict `REPEAT_MIN_OCCUPANCY` = 100, which held the line against floats but also capped genuinely discrete integers at ~60 distinct and blocked wide ones (`Binomial(1000, 0.5)` at ~90 samples/value). This decision makes impulse the strict discrete default by splitting the ceiling on dtype: integers (genuine counts) get a generous cap and floats stay tightly bounded. dtype is the clean discriminator — it distinguishes "genuinely discrete count" from "rounded continuous" without a heuristic. `REPEAT_MIN_OCCUPANCY` was lowered 100 → 20 so wide discrete integer supports actually reach the test; it stays non-binding for floats (whose ceiling of 60 already implies high occupancy).

**Alternatives Considered**
> - **A single dtype-agnostic ceiling** (the earlier implementation, `REPEAT_CEILING_FACTOR × budget` = 60 for both) — replaced; it couldn't keep genuinely discrete integers past ~60 distinct without also loosening the float guard that prevents the `is_discrete` rounded-float failure.
> - **Unbounded repeat-density (literal old `is_discrete`)** — rejected; reintroduces the rounded-float failure mode and would draw hundreds of unreadable impulse stems for a huge-support integer distribution (no "histogram when necessary" cutoff).
> - **Raising `B_1D` itself** — rejected; the base budget is a crowding anchor tied to the histogram bin count (see budget-model decision), and widening it would also loosen the 2-D tile verdict. The rescue is a separate large-n-only signal, correctly decoupled from the base budget.

---

## Decision: Outlier/Skew-Aware Histogram Binning

**Status:** Implemented (in `plot.py`'s `make_hist`); threshold constants provisional, same status as `B_1D`/`K_2D`.

**Decision**
> `make_hist`'s automatic binning (`bins=None` only — an explicit `bins=`, int or edges array, is untouched and keeps the flat equal-width scheme exactly as before) now pairs a Freedman-Diaconis bin width with a Tukey "far out" IQR fence (`[Q1 - HIST_OUTLIER_FENCE_MULT * IQR, Q3 + HIST_OUTLIER_FENCE_MULT * IQR]`, clipped to the data's own range, `HIST_OUTLIER_FENCE_MULT = 3.0`) instead of always spanning the raw min-max in `HIST_DEFAULT_BINS` (30) equal-width bins. Bin count within the fence is clamped to `[HIST_MIN_AUTO_BINS, HIST_MAX_AUTO_BINS]` = `[8, 60]`. Values beyond the fence are folded into one hatched **overflow bar** per affected side (not dropped) — the histogram's bars, regular and overflow together, still integrate to 1 (or the true total count) over every simulated value. Degenerate spread (`IQR == 0` or `n < 2`) falls back to the flat `HIST_DEFAULT_BINS`-bin scheme, unchanged.

**Rationale**
> `RV(F(5,4)).sim(10000)` has 99% of its mass below ~16 but a max of ~164 (one rare heavy-tail draw) — the old flat 30-bin-over-raw-range histogram crushed the real shape into 1-2 bins. Plain Freedman-Diaconis bin width alone isn't a fix either: computed over the full raw range it implies ~935 bins, outlier-robust in width but not in range (hundreds of empty bins past the bulk). Pairing FD width with a range clip fixes both. An overflow bar (rather than silently dropping out-of-fence data, or extending the last bin) was chosen over an optional log-scale axis: log-scale is a bigger conceptual jump for the package's target audience (general users, minimal stats/programming background, per `CLAUDE.md`'s "User Philosophy") and doesn't match a linear pmf/pdf reading elsewhere in the package. The overflow bar is visually distinguished with a hatch pattern precisely so it doesn't read as "just another equal-width bin."

**Alternatives Considered**
> Plain Freedman-Diaconis over the raw range — rejected, ~935 bins for the motivating example. Percentile-based range clip (e.g. 1st-99th) instead of a Tukey IQR fence — considered; the IQR fence was preferred for self-adapting to the data's actual spread/skew rather than a fixed percentile, and for matching a convention (Tukey's boxplot outlier fence) students commonly already see elsewhere in an intro stats course. Optional log-scale x-axis mode — rejected for this audience (see Rationale); may be worth revisiting as an opt-in for more advanced use later. Silently dropping/clipping out-of-fence data with no overflow indicator — rejected, would misrepresent the true total probability/count.

---

## Decision: Marginal Panel Rebuild (Coordinate Alignment + Type Routing)

**Status:** Implemented (`results.py`'s 2D `marginal=True` branch, `plot.py`).

**Decision**
> Marginal panels are rebuilt on the real redesigned 1D helpers
> (`make_impulse`/`make_hist`/`make_density`/`make_rug`/`make_dotplot`, each now
> accepting `orientation="vertical"|"horizontal"`) instead of the legacy
> fork code (`make_marginal_impulse`, `compute_density`, raw `ax.hist`
> calls), which is now removed. Each marginal panel's plot type is routed
> through `classify_values`/`default_plot_type` per axis (dotplot/impulse
> for a discrete axis, rug/hist for a continuous one, by that axis's own
> small-n/large-n split) instead of a hardcoded impulse-or-hist rule — except
> when the main panel itself is `density`/`density2d`, in which case both
> marginals draw density curves regardless of discreteness, matching the
> main panel's own representation.
>
> **Coordinate alignment**, fixing the confirmed tile+marginal mismatch:
> the main panel is drawn *first*, then each marginal panel is synced to
> the main panel's actual final `get_xlim()`/`get_ylim()` (via `sharex`/
> `sharey`), rather than each panel independently deriving its own idea of
> the axis range. For a discrete axis whose main-panel type packs it into
> integer rank slots instead of real values (`violin`, `box`/`boxplot`,
> `segmented_rug`, `segmented_hist`, `segmented_density` always; `tile`
> only for a categorical/non-whole-number/pathological-range axis, since
> the discrete-axis tick-crowding fix (see "Discrete-Axis Tick Label
> Thinning (2D Plots)") upgraded tile to lay out a whole-number discrete
> axis at real values instead — see `DISCRETE_INDEX_OFFSET` in `plot.py`
> and how `results.py` resolves each axis's type independently for tile),
> the marginal's own values are converted to the matching rank codes
> first, so its dots/stems land under the correct main-panel column/row.
> For `hist2d` and a real-valued `tile` axis specifically, the marginal
> histogram reuses the main panel's *exact* bin edges (`make_hist2d`'s
> return value, or `setup_tile_axis`'s, both called independently a
> second time with the same inputs — deterministic, not just
> coincidentally equal) rather than recomputing its own.
>
> **`mosaic` is an explicit exception**: `marginal=True` with `type="mosaic"`
> raises a `ValueError` pointing at mosaic's own `marginal_column=True`
> instead of building real marginal panels for it. Mosaic already visualizes
> x's marginal distribution via column width and y's marginal distribution
> via its own optional marginal column (a different, normalized-`[0,1]`
> coordinate system from every other 2D type here) — a generic marginal
> panel would be redundant with a feature it already has, not a missing one.

**Rationale**
> The legacy marginal code was confirmed byte-for-byte identical to the
> reference fork and never touched by the graphics redesign. The
> coordinate mismatch (main panel at tile's compressed index positions,
> marginal panel at real values, no `sharex`/`sharey` anywhere) was
> confirmed directly: `RV(DiscreteUniform(50,60) * Poisson(5)).sim(2000).plot(type="tile", marginal=True)`
> had a main panel x-axis of `(-0.5, 10.5)` against a marginal panel x-axis
> of `(49.5, 60.5)` — unrelated scales stacked in one figure column. Reading
> the main panel's *actual* final limits, rather than hand-deriving each
> plot type's extent formula (tile's `(-0.5, n-0.5)`, violin/boxplot's
> matplotlib-assigned category positions, ...), was the key simplification
> that made covering every main-panel type tractable in one pass, and
> avoids the exact "don't reuse a coincidental value as if guaranteed"
> trap this file's guiding principles warn about.
>
> Mosaic was scoped out deliberately, not skipped for difficulty: it already
> has an equivalent, differently-styled feature, so a second mechanism
> showing the same information would be confusing, not additive.

**Alternatives Considered**
> Building real marginal panels for mosaic too, reusing its normalized
> column-span math — rejected as redundant with `marginal_column=True`,
> which already exists and serves the same purpose. Keeping the
> hardcoded impulse-or-hist marginal rule (not routing through
> `classify_values`) — rejected once the coordinate-alignment fix was
> already touching this code; leaving small-n marginals as impulse/hist
> instead of dot/rug would have been inconsistent with how every other
> 1D plot decision in the package works.

---

## Decision: Dot-Plot Tall-Stack Fallback

**Status:** Implemented (`DOTPLOT_MAX_STACK` in `plot.py`, dispatch in `results.py`); value provisional, tune via `team/discrete_continuous_threshold_tests.ipynb`.

**Decision**
> The dot plot is the 1-D small-n discrete default, but it draws one dot per observation and stacks identical values. A **third** kind of crowding — stack height — can make it unreadable even when the two budget checks (`B_1D` distinct values, `N_SMALL_THRESHOLD` sample size) both pass: if most of the mass lands on a few values, each stacks into a tall column of specks. So when a dot plot would be the *automatic default* but the **tallest single stack** exceeds **`DOTPLOT_MAX_STACK` = 30 (provisional)**, the default is redirected to that configuration's **large-n** default — impulse for numeric 1-D discrete, bar for 1-D categorical.
>
> - The check is independent of `n` and of the distinct-value count — it's purely `max(count per distinct value)`, computed by `dotplot_tallest_stack()`.
> - Only the *default* is redirected. An explicit `type='dotplot'` is always honored (a student who asks for a dot plot gets one).
> - The suggestion note stays honest: it names the impulse/bar default it actually drew and keeps the dot plot in the alternatives list (`type='dotplot'`).
> - Anchored to the same bin count (30) as the budgets: once one stack alone is taller than the whole histogram has bins, the aggregated plot reads better.

**Rationale**
> Surfaced by raising `N_SMALL_THRESHOLD` 100 → 123, which pulls samples like `RV(Binomial(1, 0.1)).sim(111)` (≈106 dots on the value 0, ≈5 on 1) into the small-n band, where the discrete default is the dot plot. At 100 that case was large-n and already drew an impulse; widening the small-n band re-exposed it, so the fallback restores the impulse without giving up the wider small-n band elsewhere. Framing "tall stack" as "behave like large n" reuses the existing lookup rather than inventing a new plot type — the tall stack *is* a large-n symptom (lots of data on one value).

**Alternatives Considered**
> - **Guard inside `make_dotplot` itself** — rejected: it would also fire on an explicit `type='dotplot'` (this is about the *default*, not the draw), and the suggestion note built in `results.py` would still claim a dot plot. The dispatch layer is where `type`/default/suggestion stay consistent.
> - **Fold stack height into `classify_data` / the budget model** — rejected: `classify_data` is per-variable and configuration-agnostic; stack height is a rendering concern of one specific plot type, so it lives next to the dot-plot code.
> - **Hard-code the fallback to impulse for every configuration** — rejected: categorical data's natural large-n default is the bar chart, so redirecting to the configuration's own large-n default is more consistent than always picking impulse.

---

## Decision: Default Plot Lookup Table

**Status:** Provisional — filled in from the Task 1A visual sweep; team's own framing is "we will revise but just to see functionality and use with symbulate for now."

**Decision**
> The table maps each data configuration and its small-n/large-n split to a default plot type, plus a list of reasonable alternatives (these show up in the suggestion message — see "Decision: Suggestion Message Behavior" below).

| Data configuration | Small n default | Small n alternatives | Large n default | Large n alternatives |
|---|---|---|---|---|
| 1D discrete-ish | Dot plot | Impulse, Bar, Density, ECDF, Rug | Impulse | Histogram (many distinct values), Density + Rug, ECDF |
| 1D continuous-ish | Rug plot (use rug even at low n) | Rug, Boxplot (overlay?), Density + Rug | Histogram | Density (overlay?), Boxplot (overlay?), Rug |
| 2D discrete × discrete | Scatter with jitter | Tile, Scatter with size | Tile / Heatmap | Mosaic, Marginal distributions (overlay, impulse) |
| 2D continuous × continuous | Scatter | Density | 2D Histogram | Contour, Hexbin, Density, Marginal distributions (overlay) |
| 2D discrete × continuous | Segmented rug plot | Box plot / Violin plot, Ridgeline | 2D Histogram / Tile | Box plot, Violin, Scatter, Marginal distributions (overlay), Ridgeline |
| 2D continuous × discrete | Segmented rug plot | Box / violin, Ridgeline | 2D Histogram / Tile | Box plot, Violin, Scatter, Marginal distributions (overlay), Ridgeline |
| Process time point, discrete-valued | Dot plot | Impulse | Impulse | Histogram |
| Process time point, continuous-valued | Dot plot | Impulse, Rug, Boxplot (overlay?), Density + Rug | Histogram | Density (overlay?), Boxplot (overlay?), Rug |
| 1D categorical / string | Dot plot | Impulse, Bar | Impulse | Histogram (many distinct values), Density + Rug |

> **Overlay-specific defaults (more exploratory than the table above — a few cells still have a `?` in the source notes and aren't settled yet):**
>
> | Configuration | Small n | Large n |
> |---|---|---|
> | 1D discrete-ish | Stacked dot plots + color | Stacked impulse plots + color |
> | 1D continuous-ish | Stacked strip plots + color (gap between groups) | Stacked histograms + color |
> | 2D discrete × discrete | Scatter with jitter + color | Scatter with jitter + color (heatmaps can't overlay — side-by-side? unresolved) |
> | 2D continuous × continuous | Scatter + color | Scatter + color (density surfaces can't overlay — side-by-side? unresolved) |
> | 2D discrete × continuous | Stacked segmented strip plot + color | Stacked violin plots + color (+ gap between groups) |
> | 2D continuous × discrete | Same as above, horizontal orientation | Same as above, horizontal orientation; overlay spacing unresolved |
> | Process time point (discrete/continuous) | Same as the matching 1D row | Same as the matching 1D row |

**Rationale**
> The table was designed by running real examples and choosing what looked best visually (see `team/design_default_threshold_exploration.ipynb`, `team/design_plot_lookup_exploration.ipynb`, `team/phase1_symbulate_plot_comparison.ipynb`), covering 1D discrete-ish, 1D continuous-ish, all four 2D combinations, process time points (discrete and continuous), and 1D categorical/string. The reference-fork comparison notebook found that the fork's actual overlay behavior doesn't match what the "Overlay Behavior" decision says should happen (see that decision's note) — so the overlay-specific defaults above are a separate, even more tentative pass at what overlaid defaults should look like. They haven't been visually checked the way the main table has.

**Alternatives Considered**
> Reasoning abstractly about plot type without running visual test cases — rejected per Task 1A's explicit instruction ("the goal is to see what actually looks good, not to reason abstractly about it").

---

## Decision: Suggestion Message Behavior

**Status:** Finalized (wording and trigger condition); opt-out parameter name is carried over from a draft, not independently confirmed — see note below.

**Decision**
> Print a message after every plot renders, in **both** of these cases — not only when `type=` is unspecified:
>
> - **`type=` not specified (default kicked in):** `"Currently Showing: {Default Type} Plot (Default) / Alternative Plots: {Alt1} (type='{alt1}'), {Alt2} (type='{alt2}'), ..."`
>   Example: `"Currently Showing: Impulse Plot (Default) / Alternative Plots: Histogram (type='hist'), Density (type='density'), ..."`
> - **`type=` specified explicitly:** the message still prints, but now labels whichever type *would have been* the automatic default: `"Currently Showing: {Chosen Type} / Alternative Plots: {Default Type} (Default), {OtherAlt} (type='{other}'), ..."`
>   Example: `"Currently Showing: Histogram / Alternative Plots: Impulse (Default), Density (type='density'), ..."`
>
> Both cases are really one template — the "Currently Showing" line only tags `(Default)` when the default isn't what's actually showing, and the alternatives list always tells you the `type=` value you'd need to get that option.
>
> **Opt-out parameter:** `hints` (default `True`) — `.plot(hints=False)` turns the message off. **Note:** we borrowed this name from a different draft template that wasn't chosen (it used `hints=True/False`); nobody proposed an opt-out name alongside the template we did pick, so this is our best guess, not a real decision. Double-check it at the next team meeting before treating it as locked in.

**Rationale**
> Students benefit from knowing what got chosen automatically and that alternatives exist — and from learning what the default would've been even when they picked something else on purpose. This is a deliberate broadening from the original plan, which only showed the message when `type=` was unspecified.

**Alternatives Considered**
> Only showing the message when `type=` is unspecified (the original plan) — rejected in favor of always showing it. A different draft template ("Showing a {kind} plot — the default for this type of data. You can also try kind='{alt1}' or kind='{alt2}'.", shown only when unspecified) — considered, not chosen.

---

## Decision: `SymbulatePlot` Wrapper Object

**Status:** Finalized — this is already built (in `symbulate/plot.py`) and tested (`tests/test_plot.py`, `TestSymbulatePlotWrapper`), not just proposed anymore.

**Decision**
> Every `.plot()` method returns a `SymbulatePlot` object instead of `None`. `__repr__` returns `""` so Jupyter does not print it. The object stores the underlying axes as `self.ax`.

**Rationale**
> This is backwards-compatible — no existing code assigns or chains off `.plot()` return values. It lays the foundation for a future composition API (e.g., `.plot() + vline(x=0)`) and a thin interactivity conversion layer without requiring any future architecture changes. Implementation cost is ~10 lines of new code plus a one-line addition to each `plot()` method. Ship in the same PR as the first round of plot fixes.

**Alternatives Considered**
> Continue returning `None` — rejected because it permanently closes off the composition API without any migration path.

---

## Decision: Overlay Behavior

**Status:** Proposed

**Decision**
> Overlay whenever naturally possible. A second `.plot()` call always attempts to draw on the same axes. No automatic side-by-side behavior. Three outcomes: (1) **Natural overlay** — draws on shared axes, no message (scatter, impulse, 1D histogram, 1D density, rug, 2D density/contour, ecdf, most types); (2) **Readability warning** — overlay produced, warning printed (two 2D tile plots, two 2D histograms — competing color scales); (3) **Hard error** — multi-panel GridSpec layout cannot be joined (any plot using `'marginal'` layout). Color cycling via `get_next_color(ax)`, called once per `.plot()` call. Side-by-side panels are deferred.

**Rationale**
> Overlay is the expected behavior in a Jupyter teaching context. The three-tier policy avoids silently producing unreadable plots while still defaulting to overlay. Side-by-side panels require a separate design decision (likely `panel=True` or a `.compare()` method) and are deferred so the `SymbulatePlot` wrapper can be designed to support it later.

**Alternatives Considered**
> Error on any second call — too restrictive. Automatic side-by-side — requires multi-panel layout incompatible with the overlay architecture.

---

## Decision: Visual Style Guide

**Status:** Finalized. The global values here (palette, colormap, figure size, fonts, spines, grid) are implemented in `symbulate/symbulate.mplstyle` (see "Decision: `.mplstyle` Standards"); the per-plot-type values (alpha, line width) still need to be added as constants in `plot.py` as part of Phase 2 implementation.

**Decision**

> **Color palette — categorical:** replace `tab10` with **Okabe-Ito** (7 hues, excluding black): `#E69F00, #56B4E9, #009E73, #F0E442, #0072B2, #D55E00, #CC79A7`.
>
> **Color palette — sequential/continuous:** replace hardcoded `cmap="Blues"` with **viridis** everywhere, as the single default for all magnitude/density encodings (2D density, tile, hist2d).
>
> **Figure size:** keep `6.4in × 4.8in` (matplotlib factory default). No change.
>
> **Font sizes:** keep matplotlib factory defaults — `font.size=10`, `axes.titlesize='large'` (~12pt), `axes.labelsize='medium'` (10pt), tick label size `'medium'` (10pt), `legend.fontsize='medium'` (10pt). No change.
>
> **Default transparency (alpha):** histogram `0.5 → 0.65`; scatter `0.5 → 0.25`; density curve `0.5 → 0.15` (overlay-on-histogram case specifically — see Rationale).
>
> **Line widths:** impulse lines `1.5 → 2.2`; density curves `2.0 → 1.8`; sample paths stay at `1.5`. No change.
>
> **Spine style:** remove top and right spines; keep left and bottom.
>
> **Grid:** off → on, light horizontal reference grid (`axes.axisbelow=True`, low alpha, thin, light gray).
>
> **Point style:** filled → unfilled (open) circles for scatter, with edge color = series color.

**Rationale**

> **Categorical palette.** Better colorblind accessibility and print-friendliness than the current `tab10` default.
>
> **Sequential palette.** Colorblind-friendly, prints well in grayscale, and gives better visual distinction between magnitude steps than the current `Blues` default. Matches a preference already noted in this project's own prototype notes.
>
> **Figure size / font sizes.** Both are just inherited matplotlib defaults, never a deliberate Symbulate choice, and nothing in the audit suggests they need to change yet — final font sizing is pending the accessibility decision below.
>
> **Alpha.** Tuned per plot type for visual clarity: histograms get a slightly bolder fill since they don't suffer from overplotting, while scatter and density become more transparent so overlapping points and curves stay readable instead of turning into a solid blob.
>
> **Line widths.** Impulse lines get bolder for visibility since there's no overplotting risk; density curves get slightly thinner to pair with their lower alpha; sample paths are unchanged.
>
> **Spines.** Removing the top and right spines gives a cleaner, more professional look and matches standard statistical-graphics convention.
>
> **Grid.** A light reference grid makes it easier for students to read approximate values off a plot, while staying subtle enough not to distract from the data.
>
> **Point style.** Unfilled circles read more clearly when points overlap and match a look already planned for this project.

**Alternatives Considered**

> Categorical: matplotlib `tab10` (current — less colorblind-friendly), ColorBrewer `Dark2` (weaker colorblind separation than Okabe-Ito).
> Sequential: keep `Blues` (less visually distinct, no accessibility advantage), other perceptually-uniform options like `plasma`/`inferno`/`magma`/`cividis` (all reasonable, but viridis is the most widely recognized and well-rounded choice).
> Alpha: a variable, sample-size-aware transparency instead of a flat value per plot type — more precise, but more complex to implement; left as a possible future refinement.

---

## Decision: Visual Style Guide — Point Style (supersedes point-style line above)

**Status:** Finalized

**Decision**
> This replaces the "Point style: filled → unfilled (open) circles for scatter" line in the Visual Style Guide decision above. After looking at it again, scatter points are staying **filled**, matching what's already there (default size 6). This is a reversal, not a tweak — don't implement unfilled/open circles for scatter.

**Rationale**
> We reconsidered the unfilled-circle call from the original pass and decided the current filled-circle default already reads clearly at the approved alpha (0.25) — it doesn't need the extra complexity of edge-colored open markers. (`team/new_graphics/scatter.py`'s draft already uses filled circles and had flagged this exact mismatch as unresolved — this entry settles it in favor of what the draft already does.)

**Alternatives Considered**
> Unfilled (open) circles with edge color = series color — the original Task 1B proposal; rejected, not worth the added complexity for the readability gain we saw.

---

## Decision: `.mplstyle` Standards

**Status:** Finalized. File created at `symbulate/symbulate.mplstyle` (color palette, colormap, figure size, fonts, spines, grid, and a global line-width fallback); not yet wired into `results.py` — that wiring, plus the `plot.py` per-plot-type constants, is Phase 2 implementation work.

**Decision**
> Values that apply the same way across every plot type — color palette, sequential colormap, figure size, font sizes, spines, grid, and a global line-width fallback — are encoded in `symbulate.mplstyle`. Values that vary by plot type — histogram/scatter/density alpha, impulse/density line width, unfilled-scatter marker styling — cannot be expressed as matplotlib rcParams, since rcParams are global, and instead live as named constants at the top of `plot.py`. No aesthetic values are hardcoded inline in plot functions. The fragile `seaborn-colorblind` stylesheet fuzzy-match lookup and the `tab10`-overriding `init_color()` call in `results.py` will be replaced as part of Phase 2 infrastructure.

**Rationale**
> A `.mplstyle` file is the standard matplotlib mechanism for centralizing style, and hardcoded values scattered across plot functions make global changes expensive and inconsistent. The split with `plot.py` constants isn't a compromise — it's a real technical limit of rcParams (no per-plot-type granularity), discovered while drafting the file. The current seaborn stylesheet lookup is fragile (raises `IndexError` if similarity < 0.7) and must be replaced regardless.

**Alternatives Considered**
> Hardcoded constants at the top of `plot.py` for everything — rejected because global values (palette, figure size, fonts) are still scattered once per-plot functions multiply. Seaborn stylesheet lookup — rejected because it is fragile and version-sensitive. Trying to force per-plot-type alpha/line-width into rcParams — not possible; rcParams has no per-plot-type mechanism.

---

## Decision: Plot Naming and API Vocabulary

**Status:** Finalized

**Decision**
> - **`type=` parameter: KEEP.** No rename to `kind=`, despite it shadowing the Python built-in `type()`. No deprecation cycle needed since nothing is changing.
> - **`xlabel` / `ylabel` (no underscore), everywhere** — including the future composition-API geoms (see "Decision: Plot Composition API" below). The package already exports top-level `xlabel = plt.xlabel` / `ylabel = plt.ylabel` (`plot.py`); standardizing on this spelling avoids two inconsistent spellings of the same word inside one package.
> - **`color` (American spelling), never `colour=`.** `color` already flows straight through `**kwargs` into matplotlib calls (`ax.hist(..., color=color)`, `ax.scatter(..., color=color)`, etc.); supporting `colour=` would mean either a pointless translation layer or two names for one thing. Matches the spelling students will see in matplotlib, seaborn, and pandas.
> - **`normalize`: keep, don't rename.**

**Rationale**
> `type=`: we considered renaming to `kind=` (plain English, no clash with the builtin, and it matches pandas' `DataFrame.plot(kind=...)`, which students may already know) but decided to keep the existing, documented name instead — it avoids a deprecation cycle for a parameter that's already used in course materials. `xlabel`, `color`, and `normalize` all follow the same logic: match what's already shipping in this package and in the wider Python data world, instead of giving beginners a second, inconsistent spelling to trip over.

**Alternatives Considered**
> `type=` → `kind=` (rejected, see Rationale) · `x_label` (rejected — would create two spellings of the same word inside one package) · `set_xlabel` (rejected — matplotlib OO-method naming, not plain English) · `colour=` (rejected — inconsistent with the rest of the Python data ecosystem) · renaming `normalize` (rejected, no better alternative proposed).

---

## Decision: Backwards Compatibility Policy

**Status:** Partially finalized — `type=` case is now moot; `jitter=` is still open.

**Decision**
> - **`type=` parameter: moot.** We're keeping `type=`, not renaming it (see "Plot Naming and API Vocabulary" above), so there's no backwards-compatibility question left to answer.
> - **`jitter=` parameter: still open.** We're keeping the existing `jitter=True/False` boolean, but new discrete-scatter jitter *modes* are being actively discussed (7/9 and 7/13 meeting notes): `jitter="bins"` (dots placed in reading order within each bin) and `jitter="spiral"` (circular jitter). These might replace or sit alongside the `jitter="orderly"` compass-cluster mode already built in `team/new_graphics/scatter.py`. Final names, behavior, and whether `"orderly"` survives are all still undecided — see Open Decisions.

**Rationale**
> The package is used in courses, so breaking existing notebook code has a real cost — that's why `type=` was kept instead of renamed, and why any change to `jitter=` needs to add new modes rather than break the existing boolean.

**Alternatives Considered**
> Clean break / deprecation warnings with aliases / silent aliases for `type=` — moot now that `type=` isn't changing. For `jitter=`'s new modes: still open, nothing ruled out yet.

---

## Decision: Accessibility Requirements

**Status:** Partially finalized — colorblind-palette question is settled (Okabe-Ito, finalized as part of the Visual Style Guide); black-and-white legibility and font-size minimums remain open.

**Decision**
> Which specific colorblind-safe palette? **Okabe-Ito**, for the categorical palette — see the Visual Style Guide decision above for the validated CVD-separation numbers and hex values. Still open: does the package need to work in black and white (requiring line style variation in addition to color)? What are minimum font sizes for projected slides vs. printed pages?

**Rationale**
> Okabe-Ito is a well-established colorblind-friendly palette, chosen for better accessibility and print-friendliness than the current default — see the Visual Style Guide decision for the full comparison. The remaining two questions (black-and-white-only legibility, font-size minimums) still need team input; the Visual Style Guide decision left font sizes at matplotlib factory defaults specifically pending this answer.

**Alternatives Considered**
> _See Visual Style Guide § Alternatives Considered for the categorical-palette comparison. Black-and-white and font-size alternatives still to be documented._

---

## Decision: Plot Composition API

**Status:** Finalized (deferred implementation) — the vocabulary is complete now, replacing the earlier "priority geoms" list below.

**Decision**
> A composition API (e.g., `.plot() + vline(x=0)`) is out of scope for this summer but will be designed now so `SymbulatePlot` can support it later. Vocabulary must use plain English, not matplotlib or ggplot2 conventions. Full geom vocabulary (Task 1C):
>
> | Name | Signature | Plain-English description |
> |---|---|---|
> | `vline` | `vline(x, label=None)` | Draws a vertical reference line at a given value on the x-axis. |
> | `hline` | `hline(y, label=None)` | Draws a horizontal reference line at a given value on the y-axis. |
> | `shade` | `shade(from_x, to_x, label=None)` | Shades the region of the plot between two x-values (for showing a probability as an area). |
> | `curve` | `curve(distribution)` | Overlays a distribution's theoretical pdf/pmf curve on top of a simulation plot, for comparing simulated to theoretical. |
> | `text` | `text(x, y, message)` | Adds a short text label at a specific point on the plot. |
> | `title` | `title(message)` | Sets the plot's title. |
> | `xlabel` / `ylabel` | `xlabel(message)` / `ylabel(message)` | Sets the x-axis or y-axis label. Same function as the existing top-level `xlabel`/`ylabel`; also composable with `+`. |
>
> Original priority list (replaced by the table above, kept here for history): `vline`, `hline`, `title`, `x_label`, `shade`. One spelling fix: `x_label` became `xlabel`, per "Decision: Plot Naming and API Vocabulary" above. Avoid: `axvline`, `geom_vline`, `set_xlabel`, `annotate` (matplotlib already overloads "annotate" to mean "text + arrow"; `text` says exactly what it does with no baggage), `fill_between`/`shade_region` (ruled out on the same matplotlib-naming grounds as `axvline`).
>
> **Still open, and naming alone doesn't settle it:** `curve(distribution)` implies passing a Symbulate distribution object (e.g. `curve(Normal(0, 1))`), which fits the rest of the package's vocabulary better than passing a raw pdf function — worth confirming before anyone builds this (see Open Decisions).
>
> **Boundary with `.customize()`:** `title`, `xlabel`, and `ylabel` belong exclusively here, not on a future `.customize()` method — see "Decision: Customization Parameters Deferred to a Future `.customize()` Method" for the full reconciliation. `.customize()` is restricted to restyling marks that are already drawn (`color=`, `alpha=`); everything in the table above is a new visual element, which is this decision's territory.

**Rationale**
> The audience includes students with minimal programming experience. Naming tied to matplotlib or ggplot2 conventions is a barrier. Designing the vocabulary now — even without implementation — ensures the `SymbulatePlot` wrapper is built with the right interface in mind. Implementation is deferred because static graphics must be stable first.

**Alternatives Considered**
> Expose matplotlib directly — rejected (too low-level for students). Defer vocabulary design entirely — rejected (risks `SymbulatePlot` being designed without a composition interface in mind). `geom_*` prefix (ggplot2 convention) and `axvline`/`axhline` (matplotlib convention) — both rejected; a general-audience user shouldn't need to know either library to guess what `vline` composes onto a plot.

---

## Decision: Interactivity Strategy

**Status:** Finalized (deferred)

**Decision**
> Symbulate will not switch to an interactive plotting library. Interactivity and animation will be offered through `ipywidgets` template notebooks and `matplotlib.FuncAnimation` helpers. No architecture changes to Symbulate's core are required. Implementation begins after Phase 4 (static graphics) is stable.

**Rationale**
> `ipywidgets` works in Jupyter and Colab without configuration and allows sliders, dropdowns, and play buttons to wrap any `.plot()` call. Template notebooks (one per major use case) are the delivery mechanism. `FuncAnimation` handles animation within matplotlib without a new dependency. A thin Plotly conversion layer remains a possible future option (likely a second-year project) and does not need to be decided now.

**Alternatives Considered**
> Switch to Plotly or another interactive library — rejected because it breaks the overlay mechanism (see Graphics Backend decision). Build interactivity into the core — rejected because it would require architecture changes before static graphics are even stable.

---

## Decision: Customization Parameters Deferred to a Future `.customize()` Method

**Status:** Proposed — directional, not yet fully specified (Design Document Section 4 is still unwritten). Its boundary with the Plot Composition API decision below was ambiguous (both plans included `title`/`xlabel`/`ylabel`) and is now resolved — see below.

**Decision**
> Individual plot-type functions (and `.plot()` itself) shouldn't expose ad-hoc cosmetic kwargs like `color=` or `label=` to students. Those are reserved for a future chainable `.customize()` method (e.g. `.plot().customize(xlabel=..., color=...)`), which hasn't been designed yet. This doesn't touch internal plumbing like `get_next_color(ax)` — the student never types that in directly. This decision is only about what a student would type into `.plot(...)`, not about internal color-cycle bookkeeping.
>
> **Boundary with the Plot Composition API (resolved):** `.customize()` and the `+`-composable geoms (see "Decision: Plot Composition API") read as two competing ways to do the same thing once both plans existed. The scopes are now split so neither duplicates the other:
> - `.customize()` is for **restyling a mark that's already drawn** — `color=`, `alpha=`, and any other cosmetic property of existing plot elements.
> - `title`, `xlabel`, and `ylabel` belong **exclusively** to the Composition API (`+ title(...)`, `+ xlabel(...)`, `+ ylabel(...)`) — they read as *adding* a text element, not restyling an existing one, and that vocabulary is already fully specified there. `.customize()` should not grow its own `title=`/`xlabel=`/`ylabel=` kwargs.
> - Reference lines, shaded regions, and overlaid curves (`vline`, `hline`, `shade`, `curve`, `text`) were never ambiguous — they're new visual elements, so they stay on the Composition API side.
>
> When `.customize()` is eventually designed: `color=`/`alpha=` should accept a single value or a list matching the order of `type=[...]` entries (consistent with how `type=` already accepts a list), with explicit validation of list-length mismatches rather than silent cycling/truncation, and it should return `self` (or the `Axes`) so it chains, e.g. `RV(Normal()).sim(10000).plot().customize(color="green")`. It stays strictly cosmetic — no reclassification, no bin/threshold changes.

**Rationale**
> From the 7/9 meeting: this keeps `.plot()`'s signature stable and simple while we design a proper customization API separately, instead of piling up one-off cosmetic kwargs per plot type. The restyle-vs-add split avoids two supported spellings for the same operation (e.g. setting a title), which would otherwise force every student-facing doc and example to arbitrarily pick one.

**Alternatives Considered**
> Letting each new plot-type function grow its own cosmetic kwargs (`color=`, `label=`, etc.) as needed — rejected, since it's inconsistent and jumps ahead of the `.customize()` design before it even exists. Giving `.customize()` its own `title=`/`xlabel=`/`ylabel=` kwargs alongside the Composition API's `title(...)`/`xlabel(...)`/`ylabel(...)` — rejected, since it creates two redundant spellings for the same operation with no clear rule for which a student should reach for.

---

## Decision: 2D Density / Tile / Hist2D Colormap Direction

**Status:** Finalized

**Decision**
> Use plain `viridis` (not `viridis_r`) for all 2D magnitude encodings (2D density, tile, hist2d). Density/count of 0 renders as viridis's dark, low end — not light.

**Rationale**
> We confirmed this independently at both the 7/7 and 7/13 meetings ("Should 0 be dark or light? → Dark"; "2D density colorbar starts at 0"). This settles an ambiguity left open in `symbulate_graphics_plan.md`'s "Known decisions from prototype work already done," which had said `` `viridis_r` or similar `` was preferred — that phrasing is now outdated. It also matches what `team/new_graphics/density2d.py`'s draft already does (`cmap="viridis"`, no `_r` suffix, `level_edges = np.linspace(0, zmax, levels + 1)`), so no code needs to change — this entry just writes down the choice we'd already made in code.

**Alternatives Considered**
> `viridis_r` (reversed, 0 = light) — what `symbulate_graphics_plan.md` said before this entry; now outdated.

---

## Decision: Discrete-Axis Tick Label Thinning (2D Plots)

**Status:** Finalized — `make_tile` upgraded to Option B (real-value axis layout); the other five functions remain on Option A

**Decision**
> `make_tile`'s discrete axes are now laid out on a real number line instead of compacted rank-index cells, for whole-number data (a count-style variable — Binomial, Poisson, a die roll, ... — regardless of storage dtype, since Symbulate often stores these as float64). `_setup_tile_axis` builds one unit-width cell for every possible whole number across the *observed range* (`lo` to `hi`), not just the values that happened to occur, so an unobserved-but-possible value shows as a visibly empty column instead of silently vanishing. Because the axis is now a genuine number line, matplotlib's own numeric locator (`MaxNLocator(nbins=MAX_DISCRETE_TICKS, integer=True)`) picks nice, evenly spaced tick values with a **constant step** — the same mechanism the continuous axis case already relies on — instead of a hand-picked subset of whichever values happened to occur. `TILE_MAX_GAP_FILL_RANGE` (2000) guards against a pathologically wide range (e.g. two masses at 0 and 10000) allocating a huge, mostly-empty grid.
>
> Categorical data (strings/bools/objects), non-whole-number discrete data (repeated floats), and a range past `TILE_MAX_GAP_FILL_RANGE` fall back to the original Option A layout: compacted rank-index cells with `_thin_discrete_ticks()` picking an evenly spaced *subset of ranks* (capped at `MAX_DISCRETE_TICKS`, 10). Labels stay horizontal in every case, matching every other axis in the package — an earlier iteration rotated x-labels 75° once thinned (matching the reference fork's `reduce_ticks`), but that was dropped once the tick counts were tightened to `MAX_DISCRETE_TICKS` (10): straight labels at that cap read fine, and a consistent, unrotated look across every plot type was judged more important than the extra headroom rotation bought.
>
> When both axes are discrete, each is now ticked **independently** for its own consistent, evenly spaced scale — the two axes' tick *counts* can differ (e.g. 5 on one, 10 on the other) rather than being forced to match. An earlier iteration of this fix forced both axes to the same tick count (the smaller axis's own cell count, capped at 10); that was superseded once it became clear that "same count" and "evenly spaced, consistent numeric step" cannot both hold in general — matching counts across two axes with very different ranges means giving up a clean, constant step on at least one of them, and a consistent per-axis scale was judged more valuable to a student reading the axis than an identical tick count on both. The separator-line boundaries for the mixed-data case are computed directly from each axis's own `extent` (`np.arange(extent[0] + 1, extent[1])`), which works unchanged whether that axis ended up as compacted rank cells or real-valued whole-number cells.
>
> The other five functions (`make_segmented_rug`, `make_segmented_density`, `make_segmented_hist`, `make_grouped_boxplot`, `make_violin`) remain on the original Option A fix: `_thin_discrete_ticks()` thins an axis's displayed tick labels once it carries more than `MAX_DISCRETE_TICKS` (10) distinct values, keeping an evenly spaced subset of the *ranks* (not necessarily an evenly spaced subset of the *values*, if the data has gaps). Each of these only ever has one discrete axis in a given call, so they don't have `make_tile`'s two-axes-at-once complication, but they share the same underlying limitation tile just moved past: a rank-thinned axis can still show an inconsistent step between labeled values when the data has gaps. Extending the real-value + `MaxNLocator` treatment to them is a natural follow-up, not yet done.

**Rationale**
> `_setup_tile_axis` (and the equivalent tick-labeling code in the segmented rug/density/hist/box and violin helpers) originally built one forced tick label per distinct value on a discrete axis, with no thinning regardless of cardinality — confirmed with Poisson(200) (29 distinct values, 172–224) and Poisson(10) (25 distinct values), both producing dense, overlapping labels. The first fix (rank-index thinning, still what the other five functions do) solved the overlap but not a second, subtler problem: picking evenly spaced *ranks* doesn't give evenly spaced *values*. Any gap in the observed data (a count that never came up in this many simulations) makes the labeled values jump by inconsistent amounts — e.g. 172, 178, 183, 191 (jumps of 6, 5, 8) — instead of a clean, constant step like a normal numeric axis. Even *without* gaps, rounding the evenly spaced rank positions to integers introduces small inconsistencies (mostly-3 steps with one 4 thrown in). Impulse/dotplot never hit either problem because they only override ticks for categorical (string) data; for numeric data they already leave matplotlib's automatic numeric locator in charge, which both thins *and* keeps a consistent scale automatically, since it operates on real data coordinates rather than compressed ranks.
>
> `classify_data` can let a discrete axis carry up to `K_2D` (30) distinct values, so a dense, gappy discrete axis isn't a rare edge case — any 2D_dd tile plot can legitimately reach that cardinality. Once "consistent, evenly spaced scale" was confirmed as the actual goal (not just "no overlapping labels" or "same tick count on both axes"), the fix had to move past rank-based thinning entirely for at least the common case: whole-number count data.
>
> The reference fork's own fix (`reduce_ticks(x_shape, y_shape, ax)`, thinning to ~20 x-ticks / ~15 y-ticks, rotating x-labels 75°) has this same rank-vs-value limitation; it was already known not to fix the marginal-plot axis mismatch (a tile main panel and its marginal panel don't share a coordinate system) or reveal unobserved-but-possible values, which real-value positioning fixes as a side effect (see below). Its label rotation isn't carried over either — see the Decision section above.

**Alternatives Considered**
> Forcing the same tick count on both axes of a two-discrete-axes tile plot (implemented briefly, then superseded) — consistent counts, but at the cost of an uneven scale on whichever axis got force-thinned below its own natural tick count. Rejected once "evenly spaced, consistent scale" was confirmed as the higher priority.
>
> Extending real-value positioning to the other five functions (segmented rug/density/hist/box, violin) in the same pass — deferred; they don't have `make_tile`'s pixel-grid constraint (no gap-filling or range-cap needed, since a band/box/violin is just a drawing position, not a grid cell), so the fix there would be simpler, but it's still a distinct piece of work not yet done.
>
> A pcolormesh-based irregular grid (Voronoi-style cell boundaries at the midpoints between observed values, for non-whole-number or very sparse whole-number data) instead of the current gap-fill-or-fallback split — would generalize real-value positioning further, but adds real complexity (variable-width cells, no clean "empty column" story for values genuinely never possible) for a case (non-integer discrete tile data) that's rare in this package's typical usage. Not implemented.

---

## Decision: `Distribution.plot()` Discrete Rendering (True-Distribution Curve)

**Status:** Implemented (in `distributions.py`)

**Decision**
> A discrete distribution's `.plot()` (its pmf) is now drawn as a **smooth, marker-free curve** — no dots at all — by wiring up the previously-unused `overlay_true_distribution()` helper in `plot.py` as the discrete rendering path. `Distribution.plot()`'s discrete branch calls `overlay_true_distribution(self.pmf, ax, xlim=(int(xs[0]), int(xs[-1])), color=color, alpha=alpha, **kwargs)`, so the pmf reads as a rounded reference curve (cubic spline through the pmf values, clipped at zero) the way a density curve reads over a histogram — styled from the named `TRUE_DIST_LINEWIDTH` / `TRUE_DIST_LINESTYLE` (solid) constants rather than the old hardcoded `ax.scatter(..., s=40)` plus a bare solid polyline.
>
> Three coupled sub-decisions (from the finding #16 task prompt), as resolved:
> - **Line style — smooth spline, not dashed dot-to-dot.** Chosen deliberately by the team over the dashed-straight-segment alternative. (Caveat accepted: a smooth discrete pmf reads similarly to a continuous pdf; that was weighed and accepted.)
> - **Markers — none, in every case.** The prompt allowed "unfilled/open markers (or no markers)"; we took *no markers*. Because there are zero dots standalone *and* overlaid, the "filled vs. unfilled" question and its "tie it to an explicit flag vs. auto-detect overlay context" sub-question are **moot** — there is nothing to fill and no context to detect. No new `.plot()` kwarg was added (consistent with cosmetic controls being reserved for a future `.customize()`).
> - **`ax.spines["bottom"].set_position("zero")` removed.** It was unique to `Distribution.plot()` among the value-plots and could visibly misplace the axis spine for a distribution centered far from 0 (e.g. `Binomial(200, 0.5)`); removed so the spine matches every other plot type.
>
> `overlay_true_distribution()` is **kept** (not retired) and is now actually called — the literal "wire it up" reading of the prompt. It continues to auto-label its curve `"True Distribution"` and refresh the legend, so overlaying `Poisson(5).plot()` on a simulated impulse plot yields a clean two-entry legend ("Variable 1" + "True Distribution").

**Rationale**
> Finding #16 flagged that the discrete rendering used hardcoded values disconnected from the named per-plot-type constants, drew a solid straight dot-to-dot polyline that could be mistaken for something continuous, and carried a unique spine tweak — while a better-styled `overlay_true_distribution()` sat unused. Wiring that function up as the discrete path resolves all of it at once with a single implementation (no duplicated pmf-drawing logic). The "no dots at all" and "smooth curve" choices were made by the team during implementation.

**Alternatives Considered**
> Dashed straight dot-to-dot segments (keeps a discreteness cue now that there are no dots) — considered and explicitly rejected in favor of the smooth curve. Filled-standalone / unfilled-overlay markers toggled by an explicit flag or `prob=` (the prompt's default suggestion) — moot once "no markers at all" was chosen. Auto-detecting overlay context to switch marker fill — also moot for the same reason. Extracting the spline into a shared helper and deleting `overlay_true_distribution()` (cleaner, avoids a second copy) — rejected in favor of literally wiring up the existing function per the prompt.

---

## Open Decisions

The following questions must be resolved before or during Phase 2.

- [x] Discreteness thresholds: replaced the single `N_UNIQUE_THRESHOLD` with a crowding-budget model — `B_1D` and per-axis `K_2D` both anchored to the default histogram bin count (provisional 30), `N_SMALL_THRESHOLD` raised to 123 — implemented in `plot.py`/`results.py`; see "classify_data Thresholds (Budget Model)". Values still need a dedicated visual check (`team/discrete_continuous_threshold_tests.ipynb`)
- [x] Default plot lookup table: filled in provisionally from Task 1A — see "Default Plot Lookup Table". We expect to revise it
- [x] Suggestion message: wording and trigger condition are finalized — see "Suggestion Message Behavior". The opt-out parameter name (`hints`) is a guess carried over from elsewhere, not independently confirmed
- [ ] Overlay warning text: Exact student-friendly wording for warning-category plots
- [ ] Overlay hard-error text: Exact student-friendly wording for `'marginal'` layout conflict
- [ ] Overlay behavior discrepancy: `team/phase1_symbulate_plot_comparison.ipynb` found that the reference fork doesn't actually warn on two overlaid tile plots, and doesn't hard-error on a second `.plot()` after `['scatter', 'marginal']` — reconcile this against what the "Overlay Behavior" decision says before implementing it
- [x] Visual style guide: finalized with values and justification (see Decision above); `symbulate.mplstyle` is drafted; still need Phase 2 wiring plus the `plot.py` constants; point-style got reversed back to filled (see "Visual Style Guide — Point Style")
- [ ] Accessibility: colorblind palette finalized (Okabe-Ito); black-and-white legibility and font size minimums still open
- [x] `type=` parameter: KEEP — no rename (see "Plot Naming and API Vocabulary")
- [x] Backwards compatibility for `type=`: moot, since it isn't changing
- [ ] Backwards compatibility for `jitter=`: new modes (`"bins"`, `"spiral"`) are being discussed (7/9, 7/13 meetings) — final names and behavior, and whether they replace or just extend the `"orderly"` mode already drafted in `team/new_graphics/scatter.py`, are still undecided
- [x] Composition API vocabulary: finalized (full geom table — see "Plot Composition API"), even though implementation is deferred
- [ ] `curve(distribution)`: should it accept a Symbulate distribution object, or a raw pdf/pmf function? (flagged as open in the Task 1C notes themselves)
- [x] `classify_data()` architecture: resolved — `classify_data()` keeps its per-variable dtype gate and takes the count threshold as an argument; the dispatch in `results.py` passes `B_1D` for 1-D and `K_2D` per axis for 2-D (non-circular). See "classify_data Thresholds (Budget Model)"
- [ ] `.customize()` API: not yet designed (Design Document Section 4 is still unwritten) — see "Customization Parameters Deferred" decision
- [ ] Violin plot overlays at large `k`: should violins be binned? Should ridgeline plots replace violin plots entirely for the discrete×continuous / continuous×discrete configurations?
- [ ] Overlay "+color" stacking: discrete groups use the categorical (Okabe-Ito) palette — should continuous groupings use a gradient instead, and if so how does that interact with the sequential (viridis) palette already reserved for magnitude encodings?
- [x] Discrete-axis tick label crowding: `make_tile` upgraded to Option B (real-value cell positions for whole-number data, matplotlib's own locator) — see "Discrete-Axis Tick Label Thinning (2D Plots)". `make_segmented_rug/density/hist/box` and `make_violin` remain on the original Option A (rank-index + thinning); extending real-value positioning to them is still open
- [ ] Marginal-panel axis mismatch: a tile main panel and its marginal panel don't share a coordinate system — `make_tile`'s discrete axis is now real-valued for whole-number data, which should make this easier to resolve (matplotlib's `sharex`/`sharey` could line the panels up), but the marginal-panel wiring in `results.py` hasn't been touched, so this is not yet fixed
- [x] `Distribution.plot()` discrete rendering (finding #16): resolved — discrete pmf now drawn as a smooth, marker-free curve by wiring up `overlay_true_distribution()`; named `TRUE_DIST_*` constants replace the hardcoded `s=40`; the unique `set_position("zero")` spine tweak removed. See "Decision: `Distribution.plot()` Discrete Rendering (True-Distribution Curve)"
