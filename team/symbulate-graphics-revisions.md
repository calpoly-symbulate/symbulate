# Symbulate Graphics (dev branch): Prioritized Revisions & Refinements

**Scope:** This document summarizes a working conversation reviewing the graphics
overhaul on the `dev` branch of
[calpoly-symbulate/symbulate](https://github.com/calpoly-symbulate/symbulate/tree/dev),
compared against `main` and against the reference fork
[kevindavisross/symbulate](https://github.com/kevindavisross/symbulate). It is a
discussion-and-analysis pass only — **no code has been changed**. Findings below are
grounded in direct inspection of the dev-branch source, `DECISIONS.md`/`CLAUDE.md`,
and small runnable experiments (simulations, Monte Carlo checks, and rendered plots)
rather than assumptions.

Items are ordered so that foundational decisions come before the items that depend on
them (e.g., the `classify_data` rename before later sections that refer to it; the
tile axis-coordinate question before the marginal-plot bug that stems from it; the
histogram-binning discussion before the distribution-`xlim`/HDI discussion that
depends on the same "trim the tails" idea).

---

## Guiding principles established during this conversation

These came up more than once and should be applied consistently as the items below
are implemented:

1. **Discrete/continuous correctness outranks crowding or small-n/large-n concerns.**
   If a fix for visual crowding (e.g., a busy tile plot) would require calling a
   genuinely discrete variable "continuous" just to make the plot less busy, prefer a
   fix that keeps the correct discrete/continuous read and solves the crowding
   problem some other way (better tick handling, cell styling, etc.), rather than
   distorting the classification.
2. **Prefer explicit, opt-in controls over silent context-dependent behavior**, when
   both are workable. Several places in this conversation could be "fixed" either by
   auto-detecting context (e.g., "is this plot call an overlay onto existing axes?")
   or by giving the user an explicit parameter to ask for the alternate behavior. We
   consistently leaned toward the explicit parameter, since silent default changes
   based on detected context are harder to reason about and document.
3. **Don't reuse a "provisional" or coincidental numeric constant as if it were a
   deliberate cross-cutting rule** without checking — several apparent inconsistencies
   in this review turned out to be two different provisional constants that happened
   to share a value (e.g., `HIST_DEFAULT_BINS` and `B_1D`/`K_2D` are deliberately
   equal today, `HIST_DEFAULT_BINS` and `HIST2D_DEFAULT_BINS` marginal bin counts
   currently match only because both are 30). Any change to one needs an explicit
   check of what else silently depends on it.

---

## 1. Rename `classify_data` → `classify_values`

**What we found:** `classify_data(values, n_unique_threshold=B_1D, n_small_threshold=N_SMALL_THRESHOLD)`
in `symbulate/plot.py` operates on a bare array-like of already-extracted values (e.g.
`RVResults.array`, or one axis of a joint result) — never on a `Results`/`RVResults`
object itself. It is not part of the public API (only `figure, xlabel, ylabel, xlim,
ylim, plot` are re-exported from `symbulate/__init__.py`), so this is a safe internal
rename. It appears ~114 times across the repo, mostly in docstrings/comments/tests.

**Why it matters:** `classify_data` is a slightly misleading name (it doesn't operate
on "data" in the Results-object sense), and `class_results` (the originally proposed
name) would be *less* accurate, since it doesn't touch a results object either. A name
describing what it actually classifies avoids confusion for future contributors.

**Suggested direction:** Rename to `classify_values` throughout `plot.py`,
`results.py`, and `test_plot.py`, including docstrings and comments.

**Priority:** Low effort, low risk — good first cleanup item, but not urgent.

---

## 2. Discrete/continuous threshold instability at large `n`

**What we found:** The discreteness cutoff (`n_unique <= B_1D` for 1-D, `<= K_2D` for
2-D, both currently 30) is a hard cap on the number of distinct realized values,
deliberately anchored to `HIST_DEFAULT_BINS` (see `DECISIONS.md`, "classify_data
Thresholds (Budget Model)"). This means any distribution whose typical support width
lands near 30 distinct values sits on a knife-edge that flips based on the random
seed. Monte Carlo evidence (30 trials each, n=10,000):

| Distribution | mean unique values | P(classified discrete-ish) |
|---|---|---|
| `Poisson(15)` | 29.4 | 83% |
| `Binomial(100, 0.5)` at n=111* | — | — |
| `Binomial(70, 0.5)` | 31.7 | 10% |

Both `Poisson(15)` and `Binomial(70, 0.5)` have similar standard deviations
(≈3.9–4.2), so both land near the k=30 boundary; which side each lands on is
essentially a coin flip run to run, since Symbulate never seeds its RNG
(`np.random.default_rng()` with no seed, in every relevant module).

By contrast, `main`'s old `is_discrete(heights)` — "do more than 80% of distinct
values repeat at least once?" — classified **both** distributions as discrete 100% of
30 trials at n=10,000, because it scales with repeat density (sample size relative to
support width) rather than absolute support width. It was replaced because it failed
badly at *small* n (repeats haven't accumulated yet) and on rounded floats — but it
was more robust at *large* n for exactly this kind of case.

**Why it matters:** Two textbook "obviously discrete" distributions (`Poisson(15)`,
`Binomial(70, 0.5)`) get inconsistent, seed-dependent default plot types at a common
simulation size (n=10,000), which will look like a bug to instructors and students
even though it's actually a boundary-condition artifact.

**Suggested direction:** Keep `n_unique <= B_1D`/`K_2D` as the general-purpose (and
small-n-safe) rule, but add a large-n-only secondary signal resembling the old
repeat-density check (e.g., "also discrete-ish if `n` is large and a high fraction of
distinct values repeat multiple times"). This preserves the small-n fix that
motivated replacing `is_discrete` while restoring large-n robustness for
moderate-to-wide-support discrete distributions.

**A real caveat surfaced when we examined this more closely:** a naive, unbounded
version of this secondary signal would resurrect the *other* failure mode
`is_discrete` was replaced for. `classify_data` already routes a rounded-float
variable with repeats through the same `n_unique <= threshold` cap as integers — a
plain repeat-density check with no unique-value ceiling would call a rounded-float
variable with, say, 200+ distinct (but heavily-repeating) values "discrete," which is
exactly the wrong call `is_discrete` used to make. So the large-n secondary signal
needs its own bounded unique-value ceiling (which can be more generous than `B_1D`,
but must not be unlimited) — it should be an *OR* with a cap, not an unconditional
repeat-density test.

**Suggested direction (revised):** `discrete_ish = (n_unique <= B_1D) or (n is large
and n_unique <= some_higher_ceiling and repeat_fraction > X%)`. This preserves the
small-n fix, restores large-n robustness for `Poisson(15)`/`Binomial(70, 0.5)`-style
cases, and avoids reclassifying wide-support rounded-float or genuinely continuous
data as discrete purely because it repeats a lot at large n.

**Priority:** High — affects default plot type for common textbook distributions at
common sample sizes, non-deterministically.

---

## 3. 2-D tile crowding vs. the discreteness verdict

**What we found:** For `RV(Poisson(5) * Poisson(15)).sim(10000).plot()`, the
`Poisson(15)` axis lands at k≈28–30 and is classified discrete under `K_2D`,
producing a tile plot with up to 30 × 16 = 480 cells for only 10,000 points — visibly
sparser/busier than a comparable 1-D impulse plot with 30 stems, because a 2-D tile
grid crowds *multiplicatively* while a 1-D impulse crowds only *linearly*.
`DECISIONS.md` shows the team already considered and explicitly rejected a joint
`kx·ky <= budget` rule, preferring the current independent-per-axis rule for its
graceful "mixed tile" middle state.

**Why it matters — and how it interacts with principle #1 above:** Tightening `K_2D`
to reduce clutter would mean calling a genuinely discrete `Poisson(15)` axis
"continuous" purely to make the plot less busy — exactly the tradeoff principle #1
says to avoid. The crowding problem should be solved without changing the
discrete/continuous verdict.

**Suggested direction:** Leave `K_2D`/`B_1D` as-is (pending the item 2 fix above), and
instead solve tile-plot crowding through rendering (cell sizing, color scale,
tick/label handling — see item 4) rather than reclassification. Revisit the rejected
joint-budget alternative only if a rendering-level fix turns out to be insufficient in
practice.

**Priority:** Medium — real but narrower than item 2; worth a decision but not
urgent on its own.

---

## 4. Discrete-axis tick labels in 2-D plots (tile, segmented rug/box/violin, and any other plot with a discrete axis)

**What we found:** `_setup_tile_axis` unconditionally builds one forced tick label
per distinct value on a discrete axis (`ticks = (np.arange(n_cells), labels)`), with
no thinning regardless of cardinality — confirmed with `Poisson(200)` (29 distinct
values, 172–224) and `Poisson(10)` (25 distinct values) both producing dense,
overlapping labels. Impulse/dotplot never hit this problem because they only override
ticks for categorical (string) data; for numeric data they leave matplotlib's
automatic numeric locator in charge, which thins automatically since it operates on
*real data coordinates*. Tile, by contrast, plots a discrete axis using **compressed
cell-index positions** (0, 1, 2, ..., n_cells−1) with real values attached only as
labels — so matplotlib's locator has nothing to thin against. The identical
unconditional-full-label pattern also exists in the segmented rug/box/violin helpers
(`2D_mixed` configuration). This is a general issue with any 2-D plot type that has
at least one discrete axis, not something specific to tile — tile and the segmented
plots are just the two places we've confirmed it concretely so far.

**The reference fork already solved this once:** `reduce_ticks(x_shape, y_shape, ax)`
in the fork thins to ~20 x-ticks / ~15 y-ticks (rotating x-labels 75°) when a discrete
axis exceeds that count. It's not elegant (it round-trips through already-rendered
tick-label *text* rather than the underlying values), but it's a working precedent
that dev's rewrite dropped rather than a new problem to design from scratch.

**Suggested direction, two options:**
- **A (low-risk, recommended first pass):** Keep index-based cell layout, but thin
  the *label list* itself above a readability threshold (reimplementing
  `reduce_ticks`'s intent, but working from underlying values, not label text), and
  apply it consistently across every 2-D plot function that labels a discrete axis,
  not just tile.
- **B (more structural):** Position discrete cells at their real numeric values
  (matching how the continuous branch already uses a data-unit `extent`) instead of
  compacted rank-index positions, dropping the manual tick override entirely so
  matplotlib's normal locator runs free. This is a bigger change, but it directly
  fixes item 8 below (marginal-plot axis mismatch) as a side effect, since it would
  let a main panel share a coordinate system (and `sharex`/`sharey`) with a marginal
  panel. It also has a documentation-honesty upside: an unobserved-but-possible value
  shows as a visibly empty column instead of silently vanishing.

**Priority:** High — directly blocks legible 2-D plots at common cardinalities and
is the root cause of item 8's marginal-panel bug.

---

## 5. Histogram binning has no outlier/skew handling

**What we found:** Both `dev` (`HIST_DEFAULT_BINS = 30`) and `main` (hardcoded `bins =
30`) use fixed, equal-width bins spanning the raw `min`–`max` of the data, with no
outlier handling. Demonstrated on `RV(F(5,4)).sim(10000)`: mean ≈ 1.97, 99th
percentile ≈ 15.66, but max ≈ 163.95 (one rare heavy-tail draw). The resulting
histogram crushes nearly the entire distribution's shape into the first bin or two,
with ~27 of 30 bins empty. This also corrupts overlay comparisons: `F(5,4).plot()`'s
own sensible default window (`ppf(0.001)`–`ppf(0.999)` ≈ 0–52) gets stretched by the
overlay's xlim-union logic to match the simulated histogram's outlier-blown-out range
(0–164), distorting both curves together.

Naive Freedman-Diaconis doesn't fully fix this either: computed on the exact same
sample, FD's IQR-based bin width (≈0.14) implies **935 bins** over the raw range —
outlier-robust in bin *width* but not in *range*, so hundreds of bins would still sit
empty past x≈20.

**Why it matters:** This is a real, demonstrated failure mode for any noticeably
skewed/heavy-tailed distribution, not a hypothetical one, and it's inherited by both
branches (not a dev regression) — but dev is where the fix belongs given the ongoing
graphics work.

**Suggested direction:** Pair an IQR/FD-informed bin *width* with a percentile- or
IQR-based *range* clip (with an overflow/last-bin catch-all, or optional log-scale),
rather than trying to fix range or width alone.

**Depends on / interacts with:** If this becomes dynamic, `B_1D`/`K_2D` must be
explicitly re-derived or decoupled from `HIST_DEFAULT_BINS` — see item 6.

**Priority:** High — clearly demonstrated failure on realistic distributions
(F, Gamma, Pareto, LogNormal, ChiSquare are all similarly exposed).

---

## 6. Decouple the discreteness anchor from histogram bin count, if item 5 changes it

**What we found (reasoned through independently, then checked against
`DECISIONS.md`):** `B_1D`/`K_2D` = 30 is explicitly anchored to
`HIST_DEFAULT_BINS`/`HIST2D_DEFAULT_BINS`/`TILE_DEFAULT_BINS` (all 30) —
`DECISIONS.md` states directly: *"a discrete plot should stay discrete exactly while
it is no finer than the histogram it would otherwise bin into."* All three bin
constants are explicitly flagged **provisional**, pending a dedicated visual check
(`team/discrete_continuous_threshold_tests.ipynb`) — but that provisionality hasn't
yet been connected to the possibility that the bin-count *formula itself* might
change.

**Why it matters:** If item 5 makes histogram binning dynamic (FD-based), the anchor
this budget model depends on stops being a single shared constant. Reusing a dynamic,
possibly outlier-inflated bin count (recall: 935 for `F(5,4)`) directly as the
discreteness cutoff would be actively harmful — far more variables would spuriously
read as "discrete-ish." This is a case where principle #1 (protect the
discrete/continuous verdict) and principle #3 (don't silently reuse a shared
constant) both apply directly.

**Suggested direction:** If/when item 5 ships, explicitly re-derive or fully decouple
`B_1D`/`K_2D` from the (now dynamic) bin-count formula, rather than let them silently
diverge or silently break.

**Priority:** Medium on its own, but **should not be implemented independently of
item 5** — sequence them together.

---

## 7. Dot plot has no per-value stack-height cap

**What we found:** `make_dotplot` never bins and never caps stack height — confirmed
in the docstring ("Values are never binned — the dot plot is meant for discrete
data") and by direct testing. For `RV(Binomial(1, 0.1)).sim(111)`, monkeypatching
`N_SMALL_THRESHOLD` from 100 to 123 flips the default from Impulse to Dot Plot, and
because ~100 of 111 values are 0, the result is a single stack of ~100 dots at x=0
next to a short stack of ~11 at x=1 — visually lopsided and a poor use of vertical
space compared to the impulse plot's two labeled stems. By contrast, the same
threshold change for `RV(StudentT(1)).sim(111)` (continuous → rug) degrades
gracefully, since rug-plot points spread across a continuum rather than piling onto
one location.

**Why it matters:** Rule 1 of the stated design philosophy ("plot individually if not
too many *values*") is gated purely on total sample size `n`, with no check on how
concentrated those values are. Low-cardinality, imbalanced discrete data (few unique
values, uneven counts) is specifically vulnerable, even at the *current* default
threshold of 100, not just under a hypothetical higher one.

**Suggested direction:** Add a secondary crowding check independent of total `n` and
unique-value count — e.g., cap on the tallest single stack, triggering a fallback
(impulse) when exceeded.

**Note from the team:** this item may not actually be necessary — the intuition is
that raising the 1-D small-n threshold from 100 to something like 123 is worth doing
regardless of whether this cap gets built, since `RV(Binomial(1, 0.1)).sim(99)`
doesn't look too lopsided in practice. This hasn't been checked across many cases
with a single heavily-imbalanced stack, so it's worth a handful of spot checks, but
isn't worth spending much time on — the immediate action is simply to raise
`N_SMALL_THRESHOLD` from 100 to 123 and eyeball a few examples, rather than build the
crowding-cap machinery described above unless those checks turn up a real problem.

**Priority:** Medium — real but narrower in scope than items 2 and 5, and possibly
unnecessary pending the spot checks above.

---

## 8. Marginal plot rebuild, and coordinate/bin mismatches across 2-D+marginal combinations

**What we found:** The `"marginal" in type` code path in `results.py` — and its
supporting functions `make_marginal_impulse` and `compute_density` in `plot.py` — are
**byte-for-byte identical to the reference fork**, never touched by the redesign that
rebuilt `make_impulse`, `make_density`, and `make_hist`. Concretely:
- `compute_density` is a bare `gaussian_kde` wrapper with a hardcoded bandwidth factor
  (`covariance_factor = lambda: 0.25`), unrelated to the real `make_density`'s proper
  `bandwidth=` handling.
- `make_marginal_impulse` is a two-line `ax.vlines`/`ax.hlines` call with no markers,
  no `IMPULSE_LINEWIDTH`, no legend, none of `make_impulse`'s overlay bookkeeping.
- Histogram marginals bypass helpers entirely (`ax_marg_x.hist(...)` called directly),
  using a hardcoded `legacy_alpha = 0.5` — the code itself comments *"the legacy 0.5
  default still applies to the violin and marginal panels, which have no per-type
  constant yet."*
- The marginal-panel choice of plot type never consults `classify_data`/
  `default_plot_type` — it's a hardcoded two-way branch (impulse if discrete, `.hist`
  if continuous), so small-n discrete marginals never become dot plots the way a
  standalone small-n discrete variable would.

**The tick/coordinate bug is real and worse than cosmetic, and it's general — not
limited to tile+marginal.** Confirmed by direct reproduction: for
`RV(DiscreteUniform(50, 60) * Poisson(5)).sim(2000).plot(type=["tile", "marginal"])`,
the main tile panel's x-axis is `(-0.5, 10.5)` (compressed cell-index positions — see
item 4) while the marginal-x panel's x-axis is `(49.5, 60.5)` (real data values).
These are on **completely different numeric scales**, stacked in the same figure
column, with no `sharex`/`sharey` linkage anywhere in the marginal setup code. This
specific failure mode (index-based main axis vs. real-valued marginal axis) applies
to any main-panel plot type that uses tile's index-position convention for a discrete
axis (tile itself, and mixed discrete×continuous configurations) paired with a
marginal panel plotted at real values. The same class of problem needs checking
across every 2-D+marginal combination, not just this one: 2-D hist + marginal hist,
2-D density + marginal density, 2-D tile + marginal impulse, 2-D scatter + marginal
dot, mixed tile/hist + impulse/hist, etc. For fully continuous 2-D plots (hist2d+hist
marginal, density2d+density marginal), the risk is different in kind: both panels
already use real-value coordinates, so there's no index-vs-value mismatch, but bin
edges between the main panel and its marginal are only guaranteed to match today
because `HIST_DEFAULT_BINS` and `HIST2D_DEFAULT_BINS` happen to both equal 30 (see
principle #3) — not because anything enforces it, and an explicit `bins=` override
isn't verified to thread through identically to both. Scatter+marginal dot plot is
lower-risk (scatter already uses real values with no binning/compression) but should
still be confirmed once dot-plot marginals exist (see item 9's `classify_data`
routing fix).

**Checked whether the fork does any better here: it doesn't.** The fork's own
`setup_tile` uses the identical index-position scheme (`v_pos = range(len(v_lab))`)
for a discrete axis, and its own marginal-panel code (`make_marginal_impulse`,
`ax_marg_x.hist(...)`) plots at real values, with no `sharex`/`sharey` anywhere in its
marginal `GridSpec` setup either — confirmed via direct diff and inspection. The fork
does hide the marginal panel's own tick labels
(`plt.setp(ax_marg_x.get_xticklabels(), visible=False)`, kept unchanged in dev), which
papers over the mismatch's most visible symptom (you don't see two sets of
conflicting axis numbers) without fixing the underlying misalignment — the marginal
panel's bars still don't line up under the correct main-panel columns whenever the
discrete axis's support doesn't start at (and run contiguously from) 0.

**Why it matters:** This directly validates a fork-vs-dev comparison request — the
marginal feature "works" in the sense that it runs without error, but visually
misrepresents the relationship between the main panel and its marginals whenever the
discrete axis's support doesn't start at (and stay contiguous from) 0. And since the
fork has the identical bug, there's no existing fix to port for this specific
problem — unlike item 4's `reduce_ticks`.

**Suggested direction:** This depends on item 4's resolution. If item 4 adopts Option
B (real-valued cell positions), pairing the main panel and marginal panels with
`sharex`/`sharey` becomes straightforward and fixes the coordinate mismatch
structurally rather than by patching tick labels after the fact. Separately, rebuild
the marginal-panel rendering to call the actual redesigned helpers
(`make_impulse`/`make_density`/`make_hist`) instead of the legacy fork functions, and
route the marginal plot-type choice through `classify_data`/`default_plot_type` like
everywhere else. Also verify (rather than assume) that bin edges match between a 2-D
mesh plot and its marginals for `hist2d`/`tile`/`density2d`, and confirm scatter+dot
marginal alignment once implemented.

**No fork precedent to port here** — unlike item 4's `reduce_ticks`, this is new
work; the fork's marginal code is exactly what dev already (unfortunately) inherited,
bug included.

**Priority:** High — confirmed functional bug (not just stale styling), general
across multiple plot-type combinations, and it compounds a design question (item 4)
that's already high priority for other reasons.

---

## 9. Marginal API: `marginal=True` instead of `type=[..., "marginal"]`

**What we found:** `"marginal"` is checked via `"marginal" in type` scattered across
roughly six branches in `results.py`'s 2-D dispatch, functioning as a modifier that
adds two extra GridSpec panels alongside whatever the "real" plot type is — it isn't
a plot type in its own right the way `"tile"`/`"hist"`/`"scatter"` are, and (per the
dispatch code) can in principle combine with any of them.

**Why it matters:** Every other modifier in the API (`bins=`, `jitter=`,
`normalize=`) is already a separate keyword rather than folded into `type=`;
`"marginal"` being a string inside the `type` list — requiring repeated `"marginal"
in type` checks — is the same kind of API smell that shows up whenever a modifier
masquerades as a type option.

**Suggested direction:** Introduce a separate `marginal=True/False` keyword,
consistent with how other modifiers are already handled. This is naturally addressed
together with item 8's rebuild, since the marginal dispatch code is being touched
anyway.

**Priority:** Medium — a clean API win, best bundled with item 8 rather than done in
isolation.

---

## 10. New plot type: equal-width 2-D stacked bar chart for two discrete variables ("mosaic without proportional widths")

**What we found:** Reference images showed a discrete-time Markov chain's marginal
state distribution rendered as a 100%-stacked bar per time step, with every column
the same width (unlike a mosaic plot, where column width encodes the x variable's
marginal frequency). The motivating use case is a `RandomProcess`/`MarkovChain`
visualization: time on the x-axis, stacked segments showing each state's proportion
at that time step. No such plot type exists today in `main`, the fork, or `dev`.

Structurally, this is very close to `make_mosaic`, which already stacks a second
discrete variable's conditional distribution as 100%-summing segments within each
column of a first discrete variable — the only functional gap is that mosaic's column
widths are proportional to each x value's marginal count (`x_counts =
joint.sum(axis=1)`), not fixed equal-width. For a well-behaved case where every x
value (e.g. every time step, from fully-simulated sample paths with no early
termination) has an identical observation count, mosaic's proportional widths would
already come out equal by coincidence — but that's not a guarantee, and it breaks
down if paths can be censored/absorbed before some time points. Mosaic also treats
`x` purely as an ordered categorical label (equal slots divided among distinct
values, not real numeric spacing), which would misrender irregularly-spaced time
points (e.g. a `ContinuousTimeMarkovChain` snapshot at times 0, 1, 2, 5, 10).

**Why it matters:** This is a genuinely new plot-type request that a small kwarg
tweak won't fully solve, but it's also not starting from scratch — most of the
column/segment/color/legend machinery in `make_mosaic` is directly reusable.

**Suggested direction:**
1. Add an explicit `equal_width=True` (or similar) option to `make_mosaic` (or a thin
   wrapper around it) that either divides column width evenly among distinct x
   values, or — better for a literal time axis — positions columns at their true
   numeric x-value so irregular spacing renders correctly.
2. On the data side, simulate the full process once (`X.sim(n_paths)`) and evaluate
   each stored sample path at every desired time, rather than resimulating
   independently per time step — this is both more efficient and is what actually
   *guarantees* equal per-time counts (since every simulated path contributes exactly
   one observation at every time by construction), rather than relying on counts
   happening to match.
3. Decide explicitly how to handle paths that terminate/get absorbed before some time
   points, if such processes are in scope for this plot type.
4. Style to match the reference image: `annotate=False` (no in-cell numbers),
   `marginal_column=False` (keeps the external right-side legend, matching the
   reference image's legend), tighter/near-zero gaps between columns than mosaic's
   default "tile" gaps, and an explicit (not default Okabe-Ito) color choice.

**Priority:** Medium — a genuinely useful new capability with a clear implementation
path building on existing mosaic machinery, but it doesn't block any of the
correctness/crowding items above.

---

## 11. `Distribution.plot()` x-axis limits: audit result, and where HDI genuinely helps

**What we found:** A full audit of every `xlim` override in `distributions.py` shows
a single, deliberately consistent rule, not scattered inconsistency:
- **Genuinely bounded distributions show full true support:** `Bernoulli` (0,1),
  `Binomial` (0,n), `Hypergeometric` (0,n), `DiscreteUniform`/`Uniform` (a,b), `Beta`
  (0,1).
- **Distributions unbounded on at least one side clip only the hard boundary** (if
  any) and otherwise use the base class's default equal-tailed window,
  `(scipy.ppf(0.001), scipy.ppf(0.999))`: `Poisson`, `Geometric`, `NegativeBinomial`,
  `Pascal`, `Exponential`, `Gamma`, `ChiSquare`, `F`, `LogNormal`, `Pareto`.
- **Fully unbounded distributions** (`Normal`, `StudentT`, `Cauchy`) use the base
  default on both ends.

`Binomial(100, 0.5)` vs. `Poisson(100)` *look* inconsistent, but each is doing exactly
what every other distribution in its own bounded/unbounded category does.

**Why it matters — and an argument for keeping part of this as-is:** There's real
pedagogical value in showing a genuinely bounded distribution's full support by
default — trimming it to a probability threshold could visually suggest a false
boundary partway through the real (always-possible) range. The real weakness is on
the unbounded side: the default window there is *equal-tailed*, not a highest-density
interval, so skewed unbounded distributions (`F`, `Gamma`, `Pareto`, `LogNormal`,
`ChiSquare`) can look bad — this is the same failure mode item 5 demonstrated for
histograms of simulated data, just on the theoretical-curve side.

**The overlay case is a separate, real, and more clear-cut problem:** confirmed that
`RV(Binomial(100, 0.5)).sim(10000)` realizes values only from 29–68, but
`Binomial(100, 0.5).plot()`'s theoretical curve insists on its full (0,100) range;
the overlay's xlim-union logic takes the *wider* of the two axes, so the combined
plot always stretches to 0–100, wasting most of the figure on regions with
essentially zero probability and no data.

**Suggested direction:**
- Keep "bounded → full support" as the default (don't unify it away).
- For unbounded distributions, an exact HDI is cheap for **discrete** distributions
  (sort pmf descending, accumulate to target coverage) — recommend switching outright.
  For **continuous** distributions it needs a small but self-contained numerical
  routine (root-find the two equal-density points via `scipy.optimize.brentq`,
  bisect on the density threshold until coverage matches) — worth it specifically for
  the visibly skewed cases (`F`, `Gamma`, `Pareto`, `LogNormal`, `ChiSquare`); skip it
  for already-symmetric distributions (`Normal`, `StudentT`, `Cauchy`) where it
  wouldn't change anything.
- Fix the overlay case via item 12's `hdi=`/`prob=` parameter rather than by changing
  default xlim-union behavior — see below.

**Priority:** Medium-high for the overlay fix (clearly demonstrated, common use
case); medium for the general HDI upgrade (real improvement, but scoped to
specific skewed distributions).

---

## 12. Add an explicit `hdi=`/`prob=` parameter to `Distribution.plot()`

**What we found:** All the machinery item 11 needs (`pdf`/`pmf`, `cdf`, `quantile`)
is already exposed as instance attributes on the base `Distribution` class, set up
once in `__init__` — so a generic HDI-window helper can live once in the base class
and work for every subclass without per-distribution special-casing.

**Suggested direction:** Add a tri-state parameter — `None` (current default
behavior, unchanged) / `True` (HDI at a sensible default coverage, e.g. 0.99) / a
float (custom coverage) — matching the existing `suggest=None/True/False` pattern
already used elsewhere in the codebase (`should_show_suggestion`). This is preferred
over silently auto-detecting "is this call an overlay onto existing axes" (per
principle #2): instead of guessing, a user who wants a tight theoretical curve for
comparison against simulated data just writes:

```python
RV(Binomial(100, 0.5)).sim(10000).plot()
Binomial(100, 0.5).plot(hdi=True)
```

This directly and cleanly resolves item 11's overlay problem without touching the
xlim-union logic at all.

**Naming:** Once item 13 (CDF plotting) exists, calling this parameter `hdi=`
becomes semantically odd for a CDF plot ("highest density interval" is a
pdf/pmf-specific concept). **Recommend `prob=` instead of `hdi=`** — "show the
x-window covering `prob` of the probability mass" reads sensibly whether the curve
being drawn is a pdf, pmf, or cdf. Decide whether `prob=True` (opt-in, on a bounded
distribution too) should be able to override even the bounded "show full support"
default — recommend yes, since it's opt-in and the user is explicitly asking for it.

**Priority:** Medium-high — directly fixes a demonstrated overlay problem (item 11)
with a small, self-contained, low-risk addition.

---

## 13. Add CDF plotting to `Distribution.plot()`

**What we found (reference: the fork):** The fork implements `cdf=False` as a plain
boolean; when true, it evaluates `self.cdf(xs)` instead of `self.pdf(xs)`, and
correctly branches rendering: discrete → step function (`drawstyle='steps-post'`, no
markers), continuous → ordinary smooth curve. This exactly matches how dev's *own*
`make_ecdf` (for simulated data) already renders empirical CDFs via `ax.step` — so
adopting the fork's rendering choice is "free" consistency with dev's existing
conventions, and sidesteps item 16's filled/unfilled marker debate entirely (a step
function needs no point markers).

**Suggested direction:** Prefer `type="cdf"` over the fork's `cdf=True` boolean, for
naming symmetry with the already-established `RVResults.plot(type="ecdf")` — giving
a meaningful pair: `"cdf"` (exact/theoretical) vs. `"ecdf"` (empirical/estimated).
The fork's simpler boolean remains a legitimate lower-effort alternative if
minimizing new code paths matters more than naming symmetry.

**Depends on:** Item 12's parameter naming (`prob=` over `hdi=`) — the x-window
selection logic is fully reusable as-is for the CDF case (it only decides which
x-values to show, independent of whether `pdf` or `cdf` is then evaluated there), but
only reads sensibly under the more neutral name.

**Priority:** Medium — a genuinely useful, low-risk addition once items 11/12 settle
the shared x-window logic.

---

## 14. Reconcile a future plot-customization API (`.customize()` vs. the already-decided Composition API)

**What we found:** `DECISIONS.md` has two only-partially-reconciled plans for
post-`.plot()` cosmetic customization, discovered while discussing a proposed
`customize_plot(title=, xlabel=, ylabel=, color=, alpha=)`-style method (matching the
goal that `.plot()` itself should only determine plot type and necessary parameters
like `bins=`, with cosmetic customization handled separately):

- **"Decision: Customization Parameters Deferred to a Future `.customize()`
  Method"** — status *"Proposed — directional, not yet fully specified."* Sketches a
  chainable `.customize(xlabel=..., color=...)` method, motivated by keeping
  `.plot()`'s signature stable and simple.
- **"Decision: Plot Composition API"** — status *"Finalized (deferred
  implementation)."* Lays out a `+`-composable geom vocabulary (`.plot() + vline(x=0)`
  style) with a full table including `vline`, `hline`, `shade(from_x, to_x,
  label=None)`, `curve(distribution)`, `text`, `title(message)`, and
  `xlabel(message)`/`ylabel(message)`.

**`title` and `xlabel`/`ylabel` appear in both plans.** `color` appears only in the
`.customize()` example, not in the composition-API table at all. `DECISIONS.md`
doesn't reconcile this overlap anywhere — the two entries read like two different
working ideas from different points in the design process that never got merged.

**Why it matters:** Implementing both without deciding the relationship risks two
different, redundant ways to do the same thing (e.g., setting a title) — one
method-based, one `+`-composable. It also directly affects item 15
(`Distribution.shade()`), since the fork's `shade(lt=, le=, gt=, ge=)` signature and
`DECISIONS.md`'s already-decided `shade(from_x, to_x, label=None)` composition-API
geom are two different interfaces for the same concept.

**Suggested direction:** Treat a `customize_plot()`-style method as the mechanism for
*restyling something that's already drawn* (color, alpha, and possibly a post-hoc
title/label override), and reserve the `+`-composable functions for *adding a new
visual element* (reference lines, shaded regions, an overlaid theoretical curve) —
since those are genuinely additions, not restylings, and that vocabulary is already
decided. Concretely, for a `customize_plot()`:
- Allow `color=`/`alpha=` as a single value or a list matching the order of
  `type=[...]` entries, consistent with how `type=` already accepts a list.
- Validate list lengths explicitly (raise a clear error on mismatch) rather than
  silently cycling or truncating.
- Return `self` (or the `Axes`) so it chains:
  `RV(Normal()).sim(10000).plot().customize_plot(title="My Histogram", color="green")`.
- Keep it strictly cosmetic — no reclassification, no bin/threshold changes.
- Explicitly decide, rather than silently duplicate, whether `customize_plot(title=
  ...)` and a future `+ title(...)` are two supported spellings of the same thing, or
  whether `customize_plot()` should defer title/labels entirely to the composition
  API once it's built.

**Priority:** Medium — mostly a scoping/reconciliation task rather than a bug, but it
should be resolved before item 15 (shade) and ideally before the naming pass (item
19), since it affects the shape of both.

---

## 15. Add `Distribution.shade()` for tail/interval probability regions

**What we found:** The reference fork has `Distribution.shade(lt=None, le=None,
gt=None, ge=None)`, which shades the appropriate tail/interval region under the most
recently plotted curve, auto-calling `.plot()` first if the distribution hasn't been
plotted yet (tracked via a `self.plotted` flag). Dev has no equivalent — confirmed as
a genuine capability gap via the function-name diff between the two repos, not a
cosmetic difference.

**Why it matters, and the wrinkle from item 14:** This looked like a simple port at
first, but `DECISIONS.md`'s "Plot Composition API" section already specifies a
*different* signature for the same concept — `shade(from_x, to_x, label=None)` — as
part of the finalized (if unimplemented) `+`-composable vocabulary. The fork's
inequality-based interface (`lt=3` shading a half-open tail region) and the
already-decided interval-based interface (`from_x=`/`to_x=`, which would need
something like `from_x=-inf` to express the same tail) aren't quite the same shape,
and `DECISIONS.md` doesn't flag the conflict — this needs a deliberate choice, not a
silent adoption of whichever one gets implemented first.

**Suggested direction:** Resolve item 14 first (whether `shade` belongs to a
`customize()`-style method or the `+`-composable API — the composition API decision
suggests the latter), then decide explicitly between the fork's `lt=/le=/gt=/ge=`
interface (arguably more natural for expressing a tail probability) and the
already-decided `from_x=/to_x=` interval form, rather than defaulting to whichever is
easier to port. Once decided, implement it to shade under the most recently plotted
curve (pdf, pmf, or — once item 13 ships — cdf), reusing dev's `xlim`/`prob=` logic
from items 11-12 rather than the fork's pre-`prob=` version.

**Priority:** Medium — a genuinely useful, low-risk addition once item 14's
reconciliation and item 13's CDF work are settled.

---

## 16. True-distribution marker/line style, in both the "connecting line" and "filled vs. unfilled" sense

**What we found:** `Distribution.plot()`'s discrete rendering (`ax.scatter(xs, ys,
s=40, ...)` plus a bare `ax.plot(xs, ys, ...)`) uses hardcoded values disconnected
from the named per-plot-type constants used everywhere else (`IMPULSE_MARKER_SIZE =
60`, `IMPULSE_LINEWIDTH = 2.2`, the `DOTPLOT_*` family). It draws a straight dot-to-dot
polyline between integer pmf points, solid, with no distinguishing style.

There is already unused scaffolding for a better version: `overlay_true_distribution
(pmf, ax, ...)` in `plot.py` has its own dedicated constants (`TRUE_DIST_LINEWIDTH =
1.8`, `TRUE_DIST_LINESTYLE = "-"`, `TRUE_DIST_CURVE_POINTS = 600`, default legend
label `"True Distribution"`) and draws a smooth cubic spline through the pmf points
with **no markers at all** — but it is never called from anywhere in the package (no
references in `results.py`, no mention in `DECISIONS.md`/`CLAUDE.md`). This looks like
leftover scaffolding for a feature that was never wired up.

Separately: every marker type currently implemented in the package (impulse, dotplot,
scatter, `Distribution.plot()`'s discrete dots) uses **filled** markers — despite a
stray, apparently-aspirational comment in `symbulate.mplstyle` about "unfilled-circle
styling ... applied at the call site (`facecolors='none'`)," which doesn't actually
appear anywhere in `plot.py`. So there's no existing filled/unfilled convention to
defer to; this is a genuinely open design call.

**Why it matters:** When overlaying simulated impulse dots and a theoretical
distribution's dots (the whole point of which is to visually compare them), two
same-style opaque filled circles landing at nearly the same point will often just
occlude one another — defeating the purpose of the overlay. (The two markers happen
to differ in size today, 60 vs. 40, but only by accident, not by design.) A dashed or
open-marker treatment for the reference layer would both (a) avoid implying that a
discrete pmf varies continuously between integers, and (b) avoid full occlusion when
simulated and true values nearly coincide — a standard "filled = observed, open/
hollow = theoretical" convention in statistical graphics.

**Suggested direction (two coupled decisions, not one):**
1. **Connecting line for the discrete pmf:** either keep the dot-to-dot polyline but
   make it dashed/dotted (minimal change, directly signals "not continuous"), or
   adopt the existing-but-unused smooth-spline approach from
   `overlay_true_distribution` (more polished, but a spline arguably risks implying
   continuity *more*, not less — a real tradeoff to decide deliberately, not default
   into).
2. **Marker fill:** filled by default for a standalone `Distribution.plot()` call (no
   occlusion risk, consistent with the rest of the package's marker language);
   unfilled/open (or no markers, just the line) when serving as an overlay reference.
   Whether this switches automatically (context-detected) or via an explicit
   parameter is the same tension as item 12 — note that a user can *already* get
   unfilled markers today, undocumented, since `**kwargs` on `Distribution.plot()`
   passes straight to `ax.scatter` (`facecolors='none'` already works) — so this may
   be more "pick a default and document it" than "build a new feature."
3. Either way, wire up (or deliberately retire) `overlay_true_distribution` — don't
   leave two different, inconsistent answers to "how do we draw a true pmf/pdf" in
   the codebase, one of them unreachable.
4. Minor, same cluster: `ax.spines["bottom"].set_position("zero")` is unique to
   `Distribution.plot()` (not used anywhere in `plot.py`'s value-plot functions) and
   can visibly misplace the axis spine for distributions centered far from 0 (e.g.
   `Normal(100, 5)`); check whether it should match values-plot spine handling
   instead.
5. Minor: `Distribution.plot()`'s dots don't participate in `make_impulse`'s
   multi-series offset/centering (`IMPULSE_SERIES_OFFSET`) — check alignment when
   overlaying a distribution curve on 2+ impulse series.

**Priority:** Medium — real inconsistency with a clear fix path, but lower urgency
than the correctness/crowding items above.

---

## 17. Color cycling within a single combined `type=[...]` call

**What we found:** `RVResults.plot()`'s dim==1 branch fetches color once per call
(`color = get_next_color(ax)`) and reuses it across every plot type requested in that
call, so `type=["rug", "hist", "density"]` draws all three in one color, while three
separate `.plot()` calls each advance the color cycle and get different colors.
Confirmed by direct rendering comparison.

**Why it matters — and a case for *not* changing it:** Color in the rest of the
package consistently means "which batch/variable is this," not "which representation
of the same data is this" (per the overlay docstrings elsewhere in the file). Under
that reading, today's behavior is actually correct, not broken: one call on one batch
of data, shown three ways, should share one color; giving each representation its own
color would misleadingly suggest three different variables, and would need a new
legend (rug/hist/density aren't otherwise distinguished by color, only by style).

**Suggested direction:** If within-call visual separation is still wanted, cycling
color per plot-type inside the `if "X" in type:` blocks is a small, local change — but
it repurposes what color currently signals, and should ship with a legend addition
(e.g., "Rug"/"Hist"/"Density" labels) to stay legible, not as a silent styling tweak.

**Priority:** Low — a real design question, but not correctness-critical, and there's
a legitimate argument for leaving it as-is.

---

## 18. Finish the remaining/open plot types

**What we found:** Checking `DECISIONS.md`'s "Open Decisions" section directly (rather
than assuming), several plot-type questions are explicitly marked unresolved, beyond
what's covered above:
- Whether ridgeline plots should replace violin plots entirely for the
  discrete×continuous / continuous×discrete configurations, and whether violin
  overlays at large `k` should be binned.
- Overlay behavior discrepancies the team flagged but hasn't reconciled: the
  reference fork doesn't warn on two overlaid tile plots and doesn't hard-error on a
  second `.plot()` after `['scatter', 'marginal']`, contradicting what the "Overlay
  Behavior" decision currently says should happen.
- `curve(distribution)` — should it accept a Symbulate `Distribution` object (fits
  the rest of the package's vocabulary, per the open note itself) or a raw pdf/pmf
  function? Not yet decided, not yet built.
- Segmented density/hist/rug's "ridge" and "ridge hist" styling (`ridge=True` on
  `make_segmented_density`) does appear to be implemented already — worth confirming
  this is what was meant by "density ridge, ridge hist," or whether a further variant
  was intended.

**Suggested direction:** Treat this as a scoping/triage item: walk the "Open
Decisions" list in `DECISIONS.md` explicitly, confirm which are actually still open
vs. already resolved by later work, and fold genuinely-open ones into this
prioritized list rather than leaving them implicit.

**Priority:** Medium — mostly a housekeeping/scoping task, but some items (violin vs.
ridgeline for mixed data) affect default behavior directly.

---

## 19. Reconsider the plot API's argument naming and consistency, holistically

**What we found (examples, not exhaustive):**
- `suggest=` (on `RVResults.plot()`) controls whether the "Currently Showing /
  Alternative Plots" message prints — a reasonable behavior, but the name doesn't
  obviously convey *what* is being suggested without reading the docs; alternatives
  like `show_suggestion=`/`hints=` might read more clearly.
- `"segmented"` (as in `segmented_rug`/`segmented_density`/`segmented_hist`, the
  formal 2-D mixed-data plot types that `"rug"`/`"density"`/`"hist"` resolve to) is
  reasonably descriptive, but worth a deliberate check against alternatives
  (`grouped`, `faceted`, `conditional`) rather than assuming it's settled.
- The `marginal` API question (item 9), the `hdi=`/`prob=` naming question (item 12),
  the `cdf=`/`type="cdf"` question (item 13), and the `customize()`/composition-API
  and `shade()` signature questions (items 14-15) are all naming decisions raised
  individually above — they should be reconciled against each other and against the
  rest of the API in one pass, not decided in isolation per-feature.
- `DECISIONS.md` itself flags `.customize()` (a future chainable method for cosmetic
  kwargs like `color=`/`label=`) as "not yet designed," and the `jitter=` modes
  (`"bins"`, `"spiral"`, `"orderly"`) as still under active discussion with final
  names undecided.

**Suggested direction:** After the feature-level items above are settled (since
several of them mint new parameter names), do one holistic pass over every `.plot()`
signature across `RVResults`, `Distribution`, and the 2-D dispatch, checking: is the
name self-explanatory without the docstring; is it consistent with similarly-purposed
parameters elsewhere; does it match established statistical terminology where one
exists.

**Priority:** Medium, and **sequenced last among the naming-affecting items** — doing
this before items 9/12/13/14/15 settle their own naming would mean redoing it.

---

## 20. Update docstrings, error messages, and warnings — especially the suggestion messages

**What we found:** Every function touched by items above has extensive, carefully
written docstrings today — but several already reference names/behaviors that this
conversation proposes changing (e.g., docstrings mentioning `classify_data` by name,
per item 1; the "Alternative Plots" suggestion message text, which is generated from
`DEFAULT_PLOT_TYPE`'s `alternatives` lists and would need updating if item 18's
violin/ridgeline decision changes those lists; error messages listing valid `type=`
values, which would need a `"cdf"` entry if item 13 ships).

**Suggested direction:** Once the naming pass (item 19) and feature items are decided,
do a dedicated sweep specifically for:
- Docstrings referencing renamed functions/parameters (start from item 1's rename as
  a checklist template — grep-and-check, don't assume).
- The `suggestion_message`/`jitter_suggestion_message` text and the
  `DEFAULT_PLOT_TYPE` alternatives lists, since these are user-facing and are the
  most likely to drift silently out of sync with actual behavior.
- `ValueError`/`UserWarning` text listing valid `type=` values or valid parameter
  combinations (e.g., the "bins only applies when one axis is continuous" warning in
  `make_tile`, the dot plot bins warning in `results.py`).

**Priority:** Medium, but explicitly **sequenced after** the naming/feature decisions
above — this is a consistency sweep, not a source of new decisions.

---

## 21. Create a gallery of notebooks covering the graphics capability

**What we found:** The `team/` directory already has a number of exploratory/demo
notebooks (`design_plot_lookup_exploration.ipynb`, `phase1_symbulate_plot_comparison.ipynb`,
`demo_plot_lookup_showcase.ipynb`, `discrete_continuous_threshold_tests.ipynb`, the
`new_graphics_demos/` folder) but nothing that reads as a finished, organized,
user-facing gallery.

**Suggested direction:** Once the items above are implemented (or at least the
high-priority ones), build a gallery notebook (or small set of notebooks) that
systematically demonstrates: every 1-D and 2-D plot type and its default-vs-
alternative behavior; the overlay mechanism (simulated + theoretical, multiple
series); marginal plots; jitter modes; and the new `hdi=`/`prob=`/`type="cdf"`/
`shade()`/`customize_plot()` additions. This doubles as a natural place to catch any
remaining inconsistencies this conversation didn't surface, and gives the team a
single reference for what "finished" looks like.

**Priority:** Low urgency relative to the correctness/bug items above, but high value
as a wrap-up step — **naturally sequenced last**, since it should showcase the
post-fix state of the package, not the current one.

---

## Summary: suggested implementation order

1. Item 1 — rename `classify_data` → `classify_values` (foundational, touches many files)
2. Item 2 — large-n discreteness threshold instability (highest-impact correctness fix)
3. Item 6 — decouple discreteness anchor *if and only if* item 5 ships (sequence together)
4. Item 5 — outlier/skew-aware histogram binning
5. Item 3 — 2-D tile crowding vs. discreteness (decision only; defers rendering fix to item 4)
6. Item 4 — discrete-axis tick handling across 2-D plots (high priority, unblocks item 8)
7. Item 8 — marginal plot rebuild + coordinate/bin-mismatch fix (depends on item 4)
8. Item 9 — `marginal=True` API change (bundle with item 8's rebuild)
9. Item 10 — equal-width 2-D stacked bar chart / mosaic-without-proportional-widths (independent; builds on the same tile/mosaic axis conventions as items 4/8/9)
10. Item 7 — dot plot stack-height cap (possibly unnecessary; do the small-n threshold bump + spot checks first)
11. Item 11 — `Distribution.plot()` xlim/HDI audit and continuous-HDI upgrade
12. Item 12 — `prob=`/`hdi=` parameter (depends on item 11)
13. Item 13 — CDF plotting (depends on item 12's naming)
14. Item 14 — `.customize()`/Composition API reconciliation (should be resolved before item 15)
15. Item 15 — `Distribution.shade()` (depends on item 14's resolution and item 13's CDF work)
16. Item 16 — true-distribution marker/line styling (depends on item 12's opt-in-vs-auto precedent)
17. Item 17 — combined-`type=` color cycling (independent; low priority)
18. Item 18 — triage remaining open plot-type decisions
19. Item 19 — holistic API naming pass (must follow all naming-minting items: 9, 12, 13, 14, 15)
20. Item 20 — docstring/error/warning sweep (must follow item 19)
21. Item 21 — notebook gallery (must follow the implemented fixes, to showcase them)

---

*Note: sample-path plotting for random processes and Markov chains (color/dots/
jitter/line-style revisions for `Tuple`/`InfiniteVector`/`DiscreteTimeFunction`/
`ContinuousTimeFunction` and the `RVResults.plot()` ensemble branch) is tracked
separately — see the companion document `random-process-sample-paths.md`. It needs
more prototyping before it can be broken into scoped items like the ones above.*
