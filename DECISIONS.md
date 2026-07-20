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

**Alternatives Considered**
> - **One flat global `N_UNIQUE_THRESHOLD`** (the original plan, shipped as `= 40`) — replaced; it had no principled anchor and applied the same count to 1-D marks and 2-D grids without distinguishing them.
> - **A per-configuration `k` lookup table** (20 for 1-D, 5-per-axis for 2-D d×d, 10 for mixed) — more knobs than needed; the two-constant budget covers the same cases.
> - **A smaller 2-D cap than 1-D** (e.g. `K_2D = 20`, `B_1D = 40`) — considered, on the grounds that 2-D grids crowd faster; dropped once we anchored to the bin count, which makes them equal and monotonic across the tile→histogram flip (a `K_2D > 30` tile would be *finer* than the 30-bin histogram it becomes, which is backwards).
> - **Product rule `kx·ky <= B`, bin both when over** — bounds total cells but has no graceful middle state (jumps straight to a 2-D histogram); the per-axis rule was preferred for the mixed-tile intermediate.

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

**Status:** Proposed — directional, not yet fully specified (Design Document Section 4 is still unwritten).

**Decision**
> Individual plot-type functions (and `.plot()` itself) shouldn't expose ad-hoc cosmetic kwargs like `color=` or `label=` to students. Those are reserved for a future chainable `.customize()` method (e.g. `.plot().customize(xlabel=..., color=...)`), which hasn't been designed yet. This doesn't touch internal plumbing like `get_next_color(ax)` — the student never types that in directly. This decision is only about what a student would type into `.plot(...)`, not about internal color-cycle bookkeeping.

**Rationale**
> From the 7/9 meeting: this keeps `.plot()`'s signature stable and simple while we design a proper customization API separately, instead of piling up one-off cosmetic kwargs per plot type.

**Alternatives Considered**
> Letting each new plot-type function grow its own cosmetic kwargs (`color=`, `label=`, etc.) as needed — rejected, since it's inconsistent and jumps ahead of the `.customize()` design before it even exists.

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
