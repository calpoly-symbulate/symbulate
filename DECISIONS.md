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

**Status:** Proposed

**Decision**
> _To be finalized after Task 1A visual test cases. Planning document suggests 40 as a starting point, placing the discrete/continuous-ish boundary around `Poisson(30)` at n=10,000 simulations._

**Rationale**
> _Must be confirmed by seeing what looks right visually, not from numerical analysis alone._

**Alternatives Considered**
> _To be documented during Task 1A._

---

## Decision: `N_SMALL_THRESHOLD` Value

**Status:** Proposed

**Decision**
> _To be finalized after Task 1A visual test cases. Planning document suggests 100 as a starting point._

**Rationale**
> _Must be confirmed by seeing what looks right visually, not from numerical analysis alone._

**Alternatives Considered**
> _To be documented during Task 1A._

---

## Decision: Default Plot Lookup Table

**Status:** Proposed

**Decision**
> _To be filled in after Task 1A visual test cases are complete. The table maps `(discrete_ish, small_n, data_configuration)` to a default plot type and a list of reasonable alternatives._

**Rationale**
> _The table must be designed by running real examples and choosing what looks best visually — not by reasoning abstractly. Covers: 1D discrete-ish, 1D continuous-ish, all four 2D combinations, process time points (discrete and continuous), and 1D categorical/string._

**Alternatives Considered**
> _Placeholder_

---

## Decision: Suggestion Message Behavior

**Status:** Proposed

**Decision**
> When a user does not specify `type=`, print a message after the plot renders indicating what default was chosen and what alternatives exist. The message must not appear when the user specifies `type=` explicitly. Wording, frequency (always / first-time / opt-out), and opt-out parameter name are pending team decision.

**Rationale**
> Students benefit from knowing what was chosen automatically and that alternatives exist. Suppressing the message on explicit `type=` avoids noise for users who already know what they want.

**Alternatives Considered**
> Always print / print once per session / opt-out via a `verbose=False` parameter — decision pending after Task 1A draft wording.

---

## Decision: `SymbulatePlot` Wrapper Object

**Status:** Proposed

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

**Status:** Proposed

**Decision**
> _To be finalized during Task 1C. Open question: should the `type=` parameter be renamed (it shadows the Python built-in)? All naming must be intuitive to a general audience with minimal programming experience._

**Rationale**
> _Naming decisions affect backwards compatibility (see that entry). All user-facing terms need a consistent glossary before implementation begins._

**Alternatives Considered**
> _Placeholder_

---

## Decision: Backwards Compatibility Policy

**Status:** Proposed

**Decision**
> _To be finalized during Task 1D. Known cases requiring a decision: the `type=` parameter, the `jitter=` parameter._

**Rationale**
> _The package is used in courses. Breaking existing notebook code has a real cost and must be weighed explicitly._

**Alternatives Considered**
> Clean break / deprecation warnings with aliases / silent aliases — decision pending Task 1D.

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

**Status:** Finalized (deferred implementation)

**Decision**
> A composition API (e.g., `.plot() + vline(x=0)`) is out of scope for this summer but will be designed now so `SymbulatePlot` can support it later. Vocabulary must use plain English, not matplotlib or ggplot2 conventions. Priority geoms: `vline`, `hline`, `title`, `x_label`, `shade`. Avoid: `axvline`, `geom_vline`, `set_xlabel`.

**Rationale**
> The audience includes students with minimal programming experience. Naming tied to matplotlib or ggplot2 conventions is a barrier. Designing the vocabulary now — even without implementation — ensures the `SymbulatePlot` wrapper is built with the right interface in mind. Implementation is deferred because static graphics must be stable first.

**Alternatives Considered**
> Expose matplotlib directly — rejected (too low-level for students). Defer vocabulary design entirely — rejected (risks `SymbulatePlot` being designed without a composition interface in mind).

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

## Open Decisions

The following questions must be resolved before or during Phase 2.

- [ ] `N_UNIQUE_THRESHOLD`: What value? Run Task 1A visual test cases first
- [ ] `N_SMALL_THRESHOLD`: What value? Run Task 1A visual test cases first
- [ ] Default plot lookup table: Fill in all cells after Task 1A
- [ ] Suggestion message: Exact wording template for each data configuration
- [ ] Suggestion message: Always / first-time / opt-out? If opt-out, what parameter name?
- [ ] Overlay warning text: Exact student-friendly wording for warning-category plots
- [ ] Overlay hard-error text: Exact student-friendly wording for `'marginal'` layout conflict
- [x] Visual style guide: finalized with values + justification (see Decision above); `symbulate.mplstyle` drafted; still need Phase 2 wiring + `plot.py` constants
- [ ] Accessibility: colorblind palette finalized (Okabe-Ito); black-and-white legibility and font size minimums still open
- [ ] `type=` parameter: Rename to avoid shadowing Python built-in, or keep?
- [ ] Backwards compatibility: Deprecation aliases or clean break for `type=` and `jitter=`?
- [ ] Composition API vocabulary: Finalize geom names even though implementation is deferred
