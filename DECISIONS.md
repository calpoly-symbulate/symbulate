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

**Status:** Proposed

**Decision**
> _To be finalized during Task 1B. Covers: color palette (name, hex values, accessibility justification), figure size, font sizes (axes, ticks, titles, legend), alpha values (histogram, scatter, density), line widths, spine style, grid on/off, and point style._

**Rationale**
> _All plots must share consistent aesthetics. Currently values are hardcoded across plot functions — they must be consolidated into a single `.mplstyle` file before per-plot implementation begins._

**Alternatives Considered**
> _Placeholder_

---

## Decision: `.mplstyle` Standards

**Status:** Proposed

**Decision**
> All visual defaults (colors, font sizes, transparency, line widths) will be encoded in a custom `symbulate.mplstyle` file. No aesthetic values are hardcoded in plot functions. The fragile `seaborn-colorblind` stylesheet fuzzy-match lookup in `results.py` will be replaced as part of Phase 2 infrastructure.

**Rationale**
> A `.mplstyle` file is the standard matplotlib mechanism for centralizing style. Hardcoded values scattered across plot functions make global changes expensive and inconsistent. The current seaborn stylesheet lookup is fragile (raises `IndexError` if similarity < 0.7) and must be replaced regardless.

**Alternatives Considered**
> Hardcoded constants at the top of `plot.py` — rejected because values are still scattered once per-plot functions multiply. Seaborn stylesheet lookup — rejected because it is fragile and version-sensitive.

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

**Status:** Proposed

**Decision**
> _To be finalized during Task 1B. Questions: does the package need to work in black and white (requiring line style variation in addition to color)? Which specific colorblind-safe palette? What are minimum font sizes for projected slides vs. printed pages?_

**Rationale**
> _Placeholder_

**Alternatives Considered**
> _Placeholder_

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
- [ ] Visual style guide: Color palette, figure size, font sizes, alpha, line widths, spine style, grid
- [ ] Accessibility: Black-and-white legibility? Which colorblind palette? Font size minimums?
- [ ] `type=` parameter: Rename to avoid shadowing Python built-in, or keep?
- [ ] Backwards compatibility: Deprecation aliases or clean break for `type=` and `jitter=`?
- [ ] Composition API vocabulary: Finalize geom names even though implementation is deferred
