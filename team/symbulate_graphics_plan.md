# Symbulate Graphics Overhaul — Planning and Implementation Guide

## Overview

This document consolidates decisions made during project planning and provides a roadmap for the graphics overhaul, including what goes into the design document, how to sequence the work, and how to use Claude and Claude Code effectively throughout.

---

## Key Decisions Made

### Library: Stay on matplotlib

Symbulate will remain on matplotlib as its primary plotting library. The reasons:

- **The overlay mechanism is architecturally fundamental.** Two `.plot()` calls in one Jupyter cell producing one combined plot works because matplotlib maintains global figure/axes state via `plt.gcf()` and `plt.gca()`. Plotnine, Lets-Plot, Altair, and Plotly all use object-based rendering models that would break this behavior. Rebuilding it on any of these libraries is a substantial architectural cost, not a cosmetic one.
- **The existing bugs are implementation problems, not library-limitation problems.** Every specific issue in the to-do notes has a matplotlib solution.
- **seaborn handles the hard layout cases without a new dependency.** `JointGrid` (scatter with aligned marginals), `FacetGrid` (faceting), and `pairplot` (scatterplot matrix) are available through seaborn, which is already an implicit dependency. These solve the cases where raw matplotlib is most painful.
- **Migration risk is high relative to team size and timeline.** A library switch would require rewriting `plot.py`, significant parts of `results.py`, and retesting every plot type. For a small team with a busy summer this is not justified.

Approved layout helpers: `seaborn.JointGrid`, `seaborn.FacetGrid`, `seaborn.pairplot`, `mpl_toolkits.axes_grid1.make_axes_locatable`. No new top-level dependencies without team agreement.

### The `classify_data` Function: Replacing `is_discrete`

The current `is_discrete()` function asks whether most unique values appear more than once in the simulated data. This is the wrong question. It fails at small n (e.g., `Binomial(5, 0.3).sim(10)` is misclassified as continuous because values don't repeat yet), fails for wide-support discrete distributions (`Poisson(1000)` is misclassified as discrete), and misclassifies rounded continuous data as discrete.

The new approach replaces `is_discrete()` with a `classify_data()` function that makes **two separate determinations** that were previously conflated:

1. **Is the variable discrete-ish or continuous-ish?** — determines which plot geometries make sense
2. **Is n small or large?** — determines how to render

These two determinations combine in a lookup table to produce the default plot type.

#### The `classify_data` logic

```python
def classify_data(values, n_unique_threshold=40, n_small_threshold=100):
    data = np.asarray(list(values))
    n = len(data)
    n_unique = len(set(data))

    if data.dtype == object or data.dtype == bool:
        # String/categorical or boolean outcomes: always discrete-ish
        discrete_ish = True
    elif data.dtype == float and n_unique == n:
        # Float data where every value is unique: definitely continuous
        discrete_ish = False
    elif data.dtype == float and n_unique < n:
        # Float data with repeated values: user-defined finite support
        discrete_ish = n_unique <= n_unique_threshold
    else:
        # Integer dtype: use absolute unique count
        discrete_ish = n_unique <= n_unique_threshold

    small_n = n < n_small_threshold
    return discrete_ish, small_n
```

#### Why this approach is correct

- `Exponential(1).sim(10)`: float, all unique → `discrete_ish=False`, `small_n=True` → rug plot
- `Binomial(10000, 0.5).sim(1000)`: int, 200+ unique values → `discrete_ish=False`, `small_n=False` → histogram
- `Poisson(12).sim(10)`: int, ~8 unique → `discrete_ish=True`, `small_n=True` → dot/rug plot
- `Poisson(12).sim(10000)`: int, ~26 unique → `discrete_ish=True`, `small_n=False` → impulse plot
- `Poisson(2).sim(10000)`: int, ~10 unique → `discrete_ish=True`, `small_n=False` → impulse plot
- Custom float `{1, 1.5, 2.71, 4}.sim(1000)`: float with repeats, 4 unique → `discrete_ish=True` → impulse plot
- `BoxModel(['H','T']).sim(1000)`: object dtype → `discrete_ish=True` → bar/impulse plot
- `BrownianMotion()[2.5].sim(10000)`: float, all unique → `discrete_ish=False` → histogram

The threshold `n_unique_threshold=40` means the discrete/continuous-ish boundary falls around `Poisson(30)` for 10,000 simulations — which feels right visually. **The thresholds are named constants at the top of `plot.py` and must be decided by the team after running visual test cases (see design phase task below), not from numerical analysis alone.**

#### How `classify_data` is used for 2D data

For 2D data, each variable is classified independently. This feeds directly into `make_tile()`, which already accepts separate `discrete_x` and `discrete_y` flags. This produces the correct behavior for `RV(Poisson(12) * Poisson(2)).sim(10000).plot()`:

- `Poisson(2)`: `discrete_ish=True` → discrete tile axis (labels: 0, 1, 2, ...)
- `Poisson(12)`: `discrete_ish=True` → discrete tile axis (~26 integer labels)
- `Binomial(10000, 0.5)`: `discrete_ish=False` → histogram-binned axis

#### How `classify_data` interacts with stochastic processes

`X[t].sim(n)` (indexing a process at a time point) produces a standard `RVResults` object with `dim=1` and `index_set=None`. It goes through the normal 1D plotting branch, and `classify_data` works correctly on it:

- `PoissonProcess(rate=1)[2.5].sim(10000)`: int, ~12 unique → impulse plot
- `PoissonProcess(rate=5)[10].sim(10000)`: int, ~51 unique → histogram
- `BrownianMotion()[2.5].sim(10000)`: float, all unique → histogram

Full process sample paths — `X.sim(10).plot()` — take an entirely separate code path (`dim=None`, `index_set` is set) and `classify_data` is never involved. These always produce line/path plots and need no classification.

#### Extension to 3+ RVs

When 3D/4D joint distribution plotting is implemented (scatterplot matrix, faceted plots), `classify_data` extends naturally: apply it independently to each column of the data array. Each variable gets its own `discrete_ish` determination used for its marginal plot and pairwise panels. No changes to `classify_data` are needed.

**Note:** The current `else` branch in `RVResults.plot()` handles `dim > 2` by drawing each result as a connected-dot index plot, which is not appropriate for joint distribution visualization. Until proper multi-RV plotting is implemented, this branch should raise a `NotImplementedError` with a helpful message rather than silently producing a confusing plot.

#### Edge cases confirmed by testing

- Single unique value (degenerate RV always returning 5): `n_unique=1 ≤ threshold` → discrete-ish → impulse with one bar
- Boolean outcomes: explicit `dtype == bool` clause → always discrete-ish
- String/categorical outcomes: explicit `dtype == object` clause → always discrete-ish
- Small n continuous data (e.g., `Normal().sim(5)`): float, all unique → `discrete_ish=False` → rug plot

### The Default Plot Lookup Table: Design Before Coding

The output of `classify_data()` feeds into a lookup table that maps `(discrete_ish, small_n)` to a default plot type. **This table must be designed by the team before any code is written.** See the design phase task in the Work Sequence section.

The lookup operates across multiple data configurations — not just 1D discrete-ish vs. continuous-ish, but 2D mixed cases, process time points, and eventually 3+ RVs. Each cell in the table has one default plot and a short list of reasonable alternatives. The alternatives are surfaced to users via a message printed below the plot:

> *"Default plot for this data type is an impulse plot. You might also try `type='hist'` or `type='density'`."*

This message appears only when `type` is not specified by the user (i.e., only for the automatic default). It is suppressed when the user specifies `type` explicitly.

### The SymbulatePlot Wrapper Object

`.plot()` should return a `SymbulatePlot` wrapper object instead of `None`. This is a backwards-compatible change — no existing code assigns or chains off `.plot()` return values. The only implementation requirement is suppressing the repr so Jupyter doesn't print an object address below the plot:

```python
class SymbulatePlot:
    def __init__(self, ax):
        self.ax = ax

    def __repr__(self):
        return ""
```

This change has two purposes: it lays the foundation for a future plot composition API (e.g., `.plot() + vline(x=0)`), and it makes a future thin interactivity conversion layer possible without revisiting the plot architecture.

Make this change in the same PR as the first round of plot fixes. The cost is about 10 lines of new code and one-line additions to each `plot()` method. All `result.py` plot methods can be updated in the same pass.

### Overlay Policy: Overlay whenever naturally possible

The rule for multiple `.plot()` calls in a single Jupyter cell is simple: **overlay whenever naturally possible.** A second `.plot()` call always attempts to draw on the same axes. There is no automatic side-by-side behavior — if a user wants side-by-side panels, that is a separate feature (deferred).

The three outcomes:

**Natural overlay:** draws on the shared axes with the next color in the cycle. No message needed. This is the behavior for scatter, impulse, histogram (1D), density (1D), rug, density/contour (2D), ecdf, and most other plot types.

**Overlay with a readability warning:** the overlay is produced, but a warning is printed explaining that the result may be hard to read and suggesting alternatives. This applies to cases like two 2D tile plots or two 2D histograms overlaid on the same axes (competing color scales). The warning text follows the same student-friendly standard as all other messages.

**Hard error:** the plot type creates its own multi-panel GridSpec layout, and a second `.plot()` call cannot join it. This applies to any plot using the `'marginal'` layout. The error message explains what went wrong and what to do instead.

Color cycling across overlay calls is handled by `get_next_color(ax)`, which is called once per `.plot()` call and advances the matplotlib color cycle automatically.

Side-by-side panels are a separate feature from overlay and are deferred. The intended design (likely `panel=True` or a `.compare()` method) should be documented now so the `SymbulatePlot` wrapper is designed to support it later.

### Plot Composition API: Deferred, but design now

A composition API — something like `RV(Normal()).sim(1000).plot() + vline(x=0)` — is desirable but not a this-summer priority. The naming should be intuitive plain English, not tied to any particular graphics package syntax. The audience is general users including people with minimal programming experience — not people who know ggplot2 or matplotlib.

Good naming examples: `vline(x=0)`, `hline(y=0.1)`, `title("My Plot")`, `x_label("Value")`. Avoid: `geom_vline(xintercept=0)`, `axvline(x=0)`, `set_xlabel("Value")`.

The vocabulary should be small and designed around actual pedagogical use cases: vertical and horizontal reference lines, shaded probability regions, a theoretical pdf/pmf curve overlaid on a simulation histogram, text annotations. Document the intended vocabulary in the design document even if implementation is deferred.

### Interactivity and Animation: Scaffold approach, deferred

Symbulate will not switch to an interactive plotting library. Instead, interactivity and animation will be offered through Jupyter widget scaffolds and template notebooks that wrap Symbulate's static `.plot()` output. This requires no architecture changes to Symbulate's core.

The primary tool is `ipywidgets`, which works in Jupyter and Colab without configuration and allows sliders, dropdowns, and play buttons to wrap any `.plot()` call. Template notebooks (one per major use case) are the delivery mechanism — students open the notebook, put their distribution in the marked cell, and run all.

Animation within Symbulate uses `matplotlib.FuncAnimation`, wrapped in a helper function that hides the complexity.

Implement static graphics first. Interactivity scaffolds are built after the static layer is stable.

---

## Work Sequence

### Phase 1: Design (do before any code)

All design decisions must be finalized and documented before Phase 2 begins. No code should be written until the design document is complete and reviewed by the team.

#### Task 1A: Default plot lookup table design

This is the most important design task and must be completed before `classify_data` is coded. The process:

**Step 1 — Run visual test cases using the reference fork.**
Use the reference fork at `github.com/kevindavisross/symbulate`, which has more plot types available than the main branch. For each data configuration below, run a variety of representative examples and look at the plots. The goal is to see what actually looks good, not to reason abstractly about it.

For each example, try every available plot type on that data and note which looks best and which alternatives are also reasonable.

Suggested test cases to run (at minimum):

*1D discrete-ish, small n:*
- `RV(Binomial(5, 0.3)).sim(10).plot()`
- `RV(Poisson(3)).sim(10).plot()`
- `BoxModel([1,2,3,4,5,6]).sim(10).plot()`

*1D discrete-ish, large n:*
- `RV(Binomial(5, 0.3)).sim(10000).plot()`
- `RV(Poisson(3)).sim(10000).plot()`
- `RV(Poisson(20)).sim(10000).plot()` — borderline case
- `RV(DiscreteUniform(1, 20)).sim(10000).plot()`

*1D continuous-ish, small n:*
- `RV(Normal()).sim(10).plot()`
- `RV(Exponential(1)).sim(10).plot()`
- `RV(Uniform(0, 1)).sim(10).plot()`

*1D continuous-ish, large n:*
- `RV(Normal()).sim(10000).plot()`
- `RV(Exponential(1)).sim(10000).plot()`
- `RV(Binomial(10000, 0.5)).sim(10000).plot()` — discrete but continuous-ish

*2D discrete × discrete:*
- `RV(Binomial(5, 0.3) * Binomial(5, 0.3)).sim(10000).plot()`
- `RV(Poisson(3) * Poisson(2)).sim(10000).plot()`
- `RV(Binomial(5, 0.3) * Binomial(5, 0.3)).sim(50).plot()`

*2D continuous × continuous:*
- `RV(Normal() * Normal()).sim(10000).plot()`
- `RV(Normal() * Normal()).sim(50).plot()`
- `RV(BivariateNormal()).sim(10000).plot()`

*2D discrete × continuous:*
- `RV(Poisson(3) * Normal()).sim(10000).plot()`
- `RV(Binomial(5, 0.3) * Exponential(1)).sim(10000).plot()`
- `RV(Poisson(3) * Normal()).sim(50).plot()`

*2D continuous × discrete:*
- `RV(Normal() * Poisson(3)).sim(10000).plot()`

*Process time point, discrete-valued:*
- `PoissonProcess(rate=1)[2].sim(10).plot()`
- `PoissonProcess(rate=1)[2].sim(10000).plot()`
- `PoissonProcess(rate=5)[10].sim(10000).plot()` — borderline

*Process time point, continuous-valued:*
- `BrownianMotion()[1].sim(10).plot()`
- `BrownianMotion()[1].sim(10000).plot()`

*Overlay behavior — try each of these in a single cell:*
- `RV(Normal()).sim(1000).plot()` followed by `RV(Normal(1,1)).sim(1000).plot()` — should overlay cleanly
- `RV(Binomial(5,0.3)).sim(10000).plot()` followed by `RV(Binomial(5,0.5)).sim(10000).plot()` — should overlay cleanly
- `RV(Normal()*Normal()).sim(1000).plot('scatter')` followed by `RV(Normal(1,1)*Normal()).sim(1000).plot('scatter')` — should overlay cleanly
- `RV(Normal()*Normal()).sim(1000).plot('tile')` followed by `RV(Normal(1,1)*Normal()).sim(1000).plot('tile')` — overlay attempted, readability warning expected
- `RV(Normal()*Normal()).sim(1000).plot(['scatter','marginal'])` followed by a second `.plot()` — hard error expected
- `BoxModel(['H', 'T']).sim(10).plot()`
- `BoxModel(['H', 'T']).sim(10000).plot()`

**Step 2 — Fill in the lookup table.**
After running the test cases, fill in this table. Each cell gets one default and a list of reasonable alternatives (to be offered in the suggestion message). The table should be included in the design document.

| Data configuration | Small n default | Small n alternatives | Large n default | Large n alternatives |
|---|---|---|---|---|
| 1D discrete-ish | | | | |
| 1D continuous-ish | | | | |
| 2D discrete × discrete | | | | |
| 2D continuous × continuous | | | | |
| 2D discrete × continuous | | | | |
| 2D continuous × discrete | | | | |
| Process time point, discrete-valued | | | | |
| Process time point, continuous-valued | | | | |
| 1D categorical/string | | | | |

**Step 3 — Decide on the suggestion message format.**
The message printed when the default plot is used should be friendly and actionable. Draft the wording here, not in code. Example:

> *"Showing an impulse plot (the default for this type of data). You might also try `type='hist'` or `type='density'`."*

Decide: should this message always appear, or only the first time, or only in certain contexts? A warning-style message that appears every time could become annoying. Consider whether it should be opt-out via a `verbose=False` argument.

**Step 4 — Decide the thresholds.**
The `n_unique_threshold` (default 40) and `n_small_threshold` (default 100) are named constants. After seeing the visual test cases, the team should confirm these numbers feel right or adjust them. Write down the reasoning for whatever values are chosen — this goes in `DECISIONS.md`.

#### Task 1B: Visual style guide

Decide and document the aesthetic standards that all plots must meet before any plot-specific code is written:

- **Color palette:** name it, justify it (accessibility, print-friendliness), list specific hex values
- **Figure size:** default width and height in inches
- **Font sizes:** axis labels, tick labels, titles, legend text
- **Default transparency (alpha):** separate values for histograms, scatter points, density overlays
- **Line widths:** density curves, impulse lines, sample paths
- **Spine style:** which spines are visible (top/right typically removed)
- **Grid:** yes/no, and if yes what style
- **Point style:** filled or unfilled, default size

Include example images of what "good" looks like. These decisions get encoded in a custom `.mplstyle` file.

#### Task 1C: Naming and vocabulary

A glossary of every user-facing term. Decide:

- Is the plot type argument still called `type`? (Current name collides with Python built-in — worth reconsidering)
- Is it `x_label` or `xlabel`? `color` or `colour`? `normalize` or something more descriptive?
- Names for suggestion message wording
- Names for any composition API geoms (even if implementation is deferred)

All naming should be intuitive to a general audience with minimal programming experience.

#### Task 1D: Backwards compatibility policy

- Will old parameter names be kept as aliases with deprecation warnings, or will clean breaks be made?
- Specific known cases: the `type=` parameter, the `jitter=` parameter
- For a package used in courses, breaking existing notebook code is a real cost — decide now

#### Task 1E: Plot prototype design

Before any plot-specific code is written, each plot type needs a **visual prototype** — a standalone Python script that produces exactly what the final plot should look like, using fake data. The prototype is the design artifact that all subsequent implementation work is measured against. It also serves as the basis for precise Claude Code prompts.

The prototype process has four steps. The key discipline is that aesthetic decisions are made during prototyping — not during implementation. Claude Code receives a finished design to implement, not a vague instruction to make something look good.

**Step 1 — Collect reference images.**
For each plot type, find 3–5 examples of what professional output looks like. Sources can include textbooks, R packages (ggplot2 output is a good reference), published papers, or other Python packages. The goal is to have a concrete visual target before writing any code. Save these images with the design document.

**Step 2 — Write a feature list for each plot type.**
Look at the reference images and translate visual impressions into concrete, enumerable decisions. Not "looks clean" but specific choices:

- Background color
- Grid lines: yes/no, color, weight
- Colormap name
- Color scale: start from 0, or clip low values? At what threshold?
- Axis label font size and style
- Tick label font size
- Spines: all four visible, or remove top and right?
- Colorbar: position, width, label
- Padding around data: fixed value or quantile-based?
- Rendering approach: e.g., `contourf` vs `imshow` for 2D density
- Grid resolution for KDE evaluation
- Smoothing: `interpolation` parameter value

This feature list goes in the per-plot-type spec card in Section 3 of the design document.

**Step 3 — Build the prototype in Claude chat (not Claude Code).**
Share the reference images and feature list with Claude in the chat interface and ask for a standalone prototype script — a self-contained Python file that generates just that plot type with simulated data. Iterate visually in the chat conversation until the output matches the reference images. This is the right tool for prototype iteration because it allows visual feedback and back-and-forth discussion. Claude Code is not the right tool for this step — it has no way to show you the output and adjust.

For example: share a ggplot2-style contour plot as a reference, describe the feature list, and ask Claude to produce a matplotlib prototype that achieves the same visual result. Revise until satisfied.

**Step 4 — Extract and lock in the decisions.**
Once the prototype looks right, extract the specific parameter values from the working code: exact colormap name, exact `vmin` expression, exact grid resolution, exact figure size, exact font sizes, exact spine configuration. These concrete values go into:
- The per-plot-type spec card (Section 3 of the design document)
- The `.mplstyle` file (for values that apply globally)
- The Claude Code prompt that implements the final version

**Priority order for prototypes.**
Complete prototypes for the highest-priority plot types before implementing any of them. Suggested priority order based on frequency of use:

1. 1D histogram (continuous, large n)
2. 1D impulse plot (discrete, large n)
3. 2D density plot (continuous × continuous, large n) — `contourf` approach
4. 2D scatter plot (large n)
5. 1D rug/dot plot (small n, both discrete and continuous)
6. Sample path plot (stochastic processes)
7. All remaining plot types

**Known decisions from prototype work already done.**
The 2D density plot prototype has been explored. Key findings:

- `contourf` is the correct rendering approach for 2D density, not `imshow`. It produces smooth filled contour bands without pixel granularity at any grid resolution.
- Axis limits should be based on quantile bounds (0.1th–99.9th percentile of each variable) plus fixed padding, not raw `min`/`max` which is sensitive to outliers.
- KDE evaluation grid should be 300×300 minimum. The current 100×100 produces visible granularity.
- `vmin` clipping (e.g., `vmin = Z.max() * 0.02`) is needed for `imshow`-style rendering to prevent low-density regions from washing out the color scale.
- Thin white contour lines (`linewidths=0.3`, `alpha=0.4`) overlaid on filled contours improve readability.
- Colorbar belongs on the right, sized and placed using `make_axes_locatable` from `mpl_toolkits.axes_grid1`.
- `viridis_r` or similar perceptually uniform colormap is preferred over `Blues` for 2D density.

These findings should be confirmed visually by the team and then locked into the spec card for the density plot before implementation begins.

### Phase 2: Infrastructure (code — everything else inherits these decisions)

No plot-specific code should be written before these items are merged.

1. **`SymbulatePlot` wrapper class** — add to `plot.py`, update all `plot()` methods in `results.py`, `result.py`, and `distributions.py` to return it.
2. **`classify_data()` function** — replace `is_discrete()` with the new function per the spec. Add the two threshold constants at the top of `plot.py`. Wire into the 1D and 2D dispatch logic in `results.py`.
3. **Default plot lookup table** — implement the table from Task 1A as a dict or function in `plot.py`. Implement the suggestion message.
4. **Custom `.mplstyle` file** — encode the style guide from Task 1B. Replace the fragile `seaborn-colorblind` stylesheet lookup in `results.py`.
5. **`dim > 2` cleanup** — replace the current connected-dot catch-all with a `NotImplementedError` and a student-friendly message explaining that joint plotting of 3+ variables is not yet supported.

### Phase 3: Fix broken existing plots

Work through known bugs in order of dependency — 1D before 2D, since 2D decisions build on 1D. For each fix: update the plot, update the docstring (NumPy style), update error messages (student-friendly), add a pytest test. All four in the same PR.

Known bugs to address:
- 2D histogram divide-by-zero when `normalize=True` and some bins are empty
- Colorbar layout (replace hardcoded `fig.add_axes([0, 0.1, 0.05, 0.8])` with `make_axes_locatable`)
- Marginal scale mismatch when using `['tile', 'marginal']`
- Density overlay on second call (two `.plot('density')` calls in one cell)
- `MatplotlibDeprecationWarning` in `plot.py:74` (adding axes with same arguments)
- Violin plot rendering for mixed discrete/continuous (currently produces empty plot)
- Scale on blank interior plot when `type='marginal'` alone

### Phase 4: Improve existing plots that work but look poor

Aesthetics, axis labels, default transparency, point styles. Lower risk than Phase 3 — behavior isn't changing, only appearance. Each improvement should still include a docstring update and a test that the plot renders without error.

Specific improvements:
- Add default axis label "Value of RV" for 1D plots
- Scatter plot: make points unfilled circles with better transparency by default
- Impulse plot: review whether dots-and-line style is still preferred
- Jitter: make automatic for discrete scatter plots; review current `jitter=True` behavior
- Color cycle: ensure consistent behavior across overlay calls
- Tile and density plots: ensure color scale always starts from 0

### Phase 5: Add new plot types from the reference fork

The reference fork has draft implementations of several plot types. Each needs cleanup, student-friendly error messages, style consistency with the new defaults, a docstring, and tests before merging. For each: read the reference fork implementation, identify what needs cleanup, write the spec card before touching code.

- Mosaic plot (discrete RVs)
- ECDF plot
- Shading regions on pdf/pmf plots
- CDF plots with area shaded
- Stacked bar charts for discrete-time discrete-state processes
- Density ridge plots for discrete-time continuous-state processes

### Phase 6: Interactivity scaffolds

After Phase 4 is stable:
- `ipywidgets` template notebooks for major use cases (parameter exploration, watch-simulation-build, compare two distributions)
- `FuncAnimation` wrapper function for animated simulation
- Evaluate thin Plotly conversion layer as a future option (likely second-year project)

---

## The Design Document

The design document is written during Phase 1 and is the spec that Claude Code works from in all subsequent phases. If a section is too vague to prompt from, it needs more design work — not implementation.

### Section 1: Visual Style Guide

See Task 1B above. The output of this section is both a written document and the `.mplstyle` file itself.

### Section 2: The `classify_data` Specification

The complete specification for `classify_data()`, including:

- The function signature with all parameters and defaults
- The full decision logic as pseudocode (not just the final version — show the reasoning)
- The named threshold constants and their values, with justification
- All edge cases and how each is handled: string outcomes, boolean outcomes, float with repeated values, single unique value, degenerate distributions
- The complete test case table from Task 1A with results filled in
- The lookup table mapping `(discrete_ish, small_n)` to default plot type, for every data configuration
- The default alternatives list for each cell
- The suggestion message wording, finalized

### Section 3: Per-Plot-Type Specification Cards

One card per plot type, for every plot type in the package (existing and new). Each card is completed after the prototype for that plot type has been approved (Task 1E). Each card:

- **Name:** what the user passes as `type=` (or equivalent)
- **When it is the default:** which cell(s) of the lookup table it occupies
- **Allowed alternates:** what other types are valid for this data configuration, and what the suggestion message says
- **Errors:** what should produce an explicit error with a student-friendly message (e.g., scatter on 1D data)
- **Function signature:** all parameters with types, defaults, and plain-English descriptions
- **Data types accepted:** 1D discrete-ish, 2D mixed, etc.
- **Reference images:** the images collected in Task 1E that define the visual target
- **Approved prototype:** link to or copy of the standalone prototype script from Task 1E
- **Exact parameter values from prototype:** colormap name, vmin expression, grid resolution, figure size, font sizes, spine configuration, colorbar placement — every concrete value needed to reproduce the prototype
- **Default axis labels**
- **Error messages:** exact text, written for a general audience
- **Overlay behavior:** one of three values — "natural" (draws on shared axes, no message), "warning" (draws on shared axes, prints readability warning), or "error" (multi-panel layout, cannot overlay). Include the exact warning or error text where applicable.
- **Pytest test cases:** renders without error; axis labels correct; correct error raised for invalid input; suggestion message appears when type not specified

### Section 4: Customization API

All user-facing parameters documented consistently. For each parameter:

- Name (plain English, not tied to any library's conventions)
- Type and valid values
- Default
- One-sentence plain-English description
- Which plot types it applies to

Also document which things matplotlib supports that Symbulate intentionally does not expose, and why.

### Section 5: Overlay and Multi-Panel Behavior

**The overlay rule:** overlay whenever naturally possible. A second `.plot()` call in the same cell always attempts to draw on the same axes. There is no automatic side-by-side behavior. The three possible outcomes are:

**Overlays naturally** — draws on shared axes with the next color in the cycle. No message needed. Examples: scatter, impulse, histogram (1D), density (1D), rug, density/contour (2D), ecdf.

**Overlays with a readability warning** — technically possible on shared axes but the result may be hard to read. The plot is produced, and a warning is printed below it explaining the issue and suggesting alternatives. Warning text must follow the same student-friendly standard as error messages. Examples: two 2D tile plots (competing color scales), two 2D histograms (competing color scales), two violin plots (overlapping shapes).

**Hard error** — geometrically impossible because the plot type creates its own multi-panel figure layout (GridSpec). A second `.plot()` call cannot join an existing GridSpec layout. Error text must explain what went wrong and what to do instead. Examples: any plot using `'marginal'` layout.

This section of the design document must specify:

- The complete list of which plot types fall into each category (natural / warning / error)
- The exact warning text for each warning case, written for a general audience
- The exact error text for each hard-error case, written for a general audience
- How color cycling works: `get_next_color(ax)` is called once per `.plot()` call and advances the cycle, so successive overlays automatically use distinct colors
- Side-by-side panels: deferred to a future phase. Document the intended design (likely an explicit `panel=True` parameter or a separate `.compare()` method) so the `SymbulatePlot` wrapper can be designed to support it.

### Section 6: Suggestion Message Spec

- Exact wording template for each data configuration
- Whether the message always appears or is opt-out
- If opt-out: the parameter name and default
- Whether the message appears when `type` is specified explicitly (it should not)

### Section 7: Naming and Vocabulary Glossary

The output of Task 1C, as a reference table: term → definition, applicable contexts, and any rejected alternatives with reasons.

### Section 8: Composition API Vocabulary (deferred implementation)

List every geom in scope, with parameters and plain-English description, even though implementation is deferred. Naming must be intuitive to a general audience, not tied to ggplot2, matplotlib, or any other package's conventions.

### Section 9: Backwards Compatibility Policy

The output of Task 1D. Specific cases documented explicitly.

### Section 10: Accessibility Requirements

- Does the package need to work in black and white? If yes, line styles must vary as well as colors when overlaying.
- Colorblind palette: name the specific palette, verify against standard colorblindness simulations
- Font size minimums for readability in projected slides vs. printed documents

---

## Using Claude and Claude Code

### Use Claude (chat) for design decisions and prototyping

Use Claude in chat (this interface) for two things: reasoning through design decisions before committing to them, and building visual prototypes of each plot type before any implementation begins.

**For design decisions:** reasoning through tradeoffs, catching edge cases in proposed logic, reviewing draft sections of the design document, getting a second opinion on a spec.

**For plot prototypes:** this is the most important use of Claude in chat for the graphics work. The process is:

1. Share reference images of what you want the plot to look like (screenshots from ggplot2, textbooks, other packages, or example images found online).
2. Describe the specific feature decisions — colormap, grid resolution, axis limit approach, spine style, colorbar placement, etc. — in the chat.
3. Ask Claude to produce a standalone prototype script using simulated data. Claude will generate and show the output in the conversation.
4. Iterate: "the colorbar is too wide," "the axis limits are cutting off the tails," "try a different colormap." Each round is fast because it happens in the chat, not in a code editor.
5. When the prototype matches the reference, extract the exact parameter values and write them into the spec card.

This is the correct tool for prototype iteration because the conversation allows visual feedback and discussion. Claude Code cannot show you the plot output — it can only write code. Using Claude Code for aesthetic iteration means writing code, running it, looking at the result, going back to Claude Code, describing what you saw, and repeating. That is much slower and produces decisions scattered across session history rather than documented in one place.

**The boundary between chat and Claude Code:** design decisions and prototypes happen in chat. Once a prototype is finalized and its parameters are written into the spec card, implementation happens in Claude Code using a precise prompt derived from the spec. The two tools are complementary, not interchangeable.

### Use Claude Code for implementation

Claude Code works best when given a precise spec. The discipline is: **do not open Claude Code until the spec is written.**

A good Claude Code prompt for this project:

> "In `plot.py`, replace the `is_discrete()` function with `classify_data(values)`. The function should: (1) convert values to a numpy array; (2) if dtype is object or bool, return `(True, small_n)` where small_n is `len(values) < N_SMALL_THRESHOLD`; (3) if dtype is float and all values are unique, return `(False, small_n)`; (4) if dtype is float and some values repeat, return `(n_unique <= N_UNIQUE_THRESHOLD, small_n)`; (5) otherwise return `(n_unique <= N_UNIQUE_THRESHOLD, small_n)`. Use the constants `N_UNIQUE_THRESHOLD = 40` and `N_SMALL_THRESHOLD = 100` defined at the top of `plot.py`. Add a NumPy-style docstring. Update the import in `results.py` to import `classify_data` instead of `is_discrete`. Do not change any other code. Write pytest tests for: Binomial(5,0.3) n=10000 (expect discrete_ish=True, small_n=False), Normal() n=10 (expect discrete_ish=False, small_n=True), float array [1.0,1.5,2.71,4.0] n=1000 (expect discrete_ish=True, small_n=False), string array n=100 (expect discrete_ish=True)."

A bad prompt:

> "Improve the discrete/continuous detection."

### One PR per task

Each PR should do exactly one thing. This makes review tractable and test cases obvious. Never merge without running pytest and manually verifying a notebook cell.

### The decisions log

Maintain `DECISIONS.md` at the repo root. Record what was decided and why. When a Claude Code session two weeks from now needs to know why the discrete threshold is 40, the decisions log has the answer. Update it whenever a design decision is finalized.

---

## The `CLAUDE.md` File

The existing project guide (`team/symbulate_project_guide.md` Section 6) describes the general process for writing `CLAUDE.md` together as a team at the first meeting. That guidance still applies. This section adds the specific content that must be included for the graphics overhaul, and explains how to maintain the file going forward.

### What `CLAUDE.md` is and why it matters

Every time anyone opens a Claude Code session in the Symbulate repository, Claude Code automatically reads `CLAUDE.md` before doing anything else. It is a standing briefing: everything Claude Code needs to know before touching this codebase. Without it, every session starts from scratch and Claude Code may make decisions that conflict with team conventions, the existing architecture, or decisions already made in planning.

Claude Code has no memory between sessions. A decision made in a planning conversation — like "stay on matplotlib" or "classify_data returns two booleans" — is completely unknown to Claude Code unless it is written in `CLAUDE.md`. The file is the only persistent communication channel between the team's decisions and Claude Code's behavior.

### When to create it

At the first team meeting, before anyone writes any code. The project guide recommends budgeting 90 minutes: 45 minutes exploring the codebase together, 45 minutes drafting. The graphics-specific sections below should be added at the same time or immediately after the graphics design document is finalized — whichever comes first.

**The file must exist and be committed before any Claude Code session is opened for graphics work.** Opening Claude Code without it means Claude Code will make its own assumptions about library choices, naming conventions, and architecture, which will conflict with the decisions documented here.

### How to maintain it

`CLAUDE.md` is a living document. Assign one team member as the owner each week (rotate with the project lead role). That person is responsible for:

- Adding any new decisions made during the week before the next Claude Code session that would need them
- Removing or updating entries that have been superseded
- Prompting the team at weekly standup: "Did we make any decisions this week that aren't in CLAUDE.md yet?"

When a PR is merged that changes a convention or architectural decision, updating `CLAUDE.md` is part of closing that PR — not a separate task to do later.

Keep entries concise. Claude Code reads the whole file before every session. Long, discursive entries slow it down and bury the most important information. Each entry should be the minimum needed for Claude Code to make the right decision without having to ask.

### What triggers a `CLAUDE.md` update

Update `CLAUDE.md` whenever:

- A new architectural constraint is established (e.g., "all plot types must call `get_next_color(ax)` at the start")
- A threshold or constant value is finalized (e.g., `N_UNIQUE_THRESHOLD = 40`)
- A naming decision is made (e.g., the type argument will remain named `type`)
- A new file is added that Claude Code will need to know about
- A new approved dependency is added
- A prohibition is established (e.g., "do not use `plt.subplots()` inside plot type functions — use `plt.gcf()` and `plt.gca()` instead")
- A design decision is reversed or updated

### Current `CLAUDE.md` template for this project

The file lives at the repo root: `CLAUDE.md`. Placeholders marked `[FILL IN]` must be completed before the file is committed — do not commit a file with unfilled placeholders.

```markdown
# Symbulate Developer Guide for Claude

## Project purpose
Symbulate is a Python package for simulating probability models, used by
undergraduate statistics students. Users may have minimal programming or
statistics background. All output — plots, error messages, docstrings —
must be understandable by a general audience without assuming prior knowledge.

## Environment
- Python 3.13.9, conda environment named `symbulate`
- Install in editable mode: `pip install -e .`
- Run tests: `pytest tests/`
- Formatter: Black (run before every commit)

## Key files
- `symbulate/plot.py` — all plotting helpers, SymbulatePlot wrapper,
  classify_data, DEFAULT_PLOT_TYPE lookup, style constants
- `symbulate/results.py` — RVResults.plot() dispatch method (main entry point)
- `symbulate/result.py` — plot() on individual TimeFunction and Tuple objects
- `symbulate/distributions.py` — plot() on distribution objects (true pdf/pmf)
- `symbulate/__init__.py` — public API; do not change without team discussion
- `tests/` — pytest test suite; every PR adds at least one test

## Codebase map (key files)
- `symbulate/probability_space.py` — BoxModel, DeckOfCards, core spaces
- `symbulate/distributions.py` — all probability distributions
- `symbulate/random_variables.py` — RV class
- `symbulate/random_processes.py` — RandomProcess base class
- `symbulate/poisson_process.py` — PoissonProcess
- `symbulate/gaussian_process.py` — BrownianMotion, GaussianProcess

## Graphics architecture — READ BEFORE TOUCHING ANY PLOT CODE

**Library:** matplotlib only. Do not introduce plotnine, altair, plotly,
lets-plot, or any other plotting library. No exceptions without team agreement.

**Approved layout helpers (already in environment, no new install needed):**
- `seaborn.JointGrid` — scatter with aligned marginals
- `seaborn.FacetGrid` — faceting
- `seaborn.pairplot` — scatterplot matrix
- `mpl_toolkits.axes_grid1.make_axes_locatable` — colorbar placement

**Overlay mechanism:** Two `.plot()` calls in one Jupyter cell produce one
combined plot. This works because matplotlib maintains global figure/axes
state via `plt.gcf()` and `plt.gca()`. Do not use `plt.subplots()` inside
plot type functions — it creates a new figure and breaks overlay. Always use
`plt.gcf()` and `plt.gca()`.

**Return value:** Every `.plot()` method returns a `SymbulatePlot` object.
`SymbulatePlot.__repr__` returns `""` so Jupyter does not print it.

**Color cycling:** Call `get_next_color(ax)` exactly once at the start of
every `.plot()` call (after getting the axes). This advances the color cycle
so successive overlays use distinct colors automatically.

**Style:** All visual defaults (colors, font sizes, transparency, line widths)
are set in `symbulate.mplstyle`. Do not hardcode any aesthetic values.

## Overlay policy
Overlay whenever naturally possible. A second `.plot()` call always draws on
the same axes. Three outcomes:
- Natural overlay: draws on shared axes, no message. (Most plot types.)
- Readability warning: draws on shared axes, prints warning below.
  Warning cases: two 2D tile plots, two 2D histograms on same axes.
- Hard error: multi-panel GridSpec layout cannot be joined.
  Error cases: any plot using 'marginal' layout.
See design document Section 5 for exact warning and error text.

## classify_data
Replaces `is_discrete()`. Returns `(discrete_ish: bool, small_n: bool)`.
Do not call `is_discrete()` anywhere — it no longer exists.

Logic:
- dtype object or bool → discrete_ish = True
- dtype float, all values unique → discrete_ish = False
- dtype float, some values repeat → discrete_ish = (n_unique <= N_UNIQUE_THRESHOLD)
- dtype int → discrete_ish = (n_unique <= N_UNIQUE_THRESHOLD)
- small_n = (len(values) < N_SMALL_THRESHOLD)

Constants at top of plot.py:
- N_UNIQUE_THRESHOLD = [FILL IN AFTER TEAM DECISION]
- N_SMALL_THRESHOLD = [FILL IN AFTER TEAM DECISION]

See design document Section 2 for full logic, edge cases, and worked examples.

## Default plot lookup table
Maps (discrete_ish, small_n, data_configuration) to default plot type.
Implemented as DEFAULT_PLOT_TYPE in plot.py.
[FILL IN AFTER TASK 1A IS COMPLETE — paste the finalized table here]

## Suggestion messages
When the user does not specify `type`, print a suggestion message after
the plot renders. Do not print it when the user specifies `type` explicitly.
Message format: [FILL IN AFTER TEAM DECISION]
Opt-out parameter: [FILL IN AFTER TEAM DECISION]

## Code standards
- PEP 8 enforced by Black. Run Black before every commit.
- NumPy-style docstrings on all public functions. If you touch a function,
  update its docstring in the same commit.
- Error messages: student-friendly. Say what went wrong and how to fix it.
  Do not say "invalid argument" or raise bare exceptions with no message.
  Example bad: `raise ValueError("invalid type")`
  Example good: `raise ValueError("'scatter' only works with 2D data. Your
  data has 1 dimension. Try type='hist' or type='density' instead.")`
- Every new function and every bug fix includes a pytest test. No exceptions.
- Do not add new package dependencies without flagging it explicitly.
- Do not change any public function signature without flagging it explicitly.
- Do not change __init__.py without flagging it explicitly.

## dim > 2 behavior
RVResults with dim > 2 currently falls through to a catch-all branch that
produces a connected-dot index plot. This is not correct behavior for joint
distribution visualization. If you encounter dim > 2 in plot code, raise
NotImplementedError with a student-friendly message explaining that joint
plotting of 3+ variables is not yet supported and what to do instead.

## Git workflow
- Never push to main or dev directly
- Branch from dev: git checkout dev && git pull origin dev && git checkout -b type/description
- PR target: calpoly-symbulate/symbulate, base branch: dev (NOT dlsun/symbulate)
- One PR per task
- Run pytest tests/ before pushing

## Do not
- Use plt.subplots() inside plot type functions
- Hardcode colors, font sizes, or transparency values
- Call is_discrete() — it no longer exists
- Push directly to main or dev
- Change public API without flagging it
- Add dependencies without flagging it
- Open a PR against dlsun/symbulate or against main
```

### Graphics-specific additions to make when each phase completes

The template above contains several `[FILL IN]` placeholders. These must be filled in as the team completes the corresponding design tasks — not left blank.

**After Task 1A (visual test cases and lookup table):**
- Fill in `N_UNIQUE_THRESHOLD` and `N_SMALL_THRESHOLD`
- Paste the finalized default plot lookup table
- Fill in suggestion message format and opt-out parameter name

**After Task 1B (visual style guide):**
- Add the name of the `.mplstyle` file
- Add a one-line note about the color palette chosen and why (e.g., "Okabe-Ito colorblind-safe palette — do not substitute other colors")

**After Task 1C (naming and vocabulary):**
- Add any naming decisions that affect what Claude Code should generate (e.g., "the plot type argument is called `type` — do not rename it to `plot_type` or `kind`")

**After Phase 2 is merged (infrastructure):**
- Confirm and update the `classify_data` constants section with actual values
- Add any architectural constraints discovered during implementation

**When new plot types are added (Phase 5):**
- Add each new plot type name and its overlay behavior category (natural / warning / error) to the overlay policy section

---

## Summary Checklist

### Before the first Claude Code session (do at first team meeting)
- [ ] `CLAUDE.md` drafted together as a team per project guide Section 6
- [ ] Graphics-specific sections added to `CLAUDE.md` (architecture, overlay policy, `classify_data` structure with placeholders)
- [ ] `CLAUDE.md` committed to repo root on `dev` branch
- [ ] `DECISIONS.md` created at repo root
- [ ] One team member assigned as `CLAUDE.md` owner for the first week

### Phase 1: Design (before any code)
- [ ] Task 1A: Run visual test cases on reference fork for all data configurations
- [ ] Task 1A: Run overlay test cases on reference fork
- [ ] Task 1A: Fill in default plot lookup table (defaults + alternatives for each cell)
- [ ] Task 1A: Decide N_UNIQUE_THRESHOLD and N_SMALL_THRESHOLD values with justification
- [ ] Task 1A: Draft suggestion message wording and decide opt-out policy
- [ ] Task 1B: Visual style guide finalized (color palette, font sizes, figure size, transparency, etc.)
- [ ] Task 1B: `.mplstyle` file drafted
- [ ] Task 1C: Naming and vocabulary glossary finalized
- [ ] Task 1D: Backwards compatibility policy documented
- [ ] Task 1E: Reference images collected for each priority plot type
- [ ] Task 1E: Feature list written for each priority plot type
- [ ] Task 1E: Prototype built and approved in Claude chat for 1D histogram
- [ ] Task 1E: Prototype built and approved in Claude chat for 1D impulse plot
- [ ] Task 1E: Prototype built and approved in Claude chat for 2D density plot
- [ ] Task 1E: Prototype built and approved in Claude chat for 2D scatter plot
- [ ] Task 1E: Prototype built and approved in Claude chat for 1D rug/dot plot
- [ ] Task 1E: Prototype built and approved in Claude chat for sample path plot
- [ ] Task 1E: Parameter values from each prototype extracted into spec cards
- [ ] Task 1E: Global style decisions from prototypes added to `.mplstyle` file draft
- [ ] Design document complete and reviewed by team
- [ ] `CLAUDE.md` updated: fill in all `[FILL IN]` placeholders from Tasks 1A–1E
- [ ] `DECISIONS.md` updated with all Phase 1 decisions and rationale

### Phase 2: Infrastructure (merge before any other code)
- [ ] `SymbulatePlot` wrapper class in `plot.py`
- [ ] All `plot()` methods return `SymbulatePlot`
- [ ] `classify_data()` replacing `is_discrete()`
- [ ] Threshold constants at top of `plot.py`
- [ ] Default plot lookup table implemented
- [ ] Suggestion message implemented
- [ ] Overlay warning implemented for warning-category plot types
- [ ] Hard error implemented for marginal layout plots when overlay attempted
- [ ] Custom `.mplstyle` file applied
- [ ] `dim > 2` catch-all replaced with student-friendly `NotImplementedError`
- [ ] `CLAUDE.md` updated to confirm final constant values and any new constraints discovered during implementation

### For every plot-type implementation PR
- [ ] Prototype approved in Claude chat before Claude Code session opened
- [ ] Spec card complete with exact parameter values from prototype
- [ ] Spec written before Claude Code session opened
- [ ] Docstring updated (NumPy style)
- [ ] Error messages student-friendly
- [ ] Suggestion message present where applicable
- [ ] pytest test added
- [ ] Black formatter run
- [ ] Output visually compared against approved prototype before merging
- [ ] Diff reviewed by a human before merge
- [ ] `CLAUDE.md` updated if PR establishes a new convention or constraint
- [ ] `DECISIONS.md` updated if PR resolves an open design question

### Weekly maintenance
- [ ] `CLAUDE.md` owner reviews file for stale or missing entries
- [ ] Team standup includes: "Any decisions this week not yet in CLAUDE.md?"
- [ ] Rotate `CLAUDE.md` owner to next team member
