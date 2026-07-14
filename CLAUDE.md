# CLAUDE.md — Symbulate Developer Guide

[There are blanks for specific team decisions to be made as we brainstorm graphics & plotting]

## Project Overview

Symbulate is a Python package for simulating probability models, used by
undergraduate statistics students. Users may have minimal programming or
statistics background. All output — plots, error messages, docstrings —
must be understandable by a general audience without assuming prior knowledge.

## User Philosophy

- Error messages must explain what went wrong AND how to fix it. Never expose a raw Python exception to a student without context.
- Default behavior must work without configuration. No required arguments beyond what is mathematically necessary.
- Clarity over cleverness. Prefer readable code over compact code.

## Environment

- Python 3.13.9, conda environment named `symbulate`
- Install in editable mode: `pip install -e .`
- Run tests: `pytest tests/`
- Formatter: Black (run before every commit)

## Key Files

- `symbulate/plot.py` — all plotting helpers, `SymbulatePlot` wrapper
  (already implemented). `classify_data()` and the `DEFAULT_PLOT_TYPE`
  lookup are planned but **not yet implemented** — `is_discrete()` is
  still the live discreteness check today (see "classify_data" below).
- `symbulate/results.py` — RVResults.plot() dispatch method (main entry point)
- `symbulate/result.py` — plot() on individual TimeFunction and Tuple objects
- `symbulate/distributions.py` — plot() on distribution objects (true pdf/pmf)
- `symbulate/__init__.py` — public API; do not change without team discussion
- `tests/` — pytest test suite; every PR adds at least one test

## Codebase Map

| File | Contents |
|---|---|
| `distributions.py` | All distribution classes (Bernoulli, Normal, etc.), wrapping scipy.stats |
| `probability_space.py` | Core `ProbabilitySpace`, `BoxModel`, `DeckOfCards` (enhanced with suit/rank filtering), `Event` |
| `random_variables.py` | `RV` and `RVConditional` — maps probability space outcomes to numbers |
| `random_processes.py` | `RandomProcess` class for stochastic processes |
| `results.py` | `Results` and `RVResults` — stores and visualizes simulation output |
| `result.py` | Result types: `Scalar`, `Int`, `Float`, `Vector`, `InfiniteVector`, `TimeFunction` |
| `base.py` | Mixin classes: `Arithmetic`, `Transformable`, `Comparable`, `Filterable`, `Statistical`, `Logical` |
| `plot.py` | Plotting utilities |
| `table.py` | Table display for simulation results |
| `markov_chains.py` | Markov chain probability spaces |
| `gaussian_process.py` | `GaussianProcess`, `BrownianMotion` |
| `poisson_process.py` | Poisson process |
| `independence.py` | `AssumeIndependent` |
| `index_sets.py` | `Naturals`, `Integers`, `Reals`, `DiscreteTimeSequence` |
| `math.py` | Math utility functions |

## Graphics Architecture — READ BEFORE TOUCHING ANY PLOT CODE

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

**Style:** `symbulate/symbulate.mplstyle` holds every visual default that
applies the same way across all plot types — color palette, sequential
colormap, figure size, font sizes, spines, grid, and a global line-width
fallback. It does **not** hold per-plot-type values (histogram/scatter/density
alpha, impulse/density line width, unfilled-scatter styling) — matplotlib
rcParams are global and can't express "different alpha for different plot
types." Those live as named constants at the top of `plot.py` instead (see
`DECISIONS.md`, "Decision: `.mplstyle` Standards"). Do not hardcode aesthetic
values inline in a plot function — every value belongs in one of those two
places, not scattered inline.

**Color palette:** categorical palette is **Okabe-Ito** (7 hues, excluding
black) — colorblind-safe and print-friendly. Do not substitute other colors.
Sequential/continuous plots (2D density, tile, hist2d) use **viridis** —
plain `viridis`, not `viridis_r`; 0 (density/count) renders as viridis's
dark end, not light. Both are set in `symbulate.mplstyle`; see
`DECISIONS.md`, "Decision: Visual Style Guide" and "Decision: 2D Density /
Tile / Hist2D Colormap Direction" for the full palette and justification.

**Scatter point style:** filled circles (not unfilled/open) — this was
reconsidered and reversed back to filled; see `DECISIONS.md`, "Decision:
Visual Style Guide — Point Style."

## Naming Conventions

- `type=` stays `type=` — considered renaming to `kind=`, decided against
  it. Do not rename.
- `xlabel` / `ylabel` (no underscore) — everywhere, including future
  composition-API geoms. Do not use `x_label`/`y_label` or
  `set_xlabel`/`set_ylabel`.
- `color` (American spelling) — never `colour=`.
- `normalize` — keep as-is, do not rename.
- New plot-type functions should **not** add ad-hoc cosmetic override
  kwargs (`color=`, `label=`, etc.) to their user-facing surface — those
  are reserved for a future `.customize()` method (not yet designed). See
  `DECISIONS.md`, "Decision: Customization Parameters Deferred."

## Overlay Policy

Overlay whenever naturally possible. A second `.plot()` call always draws on
the same axes. Three outcomes:
- Natural overlay: draws on shared axes, no message. (Most plot types.)
- Readability warning: draws on shared axes, prints warning below.
  Warning cases: two 2D tile plots, two 2D histograms on same axes.
- Hard error: multi-panel GridSpec layout cannot be joined.
  Error cases: any plot using 'marginal' layout.
See design document Section 5 for exact warning and error text.

## classify_data

Will replace `is_discrete()`. **`is_discrete()` still exists and is still
the live discreteness check in `results.py` today** — imported at the top
of that file and called at the `is_discrete(counts.values())` /
`is_discrete(x_height)` / `is_discrete(y_height)` sites. `classify_data()`
has not been written yet. Do not remove or bypass `is_discrete()` until
`classify_data()` actually lands and every call site is migrated in the
same PR.

Planned return signature: `(discrete_ish: bool, small_n: bool)`.

Planned logic:
- dtype object or bool → discrete_ish = True
- dtype float, all values unique → discrete_ish = False
- dtype float, some values repeat → discrete_ish = (n_unique <= k)
- dtype int → discrete_ish = (n_unique <= k)
- small_n = (len(values) < n)

Provisional thresholds — **team expects to revise these; treat as a
working first pass, not settled** (see `DECISIONS.md`, "classify_data
Thresholds (Provisional)"):
- `n` (small/large-n crossover) = **40**, uniformly across every data
  configuration. Replaces the originally-proposed `N_SMALL_THRESHOLD = 100`.
- `k` (unique-value/discreteness cutoff) is **not a single global
  constant** — it varies by data configuration:
  - 1D discrete-ish / 1D categorical-string / process time point
    (discrete-valued): `k = 20`
  - 1D continuous-ish / 2D continuous×continuous / process time point
    (continuous-valued): `k = NA` (already continuous by dtype/uniqueness)
  - 2D discrete×discrete: `k = 5` per axis (25 combinations)
  - 2D discrete×continuous and continuous×discrete: `k = 10`
  - **Open implementation question:** how a per-configuration `k` composes
    with `classify_data()`'s per-variable, configuration-agnostic call
    signature is not yet decided. Resolve before wiring the lookup table
    into `results.py`.

See design document Section 2 and `DECISIONS.md` for full logic, edge
cases, and worked examples.

## Default Plot Lookup Table

Maps (data configuration, small-n vs. large-n) to a default plot type
plus alternatives. **Provisional — team expects to revise; filled in from
the Task 1A visual sweep, not a final spec.** Full table with alternatives
lives in `DECISIONS.md`, "Decision: Default Plot Lookup Table" (includes a
more tentative overlay-specific addendum too). Summary of defaults only:

| Data configuration | Small n default | Large n default |
|---|---|---|
| 1D discrete-ish | Dot plot | Impulse |
| 1D continuous-ish | Rug plot | Histogram |
| 2D discrete × discrete | Scatter with jitter | Tile / Heatmap |
| 2D continuous × continuous | Scatter | 2D Histogram |
| 2D discrete × continuous | Segmented rug plot | 2D Histogram / Tile |
| 2D continuous × discrete | Segmented rug plot | 2D Histogram / Tile |
| Process time point, discrete-valued | Dot plot | Impulse |
| Process time point, continuous-valued | Dot plot | Histogram |
| 1D categorical / string | Dot plot | Impulse |

## Suggestion Messages

Print a message after **every** plot renders — this fires whether or not
`type=` was specified (a deliberate broadening from the original plan).
See `DECISIONS.md`, "Decision: Suggestion Message Behavior" for the full
template and both wording variants. Summary:

- `type=` not specified: `"Currently Showing: {Default Type} Plot
  (Default) / Alternative Plots: {Alt1} (type='{alt1}'), ..."`
- `type=` specified explicitly: message still prints, but now labels
  whichever type *would have been* the automatic default, e.g.
  `"Currently Showing: Histogram / Alternative Plots: Impulse (Default),
  Density (type='density'), ..."`

Opt-out parameter: `hints` (default `True`), e.g. `.plot(hints=False)`.
**Note:** this name is carried over from an earlier, unchosen draft
template — not independently confirmed for the chosen wording above.

## Code Conventions

- PEP 8 enforced by Black. Run Black before every commit.
- CamelCase for classes (`ProbabilitySpace`, `RV`, `Normal`)
- `lowercase_with_underscores` for functions and variables
- No type hints — match the existing style
- Mixin-based design via multiple inheritance from `base.py`
- Operator overloading via `__add__`, `__mul__`, etc.
- Relative imports only (`.base`, `.result`, etc.)
- NumPy-style docstrings on all public functions. If you touch a function, update its docstring in the same commit.
- Do not introduce new patterns that conflict with existing ones

## Error Message Standard

Error messages must say what went wrong and how to fix it.

```python
# Bad
raise ValueError("invalid type")

# Good
raise ValueError(
    "'scatter' only works with 2D data. Your data has 1 dimension. "
    "Try type='hist' or type='density' instead."
)
```

## dim > 2 Behavior

RVResults with dim > 2 currently falls through to a catch-all branch that
produces a connected-dot index plot. This is not correct behavior for joint
distribution visualization. If you encounter dim > 2 in plot code, raise
NotImplementedError with a student-friendly message explaining that joint
plotting of 3+ variables is not yet supported and what to do instead.

## Testing Requirements

- Every new function gets at least one `pytest` test.
- Every bug fix gets a regression test.
- Tests are organized by module — one test file per source file.

| Test file | Tests for |
|---|---|
| `test_distributions.py` | All distribution classes |
| `test_probability_space.py` | `ProbabilitySpace`, `BoxModel`, `DeckOfCards`, `Event` |
| `test_random_variables.py` | `RV`, `RVConditional` |
| `test_results.py` | `Results`, `RVResults`, `tabulate()` |
| `test_result.py` | `Scalar`, `Vector`, `InfiniteVector`, etc. |
| `test_base.py` | Mixin classes |
| `test_plot.py` | Plotting utilities |
| `test_table.py` | Table display |
| `test_markov_chains.py` | Markov chains |
| `test_gaussian_process.py` | Gaussian processes |
| `test_poisson_process.py` | Poisson process |
| `test_random_processes.py` | Random processes |
| `test_independence.py` | `AssumeIndependent` |
| `test_index_sets.py` | Index sets |
| `test_math.py` | Math utilities |
| `test_smoke.py` | End-to-end import and basic usage smoke tests |

Run all tests with:
```
pytest tests/
```

## Git Workflow

- Never push to `main` or `dev` directly
- Branch from dev: `git checkout dev && git pull origin dev && git checkout -b type/description`
- Branch naming: `fix/issue-NUMBER-desc`, `feature/desc`, `docs/desc`, `test/desc`
- All PRs target **`dev`** on **`calpoly-symbulate/symbulate`** (NOT `dlsun/symbulate`)
- One PR per task
- Commit messages: `type: short description` (`fix`, `docs`, `feature`, `test`, `refactor`, `style`)
- Run `pytest tests/` before pushing

## Do Not

- Do not use `plt.subplots()` inside plot type functions
- Do not hardcode colors, font sizes, or figure/spine/grid values inline — use `symbulate.mplstyle`
- Do not hardcode per-plot-type alpha or line-width values inline — use the named constants at the top of `plot.py` (rcParams can't express per-plot-type values)
- Do not remove or bypass `is_discrete()` yet — it is still the live discreteness check in `results.py` today. `classify_data()` is planned but not yet written; only retire `is_discrete()` once `classify_data()` lands and every call site is migrated in the same PR.
- Do not rename `type=` to `kind=` or anything else — considered and decided against.
- Do not add ad-hoc cosmetic override kwargs (`color=`, `label=`, etc.) to a new plot-type function's user-facing surface — reserved for a future `.customize()` method (not yet designed).
- Do not use `viridis_r` (reversed) for 2D density/tile/hist2d — plain `viridis`, 0 = dark.
- Do not push directly to `main` or `dev`
- Do not change the public API without team discussion
- Do not add new dependencies without team agreement (current deps: `numpy`, `scipy`, `matplotlib`)
- Do not change `__init__.py` without flagging it explicitly
- Do not expose raw Python tracebacks in user-facing error messages
- Do not use type hints — they are not used in this codebase
- Do not open a PR against `dlsun/symbulate` or against `main`
