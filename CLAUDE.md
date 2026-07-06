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

- `symbulate/plot.py` — all plotting helpers, SymbulatePlot wrapper,
  classify_data, DEFAULT_PLOT_TYPE lookup, style constants
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
Sequential/continuous plots (2D density, tile, hist2d) use **viridis**. Both
are set in `symbulate.mplstyle`; see `DECISIONS.md`, "Decision: Visual Style
Guide" for the full palette and justification.

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

## Default Plot Lookup Table

Maps (discrete_ish, small_n, data_configuration) to default plot type.
Implemented as DEFAULT_PLOT_TYPE in plot.py.
[FILL IN AFTER TASK 1A IS COMPLETE — paste the finalized table here]

## Suggestion Messages

When the user does not specify `type`, print a suggestion message after
the plot renders. Do not print it when the user specifies `type` explicitly.
Message format: [FILL IN AFTER TEAM DECISION]
Opt-out parameter: [FILL IN AFTER TEAM DECISION]

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
- Do not call `is_discrete()` — it no longer exists, use `classify_data()`
- Do not push directly to `main` or `dev`
- Do not change the public API without team discussion
- Do not add new dependencies without team agreement (current deps: `numpy`, `scipy`, `matplotlib`)
- Do not change `__init__.py` without flagging it explicitly
- Do not expose raw Python tracebacks in user-facing error messages
- Do not use type hints — they are not used in this codebase
- Do not open a PR against `dlsun/symbulate` or against `main`
