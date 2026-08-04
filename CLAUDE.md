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

- `symbulate/plot.py` — all plotting helpers, `SymbulatePlot` wrapper,
  `classify_data()`, the `DEFAULT_PLOT_TYPE` lookup, and the discreteness
  budget constants (`B_1D`, `K_2D`, `N_SMALL_THRESHOLD`) — all implemented
  and live (see "classify_data" below).
- `symbulate/results.py` — RVResults.plot() dispatch method (main entry point)
- `symbulate/result.py` — plot() on individual TimeFunction and Tuple objects
- `symbulate/distributions.py` — plot() on distribution objects (true pdf/pmf),
  plus the default plotting window: `xlim` is a lazy property whose value comes
  from `Distribution._compute_xlim` (see "Distribution Plotting Window" below)
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
| `gaussian_process.py` | `GaussianProcess`, `BrownianMotion`, `OrnsteinUhlenbeck`, `BrownianBridge`, `FractionalBrownianMotion` |
| `poisson_process.py` | `PoissonProcess` and `NonHomogeneousPoissonProcess` (time-varying rate) |
| `renewal_process.py` | `RenewalProcess` — counting process with any nonnegative interarrival distribution; `CompoundPoissonProcess` — running total of a jump drawn at each Poisson arrival (see "Compound Poisson Process" below) |
| `queues.py` | `GG1`, `MG1`, `GM1`, `GGs` — general-service queues via Lindley's recursion (see "Queues" below) |
| `random_walk.py` | `RandomWalk` — running total of i.i.d. steps (simple ±1 via `p=`, or any `step_dist`) |
| `time_series.py` | `MA` — moving-average process; home for the AR/ARMA/GARCH family as it lands |
| `hitting_times.py` | `hitting_time` — when a path first reaches a level; Gaussian-process family only so far (see "Hitting Times" below) |
| `diffusion_process.py` | `DiffusionProcess` — general Ito SDE, simulated approximately; `CIR` — named special case, simulated exactly (see "Diffusion Processes" below) |
| `independence.py` | `AssumeIndependent` |
| `index_sets.py` | `Naturals`, `Integers`, `Reals`, `DiscreteTimeSequence`, `TimeInterval` |
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

`classify_data()` is **implemented and live** in `plot.py`, and it is what
`results.py` uses to choose the default plot type (imported and called in the
1D, 2D, and categorical `.plot()` branches). It returns
`(discrete_ish: bool, small_n: bool)`. `is_discrete()` still exists as a
standalone utility in `math.py` (tested in `test_math.py`), but it is **no
longer the discreteness check in `results.py`** — do not reintroduce it there.

Logic (per variable):
- dtype object / string / bool → discrete_ish = True (categorical, can't bin)
- dtype float, all values unique → discrete_ish = False (continuous)
- dtype float, some values repeat → discrete_ish = (n_unique <= threshold)
- dtype int → discrete_ish = (n_unique <= threshold)
- small_n = (len(values) < N_SMALL_THRESHOLD)

Thresholds are a **crowding budget** anchored to the default histogram bin
count (30), so a discrete plot stays discrete exactly while it is no finer than
the histogram it would bin into (constants at the top of `plot.py`; values
**provisional**, to be tuned after inspecting
`team/discrete_continuous_threshold_tests.ipynb`):
- `B_1D = 30` — 1-D discreteness budget. `n_unique_threshold` defaults to this.
- `K_2D = 30` — 2-D **per-axis** discreteness cap. Each 2-D axis is judged
  independently: both axes discrete → tile; one over → mixed tile (that axis
  binned); both over → 2-D histogram.
- `N_SMALL_THRESHOLD = 100` — small/large-n crossover (global, unchanged).

The threshold is **passed in at the dispatch layer**, not baked into
`classify_data`: `results.py` passes `B_1D` in the 1-D branch and `K_2D` in
each of the two 2-D axis calls. `classify_data` keeps its per-variable dtype
gate; the budget is applied per call site by the branch that knows the
dimensionality (this is the non-circular resolution of the old
"per-variable vs per-configuration `k`" question). Categorical data is always
discrete regardless of the threshold; all-distinct float data is always
continuous.

See `DECISIONS.md`, "classify_data Thresholds (Budget Model)", for the full
rationale, the visual justification for `K_2D < B_1D`, and edge cases.

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

## Distribution Plotting Window (`xlim`)

`Distribution.xlim` is a **lazy property**, computed on first read by
`Distribution._compute_xlim` and cached. Never compute it in `__init__`: a
window is only needed for plotting, but `__init__` runs once per *draw* inside
`PoissonProcess` and `ContinuousTimeMarkovChain` (they build a fresh
`Exponential` every draw), so eager computation charges every simulated value
for a plot nobody asked for. This was a real 26-second bug.

One rule, applied to each end of the support independently — **a fixed bound is
used as-is, an unbounded side is cut at a quantile** (`_PLOT_TAIL = 0.001`):

| Support | Default `xlim` | Example |
|---|---|---|
| bounded both ends | full support | `Binomial(1000, 0.5)` → `(0, 1000)` |
| fixed lower only | `(lower, quantile(0.999))` | `Poisson`, `Exponential` start at 0 |
| fixed upper only | `(quantile(0.001), upper)` | none currently |
| unbounded both ends | `(quantile(0.001), quantile(0.999))` | `Normal` |

Bounds are read from scipy's own `support()` via `Distribution._scipy`, so a new
distribution gets the right window with **no per-distribution code**. In
`plot(xlim=)`: `None` applies the rule, `(a, b)` sets exact limits, and
`"zoom"` cuts both ends at a quantile — framing the distribution as if it had
no fixed bounds, so `Binomial(1000, 0.5)` zooms to ~`(451, 549)`.

A highest-density interval (HDI) used to set this window. It was **removed** —
it cost a root-find per distribution to buy a window only 5–26% narrower on
most distributions, and one still unreadable on the heavy-tailed ones it was
meant to help. See `DECISIONS.md`, "Decision: Default Plotting Window (HDI
Removed)" for the measurements.

## Diffusion Processes

`DiffusionProcess(drift, diffusion, x0, tol)` in `diffusion_process.py` solves
a general Itô SDE `dX = drift(X,t) dt + diffusion(X,t) dW`. It was prototyped
outside the package and merged in; the design write-up is
`team/models-and-sim-design/gaussian_and_diffusion_processes.md`.

Three things that are easy to get wrong:

- **It is approximate**, unlike every other process here. The Gaussian
  processes (`BrownianMotion`, `OrnsteinUhlenbeck`, `BrownianBridge`,
  `FractionalBrownianMotion`) are exact, because a Gaussian process has a
  closed-form conditional distribution. A general diffusion does not, so this
  uses bridge-corrected Euler–Maruyama refined by bisection, with `tol` as the
  only accuracy knob. It **is** exact when `drift` and `diffusion` are
  constant (i.e. Brownian motion). Say "approximate" in any docstring or demo
  rather than implying otherwise.
- **Seed `diffusion_process.rng`, not `np.random.seed`.** The module draws from
  its own `np.random.default_rng()`, which does not read NumPy's legacy global
  generator, so `np.random.seed(42)` silently does nothing. This was a real bug
  in the module's demo; `test_diffusion_process.py` has a regression test that
  fails if the module ever switches back to the global generator.
- **A `sqrt` in `diffusion` needs a floor**, e.g.
  `lambda x, t: sigma * np.sqrt(max(x, 0))`. A forward step can land slightly
  below 0 even in a model that should stay positive, and one `nan` poisons the
  rest of the path.

`CIR(reversion_rate, mean, scale, initial_value)` lives in the same module as a
named special case, the way `BrownianMotion` sits inside `gaussian_process.py`
next to the general `GaussianProcess`. It does **not** go through
`DiffusionProcess`: CIR has a known transition law — its value at any later
time is a scaled noncentral chi-square, the same distribution
`ChiSquare(df, noncentrality=)` wraps — so each new time is drawn in one shot,
**exactly**, with no step size to tune. Do not reimplement it as a
`DiffusionProcess` parameterization; that would reintroduce Euler–Maruyama
error the team decision explicitly removed. `test_cir.py` pins this by checking
that one jump to a time matches fifty steps to it. The one approximate corner
is asking for a time *between* two already-computed times, since a CIR path
pinned at both ends has no simple formula.

## Compound Poisson Process

`CompoundPoissonProcess(rate, jump_dist)` is the running total of a jump
drawn at each Poisson arrival. It lives in `renewal_process.py`, **not**
`poisson_process.py` — it reuses the interarrival machinery there, and the
risk processes that will build on it (Sparre Andersen, Cramér-Lundberg) are
renewal-flavored. See `MODEL-DECISIONS.md`, "Decision: Compound Poisson
Process."

Three things that are easy to get wrong:

- **`jump_dist` is deliberately not sign-checked.** A negative jump is a
  legitimate model (gains and losses, deposits and withdrawals), so the total
  is *not* monotone. Do not copy `_validate_interarrival_dist`'s
  nonnegativity check onto it — that check protects the nondecreasing-count
  invariant, which this process does not have. `jump_dist` is checked for
  exactly two things: it is a `Distribution`, and it is not multivariate.
- **`Var[X(t)] = rate * t * E[Y**2]`**, not (number of events) × `Var(Y)` —
  both the count and the sizes vary. The mean is `rate * t * E[Y]`. Use these
  closed forms when writing a test, not a simulated reference value.
- **`states[n]` is the total held during the n-th wait**, so `states[0]` is 0
  — the index-by-index convention `ContinuousTimeMarkovChainResult` uses. A
  result hand-built from plain Python lists supports
  `get_interarrival_times()` but not `get_arrival_times()`, which needs a
  `.cumsum()` — pass a `Vector` in a test fixture that needs it.

## Queues

Queues live in **two** files, split by whether the model is Markovian:

- `markov_chains.py` — `MM1`, `MMs`, `MMsK`, `MMss`, `MMsKN`, `MMInfinity`.
  Exponential service makes the *number in the system* a birth-death CTMC, so
  these are thin generator-matrix wrappers over `BirthDeathProcess`. Each path
  is a queue **length** over continuous time, and the unbounded ones need a
  `num_states` truncation.
- `queues.py` — `GG1`, `MG1`, `GM1`, `GGs`. Once service is not exponential the
  number in the system is *not* a Markov chain (remaining service depends on
  elapsed service), so there is no generator matrix to build. Each path is
  instead the **waiting time of customer `n`**, indexed by customer number,
  built by Lindley's recursion `W[n+1] = max(W[n] + S[n] - A[n+1], 0)` — exact
  for any nonnegative distributions, with no truncation. `GGs` generalizes that
  to `s` servers by tracking when each next comes free (`heapq`, earliest at
  position 0), which is the same recursion in clock time.

All four share `_QueueResult`, which owns the input sequences, the lazily
extended `waits` list, and the derived `arrival_times` / `sojourn_times` /
`departure_times`. A new queue discipline subclasses it and implements only
`_wait_at(n)` — do not duplicate the sequence plumbing. Note `departure_times`
is increasing only for a single server; with `s > 1` a short service can
overtake a long one.

Do not try to express a general-service queue as a `ContinuousTimeMarkovChain`,
and do not add `GG1`-family classes to `markov_chains.py`. Both nonnegativity
checks reuse `renewal_process._smallest_possible_time` / `_is_always_zero`
(the arrival stream of a `GG1` genuinely *is* a renewal process); a service
distribution that is always 0 is accepted, an interarrival one is not. A
utilization of `rho >= 1` is not an error — an unstable queue is a legitimate
thing to simulate. There are no `MGs`/`GMs` classes: pass an `Exponential` on
whichever side is Markovian. See `MODEL-DECISIONS.md`, "Decision: G/G/1 Queue
via Lindley's Recursion" and "Decision: G/G/s — Multi-Server Queues."

## Hitting Times

`hitting_time(process, level, ...)` in `hitting_times.py` answers "when does the
path first reach `level`?" — added by PR #272. Two things to know before
extending it:

- **It covers the Gaussian-process family only** (`BrownianMotion`,
  `BrownianBridge`, `OrnsteinUhlenbeck`, `FractionalBrownianMotion`,
  `GeometricBrownianMotion`, hand-built `GaussianProcess`). Anything else —
  random walks, Markov chains, queues, `DiffusionProcess` — raises
  `NotImplementedError` naming itself. This is the roadmap's **Tier B**; Tier A
  (discrete-time and jump processes, which is *easier and exact*) is **not
  built**, so it is the open gap, not a Tier-C-style approximation problem.
- Between two evaluated times a path can cross and come back, so the crossing
  is decided by a Bernoulli draw using the reflection-principle probability and
  then localized by bisection. Exact for Brownian motion and bridges at any
  `step`; approximate for other Gaussian processes. It draws from
  `hitting_times.rng`, so seed **that**, not `np.random.seed` — same trap as
  `diffusion_process.rng`.

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
| `test_gaussian_process.py` | Gaussian processes (incl. Ornstein-Uhlenbeck, Brownian bridge, fractional Brownian motion) |
| `test_poisson_process.py` | Poisson process and non-homogeneous Poisson process |
| `test_renewal_process.py` | Renewal process and compound Poisson process |
| `test_queues.py` | `GG1`, `MG1`, `GM1`, `GGs` (general-service queues) |
| `test_random_walk.py` | `RandomWalk` |
| `test_time_series.py` | `MA` (and the rest of the time-series family as it lands) |
| `test_hitting_times.py` | `hitting_time` |
| `test_diffusion_process.py` | Diffusion processes |
| `test_cir.py` | The `CIR` process |
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
- Do not reintroduce `is_discrete()` into `results.py` — `classify_data()` is the live discreteness check there now. (`is_discrete()` remains a standalone utility in `math.py`; leave it.)
- Do not hardcode the discreteness thresholds — use `B_1D` (1-D) and `K_2D` (2-D per axis) from `plot.py`, passed into `classify_data()` at the `results.py` dispatch (`B_1D` for 1-D, `K_2D` per axis for 2-D). Values are provisional (see `DECISIONS.md`).
- Do not set `self.xlim` in a new distribution's `__init__`, and do not compute a window there — `Distribution._compute_xlim` derives it from scipy's `support()` on first read (see "Distribution Plotting Window"). The only exception is a degenerate branch that skips `Distribution.__init__` and so has no scipy object.
- Do not reintroduce the highest-density interval (`_discrete_hdi_xlim`, `_continuous_hdi_xlim`, `_PLOT_COVERAGE`, `_hdi_window`) — removed by team decision; there are tests asserting it stays gone.
- Do not add a nonnegativity check to `CompoundPoissonProcess`'s `jump_dist` — negative jumps are intentional (see "Compound Poisson Process")
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
