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
| `poisson_process.py` | `PoissonProcess`, `NonHomogeneousPoissonProcess` (time-varying rate), and `CoxProcess` (random rate — see "Cox Process" below) |
| `renewal_process.py` | `RenewalProcess` — counting process with any nonnegative interarrival distribution; `CompoundPoissonProcess` — running total of a jump drawn at each Poisson arrival (see "Compound Poisson Process" below) |
| `queues.py` | `GG1`, `MG1`, `GM1`, `GGs` — general-service queues via Lindley's recursion (see "Queues" below) |
| `random_walk.py` | `RandomWalk` — running total of i.i.d. steps (simple ±1 via `p=`, or any `step_dist`) |
| `time_series.py` | `MA` — moving-average process; home for the AR/ARMA/GARCH family as it lands |
| `hitting_times.py` | `hitting_time` — when a path first reaches a level; jump, discrete-time, and Gaussian paths (see "Hitting Times" below) |
| `diffusion_process.py` | `DiffusionProcess` — general Ito SDE, simulated approximately; `CIR` and `MertonJumpDiffusion` — named special cases, both simulated exactly (see "Diffusion Processes" below) |
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

`CIR(reversion_rate, mean, scale, initial)` lives in the same module as a
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

`MertonJumpDiffusion(...)` sits in the same module and is also **exact**, but
by a different route: it is pure *composition*, not a new transition law. A
path adds a Brownian motion and a `CompoundPoissonProcess` of normal jumps
together in the exponent, then exponentiates — so it is a
`GeometricBrownianMotion` that can also lurch. Each path keeps both pieces as
`path.brownian_path` and `path.jump_path`.

The one thing not to break: the drift subtracts a **compensator**,
`jump_rate * (exp(jump_mean + jump_sd**2 / 2) - 1)`. Jumps multiply, and a
multiplier averages above 1 even when its log averages 0, so without that term
adding jumps would silently raise the mean. With it, `growth_rate` keeps
meaning the growth rate of the mean — the same contract
`GeometricBrownianMotion` has. `test_merton.py` pins this by checking the mean
is unchanged across four very different jump settings. Note that with large
jumps the *simulated* mean is noisy (the sd can reach 200), so judge that
contract against the closed form, not against one simulation.

Both follow the package-wide `initial` naming (see `MODEL-DECISIONS.md`, "One
Name for a Process's Starting Condition"). `CIR` also still accepts the older
`initial_value`, via `_resolve_initial`, because it shipped under that name;
`MertonJumpDiffusion` never did, so it takes `initial` only. `CIR`'s `initial`
defaults to `None` meaning "start at `mean`", so it passes `default=None` to
the helper rather than a number.

## Cox Process

`CoxProcess(intensity, step=)` in `poisson_process.py` is a Poisson process
whose rate is drawn at random *before* any events are generated. It is one
class covering several roadmap rows — a `Distribution` intensity is the
**mixed Poisson process** (gamma intensity → negative binomial counts), a
`ContinuousTimeMarkovChain` intensity is the **Markov-modulated Poisson
process** — so do not add separate classes for those. See `MODEL-DECISIONS.md`,
"Decision: Cox Process."

Four things that are easy to get wrong:

- **It does not thin, and it does not use `_CumulativeRate`'s quadrature on a
  drawn path.** Conditional on the intensity it is an NHPP, so it reuses the
  time-change counting — `CoxProcessResult` subclasses
  `NonHomogeneousPoissonProcessResult` — and only the *integral of the drawn
  intensity* is new. The roadmap's "build on non-homogeneous Poisson thinning"
  note predates the time-change decision; thinning is still not implemented
  anywhere.
- **How that integral is taken depends on the intensity, and two of the three
  ways are exact.** A single random rate → rate × time. A path that holds one
  value at a time (`DiscreteValued` with `interarrival_times`: a CTMC, a
  Poisson or renewal count) → `_StepCumulativeRate`, a sum of rate × holding
  time. Anything else → `_PathCumulativeRate`, a **left-hand sum on a fixed
  grid** of width `step` (default `_INTENSITY_STEP = 0.01`), which is the only
  approximate case. `step` is ignored by the exact two.
- **The grid must stay fixed, anchored at 0, and left-handed.** A drawn path is
  generated as it is *looked at*, so a general-purpose integrator probing it at
  times of its own choosing would fill in a different path per probe and report
  a meaningless error estimate; and a trapezoid or right-hand rule would make
  `cumulative_rate` non-monotone within a grid cell, which would let the count
  of events go *backwards*. There are tests pinning monotonicity and
  order-independence — do not "improve" the sum's accuracy without rereading
  them.
- **A per-draw cumulative rate, not a shared one.** Each path has its own
  intensity, so each gets its own `cumulative_rate` — the opposite of NHPP,
  where one is shared across every path. Only the not-random-at-all intensity
  (a plain function or number, which reduces to an NHPP) shares one.

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

## Continuous-Time, Discrete-State Processes — Required Interface

These processes all behave the same way: they sit at one value for a random
stretch of continuous time, jump, and sit again. **Every one of them must offer
the same four things**, and `tests/test_continuous_time_processes.py` asserts it
table-driven over all of them, so a new process cannot quietly skip one:

1. A `<Name>ProbabilitySpace` class, exported from `symbulate/__init__.py`.
2. `RV(P)` — the process itself, a discrete value at each continuous time.
3. `RV(P, interarrival_times)` — the times between jumps.
4. `RV(P, arrival_times)` — the times of the jumps.
5. `RV(P, states)` — the values visited, ignoring how long each lasted.

Items 3–5 come free from `DiscreteValued`: a Result only has to set `states` and
`interarrival_times` (an object with `.cumsum()` — an `InfiniteVector`, or a
`Vector` for an eagerly simulated finite path). `arrival_times` is their running
total. See "classify_data" style conventions in `math.py` for the three free
functions themselves.

**A wrapper class needs its own space too.** `MM1`, `MMs`, `MMsK`, `MMss`,
`MMsKN`, `MMInfinity` and `BirthDeathProcess` all used to build a generator
matrix inline, which left them with no space of their own; each now has one, with
the **rate formulas living in the space** and the process class copying the
parameters back out via `_init_from_space`. `MG1`/`GM1` follow the same shape
over `GG1ProbabilitySpace`. Do not put rate or distribution logic in the process
class — the space is what `RV(P, ...)` users get.

**Known gap:** `NonHomogeneousPoissonProcess` counts events on the
"expected count" scale and never converts back to clock time, so it has no
clock-time `interarrival_times`/`arrival_times`; `CoxProcess` inherits the gap,
since `CoxProcessResult` builds on that class. Fixing it needs a numerical
inverse of the cumulative rate. Both are in `TIME_CHANGED` in
`test_continuous_time_processes.py`, which asserts the three views they *do*
support and pins the gap, so the test fails (and they move up into the main
table) once it is closed.

**One deliberate oddity:** `SIR`/`SEIR` states are whole *vectors* of
compartment counts, not single numbers, and their final holding time is `inf`
(the outbreak has ended). They are eagerly simulated, so their sequences are
finite `Vector`s rather than `InfiniteVector`s.

## Queues

Queues live in **two** files, split by whether the model is Markovian:

- `markov_chains.py` — `MM1`, `MMs`, `MMsK`, `MMss`, `MMsKN`, `MMInfinity`.
  Exponential service makes the *number in the system* a birth-death CTMC, so
  these are thin generator-matrix wrappers over `BirthDeathProcess`. Each path
  is a queue **length** over continuous time, and the unbounded ones need a
  `num_states` truncation.
- `queues.py` — `GG1`, `MG1`, `GM1`, `GGs`. Once service is not exponential the
  number in the system is *not* a Markov chain (remaining service depends on
  elapsed service), so there is no generator matrix to build. **The state is
  still the number in the system over continuous time**, same as the `MM`
  family: `X[t]` is how many customers are there at time `t`, and a path is a
  `ContinuousTimeFunction`. It is reached the long way round — Lindley's
  recursion `W[n+1] = max(W[n] + S[n] - A[n+1], 0)` gives each customer's wait,
  hence their departure time, and arrivals and departures are merged into the
  count. Exact for any nonnegative distributions, with no truncation. `GGs`
  generalizes the recursion to `s` servers by tracking when each next comes
  free (`heapq`, earliest at position 0).

All four share `_QueueResult`, which owns the customer-level sequences
(`waiting_times`, `service_times`, `sojourn_times`, `customer_arrival_times`,
`customer_departure_times`), the merged event stream, and the count itself. A
new queue discipline subclasses it and implements only `_wait_at(n)` — do not
duplicate the plumbing.

Two naming traps, both inherited from matching the `MM` queues:
- `path.interarrival_times` and `arrival_times(path)` are the **event-level**
  view — holding times between changes in the count, and the times of those
  changes — exactly as for a CTMC, where an event is an arrival *or* a
  departure. The customer-level times are `customer_arrival_times` /
  `customer_departure_times`. Don't "fix" one into the other.
- `customer_departure_times` is increasing only for a single server; with
  `s > 1` a short service can overtake a long one. The count is unaffected.

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
path first reach `level`?" — Tier B added by PR #272, Tier A since. It is **one
function with per-path-type dispatch**, not one function per family; keep it
that way (see `MODEL-DECISIONS.md`, "Decision: Hitting Times — Tier A").

- **Three families, dispatched on the drawn path, in this order:** a pure-jump
  path (`DiscreteValued` whose `get_states()` *and* `get_interarrival_times()`
  both work) is walked jump by jump; a discrete-time path (`InfiniteTuple` or
  `DiscreteTimeFunction`) is walked step by step; a Gaussian path (`cov_func` +
  `observed`) goes through `_prepare` and the reflection-principle machinery.
  Tier C (`DiffusionProcess`, `CIR`, `MertonJumpDiffusion`) still raises
  `NotImplementedError`. `MertonJumpDiffusion` is Tier C, **not** a jump path:
  it wanders continuously between its jumps.
- **Tier A is exact, and `step`/`tol` are ignored there.** They exist to deal
  with what a *continuous* path does between two evaluated times; a jump path
  can only reach a level at a jump and a discrete-time path has nothing between
  its steps, so there is nothing to scan or localize. Do not "improve" Tier A
  by adding interpolation.
- **Reaching a level means reaching it *or passing it*** for Tier A — a walk
  going 4 → 6 never sits at 5. Do not change this to equality; `_reached`
  is the single place it is decided.
- **`NonHomogeneousPoissonProcess` and `CoxProcess` are deliberately not Tier
  A**, even though their counts jump: their jump times are known on the
  expected-count scale, not the clock, so locating one means inverting
  `cumulative_rate` — which the NHPP decision rejected. They get their own
  explicit `NotImplementedError` in `_prepare`; keep it before the generic one.
- **`hitting_time` dispatches on `RV`, not `RandomProcess`.** Most Tier A
  processes (`MarkovChain`, `ContinuousTimeMarkovChain`, `RandomWalk`, `MA`,
  `GG1`, `GGs`) are plain `RV`s, so an `isinstance(..., RandomProcess)` check
  silently treats them as sample paths. This was the bug that made the first
  Tier A draft fail.
- Between two evaluated times a *Gaussian* path can cross and come back, so the
  crossing is decided by a Bernoulli draw using the reflection-principle
  probability and then localized by bisection. Exact for Brownian motion and
  bridges at any `step`; approximate for other Gaussian processes. It draws
  from `hitting_times.rng`, so seed **that**, not `np.random.seed` — same trap
  as `diffusion_process.rng`.
- **An epidemic path is turned away inside the jump branch, not before it.**
  PR #283 gave `SIR`/`SEIR` paths `states` and `interarrival_times`, so they
  match `_is_jump_path`; the compartment-vector state is then rejected by
  `_numeric_value`. Asking about one compartment (`path.I`) does not work either
  — it is a bare `_BoundedTimeFunction` with no states — so do not put that
  suggestion back into either message.
- **Still open:** `upcrossings` (the *sequence* of crossing times, as a lazy
  `InfiniteVector`) is its own roadmap row and is not built. `start_time` is
  the hook it will use. Hitting times for a single epidemic compartment are
  also unbuilt, and would need the compartment to expose its jump times.

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

**There is now a way to plot 3+ simulated variables: `.plot(pairs=True)`** — a
matrix of panels, mirroring `MultivariateDistribution.plot(pairs=True)` so a
simulation and its distribution can be read side by side. See "Pairs Matrix of
Simulated Results" below.

`.plot()` with **no** arguments on dim > 2 still falls through to the old
catch-all branch that produces a connected-dot index plot. That is still not
correct behavior for joint distribution visualization — the intended fix is to
raise NotImplementedError pointing at `pairs=True`, and it was deliberately left
alone when the pairs matrix landed (that change alters existing behavior, so it
needs its own decision).

## Pairs Matrix of Simulated Results

`RVResults.plot(pairs=True, dims=None)` in `results.py`. The panels are chosen
by the **same** classification the 1-D and 2-D dispatches use, always in the
large-sample form so a matrix never mixes a mesh with a scatter:

| Panel | Data | Type |
|---|---|---|
| diagonal | continuous-ish (`B_1D`) | `density` |
| diagonal | discrete-ish | `impulse` — a density over repeated values would smear a pmf |
| off-diagonal | both discrete (`K_2D` per axis) | `tile` |
| off-diagonal | both continuous | `hist2d` |
| off-diagonal | mixed | `tile`, continuous axis binned |

Conventions shared with the theoretical version, by design — change both or
neither: lower triangle only, `JOINT_PAIRS_MAX_DIM` cap, `JOINT_PAIRS_PANEL_SIZE`
per panel, `JOINT_PAIRS_OVERLAY_ERROR` when the figure already has a plot,
`X1`-style labels on the outer edges only, no per-panel titles (each panel's
own type title is cleared — the `"Pairs Plot"` suptitle names the layout).

**One deliberate divergence:** the simulated matrix labels the **whole** left
column, top-left panel included, so every row is named (`X1`, `X2`, `X3` down
the side — seaborn `PairGrid`'s convention). The theoretical version leaves that
one panel's y-label blank on the grounds that a diagonal panel's y-axis is a
density rather than the variable. Worth reconciling: the theoretical side needs a
one-line change (`if col == 0 and row != col:` → `if col == 0:`) to match.

Three implementation notes:
- **Diagonal panels route through `.plot()`** (full reuse of the 1-D dispatch)
  after `plt.sca(ax)` — which works because every helper draws on `plt.gca()`.
  **Joint panels call `make_tile`/`make_hist2d` directly**, because the 2-D
  dispatch hardcodes `colorbar=not marginal` and a colorbar per panel would
  spend the figure on scales instead of data.
- **One bin count for the whole matrix.** Equal-width bins over the same column
  of data with the same count give identical edges, which is what makes a column
  comparable. Don't pass `bins` to `make_tile` when both axes are discrete — it
  warns.
- `dims` is only meaningful with `pairs=True`; on its own it raises rather than
  leaking into matplotlib.

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
| `test_poisson_process.py` | Poisson process, non-homogeneous Poisson process, and Cox process |
| `test_renewal_process.py` | Renewal process and compound Poisson process |
| `test_queues.py` | `GG1`, `MG1`, `GM1`, `GGs` (general-service queues) |
| `test_random_walk.py` | `RandomWalk` |
| `test_time_series.py` | `MA` (and the rest of the time-series family as it lands) |
| `test_hitting_times.py` | `hitting_time` |
| `test_continuous_time_processes.py` | The interface **every** continuous-time discrete-state process shares (see "Continuous-Time, Discrete-State Processes" below) — table-driven over all of them |
| `test_diffusion_process.py` | Diffusion processes |
| `test_cir.py` | The `CIR` process |
| `test_merton.py` | The `MertonJumpDiffusion` process |
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
- Do not add separate `MixedPoissonProcess` or `MarkovModulatedPoissonProcess` classes — both are `CoxProcess` with a different `intensity` (see "Cox Process")
- Do not add interpolation, a `step` scan, or an equality test to `hitting_time`'s jump/discrete-time branches — they are exact, and a jump path reaching a level means reaching *or passing* it (see "Hitting Times")
- Do not narrow `hitting_time`'s process check back to `RandomProcess` — most Tier A processes are plain `RV`s (see "Hitting Times")
- Do not change `CoxProcess`'s intensity grid to a trapezoid/right-hand rule, a variable mesh, or `scipy.integrate.quad` — a fixed left-handed grid anchored at 0 is what keeps the event count from going backwards, and there are tests pinning it (see "Cox Process")
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
