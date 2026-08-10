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
| `time_series.py` | `MA`, `AR`, `ARMA`, `GARCH`, `ARCH` — time-series processes |
| `branching_process.py` | `GaltonWatson` — a family tree that may die out or grow without bound |
| `hitting_times.py` | `hitting_time` — when a path first reaches a level; jump, discrete-time, and Gaussian paths. `upcrossings` — the whole sequence of crossing times, jump and discrete-time paths only (see "Hitting Times" below) |
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

**2D density surfaces are banded by default.** `make_density2D` and
`make_joint_pdf` both default to `contour=True`, so a 2-D density (simulated
`type="density2d"`, a theoretical joint pdf, and every continuous joint panel of a
pairs matrix) is drawn as `DENSITY2D_LEVELS`/`JOINT_PDF_LEVELS` discrete bands with
thin white outlines, which can be read against the colorbar. `contour=False` gives
the old smooth gradient (`*_CONTINUOUS_LEVELS` bands). Note the panel title follows
the mode — "Contour Plot" / "Joint Contour Plot" by default, "2D Density Plot" /
"Joint PDF Plot" with `contour=False` — while `PLOT_DISPLAY_NAME["density2d"]` in
the suggestion note still reads "2D Density Plot", since that names the `type=`
token rather than the shading.

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
  Error cases: a simulated plot drawn with `marginal=True`, **every**
  theoretical two-variable plot (they always show the strips), and any pairs
  matrix.
See design document Section 5 for exact warning and error text.

A simulated 2-D plot overlays normally, because it is a single panel unless
`marginal=True` is asked for — which is exactly why that keyword stayed
opt-in. A theoretical two-variable plot has no such escape hatch: it always
builds the three panels, so it can never be overlaid.

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
- **The first two families are picked by `_tier_a_reader`, not inline.** It
  returns `(search, read_one)` — the value-by-value walk for whichever kind of
  path it is, plus a one-value probe — and is the single definition of "Tier A"
  that `hitting_time` and `upcrossings` both use. A **fourth** exactly-readable
  path kind belongs in there, not at either call site.
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
- **Still open:** hitting times for a single epidemic compartment, which would
  need the compartment to expose its jump times.

`upcrossings(process, level, max_time=100.0, start_time=0.0)` in the same module
is the *sequence* of crossing times — a lazy `InfiniteVector` for a path, an `RV`
of one for a process (index it: `upcrossings(X, level=3)[2]`). See
`MODEL-DECISIONS.md`, "Decision: Upcrossings — Tier A Only."

- **Tier A only, and Tier B is refused on purpose, not unbuilt.** A continuous
  path recrosses a level infinitely often in every stretch of time after it
  first touches it, so there is no first/second/third crossing to list and any
  count would just measure how finely the path was looked at. A Gaussian or
  `GeometricBrownianMotion` path therefore raises `NotImplementedError` pointing
  at `hitting_time(..., start_time=)`. **Do not "add Tier B"** by counting one
  crossing per `step`-sized window, and do not add a `step`/`tol` argument —
  they do not exist on this function.
- **A crossing has to arrive from strictly below**, which is what the `strict`
  flag on `_reached` is for: finding one crossing is *two* searches, `sign=-1,
  strict=True` to get properly under the level and then `sign=+1, strict=False`
  to rise back to it. Without the strict half, a path sitting *on* the level
  (4 → 5 → 5 → 6) would report a fresh crossing at every step.
- **Consequences worth knowing before "fixing" them:** a path already at or
  above the level at `start_time` is not crossing it, so `upcrossings(...)[0]`
  can be *later* than `hitting_time(...)`; and a monotone count (`PoissonProcess`,
  `RenewalProcess`) crosses at most once, so `upcrossings(N, level=0)` is all
  `inf`. Both are tested.
- **The bad-path check is deliberately eager.** The sequence is lazy, so
  `read_one()` is called up front — otherwise an epidemic path or a
  name-labelled `MarkovChain` would be accepted quietly and only complain when
  someone indexed the result.
- Exhaustion is recorded on `_UpcrossingWalk.finished`, so a sequence that has
  run past `max_time` does not re-walk the path for every later index.

## Suggestion Messages

Print a message after a plot renders — this fires whether or not
`type=` was specified (a deliberate broadening from the original plan).
See `DECISIONS.md`, "Decision: Suggestion Message Behavior" for the full
template and both wording variants. Summary:

- `type=` not specified: `"Currently Showing: {Default Type} Plot
  (Default) / Alternative Plots: {Alt1} (type='{alt1}'), ..."`
- `type=` specified explicitly: message still prints, but now labels
  whichever type *would have been* the automatic default, e.g.
  `"Currently Showing: Histogram / Alternative Plots: Impulse (Default),
  Density (type='density'), ..."`

Control parameter: `suggest` (default `None`), e.g. `.plot(suggest=False)`.
`None` shows the message only on the **first** `.plot()` of the session,
`True` shows it on every call, `False` never shows it (see
`plot.should_show_suggestion`). There is no `hints` parameter — that name
came from an earlier, unchosen draft template and was never implemented;
passing it raises rather than quietly opting out.

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

**`.plot()` on 3+ simulated variables draws the pairs matrix** — that is the
default, with no argument needed. The lookup table has an `"nD"` configuration
(`default: "pairs"`, `alternatives: ["path"]`) so the choice lives with every
other default rather than being special-cased, and the suggestion note reads
`Currently Showing: Pairs Plot (Default) / Alternative Plots: Path Plot
(type = "path")`.

`type="path"` keeps the old behavior: one connected-dot line per realization
against its index. Any other `type` on 3+ variables raises, naming both options.

**There is no `pairs=` keyword.** It was how the matrix used to be asked for; a
stray `pairs=True` now raises a message saying it is the default instead of
leaking into matplotlib as a `Rectangle.set()` error.

**The gate is `self.dim is not None and self.dim > 2`, and that `is not None`
matters.** A random process's `.sim(n)` results are time functions, not tuples of
numbers, so their `dim` is `None` — they fall through to the same path branch
they always used, untouched. Verified across `GG1`, `RandomWalk`,
`ContinuousTimeMarkovChain`, `PoissonProcess`, `BrownianMotion`, `MM1`, and
`SIR`. Don't widen that condition to `self.dim != 2` or similar.

**The theoretical side matches**: `MultivariateDistribution.plot()` on 3+
freely varying components draws the matrix too, and its `pairs=` keyword is gone
as well (a stray one raises the same kind of message). Note the sum-constrained
families count *free* variables — a 3-category `Multinomial` has 2, so it still
draws its one joint plot.

**Choosing which variables to show is asymmetric on purpose.** A distribution
takes `variables=`, Python-indexed, and the count decides what is drawn: **one**
(`variables=2` or `variables=[2]`) is that variable's own marginal pdf/pmf,
**two** is their single joint plot, **three or more** is a matrix of those.
Labels follow the numbers asked for, so `variables=[0, 1, 3]` draws
`Variable 1`, `Variable 2`, `Variable 4`. A one-variable request is honored even
when the joint plot would be refused for varying in only one direction (a
two-category `Multinomial`'s marginal is a `Binomial` — a fine thing to plot),
so that check is made *after* the single-variable case. Simulated
results have **no such argument**: index the random variable before simulating,
`X[[0, 2]].sim(1000).plot()`, since that already exists and avoids two ways to say
the same thing. Both removed keywords (`dims=` on either side, `pairs=`) raise a
message naming the replacement rather than reaching matplotlib.

## Two Variables: Joint Plus Each Variable's Own

**Three panels** — the joint distribution in the main panel, and each
variable's own distribution in a strip beside the matching axis (above for x,
right for y). `setup_marginal_axes(fig)` in `plot.py` builds the layout and
**both sides call it**, so they cannot drift apart; `MARGINAL_GRID`,
`MARGINAL_GRID_RIGHT` and `MARGINAL_COLORBAR_RECT` are its geometry.

**Who gets them is asymmetric, deliberately:**

| | Strips | Keyword |
|---|---|---|
| `RVResults.plot()` (simulated) | only with `marginal=True` | `marginal=`, default `False` |
| `MultivariateDistribution.plot()` (theoretical) | always | none — a stray `marginal=` raises |

The reason is overlay. Two plots of simulated data are routinely compared on
one set of axes, and the three-panel layout can't be shared — so making it the
default cost every 2-D overlay, and it was reverted. An exact distribution has
no such need, so it always shows the strips and takes no keyword.
`type="marginal"` is not a value on either side.

Things to respect:
- **What happens to the title differs by side.** A simulated `marginal=True`
  plot *clears* the main panel's title (no room — the strip sits on top of it)
  and sets no figure title. A theoretical two-variable plot moves it to
  `fig.suptitle`, since that is its only title. So read a theoretical joint
  plot's type with `plt.gcf().get_suptitle()`, and a simulated 2-D plot's with
  `plt.gca().get_title()`.
- **`plt.gca()` is the joint panel.** Drawing the strips and the colorbar
  moves the current axes (`fig.add_axes` makes its axes current), so both
  sides call `plt.sca(ax)` at the end.
- **`marginal=True` with `type="mosaic"` raises.** A mosaic already shows
  both marginals itself (column widths and its own marginal column), so strips
  would draw each of them twice. Pairs-matrix panels need no such guard: they
  never pass `marginal=True`.
- **The theoretical strips are exact.** They come from `_marginal_1d(i)`, the
  closed-form marginal, drawn by the univariate `Distribution.plot()` — not a
  slice or a sum over the joint surface. `_plot_marginal_panel` draws the
  right-hand strip by drawing it the normal way and then **transposing the
  artists** (`set_data`, `set_offsets`), so the sideways version cannot drift
  from the upright one in styling.
- **`ax=` on a theoretical plot draws the joint alone.** One axes has no room
  for strips, and a caller who named the axes is placing the plot in a layout
  of their own. This is the only way to get a bare joint panel.
- **Overlay is a hard error for all 2-D plots** — see "Overlay Policy" above
  for what that cost.
- **A strip's rug ticks are sized against the joint panel, not the strip.**
  `make_rug` measures its ticks as a fraction of *its own* axes (so a tick
  keeps its size when a real y-scale is drawn on it later), which in a strip
  means a visibly shorter tick. `marginal_rug_tick_height(main_ax, marg_ax,
  orientation)` cancels the size difference out; pass it as `make_rug`'s
  `tick_height`.
- **A strip's frequency axis is thinned to `MARGINAL_FREQ_TICKS`.** A strip is
  a fraction of the joint panel's size, so a full-size tick count runs
  together (`0, 2, 4, 6, 8`). `thin_marginal_frequency_ticks(marg_ax,
  orientation, integer=)` caps it, on both sides; `integer=True` for a count
  axis, since half a simulated value doesn't exist. **A dot plot needs more
  than the locator**: `_dotplot_relayout` rebuilds its own locators on every
  draw, so the helper also records the cap as `ax._symbulate_freq_ticks` and
  that rebuild honors it. Set the locator without recording the cap and the
  thinning silently reverts on the next render — it looks right until you save
  the figure.

Mixed discrete/continuous pairs are not special-cased here: the joint pmf mesh
is laid out at real values (`make_joint_pmf`'s `extent`), so a strip drawn at
real values lines up without the rank-index conversion `tile` needs. The
strips are locked to the joint panel's *own final limits* (`set_xlim` then
`sharex`) rather than re-deriving each plot type's extent formula.

## Pairs Matrix of Simulated Results

`RVResults.plot()` in `results.py`, reached by default for 3+ variables.

Every panel shows **exactly what that data would show on its own** — the same
`classify_values` + `DEFAULT_PLOT_TYPE` lookup the 1-D and 2-D dispatches use,
small-sample branch included, so a panel of the matrix matches the plot a
student gets by simulating those variables by themselves:

| Panel | Data | Large n | Small n |
|---|---|---|---|
| diagonal | continuous-ish (`B_1D`) | `hist` | `rug` |
| diagonal | discrete-ish | `impulse` | `dotplot` |
| off-diagonal | both continuous (`K_2D` per axis) | `hist2d` | `scatter` |
| off-diagonal | both discrete | `tile` | `scatter` |
| off-diagonal | mixed | `tile`, continuous axis binned | `rug` (segmented) |

Conventions shared with the theoretical version, by design — change both or
neither: lower triangle only, `JOINT_PAIRS_MAX_DIM` cap, `JOINT_PAIRS_PANEL_SIZE`
per panel, `JOINT_PAIRS_OVERLAY_ERROR` when the figure already has a plot,
`Variable 1`-style labels on the outer edges only, no per-panel titles (each
panel's own type title is cleared, and the figure carries one suptitle instead).

**A column is one variable at one scale.** Each panel type frames its own
axes differently (an impulse pads a margin around its values; a tile's cells
tile the axes edge to edge), so a diagonal panel did *not* line up with the
joint panels under it — same tick values, different limits, so a stem sat off
the center of its tile cell. `align_pairs_columns(cells)` in `plot.py` puts
every panel in a column on the joint panels' framing, and **both matrices call
it**. The last column has no joint panel beneath it, so it borrows the scale
from the joint panels along its *row*, where that same variable is the y
variable. **A dot-plot diagonal needs more than `set_xlim`**: `_dotplot_relayout`
re-frames its own value axis on every draw, so the helper also records the
range as `ax._symbulate_value_lim` and that rebuild honors it — the same trap
as `_symbulate_freq_ticks`.

**`bins=` applies to the diagonal too**, not just the joint panels, so a column
really is one binning of one variable. It is passed *only* when the diagonal is
a histogram — every other diagonal type draws each value where it falls and
warns if handed a bin count.

**The diagonal walks the categorical palette** (`advance_pairs_diagonal_color`,
called before the panel draws). Every panel is a fresh axes and each `.plot()`
takes the first color of its own cycle, so left alone the whole diagonal came
out one color.

**Colorbar pair labels are `"Variables 1 & 2"`**, from the shared
`pairs_colorbar_pair_label` — shorter than repeating the word over a narrow bar.

**The suptitle differs by design.** A theoretical matrix is titled
`"Probability Density Functions"` or `"Probability Mass Functions"` by
`self.discrete`, because every panel of it *is* an exact pdf/pmf. A simulated
matrix is titled `PAIRS_SUPTITLE` ("Joint and Marginal Distributions"), which
names what the panels are without claiming exactness: its panels are estimates,
a mixed matrix has both kinds of variable at once, `normalize=False` shows
counts, and a small simulation's panels are scatters and rugs — so no single
pdf/pmf claim would be true of it. Note `PLOT_DISPLAY_NAME["pairs"]` in the
suggestion note still reads "Pairs Plot", since that names the `type=` token
rather than the figure (the same split as `density2d`).

**A diagonal panel's y-axis says which distribution it is showing**, in both
matrices: `Marginal Density`, from the shared `pairs_marginal_label`. A matrix
shows two kinds of distribution at once — a joint one off the diagonal, a
marginal one on it — and `Density` alone doesn't say which a scale belongs to.
The word adapts to what the panel actually drew (`Marginal Count`,
`Marginal Relative Frequency`, `Marginal Probability`), because the quantity is
**read off the panel's own y-label after it draws** rather than re-derived from
the plot type — so a diagonal cannot disagree with itself.

The rest of the left column names its row's variable (`Variable 2`,
`Variable 3` down the side). The top-left panel is a diagonal, so it gives up
its `Variable 1` label to the marginal one; its row is still named by the
`Variable 1` x-label at the bottom of that same column. **The one fallback:** a
small-n continuous diagonal is a rug plot, which has no frequency axis at all
(`make_rug` hides that direction outright), so it reads as an empty quantity and
the panel keeps the variable name instead.

**A dot-plot diagonal is safe here, unlike its ticks and limits** —
`_dotplot_decorate` (which sets the `Count` y-label) runs once from
`make_dotplot`, not from `_dotplot_relayout`, so the resize handler does not
overwrite the marginal label the way it would a plain `set_xlim`.

**Colorbars go in the empty upper triangle — one per joint panel.** The matrix
fills only its lower triangle, so the cell mirroring panel `(row, col)` across
the diagonal is free and exactly the right shape for that panel's colorbar.
`add_pairs_panel_colorbar(fig, cell, mappable, pair_label, quantity_label)` in
`plot.py` places it, and **both matrices call the same helper** so they cannot
drift apart. Consequences to respect:
- Each panel keeps **its own** color scale (a dense pair and a diffuse one are
  each colored over their own range), which is *why* every bar is titled with
  the pair it explains (`"Variable 1 & Variable 2"`) — a color only means something against its
  own bar.
- `quantity_label` is `"Joint Density"`/`"Joint Count"` by `normalize` for
  simulated results and `"Joint Density"`/`"Joint Probability"` by discreteness
  for a distribution, built by the shared `pairs_joint_label` — the counterpart
  of `pairs_marginal_label` on the diagonal, so a bar can't be read as the
  marginal density of the panel it sits across from. **A count bar's ticks carry
  no decimals** (a count is a whole number of simulated values); density and
  probability keep `JOINT_CBAR_DECIMALS`. That check is
  `quantity_label.endswith("Count")`, not equality — don't narrow it back.
- The bars are placed **after `fig.tight_layout()`**, from
  `gs[col, row].get_position(fig)` — the cell rectangles are only final once the
  layout has settled. They use `fig.add_axes`, so they have no subplot spec,
  which is how tests tell a panel from a bar (`get_subplotspec() is not None`).

Three implementation notes:
- **Diagonal panels route through `.plot()`** (full reuse of the 1-D dispatch)
  after `plt.sca(ax)` — which works because every helper draws on `plt.gca()`.
  **Joint panels call `make_tile`/`make_hist2d` directly**, because the 2-D
  dispatch hardcodes `colorbar=not marginal`; they return their mappable so the
  matrix can give each one its own bar in the mirroring cell.
- **One bin count for the whole matrix.** Equal-width bins over the same column
  of data with the same count give identical edges, which is what makes a column
  comparable. Don't pass `bins` to `make_tile` when both axes are discrete — it
  warns.
- There is no `dims`/`variables` argument here: a subset is chosen by indexing
  the random variable (`X[[0, 2]].sim(n).plot()`). Passing `dims=` raises a
  message saying so.

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
| `test_time_series.py` | `MA`, `AR`, `ARMA`, `GARCH`, `ARCH` |
| `test_branching_process.py` | `GaltonWatson` |
| `test_hitting_times.py` | `hitting_time` and `upcrossings` |
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
- Do not make the three-panel layout the default for simulated 2-D plots, and do not remove `marginal=` from `RVResults.plot()` — that was tried and reverted, because it cost every 2-D overlay (see "Two Variables"). Do not add a `marginal=` keyword to `MultivariateDistribution.plot()` either; there the strips are always shown.
- Do not build the three-panel layout inline — call `setup_marginal_axes` so both sides stay identical.
- Do not set a pairs-matrix panel's limits without going through `align_pairs_columns`, and do not drop the `_symbulate_value_lim` / `_symbulate_freq_ticks` records — a dot plot rebuilds its own framing and locators on every draw, so a plain `set_xlim`/`set_major_locator` silently reverts on the next render (see "Pairs Matrix" and "Two Variables")
- Do not force whole-number ticks on a sample path unconditionally — `make_sample_path` applies them only when every time is a whole number, which is what keeps a continuous-time path (Brownian motion, a Poisson process, a queue's clock) reading as continuous
- Do not read a *theoretical* two-variable plot's type from `plt.gca().get_title()` — it is on `plt.gcf().get_suptitle()` there. A simulated 2-D plot keeps its title on the axes (see "Two Variables")
- Do not hardcode the discreteness thresholds — use `B_1D` (1-D) and `K_2D` (2-D per axis) from `plot.py`, passed into `classify_data()` at the `results.py` dispatch (`B_1D` for 1-D, `K_2D` per axis for 2-D). Values are provisional (see `DECISIONS.md`).
- Do not set `self.xlim` in a new distribution's `__init__`, and do not compute a window there — `Distribution._compute_xlim` derives it from scipy's `support()` on first read (see "Distribution Plotting Window"). The only exception is a degenerate branch that skips `Distribution.__init__` and so has no scipy object.
- Do not reintroduce the highest-density interval (`_discrete_hdi_xlim`, `_continuous_hdi_xlim`, `_PLOT_COVERAGE`, `_hdi_window`) — removed by team decision; there are tests asserting it stays gone.
- Do not add a nonnegativity check to `CompoundPoissonProcess`'s `jump_dist` — negative jumps are intentional (see "Compound Poisson Process")
- Do not add separate `MixedPoissonProcess` or `MarkovModulatedPoissonProcess` classes — both are `CoxProcess` with a different `intensity` (see "Cox Process")
- Do not add interpolation, a `step` scan, or an equality test to `hitting_time`'s jump/discrete-time branches — they are exact, and a jump path reaching a level means reaching *or passing* it (see "Hitting Times")
- Do not narrow `hitting_time`'s process check back to `RandomProcess` — most Tier A processes are plain `RV`s (see "Hitting Times")
- Do not give `upcrossings` a Gaussian branch, a `step`/`tol` argument, or a "count one crossing per window" rule — a continuous path recrosses a level infinitely often, so the refusal is the correct answer (see "Hitting Times")
- Do not drop the `strict` half of `upcrossings`' two-search cycle, or its eager `read_one()` check — the first keeps a path sitting *on* the level from reporting a crossing every step, the second keeps a lazy sequence from hiding a bad path (see "Hitting Times")
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
