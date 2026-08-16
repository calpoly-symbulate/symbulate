# Random Process Sample Path Plots: Revisions

Standalone companion document, split out from the main graphics task list because
this needs prototyping before it can be scoped into a crisp set of tasks. Covers
sample-path plotting for random processes and Markov chains — `Tuple.plot()`,
`InfiniteVector.plot()`, `DiscreteTimeFunction.plot()`, `ContinuousTimeFunction.plot()`
(all in `result.py`), the multi-path "ensemble" branch in `RVResults.plot()`
(`results.py`), and the shared `make_sample_path` helper (`plot.py`).

---

## Summary of the conversation so far

Three reference images were shared: a discrete-time Markov chain (vowel/consonant
states) rendered as (1) a single sample path with dots at each observed value
connected by lines, (2) 100 overlaid sample paths as a single-color, semi-transparent
tangle, and (3) a "100% stacked bar chart over time" showing the marginal state
distribution at each time step (handled as a separate task — see the main graphics
task-list document).

For the sample-path plots specifically, we compared `main`, the reference fork
(kevindavisross/symbulate), and `dev`:

- **The fork has not touched sample-path plotting at all.** `result.py` is
  byte-for-byte identical between `main` and the fork (confirmed via direct diff) —
  no prior art to draw on there.
- **Dev's changes are close to purely cosmetic**, as suspected, with one exception
  worth flagging as a real (small) regression rather than a neutral no-op:
  - Consolidated four duplicated one-line `plt.plot(...)` calls (in `Tuple`,
    `InfiniteVector`, `DiscreteTimeFunction`, `ContinuousTimeFunction`) into one
    shared helper, `make_sample_path`.
  - **Dropped the dot markers.** Main's style was `'.--'` (dots connected by dashed
    lines). Dev's own docstring says: *"the dot-dash marker is dropped in favor of
    a plain solid line."* Every path type is now rendered identically as a plain
    solid line, with no visual distinction between discrete-time and continuous-time
    processes.
  - Added automatic color-cycling and an automatic "Path 1"/"Path 2" legend for the
    case where a user overlays a few *distinct* paths via separate `.plot()` calls.
  - Added default axis labels ("Time"/"Value", or "Index" for `InfiniteVector`).
  - **Did not change** the many-paths "ensemble" branch in `RVResults.plot()` (what
    actually draws the "100 sample paths" image): `alpha = np.log(2) / np.log(len(self)
    + 1)` (transparency shrinks as path count grows) and a single shared color for
    every path, regardless of count — identical formula and logic to `main`.
  - **Did not fix** `ContinuousTimeFunction.plot()`'s sampling approach: it still
    evaluates the process at 200 evenly-spaced `np.linspace` points and straight-line
    connects them, identical to `main`. For a genuinely piecewise-constant
    continuous-time process (e.g. `ContinuousTimeMarkovChain`), this can smear a
    jump that falls between two grid points into a diagonal line instead of a crisp
    vertical step at the true jump time — a latent inaccuracy neither branch has
    addressed. The data needed to fix this already exists:
    `DiscreteValued.get_arrival_times()` / `get_interarrival_times()` are already
    defined.

**Five improvement ideas were proposed** (attached example images informed these):

1. Small number of paths (say <10): color each path differently. Large number:
   one shared color with transparency (the transparency half of this already
   exists; the "distinct colors for few paths" half does not).
2. Add dots at the process's actual observed values (discrete time: at each time
   step; continuous time: at jump times).
3. Add jittering of dots/paths when multiple paths are shown for a discrete-state
   process (to reduce overplotting when many paths pass through the same state at
   the same time).
4. Distinguish line style by time type: discrete time → dotted/dashed connecting
   lines; continuous time + discrete state → flat lines that jump at the actual
   transition times (a proper step function, not a linspace approximation).
5. Consider coloring by discrete state (e.g., one color per state, rather than one
   color per path) — possibly in combination with idea 1.

**Assessment of what's genuinely new vs. already-partially-supported:**

| Idea | Status today |
|---|---|
| 1 (transparency for large n) | Already implemented, unchanged since `main` |
| 1 (distinct colors for small n) | Does not exist — new |
| 2 (dots at observed/jump values) | Existed crudely in `main` (`'.--'`), removed by dev — needs deliberate reintroduction, ideally done better (true jump times, not just regular grid points) |
| 3 (jittering) | Does not exist for paths — 2D scatter's `jitter=` modes (`"spiral"`, `"bins"`, `"random"`) are a design precedent to borrow *ideas* from, not a direct reuse |
| 4 (discrete-time dotted / continuous-time step line) | Does not exist — both currently rendered identically via one generic `make_sample_path` |
| 4 (continuous-time step accuracy) | Latent inaccuracy in both `main` and `dev`; arrival-time data already exists (`get_arrival_times()`) and could drive a correct step-function renderer |
| 5 (color by discrete state) | Does not exist — architecturally different from today's "one path = one color" model (would need per-segment coloring, e.g. via a `LineCollection`, or a colored-scatter-plus-neutral-connector approach) |

---

## Why this is being handled as its own document/task

- It lives in a different part of the codebase (`result.py`'s `TimeFunction`
  hierarchy + `make_sample_path` in `plot.py`) than the rest of the graphics task
  list, which centers on `plot.py`'s classification/dispatch machinery for
  `RVResults.plot()`.
- The five ideas interact with each other in ways that are easier to resolve by
  looking at a few rendered variants than by writing one upfront spec — e.g., how
  should "distinct color per path" (idea 1) and "color by discrete state" (idea 5)
  combine when both want to use color for different things on the same path?
- The mix of effort is uneven and easy to underestimate if compressed into one
  task-list line: some of this is a small tweak (idea 1's color half), some is a
  correctness fix with existing data support (idea 4's step-function accuracy),
  and some is a real new rendering subsystem (idea 5, and to a lesser extent idea 3).

---

## Decisions that need to be made

1. **Small-n / large-n cutoff and interaction with transparency.** You suggested
   "<10" for distinct colors. Below that cutoff, should paths be fully opaque (since
   distinct colors already solve the overlap-identification problem transparency was
   solving), or still somewhat transparent? Does the existing
   `alpha = log(2)/log(n+1)` formula still apply above the cutoff unchanged, or does
   it need retuning now that the "few paths" regime is handled differently?
2. **Where do markers go, and does every process get them?** Reintroducing dots at
   observed/jump values is a deliberate reversal of a dev decision — worth deciding
   whether this applies to *all* sample-path plots (including continuous-state
   processes, e.g. Brownian motion, where "dots at discrete time steps" is less
   obviously meaningful) or specifically to discrete-time and/or discrete-state
   processes.
3. **How do color-by-path and color-by-state coexist?** If both idea 1 (distinct
   path colors) and idea 5 (color by state) are wanted, decide the priority: is
   color reserved for state (with paths distinguished some other way, e.g. line
   style or slight jitter), or is path identity the primary use of color (with state
   shown via marker shape/fill instead)? Trying to encode both with color alone on
   the same line is likely to be confusing.
4. **Jitter scope and mechanism.** Is jittering meant to spread out points *within
   a single time step* (many paths occupying the same state at the same time,
   perpendicular to the time axis), or something broader? Should it reuse/adapt the
   existing 2D scatter jitter vocabulary (`"spiral"`, `"bins"`, `"random"`) for
   naming consistency, or does a path plot need its own vocabulary since the
   geometry (points over time, connected by lines) is different from a static
   scatter?
5. **Continuous-time step-function correctness.** Should fixing
   `ContinuousTimeFunction.plot()` to use true jump/arrival times (idea 4's
   correctness half) be bundled with the new discrete-state styling work, or treated
   as an independent bug fix that could ship on its own, sooner? It doesn't depend
   on any of the other decisions above.
6. **Scope: which process/result types get which treatment?** A clear matrix would
   help — e.g., discrete-time + discrete-state (Markov chain) likely wants all five
   ideas; continuous-time + discrete-state (continuous-time Markov chain) wants the
   step-function fix plus state coloring but not "dotted line" styling (dots at
   jump times, not dashes, per your idea 4); continuous-time + continuous-state
   (e.g. Brownian motion, Poisson process arrival times aside) may want none of the
   discrete-state-specific ideas at all. This matrix should probably be written
   down explicitly before implementation, similar to how `DEFAULT_PLOT_TYPE` maps
   configurations to plot types elsewhere in the codebase.

---

## Suggested steps forward

1. **Write down the process-type matrix from decision 6 first.** Before prototyping
   any rendering, get explicit agreement on which of the five ideas apply to which
   combination of (discrete/continuous time) × (discrete/continuous state). This
   will likely resolve several of the other open decisions as a side effect.
2. **Prototype in isolation, not as one combined change.** Suggested order, roughly
   cheapest/most-isolated first:
   - Fix `ContinuousTimeFunction.plot()`'s step-function accuracy (decision 5) using
     `get_arrival_times()`/`get_interarrival_times()` — this is a self-contained
     correctness fix with no dependency on the styling decisions and can be
     validated on its own with a simple continuous-time Markov chain example.
   - Prototype idea 1 (distinct colors for <10 paths, shared+transparent above that)
     on its own, using a simple non-Markov-chain process first (so color choices
     aren't tangled up with decision 3 yet) to settle the cutoff and opacity
     questions.
   - Prototype idea 2 (dots at observed/jump values) layered on top of the above,
     using the discrete-time Markov chain example from the attached images as the
     target to match, to settle whether markers should differ by process type.
   - Only after the above are settled, prototype idea 5 (color by state) combined
     with idea 1, specifically to resolve decision 3 — this is the one most likely
     to need a few visual iterations before committing to an approach.
   - Prototype idea 3 (jittering) last, once the others are stable, since it's
     mostly a refinement on top of however markers ended up being drawn (idea 2).
3. **Use the attached reference images as concrete acceptance targets** for the
   discrete-time, discrete-state case specifically (1 path and 100 paths) — they're
   a good visual spec to prototype against directly, including checking that colors
   distinguish state (per the "State: vowel (1) / consonant (2)" legend already
   present in the reference images) rather than (or in addition to) path identity.
4. **Once prototypes settle the open decisions**, write this up as a proper set of
   scoped tasks (matching the format of the main graphics task-list document) and
   fold it back in as its own section, or keep it as a standalone companion
   document, whichever reads better once the shape of the work is clearer.

---

*Note for later: when the main graphics task-list document is next revised, add a
pointer at the end to this document for the random process sample path task.*
