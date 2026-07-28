# Matrix (Pairs) Plots for 3+ Variables: Implementation Notes

Standalone notes on extending Symbulate's automatic default-plot system — which
already covers every 1-variable and 2-variable (discrete vs. continuous, small vs.
large n) situation — to joint results with 3 or more numeric dimensions, via a
pairs-plot / scatterplot-matrix layout (every coordinate against every other, with
each variable's marginal distribution on the diagonal).

---

## Current state: this isn't just unsupported, it's actively broken

`RVResults.plot()` only branches on `self.dim == 1` and `self.dim == 2`. Anything
else — including `self.dim == 3` or higher — falls through to the final `else:`
branch, which is written for an unrelated case: the "ensemble of sample paths"
branch used for things like `MarkovChain.sim(100).plot()`, which assumes
`self.results` contains `TimeFunction`-like objects and calls `.plot()` on each one.

Confirmed directly:

```python
RV(Normal() * Normal() * Normal()).sim(30).plot()
```

runs without raising an error, but renders each simulated 3-tuple as if it were a
tiny 3-point "sample path" (via the generic `Tuple.plot()`), connected by a line —
meaningless for a joint distribution, and not remotely what a user would expect.

**This should be treated as two separate pieces of work:**
1. A quick, low-risk fix: make `dim > 2` raise a clear, deliberate error/message
   instead of silently misrendering, as an immediate stopgap.
2. The larger feature below: give `dim > 2` a real, sensible default plot, the same
   way `dim == 1` and `dim == 2` already have one — consistent with the project's own
   stated philosophy that `.plot()` should produce a reasonable default for every
   situation.

These don't have to ship together — (1) can land immediately; (2) is the real
design-and-build effort this document is about.

---

## What already exists and is directly reusable

- **Diagonal panels (each variable's own marginal):** the existing 1-D dispatch —
  `classify_values` / `default_plot_type` / `make_impulse` / `make_hist` /
  `make_dotplot` / etc. — already fully handles "pick a sensible plot for this one
  variable" and needs no changes. Call it once per dimension.
- **Off-diagonal panels (each pair):** the existing 2-D dispatch (the same logic
  that already picks `scatter`/`tile`/`hist2d`/mixed configurations for `dim == 2`)
  already handles "pick a sensible plot for this pair of variables." Call it once
  per `(i, j)` pair instead of once for the whole result.
- **A GridSpec-based multi-panel layout precedent:** the `marginal=True` machinery
  (recently rebuilt to call the real `make_density`/`make_dotplot`/`make_rug`
  helpers rather than old fork code) already composes a main panel plus two
  marginal strips into one figure via `GridSpec`. This is the closest existing
  ancestor for "compose several axes sharing coordinate context into one figure,"
  even though its current implementation is hardcoded to exactly one main panel +
  2 marginals (`gs[0, 0:3]`, `gs[1:4, 3]`) and would need real generalization, not
  just a parameter tweak, to become an N×N grid.

Roughly: the per-cell plotting logic is 100% reuse. The grid composition,
axis-sharing, and layout-consistency work below is genuinely new.

---

## Decisions that need to be made

1. **Automatic default, or explicit opt-in?** Given every 1- and 2-variable
   situation already gets an automatic default, should `dim > 2` follow the same
   philosophy (an automatic pairs-matrix default, no special argument needed), or
   should it require something like `type="pairs"`? Leaning toward "automatic,"
   for consistency with the rest of the system, but worth confirming deliberately.
2. **Layout convention.** Classic pairs plots either mirror the same panel type in
   both the upper and lower triangle (simpler, but shows each relationship twice),
   or use the upper triangle for one representation and the lower for another (e.g.,
   scatter above the diagonal, a summary statistic or density below). Needs an
   explicit choice.
3. **Per-cell type consistency vs. matrix-wide visual consistency.** If each
   off-diagonal cell independently runs the normal 2-D auto-selection, a mixed
   discrete/continuous variable set could produce a grid with `tile` in one cell,
   `hist2d` in another, `scatter` in a third — each individually correct (consistent
   with "never sacrifice the discrete/continuous read" from the main graphics
   revisions doc), but potentially inconsistent-looking across a whole matrix. Worth
   deciding whether per-cell correctness or whole-matrix visual consistency wins
   when they conflict — this tension doesn't come up for a single pairwise plot, so
   there's no existing precedent to fall back on.
4. **Shared axes per row/column.** Standard practice (e.g., seaborn's `PairGrid`) is
   that every panel in a row shares y-limits and every panel in a column shares
   x-limits. This is the same coordinate-consistency concern that came up for the
   2-variable marginal fix (main graphics revisions doc, item 8), generalized from
   "2 panels share one axis" to "N panels share one axis per row/column" — worth
   building on whatever `sharex`/`sharey` solution comes out of that item rather
   than solving it twice.
5. **Bin/threshold consistency across shared axes.** If two off-diagonal cells in
   the same row share a y-variable, their y-axis binning/discretization should
   agree — an N-variable generalization of the bin-matching concern already flagged
   for 2-variable marginals.
6. **Crowding / scale limit for large N.** An N×N grid gets dense fast (6 variables
   → 36 panels). Decide whether there's a hard size cap, a warning (consistent with
   other crowding warnings elsewhere in the codebase), or just a documented
   expectation that this is meant for a handful of variables at a time.
7. **Naming/API surface**, once (1) is decided: if it's not fully automatic, what
   should the explicit option be called, and how does it fit the existing `type=`
   vocabulary and the separate-modifier-keyword pattern (`marginal=`, etc.) already
   established elsewhere?

---

## Suggested steps forward

1. **Ship the stopgap fix first** (a clear error/message for `dim > 2` instead of
   the current silent misrender) — independent of everything else, low-risk, and
   removes the most likely source of user confusion in the meantime.
2. **Settle decisions 1–3 above before writing any grid-composition code** — they
   determine the shape of the API and the visual contract, and are cheap to change
   on paper but expensive to change after panels are being generated.
3. **Prototype the grid composition and axis-sharing (decisions 4–5) using a small,
   fixed N (e.g. 3) first**, reusing the 1-D/2-D dispatch functions as-is, before
   generalizing to arbitrary N — this isolates the genuinely new work (GridSpec
   layout, shared axes, per-row/column bin consistency) from the parts that are
   already solved.
4. **Test against a deliberately mixed-type case early** (e.g. two continuous
   variables and one small-cardinality discrete variable) to surface decision 3's
   tension concretely, rather than discovering it only once a real user hits it.
5. **Decide the crowding limit (decision 6) using a concrete example**, e.g. render
   a 3-variable, a 5-variable, and an 8-variable grid at whatever default figure
   size is chosen, and use those to set (or confirm no need for) a cap.
6. **Once the shape is settled, fold this into the main graphics task list** as a
   scoped set of tasks (in the What/Why/Prompt format used there), rather than
   keeping it as a standalone document long-term — this is being kept separate for
   now only because the design questions above haven't been resolved yet.
