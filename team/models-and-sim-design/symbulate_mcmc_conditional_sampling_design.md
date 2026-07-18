# Symbulate: Efficient Sampling from Conditional Distributions (MCMC)

**Status:** Exploratory / open — no option below is final. This is a running
document, following the same convention as the hierarchical-models design doc:
it records what's been analyzed, what's been recommended, and what depends on
decisions still being made in *other* threads (hierarchical models,
`ContinuousProbabilitySpace`). Nothing here should be read as "decided."

---

## 1. The problem

```python
X, Z = RV(Poisson(1) * Poisson(2))
Y = X + Z
(X | (Y == 4)).sim(10000)
```

Today this is pure rejection sampling. `RVConditional.draw()`
(`random_variables.py`) is:

```python
def draw(self):
    while True:
        outcome = self.prob_space.draw()
        if self.condition_event.func(outcome):
            return self.func(outcome)
```

`RV.sim(n)` just calls `draw()` in a loop
(`RVResults(_sim_with_progress(self.draw, n))`). `RVConditional` overrides
only `draw`, not `sim` — there is no `method` argument anywhere in the
codebase today.

Two concrete pain points:

1. **Low-probability discrete events** (e.g. `Y == 4` when `Y`'s mass at 4 is
   tiny) make rejection sampling arbitrarily slow — every one of the 10,000
   requested draws needs its own accepted sample, so total work scales with
   `n / P(event)`.
2. **Continuous conditioning doesn't work at all** via exact equality (`P(Y=4)
   = 0`). The existing workaround — condition on `abs(Y - y) < eps` — is
   something a user has to write by hand (there's no dedicated API for it;
   `Event` only ever wraps comparison predicates), and it's *still* rejection
   sampling on a possibly-rare band event.

The one advantage of the current approach worth preserving where possible:
**the samples are i.i.d.** Any replacement needs to be honest about giving
that up.

---

## 2. What the codebase actually gives us to work with

Read from `dev`: `probability_space.py`, `random_variables.py`,
`distributions.py`, `results.py`.

- Every `Distribution` (`distributions.py`) already exposes `.pdf` (aliased
  to `.pmf` when discrete), `.cdf`, `.quantile`, `.sd()`, `.discrete`. This is
  the single biggest asset for MCMC — no density code needs to be written for
  Poisson, Normal, etc.
- `ProbabilitySpace.__mul__` / `__pow__` (what `*` and `**` actually call)
  return a **plain `ProbabilitySpace`** wrapping a closure:

  ```python
  def draw():
      return join(self.draw(), other.draw())
  return ProbabilitySpace(draw)
  ```

  The component distributions are captured in the closure but not stored as
  inspectable attributes anywhere. Once a joint space is built, its factored
  structure is gone — this is the key architectural gap MCMC needs filled.
- `Event.func` is an opaque `outcome -> bool` predicate assembled by
  `_comparison_factory` / `_logical_factory` (`base.py`). It has no notion of
  "distance from the target," only true/false. This turns out to be *fine*
  for MH (see §4), but it rules out anything that wants a smooth/soft target.
- As of today, **no hierarchical model support exists on `dev`** — no
  distribution takes an RV/`ProbabilitySpace` as a parameter. Every joint
  model on `dev` right now is *independent components + a deterministic
  transform*. This matters a lot (§7): it's exactly what makes "joint density
  = product of independent marginals" true, and it's exactly what the
  in-progress hierarchical-models thread would change.

---

## 3. Core design: factor-preserving product spaces + component-wise MH

### 3.1 Preserve factor structure through `*` / `**`

Add `IndependentProductSpace(ProbabilitySpace)`, returned by `__mul__`/`__pow__`
in place of a bare `ProbabilitySpace`, *when both operands are known to be
density-bearing* (see §7.2 for the eligibility rule, which needs to be
duck-typed, not `isinstance(..., Distribution)`). Flatten nested products so
`(A * B) * C` gives one flat factor list, not a tree.

```python
class IndependentProductSpace(ProbabilitySpace):
    factors: list          # the component density-bearing spaces
    def log_density(self, outcome): ...   # sum of log pdf/pmf across factors
```

If either operand *isn't* density-bearing (a `BoxModel`, `DeckOfCards`, or a
bare user `ProbabilitySpace(lambda: ...)`), fall back to today's plain
`ProbabilitySpace` — no density, no MCMC, and `method="mcmc"` raises a clear
error later rather than silently doing something wrong.

### 3.2 One sampler to start: component-wise random-walk Metropolis-Hastings

Given accepted outcome `ω = (ω_1, …, ω_k)`, one factor per component, sweep
through coordinates:

1. Propose `ω_i' ~ q_i(· | ω_i)`, a **symmetric** kernel (so the Hastings
   correction cancels):
   - continuous factor: `Normal(ω_i, scale=step_i)`, default
     `step_i = 0.5 * factor.sd()`
   - discrete/count factor: symmetric integer walk (`±1` etc.), reflected at
     any known support boundary (e.g. non-negative for Poisson/Binomial)
2. Form candidate `ω'` (only coordinate `i` changed).
3. Accept with probability `min(1, f_i(ω_i') / f_i(ω_i))` **if**
   `condition_event.func(ω')` is `True`; otherwise stay at `ω`.

   All other factors' densities are unchanged and cancel — and since the
   *current* state is always already event-satisfying (an invariant kept from
   initialization), the ratio really is this simple. **This step depends on
   independence across factors — see §7.1 for how it changes if hierarchical
   models exist.**

Why this is the recommended *first* algorithm, specifically for this
codebase:

- Reuses `Distribution.pdf` and `Event.func` exactly as they exist — no new
  methods needed on either class.
- Needs only the one additive change in §3.1.
- Fixes the rare-discrete-event problem: local moves from an already-valid
  state are far more likely to stay valid than fresh i.i.d. redraws.
- Fixes the continuous-band problem too, with *zero* new `Event` machinery:
  users' existing `abs(Y - y) < eps` trick works unmodified as the MH target.
- Generalizes cleanly to a real Gibbs sampler, HMC, etc. later — the
  `.factors` / `.log_density` infrastructure is reusable, not single-purpose.

### 3.3 API sketch

```python
(X | (Y == 4)).sim(10000, method="mcmc",
                    burn_in=1000, thin=1, step_scale=1.0, init=None)
```

- `method="rejection"` stays the default — unchanged behavior, i.i.d.
  samples.
- `RVConditional.sim` gets overridden directly (not just `draw`), since a
  chain needs mutable state across calls to `draw_func()`. That closure
  still slots into the existing `_sim_with_progress(draw_func, n)` helper
  unmodified.
- **Init**: run bounded ordinary rejection sampling to find one valid `ω₀`,
  then start the chain there. If even *finding* a valid state is impractical,
  MCMC doesn't rescue that — surface a clear error rather than hang or lie.
- Return type stays `RVResults` (backward compatible with `.mean()`,
  `.plot()`, etc.); attach non-breaking diagnostics like
  `results.acceptance_rate`.
- Docstring must say plainly: **draws are not independent**. `.mean()` is
  still consistent, but naive SE/CI code assuming i.i.d. samples elsewhere
  will be wrong. Worth exposing a simple ESS/lag-1-autocorrelation helper.
- If `self.prob_space` isn't density-bearing: raise
  `NotImplementedError("method='mcmc' requires ... use method='rejection' instead.")`

### 3.4 Validation strategy

Compare `method="mcmc"` output against `method="rejection"` output (mean,
histogram/KS test) on cases cheap enough for rejection to still work. Since
both must target the *same* conditional law, this is a strong, close-to-free
regression check specific to this feature.

---

## 4. Algorithm options — comparison table

| Approach | Samples i.i.d.? | Handles rare discrete events? | Handles continuous conditioning? | New infra needed | Notes |
|---|---|---|---|---|---|
| **Current: pure rejection** | Yes | No — cost scales with `1/P(event)` | Only via manual `abs(Y-y)<eps` band, still slow | None | Status quo; kept as default |
| **Independence-sampler-as-chain** ("sticky rejection": repeat previous value on reject instead of looping) | No | Bounded cost (`n` draws total, not `n` acceptances) | No — still needs a fresh global hit on the band each step | Minimal — same `draw()`/`func()`, just don't discard on reject | Cheapest possible upgrade; doesn't fix continuous case at all |
| **Component-wise random-walk MH** (recommended first target, §3) | No | Yes — local moves stay valid much more often | Yes — local moves near an already-valid band point tend to stay in the band | `IndependentProductSpace` + small `mcmc.py` | Needs per-factor `.pdf`/`.discrete`/`.sd()` |
| **Exact manifold sampling for linear constraints** (e.g. `Y = X+Z == y`, change of variables to sample the 1-D conditional of `X` directly) | Yes | Yes, exactly | Yes, exactly (no eps-band needed at all) | Special-cased per relation type (sum, invertible maps only) | Best answer *when it applies*; doesn't generalize to arbitrary events |
| **Self-normalized importance sampling** | Yes (weighted) | Yes, if a good proposal exists | Yes, same caveat | Weighted-sample support in `RVResults` (bigger change) | Preserves independence, at the cost of needing a good proposal distribution and weight-aware downstream stats |

**Current recommendation unchanged:** ship component-wise random-walk MH
first. It's the best cost/generality tradeoff given what's actually in the
codebase today, and it builds the one piece of infra (`IndependentProductSpace`
/ density access) that every other row in this table would also need.

---

## 5. Phased roadmap

1. **Infra**: `IndependentProductSpace` + `.factors` / `.log_density`. Purely
   additive; no visible API change yet.
2. **One algorithm**: component-wise random-walk MH, `method="mcmc"`,
   restricted to pure independent-product models (no hierarchical spaces —
   see §7.1). Sensible defaults, clear failure mode when unsupported.
3. **Diagnostics**: acceptance rate, trace access, maybe an ESS estimate.
4. **Later, optional / not yet scoped:**
   - Exact manifold sampling for linear/invertible constraints (e.g. sums) —
     no MCMC needed at all for that common case, no autocorrelation.
   - `method="importance"` — complementary, preserves independence, needs
     weighted-`RVResults` support.
   - Markov-blanket-aware single-site MH for hierarchical models, if/when
     those ship (§7.1) — genuinely more work, a distinct phase, not a
     free extension of step 2.
   - Adaptive step-size tuning during burn-in only (frozen afterward, to
     keep exact stationarity).

---

## 6. Open questions (unchanged, still unresolved)

- Exact default `burn_in`/`thin` values, and whether they should scale with
  `n` or be fixed.
- Whether `step_scale` should be a single scalar or per-coordinate.
- Whether diagnostics (`acceptance_rate`, ESS) live as attributes on
  `RVResults` or are returned as a separate object.
- Whether `method="mcmc"` should be spelled `"mcmc"`, `"dependent"`, or
  something else — no strong pull either way yet.

---

## 7. Dependencies on the other two design threads

This section didn't exist in the original design and was added after reading
the hierarchical-models thread and the `ContinuousProbabilitySpace` thread.
**Neither changes the recommendation in §3/§5 — both narrow its scope or its
eligibility check.**

### 7.1 Dependency: hierarchical models (`>>` / `*` / `@`, `Hierarchical`, `AssumeHierarchical`)

The §3.2 acceptance ratio leans on today's fact that joint density = product
of independent marginals, so updating one coordinate cancels every other
factor's density term. `HierarchicalProbabilitySpace` (whichever operator
wins) removes that assumption: once `(Y | X=x) ~ Binomial(10, x)`, proposing
a new `x'` changes `Y`'s own density too, so the correct ratio is

```
f_prior(x') · f_cond(y | x')          not just          f_prior(x') / f_prior(x)
f_prior(x)  · f_cond(y | x)
```

— i.e. proposing a *parent* node requires re-instantiating
`cond_space_func(x')` and evaluating its `.pdf` at the currently-held child
value. Updating a *leaf* (nothing downstream depends on it) still gets the
cheap cancellation. This is the same single-site-Metropolis-within-Gibbs
pattern BUGS/JAGS use, keyed on each node's Markov blanket — a real, separate
feature, not a free extension.

Two extra wrinkles from the actual hierarchical sketch:

- Chained models (`A >> f >> g`) pass the **whole accumulated tuple** to each
  `cond_space_func`, so changing an early coordinate can require re-deriving
  every downstream distribution and re-evaluating its density — O(depth) per
  proposal in the worst case, not O(1).
- `AssumeHierarchical` wraps `ProbabilitySpace(prior_rv.draw)` — if
  `prior_rv` came through `.apply(transform)`, there's no way to recover a
  density for it (Symbulate doesn't track Jacobians through `apply`). The
  eligibility rule below already handles this correctly (declines MCMC on
  it) without any extra code — worth documenting as a known limitation of
  that escape hatch specifically.

**Conclusion:** ship §3.2 scoped to pure `*`/`**` independent-product models
first, and have it explicitly decline (clear error, not a silent wrong
answer) on any `HierarchicalProbabilitySpace`-based model, however that ships.
A Markov-blanket-aware sampler is Phase 4 material, added to §5 above.

### 7.2 Dependency: `ContinuousProbabilitySpace`

§3.1's eligibility check ("is this factor density-bearing?") was originally
sketched as `isinstance(factor, Distribution)`. `ContinuousProbabilitySpace`
is declared as `class ContinuousProbabilitySpace(ProbabilitySpace)` — **not**
a `Distribution` subclass — even though it hand-rolls the same
`.pdf`/`.cdf`/`.quantile` shape. An `isinstance` check would wrongly exclude
it. **Fix:** duck-type the eligibility check
(`hasattr(factor, "pdf") and callable(factor.pdf)`), or formalize a shared
informal protocol/mixin now that two independent designs both want to expose
density-like objects.

Two concrete gaps the sketch has today that MCMC would hit at runtime:

- `.discrete` is never set on `ContinuousProbabilitySpace` — the proposal
  dispatch in §3.2 would raise `AttributeError`. Either that class sets
  `self.discrete = False`, or the MCMC code defensively uses
  `getattr(factor, "discrete", False)`.
- No public `.sd()` / `.var()` / `.mean()` — the §3.2 default step size
  (`0.5 * factor.sd()`) has nothing to call. The good news: the sketch
  already computes this internally in a private `_estimate_scale()` (used to
  size its Cauchy proposal for unbounded rejection sampling) — promoting
  that to a public `.sd()` (and maybe `.mean()`/`.var()`) would fix this for
  MCMC and is probably good hygiene generally.

One thing that does **not** need fixing: `ContinuousProbabilitySpace.pdf` is
the raw, *unnormalized* density (normalization lives separately in the
private `_pdf_norm`). That's fine for MH — acceptance ratios are
`f(x')/f(x)`, so a missing normalizing constant cancels as long as `.pdf` is
consistently proportional to the true density. Worth noting only so this
isn't "fixed" later under the assumption MCMC needs a normalized density.

**Conclusion:** small, mechanical fix (duck-typing + two attributes on that
class), not a scope change — but it should happen from the start of §3.1,
since arbitrary-custom-pdf models are arguably one of the *most* compelling
MCMC use cases (custom pdfs are exactly where rejection is most likely to be
slow for a rare conditioning event).

---

## 8. Summary table across all three threads

| Design | Status | What it needs from the others | What it gives the others |
|---|---|---|---|
| Hierarchical models (`>>`/`*`/`@`) | Open, options being compared | Nothing from MCMC | Eventually needs to expose a `log_density`/Markov-blanket interface so MCMC can consume hierarchical spaces uniformly with `IndependentProductSpace` (Phase 4, not yet built) |
| `ContinuousProbabilitySpace` | Prototype built & tested | Nothing from MCMC | Needs to expose `.discrete` and a public `.sd()`/`.mean()`/`.var()`, and be included via duck-typing (not `isinstance(Distribution)`) in MCMC's eligibility check |
| MCMC / `method="mcmc"` (this doc) | Design only, no code yet | A density-bearing-factor eligibility rule that's duck-typed across `Distribution` and `ContinuousProbabilitySpace`; explicit non-support for hierarchical spaces until Phase 4 | — |

No option in any of the three threads is presented as final.
