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
- Helps with the continuous-band problem, with *zero* new `Event` machinery:
  users' existing `abs(Y - y) < eps` trick works unmodified as the MH target.
  **This is narrower than it first sounds — see §9 for what MH does and does
  not fix here.**
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
| **Component-wise random-walk MH** (recommended first target, §3) | No | Yes — local moves stay valid much more often | Partially — helps *once inside* the band, but doesn't remove the `eps` approximation or the cost of finding the first valid state; step size must be tuned relative to `eps` (see §9) | `IndependentProductSpace` + small `mcmc.py` | Needs per-factor `.pdf`/`.discrete`/`.sd()` |
| **Single-site exact-conditional Gibbs** (candidate alternative first algorithm, see §10) | No | Yes, exactly — every draw is a true conditional draw, no rejection | Same `eps`-band caveat as MH applies to *what* is being sampled, but no step-size tuning needed once the target is defined | Reuses `IndependentProductSpace`; continuous factors need `ContinuousProbabilitySpace`'s tier-1 inverse-CDF machinery | Depends on the `ContinuousProbabilitySpace` thread landing (or a smaller stand-alone inverse-CDF utility) |
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

## 9. The `eps`-band caveat, decomposed

This section exists because an earlier draft of this doc implied MH "handles
continuous conditioning" more cleanly than it actually does. That claim
conflates three genuinely separate issues, and MH only helps with one of
them.

**Issue 1 — defining a nonzero-probability target (algorithm-independent,
unavoidable).** `P(Z = 1) = 0` for continuous `Z`. No sampling method —
rejection, MH, anything — can draw from a measure-zero set by simulation.
The `abs(Z - 1) < eps` band isn't an MCMC trick; it's a redefinition of *what
distribution is being asked for*, done before any sampling method touches
it. MH doesn't remove this. The approximation error introduced by `eps` is
the same size regardless of which algorithm samples from the band.

**Issue 2 — finding the first valid state (no easier under MH).** Before any
local-move machinery can run, the chain needs a starting outcome `ω₀` with
`abs(g(ω₀) - 1) < eps`. As sketched in §3.3, that's found by ordinary
rejection sampling. If `eps` is small enough that hitting the band is rare,
*this step* costs exactly what full rejection sampling would have cost —
MH provides no advantage here, only after this point.

**Issue 3 — staying in / exploring the band once inside (the actual MH win,
with a real catch).** This is the only place local moves genuinely help, and
whether "local moves tend to stay in the band" holds depends on how the
proposal direction relates to the band's geometry, not just on locality.
Example: for `Z = X + Y`, a component-wise sweep that moves `X` alone (`Y`
fixed) shifts `Z` by exactly as much as `X` moved. If the per-coordinate step
size is comparable to or larger than `eps`, most such proposals kick `Z`
clean out of the band and get rejected — close to rejection sampling again,
just local. The fix is to shrink the step size relative to `eps` (and to how
much moving that coordinate shifts `g`), which works but forces smaller
steps, which means slower mixing. **Shrinking `eps` for accuracy shrinks the
usable step size too, which slows the chain — this tradeoff has to be tuned,
it isn't free.**

**The better fix for linear constraints.** For `Z = X + Y = z` specifically,
propose along the constraint surface directly: `X' = X + δ`, `Y' = Y - δ`
simultaneously. If the chain started exactly on `Z = z`, this keeps it
exactly on `Z = z` for *any* `δ` — no band, no rejection due to leaving the
constraint, no step-size-vs-`eps` tuning at all. This is the "exact manifold
sampling for linear constraints" row from §4's table, and it's the right
tool specifically because `X + Y` is linear. It generalizes to any
invertible linear relation but not to a generic nonlinear `g`, where staying
on the level set needs either an analytic inverse or a numerical projection
(Newton-correct back onto the constraint after each proposal) — heavier
machinery, a legitimate later-phase item, not a naive extension of
component-wise MH.

**Net effect on how this should be described to the team:** component-wise
MH should not be sold as "solves continuous conditioning." It should be
described as: `eps` is still required and still introduces approximation
error regardless of algorithm (issue 1); finding an initial valid state is
exactly as hard as it is today (issue 2); mixing efficiency inside the band
depends on tuning step size relative to `eps` and to the constraint's local
sensitivity, which the naive version does not do automatically (issue 3).
The linear-constraint exact sampler is worth prioritizing precisely because
it removes all three issues at once for a common pattern (sums), not because
it's a nice-to-have.

---

## 10. Which algorithm should actually go first: MH vs. single-site exact Gibbs

The original recommendation (§3.2) was component-wise random-walk MH. Worth
naming a real competitor rather than treating that as settled.

**Single-site Gibbs with exact full-conditional draws.** For an
independent-product model, the full conditional of coordinate `i` given
everything else and given the event is a one-dimensional distribution:

```
π(ω_i | ω_{-i}, event) ∝ f_i(ω_i) · 1{event holds when coordinate i = ω_i, others fixed}
```

This is exactly the shape two pieces of existing/proposed machinery already
handle:

- **Discrete factor with countable support** (Poisson, Geometric, …): enumerate
  this conditional on a truncated grid, normalize exactly, sample directly —
  no accept/reject step, because it's an exact draw from the true conditional,
  not a proposal.
- **Continuous factor**: this is precisely the problem `ContinuousProbabilitySpace`'s
  tier-1 inverse-CDF sampler already solves — numerically integrate an
  arbitrary (possibly unnormalized) 1-D density and invert it. Reusing that
  machinery here means each Gibbs step draws exactly from the conditional,
  up to the same numerical tolerance already accepted elsewhere in that
  design.

**Advantages over MH, specific to this problem:**
- No step-size tuning at all — §9's step-size-vs-`eps` tradeoff simply
  doesn't arise, since there's no proposal to tune.
- No wasted proposals — every Gibbs draw is accepted by construction, versus
  MH wasting some fraction of proposals even once inside the band.
- Direct architectural coherence with the `ContinuousProbabilitySpace`
  thread: rather than two features sharing only an eligibility check, Gibbs
  would actually *use* that thread's core sampler as a subroutine.

**Real costs, and why this isn't a clean swap:**
- Every continuous-factor update pays a full numerical integration +
  inversion, once per coordinate per iteration — versus MH's cheap
  `pdf(ω_i')/pdf(ω_i)` ratio. Could be slower per-iteration despite better
  mixing per-iteration; an empirical question, not obvious in advance.
- Creates a **hard dependency** on `ContinuousProbabilitySpace` landing first
  (or building a smaller stand-alone inverse-CDF utility just for this) — MH
  has no such dependency and can ship standalone.
- Needs a genuinely different code path per factor type (enumeration vs.
  numerical inversion), versus MH's uniform propose/evaluate/accept shape
  across discrete and continuous factors alike.

**Recommendation (still not final):** lean toward implementing MH first,
specifically because it can ship without waiting on the
`ContinuousProbabilitySpace` thread. But flag single-site exact-conditional
Gibbs as the stronger candidate *if* that thread is being built anyway, since
the two would compose naturally rather than duplicating effort. This
replaces §3.2's implicit "MH is simply the right first choice" with an
explicit tradeoff — both are legitimate, the tie-breaker is really about
sequencing with the other in-flight thread, not about one being better in
the abstract.

---

## 11. Hamiltonian Monte Carlo / NUTS: not now, and specifically why

Worth a concrete answer rather than a vague "too advanced," since the
blockers are specific facts about this codebase, not HMC/NUTS in general.

1. **Mixed discrete/continuous factors are the normal case here, and HMC
   doesn't handle discrete variables at all.** `Poisson(1) * Normal(0, 1)`
   is an entirely ordinary Symbulate model; HMC's leapfrog integrator
   fundamentally requires a continuous, differentiable state space. A mixed
   model would need HMC-for-continuous-factors plus a separate mechanism for
   discrete ones from day one — not "HMC," but HMC-plus-a-second-algorithm.
2. **The conditioning event is usually a hard indicator, and that's exactly
   what HMC is weak at.** HMC's efficiency comes from following the gradient
   of a smooth log-target to make large, informed proposals. A band event
   like `abs(Z - 1) < eps` is a hard wall — zero gradient almost everywhere,
   a discontinuity at the boundary. Making HMC work there needs
   reflection/refraction handling at the boundary (a real technique, used in
   "billiard" HMC variants) — additional machinery layered on top of
   already-nontrivial HMC, specifically to compensate for HMC's main
   advantage not applying well to event-conditioning in the first place.
3. **No gradients are available anywhere in Symbulate today.** HMC needs
   `d/dω log f(ω)` for every factor, chained through `Y.func` — an arbitrary
   user-built Python function assembled via operator overloading (`+`, `*`,
   `.apply(arbitrary_lambda)`, indexing, …). There's no autodiff layer in
   this codebase, and building one (or adopting `jax`/`torch` as a
   dependency) is a large, separate architectural commitment, not an add-on
   to this feature. Finite-difference gradients as a fallback are fragile
   and add cost exactly where the goal is to save cost.

**NUTS is strictly more of all three** — HMC plus adaptive trajectory-length
selection plus (typically) dual-averaging step-size adaptation during
warmup. That's the engineering underlying entire dedicated libraries (Stan,
PyMC); a correct, robust implementation is a multi-month project on its own,
not an incremental step past MH.

There's also a pedagogical angle, since this is a teaching library: MH's
accept/reject logic is readable line-by-line by an intro-stats student; HMC/
NUTS trade that transparency for efficiency in a regime (smooth,
high-dimensional, continuous-only posteriors) that mostly isn't what
Symbulate's own example set (small mixed discrete/continuous models,
event-based conditioning) needs efficiency for.

**Conclusion:** MH (or single-site exact Gibbs, per §10) first. HMC/NUTS
belong in a "not now, and here's specifically what would have to change
first" note: Symbulate would need (a) a continuous-only differentiable model
subset, (b) an autodiff layer, and (c) a smooth (not hard-indicator)
conditioning API, before HMC/NUTS became a reasonable investment.

---

## 12. Summary table across all three threads

| Design | Status | What it needs from the others | What it gives the others |
|---|---|---|---|
| Hierarchical models (`>>`/`*`/`@`) | Open, options being compared | Nothing from MCMC | Eventually needs to expose a `log_density`/Markov-blanket interface so MCMC can consume hierarchical spaces uniformly with `IndependentProductSpace` (Phase 4, not yet built) |
| `ContinuousProbabilitySpace` | Prototype built & tested | Nothing from MCMC | Needs to expose `.discrete` and a public `.sd()`/`.mean()`/`.var()`, and be included via duck-typing (not `isinstance(Distribution)`) in MCMC's eligibility check |
| MCMC / `method="mcmc"` (this doc) | Design only, no code yet | A density-bearing-factor eligibility rule that's duck-typed across `Distribution` and `ContinuousProbabilitySpace`; explicit non-support for hierarchical spaces until Phase 4 | — |

No option in any of the three threads is presented as final.
