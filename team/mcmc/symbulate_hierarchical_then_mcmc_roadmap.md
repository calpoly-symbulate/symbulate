# Symbulate: Hierarchical-Models-First, Then MCMC — Session Summary & Roadmap

**Date:** 2026-08-07
**Scope:** Session investigating `calpoly-symbulate/symbulate` at `dev` (HEAD `dce6e10`, Aug 6 2026), cross-referencing `MODEL-DECISIONS.md`, `team/models-and-sim-design/symbulate_hierarchical_models_design.md`, `team/models-and-sim-design/symbulate_mcmc_conditional_sampling_design.md`, and a prior RNG-seeding audit conversation.
**Purpose:** Record what this session found by reading the actual code (not just the design docs), answer the three original scoping questions, and lay out a concrete build order — Hierarchical models first, MCMC second — with the specific code changes each phase needs.

This is written in the same spirit as `MODEL-DECISIONS.md`: a decision log, not a ratified plan. Nothing here has been implemented; it's a recommendation based on directly reading `dev` as of the date above.

---

## 1. What's actually on `dev` right now

`MODEL-DECISIONS.md` is current — its last edit (Aug 5) trails `dev`'s tip by one unrelated commit (a plot-ticks fix), so its status labels can be trusted. Verified directly against the source, not just the doc:

- Distributions and random processes (the "Phase 1" bucket) are the mature part of the tree — the large majority of `MODEL-DECISIONS.md`'s catalog is marked Implemented, matching the user's own assessment that this part is "essentially complete for now."
- **Nothing from Phase 2 through 5 exists in code.** `symbulate/probability_space.py`'s `__mul__`/`__pow__` still return a plain `ProbabilitySpace` — no `.factors`, no `IndependentProductSpace`. There is no `__rshift__` anywhere in the codebase. `RVConditional.draw()` in `random_variables.py` is still the exact `while True: ... if condition: return` rejection loop, with no `method=` argument. There is no `ContinuousProbabilitySpace`, `Hierarchical`, or `Empirical`/`LifeTable` under `symbulate/` — those four still live only as prototype `.py` files under `team/models-and-sim-design/`.
- The RNG-consolidation patch from the prior audit conversation has **not** been merged either: `probability_space.py`, `distributions.py`, `gaussian_process.py`, `hitting_times.py`, `markov_chains.py`, `diffusion_process.py`, and `plot.py` each still declare their own independent `rng = np.random.default_rng()`. No public `seed()`/`seed_plots()` function exists yet.

## 2. Answers to the three scoping questions

**Which task is more self-contained?** Hierarchical models. `>>` (`HierarchicalProbabilitySpace`, `Hierarchical`, `__rshift__`) touches nothing that exists today — it's a new dunder, a new class, a new function. MCMC's first prerequisite, `IndependentProductSpace` (Task 2a), has to change what `*`/`**` return, and — as detailed in §4 below — that interacts with real, pre-existing performance machinery in a way the design docs didn't anticipate.

**How interrelated are they?** Less than `MODEL-DECISIONS.md`'s "Phase 2 forces one joint decision" framing implies, because that framing was written to guard against reusing `*` for both independence and hierarchy — a path this project's own prior conversations already ruled out in favor of `>>`. With `>>` settled, Hierarchical depends on nothing from MCMC. MCMC depends on `IndependentProductSpace` (2a) only — not on Hierarchical (2b) — provided its decline-guard is written as an allow-list (`isinstance(space, IndependentProductSpace)`), not a deny-list against `HierarchicalProbabilitySpace`. Written that way, the two tracks are independent regardless of build order.

**Is `ContinuousProbabilitySpace` essential to either?** No, given the paths recommended below. Hierarchical models don't touch it at all. MCMC only needs it if the team later chooses single-site exact-conditional Gibbs *with* continuous factors — the recommended first algorithm, component-wise random-walk MH, has no dependency on it. `ContinuousProbabilitySpace` itself is also still stuck on its own open question (subclass `Distribution` or stay a plain `ProbabilitySpace`), unrelated to either main track.

## 3. The `__pow__` layering problem (why MCMC's prerequisite is not "purely additive")

This is the main technical finding from this session and the reason the roadmap below front-loads specific `__pow__`/RNG/multivariate work before `IndependentProductSpace` gets built.

### 3.1 Three layers, one MRO

- `ProbabilitySpace.__pow__` — the generic fallback: loop `self.draw()` `n` times, wrap in a `Vector`.
- `Distribution.__pow__` (`distributions.py:350`) — overrides this for **every** subclass built through the standard `Distribution.__init__(params, scipy, discrete)` recipe. Instead of looping, it calls `self.sim_func(**self.params, size=exponent, random_state=rng)` — one vectorized scipy call for all `n` values. `sim_func = scipy.rvs` is set unconditionally in `Distribution.__init__`. This covers the large majority of the catalog: `Bernoulli`, `Binomial`, `Poisson`, `Normal`, `Gamma`, `Beta`, `Exponential`, `Uniform`, `Geometric`, `NegativeBinomial`, `Pascal`, `NegativeHypergeometric`, `Weibull`, and effectively all Tier 1–4 additions that follow this wrapper pattern.
- `MultivariateDistribution.__pow__`, and separately `Wishart.__pow__` / `InverseWishart.__pow__` — override it a third time, back to the plain draw-loop. These classes never call `Distribution.__init__` (their `__init__`s build `.pdf`/`draw()` by hand for vector- or matrix-valued outcomes), so they never get `self.sim_func`/`self.params`. Confirmed empirically this session: calling `Distribution.__pow__` directly on a `MultivariateNormal` instance raises `AttributeError: 'MultivariateNormal' object has no attribute 'sim_func'`. Their overrides exist purely to prevent that crash, not as a competing optimization.

### 3.2 History — this is old, deliberate machinery, not incidental

- The vectorized `Distribution.__pow__` mechanism dates to **October 7, 2018** (`43e7dadd`, "beginning overhaul of random processes," Dennis Sun) — about two years into the project's life, essentially unchanged since (one cosmetic rename in Dec 2018, one RNG-API modernization in June 2026). It is original-author, load-bearing behavior, not a shortcut to casually rework.
- `MultivariateNormal`'s protective override was added **in the same October 2018 commit** — the original author handled both sides of the conflict together, on day one.
- What's genuinely recent is the **consolidation**, not the mechanism: a `MultivariateDistribution` base class was created **July 29, 2026** (Nico Ragasa) to merge five previously-independent, near-duplicate copies of this same override (`MultivariateNormal`, `MultivariateT`, `Multinomial`, `Dirichlet`, `Hotelling`) into one shared implementation.
- `Wishart`/`InverseWishart` were added **July 28, 2026** — one day *before* that consolidation, by the same author — each carrying its own hand-written `_no_vector_summary` helper and `__pow__` override, verified to be near-verbatim duplicates of each other. They were not folded into the `MultivariateDistribution` refactor the very next day, most likely because that class's `var()`/`sd()`/`corr()` assume a *vector* draw's covariance (`np.diag(cov())`), which doesn't fit a *matrix*-valued Wishart draw — a legitimate reason not to literally subclass it, but not a reason the identical `_no_vector_summary`/`__pow__` pair needed to be duplicated between the two Wishart classes themselves.

### 3.3 Why it matters for `IndependentProductSpace`

Benchmarked this session (looped single `.rvs()` calls vs. one batched `.rvs(size=n)` call, `n=20,000`): Normal 121×, Poisson 403×, Binomial 423×, Gamma 721×, NegativeHypergeometric 1884×, TruncatedNormal 95×. The vectorized path is a broad, large win across nearly the whole catalog — not a narrow fix for one or two pathological distributions. Any change to `**` for MCMC's benefit has to **preserve** this, not bypass it.

Practical consequence: the "if both operands are density-bearing, build an `IndependentProductSpace`" logic cannot be bolted on as a fourth, independent `__pow__` layer. It has to be woven into `Distribution.__pow__` itself (since that's what actually runs for almost every `X ** n` today), preserving the existing fast `sim_func(size=n)` draw for chain initialization and plain `.sim(n)`, while separately exposing `factors = [self] * n` and a `log_density` for MCMC's per-coordinate proposal/accept step (which needs each factor's `.pdf`, not a fresh call to its `draw()`). For `MultivariateDistribution`/`Wishart`/`InverseWishart`, the simplest and safest move is to leave their existing "undo" overrides untouched and have them stay ineligible for product-space treatment — consistent with the open decision already on record in `MODEL-DECISIONS.md` to error clearly on any product containing a multivariate factor, rather than solving that alignment problem now.

There is **no need to narrow the fast path down to fewer distributions** (e.g., to just `NegativeHypergeometric`) to make this tractable — that was checked directly and rejected: it would sacrifice a broadly valuable optimization without touching the actual structural issue, which exists regardless of how many distributions use the vectorized path.

**Recommended eligibility check for `IndependentProductSpace`:** duck-type on `hasattr(x, "params") and hasattr(x, "sim_func")`, not `isinstance(x, Distribution)` and not a class-by-class allowlist. This is the one signal that actually tracks which `__pow__` behavior a given object has, and it stays correct without updates as new multivariate-style classes are added (as `Wishart`/`InverseWishart` demonstrate, new non-scalar classes keep appearing without necessarily being registered anywhere central).

## 4. Recommendation: build order

**Hierarchical models first, in full isolation. MCMC second, gated on a short prerequisite-hygiene pass.**

Hierarchical models require zero changes to `rng`, `__pow__`, or the multivariate-distribution classes — nothing below applies to that phase. The RNG/`__pow__`/multivariate work is entirely in service of `IndependentProductSpace`/MCMC, and is worth doing as dedicated prep before that phase starts rather than discovering it mid-implementation.

---

## 5. Phase-by-phase roadmap

### Phase 1 — Hierarchical models (`>>`)

No dependency on anything below. Fully additive.

1. `HierarchicalProbabilitySpace(ProbabilitySpace)` in `probability_space.py`: draws from a prior space, calls the conditional function on that draw to get a child space, draws from the child, joins the two outcomes via the existing `join()` helper (already flattens nested tuples, which is what makes multi-level chains work for free).
2. `Hierarchical(prior_space, cond_space_func)` top-level function wrapping that class.
3. `__rshift__` on `ProbabilitySpace`, so `prior >> cond_func` calls `Hierarchical(...)`.
4. A `TypeError` guard: if `cond_space_func(x)` doesn't return a `ProbabilitySpace`, raise clearly rather than failing deep inside `.draw()`.
5. Export `Hierarchical` and `HierarchicalProbabilitySpace` from `__init__.py`.
6. Tests: two-level models (Beta-Binomial), multi-level chains (3+ levels, verifying the accumulated-tuple behavior), the `TypeError` guard.
7. Optional, separable, not required for this phase to ship: `Placeholder` syntax, `AssumeHierarchical` escape hatch (needs its own guard against being called twice on the same prior RV, per the design doc's own open item).
8. Revisit, once this ships: `Mixed Poisson process`, `Markov-modulated Poisson process`, `Hidden Markov Model`, and `Tweedie` can be rebuilt using `>>` instead of hand-rolled `ProbabilitySpace(generator_function)` closures — not required, but a natural cleanup once the general mechanism exists.

**No changes needed to `rng`, `__pow__`, `__mul__`, or any `Distribution` subclass for this phase.**

### Phase 2 — MCMC prerequisites (do before writing any MCMC code)

#### 2a. RNG consolidation

Recommended to land before Phase 3, not because MCMC strictly requires it, but because reproducible chains matter far more for MH than they do for today's i.i.d. rejection sampling — debugging an acceptance-rate or mixing problem is much harder if two runs with the "same seed" silently diverge.

- Consolidate the six non-plotting generators (`probability_space`, `distributions`, `gaussian_process`, `hitting_times`, `markov_chains`, `diffusion_process`) into one shared, seedable generator owned by `probability_space.py`.
- Add a public `symbulate.seed(value)` function that actually reaches that shared generator (today's documented `np.random.seed(0)` is a no-op against Symbulate's `default_rng()`-based generators).
- Keep `plot.py`'s generator separate, with its own `seed_plots()` — it only drives cosmetic scatter jitter, and folding it into the model-seed stream would make plotting a simulation silently perturb a later `.sim()` call on the same RV.
- Fix the one test file (`test_probability_space.py`) that currently reassigns the generator directly in a way that would conflict with the shared generator.
- This patch was already drafted and verified in a prior conversation (3,277/3,281 tests passing against the patched copy, isolation between plotting and modeling confirmed) — it just hasn't been applied to `dev` yet.

#### 2b. `__pow__` reconciliation and multivariate-class hygiene

- Decide and document the eligibility rule for `IndependentProductSpace`: `hasattr(x, "params") and hasattr(x, "sim_func")` (see §3.3). Write this once, in one place, rather than as a scattered set of `isinstance` checks.
- Modify `Distribution.__pow__` (not just `ProbabilitySpace.__pow__`) so that, for a finite exponent on a density-bearing `Distribution`, it returns an `IndependentProductSpace` with `factors = [self] * exponent`, while still performing the draw via the existing `self.sim_func(**self.params, size=exponent, random_state=rng)` call — i.e., add the factor/density metadata without giving up the vectorized draw.
- Leave `MultivariateDistribution.__pow__`, `Wishart.__pow__`, and `InverseWishart.__pow__` as-is functionally (still declining, still looping `draw()`), but treat this as the moment to also do the hygiene fix flagged in §3.2: extract a small shared helper (or a thin common base under `MultivariateDistribution`) providing `_no_vector_summary`/`__pow__` for non-scalar-output distributions in general, so `Wishart`/`InverseWishart` stop duplicating that pair verbatim. **This is a recommended cleanup, not a hard blocker** — `IndependentProductSpace` doesn't require it, since the duck-typed eligibility check will correctly exclude these classes either way (none of them set `self.sim_func`/`self.params`). But it's the same area of code, the same underlying problem `MultivariateDistribution` was created to solve one day earlier, and doing it now avoids a sixth near-duplicate copy appearing the next time a new matrix- or vector-valued distribution is added.
- Add a regression test asserting `X ** n` still produces statistically correct, reasonably fast draws for a representative sample of scalar distributions (confirming the vectorized path survived the change) and that `MultivariateNormal ** n` / `Wishart(...) ** n` still work (confirming the decline path was untouched).

#### 2c. `IndependentProductSpace`

- New class in `probability_space.py`:
  ```
  class IndependentProductSpace(ProbabilitySpace):
      factors: list
      def log_density(self, outcome): ...   # sum of log pdf/pmf across factors
  ```
- Change `__mul__` so that when both operands pass the eligibility check, it returns an `IndependentProductSpace` instead of a plain `ProbabilitySpace`; flatten nested products (`(A * B) * C` → one flat `factors = [A, B, C]`, not a tree) by checking whether either operand is already an `IndependentProductSpace` and splicing its factors in.
- Fallback path for non-eligible operands (`BoxModel`, bare user `ProbabilitySpace(lambda: ...)`) — unchanged plain-join behavior, no density, `method="mcmc"` declines later with a clear error.
- Explicitly decline (raise a clear error, not a silent wrong answer) on any product containing a multivariate/matrix-valued factor, per the open decision already on record in `MODEL-DECISIONS.md` — don't try to solve the "flattened tuple with no marker between factors" alignment problem in this pass.
- Tests: 2- and 3-way products' `.factors`/`log_density` correctness, flattening, the eligibility duck-type (including against `MultivariateNormal`/`Wishart` to confirm they're correctly excluded), the non-eligible fallback path, and the multivariate-factor error.

### Phase 3 — MCMC, first algorithm

Gated on Phase 2c only, not on Phase 1.

1. Add `method="mcmc"` to `RVConditional`, restricted via an **allow-list** check (`isinstance(self.prob_space, IndependentProductSpace)`) — not a deny-list against `HierarchicalProbabilitySpace`, so this stays correct regardless of whether Phase 1 has shipped.
2. Clear, explicit error (`NotImplementedError`, not a silent wrong answer) on any non-`IndependentProductSpace`-backed model, including hierarchical-space-backed ones.
3. Build component-wise random-walk Metropolis-Hastings:
   - Propose a symmetric move on one coordinate at a time (`Normal(ω_i, scale=0.5 * factor.sd())` for continuous factors; a symmetric reflected integer walk for discrete/count factors).
   - Accept with probability `min(1, f_i(ω_i') / f_i(ω_i))` if the proposed full outcome satisfies `condition_event.func`; otherwise stay put.
   - Initialize via bounded ordinary rejection sampling to find one valid starting state; surface a clear error if even that isn't practical, rather than hanging.
4. `RVConditional.sim` overridden directly (not just `draw`), since a chain needs mutable state across calls; still slots into the existing `_sim_with_progress` helper.
5. Return type stays `RVResults`; attach non-breaking diagnostics (`results.acceptance_rate`). Docstring must say plainly that draws are not i.i.d.
6. Validation: compare `method="mcmc"` output against `method="rejection"` output (mean, histogram/KS test) on cases cheap enough for rejection to still work — a strong, close-to-free regression check specific to this feature.
7. Document the `eps`-band caveat honestly (per the design doc's own §9 revision): MH does not "solve" continuous conditioning — it only helps with staying inside/exploring a band once already inside it, not with defining the band or finding the first valid state.
8. Explicitly **not** in this phase: single-site exact-conditional Gibbs (would hard-depend on `ContinuousProbabilitySpace`'s inverse-CDF machinery for continuous factors — defer until there's a concrete reason to want exact conditional draws over MH's cheaper accept/reject), exact manifold sampling for linear constraints (worth prioritizing later specifically because it removes the `eps`-band issues entirely for sums, but is a distinct, special-cased feature), HMC/NUTS (recorded non-goal — no autodiff layer, mixed discrete/continuous factors, hard-indicator conditioning events all block it independently).

### Phase 4 — Deferred, not scheduled

- `ContinuousProbabilitySpace` merge — independent side quest, still blocked on its own open question (subclass `Distribution` or not); revisit only if Gibbs becomes the priority.
- Phase 5, Markov-blanket-aware Metropolis-within-Gibbs for `HierarchicalProbabilitySpace`-backed models — genuinely separate, unscoped work; only worth starting once both Phase 1 and Phase 3 have shipped and there's real demand for sampling from a hierarchical model's posterior.
- HMC/NUTS — explicit non-goal, not a placeholder future phase.

---

## 6. Open decisions this roadmap surfaces or inherits

- [ ] Whether to build the `Wishart`/`InverseWishart` shared-helper cleanup (§2b) as part of Phase 2, or file it separately and let it ride on its own timeline — recommended to bundle with Phase 2 since it's the same code region, but not required for correctness.
- [ ] `Placeholder` syntax and `AssumeHierarchical` — include in Phase 1's initial PR or ship as fast-follow additions (both optional per the design doc).
- [ ] `burn_in`/`thin` defaults for MCMC, and whether they scale with `n`.
- [ ] Where MCMC diagnostics (`acceptance_rate`, ESS) live — `RVResults` attributes vs. a separate object.
- [ ] Final spelling of `method="mcmc"`.
- [ ] Whether to prioritize exact manifold sampling for linear constraints ahead of other later/optional MCMC items, given it removes the `eps`-band issue entirely for sums.
- [ ] Whether `ContinuousProbabilitySpace` subclasses `Distribution` or stays a plain `ProbabilitySpace` — unresolved, independent of this roadmap, only actually blocking if/when Gibbs is prioritized.

---

## Appendix: evidence log

| Finding | Source |
|---|---|
| `MODEL-DECISIONS.md` current as of Aug 5 2026; `dev` tip Aug 6 (unrelated commit) | `git log -3 -- MODEL-DECISIONS.md`; `git log -1` |
| No `IndependentProductSpace`/`Hierarchical`/`__rshift__`/`ContinuousProbabilitySpace`/`mcmc` in `symbulate/` | `grep -rniE` across `symbulate/` |
| `ProbabilitySpace.__mul__`/`__pow__` still plain-`ProbabilitySpace` | `symbulate/probability_space.py` (read directly) |
| `RVConditional.draw()` unchanged rejection loop, no `method=` | `symbulate/random_variables.py:438-484` |
| `Distribution.__pow__` vectorized via `sim_func` | `symbulate/distributions.py:350` |
| `sim_func = scipy.rvs` set unconditionally | `symbulate/distributions.py:~210` |
| Vectorized `__pow__` origin: Oct 7 2018, `43e7dadd`, Dennis Sun | `git log -G"self\.sim_func"`, `git show 43e7dadd` |
| `MultivariateNormal`'s protective override added in the same commit | `git show 43e7dadd -- symbulate/distributions.py` |
| `MultivariateNormal` predates the conflict (2017, was `ProbabilitySpace` subclass, then `Distribution` subclass before `__pow__` existed) | `git log -G"class MultivariateNormal"` |
| `AttributeError` confirmed empirically without the override | direct `Distribution.__pow__(MultivariateNormal(...), 3)` call, this session |
| `MultivariateDistribution` consolidation: July 29 2026, Nico Ragasa, merges 5 prior duplicate overrides | `git show fa0df40` |
| `Wishart`/`InverseWishart` added July 28 2026, same author, one day before consolidation | `git show 84aa80a` |
| `Wishart`/`InverseWishart`'s `_no_vector_summary`/`__pow__` are near-verbatim duplicates of each other | direct extraction/comparison, this session |
| Benchmark: batched vs. looped `.rvs()`, 100–1900× speedup across ordinary and pathological distributions | benchmark run, this session |
| `NegativeHypergeometric` already benefits from the existing `__pow__` fast path with no override of its own | `grep -n "^class NegativeHypergeometric"`, confirmed no `__pow__` override |
| `team/fast_sim_notes.md`'s `_fast_sim(n)` proposal is unrelated to `__pow__`; targets `.sim(n)` only, not yet merged | `git show fc67ed6` |
| RNG consolidation patch drafted in prior session (3,277/3,281 tests passing) but not applied to `dev` | prior conversation; confirmed `rng = np.random.default_rng()` still independently declared in 7 files |
