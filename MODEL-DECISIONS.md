# MODEL-DECISIONS.md — Symbulate Models & Simulation Build Plan

This is a lightweight decision log for the models/simulation proposals in
`team/models-and-sim-design/`, in the same spirit as `DECISIONS.md` (which
covers the graphics overhaul only — this file does not touch or supersede
anything in it). Nothing in this file has been ratified by the team yet;
every phase below is a proposed ordering, not a committed roadmap.

- **Source proposals:** `team/models-and-sim-design/*.md` and their
  companion prototype `.py` files.
- **This file records:** the build order, why it's ordered that way, which
  parts of the order are forced by real code dependencies vs. just a
  preference, and what's still open within each phase.

Update this file when a phase actually starts or a decision is finalized.
Never remove an entry — mark it superseded instead.

---

## Decision: Overall Phasing

**Status:** Proposed

**Decision**
> Five sequential phases plus one parallel track:
>
> 1. **Merge the independent prototypes** — `RenewalProcess`,
>    `DiffusionProcess`, `ContinuousProbabilitySpace`, `Empirical`/`LifeTable`.
> 2. **Shared product-space architecture** — resolve the hierarchical-model
>    operator and build `IndependentProductSpace` jointly.
> 3. **Hierarchical models** (core mechanism, then optional `Placeholder`/
>    `AssumeHierarchical` layers).
> 4. **MCMC, first algorithm** — component-wise random-walk MH, restricted
>    to independent-product models.
> 5. **Markov-blanket-aware sampler** for hierarchical models.
>
> Running alongside all five: the distribution-additions catalog and the
> process roadmap. See "Decision: Phase 1 Scope — Process Roadmap &
> Distribution Additions" below for the full item-by-item classification —
> the short version is that essentially none of either document is gated
> by Phases 2-5; most of it is Phase-1-scope work with its own internal
> chaining (e.g. Compound Poisson needs `RenewalProcess` first).

**Rationale**
> Phase 1 has zero dependencies and is pure packaging/merge work on
> already-tested prototypes, so it can start immediately and in parallel
> across its four components. Phases 2 through 5 form a genuine dependency
> chain: both the Hierarchical-models design (§6-8) and the MCMC design
> (§3.1, §7) independently want to change what `ProbabilitySpace.__mul__`/
> `__pow__` return, and if that's decided twice instead of once, the two
> features end up with incompatible joint-space representations that a
> future Markov-blanket sampler (Phase 5) would need to unify anyway. Phase
> 2 exists specifically to force that decision to happen once. Phase 5 is
> real new work (not a free extension of Phase 4), per the MCMC design's
> own §7.1 analysis, so it's placed last and explicitly gated on both 3 and
> 4 being done.

**Alternatives Considered**
> Building Hierarchical (Phase 3) and MCMC (Phase 4) independently, each
> choosing its own product-space representation — rejected; this is exactly
> the scenario Phase 2 is inserted to prevent (see "Decision: Phase 2 —
> Shared Product-Space Architecture" below for the specific failure mode).
> Doing distribution/process-roadmap work as its own "Phase 0" before
> anything else — considered and rejected as a fixed phase, since almost
> none of it depends on the trio below; it's left as an unordered parallel
> track instead, so it doesn't artificially block or get blocked by the
> phases above.

---

## Decision: Phase 1 — Merge Independent Prototypes

**Status:** Proposed

**Decision**
> Merge `RenewalProcess` (`renewal_process.py`), `DiffusionProcess`
> (`diffusion_process.py`), `ContinuousProbabilitySpace`
> (`continuous_probability_space.py` → `symbulate/continuous.py`), and
> `Empirical`/`LifeTable` (`table_distributions.py` → `symbulate/empirical.py`)
> into the real package, each with an export in `symbulate/__init__.py` and
> a real `unittest`-style test file in `symbulate/tests/`. No shared new
> abstraction is required — each of the four was independently prototyped
> and tested against `dev` already.

**Rationale**
> All four are described as prototyped-and-tested in their source docs, and
> none of them touch `ProbabilitySpace.__mul__`/`__pow__`, `RV`, or
> `Distribution` internals — they're additive, self-contained files. This
> makes them the cheapest, lowest-risk phase, and a natural place to also
> land two small hygiene fixes Phase 2 will need anyway: promoting
> `ContinuousProbabilitySpace._estimate_scale()` to a public `.sd()` (and
> setting `.discrete = False`), since MCMC's eligibility duck-typing depends
> on both.

**Alternatives Considered**
> Deferring these merges until after Phases 2-5, on the theory that they're
> lower-priority — rejected; there's no dependency forcing that, and
> shipping them now unblocks the Compound-Poisson/Cramer-Lundberg branch of
> the process roadmap sooner.

**File-level notes (see chat for full detail):**
> - `continuous_probability_space.py` has one dead import (`numbers`,
>   never used) and no public `.sd()`/`.mean()`/`.var()` or `.discrete`
>   attribute yet — both needed before Phase 2.
> - `renewal_process.py` has no guard against a negative-support
>   `interarrival_dist` (e.g. `Normal`), which would silently violate the
>   nondecreasing-counting-function invariant — a real correctness gap, not
>   just a documentation TODO. **Resolved and merged** — see "Decision:
>   `RenewalProcess` Interarrival-Distribution Validation" below.
> - `diffusion_process_demo.py`'s `np.random.seed(42)` does **not** seed
>   the module's actual `rng = np.random.default_rng()` generator — the
>   real test file must reseed via `diffusion_process.rng = ...`, matching
>   `test_poisson_process.py`'s pattern, not the demo script's.
> - `table_distributions.py` imports `pandas`, which is not in CLAUDE.md's
>   approved dependency list (`numpy`, `scipy`, `matplotlib`) and has no
>   recorded team agreement — **this blocks merge** until the team either
>   approves pandas as a new dependency or `LifeTable`'s CSV/column parsing
>   is rewritten without it.
> - `test_table_distributions.py` is a sandbox artifact (absolute paths,
>   non-relative import, hand-rolled pass/fail harness instead of
>   `unittest.TestCase`) and must be rewritten, not copied.

---

## Decision: `RenewalProcess` Interarrival-Distribution Validation

**Status:** Implemented — `symbulate/renewal_process.py`,
`symbulate/tests/test_renewal_process.py`, exported from
`symbulate/__init__.py`. Closes the Phase 1 file-level gap noted above and
unblocks the Compound-Poisson → Cramér-Lundberg / Sparre Andersen and
G/G/1 → M/G/1 / G/M/1 branches of the process roadmap.

**Decision**
> `RenewalProcessProbabilitySpace` validates `interarrival_dist` up front —
> the renewal analogue of `PoissonProcessProbabilitySpace`'s `rate > 0`
> check — and `RenewalProcess` inherits the check by building that space.
> Four rejections, in this order:
> 1. Not a Symbulate `Distribution` → `TypeError`. A number additionally
>    gets a pointer to `PoissonProcess(rate=...)`, since passing a rate is
>    the likely mistake.
> 2. A `MultivariateDistribution` → `TypeError` (a vector per draw is not
>    one waiting time; without this, the counting loop fails later with a
>    raw numpy error).
> 3. Smallest possible time `< 0` → `ValueError`. **This is the gap being
>    fixed:** a negative interarrival time makes the counting function
>    non-monotone, so `N(t)` can decrease.
> 4. Every draw is certainly 0 (`Poisson(0)`, `DiscreteUniform(0, 0)`) →
>    `ValueError`. Not part of the original gap, but the same invariant
>    seen from the other side, and previously an **infinite hang** rather
>    than an error: infinitely many events at time 0 means the cumulative
>    sum never passes any `t`.
>
> The bound is read from scipy's own `support()` via
> `Distribution._support()`, so a newly added distribution is validated with
> no per-distribution code — the same "ask scipy once" approach
> `Distribution._compute_xlim` already uses for the plotting window.
> Parameter-level validation (`Gamma`'s shape must be positive, etc.) stays
> with the distribution's own constructor; this resolves open question 1 in
> `team/models-and-sim-design/renewal_process_notes.md` as well.

**Rationale**
> A renewal process's whole contract is a nondecreasing count of events over
> time, so an interarrival distribution with negative support isn't a
> questionable modeling choice — it produces output that is not a renewal
> process at all, silently. Checking the support analytically (rather than
> sampling draws to look for a negative one) keeps the check exact, free of
> RNG consumption, and paid once per process construction rather than once
> per draw — the same cost concern behind the lazy `xlim` property in
> `distributions.py`.

**Alternatives Considered**
> Leaving it a documented assumption, on the grounds that Symbulate
> generally trusts the user's distribution choice (option 2 in the
> prototype's notes) — rejected: the failure is silent and produces a
> plausible-looking path, which is exactly the case the CLAUDE.md error
> standard exists for. A `warnings.warn` instead of raising — rejected for
> the same reason; a warning scrolls off in a notebook while the wrong
> numbers stay on screen. Sampling a few draws and rejecting on a negative
> value — rejected: inexact (it can miss a rare negative tail), consumes
> RNG draws at construction time, and would break seeded reproducibility.

**Known limitation (accepted, not a TODO):**
> A point-mass distribution written as `Uniform(a=2, b=2)` gives scipy a
> degenerate parameterization, and scipy reports its support as `(nan, nan)`
> — so nonnegativity cannot be verified and the distribution is accepted on
> the assumption that the value is positive. Accepting is the right call
> because the nan is not evidence of a problem: `Uniform(a=2, b=2)` is a
> perfectly valid interarrival distribution. The one bad case this lets
> through, `Uniform(a=0, b=0)`, still hangs — a degenerate-parameter
> pathology in `Uniform` itself, better fixed there (require `a < b`) than
> worked around here. `LogNormal(mu, 0)` is *not* affected: it has no scipy
> object at all, so the check reads its hand-written `quantile` instead and
> correctly sees the point mass at `exp(mu)`.

**Deliberately not done:** `PoissonProcess` was left untouched, rather than
reimplemented as `RenewalProcess(Exponential(rate=rate))`. It is
mathematically the special case, but rewriting it would churn a module with
its own passing tests and docs for no user-visible gain.

---

## Decision: G/G/1 Queue via Lindley's Recursion

**Status:** Implemented — `symbulate/queues.py`,
`symbulate/tests/test_queues.py`, demo in
`team/models-and-sim-design/gg1_queue_demo.ipynb`, exports for `GG1`,
`GG1ProbabilitySpace`, `GG1Result`, `MG1`, and `GM1` in
`symbulate/__init__.py`. This is item 12 of the process roadmap's suggested
build order, and it is the branch the `RenewalProcess` decision above listed
as unblocked.

**Decision**
> The G/G/1 family is a **waiting-time process indexed by customer number**,
> not a queue-length process indexed by continuous time, and it lives in a
> **new module** rather than in `markov_chains.py`.
>
> - `GG1Result(InfiniteVector)` holds one sample path: `path[n]` is customer
>   `n`'s wait in line (not counting their own service), generated on demand
>   by Lindley's recursion `W[n+1] = max(W[n] + S[n] - A[n+1], 0)` and cached
>   in a `waits` list — the same lazily-extending pattern as
>   `RandomWalkResult.positions` and `MarkovChainResult`'s states.
> - `GG1ProbabilitySpace` validates both distributions, then builds the two
>   `dist ** inf` sequences **once** (not per draw), as `RenewalProcess` and
>   `PoissonProcess` do.
> - `GG1(RV)` is the public class, with `MG1(arrival_rate, service_dist)` and
>   `GM1(interarrival_dist, service_rate)` as thin subclasses that substitute
>   an `Exponential` on one side — mirroring how `MMss` subclasses `MMsK`.
> - Kendall-notation class names (`GG1`, `MG1`, `GM1`) to match the existing
>   `MM1`/`MMs`/`MMsK`/`MMss`/`MMsKN`/`MMInfinity` wrappers.
> - `interarrival_dist` is reused as the arrival-side parameter name,
>   verbatim from `RenewalProcess`; `service_dist` is its service-side
>   counterpart.
> - `GG1.utilization` exposes the traffic intensity `rho` (mean service /
>   mean interarrival). `rho >= 1` is **not** an error — an unstable queue is
>   a legitimate thing to simulate deliberately, and the docstring says what
>   to expect from one instead.

**Rationale**
> A general-service queue is the point where the birth-death machinery in
> `markov_chains.py` genuinely stops applying: with non-exponential service,
> the remaining service time depends on the elapsed service time, so the
> number in the system is not a Markov chain and there is no generator matrix
> to hand `ContinuousTimeMarkovChain`. Lindley's recursion sidesteps that
> entirely by changing what is indexed — customers instead of clock time —
> and is then exact for *any* pair of nonnegative distributions, with no
> truncation of the state space and no `num_states` argument to pick. Putting
> it in its own module keeps that boundary legible: `markov_chains.py` holds
> what is a Markov chain, `queues.py` holds what is not.
>
> `M/G/1` and `G/M/1` really do fall out for free, as the roadmap predicted,
> so they are subclasses rather than separate implementations.

**Alternatives Considered**
> *Making the G/G/1 result a continuous-time queue-length function `N(t)`
> instead of a waiting-time sequence* — deferred, not rejected: it is its own
> roadmap row (item 20) and needs departure/arrival event-stream merging.
> `GG1Result` is built so that work is additive later; it already exposes
> `arrival_times` and `departure_times`, which is exactly the pair such an
> `N(t)` needs. *Adding the classes to `markov_chains.py`* — rejected; they
> are the one queue family in the package that is not a Markov chain, and
> filing them there would imply otherwise. *Erroring or warning when
> `rho >= 1`* — rejected; watching an overloaded queue fail to settle down is
> a standard exercise, and a warning would scroll past in a notebook while
> the (correct, unbounded) numbers stayed on screen. *Free functions
> `waiting_times(path)` / `service_times(path)` in `math.py`, matching
> `arrival_times(path)`* — rejected: `math.py` is star-exported, and names
> that generic are ones a student is likely to use as their own variables.
> The sequences are attributes on the path instead, reachable as random
> variables through the existing `.apply()` idiom.

**Reuse of the `RenewalProcess` groundwork:** the arrival side is a renewal
process, and this is made literal rather than reimplemented. Both
nonnegativity checks call `renewal_process._smallest_possible_time` and
`_is_always_zero`, so the "ask scipy's `support()` once" approach extends to
service-time distributions with no new per-distribution code, and
`GG1Result.get_arrival_process()` hands the arrival stream back as an actual
`RenewalProcessResult` counting function.

**One deliberate asymmetry between the two distributions:** a service time
that is 0 on every draw is *accepted* (a server that finishes instantly is
degenerate but harmless — nobody waits), while an interarrival time that is 0
on every draw is *rejected*, exactly as in `RenewalProcess`, since every
customer would arrive at the same instant.

**Known limitation (accepted, inherited):** a point mass written as
`Uniform(a=b)` reports a nan mean as well as a nan support, so
`utilization` comes back `None` for it — the same degenerate-parameterization
gap documented in the `RenewalProcess` decision above, seen through
`mean()` instead of `support()`. The queue itself simulates correctly; only
the reported `rho` is unavailable. Consequently the `M/D/1` (deterministic
service) examples in the docstring and demo use a low-variance `Gamma`
rather than a degenerate `Uniform`, which also makes the
Pollaczek-Khinchine check exact.

---

## Decision: G/G/s — Multi-Server Queues

**Status:** Implemented — `GGs`, `GGsProbabilitySpace`, and `GGsResult` in
`symbulate/queues.py`, tested in `symbulate/tests/test_queues.py`, demo in
`team/models-and-sim-design/ggs_queue_demo.ipynb`, exported from
`symbulate/__init__.py`. Closes the roadmap's "G/G/s (multi-server
general-service queue)" row.

**Decision**
> Simulate by tracking **when each server next comes free**, not by
> generalizing Lindley's leftover arithmetic. Customer `n` arrives at `A[n]`,
> takes the server that frees up soonest, and waits
> `max(earliest free time - A[n], 0)`; that server is then busy until
> `A[n] + W[n] + S[n]`. The free times live in a `heapq`, so the earliest is
> always at position 0 — the only one the recursion asks about. This is the
> Kiefer–Wolfowitz vector recursion in the form that is cheapest to read.
>
> `GG1` is **not** reimplemented as `GGs(servers=1)`, and `GGs` does not
> subclass `GG1`. Instead both subclass a new private `_QueueResult`, which
> owns the two input sequences, the lazily-extended `waits` list, and the
> derived `arrival_times` / `sojourn_times` / `departure_times`; each concrete
> class supplies only `_wait_at(n)`. `GGs(servers=1)` is verified to reproduce
> `GG1`'s waits value-for-value on identical inputs.
>
> `servers` is a required argument (no default), validated as an integer `>= 1`
> by `_validate_servers`. `utilization` becomes
> `mean service / (servers * mean interarrival)`.
>
> No `MGs` / `GMs` classes — pass an `Exponential` on the Markovian side, the
> way `GG1(Exponential(...), ...)` already serves as `M/G/1`.

**Rationale**
> The roadmap called this "harder... needs order statistics over `s` servers'
> completion times," and it is harder *as a generalization of the leftover
> form* of Lindley's recursion. Reframed in clock time it is barely harder at
> all: the state is the multiset of free times, and only its minimum matters,
> which a heap gives in `O(log s)`. Choosing the *earliest* free server rather
> than an arbitrary idle one is without loss of generality — later arrivals
> come no earlier, so whichever idle server is picked, the remaining free times
> stay below the next arrival and no future wait changes.
>
> `MMs` in `markov_chains.py` already covers the exponential case as a queue
> *length*; this covers the general case as a *wait*, and the two are worth
> comparing in a course. Keeping `GG1` untouched preserves the pedagogically
> central Lindley formula as its own readable recursion, and follows the same
> reasoning as leaving `PoissonProcess` alone rather than folding it into
> `RenewalProcess`.

**Alternatives Considered**
> *`GGsResult(GG1Result)`, overriding the recursion* — rejected: a multi-server
> queue is not a kind of single-server queue, and the inheritance would assert
> otherwise for the sake of code reuse. The shared private base says what is
> actually shared. *Folding `GG1` into `GGs` with `servers=1`* — rejected;
> it would churn tested code and demote Lindley's recursion to a special case
> of a general routine, when it is the thing a course teaches first. *Per-server
> `service_dist` (heterogeneous servers)* — deferred; every server serving from
> one distribution is the standard `G/G/s`, and heterogeneous servers need a
> different assignment rule (fastest free? first free?) that should be decided
> deliberately rather than implied by an argument shape.

**Verified against exact theory:** the Erlang C formula for `M/M/2` and
`M/M/3` (mean wait and `P(wait > 0)`, at `rho` of 0.5, 0.667, and 0.75), plus
the classic equal-utilization comparison — two servers of rate `mu` keep the
*line* shorter than one server of rate `2mu` (1/3 vs 1/2) while making the
*visit* longer (4/3 vs 1), which is exactly the tradeoff a course asks
students to find.

**Known consequence, documented not fixed:** with `s > 1`,
`departure_times` is no longer increasing, since a short service can overtake
a long one on the next server. The single-server test asserting monotone
departures is therefore scoped to `GG1`, and `GGsResult` has a test asserting
the overtaking *does* happen.

---

## Decision: Non-Homogeneous Poisson Process — Time-Change, Not Thinning

**Status:** Implemented — `NonHomogeneousPoissonProcess`,
`NonHomogeneousPoissonProcessProbabilitySpace`, and
`NonHomogeneousPoissonProcessResult` live in `symbulate/poisson_process.py`,
tested in `symbulate/tests/test_poisson_process.py`, exported from
`symbulate/__init__.py`. This is roadmap step 13, the prerequisite for the
Cox process (step 14), the Hawkes process (step 17), and the
Weibull/power-law (Crow-AMSAA) reliability framing.

**Decision**
> Simulate by **time change**, not thinning. Events are drawn at a steady
> rate of 1 on the "expected count" scale (i.i.d. `Exponential(rate=1)`
> gaps, exactly as `PoissonProcess` draws clock-time gaps) and read back
> onto the clock through the cumulative rate `Λ(t) = ∫₀ᵗ rate(s) ds`.
> Counting is then the *same* cumulative-sum walk `PoissonProcessResult`
> already does, with one substitution: the running total is compared
> against `Λ(t)` instead of against `t`. `N(t) ≤ k` iff the `k`-th rate-1
> arrival exceeds `Λ(t)`, so no numerical *inversion* of `Λ` is needed
> anywhere — only its forward evaluation.
>
> `rate=` accepts a function of time, or a positive number (which gives an
> ordinary Poisson process — useful for a side-by-side comparison). `Λ` is
> exposed publicly as `.cumulative_rate`, since it is both the theoretical
> mean to check a simulation against and the object the Weibull/power-law
> framing is defined in terms of.

**Rationale**
> Thinning needs an upper bound on the rate over the region being
> simulated. That bound is either an extra required argument — against
> CLAUDE.md's "no required arguments beyond what is mathematically
> necessary" — or numerically guessed from a grid, which is silently wrong
> whenever the rate spikes between grid points. Time change needs nothing
> from the user beyond the rate function itself, is exact for *any*
> nonnegative rate, and reuses the existing counting machinery instead of
> introducing a second, parallel one. It also composes with the lazy
> infinite-vector design already in place: `N(t)` only ever needs rate-1
> arrivals up to `Λ(t)`, so nothing is generated speculatively and there is
> no simulation horizon to pick.

**Alternatives Considered**
> **Thinning (Lewis-Shedler)** — rejected as the default for the reasons
> above, and *deliberately not implemented at all yet*, even though the
> roadmap describes Cox and Hawkes as building on "non-homogeneous Poisson
> thinning." That description is right about those two and does not
> generalize backwards: Hawkes needs thinning (Ogata's method) because its
> intensity depends on the process's own past events, so `Λ` isn't known in
> advance, and a Cox process thinning against a rough realized intensity
> path (a diffusion, say) can't be integrated reliably either. Both are
> path-dependent-intensity problems. Building thinning speculatively now,
> with no such intensity to thin against, would mean shipping a second
> algorithm with a bound argument no current caller needs. It should be
> added when Cox/Hawkes land, and can then sit behind the same public class
> or its own.
> **Inversion of `Λ` by root-finding** to get explicit arrival times —
> rejected as unnecessary: it costs a root-find per event and buys nothing
> the count needs (see Decision above). Worth revisiting only if arrival
> times are exposed as a user-facing sequence.
> **A separate module** `nonhomogeneous_poisson_process.py` — rejected in
> favor of grouping the Poisson family in one module, matching how
> `markov_chains.py` holds `MarkovChain`, `ContinuousTimeMarkovChain`, the
> birth-death queues, and SIR/SEIR, and `gaussian_process.py` holds
> Brownian motion, Ornstein-Uhlenbeck, and the rest. Compound Poisson and
> Cox are expected to join it.

**Numerical policy (the one genuinely new risk this feature adds):**
> `Λ` is a numerical integral of a function Symbulate has never seen, so
> quadrature can return a confidently wrong number. Two real cases:
> `rate=lambda t: 1/t` integrates to a finite 41.7 over `(0, 1)` with a
> huge error estimate, and `rate=lambda t: t**-2` integrates to **-1.0**.
> Policy, in `_CumulativeRate`:
> - Every evaluation of the user's rate function is checked (nonnegative,
>   finite, numeric) — including the ones quadrature makes internally, so a
>   rate that goes negative only partway through is caught at the time it
>   does, naming that time.
> - An integral is trusted only if it is finite, nonnegative, and its
>   estimated error is within `_QUAD_TOL` (1e-3, relative). Otherwise it is
>   retried over 32, then 512, then 4096 pieces — which rescues a genuinely
>   hard but integrable rate (a rate oscillating 1000 times per unit time
>   integrates exactly this way) — and if it still fails, **raises** rather
>   than returning a number nobody should rely on.
> - `Λ` is cached on the probability space, and each new time is integrated
>   forward from the nearest earlier known time, so a `.sim(10000)` at one
>   time integrates once for all 10000 paths, and plotting over a grid pays
>   only for each new step.
> - scipy's `IntegrationWarning` is suppressed: it describes a subdivision
>   detail, and the checks above turn it into either a retry or a
>   student-readable error.
>
> **Accepted limitation:** a spike far narrower than the interval being
> integrated (width 1e-3 somewhere in the first 100 time units) can be
> stepped over by adaptive quadrature and its events missed, with a small
> error estimate that gives no hint. This is the one place thinning would
> be strictly more faithful (at a large efficiency cost). Documented in the
> class `Notes`; rates that vary on a scale near the times being asked
> about — daily cycles, wear, a shift change — are unaffected.

**Follow-on this unblocks, not done here:** the Weibull/power-law
(Crow-AMSAA) process is now purely a naming exercise —
`rate=lambda t: (shape / scale) * (t / scale) ** (shape - 1)` — and the
roadmap wants it exposed under its reliability-engineering name. Left as
its own roadmap row rather than smuggled in here.

**Open:** whether to add a short alias (`NHPP`) alongside
`NonHomogeneousPoissonProcess`. The queueing wrappers in `markov_chains.py`
(`MM1`, `MMs`) set a precedent for abbreviations *when the abbreviation is
the standard textbook name*; "NHPP" is standard in reliability courses but
not in intro probability. Not added — one name for now.

---

## Decision: Hitting Times — Tier B First (recorded after the fact)

**Status:** Implemented in PR #272 — `hitting_time` in
`symbulate/hitting_times.py`, tested in `symbulate/tests/test_hitting_times.py`,
demo in `team/models-and-sim-design/hitting_times_demo.ipynb`, exported from
`symbulate/__init__.py`.

**This entry is a transcription, not the author's own.** It was written from
the shipped module's docstrings and tests so the log has a record of what
landed and what did not; whoever wrote the feature should correct or expand the
rationale below.

**What was built**
> `hitting_time(process, level, max_time=100.0, start_time=0.0, step=None,
> tol=1e-6)`, covering the **Gaussian-process family only**: `BrownianMotion`,
> `BrownianBridge`, `OrnsteinUhlenbeck`, `FractionalBrownianMotion`,
> `GeometricBrownianMotion`, and a hand-built `GaussianProcess`. Given a
> process it returns a random variable; given a drawn path it returns a number.
>
> Between two evaluated times a path can cross the level and return unseen, so
> the crossing is decided by a Bernoulli draw with the reflection-principle
> probability `exp(-2 (level - x0)(level - x1) / (rate * (t1 - t0)))` and then
> localized by bisection. *Whether* a crossing happened is exact for Brownian
> motion and Brownian bridges at any `step`, approximate for other Gaussian
> processes; *exactly when* is accurate to about one `step` for all of them.
> Geometric Brownian motion is exact via the log scale, since a price reaching
> a level is its log reaching the log level.
>
> Everything else — random walks, Markov chains, the `GG1`/`GGs` queues,
> `DiffusionProcess` — raises `NotImplementedError` with a message naming
> itself and saying what it would need instead.

**Ordering note, worth a team decision:** the roadmap's build order puts
**Tier A** (discrete-time and pure-jump processes) at step 7 and Tier B at step
10, on the grounds that Tier A is both easier and *exact* — it just walks
`path[0], path[1], ...` or the jump-time sequence with a cutoff, since nothing
is hidden between queried points. PR #272 built Tier B first and left Tier A
unbuilt, so the cheap-and-exact half of the utility is still missing while the
subtle half exists. Nothing is wrong with the code; the gap is that
`RandomWalk`, `MarkovChain`, `RenewalProcess`, the birth-death queues, and the
`GG1`/`GGs` queues — most of the processes in the package — cannot answer a
hitting-time question at all. One Tier A implementation would cover all of
them, e.g. "which customer is the first to wait more than 10 minutes?"

**Seeding:** the module draws from its own `hitting_times.rng`, so
`np.random.seed` does nothing to it — the same trap `diffusion_process.rng`
already documents. Its test file reseeds both `gaussian_process.rng` and
`hitting_times.rng`.

---

## Decision: Phase 1 Scope — Process Roadmap & Distribution Additions

**Status:** Proposed

**Decision**
> Classified every item in `symbulate_process_roadmap.md` (~50 processes),
> its revision `symbulate_process_roadmap_hitting_times.md` (same roadmap,
> plus a new cross-cutting general hitting-time/upcrossing utility that
> supersedes the standalone "first-passage/ruin-time utility" row from the
> original), and `symbulate_distribution_additions.md` (55 distributions)
> against the phase plan. **Finding, reconfirmed after the revision: neither
> document contains a single item that is hard-blocked by Phase 2, 4, or
> 5.** Nothing in either list — including the new hitting-time utility —
> requires the `*`/`**` product-space change or the MCMC conditional-
> sampling work to exist. The only real link to a later phase is *soft*: a
> handful of "compound/mixture" items are natural fits for Phase 3's
> `Hierarchical` mechanism, but Symbulate's existing manual
> `ProbabilitySpace(generator_function)` pattern — already demonstrated by
> the Hierarchical design doc's own `BetaBinomialProbspace` example — means
> they can ship in Phase 1 today and would only be *revisited*, not
> blocked, once Phase 3 exists.

**Process roadmap — Phase 1 scope, no dependency on Phases 2-5:**
> Random Walk; MA(q); AR(p)/ARMA(p,q); Galton-Watson/branching process;
> GARCH(1,1)/ARCH; Bernoulli process (already expressible today);
> Ornstein-Uhlenbeck and Brownian Bridge (via the existing
> `GaussianProcessProbabilitySpace`, not even requiring Phase 1's
> `DiffusionProcess`); Fractional Brownian Motion; Telegraph process,
> multi-state life-insurance models, and multi-state/competing-risks
> survival models (all already covered structurally by the existing
> `ContinuousTimeMarkovChain` — documentation/example gaps, not new
> mechanism); Non-homogeneous Poisson Process; the birth-death-generator-
> matrix wrapper and everything it covers in one pass (Birth-death/Yule,
> M/M/1, M/M/s, M/M/s/K, M/M/s/s, M/M/s/K/N, M/M/infinity, SIR/SEIR);
> Polya urn scheme; Chinese Restaurant Process; Weibull/power-law
> (Crow-AMSAA) process framing; shot noise/filtered Poisson process (once
> Compound Poisson exists, itself Phase 1 — see chained list below);
> Little's-Law-style utility methods. (The standalone "first-passage/
> ruin-time utility" row from the original roadmap doc is now superseded —
> see the new general hitting-time/upcrossing utility below, which
> subsumes it as one instance of a general mechanism, not a separate
> build.)

**Process roadmap — Phase 1 scope, but chained internally (depends on
another Phase 1 item, not on Phases 2-5):**
> Compound Poisson Process, then Cramer-Lundberg and Sparre Andersen risk
> processes (all need `RenewalProcess` first); CIR and other diffusion-
> family SDEs (CEV, CKLS) (need `DiffusionProcess` first); Geometric
> Brownian Motion (an exact wrapper around `BrownianMotion`, which already
> exists in `dev` — no Phase 1 dependency at all, listed here only because
> it's often grouped with the diffusion-family items); Merton Jump-
> Diffusion (needs GBM + Compound Poisson); Aggregate loss process with
> actuarial severities (needs Compound Poisson + the relevant severity
> distributions, e.g. Burr/Lomax/Tweedie, themselves Phase 1 scope); G/G/1
> via Lindley's recursion, then M/G/1 and G/M/1 "for free" (all need
> `RenewalProcess`); continuous-time queue-length path for the G/G/1
> family (needs G/G/1); G/G/s multi-server queues (needs G/G/1); Cox
> process (needs Non-homogeneous Poisson thinning); Hawkes process (needs
> Compound Poisson and Non-homogeneous Poisson).
>
> **General hitting-time / upcrossing utility (new in the revised
> roadmap):** the roadmap's own key finding is that this needs **no new
> core plumbing at all** — `RV.apply(func)` already returns
> `RV(prob_space, lambda outcome: func(self.func(outcome)))`, and for a
> `RandomProcess`, `self.func(outcome)` is the whole realized path object,
> so a plain function taking a path and returning a hitting time already
> works today. The only work is writing `hitting_time`/`upcrossings`
> themselves, and Phase 1 dependency varies by tier:
> - **Tier A** (discrete-time/pure-jump: Random Walk, MarkovChain, AR/MA,
>   Renewal, Poisson/Compound Poisson, birth-death queues, CTMC) — easy
>   and exact (walk the known sequence until a crossing, bounded by
>   `max_time`/`max_steps`). No dependency beyond each underlying process
>   already being Phase 1 scope.
> - **Tier B** (Brownian motion/Gaussian processes) — moderate but exact,
>   via the reflection-principle bridge-crossing probability plus
>   bisection. Only strictly needs the existing (pre-Phase-1)
>   `GaussianProcessResult`, but the roadmap's own suggested build order
>   places this right after the `DiffusionProcess` merge (step 9 then 10)
>   so the bisection-to-localize technique can be built once and reused —
>   an ordering **preference**, not a hard dependency.
> - **Tier C** (general diffusions: `DiffusionProcess`, CIR, CEV, CKLS) —
>   moderate, approximate, and **hard-dependent** on Phase 1's
>   `DiffusionProcess` merge (reuses its Euler-Maruyama/mesh-refinement
>   approach and `tol` parameter directly).
> - **Upcrossings** (as an `InfiniteVector` sequence) — nearly free once
>   `hitting_time` exists for a given tier; depends on that tier only.
>
> None of the three tiers touch `*`/`**`, `Hierarchical`, or MCMC — this
> is Phase 1 scope end to end, just with internal tier-to-tier chaining
> like the rest of this bucket. Worth flagging as a forward-looking
> synergy (not a dependency): conditioning on a rare hitting event (e.g.
> "given ruin has not yet occurred") is exactly the kind of rare-event
> conditional query Phase 4's MCMC work targets, though `hitting_time`
> itself requires none of that machinery to ship.

**Process roadmap — soft-dependent on Phase 3 (buildable now via manual
composition; natural candidates to revisit once `Hierarchical` ships):**
> - **Mixed Poisson process** (Poisson-Gamma mixture → Negative Binomial
>   counts, listed in both the discrete-time and actuarial sections) —
>   "draw a random rate Lambda from a distribution once per path" is
>   exactly the prior→conditional pattern `Hierarchical` formalizes.
> - **Markov-modulated Poisson process** (regime-switching claim
>   intensity) — a CTMC driving a Poisson process's rate; same pattern,
>   continuous-time flavor.
> - **Hidden Markov Model** — state-dependent emission at each step is the
>   same prior→conditional shape, just per-step rather than once.
> - **Dirichlet process** — flagged in the roadmap doc itself as needing
>   its own architectural resolution (not time-indexed, likely its own
>   object rather than a `RandomProcess` subclass) *independent of Phases
>   2-5*; worth noting a real synergy though: a stick-breaking realization
>   is exactly a table of (atom, weight) pairs, which is what Phase 1's
>   `Empirical` already represents — the DP's `.draw()` could plausibly
>   return an `Empirical` instance once Phase 1 ships, regardless of
>   whether Phase 3 exists.
> None of these four are blocked — each has a working manual-generator-
> function path today (the same pattern `BetaBinomialProbspace` already
> demonstrates) — but building them via the general `>>`/`Hierarchical`
> syntax instead of a hand-rolled closure is a Phase 3 or later choice, not
> a Phase 1 one.

**Process roadmap — out of scope for this plan entirely (neither bucket):**
> Spatial point processes and Markov Decision Processes, both explicitly
> flagged out of scope by the roadmap doc itself (spatial needs a
> fundamentally different index set; MDPs are decision-theoretic, not
> purely stochastic). Nelson-Aalen/counting-process formulation is framed
> in the doc as more a data-analysis framework than a process to simulate.

**Distribution additions — Phase 1 scope, no dependency on Phases 2-5:**
> All of Tier 1 through Tier 4 follow the existing `Distribution
> (ProbabilitySpace)` wrapper pattern already used by `Bernoulli`/`Normal`/
> etc. in `distributions.py` — this is the large majority of the 55-item
> catalog (Weibull, Logistic, Laplace, Gompertz, Makeham, De Moivre, GEV,
> Noncentral t/Chi-square/F, GPD, Burr, Lomax, Inverse Gamma, Half-Normal,
> Half-Cauchy, Gumbel, Zipf, Negative Hypergeometric, Multivariate
> Hypergeometric, Triangular, Skew-Normal, Inverse Gaussian, Birnbaum-
> Saunders, Generalized Gamma, Erlang, Rice/Rician, Skellam, Fréchet,
> Log-Logistic, Von Mises, Nakagami, Lévy, Conway-Maxwell-Poisson,
> Yule-Simon, Discrete Weibull, Logarithmic, Johnson SU/SB, Singh-Maddala,
> Champernowne, Multivariate t, Wishart, Inverse Wishart, the multivariate
> Dirichlet distribution, Gaussian/t-Copula, Matrix Normal, Phase-Type
> distributions (via the existing `ContinuousTimeMarkovChain`), and the
> generic truncation/zero-modification wrapper). None of these touch
> `*`/`**`, `Hierarchical`, or MCMC.

**Distribution additions — Phase 1 scope, but a natural fit for Phase 1's
own `ContinuousProbabilitySpace` rather than a bespoke sampler:**
> Stable, Variance-Gamma, Normal-Inverse Gaussian, and Generalized
> Hyperbolic are all flagged "Hard... no closed-form density in general" —
> exactly the case `ContinuousProbabilitySpace`'s numerical inverse-CDF/
> rejection machinery exists for, once a numerically-evaluable density (or
> characteristic-function inversion) is available. Still pure Phase 1,
> just an implementation-strategy note, not a phase-ordering one.

**Distribution additions — soft-dependent on Phase 3:**
> - **Tweedie** (compound Poisson-Gamma) is itself a compound/hierarchical
>   construction (a Poisson-distributed number of Gamma jumps, summed) —
>   same soft link as Mixed Poisson process above.
> - **Beta-Binomial** is the Hierarchical design doc's flagship worked
>   example (`Beta(1,2) >> Binomial(10,x)`), but the *named distribution
>   itself* needs nothing from Phase 3 — `scipy.stats.betabinom` already
>   provides the closed-form marginal directly, so it's Phase 1 scope as a
>   distribution. The soft dependency is specifically on using it as a
>   `Hierarchical`-syntax teaching example, not on the distribution
>   existing.

**Rationale**
> The two documents are almost entirely additive-content work (new
> `Distribution` subclasses, new `...Result`/`...ProbabilitySpace`/
> `...Process` triples following the existing recipe) rather than
> architecture changes, so most of it doesn't touch the part of the
> codebase Phases 2-5 are about. The exceptions are conceptually
> interesting rather than blocking: a small number of processes and one
> distribution are *defined* as "prior, then a conditional distribution
> built from the prior's draw" — precisely what `Hierarchical` is meant to
> generalize — so building them by hand now and revisiting with the
> general mechanism later is expected, not a sign of wasted work.

**Alternatives Considered**
> Treating the soft-Phase-3-linked items as blocked until Phase 3 ships —
> rejected; the manual-generator-function pattern is already a supported,
> tested way to build these in Symbulate today, so gating them on Phase 3
> would be an artificial delay, not a real dependency.

---

## Decision: Phase 2 — Shared Product-Space Architecture

**Status:** Proposed — this is the one phase in the plan flagged as a
**forced** ordering dependency, not a preference.

**Decision**
> Before any code is written against `*`/`**` for either Hierarchical
> models or MCMC, jointly settle: (1) which operator carries hierarchical
> composition (`>>`, reused `*`, or `@` — see the comparison table in
> `symbulate_hierarchical_models_design.md` §8), and (2) the shape of
> `IndependentProductSpace(ProbabilitySpace)` (`.factors`, `.log_density()`)
> that `__mul__`/`__pow__` return when both operands are density-bearing,
> using a duck-typed eligibility check (`hasattr(x, "pdf") and
> callable(x.pdf)`), not `isinstance(x, Distribution)`.

**Rationale**
> `ContinuousProbabilitySpace` is declared as a `ProbabilitySpace` subclass,
> not a `Distribution` subclass, even though it exposes the same
> `.pdf`/`.cdf`/`.quantile` shape — an `isinstance` check would silently
> exclude exactly the custom-pdf models MCMC is most valuable for. Both the
> Hierarchical design (if it reuses `*`) and the MCMC design need a
> `ProbabilitySpace.__mul__` that can tell "independent join" apart from
> "conditional family" and/or "density-bearing product," and if either
of the two threads answers that question unilaterally, the other has to
> retrofit its own dispatch logic on top, or worse, define an incompatible
> second product-space class.

**Alternatives Considered**
> Let Hierarchical and MCMC each build their own product-space handling
> and reconcile later — rejected; explicitly the scenario this phase exists
> to prevent (see overall-phasing rationale above).

---

## Decision: Phase 3 — Hierarchical Models

**Status:** Proposed

**Decision**
> Build the core mechanism first — `HierarchicalProbabilitySpace`,
> `Hierarchical(prior, cond_func)`, the operator chosen in Phase 2, and the
> `TypeError` guard for a conditional function that doesn't return a
> `ProbabilitySpace`. Treat `Placeholder` (sub-phase 3b) and
> `AssumeHierarchical` (sub-phase 3c) as optional, separable layers on top,
> not required for 3a to ship.

**Rationale**
> The design doc itself frames `Placeholder` as "a closer visual match,"
> not a functional requirement — lambda already covers every case
> including branching logic Placeholder can't express. `AssumeHierarchical`
> is a pure escape hatch for already-written code, useful but independent
> of whether the core `>>`/`*`/`@` mechanism exists.

**Alternatives Considered**
> Shipping `Placeholder` in the same PR as the core mechanism — not
> rejected, just not required; team can decide at Phase 3 kickoff.

**Open within this phase:**
> - Lambda vs. `Placeholder` as the primarily-taught spelling.
> - `AssumeHierarchical`'s missing guard against being called twice on the
>   same prior RV (silently creates two unrelated joint spaces).

---

## Decision: Phase 4 — MCMC, First Algorithm

**Status:** Proposed

**Decision**
> Ship a first working `method="mcmc"` on `RVConditional`, restricted to
> models built purely from `*`/`**` (i.e., backed by
> `IndependentProductSpace`), with an explicit, clear error — not a silent
> wrong answer — on any hierarchical-space-backed model or any
> non-density-bearing space. The design doc's revision now names **two**
> legitimate candidates for that first algorithm, not one:
> - **Component-wise random-walk MH** (the original recommendation) — no
>   dependency beyond Phase 2's `IndependentProductSpace`; ships standalone.
> - **Single-site exact-conditional Gibbs** (new, §10) — every draw is an
>   exact conditional draw (enumeration for countable-support discrete
>   factors, `ContinuousProbabilitySpace`'s tier-1 inverse-CDF sampler for
>   continuous ones), with no step-size tuning and no wasted proposals —
>   but it **hard-depends** on `ContinuousProbabilitySpace`'s inverse-CDF
>   machinery being reusable as an actual subroutine, not just present for
>   a duck-type check.
>
> The design doc's own (still tentative) recommendation leans MH first,
> reasoning that MH can ship without waiting on `ContinuousProbabilitySpace`.
> **That reasoning is weaker in this plan than in the standalone design
> doc**, because `ContinuousProbabilitySpace` is already committed as Phase
> 1 scope here — by the time Phase 4 starts it will already exist, which
> removes the "avoid the wait" argument for defaulting to MH and
> strengthens the case for at least reconsidering Gibbs, per the design
> doc's own logic ("flag single-site exact-conditional Gibbs as the
> stronger candidate *if* that thread is being built anyway"). This is an
> open decision for the team, not settled (see Open Decisions).

**Rationale**
> The MCMC design doc's own revision (§9) walks back an earlier, overstated
> claim that MH "fixes" continuous conditioning. Conditioning on a
> continuous event (`abs(Y-y)<eps`) actually has three separable issues,
> and MH only helps with one:
> 1. **Defining a nonzero-probability target** via an `eps`-band is
>    unavoidable and algorithm-independent — no sampling method removes
>    this approximation, MH included.
> 2. **Finding the first valid state** costs exactly what ordinary
>    rejection sampling costs, regardless of algorithm — MH provides no
>    advantage here, only after this point.
> 3. **Staying in/exploring the band once inside** is the actual MH win,
>    but it requires tuning step size relative to `eps` and to the
>    constraint's local sensitivity (e.g. a component-wise sweep on
>    `Z=X+Y` shifts `Z` by exactly as much as the moved coordinate did) —
>    shrinking `eps` for accuracy shrinks the usable step size too, which
>    slows mixing. This tradeoff must be tuned, not assumed away.
>
> So Phase 4 should be scoped and described honestly: it fixes the
> rare-discrete-event problem unconditionally, but helps continuous
> conditioning only with issue 3, not issues 1-2. The component-wise
> acceptance ratio still depends on joint density factoring as a product
> of independent marginals — true today, false once hierarchical models
> exist — so declining cleanly on those remains load-bearing correctness,
> not just scope discipline.

**Alternatives Considered**
> "Sticky rejection" (repeat previous value instead of looping on reject) —
> cheaper, but doesn't fix continuous conditioning at all. Exact manifold
> sampling for linear constraints — best when it applies (removes all
> three `eps`-band issues at once for sums specifically, per §9), but
> doesn't generalize to arbitrary nonlinear events; the revision makes a
> stronger case for prioritizing this than the original doc did. Single-
> site exact-conditional Gibbs — a real competitor, not a discarded option
> (see Decision above); the team should pick between it and MH before
> Phase 4 starts, not default to MH by inertia. Self-normalized importance
> sampling — preserves i.i.d. samples, but needs weighted-`RVResults`
> support, a bigger change. Hamiltonian Monte Carlo / NUTS — considered
> and explicitly deferred, not rejected outright (see the new "Decision:
> HMC/NUTS — Explicitly Deferred" entry below). See the source doc's §4
> comparison table for the full analysis.

**Ordering note:** whether Phase 4 ships before or after Phase 3 is a
**preference** — if Hierarchical doesn't exist yet, the decline-check is
vacuously satisfied. But whichever of the two ships second must code its
check against the other's actual shipped class name, not a placeholder.

**Open within this phase:**
> - **Which algorithm ships first: component-wise MH vs. single-site
>   exact-conditional Gibbs** — not decided; see Decision above for the
>   asymmetric dependency on `ContinuousProbabilitySpace`.
> - Whether to prioritize "exact manifold sampling for linear constraints"
>   (e.g. sums) earlier than Phase 4's other "later/optional" items, since
>   it's the only approach that removes all three `eps`-band issues at
>   once for that common case.
> - Default `burn_in`/`thin` values, and whether they scale with `n`.
> - `step_scale` as one scalar vs. per-coordinate.
> - Whether diagnostics (`acceptance_rate`, ESS) live as `RVResults`
>   attributes or a separate object.
> - Final spelling of `method="mcmc"` vs. alternatives.

---

## Decision: Phase 5 — Markov-Blanket-Aware Sampler for Hierarchical Models

**Status:** Proposed — least specified of the five phases.

**Decision**
> Build single-site Metropolis-within-Gibbs for `HierarchicalProbabilitySpace`
> so that proposing a parent node correctly re-evaluates the child's
> density (`f_prior(x') · f_cond(y | x')`) instead of using Phase 4's
> cheaper cancellation, which is only valid for independent factors.

**Rationale**
> Flagged explicitly in the MCMC design (§7.1, §5) as "a real, separate
> feature, not a free extension" of Phase 4 — chained models
> (`A >> f >> g`) pass the whole accumulated tuple to each conditional
> function, so this also needs to handle O(depth)-per-proposal
> re-derivation in the worst case.

**Alternatives Considered**
> None yet — this phase starts from a problem statement in the source doc,
> not a design. No algorithm beyond "same idea as BUGS/JAGS" has been
> sketched.

**Open within this phase:** everything — this is genuinely unscoped,
unlike Phases 1-4, which all have a prototype or a detailed sketch behind
them.

---

## Decision: HMC/NUTS — Explicitly Deferred

**Status:** Proposed — a recorded non-goal, not a placeholder future phase.

**Decision**
> HMC/NUTS is not part of this build plan, now or as a scheduled later
> phase. The MCMC design doc's revision (§11) gives three concrete,
> codebase-specific blockers, not a generic "too advanced" judgment:
> 1. Symbulate models routinely mix discrete and continuous factors (e.g.
>    `Poisson(1) * Normal(0, 1)`), and HMC's leapfrog integrator
>    fundamentally requires a continuous, differentiable state space — a
>    mixed model would need HMC-for-continuous-factors plus a wholly
>    separate mechanism for discrete ones, not "HMC."
> 2. Conditioning events are typically hard indicators (e.g.
>    `abs(Z-1)<eps`) — a zero-gradient wall almost everywhere — which is
>    exactly the regime HMC's gradient-following advantage doesn't help
>    with, absent added reflection/refraction-at-boundary machinery.
> 3. Symbulate has no autodiff layer anywhere, and `Y.func` is an
>    arbitrary user-built Python function assembled via operator
>    overloading (`+`, `*`, `.apply(arbitrary_lambda)`, indexing, …) — an
>    autodiff layer (or adopting `jax`/`torch` as a dependency) is a
>    large, separate architectural commitment, not an add-on to this
>    feature.

**Rationale**
> All three blockers require independent, large prerequisite work — a
> continuous-only differentiable model subset, an autodiff layer, and a
> smooth (non-hard-indicator) conditioning API — none of which is planned
> anywhere in Phases 1-5. There's also a pedagogical argument specific to
> a teaching library: MH's accept/reject logic is readable line-by-line by
> an intro-stats student, while HMC/NUTS trade that transparency for
> efficiency in a regime (smooth, high-dimensional, continuous-only
> posteriors) that mostly isn't what Symbulate's own example set (small
> mixed discrete/continuous models, event-based conditioning) needs
> efficiency for.

**Alternatives Considered**
> Scoping HMC/NUTS as a distant "Phase 6" placeholder — rejected; without
> the three prerequisites even existing as planned work, a placeholder
> phase would imply a roadmap commitment that isn't real. Recording it as
> an explicit non-goal instead, to be revisited only if all three
> prerequisites become independently motivated (e.g. an autodiff layer
> requested for an unrelated reason).

---

## Decision: MA(q) Pre-Sample Shocks

**Status:** Implemented — `symbulate/time_series.py`,
`symbulate/tests/test_time_series.py`, demo in
`team/models-and-sim-design/ma_process_demo.ipynb`.

An MA(q) is defined over all time and is stationary by construction, so
every value has the same distribution. Simulating one has to start
somewhere, though, and `X[0]` needs the `q` shocks from *before* time 0.

**Decision**
> Simulate those `q` shocks. No parameter, no setting — the process is
> already running when the first value is read.

Leaving them out would make `X[0]` carry no echoes and `X[1]` only one, so
the first `q` values come out too small: for `coefs=[0.8, 0.5]`,
`Var(X[0])` is 1.00 instead of the correct 1.89, and the lag-2 correlation
measured from the start of a path reads 0.36 against a true 0.265. That
lands on precisely the thing an MA(q) is taught to show — the sharp cutoff
in correlation at lag `q` — so the transient is not a feature to preserve.

Unlike AR, getting it right is free and exact here: `q` extra draws, no
burn-in length to tune, and it works for any noise distribution.
`test_time_series.py::test_first_value_has_full_variance` pins it, and
fails if the pre-sample shocks are ever dropped.

This is also why `MA` takes no `initial` argument — see the next decision.

---

## Decision: One Name for a Process's Starting Condition — `initial`

**Status:** Agreed convention. New processes follow it from now on;
retrofitting the existing ones is a separate, coordinated PR (see below).

We currently spell "where does the process start" four different ways:
`initial_value` (Ornstein-Uhlenbeck, Brownian bridge, geometric Brownian
motion, random walk), `initial` (birth-death), `x0` (diffusion), and
`initial_dist` (Markov chains, where it means something different again — a
probability vector over states, not a value). Several processes that plainly
have a starting state take no such argument at all, including every M/M/\*
queue, whose parent `BirthDeathProcess` does.

**Decision**
> One parameter, named **`initial`**, accepting any of:
> - **a number** — every path starts there;
> - **a univariate distribution** — the starting value is drawn from it;
> - **a multivariate distribution** — for a process needing several
>   starting values at once, drawn jointly so their correlations are right;
> - **`"stationary"`** — the process's own long-run distribution, where
>   that has a closed form.
>
> A process with no state does not take the argument at all. `MA` is the
> case in point: it needs pre-sample *shocks*, not values, and there is
> nothing for a student to point at, so it is handled silently (see
> "Decision: MA(q) Pre-Sample Shocks").
>
> **Markov chains are carved out and keep `initial_dist`.** Unlike the
> others, that name is already accurate: what they take is a probability
> vector over states, which really is a distribution, so there is no
> mismatch to fix. It is also a *required* positional argument rather than
> an optional one, and renaming it to `initial` would invite
> `initial=0` meaning "start in state 0" — which is not supported and would
> be a functionality change, not a rename. Revisit if a `Categorical`
> distribution ever lands.

`initial` rather than `initial_value` or `initial_distribution` because it
is the only one of the three that stays honest across all of those forms — a
number is just a degenerate distribution, and `initial_distribution=0` reads
as badly as `initial_value=Normal(0, 1)`.

**Why a univariate distribution is not enough on its own.** An AR(p) needs
`p` starting values, and in the stationary distribution those values are
*correlated with each other*. Simulating an AR(2) with `phi1=0.5, phi2=0.3`
(stationary variance 2.244) from two i.i.d. `Normal(0, 1)` draws gives a
variance of 1.34 at time 0, reaching 2.25 only after about 40 steps — a
transient, just a different one. Drawing the pair jointly from the right
`MultivariateNormal` gives 2.24 immediately. So a univariate distribution is
exactly right for a one-value process (AR(1), geometric Brownian motion,
random walk) and under-specified beyond that, which is what the multivariate
form is for.

**Where `"stationary"` works.** Exactly, via Yule-Walker plus
`MultivariateNormal`, for Gaussian AR/ARMA. Not at all for non-Gaussian
noise, where no closed form exists, or for explosive coefficients, where no
stationary distribution exists — both should raise rather than quietly burn
in.

**Cost.** Cheap for the recursive processes (random walk, Markov chains,
AR, GARCH), which just draw the first value. Real work for the
Gaussian-process ones: `OrnsteinUhlenbeck` and `BrownianBridge` are built
from a `mean_func`/`cov_func` pair that assumes a deterministic start, and a
random starting value changes the covariance function itself.

**Migration.** New processes adopt `initial` immediately. Existing ones keep
their current spelling working as an accepted alias when they are retrofitted,
so notebooks and tests do not break. The retrofit touches all four groups'
code and should not ride along inside a feature branch.

---

## Open Decisions

- [ ] Who owns the `initial` retrofit, and when. Renaming
      `initial_value`/`x0`/`initial_dist` across Ornstein-Uhlenbeck,
      Brownian bridge, geometric Brownian motion, diffusion, and the Markov
      chains touches all four groups, so it wants its own PR rather than
      riding along inside a feature branch. Also whether every M/M/\* queue
      should regain the `initial` its `BirthDeathProcess` parent already has.
- [ ] Pandas as a new dependency for `LifeTable` — blocks Phase 1 merge
      until approved or the CSV-parsing is rewritten without it.
- [ ] Where test fixture data (e.g. `synthetic_soa_table.csv`) lives under
      `symbulate/tests/` — no existing precedent in the codebase.
- [ ] Whether `ContinuousProbabilitySpace` should subclass `Distribution`
      instead of `ProbabilitySpace` directly, to get `.spinner()`/plotting
      for free the way `Empirical` does.
- [ ] Hierarchical composition operator: `>>`, reused `*`, or `@` — the
      single decision Phase 2 exists to force.
- [ ] Whether `Empirical`'s integer-only outcome restriction gets the
      index-mapping fix in Phase 1 or ships as a documented limitation.
- [ ] MCMC's `burn_in`/`thin`/`step_scale` defaults and diagnostic
      placement (Phase 4).
- [ ] Which algorithm ships as Phase 4's first `method="mcmc"`:
      component-wise MH (no dependency beyond Phase 2) vs. single-site
      exact-conditional Gibbs (hard-depends on `ContinuousProbabilitySpace`'s
      inverse-CDF sampler as a subroutine — but that's already Phase 1
      scope in this plan, weakening the design doc's own case for
      defaulting to MH). See "Decision: Phase 4 — MCMC, First Algorithm."
- [ ] Whether "exact manifold sampling for linear constraints" (e.g. sums)
      should be pulled forward ahead of Phase 4's other "later/optional"
      items, since it's the only approach that removes all three
      `eps`-band issues at once for that common case.
- [ ] Phase 5's algorithm design — not yet sketched at all.
- [ ] Whether the four soft-Phase-3-linked process/distribution items
      (Mixed Poisson process, Markov-modulated Poisson process, Hidden
      Markov Model, Tweedie) should be built by hand in Phase 1 or held
      until Phase 3's `Hierarchical` mechanism ships — see "Decision:
      Phase 1 Scope — Process Roadmap & Distribution Additions."
- [ ] Dirichlet process's own architecture question (own object vs.
      `RandomProcess` subclass) — independent of Phases 2-5, but worth
      resolving alongside Phase 1's `Empirical`, which may be able to
      represent its stick-breaking realization directly.
- [ ] Whether hitting-time/upcrossing Tier B (Brownian motion/Gaussian
      processes) is built right after `DiffusionProcess` merges (to reuse
      its bisection technique, per the roadmap's suggested build order) or
      earlier/independently, since Tier B only strictly needs the
      pre-existing `GaussianProcessResult` — a preference, not a forced
      order.
