# Symbulate Random Process Roadmap

*Prioritized list of common random processes not yet built into Symbulate, with difficulty assessments*

## Scope and method

This list is grounded in the Symbulate dev branch source (random_processes.py, gaussian_process.py, poisson_process.py, markov_chains.py, index_sets.py, result.py, base.py), plus three related project conversations: implementing a general DiffusionProcess, adding a RenewalProcess, and cataloguing missing distributions. Confirmed built-in today: the generic RandomProcess, GaussianProcess / BrownianMotion, PoissonProcess, MarkovChain, and ContinuousTimeMarkovChain. Not built in: RandomWalk, ARMA family, Ornstein-Uhlenbeck, Geometric Brownian Motion, Compound Poisson, Renewal process, actuarial risk processes, and others detailed below.

Every built-in process follows the same three-piece recipe, and any new process should too, since it is what gives Symbulate's dual syntax:

- A ...Result class (one sample path) subclassing DiscreteTimeFunction / ContinuousTimeFunction / InfiniteVector.
- A ...ProbabilitySpace class whose draw() returns one ...Result.
- A convenience class subclassing both RandomProcess and RV, wiring the same probability space into both parent __init__s -- this is exactly how BrownianMotion supports both P = BrownianMotionProbabilitySpace(); X = RV(P) and X = BrownianMotion() directly.

## Infrastructure already prototyped in related work

Two pieces of groundwork from prior conversations change the difficulty ratings below and should be treated as largely done, pending a merge decision:

### DiffusionProcess (general SDE dX_t = mu(X_t,t)dt + sigma(X_t,t)dW_t)

- Implemented in the BrownianMotion/GaussianProcess style (subclasses ContinuousTimeFunction, RandomProcess, RV).
- Since a general diffusion is not Gaussian, there is no closed-form conditional distribution the way GaussianProcessResult has. Uses Euler-Maruyama forward simulation plus a locally-Gaussian Brownian-bridge correction, refined by bisection -- exact when mu, sigma are constant, a first-order-consistent approximation otherwise.
- Validated: reduces exactly to Brownian motion in the constant-coefficient case; matches known OU stationary variance; all 1514 existing tests still pass.
- Caveat to document clearly for students: this is an approximation controlled by a tol / mesh-width parameter, unlike the exact BrownianMotion / GaussianProcess conditioning.
- Consequence for this list: CIR, Vasicek, CEV/CKLS, and other finance/engineering SDEs move from Hard/Moderate down to Easy, since they become thin (mu, sigma) parameterizations of DiffusionProcess rather than new numerical work.

### RenewalProcess (general interarrival-time counting process)

- PoissonProcessResult already turns any sequence of interarrival times into a counting function N(t); nothing about it is Exponential-specific.
- ** inf (i.i.d. repeated draws) is defined generically on any ProbabilitySpace/Distribution, not just Exponential.
- So RenewalProcess is close to a copy of the Poisson process file with interarrival_dist substituted for rate. Prototyped and tested end-to-end against the real dev branch (.draw(), .sim(), indexing, error handling).
- Two open items flagged: validation is weaker than Poisson's (relies on the underlying distribution's own constructor), and nothing currently stops a distribution with negative support (e.g. Normal) from being passed in, which would break the nondecreasing-counting-function invariant.
- Consequence for this list: Renewal Process itself is essentially done. Compound Poisson, and the actuarial risk processes in Section 5, inherit a head start because they reuse this same interarrival-time machinery.

## Discrete-time processes

| Process | Priority | Why it matters | Difficulty & implementation notes |
|---|---|---|---|
| Random Walk (simple +/-1, or general i.i.d. step) | High | First process in almost every intro course; gambler's ruin, CLT intuition. | Very easy. Same skeleton as PoissonProcessResult: draw steps = (StepDist ** inf).draw(), then X[n] = cumulative sum, cached like MarkovChainResult caches states. |
| Renewal Process | High (near-done) | Generalizes Poisson; queueing and reliability courses. | Trivial -- already prototyped as a near copy of PoissonProcessResult/ProbabilitySpace with interarrival_dist swapped in for rate. Needs a negative-support guard and stronger parameter validation before merge. |
| MA(q) process | Medium-High | Standard time-series model (stats/econometrics). | Easy. X[n] is a finite weighted sum of the last q+1 terms of an i.i.d. noise InfiniteVector -- no recursion needed. |
| AR(p) / ARMA(p,q) | Medium-High | The other standard time-series model; ubiquitous in econometrics/finance. | Moderate. Needs a recursive InfiniteVector (generate up through n, caching as you go, like MarkovChainResult._func) since X[n] depends on p prior values. Requires a real design decision on initial conditions (start at 0 vs. draw from the stationary distribution). |
| Galton-Watson / branching process | Medium | Standard topic in discrete probability, population genetics, epidemics. | Moderate. Recursive like AR: X[n+1] = sum of X[n] i.i.d. offspring draws. Same InfiniteVector caching pattern, slightly more bookkeeping due to variable-length sums per step. |
| GARCH(1,1) / ARCH | Medium | The standard volatility model in financial engineering / stats courses. | Moderate. Two coupled recursive sequences (variance and value) instead of one; reuses the AR caching pattern once AR exists. |
| Mixed Poisson process (e.g., Poisson-Gamma mixture, giving Negative-Binomial counts) | Medium (actuarial) | Standard overdispersion model for claim counts in Klugman-style loss models. | Moderate. Draw a random rate Lambda from a distribution once per path, then feed it into the existing Poisson/Renewal machinery. Mostly composition once RenewalProcess exists. |
| Hidden Markov Model | Medium | Extremely common across stats, ML, bioinformatics, engineering. | Moderate-hard. Build on existing MarkovChain for the hidden state, then draw a state-dependent emission at each step. New Result class, plus validation of emission distributions per state. |
| Bernoulli process (named convenience) | Low | Pedagogically common as a named process. | Already fully expressible today as Bernoulli(p) ** inf; only value is a friendlier name. |

## Continuous-time processes

| Process | Priority | Why it matters | Difficulty & implementation notes |
|---|---|---|---|
| Ornstein-Uhlenbeck (mean-reverting) | High | Finance (Vasicek short-rate model), physics; standard second example after Brownian motion. | Very easy either way. It is Gaussian, so a GaussianProcessProbabilitySpace subclass with the closed-form OU mean_func/cov_func (same trick as BrownianMotion) gives an exact process. Also expressible as a DiffusionProcess parameterization if the approximate route is preferred. |
| Geometric Brownian Motion | High | The textbook stock-price model; the single most commonly requested finance addition. | Easy. Not Gaussian itself, so it should not go through GaussianProcessProbabilitySpace directly; instead wrap a BrownianMotionResult with S0*exp((mu - sigma^2/2)t + sigma*W(t)), using the pointwise Arithmetic that TimeFunction already supports. Preferred over routing through DiffusionProcess, since this way it stays exact rather than an Euler-Maruyama approximation. |
| Brownian Bridge | Medium-High | Standard example in probability courses (KS statistic, etc.). | Very easy. Another GaussianProcessProbabilitySpace subclass, just a different cov_func: min(s,t) - s*t/T. |
| Compound Poisson Process | High (actuarial + finance) | Insurance/finance risk models; core building block of aggregate loss models. | Easy-moderate, and benefits directly from the RenewalProcess groundwork: reuse the interarrival-time machinery, add a jump-size draw at each arrival, track the cumulative sum. |
| Fractional Brownian Motion | Medium | Increasingly common in finance/stats electives (long-range dependence). | Easy to add (just another cov_func plug-in: 1/2*(\|s\|^2H+\|t\|^2H-\|s-t\|^2H)), but flag a performance caveat: GaussianProcessResult conditions via np.linalg.solve on a growing covariance matrix each time a new point is touched, which can get slow for paths with many evaluated points under long-range dependence. |
| Cox-Ingersoll-Ross (CIR) | Medium-High (actuarial/finance) | The other classic finance short-rate model, always taught alongside Vasicek/OU. | Now easy given DiffusionProcess: mu(x,t)=kappa*(theta-x), sigma(x,t)=sigma*sqrt(x) plugged directly in. Note this makes it an approximation (Euler-Maruyama + bridge correction), not the exact noncentral-chi-square simulation some packages use -- worth a docstring caveat. |
| Merton Jump-Diffusion (GBM + compound Poisson jumps) | Medium | Standard in financial engineering courses. | Moderate, and cheap once GBM and Compound Poisson both exist -- mostly composition of the two. |
| Non-homogeneous Poisson Process (time-varying rate) | Medium | Queueing / reliability courses. | Moderate-hard. Needs thinning or time-change simulation instead of the fixed-rate exponential-interarrival trick -- genuine new algorithmic work, not a parameter swap. |
| Other diffusion-family SDEs (CEV, CKLS, and similar) | Low-Medium | Occasionally seen in grad finance electives. | Easy given DiffusionProcess -- just supply the (mu, sigma) pair, same as CIR. |
| Telegraph / random telegraph signal process | Low-Medium | Signal processing / engineering courses. | Moderate, but largely already covered: it is a continuous-time two-state Markov chain, so ContinuousTimeMarkovChain with n=2 already handles it. Mainly a documentation/example gap rather than new code. |

## Actuarial science processes (new section)

The missing-distributions review flagged that Symbulate has essentially no actuarial-specific content (Gompertz/Makeham/De Moivre mortality laws, Tweedie, Burr, Lomax for loss severity, and truncated/modified discrete distributions from the Klugman Loss Models framework are all absent). The process side has a parallel, equally real gap. If actuarial coursework (SOA Exam FAM/STAM/LTAM-style content) is in scope for Symbulate's audience, these should be treated as high priority, not niche.

| Process | Priority | Why it matters | Difficulty & implementation notes |
|---|---|---|---|
| Cramer-Lundberg / classical risk (surplus) process: U(t) = u + c*t - S(t), S(t) a compound Poisson aggregate-claims process | High (actuarial) | The foundational model of ruin theory; near-universal starting point for the actuarial risk subfield, the same way Bernoulli/Binomial are for discrete probability. | Easy once Compound Poisson exists: it is Compound Poisson plus a deterministic linear drift term. Benefits from the same RenewalProcess/PoissonProcess machinery. |
| Sparre Andersen (renewal) risk model -- Cramer-Lundberg generalized to renewal (non-Poisson) claim arrivals | Medium-High (actuarial) | The standard generalization taught right after Cramer-Lundberg in ruin theory. | Now trivial given RenewalProcess is already prototyped: swap the Poisson arrival process for a RenewalProcess and reuse the same surplus-process wrapper as Cramer-Lundberg. |
| Aggregate loss process with actuarial severity distributions (compound Poisson / renewal claim counts with Burr, Lomax, or Tweedie-type severities) | High (actuarial) | Standard in P&C ratemaking and loss modeling (Klugman Loss Models is the SOA reference text). | Easy once Compound Poisson exists and the severity distributions themselves are added -- this is really a distributions-side gap (see missing-distributions review) wearing a process-side hat; the process code itself needs no new work beyond accepting an arbitrary severity RV, which it already does by design. |
| Mixed Poisson process (Poisson-Gamma mixture -> Negative Binomial claim counts) | Medium (actuarial) | Standard overdispersion model for claim counts in Klugman-style frameworks. | Moderate. Same as the discrete-time entry above -- draw a random claim rate per path, then run the existing Poisson/Renewal machinery conditional on it. |
| Markov-modulated Poisson process (regime-switching claim intensity) | Low-Medium (actuarial) | Advanced ruin theory / environment-dependent claims, more of a grad-level topic. | Moderate-hard. Composition of ContinuousTimeMarkovChain (the regime) driving a Poisson/Compound-Poisson process (the claims) with regime-dependent rate; genuinely new coordination logic between the two existing processes. |
| Multi-state life-insurance / disability models (healthy - disabled - dead, etc.) | Low-Medium (actuarial) | Standard in life contingencies / multi-state actuarial modeling courses. | Already achievable today via ContinuousTimeMarkovChain with actuarially-labeled states; mostly a documentation/worked-example gap (like the telegraph process) rather than new code. |
| First-passage / ruin-time utility (e.g., time until a process first crosses a threshold) | Medium (actuarial, cross-cutting) | The central quantity of interest in ruin theory (probability and time of ruin), also useful for Brownian motion, GBM, and diffusions generally. | Worth scoping as a general utility method on RandomProcess / TimeFunction rather than a separate process class -- high leverage since it would apply to the surplus process, Brownian motion, GBM, and CIR/OU alike. |

## Queueing models

Queueing theory splits cleanly along the same fault line as the rest of this list. Markovian queues (Poisson arrivals, exponential service) are birth-death continuous-time Markov chains and are already almost free given ContinuousTimeMarkovChain -- the missing piece is a convenience wrapper that builds the generator matrix from (arrival rate, service rate, servers, capacity, population) rather than asking the user to hand-construct it. General-service queues (M/G/1, G/M/1, G/G/1) are not Markovian in continuous state and need a genuinely new recursive process based on Lindley's recursion, which leans directly on the RenewalProcess groundwork for the arrival side.

| Process | Priority | Why it matters | Difficulty & implementation notes |
|---|---|---|---|
| M/M/1 (single server) | High | The canonical first example in every queueing / operations-research / engineering course. | Very easy. A birth-death CTMC: birth rate lambda in every state, death rate mu in states n>=1. A direct parameterization of the existing ContinuousTimeMarkovChain; needs a thin generator-matrix-building wrapper, not new simulation logic. |
| M/M/s (multi-server) | High | Standard generalization taught immediately after M/M/1 (call centers, banks, service systems). | Very easy, same birth-death CTMC idea with death rate min(n,s)*mu. |
| M/M/s/K (finite buffer / capacity, blocking) | Medium-High | Standard finite-capacity variant; blocking probability is a core OR/telecom quantity. | Very easy -- actually an exact finite chain (state space capped at K), not a truncated approximation, so no accuracy caveat is needed. |
| M/M/s/s (Erlang loss system / Erlang B) | Medium (telecom/engineering) | Classic telecom trunking model (Erlang B blocking formula). | Very easy, special case of M/M/s/K with K = s (no waiting room). |
| M/M/s/K/N (finite population -- "machine repair" model) | Medium | Standard queueing-course staple (repairman problem). | Easy, still a birth-death CTMC, but with a state-dependent arrival rate (N-n)*lambda as well as state-dependent service rate; slightly more bookkeeping in the generator-matrix builder. |
| M/M/infinity (infinite-server queue) | Low-Medium | Self-service systems; also used to model transient/short-lived populations. | Trivial in principle (birth rate lambda, death rate n*mu), though the generator matrix still needs a practical truncation at some large N for simulation, same as any unbounded-state CTMC. |
| G/G/1 (general interarrival and service distributions) | High | The realistic queueing model; famous for having no closed-form solution, which is exactly why simulation is the standard pedagogical tool for it. | Moderate, and the natural general case to build: implement Lindley's recursion W[n+1] = max(W[n] + S[n] - A[n], 0) as a new recursive InfiniteVector (same caching pattern as the AR/branching-process entries), driven by a RenewalProcess for arrivals (any interarrival distribution) and an i.i.d. service-time RV. M/G/1 and G/M/1 fall out for free as special cases with one side set to Exponential. |
| M/G/1 (Poisson arrivals, general service) -- Pollaczek-Khinchine model | High | Classic model precisely because most real services are not exponential; standard alongside M/M/1. | Comes for free once G/G/1's Lindley recursion exists -- just fix the arrival RenewalProcess to Exponential. |
| G/M/1 (general arrivals, exponential service) | Medium | Taught alongside M/G/1 as the other half of the relaxation. | Comes for free once G/G/1 exists -- fix the service RV to Exponential. |
| Continuous-time queue-length path N(t) for the G/G/1 family | Medium | Matches the modeling style of M/M/s and PoissonProcess (an actual function of continuous time, not just a discrete waiting-time sequence). | Moderate. Once arrival times (RenewalProcess) and departure times (from the Lindley recursion) are both known, N(t) = arrivals before t minus departures before t -- structurally identical to PoissonProcessResult's counting-function logic, just merging two event streams instead of one. |
| G/G/s (multi-server general-service queue) | Low-Medium | More realistic than G/G/1, but grad-level operations-research territory. | Harder. Needs order statistics over s servers' completion times -- a genuine generalization of the single-server Lindley recursion, not a parameter swap. |
| Queueing networks (Jackson networks, tandem queues) | Low | Grad-level operations-research topic. | Hard. Needs coordinated multi-node event simulation and routing logic -- substantial new architecture, lower priority given Symbulate's mostly undergrad/intro-grad audience. |
| Little's-Law-style derived summaries (mean number in system, mean wait, utilization) | Medium (cross-cutting) | The standard quantities every queueing course asks students to compute from a simulation. | Worth scoping as generic utility methods over any queue process rather than new process classes -- same idea as the first-passage-time utility flagged for actuarial ruin theory. |

## Survival analysis and point-process extensions

Survival analysis mostly reuses machinery already on this list (Poisson/renewal processes, continuous-time Markov chains) rather than needing a new category, with one real exception: the Cox process, which is a natural unifying mechanism worth building once rather than piecemeal.

| Process | Priority | Why it matters | Difficulty & implementation notes |
|---|---|---|---|
| Cox process (doubly stochastic Poisson process) | High | A Poisson process whose intensity is itself random; unifies several items on this list rather than being a one-off. | Moderate. Build on non-homogeneous Poisson process thinning: first draw a random intensity path from any other process, then thin against that specific realization. This single mechanism subsumes the Markov-modulated Poisson process flagged in the actuarial section (intensity driven by a ContinuousTimeMarkovChain is just one special case). |
| Multi-state / competing-risks survival models (illness-death, recurrent events) | Medium | Standard framing in biostatistics and actuarial life-contingencies courses. | Already achievable today via ContinuousTimeMarkovChain with survival-labeled states; mostly a documentation/worked-example gap, the same situation as the multi-state actuarial models and the telegraph process. |
| Nelson-Aalen / counting-process (Aalen multiplicative intensity) formulation of survival data | Low-Medium | How hazard and martingale theory are actually presented in graduate biostatistics. | More a theoretical framework for analyzing data than a process to simulate; lower priority unless biostatistics is explicitly a target audience. |

## Bayesian nonparametric processes

This category is architecturally different from everything else on the list: the Dirichlet process is not indexed by time at all, it is a random probability distribution. That means it likely cannot simply subclass RandomProcess the way every other entry does -- it is worth flagging as a design question before implementation, not just a difficulty rating. Its two more elementary, sequential relatives (the Polya urn and the Chinese Restaurant Process) fit the existing recursive-caching architecture cleanly and are worth prioritizing first.

| Process | Priority | Why it matters | Difficulty & implementation notes |
|---|---|---|---|
| Polya urn scheme | High | Classic in both classical probability and Bayesian courses; directly motivates exchangeability, CRP, and DP. | Easy. A simple recursive update to an urn's composition; fits the same recursive-InfiniteVector caching pattern already used for Markov chains and branching processes. |
| Chinese Restaurant Process | High | Commonly taught by name as the sequential, exchangeable-partition view of the Dirichlet process; the standard entry point in Bayesian nonparametrics courses. | Easy-moderate. A recursive InfiniteVector over customers 1, 2, 3, ..., same caching pattern as the Markov chain / branching process code. Should be prioritized above the Dirichlet process itself since it is nearly free once that pattern exists. |
| Dirichlet process | High (Bayesian nonparametrics) | The central object in any Bayesian nonparametrics graduate course. | Harder, and architecturally distinct. A DP realization (via stick-breaking) is a countably infinite set of (weight, atom) pairs, not a function of time -- it likely needs its own object whose .draw() returns a random discrete distribution, rather than fitting the X[t] time-indexing pattern. Worth resolving as a design question first. One practical note: stick-breaking only needs Beta(1, alpha) draws, which Symbulate already has -- it does not require the (currently missing) multivariate Dirichlet distribution, so the distribution-side prerequisite is smaller than the name suggests. |
| Pitman-Yor process, Indian Buffet process, Beta process | Low | Real, but grad-level Bayesian ML topics with a narrower audience. | Same architectural issue as the Dirichlet process (not time-indexed), compounded by a more specialized audience; lower priority. |

## Other commonly requested processes across disciplines

A few processes came up on a broader scan across textbooks and course types that do not fit neatly into the sections above.

| Process | Priority | Why it matters | Difficulty & implementation notes |
|---|---|---|---|
| Birth-death process / Yule (pure-birth) process | High | The other canonical example, alongside the Poisson process, in nearly every introductory stochastic-processes textbook (e.g. Ross); also the direct ancestor of the queueing birth-death chains in Section 6. | Very easy. The simplest possible case of the birth-death generator-matrix wrapper already proposed for the queueing section -- mainly needs to be surfaced and documented as its own named entry and worked example, not new code. |
| Stochastic epidemic models (SIR/SEIR as a continuous-time Markov chain) | High (current relevance) | Increasingly common in statistics and biology curricula. | Easy-moderate. Same generator-matrix-wrapper mechanism as the queueing and birth-death entries, just over (S,I,R) counts instead of queue length. The state space grows combinatorially (pairs of S,I counts), so the wrapper needs to build a larger matrix, but ContinuousTimeMarkovChain itself needs no changes. |
| Hawkes process (self-exciting point process) | Medium-High | Increasingly taught in statistics, finance, and ML electives (high-frequency trading, earthquake and social-event data). | Moderate-hard. Intensity depends on the process's own past events via a decaying kernel, so it needs a genuinely new recursive-intensity plus thinning implementation (Ogata's algorithm) -- pairs naturally with the Cox process work in Section 7, but is real new machinery, not a parameterization of something existing. |
| Shot noise / filtered Poisson process | Medium | Common in electrical engineering / signal processing and physics courses. | Easy given Compound Poisson: identical structure, but with a general impulse-response function applied at each Poisson arrival instead of a scalar jump size. |
| Weibull / power-law process (reliability growth, Crow-AMSAA model) | Medium (reliability engineering) | The textbook-named special case of the non-homogeneous Poisson process used throughout reliability engineering. | Already covered structurally once non-homogeneous Poisson process simulation exists; worth exposing and naming explicitly since that is how reliability-engineering courses actually present it, even though no new simulation logic is required. |

## Explicitly out of scope (flagged for completeness, not prioritized)

- Spatial point processes (2D/3D Poisson process for ecology, epidemiology, disease mapping) -- these need a fundamentally different index set (a spatial region rather than time), which is a genuine architectural change rather than a new process class fitting the existing RandomProcess pattern. Not recommended unless spatial statistics becomes an explicit goal for Symbulate.
- Markov Decision Processes -- these are decision-theoretic objects (states, actions, and rewards under a policy), not purely stochastic processes. This is a genuine category difference, not an oversight, and is likely out of scope for a probability-simulation package like Symbulate.

## Cross-reference: distribution gaps that block or motivate the above

From the separate missing-distributions review, the following are the distribution-side prerequisites most relevant to the processes above, rather than a full repeat of that list:

- Tweedie (compound Poisson-Gamma) -- the standard severity/claim-modeling distribution for the aggregate loss process row above; very high value if any actuarial/GLM content is targeted.
- Burr (Burr XII) and Lomax (Pareto Type II) -- standard heavy-tailed severity distributions in Klugman's Loss Models, needed for realistic Compound Poisson / Cramer-Lundberg examples.
- Gompertz, Makeham, De Moivre -- mortality laws needed for realistic multi-state life-insurance examples on top of ContinuousTimeMarkovChain.
- Generalized Pareto Distribution (GPD) -- peaks-over-threshold severity modeling, pairs with Compound Poisson for excess-of-loss reinsurance examples.
- A generic truncation / zero-modification wrapper for discrete distributions -- flagged in the missing-distributions review as arguably higher leverage than any single new distribution, since Klugman-style actuarial courses need truncated/modified Poisson, Binomial, Negative Binomial, and Geometric constantly, and a wrapper multiplies the value of distributions Symbulate already has.
- Beta distribution (already present) is sufficient for Chinese-Restaurant-Process / stick-breaking style Dirichlet process construction -- the multivariate Dirichlet distribution flagged as missing is a smaller prerequisite for Bayesian nonparametric processes than it first appears.

## Suggested build order

Revised to weave in survival/point-process, Bayesian nonparametric, and cross-discipline additions alongside the earlier groundwork:

- 1. Random Walk -- highest pedagogical priority, cheapest to build, most conspicuous current gap.
- 2. Merge / finish RenewalProcess -- close to done; resolve the negative-support validation gap first.
- 3. Ornstein-Uhlenbeck and Brownian Bridge -- essentially free, same GaussianProcessProbabilitySpace trick as BrownianMotion.
- 4. Geometric Brownian Motion -- thin exact wrapper around BrownianMotion; high finance-course demand.
- 5. Compound Poisson Process, then Cramer-Lundberg and Sparre Andersen risk processes -- closes the entire core actuarial ruin-theory gap in one connected pass, riding on RenewalProcess.
- 6. A birth-death-generator-matrix convenience wrapper over ContinuousTimeMarkovChain -- build this once to cover M/M/1, M/M/s, M/M/s/K, M/M/s/s (Erlang loss), M/M/s/K/N (machine repair), M/M/infinity, the plain birth-death/Yule process, and SIR/SEIR epidemic models in a single pass, since all of them are the same underlying mechanism with different rate functions and state spaces.
- 7. Polya urn scheme and Chinese Restaurant Process -- cheap, high pedagogical value, reuse the existing recursive-caching pattern; do these before attempting the Dirichlet process itself.
- 8. Merge / finish DiffusionProcess -- document the approximation caveat clearly; then CIR and other SDEs (CEV, CKLS) become near-free parameterizations.
- 9. MA(q), then AR(p)/ARMA(p,q) -- MA is easy standalone; AR needs the recursive-caching design decision, ARMA combines both.
- 10. G/G/1 queue via Lindley's recursion (built on RenewalProcess for arrivals) -- M/G/1 and G/M/1 fall out for free as special cases.
- 11. Non-homogeneous Poisson process -- needed both for the Weibull/power-law reliability framing and as the direct prerequisite for the Cox process and Hawkes process below.
- 12. Cox process (built on non-homogeneous Poisson thinning) -- also subsumes the Markov-modulated Poisson process from the actuarial section as a special case, so build once rather than twice.
- 13. Mixed Poisson process, GARCH/ARCH, Galton-Watson, Hidden Markov Model -- all moderate, all benefit from the machinery built in steps 1-12.
- 14. Fractional Brownian Motion -- cheap to add, note the covariance-matrix performance caveat in docs.
- 15. Hawkes process and shot-noise/filtered Poisson process -- build once Compound Poisson and non-homogeneous Poisson exist.
- 16. Dirichlet process -- resolve the architectural question (own object vs. RandomProcess subclass) before implementing; genuinely new design work, not just new code.
- 17. Multi-state survival/actuarial CTMC worked examples, continuous-time N(t) for G/G/1, first-passage-time utility, Merton jump-diffusion -- documentation-and-examples gaps or moderate new work; lower urgency.
- 18. G/G/s multi-server queues, queueing networks (Jackson networks), Pitman-Yor/Indian Buffet/Beta process -- lower-priority grad-level topics with narrower audiences; revisit once the rest of the roadmap is in place.