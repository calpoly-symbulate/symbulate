# Design Notes: Summary-Statistic Methods & Plotting for Multivariate Distributions

## Scope

This document outlines the approach for adding `mean()`, `var()`, `sd()`,
`cov()`, `corr()`, `cdf()`, and `draw()` to Symbulate's (non-matrix-valued)
multivariate distributions, plus a design for what `Distribution.plot()`
should do for them. It covers the seven distributions requested:

- Multinomial
- MultivariateNormal
- Dirichlet
- Multivariate t (Student)
- Hotelling's T-squared
- Multivariate Hypergeometric
- Negative Multinomial

No implementation code yet — this is the plan.

---

## 1. Shared architecture: a `MultivariateDistribution` base class

Every one of these seven distributions needs the same shape of API
(`mean()`, `var()`, `sd()`, `cov()`, `corr()`, `cdf()`, `draw()`, a
disabled-or-redesigned `plot()`, and `__pow__`). Rather than writing all
seven from scratch the way `MultivariateNormal` currently does (see the
earlier discussion — it bypasses the base `Distribution` class entirely),
it's worth introducing an intermediate class:

```
Distribution (existing base class, scalar-oriented)
  └── MultivariateDistribution (new — shared multivariate machinery)
        ├── MultivariateNormal
        ├── Dirichlet
        ├── MultivariateT
        ├── Multinomial
        ├── MultivariateHypergeometric
        ├── NegativeMultinomial
        └── HotellingT2
```

`MultivariateDistribution` would define, once:
- `sd()` — always `Vector(np.sqrt(np.diag(self.cov())))`, derivable from
  `cov()` for every subclass, so this never needs to be reimplemented.
- `corr()` — always derivable from `cov()`:
  `self.cov() / np.outer(sd, sd)`, same for every subclass.
- The disabled/placeholder `plot()` from the current code, until the
  new plotting design (Section 8) replaces it.
- `__pow__` — the current hand-rolled "loop `draw()` and pack into a
  `Vector`" logic, which is identical across `MultivariateNormal` and
  `Multinomial` already and would be identical for the rest too.

Each subclass then only needs to supply `mean()`, `cov()`, `draw()`, and
`cdf()` (where feasible) — the parts that are genuinely
distribution-specific. This cuts the amount of new code roughly in half
across the seven, and guarantees `sd()`/`corr()`/`__pow__` behave
identically everywhere rather than being reimplemented (and potentially
drifting) seven separate times.

**Naming/attribute fix carried over from the `MultivariateNormal`
discussion:** every subclass stores its raw parameters as private
attributes (`self._mean`, `self._cov`, etc.), never as `self.mean`/
`self.cov`, so the public `mean()`/`cov()` methods don't collide with
stored data the way the current `MultivariateNormal.mean` attribute does.

---

## 2. What scipy already gives us for free (verified directly, not from memory)

This changes the effort estimate a lot, so it's worth listing precisely
what was checked:

| Distribution | scipy object | `.mean()` | `.cov()` | `.var()` | `.cdf()` |
|---|---|---|---|---|---|
| MultivariateNormal | `multivariate_normal` | Yes (property) | Yes (property) | — (derive from cov) | **Yes** |
| Dirichlet | `dirichlet` | Yes (method) | Yes (method) | Yes (method) | No |
| Multivariate t | `multivariate_t` | No | No | No | **Yes** |
| Multinomial | `multinomial` | Yes (method) | Yes (method) | — (derive from cov) | No |
| Multivariate Hypergeometric | `multivariate_hypergeom` | Yes (method) | Yes (method) | Yes (method) | No |
| Negative Multinomial | *(not in scipy)* | No | No | No | No |
| Hotelling's T² | *(not in scipy)* | No | No | No | No |

Concretely verified, e.g.: `stats.multinomial(n=10, p=[.5,.3,.2]).mean()`
returns `[5, 3, 2]` and `.cov()` returns the exact 3×3 covariance matrix —
so for **Multinomial, Dirichlet, and Multivariate Hypergeometric**, `mean()`
and `cov()` are direct pass-throughs to scipy, not something we compute
by hand. That's a materially smaller lift than the earlier "Medium"
integration-effort rating for these assumed.

---

## 3. Multinomial

- **`mean()` / `cov()` / `var()`**: direct pass-through to
  `stats.multinomial(n, p).mean()` / `.cov()` / (derived `var()` from
  `cov()` diagonal, or scipy's own if it turns out to expose one).
- **`draw()`**: unchanged from the current implementation
  (`rng.multinomial`).
- **`cdf()`**: this is the one genuine open problem for Multinomial.
  There's no closed form for the joint CDF of a multinomial vector —
  scipy doesn't provide one, and computing
  `P(X_1 ≤ x_1, ..., X_k ≤ x_k)` exactly means summing the joint pmf
  over a potentially large discrete region (all count-vectors
  component-wise ≤ the query point and summing to `n`). For small `n`
  and few categories this is a finite, doable brute-force sum; for
  larger `n` it becomes computationally expensive. Two options:
  1. Exact brute-force summation, with a documented "may be slow for
     large `n`" caveat, or
  2. A Monte Carlo–estimated `cdf()` (simulate many draws, report the
     empirical proportion satisfying the condition) as a fallback when
     the exact sum would be too large — mirroring how scipy itself
     falls back to randomized quadrature for `multivariate_normal.cdf()`.
  Recommendation: implement the exact brute-force sum with a size
  cutoff (e.g., total outcome count under some threshold), and raise or
  fall back to simulation above that.

---

## 4. MultivariateNormal

Already designed and tested in the previous discussion — `mean()`,
`var()`, `cov()`, `corr()`, and `cdf()` all confirmed working against
scipy's `multivariate_normal`, including an exact match against ground
truth on a correlated 3-D case. The one carried-over caveat: scipy's
`cdf()` uses randomized quasi-Monte Carlo integration internally, so two
calls with identical inputs can differ in the 5th–6th decimal place.
Worth deciding whether to pin scipy's `random_state` internally for
reproducible classroom demos.

---

## 5. Dirichlet

- **`mean()` / `cov()` / `var()`**: direct pass-through to
  `stats.dirichlet(alpha).mean()` / `.cov()` / `.var()` — confirmed all
  three exist and return correct values (e.g., `mean()` for
  `alpha=[2,3,5]` gives `[0.2, 0.3, 0.5]`, matching the textbook formula
  `alpha_i / sum(alpha)`).
- **`draw()`**: straightforward, via `stats.dirichlet(alpha).rvs()`.
- **`cdf()`**: no scipy support, and no simple closed form in general —
  the Dirichlet's CDF requires evaluating a multivariate Beta-function
  integral over a simplex region, which doesn't reduce to anything
  scipy exposes directly. Recommendation: **skip `cdf()` for Dirichlet**,
  or implement it only via numerical integration (e.g., `scipy.integrate`)
  with a clearly documented "approximate, may be slow" caveat, since
  this isn't a distribution where students typically need a joint CDF
  the way they do for MultivariateNormal.

---

## 6. Multivariate t (Student)

- **`cdf()`**: confirmed present and working in scipy
  (`stats.multivariate_t(loc, shape, df).cdf(x)`) — same numerical
  quasi-Monte Carlo caveat as MultivariateNormal's `cdf()` applies here
  too.
- **`mean()` / `cov()`**: **not** exposed by scipy's frozen object,
  unlike MultivariateNormal. These need to be derived analytically from
  the standard multivariate-t moment formulas:
  - `mean() = loc`, but **only defined for `df > 1`** — for `df ≤ 1` the
    mean doesn't exist (matches univariate Student's t, where the mean
    is undefined for 1 degree of freedom).
  - `cov() = shape * df / (df - 2)`, but **only defined for `df > 2`** —
    for `df ≤ 2` the variance is infinite/undefined.
  - This means `mean()` and `cov()` need explicit guard conditions that
    raise a clear, informative error (rather than silently returning a
    wrong number) when `df` is too small — a genuinely new kind of
    error-handling not needed by any of the other six distributions,
    since none of the others have parameter ranges where the moments
    are formally undefined.
- **`draw()`**: via `stats.multivariate_t(loc, shape, df).rvs()`.

---

## 7. Hotelling's T-squared

This is architecturally different from the other six: **Hotelling's T²
isn't a distribution scipy has as its own object**, and it isn't
typically the primary quantity a student "draws from" the way they draw
from a Multivariate Normal — it's the *sampling distribution of a test
statistic* built from other distributions. Concretely:

- If `Z ~ MultivariateNormal(0, I_p)` and `W ~ Wishart(m, I_p)`
  independently, then `T² = m · Zᵀ W⁻¹ Z` follows Hotelling's T²
  distribution with parameters `(p, m)`.
- Equivalently, and more usefully for implementation without needing
  Wishart built first: **`T²` is a direct rescaling of the F
  distribution** — `T²(p, m) = (mp)/(m-p+1) × F(p, m-p+1)`. Since `F` is
  already implemented in Symbulate, this is actually the *easiest*
  path: `HotellingT2` can be built as a thin wrapper that transforms an
  existing `F` distribution rather than needing new sampling machinery
  at all.
- **`mean()` / `var()` / `cdf()`**: all derivable directly from the
  known mean/variance/CDF of the underlying (rescaled) F distribution —
  no new numerical work needed beyond the rescaling formula itself.
- **`draw()`**: draw from the rescaled `F` and apply the same linear
  transform.
- This is the one distribution on the list where `cov()`/`corr()` don't
  really apply in the usual sense — Hotelling's T² is a **scalar-valued**
  statistic (it's univariate, despite being built from multivariate
  inputs), so it only needs `mean()`, `var()`, `sd()`, `cdf()`, and
  `draw()`, not `cov()`/`corr()`. Worth flagging: **this one might
  actually belong architecturally closer to a `Distribution` subclass
  than a `MultivariateDistribution` subclass**, since its *output* is a
  single number, not a vector — the "multivariate" part is in what it's
  derived from, not what it returns. This is a design decision worth
  confirming before writing any code.

---

## 8. Multivariate Hypergeometric

- **`mean()` / `cov()` / `var()`**: direct pass-through to
  `stats.multivariate_hypergeom(m, n).mean()` / `.cov()` / `.var()` —
  confirmed working (e.g., `m=[10,8,6], n=6` gives `mean() = [2.5, 2.0, 1.5]`
  and a full 3×3 covariance matrix).
- **`draw()`**: via `stats.multivariate_hypergeom(m, n).rvs()`.
- **`cdf()`**: same fundamental issue as Multinomial — no closed form,
  finite discrete support makes exact brute-force summation possible
  but potentially slow for large population sizes. Same recommendation:
  exact sum with a size cutoff, Monte Carlo fallback above it.

---

## 9. Negative Multinomial

- **Not in scipy at all** — this is the one distribution on the list
  requiring the most from-scratch work.
- **Construction**: the standard way to build it is as a Poisson-Gamma
  mixture generalization — draw a shared Gamma-distributed rate, then
  independent Poisson counts conditional on that rate for each
  category (the same conceptual trick as building ordinary Negative
  Binomial from Poisson-Gamma, just extended to multiple categories).
  Alternatively, it can be constructed via a direct combinatorial pmf
  formula (a multivariate generalization of the Negative Binomial pmf),
  which would need to be implemented by hand since no scipy building
  block exists.
- **`mean()` / `cov()`**: closed-form formulas exist in the standard
  references for this distribution (analogous to how Negative Binomial's
  mean/variance have simple closed forms) — implementable directly
  once the parameterization is settled, no numerical estimation needed.
- **`draw()`**: via the Poisson-Gamma mixture construction above (draw
  the Gamma rate, then draw each Poisson count) — straightforward once
  the construction is chosen, but is genuinely new simulation code, not
  a scipy pass-through.
- **`cdf()`**: same brute-force-sum-with-cutoff approach as Multinomial
  and Multivariate Hypergeometric, though the search region is
  unbounded in this case (Negative Multinomial has infinite discrete
  support, unlike Multinomial/Multivariate Hypergeometric, which are
  bounded by `n` or the population size) — this makes an exact
  brute-force sum not always finite/terminating, so `cdf()` here likely
  needs a genuine numerical truncation strategy (sum until the
  remaining tail probability is below some tolerance) or a Monte Carlo
  fallback as the primary approach rather than a backup.

---

## 10. Other multivariate (non-matrix-valued) distributions worth considering

A few legitimate additions surfaced while auditing this list that
weren't on it:

- **Dirichlet-Multinomial** — the compound distribution combining
  Dirichlet + Multinomial (integrating out the multinomial probability
  vector against a Dirichlet prior). Genuinely common in Bayesian
  text/count modeling (topic models, overdispersed multinomial data) —
  arguably belongs on this list alongside Beta-Binomial's role as the
  univariate analogue, which is already in the broader distributions
  table.
- **Multivariate Poisson** — models correlated Poisson counts (e.g., via
  a common-shock construction). Real applications in correlated
  count-data modeling (insurance claims across related lines,
  multi-sensor event counts).
- **Multivariate Log-Normal** — nearly free once `MultivariateNormal`
  has the full method set: it's just the elementwise exponential of a
  Multivariate Normal draw, so `mean()`/`cov()` have known closed-form
  transforms of the underlying MVN parameters (not simply the
  elementwise exp of the MVN's mean/cov — there's a standard correction
  formula), but `draw()` is trivial (`exp()` of an MVN draw).
- **Marshall–Olkin Multivariate Exponential** — models dependent
  failure times across multiple components; real use in reliability
  engineering. Already flagged in the broader distributions table as a
  "bivariate construction," but it generalizes to arbitrary dimension
  and is genuinely multivariate, not matrix-valued.
- **Logistic-Normal** — an alternative to Dirichlet for modeling
  compositional data (proportions that sum to 1), built by applying a
  softmax-like transform to a Multivariate Normal draw. Increasingly
  common in modern applications (microbiome data, topic modeling) as a
  more flexible alternative to Dirichlet since it can model correlation
  structure between components that Dirichlet cannot.
- **von Mises–Fisher** — the spherical generalization of Von Mises to
  higher dimensions (directional statistics). This came up in the
  broader distribution audit already; it's non-matrix-valued and
  vector-output, so it could reasonably sit in this same batch, though
  it's a smaller audience (directional/circular statistics specifically)
  than the other candidates here.
- **Multivariate Skew-Normal / Skew-t** — grad-level extensions of
  Multivariate Normal/t with an added skewness parameter; lower
  priority given the specialized audience, but worth a one-line mention
  given Skew-Normal is already in the broader table as a univariate
  addition.

Recommendation: **Dirichlet-Multinomial and Multivariate Poisson** are
the two most worth adding to the active list — both have real,
non-niche applications and would round out the "multivariate discrete"
side of this batch (which is currently Multinomial, Multivariate
Hypergeometric, and Negative Multinomial — all three "success/count"
flavored, with no compound/overdispersed option, which
Dirichlet-Multinomial would directly fill the same way Beta-Binomial
fills that role in the univariate case).

---

## 11. Design for `Distribution.plot()` on multivariate distributions

Currently, `MultivariateNormal.plot()` and `Multinomial.plot()` both
simply raise an exception. This section proposes what should replace
that. **The 2-D case and the higher-dimensional case need genuinely
different treatment** — they're not the same problem at different
sizes, they're different problems.

### 2-D case: a real joint plot is feasible and valuable

For any of these seven distributions restricted to exactly 2 dimensions
(or 2-D marginal projections of higher-dimensional ones — see below),
a joint visualization is both implementable and pedagogically useful:

- **Continuous 2-D (MultivariateNormal, Multivariate t restricted to
  2-D, Dirichlet restricted to 3 categories projected onto its 2-D
  simplex support)**: a **contour plot** or **heatmap** of the joint
  pdf over a 2-D grid — directly analogous to what the existing
  univariate `plot()` already does (evaluate `self.pdf(xs)` over a
  grid, just extended to a 2-D grid `self.pdf([x, y])` and rendered with
  `contourf`/`pcolormesh` instead of a line plot). This reuses the
  existing `xlim`-style logic, just computed independently for each of
  the two axes and combined into a meshgrid.
- **Discrete 2-D (Multinomial restricted to 3 categories — since 2
  categories reduces to Binomial — Multivariate Hypergeometric
  similarly restricted)**: a **heatmap/grid of scatter points** sized or
  colored by pmf value, the discrete analogue of the continuous contour
  plot — similar to how the existing univariate `plot()` already
  branches on `self.discrete` to choose `scatter` vs. a smooth curve.
- This means `plot()` for the 2-D case is a genuine, natural extension
  of the existing univariate plotting code, not a fundamentally new
  piece of infrastructure — it reuses the same discrete/continuous
  branching logic, just one dimension higher.

### Higher dimensions (3+): requiring an explicit `dims=(i, j)` argument

Above 2 dimensions, there's no single natural "the plot" the way there
is for 1-D or 2-D. The proposed API: `plot()` takes an optional `dims`
argument.

```python
X = MultivariateNormal(mean=[1,2,3,4], cov=cov4)   # a 4-D distribution

X.plot()             # raises a clear error:
                      # "This distribution has 4 dimensions. Specify
                      #  which two to plot, e.g. X.plot(dims=(0, 2))."

X.plot(dims=(0, 2))  # plots the joint density of X_1 and X_3 only
```

If the distribution has exactly 2 dimensions, `dims` is unused — `plot()`
just plots the one joint density that exists, the same way today's 1-D
`plot()` never needs the caller to specify which variable to show. The
requirement only kicks in at 3+ dimensions, where there's no longer a
single obvious default.

**Why require it rather than silently defaulting to, say, `dims=(0,1)`**:
a silent default risks a student believing they've seen "the
distribution" and never noticing that some other pair of dimensions has
a much stronger (or different-signed) relationship that the default view
never showed. Requiring an explicit choice makes the omission of the
other dimensions a conscious decision rather than an invisible one.

**Why `dims=(i,j)` plots a real distribution, not an approximation.**
This only works cleanly because, for these particular distributions, the
lower-dimensional marginal is *closed-form* — not something that has to
be numerically integrated out or approximated by simulation. This was
verified directly (not assumed) for MultivariateNormal: extracting the
sub-mean and sub-covariance-matrix for two coordinates out of a 4-D MVN,
and comparing against a 200,000-draw simulation of the full 4-D
distribution:

```
Closed-form 2-D marginal mean: [1, 3]
Closed-form 2-D marginal cov:  [[4, 0.5], [0.5, 1]]

Simulated marginal mean:  [1.0014, 2.9998]      <- matches
Simulated marginal cov:   [[4.006, 0.492],
                            [0.492, 0.9996]]     <- matches
```

So `X.plot(dims=(0,2))` for `MultivariateNormal` really means: pull out
those 2 entries of the mean vector and the corresponding 2×2 submatrix
of the covariance matrix, construct the corresponding 2-D
`MultivariateNormal` from them, and reuse the 2-D contour-plotting logic
above — no simulation, no approximation involved.

**Each of the seven distributions needs its own rule for how to extract
this marginal**, since the mechanics differ by family, though the
underlying "marginal stays in the same family" property holds for most
of them:
- **MultivariateNormal / Multivariate t**: submatrix/subvector
  extraction, as verified above (same degrees of freedom for the t case).
- **Dirichlet / Multinomial / Multivariate Hypergeometric**: an
  "aggregation" property instead — lumping every unselected category
  into a single "everything else" bucket gives back a smaller
  Dirichlet/Multinomial/Multivariate Hypergeometric exactly, with a
  modified parameter vector, rather than a plain submatrix operation.
- **Negative Multinomial**: likely also has an aggregation-style
  marginal, given its close relationship to Multinomial/Negative
  Binomial, but this should be confirmed against the specific
  parameterization chosen in Section 9 before assuming it carries over
  directly.

### Beyond a single pair: the pairs matrix / corner plot

For visualizing *all* pairwise relationships at once rather than one
pair at a time, the standard solution (matching R's `pairs()` and
Python's `seaborn.pairplot` / `corner.py`) is a pairs matrix / corner
plot: an `n × n` grid of panels for an `n`-dimensional distribution,
where:

- **Diagonal panels** show the 1-D marginal density/pmf for that single
  dimension — reusing the standard univariate `Distribution.plot()`
  entirely, since (per above) these marginals are closed-form for every
  distribution on this list.
- **Off-diagonal panels** show the 2-D joint density/pmf for that pair
  of dimensions — reusing the exact `dims=(i,j)` logic above, just
  rendered small and without axis decoration.
- **Only the lower triangle needs to be populated.** Panel `(i,j)` and
  panel `(j,i)` show the same pairwise relationship (just transposed),
  so the upper triangle is conventionally left blank — this halves the
  number of off-diagonal panels that actually need to be computed and
  rendered.

For an `n`-dimensional distribution this comes out to `n` diagonal
panels plus `n(n-1)/2` lower-triangle panels — very manageable at 4 or
5 dimensions, but worth a hard look at higher dimensions: a 10-category
Dirichlet would need 45 off-diagonal panels, which is a lot of
computation and screen space for arguably limited pedagogical payoff at
that scale.

**Recommendation**: implement the 2-D contour/heatmap as the real,
first-class case (used automatically whenever the distribution has
exactly 2 dimensions), require the explicit `dims=(i,j)` argument for
anything higher-dimensional (raising a clear, informative error if
`plot()` is called on a 3+ dimensional distribution without `dims`
specified), and offer the full pairs matrix as an **opt-in**
(`X.plot(pairs=True)`) rather than a default — with a size guard (e.g.,
a warning or required confirmation) once dimensionality climbs past
some threshold like 6–8, so a student can't accidentally trigger a
45-panel render without realizing it.

This also means `Hotelling's T²` doesn't need any of this — since its
output is scalar (see Section 7), its `plot()` can just reuse the
**existing univariate `plot()` logic entirely**, no new plotting code
needed at all.

---

## Summary table

| Distribution | `mean()`/`cov()` source | `cdf()` feasibility | Biggest open design question |
|---|---|---|---|
| Multinomial | scipy pass-through | Exact (small n) / Monte Carlo (large n) | cdf() cutoff strategy |
| MultivariateNormal | scipy pass-through | scipy pass-through (non-deterministic) | Whether to pin `random_state` |
| Dirichlet | scipy pass-through | Not recommended (no natural closed form) | Skip cdf() or approximate-with-caveat |
| Multivariate t | Derived formulas, `df`-conditional | scipy pass-through (non-deterministic) | Guarding undefined moments for small `df` |
| Hotelling's T² | Derived from rescaled F | Derived from rescaled F | Whether this belongs in `MultivariateDistribution` at all, given scalar output |
| Multivariate Hypergeometric | scipy pass-through | Exact (bounded support) / Monte Carlo fallback | cdf() cutoff strategy |
| Negative Multinomial | Closed-form, but no scipy backing | Numerical truncation (unbounded support) | Needs a full custom `draw()`, not a scipy wrap |
