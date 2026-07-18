# Lazy Sample Paths in Symbulate: Gaussian Processes and Diffusion Processes

This note explains the core design idea behind `gaussian_process.py` (existing
Symbulate code, which powers `BrownianMotion`) and how `diffusion_process.py`
(the new `DiffusionProcess` class) generalizes that idea to processes that
aren't Gaussian.

The one-sentence summary: **a sample path is not a pre-computed array of
values on a fixed grid — it's a lazy function that fills itself in on
demand, and every new value it produces is drawn conditional on everything
already drawn.** That's what makes "zoom in on a sub-interval" show finer
detail of the *same* path instead of a brand-new independent path.

---

## 1. The original idea: `GaussianProcessResult`

### The problem with the naive approach

The naive way to simulate Brownian motion is: pick a step size `dt`, draw
`N` i.i.d. `Normal(0, dt)` increments, and cumulatively sum them. That
produces one path on one fixed grid. If you then want to "zoom in" on
`[0, 0.1]`, there's nothing there — you'd have to draw a whole new random
walk on that interval, and it would have no relationship to the coarse path
you started with.

### The key mathematical fact

A Gaussian process is defined by a mean function `mean_func(t)` and a
covariance function `cov_func(s, t)`, with the property that **any finite
collection of time points has a multivariate normal joint distribution**.
Multivariate normals have a closed-form conditional distribution:

```
[X_A]     ( [mean_A]   [cov_AA  cov_AB] )
[X_B]  ~ N( [mean_B] , [cov_BA  cov_BB] )

X_B | X_A = a  ~  Normal( mean_B + cov_BA · cov_AA⁻¹ · (a - mean_A),
                           cov_BB - cov_BA · cov_AA⁻¹ · cov_AB )
```

This means: if you already know the path's value at some set of times `A`,
you can sample its value at any new set of times `B` from the *exact*
conditional distribution — and the result is guaranteed to be statistically
identical to having simulated all of `A ∪ B` at once in the first place.

### How the code implements it

`get_gaussian_process_result` builds a `GaussianProcessResult` object that
maintains, as instance state:

- `self.observed`: a dict `{time: value}` of every time point simulated so far
- `self.mean`: the mean vector over those times
- `self.cov`: the covariance matrix over those times

Calling `path(t)` (or `path([t1, t2, ...])`) triggers `_vfunc`, which:

1. Splits the requested times into ones already in `self.observed` (just
   look up the cached value) and genuinely new ones.
2. For the new times, computes `cov12` (cross-covariance between old and
   new times) and `cov22` (covariance among the new times) using
   `cov_func`.
3. Applies the conditional-normal formula above (`cond_mean`, `cond_var`)
   using `np.linalg.solve` instead of an explicit matrix inverse.
4. Draws the new values with `rng.multivariate_normal(cond_mean, cond_var)`.
5. Appends the new times/values into `self.mean`, `self.cov`, and
   `self.observed`, so future calls condition on them too.

A small `MACHINE_EPS * np.eye(...)` jitter is added to `cov11` before
solving, purely for numerical stability (the covariance matrix can become
near-singular as more correlated points accumulate).

Two extra details:

- If `cov_func(t, t) == 0`, the point is deterministic (`mean_func(t)`) —
  no randomness to simulate, so it's short-circuited.
- `DiscreteTimeSequence` vs. `Reals` index sets determine whether the
  result behaves like a `DiscreteTimeFunction` or `ContinuousTimeFunction`
  (this only affects indexing conventions, not the conditioning logic).

### Brownian motion is just a Gaussian process

```python
BrownianMotionProbabilitySpace(drift, scale):
    mean_func = lambda t: drift * t
    cov_func  = lambda s, t: scale**2 * min(s, t)
```

`Cov(B_s, B_t) = σ²·min(s,t)` is precisely the covariance structure of
Brownian motion. There is nothing Brownian-motion-specific in the
simulation code at all — `BrownianMotion` is 100% inherited machinery from
`GaussianProcess`, just with a particular `cov_func`. This is why it
"zooms" correctly: it's exact Gaussian-process conditioning, and Brownian
motion happens to be a Gaussian process.

---

## 2. The extension: `DiffusionProcess`

### Why Gaussian-process conditioning doesn't generalize

A general Itô diffusion

```
dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t,   X_0 = x_0
```

is Markov, but its finite-dimensional distributions are **not** jointly
Gaussian unless `μ` and `σ` are constant (or otherwise special, e.g.
affine). Geometric Brownian motion, Ornstein–Uhlenbeck-with-nonlinear-drift,
CIR-type processes, etc. have no closed-form joint density over an
arbitrary set of time points. So there's no exact analogue of step 3 above
("solve for the conditional normal") — the conditioning formula simply
doesn't exist in closed form for these processes.

What we *can* still exploit is the same lazy, lookup-and-fill-in structure
that made `GaussianProcessResult` work — we just need a different (and
necessarily approximate) rule for filling in a new point given its
neighbors.

### The replacement for exact conditioning: a bridge approximation

`DiffusionProcessResult` keeps two parallel lists instead of a dict:
`self.times` and `self.values`, kept sorted, found via `bisect.bisect_left`.
Calling `path(t)` falls into one of three cases:

1. **Already cached** (`t` matches an entry in `self.times`): return the
   cached value — identical to the Gaussian-process case.
2. **Beyond every simulated time** (`_simulate_forward`): there's nothing to
   condition on to the right, only a known starting point to the left, so
   this just runs ordinary Euler–Maruyama forward from the last cached
   `(t, x)` in fine sub-steps of size ≤ `tol`, caching every intermediate
   point along the way.
3. **Strictly between two cached times `(t0, x0)` and `(t1, x1)`**
   (`_bridge_sample`): this is the interesting case, and the direct
   generalization of Gaussian-process conditioning — we need
   `X(t) | X(t0)=x0, X(t1)=x1`, i.e. a **diffusion bridge**.

### The bridge sampling rule

Exact diffusion bridges are themselves a hard research problem in general
(this is what the Beskos–Roberts "exact algorithm" and Delyon–Hu-type MCMC
methods are for). Symbulate's `DiffusionProcess` uses the standard
practical approximation: treat the process as locally Brownian on the
sub-interval, using drift/diffusion evaluated at the left endpoint:

```
dt  = t1 - t0
mu    = drift(x0, t0)
sigma = diffusion(x0, t0)

mean = x0 + (t - t0)/dt · (x1 - x0 - mu·dt) + mu·(t - t0)
var  = sigma² · (t - t0)(t1 - t) / dt
```

The `var` term is exactly the classical **Brownian bridge variance**
formula; the `mean` term is a linear interpolation between the endpoints,
adjusted so the average drift implied over `[t0, t1]` matches `mu`. Two
important properties fall out of this:

- **It's exact for Brownian motion** (constant `μ, σ`): the formula
  collapses to the true Brownian-bridge conditional law, so in that special
  case `DiffusionProcess` reproduces `GaussianProcessResult`'s behavior
  exactly.
- **It converges as the bridge gets finer**: for general `μ(x,t), σ(x,t)`,
  this is only a first-order (Euler–Maruyama-consistent) local
  approximation — but the approximation error shrinks as `t1 - t0 → 0`,
  which is exactly what happens as you keep zooming in.

### Recursive bisection instead of solving directly for `t`

Rather than sampling directly at the requested `t`, `_bridge_sample` always
samples the **midpoint** `tm = (t0+t1)/2` first, inserts it into the cached
`times`/`values` lists, and recurses into whichever half contains `t` —
stopping once an interval is already ≤ `tol` wide. This is the same
"successive midpoint refinement" idea behind the classical Lévy–Ciesielski
construction of Brownian motion, generalized with the local drift/diffusion
correction above. Two consequences:

- Every bisection leaves behind a real cached point, so repeated zooming
  into smaller and smaller sub-intervals keeps refining the *same* path
  rather than resampling it.
- The mesh is only ever as fine as something has actually asked for —
  nothing is precomputed on a fixed grid.

### Forward simulation vs. bridge simulation

| | Extending forward past the last known point | Filling in between two known points |
|---|---|---|
| Method | Ordinary Euler–Maruyama | Brownian-bridge-corrected Euler–Maruyama |
| Conditioned on | Only the left endpoint | Both endpoints |
| Analogue in `GaussianProcessResult` | Conditioning on all previously observed times (no "future" to condition on either) | Conditioning on all previously observed times |

In the Gaussian-process code these two cases are actually unified (the
conditional-normal formula doesn't care whether the new times are "before,"
"after," or "in between" old ones — it conditions on *all* of them
simultaneously). The diffusion-process code has to split them apart because
the bridge formula needs two known endpoints, whereas forward extension
only has one.

### Where the approximation is controlled

The `tol` parameter is the only knob: it's both the maximum forward Euler–
Maruyama step size and the smallest interval the bridge sampler will bisect
down to before sampling directly. Smaller `tol` ⇒ closer to the true
diffusion (and exact in the constant-coefficient/Brownian case regardless
of `tol`) at the cost of more cached points and computation.

---

## 3. Summary comparison

| | `GaussianProcessResult` | `DiffusionProcessResult` |
|---|---|---|
| Underlying model | Any process with known `mean_func`, `cov_func` | Any Itô SDE `dX = μ(X,t)dt + σ(X,t)dW` |
| State kept | dict of observed values + running mean vector/cov matrix | sorted lists of observed `(time, value)` pairs |
| Filling in a new point | Exact multivariate-normal conditioning | Approximate diffusion bridge (Brownian-bridge-corrected Euler–Maruyama), refined by recursive bisection |
| Exactness | Exact, for any finite set of times | Exact only when `μ, σ` are constant (i.e., reduces to the Gaussian case); otherwise a converging approximation |
| Special case | `BrownianMotion` = `GaussianProcess` with `cov_func = σ²·min(s,t)` | `DiffusionProcess` with `μ=0, σ=const` reproduces `BrownianMotion`'s behavior |
