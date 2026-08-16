# Adding a Generic Continuous Probability Space to Symbulate

A design discussion and implementation for simulating from an arbitrary
continuous pdf in [Symbulate](https://github.com/calpoly-symbulate/symbulate/tree/dev),
so that:

```python
P = ContinuousProbabilitySpace(f)
X = RV(P)
X.sim(10000)
```

works for a user-supplied density `f`, falling back gracefully when `f`
is hard to integrate or invert. The full code is in the companion file
`continuous_probability_space.py`.

---

## The question

> Read the Symbulate code base:
> https://github.com/calpoly-symbulate/symbulate/tree/dev
>
> I want to add ability to simulate from a generic continuous distribution
> given its pdf. For example, if `f` represents a pdf, I want something like
>
> ```python
> P = ContinuousProbabilitySpace(f)
> X = RV(P)
> X.sim(10000)
> ```
>
> The definition of `f` would contain the possible values.
> How would you implement this? When possible, could you integrate `f` to
> get the cdf `F`, then invert `F` to get the quantile function `Q`, then
> simulate `Q(U)` where `U` is `Uniform(0, 1)`? What if `f` is not
> integrable or the cdf is not invertible? I want `.sim(10000)` to simulate
> 10000 independent values, so MCMC is not an option (that's something I
> want to discuss separately, but not now). Could you implement some form
> of rejection sampling, hopefully as efficiently as possible
> (`X.sim(10000)` should return 10000 values, but the simulation might need
> to first generate more, some of which would be rejected, but hopefully
> not too many)? Or would you recommend another method?
>
> Most common distributions would already be built in to Symbulate by
> name, so I can't think of too many pdfs where this functionality would
> actually be necessary. It's more of a teaching tool.

---

## How this fits into Symbulate's architecture

I cloned the `dev` branch and read through `probability_space.py`,
`random_variables.py`, `distributions.py`, and `results.py`. Two things
matter for this design:

1. **`ProbabilitySpace` is just a wrapper around a `draw()` function.**
   `RV(P).sim(n)` (in `random_variables.py`) calls
   `_sim_with_progress(self.draw, n)`, which is a plain Python `for` loop
   calling `draw()` one outcome at a time — `sim()` is *not* vectorized at
   this layer, even for the built-in distributions. The built-ins get
   their speed from `scipy.stats`, not from Symbulate's loop.
2. Built-in continuous distributions (`Uniform`, `Normal`, `Exponential`,
   …) all subclass `Distribution(ProbabilitySpace)` and get `pdf`, `cdf`,
   `quantile` for free from scipy (`distributions.py:98-119`). Your
   `ContinuousProbabilitySpace(f)` is essentially asking: *can we build
   that same `pdf`/`cdf`/`quantile`/`draw` bundle when scipy doesn't
   already know the distribution?*

Since `sim()` always calls `draw()` in a loop, the way to make
`X.sim(10000)` fast isn't to touch `sim()` — it's to make `draw()` cheap
on average by generating numbers in **vectorized numpy batches
internally** and dispensing them one at a time from a buffer. That's the
trick used below for both methods.

## The two-tier strategy

**Tier 1 — inverse-CDF sampling (preferred, no rejection at all).**
Build a fine grid over the support, numerically integrate `f` with
`scipy.integrate.cumulative_trapezoid` to get `F` on that grid, then
invert `F` with a monotone cubic interpolant (`PchipInterpolator`,
swapping the roles of x and y) to get `Q`. Sampling is then just `Q(U)`
for `U ~ Uniform(0,1)`, fully vectorized, and every candidate is a valid
draw — no waste.

This works for *almost anything you'd write down in a class*, including
densities with an integrable singularity (e.g. `f(x) = 1/(2√x)` on
`(0,1)`), because the singularity gets integrated away by the time you
invert `F`. It fails only when:

- `f` isn't a valid density (negative somewhere, or integrates to ~0),
- the numerical CDF isn't increasing enough to invert reliably (e.g. `f`
  is zero over almost all of the given support — often a sign the
  support/`xlim` is wrong), or
- the grid can't resolve `f` well (extremely oscillatory or a support
  that's badly mis-specified).

**Tier 2 — rejection sampling (fallback).**
If tier 1 raises, fall back automatically (or you can force
`method="rejection"`). Pick a proposal `g` and constant `M` with
`f(x) ≤ M·g(x)`:

- **bounded support** → uniform proposal on `[lo, hi]`,
- **unbounded support** → a scaled Cauchy proposal (heavy tails dominate
  most textbook densities), or a proposal you supply yourself.

`M` is found by maximizing `f(x)/g(x)` over a grid. Since `∫f = 1`, the
acceptance rate is exactly `1/M`, so instead of drawing one candidate at
a time, generate `⌈n / p_accept · 1.15⌉` candidates in one numpy batch,
accept/reject the whole batch at once, and top up with another batch
only if unlucky. That's the "generate a few extra, not too many"
behavior asked for — it's controlled, not open-ended MCMC-style
iteration.

## Usage

```python
from symbulate import *
import numpy as np

f = lambda x: np.where((0 <= x) & (x <= 1), 2 * x, 0.0)   # triangular density
P = ContinuousProbabilitySpace(f, xlim=(0, 1))
X = RV(P)
X.sim(10000).mean()   # ~0.667, via inverse-CDF, no rejection
```

Tested end to end:

- `f(x) = 2x` on `[0,1]`: uses `inverse_cdf`, `mean ≈ 0.668` (true `2/3`).
- Custom bell curve (no `xlim` given): auto-detects support `[-8, 8]`,
  `inverse_cdf`, `mean ≈ 0`, `sd ≈ 1`.
- `f(x) = 1/(2√x)` on `(0,1)` (integrable singularity): still
  `inverse_cdf`, `mean ≈ 0.336` (true `1/3`).
- Sharp bimodal density forced to `method="rejection"`: `M ≈ 8.4`,
  acceptance ≈ 12%, correct proportions, 20,000 draws in ~0.1s.
- An invalid pdf (negative somewhere / integrates to 0) raises a clear
  `ValueError` instead of silently producing garbage.

## Answering the "what if" questions

- **Not integrable numerically** (e.g. `quad`/the trapezoid grid
  genuinely fails, or the "pdf" isn't a valid density): tier 1 raises,
  `auto` catches it and warns, tier 2 kicks in. If tier 2 *also* can't
  find a finite `M` (e.g. `f`'s tails are heavier than the default Cauchy
  proposal — rare for teaching examples), it raises with a message
  pointing the user to pass `proposal=` and `M=` explicitly.
- **CDF not invertible**: for a *bona fide* pdf, `F` is always monotonic
  non-decreasing (since `f ≥ 0`), so true non-invertibility doesn't
  happen mathematically — the practical failure mode is numerical (a
  wrong/omitted `xlim` making the grid mostly zero, or a pathologically
  spiky `f`). That's exactly what the "CDF not increasing enough" check
  catches, and it's recoverable by tier 2, or by the user tightening
  `xlim`.
- **Rejection sampling efficiency**: because acceptance rate `1/M` is
  known analytically once `M` is found, batches are sized to
  `n/p_accept · 1.15` instead of guessing — so we generate "a bit more
  than needed," not open-ended amounts, and it's all vectorized numpy
  rather than a Python loop.

## A caveat and an alternative worth knowing about

For teaching purposes, tier 1 (inverse-CDF) should quietly handle the
overwhelming majority of examples you'd actually reach for this feature
— it's exact-ish, wastes nothing, and only degrades gracefully into tier
2 for genuinely awkward densities. The one thing worth flagging: the
automatic `M` search for rejection sampling is a grid-based heuristic,
not a guaranteed global bound — for extremely heavy-tailed or highly
oscillatory custom pdfs it could occasionally underestimate `M`
slightly. A more bulletproof (but heavier) alternative would be
**adaptive rejection sampling** (Gilks & Wild), which builds a
piecewise-linear envelope directly from `log f` and tightens it as
rejections occur — no `M` search needed, and it's the "gold standard"
for arbitrary log-concave densities. This wasn't implemented here
because it's a fair amount more code for a case (non-log-concave,
heavy-tailed custom pdf, taught in an intro course) that's genuinely
rare in practice.

## Wiring it into the repo

Add the file at `symbulate/continuous.py` and add
`ContinuousProbabilitySpace` to the export list in `symbulate/__init__.py`
(alongside the existing `from .probability_space import (...)` /
`from .distributions import (...)` blocks), plus a few tests in
`symbulate/tests/` following the existing `Distribution` subclass test
patterns.
