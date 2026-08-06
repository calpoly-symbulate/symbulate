# `.sim(n)` batching fix for `NegativeHypergeometric` and `TruncatedNormal`

## The problem

`.sim(n)` draws one sample at a time in a loop (`_sim_with_progress` calls
`self.draw()` exactly `n` times). For most distributions that's fine,
because a single scipy `.rvs()` call costs roughly the same whether you
ask for one sample or a thousand. But two distributions currently in
Symbulate don't work that way:

- **`NegativeHypergeometric`** is backed by `scipy.stats.nhypergeom`.
  Its `_rvs` implementation rebuilds an entire CDF table and an
  `interp1d` inverse-CDF interpolator **from scratch on every call**,
  regardless of how many samples are requested. Measured cost: about
  **1.1 ms per single draw**, vs. **~1 microsecond per sample** when
  10,000 samples are requested in one batched call. That's roughly an
  **850x** difference, and it's the reason `NegativeHypergeometric(...).sim(10000)`
  can take upwards of ten seconds.
- **`TruncatedNormal`** is backed by `scipy.stats.truncnorm`, whose
  `_rvs` inverts a uniform through `_ppf` -- cheaper than `nhypergeom`'s
  table rebuild, but still redoing real per-call work rather than
  amortizing it. Measured: **~80x** speedup batched vs. looped.

Both were confirmed directly: looping `dist.rvs(random_state=rng)` n
times vs. calling `dist.rvs(size=n, random_state=rng)` once, same total
sample count, same `rng` state.

## Why only these two

A broader scan of scipy distributions turned up a handful of others with
the same symptom to varying degrees (`Trapezoidal`, `von Mises-Fisher`,
`Zipf`, several noncentral/hypergeometric variants), and a few that look
slow but *aren't* fixable this way -- `kstwo` (the two-sided KS
statistic) is the clearest example: it's ~70x slower than an ordinary
distribution, but batching gives essentially zero benefit, because its
cost comes from a genuinely expensive per-sample calculation
(`kolmogn`), not repeated setup. `NegativeHypergeometric` and
`TruncatedNormal` are the two currently in Symbulate with both (a) a
real problem today and (b) a fix that actually works. The design below
is written so adding another override later (for `Trapezoidal` or
whichever else) is a one-line addition, not a new call-site change.

## Design: an opt-in hook, not a change to `.sim()` for everyone

The base `Distribution` class gets one new method, `_fast_sim(self, n)`,
that returns `None` by default. `NegativeHypergeometric` and
`TruncatedNormal` are the only two classes that override it, each with a
single line: call the same scipy `.rvs` already stored as `self.sim_func`,
batched with `size=n` instead of looped.

`ProbabilitySpace.sim()` and `RV.sim()` (the two places `.sim(n)` is
actually implemented) each get a small guarded check: if `_fast_sim(n)`
returns something other than `None`, use it; otherwise fall through to
the exact loop that runs today. Every other distribution's `.sim()`
pays one cheap `hasattr`/attribute check and then behaves exactly as
before.

For `RV.sim()` specifically, the fast path only fires when nothing has
been composed on top of the distribution -- checked via `self.func is
_IDENTITY`, a named sentinel pulled out of `RV.__init__`'s default
argument. The moment someone does `X.apply(f)`, conditions on
something, or otherwise changes `func`, the fast path is skipped and the
ordinary per-draw loop runs, exactly as it does today. This is the main
reason the change stays narrow: composed random variables, transformed
values, and anything built out of these two distributions rather than
being one directly, are entirely untouched.

## A deliberate choice: no `rng` argument in `_fast_sim`

Worth flagging on its own: the codebase currently has **two independent
`np.random.default_rng()` instances** -- one at module scope in
`distributions.py` (which `Distribution.draw()` uses) and a separate one
in `probability_space.py` (which `ProbabilitySpace.sim()`'s own scope
has access to). If `_fast_sim` accepted an `rng` argument from the call
site, it would be easy to accidentally pass the *wrong* one -- advancing
a random stream that ordinary `.draw()` calls never touch, which would
be a confusing, hard-to-notice inconsistency for anyone who seeds one of
these but not the other.

The patch sidesteps this by having `_fast_sim`'s overrides reference the
same module-level `rng` that `draw()` already uses, defined in
`distributions.py`, with no `rng` parameter passed in from either call
site. This is not a fix for the two-rng situation itself (out of scope
here) -- just a design choice to make sure this patch doesn't make it
worse or introduce a new place where the wrong stream could get used.

## What changes for users, and what doesn't

- **Behavior:** identical distributions, identical `.pdf`/`.cdf`/`.mean`/
  etc. Only `.sim(n)` on a bare, unmodified `NegativeHypergeometric(...)`
  or `TruncatedNormal(...)` (or an `RV` wrapping one directly) takes the
  fast path.
- **Reproducibility:** the *sequence* of values produced for a given
  seed will change for these two distributions specifically, since
  scipy's batched `rvs(size=n)` doesn't necessarily draw from the
  underlying random bits in the same order as n sequential single calls.
  Any notebook or test that hardcodes an expected sequence of
  `NegativeHypergeometric`/`TruncatedNormal` values for a fixed seed will
  need to be regenerated. Every other distribution is byte-for-byte
  unaffected.
- **Composed RVs:** anything built with `.apply()`, conditioning,
  arithmetic, or any transformation on top of these two distributions
  keeps running through the ordinary per-draw loop, unchanged.

## Suggested testing before merging

- Unit tests confirming `_fast_sim` returns `None` for every distribution
  except the two overrides.
- A statistical check (e.g. KS test against the known CDF) that the fast
  path's output distribution matches the looped path's, since the point
  of the change is speed, not a different distribution.
- A timing regression test with a generous margin (e.g. "10,000 draws of
  `NegativeHypergeometric` completes in under 1 second") to catch a
  future change that silently reintroduces the loop.
- Confirm `X.apply(lambda x: x)` on one of these two distributions still
  takes the slow path (i.e., that identity-*wrapped* RVs aren't
  accidentally caught by the `is _IDENTITY` check depending on how
  `.apply()` constructs its lambda).
