# Notes: Adding `RenewalProcess` to Symbulate

Repo examined: `calpoly-symbulate/symbulate`, `dev` branch (cloned locally and run
against, not just read).

## How `PoissonProcess` is built

File: `symbulate/poisson_process.py`. Three pieces:

1. **`PoissonProcessResult(ContinuousTimeFunction, DiscreteValued)`**
   Takes a list/iterable of interarrival times and builds a step function
   `N(t)` = number of arrivals strictly before time `t`, by walking a
   cumulative sum. Nothing about this logic assumes exponential
   interarrival times — it works for *any* sequence of nonnegative
   numbers.

2. **`PoissonProcessProbabilitySpace(ProbabilitySpace)`**
   Validates `rate` (must be a positive number), then on `draw()`:
   ```python
   interarrival_times = (Exponential(rate=self.rate) ** inf).draw()
   return PoissonProcessResult(interarrival_times)
   ```
   The `** inf` operator (repeated independent draws, evaluated lazily)
   is defined generically on `ProbabilitySpace.__pow__` in
   `probability_space.py` — it is **not** specific to `Exponential`. Any
   Symbulate `Distribution` supports `dist ** inf`.

3. **`PoissonProcess(RandomProcess, RV)`**
   Thin public wrapper: stores `rate`, builds the probability space above,
   and calls `RandomProcess.__init__` (with index set `Reals()`, since
   this is a continuous-time process) and `RV.__init__`.

## Why a Renewal Process generalizes cleanly

The only Poisson-specific line in the whole module is the one that builds
`Exponential(rate=self.rate)`. Everything else — the counting-function
construction, the `RandomProcess`/`RV` wiring, indexing via `N[t]` /
`N(t)`, `.sim()`, `.mean()`, etc. — is generic and distribution-agnostic.

So a renewal process (interarrival times i.i.d. from *any* distribution,
not just exponential) is a near copy-paste of `poisson_process.py`,
replacing the fixed `rate` argument with a user-supplied
`interarrival_dist` (a Symbulate `Distribution` instance, e.g.
`Gamma(shape=2, rate=1)`, `Uniform(a=1, b=3)`, `Exponential(rate=2)`,
etc.):

```python
def draw():
    interarrival_times = (self.interarrival_dist ** inf).draw()
    return RenewalProcessResult(interarrival_times)
```

`PoissonProcess` is, mathematically and structurally, just
`RenewalProcess(interarrival_dist=Exponential(rate=rate))` with rate
validation and a friendlier `rate=` keyword. They could share code, but
I'd keep `PoissonProcess` as-is to avoid touching its existing docs/tests.

## What I verified (not just read)

Cloned the `dev` branch, dropped in a draft `renewal_process.py`, and ran
it against the real package:

- `RenewalProcess(interarrival_dist=Gamma(shape=2, rate=1)).draw()` produces
  a valid step-function sample path (`path(3)` returned an int count).
- `N(3).sim(5)` returns a proper `Results`/`Index Result` object, same as
  `PoissonProcess`.
- Passing a non-`Distribution` (e.g. `interarrival_dist=5`) raises a
  `TypeError` with a clear message, mirroring `PoissonProcess`'s
  validation style.
- Sanity-checked that plugging in `Exponential(rate=2)` as the
  interarrival distribution reproduces Poisson-like behavior:
  `N(4).sim(3000)` had an empirical mean of ~8.01, matching the
  theoretical `rate * t = 2 * 4 = 8`.

## Open design questions before merging

1. **Validation scope.** `PoissonProcessProbabilitySpace` validates
   `rate > 0` itself. For `RenewalProcessProbabilitySpace`, I only check
   that `interarrival_dist` *is* a `Distribution` — parameter-level
   validation (e.g. `Gamma`'s shape must be positive) is already handled
   by that distribution's own constructor. Confirm this division of
   responsibility is what's wanted.

2. **Negative/invalid support.** Nothing currently stops someone from
   passing a distribution with support on negative numbers (e.g.
   `Normal`), which would violate the "nondecreasing counting function"
   invariant a renewal process assumes. Options: leave it as a documented
   assumption (matches how Symbulate generally trusts the user to supply
   sensible distributions), or add a runtime check / warning.

3. **Naming.** Went with `interarrival_dist` to read naturally
   (`RenewalProcess(interarrival_dist=Gamma(...))`). Worth confirming
   this matches whatever convention the maintainers would prefer (e.g.
   `interarrival_distribution`, `wait_dist`).

4. **Tests.** `test_poisson_process.py` is a good template to mirror:
   result-object tests (starts at zero, nondecreasing, integer-valued),
   probability-space tests (stores distribution, draw returns correct
   type), process tests (index set is `Reals()`, `__getitem__`/`__call__`
   return `RV`, reproducibility under seeding), and error-handling tests
   (non-`Distribution` argument raises `TypeError`). Happy to draft
   `test_renewal_process.py` on request.

## Files produced

- `renewal_process.py` — draft implementation, verified to run against
  the real `dev` branch.
- `renewal_process_notes.md` — this file.
