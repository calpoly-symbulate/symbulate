# The random-seed problem, and a suggested fix

## The problem

Symbulate has **seven independent `np.random.default_rng()` instances**,
one created at import time in each of these modules, with no connection
to each other and no public way to reseed any of them:

```
diffusion_process.py, distributions.py, gaussian_process.py,
hitting_times.py, markov_chains.py, plot.py, probability_space.py
```

There is no `seed=` parameter anywhere in the public API, and no
`symbulate.seed(...)` function. `np.random.seed(...)` -- the obvious
thing a user would try -- does nothing at all, because
`np.random.default_rng()` creates a `Generator` (PCG64) that is
completely independent of the legacy global state `np.random.seed()`
controls.

### This isn't hypothetical -- it's already broken today

The `RV.draw()` docstring itself demonstrates the bug:

```python
>>> np.random.seed(0)
>>> X = RV(Normal(0, 1))
>>> X.draw()
1.764052345967664          # claimed
```

Run for real:

```python
np.random.seed(0)
X.draw()   # -> 0.10369709571127243
np.random.seed(0)
X.draw()   # -> 1.7561125250878096   (different again)
```

Neither value matches, and reseeding doesn't even reproduce the *first*
result. This is the only file with a misleading example (`grep -c
"np.random.seed" symbulate/*.py` finds 2 occurrences, both in
`random_variables.py`), so the docstring fix itself is small -- but the
underlying mechanism it's demonstrating is genuinely broken.

### The test suite already works around this, badly

Since `.sim()` results need to be deterministic to test against, the
test suite monkey-patches each module's private `rng` directly, and has
to know exactly which modules a given feature touches:

```python
# test_hitting_times.py
gaussian_process.rng = np.random.default_rng(value)
hitting_times.rng = np.random.default_rng(value + 1)
distributions.rng = np.random.default_rng(value + 2)

# test_merton.py
diffusion_process.rng = np.random.default_rng(value)
gaussian_process.rng = np.random.default_rng(value + 1)
renewal_process.rng = np.random.default_rng(value + 2)   # dead code --
                                                          # renewal_process.py
                                                          # has no rng of
                                                          # its own to reseed
```

At least 11 test files independently reverse-engineer this and hand-pick
offset seeds per module. The dead `renewal_process.rng` line is a good
illustration of how easy it is to get this wrong even when writing the
test.

### It isn't even consistent within a single feature

A single simulated diffusion-process path already draws from **two
disconnected streams**:

```python
# diffusion_process.py
from .gaussian_process import get_gaussian_process_result  # uses gaussian_process.py's OWN rng
...
dw = rng.normal(0, sqrt_dt)                    # diffusion_process.py's own rng
...
brownian_path = get_gaussian_process_result(...)   # routes through gaussian_process.rng instead
```

If the split were intentional (e.g. "each process type gets its own
independent stream"), a *single* process type wouldn't fragment its own
randomness like this. It only happens because `diffusion_process.py`
calls a helper from a module that happened to set up its own generator
first -- pure accident of code organization, not a design decision.

### Is this needed for lazy evaluation to work? No.

`InfiniteVector`/`ContinuousTimeFunction` laziness (and the recursive
midpoint-bisection Brownian-bridge correction) works by **caching**: the
first time a value at a given index/time is requested, it's computed and
stored; later requests for the same slot return the cached value instead
of drawing again. That guarantee comes from the caching logic itself and
has nothing to do with which physical `Generator` object supplied the
number. A single shared `rng` behaves identically to seven separate ones
here.

Drawing many different distributions and processes from one shared
generator in a single-threaded session is standard practice (it's what
NumPy itself recommends) -- consecutive draws from one generator are
statistically independent of each other regardless of which feature
requested them. The one legitimate reason to want separate generators is
multi-threaded code, where a single `Generator` isn't safe to call
concurrently without locking; Symbulate runs single-threaded in a
notebook, so that doesn't apply.

## The fix

### The key trick: mutate in place, don't rebind

If module B does `from module_a import rng`, both names point to the
*same* object. Reassigning `module_a.rng = new_generator` breaks that
link (today's bug) -- but mutating the *existing* object's internal
state doesn't:

```python
rng.bit_generator.state = np.random.default_rng(seed_value).bit_generator.state
```

Verified directly, including across a module boundary: reseeding through
one module's function correctly changes what a second module's
already-imported reference produces next, with no rebinding needed
anywhere.

### The plan

1. **Pick one owner.** `probability_space.py` is the natural choice --
   nearly everything already imports from it.
2. **The other six modules** change their `rng = np.random.default_rng()`
   line to `from .probability_space import rng`. One line each. Zero
   changes needed at any of the ~50 actual call sites (`rng.normal(...)`,
   `rng.choice(...)`, etc. keep working exactly as written).
3. **Add one public function**, `seed(value)`, doing the in-place
   mutation above. Export it from `symbulate/__init__.py`.
4. **Fix the two misleading docstring examples** in `random_variables.py`.
5. **Replace the per-test monkey-patching** in the ~11 affected test
   files with calls to the new `seed(...)` -- simplifies the test suite
   and removes the dead `renewal_process.rng` line.

## Suggested code patches

### 1. `probability_space.py` -- add the public `seed` function

```python
# probability_space.py, near the existing `rng = np.random.default_rng()`

rng = np.random.default_rng()


def seed(value=None):
    """Reseed Symbulate's shared random generator for reproducible simulations.

    Every distribution, random process, and plot that uses randomness
    draws from the same underlying generator, so calling this once at
    the top of a notebook makes the rest of it reproducible. Call it
    again with a different value (or no value, for a fresh unpredictable
    seed) to start a new reproducible run.

    Parameters
    ----------
    value : int, optional
        The seed. If None, reseeds from fresh, unpredictable entropy.

    Examples
    --------
    >>> from symbulate import *
    >>> seed(0)
    >>> X = RV(Normal(0, 1))
    >>> X.draw()  # doctest: +SKIP
    0.12573022...
    >>> seed(0)
    >>> X.draw()  # doctest: +SKIP
    0.12573022...
    """
    rng.bit_generator.state = np.random.default_rng(value).bit_generator.state
```

### 2. The other six modules -- one-line change each

```python
# distributions.py, gaussian_process.py, markov_chains.py,
# diffusion_process.py, hitting_times.py, plot.py
#
# REMOVE:
#     rng = np.random.default_rng()
#
# REPLACE WITH:
#     from .probability_space import rng
```

(Watch for import-order/circularity: `probability_space.py` must not
import from any of these six, or this introduces a circular import.
Based on a quick check, none of them currently import from each other in
a way that would create a cycle through `probability_space.py`, but
worth double-checking at patch time.)

### 3. `symbulate/__init__.py` -- export `seed`

```python
from .probability_space import seed
```

### 4. `random_variables.py` -- fix the misleading docstring

```python
# BEFORE:
#     >>> import numpy as np
#     >>> np.random.seed(0)
#     >>> X = RV(Normal(0, 1))
#     >>> X.draw()
#     1.764052345967664
#
# AFTER:
#     >>> from symbulate import *
#     >>> seed(0)
#     >>> X = RV(Normal(0, 1))
#     >>> X.draw()  # doctest: +SKIP
#     <actual value once regenerated against the new shared rng>
```

### 5. Test suite -- replace per-module monkey-patching

```python
# BEFORE (test_hitting_times.py, and ~10 similar files):
def seed(value=42):
    gaussian_process.rng = np.random.default_rng(value)
    hitting_times.rng = np.random.default_rng(value + 1)
    distributions.rng = np.random.default_rng(value + 2)

# AFTER:
from symbulate import seed  # the new shared, public function
```

Each test file's bespoke local `seed()` helper (and the imports of
`gaussian_process`, `hitting_times`, `distributions`, etc. purely for
that purpose) can be deleted once this lands.

## What changes for users, and what doesn't

- **New:** `seed(value)` actually works, and works the same way
  regardless of which distributions or processes a notebook uses.
- **Unaffected:** anyone not calling `seed(...)` sees no behavior
  change -- the shared `rng` still starts from fresh OS entropy, exactly
  like the seven separate ones did.
- **Breaking, but only for internals:** code that currently monkey-patches
  `symbulate.distributions.rng` (or any of the other five) directly --
  i.e., the test suite -- needs to switch to the new `seed()` function,
  since after the fix most of those modules no longer own a `rng` of
  their own to reassign.

## Suggested testing before merging

- A single reproducibility test that calls `seed(42)`, draws from several
  *different* distributions/processes (e.g. a `Normal`, a `MarkovChain`,
  a `DiffusionProcess`), reseeds with the same value, and checks the
  second run matches the first -- replacing the current patchwork of
  per-file offset-seed helpers.
- Confirm `seed()` with no argument still gives an unpredictable,
  non-reproducible run (i.e. it isn't accidentally hardcoding a default
  seed).
- Regenerate the two `random_variables.py` docstring values against the
  real post-fix behavior rather than guessing them.
