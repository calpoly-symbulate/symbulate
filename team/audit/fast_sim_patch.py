"""
DRAFT PATCH -- not meant to be imported or run as-is.

A reviewable sketch of the code changes to add a batched-drawing fast
path for `.sim(n)`, scoped narrowly to NegativeHypergeometric and
TruncatedNormal. Each section below is valid Python and labeled with its
target file and approximate location in the current dev branch, so it
can be copy-pasted into place. See fast_sim_notes.md for the full
diagnosis and the reasoning behind each design choice.

Five small edits, three files:
  1. distributions.py     -- new `_fast_sim` hook on the base
                               Distribution class, returning None by
                               default (no-op for every other class).
  2. distributions.py     -- one-line override on NegativeHypergeometric.
  3. distributions.py     -- one-line override on TruncatedNormal.
  4. probability_space.py -- guarded fast path in ProbabilitySpace.sim.
  5. random_variables.py  -- guarded fast path in RV.sim, plus a named
                               identity sentinel so `func` can be
                               compared with `is`.

Nothing here changes behavior for any other distribution: `_fast_sim`
returns None unless overridden, and every call site falls straight
through to the existing `_sim_with_progress` loop when it does.
"""

# =============================================================================
# FILE: symbulate/distributions.py
# LOCATION: class Distribution, right after the `xlim` setter (~line 233),
#           before `_support`. Shown here as a standalone class so the
#           method body is valid on its own; in the real file it's one
#           more method alongside the others already there.
# =============================================================================


class _DistributionFastSimHook:
    """Sketch of the one method to add to the real `Distribution` class."""

    def _fast_sim(self, n):
        """Optional fast path for `.sim(n)`: draw all n samples in a single
        vectorized scipy call instead of one at a time.

        Returns None (the default) if there's no batched path -- true for
        every distribution unless it overrides this method. Override only
        when the underlying scipy sampler has a large, size-independent
        per-call setup cost that batching amortizes away. As of this
        patch, that's NegativeHypergeometric and TruncatedNormal; see
        fast_sim_notes.md for how that was diagnosed and why the list
        stops there for now.

        Deliberately takes no `random_state` argument: it uses the same
        module-level `rng` already in scope here (the one `draw()` uses),
        rather than accepting one from the call site. `sim()` lives in a
        different module, and passing `rng` across that boundary would
        risk quietly advancing the wrong random stream -- see the notes
        file for why that matters in this codebase specifically.
        """
        return None


# =============================================================================
# FILE: symbulate/distributions.py
# LOCATION: class NegativeHypergeometric, appended after `__init__`
#           (~line 1075, right after
#           `super().__init__(params, stats.nhypergeom, True)`).
# =============================================================================


class _NegativeHypergeometricFastSim:
    """Sketch of the one method to add to the real class."""

    def _fast_sim(self, n):
        """Draw all n samples in one vectorized call.

        Plain `.draw()` (`self.sim_func(**self.params, random_state=rng)`)
        rebuilds an entire CDF table and inverse-CDF interpolator from
        scratch on *every single call* -- see scipy's
        `nhypergeom_gen._rvs1` -- regardless of how many samples are
        requested. Paying that fixed cost once per `.sim(n)` instead of n
        times is why this went from ~1.1 ms/sample to ~1 microsecond/sample
        (~850x) in testing.
        """
        return self.sim_func(**self.params, size=n, random_state=rng)


# =============================================================================
# FILE: symbulate/distributions.py
# LOCATION: class TruncatedNormal, appended after `__init__`
#           (~line 2175, right after
#           `super().__init__(params, stats.truncnorm, False)`).
# =============================================================================


class _TruncatedNormalFastSim:
    """Sketch of the one method to add to the real class."""

    def _fast_sim(self, n):
        """Draw all n samples in one vectorized call.

        Same idea as NegativeHypergeometric: `truncnorm`'s `_rvs` inverts
        a uniform through `_ppf`, redoing non-trivial per-call work rather
        than amortizing it across a batch. ~80x speedup in testing.
        """
        return self.sim_func(**self.params, size=n, random_state=rng)


# =============================================================================
# FILE: symbulate/probability_space.py
# LOCATION: ProbabilitySpace.sim, replacing the current method body
#           (~line 74).
# =============================================================================

# Add to the existing import line:
#   from .result import Vector, InfiniteVector, join, Scalar


def _probability_space_sim(self, n):
    """Simulate n draws from probability space. [docstring unchanged]"""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")
    batch = self._fast_sim(n) if hasattr(self, "_fast_sim") else None
    if batch is not None:
        return Results([Scalar(v) for v in batch])
    return Results(_sim_with_progress(self.draw, n))


# =============================================================================
# FILE: symbulate/random_variables.py
# LOCATION: near the top of the file, alongside the other module-level
#           definitions, before the RV class.
# =============================================================================

# A named identity function, so RV.__init__'s default `func` can be
# compared with `is` in sim() below. This is the exact same object every
# no-arg call to RV(...) already got implicitly -- Python evaluates a
# default argument once -- this just gives it a name.

_IDENTITY = lambda x: x  # noqa: E731 -- named on purpose, see comment above


# =============================================================================
# FILE: symbulate/random_variables.py
# LOCATION: RV.__init__ signature (~line 43). One-word change.
# =============================================================================


def _rv_init(self, prob_space, func=_IDENTITY):
    """Create a random variable. [body otherwise unchanged]"""
    self.prob_space = prob_space
    self.func = func


# =============================================================================
# FILE: symbulate/random_variables.py
# LOCATION: RV.sim, replacing the current method body (~line 101).
# =============================================================================

# Add to the existing import line:
#   from .result import Vector, join, is_scalar, is_numeric_vector, TimeFunction, Scalar


def _rv_sim(self, n):
    """Simulate n draws from the random variable. [docstring unchanged]"""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")
    if self.func is _IDENTITY and hasattr(self.prob_space, "_fast_sim"):
        batch = self.prob_space._fast_sim(n)
        if batch is not None:
            return RVResults([Scalar(v) for v in batch])
    return RVResults(_sim_with_progress(self.draw, n))