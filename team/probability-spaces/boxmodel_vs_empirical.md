# BoxModel vs. Empirical: Do We Need a New Class?

## Background

While prototyping a way to build a Symbulate `ProbabilitySpace` from an
arbitrary table of outcomes and probabilities (motivated by SOA-style
mortality tables), we built two classes:

- **`Empirical`** — a discrete distribution from an explicit
  `(outcomes, probabilities)` table, implemented as a `Distribution`
  subclass wrapping `scipy.stats.rv_discrete`.
- **`LifeTable`** — a subclass of `Empirical` that parses an SOA-style
  mortality table (`age`, `qx` columns) into the curtate future
  lifetime distribution `K_x`.

Before proposing these as new additions, it's worth checking whether
Symbulate already solves this problem. It does, partially: **`BoxModel`**,
which already exists in `symbulate/probability_space.py`, models drawing
tickets from a box of outcomes with custom probabilities. This document
compares the two and lays out what it would take to close the gap
between them rather than shipping a fully separate class.

## What BoxModel already does

```python
BoxModel(box, size=None, replace=True, probs=None, order_matters=True)
```

`BoxModel` takes a `list` or `dict` of tickets, optional custom `probs`,
and supports single or multiple draws, with or without replacement, and
order-sensitive or order-insensitive outcomes. It's Symbulate's existing
general-purpose tool for "sample from a table of outcomes," and it
already covers the exact use case of drawing a random outcome from a
life table:

```python
box = BoxModel(list(range(66)), probs=list(K45.probabilities_))
box.draw()   # e.g. 37
```

## Where BoxModel and Empirical actually differ

| | `BoxModel` | `Empirical` / `LifeTable` |
|---|---|---|
| Base class | `ProbabilitySpace` | `Distribution` (subclass of `ProbabilitySpace`) |
| Outcome types | Any object — strings, tuples, cards, etc. | Integers only (a `scipy.stats.rv_discrete` limitation) |
| `.pmf()` / `.cdf()` / `.quantile()` | No | Yes — exact, closed-form from the table |
| `.mean()` / `.var()` / `.sd()` / `.median()` | No (must simulate and estimate) | Yes — exact, computed directly from the table |
| `.plot()` / `.spinner()` | No | Yes |
| Sampling without replacement, multiple tickets per draw, ordering | Yes (`replace`, `size`, `order_matters`) | No — one draw is one outcome |
| Designed for | Combinatorics-style experiments (urns, cards, dice) | Analytic distributions you want to reason about mathematically |

The practical gap showed up directly in testing: both were run against
the same life-table probabilities. `BoxModel`'s simulated mean over
5,000 draws was `35.66` (a noisy Monte Carlo estimate), while
`LifeTable.mean()` returned `35.538` exactly, computed directly from
the survivorship column `l_x` with no simulation involved. If a student
wants an exact expected value, an exact pmf plot, or a closed-form
actuarial present value, `BoxModel` alone can't provide that — everything
has to be estimated by simulation.

## Bottom line

`BoxModel` already answers "can I build a random draw from a table of
outcomes" — that's not a gap. What `Empirical`/`LifeTable` actually adds
is giving a table-based outcome the **full `Distribution` API**
(exact moments, pmf/cdf, plotting) the same way `Binomial` or `Normal`
already have it. Framed that way, this isn't really "a new distribution
to add" — it's "`BoxModel`'s table-based outcomes don't yet have access
to the `Distribution` machinery."

## Elaboration: how to actually close that gap

There are three broad ways to give `BoxModel`-style tables access to
`Distribution`'s analytic machinery, in increasing order of how
cleanly they fit Symbulate's existing architecture.

### Option 1: A conversion method on BoxModel

Add a method like `BoxModel.to_distribution()` that inspects `self.box`
and `self.probs`, and — if every ticket is an (or can be cast to an)
integer and `size is None` (a single draw) — returns an `Empirical`
built from the same table:

```python
box = BoxModel(list(range(66)), probs=list(K45.probabilities_))
K45_dist = box.to_distribution()   # now has .pmf(), .mean(), .plot(), etc.
```

**Pros:** Non-invasive — `BoxModel`'s existing behavior and signature
don't change at all, so no existing code or tests break. Discoverable,
since it's a method directly on the object a student already built.

**Cons:** Two objects to keep track of for the same conceptual thing
(the `BoxModel` and the `Distribution` it converts to), and it does
nothing for `BoxModel`s with `size > 1`, `replace=False`, or
non-integer/non-orderable outcomes, where "distribution" in the
`Distribution` sense doesn't cleanly apply anyway.

### Option 2: An integer-outcome check that upgrades BoxModel automatically

Have `BoxModel.__init__` detect when `size is None`, `replace=True`
doesn't matter (single draw has no with/without-replacement
distinction), and every ticket is integer-valued, and in that case
have `BoxModel` itself subclass or delegate to `Distribution`/`Empirical`
internally — so a student never has to know two classes exist; they
just get the extra methods "for free" when their box happens to
qualify.

**Pros:** Zero new API surface for students — the same `BoxModel` call
just does more when it can.

**Cons:** This is the riskiest option from a software-engineering
standpoint. `BoxModel` is a stable, widely-used class; making its
return type or available methods conditional on the *contents* of
`box` (rather than its constructor arguments) is a subtle,
hard-to-document behavior change, and it risks breaking the mental
model that `BoxModel` always returns a plain `ProbabilitySpace`. This
also doesn't resolve the deeper issue that `scipy.stats.rv_discrete`
genuinely requires integer support — non-integer tickets (e.g., a box
of dollar loss amounts) still couldn't get the analytic treatment
without a further workaround (see the note below).

### Option 3 (recommended): Keep Empirical separate, but make the relationship explicit

Keep `Empirical` as a standalone `Distribution` subclass — this is the
option we've already prototyped and tested — but:

1. **Document `Empirical` as "the `Distribution`-side counterpart to
   `BoxModel`"** directly in both classes' docstrings, so students and
   instructors understand *when* to reach for which: `BoxModel` for
   combinatorics-flavored problems (cards, urns, sampling without
   replacement), `Empirical` for "I have a table and I want the
   analytic apparatus."
2. **Add a convenience constructor**, `BoxModel.as_empirical()` or a
   module-level helper `empirical_from_box(box_model)`, that performs
   the Option 1 conversion but lives as a clearly-labeled bridge
   function rather than a core method — this keeps `BoxModel`'s
   existing contract untouched while still letting a student go from
   one to the other with one line, in either direction.
3. **Solve the integer-outcome restriction once, centrally**, rather
   than letting each future table-based class reinvent it. The
   restriction only exists because `scipy.stats.rv_discrete` requires
   integer support; the actual fix is an internal index-mapping layer
   inside `Empirical` itself — store the real (possibly non-integer,
   non-numeric) outcomes in an array, build `rv_discrete` over their
   integer *positions* `0, 1, ..., n-1`, and translate positions back
   to real outcomes at the `pmf`/`cdf`/`draw` boundary. This removes
   the integer restriction entirely for any future subclass (claim
   severities in dollars, categorical outcomes, etc.) without every
   subclass needing its own workaround.

**Why this is the better default:** it keeps `BoxModel` exactly as
stable and predictable as it is today (no behavior changes, no new
edge cases in its constructor), while still directly answering the
actual pedagogical need — a student who built a `BoxModel` and now
wants `.mean()` or `.plot()` has an obvious, named path to get there,
and the next person who wants a different kind of table-based
`Distribution` (e.g., an empirical loss-severity distribution from
claims data) inherits the index-mapping fix automatically rather than
hitting the same integer-only wall we just diagnosed for `LifeTable`.

## Recommendation

Ship `Empirical` (with the index-mapping fix from Option 3, item 3) as
a genuinely new, narrowly-scoped `Distribution` subclass — not a
duplicate of `BoxModel`, but its analytic-statistics counterpart — and
add the small bridge function so the two are discoverable from one
another. Avoid Option 2 (conditional behavior inside `BoxModel` itself)
given the risk of destabilizing a class that's already in wide use.
