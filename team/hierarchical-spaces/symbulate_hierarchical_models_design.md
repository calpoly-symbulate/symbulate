# Hierarchical Models in Symbulate — Design Summary

> **Revision note:** This document is being kept as a running record of the design
> discussion, not replaced as new ideas come up. Original content (the case for a
> `>>`-based combinator, `Placeholder`, and the code-change outline) is preserved
> below unchanged. Sections 6–8 add later alternatives that are still under active
> consideration — a `*`-based combinator built on the existing multiplication-rule
> operator, other candidate symbols, and an `AssumeHierarchical` escape hatch
> modeled on the existing `AssumeIndependent`. No single option has been finalized
> yet; Section 8 lays out the trade-offs side by side.

## 1. Goal

Symbulate currently has no direct way to express a **hierarchical (compound) model** —
a prior distribution for one random variable, and a second distribution for another
random variable whose parameters depend on the first variable's realized value. For
example:

> X ~ Beta(1, 2), and (Y | X = x) ~ Binomial(10, x)

Today this requires manually writing a custom generator function and wrapping it in a
raw `ProbabilitySpace`:

```python
def BetaBinomialProbspace(a, b, n):
    x = Beta(a, b).draw()
    y = Binomial(n, x).draw()
    return x, y

P = ProbabilitySpace(BetaBinomialProbspace(1, 2, 10))
X, Y = RV(P)
```

This works, but it doesn't compose with the rest of Symbulate's operator vocabulary
(`*` for independent joins, `**` for iid replication, `.apply()` for transformations),
and it requires writing an explicit Python function for every hierarchical model, no
matter how simple. The goal is a general-purpose, reasonably concise syntax for
expressing **any** "prior, then a conditional distribution built from the prior's
value" relationship — not just a fixed list of named textbook models
(Beta-Binomial, Gamma-Poisson, etc.).

## 2. Why this design

Four options were considered:

1. **Keep only the manual `ProbabilitySpace(generator_function)` pattern.**
   Rejected as the sole mechanism — it's a poor fit for the target audience (intro
   stats students), doesn't compose with existing operators, and every model needs
   hand-written boilerplate.

2. **Named built-ins per model** (`BetaBinomialHierarchical`, `GammaPoissonHierarchical`, …).
   Rejected as the primary mechanism — the space of "common" hierarchical models is
   unbounded, so this always sends unanticipated cases back to the manual pattern.
   It also doesn't teach the underlying idea of composing a prior with a conditional.
   Kept as an optional, thin convenience layer on top of the general mechanism.

3. **A full symbolic/graph-based model** (PyMC/NumPyro-style), which would allow the
   literal forward-reference syntax originally proposed (`Binomial(10, X)` where `X`
   is the not-yet-created RV). Rejected as disproportionate: it requires rebuilding
   every distribution as a lazy graph node and deferring all evaluation program-wide —
   a rewrite of the library's execution model for what is ultimately a cosmetic
   syntax win.

4. **A `>>` ("then") combinator on `ProbabilitySpace`, chosen.** A prior space and a
   function from a prior draw to a conditional space combine into a new
   `ProbabilitySpace`:

   ```python
   P = Beta(1, 2) >> (lambda x: Binomial(10, x))
   X, Y = RV(P)
   ```

   This was preferred because:
   - **Minimal surface area** — one new class (`HierarchicalProbabilitySpace`), one
     operator, one optional function alias (`Hierarchical(prior, cond_func)`). No
     changes to `RV`, `Distribution`, or existing operators.
   - **Composes for free** — the result is just another `ProbabilitySpace`, so it
     works with `RV(...)`, `*`, `**`, `.apply()`, and chains to arbitrary depth via
     repeated `>>` (each step's conditional function receives the accumulated tuple
     of everything drawn so far).
   - **Matches the existing operator vocabulary** — `*` already means "independent
     join" and `**` means "iid replication"; `>>` reads naturally as "then,
     conditionally," extending the same mental model rather than introducing a new
     paradigm.
   - **General rather than enumerable** — one idea covers Beta-Binomial,
     Gamma-Poisson, Normal-Normal, discrete mixtures, and arbitrary-depth chains,
     instead of a growing list of named classes.

   The trade-off: it gives up the literal forward-reference syntax
   (`Binomial(10, X)` with `X` referenced before its own assignment), since that is
   not expressible in plain Python. The **`Placeholder`** addition (below) closes
   most of that gap without requiring a symbolic engine.

   *(Note: `>>` was the first operator proposed. It is not the only one under
   consideration — see Section 6 for `*`, `@`, and why brand-new symbols like
   `*|` aren't possible in plain Python, and Section 8 for a side-by-side
   comparison of all the operator options.)*

## 3. Lambda vs. Placeholder syntax

Both forms use the same `>>` mechanism underneath; the difference is purely in how
the conditional distribution's dependence on the parent value is written.

| | **Lambda** | **Placeholder** |
|---|---|---|
| Beta-Binomial | `Beta(1, 2) >> (lambda x: Binomial(10, x))` | `X = Placeholder()`<br>`Beta(1, 2) >> Binomial(10, X)` |
| Reads like the "X" appears literally in the child distribution's call | No — `x` is a bound lambda parameter | Yes — `X` appears directly inside `Binomial(10, X)` |
| Requires an extra line to declare the dependency | No | Yes (`X = Placeholder()`) |
| Handles arithmetic on the parent value (e.g. `x + 1`) | Yes, naturally | Yes, via operator overloading on `Placeholder` (`+`, `-`, `*`, `/`, `**`, indexing) |
| Handles **branching** logic (the distribution *family* itself depends on the parent, e.g. a discrete "which model" choice) | Yes | No — genuine branching needs a real function |
| Implementation cost | None — plain Python closures | New `Placeholder`/expression-tree module, plus one hook in the shared `Distribution` base class |

**Bottom line:** `Placeholder` is a closer visual match to the original idea and
avoids the word `lambda` for the common case of "plug the parent's value into one
distribution's parameters," but `lambda`/`def` remain necessary — and fully
supported side-by-side — whenever the conditional logic branches on the parent
value rather than just substituting it into a formula.

## 4. Examples

All of the following were prototyped and verified numerically against known
theoretical values.

**Beta-Binomial**
```python
# lambda form
P = Beta(1, 2) >> (lambda x: Binomial(10, x))

# placeholder form
X = Placeholder()
P = Beta(1, 2) >> Binomial(10, X)

X, Y = RV(P)   # E[X] ≈ 1/3, E[Y] ≈ 10/3
```

**Normal-Normal (random effects / measurement error)**
```python
Mu = Placeholder()
P = Normal(0, 10) >> Normal(Mu, 1)
M, X = RV(P)   # Var[X] ≈ 10² + 1² = 101, by the law of total variance
```

**Gamma-Poisson (negative binomial via mixing)**
```python
Lam = Placeholder()
P = Gamma(shape=2, scale=3) >> Poisson(Lam)
L, N = RV(P)   # E[N] ≈ shape × scale = 6
```

**Discrete mixture — requires lambda (branching, not substitution)**
```python
Model = Bernoulli(p=0.5)          # 0 = fair coin, 1 = biased coin
def coin_given_model(m):
    return Bernoulli(p=0.5) if m == 0 else Bernoulli(p=0.9)
P = Model >> coin_given_model
M, X = RV(P)   # P(X=1) ≈ 0.5·0.5 + 0.5·0.9 = 0.7
```

**Arbitrary-depth chains** (both forms shown; `>>` associates left-to-right and each
step gets the joined tuple so far)
```python
# lambda form
P = (Uniform(0, 1)
     >> (lambda a: Uniform(0, a))
     >> (lambda ab: Uniform(0, ab[1])))

# placeholder form (indexing into the accumulated tuple)
AB = Placeholder()
P = (Uniform(0, 1) >> Uniform(0, Placeholder())) >> Uniform(0, AB[1])

A, B, C = RV(P)
```

## 5. Summary of the code changes this would require

No implementation is included here — this section only outlines the shape of the
change for a future patch.

1. **`probability_space.py`**
   - Add a `HierarchicalProbabilitySpace(ProbabilitySpace)` class: draws from a
     prior space, calls a conditional function with that draw to get the child
     space, draws from the child space, and returns the joined outcome (reusing
     the existing `join` helper, which already flattens nested tuples — this is
     what makes multi-level chaining work automatically).
   - Add a `Hierarchical(prior_space, cond_space_func)` top-level convenience
     function as a named alias.
   - Add `__rshift__` to `ProbabilitySpace` so `prior >> cond_func` calls
     `Hierarchical(prior, cond_func)`.
   - Include a clear `TypeError` if the conditional function doesn't return a
     `ProbabilitySpace` (a likely user mistake — forgetting to wrap a return
     value in a distribution).

2. **New `placeholder.py` module** (only needed if the `Placeholder` syntax is
   included, not required for the lambda-only version)
   - A `Placeholder` class representing "the value the prior will produce,"
     with operator overloads (`+`, `-`, `*`, `/`, `**`, unary `-`, `[]`) that
     build a small deferred expression tree instead of evaluating immediately.
   - A `DeferredDistribution` class: stores a distribution class plus its
     args/kwargs (some of which may be `Placeholder` expressions); calling it
     with the prior's realized value substitutes that value into the deferred
     arguments and constructs the real distribution. Being callable, it needs
     no special-casing in `Hierarchical`/`>>` — it satisfies the same
     "callable" contract as a lambda.

3. **`distributions.py`**
   - Add a `__new__` method to the shared `Distribution` base class that checks
     whether any constructor argument is a `Placeholder`/expression; if so, it
     returns a `DeferredDistribution` instead of a real instance (skipping the
     normal `__init__`/validation, which assumes concrete numeric parameters).
     Because this lives on the shared base class, it applies to every
     distribution (`Beta`, `Binomial`, `Normal`, `Poisson`, etc.) without
     touching each subclass individually.

4. **`__init__.py`**
   - Export `Hierarchical`, `HierarchicalProbabilitySpace`, and (if included)
     `Placeholder` from the top-level package.

5. **Documentation / tests**
   - Docstrings and doctest-style examples matching the repo's existing
     convention (seen throughout `probability_space.py` and `distributions.py`).
   - Unit tests covering: basic two-level hierarchical models, multi-level
     chains, the `TypeError` guard, and (if included) `Placeholder` arithmetic
     and indexing.
   - Optionally, a small set of named convenience wrappers built on top of
     `Hierarchical` (e.g. `BetaBinomial = lambda a, b, n: Beta(a, b) >>
     partial(Binomial, n)`) for direct name-recognition in a stats course,
     without making them the primary mechanism.

None of this changes the behavior of existing, non-hierarchical code — ordinary
distribution construction and `.draw()`/`.sim()` calls are unaffected in every
case checked so far.

## 6. Operator alternatives: `*`, `@`, and why brand-new symbols don't work

A later question in the discussion was whether `*` — Symbulate's existing operator
for **independent** joins (`DistX * DistY` implies `X ⊥ Y`) — could be reused for
hierarchical composition instead of introducing `>>`. The motivating observation:
independence is just the special case of the general multiplication rule

> P(A and B) = P(A) · P(B | A)

where the second factor happens not to depend on the first. So `*` could mean
"combine via the multiplication rule" in general, with independence as the
special case where the right-hand side doesn't use the left-hand draw.

**This works.** `ProbabilitySpace.__mul__` currently assumes `other` is a
`ProbabilitySpace` and calls `other.draw()` unconditionally — so today,
`Beta(1,2) * (lambda x: Binomial(10, x))` simply crashes (`AttributeError`, no
`.draw()` on a function). That's an open door, not a conflict: dispatch on the
type of `other` instead:

```python
def __mul__(self, other):
    if isinstance(other, ProbabilitySpace):
        # existing behavior: independent join, P(A)P(B)
        ...
    elif callable(other):
        # new: conditional family, P(A)P(B|A)
        return HierarchicalProbabilitySpace(self, other)
    else:
        raise TypeError(...)
```

This was prototyped and tested: independence (`die * coin`), the lambda form
(`Beta(1,2) * (lambda x: Binomial(10, x))`), the `Placeholder` form
(`Beta(1,2) * Binomial(10, X)`), and 3-level chains (`Beta(1,2) *
partial(Binomial, 10) * (lambda xy: Poisson(xy[1] + 1))`) all produced correct
sample statistics, and plain independent joins were unaffected.

**Trade-offs of reusing `*`:**
- **Pro:** one operator instead of two; independence becomes a special case of a
  single, already-taught rule rather than a separate mechanism — arguably the
  more honest match to the math.
- **Con:** `*` is no longer guaranteed commutative. `DistA * DistB` (independence)
  is symmetric; `Beta(1,2) * (lambda x: Binomial(10,x))` is not — the prior must
  be on the left, matching the reading order of P(A)P(B|A). This is consistent
  with the math notation (you always write the prior/marginal first) but is a
  subtle surprise for a symbol people otherwise associate with commuting.
- **Con:** the dispatch relies on `other` being callable vs. being a
  `ProbabilitySpace`. Safe today (no `ProbabilitySpace`/`Distribution` subclass
  defines `__call__`), but it makes "no `ProbabilitySpace` subclass may implement
  `__call__`" an implicit invariant of the library going forward.

**Could a brand-new symbol like `*|`, `|*`, or `*>` be introduced instead**, to
sidestep the commutativity concern entirely with something visually suggestive of
"multiply-then-condition"? No — this isn't something Python allows. Only a fixed,
closed set of tokens can be overloaded via dunder methods (`+ - * / // % ** @ <<
>> & | ^ ~` and comparisons); `*|`/`|*`/`*>` aren't in that set and aren't
parseable as single tokens at all (`a *| b` is a `SyntaxError` — two binary
operators back to back with nothing between them). Getting a genuinely new infix
symbol would require rewriting source code before Python parses it (an import
hook or preprocessor) — far too heavy a mechanism, and one that would create
real friction with tooling (linters, IDEs, `pytest`, Jupyter) for a single
convenience operator.

That leaves only *existing, currently-unused* operators on
`ProbabilitySpace`/`Distribution` as realistic candidates:

| Operator | Currently used elsewhere? | Fit |
|---|---|---|
| `>>` | No | Clean, unambiguous, but a genuinely new piece of vocabulary to teach. |
| `*` | Yes — independent join | Unifies with the multiplication rule; not commutative in the hierarchical case. |
| `@` (matmul) | No | Already culturally associated with *non-commutative* composition (matrix multiplication), so it doesn't fight users' intuitions the way asymmetric `*` might. A reasonable dark-horse alternative. |
| `\|` (or) | Only on `RV`, for conditioning on an event (`X \| (X > 0)`) | Thematically close ("given"), but would make `\|` mean two different things depending on whether the left operand is an `RV` or a `ProbabilitySpace`. Riskier than it looks — recommend avoiding. |

## 7. `AssumeIndependent` and an analogous `AssumeHierarchical`

`AssumeIndependent` (in `independence.py`) was examined for relevance to this
design. What it does: it exists so that a student who already wrote

```python
X = RV(Dist1)
Y = RV(Dist2)
```

(two RVs on two *separate* probability spaces — `(X & Y).sim(...)` would raise,
by design, since the library deliberately refuses to let you combine RVs from
different spaces without an explicit statement of how they relate) can add one
line, `X, Y = AssumeIndependent(X, Y)`, instead of rewriting the first two lines
as `X, Y = RV(Dist1 * Dist2)`. Mechanically, it builds a new joint
`ProbabilitySpace` whose `draw()` calls each original RV's underlying
`prob_space.draw()` independently, joins the results, and reconstructs `X` and
`Y` as RVs over that shared space, reusing each one's original `func`.

**Relevance to hierarchical models:** this is directly relevant, and prototyping
it produced a genuine (if secondary) new proposal: an `AssumeHierarchical`
escape hatch, symmetric in spirit to `AssumeIndependent` but for dependence
instead of independence.

The initial assumption was that this *couldn't* work — independence can be
patched together after the fact because the two draws are causally separate
(no information has to flow between them), whereas a hierarchical model seems to
require building the joint space from the start, since `Y`'s distribution must
depend on `X`'s draw at the moment it happens. That assumption turned out to be
wrong: `X.prob_space.draw()` composed with `X.func` — i.e., `X.draw()` itself —
is already a perfectly good `ProbabilitySpace` in disguise, so it can be reused
as the "prior" in exactly the same `Hierarchical(...)` machinery:

```python
def AssumeHierarchical(prior_rv, cond_space_func):
    if not isinstance(prior_rv, RV):
        raise TypeError(...)
    if not callable(cond_space_func):
        raise TypeError(...)
    prior_as_space = ProbabilitySpace(prior_rv.draw)   # X's realized value, as a space
    P = Hierarchical(prior_as_space, cond_space_func)  # same machinery as *, >>
    return tuple(RV(P))
```

Tested and confirmed working, including the case where the prior RV has already
been transformed (`X = RV(Beta(1,2)).apply(lambda p: 1 - p)`) — the conditional
correctly sees the *transformed* value:

```python
X = RV(Beta(1, 2))
X, Y = AssumeHierarchical(X, lambda x: Binomial(10, x))
# E[X] ≈ 1/3, E[Y] ≈ 10/3

Xflip = RV(Beta(1, 2)).apply(lambda p: 1 - p)
Xflip, Y = AssumeHierarchical(Xflip, lambda x: Binomial(10, x))
# E[Y] ≈ 10 · E[1 - Beta(1,2)] = 20/3, correctly conditioned on the transformed X
```

**What this confirms about the overall design:**
- `HierarchicalProbabilitySpace`/`Hierarchical` is the right load-bearing
  abstraction — both the primary syntax (`*` or `>>`) *and* this after-the-fact
  escape hatch reduce to thin wrappers around the same mechanism.
- The asymmetry noted for `*` in Section 6 shows up here too, and is worth
  documenting as a general property of this feature rather than an operator
  quirk: `AssumeIndependent` takes any number of RVs, order irrelevant, because
  independence has no direction. `AssumeHierarchical` necessarily takes exactly
  one prior RV plus one conditional function, because dependence has a
  direction (there is a parent and a child) that independence doesn't.
- One gap `AssumeHierarchical` has that `AssumeIndependent` doesn't:
  `AssumeIndependent` checks that its inputs are on genuinely different
  probability spaces, to catch accidental double-use. `AssumeHierarchical` has
  no equivalent cheap check yet — calling it twice on the same `prior_rv` (once
  per desired child) silently creates two unrelated joint spaces that both
  claim to extend the same `X`. Worth guarding against (or at least documenting)
  if this is built.

## 8. Summary comparison of all options considered

| Option | Literal forward-reference syntax? | New operator/symbol needed? | Escape hatch for already-written code? | Commutative? | Status |
|---|---|---|---|---|---|
| Manual `ProbabilitySpace(generator_function)` | No | No | N/A (it *is* the fallback) | N/A | Always available; poor primary UX |
| Named per-model built-ins (`BetaBinomialHierarchical`, …) | No | No | No | N/A | Optional thin layer on top of whichever mechanism is chosen, not a replacement for it |
| Full symbolic/graph model (PyMC-style) | Yes | N/A | N/A | N/A | Rejected — disproportionate rewrite for a cosmetic syntax win |
| `>>` combinator + `lambda` | No (needs `lambda`) | Yes (`>>`, currently unused) | Not yet designed (would need its own `AssumeHierarchical`-style wrapper, same as `*`) | N/A (new symbol, no baggage) | Original proposal; fully prototyped |
| `>>` combinator + `Placeholder` | Close (`Binomial(10, X)` appears literally) | Yes (`>>`) | Same as above | N/A | Prototyped; adds a small `placeholder.py` module |
| `*` combinator (multiplication-rule reuse) + `lambda`/`Placeholder` | Same as `>>` versions | No — reuses existing `*` | Yes — `AssumeHierarchical`, prototyped | No (order-dependent) | Prototyped; unifies independence and dependence under one operator |
| `@` combinator (matmul reuse) | Same as `>>`/`*` versions | Reuses existing, currently-unused `@` | Same `AssumeHierarchical` wrapper would apply | No, but `@` doesn't carry a commutativity expectation | Not yet prototyped; flagged as a candidate worth testing |
| Brand-new symbol (`*|`, `\|*`, `*>`) | — | Not possible in plain Python | — | — | Ruled out |

No option has been finalized. The two most promising directions remain (a) `*`
or `@` as the composition operator, each paired with both the `lambda` and
`Placeholder` spellings for the conditional, and (b) `AssumeHierarchical` as a
complementary escape hatch alongside whichever operator is chosen, mirroring
`AssumeIndependent`'s existing role for the independence case.

