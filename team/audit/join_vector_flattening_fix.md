# Fix: `join()` incorrectly flattens `Vector` results

## Summary

Combining a `Vector`-valued draw (e.g. from `**`, or from a multivariate
distribution) with other random variables via `&` — or joining it as part
of a hierarchical/product space — should keep that vector as one nested
component, not flatten its coordinates into the surrounding tuple.

```python
P = DiscreteUniform(a=1, b=4) ** 2
X = RV(P, sum)
Y = RV(P, max)
(RV(P) & X & Y).sim(5)
```

**Expected** (and what `main` has always produced):
```
0    ((3, 4), 7, 4)
1    ((1, 1), 2, 1)
...
```

**Currently produced on `dev`:**
```
0    (3, 4, 7, 4)
1    (1, 1, 2, 1)
...
```

The two-component vector `(3, 4)` gets silently flattened into the outer
tuple, and `Z[0]` returns a scalar instead of the original pair.

## Root cause

`join()`, in `symbulate/result.py`, decides whether to unpack a result
before joining it:

```python
a = tuple(result1.values) if isinstance(result1, Tuple) else (result1,)
b = tuple(result2.values) if isinstance(result2, Tuple) else (result2,)
```

`Vector` is a subclass of `Tuple` (`class Vector(Tuple)`), so
`isinstance(vector_instance, Tuple)` is `True` — meaning any
`Vector`-typed sub-result now gets unpacked and flattened into the joined
tuple, exactly like a plain `Tuple` would be. Before this check used
`isinstance`, it used an exact type check, which correctly distinguished
"this is literally a `Tuple`, safe to unpack" from "this is a `Vector` (or
any other `Tuple` subclass), which should stay nested as one opaque
component."

**Introduced by:** commit `478cf244437a8551eef0fe23f9d7dcfa991501b1`,
"fix: syntax and functions in result" (PR #68), June 23, 2026. That commit
made several small, unrelated hygiene fixes in the same file (a
`NotImplemented` guard, `not in` rephrasing, `int()` → `round()`, narrowing
a bare `except:`), and this `join()` change appears to have been bundled
in as a reflexive style improvement (`isinstance` is normally more
correct than an exact `type()` check) without realizing it changes real
flattening behavior for `Vector`. The PR has zero comments and zero
reviews, so there's no recorded discussion of this specific line —
this is an inference from the commit's framing, not a confirmed rationale.

**Confirmed via `git bisect`** between `main` (correct/nested) and `dev`
(flat) that this is the exact commit where the behavior changed, and
confirmed the `main` branch's `join()` still uses the exact-type check
verbatim (`type(result1) == Tuple`), matching the fix below exactly.

**Present on `dev` itself** — not specific to any feature branch. Verified
via `git bisect`, direct `git checkout` + test, and actual
`pip install git+...` into fresh virtual environments (checking the
literal installed file contents and `pip`'s own recorded resolved commit
hash), on both `dev` (`dce6e10f...`) and `feature/hierarchical-spaces`
(`646d3698...`), which branches from `dev`'s tip and does not itself touch
`result.py`.

## The fix

One line, both occurrences, in `symbulate/result.py`:

```diff
- a = tuple(result1.values) if isinstance(result1, Tuple) else (result1,)
- b = tuple(result2.values) if isinstance(result2, Tuple) else (result2,)
+ a = tuple(result1.values) if type(result1) == Tuple else (result1,)
+ b = tuple(result2.values) if type(result2) == Tuple else (result2,)
```

## Regression testing performed

- Reran the **entire current `dev` test suite** (every `test_*.py`
  module, ~2,500+ tests total, including `test_markov_chains` (121 —
  the SIR/SEIR module), `test_distributions` (918), `test_plot` (512),
  `test_hitting_times` (129), `test_poisson_process` (142),
  `test_renewal_process` (83), `test_queues` (122), `test_random_walk`
  (37), `test_time_series` (93), `test_gaussian_process` (193)) with the
  fix applied. **Zero failures anywhere.** None of the process work
  merged since June 23 depends on the flattening behavior the regression
  introduced.
- Added new tests (included in the patch below) that fail against the
  current, unfixed code and pass once the fix is applied — verified both
  directions directly, not just written.

## Patch

```diff
--- a/symbulate/result.py
+++ b/symbulate/result.py
@@ -1776,8 +1776,8 @@ def join(result1, result2):
     >>> join(t1, t2)
     (1, 2, 3, 4)
     """
-    a = tuple(result1.values) if isinstance(result1, Tuple) else (result1,)
-    b = tuple(result2.values) if isinstance(result2, Tuple) else (result2,)
+    a = tuple(result1.values) if type(result1) == Tuple else (result1,)
+    b = tuple(result2.values) if type(result2) == Tuple else (result2,)
 
     return Tuple(a + b)
 
--- a/symbulate/tests/test_random_variables.py
+++ b/symbulate/tests/test_random_variables.py
@@ -401,6 +401,23 @@ class TestRVJoint(unittest.TestCase):
         val = Z.draw()
         self.assertEqual(len(val), 2)
 
+    def test_and_vector_valued_rv_stays_nested(self):
+        # Regression test for the reported issue: joining an RV whose own
+        # draw is itself a Vector (from **) with other RVs must keep that
+        # Vector as one nested component, not flatten it into the outer
+        # tuple. E.g. (RV(P) & X & Y).sim(n) should give ((s1, s2), sum, max),
+        # not (s1, s2, sum, max).
+        seed()
+        P = DiscreteUniform(a=1, b=4) ** 2
+        X = RV(P, sum)
+        Y = RV(P, max)
+        Z = RV(P) & X & Y
+        val = Z.draw()
+        self.assertEqual(len(val), 3)
+        self.assertEqual(len(val[0]), 2)
+        self.assertEqual(val[1], sum(val[0]))
+        self.assertEqual(val[2], max(val[0]))
+
     def test_and_rv_scalar_draw_length(self):
         seed()
         X = RV(Normal(mean=0, sd=1))
--- a/symbulate/tests/test_result.py
+++ b/symbulate/tests/test_result.py
@@ -554,6 +554,46 @@ class TestJoin(unittest.TestCase):
         result = join(Tuple([1, 2]), Scalar(3))
         self.assertEqual(tuple(result), (1, 2, 3))
 
+    def test_join_vector_and_scalar_stays_nested(self):
+        # Regression test: Vector is a Tuple subclass, but join() must not
+        # unpack it -- only an exact Tuple should be flattened. A Vector
+        # result (e.g. from ** or a multivariate distribution draw) should
+        # stay as a single nested component.
+        v = Vector([1, 2])
+        result = join(v, Scalar(3))
+        self.assertEqual(len(result), 2)
+        self.assertIsInstance(result[0], Vector)
+        self.assertEqual(tuple(result[0]), (1, 2))
+        self.assertEqual(result[1], 3)
+
+    def test_join_scalar_and_vector_stays_nested(self):
+        v = Vector([1, 2])
+        result = join(Scalar(0), v)
+        self.assertEqual(len(result), 2)
+        self.assertEqual(result[0], 0)
+        self.assertIsInstance(result[1], Vector)
+        self.assertEqual(tuple(result[1]), (1, 2))
+
+    def test_join_vector_and_vector_stays_nested(self):
+        v1 = Vector([1, 2])
+        v2 = Vector([3, 4])
+        result = join(v1, v2)
+        self.assertEqual(len(result), 2)
+        self.assertIsInstance(result[0], Vector)
+        self.assertIsInstance(result[1], Vector)
+        self.assertEqual(tuple(result[0]), (1, 2))
+        self.assertEqual(tuple(result[1]), (3, 4))
+
+    def test_join_distinguishes_exact_tuple_from_vector_subclass(self):
+        # The core distinction the fix depends on: an exact Tuple flattens,
+        # but a Tuple *subclass* (Vector) does not. If join() is ever
+        # changed to use isinstance(x, Tuple) instead of type(x) == Tuple,
+        # this test will fail, since Vector would then also flatten.
+        flattened = join(Tuple([1, 2]), Scalar(3))
+        nested = join(Vector([1, 2]), Scalar(3))
+        self.assertEqual(len(flattened), 3)
+        self.assertEqual(len(nested), 2)
+
 
 # ---------------------------------------------------------------------------
 # concat
```

## Open item

The PR that introduced this had no discussion recorded, so the intent
behind the `isinstance` change is inferred, not confirmed. Worth a quick
check with the original author before merging the revert, and worth
adding a comment at the fixed line explaining *why* the exact-type check
is deliberate, so a future well-meaning cleanup pass doesn't reintroduce
`isinstance` without realizing it changes real behavior.
