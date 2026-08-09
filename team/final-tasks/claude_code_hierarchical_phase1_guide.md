# Implementing Phase 1 (Hierarchical Models) with Claude Code in VS Code

A step-by-step guide for building the `>>` hierarchical-composition mechanism
in `calpoly-symbulate/symbulate`, using Claude Code inside VS Code. Every
prompt below is grounded in what's actually in the repo today — file paths,
existing docstring conventions, test patterns, and `CLAUDE.md`'s own rules —
so Claude Code has precise context rather than having to guess.

Scope: this covers Phase 1 only — the core `>>` mechanism
(`HierarchicalProbabilitySpace`, `Hierarchical`, `__rshift__`, the `TypeError`
guard, tests, docs). `Placeholder` and `AssumeHierarchical` are included as an
optional stretch step at the end, exactly as the design doc frames them —
not required to ship the core mechanism.

---

## 0. One-time setup (skip if already done)

1. Install the Claude Code extension in VS Code (Extensions panel → search
   "Claude Code" → Install), or install the CLI (`npm install -g
   @anthropic-ai/claude-code`) and use it from VS Code's integrated terminal.
   Sign in with your Anthropic account when prompted.
2. Open the `symbulate` repo as your VS Code workspace folder (`File → Open
   Folder…`).
3. Confirm your Python environment matches `CLAUDE.md`'s "Environment"
   section: Python 3.13, conda env named `symbulate`, package installed in
   editable mode (`pip install -e .` from the repo root). If you haven't
   already: `pip install -e . && pip install pytest black`.
4. Confirm your git remote points at the org repo, not a personal fork of
   the original author's:
   ```
   git remote -v
   ```
   You want `origin` (or `upstream`) pointing at
   `calpoly-symbulate/symbulate`, per `CLAUDE.md`'s explicit note that PRs
   go there, **not** `dlsun/symbulate`. If your remote is wrong, fix it now
   (`git remote set-url origin git@github.com:calpoly-symbulate/symbulate.git`)
   or set up a fork + `upstream` remote if you don't have direct push access
   from this machine.

---

## 1. Get your branch ready

```bash
git checkout dev
git pull origin dev
git checkout -b feature/hierarchical-models
```

This matches `CLAUDE.md`'s documented convention exactly: branch from `dev`,
never commit to `dev`/`main` directly, and use the `feature/description`
naming pattern.

---

## 2. Start Claude Code and confirm it has context

Open a terminal in VS Code (or the Claude Code side panel) at the repo root
and start a session (`claude`, or open the panel). Claude Code will
automatically pick up `CLAUDE.md` — you don't need to paste it in. It will
**not** automatically read the design doc or the decision log, so your first
message should point it there explicitly.

**Prompt 1 — orient, no code yet:**

> Read `team/models-and-sim-design/symbulate_hierarchical_models_design.md`
> in full, and read the "Decision: Phase 3 — Hierarchical Models" section of
> `MODEL-DECISIONS.md`. I want to implement the **lambda-only** version of
> the `>>` operator described there — skip `Placeholder` and
> `AssumeHierarchical` for now, those are separate additions we'll do later
> if at all. Before writing any code, give me a short implementation plan:
> which file(s) change, what the new class/function/dunder method look like,
> and exactly what the `TypeError` guard should check for. Don't write code
> yet — I want to review the plan first.

Read the plan Claude Code gives you. It should mention: a
`HierarchicalProbabilitySpace` class and a `Hierarchical()` function added to
`symbulate/probability_space.py` (not a new file — that's where
`ProbabilitySpace`, `__mul__`, and `__pow__` already live), a new
`__rshift__` method on `ProbabilitySpace`, and a guard that raises when the
conditional function's return value isn't a `ProbabilitySpace` instance. If
it proposes a new module (e.g. `hierarchical.py`) or suggests touching
`Distribution`, `RV`, `__mul__`, or `__pow__`, push back — none of that is
needed for this feature; say so explicitly and ask it to revise the plan.

---

## 3. Build the core mechanism

**Prompt 2 — implement the plan:**

> Implement the plan. Specifically:
>
> 1. Add `HierarchicalProbabilitySpace(ProbabilitySpace)` to
>    `symbulate/probability_space.py`. Its `draw()` should draw from the
>    prior space, call the conditional function on that draw to get the
>    child space, draw from the child space, and join the two outcomes using
>    the existing `join()` helper already imported in that file (it already
>    flattens nested tuples, which is what makes multi-level `>>` chains work
>    without extra code).
> 2. Add a top-level `Hierarchical(prior_space, cond_space_func)` function
>    that constructs a `HierarchicalProbabilitySpace`.
> 3. Add `__rshift__` to `ProbabilitySpace` so `prior >> cond_func` calls
>    `Hierarchical(prior, cond_func)`.
> 4. Add a clear `TypeError` if `cond_space_func(x)` doesn't return a
>    `ProbabilitySpace` — follow `CLAUDE.md`'s error-message philosophy
>    (explain what went wrong *and* how to fix it; the likely mistake is
>    forgetting to wrap a return value in a distribution, so say that).
> 5. Match the existing docstring convention in this file exactly — look at
>    `ProbabilitySpace.__mul__` and `__pow__` just above where you're adding
>    this: `Parameters`/`Returns`/`Raises`/`Examples` sections, with
>    `>>> from symbulate import *` doctest blocks marked `# doctest: +SKIP`
>    wherever the output is random.
>
> Do not modify `__mul__`, `__pow__`, `Distribution`, `RV`, or anything in
> `distributions.py` — none of that is needed for this feature.

**Review before accepting.** In VS Code's diff view, check specifically:

- Nothing changed outside `probability_space.py` in this step.
- `draw()` on `HierarchicalProbabilitySpace` really does call `join()`, not
  hand-roll a tuple concatenation.
- The `TypeError` message reads like the rest of the codebase's errors (a
  sentence explaining the mistake, not a bare type name).
- The docstring's doctest examples are marked `# doctest: +SKIP` if they show
  a specific random output.

If anything's off, tell Claude Code exactly what to fix rather than
re-prompting from scratch — it keeps the rest of the diff intact.

---

## 4. Export from `__init__.py` — pause here first

`CLAUDE.md` lists `symbulate/__init__.py` as public API that should **not**
change without team discussion. This step needs that export, so before
running the prompt below, this is the natural point to loop in the rest of
the team (or at least flag it clearly in your eventual PR description) —
don't treat this step as routine just because Claude Code can do it in one
line.

**Prompt 3 — once you're ready:**

> Export `Hierarchical` and `HierarchicalProbabilitySpace` from
> `symbulate/__init__.py`, matching the existing export pattern in that file
> (check how `AssumeIndependent` or another recent addition like
> `RenewalProcess` is exported, and follow the same style).

---

## 5. Tests

**Prompt 4:**

> Add tests to `symbulate/tests/test_probability_space.py` for the new
> `Hierarchical`/`>>` mechanism, following this file's existing conventions
> exactly: `unittest.TestCase`, the module-level `seed(value=42)` helper
> already defined at the top of the file (which reassigns
> `probability_space.rng`), and `self.assertAlmostEqual(..., delta=0.1)` with
> a large `Nsim` for statistical checks — the same pattern used in
> `test_independence.py`.
>
> Cover:
> 1. The Beta-Binomial example from the design doc:
>    `Beta(1, 2) >> (lambda x: Binomial(10, x))`, checking `E[X] ≈ 1/3` and
>    `E[Y] ≈ 10/3` within tolerance.
> 2. A 3-level chain (e.g. the design doc's nested `Uniform` example) to
>    confirm the accumulated-tuple threading works — check that unpacking
>    `A, B, C = RV(P)` gives three separate random variables and that a
>    downstream lambda can correctly index into the accumulated tuple from
>    an earlier step.
> 3. The discrete-mixture example (a `Bernoulli`-based branching lambda) to
>    confirm branching logic — not just parameter substitution — works.
> 4. The `TypeError` guard: assert it raises when the conditional function
>    returns something that isn't a `ProbabilitySpace` (e.g. a bare number).
> 5. That `Hierarchical(prior, cond_func)` and `prior >> cond_func` produce
>    equivalent results, since one is defined in terms of the other.

Run this test file alone first, before the full suite, so failures are easy
to localize:

```bash
pytest symbulate/tests/test_probability_space.py -v
```

(from the repo root; equivalently `cd symbulate && pytest tests/`, matching
`CLAUDE.md`'s exact phrasing).

---

## 6. Manual smoke test — don't skip this

Automated tests confirm the numbers are close to theory; a quick REPL check
confirms the *syntax* actually reads the way the design doc promised. Open a
Python REPL or scratch notebook with the editable install active and run the
design doc's own worked examples verbatim:

```python
from symbulate import *

# Beta-Binomial
P = Beta(1, 2) >> (lambda x: Binomial(10, x))
X, Y = RV(P)
print(X.sim(10000).mean(), Y.sim(10000).mean())  # ~0.333, ~3.33

# Gamma-Poisson (negative binomial via mixing)
P = Gamma(shape=2, scale=3) >> (lambda lam: Poisson(lam))
L, N = RV(P)
print(N.sim(10000).mean())  # ~6

# Discrete mixture — branching, not substitution
Model = Bernoulli(p=0.5)
def coin_given_model(m):
    return Bernoulli(p=0.5) if m == 0 else Bernoulli(p=0.9)
P = Model >> coin_given_model
M, Xc = RV(P)
print(Xc.sim(10000).mean())  # ~0.7

# Arbitrary-depth chain
P = (Uniform(0, 1)
     >> (lambda a: Uniform(0, a))
     >> (lambda ab: Uniform(0, ab[1])))
A, B, C = RV(P)
print(A.sim(5000).mean(), B.sim(5000).mean(), C.sim(5000).mean())

# TypeError guard — this should raise, not crash deep in .draw()
try:
    (Beta(1, 2) >> (lambda x: 5)).draw()
except TypeError as e:
    print("Guard fired correctly:", e)
```

If any of these don't match the design doc's stated expectations (roughly —
simulation noise is fine, a systematically wrong mean is not), go back to
Claude Code with the specific mismatch rather than re-running the same
prompt.

---

## 7. Docstring/doctest polish pass

**Prompt 5:**

> Run the doctests in `symbulate/probability_space.py` and confirm the new
> `HierarchicalProbabilitySpace`/`Hierarchical`/`__rshift__` docstring
> examples actually pass (or are correctly marked `# doctest: +SKIP`). Also
> double check the `TypeError` guard's message reads naturally to someone
> with minimal Python background, per `CLAUDE.md`'s error-message rule.

```bash
python -m doctest symbulate/probability_space.py -v
```

---

## 8. Update the decision log

This repo tracks build decisions in `MODEL-DECISIONS.md`, and the existing
"Decision: Phase 3 — Hierarchical Models" entry is currently marked
`Status: Proposed`. Update it now that it's real.

**Prompt 6:**

> Update the "Decision: Phase 3 — Hierarchical Models" entry in
> `MODEL-DECISIONS.md`: change `Status: Proposed` to `Status: Implemented`,
> and add a line noting what actually shipped (the lambda-only core
> mechanism — `HierarchicalProbabilitySpace`, `Hierarchical`, `__rshift__`,
> the `TypeError` guard — with `Placeholder`/`AssumeHierarchical` still
> open/deferred). Follow the existing entries' format — don't rewrite
> anything else in the file, and don't delete the "Alternatives Considered"
> or "Open within this phase" sections, just update what's now resolved.

---

## 9. Full test suite, formatting, and a final review pass

```bash
cd symbulate
black .                 # CLAUDE.md: run before every commit
pytest tests/           # full suite, not just the one file
```

If `black` reformats more than the lines you touched, review that diff too
before committing — don't let a formatter silently rewrite unrelated code.

Ask Claude Code for a final self-review before you commit:

**Prompt 7:**

> Show me a diff of everything changed so far in this branch relative to
> `dev`. Confirm nothing outside `probability_space.py`,
> `symbulate/__init__.py`, `symbulate/tests/test_probability_space.py`, and
> `MODEL-DECISIONS.md` was touched, and that `Distribution.__pow__` and
> `ProbabilitySpace.__mul__`/`__pow__` are byte-for-byte unchanged.

```bash
git diff dev --stat
```

---

## 10. Commit and push

`CLAUDE.md`'s commit convention is `type: short description`
(`fix`/`docs`/`feature`/`test`/`refactor`/`style`) and **one PR per task** —
so this is naturally one feature commit (or a small handful, if you'd rather
separate the core mechanism from the tests):

```bash
git add symbulate/probability_space.py symbulate/__init__.py \
        symbulate/tests/test_probability_space.py MODEL-DECISIONS.md
git commit -m "feature: add >> operator for hierarchical probability models"
git push -u origin feature/hierarchical-models
```

You can also just ask Claude Code to do this step:

**Prompt 8:**

> Stage the changed files, write a commit message following this repo's
> `type: short description` convention, commit, and push the branch to
> `origin`.

Claude Code can run git commands directly — review what it's about to commit
(`git status`/`git diff --staged`) before approving, same as any other tool
call.

---

## 11. Open the pull request

**Prompt 9:**

> Open a pull request from `feature/hierarchical-models` targeting `dev` on
> `calpoly-symbulate/symbulate` (not `dlsun/symbulate`). Title: "Add `>>`
> operator for hierarchical probability models". In the body, summarize what
> shipped (link to
> `team/models-and-sim-design/symbulate_hierarchical_models_design.md` and
> the updated `MODEL-DECISIONS.md` entry), list the new public API
> (`Hierarchical`, `HierarchicalProbabilitySpace`, `>>` on
> `ProbabilitySpace`), note explicitly that `Placeholder` and
> `AssumeHierarchical` are deliberately out of scope for this PR, and include
> a short test-plan checklist (unit tests added, full suite passing,
> doctests passing, manual smoke test against the design doc's worked
> examples).

This runs `gh pr create` under the hood if the GitHub CLI (`gh`) is
authenticated in your terminal — check with `gh auth status` first if you
haven't used it before. If `gh` isn't set up, ask Claude Code to give you the
PR title/body text instead and open the PR manually on github.com.

---

## 12. Responding to review

When you get review comments, feed them back in directly — Claude Code
doesn't need you to translate them:

**Prompt 10 (example):**

> A reviewer left this comment on the PR: "[paste the exact comment]".
> Address it, show me the diff, and don't touch anything else in the file.

Re-run the relevant slice of Section 9 (tests + `black`) after any review
round, then push again:

```bash
git add -u
git commit -m "fix: address review feedback on >> operator"
git push
```

---

## Quick-reference: all prompts in order

1. *Plan* — read the design doc + decision log, propose a plan, no code.
2. *Implement* — `HierarchicalProbabilitySpace`, `Hierarchical`,
   `__rshift__`, `TypeError` guard, matching docstring style.
3. *Export* — add to `__init__.py` (pause for team sign-off first).
4. *Tests* — Beta-Binomial, 3-level chain, discrete mixture, `TypeError`
   guard, `Hierarchical`/`>>` equivalence.
5. *Doctest pass* — confirm docstring examples actually run or are skipped.
6. *Decision log* — flip Phase 3's status to Implemented in
   `MODEL-DECISIONS.md`.
7. *Self-review diff* — confirm scope stayed inside the four expected files.
8. *Commit + push* — one commit, `feature:` prefix.
9. *Open PR* — target `dev` on `calpoly-symbulate/symbulate`.
10. *Address review* — paste comments back in verbatim, re-test, re-push.

## Troubleshooting

- **Claude Code proposes touching `distributions.py`, `__pow__`, or
  `__mul__`:** stop and ask why — nothing in this feature needs it. If it
  can't give a concrete reason tied to the design doc, it's scope creep.
- **A test fails with a mean that's off by more than noise:** check the
  lambda's argument order first — the most common bug in a hierarchical
  chain is indexing the wrong element of the accumulated tuple (`xy[0]` vs
  `xy[1]`), exactly the ambiguity the design doc flags in its own examples.
- **`pytest tests/` can't find the tests:** you're probably running it from
  the repo root instead of inside `symbulate/` — either `cd symbulate` first
  or run `pytest symbulate/tests/` from the root.
- **PR opened against the wrong repo:** `gh repo set-default
  calpoly-symbulate/symbulate` once, then retry `gh pr create`.
