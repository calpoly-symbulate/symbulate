# Histogram / impulse plots for a single sample path's states

**Question:** For a Symbulate process with jumps (e.g. a continuous-time
Markov chain, or `MM1`), `X.apply(states).sim(1)` gives the sequence of
states visited along one sample path, and `.plot()` currently draws that
sequence as a line against its jump index (`0, 1, 2, ...`). Can `.plot()`
support `type="hist"` / `type="impulse"` for this object, and — since the
path is lazily evaluated and technically infinite — how would the code
decide how many states to include?

**Short answer:** Yes. It's a small, self-contained addition confined to
the plotting layer. It does not require any changes to the lazy-evaluation
core (`InfiniteTuple` / `InfiniteVector` / `TimeFunction`) or to
`sim()` / `draw()` / `apply()`. The only real design decision — not an
implementation difficulty — is what "how many states" should mean:
first *N* jumps, or all jumps that occurred by some elapsed time *T*.

Repo referenced: `calpoly-symbulate/symbulate`, `dev` branch.

---

## 1. How the current code path works

1. `X.apply(states)` (see `states()` in `symbulate/math.py`) just returns
   `path.get_states()`, which is the *same* `InfiniteVector` object stored
   as `.states` on the process's result class (e.g.
   `ContinuousTimeMarkovChainResult` in `symbulate/markov_chains.py`,
   line ~357):

   ```python
   self.states = InfiniteVector(lambda n: self.state_labels[self.state_indices[n]])
   ```

   It is indexed by **jump number** `n = 0, 1, 2, ...`, not by clock time.
   `interarrival_times` and `get_arrival_times()` (the holding time / clock
   time of each jump) live on the *parent* result object, not on this
   `InfiniteVector` — once you call `.apply(states)`, that connection is
   gone.

2. `.sim(1)` wraps it in an `RVResults` of length 1 (`symbulate/results.py`,
   `RVResults.__init__`, ~line 1163). Because the outcome isn't a plain
   number/vector, `self.dim` is `None`, but `self.index_set` is set because
   the outcome is a `TimeFunction`.

3. `RVResults.plot()` (`symbulate/results.py`, def at line 1904) has a
   final catch-all branch for `self.dim is None` / non-categorical data
   (~line 2757-2772):

   ```python
   else:
       ...
       for result in self.results:
           result.plot(alpha=alpha, color=color, **kwargs)
   ```

   **This branch never forwards `type=`.** It's a separate named parameter
   on `plot()`, not part of `**kwargs`, so today `type="hist"` is silently
   dropped for this kind of result.

4. The actual per-path drawing happens in `InfiniteVector.plot()`
   (`symbulate/result.py`, line 899):

   ```python
   def plot(self, tmin=0, tmax=10, **kwargs):
       xs = range(tmin, tmax)
       ys = [self[t] for t in range(tmin, tmax)]
       ...
       make_sample_path(xs, ys, ax, color, **kwargs)
   ```

   `tmin`/`tmax` here are **jump-index bounds** (default: first 10 jumps),
   and the function is hardcoded to call `make_sample_path` — there is no
   `type=` dispatch at this level at all.

5. `make_hist` and `make_impulse` (both in `symbulate/plot.py`, lines 3078
   and 2898) already exist and are exactly what `RVResults.plot()` calls
   for an ordinary `X.sim(n).plot(type="hist")` on numeric/categorical
   data. They take a flat array of values plus axes/color and are generic
   — they don't care that the values came from one path instead of `n`
   independent draws.

6. `make_impulse` currently computes frequencies by hand
   (`count_var(codes)`, ~line 2979) — it has **no `weights=` support**.
   `make_hist` delegates to `ax.hist(...)`, which *does* support
   `weights=` natively, but nothing currently plumbs a `weights` argument
   into it from `InfiniteVector.plot()`.

---

## 2. Concrete code changes

### 2a. Thread `type=` through the path branch of `RVResults.plot()`

File: `symbulate/results.py`, inside `RVResults.plot()`, the final `else`
branch (~line 2757).

```python
else:
    ...
    for result in self.results:
        result.plot(type=type, alpha=alpha, color=color, **kwargs)
```

Also update the docstring's list of valid `type` values / validation logic
near the top of `plot()` so `"hist"` / `"impulse"` are documented as valid
for path-shaped (`TimeFunction`) results, not just scalar/vector ones.

### 2b. Add `type=` dispatch to `InfiniteVector.plot()`

File: `symbulate/result.py`, `InfiniteVector.plot()` (line 899).

```python
def plot(self, tmin=0, tmax=10, type=None, weights=None, **kwargs):
    xs = range(tmin, tmax)
    ys = [self[t] for t in range(tmin, tmax)]
    ax = plt.gca()
    color = kwargs.pop("color", None)
    if color is None:
        color = get_next_color(ax)

    if type in ("hist", "impulse"):
        w = None
        if weights is not None:
            w = [weights[t] for t in range(tmin, tmax)]
        if type == "hist":
            make_hist(ys, ax, color, weights=w, **kwargs)
        else:
            make_impulse(ys, ax, color, weights=w, **kwargs)
    else:
        kwargs.setdefault("xlabel", "Index")
        make_sample_path(xs, ys, ax, color, **kwargs)
    return SymbulatePlot(ax)
```

This reuses the exact same value-gathering logic already in place (`ys`
over `range(tmin, tmax)`); it just routes to a different renderer. No new
plot primitives are needed — `make_hist` / `make_impulse` already exist
and are already used elsewhere for the same visual output.

Do the analogous thing in `DiscreteTimeFunction.plot()` (line 1281) and
`ContinuousTimeFunction.plot()` (line 1538) only if hist/impulse should
also be offered for a *time-indexed* path evaluated at dense time points —
see the caveat in section 3 about why that's a worse fit than the
jump-indexed case.

### 2c. (Optional, for the time-weighted version) Add `weights=` support to `make_impulse`

File: `symbulate/plot.py`, `make_impulse()` (line 2898). It currently does:

```python
counts = count_var(codes)
xs = sorted(counts.keys()) if categories is not None else list(counts.keys())
freqs = [counts[x] for x in xs]
```

This needs to become a weighted sum instead of a plain count when
`weights` is provided, e.g. accumulate `sum(weights[i] for i where codes[i] == x)`
per distinct `x` instead of `counts[x]`. `make_hist` needs no change here
since `ax.hist(..., weights=weights)` already works — just pass `weights`
through in `make_hist`'s signature and down to the `ax.hist(...)` call.

### 2d. (Optional, for time-bounded rather than jump-bounded paths) A jump-count-by-time helper

This is the piece that answers "how do we know how many states to plot
along an infinite path" when the bound should be a **clock time** `T`
rather than a **jump count** `N`. It only matters if you plot from the
*full path* (which still has `interarrival_times` / `get_arrival_times()`)
rather than from the bare object returned by `.apply(states)`.

Add a small method, e.g. on `DiscreteValued`
(`symbulate/result.py`, ~line 1617), modeled directly on the existing
forward-walk in `ContinuousTimeMarkovChainResult.__init__`'s `_func`
(`symbulate/markov_chains.py`, lines 366-374):

```python
def num_jumps_by(self, t):
    """Return how many jumps of this path occurred by time t."""
    arrival_times = self.get_arrival_times()
    n = 0
    while arrival_times[n] <= t:
        n += 1
    return n
```

Then a path-level convenience (e.g. on `ContinuousTimeMarkovChainResult`,
or a shared mixin) could accept a `tmax` in *time* units for a states
histogram and convert it internally:

```python
n = path.num_jumps_by(tmax)
states(path)[0:n].plot(type="hist", weights=... )
```

This is new code, but small (~10 lines) and directly mirrors a pattern
already in the codebase.

---

## 3. Design decisions to make (not implementation cost)

These matter more than the code itself:

1. **What does "tmax" mean for a states histogram?**
   - On the object returned by `.apply(states)` alone: naturally means
     "first N jumps," since that's the only information available (no
     `arrival_times` survive `.apply(states)`). Cheapest option — matches
     today's existing `tmin`/`tmax` semantics on `InfiniteVector.plot()`
     exactly.
   - On the full path object: could mean "all jumps by elapsed time T,"
     which is more intuitive for a continuous-time process but requires
     keeping the whole path (not just `.apply(states)`) and the
     `num_jumps_by` helper from 2d.

2. **Per-jump count vs. time-weighted histogram — these answer different
   questions.** A plain histogram of "the first N states visited," one
   count per jump, estimates the distribution of the **embedded jump
   chain**. It does *not* estimate the fraction of wall-clock time the
   process spent in each state, because holding times differ by state
   (e.g., in `MM1`, the time spent at state 0 has a different rate than at
   a congested state). If the goal is "what fraction of time was the
   queue in each state along this path," the histogram needs to weight
   each visited state's bar by its holding time (`interarrival_times[n]`),
   not count it once — this is the `weights=` path in 2c above. Worth
   surfacing this distinction explicitly in the docstring/tutorial so
   users don't assume the naive per-jump histogram is a time-average.

3. **Time-indexed (not jump-indexed) hist/impulse is a worse fit.**
   `ContinuousTimeFunction.plot()` currently samples 200 evenly-spaced
   points in `[tmin, tmax]` and evaluates the (step-function) path at
   each. Histogramming *those* 200 samples would approximate a
   time-weighted histogram only as the sample density goes to infinity,
   and would jitter based on where the samples happen to fall relative to
   jump boundaries. The jump-indexed route (2b/2d, driven by
   `interarrival_times` as exact weights) gives an exact answer instead of
   an approximation, so it's the recommended path even for continuous-time
   processes.

---

## 4. Overall scope estimate

| Piece | File | Size | Required for basic feature? |
|---|---|---|---|
| Forward `type=` through path branch | `results.py`, `RVResults.plot()` | ~1 line + docstring update | Yes |
| `type=` dispatch in `InfiniteVector.plot()` | `result.py` | ~15 lines | Yes |
| Same dispatch in `DiscreteTimeFunction.plot()` / `ContinuousTimeFunction.plot()` | `result.py` | ~15 lines each | No (optional; see §3.3) |
| `weights=` support in `make_impulse` | `plot.py` | ~10 lines | Only for time-weighted histograms |
| `weights=` passthrough in `make_hist` | `plot.py` | ~2 lines | Only for time-weighted histograms |
| `num_jumps_by(t)` helper | `result.py`, `DiscreteValued` | ~8 lines | Only for time-bounded (not jump-bounded) plots |

No changes are needed to `InfiniteTuple`/`InfiniteVector`'s lazy
evaluation, `RV.apply()`, `.sim()`/`.draw()`, or any probability-space
class. This is a plotting-layer feature that reuses existing rendering
helpers (`make_hist`, `make_impulse`) end to end — the work is mostly
wiring plus the `weights=`/time-vs-jump-count decisions above.
