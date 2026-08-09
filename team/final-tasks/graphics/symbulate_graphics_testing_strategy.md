# Testing Strategy for Symbulate Graphics

*Grounded in the actual `dev` branch (cloned and run against directly — every claim and code sample below was executed, not just reasoned about) as of August 2026.*

## The core problem, restated precisely

`test_plot.py` (5,200+ lines) is already unusually thorough. It asserts *specific, known* properties of *specific* calls — "`Normal(0,1).plot()` draws exactly one line," "a pmf's y-values are non-negative," and so on, across dozens of test classes. That style of test is precise, but it has a structural limit: it only catches what someone already thought to check, one call at a time. It can't scale to "does `.plot()` look reasonable across every distribution × every parameter regime × every kwarg combination," and by construction it cannot catch a failure mode nobody has written an assertion for yet — which is exactly the category your `Normal().plot(cdf=True)` example falls into.

So the right frame isn't "add more tests like the existing ones" — it's **add a second, complementary layer** that doesn't require knowing in advance what could go wrong. That layer has three parts:

1. A generic **"is this actually a plot" detector** — catches blank/near-blank renders regardless of *why* they're blank.
2. A **table-driven sweep** that runs every distribution (and eventually every plot type) through that detector, plus "did it raise, did it warn," so you get combinatorial coverage without combinatorial hand-written assertions.
3. A small number of **guards added to the library itself**, not just to tests — because a test suite only tells you *after the fact* that something's wrong; a guard stops it from reaching the student in the first place. Two concrete guards are proposed below, both reproducing and fixing real bugs found while writing this.

## What I found by actually running it

Before writing the recommendation, I ran the current `dev` branch against itself. Two distinct failure modes turned up, both real:

**1. Your `cdf=True` example — confirmed and diagnosed.** `Normal(0,1).plot(); Normal(0,1).plot(cdf=True)` overlays a pdf and a cdf on one axes. Checking the resulting axes directly:

```
>>> ax.get_title()
'CDF Plot'
>>> ax.get_ylabel()
'Density'
```

The title comes from the second call (it always overwrites), but the y-axis label is `"Density"` from the *first* call — because `Distribution.plot()` deliberately only sets an axis label when one isn't already set (so a theoretical curve overlaid on simulated data doesn't clobber that plot's own label — see `DECISIONS.md`, "Distribution Plotting Window"). That's the right rule for pdf-onto-histogram overlays; it's exactly the wrong rule when the second curve is a *different function of the same variable*. The result is two curves with incompatible y-scales (a density that can exceed 1, and a probability bounded in [0,1]) sharing one axis, titled and labeled in a way that doesn't describe either curve correctly. This is worse than a blank plot: it looks complete and confidently mislabels itself.

I checked whether the project's own Overlay Policy (`CLAUDE.md`, `DECISIONS.md`) already covers this — it doesn't. The three-tier policy (natural overlay / readability warning / hard error) was designed around *simulated-data* overlays (two tiles, two histograms, `marginal=True` layouts) and the pdf-onto-simulated-data case. A pdf/cdf-on-the-same-distribution overlay was never one of the cases considered, which is exactly why it slips through — mechanically the two calls share an axes just fine, so nothing objects.

**2. Degenerate parameters crash with raw, unhelpful errors.** Fuzzing `Distribution.plot()` toward parameter boundaries (`n=0`, `p∈{0,1}`, `sd→0`, `shape→0`, `a==b`, etc. — the kind of values a curious student *will* type) turned up real crashes:

```
Normal(mean=0, sd=0).plot()
  -> ValueError: zero-size array to reduction operation maximum which has no identity

Uniform(a=5, b=5).plot()
  -> ValueError: zero-size array to reduction operation maximum which has no identity

Gamma(shape=1e-6, rate=1).plot()
  -> ValueError: zero-size array to reduction operation maximum which has no identity

Pareto(shape=1e-3, scale=1).plot()
  -> ValueError: Axis limits cannot be NaN or Inf
```

All four come from the same place: `distributions.py:499`, `ymax = ys[np.isfinite(ys)].max()`. When every evaluated y-value is non-finite (a degenerate/zero-variance distribution, or a shape parameter so extreme the pdf overflows or underflows to `nan`/`inf` everywhere in the plotting window), `ys[np.isfinite(ys)]` is empty and `.max()` throws a bare NumPy internals error straight at the student. This is a direct violation of `CLAUDE.md`'s own **Error Message Standard** ("Error messages must say what went wrong and how to fix it… never expose a raw Python exception to a student without context") — it's just that nobody has hit these particular parameter values in a test yet.

Neither of these is a hypothetical "there could be many ways plots could be wrong" — they're two concrete ways, reproduced, with the exact line numbers.

## Layer 1: a generic blank-plot detector

Your instinct — "check if the plot is blank, raise a catch-all warning" — is right, but the obvious version of it doesn't work. `ax.has_data()` says a plot is non-blank as soon as *any* artist was added, even if that artist renders nothing:

| Call | `ax.has_data()` | Actually visible? |
|---|---|---|
| `ax.plot([], [])` | `True` | No |
| `ax.plot([nan]*5, [nan]*5)` | `True` | No |
| `ax.hist([])` (10 zero-height bars) | `True` | No |
| `ax.bar([], [])` | `False` | No |
| `ax.plot([1,2,3],[1,4,9])` | `True` | Yes |

So `has_data()` and "count the artists" both miss real blank-plot cases and can't be trusted as a catch-all — they're checking whether something was *attempted*, not whether anything is *visible*. The reliable check is to render the figure and look at the actual pixels:

```python
def axes_is_blank(ax, shrink=0.06, pixel_tol=8, area_tol=0.0005):
    """Render the figure and check whether anything is visible inside `ax`.

    Looks at rendered pixels rather than artist objects, so it catches a
    blank plot regardless of *why* it's blank (empty arrays, all-NaN data,
    zero-height bars, alpha=0, size=0 markers, ...) without needing a
    separate check per failure mode.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    buf = np.asarray(renderer.buffer_rgba())

    bbox = ax.get_window_extent(renderer=renderer)
    x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    dx, dy = (x1 - x0) * shrink, (y1 - y0) * shrink   # shrink inward, away from spines
    x0, x1, y0, y1 = x0 + dx, x1 - dx, y0 + dy, y1 - dy

    h = buf.shape[0]
    row0, row1 = int(h - y1), int(h - y0)
    col0, col1 = int(x0), int(x1)
    region = buf[max(row0, 0):row1, max(col0, 0):col1, :3]
    if region.size == 0:
        return True

    bg = np.array(fig.get_facecolor())[:3] * 255
    diff = np.abs(region.astype(int) - bg.astype(int)).max(axis=-1)
    return float((diff > pixel_tol).mean()) < area_tol
```

I validated this against the table above plus real Symbulate plots before recommending it:

| Case | `axes_is_blank` |
|---|---|
| truly empty axes | **True** |
| `plot([], [])` | **True** |
| all-NaN line | **True** |
| `bar([], [])` | **True** |
| `hist([])` | **True** |
| `Normal(0,1).plot()` (real) | False |
| `Binomial(10,.4).sim(500).plot()` (real) | False |
| single small scatter point (sparsest real case tested) | False (0.00026 vs. 0.0005 threshold — has margin, but not a lot; see note below) |

Two honest caveats, so this doesn't get treated as more precise than it is:

- The `shrink=0.06` and `area_tol=0.0005` values were tuned against default `matplotlib` figure size/DPI. If Symbulate's `.mplstyle` uses a notably different default figure size, re-run the calibration above against it before trusting the threshold.
- It's a *blunt* instrument by design — it tells you "nothing rendered," not "the wrong thing rendered." It would not have caught the `cdf=True` overlay bug (that plot is very much non-blank — two full curves are drawn, just mislabeled). That's expected and fine: blank-detection and semantic guards are two different layers solving two different problems, not one solving both. Don't reach for a fancier blank-detector to try to catch semantic bugs — reach for Layer 3 instead.

## Layer 2: table-driven sweep, not hand-enumeration

A blank-detector is only useful if it's run against enough cases to matter. Hand-writing a test per distribution (which `test_plot.py` mostly does today for the handful it covers) doesn't scale and silently misses new distributions. The fix already has a precedent in this codebase: `test_continuous_time_processes.py` asserts a required interface *table-driven* over every process, specifically so a new process can't be added without also being covered. The same pattern applies directly here.

I built this out and ran it against all 55 univariate `Distribution` subclasses (via `Distribution.__subclasses__()` introspection, so it can't silently miss a class added later) — full code in `test_plot_regression_sweep.py`, delivered alongside this document. Structure:

- `REGISTRY`: one entry per distribution with minimal valid constructor kwargs.
- `SKIPPED`: distributions deliberately excluded, each with a stated reason (currently just the three matrix-variate ones — `Hotelling`, `Wishart`, `InverseWishart` — which aren't really 1-D and need their own coverage).
- `test_registry_covers_every_univariate_distribution`: fails the build if a class is in neither dict. This is the piece that keeps the sweep from quietly rotting as distributions are added — the same contract `test_continuous_time_processes.py` already enforces elsewhere.
- `TestDistributionPlotSweep`: for every registered distribution, both `.plot()` and `.plot(cdf=True)` must not raise, not warn, and not render blank (via `axes_is_blank`).

Result against current `dev`, with default constructor parameters: **0 failures across 104 calls (52 distributions × 2 plot modes)**. That's a genuinely useful data point — the baseline pdf/pmf/cdf rendering is solid; the bugs are in *combinations* (overlay, edge-case parameters), which is exactly what motivated pushing further:

- **Parameter fuzzing toward boundaries** (a lightweight version of property-based testing, no `hypothesis` dependency required to start) is what actually found the four degenerate-parameter crashes above. I'd treat this as the next thing to build out, not `REGISTRY` itself — walk each distribution's parameters toward 0, toward its stated valid-range edges, and toward large values, running the same three checks. This is the practical way to cover "many ways plots could be wrong" without hand-writing a case for each one; you're delegating case *generation* to a systematic sweep and only hand-writing the *check*.
- If a boundary sweep turns up more than a handful of crashes, that's the point to bring in `hypothesis` for real — `@given` a parameter strategy per distribution family (positive floats for shape/scale/rate, `[0,1]` for probabilities, etc.) and let it shrink failures to minimal reproductions automatically, rather than manually guessing which boundary values matter.

## Layer 3: guards in the library, not just in tests

A test suite tells you something broke; it doesn't stop the plot from reaching a student mid-lab. Both bugs found above are exactly the shape CLAUDE.md already has conventions for — so the fixes follow those conventions rather than inventing new ones.

**Guard 1 — friendly error on non-finite everywhere.** Same spot the crash comes from (`distributions.py`, in `plot()`, right after `ys` is computed):

```python
ys = self.cdf(xs) if cdf else self.pdf(xs)

finite = np.isfinite(ys)
if not np.any(finite):
    raise ValueError(
        f"{type(self).__name__}'s parameters make the "
        f"{'cdf' if cdf else 'pdf/pmf'} undefined or infinite everywhere in "
        f"the plotting window {tuple(xlim)}. This usually means a parameter "
        f"is at a degenerate edge (e.g. zero variance, a shape parameter at "
        f"0, or bounds that collapse to a single point) rather than a bug "
        f"in your code -- check the distribution's parameters."
    )

ymax = ys[finite].max()
```

This follows the Error Message Standard's own template (what went wrong + how to fix it) and costs one `np.any()` check on data already being computed — no new dependency, no behavior change for any of the 104 sweep cases that currently pass.

**Guard 2 — stop the pdf/cdf overlay from silently mislabeling itself.** The codebase already has a working pattern for exactly this shape of problem: tag the axes/figure with a private attribute when a plot is drawn, check the tag before drawing again, and raise a named, student-facing error constant. `MARGINAL_OVERLAY_ERROR` and the `_symbulate_marginal` / `_symbulate_freq_ticks` / `_symbulate_value_lim` tags in `plot.py` are the precedent — this reuses that idiom rather than adding a new one:

```python
# near the other overlay-error constants in plot.py
THEORETICAL_CDF_PDF_OVERLAY_ERROR = (
    "This axes already has a {existing} plotted on it, and you're now "
    "trying to add a {new}. A density/mass curve and a cumulative "
    "distribution curve use different y-axis scales (one can exceed 1, the "
    "other runs from 0 to 1), so overlaying them on the same axes produces "
    "a plot that can't be read correctly. Draw them in separate cells, or "
    "pass ax= to place them in your own subplots (e.g. via plt.subplots())."
)
```

```python
# in Distribution.plot(), right after `ax` is resolved (ax = ax or plt.gca())
existing = getattr(ax, "_symbulate_theoretical_kind", None)
new_kind = "cdf" if cdf else "pdf/pmf"
if existing is not None and existing != new_kind and ax.has_data():
    raise ValueError(
        THEORETICAL_CDF_PDF_OVERLAY_ERROR.format(existing=existing, new=new_kind)
    )
...
# at the end of plot(), right before returning
ax._symbulate_theoretical_kind = new_kind
```

I validated all three behaviors this needs to preserve before recommending it:

- Same-kind overlay still works (`Normal(0,1).plot(); Normal(2,1).plot()` → 2 lines, no error) — matches existing `test_two_distributions_overlay_on_one_axes`.
- Mismatched overlay now raises instead of silently drawing garbage.
- The escape hatch works for anyone who *does* want both curves side by side: `ax2 = ax1.twinx(); Normal(0,1).plot(ax=ax1); Normal(0,1).plot(ax=ax2, cdf=True)` — different `Axes` objects, so the tag on `ax1` never conflicts with `ax2`. No new keyword needed to unlock this; it falls out of the tag being per-Axes.

One open call for the team, not something I'd resolve unilaterally: this is written as a **hard error**, the strictest of the three overlay tiers. I'd default to that given the teaching context (a silently mislabeled plot actively teaches something wrong, which is worse than the other warning-tier cases like two overlapping tiles, which are just hard to read) — but it's a genuine product decision the same way the marginal-layout tier was, and belongs in `DECISIONS.md` alongside the rest of the Overlay Policy rather than being decided by whoever happens to write the patch.

## What I'd deliberately *not* do (yet)

- **`pytest-mpl` golden-image regression testing** — comparing rendered output pixel-for-pixel against saved baseline images. This is the right tool for catching *unintended visual regressions* between PRs (e.g., a refactor of `make_hist` accidentally shifts bar alpha), but it's expensive to maintain during a period where the visual design is still actively being decided (per `DECISIONS.md`, most of the plot lookup table and styling is explicitly "provisional"). Every deliberate style change means regenerating baselines. I'd introduce this once the visual design stabilizes, not now — right now it would mostly generate noise, not signal.
- **Full `hypothesis` property-based testing across every distribution family** — powerful, but a bigger lift (writing a parameter strategy per family, tuning `@given` settings) than the team needs to pay for right now. The lightweight boundary-fuzzing in Layer 2 gets most of the value for a fraction of the setup cost; escalate to `hypothesis` if that sweep keeps finding new failures.
- **CI enforcement** — worth flagging: there's no `.github/workflows/` in the repo yet, so none of this runs automatically on a PR today; it only runs when someone remembers `pytest tests/` locally. That's a separate, smaller task (one `pytest` workflow file) but it's the difference between "these tests exist" and "these tests actually gate merges" — I'd treat wiring that up as a near-term follow-on to whichever of the above the team picks up first.

## Suggested next steps, in priority order

1. Add Guard 1 (friendly error on non-finite `ys`) and its regression test — smallest change, fixes a real crash, no design decision required.
2. Bring the pdf/cdf overlay question (Guard 2) to the team as a `DECISIONS.md` entry — the mechanism is validated and ready to drop in; the only open question is which overlay tier it belongs to.
3. Land `test_plot_regression_sweep.py` (delivered alongside this doc) as-is, then extend its parameter-boundary fuzzing pass across `REGISTRY` to look for more crashes of the same shape as Guard 1's.
4. Add a bare-bones `pytest` GitHub Actions workflow so all of the above actually gates merges instead of relying on someone remembering to run it.
