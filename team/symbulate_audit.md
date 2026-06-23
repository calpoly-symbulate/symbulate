# Symbulate Codebase Compatibility Audit — `dev` Branch
*June 2026 — Python 3.13.9, NumPy 2.4.4, Matplotlib 3.10.8, SciPy current*

---

## What's Changed Since the `main` Branch Audit

Three of the five original issues have been fixed on `dev`:

| Issue | `main` | `dev` |
|---|---|---|
| `get_next_color()` — removed `.prop_cycler` attribute | 🔴 Broken | ✅ Fixed (PR #49) |
| `init_color()` — deprecated `plt.cm.get_cmap()` | 🔴 Broken | ✅ Fixed (PR #49, uses `matplotlib.colormaps`) |
| `stats.frechet_r` removed from SciPy | 🔴 Broken | ✅ Fixed (removed from tests) |
| Seaborn stylesheet name — fragile fuzzy match | 🟡 Fragile | 🟡 Still fragile |
| `hist2d` normalize — `int(label.get_text())` on float string | 🟡 Warning | 🟡 Still present |
| `violinplot(vert=...)` — deprecated keyword | 🟡 Warning | 🟡 Still present |
| Legacy `np.random` API — 13 call sites | 🟡 Pending | 🟡 Still present |
| Multivariate PDF bug — returns object not float | 🔴 Wrong | 🔴 Still present |

---

## 🟡 Remaining Issues

### 1. Seaborn stylesheet name — fragile fuzzy match (`results.py:45–49`)

```python
seaborn_colorblind_closest = difflib.get_close_matches(
    'seaborn-colorblind', plt.style.available, cutoff=0.7
)
stylesheet = seaborn_colorblind_closest[0]   # IndexError if list is empty
```

Matplotlib renamed `'seaborn-colorblind'` to `'seaborn-v0_8-colorblind'` in version 3.6. The fuzzy match currently finds it (similarity score 0.878, well above the 0.7 cutoff), so this is working. But it is one Matplotlib version bump away from silently breaking, and it will raise an unhandled `IndexError` rather than degrading gracefully. It is also slow — `difflib.get_close_matches` runs on every module import.

**Fix (3 lines):**
```python
stylesheet = (
    'seaborn-v0_8-colorblind'
    if 'seaborn-v0_8-colorblind' in plt.style.available
    else 'ggplot'
)
```

---

### 2. `init_color()` still uses `cm.get_cmap()` — deprecated in Matplotlib 3.9 (`plot.py:22`)

PR #49 correctly switched from `plt.cm.get_cmap()` (already deprecated) to importing `colormaps as cm` from matplotlib. However, it still calls `cm.get_cmap()`, which is also deprecated in favor of `cm['tab10']` or `cm.get_cmap()` on the `ColormapRegistry`.

Currently this does **not** raise a warning because `ColormapRegistry.get_cmap()` is the new API method, not the old `plt.cm.get_cmap()`. But the intent of the fix was to use the dictionary-style access, and the function is also now unnecessary since `init_color()` is never called at module load and the PR's new color cycling approach in `get_next_color()` doesn't depend on it. The function should either be removed or simplified.

**Fix (cleaner, removes the dead function):**
```python
# init_color() can be deleted entirely — get_next_color() now reads
# directly from plt.rcParams["axes.prop_cycle"] via itertools.cycle
```

---

### 3. `violinplot(vert=...)` keyword deprecated (`plot.py:139`)

```python
violins = ax.violinplot(
    dataset=values, showmedians=True, vert=False if axis == 'y' else True
)
```

The `vert` keyword was deprecated in Matplotlib 3.9 in favor of `orientation`. It raises a `PendingDeprecationWarning` on every violin plot, and will become an error in a future release.

**Fix:**
```python
orientation = 'horizontal' if axis == 'y' else 'vertical'
violins = ax.violinplot(dataset=values, showmedians=True, orientation=orientation)
```

---

### 4. `hist2d` normalize labels — `int()` on float string (`results.py:1272`)

```python
for label in caxes.get_yticklabels():
    new_labels.append(int(label.get_text()) / len(x))  # 'int' raises on '0.0'
```

Matplotlib now formats colorbar tick labels as floats (`'0.0'`, `'50.0'`, etc.) rather than integers. Calling `int()` on these strings raises a `ValueError`. Separately, `set_yticklabels()` on an auto-scaled axis also raises a `UserWarning` about using a `FixedLocator` first.

This does **not** crash during normal use because the `ValueError` is only triggered at specific tick label values. But it is unreliable and the normalization logic is also incorrect — it divides the count-axis labels by `len(x)` rather than computing a proper density.

**Fix (replaces the fragile label-manipulation approach):**
```python
if normalize:
    # Re-plot using density=True directly instead of rescaling labels after
    ax.cla()
    histo = ax.hist2d(x, y, bins=bins, cmap='Blues', density=True)
    caxes = add_colorbar(fig, type, histo[3], 'Density')
else:
    caxes = add_colorbar(fig, type, histo[3], 'Count')
```

---

### 5. Legacy `np.random` API — 13 call sites across 5 files

The legacy NumPy random API (`np.random.choice`, `np.random.normal`, etc.) is used for all custom `draw()` overrides and for jitter in plots. NumPy 1.17 introduced `np.random.default_rng()` as the preferred interface. The legacy functions are not yet removed but generate `DeprecationWarning` in strict mode and cannot be seeded independently, which makes tests less reproducible.

**Remaining call sites in production code (not docstrings):**

| File | Line | Call |
|------|------|------|
| `distributions.py` | 480 | `np.random.negative_binomial(...)` — `NegativeBinomial.draw()` |
| `distributions.py` | 1119 | `np.random.standard_cauchy()` — `Cauchy.draw()` |
| `distributions.py` | 1246 | `np.random.pareto(...)` — `Pareto.draw()` |
| `distributions.py` | 1358 | `np.random.multivariate_normal(...)` — `MultivariateNormal.draw()` |
| `distributions.py` | 1539 | `np.random.multinomial(...)` — `Multinomial.draw()` |
| `results.py` | 1160 | `np.random.uniform(...)` — jitter in impulse plot |
| `results.py` | 1170 | `np.random.normal(...)` — rug noise |
| `results.py` | 1255–1258 | `np.random.normal(...)` ×2 — jitter in scatter plot |
| `markov_chains.py` | 93 | `np.random.choice(...)` — initial state |
| `markov_chains.py` | 102 | `np.random.choice(...)` — transition |
| `gaussian_process.py` | 172 | `np.random.multivariate_normal(...)` |
| `probability_space.py` | 428 | `np.random.choice(...)` — `BoxModel.draw()` |

Note: The `np.random.seed(42)` calls in `test_distributions.py` also use the legacy API and will not seed the new Generator-based functions if those are adopted.

**Recommended fix:** Add a module-level `rng = np.random.default_rng()` to each affected file and replace all legacy calls with `rng.choice(...)`, `rng.normal(...)`, etc.

---

### 6. Multivariate PDF methods return wrong type (`distributions.py`)

This is a logic bug, not a deprecation. `MultivariateNormal.pdf` and `Multinomial.pdf` are set to return a frozen distribution object rather than a float:

```python
# MultivariateNormal.__init__ (line ~1340)
self.pdf = lambda x: stats.multivariate_normal(x, mean, cov)   # returns frozen dist, not float

# Multinomial.__init__ (line ~1530)
self.pdf = lambda x: stats.multinomial(x, n, p)                 # same problem
```

`stats.multivariate_normal(x, mean, cov)` constructs a frozen distribution; to get the density at point `x` you need `.pdf(x)`. This is masked because `MultivariateNormal.plot()` and `Multinomial.plot()` raise `NotImplementedError` before `.pdf` is ever called in normal use, but any code that calls `.pdf()` directly will receive an object instead of a number.

**Fix:**
```python
self.pdf = lambda x: stats.multivariate_normal(mean=mean, cov=cov).pdf(x)
self.pdf = lambda x: stats.multinomial(n=n, p=p).pmf(x)
```

---

## ✅ Things That Are Clean on `dev`

- **`get_next_color()`**: Now uses `itertools.cycle` over `plt.rcParams["axes.prop_cycle"]` — clean and forward-compatible.
- **`init_color()`**: Uses `matplotlib.colormaps` (new API) rather than the deprecated `plt.cm.get_cmap()`.
- **All other plot types** (hist, density, rug, scatter, tile, marginal, Distribution.plot, process paths): Smoke-tested clean with no warnings.
- **Core simulation engine** (`probability_space.py`, `random_variables.py`, `results.py` logic): No issues.
- **SciPy distribution wrappers** in `Distribution.__init__()`: Current and correct.
- **NumPy type aliases** (`np.bool`, `np.int`, etc.): None appear anywhere in the codebase.

---

## Prioritized Fix List

| Priority | Issue | File(s) | Effort |
|---|---|---|---|
| 1 | `violinplot(vert=...)` deprecated | `plot.py:139` | 2 lines |
| 2 | Seaborn style name fragile | `results.py:45–49` | 3 lines |
| 3 | `hist2d` normalize labels — int cast + UserWarning | `results.py:1265–1273` | ~8 lines |
| 4 | Multivariate PDF logic bug | `distributions.py` ×2 | 2 lines |
| 5 | `init_color()` cleanup / removal | `plot.py:21–23` | minor |
| 6 | Legacy `np.random` API — 13 sites | 5 files | medium, systematic |
