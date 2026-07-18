import sys
sys.path.insert(0, "/home/claude/symbulate")
sys.path.insert(0, "/home/claude/work")

import numpy as np
import pandas as pd

from table_distributions import Empirical, LifeTable

np.random.seed(0)

results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


# ---------------------------------------------------------------------
# 1. Basic Empirical correctness
# ---------------------------------------------------------------------
print("\n=== Empirical: basic correctness ===")
X = Empirical([1, 2, 3], [0.2, 0.3, 0.5])

check("pmf sums to 1", np.isclose(sum(X.pmf(k) for k in [1, 2, 3]), 1.0))
check("pmf(1) == 0.2", np.isclose(X.pmf(1), 0.2))
check("pmf(2) == 0.3", np.isclose(X.pmf(2), 0.3))
check("pmf(3) == 0.5", np.isclose(X.pmf(3), 0.5))
check("mean == 1*.2+2*.3+3*.5 == 2.3", np.isclose(X.mean(), 2.3), f"got {X.mean()}")
check("cdf(2) == 0.5", np.isclose(X.cdf(2), 0.5))

# draw() returns only outcomes in the support
draws = [X.draw() for _ in range(500)]
check("all draws are in {1,2,3}", set(draws).issubset({1, 2, 3}))

# large-sample frequency check against true probabilities
sim = X.sim(20000)
tab = sim.tabulate(normalize=True)
check(
    "simulated frequencies close to true probs (tol .02)",
    all(abs(tab[k] - p) < 0.02 for k, p in zip([1, 2, 3], [0.2, 0.3, 0.5])),
    detail=str(dict(tab)),
)

# renormalization of probabilities that don't sum to 1
Y = Empirical([10, 20], [2, 2])  # unnormalized (sums to 4, not 1)
check("renormalizes probabilities", np.isclose(Y.pmf(10), 0.5) and np.isclose(Y.pmf(20), 0.5))

# error handling
try:
    Empirical([1, 2], [0.5, 0.5, 0.1])
    check("raises on mismatched lengths", False)
except ValueError:
    check("raises on mismatched lengths", True)

try:
    Empirical([1.5, 2.5], [0.5, 0.5])
    check("raises on non-integer outcomes", False)
except ValueError:
    check("raises on non-integer outcomes", True)

try:
    Empirical([1, 2], [-0.5, 1.5])
    check("raises on negative probabilities", False)
except ValueError:
    check("raises on negative probabilities", True)


# ---------------------------------------------------------------------
# 2. LifeTable: build from synthetic SOA-style table
# ---------------------------------------------------------------------
print("\n=== LifeTable: build & sanity checks ===")
df = pd.read_csv("/home/claude/work/synthetic_soa_table.csv")

K45 = LifeTable(df, age=45)

check("LifeTable has 'age' attribute set correctly", K45.age == 45)
check("support starts at k=0", K45.pmf(0) > 0)
check(
    "support length == number of remaining table rows (ages 45..110 inclusive = 66)",
    len(K45.lx_) - 1 == (110 - 45 + 1),
    detail=f"got {len(K45.lx_) - 1}",
)
check("lx terminates at exactly 0 at the omega age", K45.lx_[-1] == 0.0, detail=f"lx[-1]={K45.lx_[-1]}")
check("pmf sums to (approximately) 1 over full support",
      np.isclose(sum(K45.pmf(k) for k in range(len(K45.lx_) - 1)), 1.0, atol=1e-6))

# lx must be non-increasing
check("lx is non-increasing", np.all(np.diff(K45.lx_) <= 0))

# mean curtate future lifetime should be a plausible number of years
# for a 45-year-old under this table (definitely not negative, definitely
# less than the remaining table length)
mean_k = K45.mean()
check(
    "mean curtate future lifetime is plausible (0 < mean < 66)",
    0 < mean_k < 66,
    detail=f"E[K_45] = {mean_k:.2f} years -> expected age at death ~ {45+mean_k:.1f}",
)

# simulate and confirm draws are integers within the valid range
draws45 = K45.sim(5000)
draws_list = list(draws45)
check(
    "all simulated K_45 draws within valid range [0, 65]",
    all(0 <= d <= 65 for d in draws_list),
)
check(
    "simulated mean roughly matches theoretical mean (tol 1.0 yr)",
    abs(np.mean(draws_list) - mean_k) < 1.0,
    detail=f"sim mean={np.mean(draws_list):.2f}, theoretical={mean_k:.2f}",
)

# probability of surviving at least 20 more years (K_45 >= 20)
# should equal l_65 / l_45 from the life table directly
p_survive_20 = 1 - K45.cdf(19)  # P(K >= 20) = 1 - P(K <= 19)
p_from_lx = K45.lx_[20] / K45.lx_[0]
check(
    "P(K_45 >= 20) matches l_65/l_45 from the table directly (exact)",
    np.isclose(p_survive_20, p_from_lx, atol=1e-9),
    detail=f"{p_survive_20:.9f} vs {p_from_lx:.9f}",
)

# ---------------------------------------------------------------------
# 3. LifeTable: different entry ages behave consistently
# ---------------------------------------------------------------------
print("\n=== LifeTable: consistency across entry ages ===")
K0 = LifeTable(df, age=0)
K70 = LifeTable(df, age=70)

check("older entry age has shorter remaining lifetime on average",
      K70.mean() < K0.mean(),
      detail=f"E[K_0]={K0.mean():.2f}, E[K_70]={K70.mean():.2f}")

# ---------------------------------------------------------------------
# 4. LifeTable: error handling
# ---------------------------------------------------------------------
print("\n=== LifeTable: error handling ===")
try:
    LifeTable(df, age=200)
    check("raises on out-of-range age", False)
except ValueError:
    check("raises on out-of-range age", True)

try:
    LifeTable(df, age=45, qx_column="not_a_real_column")
    check("raises on missing qx column", False)
except ValueError:
    check("raises on missing qx column", True)

bad_df = df.copy()
bad_df.loc[bad_df.index[-1], "qx"] = 0.5  # doesn't reach 1.0 at omega age
try:
    LifeTable(bad_df, age=45)
    check("raises when table doesn't terminate at qx=1.0", False)
except ValueError:
    check("raises when table doesn't terminate at qx=1.0", True)

# ---------------------------------------------------------------------
# 5. Integration check: works with Symbulate's RV() wrapper too
# ---------------------------------------------------------------------
print("\n=== Integration with Symbulate RV/plot machinery ===")
from symbulate import RV

rv = RV(K45)
rv_draws = rv.sim(1000)
check("RV(LifeTable) simulates successfully", len(rv_draws) == 1000)

import matplotlib
matplotlib.use("Agg")
try:
    K45.plot()
    check("plot() runs without error", True)
except Exception as e:
    check("plot() runs without error", False, detail=str(e))


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\n=== SUMMARY ===")
n_pass = sum(1 for _, s, _ in results if s == "PASS")
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"{n_pass} passed, {n_fail} failed, {len(results)} total")
if n_fail:
    print("\nFAILED CHECKS:")
    for name, status, detail in results:
        if status == "FAIL":
            print(f" - {name}: {detail}")
    sys.exit(1)
