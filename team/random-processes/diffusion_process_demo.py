"""
Demo: DiffusionProcess in Symbulate

DiffusionProcess now ships as part of the symbulate package
(symbulate/diffusion_process.py), so this script runs as-is -- there is
nothing to copy into place first.
"""

import numpy as np
import matplotlib.pyplot as plt
from symbulate import DiffusionProcess, diffusion_process

# Seed the generator the module actually draws from. numpy.random.seed only
# seeds the old global generator, which default_rng() does not read, so
# seeding that way leaves the path different on every run.
diffusion_process.rng = np.random.default_rng(42)

# Geometric Brownian motion: dX = mu*X dt + sigma*X dW  (e.g., a stock price model)
mu, sigma = 0.05, 0.4
X = DiffusionProcess(
    drift=lambda x, t: mu * x,
    diffusion=lambda x, t: sigma * x,
    x0=100,
    tol=1e-3,  # finest mesh resolution the path can be refined to
)

path = X.draw()  # ONE realization -- a lazily-evaluated sample path

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Look at the path on the full time scale [0, 10] ...
ts_full = np.linspace(0, 10, 2000)
axes[0].plot(ts_full, [path(t) for t in ts_full], lw=1)
axes[0].axvspan(0, 0.1, color="orange", alpha=0.2)
axes[0].set_title("Full path on [0, 10]")
axes[0].set_xlabel("t")
axes[0].set_ylabel("X(t)")

# ... then zoom into [0, 0.1]. This is the SAME draw: values already
# computed (e.g. path(0), path(0.1) if queried before) are cached and
# reused; new in-between values are drawn as bridge samples consistent
# with them, not as an independent walk.
ts_zoom = np.linspace(0, 0.1, 2000)
axes[1].plot(ts_zoom, [path(t) for t in ts_zoom], lw=1, color="darkorange")
axes[1].set_title("Zoomed in on [0, 0.1] -- same draw")
axes[1].set_xlabel("t")
axes[1].set_ylabel("X(t)")

plt.tight_layout()
plt.savefig("zoom_demo.png", dpi=130)
print("Saved zoom_demo.png")
