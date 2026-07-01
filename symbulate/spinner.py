"""Static spinner wheel visualization for probability distributions.

Called via Distribution.spinner():
    Normal(0, 1).spinner()
    Binomial(10, 0.3).spinner(mode='equal')

mode = 'proportional'  – each slice sized by probability (default)
       'equal'         – all slices the same size
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── colors ───────────────────────────────────────────────────────────────────

_COLORS = [
    '#4e9af1', '#f5a63c', '#5cb85c', '#e74c3c', '#9b59b6',
    '#1abc9c', '#e67e22', '#3498db', '#e91e63', '#2ecc71',
    '#ff9800', '#8e44ad', '#00bcd4', '#c0392b', '#27ae60',
    '#d35400', '#2980b9', '#f06292', '#16a085', '#7f8c8d',
]


def _color(i):
    return _COLORS[i % len(_COLORS)]


def _is_dark(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 < 0.55


# ─── formatting ───────────────────────────────────────────────────────────────

def _fmt(x):
    a = abs(x)
    if a >= 100:
        return f'{x:.0f}'
    if a >= 10:
        return f'{x:.1f}'
    return f'{x:.2f}'


# ─── internal defaults for wheel sections ─────────────────────────────────────

_DISC_EQUAL_N = 20
_CONT_BINS    = 16


# ─── slice builders ────────────────────────────────────────────────────────────

def _discrete_proportional(dist):
    lo = int(math.floor(float(dist.quantile(0.0005))))
    hi = int(math.ceil(float(dist.quantile(0.9995))))
    values = list(range(lo, hi + 1))
    raw = [max(0.0, float(dist.pmf(k))) for k in values]
    tail = max(0.0, 1.0 - sum(raw))
    slices = [{'label': str(v), 'prob': p}
              for v, p in zip(values, raw) if p > 1e-7]
    if tail > 0.004:
        slices.append({'label': f'≥{hi + 1}', 'prob': tail})
    total = sum(s['prob'] for s in slices) or 1.0
    for s in slices:
        s['prob'] /= total
    return slices


def _discrete_equal(dist, n):
    slices = []
    for i in range(n):
        val = int(round(float(dist.quantile((i + 0.5) / n))))
        slices.append({'label': str(val), 'prob': 1.0 / n})
    return slices


def _continuous_proportional(dist, n):
    lo = float(dist.quantile(0.005))
    hi = float(dist.quantile(0.995))
    edges = np.linspace(lo, hi, n + 1)
    slices = []
    for i in range(n):
        prob = max(0.0, float(dist.cdf(edges[i + 1]) - dist.cdf(edges[i])))
        mid = (edges[i] + edges[i + 1]) / 2.0
        slices.append({'label': _fmt(mid), 'prob': prob})
    total = sum(s['prob'] for s in slices) or 1.0
    for s in slices:
        s['prob'] /= total
    return slices


def _continuous_equal(dist, n):
    slices = []
    for i in range(n):
        val = float(dist.quantile((i + 0.5) / n))
        slices.append({'label': _fmt(val), 'prob': 1.0 / n})
    return slices


def _build_slices(dist, mode):
    if getattr(dist, 'discrete', False):
        return (_discrete_proportional(dist) if mode == 'proportional'
                else _discrete_equal(dist, _DISC_EQUAL_N))
    else:
        return (_continuous_proportional(dist, _CONT_BINS) if mode == 'proportional'
                else _continuous_equal(dist, _CONT_BINS))


# ─── drawing ───────────────────────────────────────────────────────────────────

def _label_color_map(slices):
    """Assign a consistent color to each unique label, in order of first appearance."""
    seen = {}
    for s in slices:
        if s['label'] not in seen:
            seen[s['label']] = _color(len(seen))
    return seen


def _draw_spinner(slices, dist, mode):
    n_sl = len(slices)
    label_colors = _label_color_map(slices)

    fig, ax = plt.subplots(figsize=(6, 6.5), facecolor='white')
    ax.set_aspect('equal')
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.65)
    ax.axis('off')

    # Wedges clockwise from top (90° in matplotlib convention)
    current = 90.0
    for s in slices:
        sweep = s['prob'] * 360.0
        theta1 = current - sweep
        c = label_colors[s['label']]
        ax.add_patch(mpatches.Wedge(
            (0, 0), 1.0, theta1, current,
            facecolor=c, edgecolor='white', linewidth=1.5
        ))
        if sweep >= 4:
            mid_rad = math.radians((theta1 + current) / 2.0)
            lx, ly = 0.70 * math.cos(mid_rad), 0.70 * math.sin(mid_rad)
            fs = max(5, min(10, int(180 / n_sl)))
            tc = 'white' if _is_dark(c) else '#1a1a2e'
            ax.text(lx, ly, s['label'], ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=tc, clip_on=True)
        current = theta1

    # Outer border
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False,
                             edgecolor='#2c3e50', linewidth=2.5))

    # Center hub
    ax.add_patch(plt.Circle((0, 0), 0.06, facecolor='#2c3e50',
                             edgecolor='white', linewidth=2, zorder=10))

    # Pointer triangle
    ax.add_patch(plt.Polygon(
        [[0, 1.07], [-0.06, 1.26], [0.06, 1.26]],
        closed=True, facecolor='#e74c3c', edgecolor='white',
        linewidth=1.5, zorder=11
    ))

    # Title
    dist_name = type(dist).__name__
    mode_str = 'Equal sections' if mode == 'equal' else 'Sized by probability'
    ax.text(0, 1.55, dist_name, ha='center', va='center',
            fontsize=13, fontweight='bold', color='#1a1a2e')
    ax.text(0, 1.42, mode_str, ha='center', va='center',
            fontsize=9, color='#666', style='italic')

    plt.tight_layout()
    plt.show()


# ─── entry point ──────────────────────────────────────────────────────────────

_MULTIVARIATE = {'BivariateNormal', 'MultivariateNormal', 'Multinomial'}


def show_spinner(dist, mode='proportional'):
    """Render a static spinner wheel image. Called by Distribution.spinner()."""
    name = type(dist).__name__
    if name in _MULTIVARIATE:
        raise ValueError(
            f"{name} is a multivariate distribution and cannot be shown "
            "as a single spinner wheel."
        )
    slices = _build_slices(dist, mode)
    _draw_spinner(slices, dist, mode)
