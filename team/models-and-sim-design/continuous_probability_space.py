"""Generic continuous probability space defined by an arbitrary pdf.

Given only a probability density function f (and, optionally, its
support), :class:`ContinuousProbabilitySpace` tries, in order:

1. **Inverse-CDF (inverse transform) sampling.** Numerically integrate
   f to build F, invert F with a monotone interpolant to get the
   quantile function Q, then sample Q(U) for U ~ Uniform(0, 1).
   This produces exactly n iid draws with no waste and is used
   whenever it is numerically well-behaved.
2. **Rejection sampling**, if step 1 fails (e.g. the numerical CDF is
   not usable, or the user asks for method="rejection" directly).
   A proposal distribution g and envelope constant M with
   f(x) <= M*g(x) are found automatically (uniform proposal on a
   finite support, or a scaled Cauchy proposal on an unbounded one),
   and batches of candidates are generated and accepted/rejected
   with numpy, so this is vectorized instead of a Python-level loop.

Either way, draws are cached internally in batches so that repeated
calls to draw() (which is how Symbulate's sim() consumes a
probability space) are cheap on average.
"""

import numbers
import warnings

import numpy as np
from scipy import integrate, interpolate

from .probability_space import ProbabilitySpace
from .result import Scalar

rng = np.random.default_rng()


class ContinuousProbabilitySpace(ProbabilitySpace):
    """Probability space for simulating from an arbitrary continuous pdf.

    Parameters
    ----------
    pdf : callable
        The probability density function, f(x). Must accept a numpy
        array and return an array of the same shape (vectorized), be
        non-negative, and be zero outside its intended support.
    xlim : tuple of float, optional
        The support of the density, as (lo, hi). Either side may be
        ``-np.inf`` / ``np.inf``. If omitted, the support is guessed
        by expanding outward from (-8, 8) until the density is
        negligible at both ends -- this is convenient but less
        reliable than specifying it, especially for densities with
        heavy tails or supports far from the origin.
    method : {"auto", "inverse_cdf", "rejection"}, optional
        "auto" (default) tries inverse-CDF sampling first and falls
        back to rejection sampling if that fails. The other two force
        a specific method and raise an error if it can't be used.
    proposal, M : optional
        Only used for rejection sampling. ``proposal`` is a Symbulate
        distribution (something with a ``.pdf`` and a way to draw,
        e.g. ``Normal(0, 3)``) to sample candidates from, and ``M`` is
        a known constant with pdf(x) <= M * proposal.pdf(x) everywhere.
        If omitted, both are chosen automatically.
    grid_size : int, optional
        Number of grid points used to numerically integrate pdf and
        (for rejection sampling) to search for the envelope constant.
    buffer_size : int, optional
        Number of draws generated per internal batch.

    Attributes
    ----------
    pdf : callable
        The (unnormalized) density function passed in.
    cdf : callable
        Numerically-estimated CDF, F(x), vectorized. Always
        available, even when rejection sampling is used for draws.
    quantile : callable or None
        Inverse CDF, Q(u), if the inverse-CDF method could be used;
        otherwise ``None``.
    method_used : {"inverse_cdf", "rejection"}
        Which sampling method ended up being used.

    Examples
    --------
    >>> from symbulate import *
    >>> import numpy as np
    >>> f = lambda x: np.where((0 <= x) & (x <= 1), 2 * x, 0.0)
    >>> P = ContinuousProbabilitySpace(f, xlim=(0, 1))
    >>> X = RV(P)
    >>> X.sim(10000).mean()  # doctest: +SKIP
    0.667
    """

    def __init__(
        self,
        pdf,
        xlim=None,
        method="auto",
        proposal=None,
        M=None,
        grid_size=4000,
        buffer_size=2000,
    ):
        if not callable(pdf):
            raise TypeError(
                "pdf must be callable (e.g., a lambda or function), "
                f"but got {type(pdf).__name__}."
            )
        if method not in ("auto", "inverse_cdf", "rejection"):
            raise ValueError(
                f"method must be 'auto', 'inverse_cdf', or 'rejection', got {method!r}."
            )

        self.pdf = pdf
        self.buffer_size = buffer_size
        self._buffer = []

        self.lo, self.hi = self._resolve_support(pdf, xlim)

        total, _ = integrate.quad(pdf, self.lo, self.hi, limit=200)
        if not (total > 0) or not np.isfinite(total):
            raise ValueError(
                f"pdf must integrate to a positive, finite value over "
                f"[{self.lo}, {self.hi}], but got {total!r}. Check that "
                "pdf(x) >= 0 and that xlim covers the support."
            )
        if abs(total - 1.0) > 1e-2:
            warnings.warn(
                f"pdf integrates to {total:.4f} over [{self.lo}, {self.hi}], "
                "not 1. Draws will be generated as if pdf were normalized "
                "by this constant, but double check pdf and xlim.",
                UserWarning,
            )
        self._norm = total
        self._pdf_norm = lambda x: pdf(x) / self._norm

        self.cdf = None
        self.quantile = None
        self.method_used = None

        if method in ("auto", "inverse_cdf"):
            try:
                self._build_inverse_cdf_sampler(grid_size)
                self.method_used = "inverse_cdf"
            except Exception as exc:
                if method == "inverse_cdf":
                    raise ValueError(
                        f"Could not build an invertible CDF for this pdf: {exc}"
                    ) from exc
                warnings.warn(
                    f"Falling back to rejection sampling because the CDF "
                    f"could not be inverted numerically ({exc}).",
                    UserWarning,
                )

        if self.method_used is None:
            self._build_rejection_sampler(proposal, M, grid_size)
            self.method_used = "rejection"

    # ------------------------------------------------------------------
    # Support detection
    # ------------------------------------------------------------------

    def _resolve_support(self, pdf, xlim):
        if xlim is not None:
            lo, hi = xlim
            if lo >= hi:
                raise ValueError(f"xlim must satisfy lo < hi, got {xlim!r}.")
            return float(lo), float(hi)

        lo, hi = -8.0, 8.0
        for _ in range(20):
            xs = np.linspace(lo, hi, 4001)
            ys = np.nan_to_num(pdf(xs), nan=0.0, posinf=0.0, neginf=0.0)
            peak = ys.max()
            if peak > 0 and ys[0] <= peak * 1e-6 and ys[-1] <= peak * 1e-6:
                return lo, hi
            lo, hi = lo * 2, hi * 2
        warnings.warn(
            "Could not automatically determine the support of pdf; "
            "pass xlim=(lo, hi) explicitly for reliable results. "
            f"Falling back to [{lo}, {hi}].",
            UserWarning,
        )
        return lo, hi

    # ------------------------------------------------------------------
    # Method 1: inverse-CDF (inverse transform) sampling
    # ------------------------------------------------------------------

    def _build_inverse_cdf_sampler(self, grid_size):
        xs = np.linspace(self.lo, self.hi, grid_size)
        ys = np.asarray(self.pdf(xs), dtype=float)
        if np.any(ys < -1e-8):
            raise ValueError("pdf takes negative values; not a valid density.")
        ys = np.clip(ys, 0, None)

        cdf_vals = integrate.cumulative_trapezoid(ys, xs, initial=0.0)
        total = cdf_vals[-1]
        if not (total > 0):
            raise ValueError("pdf integrates to zero over the given support.")
        cdf_vals = cdf_vals / total

        # PchipInterpolator needs a strictly increasing x-array; drop
        # points where the numerical CDF is exactly flat (pdf ~ 0).
        cdf_unique, idx = np.unique(cdf_vals, return_index=True)
        xs_unique = xs[idx]
        if len(cdf_unique) < max(10, grid_size * 0.05):
            raise ValueError(
                "the numerical CDF is not increasing enough to invert "
                "reliably (pdf may be zero almost everywhere on xlim)."
            )

        self.quantile = interpolate.PchipInterpolator(
            cdf_unique, xs_unique, extrapolate=True
        )
        self.cdf = interpolate.interp1d(
            xs, cdf_vals, bounds_error=False, fill_value=(0.0, 1.0)
        )

    def _sample_inverse_cdf(self, n):
        u = rng.random(n)
        return self.quantile(u)

    # ------------------------------------------------------------------
    # Method 2: rejection sampling
    # ------------------------------------------------------------------

    def _build_rejection_sampler(self, proposal, M, grid_size):
        finite_support = np.isfinite(self.lo) and np.isfinite(self.hi)

        if proposal is None:
            if finite_support:
                width = self.hi - self.lo
                self._propose = lambda n: rng.uniform(self.lo, self.hi, n)
                self._proposal_pdf = lambda x: np.full_like(x, 1.0 / width, dtype=float)
            else:
                # Generic heavy-tailed default proposal for unbounded
                # support. A Cauchy's tails decay slower than most
                # textbook densities, so f(x)/g(x) usually stays bounded.
                scale = self._estimate_scale()
                self._propose = lambda n: scale * rng.standard_cauchy(n)
                self._proposal_pdf = lambda x: 1.0 / (
                    np.pi * scale * (1.0 + (x / scale) ** 2)
                )
        else:
            self._propose = lambda n: np.asarray(proposal.sim(n))
            self._proposal_pdf = proposal.pdf

        if M is None:
            lo = self.lo if np.isfinite(self.lo) else -50.0 * self._estimate_scale()
            hi = self.hi if np.isfinite(self.hi) else 50.0 * self._estimate_scale()
            xs = np.linspace(lo, hi, grid_size)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = self._pdf_norm(xs) / self._proposal_pdf(xs)
            ratio = ratio[np.isfinite(ratio)]
            if ratio.size == 0 or ratio.max() <= 0:
                raise ValueError(
                    "Could not find a finite envelope constant M for "
                    "rejection sampling; supply proposal and M explicitly."
                )
            M = ratio.max() * 1.05  # small safety margin
        if not np.isfinite(M) or M <= 0:
            raise ValueError(f"M must be a positive, finite number, got {M!r}.")

        self._M = M
        self._p_accept = max(1.0 / M, 1e-6)

    def _estimate_scale(self):
        # Rough spread estimate (used only to size a default proposal),
        # via a numerical standard deviation of the normalized pdf.
        lo = self.lo if np.isfinite(self.lo) else -1e3
        hi = self.hi if np.isfinite(self.hi) else 1e3
        mean, _ = integrate.quad(lambda x: x * self._pdf_norm(x), lo, hi, limit=200)
        var, _ = integrate.quad(
            lambda x: (x - mean) ** 2 * self._pdf_norm(x), lo, hi, limit=200
        )
        return max(np.sqrt(var), 1e-6)

    def _sample_rejection(self, n):
        accepted = []
        n_have = 0
        while n_have < n:
            need = n - n_have
            batch = int(np.ceil(need / self._p_accept * 1.15)) + 16
            x = np.asarray(self._propose(batch), dtype=float)
            u = rng.random(batch)
            with np.errstate(divide="ignore", invalid="ignore"):
                accept_prob = self._pdf_norm(x) / (self._M * self._proposal_pdf(x))
            keep = u <= np.nan_to_num(accept_prob, nan=0.0)
            x = x[keep]
            accepted.append(x)
            n_have += x.size
        return np.concatenate(accepted)[:n]

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _refill(self):
        if self.method_used == "inverse_cdf":
            batch = self._sample_inverse_cdf(self.buffer_size)
        else:
            batch = self._sample_rejection(self.buffer_size)
        self._buffer = list(batch)

    def draw(self):
        """Draw a single random sample from the density.

        Returns
        -------
        Scalar
            One random value drawn from the (approximate) distribution
            described by ``pdf``.
        """
        if not self._buffer:
            self._refill()
        return Scalar(self._buffer.pop())
