"""When does a random process first reach a level?

This module provides :func:`hitting_time`, which answers "how long until the
process first gets to ``level``?" for a sample path -- or, given a whole
process, hands back a random variable you can simulate like any other.

It also provides :func:`upcrossings`, which asks the same question over and
over: when does the path get to the level, and then when does it get back there
again, and again? For the jump and discrete-time processes described below that
is all it needs, and the answer is exact.

A path that moves continuously needs one thing more, and the reason is worth
understanding rather than working around. Such a path recrosses a level
infinitely often the instant it touches it, so for a Brownian motion there is no
second or third crossing of that level to report -- any count of them would
measure how finely the path was looked at rather than anything about the path.
What fixes it is a second level: give :func:`upcrossings` a ``reset``, and a
crossing counts only once the path has been back to it, so the path has a finite
distance to cover between one crossing and the next and only finitely many fit
into a finite stretch of time. Ask for a single level on a continuous path and
the function says all this rather than returning a number that depends on a step
size.

The hard part is that a sample path is only ever computed at the times
somebody asks about. Between two of those times the path is not a straight
line: it wiggles, and it might have crossed the level and come back without
either endpoint noticing. Simply checking the times you happened to look at
would therefore report hitting times that are too late, or miss crossings
altogether.

For a Brownian motion there is an exact fix, and it is what this module
uses. Given the values at two times, the *probability* that the path crossed
a level somewhere in between has a closed form -- a classical result that
follows from the reflection principle:

    P(crossed) = exp(-2 * (level - x0) * (level - x1) / (rate * (t1 - t0)))

So instead of guessing, the crossing is decided by flipping a coin weighted
by that probability. When the coin says a crossing happened, the interval is
cut in half repeatedly to pin down when -- each halving asking the path for
its value at the midpoint, which is exactly the "zoom in" the process
already supports.

Two different things are worth separating when asking how accurate this is
(see :func:`hitting_time` for the details):

- **Did the process reach the level, and in which stretch?** For Brownian
  motion and Brownian bridges this is exact, at any ``step``, because the
  crossing formula above is exact for them. For other Gaussian processes it
  is an approximation that improves as ``step`` shrinks.
- **Exactly when inside that stretch?** Approximate for every process. The
  crossing is placed inside the right stretch by halving, but the halving
  rule is slightly biased toward later times, so the answer is reliable to
  about one ``step``. Use a smaller ``step`` when the precise time matters.

Geometric Brownian motion is handled too, and exactly, even though it is not
a Gaussian process itself: a price reaching a level is the same event as its
*log* reaching the log of that level, and the log of a geometric Brownian
motion is an ordinary Brownian motion with drift. So the question is simply
restated on the log scale, where the formula above applies exactly.

None of that guesswork is needed for a process that moves in jumps, and those
are handled separately and **exactly**. A pure-jump path -- a Poisson or
renewal count, a compound Poisson total, a continuous-time Markov chain, a
queue length -- holds one value at a time, so it can only reach a level *at* a
jump; walking the jumps therefore cannot miss a crossing, and there is nothing
to localize. A discrete-time path -- a random walk, a Markov chain, a
moving-average series -- has no values between its steps at all, so walking the
steps is likewise the whole story. Both are read with one difference from the
Gaussian case worth knowing about: such a path can *leap over* a level without
ever equaling it, so reaching a level means getting to it **or past it**.

General diffusions (:class:`~symbulate.diffusion_process.DiffusionProcess`,
:class:`~symbulate.diffusion_process.CIR`,
:class:`~symbulate.diffusion_process.MertonJumpDiffusion`) still need their own
approximation, and :func:`hitting_time` says so plainly rather than guessing.
A Merton jump diffusion is *not* one of the jump paths above, despite its name:
it wanders continuously between its jumps, so walking the jumps would miss
everything that happens in between.
"""

import math
import numbers

import numpy as np

from .random_variables import RV
from .result import (
    DiscreteTimeFunction,
    DiscreteValued,
    InfiniteTuple,
    InfiniteVector,
)

# Guard against a singular covariance matrix when two times are nearly equal,
# the same trick and constant gaussian_process.py uses for the same reason.
MACHINE_EPS = 1e-12

# A backstop on walking a jump path, so that a process jumping far more often
# than the stretch of time being searched reports a readable error instead of
# appearing to hang.
MAX_JUMPS = 1_000_000

# The shared generator, owned by probability_space.py -- this module used to
# create its own, which made seed() unable to reach it. Do not reintroduce a
# local `rng = np.random.default_rng()` here.
from .probability_space import rng


def _local_variance_rate(cov_func, t0, t1):
    """Return the Brownian variance rate matching this process over ``[t0, t1]``.

    A Brownian motion with variance rate ``r`` has, between two known
    endpoints, a midpoint variance of ``r * (t1 - t0) / 4``. This works
    backwards from that: it computes what the process's own midpoint
    variance actually is, then reports the ``r`` that a Brownian motion
    would need to match it. Feeding that ``r`` into the crossing formula is
    what lets one formula serve every Gaussian process.

    For a Brownian motion (or a Brownian bridge) this returns the process's
    own variance rate exactly, for any pair of times, which is why the
    crossing probability is exact for them. For other Gaussian processes it
    returns a local stand-in that becomes more accurate the closer together
    ``t0`` and ``t1`` are.

    Parameters
    ----------
    cov_func : callable
        The process's covariance function ``k(s, t)``.
    t0, t1 : float
        The two times, with ``t0 < t1``.

    Returns
    -------
    float
        The matching variance rate, or 0 if the process cannot move at all
        between these two times.
    """
    if t1 <= t0:
        return 0.0

    midpoint = 0.5 * (t0 + t1)
    endpoint_cov = np.array(
        [
            [cov_func(t0, t0), cov_func(t0, t1)],
            [cov_func(t0, t1), cov_func(t1, t1)],
        ],
        dtype=float,
    )
    midpoint_cov = np.array(
        [cov_func(midpoint, t0), cov_func(midpoint, t1)], dtype=float
    )

    try:
        weights = np.linalg.solve(endpoint_cov + MACHINE_EPS * np.eye(2), midpoint_cov)
    except np.linalg.LinAlgError:
        return 0.0

    midpoint_variance = cov_func(midpoint, midpoint) - midpoint_cov @ weights
    if not np.isfinite(midpoint_variance) or midpoint_variance <= 0:
        return 0.0
    return float(4.0 * midpoint_variance / (t1 - t0))


def _crossing_probability(gap0, gap1, variance_rate, elapsed):
    """Return the chance the path crossed the level between two known values.

    This is the reflection-principle formula. ``gap0`` and ``gap1`` are how
    far each endpoint sits from the level, measured so that both are
    positive when neither endpoint has reached it.

    Parameters
    ----------
    gap0, gap1 : float
        Distance from the level at the start and end of the stretch.
    variance_rate : float
        The Brownian variance rate over this stretch, from
        :func:`_local_variance_rate`.
    elapsed : float
        How much time the stretch covers.

    Returns
    -------
    float
        A probability between 0 and 1. Returns 1 if either endpoint has
        already reached the level, and 0 if the process cannot move.
    """
    if gap0 <= 0 or gap1 <= 0:
        return 1.0
    if variance_rate <= 0 or elapsed <= 0:
        return 0.0
    return float(np.exp(-2.0 * gap0 * gap1 / (variance_rate * elapsed)))


def _localize_crossing(read, level, sign, cov_func, t0, x0, t1, x1, tol):
    """Narrow down *when* a crossing happened, once it is known that one did.

    Repeatedly halves ``[t0, t1]``, asking the path for its midpoint value
    each time and keeping whichever half holds the *first* crossing. If the
    midpoint has already reached the level, the first crossing must be in the
    left half. Otherwise either half could hide it, so the half is chosen
    with the probability that it is the one -- again using the reflection
    formula.

    Parameters
    ----------
    read : callable
        ``read(t)`` gives the value being tracked at time ``t``, on whatever
        scale the crossing formula applies to. For a Gaussian process that is
        just the path itself; for a geometric Brownian motion it is the log
        of the path (see :func:`hitting_time`).
    level : float
        The level being crossed, on the same scale as ``read``.
    sign : float
        ``1`` when looking for the value to rise to the level, ``-1`` when
        looking for it to fall to the level. Multiplying by this lets one
        piece of code handle both directions.
    cov_func : callable
        The covariance function of the process ``read`` reports.
    t0, t1 : float
        A stretch of time known to contain a crossing.
    x0, x1 : float
        Already-adjusted distances from the level at ``t0`` and ``t1``
        (positive means "has not reached it").
    tol : float
        Stop once the stretch is this short, and report its midpoint.

    Returns
    -------
    float
        The crossing time, to within ``tol``.
    """
    while t1 - t0 > tol:
        midpoint = 0.5 * (t0 + t1)
        gap_mid = sign * (level - read(midpoint))

        # The midpoint itself has reached the level, so the first crossing is
        # certainly no later than the midpoint.
        if gap_mid <= 0:
            t1, x1 = midpoint, gap_mid
            continue

        rate_left = _local_variance_rate(cov_func, t0, midpoint)
        prob_left = _crossing_probability(x0, gap_mid, rate_left, midpoint - t0)

        if x1 <= 0:
            # The right half ends at or past the level, so it definitely
            # contains a crossing. The first one is in the left half exactly
            # when the left half has one at all.
            prob_first_left = prob_left
        else:
            rate_right = _local_variance_rate(cov_func, midpoint, t1)
            prob_right = _crossing_probability(gap_mid, x1, rate_right, t1 - midpoint)
            neither = (1.0 - prob_left) * (1.0 - prob_right)
            prob_first_left = prob_left / (1.0 - neither) if neither < 1.0 else 0.0

        if rng.uniform() < prob_first_left:
            t1, x1 = midpoint, gap_mid
        else:
            t0, x0 = midpoint, gap_mid

    return 0.5 * (t0 + t1)


def _numeric_value(value, where):
    """Check one value read off a path, and return it as a plain number.

    Parameters
    ----------
    value : object
        The value the path reported. A number, if all is well.
    where : str
        Where it came from, phrased to begin a sentence: ``"At step 3,"``.
        Used in the error message below.

    Returns
    -------
    float
        The value.

    Raises
    ------
    NotImplementedError
        If the value is several numbers at once, as an epidemic model's
        compartment counts are. Such a path has no single level to reach, so
        this is a missing feature rather than a mistake.
    TypeError
        If the value is not a number for any other reason, so cannot be
        compared with a level at all.
    """
    # An epidemic path reports its whole compartment vector as a state, so it
    # reaches this check rather than the general one at the bottom of
    # `_prepare`. The advice a student needs is the same either way.
    if isinstance(value, (tuple, list, np.ndarray)):
        raise NotImplementedError(
            f"{where} the process is at {tuple(value)!r} -- several numbers at "
            "once rather than one. A level is a single number, so there is "
            "nothing here to compare it with, and neither a hitting time nor "
            "its upcrossings can be worked out for a process like this yet. "
            "That holds for an epidemic model's compartments taken one at a "
            "time as well, so path.I is not a way round it for now."
        )

    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"{where} the process is at {value!r}, which is not a number, so "
            "there is no way to tell whether it has reached a level. Asking "
            "when a process reaches a level only makes sense for one whose "
            "values are numbers. If this is a Markov chain, label its states "
            "with numbers -- MarkovChain(P, initial, state_labels=[0, 1, 2]) "
            "-- rather than with names."
        )
    return float(value)


def _reached(value, level, sign, strict=False):
    """Whether a value has got to the level, or past it.

    A path that moves in jumps can step straight over a level without ever
    equaling it, so "reached" has to mean reached *or passed* -- unlike the
    Gaussian case, where a continuous path cannot get to the other side
    without touching it.

    Parameters
    ----------
    value : float
        The value the path is at.
    level : float
        The level being watched for.
    sign : float
        ``1`` when the path has to rise to the level, ``-1`` when it has to
        fall to it.
    strict : bool, optional
        Whether sitting exactly *on* the level counts. Default is ``False``,
        which is what a hitting time wants. :func:`upcrossings` needs the
        strict version for the other half of its cycle: a path has only really
        come back below a level once it is properly below it, not merely
        touching it, or every touch would count as a fresh crossing.

    Returns
    -------
    bool
        ``True`` once the level is reached or passed.
    """
    gap = sign * (value - level)
    return gap > 0 if strict else gap >= 0


def _jump_hitting_time(path, level, max_time, start_time, sign=None, strict=False):
    """Return when a pure-jump path first reaches ``level``. Exact.

    A pure-jump path holds one value at a time and changes only at its jumps,
    so it can only reach a level at the moment of a jump -- which makes
    walking the jumps exact, with nothing between them to miss and nothing to
    narrow down afterwards. This is the same states-and-holding-times walk
    used to read such a path anywhere else in the package: the path sits at
    ``states[n]`` for ``interarrival_times[n]`` units of time.

    Parameters
    ----------
    path : DiscreteValued
        The sample path, providing ``get_states()`` and
        ``get_interarrival_times()``.
    level : float
        The level to reach.
    max_time : float
        Give up after this time and report ``inf``.
    start_time : float
        Start looking from here. The direction -- rise to the level or fall to
        it -- is taken from where the path is at this time, not from where it
        started out, unless ``sign`` says otherwise.
    sign : float, optional
        ``1`` to wait for a rise to the level, ``-1`` for a fall. Default is
        ``None``, which reads the direction off the path at ``start_time``.
        :func:`upcrossings` passes it explicitly, because it is looking for a
        specific one of the two.
    strict : bool, optional
        Whether to require passing the level rather than merely touching it.
        Default is ``False``. See :func:`_reached`.

    Returns
    -------
    float
        The clock time of the jump that first reached ``level``, or ``inf``.

    Raises
    ------
    TypeError
        If a state of the path is not a number.
    ValueError
        If the path jumps more than ``MAX_JUMPS`` times before ``max_time``.
    """
    states = path.get_states()
    holding_times = path.get_interarrival_times()

    entered = 0.0  # the clock time at which the current value was taken on
    n = 0

    while entered <= max_time:
        if n >= MAX_JUMPS:
            raise ValueError(
                f"The process jumps more than {MAX_JUMPS:,} times before time "
                f"{max_time:g}, which is too many to walk through. It is "
                "changing far more often than the stretch of time being "
                "searched, so either lower max_time or slow the process down."
            )

        value = _numeric_value(states[n], f"In state number {n},")
        left = entered + float(holding_times[n])

        # Only stretches that reach past `start_time` are being searched; the
        # earlier ones are walked through just to keep the clock.
        if left > start_time:
            reached_at = max(entered, start_time)
            if reached_at > max_time:
                break
            if sign is None:
                # The first value inside the search window fixes the direction:
                # below the level means wait for a rise, above means a fall.
                if value == level:
                    return float(reached_at)
                sign = 1.0 if value < level else -1.0
            if _reached(value, level, sign, strict):
                return float(reached_at)

        entered = left
        n += 1

    return float("inf")


def _step_hitting_time(
    path, level, max_time, start_time, samples_per_time, sign=None, strict=False
):
    """Return when a discrete-time path first reaches ``level``. Exact.

    A discrete-time path has no values between its steps, so there is nothing
    to miss: reading the steps in order *is* the whole path.

    Parameters
    ----------
    path : InfiniteTuple or DiscreteTimeFunction
        The sample path, read as ``path[n]`` at step ``n``.
    level : float
        The level to reach.
    max_time : float
        Give up after this time and report ``inf``.
    start_time : float
        Start looking from here.
    samples_per_time : float
        How many steps make up one unit of time, so that a step index can be
        reported as a time. This is 1 for the usual case, where a step *is*
        the unit of time (a random walk, a Markov chain, a moving-average
        series), and the process's sampling rate otherwise.
    sign : float, optional
        ``1`` to wait for a rise to the level, ``-1`` for a fall. Default is
        ``None``, which reads the direction off the path at ``start_time``.
    strict : bool, optional
        Whether to require passing the level rather than merely touching it.
        Default is ``False``. See :func:`_reached`.

    Returns
    -------
    float
        The time of the step that first reached ``level``, or ``inf``.

    Raises
    ------
    TypeError
        If a value of the path is not a number.
    """
    first_step = math.ceil(start_time * samples_per_time)
    last_step = math.floor(max_time * samples_per_time)

    for n in range(first_step, last_step + 1):
        value = _numeric_value(path[n], f"At step {n},")
        if sign is None:
            if value == level:
                return n / samples_per_time
            sign = 1.0 if value < level else -1.0
        if _reached(value, level, sign, strict):
            return n / samples_per_time

    return float("inf")


def _gaussian_hitting_time(
    read, cov_func, level, max_time, start_time, step, tol, sign=None
):
    """Return when a continuous path first reaches ``level``. Approximate.

    This is the scan described at the top of the module: step forward, and at
    each step either see the level reached outright or decide with the
    reflection-principle probability that it was reached and left again in
    between, then narrow down when. Whether the level was reached is exact for
    a Brownian motion or bridge; where inside the step it happened is
    approximate for every process. See :func:`hitting_time` for the full
    accuracy story.

    Parameters
    ----------
    read : callable
        ``read(t)`` gives the value to track at time ``t``, on the scale the
        crossing formula applies to. From :func:`_prepare`.
    cov_func : callable
        The covariance function of the process ``read`` reports.
    level : float
        The level to reach, on the same scale as ``read``.
    max_time : float
        Give up after this time and report ``inf``.
    start_time : float
        Start looking from here.
    step : float or None
        How far ahead to look at a time. ``None`` divides the window into 100.
    tol : float
        When to stop halving while narrowing a crossing down.
    sign : float, optional
        ``1`` to wait for a rise to the level, ``-1`` for a fall. Default is
        ``None``, which reads the direction off the path at ``start_time``.
        :func:`upcrossings` passes it explicitly, because it alternates
        between two levels and so is looking for a specific one of the two.

    Returns
    -------
    float
        The crossing time, or ``inf``.
    """
    start_value = read(start_time)

    if sign is None:
        if start_value == level:
            return float(start_time)
        # One piece of code covers rising to the level and falling to it:
        # measure everything as "distance still to go", positive either way.
        sign = 1.0 if start_value < level else -1.0

    gap0 = sign * (level - start_value)
    if gap0 <= 0:
        # Already at or past the level. With the direction read off the path
        # this cannot happen -- equality was dealt with just above, and the
        # sign was then chosen to make the gap positive -- but a caller asking
        # for a particular direction may well start on the far side.
        return float(start_time)

    if step is None:
        step = (max_time - start_time) / 100.0

    t0 = float(start_time)
    while t0 < max_time:
        t1 = min(t0 + step, max_time)
        gap1 = sign * (level - read(t1))

        if gap1 <= 0:
            # The level was reached by t1. Find out when.
            return _localize_crossing(
                read, level, sign, cov_func, t0, gap0, t1, gap1, tol
            )

        # Neither end reached the level, but the path may have gone there and
        # come back. Decide with the reflection-principle probability.
        rate = _local_variance_rate(cov_func, t0, t1)
        if rng.uniform() < _crossing_probability(gap0, gap1, rate, t1 - t0):
            return _localize_crossing(
                read, level, sign, cov_func, t0, gap0, t1, gap1, tol
            )

        t0, gap0 = t1, gap1

    return float("inf")


def _is_jump_path(path):
    """Whether a path can be walked jump by jump.

    A path that reports both its states and how long it holds each one changes
    only at its jumps -- a Poisson or renewal count, a compound Poisson total,
    a continuous-time Markov chain, a birth-death chain, a queue length.

    Parameters
    ----------
    path : object
        The sample path.

    Returns
    -------
    bool
        ``True`` if the path can be walked jump by jump.
    """
    if not isinstance(path, DiscreteValued):
        return False
    try:
        path.get_states()
        path.get_interarrival_times()
    except AttributeError:
        # A discrete-time Markov chain reports states but no holding times --
        # its steps *are* its jumps -- so it is walked as a discrete-time path
        # instead.
        return False
    return True


def _is_continuous_path(path):
    """Whether a path moves continuously, so has no values to walk one by one.

    A Gaussian-process path or a geometric Brownian motion. Such a path is
    everywhere between the times it has been asked about, which is what the
    reflection-principle machinery at the top of this module exists to deal
    with, and what stops its crossings of a single level from forming a
    sequence (see :func:`upcrossings`).

    Parameters
    ----------
    path : object
        The sample path.

    Returns
    -------
    bool
        ``True`` if the path is one of the two continuous kinds handled here.
    """
    is_gaussian = hasattr(path, "cov_func") and hasattr(path, "observed")
    is_geometric = hasattr(path, "brownian_path") and hasattr(path, "scale")
    return is_gaussian or is_geometric


def _steps_per_time(path):
    """How many steps of a discrete-time path make up one unit of time.

    Almost always 1: for a random walk, a Markov chain, or a moving-average
    series a step *is* the unit of time, and the index and the clock are the
    same thing. A path built on a
    :class:`~symbulate.index_sets.DiscreteTimeSequence` can be sampled faster
    than that, and then a step index has to be divided by the sampling rate to
    be a time.

    Parameters
    ----------
    path : object
        The sample path.

    Returns
    -------
    float
        The number of steps per unit of time.
    """
    index_set = getattr(path, "index_set", None)
    return float(getattr(index_set, "fs", 1))


def _tier_a_reader(path):
    """Return how to read ``path`` value by value, or ``None``.

    Both kinds of path that can be read exactly -- a pure-jump path and a
    discrete-time path -- are walked value by value, but by different walks.
    This decides which walk fits and hands it back behind one signature, so
    that a caller which has to look for a level over and over
    (:func:`upcrossings`) does not re-decide on every pass, and so that
    :func:`hitting_time` and :func:`upcrossings` cannot drift apart on what
    counts as an exactly-readable path. **A third kind of such path belongs
    here**, not at either call site.

    Parameters
    ----------
    path : object
        The sample path.

    Returns
    -------
    tuple or None
        ``(search, read_one)``, or ``None`` when the path cannot be walked value
        by value -- a continuous path, whose crossings need the machinery above.

        ``search(level, max_time, start_time, sign, strict)`` gives the first
        time from ``start_time`` onwards at which the path has reached ``level``
        in the direction ``sign``, or ``inf`` if it has not by ``max_time``.

        ``read_one()`` reads the path's very first value and checks that it is a
        number, raising if not. Whether a path's values can be compared with a
        level at all is a property of the path rather than of where you look, so
        one value settles it. It is for a caller that does not walk the path
        straight away and so would otherwise report a bad path long after the
        event.
    """
    if _is_jump_path(path):

        def search(level, max_time, start_time, sign, strict):
            return _jump_hitting_time(
                path, level, max_time, start_time, sign=sign, strict=strict
            )

        def read_one():
            return _numeric_value(path.get_states()[0], "In state number 0,")

        return search, read_one

    if isinstance(path, (InfiniteTuple, DiscreteTimeFunction)):
        samples_per_time = _steps_per_time(path)

        def search(level, max_time, start_time, sign, strict):
            return _step_hitting_time(
                path,
                level,
                max_time,
                start_time,
                samples_per_time,
                sign=sign,
                strict=strict,
            )

        def read_one():
            return _numeric_value(path[0], "At step 0,")

        return search, read_one

    return None


def _tier_b_reader(path, level, reset, step, tol):
    """Return how to search a continuous path between two levels.

    The counterpart of :func:`_tier_a_reader` for a path that cannot be walked
    value by value. Only :func:`upcrossings` with a ``reset`` level needs this:
    a band gives a continuous path's crossings something to be separated by,
    which is what makes a sequence of them exist at all.

    Parameters
    ----------
    path : GaussianProcessResult or GeometricBrownianMotionResult
        The sample path.
    level : float
        The level being crossed, on the path's own scale.
    reset : float
        The level that re-arms a crossing, on the path's own scale.
    step : float or None
        How far ahead to look at a time while scanning.
    tol : float
        When to stop halving while narrowing a crossing down.

    Returns
    -------
    tuple
        ``(search, level, reset)`` -- the search, plus the two levels restated
        on whatever scale ``search`` works on. They are handed back rather than
        left to the caller because for a geometric Brownian motion that scale
        is the log scale, and the walk should not have to know.
    """
    read, cov_func, scaled_level = _prepare(path, level)
    # _prepare's third value is the level restated on the scale `read` reports,
    # so asking it a second time is how the reset level is put on that same
    # scale -- and it is also what rejects a reset of 0 or below for a
    # geometric Brownian motion. The read and cov_func are the same pair both
    # times, so the second pair is discarded. Taking the log is increasing, so
    # whichever of the two levels is the lower stays the lower.
    _, _, scaled_reset = _prepare(path, reset)

    def search(target, max_time, start_time, sign, strict):
        # `strict` has nothing to do here. It exists because a jump path can
        # leap over a level, so "reached" and "properly past" differ; a
        # continuous path cannot get to the far side of a level without
        # touching it. Keeping the level and the reset apart is the band's job
        # instead.
        return _gaussian_hitting_time(
            read, cov_func, target, max_time, start_time, step, tol, sign=sign
        )

    return search, scaled_level, scaled_reset


def _prepare(path, level):
    """Work out how to read a path, and on what scale to compare it.

    The crossing formula needs a process whose stretches between two known
    values are Brownian bridges. Gaussian-process paths qualify directly. A
    geometric Brownian motion does not -- but taking its log gives back a
    Brownian motion with drift, which does, and a price reaching a level is
    exactly its log reaching the log of that level. So the whole question is
    moved onto the log scale, where it can be answered exactly.

    Parameters
    ----------
    path : GaussianProcessResult or GeometricBrownianMotionResult
        The sample path.
    level : float
        The level to reach, on the path's own scale.

    Returns
    -------
    tuple
        ``(read, cov_func, level)`` -- a function giving the value to track at
        a time, the covariance function of whatever that function reports, and
        the level restated on the same scale.

    Raises
    ------
    NotImplementedError
        If the path is neither of the two supported kinds.
    ValueError
        If a geometric Brownian motion is asked about a level of 0 or below,
        which it can never reach.
    """
    # A geometric Brownian motion: undo the exponential and work in log space.
    brownian_path = getattr(path, "brownian_path", None)
    if brownian_path is not None and hasattr(path, "scale"):
        if level <= 0:
            raise ValueError(
                f"level must be positive for a geometric Brownian motion, got "
                f"{level}. It multiplies its starting value by positive "
                f"numbers, so it never reaches 0 or goes below it."
            )
        initial = path.initial
        scale = path.scale

        def read(t):
            return float(np.log(float(path(t)) / initial))

        # log(price) is a Brownian motion with drift, whose covariance is the
        # plain Brownian one. The drift does not appear: a bridge's law does
        # not depend on it once both endpoints are known.
        def cov_func(s, t):
            return scale**2 * min(s, t)

        return read, cov_func, float(np.log(level / initial))

    # An ordinary Gaussian-process path: read it directly.
    cov_func = getattr(path, "cov_func", None)
    if cov_func is not None and hasattr(path, "observed"):

        def read(t):
            return float(path(t))

        return read, cov_func, float(level)

    # A non-homogeneous Poisson or Cox count. This one is worth its own message
    # rather than the general one below: it *is* a jump process, so being told
    # only that it is unsupported would be baffling. What it lacks is its jump
    # times on the clock -- it knows them on the "expected count" scale, and
    # turning those back into clock times means undoing the cumulative rate.
    if hasattr(path, "cumulative_rate") and hasattr(
        path, "standard_interarrival_times"
    ):
        raise NotImplementedError(
            "hitting_time cannot read a "
            "NonHomogeneousPoissonProcess or CoxProcess path yet. Its count "
            "does move in jumps, but unlike a PoissonProcess it does not know "
            "when those jumps happen on the clock -- it knows them on the "
            "'expected number of events' scale, and working back to a time "
            "from that is not built. What you can do meanwhile is ask for the "
            "count itself at a time, which is exact: path(t) for one path, or "
            "N[t].sim(10000) for the whole distribution, and the chance of "
            "having reached a level k by time t is the share of those that are "
            "k or more."
        )

    raise NotImplementedError(
        f"hitting_time does not know how to read a {type(path).__name__} yet. "
        "It handles Gaussian processes (BrownianMotion, BrownianBridge, "
        "OrnsteinUhlenbeck, FractionalBrownianMotion, "
        "GeometricBrownianMotion, or a GaussianProcess you built yourself), "
        "processes that move in jumps (PoissonProcess, RenewalProcess, "
        "CompoundPoissonProcess, ContinuousTimeMarkovChain, the birth-death "
        "and M/M queues, and the G/G queues), and discrete-time processes "
        "(RandomWalk, MarkovChain, MA). A DiffusionProcess, CIR, or "
        "MertonJumpDiffusion needs its own approximation, which is not built "
        "yet -- a Merton jump diffusion wanders continuously between its "
        "jumps, so walking the jumps would miss everything in between. A "
        "process whose value is several numbers at once, such as an epidemic "
        "model, has no single level to reach, and neither it nor one of its "
        "compartments on its own is handled yet."
    )


def _validate_window(level, max_time, start_time):
    """Check the level and the stretch of time to search.

    Shared by :func:`hitting_time` and :func:`upcrossings`, which ask for a
    level over the same kind of window and so reject the same things.

    Raises
    ------
    TypeError
        If any argument is not a number.
    ValueError
        If ``start_time`` is negative, or ``max_time`` is not after it.
    """
    for name, value in [
        ("level", level),
        ("max_time", max_time),
        ("start_time", start_time),
    ]:
        if not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}.")

    if start_time < 0:
        raise ValueError(
            f"start_time must be at least 0, got {start_time}. Processes here "
            f"begin at time 0."
        )
    if max_time <= start_time:
        raise ValueError(
            f"max_time must come after start_time, got start_time="
            f"{start_time} and max_time={max_time}. There is no stretch of "
            f"time to search."
        )


def _validate_precision(step, tol):
    """Check how finely a continuous path is to be looked at.

    Shared by :func:`hitting_time` and :func:`upcrossings`, which scan a
    continuous path the same way and so reject the same things.

    Raises
    ------
    TypeError
        If ``step`` or ``tol`` is not a number.
    ValueError
        If ``step`` or ``tol`` is not positive.
    """
    if not isinstance(tol, numbers.Real):
        raise TypeError(f"tol must be a number, got {type(tol).__name__}.")
    if step is not None and not isinstance(step, numbers.Real):
        raise TypeError(f"step must be a number, got {type(step).__name__}.")

    if step is not None and step <= 0:
        raise ValueError(
            f"step must be positive, got {step}. It is how far ahead to look "
            f"at a time while scanning for the level."
        )
    if tol <= 0:
        raise ValueError(
            f"tol must be positive, got {tol}. It is how precisely to pin "
            f"down the crossing time."
        )


def _validate(level, max_time, start_time, step, tol):
    """Check the arguments of :func:`hitting_time`.

    Raises
    ------
    TypeError
        If any argument is not a number.
    ValueError
        If ``max_time`` is not after ``start_time``, or if ``step`` or ``tol``
        is not positive.
    """
    _validate_window(level, max_time, start_time)
    _validate_precision(step, tol)


def _validate_upcrossings(level, reset, max_time, start_time, step, tol):
    """Check the arguments of :func:`upcrossings`.

    Raises
    ------
    TypeError
        If any argument is not a number.
    ValueError
        If ``max_time`` is not after ``start_time``, if ``reset`` equals
        ``level``, or if ``step`` or ``tol`` is not positive.
    """
    _validate_window(level, max_time, start_time)
    _validate_precision(step, tol)

    if reset is None:
        return

    if not isinstance(reset, numbers.Real):
        raise TypeError(f"reset must be a number, got {type(reset).__name__}.")

    if reset == level:
        raise ValueError(
            f"reset must be different from level, and both are {level}. reset "
            f"is the level the process has to get back to before it can cross "
            f"level again, so the two have to be apart: reset below level "
            f"counts crossings upward, reset above it counts them downward. "
            f"Leave reset out altogether to count every crossing of level."
        )


def hitting_time(process, level, max_time=100.0, start_time=0.0, step=None, tol=1e-6):
    """Return the first time a process reaches ``level``.

    Pass a whole process and you get a random variable back, which you can
    simulate, plot, and summarize like any other. Pass a single sample path
    and you get one number.

    Whether the process has to rise or fall to reach the level is worked out
    automatically from where the path starts: starting below ``level`` this
    finds the first time it rises to it, and starting above it finds the
    first time it falls to it.

    Three kinds of process are handled, and which one you have decides both
    how the search is done and how accurate the answer is (see the notes):

    - **Processes that move in jumps**, holding one value at a time --
      :class:`~symbulate.poisson_process.PoissonProcess`,
      :class:`~symbulate.renewal_process.RenewalProcess`,
      :class:`~symbulate.renewal_process.CompoundPoissonProcess`,
      :class:`~symbulate.markov_chains.ContinuousTimeMarkovChain`, the
      birth-death and M/M queues, and the G/G queues. **Exact.** (Not
      :class:`~symbulate.poisson_process.NonHomogeneousPoissonProcess` or
      :class:`~symbulate.poisson_process.CoxProcess`, whose jumps are known on
      the expected-count scale rather than on the clock -- see the error they
      raise.)
    - **Discrete-time processes**, which have a value at each step and nothing
      in between -- :class:`~symbulate.random_walk.RandomWalk`,
      :class:`~symbulate.markov_chains.MarkovChain`,
      :class:`~symbulate.time_series.MA`. **Exact.**
    - **Gaussian processes** and
      :class:`~symbulate.gaussian_process.GeometricBrownianMotion`, whose paths
      wiggle between any two times you look at. Exact for Brownian motion and
      bridges, approximate otherwise.

    Parameters
    ----------
    process : RV, RandomProcess, or a sample path
        The process, or a single sample path drawn from one.
    level : float
        The level to reach. Must be positive for a geometric Brownian motion,
        which never reaches 0.
    max_time : float, optional
        Give up after this time and report ``inf``. Needed because a level may
        simply never be reached. Default is 100. For a discrete-time process a
        step *is* the unit of time, so this counts steps.
    start_time : float, optional
        Start looking from here rather than from time 0. Default is 0. Useful
        for finding a *later* crossing after an earlier one. The direction --
        rise to the level or fall to it -- is then taken from where the path is
        at ``start_time``, not from where it started out.
    step : float, optional
        How far ahead to look at a time while scanning a Gaussian process.
        Defaults to one hundredth of the stretch being searched. This sets how
        precise the reported time is -- see the notes -- as well as how long
        the search takes, so smaller is sharper and slower. **Ignored** for a
        jump or discrete-time process, which is walked value by value and needs
        no scanning.
    tol : float, optional
        When to stop halving while narrowing a crossing down. Default is
        ``1e-6``. This is not the accuracy of the answer; ``step`` is. Also
        **ignored** for a jump or discrete-time process, which has nothing to
        narrow down.

    Returns
    -------
    RV or float
        A random variable, if given a process; the crossing time as a number,
        if given a single path. ``inf`` means the level was not reached
        before ``max_time``.

    Raises
    ------
    TypeError
        If an argument has the wrong type, or the path's values are not
        numbers -- a Markov chain labelled with names rather than numbers has
        no level to reach.
    ValueError
        If ``max_time`` is not after ``start_time``, ``step`` or ``tol`` is not
        positive, or a jump process jumps too many times before ``max_time`` to
        walk through.
    NotImplementedError
        If the path is from a general diffusion
        (:class:`~symbulate.diffusion_process.DiffusionProcess`,
        :class:`~symbulate.diffusion_process.CIR`,
        :class:`~symbulate.diffusion_process.MertonJumpDiffusion`), or is not a
        single number at each time. Neither is handled yet.

    Notes
    -----
    **Reaching a level means getting to it or past it** for a jump or
    discrete-time process. Such a path can step straight over a level without
    ever equaling it -- a random walk going from 4 to 6 never sits at 5 -- so
    the first time it is *at or beyond* the level is the only sensible reading,
    and it is the standard definition for these processes. For a Gaussian
    process the question does not arise: a continuous path cannot get to the
    other side without touching it.

    **Jump and discrete-time processes are exact**, and no argument makes them
    more or less so. A jump path holds one value at a time, so it can only
    reach a level at the moment of a jump, and walking the jumps sees every one
    of them; a discrete-time path has nothing between its steps at all. Neither
    has anything hiding between the values it reports, which is what ``step``
    and ``tol`` exist to deal with -- so both are ignored for these.

    **How accurate this is** for a Gaussian process, separated into the two
    questions it answers:

    *Whether the level is reached, and in which ``step``-sized stretch.* For
    :class:`BrownianMotion` (with or without drift), :class:`BrownianBridge`,
    and :class:`GeometricBrownianMotion` this is exact, at any ``step``,
    because the stretch between two known values of such a process really is
    a Brownian bridge -- on the log scale, for the geometric case -- and the
    crossing probability for one is a closed form. For other Gaussian
    processes -- :class:`OrnsteinUhlenbeck`,
    :class:`FractionalBrownianMotion` -- the same formula is applied to a
    Brownian stand-in fitted to the process locally, so it is an
    approximation that improves as ``step`` shrinks.

    *Exactly where inside that stretch.* Approximate for every process,
    including Brownian motion. Halving the stretch narrows the crossing down,
    but each halving asks the path for a midpoint value drawn without
    accounting for the crossing already known to have happened, which tilts
    the reported time slightly late. The error is therefore bounded by about
    one ``step``, and ``step`` -- not ``tol`` -- is what controls how precise
    the reported time is. ``tol`` only says when to stop halving.

    So treat ``step`` as a precision setting, not just a speed one. The
    default divides the search window into 100 pieces, which is fine for
    plotting a distribution of hitting times; shrink it if you need the
    individual times to be sharp.

    **Why a plain scan is not enough.** A path only knows the times it has
    been asked about. Checking just those times would miss a crossing that
    happened and reversed in between, which biases hitting times upward. This
    is why the crossing is decided by a weighted coin flip rather than by
    looking at endpoint values alone.

    Examples
    --------
    >>> from symbulate import *
    >>> B = BrownianMotion()
    >>> T = hitting_time(B, level=2)
    >>> T.sim(100).mean()      # doctest: +SKIP
    31.5
    >>> # A single path gives a single number
    >>> path = B.draw()
    >>> hitting_time(path, level=2)   # doctest: +SKIP
    5.72

    A random walk, where the answer is a step number and is exact:

    >>> steps = hitting_time(RandomWalk(p=0.5), level=3, max_time=200)
    >>> steps.sim(100).mean()   # doctest: +SKIP
    88.4

    How long until a queue first has 5 people in it:

    >>> busy = hitting_time(GG1(Exponential(rate=1), Exponential(rate=1.2)), level=5)
    >>> busy.sim(100).mean()   # doctest: +SKIP
    37.2

    See Also
    --------
    BrownianMotion : The process this is exact for.
    """
    _validate(level, max_time, start_time, step, tol)

    # Handed a whole process, hand back a random variable. apply() passes the
    # entire sample path through, so this is the same computation done once
    # per simulated path. Every process here is an RV; only some of them are
    # also a RandomProcess, so RV is the check that catches all of them.
    if isinstance(process, RV):
        return process.apply(
            lambda path: hitting_time(
                path,
                level=level,
                max_time=max_time,
                start_time=start_time,
                step=step,
                tol=tol,
            )
        )

    # A path that moves in jumps, or in steps, is walked exactly -- there is
    # nothing hiding between the values it reports, so `step` and `tol`, which
    # exist to deal with what a continuous path does in between, have nothing
    # to do here. The direction is left to the walk to read off the path, and
    # reaching the level counts even without landing on it.
    reader = _tier_a_reader(process)
    if reader is not None:
        search, _ = reader
        return search(float(level), float(max_time), float(start_time), None, False)

    read, cov_func, level = _prepare(process, level)
    return _gaussian_hitting_time(
        read, cov_func, level, max_time, start_time, step, tol
    )


def _upcrossings_unsupported(path):
    """Explain why ``path`` has no list of crossing times, and raise.

    Reached only for a path that can be neither walked value by value nor
    scanned between a pair of levels. There are three such cases and they fail
    for genuinely different reasons, so each gets its own message rather than
    one generic "unsupported".

    Parameters
    ----------
    path : object
        The sample path.

    Raises
    ------
    NotImplementedError
        Always.
    """
    # A Gaussian-process path, or a geometric Brownian motion, asked for the
    # crossings of a single level. This is not a missing feature: a level on its
    # own has no *sequence* of crossings for a continuous path, and a reset
    # level is what gives it one -- see the Notes in `upcrossings`.
    if _is_continuous_path(path):
        raise NotImplementedError(
            f"upcrossings needs a reset level to list the crossing times of a "
            f"{type(path).__name__} path, because a level on its own has no "
            "first, second, third crossing to list. A path that moves "
            "continuously recrosses a level infinitely often in every stretch "
            "of time after it first touches it, however short that stretch is, "
            "so its crossings of that one level are packed together rather "
            "than coming one after another, and any count of them would only "
            "report how finely the path happened to be looked at. Say how far "
            "back the path has to come before a crossing counts again and the "
            "question has an answer: upcrossings(path, level=1, reset=0) is "
            "the times it reaches 1 having been down to 0 in between. "
            "Alternatively hitting_time works here as it is, and asking it "
            "again from a later start_time gives a later crossing."
        )

    # A non-homogeneous Poisson or Cox count -- the same gap `hitting_time`
    # documents, worth repeating here because it *is* a jump process and being
    # told only that it is unsupported would be baffling.
    if hasattr(path, "cumulative_rate") and hasattr(
        path, "standard_interarrival_times"
    ):
        raise NotImplementedError(
            "upcrossings cannot read a NonHomogeneousPoissonProcess or "
            "CoxProcess path yet, for the same reason hitting_time cannot: its "
            "count does move in jumps, but it does not know when those jumps "
            "happen on the clock -- it knows them on the 'expected number of "
            "events' scale, and working back to a time from that is not built. "
            "Bear in mind that a count only ever goes up, so it crosses a "
            "level at most once in any case. What you can do meanwhile is ask "
            "for the count itself at a time, which is exact: path(t) for one "
            "path, or N[t].sim(10000) for the whole distribution."
        )

    raise NotImplementedError(
        f"upcrossings does not know how to read a {type(path).__name__} yet. It "
        "handles processes that move in jumps (PoissonProcess, RenewalProcess, "
        "CompoundPoissonProcess, ContinuousTimeMarkovChain, the birth-death "
        "and M/M queues, and the G/G queues) and discrete-time processes "
        "(RandomWalk, MarkovChain, MA) -- the ones whose crossings of a level "
        "are separated from one another. A DiffusionProcess, CIR, or "
        "MertonJumpDiffusion moves continuously between the times you look at "
        "it, so it has no such sequence of crossings, and its hitting time is "
        "not built yet either. A process whose value is several numbers at "
        "once, such as an epidemic model, has no single level to cross, and "
        "neither it nor one of its compartments on its own is handled yet."
    )


class _UpcrossingWalk:
    """Finds the upcrossing times of one path, in order, as they are asked for.

    An upcrossing has to come *from below*, so finding one is two searches
    rather than one: get to where the path is strictly below the level, then
    find where it next rises back to it. Repeating that pair walks up the
    sequence. The times found are kept, so asking for the same one twice does
    not walk the path again -- the caching pattern
    :class:`~symbulate.result.InfiniteVector` is built around.

    Once a search runs past ``max_time`` there are no more crossings to find,
    and every later position of the sequence is ``inf``. That is recorded rather
    than rediscovered, so a sequence that has run out does not re-walk the whole
    path on each new index.

    Given a ``reset`` level the first of the two searches aims at that level
    instead, and the crossing then has to come from beyond it rather than
    merely from the other side of ``level``. Everything else is the same walk:
    ``reset`` below ``level`` counts crossings upward, ``reset`` above it counts
    crossings downward, and the direction is taken from which of the two is the
    lower rather than asked for separately.

    Parameters
    ----------
    search : callable
        The walk or scan for this path, from :func:`_tier_a_reader` or
        :func:`_tier_b_reader`.
    level : float
        The level being crossed.
    max_time : float
        Stop looking after this time.
    start_time : float
        Start looking from here.
    reset : float, optional
        The level the path has to get back to before another crossing of
        ``level`` counts. Default is ``None``, which asks only that the path be
        strictly the other side of ``level`` itself.

    Attributes
    ----------
    times : list of float
        The crossing times found so far, in order.
    """

    def __init__(self, search, level, max_time, start_time, reset=None):
        self.search = search
        self.level = level
        self.max_time = max_time
        self.times = []
        # Where the next pair of searches begins. After a crossing is found this
        # is the crossing itself, which is at or beyond the level -- so the next
        # "get back" search starts by moving off it, as it should.
        self.cursor = start_time
        self.finished = False

        if reset is None:
            # No band: getting back means getting *strictly* the other side of
            # the level itself, or a path sitting on it would cross it again at
            # every step.
            self.arm_level = level
            self.arm_strict = True
            self.sign = 1.0
        else:
            self.arm_level = reset
            self.arm_strict = False
            self.sign = 1.0 if reset < level else -1.0

    def _find_one_more(self):
        """Add the next crossing time, or record that there are none left."""
        armed_at = self.search(
            self.arm_level, self.max_time, self.cursor, -self.sign, self.arm_strict
        )
        if math.isinf(armed_at):
            self.finished = True
            return

        crossed_at = self.search(self.level, self.max_time, armed_at, self.sign, False)
        if math.isinf(crossed_at):
            self.finished = True
            return

        self.times.append(crossed_at)
        self.cursor = crossed_at

    def crossing_at(self, n):
        """Return the time of the ``n``-th upcrossing, counting from 0.

        Parameters
        ----------
        n : int
            Which crossing to report.

        Returns
        -------
        float
            Its time, or ``inf`` if the path does not cross the level that many
            times before ``max_time``.
        """
        while not self.finished and len(self.times) <= n:
            self._find_one_more()
        return self.times[n] if n < len(self.times) else float("inf")


def upcrossings(
    process, level, reset=None, max_time=100.0, start_time=0.0, step=None, tol=1e-6
):
    """Return every time a process crosses through ``level``, in order.

    Where :func:`hitting_time` answers "when does it first get there?", this
    answers "when does it get there, and then when again, and again?" Pass a
    whole process and you get a random variable back; pass a single sample path
    and you get the sequence of times for that one path.

    The sequence is an :class:`~symbulate.result.InfiniteVector`, so it is
    computed only as far as you look: ``times[0]`` is the first crossing,
    ``times[3]`` the fourth, and nothing beyond what you ask for is worked out.

    Processes that move in **jumps** or in **steps** need nothing further: their
    crossings of a level are already separated from one another, and are found
    exactly. A process with **continuous** paths, such as
    :class:`BrownianMotion`, needs a ``reset`` level as well, because its
    crossings of a single level are not separated at all -- see the notes.

    ``reset`` is how far back the path has to come before another crossing of
    ``level`` counts, and it decides the direction: below ``level`` it counts
    crossings upward, above ``level`` it counts them downward. It is allowed for
    every process, and is worth using on a jump or step process too whenever
    small wobbles about the level should not each count as a crossing.

    Parameters
    ----------
    process : RV or a sample path
        The process, or a single sample path drawn from one.
    level : float
        The level to cross.
    reset : float, optional
        The level the path has to get back to before another crossing of
        ``level`` counts. Default is ``None``, which asks only that the path go
        strictly the other side of ``level`` itself -- enough for a jump or step
        process, and not enough for a continuous one, which then raises. Below
        ``level`` counts crossings upward, above it counts them downward, and it
        must not equal ``level``.
    max_time : float, optional
        Stop looking after this time. Default is 100. Crossings after it are
        reported as ``inf``. For a discrete-time process a step *is* the unit
        of time, so this counts steps.
    start_time : float, optional
        Start looking from here rather than from time 0. Default is 0.
    step : float, optional
        How far ahead to look at a time while scanning a **continuous** path.
        Defaults to one hundredth of the window being searched. Ignored by jump
        and step processes, which are read exactly. This is the accuracy knob
        here -- see the notes.
    tol : float, optional
        When to stop halving while narrowing a crossing of a continuous path
        down. Default is ``1e-6``. Ignored by jump and step processes. As in
        :func:`hitting_time` this is not the accuracy of the answer; ``step``
        is.

    Returns
    -------
    RV or InfiniteVector
        A random variable, if given a process -- index it to get the ``n``-th
        crossing time as a random variable of its own,
        ``upcrossings(X, level=3)[0]``. The sequence of times itself, if given a
        single path. ``inf`` at position ``n`` means the path did not cross the
        level ``n + 1`` times before ``max_time``, and every position after an
        ``inf`` is ``inf`` too.

    Raises
    ------
    TypeError
        If an argument has the wrong type, or the path's values are not numbers
        -- a Markov chain labelled with names rather than numbers has no level
        to cross.
    ValueError
        If ``max_time`` is not after ``start_time``, if ``reset`` equals
        ``level``, or the process jumps too many times before ``max_time`` to
        walk through.
    NotImplementedError
        If the path is continuous (any Gaussian process or
        :class:`GeometricBrownianMotion`) and no ``reset`` was given, if it is a
        general diffusion, if it is a
        :class:`~symbulate.poisson_process.NonHomogeneousPoissonProcess` or
        :class:`~symbulate.poisson_process.CoxProcess` count, or if it is not a
        single number at each time. Each case says why.

    Notes
    -----
    **What counts as an upcrossing.** The path has to arrive at the level
    *from below*: a crossing is a time when the path is at or above ``level``
    having been strictly below it beforehand. So the path must drop under the
    level again between one crossing and the next, which is what keeps the
    crossings apart and makes them a sequence. It also means that if the path is
    already at or above ``level`` when you start looking, that is not counted as
    a crossing -- it has to go below first -- so the first entry here can come
    later than :func:`hitting_time`. When the path starts strictly below the
    level, ``upcrossings(X, level=a)[0]`` and ``hitting_time(X, level=a)`` are
    the same thing.

    With a ``reset`` level it is that level the path has to get back to, rather
    than merely the other side of ``level``: a crossing is a time the path
    reaches ``level`` having been to ``reset`` since the previous one. Set
    ``reset`` above ``level`` and everything above is mirrored -- the path
    arrives from above, and the crossings counted are downward.

    Reaching the level means reaching it **or passing it**, exactly as for
    :func:`hitting_time`: a walk going from 4 to 6 is never at 5, so the step
    that took it to 6 is the crossing of 5.

    **For a jump or step process this is exact**, for the same reason Tier A of
    :func:`hitting_time` is: a jump path holds one value at a time so it can
    only cross at a jump, and a discrete-time path has no values between its
    steps. There is nothing hidden between the values the path reports, so there
    is no step size to choose and no accuracy to trade off. ``step`` and ``tol``
    are accepted and ignored for these, so that one call works across a mixture
    of processes.

    **Why a continuous process needs a reset level.** A Brownian motion
    recrosses a level infinitely often in every stretch of time after it first
    touches it, however short -- the crossing times of a single level are packed
    together, not spread out one after another. So there is no second or third
    crossing of ``level`` alone to report: any count would be a count of how
    finely the path was looked at, not a property of the path. It would grow
    without limit as ``step`` shrank.

    A reset level fixes that, because the path then has a finite distance to
    travel between one crossing and the next, so only finitely many fit into a
    finite stretch of time. That is what makes the sequence exist. It is also
    what makes ``reset`` a modelling choice rather than a technicality: crossing
    110 having been down to 90 and crossing 110 having been down to 109 are
    different questions, both perfectly good, and the answers differ.

    **For a continuous process this is approximate, but stable.** Each leg of
    the search is a :func:`hitting_time` scan, so it carries that function's
    accuracy: whether the level was reached is settled by the
    reflection-principle coin flip rather than by looking, which is exact for a
    Brownian motion, and where inside a step it happened leans slightly late.

    What matters more here is that the *number* of crossings does not run away
    with ``step``, which is what makes the question well-posed. Each crossing is
    pinned down to within ``tol`` before the next is looked for, so a coarse
    ``step`` does not stop two nearby crossings being told apart, and over a 25x
    range of ``step`` the mean count of a Brownian motion's crossings moves by
    less than its simulation error (see ``MODEL-DECISIONS.md``). Contrast the
    single-level case, where the count grows without limit. Halving ``step`` is
    still the way to check on a process of your own.

    **A count crosses a level at most once.** A
    :class:`~symbulate.poisson_process.PoissonProcess` or
    :class:`~symbulate.renewal_process.RenewalProcess` count only ever goes up,
    so it never comes back below a level to cross it again: the answer is the
    hitting time followed by ``inf`` forever. This is worth knowing rather than
    surprising -- upcrossings are interesting for processes that move both ways,
    such as a random walk, a queue length, a continuous-time Markov chain, or a
    :class:`~symbulate.renewal_process.CompoundPoissonProcess` with negative
    jumps.

    Examples
    --------
    >>> from symbulate import *
    >>> walk = RandomWalk(p=0.5).draw()
    >>> times = upcrossings(walk, level=2, max_time=500)
    >>> times[0], times[1]     # doctest: +SKIP
    (14.0, 22.0)

    How often a queue reaches five customers in its first 200 time units -- the
    third such time, over many simulated queues:

    >>> queue = GG1(Exponential(rate=1), Exponential(rate=1.2))
    >>> third = upcrossings(queue, level=5, max_time=200)[2]
    >>> third.sim(100).mean()      # doctest: +SKIP
    92.7

    A continuous process needs the reset level. When does a share worth \\$100
    recover to \\$110, having dipped to \\$90 in between?

    >>> stock = GeometricBrownianMotion(initial=100, growth_rate=0, scale=0.4)
    >>> recoveries = upcrossings(stock, level=110, reset=90, max_time=5, step=0.01)
    >>> recoveries[0].sim(100).mean()      # doctest: +SKIP
    2.03

    See Also
    --------
    hitting_time : The first time a process reaches a level.
    """
    _validate_upcrossings(level, reset, max_time, start_time, step, tol)

    # Handed a whole process, hand back a random variable, whose value on each
    # simulated path is that path's sequence of crossing times. Every process
    # here is an RV; only some are also a RandomProcess, so RV is the check that
    # catches all of them.
    if isinstance(process, RV):
        return process.apply(
            lambda path: upcrossings(
                path,
                level=level,
                reset=reset,
                max_time=max_time,
                start_time=start_time,
                step=step,
                tol=tol,
            )
        )

    walk_level = float(level)
    walk_reset = None if reset is None else float(reset)

    reader = _tier_a_reader(process)
    if reader is not None:
        # A path that can be walked value by value, which is exact and needs
        # neither `step` nor `tol`.
        search, read_one = reader

        # The sequence below is computed only as far as it is looked at, so a
        # path whose values cannot be compared with a level -- an epidemic
        # model's compartment counts, a Markov chain labelled with names --
        # would otherwise be accepted quietly here and complain only once
        # somebody indexed the result. One value read now puts the error where
        # the mistake is, which is what hitting_time gets for free by walking
        # straight away.
        read_one()
    elif walk_reset is not None and _is_continuous_path(process):
        # A continuous path, with a band to keep its crossings apart. The two
        # levels come back restated on whatever scale the scan works on, which
        # is the log scale for a geometric Brownian motion.
        search, walk_level, walk_reset = _tier_b_reader(
            process, walk_level, walk_reset, step, tol
        )
    else:
        _upcrossings_unsupported(process)

    walk = _UpcrossingWalk(
        search, walk_level, float(max_time), float(start_time), reset=walk_reset
    )
    return InfiniteVector(walk.crossing_at)
