"""When does a random process first reach a level?

This module provides :func:`hitting_time`, which answers "how long until the
process first gets to ``level``?" for a sample path -- or, given a whole
process, hands back a random variable you can simulate like any other.

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
:class:`~symbulate.diffusion_process.CIR`) still need their own approximation,
and :func:`hitting_time` says so plainly rather than guessing.
"""

import math
import numbers

import numpy as np

from .random_variables import RV
from .result import DiscreteTimeFunction, DiscreteValued, InfiniteTuple

# Guard against a singular covariance matrix when two times are nearly equal,
# the same trick and constant gaussian_process.py uses for the same reason.
MACHINE_EPS = 1e-12

# A backstop on walking a jump path, so that a process jumping far more often
# than the stretch of time being searched reports a readable error instead of
# appearing to hang.
MAX_JUMPS = 1_000_000

rng = np.random.default_rng()


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
    TypeError
        If the value is not a number, so cannot be compared with a level.
    """
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


def _reached(value, level, sign):
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

    Returns
    -------
    bool
        ``True`` once the level is reached or passed.
    """
    return sign * (value - level) >= 0


def _jump_hitting_time(path, level, max_time, start_time):
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
        started out.

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
    sign = None
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
            if _reached(value, level, sign):
                return float(reached_at)

        entered = left
        n += 1

    return float("inf")


def _step_hitting_time(path, level, max_time, start_time, samples_per_time):
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

    sign = None
    for n in range(first_step, last_step + 1):
        value = _numeric_value(path[n], f"At step {n},")
        if sign is None:
            if value == level:
                return n / samples_per_time
            sign = 1.0 if value < level else -1.0
        if _reached(value, level, sign):
            return n / samples_per_time

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
        initial_value = path.initial_value
        scale = path.scale

        def read(t):
            return float(np.log(float(path(t)) / initial_value))

        # log(price) is a Brownian motion with drift, whose covariance is the
        # plain Brownian one. The drift does not appear: a bridge's law does
        # not depend on it once both endpoints are known.
        def cov_func(s, t):
            return scale**2 * min(s, t)

        return read, cov_func, float(np.log(level / initial_value))

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
        "(RandomWalk, MarkovChain, MA). A DiffusionProcess or CIR needs its "
        "own approximation, which is not built yet. A process whose value is "
        "several numbers at once, such as an epidemic model, has no single "
        "level to reach -- ask about one compartment of it instead."
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
    for name, value in [
        ("level", level),
        ("max_time", max_time),
        ("start_time", start_time),
        ("tol", tol),
    ]:
        if not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if step is not None and not isinstance(step, numbers.Real):
        raise TypeError(f"step must be a number, got {type(step).__name__}.")

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
        :class:`~symbulate.diffusion_process.CIR`), or is not a single number
        at each time. Neither is handled yet.

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
    # to do here.
    if _is_jump_path(process):
        return _jump_hitting_time(
            process, float(level), float(max_time), float(start_time)
        )

    if isinstance(process, (InfiniteTuple, DiscreteTimeFunction)):
        return _step_hitting_time(
            process,
            float(level),
            float(max_time),
            float(start_time),
            _steps_per_time(process),
        )

    read, cov_func, level = _prepare(process, level)

    start_value = read(start_time)
    if start_value == level:
        return float(start_time)

    # One piece of code covers rising to the level and falling to it: measure
    # everything as "distance still to go", which is positive either way.
    sign = 1.0 if start_value < level else -1.0
    gap0 = sign * (level - start_value)

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
