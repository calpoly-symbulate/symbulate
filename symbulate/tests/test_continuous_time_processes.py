"""The shared interface every continuous-time, discrete-state process offers.

These processes all behave the same way: they sit at one value for a random
stretch of continuous time, jump to another, and sit there. So each of them
should support the same four things, and this file checks all of them against
every such process in one table, so that a newly added process cannot quietly
skip one:

- a ``...ProbabilitySpace`` class, exported from ``symbulate``;
- ``RV(P)`` -- the process itself, a discrete value at each continuous time;
- ``RV(P, interarrival_times)`` -- the times between jumps;
- ``RV(P, arrival_times)`` -- the times of the jumps;
- ``RV(P, states)`` -- the values visited, ignoring how long each lasted.

The convenience classes (``MM1``, ``GG1``, ...) are checked to draw the same
kind of path, and to reach the same three views through ``.apply()``.

``NonHomogeneousPoissonProcess`` and ``CoxProcess`` support only three of the
four (see ``TIME_CHANGED`` below): they count events on the "expected count"
scale and never convert back to clock time, so neither has clock-time
``interarrival_times``, and ``CoxProcess`` inherits the gap by building on the
non-homogeneous one. Closing it needs a numerical inverse of the cumulative
rate, a decision for that module's author -- so they are checked here for what
they do offer, and this file is where they move once it lands.
"""

import unittest

import numpy as np

from symbulate import *
from symbulate import distributions, markov_chains

# name -> (build a probability space, build the convenience process or None)
PROCESSES = {
    "ContinuousTimeMarkovChain": (
        lambda: ContinuousTimeMarkovChainProbabilitySpace(
            [[-1, 1], [2, -2]], [1.0, 0.0]
        ),
        lambda: ContinuousTimeMarkovChain([[-1, 1], [2, -2]], [1.0, 0.0]),
    ),
    "PoissonProcess": (
        lambda: PoissonProcessProbabilitySpace(rate=1),
        lambda: PoissonProcess(rate=1),
    ),
    "RenewalProcess": (
        lambda: RenewalProcessProbabilitySpace(Gamma(shape=2, rate=1)),
        lambda: RenewalProcess(Gamma(shape=2, rate=1)),
    ),
    "BirthDeathProcess": (
        lambda: BirthDeathProcessProbabilitySpace(1, 1.5, 21),
        lambda: BirthDeathProcess(birth_rates=1, death_rates=1.5, num_states=21),
    ),
    "MM1": (
        lambda: MM1ProbabilitySpace(1, 1.5, 21),
        lambda: MM1(arrival_rate=1, service_rate=1.5, num_states=21),
    ),
    "MMs": (
        lambda: MMsProbabilitySpace(3, 2, 2, 21),
        lambda: MMs(arrival_rate=3, service_rate=2, servers=2, num_states=21),
    ),
    "MMsK": (
        lambda: MMsKProbabilitySpace(4, 1, 2, 5),
        lambda: MMsK(arrival_rate=4, service_rate=1, servers=2, capacity=5),
    ),
    "MMss": (
        lambda: MMssProbabilitySpace(3, 1, 3),
        lambda: MMss(arrival_rate=3, service_rate=1, servers=3),
    ),
    "MMsKN": (
        lambda: MMsKNProbabilitySpace(0.5, 1, 2, 6, 6),
        lambda: MMsKN(
            arrival_rate=0.5, service_rate=1, servers=2, capacity=6, population=6
        ),
    ),
    "MMInfinity": (
        lambda: MMInfinityProbabilitySpace(3, 1, 21),
        lambda: MMInfinity(arrival_rate=3, service_rate=1, num_states=21),
    ),
    "YuleProcess": (
        lambda: YuleProcessProbabilitySpace(birth_rate=1),
        lambda: YuleProcess(birth_rate=1),
    ),
    "GG1": (
        lambda: GG1ProbabilitySpace(Exponential(rate=1), Exponential(rate=2)),
        lambda: GG1(Exponential(rate=1), Exponential(rate=2)),
    ),
    "MG1": (
        lambda: MG1ProbabilitySpace(1, Exponential(rate=2)),
        lambda: MG1(arrival_rate=1, service_dist=Exponential(rate=2)),
    ),
    "GM1": (
        lambda: GM1ProbabilitySpace(Exponential(rate=1), 2),
        lambda: GM1(interarrival_dist=Exponential(rate=1), service_rate=2),
    ),
    "GGs": (
        lambda: GGsProbabilitySpace(Exponential(rate=1), Exponential(rate=1), 2),
        lambda: GGs(Exponential(rate=1), Exponential(rate=1), servers=2),
    ),
    # A running total of random jumps rather than a count, so its state is a
    # real number instead of an integer -- but it holds that value for a
    # stretch of continuous time and jumps, so the same interface applies.
    "CompoundPoissonProcess": (
        lambda: CompoundPoissonProcessProbabilitySpace(rate=1, jump_dist=Normal(0, 1)),
        lambda: CompoundPoissonProcess(rate=1, jump_dist=Normal(0, 1)),
    ),
    "SIR": (
        lambda: SIRProbabilitySpace(population=50, infection_rate=2, recovery_rate=1),
        lambda: SIR(population=50, infection_rate=2, recovery_rate=1),
    ),
    "SEIR": (
        lambda: SEIRProbabilitySpace(
            population=50, infection_rate=2, incubation_rate=1, recovery_rate=1
        ),
        lambda: SEIR(
            population=50, infection_rate=2, incubation_rate=1, recovery_rate=1
        ),
    ),
}


# The two that count events on the "expected count" scale, so they have a
# space, RV(P), and states, but no clock-time interarrival or arrival times.
# Move an entry up into PROCESSES once it gains them.
TIME_CHANGED = {
    "NonHomogeneousPoissonProcess": (
        lambda: NonHomogeneousPoissonProcessProbabilitySpace(rate=lambda t: 1 + t),
        lambda: NonHomogeneousPoissonProcess(rate=lambda t: 1 + t),
    ),
    "CoxProcess": (
        lambda: CoxProcessProbabilitySpace(intensity=Gamma(shape=2, rate=1)),
        lambda: CoxProcess(intensity=Gamma(shape=2, rate=1)),
    ),
}


def seed(value=42):
    """Reseed every generator a process here might draw through."""
    distributions.rng = np.random.default_rng(value)
    markov_chains.rng = np.random.default_rng(value)


class TestProbabilitySpaceExists(unittest.TestCase):

    def test_every_process_has_an_exported_probability_space(self):
        import symbulate

        for name in PROCESSES:
            with self.subTest(process=name):
                self.assertTrue(
                    hasattr(symbulate, name + "ProbabilitySpace"),
                    f"{name}ProbabilitySpace is not exported from symbulate",
                )

    def test_every_space_draws_a_path(self):
        for name, (make_space, _) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                self.assertIsNotNone(make_space().draw())


class TestRVOfTheSpace(unittest.TestCase):
    """RV(P) and the three event-by-event views, for every process."""

    def test_rv_of_the_space_is_the_process(self):
        for name, (make_space, _) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                X = RV(make_space())
                self.assertIsInstance(X, RV)
                # A value at a continuous time, whatever kind of value it is.
                self.assertIsNotNone(X.draw()(1.0))

    def test_interarrival_times(self):
        for name, (make_space, _) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                gap = RV(make_space(), interarrival_times)[0].draw()
                self.assertGreaterEqual(gap, 0)

    def test_arrival_times(self):
        for name, (make_space, _) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                first_jump = RV(make_space(), arrival_times)[0].draw()
                self.assertGreaterEqual(first_jump, 0)

    def test_states(self):
        for name, (make_space, _) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                self.assertIsNotNone(RV(make_space(), states)[0].draw())

    def test_arrival_times_are_the_running_total_of_the_gaps(self):
        for name, (make_space, _) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                P = make_space()
                # One drawn path, read both ways.
                path = P.draw()
                gaps = interarrival_times(path)
                jumps = arrival_times(path)
                self.assertAlmostEqual(jumps[0], gaps[0])
                self.assertAlmostEqual(jumps[1], gaps[0] + gaps[1])

    def test_views_can_be_simulated(self):
        """The three views are ordinary random variables."""
        for name, (make_space, _) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                P = make_space()
                self.assertEqual(len(RV(P, interarrival_times)[0].sim(3)), 3)


class TestConvenienceClass(unittest.TestCase):
    """The named process classes match their spaces."""

    def test_class_draws_the_same_kind_of_path(self):
        for name, (make_space, make_process) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                from_space = make_space().draw()
                seed()
                from_class = make_process().draw()
                self.assertIs(type(from_space), type(from_class))

    def test_class_reaches_the_views_through_apply(self):
        for name, (_, make_process) in PROCESSES.items():
            with self.subTest(process=name):
                seed()
                X = make_process()
                self.assertGreaterEqual(X.apply(interarrival_times)[0].draw(), 0)
                self.assertGreaterEqual(X.apply(arrival_times)[0].draw(), 0)
                self.assertIsNotNone(X.apply(states)[0].draw())

    def test_class_uses_its_own_probability_space(self):
        for name, (_, make_process) in PROCESSES.items():
            with self.subTest(process=name):
                X = make_process()
                self.assertEqual(type(X.prob_space).__name__, name + "ProbabilitySpace")


class TestTimeChangedProcesses(unittest.TestCase):
    """The two that count on the expected-count scale, for what they do offer."""

    def test_probability_space_is_exported(self):
        import symbulate

        for name in TIME_CHANGED:
            with self.subTest(process=name):
                self.assertTrue(hasattr(symbulate, name + "ProbabilitySpace"))

    def test_rv_of_the_space_is_the_process(self):
        for name, (make_space, _) in TIME_CHANGED.items():
            with self.subTest(process=name):
                seed()
                self.assertIsNotNone(RV(make_space()).draw()(1.0))

    def test_states(self):
        for name, (make_space, _) in TIME_CHANGED.items():
            with self.subTest(process=name):
                seed()
                self.assertIsNotNone(RV(make_space(), states)[0].draw())

    def test_clock_time_jump_views_are_the_documented_gap(self):
        """Fails once they gain clock-time times -- move them up to PROCESSES."""
        for name, (make_space, _) in TIME_CHANGED.items():
            with self.subTest(process=name):
                seed()
                path = make_space().draw()
                with self.assertRaises(AttributeError):
                    interarrival_times(path)
                with self.assertRaises(AttributeError):
                    arrival_times(path)


class TestEpidemicStatesAreVectors(unittest.TestCase):
    """SIR and SEIR are the one family whose state is a vector of counts."""

    def test_sir_state_is_the_compartment_counts(self):
        seed()
        path = SIR(population=50, infection_rate=2, recovery_rate=1).draw()
        self.assertEqual(len(states(path)[0]), 3)
        self.assertEqual(sum(states(path)[0]), 50)

    def test_seir_state_has_four_compartments(self):
        seed()
        path = SEIR(
            population=50, infection_rate=2, incubation_rate=1, recovery_rate=1
        ).draw()
        self.assertEqual(len(states(path)[0]), 4)

    def test_last_state_lasts_forever(self):
        """The outbreak ends, so the final state is never left."""
        seed()
        path = SIR(population=50, infection_rate=2, recovery_rate=1).draw()
        self.assertEqual(interarrival_times(path)[-1], float("inf"))

    def test_holding_times_line_up_with_the_event_times(self):
        seed()
        path = SIR(population=50, infection_rate=2, recovery_rate=1).draw()
        # An outbreak that fizzles out early has only a few events, so check
        # however many this one had (bar the last, which lasts forever).
        for k in range(len(path.event_times) - 1):
            self.assertAlmostEqual(
                interarrival_times(path)[k],
                path.event_times[k + 1] - path.event_times[k],
            )


if __name__ == "__main__":
    unittest.main()
