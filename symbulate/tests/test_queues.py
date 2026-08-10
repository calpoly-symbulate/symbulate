"""Tests for symbulate.queues.

The state of every queue here is the **number of customers in the system** as
a function of continuous time, matching the M/M/... queues in markov_chains.py.
The waiting times that the count is built from (Lindley's recursion for one
server, the servers' next-free times for several) stay available on a drawn
path, and are tested through it.

Covers the sample-path result object (the count, the event-level
states/holding-times view, the customer-level times, caching), the probability
spaces, the GG1 queue with its MG1 / GM1 special cases, the multi-server GGs
queue, the validation of both distributions, and these exact queueing-theory
results the simulation is checked against:

- M/M/1 (both distributions exponential): the number in the system is
  Geometric with mean rho / (1 - rho), and P(empty) = 1 - rho;
- Little's law, L = lambda * W, tying the count to the sojourn times;
- M/G/1: the Pollaczek-Khinchine formula for the wait,
  lambda * E[S^2] / (2 * (1 - rho)), checked for two service distributions
  with the same mean but very different variance;
- M/M/s (the multi-server GGs queue with both distributions exponential): the
  Erlang C formula, for P(wait > 0), the mean wait, and the mean number
  waiting;
- a queue with rho >= 1 grows without bound.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_renewal_process.py.
"""

import unittest
from math import factorial

import numpy as np

from symbulate import *
from symbulate import distributions
from symbulate.queues import (
    GG1,
    GG1ProbabilitySpace,
    GG1Result,
    GGs,
    GGsProbabilitySpace,
    GGsResult,
    MG1,
    GM1,
)
from symbulate.renewal_process import RenewalProcessResult
from symbulate.result import ContinuousTimeFunction, DiscreteValued, InfiniteVector


def erlang_c(servers, arrival_rate, service_rate):
    """P(an M/M/s arrival has to wait), the Erlang C formula."""
    offered = arrival_rate / service_rate
    rho = offered / servers
    tail = offered**servers / (factorial(servers) * (1 - rho))
    idle = sum(offered**k / factorial(k) for k in range(servers))
    return tail / (idle + tail)


# The count and the waiting times both settle into their steady-state
# distributions geometrically fast, so time 50 and customer 50 are already
# (well within simulation error of) draws from them. Each simulated path costs
# about 2 * 50 distribution draws, which is what keeps Nsim smaller here than
# in the other process test files.
Nsim = 500
TIME = 50.0
CUSTOMER = 50


def counts_at(queue, t=TIME, nsim=Nsim):
    """Simulate the number in the system at time ``t``, ``nsim`` times."""
    return [queue.draw()(t) for _ in range(nsim)]


def waits_at(queue, customer=CUSTOMER, nsim=Nsim):
    """Simulate one customer's waiting time ``nsim`` times."""
    return [queue.draw().waiting_times[customer] for _ in range(nsim)]


class TestGG1ResultCount(unittest.TestCase):
    """The path itself: the number of customers in the system."""

    def test_is_continuous_time_function(self):
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_is_discrete_valued(self):
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertIsInstance(path, DiscreteValued)

    def test_starts_empty(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertEqual(path(0), 0)

    def test_count_by_hand(self):
        # A customer every 1.0, each served in 1.5, one server: arrivals at
        # 1, 2, 3, ... and departures at 2.5, 4.0, 5.5, 7.0, ...
        path = GG1Result([1.0] * 8, [1.5] * 8)
        self.assertEqual(
            [path(t) for t in [0.5, 1.0, 2.0, 2.5, 3.0, 6.0]], [0, 1, 2, 1, 2, 3]
        )

    def test_count_stays_at_zero_while_the_server_keeps_up(self):
        # Every service (1.0) is shorter than every gap (2.0), so each customer
        # is gone before the next arrives and the system is empty in between.
        path = GG1Result([2.0] * 6, [1.0] * 6)
        self.assertEqual([path(t) for t in [1.9, 2.5, 3.5, 4.5]], [0, 1, 0, 1])

    def test_counts_are_nonnegative_integers(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=1.2)).draw()
        for t in [0.5, 5.0, 20.0, 50.0]:
            value = path(t)
            self.assertIsInstance(value, int)
            self.assertGreaterEqual(value, 0)

    def test_getitem_matches_call(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        for t in [0.5, 5.0, 20.0]:
            self.assertEqual(path[t], path(t))

    def test_path_is_cached_and_stable(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        times = [1.0, 7.5, 20.0, 41.0]
        self.assertEqual([path(t) for t in times], [path(t) for t in times])

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        early = [path(t) for t in [1.0, 3.0, 5.0]]
        path(200.0)
        self.assertEqual([path(t) for t in [1.0, 3.0, 5.0]], early)

    def test_number_waiting_excludes_the_one_being_served(self):
        path = GG1Result([1.0] * 8, [1.5] * 8)
        waiting = path.get_number_waiting()
        self.assertEqual(path(3.2), 2)
        self.assertEqual(waiting(3.2), 1)
        # Nobody is waiting when the system is empty.
        self.assertEqual(waiting(0.5), 0)

    def test_count_agrees_with_the_customer_times(self):
        # The count is exactly: arrived by t, minus departed by t.
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=1.2)).draw()
        for t in [2.0, 10.0, 30.0]:
            arrived = sum(1 for n in range(200) if path.customer_arrival_times[n] <= t)
            departed = sum(
                1 for n in range(200) if path.customer_departure_times[n] <= t
            )
            self.assertEqual(path(t), arrived - departed)


class TestGG1ResultEvents(unittest.TestCase):
    """The event-level view, shared with the continuous-time Markov chains."""

    def test_states_are_the_counts_in_order(self):
        # Events: arrival(1), arrival(2), departure(1), arrival(2), ...
        path = GG1Result([1.0] * 8, [1.5] * 8)
        self.assertEqual([path.states[n] for n in range(6)], [0, 1, 2, 1, 2, 3])

    def test_first_state_is_empty(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertEqual(path.states[0], 0)

    def test_holding_times_are_the_gaps_between_events(self):
        path = GG1Result([1.0] * 8, [1.5] * 8)
        # First event at 1.0, then 2.0, then the first departure at 2.5.
        self.assertEqual(
            [path.interarrival_times[n] for n in range(3)], [1.0, 1.0, 0.5]
        )

    def test_event_times_are_the_cumulative_holding_times(self):
        path = GG1Result([1.0] * 8, [1.5] * 8)
        self.assertEqual(
            [path.get_arrival_times()[n] for n in range(3)], [1.0, 2.0, 2.5]
        )

    def test_event_times_are_nondecreasing(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=1.1)).draw()
        events = [path.get_arrival_times()[n] for n in range(40)]
        for earlier, later in zip(events, events[1:]):
            self.assertLessEqual(earlier, later)

    def test_states_and_holding_times_describe_the_same_path(self):
        # Reading the path at a time just after the k-th event must give the
        # state the event-level view says it is in.
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=1.2)).draw()
        events = path.get_arrival_times()
        for k in range(1, 15):
            midpoint = (events[k - 1] + events[k]) / 2
            self.assertEqual(path(midpoint), path.states[k])

    def test_free_functions_work_on_a_queue_path(self):
        """states/interarrival_times/arrival_times, as for an MM1 path."""
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertEqual(states(path)[0], 0)
        self.assertGreaterEqual(interarrival_times(path)[0], 0)
        self.assertGreaterEqual(arrival_times(path)[0], 0)

    def test_getters_return_the_event_sequences(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertIs(path.get_states(), path.states)
        self.assertIs(path.get_interarrival_times(), path.interarrival_times)


class TestGG1ResultCustomers(unittest.TestCase):
    """The customer-level view the count is built from."""

    def test_lindley_recursion_by_hand(self):
        # Arrivals 1 apart, every service taking 1.5: the server falls half a
        # unit further behind on each customer.
        path = GG1Result([1.0] * 5, [1.5] * 5)
        self.assertEqual(
            [path.waiting_times[n] for n in range(4)], [0.0, 0.5, 1.0, 1.5]
        )

    def test_first_customer_never_waits(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertEqual(path.waiting_times[0], 0)

    def test_idle_server_resets_the_wait_to_zero(self):
        # The 5.0 gap before customer 2 is longer than the 3.0 of work left
        # over, so the server goes idle and customer 2 walks straight up.
        path = GG1Result([1.0, 1.0, 5.0, 1.0], [3.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            [path.waiting_times[n] for n in range(4)], [0.0, 2.0, 0.0, 0.0]
        )

    def test_waits_are_nonnegative(self):
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=1.2)).draw()
        for n in range(30):
            self.assertGreaterEqual(path.waiting_times[n], 0)

    def test_arrival_times_accumulate_the_gaps(self):
        path = GG1Result([1.0, 2.0, 0.5], [0.1, 0.1, 0.1])
        self.assertEqual(
            [path.customer_arrival_times[n] for n in range(3)], [1.0, 3.0, 3.5]
        )

    def test_sojourn_time_is_wait_plus_service(self):
        path = GG1Result([1.0] * 4, [1.5] * 4)
        for n in range(3):
            self.assertAlmostEqual(
                path.sojourn_times[n], path.waiting_times[n] + path.service_times[n]
            )

    def test_departure_time_is_arrival_plus_sojourn(self):
        path = GG1Result([1.0] * 4, [1.5] * 4)
        for n in range(3):
            self.assertAlmostEqual(
                path.customer_departure_times[n],
                path.customer_arrival_times[n] + path.sojourn_times[n],
            )

    def test_departures_are_in_order_for_a_single_server(self):
        # One server serving first come, first served cannot let a later
        # customer leave before an earlier one.
        seed(42)
        path = GG1(Exponential(rate=1), Exponential(rate=1.1)).draw()
        departures = [path.customer_departure_times[n] for n in range(30)]
        for earlier, later in zip(departures, departures[1:]):
            self.assertLessEqual(earlier, later)

    def test_arrival_process_counts_the_arrivals(self):
        path = GG1Result([1.0] * 5, [1.5] * 5)
        arrivals = path.get_arrival_process()
        self.assertIsInstance(arrivals, RenewalProcessResult)
        # Customers arrive at times 1, 2, 3, ...
        self.assertEqual(arrivals(0.5), 0)
        self.assertEqual(arrivals(2.5), 2)

    def test_arrival_process_counts_more_than_the_path_holds(self):
        """Arrivals never leave; the path's own count does."""
        path = GG1Result([2.0] * 6, [1.0] * 6)
        self.assertEqual(path.get_arrival_process()(5.0), 2)
        self.assertEqual(path(5.0), 0)  # both have already been served


class TestGG1ProbabilitySpace(unittest.TestCase):

    def test_distributions_stored(self):
        interarrival, service = Exponential(rate=1), Gamma(shape=2, rate=4)
        space = GG1ProbabilitySpace(interarrival, service)
        self.assertIs(space.interarrival_dist, interarrival)
        self.assertIs(space.service_dist, service)

    def test_draw_returns_result(self):
        seed(42)
        space = GG1ProbabilitySpace(Exponential(rate=1), Exponential(rate=2))
        self.assertIsInstance(space.draw(), GG1Result)

    def test_successive_draws_are_independent(self):
        # The two i.i.d. sequences are built once and reused, so each draw
        # must still give a fresh path rather than repeating the first one.
        seed(42)
        space = GG1ProbabilitySpace(Exponential(rate=1), Exponential(rate=1.2))
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first(t) for t in range(1, 30)], [second(t) for t in range(1, 30)]
        )


class TestGG1(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(GG1(Exponential(rate=1), Exponential(rate=2)), RV)

    def test_distributions_stored(self):
        interarrival, service = Exponential(rate=1), Gamma(shape=2, rate=4)
        queue = GG1(interarrival, service)
        self.assertIs(queue.interarrival_dist, interarrival)
        self.assertIs(queue.service_dist, service)

    def test_accepts_keyword_arguments(self):
        queue = GG1(
            interarrival_dist=Exponential(rate=1), service_dist=Exponential(rate=2)
        )
        self.assertEqual(queue.utilization, 0.5)

    def test_utilization_is_mean_service_over_mean_interarrival(self):
        queue = GG1(Exponential(rate=1), Uniform(a=0, b=1))
        self.assertAlmostEqual(queue.utilization, 0.5)

    def test_utilization_unavailable_for_a_degenerate_point_mass(self):
        # Accepted limitation, shared with RenewalProcess's validation: a
        # point mass written as Uniform(a=b) gives scipy a degenerate
        # parameterization, whose mean comes back nan.
        queue = GG1(Exponential(rate=1), Uniform(a=0.5, b=0.5))
        self.assertIsNone(queue.utilization)

    def test_getitem_at_a_time_returns_rv(self):
        self.assertIsInstance(GG1(Exponential(rate=1), Exponential(rate=2))[5.0], RV)

    def test_getitem_matches_drawn_path(self):
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        seed(7)
        indexed = list(queue[5.0].sim(10))
        seed(7)
        drawn = [queue.draw()(5.0) for _ in range(10)]
        self.assertEqual(indexed, drawn)

    def test_reproducible_under_same_seed(self):
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        seed(123)
        first = [queue.draw()(10.0) for _ in range(20)]
        seed(123)
        second = [queue.draw()(10.0) for _ in range(20)]
        self.assertEqual(first, second)

    def test_sim_gives_independent_paths(self):
        seed(42)
        queue = GG1(Exponential(rate=1), Exponential(rate=1.2))
        paths = list(queue.sim(2))
        self.assertNotEqual(
            [paths[0](t) for t in range(1, 30)], [paths[1](t) for t in range(1, 30)]
        )

    def test_waiting_time_via_apply(self):
        seed(42)
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        wait = queue.apply(lambda path: path.waiting_times[10])
        self.assertIsInstance(wait, RV)
        self.assertGreaterEqual(wait.draw(), 0)

    def test_sojourn_time_via_apply(self):
        seed(42)
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        sojourn = queue.apply(lambda path: path.sojourn_times[10])
        self.assertGreater(sojourn.draw(), 0)


class TestMG1(unittest.TestCase):

    def test_arrival_rate_becomes_exponential_interarrivals(self):
        queue = MG1(arrival_rate=3, service_dist=Uniform(a=0, b=0.2))
        self.assertEqual(queue.arrival_rate, 3)
        self.assertIsInstance(queue.interarrival_dist, Exponential)
        self.assertAlmostEqual(queue.interarrival_dist.mean(), 1 / 3)

    def test_is_a_gg1_queue(self):
        self.assertIsInstance(MG1(arrival_rate=1, service_dist=Uniform(a=0, b=1)), GG1)

    def test_utilization_is_rate_times_mean_service(self):
        queue = MG1(arrival_rate=2, service_dist=Uniform(a=0, b=0.5))
        self.assertAlmostEqual(queue.utilization, 0.5)

    def test_matches_gg1_with_exponential_interarrivals(self):
        service = Gamma(shape=2, rate=8)
        seed(5)
        from_mg1 = MG1(arrival_rate=1, service_dist=service).draw()(10.0)
        seed(5)
        from_gg1 = GG1(Exponential(rate=1), service).draw()(10.0)
        self.assertEqual(from_mg1, from_gg1)


class TestGM1(unittest.TestCase):

    def test_service_rate_becomes_exponential_services(self):
        queue = GM1(interarrival_dist=Gamma(shape=8, rate=8), service_rate=4)
        self.assertEqual(queue.service_rate, 4)
        self.assertIsInstance(queue.service_dist, Exponential)
        self.assertAlmostEqual(queue.service_dist.mean(), 0.25)

    def test_is_a_gg1_queue(self):
        self.assertIsInstance(
            GM1(interarrival_dist=Exponential(rate=1), service_rate=2), GG1
        )

    def test_utilization_uses_the_given_arrivals(self):
        queue = GM1(interarrival_dist=Uniform(a=0, b=2), service_rate=2)
        self.assertAlmostEqual(queue.utilization, 0.5)


class TestGGsResult(unittest.TestCase):

    def test_is_continuous_time_function(self):
        path = GGs(Exponential(rate=1), Exponential(rate=1), servers=2).draw()
        self.assertIsInstance(path, ContinuousTimeFunction)

    def test_servers_stored(self):
        self.assertEqual(GGsResult([1.0] * 3, [1.0] * 3, servers=4).servers, 4)

    def test_one_server_reproduces_lindley_exactly(self):
        # Same inputs, so GGsResult with one server and GG1Result must agree
        # value for value -- both the waits and the count they produce.
        interarrival = [1.0, 1.0, 5.0, 1.0, 2.0, 1.0]
        service = [3.0, 1.0, 1.0, 1.0, 4.0, 1.0]
        lindley = GG1Result(interarrival, service)
        one = GGsResult(interarrival, service, servers=1)
        self.assertEqual(
            [lindley.waiting_times[n] for n in range(6)],
            [one.waiting_times[n] for n in range(6)],
        )
        self.assertEqual(
            [lindley(t) for t in [0.5, 2.0, 4.0, 8.0]],
            [one(t) for t in [0.5, 2.0, 4.0, 8.0]],
        )

    def test_count_by_hand_with_two_servers(self):
        # A customer every 1.0, each served in 1.5, two servers: customer 0 is
        # served 1.0-2.5, customer 1 2.0-3.5, customer 2 3.0-4.5, ... so two
        # customers are in the system from 2.0 onwards and nobody ever waits.
        path = GGsResult([1.0] * 8, [1.5] * 8, servers=2)
        self.assertEqual([path(t) for t in [0.5, 1.0, 2.0, 3.0, 6.0]], [0, 1, 2, 2, 2])
        self.assertEqual([path.waiting_times[n] for n in range(5)], [0.0] * 5)

    def test_second_server_holds_a_shorter_line(self):
        interarrival, service = [1.0] * 8, [1.5] * 8
        one = GGsResult(interarrival, service, servers=1)
        two = GGsResult(interarrival, service, servers=2)
        self.assertEqual((one(6.0), two(6.0)), (3, 2))

    def test_waits_by_hand_when_both_servers_are_busy(self):
        # Two servers, a customer every 1.0. Services 4, 4, 1, 1: customers 0
        # and 1 take both servers until times 5 and 6, so customer 2 (arriving
        # at 3.0) waits until 5.0 and customer 3 (arriving at 4.0) until 6.0.
        path = GGsResult([1.0] * 4, [4.0, 4.0, 1.0, 1.0], servers=2)
        self.assertEqual(
            [path.waiting_times[n] for n in range(4)], [0.0, 0.0, 2.0, 2.0]
        )

    def test_customer_takes_whichever_server_frees_first(self):
        # Two servers: customer 0 is served until 1.5, customer 1 until 4.0.
        # Customer 2 arrives at 3.0 and takes the server that is already free.
        path = GGsResult([1.0, 1.0, 1.0], [0.5, 2.0, 1.0], servers=2)
        self.assertEqual([path.waiting_times[n] for n in range(3)], [0.0, 0.0, 0.0])

    def test_number_waiting_allows_for_every_server(self):
        # Two servers busy and a third customer in line.
        path = GGsResult([1.0] * 4, [4.0, 4.0, 1.0, 1.0], servers=2)
        self.assertEqual(path(3.5), 3)
        self.assertEqual(path.get_number_waiting()(3.5), 1)

    def test_counts_are_nonnegative(self):
        seed(42)
        path = GGs(Exponential(rate=2), Exponential(rate=1.2), servers=2).draw()
        for t in [1.0, 10.0, 40.0]:
            self.assertGreaterEqual(path(t), 0)

    def test_path_is_cached_and_stable(self):
        seed(42)
        path = GGs(Exponential(rate=2), Exponential(rate=1), servers=2).draw()
        times = [1.0, 8.0, 25.0]
        self.assertEqual([path(t) for t in times], [path(t) for t in times])

    def test_event_view_is_inherited(self):
        path = GGsResult([1.0] * 8, [1.5] * 8, servers=2)
        self.assertEqual([path.states[n] for n in range(3)], [0, 1, 2])
        self.assertEqual([path.get_arrival_times()[n] for n in range(2)], [1.0, 2.0])
        self.assertIsInstance(path.get_arrival_process(), RenewalProcessResult)

    def test_departures_can_be_out_of_order_with_two_servers(self):
        # Customer 0 takes 5.0 while customer 1 takes 0.5 beside them, so
        # customer 1 leaves first -- something a single server cannot do.
        path = GGsResult([1.0, 1.0, 1.0], [5.0, 0.5, 0.5], servers=2)
        self.assertGreater(
            path.customer_departure_times[0], path.customer_departure_times[1]
        )

    def test_count_is_right_even_when_departures_are_out_of_order(self):
        # Arrivals at 1, 2, 3; departures at 6.0, 2.5, 3.5. At t = 4 only
        # customer 0 is still there, and by t = 6 the system is empty. Reading
        # past the last of three given arrivals also has to work, which is why
        # this is a regression test: it used to raise IndexError.
        path = GGsResult([1.0, 1.0, 1.0], [5.0, 0.5, 0.5], servers=2)
        self.assertEqual([path(t) for t in [1.5, 2.0, 2.5, 4.0, 6.0]], [1, 2, 1, 1, 0])

    def test_hand_built_path_runs_out_of_customers_gracefully(self):
        # Three customers only: long after the last one leaves the system is
        # empty, the count never changes again, and nothing raises.
        path = GGsResult([1.0, 1.0, 1.0], [5.0, 0.5, 0.5], servers=2)
        self.assertEqual(path(1000.0), 0)
        self.assertEqual(path.states[20], 0)
        self.assertEqual(path.interarrival_times[20], float("inf"))


class TestGGsProbabilitySpace(unittest.TestCase):

    def test_parameters_stored(self):
        interarrival, service = Exponential(rate=2), Gamma(shape=2, rate=2)
        space = GGsProbabilitySpace(interarrival, service, 3)
        self.assertIs(space.interarrival_dist, interarrival)
        self.assertIs(space.service_dist, service)
        self.assertEqual(space.servers, 3)

    def test_draw_returns_result_with_the_servers(self):
        seed(42)
        path = GGsProbabilitySpace(Exponential(rate=2), Exponential(rate=1), 3).draw()
        self.assertIsInstance(path, GGsResult)
        self.assertEqual(path.servers, 3)

    def test_successive_draws_are_independent(self):
        seed(42)
        space = GGsProbabilitySpace(Exponential(rate=2), Exponential(rate=1), 2)
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first(t) for t in range(1, 30)], [second(t) for t in range(1, 30)]
        )


class TestGGs(unittest.TestCase):

    def test_is_rv(self):
        self.assertIsInstance(
            GGs(Exponential(rate=1), Exponential(rate=1), servers=2), RV
        )

    def test_parameters_stored(self):
        interarrival, service = Exponential(rate=2), Gamma(shape=2, rate=2)
        queue = GGs(interarrival, service, servers=3)
        self.assertIs(queue.interarrival_dist, interarrival)
        self.assertIs(queue.service_dist, service)
        self.assertEqual(queue.servers, 3)

    def test_accepts_keyword_arguments(self):
        queue = GGs(
            interarrival_dist=Exponential(rate=1),
            service_dist=Exponential(rate=1),
            servers=2,
        )
        self.assertEqual(queue.servers, 2)

    def test_utilization_is_shared_out_over_the_servers(self):
        # Mean service 1, mean gap 1: one server is saturated, two are half busy.
        for servers, expected in [(1, 1.0), (2, 0.5), (4, 0.25)]:
            with self.subTest(servers=servers):
                queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=servers)
                self.assertAlmostEqual(queue.utilization, expected)

    def test_second_server_can_make_an_unstable_queue_stable(self):
        overloaded = GGs(Exponential(rate=1), Exponential(rate=0.8), servers=1)
        rescued = GGs(Exponential(rate=1), Exponential(rate=0.8), servers=2)
        self.assertGreater(overloaded.utilization, 1)
        self.assertLess(rescued.utilization, 1)

    def test_getitem_at_a_time_returns_rv(self):
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        self.assertIsInstance(queue[5.0], RV)

    def test_reproducible_under_same_seed(self):
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        seed(123)
        first = [queue.draw()(10.0) for _ in range(20)]
        seed(123)
        self.assertEqual(first, [queue.draw()(10.0) for _ in range(20)])

    def test_sim_gives_independent_paths(self):
        seed(42)
        queue = GGs(Exponential(rate=2), Exponential(rate=1.1), servers=2)
        paths = list(queue.sim(2))
        self.assertNotEqual(
            [paths[0](t) for t in range(1, 30)], [paths[1](t) for t in range(1, 30)]
        )


class TestQueueTheory(unittest.TestCase):
    """The single-server simulation against exact queueing-theory results."""

    def test_mm1_mean_number_in_system(self):
        # Both distributions exponential is the M/M/1 queue, whose long-run
        # mean number in the system is rho / (1 - rho).
        seed(42)
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        rho = queue.utilization
        expected = rho / (1 - rho)
        self.assertAlmostEqual(np.mean(counts_at(queue)), expected, delta=0.15)

    def test_mm1_probability_the_system_is_empty(self):
        seed(42)
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        empty = np.mean([count == 0 for count in counts_at(queue)])
        self.assertAlmostEqual(empty, 1 - queue.utilization, delta=0.06)

    def test_mm1_number_in_system_is_geometric(self):
        # P(N = k) = (1 - rho) * rho ** k.
        seed(42)
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        rho = queue.utilization
        counts = counts_at(queue)
        for k in range(4):
            with self.subTest(k=k):
                expected = (1 - rho) * rho**k
                observed = np.mean([count == k for count in counts])
                self.assertAlmostEqual(observed, expected, delta=0.05)

    def test_littles_law(self):
        # L = lambda * W: the mean number in the system is the arrival rate
        # times the mean time each customer spends in it.
        seed(42)
        arrival_rate = 1
        queue = GG1(Exponential(rate=arrival_rate), Exponential(rate=2))
        number = np.mean(counts_at(queue))
        sojourn = np.mean([queue.draw().sojourn_times[CUSTOMER] for _ in range(Nsim)])
        self.assertAlmostEqual(number, arrival_rate * sojourn, delta=0.2)

    def test_mm1_mean_wait(self):
        # The wait behind that count: rho / (mu - lambda).
        seed(42)
        arrival_rate, service_rate = 1, 2
        expected = (arrival_rate / service_rate) / (service_rate - arrival_rate)
        queue = GG1(Exponential(rate=arrival_rate), Exponential(rate=service_rate))
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.12)

    def test_pollaczek_khinchine_with_erratic_service(self):
        # M/G/1: mean wait = lambda * E[S^2] / (2 * (1 - rho)). With
        # Exponential(rate=2) service, E[S^2] = 0.5 and rho = 0.5, so 0.5.
        seed(42)
        queue = MG1(arrival_rate=1, service_dist=Exponential(rate=2))
        self.assertAlmostEqual(np.mean(waits_at(queue)), 0.5, delta=0.12)

    def test_pollaczek_khinchine_with_steady_service(self):
        # The same mean service time (0.5) but far less variable:
        # Gamma(shape=20, rate=40) has E[S^2] = 0.2625, so the wait is halved.
        seed(42)
        service = Gamma(shape=20, rate=40)
        second_moment = service.var() + service.mean() ** 2
        expected = 1 * second_moment / (2 * (1 - 0.5))
        queue = MG1(arrival_rate=1, service_dist=service)
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.08)

    def test_steadier_service_holds_a_shorter_line(self):
        # The same lesson read off the count instead of the wait.
        seed(42)
        steady = MG1(arrival_rate=1, service_dist=Gamma(shape=20, rate=40))
        erratic = MG1(arrival_rate=1, service_dist=Exponential(rate=2))
        self.assertAlmostEqual(steady.utilization, erratic.utilization)
        self.assertLess(np.mean(counts_at(steady)), np.mean(counts_at(erratic)))

    def test_steadier_arrivals_shorten_the_wait(self):
        # G/M/1 against M/M/1 at identical utilization: Gamma(shape=8, rate=8)
        # arrivals are far more regular than Exponential(rate=1) ones, and
        # regularity is worth a shorter line.
        seed(42)
        steady = GM1(interarrival_dist=Gamma(shape=8, rate=8), service_rate=2)
        erratic = GM1(interarrival_dist=Exponential(rate=1), service_rate=2)
        self.assertAlmostEqual(steady.utilization, erratic.utilization)
        self.assertLess(np.mean(waits_at(steady)), np.mean(waits_at(erratic)))

    def test_unstable_queue_grows_without_bound(self):
        # With rho > 1 the server cannot keep up, so the line keeps growing
        # instead of settling down.
        seed(42)
        queue = GG1(Exponential(rate=2), Exponential(rate=1))
        self.assertGreater(queue.utilization, 1)
        path = queue.draw()
        self.assertLess(path(20.0), path(200.0))

    def test_critically_loaded_queue_has_no_steady_state(self):
        # rho exactly 1 is also unstable, if more slowly.
        seed(42)
        queue = GG1(Exponential(rate=1), Exponential(rate=1))
        self.assertEqual(queue.utilization, 1)
        early = np.mean([queue.draw()(10.0) for _ in range(100)])
        late = np.mean([queue.draw()(400.0) for _ in range(100)])
        self.assertGreater(late, early)


class TestGGsTheory(unittest.TestCase):
    """The multi-server simulation against exact M/M/s results."""

    def test_erlang_c_probability_of_waiting_two_servers(self):
        seed(42)
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        waited = np.mean([wait > 0 for wait in waits_at(queue)])
        self.assertAlmostEqual(waited, erlang_c(2, 1, 1), delta=0.07)

    def test_erlang_c_mean_wait_two_servers(self):
        # M/M/2 with lambda = mu = 1: mean wait in line is
        # C / (s * mu - lambda) = (1/3) / 1 = 1/3.
        seed(42)
        expected = erlang_c(2, 1, 1) / (2 * 1 - 1)
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.12)

    def test_mms_mean_number_in_system(self):
        # M/M/2 with lambda = mu = 1: L = L_q + lambda / mu, where
        # L_q = C * rho / (1 - rho) = 1/3, so L = 4/3.
        seed(42)
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        number_waiting = erlang_c(2, 1, 1) * 0.5 / (1 - 0.5)
        expected = number_waiting + 1 / 1
        self.assertAlmostEqual(np.mean(counts_at(queue, t=60.0)), expected, delta=0.25)

    def test_mms_mean_number_waiting(self):
        seed(42)
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        expected = erlang_c(2, 1, 1) * 0.5 / (1 - 0.5)
        observed = np.mean(
            [queue.draw().get_number_waiting()(60.0) for _ in range(Nsim)]
        )
        self.assertAlmostEqual(observed, expected, delta=0.15)

    def test_erlang_c_mean_wait_three_servers(self):
        # M/M/3 with lambda = 2, mu = 1: rho = 2/3, mean wait 4/9.
        seed(42)
        expected = erlang_c(3, 2, 1) / (3 * 1 - 2)
        queue = GGs(Exponential(rate=2), Exponential(rate=1), servers=3)
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.12)

    def test_two_slow_servers_shorten_the_line_but_not_the_visit(self):
        # The classic comparison at equal utilization: two servers of rate 1
        # against one server of rate 2. Exact M/M/ values are 1/3 vs 1/2 for
        # the wait in line, and 4/3 vs 1 for the time in system.
        seed(42)
        fast = GG1(Exponential(rate=1), Exponential(rate=2))
        pair = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        self.assertAlmostEqual(fast.utilization, pair.utilization)

        self.assertLess(np.mean(waits_at(pair)), np.mean(waits_at(fast)))
        fast_visit = np.mean([fast.draw().sojourn_times[CUSTOMER] for _ in range(Nsim)])
        pair_visit = np.mean([pair.draw().sojourn_times[CUSTOMER] for _ in range(Nsim)])
        self.assertGreater(pair_visit, fast_visit)

    def test_more_servers_never_lengthen_the_line(self):
        seed(42)
        means = []
        for servers in [1, 2, 3]:
            queue = GGs(Exponential(rate=1), Exponential(rate=1.2), servers=servers)
            means.append(np.mean(counts_at(queue, nsim=200)))
        self.assertLess(means[1], means[0])
        self.assertLessEqual(means[2], means[1])

    def test_unstable_multi_server_queue_grows_without_bound(self):
        seed(42)
        queue = GGs(Exponential(rate=3), Exponential(rate=1), servers=2)
        self.assertGreater(queue.utilization, 1)
        path = queue.draw()
        self.assertLess(path(20.0), path(200.0))


class TestQueueValidation(unittest.TestCase):
    """Validation of both distributions, for the spaces and the queues."""

    def test_arrival_rate_in_place_of_a_distribution_raises_type_error(self):
        with self.assertRaises(TypeError):
            GG1(1, Exponential(rate=2))

    def test_interarrival_number_error_suggests_mg1(self):
        """The message points a student who passed a rate to MG1."""
        with self.assertRaises(TypeError) as context:
            GG1(1, Exponential(rate=2))
        self.assertIn("MG1", str(context.exception))

    def test_service_rate_in_place_of_a_distribution_raises_type_error(self):
        with self.assertRaises(TypeError):
            GG1(Exponential(rate=1), 2)

    def test_service_number_error_suggests_gm1(self):
        with self.assertRaises(TypeError) as context:
            GG1(Exponential(rate=1), 2)
        self.assertIn("GM1", str(context.exception))

    def test_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            GG1("Exponential", Exponential(rate=2))
        with self.assertRaises(TypeError):
            GG1(Exponential(rate=1), "Exponential")

    def test_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            GG1(None, Exponential(rate=2))
        with self.assertRaises(TypeError):
            GG1(Exponential(rate=1), None)

    def test_uninstantiated_distribution_class_raises_type_error(self):
        """Passing the class Exponential rather than Exponential(rate=1)."""
        with self.assertRaises(TypeError):
            GG1(Exponential, Exponential(rate=2))
        with self.assertRaises(TypeError):
            GG1(Exponential(rate=1), Exponential)

    def test_multivariate_distribution_raises_type_error(self):
        """A vector per draw is neither one waiting time nor one service."""
        with self.assertRaises(TypeError):
            GG1(BivariateNormal(mean1=1, mean2=1), Exponential(rate=2))
        with self.assertRaises(TypeError):
            GG1(Exponential(rate=1), BivariateNormal(mean1=1, mean2=1))

    def test_negative_interarrival_support_raises_value_error(self):
        with self.assertRaises(ValueError):
            GG1(Normal(mean=5, sd=1), Exponential(rate=2))

    def test_negative_service_support_raises_value_error(self):
        with self.assertRaises(ValueError):
            GG1(Exponential(rate=1), Normal(mean=0.5, sd=0.1))

    def test_negative_support_error_names_the_distribution(self):
        with self.assertRaises(ValueError) as context:
            GG1(Exponential(rate=1), Uniform(a=-1, b=3))
        message = str(context.exception)
        self.assertIn("Uniform", message)
        self.assertIn("service_dist", message)

    def test_always_zero_interarrivals_raise_value_error(self):
        """Every customer arriving at once is not a queue."""
        with self.assertRaises(ValueError):
            GG1(Poisson(0), Exponential(rate=2))

    def test_always_zero_service_is_accepted(self):
        """A server that finishes instantly is degenerate, not impossible."""
        seed(42)
        path = GG1(Exponential(rate=1), Poisson(0)).draw()
        # Every customer leaves the instant they arrive, so nobody waits and
        # the system is empty except at the arrival instants themselves.
        self.assertEqual([path.waiting_times[n] for n in range(5)], [0.0] * 5)
        self.assertEqual(path(10.5), 0)

    def test_probability_space_also_validates(self):
        """The guard lives in the space, so it fires there too."""
        with self.assertRaises(TypeError):
            GG1ProbabilitySpace(1, Exponential(rate=2))
        with self.assertRaises(ValueError):
            GG1ProbabilitySpace(Exponential(rate=1), Normal(mean=1, sd=1))

    def test_nonnegative_distribution_pairs_accepted(self):
        for interarrival, service in [
            (Exponential(rate=1), Exponential(rate=2)),
            (Gamma(shape=2, rate=2), Uniform(a=0, b=0.5)),
            (Uniform(a=0, b=2), LogNormal(-1.5, 0.75)),
            (LogNormal(0, 0.5), Weibull(shape=2, scale=0.5)),
            (Poisson(2), Beta(2, 3)),
        ]:
            with self.subTest(
                interarrival=type(interarrival).__name__,
                service=type(service).__name__,
            ):
                seed(42)
                self.assertGreaterEqual(GG1(interarrival, service).draw()(5.0), 0)

    def test_bad_arrival_rate_raises(self):
        with self.assertRaises(TypeError):
            MG1(arrival_rate="fast", service_dist=Exponential(rate=2))
        with self.assertRaises(ValueError):
            MG1(arrival_rate=0, service_dist=Exponential(rate=2))
        with self.assertRaises(ValueError):
            MG1(arrival_rate=-1, service_dist=Exponential(rate=2))

    def test_arrival_rate_error_names_the_parameter(self):
        with self.assertRaises(ValueError) as context:
            MG1(arrival_rate=-1, service_dist=Exponential(rate=2))
        self.assertIn("arrival_rate", str(context.exception))

    def test_bad_service_rate_raises(self):
        with self.assertRaises(TypeError):
            GM1(interarrival_dist=Exponential(rate=1), service_rate=None)
        with self.assertRaises(ValueError):
            GM1(interarrival_dist=Exponential(rate=1), service_rate=0)

    def test_mg1_still_validates_the_service_distribution(self):
        with self.assertRaises(TypeError):
            MG1(arrival_rate=1, service_dist=2)
        with self.assertRaises(ValueError):
            MG1(arrival_rate=1, service_dist=Normal(mean=1, sd=1))

    def test_gm1_still_validates_the_interarrival_distribution(self):
        with self.assertRaises(TypeError):
            GM1(interarrival_dist=2, service_rate=1)
        with self.assertRaises(ValueError):
            GM1(interarrival_dist=Normal(mean=5, sd=1), service_rate=1)


class TestGGsValidation(unittest.TestCase):

    def test_non_integer_servers_raises_type_error(self):
        with self.assertRaises(TypeError):
            GGs(Exponential(rate=1), Exponential(rate=1), servers=2.5)

    def test_string_servers_raises_type_error(self):
        with self.assertRaises(TypeError):
            GGs(Exponential(rate=1), Exponential(rate=1), servers="two")

    def test_servers_error_names_the_parameter(self):
        with self.assertRaises(TypeError) as context:
            GGs(Exponential(rate=1), Exponential(rate=1), servers=2.5)
        self.assertIn("servers", str(context.exception))

    def test_zero_servers_raises_value_error(self):
        with self.assertRaises(ValueError):
            GGs(Exponential(rate=1), Exponential(rate=1), servers=0)

    def test_negative_servers_raises_value_error(self):
        with self.assertRaises(ValueError):
            GGs(Exponential(rate=1), Exponential(rate=1), servers=-2)

    def test_distributions_are_still_validated(self):
        with self.assertRaises(TypeError):
            GGs(1, Exponential(rate=1), servers=2)
        with self.assertRaises(TypeError):
            GGs(Exponential(rate=1), 2, servers=2)
        with self.assertRaises(ValueError):
            GGs(Normal(mean=5, sd=1), Exponential(rate=1), servers=2)
        with self.assertRaises(ValueError):
            GGs(Exponential(rate=1), Normal(mean=1, sd=1), servers=2)

    def test_probability_space_also_validates(self):
        with self.assertRaises(ValueError):
            GGsProbabilitySpace(Exponential(rate=1), Exponential(rate=1), 0)
        with self.assertRaises(TypeError):
            GGsProbabilitySpace(1, Exponential(rate=1), 2)

    def test_base_result_class_cannot_be_used_on_its_own(self):
        """The shared machinery has no waiting-time recursion of its own."""
        from symbulate.queues import _QueueResult

        with self.assertRaises(NotImplementedError):
            _QueueResult([1.0] * 3, [1.0] * 3)(1.0)


if __name__ == "__main__":
    unittest.main()
