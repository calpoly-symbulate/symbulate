"""Tests for symbulate.queues.

Covers the sample-path result object (Lindley's recursion, the derived
arrival/sojourn/departure sequences, caching), the probability space, the
GG1 queue and its MG1 / GM1 special cases, the validation of both
distributions, and four exact queueing-theory results the simulation is
checked against:

- M/M/1 (both distributions exponential): mean wait rho / (mu - lambda), and
  P(wait = 0) = 1 - rho;
- M/G/1: the Pollaczek-Khinchine formula
  lambda * E[S^2] / (2 * (1 - rho)), checked for two service distributions
  with the same mean but very different variance;
- M/M/s (the multi-server GGs queue with both distributions exponential):
  the Erlang C formula, for both the mean wait and P(wait > 0);
- a queue with rho >= 1 has waiting times that grow without bound.

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
from symbulate.result import InfiniteVector


def erlang_c(servers, arrival_rate, service_rate):
    """P(an M/M/s arrival has to wait), the Erlang C formula."""
    offered = arrival_rate / service_rate
    rho = offered / servers
    tail = offered**servers / (factorial(servers) * (1 - rho))
    idle = sum(offered**k / factorial(k) for k in range(servers))
    return tail / (idle + tail)


# Waiting times converge to their steady-state distribution geometrically
# fast, so customer 50 is already (well within simulation error of) a draw
# from it. Each simulated path costs 2 * 50 distribution draws, which is what
# keeps Nsim smaller here than in the other process test files.
Nsim = 500
CUSTOMER = 50


def seed(value=42):
    """Reseed the generator that distribution draws route through."""
    distributions.rng = np.random.default_rng(value)


def waits_at(queue, customer=CUSTOMER, nsim=Nsim):
    """Simulate one customer's waiting time ``nsim`` times."""
    return [queue.draw()[customer] for _ in range(nsim)]


class TestGG1Result(unittest.TestCase):

    def test_is_infinite_vector(self):
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertIsInstance(path, InfiniteVector)

    def test_first_customer_never_waits(self):
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        self.assertEqual(path[0], 0)

    def test_lindley_recursion_by_hand(self):
        # Arrivals 1 apart, every service taking 1.5: the server falls half a
        # unit further behind on each customer.
        path = GG1Result([1.0] * 5, [1.5] * 5)
        self.assertEqual([path[n] for n in range(4)], [0.0, 0.5, 1.0, 1.5])

    def test_idle_server_resets_the_wait_to_zero(self):
        # The 5.0 gap before customer 2 is longer than the 3.0 of work left
        # over, so the server goes idle and customer 2 walks straight up.
        path = GG1Result([1.0, 1.0, 5.0, 1.0], [3.0, 1.0, 1.0, 1.0])
        self.assertEqual([path[n] for n in range(4)], [0.0, 2.0, 0.0, 0.0])

    def test_waits_are_nonnegative(self):
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=1.2)).draw()
        for n in range(30):
            self.assertGreaterEqual(path[n], 0)

    def test_path_is_cached_and_stable(self):
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        first = [path[n] for n in range(20)]
        second = [path[n] for n in range(20)]
        self.assertEqual(first, second)

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        early = [path[n] for n in range(5)]
        path[60]
        self.assertEqual([path[n] for n in range(5)], early)

    def test_no_wait_when_service_is_always_faster_than_arrivals(self):
        # Every service (1.0) is shorter than every gap between arrivals
        # (2.0), so the server is always free when the next customer arrives.
        path = GG1Result([2.0] * 6, [1.0] * 6)
        self.assertEqual([path[n] for n in range(5)], [0.0] * 5)

    def test_arrival_times_accumulate_the_interarrival_times(self):
        path = GG1Result([1.0, 2.0, 0.5], [0.1, 0.1, 0.1])
        self.assertEqual([path.arrival_times[n] for n in range(3)], [1.0, 3.0, 3.5])

    def test_sojourn_time_is_wait_plus_service(self):
        path = GG1Result([1.0] * 4, [1.5] * 4)
        for n in range(3):
            self.assertAlmostEqual(
                path.sojourn_times[n], path[n] + path.service_times[n]
            )

    def test_departure_time_is_arrival_plus_sojourn(self):
        path = GG1Result([1.0] * 4, [1.5] * 4)
        for n in range(3):
            self.assertAlmostEqual(
                path.departure_times[n], path.arrival_times[n] + path.sojourn_times[n]
            )

    def test_departures_are_in_order_for_a_single_server(self):
        # One server serving first come, first served cannot let a later
        # customer leave before an earlier one.
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=1.1)).draw()
        departures = [path.departure_times[n] for n in range(30)]
        for earlier, later in zip(departures, departures[1:]):
            self.assertLessEqual(earlier, later)

    def test_getters_return_the_sequences(self):
        path = GG1Result([1.0] * 4, [1.5] * 4)
        self.assertIs(path.get_waiting_times(), path)
        self.assertIs(path.get_service_times(), path.service_times)
        self.assertIs(path.get_interarrival_times(), path.interarrival_times)
        self.assertIs(path.get_arrival_times(), path.arrival_times)
        self.assertIs(path.get_sojourn_times(), path.sojourn_times)
        self.assertIs(path.get_departure_times(), path.departure_times)

    def test_arrival_process_counts_the_arrivals(self):
        path = GG1Result([1.0] * 5, [1.5] * 5)
        arrivals = path.get_arrival_process()
        self.assertIsInstance(arrivals, RenewalProcessResult)
        # Customers arrive at times 1, 2, 3, ...
        self.assertEqual(arrivals(0.5), 0)
        self.assertEqual(arrivals(2.5), 2)

    def test_arrival_process_agrees_with_arrival_times(self):
        seed()
        path = GG1(Exponential(rate=1), Exponential(rate=2)).draw()
        arrivals = path.get_arrival_process()
        # Exactly the customers whose arrival time is before t are counted.
        for t in [0.5, 2.0, 5.0]:
            counted = sum(1 for n in range(40) if path.arrival_times[n] < t)
            self.assertEqual(arrivals(t), counted)


class TestGG1ProbabilitySpace(unittest.TestCase):

    def test_distributions_stored(self):
        interarrival, service = Exponential(rate=1), Gamma(shape=2, rate=4)
        space = GG1ProbabilitySpace(interarrival, service)
        self.assertIs(space.interarrival_dist, interarrival)
        self.assertIs(space.service_dist, service)

    def test_draw_returns_result(self):
        seed()
        space = GG1ProbabilitySpace(Exponential(rate=1), Exponential(rate=2))
        self.assertIsInstance(space.draw(), GG1Result)

    def test_successive_draws_are_independent(self):
        # The two i.i.d. sequences are built once and reused, so each draw
        # must still give a fresh path rather than repeating the first one.
        seed()
        space = GG1ProbabilitySpace(Exponential(rate=1), Exponential(rate=1.2))
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first[n] for n in range(20)], [second[n] for n in range(20)]
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

    def test_getitem_returns_rv(self):
        self.assertIsInstance(GG1(Exponential(rate=1), Exponential(rate=2))[5], RV)

    def test_getitem_matches_drawn_path(self):
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        seed(7)
        indexed = list(queue[5].sim(10))
        seed(7)
        drawn = [queue.draw()[5] for _ in range(10)]
        self.assertEqual(indexed, drawn)

    def test_reproducible_under_same_seed(self):
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        seed(123)
        first = [queue.draw()[10] for _ in range(20)]
        seed(123)
        second = [queue.draw()[10] for _ in range(20)]
        self.assertEqual(first, second)

    def test_sim_gives_independent_paths(self):
        seed()
        queue = GG1(Exponential(rate=1), Exponential(rate=1.2))
        paths = list(queue.sim(2))
        self.assertNotEqual(
            [paths[0][n] for n in range(20)], [paths[1][n] for n in range(20)]
        )

    def test_sojourn_time_via_apply(self):
        seed()
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        sojourn = queue.apply(lambda path: path.sojourn_times[10])
        self.assertIsInstance(sojourn, RV)
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
        from_mg1 = [MG1(arrival_rate=1, service_dist=service).draw()[10]]
        seed(5)
        from_gg1 = [GG1(Exponential(rate=1), service).draw()[10]]
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


class TestQueueTheory(unittest.TestCase):
    """The simulation against exact queueing-theory results."""

    def test_mm1_mean_wait(self):
        # Both distributions exponential is the M/M/1 queue, whose long-run
        # mean wait in line is rho / (mu - lambda) with rho = lambda / mu.
        seed()
        rate_in, rate_out = 1, 2
        expected = (rate_in / rate_out) / (rate_out - rate_in)
        queue = GG1(Exponential(rate=rate_in), Exponential(rate=rate_out))
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.12)

    def test_mm1_probability_of_no_wait(self):
        # An M/M/1 arrival finds the system empty -- and so waits 0 -- with
        # probability 1 - rho.
        seed()
        queue = GG1(Exponential(rate=1), Exponential(rate=2))
        no_wait = np.mean([wait == 0 for wait in waits_at(queue)])
        self.assertAlmostEqual(no_wait, 1 - queue.utilization, delta=0.06)

    def test_pollaczek_khinchine_with_erratic_service(self):
        # M/G/1: mean wait = lambda * E[S^2] / (2 * (1 - rho)). With
        # Exponential(rate=2) service, E[S^2] = 2 / 2^2 = 0.5, rho = 0.5, so
        # the mean wait is 0.5.
        seed()
        queue = MG1(arrival_rate=1, service_dist=Exponential(rate=2))
        self.assertAlmostEqual(np.mean(waits_at(queue)), 0.5, delta=0.12)

    def test_pollaczek_khinchine_with_steady_service(self):
        # The same formula with the same mean service time (0.5) but far less
        # variable service: Gamma(shape=20, rate=40) has E[S^2] = 0.2625, so
        # the mean wait is 0.2625 -- half as long on the same load, which is
        # the point of the formula depending on E[S^2] rather than E[S].
        seed()
        service = Gamma(shape=20, rate=40)
        second_moment = service.var() + service.mean() ** 2
        expected = 1 * second_moment / (2 * (1 - 0.5))
        queue = MG1(arrival_rate=1, service_dist=service)
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.08)

    def test_steadier_arrivals_shorten_the_wait(self):
        # G/M/1 against M/M/1 at identical utilization: Gamma(shape=8, rate=8)
        # arrivals are far more regular than Exponential(rate=1) ones, and
        # regularity is worth a shorter line.
        seed()
        steady = GM1(interarrival_dist=Gamma(shape=8, rate=8), service_rate=2)
        erratic = GM1(interarrival_dist=Exponential(rate=1), service_rate=2)
        self.assertAlmostEqual(steady.utilization, erratic.utilization)
        self.assertLess(np.mean(waits_at(steady)), np.mean(waits_at(erratic)))

    def test_unstable_queue_grows_without_bound(self):
        # With rho > 1 the server cannot keep up, so waits keep growing
        # instead of settling down.
        seed()
        queue = GG1(Exponential(rate=2), Exponential(rate=1))
        self.assertGreater(queue.utilization, 1)
        path = queue.draw()
        self.assertLess(path[20], path[200])

    def test_critically_loaded_queue_has_no_steady_state(self):
        # rho exactly 1 is also unstable, if more slowly.
        seed()
        queue = GG1(Exponential(rate=1), Exponential(rate=1))
        self.assertEqual(queue.utilization, 1)
        early = np.mean([queue.draw()[10] for _ in range(100)])
        late = np.mean([queue.draw()[400] for _ in range(100)])
        self.assertGreater(late, early)


class TestQueueValidation(unittest.TestCase):
    """Validation of both distributions, for the space and the queue."""

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
        seed()
        path = GG1(Exponential(rate=1), Poisson(0)).draw()
        self.assertEqual([path[n] for n in range(5)], [0.0] * 5)

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
                seed()
                self.assertGreaterEqual(GG1(interarrival, service).draw()[5], 0)

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


class TestGGsResult(unittest.TestCase):

    def test_is_infinite_vector(self):
        path = GGs(Exponential(rate=1), Exponential(rate=1), servers=2).draw()
        self.assertIsInstance(path, InfiniteVector)

    def test_servers_stored(self):
        self.assertEqual(GGsResult([1.0] * 3, [1.0] * 3, servers=4).servers, 4)

    def test_one_server_reproduces_lindley_exactly(self):
        # Same inputs, so GGsResult with one server and GG1Result must agree
        # value for value -- including the idle stretch in the middle.
        interarrival = [1.0, 1.0, 5.0, 1.0, 2.0, 1.0]
        service = [3.0, 1.0, 1.0, 1.0, 4.0, 1.0]
        lindley = [GG1Result(interarrival, service)[n] for n in range(6)]
        servers = [GGsResult(interarrival, service, servers=1)[n] for n in range(6)]
        self.assertEqual(lindley, servers)

    def test_second_server_removes_the_wait(self):
        # A customer every 1.0 with each service taking 1.5: one server falls
        # steadily behind, two keep up and nobody waits at all.
        interarrival, service = [1.0] * 6, [1.5] * 6
        one = [GGsResult(interarrival, service, servers=1)[n] for n in range(5)]
        two = [GGsResult(interarrival, service, servers=2)[n] for n in range(5)]
        self.assertEqual(one, [0.0, 0.5, 1.0, 1.5, 2.0])
        self.assertEqual(two, [0.0] * 5)

    def test_waits_by_hand_when_servers_are_overtaken(self):
        # Two servers, a customer every 1.0. Services 4, 4, 1, 1: customers 0
        # and 1 take both servers until times 5 and 6, so customer 2 (arriving
        # at 3.0) waits until 5.0 and customer 3 (arriving at 4.0) until 6.0.
        path = GGsResult([1.0] * 4, [4.0, 4.0, 1.0, 1.0], servers=2)
        self.assertEqual([path[n] for n in range(4)], [0.0, 0.0, 2.0, 2.0])

    def test_third_customer_takes_whichever_server_frees_first(self):
        # Two servers: customer 0 is served until 1.5, customer 1 until 4.0.
        # Customer 2 arrives at 3.0 and takes the server that is already free.
        path = GGsResult([1.0, 1.0, 1.0], [0.5, 2.0, 1.0], servers=2)
        self.assertEqual([path[n] for n in range(3)], [0.0, 0.0, 0.0])

    def test_waits_are_nonnegative(self):
        seed()
        path = GGs(Exponential(rate=2), Exponential(rate=1.2), servers=2).draw()
        for n in range(30):
            self.assertGreaterEqual(path[n], 0)

    def test_path_is_cached_and_stable(self):
        seed()
        path = GGs(Exponential(rate=2), Exponential(rate=1), servers=2).draw()
        first = [path[n] for n in range(20)]
        self.assertEqual(first, [path[n] for n in range(20)])

    def test_reading_far_ahead_keeps_earlier_values(self):
        seed()
        path = GGs(Exponential(rate=2), Exponential(rate=1), servers=2).draw()
        early = [path[n] for n in range(5)]
        path[60]
        self.assertEqual([path[n] for n in range(5)], early)

    def test_derived_sequences_are_inherited(self):
        path = GGsResult([1.0] * 4, [1.5] * 4, servers=2)
        self.assertEqual(path.arrival_times[2], 3.0)
        self.assertAlmostEqual(path.sojourn_times[1], path[1] + 1.5)
        self.assertAlmostEqual(
            path.departure_times[1], path.arrival_times[1] + path.sojourn_times[1]
        )
        self.assertIs(path.get_waiting_times(), path)
        self.assertIsInstance(path.get_arrival_process(), RenewalProcessResult)

    def test_departures_can_be_out_of_order_with_two_servers(self):
        # Customer 0 takes 5.0 while customer 1 takes 0.5 beside them, so
        # customer 1 leaves first -- something a single server cannot do.
        path = GGsResult([1.0, 1.0, 1.0], [5.0, 0.5, 0.5], servers=2)
        self.assertGreater(path.departure_times[0], path.departure_times[1])


class TestGGsProbabilitySpace(unittest.TestCase):

    def test_parameters_stored(self):
        interarrival, service = Exponential(rate=2), Gamma(shape=2, rate=2)
        space = GGsProbabilitySpace(interarrival, service, 3)
        self.assertIs(space.interarrival_dist, interarrival)
        self.assertIs(space.service_dist, service)
        self.assertEqual(space.servers, 3)

    def test_draw_returns_result_with_the_servers(self):
        seed()
        path = GGsProbabilitySpace(Exponential(rate=2), Exponential(rate=1), 3).draw()
        self.assertIsInstance(path, GGsResult)
        self.assertEqual(path.servers, 3)

    def test_successive_draws_are_independent(self):
        seed()
        space = GGsProbabilitySpace(Exponential(rate=2), Exponential(rate=1), 2)
        first, second = space.draw(), space.draw()
        self.assertNotEqual(
            [first[n] for n in range(20)], [second[n] for n in range(20)]
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

    def test_getitem_returns_rv(self):
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        self.assertIsInstance(queue[5], RV)

    def test_reproducible_under_same_seed(self):
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        seed(123)
        first = [queue.draw()[10] for _ in range(20)]
        seed(123)
        self.assertEqual(first, [queue.draw()[10] for _ in range(20)])

    def test_sim_gives_independent_paths(self):
        seed()
        queue = GGs(Exponential(rate=2), Exponential(rate=1.1), servers=2)
        paths = list(queue.sim(2))
        self.assertNotEqual(
            [paths[0][n] for n in range(20)], [paths[1][n] for n in range(20)]
        )

    def test_one_server_matches_gg1_distributionally(self):
        # The two consume the RNG in a different order, so the check is on the
        # mean rather than value by value: M/M/1 with rho = 0.5 waits 0.5.
        seed()
        queue = GGs(Exponential(rate=1), Exponential(rate=2), servers=1)
        self.assertAlmostEqual(np.mean(waits_at(queue)), 0.5, delta=0.12)


class TestGGsTheory(unittest.TestCase):
    """The multi-server simulation against exact M/M/s results."""

    def test_erlang_c_mean_wait_two_servers(self):
        # M/M/2 with lambda = mu = 1: mean wait in line is
        # C / (s * mu - lambda) = (1/3) / 1 = 1/3.
        seed()
        expected = erlang_c(2, 1, 1) / (2 * 1 - 1)
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.12)

    def test_erlang_c_probability_of_waiting_two_servers(self):
        seed()
        queue = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        waited = np.mean([wait > 0 for wait in waits_at(queue)])
        self.assertAlmostEqual(waited, erlang_c(2, 1, 1), delta=0.07)

    def test_erlang_c_mean_wait_three_servers(self):
        # M/M/3 with lambda = 2, mu = 1: rho = 2/3, mean wait 4/9.
        seed()
        expected = erlang_c(3, 2, 1) / (3 * 1 - 2)
        queue = GGs(Exponential(rate=2), Exponential(rate=1), servers=3)
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.12)

    def test_erlang_c_holds_for_a_saturated_queue(self):
        # M/M/2 with lambda = 3, mu = 2: rho = 0.75, a busier system.
        seed()
        expected = erlang_c(2, 3, 2) / (2 * 2 - 3)
        queue = GGs(Exponential(rate=3), Exponential(rate=2), servers=2)
        self.assertAlmostEqual(np.mean(waits_at(queue)), expected, delta=0.2)

    def test_two_slow_servers_shorten_the_line_but_not_the_visit(self):
        # The classic comparison at equal utilization: two servers of rate 1
        # against one server of rate 2. Exact M/M/ values are 1/3 vs 1/2 for
        # the wait in line, and 4/3 vs 1 for the time in system.
        seed()
        fast = GG1(Exponential(rate=1), Exponential(rate=2))
        pair = GGs(Exponential(rate=1), Exponential(rate=1), servers=2)
        self.assertAlmostEqual(fast.utilization, pair.utilization)

        fast_line, pair_line = np.mean(waits_at(fast)), np.mean(waits_at(pair))
        fast_visit = np.mean([fast.draw().sojourn_times[CUSTOMER] for _ in range(Nsim)])
        pair_visit = np.mean([pair.draw().sojourn_times[CUSTOMER] for _ in range(Nsim)])
        self.assertLess(pair_line, fast_line)  # shorter line with two servers
        self.assertGreater(pair_visit, fast_visit)  # but a longer visit

    def test_more_servers_never_lengthen_the_wait(self):
        seed()
        means = []
        for servers in [1, 2, 3]:
            queue = GGs(Exponential(rate=1), Exponential(rate=1.2), servers=servers)
            means.append(np.mean(waits_at(queue, nsim=200)))
        self.assertLess(means[1], means[0])
        self.assertLessEqual(means[2], means[1])

    def test_unstable_multi_server_queue_grows_without_bound(self):
        seed()
        queue = GGs(Exponential(rate=3), Exponential(rate=1), servers=2)
        self.assertGreater(queue.utilization, 1)
        path = queue.draw()
        self.assertLess(path[20], path[200])


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
        """The shared machinery has no recursion of its own."""
        from symbulate.queues import _QueueResult

        with self.assertRaises(NotImplementedError):
            _QueueResult([1.0] * 3, [1.0] * 3)[0]


if __name__ == "__main__":
    unittest.main()
