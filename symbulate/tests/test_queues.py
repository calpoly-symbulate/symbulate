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
- a queue with rho >= 1 has waiting times that grow without bound.

Reproducibility is obtained by reseeding the distributions module's
generator (``distributions.rng``), matching test_renewal_process.py.
"""

import unittest

import numpy as np

from symbulate import *
from symbulate import distributions
from symbulate.queues import GG1, GG1ProbabilitySpace, GG1Result, MG1, GM1
from symbulate.renewal_process import RenewalProcessResult
from symbulate.result import InfiniteVector

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


if __name__ == "__main__":
    unittest.main()
