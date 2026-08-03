"""Tests for symbulate.probability_space.

Covers ProbabilitySpace (draw, sim, apply, multiply, power),
Event (draw, sim, logical operators, type guards),
BoxModel (list/dict init, size, replacement, ordering, error handling),
and DeckOfCards.
"""

import unittest
import numpy as np

from symbulate import *
from symbulate import probability_space
from symbulate.probability_space import Event
from symbulate.result import Vector, InfiniteVector


def seed(value=42):
    probability_space.rng = np.random.default_rng(value)


class TestProbabilitySpace(unittest.TestCase):

    def test_draw_returns_outcome(self):
        P = ProbabilitySpace(lambda: 7)
        self.assertEqual(P.draw(), 7)

    def test_sim_returns_results(self):
        P = ProbabilitySpace(lambda: 1)
        self.assertIsInstance(P.sim(5), Results)

    def test_sim_length(self):
        P = ProbabilitySpace(lambda: 1)
        self.assertEqual(len(P.sim(10)), 10)

    def test_apply_transforms_outcome(self):
        P = ProbabilitySpace(lambda: 3)
        Q = P.apply(lambda x: x * 2)
        self.assertEqual(Q.draw(), 6)

    def test_apply_returns_probability_space(self):
        P = ProbabilitySpace(lambda: 3)
        self.assertIsInstance(P.apply(lambda x: x), ProbabilitySpace)

    def test_mul_returns_probability_space(self):
        P1 = ProbabilitySpace(lambda: 1)
        P2 = ProbabilitySpace(lambda: 2)
        self.assertIsInstance(P1 * P2, ProbabilitySpace)

    def test_mul_draws_both_outcomes(self):
        P1 = ProbabilitySpace(lambda: "a")
        P2 = ProbabilitySpace(lambda: "b")
        result = list((P1 * P2).draw())
        self.assertIn("a", result)
        self.assertIn("b", result)

    def test_pow_returns_vector(self):
        seed()
        P = ProbabilitySpace(lambda: 1)
        result = (P**3).draw()
        self.assertIsInstance(result, Vector)

    def test_pow_vector_has_correct_length(self):
        seed()
        P = ProbabilitySpace(lambda: 1)
        result = (P**4).draw()
        self.assertEqual(len(result), 4)

    def test_pow_inf_returns_infinite_vector(self):
        P = ProbabilitySpace(lambda: 1)
        result = (P ** float("inf")).draw()
        self.assertIsInstance(result, InfiniteVector)

    def test_non_callable_draw_raises_type_error(self):
        with self.assertRaises(TypeError):
            ProbabilitySpace(5)

    def test_sim_negative_n_raises(self):
        P = ProbabilitySpace(lambda: 1)
        with self.assertRaises(ValueError):
            P.sim(-1)

    def test_sim_zero_n_raises(self):
        P = ProbabilitySpace(lambda: 1)
        with self.assertRaises(ValueError):
            P.sim(0)

    def test_sim_float_n_raises(self):
        P = ProbabilitySpace(lambda: 1)
        with self.assertRaises(ValueError):
            P.sim(2.5)

    def test_pow_zero_raises(self):
        P = ProbabilitySpace(lambda: 1)
        with self.assertRaises(ValueError):
            P**0

    def test_pow_negative_raises(self):
        P = ProbabilitySpace(lambda: 1)
        with self.assertRaises(ValueError):
            P**-1

    def test_check_same_passes_for_same_space(self):
        P = ProbabilitySpace(lambda: 1)
        P.check_same(P)  # should not raise

    def test_check_same_raises_for_different_spaces(self):
        P1 = ProbabilitySpace(lambda: 1)
        P2 = ProbabilitySpace(lambda: 2)
        with self.assertRaises(ValueError):
            P1.check_same(P2)


class TestEvent(unittest.TestCase):

    def test_draw_returns_bool(self):
        seed()
        P = ProbabilitySpace(lambda: probability_space.rng.choice(range(1, 7)))
        event = Event(P, lambda x: x > 3)
        self.assertIsInstance(event.draw(), (bool, np.bool_))

    def test_sim_contains_only_bools(self):
        seed()
        P = ProbabilitySpace(lambda: probability_space.rng.choice(range(1, 7)))
        event = Event(P, lambda x: x > 3)
        results = event.sim(20)
        self.assertTrue(all(isinstance(v, (bool, np.bool_)) for v in results))

    def test_and_both_true(self):
        P = ProbabilitySpace(lambda: True)
        A = Event(P, lambda x: x)
        B = Event(P, lambda x: x)
        self.assertTrue((A & B).draw())

    def test_or_one_true(self):
        P = ProbabilitySpace(lambda: True)
        A = Event(P, lambda x: not x)  # always False
        B = Event(P, lambda x: x)  # always True
        self.assertTrue((A | B).draw())

    def test_not_inverts_result(self):
        P = ProbabilitySpace(lambda: True)
        A = Event(P, lambda x: x)  # always True
        self.assertFalse((~A).draw())

    def test_bool_raises_type_error(self):
        P = ProbabilitySpace(lambda: 1)
        A = Event(P, lambda x: x > 0)
        with self.assertRaises(TypeError):
            bool(A)

    def test_logical_op_different_spaces_raises(self):
        P1 = ProbabilitySpace(lambda: 1)
        P2 = ProbabilitySpace(lambda: 1)
        A = Event(P1, lambda x: True)
        B = Event(P2, lambda x: True)
        with self.assertRaises(ValueError):
            A & B

    def test_and_with_non_event_raises(self):
        P = ProbabilitySpace(lambda: 1)
        A = Event(P, lambda x: True)
        with self.assertRaises(TypeError):
            A & 5


class TestBoxModel(unittest.TestCase):

    def test_draw_from_list_in_box(self):
        seed()
        bm = BoxModel(["a", "b", "c"])
        self.assertIn(bm.draw(), ["a", "b", "c"])

    def test_draw_from_dict_in_box(self):
        seed()
        bm = BoxModel({"H": 1, "T": 1})
        self.assertIn(bm.draw(), ["H", "T"])

    def test_dict_expands_counts(self):
        bm = BoxModel({"a": 3, "b": 1})
        self.assertEqual(len(bm.box), 4)
        self.assertEqual(bm.box.count("a"), 3)

    def test_size_none_returns_scalar(self):
        seed()
        bm = BoxModel(["x", "y", "z"], size=None)
        self.assertNotIsInstance(bm.draw(), Vector)

    def test_size_one_normalized_to_none(self):
        # size=1 is stored internally as None (returns a scalar, not a Vector)
        bm = BoxModel(["x", "y", "z"], size=1)
        self.assertIsNone(bm.size)

    def test_size_n_returns_vector(self):
        seed()
        bm = BoxModel(["a", "b", "c"], size=2)
        self.assertIsInstance(bm.draw(), Vector)

    def test_size_n_vector_has_correct_length(self):
        seed()
        bm = BoxModel(["a", "b", "c"], size=2)
        self.assertEqual(len(bm.draw()), 2)

    def test_size_inf_returns_infinite_vector(self):
        seed()
        bm = BoxModel(["a", "b"], size=float("inf"))
        self.assertIsInstance(bm.draw(), InfiniteVector)

    def test_replace_false_no_duplicates(self):
        seed()
        bm = BoxModel([1, 2, 3, 4, 5], size=5, replace=False)
        result = list(bm.draw())
        self.assertEqual(len(result), len(set(result)))

    def test_order_matters_false_result_is_sorted(self):
        seed()
        bm = BoxModel([3, 1, 2], size=3, replace=False, order_matters=False)
        result = list(bm.draw())
        self.assertEqual(result, sorted(result, key=str))

    def test_invalid_box_type_raises(self):
        with self.assertRaises(TypeError):
            BoxModel("abc")

    def test_size_exceeds_box_without_replacement_raises(self):
        with self.assertRaises(ValueError):
            BoxModel(["a", "b", "c"], size=4, replace=False)

    def test_probs_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            BoxModel(["a", "b", "c"], probs=[0.5, 0.5])


class TestDeckOfCards(unittest.TestCase):

    def test_deck_has_52_cards(self):
        deck = DeckOfCards()
        self.assertEqual(len(deck.box), 52)

    def test_each_card_is_rank_suit_tuple(self):
        deck = DeckOfCards()
        for card in deck.box:
            self.assertIsInstance(card, tuple)
            self.assertEqual(len(card), 2)

    def test_all_suits_present(self):
        deck = DeckOfCards()
        suits = {card[1] for card in deck.box}
        self.assertEqual(suits, {"Diamonds", "Hearts", "Clubs", "Spades"})

    def test_all_ranks_present(self):
        deck = DeckOfCards()
        ranks = {card[0] for card in deck.box}
        self.assertEqual(ranks, set(range(2, 11)) | {"J", "Q", "K", "A"})

    def test_default_no_replacement(self):
        deck = DeckOfCards()
        self.assertFalse(deck.replace)

    def test_draw_single_card_is_tuple(self):
        seed()
        deck = DeckOfCards()
        self.assertIsInstance(deck.draw(), tuple)

    def test_draw_size_returns_vector(self):
        seed()
        deck = DeckOfCards(size=5)
        result = deck.draw()
        self.assertIsInstance(result, Vector)
        self.assertEqual(len(result), 5)


class TestPokerHands(unittest.TestCase):

    # One fixed, hand-built example of each category. Deterministic — no
    # simulation, so these assert exact behavior.
    HANDS = {
        "royal flush": [
            (10, "Hearts"),
            ("J", "Hearts"),
            ("Q", "Hearts"),
            ("K", "Hearts"),
            ("A", "Hearts"),
        ],
        "straight flush": [
            (5, "Clubs"),
            (6, "Clubs"),
            (7, "Clubs"),
            (8, "Clubs"),
            (9, "Clubs"),
        ],
        "four of a kind": [
            (7, "Clubs"),
            (7, "Hearts"),
            (7, "Spades"),
            (7, "Diamonds"),
            (2, "Clubs"),
        ],
        "full house": [
            (3, "Clubs"),
            (3, "Hearts"),
            (3, "Spades"),
            (8, "Clubs"),
            (8, "Diamonds"),
        ],
        "flush": [
            (2, "Spades"),
            (5, "Spades"),
            (7, "Spades"),
            (9, "Spades"),
            ("J", "Spades"),
        ],
        "straight": [
            (4, "Clubs"),
            (5, "Hearts"),
            (6, "Spades"),
            (7, "Clubs"),
            (8, "Diamonds"),
        ],
        "three of a kind": [
            ("Q", "Clubs"),
            ("Q", "Hearts"),
            ("Q", "Spades"),
            (2, "Clubs"),
            (5, "Diamonds"),
        ],
        "two pair": [
            (9, "Clubs"),
            (9, "Hearts"),
            (4, "Spades"),
            (4, "Clubs"),
            ("K", "Diamonds"),
        ],
        "pair": [
            (6, "Clubs"),
            (6, "Hearts"),
            (2, "Spades"),
            (9, "Clubs"),
            ("J", "Diamonds"),
        ],
        "high card": [
            (2, "Clubs"),
            (5, "Hearts"),
            (7, "Spades"),
            (9, "Clubs"),
            ("J", "Diamonds"),
        ],
    }

    PREDICATES = {
        "royal flush": is_royal_flush,
        "straight flush": is_straight_flush,
        "four of a kind": is_four_of_a_kind,
        "full house": is_full_house,
        "flush": is_flush,
        "straight": is_straight,
        "three of a kind": is_three_of_a_kind,
        "two pair": is_two_pair,
        "pair": is_pair,
        "high card": is_high_card,
    }

    def test_classify_each_category(self):
        for category, hand in self.HANDS.items():
            with self.subTest(category=category):
                self.assertEqual(classify_hand(hand), category)

    def test_matching_predicate_true(self):
        # The is_X for a hand's own category returns True...
        for category, hand in self.HANDS.items():
            with self.subTest(category=category):
                self.assertIs(self.PREDICATES[category](hand), True)

    def test_other_predicates_false(self):
        # ...and every other is_X returns False (mutually exclusive).
        for category, hand in self.HANDS.items():
            for other, predicate in self.PREDICATES.items():
                if other == category:
                    continue
                with self.subTest(hand=category, predicate=other):
                    self.assertIs(predicate(hand), False)

    def test_every_category_is_in_poker_hands(self):
        self.assertEqual(set(self.HANDS), set(POKER_HANDS))

    def test_wheel_straight_ace_low(self):
        # A-2-3-4-5 is a straight even though the Ace is high-valued.
        wheel = [
            ("A", "Clubs"),
            (2, "Hearts"),
            (3, "Spades"),
            (4, "Clubs"),
            (5, "Diamonds"),
        ]
        self.assertEqual(classify_hand(wheel), "straight")

    def test_ace_high_straight(self):
        # 10-J-Q-K-A across suits is a plain straight, not a flush.
        hand = [
            (10, "Clubs"),
            ("J", "Hearts"),
            ("Q", "Spades"),
            ("K", "Clubs"),
            ("A", "Diamonds"),
        ]
        self.assertEqual(classify_hand(hand), "straight")

    def test_full_house_is_not_a_pair(self):
        # Exclusivity: a full house is not reported as a pair.
        self.assertFalse(is_pair(self.HANDS["full house"]))

    def test_straight_flush_is_not_flush_or_straight(self):
        sf = self.HANDS["straight flush"]
        self.assertFalse(is_flush(sf))
        self.assertFalse(is_straight(sf))

    def test_too_few_cards_raises(self):
        with self.assertRaises(ValueError):
            classify_hand([(2, "Clubs"), (3, "Hearts"), (4, "Spades"), (5, "Clubs")])

    def test_too_many_cards_raises(self):
        with self.assertRaises(ValueError):
            classify_hand(
                [
                    (2, "Clubs"),
                    (3, "Hearts"),
                    (4, "Spades"),
                    (5, "Clubs"),
                    (6, "Diamonds"),
                    (7, "Hearts"),
                    (8, "Spades"),
                ]
            )

    def test_invalid_rank_raises(self):
        with self.assertRaises(ValueError):
            classify_hand(
                [
                    ("Z", "Clubs"),
                    (3, "Hearts"),
                    (4, "Spades"),
                    (5, "Clubs"),
                    (6, "Diamonds"),
                ]
            )

    def test_works_on_a_real_draw(self):
        # classify_hand should accept a Vector straight from a deck draw.
        seed()
        hand = DeckOfCards(size=5).draw()
        self.assertIn(classify_hand(hand), POKER_HANDS)


if __name__ == "__main__":
    unittest.main()
