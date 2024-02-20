import unittest
import warnings
from unittest.mock import Mock

from src.games.two_players_pd import TwoPlayersPD
from src.games.two_players_pd_utils import two_players_pd_payoff, from_nat_lang, to_nat_lang
from src.player import Player


class TestTwoPlayersPD(unittest.TestCase):

    def setUp(self):
        self.player1 = Mock()
        self.player1.__class__ = Player
        self.player1.get_name.return_value = "AlwaysCooperate"
        self.player1.play_round.return_value = 1
        self.player2 = Mock()
        self.player2.__class__ = Player
        self.player2.get_name.return_value = "AlwaysDefect"
        self.player2.play_round.return_value = 0

    def test_init(self):
        game = TwoPlayersPD()
        self.assertEqual(game.get_iterations(), 10)
        self.assertEqual(game.get_action_space(), {1, 0})
        self.assertEqual(game.get_payoff_function(), two_players_pd_payoff)
        self.assertEqual(game.get_players(), [])
        game = TwoPlayersPD(self.player1)
        self.assertEqual(game.get_players(), ["AlwaysCooperate"])
        game = TwoPlayersPD(self.player1, self.player2)
        self.assertEqual(game.get_players(), ["AlwaysCooperate", "AlwaysDefect"])

    def test_add_player(self):
        game = TwoPlayersPD()
        game.add_player(self.player1)
        self.assertEqual(game.get_players(), ["AlwaysCooperate"])
        self.assertRaises(ValueError, game.add_player, "I'm not a player")
        game.add_player(self.player2)
        self.assertEqual(game.get_players(), ["AlwaysCooperate", "AlwaysDefect"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(UserWarning, game.add_player, self.player1)
            warnings.simplefilter("ignore")
            self.assertEqual(game.get_players(), ["AlwaysCooperate", "AlwaysDefect"])

    # Tests for two_players_pd_utils.py

    def test_two_players_pd_payoff(self):
        self.assertEqual(two_players_pd_payoff(1, 1), 3)
        self.assertEqual(two_players_pd_payoff(1, 0), 0)
        self.assertEqual(two_players_pd_payoff(0, 1), 5)
        self.assertEqual(two_players_pd_payoff(0, 0), 1)
        self.assertRaises(ValueError, two_players_pd_payoff, 0, 2)

    def test_from_nat_lang(self):
        self.assertEqual(from_nat_lang("Cooperate"), 1)
        self.assertEqual(from_nat_lang('"Cooperate"'), 1)
        self.assertEqual(from_nat_lang("Defect"), 0)
        self.assertEqual(from_nat_lang('"Defect"'), 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(UserWarning, from_nat_lang, "I'm an error")
            warnings.simplefilter("ignore")
            self.assertEqual(from_nat_lang("I'm an error"), 0)

    def test_to_nat_lang(self):
        self.assertEqual(to_nat_lang(1), "Cooperate")
        self.assertEqual(to_nat_lang(0), "Defect")
        self.assertEqual(to_nat_lang({1, 0}), '{"Cooperate", "Defect"}')
        self.assertEqual(to_nat_lang({0, 1}), '{"Cooperate", "Defect"}')
        self.assertRaises(ValueError, to_nat_lang, 2)
        self.assertRaises(ValueError, to_nat_lang, {})
        self.assertRaises(ValueError, to_nat_lang, {1, 0, 2})
        self.assertRaises(ValueError, to_nat_lang, {1, 2})
        self.assertRaises(ValueError, to_nat_lang, {0, 2})
        self.assertEqual(to_nat_lang(1, string_of_string=True), '"Cooperate"')
        self.assertEqual(to_nat_lang(0, string_of_string=True), '"Defect"')
        self.assertEqual(to_nat_lang({1, 0}, string_of_string=True), '"{"Cooperate", "Defect"}"')
        self.assertEqual(to_nat_lang({0, 1}, string_of_string=True), '"{"Cooperate", "Defect"}"')

    # TODO? Test play_round?
