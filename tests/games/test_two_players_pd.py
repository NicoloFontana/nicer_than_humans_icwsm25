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
        self.assertEqual(10, game.get_iterations())
        self.assertEqual({1, 0}, game.get_action_space())
        self.assertEqual(two_players_pd_payoff, game.get_payoff_function())
        self.assertEqual([], game.get_players())
        game = TwoPlayersPD(self.player1)
        self.assertEqual(["AlwaysCooperate"], game.get_players())
        game = TwoPlayersPD(self.player1, self.player2)
        self.assertEqual(["AlwaysCooperate", "AlwaysDefect"], game.get_players())

    def test_add_player(self):
        game = TwoPlayersPD()
        game.add_player(self.player1)
        self.assertEqual(["AlwaysCooperate"], game.get_players())
        self.assertRaises(TypeError, game.add_player, "I'm not a player")
        game.add_player(self.player2)
        self.assertEqual(["AlwaysCooperate", "AlwaysDefect"], game.get_players())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(UserWarning, game.add_player, self.player1)
            warnings.simplefilter("ignore")
            self.assertEqual(["AlwaysCooperate", "AlwaysDefect"], game.get_players())

    # Tests for two_players_pd_utils.py

    def test_two_players_pd_payoff(self):
        self.assertEqual(3, two_players_pd_payoff(1, 1))
        self.assertEqual(0, two_players_pd_payoff(1, 0))
        self.assertEqual(5, two_players_pd_payoff(0, 1))
        self.assertEqual(1, two_players_pd_payoff(0, 0))
        self.assertRaises(ValueError, two_players_pd_payoff, 0, 2)

    def test_from_nat_lang(self):
        self.assertEqual(1, from_nat_lang("Cooperate"))
        self.assertEqual(1, from_nat_lang('"Cooperate"'))
        self.assertEqual(0, from_nat_lang("Defect"))
        self.assertEqual(0, from_nat_lang('"Defect"'))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(UserWarning, from_nat_lang, "I'm an error")
            warnings.simplefilter("ignore")
            self.assertEqual(0, from_nat_lang("I'm an error"))

    def test_to_nat_lang(self):
        self.assertEqual("Cooperate", to_nat_lang(1))
        self.assertEqual("Defect", to_nat_lang(0))
        self.assertEqual('{"Cooperate", "Defect"}', to_nat_lang({1, 0}))
        self.assertEqual('{"Cooperate", "Defect"}', to_nat_lang({0, 1}))
        self.assertRaises(ValueError, to_nat_lang, 2)
        self.assertRaises(ValueError, to_nat_lang, {})
        self.assertRaises(ValueError, to_nat_lang, {1, 0, 2})
        self.assertRaises(ValueError, to_nat_lang, {1, 2})
        self.assertRaises(ValueError, to_nat_lang, {0, 2})
        self.assertEqual('"Cooperate"', to_nat_lang(1, string_of_string=True))
        self.assertEqual('"Defect"', to_nat_lang(0, string_of_string=True))
        self.assertEqual('"{"Cooperate", "Defect"}"', to_nat_lang({1, 0}, string_of_string=True))
        self.assertEqual('"{"Cooperate", "Defect"}"', to_nat_lang({0, 1}, string_of_string=True))

    # TODO? Test play_round?
