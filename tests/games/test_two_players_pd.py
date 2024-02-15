import unittest
from unittest.mock import Mock

from src.games.two_players_pd import TwoPlayersPD, two_players_pd_payoff


class TestTwoPlayersPD(unittest.TestCase):

    def setUp(self):
        self.player1 = Mock()
        self.player1.get_name.return_value = "AlwaysCooperate"
        self.player1.play_round.return_value = 1
        self.player2 = Mock()
        self.player2.get_name.return_value = "AlwaysDefect"
        self.player2.play_round.return_value = 0
        self.game = TwoPlayersPD(self.player1, self.player2)

    def test_two_players_pd_payoff(self):
        self.assertEqual(two_players_pd_payoff(1, 1), 3)
        self.assertEqual(two_players_pd_payoff(1, 0), 0)
        self.assertEqual(two_players_pd_payoff(0, 1), 5)
        self.assertEqual(two_players_pd_payoff(0, 0), 1)
        self.assertRaises(ValueError, two_players_pd_payoff, 0, 2)

    def test_init(self):
        self.assertEqual(self.game.iterations, 10)
        self.assertEqual(self.game.action_space, {1, 0})
        self.assertEqual(self.game.payoff_function(1, 1), 3)
        self.assertEqual(self.game.payoff_function(1, 0), 0)
        self.assertEqual(self.game.payoff_function(0, 1), 5)
        self.assertEqual(self.game.payoff_function(0, 0), 1)
        self.assertEqual(self.game.players, {"AlwaysCooperate": self.player1, "AlwaysDefect": self.player2})

    # TODO: test play_round

    # TODO: test add_player
