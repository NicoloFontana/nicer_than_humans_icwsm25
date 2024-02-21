import unittest
import warnings

from src.games.gt_game import GTGame
from src.player import Player


class TestGTGame(unittest.TestCase):

    def test_play_round(self):
        game = GTGame(iterations=2)
        self.assertEqual(1, game.get_current_round())
        self.assertFalse(game.is_ended)
        game.play_round()
        self.assertEqual(2, game.get_current_round())
        self.assertFalse(game.is_ended)
        game.play_round()
        self.assertEqual(3, game.get_current_round())
        self.assertTrue(game.is_ended)

    def test_get_player(self):
        empty_game = GTGame()
        self.assertEqual([], empty_game.get_players())
        game = GTGame(players={"Alice": Player("Alice"), "Bob": Player("Bob")})
        self.assertEqual(["Alice", "Bob"], game.get_players())

    def test_add_player(self):
        game = GTGame()
        self.assertEqual([], game.get_players())
        self.assertRaises(TypeError, game.add_player, "Alice")
        alice = Player("Alice")
        game.add_player(alice)
        self.assertEqual([alice.get_name()], game.get_players())
        bob = Player("Bob")
        game.add_player(bob)
        self.assertEqual([alice.get_name(), bob.get_name()], game.get_players())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(UserWarning, game.add_player, alice)
            warnings.simplefilter("ignore")
            self.assertEqual([alice.get_name(), bob.get_name()], game.get_players())
