import unittest

from src.player import Player
from src.player_memory import PlayerMemory


class TestPlayer(unittest.TestCase):

    def test_init(self):
        player = Player("a")
        self.assertEqual(player.get_name(), "a")
        self.assertEqual(player.get_score(), 0)

    def test_update_score(self):
        player = Player("a")
        player.update_score(10)
        self.assertEqual(player.get_score(), 10)
        player.update_score(-5)
        self.assertEqual(player.get_score(), 5)
        self.assertRaises(ValueError, player.update_score, "I'm a runtime error")

    def test_set_memory(self):
        player = Player("a")
        self.assertRaises(ValueError, player.set_memory, "I'm a runtime error")
        player.set_memory(PlayerMemory())
        self.assertFalse(bool(player.get_memory()))
        player.set_memory(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        self.assertTrue(bool(player.get_memory()))

    def test_update_memory(self):
        player = Player("a")
        self.assertRaises(ValueError, player.update_memory, "I'm a runtime error")
        player.update_memory(PlayerMemory())
        self.assertFalse(bool(player.get_memory()))
        new_memory = PlayerMemory(('a', 'b'), [(1, 2), (2, 3)])
        player.update_memory(new_memory)
        self.assertTrue(bool(player.get_memory()))
        self.assertEqual(player.get_memory(), new_memory)

    def test_eq(self):
        player1 = Player("a")
        player2 = Player("a")
        self.assertEqual(player1, player2)
        player3 = Player("b")
        self.assertNotEqual(player1, player3)
        player2.update_score(10)
        self.assertNotEqual(player1, player2)
        player1.update_score(10)
        self.assertEqual(player1, player2)
        player2.set_memory(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        self.assertNotEqual(player1, player2)
        player1.set_memory(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        self.assertEqual(player1, player2)
        player2.update_memory(PlayerMemory(('a', 'b'), [(5, 6), (6, 7)]))
        self.assertNotEqual(player1, player2)
        player1.update_memory(PlayerMemory(('a', 'b'), [(5, 6), (6, 7)]))
        self.assertEqual(player1, player2)
