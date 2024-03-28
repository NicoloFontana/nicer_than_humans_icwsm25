# import unittest
#
# from src.player import Player
# from src.player_memory import PlayerMemory
#
#
# class TestPlayer(unittest.TestCase):
#
#     def test_init(self):
#         player = Player("a")
#         self.assertEqual("a", player.get_name())
#         self.assertEqual(0, player.get_total_payoff())
#
#     def test_update_payoff(self):
#         player = Player("a")
#         player.update_total_payoff(10)
#         self.assertEqual(10, player.get_total_payoff())
#         player.update_total_payoff(-5)
#         self.assertEqual(5, player.get_total_payoff())
#         self.assertRaises(TypeError, player.update_total_payoff, "I'm not a number")
#
#     def test_set_memory(self):
#         player = Player("a")
#         self.assertRaises(TypeError, player.set_memory, "I'm not a PlayerMemory")
#         player.set_memory(PlayerMemory())
#         self.assertFalse(bool(player.get_memory()))
#         player.set_memory(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
#         self.assertTrue(bool(player.get_memory()))
#
#     def test_update_memory(self):
#         player = Player("a")
#         self.assertRaises(TypeError, player.update_memory, "I'm not a PlayerMemory")
#         player.update_memory(PlayerMemory())
#         self.assertFalse(bool(player.get_memory()))
#         new_memory = PlayerMemory(('a', 'b'), [(1, 2), (2, 3)])
#         player.update_memory(new_memory)
#         self.assertTrue(bool(player.get_memory()))
#         self.assertEqual(new_memory, player.get_memory())
#
#     def test_eq(self):
#         player1 = Player("a")
#         player2 = Player("a")
#         self.assertEqual(player1, player2)
#         player3 = Player("b")
#         self.assertNotEqual(player1, player3)
#         player2.update_total_payoff(10)
#         self.assertNotEqual(player1, player2)
#         player1.update_total_payoff(10)
#         self.assertEqual(player1, player2)
#         player2.set_memory(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
#         self.assertNotEqual(player1, player2)
#         player1.set_memory(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
#         self.assertEqual(player1, player2)
#         player2.update_memory(PlayerMemory(('a', 'b'), [(5, 6), (6, 7)]))
#         self.assertNotEqual(player1, player2)
#         player1.update_memory(PlayerMemory(('a', 'b'), [(5, 6), (6, 7)]))
#         self.assertEqual(player1, player2)
