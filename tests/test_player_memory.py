import unittest
import warnings

from src.player_memory import PlayerMemory


class TestPlayerMemory(unittest.TestCase):
    """
    Test class for the PlayerMemory class.\n
    """

    def test_init(self):
        player_memory = PlayerMemory()
        self.assertFalse(player_memory)
        player_memory = PlayerMemory(('a', 'b'), [(1, 2), (2, 3)])
        self.assertTrue(player_memory)
        self.assertRaises(TypeError, PlayerMemory, "I'm not a tuple", [])
        self.assertRaises(TypeError, PlayerMemory, ('a', 'b'), "I'm not a list of tuples")
        self.assertRaises(TypeError, PlayerMemory, ('a', 'b'), [1, 2])
        self.assertRaises(TypeError, PlayerMemory, ('a', 'b'), [(1, 2), 2])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, PlayerMemory, ('a', 'b'), (1, 2))

    def test_add_element(self):
        player_memory = PlayerMemory()
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertTrue(player_memory)
        self.assertRaises(TypeError, player_memory.add_element, "I'm not a tuple", [])
        self.assertRaises(TypeError, player_memory.add_element, ('a', 'b'), "I'm not a list of tuples")
        self.assertRaises(TypeError, player_memory.add_element, ('a', 'b'), [1, 2])
        self.assertRaises(TypeError, player_memory.add_element, ('a', 'b'), [(1, 2), 2])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.add_element, ('a', 'b'), (1, 2))

    def test_get_players_names_pairings(self):
        player_memory = PlayerMemory()
        self.assertEqual([], player_memory.get_players_names_pairings())
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual([('a', 'b')], player_memory.get_players_names_pairings())
        player_memory.add_element(('c', 'd'), [(3, 4), (4, 5)])
        self.assertEqual([('a', 'b'), ('c', 'd')], player_memory.get_players_names_pairings())

    def test_get_actions_by_pairing(self):
        player_memory = PlayerMemory()
        self.assertRaises(TypeError, player_memory.get_actions_by_pairing, "I'm not a tuple")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.get_actions_by_pairing, ('a', 'b'))
            warnings.simplefilter("ignore")
            self.assertEqual([], player_memory.get_actions_by_pairing(('a', 'b')))
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual([(1, 2), (2, 3)], player_memory.get_actions_by_pairing(('a', 'b')))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.get_actions_by_pairing, ('c', 'd'))
            warnings.simplefilter("ignore")
            self.assertEqual([], player_memory.get_actions_by_pairing(('c', 'd')))
        player_memory.add_element(('c', 'd'), [(3, 4), (4, 5)])
        self.assertEqual([(3, 4), (4, 5)], player_memory.get_actions_by_pairing(('c', 'd')))
        self.assertEqual([(1, 2), (2, 3)], player_memory.get_actions_by_pairing(('a', 'b')))

    def test_append(self):
        player_memory = PlayerMemory()
        player_memory.append(PlayerMemory())
        self.assertFalse(player_memory)
        player_memory.append(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        self.assertTrue(player_memory)
        self.assertEqual([(1, 2), (2, 3)], player_memory.get_actions_by_pairing(('a', 'b')))
        self.assertRaises(TypeError, player_memory.append, "I'm not a PlayerMemory")

    def test_get_memory_by_player_name(self):
        player_memory = PlayerMemory()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.get_memory_by_player_name, 0)
            self.assertRaises(Warning, player_memory.get_memory_by_player_name, 'a')
            warnings.simplefilter("ignore")
            self.assertEqual(PlayerMemory(), player_memory.get_memory_by_player_name(0))
            self.assertEqual(PlayerMemory(), player_memory.get_memory_by_player_name('a'))
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]), player_memory.get_memory_by_player_name('a'))
        player_memory.add_element(('c', 'd'), [(3, 4), (4, 5)])
        self.assertEqual(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]), player_memory.get_memory_by_player_name('a'))
        self.assertEqual(PlayerMemory(('c', 'd'), [(3, 4), (4, 5)]), player_memory.get_memory_by_player_name('c'))

    def test_bool(self):
        player_memory = PlayerMemory()
        self.assertFalse(bool(player_memory))
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertTrue(bool(player_memory))

    def test_eq(self):
        player_memory = PlayerMemory()
        self.assertEqual(PlayerMemory(), player_memory)
        self.assertNotEqual(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]), player_memory)
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]), player_memory)
        self.assertNotEqual(PlayerMemory(('c', 'd'), [(3, 4), (4, 5)]), player_memory)
