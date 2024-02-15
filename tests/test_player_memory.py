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
        self.assertRaises(ValueError, PlayerMemory, "I'm a runtime error", [])
        self.assertRaises(ValueError, PlayerMemory, ('a', 'b'), "I'm a runtime error")
        self.assertRaises(ValueError, PlayerMemory, ('a', 'b'), [1, 2])
        self.assertRaises(ValueError, PlayerMemory, ('a', 'b'), [(1, 2), 2])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, PlayerMemory, ('a', 'b'), (1, 2))

    def test_add_element(self):
        player_memory = PlayerMemory()
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertTrue(player_memory)
        self.assertRaises(ValueError, player_memory.add_element, "I'm a runtime error", [])
        self.assertRaises(ValueError, player_memory.add_element, ('a', 'b'), "I'm a runtime error")
        self.assertRaises(ValueError, player_memory.add_element, ('a', 'b'), [1, 2])
        self.assertRaises(ValueError, player_memory.add_element, ('a', 'b'), [(1, 2), 2])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.add_element, ('a', 'b'), (1, 2))

    def test_get_players_names_pairings(self):
        player_memory = PlayerMemory()
        self.assertEqual(player_memory.get_players_names_pairings(), [])
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual(player_memory.get_players_names_pairings(), [('a', 'b')])
        player_memory.add_element(('c', 'd'), [(3, 4), (4, 5)])
        self.assertEqual(player_memory.get_players_names_pairings(), [('a', 'b'), ('c', 'd')])

    def test_get_actions_by_pairing(self):
        player_memory = PlayerMemory()
        self.assertRaises(ValueError, player_memory.get_actions_by_pairing, "I'm a runtime error")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.get_actions_by_pairing, ('a', 'b'))
            warnings.simplefilter("ignore")
            self.assertEqual(player_memory.get_actions_by_pairing(('a', 'b')), [])
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual(player_memory.get_actions_by_pairing(('a', 'b')), [(1, 2), (2, 3)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.get_actions_by_pairing, ('c', 'd'))
            warnings.simplefilter("ignore")
            self.assertEqual(player_memory.get_actions_by_pairing(('c', 'd')), [])
        player_memory.add_element(('c', 'd'), [(3, 4), (4, 5)])
        self.assertEqual(player_memory.get_actions_by_pairing(('c', 'd')), [(3, 4), (4, 5)])
        self.assertEqual(player_memory.get_actions_by_pairing(('a', 'b')), [(1, 2), (2, 3)])

    def test_append(self):
        player_memory = PlayerMemory()
        player_memory.append(PlayerMemory())
        self.assertFalse(player_memory)
        player_memory.append(PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        self.assertTrue(player_memory)
        self.assertEqual(player_memory.get_actions_by_pairing(('a', 'b')), [(1, 2), (2, 3)])
        self.assertRaises(ValueError, player_memory.append, "I'm a runtime error")

    def test_get_memory_by_player_name(self):
        player_memory = PlayerMemory()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, player_memory.get_memory_by_player_name, 0)
            self.assertRaises(Warning, player_memory.get_memory_by_player_name, 'a')
            warnings.simplefilter("ignore")
            self.assertEqual(player_memory.get_memory_by_player_name(0), PlayerMemory())
            self.assertEqual(player_memory.get_memory_by_player_name('a'), PlayerMemory())
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual(player_memory.get_memory_by_player_name('a'), PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        player_memory.add_element(('c', 'd'), [(3, 4), (4, 5)])
        self.assertEqual(player_memory.get_memory_by_player_name('a'), PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        self.assertEqual(player_memory.get_memory_by_player_name('c'), PlayerMemory(('c', 'd'), [(3, 4), (4, 5)]))

    def test_bool(self):
        player_memory = PlayerMemory()
        self.assertFalse(bool(player_memory))
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertTrue(bool(player_memory))

    def test_eq(self):
        player_memory = PlayerMemory()
        self.assertEqual(player_memory, PlayerMemory())
        self.assertNotEqual(player_memory, PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        player_memory.add_element(('a', 'b'), [(1, 2), (2, 3)])
        self.assertEqual(player_memory, PlayerMemory(('a', 'b'), [(1, 2), (2, 3)]))
        self.assertNotEqual(player_memory, PlayerMemory(('c', 'd'), [(3, 4), (4, 5)]))
