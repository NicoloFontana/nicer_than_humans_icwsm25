import unittest
import warnings

from src.games.game_history import GameHistory


class TestGameHistory(unittest.TestCase):

    def test_init(self):
        game_history = GameHistory()
        self.assertFalse(bool(game_history))

    def test_add_last_iteration(self):
        game_history = GameHistory()
        self.assertRaises(ValueError, game_history.add_last_iteration, ['a', 'b'], [1, 2, 3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, game_history.add_last_iteration, 'a', [])
            self.assertRaises(Warning, game_history.add_last_iteration, ['a', 'b'], "I'm a runtime error")
            self.assertRaises(Warning, game_history.add_last_iteration, ['a', 'b'], [1])
            self.assertRaises(Warning, game_history.add_last_iteration, ['a', 'b'], (1, 2))
            self.assertRaises(Warning, game_history.add_last_iteration, 'a', [1])
            warnings.simplefilter("ignore")
        game_history.add_last_iteration(['a', 'b'], [1, 2])
        self.assertTrue(bool(game_history))

    def test_get_actions_by_player(self):
        game_history = GameHistory()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, game_history.get_actions_by_player, 'a')
            warnings.simplefilter("ignore")
            self.assertEqual(game_history.get_actions_by_player('a'), [])
        game_history.add_last_iteration(['a', 'b'], [1, 2])
        self.assertEqual(game_history.get_actions_by_player('a'), [1])
        self.assertEqual(game_history.get_actions_by_player('b'), [2])

    def test_get_actions_by_iteration(self):
        game_history = GameHistory()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, game_history.get_actions_by_iteration, 0)
            warnings.simplefilter("ignore")
            self.assertEqual(game_history.get_actions_by_iteration(0), {})
        game_history.add_last_iteration(['a', 'b'], [1, 2])
        self.assertEqual(game_history.get_actions_by_iteration(0), {'a': 1, 'b': 2})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, game_history.get_actions_by_iteration, 0.2)
            self.assertRaises(Warning, game_history.get_actions_by_iteration, 0.5)
            self.assertRaises(Warning, game_history.get_actions_by_iteration, 0.7)
            warnings.simplefilter("ignore")
            self.assertEqual(game_history.get_actions_by_iteration(0.2), {'a': 1, 'b': 2})
            self.assertEqual(game_history.get_actions_by_iteration(0.5), {'a': 1, 'b': 2})
            self.assertEqual(game_history.get_actions_by_iteration(0.7), {'a': 1, 'b': 2})

    def test_bool(self):
        game_history = GameHistory()
        self.assertFalse(bool(game_history))
        game_history.add_last_iteration(['a', 'b'], [1, 2])
        self.assertTrue(bool(game_history))

    def test_eq(self):
        game_history1 = GameHistory()
        game_history2 = GameHistory()
        self.assertEqual(game_history1, game_history2)
        game_history1.add_last_iteration(['a', 'b'], [1, 2])
        self.assertNotEqual(game_history1, game_history2)
        game_history2.add_last_iteration(['a', 'b'], [1, 2])
        self.assertEqual(game_history1, game_history2)

    def test_len(self):
        game_history = GameHistory()
        self.assertEqual(len(game_history), 0)
        game_history.add_last_iteration(['a', 'b'], [1, 2])
        self.assertEqual(len(game_history), 1)
        game_history.add_last_iteration(['a', 'b'], [1, 2])
        self.assertEqual(len(game_history), 2)
