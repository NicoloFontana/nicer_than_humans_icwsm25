import unittest
import warnings
from unittest.mock import Mock

from src.games.two_players_pd import TwoPlayersPD
from src.player import Player
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy


class TestOneVsOnePDLlmStrategy(unittest.TestCase):

    def setUp(self):
        self.client = Mock()

    def test_init(self):
        game = TwoPlayersPD(Player("Alice"), Player("Bob"))
        self.assertRaises(Exception, OneVsOnePDLlmStrategy, game, "Alice", "Bob", model="I'm not a model")
        self.assertRaises(Exception, OneVsOnePDLlmStrategy, game, "Alice", "Bob", token="I'm not a token")
        self.assertRaises(TypeError, OneVsOnePDLlmStrategy, "I'm not a game", "Alice", "Bob")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            game = TwoPlayersPD(Player("1"), Player("Bob"))
            self.assertRaises(Warning, OneVsOnePDLlmStrategy, game, 1, "Bob")
            game = TwoPlayersPD(Player("Alice"), Player("1"))
            self.assertRaises(Warning, OneVsOnePDLlmStrategy, game, "Alice", 1)
            warnings.simplefilter("ignore")
            game = TwoPlayersPD(Player("1"), Player("Bob"))
            self.assertEqual(OneVsOnePDLlmStrategy(game, 1, "Bob").player_name, "1")
            game = TwoPlayersPD(Player("Alice"), Player("1"))
            self.assertEqual(OneVsOnePDLlmStrategy(game, "Alice", 1).opponent_name, "1")
            game = TwoPlayersPD(Player("Alice"), Player("Bob"))
        self.assertRaises(ValueError, OneVsOnePDLlmStrategy, game, "Charlie", "Bob")
        self.assertRaises(ValueError, OneVsOnePDLlmStrategy, game, "Alice", "Charlie")

    def test_play(self):
        game = TwoPlayersPD(Player("Alice"), Player("Bob"))
        strategy = OneVsOnePDLlmStrategy(game, "Alice", "Bob", update_client=False)
        strategy.client = self.client
        self.client.text_generation.return_value = None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, strategy.play)
            warnings.simplefilter("ignore")
            self.assertEqual(strategy.play(), 0)
        self.client.text_generation.return_value = '{I\'m not a valid JSON'
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, strategy.play)
            warnings.simplefilter("ignore")
            self.assertEqual(strategy.play(), 0)
        self.client.text_generation.return_value = '{"choice": "Cooperate"}'
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, strategy.play)
            warnings.simplefilter("ignore")
            self.assertEqual(strategy.play(), 0)
        self.client.text_generation.return_value = '{"action": "Cooperate"}'
        self.assertEqual(strategy.play(), 1)
        self.client.text_generation.return_value = '{"action": "Defect"}'
        self.assertEqual(strategy.play(), 0)
        self.client.text_generation.return_value = '{"action": "I\'m not a valid action"}'
        self.assertEqual(strategy.play(), 0)
