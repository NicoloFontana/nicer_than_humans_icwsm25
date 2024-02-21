import unittest
from unittest.mock import Mock

from src.checkers.rule_checker import RuleChecker
from src.games.two_players_pd_utils import two_players_pd_payoff


class TestRuleChecker(unittest.TestCase):

    def setUp(self):
        self.client = Mock()
        self.questions = [
            "single answer",
            "unordered list answer",
            "ordered list answer",
        ]

    # check_payoff_bounds is not tested because it works the same way as TimeChecker.check_current_round

    def test_check_allowed_actions(self):
        checker = RuleChecker(0)
        checker.inference_client = self.client
        checker.system_prompt = "system prompt"
        self.client.text_generation.return_value = '{"answer": ["Cooperate", "Defect"]}'
        checker.check_allowed_actions({1, 0})
        self.assertEqual(1, checker.scores[checker.questions[2]])
        self.assertEqual(1, checker.checks[checker.questions[2]])
        self.client.text_generation.return_value = '{"answer": ["Defect", "Cooperate"]}'
        checker.check_allowed_actions({1, 0})
        self.assertEqual(2, checker.scores[checker.questions[2]])
        self.assertEqual(2, checker.checks[checker.questions[2]])
        self.client.text_generation.return_value = '{"answer": ["Cooperate", "Not a valid action"]}'
        checker.check_allowed_actions({1, 0})
        self.assertEqual(2, checker.scores[checker.questions[2]])
        self.assertEqual(3, checker.checks[checker.questions[2]])

    # check_payoff_of_combo is not tested because it works the same way as TimeChecker.check_current_round

    def test_check_exist_combo_for_payoff(self):
        checker = RuleChecker(0)
        checker.inference_client = self.client
        checker.system_prompt = "system prompt"
        self.client.text_generation.return_value = '{"answer": "Yes"}'
        checker.check_exists_combo_for_payoff({1, 0}, two_players_pd_payoff, 0)
        self.assertEqual(1, checker.scores[checker.questions[5]])
        self.assertEqual(1, checker.checks[checker.questions[5]])
        self.client.text_generation.return_value = '{"answer": "No"}'
        checker.check_exists_combo_for_payoff({1, 0}, two_players_pd_payoff, 0)
        self.assertEqual(1, checker.scores[checker.questions[5]])
        self.assertEqual(2, checker.checks[checker.questions[5]])
        self.client.text_generation.return_value = '{"answer": "Yes. It exists."}'
        checker.check_exists_combo_for_payoff({1, 0}, two_players_pd_payoff, 0)
        self.assertEqual(2, checker.scores[checker.questions[5]])
        self.assertEqual(3, checker.checks[checker.questions[5]])
        self.client.text_generation.return_value = '{"answer": "I think yes."}'
        checker.check_exists_combo_for_payoff({1, 0}, two_players_pd_payoff, 0)
        self.assertEqual(3, checker.scores[checker.questions[5]])
        self.assertEqual(4, checker.checks[checker.questions[5]])

    def test_check_combo_for_payoff(self):
        checker = RuleChecker(0)
        checker.inference_client = self.client
        checker.system_prompt = "system prompt"
        self.client.text_generation.return_value = '{"answer": ["Cooperate", "Defect"]}'
        checker.check_combo_for_payoff({1, 0}, two_players_pd_payoff, 0)
        self.assertEqual(1, checker.scores[checker.questions[6]])
        self.assertEqual(1, checker.checks[checker.questions[6]])
        self.client.text_generation.return_value = '{"answer": ["Defect", "Cooperate"]}'
        checker.check_combo_for_payoff({1, 0}, two_players_pd_payoff, 0)
        self.assertEqual(1, checker.scores[checker.questions[6]])
        self.assertEqual(2, checker.checks[checker.questions[6]])
