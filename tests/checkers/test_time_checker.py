import unittest
from unittest.mock import Mock

from src.checkers.time_checker import TimeChecker


class TestTimeChecker(unittest.TestCase):
    """
    Test the TimeChecker class.\n
    The AggregationChecker class works the same way.
    """

    def setUp(self):
        self.client = Mock()
        self.questions = [
            "single answer",
            "unordered list answer",
            "ordered list answer",
        ]

    def test_check_current_round(self):
        # check_points_collected works the same way

        checker = TimeChecker(0)
        checker.inference_client = self.client
        checker.system_prompt = "system prompt"
        self.client.text_generation.return_value = '{"answer": 1}'
        checker.check_current_round(1)
        self.assertEqual(1, checker.scores[checker.questions[0]])
        self.assertEqual(1, checker.checks[checker.questions[0]])
        self.client.text_generation.return_value = '{"answer": 2}'
        checker.check_current_round(1)
        self.assertEqual(1, checker.scores[checker.questions[0]])
        self.assertEqual(2, checker.checks[checker.questions[0]])
        self.client.text_generation.return_value = '{"answer": "1"}'
        checker.check_current_round(1)
        self.assertEqual(2, checker.scores[checker.questions[0]])
        self.assertEqual(3, checker.checks[checker.questions[0]])
        self.client.text_generation.return_value = '{"answer": "Round 1"}'
        checker.check_current_round(1)
        self.assertEqual(3, checker.scores[checker.questions[0]])
        self.assertEqual(4, checker.checks[checker.questions[0]])
        self.client.text_generation.return_value = '{"answer": "Round 1 has been played."}'
        checker.check_current_round(1)
        self.assertEqual(4, checker.scores[checker.questions[0]])
        self.assertEqual(5, checker.checks[checker.questions[0]])

    def test_check_action_played(self):
        checker = TimeChecker(0)
        checker.inference_client = self.client
        checker.system_prompt = "system prompt"
        action_space = {1, 0}
        self.client.text_generation.return_value = '{"answer": "Cooperate"}'
        checker.check_action_played(True, 1, 1, action_space)
        self.assertEqual(1, checker.scores[checker.questions[1]])
        self.assertEqual(1, checker.checks[checker.questions[1]])
        self.client.text_generation.return_value = '{"answer": "Defect"}'
        checker.check_action_played(True, 1, 1, action_space)
        self.assertEqual(1, checker.scores[checker.questions[1]])
        self.assertEqual(2, checker.checks[checker.questions[1]])
        self.client.text_generation.return_value = '{"answer": "The action played is Cooperate."}'
        checker.check_action_played(True, 1, 1, action_space)
        self.assertEqual(2, checker.scores[checker.questions[1]])
        self.assertEqual(3, checker.checks[checker.questions[1]])
        self.client.text_generation.return_value = '{"answer": "Cooperate would be the best option."}'
        checker.check_action_played(True, 1, 1, action_space)
        self.assertEqual(3, checker.scores[checker.questions[1]])
        self.assertEqual(4, checker.checks[checker.questions[1]])
