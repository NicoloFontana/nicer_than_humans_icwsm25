import unittest
import warnings
from unittest.mock import Mock

from src.checkers.checker import Checker


class TestChecker(unittest.TestCase):

    def setUp(self):
        self.client = Mock()
        self.questions = [
            "single answer",
            "unordered list answer",
            "ordered list answer",
        ]

    def test_get_answer_from_llm(self):
        checker = Checker("checker", self.questions, 0)
        checker.inference_client = self.client
        self.client.text_generation.side_effect = Mock(side_effect=Exception('Generic error'))
        question = self.questions[0]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question, True)
            warnings.simplefilter("ignore")
            self.assertEqual("", checker.get_answer_from_llm("prompt", question,  True))
            self.client.text_generation.side_effect = None
            self.client.text_generation.return_value = None
            warnings.simplefilter("error")
            self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question,  True)
            warnings.simplefilter("ignore")
            self.assertEqual("", checker.get_answer_from_llm("prompt", question,  True))
            self.client.text_generation.return_value = "I'm not a valid json"
            warnings.simplefilter("error")
            self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question,  True)
            warnings.simplefilter("ignore")
            self.assertEqual("", checker.get_answer_from_llm("prompt", question,  True))
            self.client.text_generation.return_value = '{"asnwer": "valid json but no answer key"}'
            warnings.simplefilter("error")
            self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question,  True)
            warnings.simplefilter("ignore")
            self.assertEqual("", checker.get_answer_from_llm("prompt", question,  True))
            self.client.text_generation.return_value = '{"answer": 1}'
            self.assertEqual("1", checker.get_answer_from_llm("prompt", question,  True))

    def test_check_answer(self):
        checker = Checker("checker", self.questions, 0)
        checker.inference_client = self.client
        # Single answer
        question = self.questions[0]
        idx = 0
        correct_answer = "1"
        llm_answer = "1"
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual("1", checker.answers[question][idx]["correct_answer"])
        self.assertEqual("1", checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(1, checker.checks[question])
        idx += 1
        llm_answer = "2"
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual("1", checker.answers[question][idx]["correct_answer"])
        self.assertEqual("2", checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(2, checker.checks[question])
        # Ordered list
        question = self.questions[1]
        idx = 0
        correct_answer = (1, 2)
        llm_answer = (1, 2)
        checker.check_answer(llm_answer, correct_answer, question, is_list=True)
        self.assertEqual((1, 2), checker.answers[question][idx]["correct_answer"])
        self.assertEqual((1, 2), checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(1, checker.checks[question])
        idx += 1
        llm_answer = (2, 1)
        checker.check_answer(llm_answer, correct_answer, question, is_list=True)
        self.assertEqual((1, 2), checker.answers[question][idx]["correct_answer"])
        self.assertEqual((2, 1), checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(2, checker.checks[question])
        idx += 1
        llm_answer = (1, 3)
        checker.check_answer(llm_answer, correct_answer, question, is_list=True)
        self.assertEqual((1, 2), checker.answers[question][idx]["correct_answer"])
        self.assertEqual((1, 3), checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(3, checker.checks[question])
        idx += 1
        llm_answer = ("1", "2")
        checker.check_answer(llm_answer, correct_answer, question, is_list=True)
        self.assertEqual((1, 2), checker.answers[question][idx]["correct_answer"])
        self.assertEqual(("1", "2"), checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(4, checker.checks[question])
        # Unordered list
        question = self.questions[2]
        idx = 0
        correct_answer = (1, 2)
        llm_answer = (1, 2)
        checker.check_answer(llm_answer, correct_answer, question, is_list=True, is_ordered=True)
        self.assertEqual((1, 2), checker.answers[question][idx]["correct_answer"])
        self.assertEqual((1, 2), checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(1, checker.checks[question])
        idx += 1
        llm_answer = (2, 1)
        checker.check_answer(llm_answer, correct_answer, question, is_list=True, is_ordered=True)
        self.assertEqual((1, 2), checker.answers[question][idx]["correct_answer"])
        self.assertEqual((2, 1), checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(2, checker.checks[question])
        idx += 1
        llm_answer = (1, 3)
        checker.check_answer(llm_answer, correct_answer, question, is_list=True, is_ordered=True)
        self.assertEqual((1, 2), checker.answers[question][idx]["correct_answer"])
        self.assertEqual((1, 3), checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(3, checker.checks[question])
