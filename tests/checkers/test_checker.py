import unittest
import warnings
from unittest.mock import Mock

from src.checkers.checker import Checker


class TestChecker(unittest.TestCase):

    def setUp(self):
        self.client = Mock()
        self.questions = [
            "single answer",
            "set answer",
            "list answer",
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
        idx += 1
        correct_answer = "Cooperate"
        llm_answer = "Cooperate"
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual("Cooperate", checker.answers[question][idx]["correct_answer"])
        self.assertEqual("Cooperate", checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(3, checker.checks[question])
        idx += 1
        llm_answer = "cooperate"
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual("Cooperate", checker.answers[question][idx]["correct_answer"])
        self.assertEqual("cooperate", checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(3, checker.scores[question])
        self.assertEqual(4, checker.checks[question])
        # Set
        question = self.questions[1]
        idx = 0
        correct_answer = {1, 2}
        llm_answer = {1, 2}
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual({1, 2}, checker.answers[question][idx]["correct_answer"])
        self.assertEqual({1, 2}, checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(1, checker.checks[question])
        idx += 1
        llm_answer = {2, 1}
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual({1, 2}, checker.answers[question][idx]["correct_answer"])
        self.assertEqual({2, 1}, checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(2, checker.checks[question])
        idx += 1
        llm_answer = {1, 3}
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual({1, 2}, checker.answers[question][idx]["correct_answer"])
        self.assertEqual({1, 3}, checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(3, checker.checks[question])
        idx += 1
        llm_answer = {"1", "2"}
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual({1, 2}, checker.answers[question][idx]["correct_answer"])
        self.assertEqual({"1", "2"}, checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(4, checker.checks[question])
        idx += 1
        correct_answer = {"Cooperate", "Defect"}
        llm_answer = {"Cooperate", "Defect"}
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual({"Cooperate", "Defect"}, checker.answers[question][idx]["correct_answer"])
        self.assertEqual({"Cooperate", "Defect"}, checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(3, checker.scores[question])
        self.assertEqual(5, checker.checks[question])
        idx += 1
        llm_answer = {"defect", "cooperate"}
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual({"Cooperate", "Defect"}, checker.answers[question][idx]["correct_answer"])
        self.assertEqual({"defect", "cooperate"}, checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(4, checker.scores[question])
        self.assertEqual(6, checker.checks[question])
        # List
        question = self.questions[2]
        idx = 0
        correct_answer = [1, 2]
        llm_answer = [1, 2]
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual([1, 2], checker.answers[question][idx]["correct_answer"])
        self.assertEqual([1, 2], checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(1, checker.checks[question])
        idx += 1
        llm_answer = [2, 1]
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual([1, 2], checker.answers[question][idx]["correct_answer"])
        self.assertEqual([2, 1], checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(2, checker.checks[question])
        idx += 1
        llm_answer = [1, 3]
        checker.check_answer(llm_answer, correct_answer, question)
        self.assertEqual([1, 2], checker.answers[question][idx]["correct_answer"])
        self.assertEqual([1, 3], checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(1, checker.scores[question])
        self.assertEqual(3, checker.checks[question])
        idx += 1
        right_answer = ["Cooperate", "Defect"]
        llm_answer = ["Cooperate", "Defect"]
        checker.check_answer(llm_answer, right_answer, question)
        self.assertEqual(["Cooperate", "Defect"], checker.answers[question][idx]["correct_answer"])
        self.assertEqual(["Cooperate", "Defect"], checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(2, checker.scores[question])
        self.assertEqual(4, checker.checks[question])
        idx += 1
        llm_answer = ["cooperate", "defect"]
        checker.check_answer(llm_answer, right_answer, question)
        self.assertEqual(["Cooperate", "Defect"], checker.answers[question][idx]["correct_answer"])
        self.assertEqual(["cooperate", "defect"], checker.answers[question][idx]["llm_answer"])
        self.assertEqual(True, checker.answers[question][idx]["is_correct"])
        self.assertEqual(3, checker.scores[question])
        self.assertEqual(5, checker.checks[question])
        idx += 1
        llm_answer = ["defect", "cooperate"]
        checker.check_answer(llm_answer, right_answer, question)
        self.assertEqual(["Cooperate", "Defect"], checker.answers[question][idx]["correct_answer"])
        self.assertEqual(["defect", "cooperate"], checker.answers[question][idx]["llm_answer"])
        self.assertEqual(False, checker.answers[question][idx]["is_correct"])
        self.assertEqual(3, checker.scores[question])
        self.assertEqual(6, checker.checks[question])
