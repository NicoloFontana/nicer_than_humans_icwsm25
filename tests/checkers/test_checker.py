# import os
# import unittest
# import warnings
# from unittest.mock import Mock
#
# from src.checkers.checker import Checker
#
#
# class TestChecker(unittest.TestCase):
#
#     def setUp(self):
#         self.client = Mock()
#         self.questions = [
#             "single answer",
#             "set answer",
#             "list answer",
#             "weighted answer",
#         ]
#         self.questions_labels = [
#             "single",
#             "set",
#             "list",
#             "weighted",
#         ]
#
#     def test_get_answer_from_llm(self):
#         checker = Checker("checker", self.questions, self.questions_labels, 0)
#         checker.inference_client = self.client
#         question = self.questions[0]
#         with warnings.catch_warnings():
#             # Catch inference client errors
#             self.client.text_generation.side_effect = Mock(side_effect=Exception('Generic error'))
#             warnings.simplefilter("error")
#             self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question, True)
#             warnings.simplefilter("ignore")
#             self.assertEqual("", checker.get_answer_from_llm("prompt", question, True))
#             self.client.text_generation.side_effect = None
#             # Convert None to empty string
#             string = "None"
#             self.client.text_generation.return_value = string
#             warnings.simplefilter("error")
#             self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question, True)
#             warnings.simplefilter("ignore")
#             self.assertEqual(string, checker.get_answer_from_llm("prompt", question, True))
#             # Convert invalid json to empty string
#             string = "I'm not a valid json"
#             self.client.text_generation.return_value = string
#             warnings.simplefilter("error")
#             self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question, True)
#             warnings.simplefilter("ignore")
#             self.assertEqual(string, checker.get_answer_from_llm("prompt", question, True))
#             # Convert valid json but no answer key to empty string
#             string = '{"not_answer": "valid json but no answer key"}'
#             self.client.text_generation.return_value = string
#             warnings.simplefilter("error")
#             self.assertRaises(Warning, checker.get_answer_from_llm, "prompt", question, True)
#             warnings.simplefilter("ignore")
#             self.assertEqual(string, checker.get_answer_from_llm("prompt", question, True))
#             # Convert non-string answer to string equivalent
#             self.client.text_generation.return_value = '{"answer": 1}'
#             self.assertEqual("1", checker.get_answer_from_llm("prompt", question, True))
#
#         # Remove empty directory
#         os.rmdir(checker.dir_path)
#
#     def test_check_answer(self):
#         checker = Checker("checker", self.questions, self.questions_labels, 0)
#         checker.inference_client = self.client
#
#         # Single answer
#         question = self.questions[0]
#         idx = 0
#         correct_answer = "1"
#         # Correct single numeric answer
#         llm_answer = "1"
#         checker.check_answer(llm_answer, correct_answer, question)
#         # Answers storing
#         self.assertEqual(correct_answer,
#                          checker.questions_results[question][checker.answer_str][idx]["correct_answer"])
#         self.assertEqual(llm_answer, checker.questions_results[question][checker.answer_str][idx]["llm_answer"])
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Incorrect single numeric answer
#         llm_answer = "2"
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(False, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Correct single string answer
#         correct_answer = "Cooperate"
#         llm_answer = "Cooperate"
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Case insensitivity for string answer
#         llm_answer = "cooperate"
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#
#         # Set
#         question = self.questions[1]
#         idx = 0
#         correct_answer = {1, 2}
#         # Correct set of numeric answers
#         llm_answer = {1, 2}
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Incorrect set of numeric answers
#         llm_answer = {1, 3}
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(False, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Set of strings of integers is not the same as set of integers
#         llm_answer = {"1", "2"}
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(False, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         correct_answer = {"Cooperate", "Defect"}
#         # Correct set of string answers
#         llm_answer = {"Cooperate", "Defect"}
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Set order and case insensitivity
#         llm_answer = {"defect", "cooperate"}
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#
#         # List
#         question = self.questions[2]
#         idx = 0
#         correct_answer = [1, 2]
#         # Correct list of numeric answers
#         llm_answer = [1, 2]
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Incorrect list of numeric answers
#         llm_answer = [1, 3]
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(False, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         correct_answer = ["Cooperate", "Defect"]
#         # Correct list of string answers
#         llm_answer = ["Cooperate", "Defect"]
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Case insensitivity
#         llm_answer = ["cooperate", "defect"]
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(True, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#         idx += 1
#         # Order sensitivity
#         llm_answer = ["defect", "cooperate"]
#         checker.check_answer(llm_answer, correct_answer, question)
#         self.assertEqual(False, checker.questions_results[question][checker.answer_str][idx]["is_correct"])
#
#         # Remove empty directory
#         os.rmdir(checker.dir_path)
#     #
#     #
#     # def test_update_aggregates_for_question(self):
#     #     checker = Checker("checker", self.questions, self.questions_labels, 0)
#     #     question = self.questions[3]
#     #     is_correct = True
#     #     weight = 1
#     #     checker.update_aggregates_for_question(question, is_correct, weight)
#     #     self.assertEqual(1, checker.questions_results[question][checker.positives_str])
#     #     self.assertEqual(1, checker.questions_results[question][checker.total_str])
#     #     self.assertEqual(1, checker.questions_results[question][checker.sample_mean_str])
#     #     self.assertEqual(0, checker.questions_results[question][checker.squared_diffs_sum_str])
#     #     self.assertEqual(0, checker.questions_results[question][checker.sample_variance_str])
#     #     is_correct = False
#     #     checker.update_aggregates_for_question(question, is_correct, weight)
#     #     self.assertEqual(1, checker.questions_results[question][checker.positives_str])
#     #     self.assertEqual(2, checker.questions_results[question][checker.total_str])
#     #     self.assertEqual(0.5, checker.questions_results[question][checker.sample_mean_str])
#     #     self.assertEqual(0, checker.questions_results[question][checker.squared_diffs_sum_str])
#     #     self.assertEqual(0.5, checker.questions_results[question][checker.sample_variance_str])
#     #     weight = 0.5
#     #     is_correct = True
#     #     checker.update_aggregates_for_question(question, is_correct, weight)
#     #     self.assertEqual(1.5, checker.questions_results[question][checker.positives_str])
#     #
#     #
#     #
#     #     # Weighted
#     #     question = self.questions[3]
#     #     correct_answer = 1
#     #     llm_answer = 1
#     #     checker.check_answer(llm_answer, correct_answer, question, weight=0.5)
#     #     self.assertEqual(0.5, checker.questions_results[question][checker.positives_str])
#     #     self.assertEqual(0.5, checker.questions_results[question][checker.total_str])
#     #     llm_answer = 0
#     #     checker.check_answer(llm_answer, correct_answer, question, weight=0.5)
#     #     self.assertEqual(0.5, checker.questions_results[question][checker.positives_str])
#     #     self.assertEqual(1, checker.questions_results[question][checker.total_str])
#     #     llm_answer = 1
#     #     checker.check_answer(llm_answer, correct_answer, question, weight=0.25)
#     #     self.assertEqual(0.75, checker.questions_results[question][checker.positives_str])
#     #     self.assertEqual(1.25, checker.questions_results[question][checker.total_str])
