# import os
# import unittest
# from unittest.mock import Mock
#
# from src.checkers.time_checker import TimeChecker
#
#
# class TestTimeChecker(unittest.TestCase):
#     """
#     Test the TimeChecker class.\n
#     The AggregationChecker class works the same way.
#     """
#
#     def setUp(self):
#         self.client = Mock()
#         self.questions = [
#             "single answer",
#             "unordered list answer",
#             "ordered list answer",
#         ]
#     #
#     # def test_check_current_round(self):
#     #     # check_points_collected works the same way
#     #
#     #     checker = TimeChecker(0)
#     #     checker.inference_client = self.client
#     #     checker.system_prompt = "system prompt"
#     #     # Correct answer
#     #     self.client.text_generation.return_value = '{"answer": 1}'
#     #     checker.check_current_round(1)
#     #     self.assertEqual(1, checker.questions_results[checker.questions[0]][checker.positives_str])
#     #     self.assertEqual(1, checker.questions_results[checker.questions[0]][checker.n_samples_str])
#     #     # Incorrect answer
#     #     self.client.text_generation.return_value = '{"answer": 2}'
#     #     checker.check_current_round(1)
#     #     self.assertEqual(1, checker.questions_results[checker.questions[0]][checker.positives_str])
#     #     self.assertEqual(2, checker.questions_results[checker.questions[0]][checker.n_samples_str])
#     #
#     #     # Remove empty directory
#     #     os.rmdir(checker.out_path)
#     #
#     # def test_check_action_played(self):
#     #     checker = TimeChecker(0)
#     #     checker.inference_client = self.client
#     #     checker.system_prompt = "system prompt"
#     #     action_space = {1, 0}
#     #     # Correct answer
#     #     self.client.text_generation.return_value = '{"answer": "Cooperate"}'
#     #     checker.check_action_played(True, 1, 1, action_space)
#     #     self.assertEqual(1, checker.questions_results[checker.questions[1]][checker.positives_str])
#     #     self.assertEqual(1, checker.questions_results[checker.questions[1]][checker.n_samples_str])
#     #     # Incorrect answer
#     #     self.client.text_generation.return_value = '{"answer": "Defect"}'
#     #     checker.check_action_played(True, 1, 1, action_space)
#     #     self.assertEqual(1, checker.questions_results[checker.questions[1]][checker.positives_str])
#     #     self.assertEqual(2, checker.questions_results[checker.questions[1]][checker.n_samples_str])
#     #
#     #     # Remove empty directory
#     #     os.rmdir(checker.out_path)
