import unittest
import warnings

from src.utils import find_json_object, find_first_int, find_first_substring


class TestUtils(unittest.TestCase):

    def test_find_json_object(self):
        string = """   {"action": "Cooperate", "reason": "A should cooperate to maximize the total points collected in the long run, considering the previous rounds, outcomes."}\n\nIn this scenario"""  # Without escaping the comma was a single apex, it skips logically valid, but grammaticaly invalid jsons
        self.assertEqual({"action": "Cooperate", "reason": "A should cooperate to maximize the total points collected in the long run, considering the previous rounds, outcomes."},
                         find_json_object(string))
        string = '{"action": "Cooperate", "reason": "Since it\'s optimal"}'
        self.assertEqual({"action": "Cooperate", "reason": "Since it,s optimal"}, find_json_object(string))
        string = "{\"answer\": ['Defect', 'Cooperate']}"
        self.assertEqual({"answer": ["Defect", "Cooperate"]}, find_json_object(string))
        string = '{"action": "Cooperate", "reason": "I want to start with a cooperative move."}'
        self.assertEqual({"action": "Cooperate", "reason": "I want to start with a cooperative move."},
                         find_json_object(string))
        string = '{"answer": ["Cooperate", "Cooperate"]} Alternatively: {"answer": ["Defect", "Cooperate"]}'
        self.assertEqual({"answer": ["Cooperate", "Cooperate"]}, find_json_object(string))
        with warnings.catch_warnings():
            string = '{"answer": "Cooperate"'
            warnings.simplefilter("error")
            self.assertRaises(Warning, find_json_object, string)
            warnings.simplefilter("ignore")
            self.assertEqual(None, find_json_object(string))
            string = '"answer": "Cooperate"}'
            warnings.simplefilter("error")
            self.assertRaises(Warning, find_json_object, string)
            warnings.simplefilter("ignore")
            self.assertEqual(None, find_json_object(string))
            string = '{this is not a valid json}'
            warnings.simplefilter("error")
            self.assertRaises(Warning, find_json_object, string)
            warnings.simplefilter("ignore")
            self.assertEqual(None, find_json_object(string))

    def test_find_first_int(self):
        string = "I have 10 apples"
        self.assertEqual("10", find_first_int(string))
        string = "I have 10 apples and 5 oranges"
        self.assertEqual("10", find_first_int(string))
        string = "I have '10' apples"
        self.assertEqual("10", find_first_int(string))
        string = "I have 10.5 apples"
        self.assertEqual("10", find_first_int(string))

    def test_find_first_substring(self):
        substrings = ["Cooperate", "Defect"]
        string = "Cooperate"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "Defect"
        self.assertEqual("Defect", find_first_substring(string, substrings))
        string = "I will Cooperate"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "Cooperate would be the best option"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "I don't know what to do"
        self.assertEqual("", find_first_substring(string, substrings))
        string = '"Cooperate"'
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "'Cooperate'"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "Cooperate_"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "Cooperate."
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "_Cooperate"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = ",Cooperate"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
        string = "cooperate"
        self.assertEqual("Cooperate", find_first_substring(string, substrings))
