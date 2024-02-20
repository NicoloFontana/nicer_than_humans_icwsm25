import unittest
import warnings

from src.utils import find_json_object


class TestUtils(unittest.TestCase):

    def test_find_json_object(self):
        string = '{"action": "Cooperate", "reason": "I want to start with a cooperative move."}'
        self.assertEqual(find_json_object(string), {"action": "Cooperate", "reason": "I want to start with a cooperative move."})
        string = '{"answer": ["Cooperate", "Cooperate"]} Alternatively: {"answer": ["Defect", "Cooperate"]}'
        self.assertEqual(find_json_object(string), {"answer": ["Cooperate", "Cooperate"]})
        with warnings.catch_warnings():
            string = '{"answer": "Cooperate"'
            warnings.simplefilter("error")
            self.assertRaises(Warning, find_json_object, string)
            warnings.simplefilter("ignore")
            self.assertEqual(find_json_object(string), None)
            string = '"answer": "Cooperate"}'
            warnings.simplefilter("error")
            self.assertRaises(Warning, find_json_object, string)
            warnings.simplefilter("ignore")
            self.assertEqual(find_json_object(string), None)
        string = '{this is not a valid json}'
        self.assertRaises(ValueError, find_json_object, string)
