import unittest
import warnings

from src.utils import find_json_object


class TestUtils(unittest.TestCase):

    def test_find_json_object(self):
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
