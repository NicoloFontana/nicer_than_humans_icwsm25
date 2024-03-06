import datetime as dt
import json
import logging
import os
import re
import warnings

from src.llm_utils import OUT_BASE_PATH


timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
dir_path = OUT_BASE_PATH / str(timestamp)
os.makedirs(dir_path, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=dir_path / f"{timestamp}.log")
log = logging.getLogger()

def find_json_object(string):
    if not isinstance(string, str):
        warnings.warn(f"Input string is not of type str: {type(string)}. Returning None.")
        return None
    start_json = False
    json_end = False
    match = 0
    probably_json_parsable = ""
    for char in string:
        if not start_json and char == "{":
            start_json = True
        if start_json and not json_end:
            probably_json_parsable += char
            if char == "{":
                match += 1
            if char == "}" and match > 0:
                match -= 1
                if match == 0:
                    json_end = True
    if json_end:
        try:
            return json.loads(probably_json_parsable)
        except json.JSONDecodeError:
            warnings.warn(f"Could not parse JSON: {probably_json_parsable}. Returning None.")
            return None
    else:
        warnings.warn(f"No JSON object found in the input string:\n {string}. Returning None.")
        return None


def find_first_int(string):
    re_findall = re.findall(r'\d+', string)
    if len(re_findall) > 0:
        return re_findall[0]
    return ""


def find_first_substring(string, substrings):
    insensitive_string = string.casefold()
    for sub in substrings:
        if insensitive_string.find(sub.casefold()) != -1:
            return sub
    return ""
