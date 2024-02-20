import json
import warnings

HF_API_TOKEN = "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"
CHECKS_OUT_BASE_PATH = "C:\\Users\\fonta\\PycharmProjects\\random_stuff\\masters_thesis_draft\\code\\out\\llm_checks\\"
MODEL = "meta-llama/Llama-2-70b-chat-hf"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7


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
