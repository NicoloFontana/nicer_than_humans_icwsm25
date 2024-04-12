import datetime as dt
import json
import logging
import math
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np

dt_start_time = dt.datetime.now()
start_time = time.mktime(dt_start_time.timetuple())
timestamp = dt_start_time.strftime("%Y%m%d%H%M%S")
OUT_BASE_PATH = Path("out")
out_path = OUT_BASE_PATH / str(timestamp)
os.makedirs(out_path, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", filename=out_path / f"{timestamp}.log")
log = logging.getLogger()


def find_json_object(string):
    string = re.sub(r"([a-zA-Z])'([a-zA-Z])", r'\1,\2', string)
    string = re.sub(r"'", '"', string)
    # string = str(string).replace("'", '"')
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


def compute_average_vector(vectors):
    n_vectors = len(vectors)
    if n_vectors == 0:
        return []
    n_elements = len(vectors[0])
    average_vector = [sum([vectors[i][j] for i in range(n_vectors)]) / n_vectors for j in range(n_elements)]
    return average_vector


def compute_estimators_of_ts(ts):
    sample_means = []
    sample_variances = []
    sample_std_devs = []
    for i in range(len(ts)):  # i is already (#samples - 1) => i + 1 is #samples
        sample_mean = sum(ts[:i + 1]) / (i + 1)  # sample mean of #samples
        sample_means.append(sample_mean)
        sum_squared_diffs = sum([(el - sample_mean) ** 2 for el in ts[:i + 1]])  # sum of squared differences between last #samples and current mean
        sample_variance = sum_squared_diffs / i if i > 0 else 0  # unbiased sample variance of #samples
        sample_variances.append(sample_variance)
        sample_std_devs.append(sample_variance ** 0.5)  # unbiased sample standard deviation of #samples
    return sample_means, sample_variances, sample_std_devs


def extract_infixes(extraction_timestamp, file_name, subdir=None):
    dir_path = OUT_BASE_PATH / str(extraction_timestamp)
    if subdir is not None:
        dir_path = dir_path / subdir
    infixes = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_name in file_path.name:
            infix = file_path.name.split("_")[-1].split(".")[0]
            infixes.append(infix)
    return infixes


def convert_matrix_to_percentage(matrix):
    """
    Converts the specified matrix to a percentage matrix.
    :param matrix: matrix to be converted
    :return: percentage matrix
    """
    percentage_matrix = np.zeros(matrix.shape)
    total_sum = sum([sum(row) for row in matrix])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            percentage_matrix[i, j] = matrix[i, j] / total_sum if total_sum != 0 else 0
    return percentage_matrix
