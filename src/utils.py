import datetime as dt
import json
import logging
import math
import os
import re
import shutil
import time
import warnings
from pathlib import Path
import scipy.stats as st

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


def compute_average_vector(vectors, start=0, end=None):
    n_vectors = len(vectors)
    if n_vectors == 0:
        return []
    if end is None:
        end = len(vectors[0])
    average_vector = [sum([vectors[i][j] for i in range(n_vectors)]) / n_vectors for j in range(start, end)]
    return average_vector


def compute_variance_vector(vectors, start=0, end=None):
    n_vectors = len(vectors)
    if n_vectors == 0:
        return []
    if end is None:
        end = len(vectors[0])
    average_vector = compute_average_vector(vectors, start, end)
    variance_vector = [sum([(vectors[i][j] - average_vector[j]) ** 2 for i in range(n_vectors)]) / n_vectors for j in range(start, end)]
    return variance_vector


def compute_confidence_interval_vectors(vectors, start=0, end=None, confidence=0.95):
    n_vectors = len(vectors)
    is_normal = n_vectors > 30
    if n_vectors == 0:
        return []
    if end is None:
        end = len(vectors[0])
    confidence_interval_vector = []
    for iteration in range(start, end):
        iteration_data = [vectors[i][iteration] for i in range(n_vectors)]
        if is_normal:
            ci = st.norm.interval(confidence=confidence, loc=np.mean(iteration_data), scale=st.sem(iteration_data))
        else:
            ci = st.t.interval(confidence=confidence, df=len(iteration_data) - 1, loc=np.mean(iteration_data), scale=st.sem(iteration_data))
        confidence_interval_vector.append(ci)
    lower_bounds = [ci[0] for ci in confidence_interval_vector]
    upper_bounds = [ci[1] for ci in confidence_interval_vector]
    return lower_bounds, upper_bounds


def compute_std_dev_vector(vectors, start=0, end=None):
    variance_vector = compute_variance_vector(vectors, start, end)
    std_dev_vector = [math.sqrt(variance_vector[j]) for j in range(len(variance_vector))]
    return std_dev_vector


def compute_estimators_of_ts(ts):
    if not ts:
        return 0, 0, 0
    means, variances, std_devs = compute_cumulative_estimators_of_ts(ts)
    return means[-1], variances[-1], std_devs[-1]


def compute_cumulative_estimators_of_ts(ts):
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


def extract_infixes(extraction_timestamp, file_name, subdir=None, max_infix=None):
    dir_path = OUT_BASE_PATH / str(extraction_timestamp)
    if subdir is not None:
        dir_path = dir_path / subdir
    infixes = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_name in file_path.name:
            infix = file_path.name.split("_")[-1].split(".")[0]
            if max_infix is None or int(infix) <= max_infix:
                infixes.append(infix)
    return infixes


def extract_infixes_(dir_path, file_name, max_infix=None):
    infixes = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_name in file_path.name:
            infix = file_path.name.split("_")[-1].split(".")[0]
            if max_infix is None or int(infix) <= max_infix:
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


def convert_time_string_to_seconds(time_str):
    total_seconds = None
    # Extract hours, minutes, and seconds using regular expressions
    match_hour = re.match(r'(\d+)h(\d+)m(\d+\.\d+)s', time_str)
    match_minute = re.match(r'(\d+)m(\d+\.\d+)s', time_str)
    match_second = re.match(r'(\d+\.\d+)s', time_str)
    if match_hour:
        hours = int(match_hour.group(1))
        minutes = int(match_hour.group(2))
        seconds = float(match_hour.group(3))

        # Convert all to seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds

    elif match_minute:
        minutes = int(match_minute.group(1))
        seconds = float(match_minute.group(2))

        # Convert all to seconds
        total_seconds = minutes * 60 + seconds
    elif match_second:
        total_seconds = float(match_second.group(1))

    if total_seconds is not None:
        return total_seconds
    else:
        raise ValueError("Invalid time format")


def shutdown_run():
    # Remove empty directory
    logging.shutdown()
    shutil.rmtree(out_path)
