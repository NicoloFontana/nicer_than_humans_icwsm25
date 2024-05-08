import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from src.utils import OUT_BASE_PATH
#
# aggregation_complete_answers_1 = {}
# rule_complete_answers_1 = {}
# time_complete_answers_1 = {}
#
#
# aggregation_complete_answers_2 = {}
# rule_complete_answers_2 = {}
# time_complete_answers_2 = {}
#
#
# aggregation_complete_answers_3 = {}
# rule_complete_answers_3 = {}
# time_complete_answers_3 = {}
#
timestamps = ["20240315190502", "20240318120135", "20240318120155"]
checkers = ["aggregation", "rule", "time"]
confidence = 0.95
#
# for extraction_timestamp in timestamps:
#     base_path = OUT_BASE_PATH / extraction_timestamp
#     for checker_name in checkers:
#         checker_path = base_path / checker_name
#         file_name = f"{checker_name}_checker_complete_answers.json"
#         complete_answers = json.load(checker_path / file_name)
#         for label in complete_answers.keys():
#             if checker_name == "aggregation":
#                 pass
#                 # aggregation_complete_answers
#                 # TODO does it make sense to merge all answers for a question?
#                 # for "which action played at X" I have 3 answers from the first round and 300 from the last round

# for each run:
    # load X_checker_complete_answers.json
    # merge by questions
    # save to X_checker_complete_answers.json
# for each question:
    # get results
    # compute mean
    # compute CI
# store results in .csv as questionX: text, label, accuracy, lb, ub


# create tmp .csv
# extraction_timestamp = "20240315190502"
csv_dir_path = Path("csv_files_for_plots") / "accuracy-question"
csv_dir_path.mkdir(parents=True, exist_ok=True)
for checker_name in checkers:
    is_first = True
    data = {}
    for extraction_timestamp in timestamps:
        print(extraction_timestamp)
        checker_path = OUT_BASE_PATH / extraction_timestamp / f"{checker_name}_checker"
        file_name = f"{checker_name}_checker_complete_answers.json"
        with open(checker_path / file_name, "r") as f:
            complete_answers = json.load(f)
        for label in complete_answers.keys():
            if is_first:
                data[label] = {}
                data[label][checker_name] = checker_name
                data[label]["full_question"] = complete_answers[label]["0"]["question"]
                data[label]["values"] = []
                data[label]["mean"] = None
                data[label]["lb"] = None
                data[label]["ub"] = None
            for idx in complete_answers[label].keys():
                data[label]["values"].append(complete_answers[label][idx]["answer"]["is_correct"])
        is_first = False
    for label in data.keys():
        data[label]["mean"] = np.mean(data[label]["values"])
        ci = st.norm.interval(confidence, loc=data[label]["mean"], scale=st.sem(data[label]["values"]))
        data[label]["lb"] = ci[0] if not np.isnan(ci[0]) else data[label]["mean"]
        data[label]["ub"] = ci[1] if not np.isnan(ci[1]) else data[label]["mean"]
    checker_csv = []
    fieldnames = ["full_question", "label", "accuracy", "ci_lb", "ci_ub"]
    for label in data.keys():
        question = {}
        question["full_question"] = data[label]["full_question"]
        question["label"] = label
        question["accuracy"] = data[label]["mean"]
        question["ci_lb"] = data[label]["lb"]
        question["ci_ub"] = data[label]["ub"]
        checker_csv.append(question)
    df = pd.DataFrame(checker_csv)
    df.to_csv(csv_dir_path / f"{checker_name}_checker_results.csv")


from src.utils import shutdown_run

shutdown_run()
