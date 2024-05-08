import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from src.utils import OUT_BASE_PATH

timestamps = ["20240315190502", "20240318120135", "20240318120155"]
checkers = ["aggregation", "rule", "time"]
confidence = 0.95


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
