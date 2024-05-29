import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from src.utils import OUT_BASE_PATH

final_timestamps = ["20240528145105"]#["20240315190502", "20240318120135", "20240318120155"]
# initial_timestamps = ["20240403094750"]
checkers = ["aggregation", "rule", "time"]
confidence = 0.95


def old_to_new_label(old_label):
    # time
    if old_label == "current_round":
        return "round"
    if old_label == "action_A" or old_label == "action_B" or old_label == "action_you_at_round" or old_label == "action_opponent_at_round":
        return "action_i"
    if old_label == "points_A" or old_label == "points_B" or old_label == "points_you_at_round" or old_label == "points_opponent_at_round":
        return "points_i"
    # rule
    if old_label == "min_payoff" or old_label == "max_payoff":
        return "min_max"
    if old_label == "allowed_actions":
        return "actions"
    if old_label == "round_payoff_A" or old_label == "round_payoff_B" or old_label == "payoff_you_given_combo" or old_label == "payoff_opponent_given_combo":
        return "payoff"
    if old_label == "exists_combo" or old_label == "exists_combo_for_payoff_you":
        return "exist_combo"
    if old_label == "combo_for_payoff_A" or old_label == "which_combo_for_payoff_you":
        return "combo"
    # aggreg
    if old_label == "#actions_A" or old_label == "#actions_B" or old_label == "#actions_you" or old_label == "#actions_opponent":
        return "#actions"
    if old_label == "total_payoff_A" or old_label == "total_payoff_B" or old_label == "total_payoff_you" or old_label == "total_payoff_opponent":
        return "#points"


def new_label_to_full_question(new_label):
    if new_label == "round":
        return "Which is the current round of the game?"
    if new_label == "action_i":
        return "Which action did player X play in round i?"
    if new_label == "points_i":
        return "How many points did player X collect in round i?"
    if new_label == "min_max":
        return "What is the lowest/highest payoff player A can get in a single round?"
    if new_label == "actions":
        return "Which actions is player A allowed to play?"
    if new_label == "payoff":
        return "Which is player X's payoff in a single round if X plays p and Y plays q?"
    if new_label == "exist_combo":
        return "Does exist a combination of actions that gives a player a payoff of K in a single round?"
    if new_label == "combo":
        return "Which combination of actions gives a payoff of K to A in a single round?"
    if new_label == "#actions":
        return "How many times did player X choose p?"
    if new_label == "#points":
        return "What is player X's current total payoff?"

def old_to_new_checker_name(old_checker_name):
    if old_checker_name == "aggregation":
        return "state"
    if old_checker_name == "rule":
        return "rules"
    if old_checker_name == "time":
        return "time"


csv_dir_path = Path("csv_files_for_plots") / "accuracy-question"
csv_dir_path.mkdir(parents=True, exist_ok=True)
checkers_csv = []

for checker_name in checkers:
    new_checker_name = old_to_new_checker_name(checker_name)
    data = {}
    for extraction_timestamp in final_timestamps:
        print(extraction_timestamp)
        checker_path = OUT_BASE_PATH / extraction_timestamp / f"{checker_name}_checker"
        file_name = f"{checker_name}_checker_complete_answers.json"
        with open(checker_path / file_name, "r") as f:
            complete_answers = json.load(f)

        for old_label in complete_answers.keys():
            if "combo" in old_label:
                continue
            new_label = old_to_new_label(old_label)
            if new_label not in data.keys():
                data[new_label] = {}
                data[new_label][checker_name] = checker_name
                data[new_label]["full_question"] = new_label_to_full_question(new_label)
                data[new_label]["final_values"] = []
                data[new_label]["final_mean"] = None
                data[new_label]["final_lb"] = None
                data[new_label]["final_ub"] = None
                # data[new_label]["initial_values"] = []
                # data[new_label]["initial_mean"] = None
                # data[new_label]["initial_lb"] = None
                # data[new_label]["initial_ub"] = None
            for idx in complete_answers[old_label].keys():
                data[new_label]["final_values"].append(complete_answers[old_label][idx]["answer"]["is_correct"])
    # for extraction_timestamp in initial_timestamps:
    #     print(extraction_timestamp)
    #     checker_path = OUT_BASE_PATH / "main_runs" / extraction_timestamp / f"{checker_name}_checker"
    #     file_name = f"{checker_name}_checker_complete_answers.json"
    #     with open(checker_path / file_name, "r") as f:
    #         complete_answers = json.load(f)
    #
    #     for old_label in complete_answers.keys():
    #         if "combo_for" in old_label:
    #             continue
    #         new_label = old_to_new_label(old_label)
    #         for idx in complete_answers[old_label].keys():
    #             data[new_label]["initial_values"].append(complete_answers[old_label][idx]["answer"]["is_correct"])
    for new_label in data.keys():
        data[new_label]["final_mean"] = np.mean(data[new_label]["final_values"])
        ci = st.norm.interval(confidence, loc=data[new_label]["final_mean"], scale=st.sem(data[new_label]["final_values"]))
        data[new_label]["final_lb"] = ci[0] if not np.isnan(ci[0]) else data[new_label]["final_mean"]
        data[new_label]["final_ub"] = ci[1] if not np.isnan(ci[1]) else data[new_label]["final_mean"]

        # data[new_label]["initial_mean"] = np.mean(data[new_label]["initial_values"])
        # ci = st.norm.interval(confidence, loc=data[new_label]["initial_mean"], scale=st.sem(data[new_label]["initial_values"]))
        # data[new_label]["initial_lb"] = ci[0] if not np.isnan(ci[0]) else data[new_label]["initial_mean"]
        # data[new_label]["initial_ub"] = ci[1] if not np.isnan(ci[1]) else data[new_label]["initial_mean"]

    for new_label in data.keys():
        question = {}
        question["type"] = new_checker_name
        question["full_question"] = data[new_label]["full_question"]
        question["label"] = new_label
        # question["initial_accuracy"] = data[new_label]["initial_mean"]
        # question["initial_ci_lb"] = data[new_label]["initial_lb"]
        # question["initial_ci_ub"] = data[new_label]["initial_ub"]
        question["final_accuracy"] = data[new_label]["final_mean"]
        question["final_ci_lb"] = data[new_label]["final_lb"]
        question["final_ci_ub"] = data[new_label]["final_ub"]
        checkers_csv.append(question)
df = pd.DataFrame(checkers_csv)
df.to_csv(csv_dir_path / f"all_checkers_final_results-llama3.csv")

from src.utils import shutdown_run

shutdown_run()
