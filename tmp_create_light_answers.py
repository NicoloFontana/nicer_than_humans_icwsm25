import json
from pathlib import Path

from src.analysis.checkers_utils import old_to_new_label, new_label_to_full_question

# 20240311135514: kojima
# 20240311140702: zhou
# 20240311115510: no zscot
# 20240313114605: AvB
# 20240313160040: FvJ
# 20240314150845: BvA

# 20240305134839: v0.1
# 20240307183455: v0.2
# 20240307185046: v0.2.1
# 20240308162745: v0.2.2
run_dir = Path("relevant_runs_copies") / "others" / "20240308162745"
checkers_names = ["time_checker", "rule_checker", "aggregation_checker"]

# def convert_label(old_label):
#     if old_label == "round_payoff":
#         return "first_payoff"
#     if old_label == "opponent_round_payoff":
#         return "second_payoff_inverse"
#     if old_label == "opponent_round_payoff_inverse":
#         return "second_payoff"
#     return None
#
# def label_to_question(label):
#     if label == "first_payoff":
#         return "Which is the first player's payoff in a single round if the first player plays {} and the second player plays {}?"
#     if label == "second_payoff":
#         return "What was the second player's payoff in the round if the first player plays {} and the second player plays {}?"
#     if label == "second_payoff_inverse":
#         return "What was the second player's payoff in the round if the second player plays {} and the first player plays {}?"
#     return None



for checker_name in checkers_names:
    complete_answers_dir = run_dir / checker_name
    complete_answers_file = complete_answers_dir / f"{checker_name}_90.json"
    with open(complete_answers_file, "r") as complete_answers_file:
        complete_answers = json.load(complete_answers_file)
    light_complete_answers = {}
    for question in complete_answers.keys():
        label = old_to_new_label(complete_answers[question]["label"])
        if label is None:
            continue
        if label not in light_complete_answers.keys():
            light_single_complete_answer = {
                "question": new_label_to_full_question(label),
                "answers": [1] * int(complete_answers[question]["positives"]) + [0] * int(
                    complete_answers[question]["total"] - complete_answers[question]["positives"])
            }
            light_complete_answers[label] = light_single_complete_answer
        else:
            light_complete_answers[label]["answers"] += ([1] * int(complete_answers[question]["positives"]) + [0] * int(
                complete_answers[question]["total"] - complete_answers[question]["positives"]))
    light_complete_answers_file = run_dir / checker_name / f"light_answers.json"
    with open(light_complete_answers_file, "w") as light_complete_answers_file:
        json.dump(light_complete_answers, light_complete_answers_file, indent=4)
    print(f"Light complete answers of {checker_name} saved.")

from src.utils import shutdown_run

shutdown_run()
