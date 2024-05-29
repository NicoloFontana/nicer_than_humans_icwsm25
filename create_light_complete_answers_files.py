import json
from pathlib import Path

run_dir = Path("relevant_runs_copies") / "main_runs_copies" / "llama2_v10_questions_3runs" / "run3"
checkers_names = ["time_checker", "rule_checker", "aggregation_checker"]

for checker_name in checkers_names:
    complete_answers_dir = run_dir / checker_name / "complete_answers"
    complete_answers_file = complete_answers_dir / f"{checker_name}_complete_answers.json"
    with open(complete_answers_file, "r") as complete_answers_file:
        complete_answers = json.load(complete_answers_file)
    light_complete_answers = {}
    for label in complete_answers.keys():
        light_single_complete_answer = {
            "question": complete_answers[label]["0"]["question"],
            "answers": []
        }
        for iteration in complete_answers[label].keys():
            light_single_complete_answer["answers"].append(complete_answers[label][iteration]["answer"]["is_correct"])
        light_complete_answers[label] = light_single_complete_answer
    light_complete_answers_file = run_dir / checker_name / f"light_complete_answers.json"
    with open(light_complete_answers_file, "w") as light_complete_answers_file:
        json.dump(light_complete_answers, light_complete_answers_file, indent=4)
    print(f"Light complete answers of {checker_name} saved.")

from src.utils import shutdown_run

shutdown_run()
