import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.games.two_players_pd_utils import to_nat_lang, two_players_pd_payoff, player_1_, player_2_
from src.utils import OUT_BASE_PATH

HF_API_TOKEN = "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"
MODEL = "meta-llama/Llama-2-70b-chat-hf"
# MODEL = "CohereForAI/c4ai-command-r-plus"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
history_window_size = 14

OVERALL = "overall"


def generate_text(prompt, inference_client, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE):
    generated_text = ""
    generated = False
    while not generated:
        try:
            generated_text = inference_client.text_generation(prompt, max_new_tokens=max_new_tokens,
                                                              temperature=temperature)
            generated = True
        except Exception as e:
            if e.__class__.__name__ == "HfHubHTTPError" or e.__class__.__name__ == "OverloadedError":
                warnings.warn("Model is overloaded. Waiting 2 seconds and retrying.")
                time.sleep(2)
            else:
                warnings.warn(
                    f"Error {str(e)} in text generation with prompt: {prompt}. Substituting with empty string.")
                generated_text = ""
                generated = True
    return generated_text


# CHECKERS UTILS

def compact_format(x, is_percentage=False):
    if x == int(x):
        if x == 0:
            new_text = '-'
        else:
            new_text = '{:.0f}%'.format(x) if is_percentage else '{:.0f}'.format(x)
    else:
        new_text = '{:.2f}%'.format(x) if is_percentage else '{:.2f}'.format(x)
    return new_text


def extract_labels_and_questions_from_checker_answers(answers_dir_path, in_file_name=None):
    if in_file_name is None:
        in_file_name = answers_dir_path.name + "_complete_answers.json"
    in_file_path = answers_dir_path / in_file_name
    with open(in_file_path, "r") as in_file:
        json_data = in_file.read()
        dict_data = json.loads(json_data)
        labels = list(dict_data.keys())
        questions = {}
        for label in dict_data.keys():
            questions[label] = dict_data[label][str(0)]["question"]
        return labels, questions


def plot_confusion_matrix_for_question(answers_dir_path, label, in_file_name=None, title=None, infix=None):
    if in_file_name is None:
        if infix is None:
            in_file_name = answers_dir_path.name + "_complete_answers.json"
        else:
            in_file_name = answers_dir_path.name + f"_complete_answers_{infix}.json"
    in_file_path = answers_dir_path / in_file_name
    with open(in_file_path, "r") as in_file:
        out_path = answers_dir_path / "confusion_matrices"
        os.makedirs(out_path, exist_ok=True)
        json_data = in_file.read()
        dict_data = json.loads(json_data)
        answers = dict_data[label]

        # Get the correct labels
        y_labels = []
        other_str = "Other"
        for quest in answers.values():
            correct_ans = quest["answer"]["correct_answer"]
            if correct_ans not in y_labels:
                y_labels.append(correct_ans)

        # Get the LLM's answers
        actual = []
        predicted = []
        other = False
        for quest in answers.values():
            actual.append(quest["answer"]["correct_answer"])
            if quest["answer"]["llm_answer"] not in y_labels:
                predicted.append(other_str)  # Group unexpected answers as "Other"
                other = True
            else:
                predicted.append(quest["answer"]["llm_answer"])

        # Get the LLM labels
        x_labels = y_labels.copy()
        if other:
            x_labels.append(other_str)

        # Compute the confusion matrix
        confusion_matrix = np.zeros((len(y_labels), len(x_labels)), dtype=int)
        for correct_answer, llm_answer in zip(actual, predicted):
            correct_idx = y_labels.index(correct_answer)
            llm_idx = x_labels.index(llm_answer)
            confusion_matrix[correct_idx, llm_idx] += 1

        # Create the plot and save it
        if title is None:
            title = label
        ax = sns.heatmap(confusion_matrix, annot=True, xticklabels=x_labels, yticklabels=y_labels, cmap="Blues", fmt='d')
        for text in ax.texts:
            x = float(text.get_text())
            text.set_text(compact_format(x))
        plt.title(title, loc="center", wrap=True)
        plt.xlabel('LLM answer')
        plt.ylabel('Correct answer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if infix is None:
            plt.savefig(out_path / f"{label}.svg")
            plt.savefig(out_path / f"{label}.png")
        else:
            plt.savefig(out_path / f"{label}_{infix}.svg")
            plt.savefig(out_path / f"{label}_{infix}.png")
        plt.show()
        ax = sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot=True, xticklabels=x_labels, yticklabels=y_labels, cmap="Blues", fmt='.2%')
        for text in ax.texts:
            x = float(text.get_text().replace('%', ''))
            text.set_text(compact_format(x, is_percentage=True))
        plt.title(title, loc="center", wrap=True)
        plt.xlabel('LLM answer')
        plt.ylabel('Correct answer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if infix is None:
            plt.savefig(out_path / f"{label}_percentage.svg")
            plt.savefig(out_path / f"{label}_percentage.png")
        else:
            plt.savefig(out_path / f"{label}_percentage_{infix}.svg")
            plt.savefig(out_path / f"{label}_percentage_{infix}.png")
        plt.show()


def plot_checkers_results(checkers_names: list, timestamp, n_iterations, infix=None):
    if checkers_names is None or len(checkers_names) == 0:
        return
    results_file_path = merge_checkers_results(checkers_names, timestamp, infix=infix)

    with open(results_file_path, "r") as results_file:
        results = json.load(results_file)

    entries = [label for label in results.keys()]
    means = [result["sample_mean"] for result in results.values()]
    variances = [result["sample_variance"] for result in results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))

    first_cmap = plt.get_cmap('Dark2')

    checker_color_map = {checker: first_cmap(i / len(checkers_names)) for i, checker in enumerate(checkers_names)}
    entry_color_map = {}
    for label in results.keys():
        if label in checkers_names:
            entry_color_map[label] = 'red'
        else:
            entry_color_map[label] = checker_color_map[results[label]['checker']]
    for entry, mean, variance in zip(entries, means, variances):
        ax.plot([entry, entry], [mean - variance, mean + variance], '_:k', markersize=10, label='Variance')

    for entry in entries:
        plt.scatter(entry, means[entries.index(entry)], color=entry_color_map[entry], label=entry, s=100)

    for checker in checkers_names:
        checker_idx = entries.index(checker)
        plt.axvline(x=checker_idx - 0.5, linestyle='--', color='black', lw=0.5)

    plt.axhline(y=1.0, color='red', linestyle='-.', lw=0.25)
    plt.axhline(y=0.75, color='red', linestyle='-.', lw=0.5)
    plt.axhline(y=0.5, color='red', linestyle='-.', lw=0.75)
    plt.axhline(y=0.25, color='red', linestyle='-.', lw=0.5)
    plt.axhline(y=0.0, color='red', linestyle='-.', lw=0.25)

    ax.set_xlabel('Questions')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'LLM checks - {timestamp} - {n_iterations} iterations')

    labels = []
    for label in results.keys():
        labels.append(label)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(labels)
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(entry_color_map[tick_label.get_text()])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = OUT_BASE_PATH / str(timestamp)
    if infix is None:
        plt.savefig(out_path / f'{OVERALL}.svg')
        plt.savefig(out_path / f'{OVERALL}.png')
    else:
        plt.savefig(out_path / f'{OVERALL}_{infix}.svg')
        plt.savefig(out_path / f'{OVERALL}_{infix}.png')
    plt.show()


def merge_checkers_results(checkers_names, timestamp, infix=None):
    python_objects = {}
    out_path = OUT_BASE_PATH / str(timestamp)

    for checker in checkers_names:
        if infix is None:
            in_file_path = out_path / checkers_names[checkers_names.index(checker)] / f"{checkers_names[checkers_names.index(checker)]}.json"
        else:
            in_file_path = out_path / checkers_names[checkers_names.index(checker)] / f"{checkers_names[checkers_names.index(checker)]}_{infix}.json"
        with open(in_file_path, "r") as f:
            python_object = json.load(f)
            for key in python_object.keys():
                python_objects[key] = python_object[key]

    if infix is None:
        out_file_path = out_path / f"{OVERALL}.json"
    else:
        out_file_path = out_path / f"{OVERALL}_{infix}.json"
    with open(out_file_path, "w") as out_file:
        json_out = json.dumps(python_objects, indent=4)
        out_file.write(json_out)
        return out_file_path


# PROMPT UTILS

def generate_game_rules_prompt(action_space, payoff_function, n_iterations):
    single_payoff_prompt = ("If {} plays {} and {} plays {}, "  # If A plays "Cooperate" and B plays "Defect",
                            "{} collects {} points and {} collects {} points.\n")  # A collects 0 points and B collects 5 points.
    payoff_prompt = "".join([single_payoff_prompt.format(player_1_, to_nat_lang(own_action), player_2_, to_nat_lang(opponent_action), player_1_, payoff_function(own_action, opponent_action), player_2_, payoff_function(opponent_action, own_action)) for own_action in action_space for opponent_action in action_space])

    game_rules_prompt = (f"<<SYS>>\n"
                         f"Context: Player {player_1_} and player {player_2_} are playing a multi-round game.\n"
                         f"At each turn player {player_1_} and player {player_2_} simultaneously perform one of the following actions: {to_nat_lang(action_space)}\n"
                         f"The payoffs for each combination of chosen action are the following:\n"
                         f"{payoff_prompt}"
                         f"They will play a total of {n_iterations} rounds of this game.\n"
                         f"Remember that a player's objective is to get the highest possible amount of points in the long run.<<SYS>>\n")

    return game_rules_prompt


def generate_history_prompt(own_history, opponent_history, payoff_function, window_size=None, is_ended=False):
    if len(own_history) == 0:
        return "This is the first round of the game.\n"
    if window_size is None:
        window_size = len(own_history)
    history_prompt_parts = [f"The history of the game in the last {min(len(own_history), window_size)} rounds is the following:\n"]

    start = max(0, len(own_history) - window_size)
    end = len(own_history)
    own_coop = sum(own_history[start:end])
    own_defect = sum([1 for action in own_history[start:end] if not action])
    opponent_coop = sum(opponent_history[start:end])
    opponent_defect = sum([1 for action in opponent_history[start:end] if not action])
    own_total_payoff = sum([payoff_function(own_history[i], opponent_history[i]) for i in range(start, end)])
    opponent_total_payoff = sum([payoff_function(opponent_history[i], own_history[i]) for i in range(start, end)])
    single_round_prompt = ("Round {}: {} played {} and {} played {} "  # Round 1: A played "Cooperate" and B played "Defect"
                           "{} collected {} points and {} collected {} points.\n")  # A collected 0 points and B collected 5 points.
    rounds_prompt = "".join([single_round_prompt.format(i+1, player_1_, to_nat_lang(own_history[i]), player_2_, to_nat_lang(opponent_history[i]),
                                                        player_1_, payoff_function(own_history[i], opponent_history[i]), player_2_,
                                                        payoff_function(opponent_history[i], own_history[i])) for i in range(start, end)])
    history_prompt_parts.append(rounds_prompt)
    history_prompt_parts.append(f'In total, {player_1_} chose {to_nat_lang(1)} {own_coop} times and chose {to_nat_lang(0)} {own_defect} times, '
                       f'{player_2_} chose {to_nat_lang(1)} {opponent_coop} times and chose {to_nat_lang(0)} {opponent_defect} times.\n')
    history_prompt_parts.append(f"In total, {player_1_} collected {own_total_payoff} points and {player_2_} collected {opponent_total_payoff} points.\n")
    if not is_ended:
        history_prompt_parts.append(f"Current round: {len(own_history) + 1}.\n")
    else:
        history_prompt_parts.append(f"The game has ended.\n")

    history_prompt = "".join(history_prompt_parts)
    return history_prompt


def generate_prompt(action_space, payoff_function, n_iterations, own_history, opponent_history, custom_prompt="", history_window_size=None, zero_shot=False):
    game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)

    is_ended = len(own_history) >= n_iterations
    history_prompt = generate_history_prompt(own_history, opponent_history, payoff_function, window_size=history_window_size, is_ended=is_ended)

    prompt = generate_prompt_from_sub_prompts([game_rules_prompt, history_prompt, custom_prompt], zero_shot=zero_shot)

    return prompt


def generate_prompt_from_sub_prompts(sub_prompts, zero_shot=False):
    prompt_parts = ["<s>[INST] ", "".join(sub_prompts), "Remember to answer using the right format.[/INST]\n"]

    if zero_shot:
        prompt_parts.append(f"Let's work this out in a step-by-step way to be sure we have the right answer in the right format\n")

    prompt = "".join(prompt_parts)
    return prompt


def save_prompt(version, description=None):
    if description is None or description == "":
        print("Remember to add a description to the prompt.")
        return
    out_path = Path("prompts") / f"v{version}"
    out_path.mkdir(parents=True, exist_ok=True)
    custom_prompt = ('Remember to use only the following JSON format: {"action": <ACTION_of_A>, "reason": <YOUR_REASON>}\n'
                     f'Answer saying which action player {player_1_} should play.')
    with open(out_path / "prompt.txt", "w") as f:
        f.write(generate_prompt({1, 0}, two_players_pd_payoff, 100, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], custom_prompt))
    with open(out_path / "description.txt", "w") as f:
        f.write(description)

    llm_utils_path = Path("src") / "llm_utils.py"
    with open(llm_utils_path, "r") as infile:
        with open(out_path / "def_generate_history_prompt.txt", 'w') as outfile:
            copy = False
            for line in infile:
                if "def generate_history_prompt(" in line.strip():
                    outfile.write(line)
                    copy = True
                elif "return history_prompt" in line.strip():
                    outfile.write(line)
                    copy = False
                elif copy:
                    outfile.write(line)
    with open(llm_utils_path, "r") as infile:
        with open(out_path / "def_generate_game_rules_prompt.txt", 'w') as outfile:
            copy = False
            for line in infile:
                if "def generate_game_rules_prompt(" in line.strip():
                    outfile.write(line)
                    copy = True
                elif "return game_rules_prompt" in line.strip():
                    outfile.write(line)
                    copy = False
                elif copy:
                    outfile.write(line)
