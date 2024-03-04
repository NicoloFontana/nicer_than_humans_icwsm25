import json
import time
import warnings
from pathlib import Path

from matplotlib import pyplot as plt

from src.games.two_players_pd_utils import to_nat_lang

HF_API_TOKEN = "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"
OUT_BASE_PATH = Path("out")
MODEL = "meta-llama/Llama-2-70b-chat-hf"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7

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
            if str(e) == "Model is overloaded":
                warnings.warn("Model is overloaded. Waiting 2 seconds and retrying.")
                time.sleep(2)
            else:
                warnings.warn(
                    f"Error {str(e)} in text generation with prompt: {prompt}. Substituting with empty string.")
                generated_text = ""
                generated = True
    return generated_text


# CHECKERS UTILS

def plot_checkers_results(checkers_names, timestamp, n_iterations, infix=None):
    results_file_path = merge_checkers_results(checkers_names, timestamp, infix=infix)

    with open(results_file_path, "r") as results_file:
        results = json.load(results_file)

    entries = [result["label"] for result in results.values()]
    means = [result["sample_mean"] for result in results.values()]
    variances = [result["sample_variance"] for result in results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))

    first_cmap = plt.get_cmap('Dark2')

    checker_color_map = {checker: first_cmap(i / len(checkers_names)) for i, checker in enumerate(checkers_names)}
    entry_color_map = {}
    for result in results.values():
        if result['label'] in checkers_names:
            entry_color_map[result['label']] = 'red'
        else:
            entry_color_map[result['label']] = checker_color_map[result['checker']]
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
    for result in results.values():
        labels.append(result['label'])
    ax.set_xticks(range(len(entries)))  #
    ax.set_xticklabels(labels)
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(entry_color_map[tick_label.get_text()])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if infix is None:
        plt.savefig(OUT_BASE_PATH / str(timestamp) / f'{OVERALL}.svg')
        plt.savefig(OUT_BASE_PATH / str(timestamp) / f'{OVERALL}.png')
    else:
        plt.savefig(OUT_BASE_PATH / str(timestamp) / f'{OVERALL}_{infix}.svg')
        plt.savefig(OUT_BASE_PATH / str(timestamp) / f'{OVERALL}_{infix}.png')
    plt.show()


def merge_checkers_results(checkers_names, timestamp, infix=None):
    python_objects = {}

    for checker in checkers_names:
        if infix is None:
            in_file_path = OUT_BASE_PATH / str(
                timestamp) / checkers_names[checkers_names.index(checker)] / f"{checkers_names[checkers_names.index(checker)]}.json"
        else:
            in_file_path = OUT_BASE_PATH / str(
                timestamp) / checkers_names[checkers_names.index(checker)] / f"{checkers_names[checkers_names.index(checker)]}_{infix}.json"
        with open(in_file_path, "r") as f:
            python_object = json.load(f)
            for key in python_object.keys():
                python_objects[key] = python_object[key]

    # Dump all the Python objects into a single JSON file.
    if infix is None:
        out_file_path = OUT_BASE_PATH / str(timestamp) / f"{OVERALL}.json"
    else:
        out_file_path = OUT_BASE_PATH / str(timestamp) / f"{OVERALL}_{infix}.json"
    with open(out_file_path, "w") as f:
        json.dump(python_objects, f, indent=4)
        return out_file_path


# PROMPT UTILS

def generate_game_rules_prompt(action_space, payoff_function, n_iterations):
    payoff_prompt = ""
    for action in action_space:
        for opponent_action in action_space:
            payoff_prompt += f"If you play {to_nat_lang(action, True)} and the opponent plays {to_nat_lang(opponent_action, True)}, you get {payoff_function(action, opponent_action)} points and the opponent gets {payoff_function(opponent_action, action)} points.\n\t"

    return (f"<<SYS>>\n"
            f"\tContext: You are playing a multi-round game against an opponent.\n"
            f"\tAt each turn you and your opponent simultaneously perform one of the following actions: {to_nat_lang(action_space, True)}\n"
            f"\tThe payoffs for each combination of chosen action are the following:\n"
            f"\t{payoff_prompt}"
            f"You and your opponent will play a total of {n_iterations} rounds of this game.\n"
            f"\tRemember that your objective is to get the highest possible amount of points in the long run.\n")


def generate_history_prompt(own_history, opponent_history, payoff_function):
    history_prompt = ""
    own_total_payoff = 0
    own_coop = 0
    own_defect = 0
    opponent_total_payoff = 0
    opponent_coop = 0
    opponent_defect = 0
    for i in range(len(own_history)):
        own_coop += 1 if own_history[i] else 0
        own_defect += 1 if not own_history[i] else 0
        opponent_coop += 1 if opponent_history[i] else 0
        opponent_defect += 1 if not opponent_history[i] else 0
        own_payoff = payoff_function(own_history[i], opponent_history[i])
        opponent_payoff = payoff_function(opponent_history[i], own_history[i])
        history_prompt += (
            f"\tIn round {i + 1} you played {to_nat_lang(own_history[i], True)} and your opponent played {to_nat_lang(opponent_history[i], True)}. "
            f"You got {own_payoff} points and your opponent got {opponent_payoff} points.\n")
        own_total_payoff += own_payoff
        opponent_total_payoff += opponent_payoff
    history_prompt += (f"\tIn total, you have cooperated {own_coop} times and defected {own_defect} times, "
                       f"your opponent has cooperated {opponent_coop} times and defected {opponent_defect} times.\n")
    history_prompt += f"\tIn total, you have collected {own_total_payoff} points and your opponent has collected {opponent_total_payoff} points.\n"
    history_prompt += f"\tNow it is round {len(own_history) + 1}.\n"

    return history_prompt


def generate_prompt(action_space, payoff_function, n_iterations, own_history, opponent_history, custom_prompt=""):
    start_prompt = "<s>[INST] "

    game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)

    history_prompt = generate_history_prompt(own_history, opponent_history, payoff_function)

    if custom_prompt == "":
        custom_prompt = f"<<SYS>>"

    end_prompt = "[/INST]"

    return start_prompt + game_rules_prompt + history_prompt + custom_prompt + end_prompt


def generate_prompt_from_sub_prompts(sub_prompts):
    start_prompt = "<s>[INST] "

    body_prompt = "".join(sub_prompts)

    end_prompt = "[/INST]"

    return start_prompt + body_prompt + end_prompt
