import json
import time
import warnings
from pathlib import Path

from matplotlib import pyplot as plt

from src.games.two_players_pd_utils import to_nat_lang, two_players_pd_payoff

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

def plot_checkers_results(checkers_names: list, timestamp, n_iterations, infix=None):
    if checkers_names is None or len(checkers_names) == 0:
        return
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

    # Dump all the Python objects into a single JSON file.
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
    # payoff_prompt = ""
    # for action in action_space:
    #     for opponent_action in action_space:
    #         payoff_dict = {
    #             "action_of_A": to_nat_lang(action),
    #             "action_of_B": to_nat_lang(opponent_action),
    #             "payoff_of_A": payoff_function(action, opponent_action),
    #             "payoff_of_B": payoff_function(opponent_action, action)
    #         }
    #         payoff_prompt += str(payoff_dict) + "\n"
    #
    # return (f"<<SYS>>\n"
    #         f"\tContext: Player A is playing a multi-round game against player B.\n"
    #         f"\tAt each turn player A and player B simultaneously perform one of the following actions: {to_nat_lang(action_space, True)}\n"
    #         f"\tThe payoffs for each combination of chosen action are the following:\n"
    #         f"\t{payoff_prompt}"
    #         f"They will play a total of {n_iterations} rounds of this game.\n"
    #         f"\tRemember that a player's objective is to get the highest possible amount of points in the long run.\n")

    ### v0.1
    payoff_prompt = ""
    for action in action_space:
        for opponent_action in action_space:
            payoff_prompt += (f"If you play {to_nat_lang(action, True)} and the opponent plays {to_nat_lang(opponent_action, True)}, "
                              f"you get {payoff_function(action, opponent_action)} points and the opponent gets {payoff_function(opponent_action, action)} points.\n\t")

    return (f"<<SYS>>\n"
            f"\tContext: You are playing a multi-round game against an opponent.\n"
            f"\tAt each turn you and your opponent simultaneously perform one of the following actions: {to_nat_lang(action_space, True)}\n"
            f"\tThe payoffs for each combination of chosen action are the following:\n"
            f"\t{payoff_prompt}"
            f"You and your opponent will play a total of {n_iterations} rounds of this game.\n"
            f"\tRemember that your objective is to get the highest possible amount of points in the long run.\n")


def generate_history_prompt(own_history, opponent_history, payoff_function, is_ended=False):
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
        #     round_dict = {
        #         "round": i + 1,
        #         "action_of_A": to_nat_lang(own_history[i]),
        #         "action_of_B": to_nat_lang(opponent_history[i]),
        #         "payoff_of_A": own_payoff,
        #         "payoff_of_B": opponent_payoff
        #     }
        #     history_prompt += str(round_dict) + "\n"
        #     own_total_payoff += own_payoff
        #     opponent_total_payoff += opponent_payoff
        # aggregate_dict = {
        #     "n_times_coop_A": own_coop,
        #     "n_times_defect_A": own_defect,
        #     "n_times_coop_B": opponent_coop,
        #     "n_times_defect_B": opponent_defect,
        #     "total_payoff_A": own_total_payoff,
        #     "total_payoff_B": opponent_total_payoff
        # }
        # history_prompt += str(aggregate_dict) + "\n"
        # if not is_ended:
        #     history_prompt += f"\tNow it is round {len(own_history) + 1}.\n"
        # else:
        #     history_prompt += f"\tThe game has ended.\n"

        ### v0.1
        history_prompt += (
            f"\tIn round {i + 1} you played {to_nat_lang(own_history[i], True)} and your opponent played {to_nat_lang(opponent_history[i], True)}. "
            f"You got {own_payoff} points and your opponent got {opponent_payoff} points.\n")
        own_total_payoff += own_payoff
        opponent_total_payoff += opponent_payoff
    history_prompt += (f'\tIn total, you chose "Cooperate" {own_coop} times and chose "Defect" {own_defect} times, '
                       f'your opponent chose "Cooperate" {opponent_coop} times and chose "Defect" {opponent_defect} times.\n')
    history_prompt += f"\tIn total, you collected {own_total_payoff} points and your opponent collected {opponent_total_payoff} points.\n"
    if not is_ended:
        history_prompt += f"\tNow it is round {len(own_history) + 1}.\n"
    else:
        history_prompt += f"\tThe game has ended.\n"

    return history_prompt


def generate_prompt(action_space, payoff_function, n_iterations, own_history, opponent_history, custom_prompt=""):
    start_prompt = "<s>[INST] "

    game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)

    is_ended = len(own_history) >= n_iterations
    history_prompt = generate_history_prompt(own_history, opponent_history, payoff_function, is_ended=is_ended)

    if custom_prompt == "":
        custom_prompt = f"<<SYS>>"

    end_prompt = "[/INST]"

    return start_prompt + game_rules_prompt + history_prompt + custom_prompt + end_prompt


def generate_prompt_from_sub_prompts(sub_prompts):
    start_prompt = "<s>[INST] "

    body_prompt = "".join(sub_prompts)

    end_prompt = "[/INST]"

    return start_prompt + body_prompt + end_prompt


def save_prompt(version, description=None):
    out_path = Path("prompts") / f"v{version}"
    out_path.mkdir(parents=True, exist_ok=True)
    custom_prompt = ('\tRemember to use only the following JSON format: {"action": <ACTION_of_A>, "reason": <YOUR_REASON>}<<SYS>>\n'
                     '\tAnswer saying which action player A should play.')
    with open(out_path / "prompt.txt", "w") as f:
        f.write(generate_prompt({1, 0}, two_players_pd_payoff, 100, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], custom_prompt))
    if description is not None:
        with open(out_path / "description.txt", "w") as f:
            f.write(description)
