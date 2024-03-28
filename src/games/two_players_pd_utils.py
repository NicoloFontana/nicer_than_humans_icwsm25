import json
import warnings

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.utils import OUT_BASE_PATH

action_0_ = "Defect"
action_1_ = "Cooperate"

player_1_ = "A"
player_2_ = "B"


def to_nat_lang(action, string_of_string=True):
    if isinstance(action, set) and len(action) == 2 and 1 in action and 0 in action:
        return f'{{\"{action_1_}\", \"{action_0_}\"}}' if string_of_string else f'{{{action_1_}, {action_0_}}}'
    if action == 1:
        return f'\"{action_1_}\"' if string_of_string else action_1_
    if action == 0:
        return f'\"{action_0_}\"' if string_of_string else action_0_
    raise ValueError(f"Invalid action: {action}")


def from_nat_lang(action):
    if action == action_1_ or action == f'{action_1_}':
        return 1
    if action == action_0_ or action == f'{action_0_}':
        return 0
    warnings.warn(f"Invalid action: {action}. Returning '{action_0_}' as 0.")
    return 0


def two_players_pd_payoff(own_action: int, other_action: int) -> int:
    """
    Default payoff function for the two-player version of the Prisoner's Dilemma game
    :param own_action: action of the player for which the payoff is computed
    :param other_action: opponent's action
    :return: computed payoff
    """
    if own_action == 1 and other_action == 1:
        return 3
    elif own_action == 1 and other_action == 0:
        return 0
    elif own_action == 0 and other_action == 1:
        return 5
    elif own_action == 0 and other_action == 0:
        return 1
    else:
        raise ValueError("Invalid actions")


def plot_occurrences_histogram(timestamp, history, infix=None):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots"
    out_path.mkdir(exist_ok=True)

    n_iterations = len(history)

    A_n_coop = history.count(1)
    A_coop_perc = A_n_coop / len(history)
    A_n_defect = history.count(0)
    A_defect_perc = A_n_defect / len(history)

    # Histogram
    plt.bar([f"Cooperate - {(A_coop_perc * 100):.2f}%", f"Defect - {(A_defect_perc * 100):.2f}%"], [A_n_coop, A_n_defect], color=['green', 'red'])

    plt.axhline(y=1.0 * n_iterations, color='red', linestyle='-.', lw=0.25)
    plt.axhline(y=0.75 * n_iterations, color='red', linestyle='-.', lw=0.5)
    plt.axhline(y=0.5 * n_iterations, color='red', linestyle='-.', lw=0.75)
    plt.axhline(y=0.25 * n_iterations, color='red', linestyle='-.', lw=0.5)
    plt.axhline(y=0.0 * n_iterations, color='red', linestyle='-.', lw=0.25)

    plt.title(f'Occurrences of "Cooperate" and "Defect" - {timestamp}')
    plt.ylabel('Occurrences')
    plt.tight_layout()
    if infix is None:
        plt.savefig(out_path / f"occurrences.svg")
        plt.savefig(out_path / f"occurrences.png")
    else:
        plt.savefig(out_path / f"occurrences_{infix}.svg")
        plt.savefig(out_path / f"occurrences_{infix}.png")
    plt.show()


def compute_transition_matrix(history, transition_matrix=None):
    if transition_matrix is None:
        transition_matrix = np.zeros((2, 2), dtype=int)
    for i in range(len(history) - 1):
        transition_matrix[history[i], history[i + 1]] += 1
    return transition_matrix


def plot_transition_matrix(timestamp, transition_matrix, infix=None):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots"
    out_path.mkdir(exist_ok=True)

    ax = sns.heatmap(transition_matrix, annot=True, xticklabels=[0, 1], yticklabels=[0, 1], cmap="Blues", fmt='d')
    if infix is None:
        plt.title(f'Transition matrix" - {timestamp}')
    else:
        plt.title(f'Transition matrix - {timestamp} - Game {infix}')
    plt.xlabel('Next action')
    plt.ylabel('Current action')
    plt.tight_layout()
    if infix is None:
        plt.savefig(out_path / f"transition_matrix.svg")
        plt.savefig(out_path / f"transition_matrix.png")
    else:
        plt.savefig(out_path / f"transition_matrix_{infix}.svg")
        plt.savefig(out_path / f"transition_matrix_{infix}.png")
    plt.show()

    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots"
    out_path.mkdir(exist_ok=True)


def compute_second_order_transition_matrix(history, second_order_transition_matrix=None):
    if second_order_transition_matrix is None:
        second_order_transition_matrix = np.zeros((4, 2), dtype=int)
    for i in range(len(history) - 2):
        second_order_transition_matrix[2 * history[i] + history[i + 1], history[i + 2]] += 1
    return second_order_transition_matrix


def plot_second_order_transition_matrix(timestamp, second_order_transition_matrix, infix=None):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots"
    out_path.mkdir(exist_ok=True)

    y_labels = ["00", "01", "10", "11"]

    ax = sns.heatmap(second_order_transition_matrix, annot=True, xticklabels=[0, 1], yticklabels=y_labels, cmap="Blues", fmt='d')
    if infix is None:
        plt.title(f'Second-order transition matrix - {timestamp}')
    else:
        plt.title(f'Second-order transition matrix - {timestamp} - Game {infix}')
    plt.xlabel('Next action')
    plt.ylabel('Current action')
    plt.tight_layout()
    if infix is None:
        plt.savefig(out_path / f"second_order_transition_matrix.svg")
        plt.savefig(out_path / f"second_order_transition_matrix.png")
    else:
        plt.savefig(out_path / f"second_order_transition_matrix_{infix}.svg")
        plt.savefig(out_path / f"second_order_transition_matrix_{infix}.png")
    plt.show()


def plot_history_analysis(timestamp, infix=None):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    if infix is None:
        history_file_path = run_dir_path / "game_history.json"
    else:
        history_file_path = run_dir_path / f"game_history_{infix}.json"

    with open(history_file_path, "r") as history_file:
        history = json.load(history_file)

    A_history = history[player_1_]

    plot_occurrences_histogram(timestamp, A_history, infix)
    transition_matrix = compute_transition_matrix(A_history)
    plot_transition_matrix(timestamp, transition_matrix, infix)
    second_order_transition_matrix = compute_second_order_transition_matrix(A_history)
    plot_second_order_transition_matrix(timestamp, second_order_transition_matrix, infix)


def plot_aggregated_histories_analysis(timestamp, infixes):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots"
    out_path.mkdir(exist_ok=True)

    histories = []
    aggregated_history = []
    for infix in infixes:
        history_file_path = run_dir_path / f"game_history_{infix}.json"
        with open(history_file_path, "r") as history_file:
            history = json.load(history_file)
            histories.append(history)
        aggregated_history.extend(history[player_1_])

    plot_occurrences_histogram(timestamp, aggregated_history)

    transition_matrix = None
    for hist in histories:
        transition_matrix = compute_transition_matrix(hist[player_1_], transition_matrix)
    plot_transition_matrix(timestamp, transition_matrix)

    second_order_transition_matrix = None
    for hist in histories:
        second_order_transition_matrix = compute_second_order_transition_matrix(hist[player_1_], second_order_transition_matrix)
    plot_second_order_transition_matrix(timestamp, second_order_transition_matrix)


def plot_actions_ts(timestamp, infixes):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots"
    out_path.mkdir(exist_ok=True)

    histories = []
    A_histories = []
    for infix in infixes:
        history_file_path = run_dir_path / f"game_history_{infix}.json"
        with open(history_file_path, "r") as history_file:
            history = json.load(history_file)
            histories.append(history)
            A_histories.append(history[player_1_])

    n_games = len(histories)
    n_iterations = len(A_histories[0])

    averaged_history = []
    for i in range(n_iterations):
        averaged_history.append(sum([A_histories[j][i] for j in range(len(A_histories))]) / len(A_histories))

    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(n_iterations)], averaged_history, linestyle='-', marker='o', color='b')
    plt.title(f'Average action per round - {timestamp} - {n_games} games')
    plt.xlabel('Iteration')
    plt.ylabel('Avg action')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path / f"average_action_per_round.svg")
    plt.savefig(out_path / f"average_action_per_round.png")
    plt.show()
