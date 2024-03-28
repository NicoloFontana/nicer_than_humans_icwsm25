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


def plot_history_analysis(timestamp, infix=None):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    history_file_path = run_dir_path / f"game_history_{infix}.json"
    out_path = run_dir_path / "plots"
    out_path.mkdir(exist_ok=True)

    with open(history_file_path, "r") as history_file:
        history = json.load(history_file)

    A_history = history[player_1_]

    A_n_coop = A_history.count(1)
    A_coop_perc = A_n_coop / len(A_history)
    A_n_defect = A_history.count(0)
    A_defect_perc = A_n_defect / len(A_history)

    # Histogram
    plt.bar([f"Cooperate - {A_coop_perc}%", f"Defect - {A_defect_perc}%"], [A_n_coop, A_n_defect], color=['green', 'red'])
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

    # Transition matrix
    transition_matrix = np.zeros((2, 2), dtype=int)
    for i in range(len(A_history) - 1):
        transition_matrix[A_history[i], A_history[i + 1]] += 1

    ax = sns.heatmap(transition_matrix, annot=True, xticklabels=[0, 1], yticklabels=[0, 1], cmap="Blues", fmt='d')
    plt.title(f'Transition matrix" - {timestamp}')
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

    triple_transition_matrix = np.zeros((4, 2), dtype=int)
    y_labels = ["00", "01", "10", "11"]
    for i in range(len(A_history) - 2):
        triple_transition_matrix[y_labels.index(f"{A_history[i]}{A_history[i+1]}"), A_history[i+2]] += 1

    ax = sns.heatmap(triple_transition_matrix, annot=True, xticklabels=[0, 1], yticklabels=y_labels, cmap="Blues", fmt='d')
    plt.title(f'Second-order transition matrix - {timestamp}')
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
