import warnings

from src.game.game_history import GameHistory
from src.utils import extract_infixes_

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


def extract_histories_from_files(out_dir, file_name=None, numerical_infixes=None, max_infix=None):
    game_histories = []
    if file_name is None:
        file_name = "game_history"
    if numerical_infixes is None:
        numerical_infixes = extract_infixes_(out_dir, file_name=file_name, max_infix=max_infix)
    for infix in numerical_infixes:
        game_history = GameHistory()
        file_path = out_dir / f"{file_name}_{infix}.json"
        game_history.load_from_file(file_path)
        game_histories.append(game_history)
    return game_histories


def two_players_pd_axelrod_payoff(own_action: int, other_action: int) -> int:
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
