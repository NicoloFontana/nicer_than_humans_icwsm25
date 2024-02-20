import warnings


def to_nat_lang(action, string_of_string=False):
    if isinstance(action, set) and len(action) == 2 and 1 in action and 0 in action:
        return '"{"Cooperate", "Defect"}"' if string_of_string else '{"Cooperate", "Defect"}'
    if action == 1:
        return '"Cooperate"' if string_of_string else "Cooperate"
    if action == 0:
        return '"Defect"' if string_of_string else "Defect"
    raise ValueError(f"Invalid action: {action}")


def from_nat_lang(action):
    if action == "Cooperate" or action == '"Cooperate"':
        return 1
    if action == "Defect" or action == '"Defect"':
        return 0
    warnings.warn(f"Invalid action: {action}. Returning 'Defect' as 0.")
    return 0


def two_players_pd_payoff(own_action: int, other_action: int) -> int:
    """
    Default payoff function for the two-player version of the Prisoner's Dilemma game
    :param own_action: action of the player for which the payoff is computed
    :param other_action: oppent's action
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
