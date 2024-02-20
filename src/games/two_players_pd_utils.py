import warnings


def generate_game_rules_prompt(action_space, payoff_function, n_iterations):
    payoff_prompt = ""
    for action in action_space:
        for opponent_action in action_space:
            payoff_prompt += f"If you play {to_nat_lang(action, True)} and the opponent plays {to_nat_lang(opponent_action, True)}, you get {payoff_function(action, opponent_action)} points and the opponent gets {payoff_function(opponent_action, action)} points.\n\t"

    return (f"<<SYS>>\n"
            f"\tContext: You are playing a multi-round game against an opponent.\n"
            f"\tAt each turn you and your opponent simultaneously perform one of the following actions: {to_nat_lang(action_space,True)}\n"
            f"\tThe payoffs for each combination of chosen action are the following:\n"
            f"\t{payoff_prompt}"
            f"You and your opponent will play a total of {n_iterations} rounds of this game.\n"
            f"\tRemember that your objective is to get the highest possible amount of points in the long run.\n")


def generate_history_prompt(own_history, opponent_history):
    history_prompt = ""
    for i in range(len(own_history)):
        history_prompt += f"\tIn round {i + 1} you played {to_nat_lang(own_history[i], True)} and your opponent played {to_nat_lang(opponent_history[i], True)}.\n"

    return history_prompt


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
    warnings.warn(f"Invalid action: {action}")
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
