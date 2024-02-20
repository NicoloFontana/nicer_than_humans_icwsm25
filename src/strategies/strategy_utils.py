from src.games.two_players_pd_utils import to_nat_lang


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


def generate_prompt(action_space, payoff_function, n_iterations, own_history, opponent_history):
    start_prompt = "<s>[INST] "

    game_rules_prompt = generate_game_rules_prompt(action_space, payoff_function, n_iterations)

    json_prompt = 'Remember to answer using only the following JSON format: {"action": <YOUR_ACTION>, "reason": <YOUR_REASON>}'

    history_prompt = generate_history_prompt(own_history, opponent_history)

    next_action_prompt = f"<<SYS>>Now it is round {len(own_history) + 1}.\nAnswer saying which action you choose to play."

    end_prompt = "[/INST]"

    return start_prompt + game_rules_prompt + json_prompt + history_prompt + next_action_prompt + end_prompt
