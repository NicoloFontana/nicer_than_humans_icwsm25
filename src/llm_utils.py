from pathlib import Path

from src.game.two_players_pd_utils import to_nat_lang, player_1_, player_2_, two_players_pd_axelrod_payoff

# TODO 1/3: [START] check model, max_new_tokens, temperature, history_window_size
KEY = "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"  # huggingface
# KEY = "sk-proj-WUY3EjWIgbwhS3UbY6DTT3BlbkFJohhB3HQl5D3yyxWxRJcH"  # openai
MODEL_NAME = "llama3"
# MODEL_NAME = "gpt35t"
# MODEL_URL = "meta-llama/Llama-2-70b-chat-hf"
# MODEL_URL = "gpt-3.5-turbo"
MODEL_URL = "meta-llama/Meta-Llama-3-70B-Instruct"
# MODEL_URL = "CohereForAI/c4ai-command-r-plus"
PROVIDER = "huggingface"
# PROVIDER = "openai"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
history_window_size = 10

daily_requests = 0
minute_requests = 0
daily_requests_limit = 10000
minute_requests_limit = 3500

OVERALL = "overall"


# PROMPT UTILS

def generate_game_rules_prompt(action_space, payoff_function, n_iterations):
    single_payoff_prompt = ("If {} plays {} and {} plays {}, "  # If A plays "Cooperate" and B plays "Defect",
                            "{} collects {} points and {} collects {} points.\n")  # A collects 0 points and B collects 5 points.
    payoff_prompt = "".join([single_payoff_prompt.format(player_1_, to_nat_lang(own_action), player_2_, to_nat_lang(opponent_action), player_1_,
                                                         payoff_function(own_action, opponent_action), player_2_, payoff_function(opponent_action, own_action)) for own_action in
                             action_space for opponent_action in action_space])

    game_rules_prompt = (f"<<SYS>>\n"
                         f"Context: Player {player_1_} and player {player_2_} are playing a multi-round game.\n"
                         f"At each turn player {player_1_} and player {player_2_} simultaneously perform one of the following actions: {to_nat_lang(action_space)}\n"
                         f"The payoffs for each combination of chosen actions are the following:\n"
                         f"{payoff_prompt}"
                         # TODO 3/3: check IIPD vs IPD --> goto one_vs_one_pd_llm_strategy.py
                         f"They will play a total of {n_iterations} rounds of this game.\n"  # INDEFINITELY vs DEFINITELY IPD
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
    own_coop = sum(own_history)
    own_defect = sum([1 for action in own_history if not action])
    opponent_coop = sum(opponent_history)
    opponent_defect = sum([1 for action in opponent_history if not action])
    own_total_payoff = sum([payoff_function(own_history[i], opponent_history[i]) for i in range(0, end)])
    opponent_total_payoff = sum([payoff_function(opponent_history[i], own_history[i]) for i in range(0, end)])
    single_round_prompt = ("Round {}: {} played {} and {} played {} "  # Round 1: A played "Cooperate" and B played "Defect"
                           "{} collected {} points and {} collected {} points.\n")  # A collected 0 points and B collected 5 points.
    rounds_prompt = "".join([single_round_prompt.format(i + 1, player_1_, to_nat_lang(own_history[i]), player_2_, to_nat_lang(opponent_history[i]),
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


def save_prompt(version, description=None):  # TODO parametrize
    if description is None or description == "":
        print("Remember to add a description to the prompt.")
        return
    out_path = Path("prompts") / f"v{version}"
    out_path.mkdir(parents=True, exist_ok=True)
    custom_prompt = ('Remember to use only the following JSON format: {"action": <ACTION_of_A>}\n'  # , "reason": <YOUR_REASON>}\n'
                     f'Answer saying which action player {player_1_} should play.')
    with open(out_path / "prompt.txt", "w") as f:
        f.write(generate_prompt({1, 0}, two_players_pd_axelrod_payoff, 100, [1, 0, 1, 0, 1], [0, 1, 0, 1, 0], custom_prompt))
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
