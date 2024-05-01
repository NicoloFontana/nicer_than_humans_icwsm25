from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.games.gt_game import GTGame
from src.games.two_players_pd_utils import extract_histories_from_files
from src.player import Player
from src.strategies.strategy_utils import get_strategies, main_hard_coded_strategies, main_blind_strategies, get_strategy_by_name, get_strategy_by_label, get_strategies_by_labels


def compute_likelihood(game_histories, strategy, strategies, parameters, main_player_name):
    # strategies is only ALLD, main_player_name is ALLD
    n_strategies = len(strategies)
    strategy_idx = strategies.index(strategy)
    likelihood = 1.0
    for game_history in game_histories:
        strategy_history = strategy.generate_alternative_history_for_player(game_history, main_player_name)  # generate ALLD history
        game_likelihood = 1.0
        main_history = game_history.get_actions_by_player(main_player_name)  # get ALLD history
        for iteration in range(len(main_history)):
            fidelity = 1 if main_history[iteration] == strategy_history[iteration] else 0  # I = 1 if s=c, else I = 0
            iteration_likelihood = (parameters[0] ** fidelity) * ((1 - parameters[0]) ** (1 - fidelity))  # β^I * (1-β)^(1-I)
            game_likelihood *= iteration_likelihood  # Π_r β^I * (1-β)^(1-I)
        likelihood *= game_likelihood  # Π_m Π_r β^I * (1-β)^(1-I)
    return likelihood


def compute_neg_log_likelihood(parameters, game_histories, strategies, main_player_name):
    likelihoods = []
    for strategy in strategies:
        l = compute_likelihood(game_histories, strategy, strategies, parameters, main_player_name)
        likelihoods.append(parameters[strategies.index(strategy) + 1]*l)  # Φ_k * Π_m Π_r β^I * (1-β)^(1-I)
    return -np.log(sum(likelihoods)) if sum(likelihoods) > 0 else 1e6  # -ln(Σ_k Φ_k * Π_m Π_r β^I * (1-β)^(1-I))


player_name = "ALLD"
extraction_timestamp = "20240422133439"  # Try with hard vs RND
subdir = Path("game_histories") / player_name
file_name = f"game_history_{player_name}-rnd_baseline"
game_histories_analyzed = extract_histories_from_files(extraction_timestamp, subdir=subdir, file_name=file_name)  # get both rnd_bseline and ALLD histories
# strategies_analyzed = get_strategies()
game = GTGame(players={player_name: Player(player_name), "rnd_baseline": Player("rnd_baseline")})

# strategies_array = []
# for strategy_name in strategies_analyzed.keys():
#     if strategy_name in main_hard_coded_strategies:
#         strategies_array.append(main_hard_coded_strategies[strategy_name]["strategy"](game, None))
#     elif strategy_name == "unfair_random":
#         for p in range(6, 10):
#             p = p / 10
#             strategies_array.append(main_blind_strategies["unfair_random"]["strategy"](p))
#     else:
#         strategies_array.append(strategies_analyzed[strategy_name]["strategy"]())
strategy_entries = get_strategies_by_labels([player_name, "ALLC"])  # get ALLD strategy
strategy_names = list(strategy_entries.keys())
print(strategy_names)
strategies_array = [strategy_entries[strategy_name]["strategy"]() for strategy_name in strategy_names]
num_strategies = len(strategies_array)

# TODO FIX MLE!
beta = np.array([0.9])  # with beta=0.9 doesn't work, with beta=1 works...
# initial_phis = np.random.uniform(0.0, 1, size=num_strategies)
initial_phis = np.array([0.5] * num_strategies)
initial_parameters = np.concatenate((beta, initial_phis))
bounds = [(0.5, 1)] + [(0.0, 1)] * num_strategies

result = minimize(compute_neg_log_likelihood, initial_parameters, args=(game_histories_analyzed, strategies_array, player_name), method='L-BFGS-B',
                  bounds=bounds)

estimated_parameters = result.x
estimated_phis = estimated_parameters[1:]
estimated_beta = estimated_parameters[0]

print(f"Estimated Phis: {estimated_phis}")
print(f"Estimated Beta: {estimated_beta}")


from src.utils import shutdown_run

shutdown_run()
