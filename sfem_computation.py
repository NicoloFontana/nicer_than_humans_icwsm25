from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from src.games.gt_game import GTGame
from src.games.two_players_pd_utils import extract_histories_from_files, player_1_
from src.player import Player
from src.strategies.strategy_utils import get_strategies, main_hard_coded_strategies, main_blind_strategies, get_strategy_by_name, get_strategy_by_label, get_strategies_by_labels

"""
Implementation of the SFEM algorithm to compute the behavioral profile of a player against a set of strategies.

Adapted from:
Romero, J. and Rosokha, Y. (2018) ‘Constructing strategies in the indefinitely repeated prisoner’s dilemma game’, European Economic Review. Available at: https://doi.org/10.1016/j.euroecorev.2018.02.008.
"""


# Likelihood function takes as an input a vector of proportions of strategies and returns the likelihood value
def objective(x, args):
    C = args[0]
    E = args[1]

    bc = np.power(x[0], C)  # beta to the power of C
    be = np.power(1 - x[0], E)  # beta to the power of E
    prodBce = np.multiply(bc, be)  # Hadamard product

    # maximum is taken so that there is no log(0) warning/error
    res = np.log(np.maximum(np.dot(x[1:], prodBce), np.nextafter(0, 1))).sum()

    return -res


def constraint(x):
    return x[1:].sum() - 1


def sfem_computation(game_histories, strategies, main_player_name):
    n_games = len(game_histories)
    opponent_name = game_histories[0].get_opponents_names(main_player_name)[0]
    game = GTGame(players={main_player_name: Player(main_player_name), opponent_name: Player(opponent_name)})

    strategies_names = []
    strategies_instances = []
    for strategy_name in strategies.keys():
        if strategy_name in main_hard_coded_strategies:
            strategies_names.append(main_hard_coded_strategies[strategy_name]["label"])
            strategies_instances.append(main_hard_coded_strategies[strategy_name]["strategy"](game, None))
        elif strategy_name == "unfair_random":
            for p in range(6, 10):
                p = p / 10
                strategies_names.append(f"URND{p}")
                strategies_instances.append(main_blind_strategies["unfair_random"]["strategy"](p))
        else:
            strategies_names.append(strategies[strategy_name]["label"])
            strategies_instances.append(strategies[strategy_name]["strategy"]())
    num_strategies = len(strategies_instances)

    scores = np.zeros(num_strategies)
    for game_history in game_histories:
        main_history = game_history.get_actions_by_player(main_player_name)
        # For each strategy k compare LLM's actual play with how strategy k would have played.
        C = np.zeros(num_strategies)  # Number of periods in which play matches
        E = np.zeros(num_strategies)  # Number of periods in which play does not match
        for k in range(num_strategies):
            strategy_history = strategies_instances[k].generate_alternative_history_for_player(game_history, main_player_name)
            delta_history = [0 if main_history[i] == strategy_history[i] else 1 for i in range(len(main_history))]
            C[k] = np.sum(delta_history)
            E[k] = len(main_history) - C[k]

        # Set up the boundaries and constraints
        beta = (np.nextafter(0.5, 1), 1 - np.nextafter(0, 1))  # Beta is at least .5
        phis = (np.nextafter(0, 1), 1 - np.nextafter(0, 1))  # Phis are between 0 and 1
        bounds = tuple([beta] + [phis] * num_strategies)
        constraints = {'type': 'eq', 'fun': constraint}

        # Some random starting point
        x0 = np.zeros(num_strategies + 1)
        x0[0] = .5 + .5 * np.random.random()
        temp = np.random.random(num_strategies)
        x0[1:] = temp / temp.sum()

        bestX = x0
        bestObjective = objective(x0, [C, E])

        for k in range(50):  # Do many times so that there is low chance of getting stuck in local optimum

            x0 = np.zeros(num_strategies + 1)
            x0[0] = .5 + .5 * np.random.random()
            temp = np.random.random(num_strategies)
            x0[1:] = temp / temp.sum()

            # Notice that we are minimizing the negative
            solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, args=([C, E]))
            x = solution.x
            obj = solution.fun
            check = solution.success  # solution.success checks whether optimizer exited successfully

            if bestObjective > obj and check:
                bestObjective = obj
                bestX = x

        scores[np.argmax(bestX[1:]) + 1] += 1

    return [scores[i] / n_games for i in range(num_strategies)]


# player_name = player_1_
# extraction_timestamp = "20240422164401"
# # subdir = Path("game_histories") / player_name
# # file_name = f"game_history_{player_name}-rnd_baseline"
# game_histories_analyzed = extract_histories_from_files(extraction_timestamp, max_infix=100)
#
# strategies_analyzed = get_strategies()
# # strategies_analyzed = get_strategies_by_labels(["URND", "RND"])
# scores = sfem_computation(game_histories_analyzed, strategies_analyzed, player_name)
# for i in range(len(scores)):
#     print(f"{list(strategies_analyzed.keys())[i]} frequency: {scores[i]}")

from src.utils import shutdown_run

shutdown_run()
