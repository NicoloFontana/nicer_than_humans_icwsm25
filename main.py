import datetime
import time

from src.checkers.aggregation_checker import AggregationChecker
from src.checkers.checkers_utils import plot_checkers_results
from src.checkers.rule_checker import RuleChecker
from src.checkers.time_checker import TimeChecker
from src.games.two_players_pd import TwoPlayersPD
from src.player import Player
from src.strategies.basic_pd_strategies import RndStrategy
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy

n_iterations = 100
checkpoint = 10

verbose = False
game = TwoPlayersPD(iterations=n_iterations)
payoff_function = game.get_payoff_function()
game.add_player(Player("Alice"))
game.add_player(Player("Bob"))
start_time = time.time()
timestamp = int(time.time())
checkers = [
    TimeChecker(timestamp),
    RuleChecker(timestamp),
    AggregationChecker(timestamp),
]
checkers_names = [checker.get_name() for checker in checkers]

for iteration in range(n_iterations):
    print("\n\n") if verbose else None
    curr_round = iteration + 1
    print(f"Round {curr_round}")
    if not game.is_ended:
        for player in game.players.values():
            own_history = game.get_history().get_actions_by_player(player.get_name())
            opponent_history = game.get_history().get_actions_by_player(
                [p for p in game.players if p != player.get_name()][0])
            if player.get_name() == "Alice":
                strategy = OneVsOnePDLlmStrategy(game, player.get_name(), game.get_opponent_name(player.get_name()))
                player.set_strategy(strategy, checkers, verbose)
            else:
                player.set_strategy(RndStrategy())
        game.play_round()
        if curr_round % checkpoint == 0 and curr_round < n_iterations:
            for checker in checkers:
                checker.save_results(infix=curr_round)
            plot_checkers_results(checkers_names, timestamp, curr_round, infix=curr_round)
            print(f"Time elapsed: {datetime.timedelta(seconds=int(time.time() - start_time))}")
for checker in checkers:
    checker.save_results()
    checker.save_complete_answers()
print("Results and answers saved.")
print("\n\n") if verbose else None
print(game.get_history()) if verbose else None

plot_checkers_results(checkers_names, timestamp, n_iterations)
print(f"Time elapsed: {datetime.timedelta(seconds=int(time.time() - start_time))}")
