import datetime as dt
import sys
import time

from huggingface_hub import InferenceClient

from src.checkers.aggregation_checker import AggregationChecker
from src.llm_utils import plot_checkers_results, HF_API_TOKEN, MODEL
from src.checkers.rule_checker import RuleChecker
from src.checkers.time_checker import TimeChecker
from src.games.two_players_pd import TwoPlayersPD
from src.player import Player
from src.strategies.basic_pd_strategies import RndStrategy
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy
from src.utils import timestamp, log

n_iterations = 100
checkpoint = 10
ALICE = "Alice"
verbose = False
checkers = True
save = True
msg = "Trying natural language format with A-vs-B perspective."


if msg == "":
    log.info("Set a message.")
    sys.exit()
log.info(msg)
game = TwoPlayersPD(iterations=n_iterations)
# payoff_function = game.get_payoff_function()
game.add_player(Player(ALICE))
game.add_player(Player("Bob"))
start_time = time.time()
log.info(f"Starting time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Starting time: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
if checkers:
    checkers = [
        TimeChecker(timestamp),
        RuleChecker(timestamp),
        AggregationChecker(timestamp),
    ]
else:
    checkers = []
checkers_names = [checker.get_name() for checker in checkers]
client = InferenceClient(model=MODEL, token=HF_API_TOKEN)
client.headers["x-use-cache"] = "0"
strategy = OneVsOnePDLlmStrategy(game, ALICE, game.get_opponent_name(ALICE), client)

for iteration in range(n_iterations):
    curr_round = iteration + 1
    log.info(f"Round {curr_round}")
    print(f"Round {curr_round}")
    if not game.is_ended:
        for player in game.players.values():
            # own_history = game.get_history().get_actions_by_player(player.get_name())
            # opponent_history = game.get_history().get_actions_by_player(
            #     [p for p in game.players if p != player.get_name()][0])
            if player.get_name() == ALICE:
                player.set_strategy(strategy, verbose)
            else:
                player.set_strategy(RndStrategy())
        game.play_round()
        game.get_player_by_name(ALICE).get_strategy().ask_questions(checkers, game, verbose)
        if checkpoint != 0 and curr_round % checkpoint == 0 and curr_round < n_iterations:
            if save:
                for checker in checkers:
                    checker.save_results(infix=curr_round)
                plot_checkers_results(checkers_names, timestamp, curr_round, infix=curr_round)
            log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
            print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
if save:
    game.save_history(timestamp)
    for checker in checkers:
        checker.save_results()
        checker.save_complete_answers()
    plot_checkers_results(checkers_names, timestamp, n_iterations)
log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
