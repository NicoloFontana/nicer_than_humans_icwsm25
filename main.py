import datetime as dt
import sys
import time
from pathlib import Path

from huggingface_hub import InferenceClient

from src.checkers.aggregation_checker import AggregationChecker
from src.llm_utils import plot_checkers_results, HF_API_TOKEN, MODEL, plot_confusion_matrix_for_question
from src.games.two_players_pd_utils import player_1_, player_2_
from src.checkers.rule_checker import RuleChecker
from src.checkers.time_checker import TimeChecker
from src.games.two_players_pd import TwoPlayersPD
from src.player import Player
from src.strategies.basic_pd_strategies import RndStrategy
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy
from src.utils import timestamp, log, start_time, dt_start_time

n_iterations = 100
checkpoint = 10
verbose = False
checkers = False
save = True
msg = "Run without questions. Defect, Cooperate actions."

if msg == "":
    log.info("Set a message.")
    sys.exit()
log.info(msg)
print(msg)
game = TwoPlayersPD(iterations=n_iterations)
# payoff_function = game.get_payoff_function()
game.add_player(Player(player_1_))
game.add_player(Player(player_2_))
time.sleep(86400)  # TODO remove this line. Added just to avoid having too many jobs on hpc querying the model.
new_dt_start_time = dt.datetime.now()  # Change
new_start_time = time.mktime(dt_start_time.timetuple())  # Change
log.info(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")  # Change
print(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")  # Change
if checkers:
    checkers = [
        TimeChecker(),
        RuleChecker(),
        AggregationChecker(),
    ]
else:
    checkers = []
checkers_names = [checker.get_name() for checker in checkers]
client = InferenceClient(model=MODEL, token=HF_API_TOKEN)
client.headers["x-use-cache"] = "0"
strategy = OneVsOnePDLlmStrategy(game, player_1_, game.get_opponent_name(player_1_), client)

for iteration in range(n_iterations):
    curr_round = iteration + 1
    log.info(f"Round {curr_round}")
    print(f"Round {curr_round}")
    if not game.is_ended:
        for player in game.players.values():
            # own_history = game.get_history().get_actions_by_player(player.get_name())
            # opponent_history = game.get_history().get_actions_by_player(
            #     [p for p in game.players if p != player.get_name()][0])
            if player.get_name() == player_1_:
                player.set_strategy(strategy, verbose)
            else:
                player.set_strategy(RndStrategy())
        game.play_round()
        game.get_player_by_name(player_1_).get_strategy().ask_questions(checkers, game, verbose)
        if checkpoint != 0 and curr_round % checkpoint == 0 and curr_round < n_iterations:
            if save:
                for checker in checkers:
                    checker.save_results(infix=curr_round)
                    checker.save_complete_answers(infix=curr_round)
                    # labels = checker.questions_labels
                    # for label in labels:
                    #     plot_confusion_matrix_for_question(checker.dir_path, label, infix=curr_round)
                plot_checkers_results(checkers_names, timestamp, curr_round, infix=curr_round)
                strategy.save_action_answers(infix=curr_round)
            log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # Change
            print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # Change
if save:
    game.save_history(timestamp)
    for checker in checkers:
        checker.save_results()
        checker.save_complete_answers()
        # labels = checker.questions_labels
        # for label in labels:
        #     plot_confusion_matrix_for_question(checker.dir_path, label)
    plot_checkers_results(checkers_names, timestamp, n_iterations)
    strategy.save_action_answers()
log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # Change
print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # Change
