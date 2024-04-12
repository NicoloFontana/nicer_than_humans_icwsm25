import datetime as dt
import sys
import time
from pathlib import Path

from huggingface_hub import InferenceClient

from src.checkers.aggregation_checker import AggregationChecker
from src.llm_utils import plot_checkers_results, HF_API_TOKEN, MODEL, plot_confusion_matrix_for_question, history_window_size
from src.games.two_players_pd_utils import player_1_, player_2_
from src.checkers.rule_checker import RuleChecker
from src.checkers.time_checker import TimeChecker
from src.games.two_players_pd import TwoPlayersPD
from src.player import Player
from src.strategies.blind_pd_strategies import RndStrategy, AlwaysCoopStrategy, AlwaysDefectStrategy
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy
from src.utils import timestamp, log, start_time, dt_start_time

n_games = 50
n_iterations = 100
checkpoint = 0
verbose = False
checkers = False
save = True
msg = "Run LLM against Always-Defect strategy with sliding window of 15."

if msg == "":
    log.info("Set a message.")
    sys.exit()
log.info(msg)
print(msg)

log.info(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sleeping routine # TODO remove
print("Going to sleep")
time.sleep(30000)
new_dt_start_time = dt.datetime.now()
new_start_time = time.mktime(dt_start_time.timetuple())
log.info(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")


for n_game in range(n_games):
    log.info(f"Game {n_game + 1}") if n_games > 1 else None
    print(f"Game {n_game + 1}") if n_games > 1 else None
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
    game = TwoPlayersPD(iterations=n_iterations)
    # payoff_function = game.get_payoff_function()
    game.add_player(Player(player_1_))
    game.add_player(Player(player_2_))
    strategy = OneVsOnePDLlmStrategy(game, player_1_, game.get_opponent_name(player_1_), client)
    for iteration in range(n_iterations):
        curr_round = iteration + 1
        log.info(f"Round {curr_round}") if n_games == 1 or curr_round % 10 == 0 else None
        print(f"Round {curr_round}") if n_games == 1 or curr_round % 10 == 0 else None
        if not game.is_ended:
            for player in game.players.values():
                # own_history = game.get_history().get_actions_by_player(player.get_name())
                # opponent_history = game.get_history().get_actions_by_player(
                #     [p for p in game.players if p != player.get_name()][0])
                if player.get_name() == player_1_:
                    player.set_strategy(strategy, verbose)
                else:
                    player.set_strategy(AlwaysDefectStrategy())
            game.play_round()
            game.get_player_by_name(player_1_).get_strategy().ask_questions(checkers, game, history_window_size=history_window_size, verbose=verbose)
            if checkpoint != 0 and curr_round % checkpoint == 0 and curr_round < n_iterations:
                if save:
                    for checker in checkers:
                        checker.save_results(infix=curr_round)
                        checker.save_complete_answers(infix=curr_round)
                        # labels = checker.questions_labels
                        # for label in labels:
                        #     plot_confusion_matrix_for_question(checker.dir_path, label, infix=curr_round)
                    plot_checkers_results(checkers_names, timestamp, curr_round, infix=curr_round)
                    infix = f"{n_game+1}_{curr_round}" if n_games > 1 else curr_round
                    strategy.save_action_answers(infix=infix)
                log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
                print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
    if save:
        for checker in checkers:
            checker.save_results()
            checker.save_complete_answers()
            # labels = checker.questions_labels
            # for label in labels:
            #     plot_confusion_matrix_for_question(checker.dir_path, label)
        plot_checkers_results(checkers_names, timestamp, n_iterations)
        infix = n_game + 1 if n_games > 1 else None
        game.save_history(timestamp, infix=infix)
        strategy.save_action_answers(infix=infix)

    log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # TODO remove "new"
    print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # TODO remove "new"
