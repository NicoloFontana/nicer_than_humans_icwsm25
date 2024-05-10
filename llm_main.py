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
from src.strategies.blind_pd_strategies import RandomStrategy, AlwaysCooperate, AlwaysDefect, FixedSequence, UnfairRandom
from src.strategies.hard_coded_pd_strategies import TitForTat
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy
from src.utils import timestamp, log, start_time, dt_start_time

n_games = 50
n_iterations = 100
checkpoint = 0
# checkers = False
msg = "Run LLM against ALLD with window size 50"
# coop_prob = 0.9

if msg == "":
    log.info("Set a message.")
    sys.exit()
log.info(msg)
print(msg)

log.info(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Sleeping routine # TODO remove
log.info("Going to sleep")
print("Going to sleep")
time.sleep(172000)
new_dt_start_time = dt.datetime.now()
new_start_time = time.mktime(new_dt_start_time.timetuple())
log.info(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

for n_game in range(n_games):
    log.info(f"Game {n_game + 1}") if n_games > 1 else None
    print(f"Game {n_game + 1}") if n_games > 1 else None
    # checkers = [
    #     TimeChecker(),
    #     RuleChecker(),
    #     AggregationChecker(),
    # ] if checkers else []
    game = TwoPlayersPD(iterations=n_iterations)
    game.add_player(Player(player_1_))
    game.add_player(Player(player_2_))
    client = InferenceClient(model=MODEL, token=HF_API_TOKEN)
    client.headers["x-use-cache"] = "0"
    strat1 = OneVsOnePDLlmStrategy(game, player_1_, client, history_window_size=history_window_size)  # TODO checkers
    strat2 = AlwaysDefect()
    for player in game.players.values():
        if player.get_name() == player_1_:
            player.set_strategy(strat1)
        else:
            player.set_strategy(strat2)
    for iteration in range(n_iterations):
        curr_round = iteration + 1
        log.info(f"Round {curr_round}") if n_games == 1 or curr_round % 10 == 0 else None
        print(f"Round {curr_round}") if n_games == 1 or curr_round % 10 == 0 else None
        if not game.is_ended:
            game.play_round()
            save = checkpoint != 0 and curr_round % checkpoint == 0
            infix = f"{n_game + 1}_{curr_round}" if n_games > 1 else curr_round
            strat1.wrap_up_round(save=save, infix=infix)
            log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if save else None
            print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if save else None
    infix = n_game + 1 if n_games > 1 else None
    strat1.wrap_up_round(save=True, infix=infix)
    game.save_history(timestamp, infix=infix)

    # log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
    # print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")

    log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # TODO remove "new"
    print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")  # TODO remove "new"
