import datetime as dt
import sys
import time

from src.analysis.aggregation_checker import AggregationChecker
from src.llm_utils import KEY, MODEL_URL, history_window_size, MODEL_NAME, PROVIDER
from src.game.two_players_pd_utils import player_1_, player_2_
from src.analysis.rule_checker import RuleChecker
from src.analysis.time_checker import TimeChecker
from src.game.two_players_pd import TwoPlayersPD
from src.model_client import ModelClient
from src.game.player import Player
from src.strategies.blind_pd_strategies import RandomStrategy, AlwaysCooperate, AlwaysDefect, FixedSequence, UnfairRandom
from src.strategies.hard_coded_pd_strategies import TitForTat, Grim, WinStayLoseShift, SuspiciousTitForTat
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy
from src.utils import timestamp, log, start_time, OUT_BASE_PATH

n_games = 50
n_iterations = 100
checkpoint = 0
checkers = False
t = 0.7

msg = "Run llama3 vs URND[1.0] for 50 games, 100 iterations, window size 10 with fixed prompt."

if msg == "":
    log.info("Set a message.")
    sys.exit()
log.info(msg)
print(msg)

# log.info(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#
# # Sleeping routine
# log.info("Going to sleep")
# print("Going to sleep")
# time.sleep(129600)
new_dt_start_time = dt.datetime.now()
new_start_time = time.mktime(new_dt_start_time.timetuple())
log.info(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
# daily_requests = 3275

for p in {10}:
    coop_prob = p / 10
    urnd_str = f"URND{p:02}"
    print(f"URND{p:02}")
    log.info(f"URND{p:02}")
    for n_game in range(50, 50+n_games):
        log.info(f"Game {n_game + 1}") if n_games > 1 else None
        print(f"Game {n_game + 1}") if n_games > 1 else None
        checkers = [
            TimeChecker(),
            RuleChecker(),
            AggregationChecker(),
        ] if checkers else []

        model_client = ModelClient(model_name=MODEL_NAME, model_url=MODEL_URL, api_key=KEY, provider=PROVIDER)
        # model_client.daily_requests = daily_requests

        # Set up the game
        game = TwoPlayersPD(iterations=n_iterations)

        llm_player = Player(player_1_)
        llm_strategy = OneVsOnePDLlmStrategy(game, model_client, history_window_size=history_window_size, checkers=checkers)
        llm_strategy.set_temperature(t)
        llm_player.set_strategy(llm_strategy)
        game.add_player(llm_player)

        second_player = Player(player_2_)
        second_player_strategy = UnfairRandom(coop_prob)
        second_player.set_strategy(second_player_strategy)
        game.add_player(second_player)

        for iteration in range(n_iterations):
            curr_round = iteration + 1
            log.info(f"Round {curr_round}") if n_games == 1 or curr_round % 10 == 0 else None
            print(f"Round {curr_round}") if n_games == 1 or curr_round % 10 == 0 else None
            if not game.is_ended:
                game.play_game_round()
                out_dir = OUT_BASE_PATH / str(timestamp) / urnd_str if checkpoint != 0 and curr_round % checkpoint == 0 else None
                infix = f"{n_game + 1}_{curr_round}" if n_games > 1 else curr_round
                # infix = f"{n_game + 1}_{curr_round}" if n_games > 1 else curr_round
                llm_strategy.wrap_up_round(out_dir=out_dir, infix=infix)
                log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if out_dir is not None else None
                print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if out_dir is not None else None
        out_dir = OUT_BASE_PATH / str(timestamp) / urnd_str
        infix = f"{n_game + 1}" if n_games > 1 else None
        # infix = f"{n_game + 1}" if n_games > 1 else None
        llm_strategy.wrap_up_round(out_dir=out_dir, infix=infix)
        game.save_history(out_dir=out_dir, infix=infix)

        # log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")
        # print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")

        # daily_requests = model_client.daily_requests
        # time.sleep(60)

        log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")
        print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")
