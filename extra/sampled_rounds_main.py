import random
import datetime as dt
import sys
import time
from pathlib import Path

import pandas as pd

from src.game.game_history import GameHistory
from src.game.player import Player
from src.game.two_players_pd import TwoPlayersPD
from src.llm_utils import MODEL_NAME, MODEL_URL, KEY, PROVIDER, history_window_size
from src.model_client import ModelClient
from src.strategies.blind_pd_strategies import UnfairRandom
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy
from src.utils import log, OUT_BASE_PATH, timestamp

model = "llama3"
initial_game = 0
n_games = 100
sampled_rounds = [1, 10, 14, 15, 38, 50, 54, 80, 95, 100]
# sampled_rounds = {1, 50, 100}
# for i in range(7):
#     rnd_round = random.choice([ele for ele in range(1, 100) if ele not in sampled_rounds])
#     sampled_rounds.add(rnd_round)
# sampled_rounds = list(sampled_rounds)
# sampled_rounds.sort()

msg = f"Test temperature 1.0 for GPT3.5 on sampled rounds {sampled_rounds} for alpha 0.9,1.0, for every game."

if msg == "":
    log.info("Set a message.")
    sys.exit()
log.info(msg)
print(msg)

# new_dt_start_time = dt.datetime.now()
# log.info(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
#
# # Sleeping routine
# log.info("Going to sleep")
# print("Going to sleep")
# time.sleep(21600)

new_dt_start_time = dt.datetime.now()
new_start_time = time.mktime(new_dt_start_time.timetuple())
log.info(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Starting time: {new_dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

model_client = ModelClient(model_name=MODEL_NAME, model_url=MODEL_URL, api_key=KEY, provider=PROVIDER)

for t in {1.0}:
    temp = f"{t:.1f}".replace(".", "")
    temp_dir = f"T{temp}"
    log.info(f"Temperature: {t}")
    print(f"Temperature: {t}")
    for p in range(9, 11):
        coop_prob = p / 10
        urnd_dir = f"URND{p:02}"
        print(f"URND{p:02}")
        log.info(f"URND{p:02}")
        csv_dir = OUT_BASE_PATH / str(timestamp) / temp_dir / urnd_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
        if p == 9:
            initial_game = 57
        else:
            initial_game = 0
        for n_game in range(initial_game, n_games):
            log.info(f"Game {n_game + 1}") if n_games > 1 else None
            print(f"Game {n_game + 1}") if n_games > 1 else None

            dir_path = Path("../relevant_runs_copies") / "main_runs_copies" / f"{model}_URNDx_ws10" / urnd_dir / "game_histories"
            file_name = f"game_history_{n_game + 1}"
            game_history = GameHistory()
            file_path = dir_path / f"{file_name}.json"

            sample_csv = []

            for sampled_round in sampled_rounds:
                log.info(f"Sampled round: {sampled_round}")
                print(f"Sampled round: {sampled_round}")
                game_history.load_from_file_with_limit(file_path, sampled_round - 1)
                first_player_name = game_history.get_players_names()[0]
                second_player_name = game_history.get_players_names()[1]

                tmp_gh = GameHistory()
                tmp_gh.load_from_file_with_limit(file_path, sampled_round)
                original_move = tmp_gh.get_actions_by_player(first_player_name)[-1]

                game = TwoPlayersPD(iterations=100)

                llm_player = Player(first_player_name)
                llm_strategy = OneVsOnePDLlmStrategy(game, model_client, history_window_size=history_window_size)
                llm_strategy.set_temperature(t)
                llm_player.set_strategy(llm_strategy)
                game.add_player(llm_player)

                second_player = Player(second_player_name)
                second_player_strategy = UnfairRandom(0.5)
                second_player.set_strategy(second_player_strategy)
                game.add_player(second_player)

                game.set_history(game_history)

                game.play_game_round()
                llm_strategy.wrap_up_round(out_dir=csv_dir, infix=f"{n_game + 1}_{sampled_round}")

                new_move = game.get_actions_by_player(first_player_name)[-1]

                sample = {
                    "sampled_round": sampled_round,
                    "original_move": original_move,
                    "new_move": new_move,
                }
                sample_csv.append(sample)

            df = pd.DataFrame(sample_csv)
            df.to_csv(csv_dir / f"sample_game_{n_game + 1}.csv")

        log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")
        print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - new_start_time))}")
