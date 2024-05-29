from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st

from src.games.two_players_pd_utils import player_1_, player_2_, extract_histories_from_files
from src.unused_functions import plot_ts_, plot_fill
from src.llm_utils import history_window_size
from src.utils import OUT_BASE_PATH

history_main_name = player_1_
history_opponent_name = player_2_
cmap = plt.get_cmap('Dark2')
confidence = 0.95

min_urnd_coop = 0
max_urnd_coop = 11
#
# means = []
# cis = []
# base_path = Path("../../behavioral_profiles_analysis")
# strat_name = "llama2"
# strat_dir_path = base_path / strat_name
#
# # opponent_name = "URND05"  # vs RND
# opponent_name = "URND00"  # vs ALLD
# urnd_dir_path = strat_dir_path / opponent_name
# game_histories_dir_path = urnd_dir_path / "game_histories"
# file_name = f"game_history"
# game_histories = extract_histories_from_files_(game_histories_dir_path, file_name)
#
# main_histories = [game_history.get_actions_by_player(history_main_name) for game_history in game_histories]
# n_iterations = len(main_histories[0])
# mean_history = []
# lb_history = []
# ub_history = []
# for iteration in range(n_iterations):
#     iterations = [main_history[iteration] for main_history in main_histories]
#     mean_iteration = sum(iterations) / len(iterations)
#     mean_history.append(mean_iteration)
#     ci = st.norm.interval(confidence=confidence, loc=np.mean(iterations), scale=st.sem(iterations))
#     lb_history.append(ci[0])
#     ub_history.append(ci[1])
#
# plt_fig = plot_ts_(mean_history, "blue", "window size 10", axhlines=[0.0, 0.5, 1.0])
# plt_fig = plot_fill(lb_history, ub_history, "blue", fig=plt_fig, alpha=0.1)

# timestamp = "20240328173931-20240329014202"  # vs RND
# timestamp = "20240409114154"  # vs ALLD
# timestamp = "20240523140851"  # gpt35t vs ALLD
timestamp = "20240523181505"  # gpt35t vs ALLD wdw=10
# timestamp = "20240524133431"  # gpt35t vs ALLD IIPD
no_wdw_path = OUT_BASE_PATH / timestamp / "game_histories"
no_wdw_path = Path("C:\\Users\\fonta\\PycharmProjects\\masters_thesis_PoliMi_ITU\\out") / timestamp / "game_histories"
no_wdw_file = "game_history"
no_wdw_histories = extract_histories_from_files(no_wdw_path, no_wdw_file)

no_wdw_main_histories = [game_history.get_actions_by_player(history_main_name) for game_history in no_wdw_histories]
n_iterations = len(no_wdw_main_histories[0])
no_wdw_mean_history = []
no_wdw_lb_history = []
no_wdw_ub_history = []
for iteration in range(n_iterations):
    iterations = [main_history[iteration] for main_history in no_wdw_main_histories]
    mean_iteration = sum(iterations) / len(iterations)
    no_wdw_mean_history.append(mean_iteration)
    ci = st.norm.interval(confidence=confidence, loc=np.mean(iterations), scale=st.sem(iterations))
    no_wdw_lb_history.append(ci[0])
    no_wdw_ub_history.append(ci[1])

plt_fig = plot_ts_(no_wdw_mean_history, "red", "no sliding window", fig=0, linestyle=":", marker=",")
plt_fig = plot_fill(no_wdw_lb_history, no_wdw_ub_history, "red", fig=plt_fig, alpha=0.1)
plt.figure(plt_fig)
plt.xlabel("Round")
plt.ylabel("Average Llama2 cooperation")
# plt.title(f"Llama2 vs RND")  # vs RND
plt.title(f"Llama2 vs ALLD")  # vs ALLD
plt.tight_layout()
plt.legend()
plt.show()


from src.utils import shutdown_run

shutdown_run()
