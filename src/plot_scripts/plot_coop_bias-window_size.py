import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st

from src.games.two_players_pd_utils import player_1_, extract_histories_from_files
from src.unused_functions import plot_ts_, plot_fill, save_plot
from src.utils import OUT_BASE_PATH

runs = {
    "20240417174456": 1,
    "20240417174800": 2,
    "20240417175008": 3,
    "20240417175215": 4,
    "20240412104455": 5,
    "20240412100726": 10,
    "20240503171507": 11,
    "20240503171247": 12,
    "20240506100825": 13,
    "20240506101103": 14,
    "20240412103555": 15,
    "20240510104016": 16,
    "20240510104526": 17,
    "20240510105136": 18,
    "20240510104201": 19,
    "20240412104155": 20,
    "20240415144013": 25,
    "20240510105405": 50,
    "20240510105526": 75,
    "20240409114154": 0,
}
confidence = 0.95

mean_coops = []
lbs = []
ubs = []
# std_avg = []
for timestamp in runs.keys():
    window_size = runs[timestamp]
    game_histories = extract_histories_from_files(timestamp)
    main_histories = [game_history.get_actions_by_player(player_1_) for game_history in game_histories]
    mean_main_histories = [sum(main_history[:-10]) / len(main_history[:-10]) for main_history in main_histories]
    ci = st.norm.interval(confidence=confidence, loc=np.mean(mean_main_histories), scale=st.sem(mean_main_histories))
    lbs.append(ci[0])
    ubs.append(ci[1])
    mean_coops.append(sum(mean_main_histories) / len(mean_main_histories))

plt_fig = plot_ts_(mean_coops, "blue", "Average cooperativeness", axhlines=[0.0, 0.5, 1.0])
plt_fig = plot_fill(lbs, ubs, "blue", fig=plt_fig, alpha=0.3)
n_wdws = len(mean_coops)

plt.title("LLM cooperation bias and window size")
plt.xlabel("Window size")
plt.ylabel("Average cooperativeness")
plt.xticks([i for i in range(n_wdws)], [f"{runs[timestamp]}" if runs[timestamp] != 0 else "100" for timestamp in runs.keys()])
plt.legend()
plt.tight_layout()
save_plot(plt_fig, OUT_BASE_PATH / "cooperation_bias-window_size_last10")
plt.show()

from src.utils import shutdown_run

shutdown_run()
