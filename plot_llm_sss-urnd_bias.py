from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st

from src.games.two_players_pd_utils import player_1_, player_2_, plot_ts_, plot_fill, extract_histories_from_files_
from src.llm_utils import history_window_size

history_main_name = player_1_
history_opponent_name = player_2_
cmap = plt.get_cmap('Dark2')
confidence = 0.95

min_urnd_coop = 0
max_urnd_coop = 9

means = []
cis = []
base_path = Path("behavioral_profiles_analysis")
strat_name = "llama2"
strat_dir_path = base_path / strat_name
for i in range(min_urnd_coop, max_urnd_coop):
    p = i / 10
    p_str = str(p).replace(".", "")
    opponent_name = f"URND{p_str}"
    urnd_dir_path = strat_dir_path / opponent_name
    game_histories_dir_path = urnd_dir_path / "game_histories"
    # file_name = f"game_history_{strat_name}-{opponent_name}"
    file_name = f"game_history"
    game_histories = extract_histories_from_files_(game_histories_dir_path, file_name)

    main_history_ends = [game_history.get_actions_by_player(history_main_name)[10:] for game_history in game_histories]
    mean_main_history_ends = [sum(main_history_end) / len(main_history_end) for main_history_end in main_history_ends]
    mean = sum(mean_main_history_ends) / len(mean_main_history_ends)
    means.append(mean)
    ci = st.norm.interval(confidence=confidence, loc=np.mean(mean_main_history_ends), scale=st.sem(mean_main_history_ends))
    cis.append(ci)
plt_fig = plot_ts_(means, "blue", "mean cooperation", axhlines=[0.0, 0.5, 1.0])
plt_fig = plot_fill([ci[0] for ci in cis], [ci[1] for ci in cis], "blue", plt_figure=plt_fig, alpha=0.3)
plt.figure(plt_fig)
plt.xticks(range(min_urnd_coop, max_urnd_coop), [str(i / 10) for i in range(min_urnd_coop, max_urnd_coop)])
plt.xlabel("URND cooperation")
plt.ylabel("Llama2 cooperation")
plt.title(f"Llama2 vs URND variations in last {history_window_size} iterations")
plt.tight_layout()
plt.legend()
plt.show()


from src.utils import shutdown_run

shutdown_run()
