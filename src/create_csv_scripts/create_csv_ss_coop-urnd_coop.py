from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st

from src.games.two_players_pd_utils import player_1_, player_2_, plot_ts_, plot_fill, extract_histories_from_files_
from src.llm_utils import history_window_size

history_main_name = player_1_
history_opponent_name = player_2_
cmap = plt.get_cmap('Dark2')
confidence = 0.95

csv_dir_path = Path("../../csv_files_for_plots") / "steady_state_coop-urnd_coop"
csv_dir_path.mkdir(parents=True, exist_ok=True)

min_urnd_coop = 0
max_urnd_coop = 11

base_path = Path("../../behavioral_profiles_analysis")
strat_name = "llama2"
strat_dir_path = base_path / strat_name
csv_file = []
for i in range(min_urnd_coop, max_urnd_coop):
    p = i / 10
    p_str = str(p).replace(".", "")
    opponent_name = f"URND{p_str}"
    urnd_dir_path = strat_dir_path / opponent_name
    game_histories_dir_path = urnd_dir_path / "game_histories"
    # file_name = f"game_history_{strat_name}-{opponent_name}"
    file_name = f"game_history"
    game_histories = extract_histories_from_files_(game_histories_dir_path, file_name)

    main_history_ends = [game_history.get_actions_by_player(history_main_name)[history_window_size:] for game_history in game_histories]
    mean_main_history_ends = [sum(main_history_end) / len(main_history_end) for main_history_end in main_history_ends]
    mean = sum(mean_main_history_ends) / len(mean_main_history_ends)
    ci = st.norm.interval(confidence=confidence, loc=np.mean(mean_main_history_ends), scale=st.sem(mean_main_history_ends))
    element = {
        "URND_coop": p,
        "Llama2_coop": mean,
        "ci_lb": ci[0] if not np.isnan(ci[0]) else mean,
        "ci_ub": ci[1] if not np.isnan(ci[1]) else mean
    }

    csv_file.append(element)
df = pd.DataFrame(csv_file)
df.to_csv(csv_dir_path / f"llama2_steady_state_coop-urnd_coop.csv")


from src.utils import shutdown_run

shutdown_run()
