import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

from src.games.two_players_pd_utils import extract_histories_from_files, player_1_, extract_histories_from_files
from src.utils import OUT_BASE_PATH, compute_average_vector, compute_confidence_interval_vectors

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
csv_dir_path = Path("../../csv_files_for_plots") / "coop_bias-window_size"
csv_dir_path.mkdir(parents=True, exist_ok=True)

csv_file = []
for timestamp in runs.keys():
    window_size = runs[timestamp]
    game_histories = extract_histories_from_files(Path("../../") / OUT_BASE_PATH / str(timestamp) / "game_histories")
    main_histories = [game_history.get_actions_by_player(player_1_) for game_history in game_histories]
    mean_main_histories = [sum(main_history[-10:]) / len(main_history[-10:]) for main_history in main_histories]
    ci = st.norm.interval(confidence=confidence, loc=np.mean(mean_main_histories), scale=st.sem(mean_main_histories))
    mean = sum(mean_main_histories) / len(mean_main_histories)
    element = {
        "window_size": window_size if window_size != 0 else 100,
        "mean": mean,
        "ci_lb": ci[0] if not np.isnan(ci[0]) else mean,
        "ci_ub": ci[1] if not np.isnan(ci[1]) else mean
    }
    csv_file.append(element)
    df = pd.DataFrame(csv_file)
    df.to_csv(csv_dir_path / f"llama2_coop_bias_per_window_size_fixed.csv")


from src.utils import shutdown_run

shutdown_run()
