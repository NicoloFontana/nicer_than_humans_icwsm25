import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.games.two_players_pd_utils import extract_histories_from_files, player_1_
from src.utils import OUT_BASE_PATH, compute_average_vector, compute_confidence_interval_vectors

# no_wdw_extraction_timestamp = "20240328173931-20240329014202"  # vs RND
# no_wdw_extraction_timestamp = "20240409114154"  # vs ALLD
# no_wdw_extraction_timestamp = "20240523140851"  # gpt35t vs ALLD
no_wdw_extraction_timestamp = "20240529102526"  # llama3 vs ALLD
no_wdw_path = Path("../../") / OUT_BASE_PATH / no_wdw_extraction_timestamp / "game_histories"
confidence = 0.95
csv_dir_path = Path("../../csv_files_for_plots") / "gpt35t_coop_vs_opponent"
csv_dir_path.mkdir(parents=True, exist_ok=True)

no_wdw_game_histories = extract_histories_from_files(no_wdw_path, "game_history")
no_wdw_main_histories = [game_history.get_actions_by_player(player_1_) for game_history in no_wdw_game_histories]
no_wdw_avg_main_history = compute_average_vector(no_wdw_main_histories)
no_wdw_ci_lbs, no_wdw_ci_ubs = compute_confidence_interval_vectors(no_wdw_main_histories, confidence=confidence)

# wdw_extraction_timestamp = "20240523181505"  # gpt35t vs ALLD
wdw_extraction_timestamp = "20240529151748"  # llama3 vs ALLD
# wdw_path = Path("../../behavioral_profiles_analysis") / "llama2" / "URND05" / "game_histories"  # vs RND
# wdw_path = Path("../../behavioral_profiles_analysis") / "llama2" / "URND00" / "game_histories"  # vs ALLD
wdw_path = Path("../../") / OUT_BASE_PATH / wdw_extraction_timestamp / "game_histories"  # gpt35t vs ALLD
wdw_game_histories = extract_histories_from_files(wdw_path, "game_history")
wdw_main_histories = [game_history.get_actions_by_player(player_1_) for game_history in wdw_game_histories]
wdw_avg_main_history = compute_average_vector(wdw_main_histories)
wdw_ci_lbs, wdw_ci_ubs = compute_confidence_interval_vectors(wdw_main_histories, confidence=confidence)

csv_file = []
for iteration_idx in range(len(no_wdw_main_histories[0])):
    iteration = {
        "iteration": iteration_idx+1,
        "no_wdw_mean": no_wdw_avg_main_history[iteration_idx],
        "no_wdw_ci_lb": no_wdw_ci_lbs[iteration_idx] if not np.isnan(no_wdw_ci_lbs[iteration_idx]) else no_wdw_avg_main_history[iteration_idx],
        "no_wdw_ci_ub": no_wdw_ci_ubs[iteration_idx] if not np.isnan(no_wdw_ci_ubs[iteration_idx]) else no_wdw_avg_main_history[iteration_idx],
        "wdw_mean": wdw_avg_main_history[iteration_idx],
        "wdw_ci_lb": wdw_ci_lbs[iteration_idx] if not np.isnan(wdw_ci_lbs[iteration_idx]) else wdw_avg_main_history[iteration_idx],
        "wdw_ci_ub": wdw_ci_ubs[iteration_idx] if not np.isnan(wdw_ci_ubs[iteration_idx]) else wdw_avg_main_history[iteration_idx],
    }
    csv_file.append(iteration)
    df = pd.DataFrame(csv_file)
    df.to_csv(csv_dir_path / f"llama3_vs_ad_wdw-no_wdw.csv")


from src.utils import shutdown_run

shutdown_run()
