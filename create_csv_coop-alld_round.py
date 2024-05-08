import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src.games.two_players_pd_utils import extract_histories_from_files_, player_1_
from src.utils import OUT_BASE_PATH, compute_average_vector, compute_confidence_interval_vectors

extraction_timestamp = "20240409114154"
confidence = 0.95
csv_dir_path = Path("csv_files_for_plots") / "coop-alld_round"
csv_dir_path.mkdir(parents=True, exist_ok=True)

game_histories = extract_histories_from_files_(OUT_BASE_PATH / extraction_timestamp / "game_histories", "game_history")
main_histories = [game_history.get_actions_by_player(player_1_) for game_history in game_histories]
avg_main_history = compute_average_vector(main_histories)
ci_lbs, ci_ubs = compute_confidence_interval_vectors(main_histories, confidence=confidence)
csv_file = []
for iteration_idx in range(len(main_histories[0])):
    fieldnames = ["iteration", "mean", "ci_lb", "ci_ub"]
    iteration = {
        "iteration": iteration_idx+1,
        "mean": avg_main_history[iteration_idx],
        "ci_lb": ci_lbs[iteration_idx] if not np.isnan(ci_lbs[iteration_idx]) else avg_main_history[iteration_idx],
        "ci_ub": ci_ubs[iteration_idx] if not np.isnan(ci_ubs[iteration_idx]) else avg_main_history[iteration_idx]
    }
    csv_file.append(iteration)
    df = pd.DataFrame(csv_file)
    df.to_csv(csv_dir_path / f"llama2_coop_per_round_vs_ALLD.csv")


from src.utils import shutdown_run

shutdown_run()
