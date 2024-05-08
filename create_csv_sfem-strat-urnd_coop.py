from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from sfem_computation import compute_sfem
from src.games.two_players_pd_utils import player_1_, extract_histories_from_files_
from src.strategies.strategy_utils import get_strategies, plot_errorbar



base_path = Path("behavioral_profiles_analysis")
strat_name = "llama2"
history_main_name = player_1_
strat_dir_path = base_path / strat_name
cmap = plt.get_cmap('Dark2')
plt_fig = plt.figure()

csv_dir_path = Path("csv_files_for_plots") / "sfem-urnd_coop"
csv_dir_path.mkdir(parents=True, exist_ok=True)

available_strategies = get_strategies()
strategies_names = list(available_strategies.keys())
csv_file = []
for i in range(0, 11):
    p = i / 10
    element = {
        "URND_coop": p,
    }
    p_str = str(p).replace(".", "")
    opponent_name = f"URND{p_str}"
    print(opponent_name)
    urnd_dir_path = strat_dir_path / opponent_name
    game_histories_dir_path = urnd_dir_path / "game_histories"
    # file_name = f"game_history_{strat_name}-{opponent_name}"
    file_name = f"game_history"
    game_histories = extract_histories_from_files_(game_histories_dir_path, file_name)

    scores, strategies_names = compute_sfem(game_histories, available_strategies, history_main_name)

    for score, strategy_name in zip(scores, strategies_names):
        element[f"{strategy_name}_score"] = score
    csv_file.append(element)
df = pd.DataFrame(csv_file)
df.to_csv(csv_dir_path / f"llama2_sfem-urnd_coop.csv")

from src.utils import shutdown_run

shutdown_run()
