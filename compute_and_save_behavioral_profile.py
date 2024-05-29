from pathlib import Path

from src.behavioral_analysis.main_behavioral_dimensions import main_behavioral_dimensions
from src.games.two_players_pd_utils import extract_histories_from_files
from src.strategies.strategy_utils import main_hard_coded_strategies, compute_behavioral_profile_
from src.games.two_players_pd_utils import player_2_, player_1_

base_path = Path("behavioral_profiles_analysis")
# strats = main_hard_coded_strategies
strats = {"llama2": None}

for strat_name in strats.keys():
    strat_dir_path = base_path / strat_name
    for i in range(0, 11):
        p = i / 10
        p_str = str(p).replace(".", "")
        opponent_name = f"URND{p_str}"
        urnd_dir_path = strat_dir_path / opponent_name
        game_histories_dir_path = urnd_dir_path / "game_histories"
        # file_name = f"game_history_{strat_name}-{opponent_name}"
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)

        history_main_name = player_1_
        history_opponent_name = player_2_
        profile = compute_behavioral_profile_(game_histories, main_behavioral_dimensions, history_main_name, history_opponent_name)
        profile.strategy_name = strat_name
        profile.opponent_name = opponent_name
        profile.save_to_file(urnd_dir_path)

from src.utils import shutdown_run

shutdown_run()
