from pathlib import Path

from matplotlib import pyplot as plt

from sfem_computation import compute_sfem
from src.games.two_players_pd_utils import player_1_, extract_histories_from_files_
from src.strategies.strategy_utils import get_strategies, plot_errorbar


def fmt_map(idx):
    fmts = ['.', 'o', '_', '2', 's',
            'P', '^', '1', 'D', 'v', 'x', '*']
    return fmts[idx % len(fmts)]


base_path = Path("behavioral_profiles_analysis")
strat_name = "llama2"
history_main_name = player_1_
strat_dir_path = base_path / strat_name
cmap = plt.get_cmap('Dark2')
plt_fig = plt.figure()

available_strategies = get_strategies()
strategies_names = list(available_strategies.keys())
for i in range(0, 9):
    p = i / 10
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
        print(f"{strategy_name}: {score}")
    plt_fig = plot_errorbar(scores, cmap(i), opponent_name, plt_figure=plt_fig, fmt=fmt_map(i))
plt.xticks(range(len(strategies_names)), strategies_names)
plt.title(f"SFEM scores for {strat_name}")
plt.xlabel("Estimated strategies")
plt.ylabel("SFEM score")
plt.legend(title="Faced strategies", bbox_to_anchor=(1, 0.75))
plt.tight_layout()
plt.show()

from src.utils import shutdown_run

shutdown_run()
