from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as st

from src.behavioral_analysis.behavioral_profile import BehavioralProfile
from src.behavioral_analysis.main_behavioral_dimensions import main_behavioral_dimensions
from src.games.two_players_pd_utils import extract_histories_from_files, player_1_, player_2_
from src.strategies.strategy_utils import compute_behavioral_profile_
from src.unused_functions import plot_errorbar, plot_ts_, plot_fill

base_path = Path("../../behavioral_profiles_analysis")
strat_name = "llama2"
strat_dir_path = base_path / strat_name
rnd_name = "RND"
rnd_dir_path = base_path / rnd_name


features_analyzed = [
    "niceness",
    "forgiveness",
    "provocability",
    # "cooperativeness",
    "troublemaking",
    "emulation",
    # "consistency",
]


min_urnd_coop = 0
max_urnd_coop = 11

for i in range(min_urnd_coop, max_urnd_coop):
    p = i / 10
    p_str = str(p).replace(".", "")
    opponent_name = f"URND{p_str}"
    urnd_dir_path = strat_dir_path / opponent_name
    game_histories_dir_path = urnd_dir_path / "game_histories"
    # file_name = f"game_history_{strat_name}-{opponent_name}"
    file_name = f"game_history"
    game_histories = extract_histories_from_files(game_histories_dir_path, file_name)

    for game_history in game_histories:
        rng = np.random.default_rng()
        new_rnd_history = {rnd_name: list(rng.integers(2, size=len(game_history.get_actions_by_player(player_1_))))}
        game_history.add_history(new_rnd_history)

    history_opponent_name = player_2_
    profile = compute_behavioral_profile_(game_histories, main_behavioral_dimensions, rnd_name, history_opponent_name)
    profile.strategy_name = rnd_name
    profile.opponent_name = opponent_name
    out_profile_dir = rnd_dir_path / opponent_name
    out_profile_dir.mkdir(parents=True, exist_ok=True)
    # out_profile_path = out_profile_dir / f"behavioral_profile_{profile.strategy_name}-{profile.opponent_name}.json"
    profile.save_to_file(out_profile_dir)

cmap = plt.get_cmap('Dark2')
confidence = 0.95

plt_fig = plt.figure()
sup_fig, axs = plt.subplots(len(features_analyzed), 1, figsize=(12, 15))
plt_fig = sup_fig
for feature_name in features_analyzed:
    ax = axs[features_analyzed.index(feature_name)]
    feature_mean_ts = []
    feature_lb_ts = []
    feature_ub_ts = []
    feature_yerr_ts = []
    idx = 0
    for i in range(1, 10):
        coop_prob = i / 10
        coop_prob_str = str(coop_prob).replace(".", "")
        opponent_name = f"URND{coop_prob_str}"
        urnd_dir_path = rnd_dir_path / opponent_name
        file_name = f"behavioral_profile_{rnd_name}-{{}}.json"
        file_path = urnd_dir_path / file_name.format(opponent_name)
        profile = BehavioralProfile(strat_name, opponent_name)
        profile.load_from_file(file_path, load_values=True)
        feature = profile.dimensions[feature_name]
        feature_mean_ts.append(feature.mean)
        cis = st.norm.interval(confidence, loc=feature.mean, scale=st.sem(feature.values))
        feature_lb_ts.append(cis[0])
        feature_ub_ts.append(cis[1])
        yerr = (cis[1] - cis[0]) / 2
        feature_yerr_ts.append(yerr)
    axhlines = [0.0, 0.5, 1.0]
    plt_fig = plot_ts_(feature_mean_ts, "blue", feature_name, axhlines=axhlines, fig=plt_fig, ax=ax)
    plt_fig = plot_fill(feature_lb_ts, feature_ub_ts, "blue", fig=plt_fig, ax=ax)
    plt.ylabel(feature_name)
    # plt.xlabel("URND cooperation") if features_analyzed.index(feature_name) == len(features_analyzed) - 1 else None
    plt.xticks([i for i in range(9)], [str(i / 10) for i in range(9)])
    plt.tight_layout()
plt.figure(plt_fig)
plt.suptitle(f"{rnd_name} behavioral dimensions against different unfair RND")
sup_fig.supxlabel("URND cooperation")
sup_fig.supylabel("Behavioral dimensions")
plt.tight_layout()
plt.show()

#
# def fmt_map(idx):
#     fmts = ['.', 'o', '_', '2', 's',
#             'P', '^', '1', 'D', 'v', 'x', '*']
#     return fmts[idx % len(fmts)]
#
#
# plt_fig = plt.figure()
# idx = 0
# for i in range(0, 9):
#     coop_prob = i / 10
#     coop_prob_str = str(coop_prob).replace(".", "")
#     opponent_name = f"URND{coop_prob_str}"
#     urnd_dir_path = strat_dir_path / opponent_name
#     file_name = f"behavioral_profile_{strat_name}-{{}}.json"
#     file_path = urnd_dir_path / file_name.format(opponent_name)
#     profile = BehavioralProfile(strat_name, opponent_name)
#     profile.load_from_file(file_path, load_values=True)
#     means = []
#     cis = []
#     yerrs = []
#     for feature_name in features_analyzed:
#         if feature_name not in profile.dimensions:
#             continue
#         feature = profile.dimensions[feature_name]
#         means.append(feature.mean)
#         cis.append(st.norm.interval(confidence, loc=feature.mean, scale=st.sem(feature.values)))
#         yerrs.append((cis[-1][1] - cis[-1][0]) / 2)
#     axhlines = [0.0, 0.5, 1.0]
#     label = f"vs {opponent_name}"
#     # plt_fig = plot_errorbar(means, cmap(idx), label, fig=plt_fig, axhlines=axhlines, yerr=yerrs, fmt=fmt_map(idx))
#     plt_fig = plot_errorbar(means, cmap(idx), label, fig=plt_fig, axhlines=axhlines, fmt=fmt_map(idx))
#     idx += 1
#
# plt.figure(plt_fig)
# plt.ylabel("Level")
# plt.xlabel("Behavioral dimensions")
# plt.xticks(range(len(features_analyzed)), features_analyzed, rotation=45, ha='right')
# plt.title(f"{strat_name} behavioral profiles")
# plt.legend(title="Faced strategies", bbox_to_anchor=(1, 0.75))
# plt.tight_layout()
#
# # out_file_path = OUT_BASE_PATH / "behavioral_profiles" / "plots" / f"Llama2-vs-URND_test"  # TODO
# # plt.savefig(out_file_path.with_suffix('.png'))
# # plt.savefig(out_file_path.with_suffix('.svg'))
# plt.show()


from src.utils import shutdown_run

shutdown_run()
