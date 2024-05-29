from pathlib import Path

from matplotlib import pyplot as plt
import scipy.stats as st

from src.behavioral_analysis.behavioral_profile import BehavioralProfile
from src.unused_functions import plot_errorbar, plot_ts_, plot_fill

base_path = Path("../../behavioral_profiles_analysis")
strat_name = "llama2"
strat_dir_path = base_path / strat_name
rnd_name = "RND"
rnd_dir_path = base_path / rnd_name

cmap = plt.get_cmap('Dark2')
confidence = 0.95
features_analyzed = [
    "niceness",
    "forgiveness",
    "provocability",
    # "cooperativeness",
    "troublemaking",
    "emulation",
    # "consistency",
]

# plt_fig = plt.figure()
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
        urnd_dir_path = strat_dir_path / opponent_name
        file_name = f"behavioral_profile_{strat_name}-{{}}.json"
        file_path = urnd_dir_path / file_name.format(opponent_name)
        profile = BehavioralProfile(strat_name, opponent_name)
        profile.load_from_file(file_path, load_values=True)
        feature = profile.dimensions[feature_name]

        rnd_file_name = f"behavioral_profile_{rnd_name}-{opponent_name}.json"
        rnd_file_path = rnd_dir_path / opponent_name / rnd_file_name
        rnd_profile = BehavioralProfile(rnd_name, opponent_name)
        rnd_profile.load_from_file(rnd_file_path, load_values=True)
        rnd_feature = rnd_profile.dimensions[feature_name]

        delta_values = [feature.values[i] - rnd_feature.values[i] for i in range(len(feature.values))]
        delta_feature_mean = sum(delta_values) / len(delta_values)
        feature_mean_ts.append(delta_feature_mean)
        cis = st.norm.interval(confidence, loc=delta_feature_mean, scale=st.sem(delta_values))
        feature_lb_ts.append(cis[0])
        feature_ub_ts.append(cis[1])
        yerr = (cis[1] - cis[0]) / 2
        feature_yerr_ts.append(yerr)
    axhlines = [0.0, 0.5, 1.0]
    plt_fig = plot_ts_(feature_mean_ts, "blue", feature_name, axhlines=axhlines, fig=plt_fig, ax=ax)
    plt_fig = plot_fill(feature_lb_ts, feature_ub_ts, "blue", fig=plt_fig, ax=ax)
    plt.ylabel(feature_name)
    # plt.xlabel("URND cooperation") if features_analyzed.index(feature_name) == len(features_analyzed) - 1 else None
    plt.xticks([i for i in range(1, 10)], [str(i / 10) for i in range(1, 10)])
    plt.tight_layout()
plt.figure(plt_fig)
plt.suptitle(f"{strat_name} - {rnd_name} behavioral dimensions against different unfair RND")
sup_fig.supxlabel("URND cooperation")
sup_fig.supylabel("Behavioral dimensions")
plt.tight_layout()
plt.show()
