from matplotlib import pyplot as plt
import scipy.stats as st

from src.behavioral_analysis.behavioral_profile import BehavioralProfile
from src.strategies.strategy_utils import get_strategies, compute_behavioral_profile, get_strategy_by_name, plot_behavioral_profile, plot_errorbar
from src.utils import OUT_BASE_PATH, compute_average_vector, out_path, timestamp

behavioral_profiles_path = OUT_BASE_PATH / "behavioral_profiles" / "llama2_vs_urnd"
file_name = "behavioral_profile_Llama2-{}.json"
cmap = plt.get_cmap('Dark2')
confidence = 0.95
features_analyzed = [
    "niceness",
    "forgiveness2",
    "provocability",
    "cooperativeness",
    "troublemaking",
    "emulation",
    "consistency",
]


# features_analyzed = main_behavioral_features.keys()

def fmt_map(idx):
    fmts = ['.', 'o', '|', '2', 's',
            'p', 'h', '+', 'D', '^', 'x', '*']
    return fmts[idx % len(fmts)]


plt_fig = plt.figure()
idx = 0
for i in range(0, 9):
    coop_prob = i / 10
    coop_prob_str = str(coop_prob).replace(".", "")
    if i != 0 and i != 5 and i != 10:
        opponent_name = f"URND{coop_prob_str}"
    elif i == 0:
        opponent_name = "ALLD"
    elif i == 5:
        opponent_name = "RND"
    else:
        opponent_name = "ALLC"
    label = f"vs {opponent_name}"
    file_path = behavioral_profiles_path / file_name.format(opponent_name)
    profile = BehavioralProfile("Llama2", opponent_name)
    profile.load_from_file(file_path, load_values=True)
    means = []
    cis = []
    yerrs = []
    for feature_name in features_analyzed:
        if feature_name not in profile.features:
            continue
        feature = profile.features[feature_name]
        means.append(feature.mean)
        cis.append(st.norm.interval(confidence, loc=feature.mean, scale=st.sem(feature.values)))
        yerrs.append((cis[-1][1] - cis[-1][0]) / 2)
    axhlines = [0.0, 0.5, 1.0]
    plt_fig = plot_errorbar(means, cmap(idx), label, plt_figure=plt_fig, axhlines=axhlines, yerr=yerrs, fmt=fmt_map(idx))
    idx += 1

plt.figure(plt_fig)
plt.ylabel("Level")
plt.xlabel("Behavioral features")
plt.xticks(range(len(features_analyzed)), features_analyzed, rotation=45, ha='right')
plt.title(f"Llama2 behavioral profiles")
plt.legend(bbox_to_anchor=(1, 0.75))
plt.tight_layout()

# out_file_path = OUT_BASE_PATH / "behavioral_profiles" / "plots" / f"Llama2-vs-URND_test"  # TODO
# plt.savefig(out_file_path.with_suffix('.png'))
# plt.savefig(out_file_path.with_suffix('.svg'))
plt.show()


from src.utils import shutdown_run

shutdown_run()
