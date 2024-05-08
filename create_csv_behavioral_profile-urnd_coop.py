from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st

from src.behavioral_analysis.behavioral_profile import BehavioralProfile
from src.games.two_players_pd_utils import plot_ts_, plot_fill
from src.strategies.strategy_utils import plot_errorbar

base_path = Path("behavioral_profiles_analysis")
strat_name = "llama2"
strat_dir_path = base_path / strat_name

cmap = plt.get_cmap('Dark2')
confidence = 0.95

csv_dir_path = Path("csv_files_for_plots") / "behavioral_profile-urnd_coop"
csv_dir_path.mkdir(parents=True, exist_ok=True)

features_analyzed = [
    "niceness",
    "forgiveness",
    "provocability",
    "cooperativeness",
    "troublemaking",
    "emulation",
    "consistency",
]

# plt_fig = plt.figure()
sup_fig, axs = plt.subplots(len(features_analyzed), 1, figsize=(15, 15))
plt_fig = sup_fig
csv_file = []
for i in range(0, 11):
    coop_prob = i / 10
    element = {
        "URND_coop": coop_prob,
    }
    coop_prob_str = str(coop_prob).replace(".", "")
    opponent_name = f"URND{coop_prob_str}"
    urnd_dir_path = strat_dir_path / opponent_name
    file_name = f"behavioral_profile_{strat_name}-{{}}.json"
    file_path = urnd_dir_path / file_name.format(opponent_name)
    profile = BehavioralProfile(strat_name, opponent_name)
    profile.load_from_file(file_path, load_values=True)
    for feature_name in features_analyzed:
        feature = profile.features[feature_name]
        cis = st.norm.interval(confidence, loc=feature.mean, scale=st.sem(feature.values))
        element[f"{feature_name}_mean"] = feature.mean
        element[f"{feature_name}_ci_lb"] = cis[0]
        element[f"{feature_name}_ci_ub"] = cis[1]
    csv_file.append(element)
df = pd.DataFrame(csv_file)
df.to_csv(csv_dir_path / f"llama2_behavioral_profile-urnd_coop.csv")

from src.utils import shutdown_run

shutdown_run()
