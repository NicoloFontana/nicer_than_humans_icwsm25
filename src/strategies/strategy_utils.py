from pathlib import Path

from matplotlib import pyplot as plt

from src.behavioral_analysis.behavioral_profile import BehavioralProfile
from src.games.two_players_pd_utils import player_1_, player_2_
from src.strategies.blind_pd_strategies import RandomStrategy, UnfairRandom, AlwaysCooperate, AlwaysDefect
from src.strategies.hard_coded_pd_strategies import TitForTat, SuspiciousTitForTat, Grim, WinStayLoseShift
from src.utils import OUT_BASE_PATH

main_blind_strategies = {
    "random_strategy": {
        "strategy": RandomStrategy,
        "label": "RND",
    },
    "unfair_random": {
        "strategy": UnfairRandom,
        "label": "URND",
    },
    "always_cooperate": {
        "strategy": AlwaysCooperate,
        "label": "ALLC",
    },
    "always_defect": {
        "strategy": AlwaysDefect,
        "label": "ALLD",
    },
}
main_hard_coded_strategies = {
    "tit_for_tat": {
        "strategy": TitForTat,
        "label": "TFT",
    },
    "suspicious_tit_for_tat": {
        "strategy": SuspiciousTitForTat,
        "label": "STFT",
    },
    "grim": {
        "strategy": Grim,
        "label": "GRIM",
    },
    "win_stay_lose_shift": {
        "strategy": WinStayLoseShift,
        "label": "WSLS",
    },
}


def get_strategy_by_name(strategy_name):
    strategies = get_strategies()
    for sn in strategies.keys():
        if sn == strategy_name:
            return {strategy_name: strategies[strategy_name]}
    return None


def get_strategies_by_names(strategies_names):
    strategies = get_strategies()
    ret = {}
    for sn in strategies_names:
        if sn in strategies.keys():
            ret[sn] = strategies[sn]
    return ret


def get_strategy_by_label(strategy_label):
    strategies = get_strategies()
    for strategy_name in strategies.keys():
        if strategies[strategy_name]["label"] == strategy_label:
            return {strategy_name: strategies[strategy_name]}
    return None


def get_strategies_by_labels(strategies_labels):
    strategies = get_strategies()
    ret = {}
    for sl in strategies_labels:
        for sk in strategies.keys():
            if strategies[sk]["label"] == sl:
                ret[sk] = strategies[sk]
    return ret


def get_strategies():
    main_strategies = {}
    for strategy_name in main_blind_strategies:
        main_strategies[strategy_name] = main_blind_strategies[strategy_name]
    for strategy_name in main_hard_coded_strategies:
        main_strategies[strategy_name] = main_hard_coded_strategies[strategy_name]
    return main_strategies


def compute_behavioral_profile(timestamp, game_histories, behavioral_features, main_player_name=player_1_, opponent_name=player_2_):  # TODO remove
    profile = BehavioralProfile(main_player_name, opponent_name)
    profile.add_features(behavioral_features)
    for game_history in game_histories:
        main_history = game_history.get_actions_by_player(main_player_name)
        opponent_history = game_history.get_actions_by_player(opponent_name)
        profile.compute_features(main_history, opponent_history)
    # profile.save_to_file(timestamp)
    return profile


def compute_behavioral_profile_(game_histories, behavioral_features, main_player_name=player_1_, opponent_name=player_2_):
    profile = BehavioralProfile(main_player_name, opponent_name)
    profile.add_features(behavioral_features)
    for game_history in game_histories:
        main_history = game_history.get_actions_by_player(main_player_name)
        opponent_history = game_history.get_actions_by_player(opponent_name)
        profile.compute_features(main_history, opponent_history)
    return profile


def plot_behavioral_profile(profile, title=None, out_file_path=None, show=False, color=None, plt_figure=None, label=None):
    # file_path = OUT_BASE_PATH / f"{extraction_timestamp}" / "behavioral_profiles" / f"behavioral_profile_{main_name}-{opponent_name}.json"
    # profile = BehavioralProfile(main_name, opponent_name)
    # profile.load_from_file(file_path)

    color = color if color is not None else 'blue'

    plt.figure(plt_figure)

    feature_names = list(profile.features.keys())
    means = [profile.features[feature_name].mean for feature_name in feature_names]
    std_devs = [profile.features[feature_name].std_dev for feature_name in feature_names]
    plt.errorbar(feature_names, means, yerr=std_devs, fmt='o', color=color, label=label)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.1)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.1)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.1)

    # plt.title(f"{main_name} vs {opponent_name} - {n_games} games") if n_games is not None else plt.title(" ")#f"{main_name} vs {opponent_name}")
    plt.title(title) if title is not None else plt.title(" ")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # out_path = OUT_BASE_PATH / f"{extraction_timestamp}" / "behavioral_profiles" / "plots"
    if out_file_path is not None:
        out_parents = Path(out_file_path.parent)
        out_parents.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file_path.with_suffix('.png'))
        plt.savefig(out_file_path.with_suffix('.svg'))
    plt.show() if show else None
    return plt.gcf().number


def plot_errorbar(values, values_color, values_label, yerr=None, axhlines=None, plt_figure=None, alpha=1.0, fmt='o'):
    plt.figure(plt_figure)

    plt.errorbar([i for i in range(len(values))], values, yerr=yerr, fmt=fmt, color=values_color, label=values_label, alpha=alpha)

    if axhlines is not None:
        for axhline in axhlines:
            plt.axhline(y=axhline, color='black', linestyle='--', lw=0.5, alpha=0.3)

    plt.title(" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    return plt.gcf().number
