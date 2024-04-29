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
        "label": "susTFT",
    },
    "grim": {
        "strategy": Grim,
        "label": "Grim",
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


def get_strategy_by_label(strategy_label):
    strategies = get_strategies()
    for strategy_name in strategies.keys():
        if strategies[strategy_name]["label"] == strategy_label:
            return {strategy_name: strategies[strategy_name]}
    return None


def get_strategies():
    main_strategies = {}
    for strategy_name in main_blind_strategies:
        main_strategies[strategy_name] = main_blind_strategies[strategy_name]
    for strategy_name in main_hard_coded_strategies:
        main_strategies[strategy_name] = main_hard_coded_strategies[strategy_name]
    return main_strategies


def compute_behavioral_profile(timestamp, game_histories, behavioral_features, main_player_name=player_1_, opponent_name=player_2_):
    profile = BehavioralProfile(main_player_name, opponent_name)
    profile.add_features(behavioral_features)
    for game_history in game_histories:
        main_history = game_history.get_actions_by_player(main_player_name)
        opponent_history = game_history.get_actions_by_player(opponent_name)
        profile.compute_features(main_history, opponent_history)
    # profile.save_to_file(timestamp)
    return profile


def plot_behavioral_profile(extraction_timestamp, main_name, opponent_name):
    file_path = OUT_BASE_PATH / f"{extraction_timestamp}" / "behavioral_profiles" / f"behavioral_profile_{main_name}-{opponent_name}.json"
    profile = BehavioralProfile(main_name, opponent_name)
    profile.load_from_file(file_path)
    n_games = profile.n_games

    fig, ax = plt.subplots(figsize=(12, 6))
    for feature_name in profile.features.keys():
        feature = profile.features[feature_name]
        mean = feature.mean
        std_dev = feature.std_dev
        ax.plot([feature_name, feature_name], [mean - std_dev, mean + std_dev], label=feature_name)
        plt.scatter(feature_name, mean, label=feature_name, s=100)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.1)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.1)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.1)

    plt.title(f"{main_name} vs {opponent_name} - {n_games} games")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = OUT_BASE_PATH / f"{extraction_timestamp}" / "behavioral_profiles" / "plots"
    out_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path / f"{main_name}-{opponent_name}_{n_games}.png")
    plt.savefig(out_path / f"{main_name}-{opponent_name}_{n_games}.svg")
    plt.show()
