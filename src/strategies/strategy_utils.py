from src.behavioral_analysis.behavioral_profile import BehavioralProfile
from src.games.two_players_pd_utils import player_1_, player_2_
from src.strategies.blind_pd_strategies import RandomStrategy, UnfairRandom, AlwaysCooperate, AlwaysDefect
from src.strategies.hard_coded_pd_strategies import TitForTat, SuspiciousTitForTat, Grim, WinStayLoseShift

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


def get_strategies_by_names(strategies_names):
    if isinstance(strategies_names, str):
        strategies_names = [strategies_names]
    strategies = get_strategies()
    ret = {}
    for sn in strategies_names:
        if sn in strategies.keys():
            ret[sn] = strategies[sn]
    return ret


def get_strategies_by_labels(strategies_labels):
    if isinstance(strategies_labels, str):
        strategies_labels = [strategies_labels]
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


def compute_behavioral_profile_(game_histories, behavioral_features, main_player_name=player_1_, opponent_name=player_2_):
    profile = BehavioralProfile(main_player_name, opponent_name)
    profile.add_dimensions(behavioral_features)
    for game_history in game_histories:
        main_history = game_history.get_actions_by_player(main_player_name)
        opponent_history = game_history.get_actions_by_player(opponent_name)
        profile.compute_dimensions(main_history, opponent_history)
    return profile
