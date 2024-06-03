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
        "label": "AC",
    },
    "always_defect": {
        "strategy": AlwaysDefect,
        "label": "AD",
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


def get_strategy_instance_by_name(strategy_name, args=None):
    strategy_dict = get_strategies_by_names(strategy_name)
    if len(strategy_dict) == 1:
        strategy_dict_element = list(strategy_dict.values())[0]
        if strategy_dict_element["label"] == "URND":
            return strategy_dict_element["strategy"](args)
        return strategy_dict_element["strategy"]()
    else:
        return None


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


def get_strategy_instance_by_label(strategy_label, args=None):
    strategy_dict = get_strategies_by_labels(strategy_label)
    if len(strategy_dict) == 1:
        strategy_dict_element = list(strategy_dict.values())[0]
        if strategy_dict_element["label"] == "URND":
            return strategy_dict_element["strategy"](args)
        return strategy_dict_element["strategy"]()
    else:
        return None


def get_strategy_instance(strategy_str, args=None):
    strategy_instance = get_strategy_instance_by_name(strategy_str, args=args)
    if strategy_instance is None:
        strategy_instance = get_strategy_instance_by_label(strategy_str, args=args)
    if strategy_instance is None:
        raise ValueError(f"Invalid strategy: {strategy_str}.\n"
                         f"Provide a strategy name among these:\n"
                         f"{get_strategies_names()}\n"
                         f"or a label among these:\n"
                         f"{get_strategies_labels()}")
    return strategy_instance


def get_strategies():
    main_strategies = {}
    for strategy_name in main_blind_strategies:
        main_strategies[strategy_name] = main_blind_strategies[strategy_name]
    for strategy_name in main_hard_coded_strategies:
        main_strategies[strategy_name] = main_hard_coded_strategies[strategy_name]
    return main_strategies


def get_strategies_labels():
    strategies = get_strategies()
    return [strategies[sk]["label"] for sk in strategies.keys()]


def get_strategies_names():
    strategies = get_strategies()
    return [sk for sk in strategies.keys()]


def compute_behavioral_profile(game_histories, behavioral_dimensions_names, main_player_name=player_1_, opponent_name=player_2_):
    profile = BehavioralProfile(main_player_name, opponent_name)
    profile.add_dimensions(behavioral_dimensions_names)
    for game_history in game_histories:
        main_history = game_history.get_actions_by_player(main_player_name)
        opponent_history = game_history.get_actions_by_player(opponent_name)
        profile.compute_dimensions(main_history, opponent_history)
    return profile
