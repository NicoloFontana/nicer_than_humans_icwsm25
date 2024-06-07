from src.checkers.aggregation_checker import AggregationChecker
from src.checkers.rule_checker import RuleChecker
from src.checkers.time_checker import TimeChecker


def old_to_new_label(old_label):
    # time
    if old_label == "current_round":
        return "round"
    if old_label == "action_A" or old_label == "action_B" or old_label == "action_you_at_round" or old_label == "action_opponent_at_round":
        return "action_i"
    if old_label == "points_A" or old_label == "points_B" or old_label == "points_you_at_round" or old_label == "points_opponent_at_round":
        return "points_i"
    # rule
    if old_label == "min_payoff" or old_label == "max_payoff":
        return "min_max"
    if old_label == "allowed_actions":
        return "actions"
    if old_label == "round_payoff_A" or old_label == "round_payoff_B" or old_label == "payoff_you_given_combo" or old_label == "payoff_opponent_given_combo":
        return "payoff"
    # if old_label == "exists_combo" or old_label == "exists_combo_for_payoff_you":
    #     return "exist_combo"
    # if old_label == "combo_for_payoff_A" or old_label == "which_combo_for_payoff_you":
    #     return "combo"
    # aggreg
    if old_label == "#actions_A" or old_label == "#actions_B" or old_label == "#actions_you" or old_label == "#actions_opponent":
        return "#actions"
    if old_label == "total_payoff_A" or old_label == "total_payoff_B" or old_label == "total_payoff_you" or old_label == "total_payoff_opponent":
        return "#points"
    return None


def new_label_to_full_question(new_label):
    if new_label == "round":
        return "Which is the current round of the game?"
    if new_label == "action_i":
        return "Which action did player X play in round i?"
    if new_label == "points_i":
        return "How many points did player X collect in round i?"
    if new_label == "min_max":
        return "What is the lowest/highest payoff player A can get in a single round?"
    if new_label == "actions":
        return "Which actions is player A allowed to play?"
    if new_label == "payoff":
        return "Which is player X's payoff in a single round if X plays p and Y plays q?"
    if new_label == "exist_combo":
        return "Does exist a combination of actions that gives a player a payoff of K in a single round?"
    if new_label == "combo":
        return "Which combination of actions gives a payoff of K to A in a single round?"
    if new_label == "#actions":
        return "How many times did player X choose p?"
    if new_label == "#points":
        return "What is player X's current total payoff?"


def old_to_new_checker_name(old_checker_name):
    if old_checker_name == "aggregation":
        return "state"
    if old_checker_name == "rule":
        return "rules"
    if old_checker_name == "time":
        return "time"


def get_checkers_by_names(checkers_names):
    checkers = []
    if "time" in checkers_names:
        checkers.append(TimeChecker())
    if "rule" in checkers_names or "rules" in checkers_names:
        checkers.append(RuleChecker())
    if "aggregation" in checkers_names or "aggregate" in checkers_names or "state" in checkers_names:
        checkers.append(AggregationChecker())
    return checkers
