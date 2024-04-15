import numpy as np

from src.strategies.strategy import Strategy


class AlwaysCooperate(Strategy):
    def __init__(self):
        super().__init__("AlwaysCoop")

    def play(self):
        return 1

    def generate_alternative_history_for_player(self, game_history, player_name):
        current_round = len(game_history.get_actions_by_player(player_name))
        return [1] * current_round


class AlwaysDefect(Strategy):
    def __init__(self):
        super().__init__("AlwaysDefect")

    def play(self):
        return 0

    def generate_alternative_history_for_player(self, game_history, player_name):
        current_round = len(game_history.get_actions_by_player(player_name))
        return [0] * current_round


class RandomStrategy(Strategy):
    def __init__(self):
        self.rng = np.random.default_rng()
        super().__init__("Rnd")

    def play(self):
        choice = self.rng.choice(np.array(list({1, 0})))
        return choice

    def generate_alternative_history_for_player(self, game_history, player_name):
        current_round = len(game_history.get_actions_by_player(player_name))
        return self.rng.integers(2, size=current_round)
