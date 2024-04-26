import numpy as np

from src.strategies.strategy import Strategy


class AlwaysCooperate(Strategy):
    def __init__(self):
        super().__init__("AlwaysCoop")

    def play(self):
        return 1

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        current_round = len(game_history.get_actions_by_player(player_name))
        return [1] * current_round


class AlwaysDefect(Strategy):
    def __init__(self):
        super().__init__("AlwaysDefect")

    def play(self):
        return 0

    def wrap_up_round(self):
        pass

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

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        current_round = len(game_history.get_actions_by_player(player_name))
        return self.rng.integers(2, size=current_round)


class UnfairRandom(Strategy):
    def __init__(self, defection_prob=0.5):
        self.defection_prob = defection_prob
        self.rng = np.random.default_rng()
        super().__init__("UnfairRnd")

    def play(self):
        choice = self.rng.choice(np.array(list({1, 0})), p=[self.defection_prob, 1 - self.defection_prob])
        return choice

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        current_round = len(game_history.get_actions_by_player(player_name))
        return self.rng.choice(np.array(list({1, 0})), p=[self.defection_prob, 1 - self.defection_prob], size=current_round)


class FixedSequence(Strategy):
    def __init__(self, sequence):
        self.sequence = sequence
        super().__init__("FixedSeq")

    def play(self):
        return self.sequence.pop(0)

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        current_round = len(game_history.get_actions_by_player(player_name))
        if current_round > len(self.sequence):
            return self.sequence[-1]
        return self.sequence[:current_round]
