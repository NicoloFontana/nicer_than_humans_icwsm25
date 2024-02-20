import numpy as np

from src.strategies.strategy import Strategy


class AlwaysCoopStrategy(Strategy):
    def __init__(self):
        super().__init__("AlwaysCoop")

    def play(self):
        return 1


class AlwaysDefectStrategy(Strategy):
    def __init__(self):
        super().__init__("AlwaysDefect")

    def play(self):
        return 0


class RndStrategy(Strategy):
    def __init__(self):
        super().__init__("Rnd")

    def play(self):
        rng = np.random.default_rng()
        choice = rng.choice(np.array(list({1, 0})))
        return choice
