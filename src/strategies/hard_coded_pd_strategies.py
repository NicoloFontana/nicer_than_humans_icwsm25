import random

import numpy as np

from src.game.gt_game import GTGame
from src.strategies.strategy import Strategy


class TitForTat(Strategy):
    def __init__(self, game: GTGame, player_name: str, sensitivity=0):
        self.game = game
        self.player_name = player_name
        self.sensitivity = sensitivity
        super().__init__("TitForTat")

    def play(self):
        noise = random.uniform(0, 1)
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if opponent_history is None or len(opponent_history) == 0:
            return 1 if noise > self.sensitivity else 0
        return opponent_history[-1] if noise > self.sensitivity else 1 - opponent_history[-1]

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        noise = random.uniform(0, 1)
        alternative_history = [1] if noise > self.sensitivity else [0]
        for i in range(1, len(opponent_history)):
            noise = random.uniform(0, 1)
            alternative_history.append(opponent_history[i] if noise > self.sensitivity else 1 - opponent_history[i])
        return alternative_history


class SuspiciousTitForTat(Strategy):
    def __init__(self, game: GTGame, player_name: str, sensitivity=0):
        self.game = game
        self.player_name = player_name
        self.sensitivity = sensitivity
        super().__init__("SuspiciousTitForTat")

    def play(self):
        noise = random.uniform(0, 1)
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if opponent_history is None or len(opponent_history) == 0:
            return 0 if noise > self.sensitivity else 1
        return opponent_history[-1] if noise > self.sensitivity else 1 - opponent_history[-1]

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        noise = random.uniform(0, 1)
        alternative_history = [0] if noise > self.sensitivity else [1]
        for i in range(1, len(opponent_history)):
            noise = random.uniform(0, 1)
            alternative_history.append(opponent_history[i] if noise > self.sensitivity else 1 - opponent_history[i])
        return alternative_history


class Grim(Strategy):
    def __init__(self, game: GTGame, player_name: str, sensitivity=0):
        self.game = game
        self.player_name = player_name
        self.sensitivity = sensitivity
        self.defected = False
        super().__init__("Grim")

    def play(self):
        noise = random.uniform(0, 1)
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if opponent_history is not None and len(opponent_history) > 0:
            if not opponent_history[-1]:
                self.defected = True
        if self.defected:
            return 0 if noise > self.sensitivity else 1
        return 1 if noise > self.sensitivity else 0

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        noise = random.uniform(0, 1)
        alternative_history = [1] if noise > self.sensitivity else [0]
        triggered = False
        for i in range(len(opponent_history) - 1):
            if opponent_history[i] == 0:
                triggered = True
            noise = random.uniform(0, 1)
            if triggered:
                alternative_history.append(0 if noise > self.sensitivity else 1)
            else:
                alternative_history.append(1 if noise > self.sensitivity else 0)
        return alternative_history


class WinStayLoseShift(Strategy):  # Also known as Pavlov
    def __init__(self, game: GTGame, player_name: str, sensitivity=0):
        self.game = game
        self.player_name = player_name
        self.sensitivity = sensitivity
        super().__init__("WinStayLoseShift")

    def play(self):
        noise = random.uniform(0, 1)
        self_history = self.game.get_actions_by_player(self.player_name)
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if self_history is None or len(self_history) == 0:
            return 1 if noise > self.sensitivity else 0
        if self_history[-1] == opponent_history[-1]:
            # Cooperate if both played the same action
            return 1 if noise > self.sensitivity else 0
        # Defect if players played different actions
        return 0 if noise > self.sensitivity else 1

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        noise = random.uniform(0, 1)
        alternative_history = [1] if noise > self.sensitivity else [0]
        for i in range(1, len(opponent_history)):
            noise = random.uniform(0, 1)
            if (alternative_history[i - 1] == 1 and opponent_history[i - 1] == 1) or (alternative_history[i - 1] == 0 and opponent_history[i - 1] == 1):
                alternative_history.append(1 if noise > self.sensitivity else 0)
            else:
                alternative_history.append(0 if noise > self.sensitivity else 1)
        return alternative_history
