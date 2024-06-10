from src.game.gt_game import GTGame
from src.strategies.strategy import Strategy


class TitForTat(Strategy):
    def __init__(self, game: GTGame, player_name: str):
        self.game = game
        self.player_name = player_name
        super().__init__("TitForTat")

    def play(self):
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if opponent_history is None or len(opponent_history) == 0:
            return 1
        return opponent_history[-1]

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        return [1] + opponent_history[:(len(opponent_history) - 1)]


class SuspiciousTitForTat(Strategy):
    def __init__(self, game: GTGame, player_name: str):
        self.game = game
        self.player_name = player_name
        super().__init__("SuspiciousTitForTat")

    def play(self):
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if opponent_history is None or len(opponent_history) == 0:
            return 0
        return opponent_history[-1]

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        return [0] + opponent_history[:(len(opponent_history) - 1)]


class Grim(Strategy):
    def __init__(self, game: GTGame, player_name: str):
        super().__init__("Grim")
        self.game = game
        self.player_name = player_name
        self.defected = False

    def play(self):
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if opponent_history is not None and len(opponent_history) > 0:
            if not opponent_history[-1]:
                self.defected = True
        return 1 if not self.defected else 0

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        alt_history = [1]
        triggered = False
        for i in range(len(opponent_history) - 1):
            if opponent_history[i] == 0:
                triggered = True
            alt_history.append(0 if triggered else 1)
        return alt_history


class WinStayLoseShift(Strategy):  # Also known as Pavlov
    def __init__(self, game: GTGame, player_name: str):
        super().__init__("WinStayLoseShift")
        self.game = game
        self.player_name = player_name

    def play(self):
        self_history = self.game.get_actions_by_player(self.player_name)
        opponent_name = self.game.get_opponents_names(self.player_name)[0]
        opponent_history = self.game.get_actions_by_player(opponent_name)
        if self_history is None or len(self_history) == 0:
            return 1
        if self_history[-1] == opponent_history[-1]:
            return 1  # Cooperate if both played the same action
        return 0  # Defect if both played different actions

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        opponent_name = self.game.get_opponents_names(player_name)[0]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        alt_history = [1]
        for i in range(1, len(opponent_history)):
            if (alt_history[i - 1] == 1 and opponent_history[i - 1] == 1) or (alt_history[i - 1] == 0 and opponent_history[i - 1] == 1):
                alt_history.append(1)
            else:
                alt_history.append(0)
        return alt_history
