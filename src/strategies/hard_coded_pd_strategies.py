from src.games.gt_game import GTGame
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

    def generate_alternative_history_for_player(self, game_history, player_name):
        players_names = game_history.get_players_names()
        opponent_name = players_names[0] if player_name == players_names[1] else players_names[1]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        return [1] + opponent_history[:(len(opponent_history) - 1)]
