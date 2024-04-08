from src.strategies.strategy import Strategy


class TitForTat(Strategy):
    def __init__(self):
        super().__init__("TitForTat")

    def play(self, opponent_history=None, verbose=False):
        if opponent_history is None:
            return 1
        return opponent_history[-1]

    def generate_alternative_history_for_player(self, game_history, player_name):
        players_names = game_history.get_players_names()
        opponent_name = players_names[0] if player_name == players_names[1] else players_names[1]
        opponent_history = game_history.get_actions_by_player(opponent_name)
        return [1] + opponent_history[:(len(opponent_history) - 1)]
