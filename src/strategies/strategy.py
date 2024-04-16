import warnings


class Strategy:
    def __init__(self, strategy_name: str):
        if not isinstance(strategy_name, str):
            warnings.warn(f"strategy_name must be of type str, not {type(strategy_name)}. Converting to str.")
            strategy_name = str(strategy_name)
        self.strategy_name = strategy_name

    def play(self):
        pass

    def wrap_up_round(self):
        pass

    def generate_alternative_history_for_player(self, game_history, player_name):
        pass
