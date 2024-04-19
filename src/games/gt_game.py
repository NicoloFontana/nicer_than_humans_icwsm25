import warnings

from src.games.game_history import GameHistory
from src.player import Player


class GTGame:
    """
    Superclass to represent an iterated game

    Parameters
    ----------
    players : dict
        A dictionary with the players in the game. The key is the player's name and the value is the player object
    iterations : int
        The number of iterations of the game
    action_space : set
        The set of actions that can be taken by the players at each iteration
    payoff_function : function
        The function to compute the payoff given one's own action and the other players' actions
    """

    def __init__(self, players: dict[str, Player] = None, iterations: int = 10, action_space: set = None,
                 payoff_function: callable = None):
        self.players = players
        self.iterations = iterations
        self.action_space = action_space
        self.payoff_function = payoff_function
        # Store for each player their history of actions
        self.history = GameHistory()
        self.current_round = 1
        self.is_ended = False

    def play_round(self):
        self.current_round += 1
        if self.current_round > self.iterations:
            self.is_ended = True
        pass

    def get_history(self) -> GameHistory:
        return self.history

    def add_player(self, new_player: Player):
        if self.players is None:
            self.players = {}
        if not isinstance(new_player, Player):
            raise TypeError("The player must be an instance of the class Player")
        if new_player.get_name() in self.players:
            warnings.warn(f"The player {new_player.get_name()} is already in the game")
            return
        self.players[new_player.get_name()] = new_player
        self.history.add_player(new_player.get_name())

    def save_history(self, timestamp, infix=None, subdir=None):
        self.history.save(timestamp, infix=infix, subdir=subdir)

    def get_players(self):
        if self.players is None:
            return []
        return list(self.players.keys())

    def get_opponents_names(self, player_name):
        return [name for name in self.players.keys() if name != player_name]

    def get_player_by_name(self, name):
        return self.players[name]

    def get_iterations(self):
        return self.iterations

    def get_payoff_function(self):
        return self.payoff_function

    def get_action_space(self):
        return self.action_space

    def get_current_round(self):
        return self.current_round

    def get_actions_by_player(self, player_name: str):
        return self.history.get_actions_by_player(player_name)

    def get_actions_by_iteration(self, iteration: int):
        return self.history.get_actions_by_iteration(iteration)

    def get_total_payoff_by_player(self, player_name: str):
        return self.players[player_name].get_total_payoff()
