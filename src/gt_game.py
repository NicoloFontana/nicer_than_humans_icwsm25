from src.player import Player
from src.game_history import GameHistory


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
        self.current_round = 0
        self.is_ended = False

    def play_round(self):
        self.current_round += 1
        if self.current_round > self.iterations:
            self.is_ended = True
        pass

    def get_history(self) -> GameHistory:
        return self.history

    def add_player(self, new_player: Player):
        self.players[new_player.get_name()] = new_player
        self.history.add_player(new_player.get_name())

    def get_payoff_function(self):
        return self.payoff_function

    def get_action_space(self):
        return self.action_space
