import warnings

from src.games.gt_game import GTGame
from src.games.two_players_pd_utils import two_players_pd_payoff
from src.player import Player
from src.player_memory import PlayerMemory


class TwoPlayersPD(GTGame):
    """
    Specific class to represent the Prisoner's Dilemma game with two players
    """

    def __init__(self, player_one: Player = None, player_two: Player = None, iterations: int = 10):
        players = {}
        if player_one is not None:
            players[player_one.get_name()] = player_one
        if player_two is not None:
            players[player_two.get_name()] = player_two
        super().__init__(players, iterations, action_space={1, 0}, payoff_function=two_players_pd_payoff)

    def play_round(self, memories=None):
        player_one = self.players[list(self.players)[0]]
        player_two = self.players[list(self.players)[1]]
        if memories is None:
            player_one.set_memory(PlayerMemory())
            player_two.set_memory(PlayerMemory())
        else:
            player_one.set_memory(memories[player_one.get_name()])
            player_two.set_memory(memories[player_two.get_name()])

        action_one = player_one.play_round()
        if action_one not in self.action_space:
            raise ValueError(f"The action {action_one} is not in the action space")
        action_two = player_two.play_round()
        if action_two not in self.action_space:
            raise ValueError(f"The action {action_two} is not in the action space")
        self.history.add_last_iteration([player_one.get_name(), player_two.get_name()], [action_one, action_two])
        # Update each player's total_payoff
        payoff_one = self.payoff_function(action_one, action_two)
        player_one.update_total_payoff(payoff_one)
        payoff_two = self.payoff_function(action_two, action_one)
        player_two.update_total_payoff(payoff_two)
        super().play_round()

    def add_player(self, new_player: Player):
        if isinstance(new_player, Player):
            if len(self.players.keys()) < 2:
                super().add_player(new_player)
            else:
                warnings.warn("There are already 2 players")
        else:
            raise ValueError("The player must be an instance of the class Player")

    def get_opponent_name(self, player_name):
        for name in self.players.keys():
            if name != player_name:
                return name
        warnings.warn(f"The player {player_name} is not present in the game")
        return None
