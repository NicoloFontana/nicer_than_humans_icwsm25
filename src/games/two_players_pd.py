import warnings

from src.games.gt_game import GTGame
from src.player import Player
from src.player_memory import PlayerMemory


def two_players_pd_payoff(own_action: int, other_action: int) -> int:
    """
    Default payoff function for the two-player version of the Prisoner's Dilemma game
    :param own_action: action of the player for which the payoff is computed
    :param other_action: opponent's action
    :return: computed payoff
    """
    if own_action == 1 and other_action == 1:
        return 3
    elif own_action == 1 and other_action == 0:
        return 0
    elif own_action == 0 and other_action == 1:
        return 5
    elif own_action == 0 and other_action == 0:
        return 1
    else:
        raise ValueError("Invalid actions")


class TwoPlayersPD(GTGame):
    """
    Specific class to represent the Prisoner's Dilemma game with two players
    """

    def __init__(self, player_one: Player = None, player_two: Player = None, iterations: int = 10):
        players = {}
        if player_one is not None and player_two is not None:
            players = {player_one.get_name(): player_one, player_two.get_name(): player_two}
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
        # Update each player's score
        payoff_one = self.payoff_function(action_one, action_two)
        player_one.update_score(payoff_one)
        payoff_two = self.payoff_function(action_two, action_one)
        player_two.update_score(payoff_two)
        super().play_round()

    def add_player(self, new_player: Player):
        if len(self.players.keys()) < 2:
            super().add_player(new_player)
        else:
            warnings.warn("There are already 2 players")
