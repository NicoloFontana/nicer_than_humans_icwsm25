import numbers

from src.player_memory import PlayerMemory
from src.strategies.strategy import Strategy


class Player:
    """
    Superclass to represent a player in a game

    Parameters
    ----------
    name : str
        The name of the player
    """

    def __init__(self, name: str):
        self.name = name
        self.total_payoff = 0
        # Store the memory of the faced opponents and the relative actions taken
        self.memory = PlayerMemory()
        self.strategy = None
        self.strategy_args = None

    def get_name(self) -> str:
        return self.name

    def update_total_payoff(self, payoff: numbers.Number):
        """
        Add the payoff to the player's total_payoff
        :param payoff: number to be added to the player's total_payoff
        """
        if not isinstance(payoff, numbers.Number):
            raise TypeError(f"The payoff must be a number")
        self.total_payoff += payoff

    def get_total_payoff(self) -> numbers.Number:
        """
        Get the current total_payoff of the player
        :return: the current total_payoff of the player
        """
        return self.total_payoff

    def set_memory(self, memory: PlayerMemory):
        """
        Set the memory of the player
        :param memory: memory of the faced opponents and the relative actions taken
        """
        if not isinstance(memory, PlayerMemory):
            raise TypeError(f"The memory must be a PlayerMemory object")
        self.memory.append(memory)

    def update_memory(self, new_memory: PlayerMemory):
        """
        Update the memory of the player by concatenating the actions played in the last iterations by every opponent
        :param new_memory: the new memory to be added to the player's memory
        """
        if not isinstance(new_memory, PlayerMemory):
            raise TypeError(f"The memory to be added must be a PlayerMemory object")
        self.memory.append(new_memory)

    def get_memory(self) -> PlayerMemory:
        """
        Get the memory of the player
        :return: the memory of the player
        """
        return self.memory

    def get_strategy(self) -> Strategy:
        """
        Get the strategy used by the player and implemented in the play_round method
        :return: the strategy used by the player
        """
        return self.strategy

    def set_strategy(self, strategy: Strategy, *strategy_args):
        """
        Set the strategy used by the player and implemented in the play_round method
        :param strategy: A function to compute the action to be taken by the player
        """
        if not isinstance(strategy, Strategy):
            raise TypeError(f"The strategy must be a Strategy object")
        self.strategy = strategy
        self.strategy_args = strategy_args

    def play_round(self):
        """
        Play a round of the game using the strategy of the player.
        :return: one of the possible actions.
        """
        if self.strategy:
            if self.strategy_args:
                return self.strategy.play(*self.strategy_args)
            return self.strategy.play()
        return None

    def __eq__(self, other):
        return self.name == other.name and self.total_payoff == other.total_payoff and self.memory == other.memory

    def __str__(self):
        return f"Player {self.name}\ntotal_payoff: {self.total_payoff}\n"
