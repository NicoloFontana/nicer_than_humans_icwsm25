import numbers

from src.player_memory import PlayerMemory


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
        self.score = 0
        # Store the memory of the faced opponents and the relative actions taken
        self.memory = PlayerMemory()
        self.strategy = None
        self.strategy_args = None

    def get_name(self) -> str:
        return self.name

    def update_score(self, payoff: numbers.Number):
        """
        Add the payoff to the player's score
        :param payoff: number to be added to the player's score
        """
        if not isinstance(payoff, numbers.Number):
            raise ValueError(f"The payoff must be a number")
        self.score += payoff

    def get_score(self) -> numbers.Number:
        """
        Get the current score of the player
        :return: the current score of the player
        """
        return self.score

    def set_memory(self, memory: PlayerMemory):
        """
        Set the memory of the player
        :param memory: memory of the faced opponents and the relative actions taken
        """
        if not isinstance(memory, PlayerMemory):
            raise ValueError(f"The memory must be a PlayerMemory object")
        self.memory.append(memory)

    def update_memory(self, new_memory: PlayerMemory):
        """
        Update the memory of the player by concatenating the actions played in the last iterations by every opponent
        :param new_memory: the memory to be added
        """
        if not isinstance(new_memory, PlayerMemory):
            raise ValueError(f"The memory to be added must be a PlayerMemory object")
        self.memory.append(new_memory)

    def get_memory(self) -> PlayerMemory:
        """
        Get the memory of the player
        :return: the memory of the player
        """
        return self.memory

    def set_strategy(self, strategy: callable, *args):
        """
        Set the strategy used by the player and implemented in the play_round method
        :param strategy: A function to compute the action to be taken by the player
        """
        if not callable(strategy):
            raise ValueError(f"The strategy must be a callable")
        self.strategy = strategy
        self.strategy_args = args

    def play_round(self) -> object:
        """
        Play a round of the game using the assigned strategy.
        :return: one of the possible actions.
        """
        if self.strategy:
            return self.strategy(*self.strategy_args)
        return None

    def __eq__(self, other):
        return self.name == other.name and self.score == other.score and self.memory == other.memory

    def __str__(self):
        return f"Player {self.name}\nscore: {self.score}\n"
