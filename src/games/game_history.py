import warnings


class GameHistory:
    """
    Class to store the history of actions of the players in a game.\n
    The counting of the iterations starts from 0.\n
    The behaviour for iterations smaller than 0 is the same as in Python lists.\n
    """
    def __init__(self):
        self.history = {}

    def add_player(self, player_name: str):
        """
        Add a player to the history.\n
        :param player_name: name of the player
        """
        if player_name not in self.history.keys():
            self.history[player_name] = []

    def add_last_iteration(self, players_names: list[str], actions: list[object]):
        """
        Add the history of the last iteration played.\n
        Each action is paired with the player's name in the corresponding position.\n
        Players without an action are paired with None.\n
        :param players_names: names of the players involved in the last iteration
        :param actions: actions taken by the players in the last iteration
        """
        if not isinstance(players_names, list) and isinstance(players_names, str):
            warnings.warn("The players' names must be a list. Encapsulating the input in a list")
            players_names = [players_names]
        if not isinstance(actions, list):
            warnings.warn("The actions played must be a list. Encapsulating the input in a list")
            actions = [actions]
        if not isinstance(players_names, list) or not isinstance(actions, list):
            raise TypeError("The players' names and the actions played must be lists")
        if len(players_names) < len(actions):
            raise IndexError("The number of players' names must be greater or equal to the number of actions played")
        if len(players_names) > len(actions):
            # Append None to the actions list for the players that did not play
            warnings.warn("The number of players' names is greater than the number of actions played. Appending None to the actions list for the players that did not play.")
            for i in range(len(players_names) - len(actions)):
                actions.append(None)
        for idx, player_name in enumerate(players_names):
            if player_name not in self.history.keys():
                self.history[player_name] = []
            self.history[player_name].append(actions[idx])

    def get_actions_by_player(self, player_name: str) -> list:
        """
        Get the history of actions played by a specific player.\n
        :param player_name: name of the player
        :return: list of actions played up to now by the player
        """
        if player_name not in self.history.keys():
            warnings.warn(f"The player {player_name} is not present in the history")
            return []
        return self.history[player_name]

    def get_actions_by_iteration(self, iteration: int) -> dict:
        """
        Get the actions played by all the players at a specific iteration of the game.\n
        :param iteration: iteration to be considered
        :return: dictionary with the action played by each player
        """
        if not isinstance(iteration, int):
            warnings.warn("The iteration must be an integer. Rounding the input to the closest smaller integer")
            iteration = int(iteration)
        if not bool(self.history) or iteration >= len(self.history[list(self.history.keys())[0]]):
            warnings.warn(f"The iteration {iteration} is not present in the history")
            return {}
        iteration_history = {}
        for player in self.history.keys():
            iteration_history[player] = self.history[player][iteration]
        return iteration_history

    def __str__(self):
        to_str = ""
        for player in self.history.keys():
            to_str += f"{player}: {self.history[player]}\n"
        return to_str

    def __bool__(self):
        return bool(self.history)

    def __eq__(self, other):
        return self.history == other.history

    def __len__(self):
        if not bool(self.history):
            return 0
        first_player = list(self.history.keys())[0]
        return len(self.history[first_player])
