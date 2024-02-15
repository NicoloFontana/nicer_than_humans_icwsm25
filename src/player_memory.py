import warnings


class PlayerMemory:
    """
    Class to store the memory of the faced opponents and the relative actions taken.\n

    Parameters
    ----------
    players_names_pairing : tuple
        Tuple of players' names paired during the iterations considered
    pairing_actions : list[tuple]
        List of actions taken by the players in the pairing at each iteration considered
    """

    def __init__(self, players_names_pairing: tuple = None, pairing_actions: list[tuple] = None):
        self.memory = {}
        if players_names_pairing is not None and pairing_actions is not None:
            if not isinstance(players_names_pairing, tuple):
                raise ValueError("The pairing of players must be a tuple")
            if not isinstance(pairing_actions, list):
                if isinstance(pairing_actions, tuple):
                    warnings.warn("The actions pairing must be a list. Encapsulating the input in a list")
                    pairing_actions = [pairing_actions]
                else:
                    raise ValueError("The actions played must be tuples in a list")
            elif not all(isinstance(action, tuple) for action in pairing_actions):
                raise ValueError("The actions played must be tuples")
            self.memory[players_names_pairing] = pairing_actions

    def add_element(self, players_names_pairing: tuple, pairing_actions: list[tuple]):
        """
        Add the history of action of a specific pairing of players.\n
        If the pairing already exists, the history is appended.\n
        :param players_names_pairing: tuple of players' names paired during the iterations considered
        :param pairing_actions: actions played by the players in the pairing at each iteration considered
        """
        if not isinstance(players_names_pairing, tuple):
            raise ValueError("The pairing of players must be a tuple")
        if not isinstance(pairing_actions, list):
            if isinstance(pairing_actions, tuple):
                warnings.warn("The actions pairings must be a list. Encapsulating the input in a list")
                pairing_actions = [pairing_actions]
            else:
                raise ValueError("The actions played must be tuples in a list")
        elif not all(isinstance(action, tuple) for action in pairing_actions):
            raise ValueError("The actions played must be tuples")
        self.memory[players_names_pairing] = pairing_actions

    def get_players_names_pairings(self) -> list[tuple]:
        return list(self.memory.keys())

    def get_actions_by_pairing(self, players_names_pairing: tuple) -> list[tuple]:
        if not isinstance(players_names_pairing, tuple):
            raise ValueError("The pairing of players must be a tuple")
        if players_names_pairing not in self.memory.keys():
            warnings.warn(f"The pairing {players_names_pairing} is not present in the memory")
            return []
        return self.memory[players_names_pairing]

    def append(self, new_memory):
        """
        Update the memory of the player by concatenating the actions played in the last iterations.\n
        :param new_memory: memory to be added
        """
        if not isinstance(new_memory, PlayerMemory):
            raise ValueError("The memory to be added must be a PlayerMemory object")
        for players_names_pairing in new_memory.get_players_names_pairings():
            if players_names_pairing not in self.memory.keys():
                self.memory[players_names_pairing] = new_memory.get_actions_by_pairing(players_names_pairing)
            else:
                self.memory[players_names_pairing] += new_memory.get_actions_by_pairing(players_names_pairing)

    def get_memory_by_player_name(self, player_name: str):
        """
        Get the memory of a specific player as a dictionary.\n
        :param player_name: name of the player whose memory is requested
        :return: a PlayerMemory with the list of action where the player is involved
        """
        if not isinstance(player_name, str):
            warnings.warn("The player name should be a string")
        if not any(player_name in pairing for pairing in self.memory.keys()):
            warnings.warn(f"The player {player_name} is not present in the memory")
            return PlayerMemory()
        requested_player_memory = PlayerMemory()
        for pairing in self.memory.keys():
            if player_name in pairing:
                requested_player_memory.add_element(pairing, self.memory[pairing])
        return requested_player_memory

    def __str__(self):
        to_str = ""
        for pairing in self.memory.keys():
            to_str += f"{pairing}: {self.memory[pairing]}\n"
        return to_str

    def __bool__(self):
        return bool(self.memory)

    # Python 2 compatibility
    __nonzero__ = __bool__

    def __eq__(self, other):
        if not isinstance(other, PlayerMemory):
            return False
        return self.memory == other.memory
