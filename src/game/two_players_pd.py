import datetime as dt
import time
import warnings

from src.analysis.checkers_utils import get_checkers_by_names
from src.game.gt_game import GTGame
from src.game.two_players_pd_utils import player_1_, player_2_, two_players_pd_axelrod_payoff
from src.model_client import ModelClient
from src.game.player import Player
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy
from src.strategies.strategy_utils import get_strategy_instance


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
        super().__init__(players, iterations, action_space={1, 0}, payoff_function=two_players_pd_axelrod_payoff)

    def play_game_round(self):  # , memories=None):
        player_one = self.players[list(self.players)[0]]
        player_two = self.players[list(self.players)[1]]
        # if memories is None:
        #     player_one.set_memory(PlayerMemory())
        #     player_two.set_memory(PlayerMemory())
        # else:
        #     player_one.set_memory(memories[player_one.get_name()])
        #     player_two.set_memory(memories[player_two.get_name()])

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
        super().play_game_round()

    def add_player(self, new_player: Player):
        if isinstance(new_player, Player):
            if len(self.players.keys()) < 2:
                super().add_player(new_player)
            else:
                warnings.warn("There are already 2 players")
        else:
            raise TypeError("The player must be an instance of the class Player")

    def get_opponent_name(self, player_name):
        for name in self.players.keys():
            if name != player_name:
                return name
        warnings.warn(f"The player {player_name} is not present in the game")
        return None


def play_two_players_pd(out_dir, first_strategy_id, second_strategy_id, first_strategy_args=None, second_strategy_args=None, n_games=2, n_iterations=5, history_window_size=5,
                        checkpoint=2, run_description=None, ask_questions=False, verbose=True):
    if checkpoint == 0:
        checkpoint = n_iterations + 1
    if isinstance(first_strategy_id, ModelClient):
        first_strategy_name = first_strategy_id.model_name
    else:
        first_strategy_name = first_strategy_id
    if isinstance(second_strategy_id, ModelClient):
        second_strategy_name = second_strategy_id.model_name
    else:
        second_strategy_name = second_strategy_id
    dt_start_time = dt.datetime.now()
    start_time = time.mktime(dt_start_time.timetuple())
    out_dir.mkdir(parents=True, exist_ok=True)
    first_player_name = player_1_
    second_player_name = player_2_
    if run_description is None:
        run_description = f"Running {first_strategy_name} as '{first_player_name}' against {second_strategy_name} as '{second_player_name}' in {n_games} games of {n_iterations} iterations each with window size {history_window_size}."
        if ask_questions:
            run_description += " Asking comprehension questions."
    print(run_description)
    for n_game in range(n_games):
        current_game = n_game + 1
        print(f"Game {current_game}") if verbose else None
        checkers_names = ["time", "rule", "aggregation"] if ask_questions else []
        checkers = get_checkers_by_names(checkers_names)
        # Set up the game
        game = TwoPlayersPD(iterations=n_iterations)

        first_player = Player(first_player_name)

        if isinstance(first_strategy_id, ModelClient):
            first_player_strategy = OneVsOnePDLlmStrategy(game, first_strategy_id, history_window_size=history_window_size, checkers=checkers)
        else:
            first_player_strategy = get_strategy_instance(first_strategy_id, args=first_strategy_args)
        first_player.set_strategy(first_player_strategy)
        game.add_player(first_player)

        second_player = Player(second_player_name)
        if isinstance(second_strategy_id, ModelClient):
            second_player_strategy = OneVsOnePDLlmStrategy(game, second_strategy_id, history_window_size=history_window_size, checkers=checkers)
        else:
            second_player_strategy = get_strategy_instance(second_strategy_id, args=second_strategy_args)
        second_player.set_strategy(second_player_strategy)
        game.add_player(second_player)

        for iteration in range(n_iterations):
            current_round = iteration + 1
            print(f"Round {current_round}") if current_round % checkpoint == 0 else None
            if not game.is_ended:
                game.play_game_round()
                checkpoint_dir = out_dir if current_round % checkpoint == 0 else None
                infix = f"{current_game}_{current_round}"
                if isinstance(first_strategy_id, ModelClient):
                    first_player_strategy.wrap_up_round(out_dir=checkpoint_dir, infix=infix)
                if isinstance(second_strategy_id, ModelClient):
                    second_player_strategy.wrap_up_round(out_dir=checkpoint_dir, infix=infix)
                print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if checkpoint_dir is not None else None
        infix = f"{current_game}"
        if isinstance(first_strategy_id, ModelClient):
            first_player_strategy.wrap_up_round(out_dir=out_dir, infix=infix)
        if isinstance(second_strategy_id, ModelClient):
            second_player_strategy.wrap_up_round(out_dir=out_dir, infix=infix)
        game.save_history(out_dir=out_dir, infix=infix)
        print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if verbose or n_game == n_games - 1 else None
