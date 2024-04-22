import datetime as dt
import sys
import time
from src.games.two_players_pd_utils import player_1_, player_2_, main_blind_strategies, main_hard_coded_strategies, get_main_strategies, compute_behavioral_profile, \
    main_behavioral_features
from src.games.two_players_pd import TwoPlayersPD
from src.player import Player
from src.strategies.blind_pd_strategies import RandomStrategy
from src.utils import timestamp, log, start_time, dt_start_time

n_games = 1000
n_iterations = 100
checkpoint = 100
main_strategies = get_main_strategies()

msg = ("Run the main strategies against both RND and themselves, and compute the behavioral profile for each one in both cases."
       f"The main strategies are the following: {main_strategies.keys()}."
       f"The behavioral features composing the behavioral profile are the following: {main_behavioral_features.keys()}.")

if msg == "":
    log.info("Set a message.")
    sys.exit()
log.info(msg)
print(msg)

log.info(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Starting time: {dt_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

for strategy_name in main_strategies.keys():
    log.info(f"Strategy: {strategy_name}")
    print(f"Strategy: {strategy_name}")
    args = strategy_name in main_hard_coded_strategies.keys()
    for i in range(2):  # 0: against RND, 1: against self
        log.info(f"Opponent: RND") if i == 0 else log.info(f"Opponent: self")
        print(f"Opponent: RND") if i == 0 else print(f"Opponent: self")
        game_histories = []
        opponent_name = "rnd_baseline" if i == 0 else "self"
        main_name = main_strategies[strategy_name]["label"]
        for n_game in range(n_games):
            log.info(f"Game {n_game + 1}") if checkpoint != 0 and (n_game + 1) % checkpoint == 0 else None
            print(f"Game {n_game + 1}") if checkpoint != 0 and (n_game + 1) % checkpoint == 0 else None
            game = TwoPlayersPD(iterations=n_iterations)
            game.add_player(Player(opponent_name))
            game.add_player(Player(main_name))
            if i == 0:
                strat1 = RandomStrategy()
            else:
                strat1 = main_strategies[strategy_name]["strategy"](game, main_name) if args else main_strategies[strategy_name]["strategy"]()
            strat2 = main_strategies[strategy_name]["strategy"](game, main_name) if args else main_strategies[strategy_name]["strategy"]()
            for player in game.players.values():
                if player.get_name() == opponent_name:
                    player.set_strategy(strat1)
                else:
                    player.set_strategy(strat2)
            for iteration in range(n_iterations):
                curr_round = iteration + 1
                if not game.is_ended:
                    game.play_round()
            infix = f"{main_name}-{opponent_name}_{n_game + 1}"
            game.save_history(timestamp, infix=infix, subdir=main_name)
            game_histories.append(game.get_history())
            log.info(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if checkpoint != 0 and (n_game + 1) % checkpoint == 0 else None
            print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if checkpoint != 0 and (n_game + 1) % checkpoint == 0 else None

        behavioral_profile = compute_behavioral_profile(game_histories, main_behavioral_features, main_player_name=main_name, opponent_name=opponent_name)
