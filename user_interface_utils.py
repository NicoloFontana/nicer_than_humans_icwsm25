import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from src.checkers.checkers_utils import old_to_new_label, new_label_to_full_question, old_to_new_checker_name, get_checkers_by_names
from src.games.two_players_pd import TwoPlayersPD
from src.player import Player
from src.strategies.blind_pd_strategies import RandomStrategy
from src.strategies.one_vs_one_pd_llm_strategy import OneVsOnePDLlmStrategy

c_blue1 = '#5c509d'
c_green1 = '#7dc0a7'
c_green2 = '#39d05c'
c_violet1 = '#301466'
c_gray1 = '#444444'
c_orange1 = '#ECA72C'

checkers_names = ["time", "rule", "aggregation"]  # TODO FIX CHECKERS NAMES
base_out_dir = Path("out")
comprehension_questions_out_dir = base_out_dir / "comprehension_questions"


def evaluate_comprehension_questions(model_client, n_games=2, n_iterations=5, history_window_size=5, checkpoint=2, run_description=None):
    if checkpoint == 0:
        checkpoint = n_iterations + 1
    model_name = model_client.model_name
    dt_start_time = dt.datetime.now()
    start_time = time.mktime(dt_start_time.timetuple())
    comprehension_questions_out_dir.mkdir(parents=True, exist_ok=True)
    if run_description is None:
        run_description = f"Running {model_name} against a random opponent in {n_games} games of {n_iterations} iterations each."
    print(run_description)
    first_player_name = "A"
    second_player_name = "B"
    for n_game in range(n_games):
        current_game = n_game + 1
        print(f"Game {current_game}")
        checkers = get_checkers_by_names(checkers_names)
        # Set up the game
        game = TwoPlayersPD(iterations=n_iterations)

        first_player = Player(first_player_name)
        first_player_strategy = OneVsOnePDLlmStrategy(game, model_client, history_window_size=history_window_size, checkers=checkers)
        first_player.set_strategy(first_player_strategy)
        game.add_player(first_player)

        second_player = Player(second_player_name)
        second_player.set_strategy(RandomStrategy())
        game.add_player(second_player)

        for iteration in range(n_iterations):
            current_round = iteration + 1
            print(f"Round {current_round}") if current_round % checkpoint == 0 else None
            if not game.is_ended:
                game.play_game_round()
                checkpoint_dir = comprehension_questions_out_dir if current_round % checkpoint == 0 else None
                infix = f"{current_game}_{current_round}"
                first_player_strategy.wrap_up_round(out_dir=checkpoint_dir, infix=infix)
                print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}") if checkpoint_dir is not None else None
        infix = f"{current_game}"
        first_player_strategy.wrap_up_round(out_dir=comprehension_questions_out_dir, infix=infix)
        game.save_history(out_dir=comprehension_questions_out_dir, infix=infix)
        print(f"Time elapsed: {dt.timedelta(seconds=int(time.time() - start_time))}")

    create_csv_for_results(n_games)


def plot_comprehension_questions_results(model_name, n_games, n_iterations, history_window_size):
    df_questions = pd.read_csv(comprehension_questions_out_dir / f"comprehension_questions_results.csv")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(df_questions['label'], df_questions['accuracy'], marker='o',
               s=70, zorder=10, alpha=0.9, c=c_blue1)
    ax.errorbar(df_questions['label'], df_questions['accuracy'],
                # TODO check yerr
                yerr=df_questions['ci_ub']-df_questions['ci_lb'],
                # yerr=(df_questions['ci_ub'] - df_questions['ci_lb']) / 2,
                capsize=2, fmt='none', c=c_blue1)

    ax.set_xlabel('Comprehension questions', fontsize=24)
    ax.set_ylabel('Response accuracy', fontsize=24)

    x_labels = ['min_max', 'actions', 'payoff', 'round', 'action$_i$', 'points$_i$', '#actions', '#points']
    ax.set_ylim(-0.01, 1.1)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=18)

    ax.spines[['right', 'top', 'bottom']].set_visible(False)

    # lgnd = plt.legend(fontsize=15, loc="lower left", handletextpad=0.05,
    #                   ncol=1)  # bbox_to_anchor=(0.1, 0.1), frameon=False
    # lgnd.get_frame().set_facecolor('white')
    # lgnd.get_frame().set_linewidth(0.0)

    ax.vlines(2.5, 0, 1.1, color='gray', linewidth=1)
    ax.vlines(5.5, 0, 1.1, color='gray', linewidth=1)

    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1)
    ax.text(0.18, 1.05, 'Rules', fontsize=20,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.56, 1.05, 'Time', fontsize=20,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.89, 1.05, 'State', fontsize=20,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f'{model_name}_{n_games}games_{n_iterations}rounds_{history_window_size}window.pdf')
    plt.savefig(f'{model_name}_{n_games}games_{n_iterations}rounds_{history_window_size}window.png')


def create_csv_for_results(n_games):
    checkers_csv = []
    confidence = 0.95
    for checker_name in checkers_names:
        new_checker_name = old_to_new_checker_name(checker_name)
        data = {}
        for n_game in range(n_games):
            checker_path = comprehension_questions_out_dir / f"{checker_name}_checker" / "complete_answers"
            file_name = f"complete_answers_{n_game + 1}.json"
            with open(checker_path / file_name, "r") as f:
                complete_answers = json.load(f)
            for old_label in complete_answers.keys():
                # if "combo" in old_label:
                #     continue
                new_label = old_to_new_label(old_label)
                if new_label not in data.keys():
                    data[new_label] = {}
                    data[new_label][checker_name] = checker_name
                    data[new_label]["full_question"] = new_label_to_full_question(new_label)
                    data[new_label]["values"] = []
                    data[new_label]["mean"] = None
                    data[new_label]["lb"] = None
                    data[new_label]["ub"] = None
                for idx in complete_answers[old_label].keys():
                    data[new_label]["values"].append(complete_answers[old_label][idx]["answer"]["is_correct"])
        for new_label in data.keys():
            data[new_label]["mean"] = np.mean(data[new_label]["values"])
            ci = st.norm.interval(confidence, loc=data[new_label]["mean"], scale=st.sem(data[new_label]["values"]))
            data[new_label]["lb"] = ci[0] if not np.isnan(ci[0]) else data[new_label]["mean"]
            data[new_label]["ub"] = ci[1] if not np.isnan(ci[1]) else data[new_label]["mean"]

        for new_label in data.keys():
            question = {}
            question["type"] = new_checker_name
            question["full_question"] = data[new_label]["full_question"]
            question["label"] = new_label
            question["accuracy"] = data[new_label]["mean"]
            question["ci_lb"] = data[new_label]["lb"]
            question["ci_ub"] = data[new_label]["ub"]
            checkers_csv.append(question)
    df = pd.DataFrame(checkers_csv)
    df.to_csv(comprehension_questions_out_dir / f"comprehension_questions_results.csv")
