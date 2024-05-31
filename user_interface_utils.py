import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt

from src.checkers.checkers_utils import old_to_new_label, new_label_to_full_question, old_to_new_checker_name
from src.games.two_players_pd import play_two_players_pd
from src.games.two_players_pd_utils import extract_histories_from_files, player_1_
from src.utils import compute_average_vector, compute_confidence_interval_vectors

c_blue1 = '#5c509d'
c_green1 = '#7dc0a7'
c_green2 = '#39d05c'
c_violet1 = '#301466'
c_gray1 = '#444444'
c_orange1 = '#ECA72C'

checkers_names = ["time", "rule", "aggregation"]  # TODO FIX CHECKERS NAMES
base_out_dir = Path("out")
comprehension_questions_dir = "comprehension_questions"
window_size_effect_dir = "window_size_effect"
confidence = 0.95


def evaluate_comprehension_questions(model_client, n_games=2, n_iterations=5, history_window_size=5, checkpoint=2, run_description=None):
    out_dir = base_out_dir / model_client.model_name / comprehension_questions_dir / f"g{n_games}_i{n_iterations}_w{history_window_size}"
    play_two_players_pd(out_dir, model_client, "RND", n_games=n_games, n_iterations=n_iterations, history_window_size=history_window_size, checkpoint=checkpoint,
                        run_description=run_description, ask_questions=True)

    create_csv_comprehension_questions_results(out_dir, n_games)


def create_csv_comprehension_questions_results(out_dir, n_games):
    checkers_csv = []
    for checker_name in checkers_names:
        new_checker_name = old_to_new_checker_name(checker_name)
        data = {}
        for n_game in range(n_games):
            checker_path = out_dir / f"{checker_name}_checker" / "complete_answers"
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
    df.to_csv(out_dir / f"comprehension_questions_results.csv")


def plot_comprehension_questions_results(model_name, n_games, n_iterations, history_window_size):
    out_dir = base_out_dir / model_name / comprehension_questions_dir / f"g{n_games}_i{n_iterations}_w{history_window_size}"
    df_questions = pd.read_csv(out_dir / f"comprehension_questions_results.csv")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(df_questions['label'], df_questions['accuracy'], marker='o',
               s=70, zorder=10, alpha=0.9, c=c_blue1)
    ax.errorbar(df_questions['label'], df_questions['accuracy'],
                # TODO check yerr
                yerr=df_questions['ci_ub'] - df_questions['ci_lb'],
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
    file_path = out_dir / f'{model_name}_comprehension_questions'
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def get_window_size_effect_dir(model_name, n_games, n_iterations, history_window_size):
    model_window_size_effect_dir = get_model_window_size_effect_dir(model_name)
    return model_window_size_effect_dir / f"g{n_games}_i{n_iterations}_w{history_window_size}"


def get_model_window_size_effect_dir(model_name):
    return base_out_dir / model_name / window_size_effect_dir


def evaluate_window_size_effect(model_client, history_window_size, n_games=2, n_iterations=5, checkpoint=2, run_description=None):
    out_dir = get_window_size_effect_dir(model_client.model_name, n_games, n_iterations, history_window_size)
    play_two_players_pd(out_dir, model_client, "AD", n_games=n_games, n_iterations=n_iterations, history_window_size=history_window_size, checkpoint=checkpoint,
                        run_description=run_description, ask_questions=False)

    create_csv_window_size_effect_results(out_dir, player_1_, n_iterations, history_window_size)


def create_csv_window_size_effect_results(out_dir, player_name, n_iterations, history_window_size):
    game_histories = extract_histories_from_files(out_dir / "game_histories", f"game_history")
    main_histories = [game_history.get_actions_by_player(player_name) for game_history in game_histories]
    avg_main_history = compute_average_vector(main_histories)
    avg_ci_lbs, avg_ci_ubs = compute_confidence_interval_vectors(main_histories, confidence=confidence)

    avg_coop_csv_file = []
    for iteration_idx in range(len(main_histories[0])):
        iteration = {
            "iteration": iteration_idx + 1,
            "mean": avg_main_history[iteration_idx],
            "ci_lb": avg_ci_lbs[iteration_idx] if not np.isnan(avg_ci_lbs[iteration_idx]) else avg_main_history[iteration_idx],
            "ci_ub": avg_ci_ubs[iteration_idx] if not np.isnan(avg_ci_ubs[iteration_idx]) else avg_main_history[iteration_idx],
        }
        avg_coop_csv_file.append(iteration)
    df = pd.DataFrame(avg_coop_csv_file)
    df.to_csv(out_dir / f"average_cooperation.csv")

    steady_state_round = 10 if n_iterations > 10 else 0
    steady_state_main_histories = [sum(main_history[-min(n_iterations, 10):]) / len(main_history[-min(n_iterations, 10):]) for main_history in main_histories]
    steady_state_ci = st.norm.interval(confidence=confidence, loc=np.mean(steady_state_main_histories), scale=st.sem(steady_state_main_histories))
    steady_state_mean = sum(steady_state_main_histories) / len(steady_state_main_histories)

    steady_state_csv_file = []
    element = {
        "window_size": history_window_size,
        "mean": steady_state_mean,
        "ci_lb": steady_state_ci[0] if not np.isnan(steady_state_ci[0]) else steady_state_mean,
        "ci_ub": steady_state_ci[1] if not np.isnan(steady_state_ci[1]) else steady_state_mean
    }
    steady_state_csv_file.append(element)
    df = pd.DataFrame(steady_state_csv_file)
    df.to_csv(out_dir / f"steady_state_cooperation.csv")


def plot_window_size_effect_comparison(model_name, first_dir, first_window_size, second_dir, second_window_size, with_confidence_intervals=True):
    out_dir = get_model_window_size_effect_dir(model_name)
    df_win1 = pd.read_csv(first_dir / "average_cooperation.csv")
    df_win2 = pd.read_csv(second_dir / "average_cooperation.csv")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(df_win1['iteration'], df_win1['mean'], marker='.', markersize=5, zorder=2, alpha=1.0, c=c_blue1, label=f"ws {first_window_size}")
    ax.fill_between(df_win1['iteration'], df_win1['ci_lb'], df_win1['ci_ub'], color=c_blue1, alpha=0.3, edgecolor="none", zorder=0) if with_confidence_intervals else None

    ax.plot(df_win2['iteration'], df_win2['mean'], marker='.', linestyle='-', markersize=5, zorder=2, alpha=1.0, c=c_green1, label=f"ws {second_window_size}")
    ax.fill_between(df_win2['iteration'], df_win2['ci_lb'], df_win2['ci_ub'], color=c_green1, alpha=0.3, edgecolor="none", zorder=0) if with_confidence_intervals else None

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1)

    ax.set_ylabel('$p_{coop}$ vs. Always Defect', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Round', fontsize=18)

    ax.set_ylim(-0.01, 1.1)

    plt.legend()
    plt.tight_layout()
    file_path = out_dir / f'{model_name}_{first_window_size}w_vs_{second_window_size}w'
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_steady_state_cooperation_per_window_sizes(model_name, dirs, window_sizes):
    out_dir = get_model_window_size_effect_dir(model_name)
    csv_file = []
    for window_size in window_sizes:
        wdw_df = pd.read_csv(dirs[window_sizes.index(window_size)] / "steady_state_cooperation.csv")
        element = {
            "window_size": window_size,
            "mean": wdw_df["mean"][0],
            "ci_lb": wdw_df["ci_lb"][0],
            "ci_ub": wdw_df["ci_ub"][0]
        }
        csv_file.append(element)
    df = pd.DataFrame(csv_file)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(df['window_size'], df['mean'], marker='.', markersize=5, zorder=2, alpha=1.0, c=c_orange1)
    ax.fill_between(df['window_size'], df['ci_lb'], df['ci_ub'], color=c_orange1, alpha=0.3, edgecolor="none", zorder=0)

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.set_xlim([0, 40])
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xlabel('Window size', fontsize=18)
    ax.set_yticks([0, 0.2, 0.4, 0.6])

    plt.legend()
    plt.tight_layout()
    min_wdw = min(window_sizes)
    max_wdw = max(window_sizes)
    file_path = out_dir / f'{model_name}_steady_state_cooperation_{min_wdw}w-{max_wdw}w'
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))
