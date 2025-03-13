import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
import seaborn as sns

from sfem_computation import compute_sfem_from_labels
from src.analysis.behavioral_profile import behavioral_dimensions
from src.analysis.checkers_utils import old_to_new_label, new_label_to_full_question, old_to_new_checker_name
from src.game.two_players_pd import play_two_players_pd
from src.game.two_players_pd_utils import extract_histories_from_files, player_1_, player_2_
from src.strategies.strategy_utils import compute_behavioral_profile
from src.utils import compute_average_vector, compute_confidence_interval_vectors

c_blue1 = '#5c509d'
c_green1 = '#7dc0a7'
c_green2 = '#39d05c'
c_violet1 = '#301466'
c_gray1 = '#444444'
c_orange1 = '#ECA72C'

checkers_names = ["time", "rule", "aggregation"]
comprehension_questions_labels = ['min_max', 'actions', 'payoff', 'round', 'action$_i$', 'points$_i$', '#actions', '#points']
sfem_strategies_labels = ['AD', 'RND', 'AC', 'TFT', 'STFT', 'GRIM', 'WSLS']
base_out_dir = Path("out") / "analysis"
comprehension_questions_dir = "comprehension_questions"
window_size_effect_dir = "window_size_effect"
behavioral_analysis_dir = "behavioral_analysis"
urnd_str = "URND{p:02}"
rnd_str = "RND"
confidence = 0.95

max_p = 10


def get_comprehension_questions_dir(model_name, timestamp, n_games, n_iterations, history_window_size):
    return base_out_dir / model_name / comprehension_questions_dir / f"{timestamp}_g{n_games}_i{n_iterations}_w{history_window_size}"


def evaluate_comprehension_questions(out_dir, logger, model_client, n_games=2, n_iterations=5, history_window_size=5, checkpoint=2, run_description=None):
    play_two_players_pd(out_dir, logger, model_client, "RND", n_games=n_games, n_iterations=n_iterations, history_window_size=history_window_size, checkpoint=checkpoint,
                        run_description=run_description, ask_questions=True)


def create_csv_comprehension_questions_results(out_dir, n_games, use_light_complete_answers=False):
    checkers_csv = []
    for checker_name in checkers_names:
        new_checker_name = old_to_new_checker_name(checker_name)
        data = {}
        for n_game in range(n_games):
            if use_light_complete_answers:
                checker_path = out_dir / f"{checker_name}_checker" / "light_answers"
                file_name = f"light_answers_{n_game + 1}.json"
            else:
                checker_path = out_dir / f"{checker_name}_checker" / "complete_answers"
                file_name = f"complete_answers_{n_game + 1}.json"
            with open(checker_path / file_name, "r") as f:
                complete_answers = json.load(f)
            for old_label in complete_answers.keys():
                new_label = old_to_new_label(old_label)
                if new_label is None:
                    continue
                if new_label not in data.keys():
                    data[new_label] = {}
                    data[new_label][checker_name] = checker_name
                    data[new_label]["full_question"] = new_label_to_full_question(new_label)
                    data[new_label]["values"] = []
                    data[new_label]["mean"] = None
                    data[new_label]["lb"] = None
                    data[new_label]["ub"] = None
                if use_light_complete_answers:
                    data[new_label]["values"] += complete_answers[old_label]["answers"]
                else:
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


def plot_comprehension_questions_results(out_dir, model_name, out_fig_name=None):
    df_questions = pd.read_csv(out_dir / f"comprehension_questions_results.csv")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(df_questions['label'], df_questions['accuracy'], marker='o',
               s=70, zorder=10, alpha=0.9, c=c_blue1)
    ax.errorbar(df_questions['label'], df_questions['accuracy'],
                yerr=df_questions['ci_ub'] - df_questions['ci_lb'],
                capsize=2, fmt='none', c=c_blue1)

    # ax.set_xlabel('Comprehension questions', fontsize=24)
    # ax.set_ylabel(f'{model_name} response accuracy', fontsize=24)

    x_labels = comprehension_questions_labels
    ax.set_ylim(-0.05, 1.05)
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
    out_fig_name = f'{model_name}_comprehension_questions' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def get_window_size_comparison_dir(model_name, timestamp):
    return base_out_dir / model_name / window_size_effect_dir / f"{timestamp}"


def get_window_size_effect_dir(model_name, timestamp, n_games, n_iterations, history_window_size):
    model_window_size_effect_dir = get_window_size_comparison_dir(model_name, timestamp)
    return model_window_size_effect_dir / f"g{n_games}_i{n_iterations}_w{history_window_size}"


def evaluate_window_size_effect(out_dir, logger, model_client, history_window_size, n_games=2, n_iterations=5, checkpoint=2, run_description=None):
    play_two_players_pd(out_dir, logger, model_client, "AD", n_games=n_games, n_iterations=n_iterations, history_window_size=history_window_size, checkpoint=checkpoint,
                        run_description=run_description, ask_questions=False)


def create_csv_average_cooperation_per_round(out_dir):
    game_histories = extract_histories_from_files(out_dir / "game_histories", f"game_history")
    main_histories = [game_history.get_actions_by_player(player_1_) for game_history in game_histories]
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
    df.to_csv(out_dir / f"average_cooperation_per_round.csv")


def create_csv_window_size_effect_results(out_dir, history_window_size):
    game_histories = extract_histories_from_files(out_dir / "game_histories", f"game_history")
    main_histories = [game_history.get_actions_by_player(player_1_) for game_history in game_histories]
    n_iterations = len(main_histories[0])
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
    df.to_csv(out_dir / f"average_cooperation_per_round.csv")

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


def create_csv_steady_state_cooperation_per_window_sizes(out_dir, dirs):
    csv_file = []
    for window_size in dirs.keys():
        wdw_df = pd.read_csv(dirs[window_size] / "steady_state_cooperation.csv")
        element = {
            "window_size": window_size,
            "mean": wdw_df["mean"][0],
            "ci_lb": wdw_df["ci_lb"][0],
            "ci_ub": wdw_df["ci_ub"][0]
        }
        csv_file.append(element)
    df = pd.DataFrame(csv_file)
    df.to_csv(out_dir / f"steady_state_cooperation_per_window_sizes.csv")


def plot_window_size_effect_comparison(out_dir, model_name, first_dir, first_window_size, second_dir, second_window_size, with_confidence_intervals=True, out_fig_name=None):
    df_win1 = pd.read_csv(first_dir / "average_cooperation_per_round.csv")
    df_win2 = pd.read_csv(second_dir / "average_cooperation_per_round.csv")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(df_win1['iteration'], df_win1['mean'], marker='.', linestyle='--', markersize=5, zorder=2, alpha=0.5, c=c_gray1, label=f"ws {first_window_size}")
    ax.fill_between(df_win1['iteration'], df_win1['ci_lb'], df_win1['ci_ub'], color=c_gray1, alpha=0.1, edgecolor="none", zorder=0) if with_confidence_intervals else None

    ax.plot(df_win2['iteration'], df_win2['mean'], marker='.', markersize=5, zorder=2, alpha=1.0, c=c_green2, label=f"ws {second_window_size}")
    ax.fill_between(df_win2['iteration'], df_win2['ci_lb'], df_win2['ci_ub'], color=c_green2, alpha=0.3, edgecolor="none", zorder=0) if with_confidence_intervals else None

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1)

    # ax.set_ylabel(f'{model_name} $p_{{coop}}$ vs. Always Defect', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Round', fontsize=24)

    ax.set_ylim(-0.05, 1.05)

    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout()
    # out_fig_name = f'{model_name}_{first_window_size}w_vs_{second_window_size}w' if out_fig_name is None else out_fig_name
    out_fig_name = f'{model_name}_ad_ws100_ws10'
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))
    return fig, ax


def plot_steady_state_cooperation_per_window_sizes(out_dir, model_name, out_fig_name=None, fig_ax=None):
    df = pd.read_csv(out_dir / "steady_state_cooperation_per_window_sizes.csv")
    window_sizes = df['window_size']
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)) if fig_ax is None else fig_ax

    ax.plot(df['window_size'], df['mean'], marker='.', markersize=5, zorder=2, alpha=1.0, c=c_green1)
    ax.fill_between(df['window_size'], df['ci_lb'], df['ci_ub'], color=c_green1, alpha=0.3, edgecolor="none", zorder=0)

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.set_ylim([-0.05, 0.65])
    ax.set_xlim([0, 40])
    ax.tick_params(axis='both', labelsize=14)
    # ax.set_ylabel(f'{model_name} $p_{{coop}}$ vs. Always Defect', fontsize=18)
    ax.set_xlabel('Window size', fontsize=18)
    ax.set_yticks([0, 0.2, 0.4, 0.6])

    # plt.tight_layout()
    min_wdw = min(window_sizes)
    max_wdw = max(window_sizes)
    out_fig_name = f'{model_name}_steady_state_cooperation_inset' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def get_behavioral_analysis_dir(model_name, timestamp, n_games, n_iterations, history_window_size):
    return base_out_dir / model_name / behavioral_analysis_dir / f"{timestamp}_g{n_games}_i{n_iterations}_w{history_window_size}"


def compute_response_to_different_hostilities(out_dir, logger, model_client, n_games=2, n_iterations=5, history_window_size=5, checkpoint=2, run_description=None,
                                              already_run_against_ad=False):
    if already_run_against_ad:
        initial_p = 1
        in_dir = get_window_size_effect_dir(model_client.model_name, n_games, n_iterations, history_window_size)
        game_histories_dir_path = in_dir / "game_histories"
        shutil.copytree(str(game_histories_dir_path), str(out_dir / urnd_str.format(p=0) / "game_histories"), dirs_exist_ok=True)
    else:
        initial_p = 0
    for p in range(initial_p, max_p + 1):
        alpha = p / 10
        logger.info(urnd_str.format(p=p))
        print(urnd_str.format(p=p))
        urnd_out_dir = out_dir / urnd_str.format(p=p)
        play_two_players_pd(urnd_out_dir, model_client, "URND", second_strategy_args=alpha, n_games=n_games, n_iterations=n_iterations, history_window_size=history_window_size,
                            checkpoint=checkpoint,
                            run_description=run_description, ask_questions=False)


def compute_rnd_response_to_different_hostilities(rnd_out_dir):
    run_description = f"Running RND as 'A' against URNDx as 'B' in 100 games of 100 iterations."
    for p in range(max_p + 1):
        alpha = p / 10
        print(urnd_str.format(p=p))
        urnd_out_dir = rnd_out_dir / urnd_str.format(p=p)
        play_two_players_pd(urnd_out_dir, rnd_str, "URND", second_strategy_args=alpha, n_games=100, n_iterations=100, history_window_size=5,
                            checkpoint=0,
                            run_description=run_description, ask_questions=False, verbose=False)


def create_csv_cooperation_probability_results(out_dir):
    csv_file = []
    for p in range(max_p + 1):
        alpha = p / 10
        urnd_out_dir = out_dir / urnd_str.format(p=p)
        game_histories_dir_path = urnd_out_dir / "game_histories"
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)
        n_iterations = len(game_histories[0].get_actions_by_player(player_1_))

        main_history_ends = [game_history.get_actions_by_player(player_1_) for game_history in game_histories]
        mean_main_history_ends = [sum(main_history_end) / len(main_history_end) for main_history_end in main_history_ends]
        mean = sum(mean_main_history_ends) / len(mean_main_history_ends)
        ci = st.norm.interval(confidence=confidence, loc=np.mean(mean_main_history_ends), scale=st.sem(mean_main_history_ends))
        element = {
            "URND_alpha": alpha,
            "model_coop": mean,
            "ci_lb": ci[0] if not np.isnan(ci[0]) else mean,
            "ci_ub": ci[1] if not np.isnan(ci[1]) else mean
        }

        csv_file.append(element)
    df = pd.DataFrame(csv_file)
    df.to_csv(out_dir / f"coop_probability_vs_urnd_alpha.csv")


def create_csv_cooperation_probability_other_strats_results(out_dir, opponents_names):
    csv_file = []
    for strat_name in opponents_names:
        element = {
            "opponent": strat_name,
        }
        strat_out_dir = out_dir / strat_name
        game_histories_dir_path = strat_out_dir / "game_histories"
        if not game_histories_dir_path.exists():
            continue
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)

        main_history_ends = [game_history.get_actions_by_player(player_1_) for game_history in game_histories]
        mean_main_history_ends = [sum(main_history_end) / len(main_history_end) for main_history_end in main_history_ends]
        mean = sum(mean_main_history_ends) / len(mean_main_history_ends)
        ci = st.norm.interval(confidence=confidence, loc=np.mean(mean_main_history_ends), scale=st.sem(mean_main_history_ends))
        element["model_coop"] = mean
        element["ci_lb"] = ci[0] if not np.isnan(ci[0]) else mean
        element["ci_ub"] = ci[1] if not np.isnan(ci[1]) else mean
        csv_file.append(element)
    df = pd.DataFrame(csv_file)
    df.to_csv(out_dir / f"coop_probability_vs_other_strats.csv")


def create_csv_sfem_results(out_dir):
    csv_file = []
    for p in range(1, max_p):  # Skip alpha = 0.0 and alpha = 1.0 because SFEM results don't hold
        alpha = p / 10
        element = {
            "URND_alpha": alpha,
        }
        urnd_out_dir = out_dir / urnd_str.format(p=p)
        game_histories_dir_path = urnd_out_dir / "game_histories"
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)
        print(f"Computing SFEM for run against {urnd_str.format(p=p)}")
        scores = compute_sfem_from_labels(game_histories, sfem_strategies_labels, player_1_)

        for score, strategy_label in zip(scores, sfem_strategies_labels):
            element[f"{strategy_label}_score"] = score
        csv_file.append(element)
    df = pd.DataFrame(csv_file)
    df.to_csv(out_dir / f"sfem_scores_vs_urnd_alpha.csv")


def create_csv_sfem_other_strats_results(out_dir, opponents_names):
    csv_file = []
    for strat_name in opponents_names:
        element = {
            "opponent": strat_name,
        }
        strat_out_dir = out_dir / strat_name
        game_histories_dir_path = strat_out_dir / "game_histories"
        if not game_histories_dir_path.exists():
            continue
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)
        print(f"Computing SFEM for run against {strat_name}")
        scores = compute_sfem_from_labels(game_histories, sfem_strategies_labels, player_1_)
        for score, strategy_label in zip(scores, sfem_strategies_labels):
            element[f"{strategy_label}_score"] = score
        csv_file.append(element)
    df = pd.DataFrame(csv_file)
    df.to_csv(out_dir / f"sfem_scores_vs_other_strats.csv")


def create_csv_behavioral_profile_results(out_dir, model_name):
    model_csv_file = []
    for p in range(max_p + 1):
        alpha = p / 10
        urnd_out_dir = out_dir / urnd_str.format(p=p)

        # Compute the behavioral profile for the model
        model_element = {
            "URND_alpha": alpha,
        }
        game_histories_dir_path = urnd_out_dir / "game_histories"
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)
        history_main_name = player_1_
        history_opponent_name = player_2_
        profile = compute_behavioral_profile(game_histories, history_main_name, history_opponent_name)
        profile.strategy_name = model_name
        profile.opponent_name = urnd_str.format(p=p)
        for dimension_name in behavioral_dimensions.keys():
            dimension_values = profile.dimensions[dimension_name]
            mean = np.mean(dimension_values)
            cis = st.norm.interval(confidence, loc=mean, scale=st.sem(dimension_values))
            model_element[f"{dimension_name}_mean"] = mean
            model_element[f"{dimension_name}_ci_lb"] = cis[0] if not np.isnan(cis[0]) else mean
            model_element[f"{dimension_name}_ci_ub"] = cis[1] if not np.isnan(cis[1]) else mean
        model_csv_file.append(model_element)

    model_df = pd.DataFrame(model_csv_file)
    model_df.to_csv(out_dir / f"behavioral_profile_vs_urnd_alpha.csv")


def create_csv_behavioral_profile_other_strats_results(out_dir, model_name, opponents_names):
    model_csv_file = []
    for strat_name in opponents_names:
        model_element = {
            "opponent": strat_name,
        }
        strat_out_dir = out_dir / strat_name
        game_histories_dir_path = strat_out_dir / "game_histories"
        if not game_histories_dir_path.exists():
            continue
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)
        history_main_name = player_1_
        history_opponent_name = player_2_
        profile = compute_behavioral_profile(game_histories, history_main_name, history_opponent_name)
        profile.strategy_name = model_name
        profile.opponent_name = strat_name
        for dimension_name in behavioral_dimensions.keys():
            dimension_values = profile.dimensions[dimension_name]
            mean = np.mean(dimension_values)
            cis = st.norm.interval(confidence, loc=mean, scale=st.sem(dimension_values))
            model_element[f"{dimension_name}_mean"] = mean
            model_element[f"{dimension_name}_ci_lb"] = cis[0] if not np.isnan(cis[0]) else mean
            model_element[f"{dimension_name}_ci_ub"] = cis[1] if not np.isnan(cis[1]) else mean
        model_csv_file.append(model_element)

    model_df = pd.DataFrame(model_csv_file)
    model_df.to_csv(out_dir / f"behavioral_profile_vs_other_strats.csv")


def plot_coop_probability_vs_urnd_alpha(out_dir, model_name, out_fig_name=None):
    df_coop = pd.read_csv(out_dir / f"coop_probability_vs_urnd_alpha.csv")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    ax.plot(df_coop['URND_alpha'], df_coop[f'model_coop'], markersize=5, marker='o',
            c=c_violet1, alpha=1.0, zorder=4)
    ax.fill_between(df_coop['URND_alpha'], df_coop[f'ci_lb'], df_coop[f'ci_ub'],
                    color=c_blue1, edgecolor="none", alpha=0.5, zorder=0)

    ax.set_facecolor('white')

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.set_xticks(df_coop['URND_alpha'])
    # ax.set_ylabel(f'{model_name} $p_{{coop}}$', fontsize=14)
    ax.set_ylabel(f'$p_{{coop}}$', fontsize=14)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Opponent cooperation probability ($α$)', fontsize=14)
    plt.tight_layout()

    out_fig_name = f'{model_name}_coop_vs_alpha' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_coop_probability_vs_other_strats(out_dir, model_name, out_fig_name=None):
    df_coop = pd.read_csv(out_dir / f"coop_probability_vs_other_strats.csv")
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    ax.scatter(df_coop['opponent'], df_coop['model_coop'], marker='o',
               s=70, zorder=10, alpha=0.9, c=c_orange1)
    ax.errorbar(df_coop['opponent'], df_coop['model_coop'],
                # TODO check yerr
                yerr=df_coop['ci_ub'] - df_coop['ci_lb'],
                # yerr=(df_questions['ci_ub'] - df_questions['ci_lb']) / 2,
                capsize=2, fmt='none', c=c_orange1)

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.set_xticks(df_coop['opponent'])
    ax.set_ylabel(f'$p_{{coop}}$', fontsize=14)
    ax.set_ylim([-0.05, 1.05])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Opponent', fontsize=14)
    plt.tight_layout()

    out_fig_name = f'{model_name}_coop_vs_other_strats' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_sfem_results_vs_urnd_alpha(out_dir, model_name, out_fig_name=None):
    df_sfem = pd.read_csv(out_dir / f"sfem_scores_vs_urnd_alpha.csv")
    threshold = 0.2
    best_strategies = []
    for strategy_label in sfem_strategies_labels:
        if df_sfem[f'{strategy_label}_score'].max() > threshold:
            best_strategies.append(strategy_label)
    best_strategies.sort(key=lambda x: df_sfem[f'{x}_score'].max(), reverse=True)
    best_strategies = best_strategies[:min(3, len(best_strategies))]
    colors = [c_blue1, c_orange1, c_green1]
    markers = ['o', 'x', '^']
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    others_already_set = False
    for strategy_label in sfem_strategies_labels:
        alpha = 1 if strategy_label in best_strategies else 0.7
        color = colors[best_strategies.index(strategy_label)] if strategy_label in best_strategies else c_gray1
        marker = markers[best_strategies.index(strategy_label)] if strategy_label in best_strategies else ','
        label = strategy_label if strategy_label in best_strategies else None
        if label is None and not others_already_set:
            label = 'Others'
            others_already_set = True
        ax.plot(df_sfem['URND_alpha'], df_sfem[f'{strategy_label}_score'], marker=marker, markersize=5,
                c=color, zorder=4, alpha=alpha, label=label)

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.set_xticks(df_sfem['URND_alpha'])
    ax.set_ylabel(f'{model_name} SFEM score', fontsize=14)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0.05, 0.95])
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Opponent cooperation probability ($α$)', fontsize=14)
    plt.legend()
    plt.tight_layout()

    out_fig_name = f'{model_name}_sfem_vs_alpha' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_sfem_matrix(out_dir, sfem_candidates_names, out_fig_name=None):
    df_sfem = pd.read_csv(out_dir / f"sfem_scores_vs_other_strats.csv")
    df_sfem = df_sfem.drop(columns=df_sfem.columns[0])
    opponents_names = df_sfem['opponent']
    df_sfem = df_sfem.drop(columns=df_sfem.columns[0])
    matrix = df_sfem.to_numpy()
    matrix = matrix.T
    title = 'SFEM scores'
    xlabel = 'Opponent'
    ylabel = 'Candidate'
    vmin = 0
    xticklabels = opponents_names
    yticklabels = sfem_candidates_names
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(matrix, annot=True, xticklabels=xticklabels, yticklabels=yticklabels, cmap="YlOrBr", fmt='.0%', vmin=vmin)
    plt.title(title) if title is not None else None
    plt.xlabel(xlabel) if xlabel is not None else None
    plt.ylabel(ylabel) if ylabel is not None else None
    plt.tight_layout()

    out_fig_name = f'sfem_matrix' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_behavioral_profile_vs_urnd_alpha(out_dir, model_name, rnd_out_dir, out_fig_name=None):
    df_profile = pd.read_csv(out_dir / 'behavioral_profile_vs_urnd_alpha.csv')
    df_profile.at[10, 'forgiving_mean'] = float("NaN")
    df_profile.at[10, 'forgiving_ci_lb'] = float("NaN")
    df_profile.at[10, 'forgiving_ci_ub'] = float("NaN")
    df_profile.at[10, 'retaliatory_mean'] = float("NaN")
    df_profile.at[10, 'retaliatory_ci_lb'] = float("NaN")
    df_profile.at[10, 'retaliatory_ci_ub'] = float("NaN")
    df_profile.at[0, 'troublemaking_mean'] = float("NaN")
    df_profile.at[0, 'troublemaking_ci_lb'] = float("NaN")
    df_profile.at[0, 'troublemaking_ci_ub'] = float("NaN")

    df_profile_rnd = pd.read_csv(rnd_out_dir / f'behavioral_profile_vs_urnd_alpha.csv')
    df_profile_rnd.at[10, 'forgiving_mean'] = float("NaN")
    df_profile_rnd.at[10, 'forgiving_ci_lb'] = float("NaN")
    df_profile_rnd.at[10, 'forgiving_ci_ub'] = float("NaN")
    df_profile_rnd.at[10, 'retaliatory_mean'] = float("NaN")
    df_profile_rnd.at[10, 'retaliatory_ci_lb'] = float("NaN")
    df_profile_rnd.at[10, 'retaliatory_ci_ub'] = float("NaN")
    df_profile_rnd.at[0, 'troublemaking_mean'] = float("NaN")
    df_profile_rnd.at[0, 'troublemaking_ci_lb'] = float("NaN")
    df_profile_rnd.at[0, 'troublemaking_ci_ub'] = float("NaN")
    fig, axs = plt.subplots(5, 1, figsize=(5, 7))
    for i, (d, ax) in enumerate(zip(behavioral_dimensions.keys(), axs)):
        #    ax= axs[0]
        ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
        ax.plot(df_profile['URND_alpha'], df_profile[f'{d}_mean'], '-o', markersize=5, color=c_blue1,
                zorder=4, label=f'Llama2')
        ax.fill_between(df_profile['URND_alpha'], df_profile[f'{d}_ci_lb'], df_profile[f'{d}_ci_ub'],
                        color=c_blue1, edgecolor="none", alpha=0.5, zorder=3)

        ax.plot(df_profile_rnd['URND_alpha'], df_profile_rnd[f'{d}_mean'], '-', markersize=5, color=c_green1,
                zorder=2, alpha=0.8, label='Random')
        ax.fill_between(df_profile_rnd['URND_alpha'], df_profile_rnd[f'{d}_ci_lb'], df_profile_rnd[f'{d}_ci_ub'],
                        color=c_green1, edgecolor="none", alpha=0.2, zorder=1)

        ax.set_ylabel(d.capitalize(), fontsize=12)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.05, 1.05])
        ax.spines[['right', 'top']].set_visible(False)
        if i < 4:
            ax.spines[['bottom']].set_visible(False)
            ax.set_xticks([])

    lgnd = plt.legend(fontsize=12, loc="lower left", handletextpad=0.3,
                      ncol=2, frameon=False)
    lgnd.get_frame().set_facecolor('white')

    ax.set_xticks(df_profile['URND_alpha'])
    ax.set_xlabel('Opponent cooperation probability ($α$)', fontsize=14)
    plt.tight_layout()

    out_fig_name = f'{model_name}_behavioral_profile_vs_alpha' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_behavioral_profile_vs_other_strats(out_dir, model_name, opponents_names, out_fig_name=None):
    df_profile = pd.read_csv(out_dir / 'behavioral_profile_vs_other_strats.csv')
    fig, axs = plt.subplots(5, 1, figsize=(5, 7))
    for i, (d, ax) in enumerate(zip(behavioral_dimensions.keys(), axs)):
        ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
        ax.scatter(df_profile['opponent'], df_profile[f'{d}_mean'], marker='o',
                   s=70, zorder=10, alpha=0.9, c=c_orange1)
        ax.errorbar(df_profile['opponent'], df_profile[f'{d}_mean'],
                    # TODO check yerr
                    yerr=df_profile[f'{d}_ci_ub'] - df_profile[f'{d}_ci_lb'],
                    # yerr=(df_questions['ci_ub'] - df_questions['ci_lb']) / 2,
                    capsize=2, fmt='none', c=c_orange1)

        ax.set_ylabel(d.capitalize(), fontsize=12)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([-0.5, len(opponents_names) - 0.5])
        ax.hlines(0, -1.5, len(opponents_names) + 0.5, color='black', linewidth=1, linestyle='--')
        ax.spines[['right', 'top']].set_visible(False)
        if i < 4:
            ax.spines[['bottom']].set_visible(False)
            ax.set_xticks([])

    ax.set_xticks(df_profile['opponent'])
    ax.set_xlabel('Opponent', fontsize=14)
    plt.tight_layout()

    out_fig_name = f'{model_name}_behavioral_profile_vs_other_strats' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def create_csv_delta_behavioral_profile(out_dir, model_name, rnd_out_dir, abs_values=True):
    model_csv_file = []
    dimensions_overall_values = {d: [] for d in behavioral_dimensions.keys()}
    for p in range(max_p + 1):
        urnd_out_dir = out_dir / urnd_str.format(p=p)

        # Compute the delta behavioral profile for the model
        game_histories_dir_path = urnd_out_dir / "game_histories"
        file_name = f"game_history"
        game_histories = extract_histories_from_files(game_histories_dir_path, file_name)
        rnd_game_histories_dir_path = rnd_out_dir / urnd_str.format(p=p) / "game_histories"
        rnd_game_histories = extract_histories_from_files(rnd_game_histories_dir_path, file_name)
        n_games = min(len(game_histories), len(rnd_game_histories))
        for rnd_idx in range(n_games):
            urnd_history = game_histories[rnd_idx].get_actions_by_player(player_2_)
            rnd_game_histories[rnd_idx].substitute_history_of_player(player_2_, urnd_history)
        history_main_name = player_1_
        history_opponent_name = player_2_
        profile = compute_behavioral_profile(game_histories[:n_games], history_main_name, history_opponent_name)
        profile.strategy_name = model_name
        rnd_profile = compute_behavioral_profile(rnd_game_histories[:n_games], history_main_name, history_opponent_name)
        rnd_profile.strategy_name = rnd_str
        delta_profile = abs(profile - rnd_profile) if abs_values else profile - rnd_profile
        for dimension_name in behavioral_dimensions.keys():
            dimensions_overall_values[dimension_name].extend(delta_profile.dimensions[dimension_name])
    for dimension_name in behavioral_dimensions.keys():
        model_element = {
            "dimension": dimension_name,
        }
        dimension_values = dimensions_overall_values[dimension_name]
        mean = np.mean(dimension_values)
        cis = st.norm.interval(confidence, loc=mean, scale=st.sem(dimension_values))
        model_element[f"mean"] = mean
        model_element[f"ci_lb"] = cis[0] if not np.isnan(cis[0]) else mean
        model_element[f"ci_ub"] = cis[1] if not np.isnan(cis[1]) else mean
        model_csv_file.append(model_element)
    model_df = pd.DataFrame(model_csv_file)
    csv_file_name = "abs_delta_behavioral_profile_vs_urnd_alpha.csv" if abs_values else "delta_behavioral_profile_vs_urnd_alpha.csv"
    model_df.to_csv(out_dir / csv_file_name)


def plot_delta_behavioral_profile(out_dir, model_name, abs_values=True):
    csv_file_name = "abs_delta_behavioral_profile_vs_urnd_alpha.csv" if abs_values else "delta_behavioral_profile_vs_urnd_alpha.csv"
    df_profile = pd.read_csv(out_dir / csv_file_name)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    # for i, (d, ax) in enumerate(zip(behavioral_dimensions.keys(), axs)):

    ax.scatter(list(behavioral_dimensions.keys()), df_profile[f'mean'], marker='o',
               s=70, zorder=10, alpha=0.9, c=c_blue1)
    ax.errorbar(list(behavioral_dimensions.keys()), df_profile[f'mean'],
                # TODO check yerr
                yerr=df_profile[f'ci_ub'] - df_profile[f'ci_lb'],
                # yerr=(df_questions['ci_ub'] - df_questions['ci_lb']) / 2,
                capsize=2, fmt='none', c=c_blue1)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.grid(axis='x', color='gray', linestyle=':', linewidth=1, zorder=0)

    ax.hlines(0, -0.5, 4.5, color=c_gray1, linewidth=1, linestyle='--')

    ax.set_ylim([-0.05, 1.05]) if abs_values else ax.set_ylim([-1.05, 1.05])
    plt.tight_layout()

    img_file_name = f"{model_name}_abs_delta_behavioral_profile_vs_urnd_alpha" if abs_values else f"{model_name}_delta_behavioral_profile_vs_urnd_alpha"
    file_path = out_dir / img_file_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_comparison_delta_behavioral_profiles(first_out_dir, first_model_name, second_out_dir, second_model_name, abs_values=False):
    # TODO fix title, xticks and stuff...
    csv_file_name = "abs_delta_behavioral_profile_vs_urnd_alpha.csv" if abs_values else "delta_behavioral_profile_vs_urnd_alpha.csv"
    first_df_profile = pd.read_csv(first_out_dir / csv_file_name)
    second_df_profile = pd.read_csv(second_out_dir / csv_file_name)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))

    ax.scatter(list(behavioral_dimensions.keys()), first_df_profile[f'mean'], marker='o',
               s=70, zorder=10, alpha=0.9, c=c_blue1, label=first_model_name)
    ax.errorbar(list(behavioral_dimensions.keys()), first_df_profile[f'mean'],
                yerr=first_df_profile[f'ci_ub'] - first_df_profile[f'ci_lb'],
                capsize=2, fmt='none', c=c_blue1)

    ax.scatter(list(behavioral_dimensions.keys()), second_df_profile[f'mean'], marker='o',
               s=70, zorder=10, alpha=0.9, c=c_orange1, label=second_model_name)
    ax.errorbar(list(behavioral_dimensions.keys()), second_df_profile[f'mean'],
                yerr=second_df_profile[f'ci_ub'] - second_df_profile[f'ci_lb'],
                capsize=2, fmt='none', c=c_orange1)

    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.grid(axis='x', color='gray', linestyle=':', linewidth=1, zorder=0)

    ax.hlines(0, -0.5, 4.5, color=c_gray1, linewidth=1, linestyle='--')

    ax.set_ylim([-0.05, 1.05]) if abs_values else ax.set_ylim([-1.05, 1.05])
    plt.tight_layout()
    plt.legend()

    img_file_name = f"{first_model_name}_vs_{second_model_name}_abs_delta_behavioral_profile_vs_urnd_alpha" if abs_values else f"{first_model_name}_vs_{second_model_name}_delta_behavioral_profile_vs_urnd_alpha"
    file_path = first_out_dir / img_file_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def create_csv_statistical_tests(out_dir):
    first_defe_csv = []
    avg_defe_csv = []
    std_defe_csv = []
    game_histories_dir_path = out_dir / "game_histories"
    file_name = f"game_history"
    game_histories = extract_histories_from_files(game_histories_dir_path, file_name)
    for game_history in game_histories:
        main_history = game_history.get_actions_by_player(player_1_)
        n_iterations = len(main_history)
        first_defe_csv.append({
            'first_defection': main_history.index(0) / n_iterations if 0 in main_history else float("NaN")
        })
        defes = [i / n_iterations for i, x in enumerate(main_history) if x == 0]
        avg_defe_csv.append({
            'avg_defection': np.mean(defes) if len(defes) > 0 else float("NaN")
        })
        std_defe_csv.append({
            'std_defection': np.std(defes) if len(defes) > 0 else float("NaN")
        })
    first_defe_df = pd.DataFrame(first_defe_csv)
    avg_defe_df = pd.DataFrame(avg_defe_csv)
    std_defe_df = pd.DataFrame(std_defe_csv)
    first_defe_df.to_csv(out_dir / "first_defection.csv")
    avg_defe_df.to_csv(out_dir / "avg_defection.csv")
    std_defe_df.to_csv(out_dir / "std_defection.csv")


def plot_statistical_tests(out_dir, first_dir, first_label, second_dir, second_label):
    first_defe_name = "first_defection.csv"
    avg_defe_name = "avg_defection.csv"
    std_defe_name = "std_defection.csv"
    first_dfs = [pd.read_csv(first_dir / first_defe_name), pd.read_csv(first_dir / avg_defe_name), pd.read_csv(first_dir / std_defe_name)]
    second_dfs = [pd.read_csv(second_dir / first_defe_name), pd.read_csv(second_dir / avg_defe_name), pd.read_csv(second_dir / std_defe_name)]
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    first_avg_different = False
    avg_avg_different = False
    std_avg_different = False
    p_value_threshold = 0.05

    for i, (first_df, second_df) in enumerate(zip(first_dfs, second_dfs)):
        plt.figure(fig)
        plt.sca(ax)
        first_values = first_df.iloc[:, 1]
        second_values = second_df.iloc[:, 1]
        first_mean = np.mean(first_values)
        second_mean = np.mean(second_values)
        ax.scatter([i], [first_mean], marker='o', s=70, zorder=10, alpha=1.0, c=c_blue1)
        first_ci = st.norm.interval(confidence, loc=first_mean, scale=st.sem(first_values))
        ax.errorbar([i], [first_mean], yerr=[first_ci[1] - first_mean], capsize=2, fmt='none', c=c_blue1)
        ax.scatter([i], [second_mean], marker='^', s=70, zorder=10, alpha=1.0, c=c_orange1)
        second_ci = st.norm.interval(confidence, loc=second_mean, scale=st.sem(second_values))
        ax.errorbar([i], [second_mean], yerr=[second_ci[1] - second_mean], capsize=2, fmt='none', c=c_orange1)
        if i == 0:
            first_avg_different = st.ttest_ind(first_values, second_values, nan_policy='omit').pvalue < p_value_threshold
        if i == 1:
            avg_avg_different = st.ttest_ind(first_values, second_values, nan_policy='omit').pvalue < p_value_threshold
        if i == 2:
            std_avg_different = st.ttest_ind(first_values, second_values, nan_policy='omit').pvalue < p_value_threshold

    plt.figure(fig)
    plt.sca(ax)
    ax.scatter([], [], marker='o', s=70, zorder=10, alpha=1.0, c=c_blue1, label=first_label)
    ax.scatter([], [], marker='^', s=70, zorder=10, alpha=1.0, c=c_orange1, label=second_label)
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticklabels(["First defection" if first_avg_different else "First defection\nEQUAL", "Average defection" if avg_avg_different else "Average defection\nEQUAL",
                        "Std deviation" if std_avg_different else "Std deviation\nEQUAL"])
    plt.legend()

    plt.tight_layout()
    img_file_name = f"{first_label}-{second_label}_statistical_tests"
    file_path = out_dir / img_file_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))


def plot_cooperation_comparison(out_dir, model_name, dirs, labels, with_confidence_intervals=True, out_fig_name=None):
    dfs = [pd.read_csv(d / "average_cooperation_per_round.csv") for d in dirs]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    for df, label in zip(dfs, labels):
        ax.plot(df['iteration'], df['mean'], marker='.', markersize=5, zorder=2, alpha=1.0, label=label)
        ax.fill_between(df['iteration'], df['ci_lb'], df['ci_ub'], alpha=0.1, edgecolor="none", zorder=0) if with_confidence_intervals else None

    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', linewidth=1)

    # ax.set_ylabel(f'{model_name} $p_{{coop}}$ vs. Always Defect', fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Round', fontsize=24)

    ax.set_ylim(-0.05, 1.05)

    plt.legend(loc='upper left', fontsize=14)
    plt.tight_layout()
    out_fig_name = f'{model_name}_coop_comparison' if out_fig_name is None else out_fig_name
    file_path = out_dir / out_fig_name
    plt.savefig(file_path.with_suffix('.pdf'))
    plt.savefig(file_path.with_suffix('.png'))
    return fig, ax
