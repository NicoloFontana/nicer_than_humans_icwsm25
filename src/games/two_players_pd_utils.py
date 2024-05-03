import json
import warnings
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.games.game_history import GameHistory
from src.utils import OUT_BASE_PATH, compute_cumulative_estimators_of_ts, extract_infixes, compute_average_vector, convert_matrix_to_percentage, extract_infixes_

action_0_ = "Defect"
action_1_ = "Cooperate"

player_1_ = "A"
player_2_ = "B"


def to_nat_lang(action, string_of_string=True):
    if isinstance(action, set) and len(action) == 2 and 1 in action and 0 in action:
        return f'{{\"{action_1_}\", \"{action_0_}\"}}' if string_of_string else f'{{{action_1_}, {action_0_}}}'
    if action == 1:
        return f'\"{action_1_}\"' if string_of_string else action_1_
    if action == 0:
        return f'\"{action_0_}\"' if string_of_string else action_0_
    raise ValueError(f"Invalid action: {action}")


def from_nat_lang(action):
    if action == action_1_ or action == f'{action_1_}':
        return 1
    if action == action_0_ or action == f'{action_0_}':
        return 0
    warnings.warn(f"Invalid action: {action}. Returning '{action_0_}' as 0.")
    return 0


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


# PLOTTING FUNCTIONS

def plot_histogram(height, out_file_path=None, show=False, title=None, xlabel=None, ylabel=None, axhlines=None, xticklabels=None, plt_figure=None):
    """
    Plots a histogram with the specified parameters.
    :param height: values for each entry
    :param title: title of the plot
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param out_file_path: path where to save the plot
    :param show: whether to show the plot or not
    :param axhlines: horizontal lines to be highlighted in the plot
    :param xticklabels: labels for each entry
    :param plt_figure: number of the figure to eventually plot on
    :return: number of the figure used
    """
    plt.figure(plt_figure)
    if xticklabels is None:
        xticklabels = [str(i) for i in range(len(height))]
    plt.bar(xticklabels, height, color=['green', 'red'])

    if axhlines is not None:
        for axhline in axhlines:
            plt.axhline(y=axhline, color='red', linestyle='-.', lw=0.25)

    plt.title(title) if title is not None else None
    plt.xlabel(xlabel) if xlabel is not None else None
    plt.ylabel(ylabel) if ylabel is not None else None
    plt.tight_layout()
    if out_file_path is not None:
        plt.savefig(out_file_path.with_suffix('.svg'))
        plt.savefig(out_file_path.with_suffix('.png'))
    plt.show() if show else None
    return plt.gcf().number


def plot_occurrences_histogram(timestamp, history, infix=None, show=False, plt_figure=None):
    """
    Plots the occurrences of "Cooperate" and "Defect" in the specified history.
    :param timestamp: run to be analyzed
    :param history: history to be analyzed
    :param infix: number of the history to distinguish it from the others
    :param show: whether to show the plot or not
    :param plt_figure: number of the figure to eventually plot on
    :return: number of the figure used
    """
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots" / "occurrences"
    out_path.mkdir(parents=True, exist_ok=True)

    n_iterations = len(history)

    A_n_coop = history.count(1)
    A_coop_perc = A_n_coop / len(history)
    A_n_defect = history.count(0)
    A_defect_perc = A_n_defect / len(history)

    height = [A_n_coop, A_n_defect]
    title = f'# of "Cooperate" and "Defect" - {timestamp}'
    title += f' - G{infix}' if infix is not None else ''

    xlabel = 'Actions'
    ylabel = 'Occurrences'
    out_file_path = out_path / f"occurrences"
    out_file_path = out_file_path.parent / (out_file_path.name + f"_{infix}") if infix is not None else out_file_path

    axhlines = [0.0 * n_iterations, 0.25 * n_iterations, 0.5 * n_iterations, 0.75 * n_iterations, 1.0 * n_iterations]
    xticklabels = [f"Cooperate - {(A_coop_perc * 100):.2f}%", f"Defect - {(A_defect_perc * 100):.2f}%"]

    return plot_histogram(height, out_file_path, show=show, title=title, xlabel=xlabel, ylabel=ylabel, axhlines=axhlines, xticklabels=xticklabels, plt_figure=plt_figure)


def compute_transition_matrix(history, transition_matrix=None):
    """
    Computes the transition matrix for the specified history.
    For each action played, count the number of times a specific action is played next.
    :param history: history to be analyzed
    :param transition_matrix: eventual transition matrix to be updated
    :return: updated transition matrix
    """
    if transition_matrix is None:
        transition_matrix = np.zeros((2, 2), dtype=int)
    for i in range(len(history) - 1):
        transition_matrix[history[i], history[i + 1]] += 1
    return transition_matrix


def compute_second_order_transition_matrix(history, second_order_transition_matrix=None):
    """
    Computes the second-order transition matrix for the specified history.
    For each sequence of 2 actions played, count the number of times a specific action is played next.
    :param history: history to be analyzed
    :param second_order_transition_matrix: eventual second-order transition matrix to be updated
    :return: updated second-order transition matrix
    """
    if second_order_transition_matrix is None:
        second_order_transition_matrix = np.zeros((4, 2), dtype=int)
    for i in range(len(history) - 2):
        second_order_transition_matrix[2 * history[i] + history[i + 1], history[i + 2]] += 1
    return second_order_transition_matrix


def compute_reaction_matrix(main_history, opponent_history, reaction_matrix=None):
    """
    Computes the reaction matrix for the specified histories.
    For each action played by the opponent, count the number of times the main player played a specific action next.
    :param main_history: history of the main player to be analyzed
    :param opponent_history: history of the opponent to be considered
    :param reaction_matrix: eventual reaction matrix to be updated
    :return: updated reaction matrix
    """
    if reaction_matrix is None:
        reaction_matrix = np.zeros((2, 2), dtype=int)
    for i in range(len(main_history) - 1):
        reaction_matrix[opponent_history[i], main_history[i + 1]] += 1
    return reaction_matrix


def plot_matrix(matrix, out_file_path=None, show=False, title=None, xlabel=None, ylabel=None, vmin=0, vmax=None, xticklabels: Any = "auto", yticklabels: Any = "auto",
                plt_figure=None):
    """
    Plots the specified matrix.
    :param matrix: matrix to be plotted
    :param out_file_path: where to save the plot
    :param show: whether to show the plot or not
    :param title: title of the plot
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param vmin: minimum value for the color scale
    :param vmax: maximum value for the color scale
    :param xticklabels: labels for each x-axis entry
    :param yticklabels: labels for each y-axis entry
    :param plt_figure: number of the figure to eventually plot on
    :return: number of the figure used
    """
    plt.figure(plt_figure)
    sns.heatmap(matrix, annot=True, xticklabels=xticklabels, yticklabels=yticklabels, cmap="Blues", fmt='.2%', vmin=vmin, vmax=vmax)
    plt.title(title) if title is not None else None
    plt.xlabel(xlabel) if xlabel is not None else None
    plt.ylabel(ylabel) if ylabel is not None else None
    plt.tight_layout()
    if out_file_path is not None:
        plt.savefig(out_file_path.with_suffix('.svg'))
        plt.savefig(out_file_path.with_suffix('.png'))
    plt.show() if show else None
    return plt.gcf().number


def plot_transition_matrix(timestamp, transition_matrix, infix=None, show=False, is_reaction=False, is_second_order=False, plt_figure=None):
    """
    Plots the transition matrix for the specified history.
    :param timestamp: run to be analyzed
    :param transition_matrix: matrix to be plotted
    :param infix: number of the history to distinguish it from the others
    :param show: whether to show the plot or not
    :param is_reaction: whether the matrix is a reaction matrix or not
    :param is_second_order: whether the matrix is a second-order transition matrix or not
    :param plt_figure: number of the figure to eventually plot on
    :return: number of the figure used
    """
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    if is_second_order:
        out_path = run_dir_path / "plots" / "second_order_transition_matrices"
    elif is_reaction:
        out_path = run_dir_path / "plots" / "reaction_matrices"
    else:
        out_path = run_dir_path / "plots" / "transition_matrices"
    out_path.mkdir(parents=True, exist_ok=True)

    if is_second_order:
        title = f'Second-order transition matrix'
        out_file_path = out_path / f"second_order_transition_matrix"
    elif is_reaction:
        title = f'Reaction matrix'
        out_file_path = out_path / f"reaction_matrix"
    else:
        title = f'Transition matrix'
        out_file_path = out_path / f"transition_matrix"

    title += f" - {timestamp}"
    title += f" - G{infix}" if infix is not None else ""

    out_file_path = out_file_path.parent / (out_file_path.name + f"_{infix}") if infix is not None else out_file_path

    xlabel = 'Next action' if not is_reaction else 'Own current action'
    if is_second_order:
        ylabel = 'Last actions'
    elif is_reaction:
        ylabel = 'Opponent previous action'
    else:
        ylabel = 'Last action'

    vmax = sum([sum(row) for row in transition_matrix])
    xticklabels = [action_0_, action_1_]
    if is_second_order:
        yticklabels = [f"{action_0_}-{action_0_}", f"{action_0_}-{action_1_}", f"{action_1_}-{action_0_}", f"{action_1_}-{action_1_}"]
    else:
        yticklabels = [action_0_, action_1_]
    return plot_matrix(transition_matrix, out_file_path, show=show, title=title, xlabel=xlabel, ylabel=ylabel, xticklabels=xticklabels, yticklabels=yticklabels,
                       plt_figure=plt_figure, vmax=vmax)


def plot_ts_(ts, ts_color, ts_label, lw=1.0, axhlines=None, fig=None, ax=None):
    n_iterations = len(ts)
    plt.figure(fig)
    plt.sca(ax)
    plt.plot([i for i in range(n_iterations)], ts, linestyle='-', marker=',', color=ts_color, label=ts_label, lw=lw)

    if axhlines is not None:
        for axhline in axhlines:
            plt.axhline(y=axhline, color='black', linestyle='--', lw=0.1)
    plt.title(" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.tight_layout()
    return plt.gcf().number


def plot_fill(lower_bounds, upper_bounds, fill_color, fig=None, ax=None, alpha=0.3):
    n_iterations = len(upper_bounds)
    plt.figure(fig)
    plt.sca(ax)
    plt.fill_between([i for i in range(n_iterations)], upper_bounds, lower_bounds, color=fill_color, alpha=alpha)
    plt.title(" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    plt.tight_layout()
    return plt.gcf().number


def save_plot(plt_figure, out_file_path, show=False):
    plt.figure(plt_figure)
    if out_file_path is not None:
        plt.savefig(out_file_path.with_suffix('.svg'))
        plt.savefig(out_file_path.with_suffix('.png'))
    plt.show() if show else None


def plot_ts(ts, ts_label, ts_color, out_file_path=None, show=False, title=None, xlabel=None, ylabel=None, loc="best", axhlines=None, lw=1.0, mean_color=None, fill=False,
            plt_figure=None, std_dev_ts=None):
    """
    Plots the time series `ts` with the specified parameters.
    :param ts: time series to be plotted
    :param ts_label: label to identify the time series
    :param ts_color: color of the time series plot
    :param out_file_path: where to save the plot
    :param show: whether to show the plot or not
    :param title: title of the plot
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param loc: location of the legend
    :param axhlines: horizontal lines to be highlighted in the plot
    :param lw: line width of the plot (used as baseline for opponent's and mean's widths)
    :param mean_color: color of the mean line (used to check whether to plot the mean line or not)
    :param fill: whether to fill the area between the ts or its mean and the standard deviation or not
    :param plt_figure: number of the figure to eventually plot on
    :return: number of the figure used
    """
    n_iterations = len(ts)
    sample_means, sample_variances, sample_std_devs = compute_cumulative_estimators_of_ts(ts)
    plt.figure(plt_figure)
    plt.plot([i for i in range(n_iterations)], ts, linestyle='-', marker=',', color=ts_color, label=ts_label, lw=lw)
    if mean_color is not None:
        plt.plot([i for i in range(n_iterations)], sample_means, linestyle='-', marker=',', color=mean_color, lw=lw * 0.5, alpha=0.5, label=f"{ts_label} mean")

    if axhlines is not None:
        for axhline in axhlines:
            plt.axhline(y=axhline, color='red', linestyle='--', lw=0.15)

    if std_dev_ts is not None:
        ub = [ts[i] + std_dev_ts[i] for i in range(n_iterations)]
        lb = [ts[i] - std_dev_ts[i] for i in range(n_iterations)]
        plt.fill_between([i for i in range(n_iterations)], ub, lb, color=ts_color, alpha=lw * 0.2)

    if fill:
        np_sample_std_devs = np.array(sample_std_devs)
        if mean_color is not None:
            np_sample_means = np.array(sample_means)
            plt.fill_between([i for i in range(n_iterations)], np_sample_means + np_sample_std_devs, np_sample_means - np_sample_std_devs, color=mean_color, alpha=lw * 0.2)
        else:
            np_ts = np.array(ts)
            plt.fill_between([i for i in range(n_iterations)], np_ts + np_sample_std_devs, np_ts - np_sample_std_devs, color=ts_color, alpha=lw * 0.2)

    plt.title(title) if title is not None else None
    plt.xlabel(xlabel) if xlabel is not None else None
    plt.ylabel(ylabel) if ylabel is not None else None
    plt.legend(loc=loc) if loc is not None else None
    plt.tight_layout()
    if out_file_path is not None:
        plt.savefig(out_file_path.with_suffix('.svg'))
        plt.savefig(out_file_path.with_suffix('.png'))
    plt.show() if show else None
    return plt.gcf().number


def plot_average_history_with_comparison(timestamp, game_histories, main_player_name, show=True, opponent_name=None, opponent_label=None, fill=False, plt_figure=None):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots" / "average_history"
    out_path.mkdir(parents=True, exist_ok=True)

    n_games = len(game_histories)

    main_histories = [game_history.get_actions_by_player(main_player_name) for game_history in game_histories]
    main_average_history = compute_average_vector(main_histories)

    title = f'Average action - {timestamp} - {n_games} games'
    xlabel = 'Iteration'
    ylabel = 'Avg action'
    out_file_path = out_path / f"average_action"
    out_file_path = out_file_path.parent / (out_file_path.name + f"_fill") if fill else out_file_path
    loc = 'lower right'

    axhlines = [0.0, 0.5, 1.0]

    if opponent_name is None:
        mean_color = 'r'
    else:
        mean_color = 'b'
    main_figure = plot_ts(main_average_history, "LLM", 'b', out_file_path, mean_color=mean_color, show=show, title=title, xlabel=xlabel, ylabel=ylabel, loc=loc, fill=fill,
                          plt_figure=plt_figure, axhlines=axhlines)

    opponent_figure = None
    if opponent_name is not None:
        if opponent_label is None:
            opponent_label = opponent_name
        opponent_histories = []
        for game_history in game_histories:
            opponent_histories.append(game_history.get_actions_by_player(opponent_name))
        opponent_average_history = compute_average_vector(opponent_histories)

        plot_infix = f"_{opponent_label}" if opponent_label is not None else ""
        plot_infix += "_fill" if fill else ""
        out_file_path = out_path / f"average_action{plot_infix}"

        opponent_figure = plot_ts(opponent_average_history, opponent_label, 'r', out_file_path, lw=0.5, mean_color='r', show=show, fill=fill, plt_figure=main_figure)

    return main_figure, opponent_figure


def compute_similarity_between_single_histories(main_history, opponent_history):
    n_iterations = len(main_history)
    similarity = [sum([(1 - abs(main_history[j + 1] - opponent_history[j])) for j in range(i + 1)]) / (i + 1) for i in range(n_iterations - 1)]
    return similarity


def plot_similarity_between_histories(timestamp, game_histories, main_player_name, opponent_name, opponent_label=player_2_):
    run_dir_path = OUT_BASE_PATH / str(timestamp)
    out_path = run_dir_path / "plots" / "similarity"
    out_path.mkdir(parents=True, exist_ok=True)

    main_histories = []
    opponent_histories = []
    for game_history in game_histories:
        main_histories.append(game_history.get_actions_by_player(main_player_name))
        opponent_histories.append(game_history.get_actions_by_player(opponent_name))

    n_games = len(game_histories)

    # Compute how much (on average) the main player's actions are similar to the opponent's actions
    similarities = []
    for i in range(n_games):
        similarity = compute_similarity_between_single_histories(main_histories[i], opponent_histories[i])
        similarities.append(similarity)
    average_similarity = compute_average_vector(similarities)

    title = f'Similarity LLM-{opponent_label} - {timestamp} - {n_games} games'
    xlabel = "Iteration"
    ylabel = "Similarity"
    axhlines = [0.0, 0.5, 1.0]
    out_file_path = out_path / "similarity"

    plot_ts(average_similarity, f"LLM-{opponent_label}", 'b', out_file_path, mean_color='r', show=False, title=title, xlabel=xlabel, ylabel=ylabel, axhlines=axhlines, fill=True)

    # # Compute how much the average main player's actions are similar to the average opponent's actions
    # # TODO: may it be interesting??
    # main_average_history = compute_average_vector(main_histories)
    # opponent_average_history = compute_average_vector(opponent_histories)
    # similarity_average = compute_similarity_between_single_histories(main_average_history, opponent_average_history)
    #
    # title = f'Similarity avgLLM-avgRND - {timestamp} - {n_games} games'
    #
    # plot_ts(similarity_average, "avgLLM-avgRND", 'r', out_file_path, show=True, title=title, xlabel="Iteration", ylabel="Similarity", axhlines=axhlines, fill=True)


def plot_histories_analysis(timestamp, game_histories, show=False, main_player_name=player_1_, opponent_name=player_2_, opponent_label=player_2_, percentage_matrices=True):
    """
    Plots the occurrences histogram, the transition matrix, the second-order transition matrix and the average action played at each turn for the aggregated game histories of run `timestamp`.
    The histories are NOT concatenated, instead their single metrics are aggregated.

    Example:
    - history 1: [0, 1]
    - history 2: [1, 0]
    The number of transitions from 1 to 1 is 0 and remains 0 in the aggregated analysis.

    :param timestamp: timestamp of the run to be analyzed
    :param game_histories: list of game histories to be analyzed
    :param show: whether to show the plot or not
    :param main_player_name: name of the main player to be analyzed
    :param opponent_name: name of the opponent player to be considered
    :param opponent_label: label of the opponent player to be shown in the plots
    :param percentage_matrices: whether to convert the transition matrices to percentage matrices or not
    """
    aggregated_history = []
    for game_history in game_histories:
        main_actions = game_history.get_actions_by_player(main_player_name)
        aggregated_history.extend(main_actions)
    plot_occurrences_histogram(timestamp, aggregated_history, show=show)

    transition_matrix = None
    for game_history in game_histories:
        main_actions = game_history.get_actions_by_player(main_player_name)
        transition_matrix = compute_transition_matrix(main_actions, transition_matrix)
    if percentage_matrices:
        transition_matrix = convert_matrix_to_percentage(transition_matrix)
    plot_transition_matrix(timestamp, transition_matrix, show=show)

    second_order_transition_matrix = None
    for game_history in game_histories:
        main_actions = game_history.get_actions_by_player(main_player_name)
        second_order_transition_matrix = compute_second_order_transition_matrix(main_actions, second_order_transition_matrix)
    if percentage_matrices:
        second_order_transition_matrix = convert_matrix_to_percentage(second_order_transition_matrix)
    plot_transition_matrix(timestamp, second_order_transition_matrix, show=show, is_second_order=True)

    reaction_matrix = None
    for game_history in game_histories:
        main_actions = game_history.get_actions_by_player(main_player_name)
        opponent_actions = game_history.get_actions_by_player(opponent_name)
        reaction_matrix = compute_reaction_matrix(main_actions, opponent_actions, reaction_matrix)
    if percentage_matrices:
        reaction_matrix = convert_matrix_to_percentage(reaction_matrix)
    plot_transition_matrix(timestamp, reaction_matrix, show=show, is_reaction=True)

    plot_average_history_with_comparison(timestamp, game_histories, main_player_name, show=show, opponent_name=opponent_name, opponent_label=opponent_label, fill=True)

    plot_similarity_between_histories(timestamp, game_histories, main_player_name, opponent_name, opponent_label=opponent_label)


def extract_histories_from_files(extraction_timestamp, subdir=None, file_name=None, numerical_infixes=None, max_infix=None):
    game_histories = []
    if subdir is None:
        subdir = "game_histories"
    if file_name is None:
        file_name = "game_history"
    if numerical_infixes is None:
        numerical_infixes = extract_infixes(extraction_timestamp, file_name=file_name, subdir=subdir, max_infix=max_infix)
    for infix in numerical_infixes:
        game_history = GameHistory()
        file_path = OUT_BASE_PATH / str(extraction_timestamp) / subdir / f"{file_name}_{infix}.json"
        game_history.load_from_file(file_path)
        game_histories.append(game_history)
    return game_histories


def extract_histories_from_files_(dir_path, file_name=None, numerical_infixes=None, max_infix=None):
    game_histories = []
    if file_name is None:
        file_name = "game_history"
    if numerical_infixes is None:
        numerical_infixes = extract_infixes_(dir_path, file_name=file_name, max_infix=max_infix)
    for infix in numerical_infixes:
        game_history = GameHistory()
        file_path = dir_path / f"{file_name}_{infix}.json"
        game_history.load_from_file(file_path)
        game_histories.append(game_history)
    return game_histories


def merge_runs(runs):
    min_timestamp = str(min(runs.keys()))
    max_timestamp = str(max(runs.keys()))
    storing_timestamp = str(min_timestamp[-8:]) + "-" + str(max_timestamp[-8:])
    dir_path = OUT_BASE_PATH / str(storing_timestamp)
    dir_path.mkdir(parents=True, exist_ok=True)
    idx = 1
    for timestamp in runs.keys():
        infixes = runs[timestamp]
        game_histories = extract_histories_from_files(timestamp, infixes)
        for game_history in game_histories:
            game_history.save_to_file(storing_timestamp, infix=idx)
            idx += 1
