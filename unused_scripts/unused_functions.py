import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.behavioral_analysis.behavioral_dimension import BehavioralDimension
from src.games.two_players_pd_utils import action_0_, action_1_, player_2_, player_1_, extract_histories_from_files
from src.llm_utils import OVERALL
from src.utils import OUT_BASE_PATH, compute_cumulative_estimators_of_ts, compute_average_vector, convert_matrix_to_percentage


# response = client.chat.completions.with_raw_response.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         # {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#         {"role": "user", "content": "test"}
#     ],
#     temperature=TEMPERATURE,
#     max_tokens=MAX_NEW_TOKENS
# )
# print(response.headers)
# completion = response.parse()
#
# print(completion.choices[0].message.content)


def plot_confusion_matrix_for_question(answers_dir_path, label, in_file_name=None, title=None, infix=None):
    if in_file_name is None:
        if infix is None:
            in_file_name = answers_dir_path.name + "_complete_answers.json"
        else:
            in_file_name = answers_dir_path.name + f"_complete_answers_{infix}.json"
    in_file_path = answers_dir_path / in_file_name
    with open(in_file_path, "r") as in_file:
        out_path = answers_dir_path / "confusion_matrices"
        os.makedirs(out_path, exist_ok=True)
        json_data = in_file.read()
        dict_data = json.loads(json_data)
        answers = dict_data[label]

        # Get the correct labels
        y_labels = []
        other_str = "Other"
        for quest in answers.values():
            correct_ans = quest["answer"]["correct_answer"]
            if correct_ans not in y_labels:
                y_labels.append(correct_ans)

        # Get the LLM's answers
        actual = []
        predicted = []
        other = False
        for quest in answers.values():
            actual.append(quest["answer"]["correct_answer"])
            if quest["answer"]["llm_answer"] not in y_labels:
                predicted.append(other_str)  # Group unexpected answers as "Other"
                other = True
            else:
                predicted.append(quest["answer"]["llm_answer"])

        # Get the LLM labels
        x_labels = y_labels.copy()
        if other:
            x_labels.append(other_str)

        # Compute the confusion matrix
        confusion_matrix = np.zeros((len(y_labels), len(x_labels)), dtype=int)
        for correct_answer, llm_answer in zip(actual, predicted):
            correct_idx = y_labels.index(correct_answer)
            llm_idx = x_labels.index(llm_answer)
            confusion_matrix[correct_idx, llm_idx] += 1

        # Create the plot and save it
        if title is None:
            title = label
        ax = sns.heatmap(confusion_matrix, annot=True, xticklabels=x_labels, yticklabels=y_labels, cmap="Blues", fmt='d')
        for text in ax.texts:
            x = float(text.get_text())
            text.set_text(compact_format(x))
        plt.title(title, loc="center", wrap=True)
        plt.xlabel('LLM answer')
        plt.ylabel('Correct answer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if infix is None:
            plt.savefig(out_path / f"{label}.svg")
            plt.savefig(out_path / f"{label}.png")
        else:
            plt.savefig(out_path / f"{label}_{infix}.svg")
            plt.savefig(out_path / f"{label}_{infix}.png")
        plt.show()
        ax = sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot=True, xticklabels=x_labels, yticklabels=y_labels, cmap="Blues", fmt='.2%')
        for text in ax.texts:
            x = float(text.get_text().replace('%', ''))
            text.set_text(compact_format(x, is_percentage=True))
        plt.title(title, loc="center", wrap=True)
        plt.xlabel('LLM answer')
        plt.ylabel('Correct answer')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if infix is None:
            plt.savefig(out_path / f"{label}_percentage.svg")
            plt.savefig(out_path / f"{label}_percentage.png")
        else:
            plt.savefig(out_path / f"{label}_percentage_{infix}.svg")
            plt.savefig(out_path / f"{label}_percentage_{infix}.png")
        plt.show()


def compact_format(x, is_percentage=False):
    if x == int(x):
        if x == 0:
            new_text = '-'
        else:
            new_text = '{:.0f}%'.format(x) if is_percentage else '{:.0f}'.format(x)
    else:
        new_text = '{:.2f}%'.format(x) if is_percentage else '{:.2f}'.format(x)
    return new_text


def extract_labels_and_questions_from_checker_answers(answers_dir_path, in_file_name=None):
    if in_file_name is None:
        in_file_name = answers_dir_path.name + "_complete_answers.json"
    in_file_path = answers_dir_path / in_file_name
    with open(in_file_path, "r") as in_file:
        json_data = in_file.read()
        dict_data = json.loads(json_data)
        labels = list(dict_data.keys())
        questions = {}
        for label in dict_data.keys():
            questions[label] = dict_data[label][str(0)]["question"]
        return labels, questions


def plot_checkers_results(checkers_names: list, timestamp, n_iterations, infix=None):
    if checkers_names is None or len(checkers_names) == 0:
        return
    results_file_path = merge_checkers_results(checkers_names, timestamp, infix=infix)

    with open(results_file_path, "r") as results_file:
        results = json.load(results_file)

    entries = [label for label in results.keys()]
    means = [result["sample_mean"] for result in results.values()]
    variances = [result["sample_variance"] for result in results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))

    first_cmap = plt.get_cmap('Dark2')

    checker_color_map = {checker: first_cmap(i / len(checkers_names)) for i, checker in enumerate(checkers_names)}
    entry_color_map = {}
    for label in results.keys():
        if label in checkers_names:
            entry_color_map[label] = 'red'
        else:
            entry_color_map[label] = checker_color_map[results[label]['checker']]
    for entry, mean, variance in zip(entries, means, variances):
        ax.plot([entry, entry], [mean - variance, mean + variance], '_:k', markersize=10, label='Variance')

    for entry in entries:
        plt.scatter(entry, means[entries.index(entry)], color=entry_color_map[entry], label=entry, s=100)

    for checker in checkers_names:
        checker_idx = entries.index(checker)
        plt.axvline(x=checker_idx - 0.5, linestyle='--', color='black', lw=0.5)

    plt.axhline(y=1.0, color='red', linestyle='-.', lw=0.25)
    plt.axhline(y=0.75, color='red', linestyle='-.', lw=0.5)
    plt.axhline(y=0.5, color='red', linestyle='-.', lw=0.75)
    plt.axhline(y=0.25, color='red', linestyle='-.', lw=0.5)
    plt.axhline(y=0.0, color='red', linestyle='-.', lw=0.25)

    ax.set_xlabel('Questions')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'LLM checks - {timestamp} - {n_iterations} iterations')

    labels = []
    for label in results.keys():
        labels.append(label)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels(labels)
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(entry_color_map[tick_label.get_text()])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    out_path = OUT_BASE_PATH / str(timestamp)
    if infix is None:
        plt.savefig(out_path / f'{OVERALL}.svg')
        plt.savefig(out_path / f'{OVERALL}.png')
    else:
        plt.savefig(out_path / f'{OVERALL}_{infix}.svg')
        plt.savefig(out_path / f'{OVERALL}_{infix}.png')
    plt.show()


def merge_checkers_results(checkers_names, timestamp, infix=None):
    python_objects = {}
    out_path = OUT_BASE_PATH / str(timestamp)

    for checker in checkers_names:
        if infix is None:
            in_file_path = out_path / checkers_names[checkers_names.index(checker)] / f"{checkers_names[checkers_names.index(checker)]}.json"
        else:
            in_file_path = out_path / checkers_names[checkers_names.index(checker)] / f"{checkers_names[checkers_names.index(checker)]}_{infix}.json"
        with open(in_file_path, "r") as f:
            python_object = json.load(f)
            for key in python_object.keys():
                python_objects[key] = python_object[key]

    if infix is None:
        out_file_path = out_path / f"{OVERALL}.json"
    else:
        out_file_path = out_path / f"{OVERALL}_{infix}.json"
    with open(out_file_path, "w") as out_file:
        json_out = json.dumps(python_objects, indent=4)
        out_file.write(json_out)
        return out_file_path


def plot_behavioral_profile(profile, title=None, out_file_path=None, show=False, color=None, plt_figure=None, label=None):
    # file_path = OUT_BASE_PATH / f"{extraction_timestamp}" / "behavioral_profiles" / f"behavioral_profile_{main_name}-{opponent_name}.json"
    # profile = BehavioralProfile(main_name, opponent_name)
    # profile.load_from_file(file_path)

    color = color if color is not None else 'blue'

    plt.figure(plt_figure)

    feature_names = list(profile.dimensions.keys())
    means = [profile.dimensions[feature_name].mean for feature_name in feature_names]
    std_devs = [profile.dimensions[feature_name].std_dev for feature_name in feature_names]
    plt.errorbar(feature_names, means, yerr=std_devs, fmt='o', color=color, label=label)

    plt.axhline(y=0, color='black', linestyle='--', alpha=0.1)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.1)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.1)

    # plt.title(f"{main_name} vs {opponent_name} - {n_games} games") if n_games is not None else plt.title(" ")#f"{main_name} vs {opponent_name}")
    plt.title(title) if title is not None else plt.title(" ")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # out_path = OUT_BASE_PATH / f"{extraction_timestamp}" / "behavioral_profiles" / "plots"
    if out_file_path is not None:
        out_parents = Path(out_file_path.parent)
        out_parents.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_file_path.with_suffix('.png'))
        plt.savefig(out_file_path.with_suffix('.svg'))
    plt.show() if show else None
    return plt.gcf().number


def plot_errorbar(values, values_color, values_label, yerr=None, axhlines=None, plt_figure=None, alpha=1.0, fmt='o'):
    plt.figure(plt_figure)

    plt.errorbar([i for i in range(len(values))], values, yerr=yerr, fmt=fmt, color=values_color, label=values_label, alpha=alpha)

    if axhlines is not None:
        for axhline in axhlines:
            plt.axhline(y=axhline, color='black', linestyle='--', lw=0.5, alpha=0.3)

    plt.title(" ")
    plt.xlabel(" ")
    plt.ylabel(" ")
    return plt.gcf().number


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


def plot_ts_(ts, ts_color, ts_label, lw=1.0, linestyle='-', marker=',', axhlines=None, fig=None, ax=None):
    n_iterations = len(ts)
    plt.figure(fig) if fig is not None else None
    plt.sca(ax) if ax is not None else None
    plt.plot([i for i in range(n_iterations)], ts, linestyle=linestyle, marker=marker, color=ts_color, label=ts_label, lw=lw)

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
    plt.figure(fig) if fig is not None else None
    plt.sca(ax) if ax is not None else None
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


#
# def plot_ts(ts, ts_label, ts_color, out_file_path=None, show=False, title=None, xlabel=None, ylabel=None, loc="best", axhlines=None, lw=1.0, mean_color=None, fill=False,
#             plt_figure=None, std_dev_ts=None):
#     """
#     Plots the time series `ts` with the specified parameters.
#     :param ts: time series to be plotted
#     :param ts_label: label to identify the time series
#     :param ts_color: color of the time series plot
#     :param out_file_path: where to save the plot
#     :param show: whether to show the plot or not
#     :param title: title of the plot
#     :param xlabel: label for the x-axis
#     :param ylabel: label for the y-axis
#     :param loc: location of the legend
#     :param axhlines: horizontal lines to be highlighted in the plot
#     :param lw: line width of the plot (used as baseline for opponent's and mean's widths)
#     :param mean_color: color of the mean line (used to check whether to plot the mean line or not)
#     :param fill: whether to fill the area between the ts or its mean and the standard deviation or not
#     :param plt_figure: number of the figure to eventually plot on
#     :return: number of the figure used
#     """
#     n_iterations = len(ts)
#     sample_means, sample_variances, sample_std_devs = compute_cumulative_estimators_of_ts(ts)
#     plt.figure(plt_figure)
#     plt.plot([i for i in range(n_iterations)], ts, linestyle='-', marker=',', color=ts_color, label=ts_label, lw=lw)
#     if mean_color is not None:
#         plt.plot([i for i in range(n_iterations)], sample_means, linestyle='-', marker=',', color=mean_color, lw=lw * 0.5, alpha=0.5, label=f"{ts_label} mean")
#
#     if axhlines is not None:
#         for axhline in axhlines:
#             plt.axhline(y=axhline, color='red', linestyle='--', lw=0.15)
#
#     if std_dev_ts is not None:
#         ub = [ts[i] + std_dev_ts[i] for i in range(n_iterations)]
#         lb = [ts[i] - std_dev_ts[i] for i in range(n_iterations)]
#         plt.fill_between([i for i in range(n_iterations)], ub, lb, color=ts_color, alpha=lw * 0.2)
#
#     if fill:
#         np_sample_std_devs = np.array(sample_std_devs)
#         if mean_color is not None:
#             np_sample_means = np.array(sample_means)
#             plt.fill_between([i for i in range(n_iterations)], np_sample_means + np_sample_std_devs, np_sample_means - np_sample_std_devs, color=mean_color, alpha=lw * 0.2)
#         else:
#             np_ts = np.array(ts)
#             plt.fill_between([i for i in range(n_iterations)], np_ts + np_sample_std_devs, np_ts - np_sample_std_devs, color=ts_color, alpha=lw * 0.2)
#
#     plt.title(title) if title is not None else None
#     plt.xlabel(xlabel) if xlabel is not None else None
#     plt.ylabel(ylabel) if ylabel is not None else None
#     plt.legend(loc=loc) if loc is not None else None
#     plt.tight_layout()
#     if out_file_path is not None:
#         plt.savefig(out_file_path.with_suffix('.svg'))
#         plt.savefig(out_file_path.with_suffix('.png'))
#     plt.show() if show else None
#     return plt.gcf().number


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


def convert_time_string_to_seconds(time_str):
    total_seconds = None
    # Extract hours, minutes, and seconds using regular expressions
    match_hour = re.match(r'(\d+)h(\d+)m(\d+\.\d+)s', time_str)
    match_minute = re.match(r'(\d+)m(\d+\.\d+)s', time_str)
    match_second = re.match(r'(\d+\.\d+)s', time_str)
    if match_hour:
        hours = int(match_hour.group(1))
        minutes = int(match_hour.group(2))
        seconds = float(match_hour.group(3))

        # Convert all to seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds

    elif match_minute:
        minutes = int(match_minute.group(1))
        seconds = float(match_minute.group(2))

        # Convert all to seconds
        total_seconds = minutes * 60 + seconds
    elif match_second:
        total_seconds = float(match_second.group(1))

    if total_seconds is not None:
        return total_seconds
    else:
        warnings.warn(f"Invalid time format: {time_str}. Returning 60 seconds.")
        return 60


class Cooperativeness(BehavioralDimension):
    def __init__(self):
        super().__init__("cooperativeness")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        cooperation = sum(main_history)
        cooperativeness = cooperation / n if n > 0 else 0
        self.values.append(cooperativeness)
        self.update_aggregates()
        return cooperativeness


class Naivety(BehavioralDimension):
    def __init__(self):
        super().__init__("naivety")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        uncalled_cooperation = 1 if main_history[0] == 1 else 0  # first cooperation was uncalled
        occasions = 1
        for i in range(n - 1):
            if opponent_history[i] == 0:  # opponent's defection
                occasions += 1
                uncalled_cooperation += 1 if main_history[i + 1] == 1 else 0  # main's uncalled cooperation
        naivety = uncalled_cooperation / occasions
        self.values.append(naivety)
        self.update_aggregates()
        return naivety


class Consistency(BehavioralDimension):
    def __init__(self):
        super().__init__("consistency")
        self.values = []
        self.mean = None
        self.variance = None
        self.std_dev = None

    def compute_dimension(self, main_history: list, opponent_history: list) -> float:
        n = len(main_history)
        changes = 0
        for i in range(n - 1):
            changes += 1 if main_history[i] != main_history[i + 1] else 0
        consistency = 1 - (changes / (n - 1)) if n > 1 else 1
        self.values.append(consistency)
        self.update_aggregates()
        return consistency


#
# class Forgiveness1(BehavioralDimension):
#     # 1-\frac{\sum(\frac{\mathsf{\#waited\_rounds}}{\mathsf{\#remaining\_rounds}})}{\mathsf{\#opponent\_defections}}
#     def __init__(self):
#         super().__init__("forgiveness1")
#         self.values = []
#         self.mean = None
#         self.variance = None
#         self.std_dev = None
#
#     def compute_feature(self, main_history: list, opponent_history: list) -> float:
#         n = len(main_history)
#         total_unforgiveness = 0
#         opponent_defection = 0  # for each opponent's defection, there is a chance to forgive
#         for i in range(n - 1):
#             if opponent_history[i] == 0:  # opponent's defection
#                 opponent_defection += 1
#                 start = i + 1
#                 forgiving_round = -1
#                 for j in range(start, n):
#                     if main_history[j] == 1:  # main cooperates after opponent's defection
#                         forgiving_round = j
#                         break
#                 if forgiving_round == -1:
#                     forgiving_round = n
#                 total_unforgiveness += (forgiving_round - start) / (n - start) if n > start else 0  # ratio between waited rounds to cooperate and remaining rounds
#                 # 0 if immediate cooperation, 1 if no cooperation
#         relative_unforgiveness = total_unforgiveness / opponent_defection if opponent_defection > 0 else 0  # ratio between total unforgiveness and occasions to forgive
#         # 0 if always forgives immediately, 1 if never forgives
#         forgiveness = 1 - relative_unforgiveness
#         self.values.append(forgiveness)
#         self.update_aggregates()
#         return forgiveness
#
#
# class Forgiveness2(BehavioralDimension):
#     # 1-\frac{\sum{\mathsf{\#waited\_rounds}}}{\sum{\mathsf{\#remaining\_rounds}}}
#     def __init__(self):
#         super().__init__("forgiveness2")
#         self.values = []
#         self.mean = None
#         self.variance = None
#         self.std_dev = None
#
#     def compute_feature(self, main_history: list, opponent_history: list) -> float:
#         n = len(main_history)
#         waited = 0  # how many rounds the main player waited to forgive in total
#         remaining = 0  # how many rounds the main player had to forgive in total
#         for i in range(n - 1):
#             if opponent_history[i] == 0:  # opponent's defection
#                 start = i + 1
#                 forgiving_round = -1
#                 for j in range(start, n):
#                     if main_history[j] == 1:  # main cooperates after opponent's defection
#                         forgiving_round = j
#                         break
#                 if forgiving_round == -1:
#                     forgiving_round = n
#                 waited += forgiving_round - start
#                 remaining += n - start
#         unforgiveness = waited / remaining if remaining > 0 else 0  # ratio between waited rounds to forgive and remaining rounds
#         # 0 if always forgives immediately, 1 if never forgives
#         forgiveness = 1 - unforgiveness
#         self.values.append(forgiveness)
#         self.update_aggregates()
#         return forgiveness
#
#
# class Forgiveness3(BehavioralDimension):
#     def __init__(self):
#         super().__init__("forgiveness3")
#         self.values = []
#         self.mean = None
#         self.variance = None
#         self.std_dev = None
#
#     def compute_feature(self, main_history: list, opponent_history: list) -> float:
#         n = len(main_history)
#         opponent_defection = 0  # for each opponent's defection, there is a chance to forgive
#         unforgiven = 0
#         for i in range(n):
#             if main_history[i] == 1:  # main cooperates
#                 unforgiven = max(0, unforgiven - 1)
#             if opponent_history[i] == 0 and i < n - 1:  # opponent's defection
#                 opponent_defection += 1
#                 unforgiven += 1
#         unforgiveness = unforgiven / opponent_defection if opponent_defection > 0 else 0  # ratio between total unforgiveness and occasions to forgive
#         # 0 if always forgives immediately, 1 if never forgives
#         forgiveness = 1 - unforgiveness
#         self.values.append(forgiveness)
#         self.update_aggregates()
#         return forgiveness


def old_to_new_dimension(old_dimension):
    if old_dimension == "niceness":
        return "nice"
    if old_dimension == "forgiveness":
        return "forgiving"
    if old_dimension == "provocability":
        return "retaliatory"
    if old_dimension == "cooperativeness":
        return "cooperative"
    if old_dimension == "troublemaking":
        return "troublemaking"
    if old_dimension == "emulation":
        return "emulative"
