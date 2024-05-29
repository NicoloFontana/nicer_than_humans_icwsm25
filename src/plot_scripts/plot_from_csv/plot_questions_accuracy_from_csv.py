import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import lines as lin
from matplotlib import patches

from src.unused_functions import plot_errorbar

dfs = []
dfs.append(pd.read_csv("../../../csv_files_for_plots/accuracy-question/rules_checker_initial-final_results.csv"))
dfs.append(pd.read_csv("../../../csv_files_for_plots/accuracy-question/time_checker_initial-final_results.csv"))
dfs.append(pd.read_csv("../../../csv_files_for_plots/accuracy-question/state_checker_initial-final_results.csv"))

labels = []
initial_means = []
initial_yerrs = []
final_means = []
final_yerrs = []

for df in dfs:
    for index, row in df.iterrows():
        labels.append(row['label'])
        initial_means.append(row['initial_accuracy'])
        initial_yerrs.append((row['initial_ci_ub'] - row['initial_ci_lb']) / 2)
        final_means.append(row['final_accuracy'])
        final_yerrs.append((row['final_ci_ub'] - row['final_ci_lb']) / 2)

plt_fig = plot_errorbar(initial_means, "blue", labels, initial_yerrs, [0.0, 0.5, 1.0], fmt='^', alpha=0.3)
plt_fig = plot_errorbar(final_means, "red", labels, final_yerrs, [0.0, 0.5, 1.0], plt_figure=plt_fig, fmt='o')


implicit = lin.Line2D([], [], color='blue', marker='^', label='First prompt', alpha=0.3)
explicit = lin.Line2D([], [], color='red', marker='o', label='Last prompt')

plt.figure(plt_fig)
plt.axvline(x=4.5, color='black', linestyle='--', alpha=0.3)
plt.axvline(x=7.5, color='black', linestyle='--', alpha=0.3)
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
plt.title("Accuracy of questions")
plt.ylabel("Accuracy")
plt.xlabel("Questions (rules - time - state)")
plt.legend(handles=[implicit, explicit])
plt.tight_layout()
plt.show()
