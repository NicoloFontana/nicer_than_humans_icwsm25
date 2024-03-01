import json

from matplotlib import pyplot as plt

from src.utils import CHECKS_OUT_BASE_PATH

OVERALL = "overall"


def plot_checkers_results(checkers_names, timestamp, n_iterations, infix=None):
    results_file_path = merge_checkers_results(checkers_names, timestamp, infix=infix)

    with open(results_file_path, "r") as results_file:
        results = json.load(results_file)

    entries = [result["label"] for result in results.values()]
    means = [result["sample_mean"] for result in results.values()]
    variances = [result["sample_variance"] for result in results.values()]

    fig, ax = plt.subplots(figsize=(12, 6))

    first_cmap = plt.get_cmap('Dark2')

    checker_color_map = {checker: first_cmap(i / len(checkers_names)) for i, checker in enumerate(checkers_names)}
    entry_color_map = {}
    for result in results.values():
        if result['label'] in checkers_names:
            entry_color_map[result['label']] = 'red'
        else:
            entry_color_map[result['label']] = checker_color_map[result['checker']]
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
    for result in results.values():
        labels.append(result['label'])
    ax.set_xticklabels(labels)
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(entry_color_map[tick_label.get_text()])

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if infix is None:
        plt.savefig(CHECKS_OUT_BASE_PATH / str(timestamp) / f'{OVERALL}.svg')
        plt.savefig(CHECKS_OUT_BASE_PATH / str(timestamp) / f'{OVERALL}.png')
    else:
        plt.savefig(CHECKS_OUT_BASE_PATH / str(timestamp) / f'{OVERALL}_{infix}.svg')
        plt.savefig(CHECKS_OUT_BASE_PATH / str(timestamp) / f'{OVERALL}_{infix}.png')
    plt.show()


def merge_checkers_results(checkers_names, timestamp, infix=None):
    python_objects = {}

    for checker in checkers_names:
        if infix is None:
            in_file_path = CHECKS_OUT_BASE_PATH / str(timestamp) / f"{checkers_names[checkers_names.index(checker)]}.json"
        else:
            in_file_path = CHECKS_OUT_BASE_PATH / str(timestamp) / f"{checkers_names[checkers_names.index(checker)]}_{infix}.json"
        with open(in_file_path, "r") as f:
            python_object = json.load(f)
            for key in python_object.keys():
                python_objects[key] = python_object[key]

    # Dump all the Python objects into a single JSON file.
    if infix is None:
        out_file_path = CHECKS_OUT_BASE_PATH / str(timestamp) / f"{OVERALL}.json"
    else:
        out_file_path = CHECKS_OUT_BASE_PATH / str(timestamp) / f"{OVERALL}_{infix}.json"
    with open(out_file_path, "w") as f:
        json.dump(python_objects, f, indent=4)
        return out_file_path
