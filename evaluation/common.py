from typing import List, Dict, Any

import numpy as np
from matplotlib import pyplot as plt


def aggregate_without_voting(stats: List[Dict[str, Any]]) -> Dict[str, float]:
    average_accuracies: List[float] = [stat['outcome_counts'][stat['label']] / (stat['n_tries'] - stat['data_errors'] - stat['outcome_counts'][None])
                                       for stat in stats if stat['n_tries'] - stat['data_errors'] - stat['outcome_counts'][None] > 0]
    print(f'w/o voting', len(average_accuracies))
    accuracy_mean = np.mean(average_accuracies)
    accuracy_std = np.std(average_accuracies)
    confidence_interval_95 = 1.96 * accuracy_std / np.sqrt(len(average_accuracies))
    return dict(
        accuracy_mean=accuracy_mean,
        confidence_interval_95=confidence_interval_95,
    )


def aggregate_with_voting(stats: List[Dict[str, Any]]) -> float:
    none_null_outcome_counts = [{k: v for k, v in stat['outcome_counts'].items() if k is not None}
                                for stat in stats]
    majority_correct = [max(outcome_counts.items(), key=lambda x: x[1])[0] == stat['label']
                        if len(outcome_counts) > 0 else False
                        for stat, outcome_counts in zip(stats, none_null_outcome_counts)
                        if stat['n_tries'] > stat['data_errors']]
    print('w/ voting', len(majority_correct))
    accuracy_mean = np.mean(majority_correct)
    return accuracy_mean


def build_figure(results: Dict[str, Dict[str, Any]]) -> plt.Figure:
    fig, ax = plt.subplots()

    r = np.arange(len(results))
    w = 0.4

    without_voting_means = np.array([result['without_voting']['accuracy_mean']
                                     for result in results.values()]) * 100
    without_voting_errors = np.array([result['without_voting']['confidence_interval_95']
                                      for result in results.values()]) * 100
    with_voting_means = np.array([result['with_voting'] for result in results.values()]) * 100

    ax.bar(r, without_voting_means, yerr=without_voting_errors, width=w, capsize=5, label='w/o voting')
    ax.bar(r + w, with_voting_means, width=w, label='w/ voting')

    y_min = min((without_voting_means - without_voting_errors).min(), with_voting_means.min()) - 2
    y_max = max((without_voting_means + without_voting_errors).max(), with_voting_means.max()) + 2
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(r + w / 2)
    ax.set_xticklabels(results.keys())
    ax.set_yticks(np.arange(int(np.ceil(y_min)), int(np.floor(y_max)) + 1, 2))

    ax.legend()

    return fig
