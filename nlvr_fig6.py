import argparse
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from evaluation.common import aggregate_without_voting, aggregate_with_voting
from evaluation.nlvr import compute_stats


def process_results(results_file: str) -> Dict[str, Any]:
    stats = compute_stats(results_file)
    without_voting = aggregate_without_voting(stats)
    with_voting = aggregate_with_voting(stats)
    return dict(
        without_voting=without_voting,
        with_voting=with_voting,
    )


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


def main():
    parser = argparse.ArgumentParser(
        description='Generate figure 6 bar plot for given the NLVR evaluation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'in_context_2_results_file',
        type=str,
    )
    parser.add_argument(
        'in_context_4_results_file',
        type=str,
    )
    parser.add_argument(
        'in_context_8_results_file',
        type=str,
    )
    parser.add_argument(
        'in_context_12_results_file',
        type=str,
    )
    parser.add_argument(
        'figure_file',
        type=str,
    )

    args = parser.parse_args()

    results = {
        '2': process_results(args.in_context_2_results_file),
        '4': process_results(args.in_context_4_results_file),
        '8': process_results(args.in_context_8_results_file),
        '12': process_results(args.in_context_12_results_file),
    }

    fig = build_figure(results)
    fig.savefig(args.figure_file)


if __name__ == '__main__':
    main()
