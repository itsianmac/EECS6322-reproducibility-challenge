import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Any

import numpy as np
import yaml
import matplotlib.pyplot as plt


def compute_stats(results_file: str) -> List[Dict[str, Any]]:
    with open(results_file, 'r') as f:
        results = yaml.safe_load(f)

    stats: List[Dict[str, Any]] = []
    for prompt in results:
        for pair in prompt['pairs']:
            results = [program_object['results'][pair['id']] for program_object in prompt['programs']
                       if pair['id'] in program_object['results']]  # TODO: remove this line
            label = pair['label']
            outcome_counts = defaultdict(lambda: 0, Counter(result['prediction'] for result in results
                                                            if result['execution_error'] is None
                                                            and result['data_error'] is None))
            execution_errors = len([result['execution_error'] for result in results
                                    if result['execution_error'] is not None])
            data_errors = len([result['data_error'] for result in results
                               if result['data_error'] is not None])
            assert sum(outcome_counts.values()) + execution_errors + data_errors == len(results), \
                f'Inconsistent results for prompt {prompt["id"]}, pair {pair["id"]} in {results_file}'
            stats.append(dict(
                label=label,
                outcome_counts=outcome_counts,
                execution_errors=execution_errors,
                data_errors=data_errors,
                n_tries=len(results),
            ))
    return stats


def aggregate_without_voting(stats: List[Dict[str, Any]]) -> Dict[str, float]:
    average_accuracies = [stat['outcome_counts'][stat['label']] / (stat['n_tries'] - stat['data_errors'])
                          for stat in stats if stat['n_tries'] > stat['data_errors']]
    accuracy_mean = np.mean(average_accuracies)
    accuracy_std = np.std(average_accuracies)
    confidence_interval_95 = 1.96 * accuracy_std / np.sqrt(len(average_accuracies))
    return dict(
        accuracy_mean=accuracy_mean,
        confidence_interval_95=confidence_interval_95,
    )


def aggregate_with_voting(stats: List[Dict[str, Any]]) -> float:
    majority_correct = [max(stat['outcome_counts'].items(), key=lambda x: x[1])[0] == stat['label']
                         if len(stat['outcome_counts']) > 0 else False
                         for stat in stats if stat['n_tries'] > stat['data_errors']]
    accuracy_mean = np.mean(majority_correct)
    return accuracy_mean


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
