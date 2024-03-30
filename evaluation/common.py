from typing import List, Dict, Any

import numpy as np


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
