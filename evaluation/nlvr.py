from collections import defaultdict, Counter
from typing import List, Dict, Any, Hashable

import numpy as np
import yaml


def compute_stats(results_file: str) -> List[Dict[str, Any]]:
    with open(results_file, 'r') as f:
        results = yaml.safe_load(f)

    stats: List[Dict[str, Any]] = []
    for prompt in results:
        for pair in prompt['pairs']:
            try:
                execution_results = [program_object['results'][pair['id']]
                                     for program_object in prompt['programs']
                                     if isinstance(program_object, dict)  # TODO: remove this
                                     and pair['id'] in program_object['results']]  # TODO: remove this
                if len(execution_results) < 5:    # TODO: this should not be necessary
                    continue
                label = pair['label']
                outcome_counts = defaultdict(lambda: 0, Counter(result['prediction']
                                                                if isinstance(result['prediction'], Hashable)
                                                                else None   # for unhashable results like dictionaries which are program errors
                                                                for result in execution_results
                                                                if result['execution_error'] is None
                                                                and result['data_error'] is None))
                execution_errors = len([result['execution_error'] for result in execution_results
                                        if result['execution_error'] is not None])
                data_errors = len([result['data_error'] for result in execution_results
                                   if result['data_error'] is not None])
                assert sum(outcome_counts.values()) + execution_errors + data_errors == len(execution_results), \
                    f'Inconsistent results for prompt {prompt["id"]}, pair {pair["id"]} in {results_file}'
                stats.append(dict(
                    label=label,
                    outcome_counts=outcome_counts,
                    execution_errors=execution_errors,
                    data_errors=data_errors,
                    n_tries=len(execution_results),
                ))
            except:
                print(f'Error in prompt {prompt["id"]}, pair {pair["id"]} in {results_file}. results: {execution_results}')
                raise
    return stats


def compute_one_run_accuracy(results_file: str, seed: int) -> float:
    with open(results_file, 'r') as f:
        results = yaml.safe_load(f)

    rng = np.random.RandomState(seed)
    correct: List[bool] = []
    for prompt in results:
        for pair in prompt['pairs']:
            try:
                execution_results = [program_object['results'][pair['id']]
                                     for program_object in prompt['programs']
                                     if isinstance(program_object, dict)            # TODO: remove this
                                     and pair['id'] in program_object['results']    # TODO: remove this
                                     and program_object['results'][pair['id']]['prediction'] is not None]
                result = rng.choice(execution_results or [None])
                if result is None:
                    continue
                label = pair['label']
                outcome = result['prediction']
                correct.append(outcome == label)
            except:
                print(f'Error in prompt {prompt["id"]}, pair {pair["id"]} in {results_file}. results: {execution_results}')
                raise

    accuracy = np.mean(correct)
    return accuracy
