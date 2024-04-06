import argparse
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Hashable, List

import numpy as np
import pudb
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate results for GQA programs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "results_yaml",
        type=str,
    )

    parser.add_argument(
        "output_file",
        type=str,
    )

    return parser.parse_args()


def compute_stats(results_file: str) -> List[Dict[str, Any]]:
    with open(results_file, "r") as f:
        results = yaml.safe_load(f)

    stats: List[Dict[str, Any]] = []
    for prompt in results:
        try:
            execution_results = [
                program_object["results"] for program_object in prompt["programs"]
            ]
            label = prompt["answers"]["answer"]
            outcome_counts = defaultdict(
                lambda: 0,
                Counter(
                    (
                        result["prediction"]
                        if isinstance(result["prediction"], Hashable)
                        else None
                    )  # for unhashable results like dictionaries which are program errors
                    for result in execution_results
                    if "execution_error" in result
                    if "data_error" in result
                    if result["execution_error"] is None
                    and result["data_error"] is None
                ),
            )
            execution_errors = len(
                [
                    result["execution_error"]
                    for result in execution_results
                    if "execution_error" in result
                    if result["execution_error"] is not None
                ]
            )
            data_errors = len(
                [
                    result["data_error"]
                    for result in execution_results
                    if "data_error" in result
                    if result["data_error"] is not None
                ]
            )
            assert sum(outcome_counts.values()) + execution_errors + data_errors == len(
                execution_results
            ), f'Inconsistent results for prompt {prompt["id"]}, in {results_file}'
            stats.append(
                dict(
                    label=label,
                    outcome_counts=outcome_counts,
                    execution_errors=execution_errors,
                    data_errors=data_errors,
                    n_tries=len(execution_results),
                )
            )
        except:
            print(
                f'Error in prompt {prompt["id"]}, in {results_file}. results: {execution_results}'
            )
            raise
    return stats


def compute_one_run_accuracy(results_file: str, seed: int) -> float:
    with open(results_file, "r") as f:
        results = yaml.safe_load(f)

    rng = np.random.RandomState(seed)
    correct: List[bool] = []
    for prompt in results:
        try:
            execution_results = [
                program_object["results"]["prediction"]
                for program_object in prompt["programs"]
                if "prediction" in program_object["results"]
                if program_object["results"]["prediction"] is not None
            ]
            result = rng.choice(execution_results or [None])
            if result is None:
                continue
            label = prompt["answers"]["answer"]
            outcome = result
            correct.append(outcome == label)
        except:
            print(
                f'Error in prompt {prompt["id"]}, in {results_file}. results: {execution_results}'
            )
            raise

    accuracy = np.mean(correct)
    return accuracy


def main():
    args = parse_args()

    one_run_accuracy = compute_one_run_accuracy(args.results_yaml, 42)
    print(f"One run accuracy: {one_run_accuracy}")
    stats = compute_stats(args.results_yaml)
    print(f"Stats: {stats}")

    # with open(args.results_yaml, "r") as f:
    #     results = yaml.safe_load(f)
    #
    #     correct_preds = 0
    #     total_prompts = 0
    #
    #     for result in results:
    #         total_prompts += 1
    #         if len(result["programs"]) > 0:
    #             predictions = [
    #                 program_res["results"]["prediction"]
    #                 for program_res in result["programs"]
    #                 if "prediction" in program_res["results"]
    #             ]
    #             gt = result["answers"]["answer"]
    #
    #             # Calculate results without voting, just wether there is a correct prediction across the answers
    #             for prediction in predictions:
    #                 pred_str = str(prediction)
    #                 print(f"Prompt: {result['prompt']}")
    #                 print(f"Prediction: {pred_str}")
    #                 print(f"Ground truth: {gt}")
    #                 if pred_str is not None:
    #                     if pred_str in gt or gt in pred_str:
    #                         print("Correct")
    #                         correct_preds += 1
    #                         break
    #                     else:
    #                         print("Incorrect")
    #
    # print(f"Correct predictions: {correct_preds}/{total_prompts}")
    # print(f"Correct predictions %: {(correct_preds/total_prompts) * 100}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pudb.post_mortem()
