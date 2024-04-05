import argparse
import os

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


def main():
    args = parse_args()

    with open(args.results_yaml, "r") as f:
        results = yaml.safe_load(f)

        correct_preds = 0
        total_prompts = 0

        for result in results:
            total_prompts += 1
            if len(result["programs"]) > 0:
                predictions = [
                    program_res["results"]["prediction"]
                    for program_res in result["programs"]
                    if "prediction" in program_res["results"]
                ]
                gt = result["answers"]["answer"]

                for prediction in predictions:
                    pred_str = str(prediction)
                    print(f"Prompt: {result['prompt']}")
                    print(f"Prediction: {pred_str}")
                    print(f"Ground truth: {gt}")
                    if pred_str is not None:

                        if pred_str in gt or gt in pred_str:
                            print("Correct")
                            correct_preds += 1
                            break
                        else:
                            print("Incorrect")

    print(f"Correct predictions: {correct_preds}/{total_prompts}")
    print(f"Correct predictions %: {(correct_preds/total_prompts) * 100}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pudb.post_mortem()
