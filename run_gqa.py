import argparse
import os
import time
import traceback
from typing import Any, List, Optional, Tuple

import pudb
import yaml
from PIL import Image
from tqdm import tqdm

from modules import (VQA, Count, Crop, CropAbove, CropBelow, CropLeft,
                     CropRight, Eval, ExecutionError, Loc, Result)
from visprog import ProgramRunner


def do_gqa(
    program_runner: ProgramRunner, program: str, image: Image.Image
) -> Tuple[Optional[bool], List[Any], Optional[str]]:

    initial_state = {
        "IMAGE": image,
    }
    print("===== Visprog PROGRAM =====: ", program)

    try:
        steps, result = program_runner.execute_program(program, initial_state)
    except ExecutionError as e:
        return None, [d.get("output", None) for d in e.previous_step_details], e.error
    if not isinstance(result.output, dict):
        return (
            None,
            [],
            f"Expected output to be a dictionary, got {type(result.output)} with value {result.output}",
        )

    prediction = result.output.get("var", None)

    print("===== Visprog PREDICTION =====: ", prediction)

    step_details = [d.get("output", None) for d in result.step_details[:-1]]

    return prediction, step_details, None


def read_gqa(statement_details: Any, images_dir: str):
    try:
        for i, statement_detail in tqdm(
            enumerate(statement_details),
            desc="Reading GQA programs",
            total=len(statement_details),
        ):
            programs = statement_detail["programs"]
            statement_detail["image"] = os.path.join(
                images_dir, statement_detail["image"]
            )

            for j in range(len(programs)):
                if isinstance(programs[j], str):
                    programs[j] = dict(program=programs[j])
                if "results" in programs[j]:
                    continue
                if "results" not in programs[j]:
                    programs[j]["results"] = {}

    except:
        traceback.print_exc()
        time.sleep(1)
    finally:
        print("Done reading GQA")
        return statement_details


def write_gqa_results(
    output_file: str,
    statements: Any,
    prediction: Any,
    step_details: Any,
    error: Any,
    i: int,
    j: int,
):
    statements[i]["programs"][j]["results"] = dict(
        prediction=prediction,
        # steps=step_details,
        execution_error=error,
        data_error=None,
    )

    try:
        with open(output_file, "w") as f:
            yaml.dump(statements, f)
    except:
        traceback.print_exc()
        time.sleep(1)
    finally:
        print(f"Done writing GQA results for statement {i} program {j}")


def main():

    print("Running GQA programs")
    parser = argparse.ArgumentParser(
        description="Run all programs in a NLVR yaml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "images_dir",
        type=str,
    )
    parser.add_argument(
        "input_file",
        type=str,
    )
    parser.add_argument(
        "output_file",
        type=str,
    )

    args = parser.parse_args()

    # Define modules based on what is given in the in-context examples for GQA
    loc_module = Loc()
    crop = Crop()
    crop_right = CropRight()
    crop_left = CropLeft()
    crop_above = CropAbove()
    crop_below = CropBelow()
    count = Count()
    _eval = Eval()
    result = Result()
    vqa = VQA(device=args.device, cast_from_string=True)

    modules = [
        loc_module,
        crop,
        crop_right,
        crop_left,
        crop_above,
        crop_below,
        count,
        _eval,
        result,
        vqa,
    ]

    # Pass modules to the program runner
    program_runner = ProgramRunner(modules)

    # Open the json file containing the chat-gpt generated programs
    with open(args.input_file, "r") as f:
        statement_details = yaml.safe_load(f)

    # Make sure the output directory / file exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Read GQA
    statements = read_gqa(statement_details, args.images_dir)
    for i, statement in enumerate(statements):
        img_pth = statement["image"]
        img = Image.open(img_pth).convert("RGB")

        for j, program in enumerate(statement["programs"]):
            # Print question
            print(f"Question: {statement['prompt']['question']}")
            print(f"Ground truth answers: {statement['answers']}")
            print(f"id: {statement['id']}")

            # If there is a results key, we have already generated the program
            try:
                prediction, step_details, error = do_gqa(
                    program_runner, program["program"], img
                )

                write_gqa_results(
                    args.output_file, statements, prediction, step_details, error, i, j
                )
            except KeyboardInterrupt:
                raise
            except:
                continue


if __name__ == "__main__":
    try:
        main()
    except:
        pudb.post_mortem()
