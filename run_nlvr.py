import argparse
import os
from typing import Optional, Tuple, List, Any

import yaml
from PIL import Image
from tqdm import tqdm

from modules import VQA, Eval, Result, ExecutionError
from visprog import ProgramRunner


def do_nlvr(program_runner: ProgramRunner, program: str, images_dir: str,
            left_image_name: str, right_image_name: str) -> Tuple[Optional[bool], List[Any], Optional[str]]:
    try:
        left_image_path = os.path.join(images_dir, left_image_name)
        right_image_path = os.path.join(images_dir, right_image_name)
        left_image = Image.open(left_image_path).convert('RGB')
        right_image = Image.open(right_image_path).convert('RGB')
    except OSError as e:
        return None, [], str(e)
    initial_state = {
        'LEFT': left_image,
        'RIGHT': right_image,
    }
    try:
        steps, result = program_runner.execute_program(program, initial_state)
    except ExecutionError as e:
        return None, [d.get('output', None) for d in e.previous_step_details], e.error
    if not isinstance(result.output, dict):
        return None, [], f'Expected output to be a dictionary, got {type(result.output)} with value {result.output}'
    prediction = result.output.get('var', None)
    step_details = [d.get('output', None) for d in result.step_details[:-1]]
    return prediction, step_details, None


def main():
    parser = argparse.ArgumentParser(
        description='Run all programs in a NLVR yaml file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        default='cpu',
    )
    parser.add_argument(
        'images_dir',
        type=str,
    )
    parser.add_argument(
        'input_file',
        type=str,
    )
    parser.add_argument(
        'output_file',
        type=str,
    )

    args = parser.parse_args()

    vqa = VQA(device=args.device)
    eval_ = Eval()
    result = Result()
    modules = [vqa, eval_, result]
    program_runner = ProgramRunner(modules)

    with open(args.input_file, 'r') as f:
        statement_details = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    for statement_detail in tqdm(statement_details, desc='running programs', total=len(statement_details)):
        programs = statement_detail['programs']
        pairs = statement_detail['pairs']
        for i in range(len(programs)):
            if isinstance(programs[i], str):
                programs[i] = dict(program=programs[i])
            if 'results' not in programs[i]:
                programs[i]['results'] = {}
            for pair_object in pairs:
                if pair_object['id'] in programs[i]['results']:
                    continue
                prediction, step_details, error = do_nlvr(program_runner, programs[i]['program'], args.images_dir,
                                                          pair_object['left_image'], pair_object['right_image'])
                programs[i]['results'][pair_object['id']] = dict(prediction=prediction,
                                                                 steps=step_details,
                                                                 error=error)
                with open(args.output_file, 'w') as f:
                    yaml.dump(statement_details, f, default_style='|', sort_keys=False)


if __name__ == '__main__':
    main()