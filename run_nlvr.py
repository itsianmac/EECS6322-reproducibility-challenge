import argparse
import os
import threading
import traceback
from queue import Queue
from typing import Optional, Tuple, List, Any

import yaml
from PIL import Image
from tqdm import tqdm

from modules import VQA, Eval, Result, ExecutionError
from visprog import ProgramRunner


object_lock = threading.Lock()


def do_nlvr(program_runner: ProgramRunner, program: str,
            left_image: Image.Image, right_image: Image) -> Tuple[Optional[bool], List[Any], Optional[str]]:
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


def write_results(output_file: str, write_queue: Queue, statement_details: Any):
    try:
        while True:
            element = write_queue.get(block=True)
            if element is None:
                break
            i, j, pair_id, prediction, step_details, error = element
            with object_lock:
                statement_details[i]['programs'][j]['results'][pair_id] = dict(prediction=prediction,
                                                                               steps=step_details,
                                                                               error=error)
                with open(output_file, 'w') as f:
                    yaml.dump(statement_details, f, default_style='|', sort_keys=False)
    except Exception:
        traceback.print_exc()


def read_nlvr(statement_details: Any, images_dir: str, run_queue: Queue, write_queue: Queue):
    try:
        for i, statement_detail in tqdm(enumerate(statement_details), desc='running programs', total=len(statement_details)):
            programs = statement_detail['programs']
            pairs = statement_detail['pairs']
            for j in range(len(programs)):
                if isinstance(programs[j], str):
                    with object_lock:
                        programs[j] = dict(program=programs[j])
                if 'results' not in programs[j]:
                    with object_lock:
                        programs[j]['results'] = {}
                for pair_object in pairs:
                    if pair_object['id'] in programs[j]['results']:
                        continue
                    try:
                        left_image_path = os.path.join(images_dir, pair_object['left_image'])
                        right_image_path = os.path.join(images_dir, pair_object['right_image'])
                        left_image = Image.open(left_image_path).convert('RGB')
                        right_image = Image.open(right_image_path).convert('RGB')
                        if left_image.size[0] <= 3 or left_image.size[1] <= 3:
                            return None, [], f'Image {left_image_path} is too small'
                        if right_image.size[0] <= 3 or right_image.size[1] <= 3:
                            return None, [], f'Image {right_image_path} is too small'
                    except OSError as e:
                        write_queue.put((i, j,  pair_object['id'], None, [], str(e)))
                        continue
                    run_queue.put((i, j, pair_object['id'], programs[j]['program'], left_image, right_image))
    except Exception:
        traceback.print_exc()
    finally:
        run_queue.put(None)


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
    write_queue = Queue(maxsize=-1)
    run_queue = Queue(maxsize=64)
    write_results_thread = threading.Thread(target=write_results, args=(args.output_file, write_queue, statement_details))
    write_results_thread.start()
    read_thread = threading.Thread(target=read_nlvr, args=(statement_details, args.images_dir, run_queue, write_queue))
    read_thread.start()

    while True:
        run_element = run_queue.get(block=True)
        if run_element is None:
            break
        i, j, pair_id, program, left_image, right_image = run_element
        prediction, step_details, error = do_nlvr(program_runner, program, left_image, right_image)
        write_queue.put((i, j, pair_id, prediction, step_details, error))

    write_queue.put(None)
    write_results_thread.join()
    read_thread.join()


if __name__ == '__main__':
    main()
