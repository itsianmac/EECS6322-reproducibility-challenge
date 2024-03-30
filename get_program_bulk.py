import argparse
import importlib
import os
import re
import time
import random

import yaml
from selenium.common import TimeoutException

from gpt import GPTClient
from instructions import BulkPromptFactory

CODE_REGEX = re.compile(r'\S+\s*=\s*\S+\s*\([^)]*\)')


def main():
    parser = argparse.ArgumentParser(
        description='Ask ChatGPT to generate programs for each instruction in a bulk fashion',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-c', '--context',
        type=str,
        choices=['gqa', 'imgedit', 'nlvr'],
        default='gqa',
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        choices=['all', 'random'],
        default='random',
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=2,
    )
    parser.add_argument(
        '-np', '--num-prompts',
        type=int,
        default=8,
    )
    parser.add_argument(
        '-nt', '--num-tries',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-ud', '--gpt-user-dirs',
        type=str,
        nargs='+',
        default=['./user-data'],
    )
    parser.add_argument(
        '--split-by',
        type=str,
        default='FINAL_ANSWER',
    )
    parser.add_argument(
        '--gpt-auth',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        'prompts_file',
        type=str,
    )
    parser.add_argument(
        'output_file',
        type=str,
    )
    args = parser.parse_args()

    get_prompt_factory = importlib.import_module(f'instructions.{args.context}').get_bulk_prompt_factory
    prompt_factory: BulkPromptFactory = get_prompt_factory(method=args.method, num_prompts=args.num_prompts)

    print('Initializing GPT client')
    current_gpt_index = 0
    gpt = GPTClient(args.gpt_user_dirs[current_gpt_index], wait_for_login=args.gpt_auth)
    time.sleep(1)
    print('GPT client initialized')

    with open(args.prompts_file, 'r') as f:
        prompts = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    seed = args.seed
    rng = random.Random(seed)
    try:
        while True:
            indices = list(range(len(prompts)))
            indices = [i for i in indices if len(prompts[i].get('programs', [])) < args.num_tries]
            print(f'Remaining prompts: {len(indices)}')
            if not indices:
                break
            for i in range(0, len(indices), args.batch_size):
                batch_indices = indices[i:i + args.batch_size]
                prompt_objects = [prompts[j] for j in batch_indices]
                print(f'Generating programs for prompts {', '.join(prompt_object["id"] for prompt_object in prompt_objects)}')
                for prompt_object in prompt_objects:
                    prompt_object['programs'] = prompt_object.get('programs', [])

                prompt = prompt_factory([o['prompt'] for o in prompt_objects], seed=rng.randint(0, 2**32-1))
                try:
                    program = gpt.ask(prompt)
                    if program.strip() == "You've reached our limit of messages per hour. Please try again later.":
                        gpt.quit()
                        current_gpt_index = (current_gpt_index + 1) % len(args.gpt_user_dirs)
                        gpt = GPTClient(args.gpt_user_dirs[current_gpt_index], wait_for_login=args.gpt_auth)
                        continue
                    lines = [line.strip() for line in program.strip().split('\n') if CODE_REGEX.match(line)]
                    final_lines = [-1] + [i for i, line in enumerate(lines) if args.split_by in line]
                    if not final_lines:
                        raise ValueError('No program generated')
                    if len(final_lines) != len(prompt_objects) + 1:
                        if len(final_lines) == 1:
                            print(f'Answer: {program}')
                        raise ValueError(f'Ambiguous program generated. Expected {len(prompt_objects)} programs, '
                                         f'got {len(final_lines) - 1}')
                    programs = ['\n'.join(lines[before_line + 1:final_line + 1])
                                for before_line, final_line in zip(final_lines[:-1], final_lines[1:])]
                    for prompt_object, program in zip(prompt_objects, programs):
                        prompt_object['programs'].append(program)
                    print(f'Generated programs: {"\n\n".join(programs)}')
                    with open(args.output_file, 'w') as f:
                        yaml.dump(prompts, f, default_style='|', sort_keys=False)
                except (TimeoutException, ValueError) as e:
                    if isinstance(e, ValueError):
                        if not e.args or not (e.args[0] == 'No program generated' or e.args[0].startswith('Ambiguous')):
                            raise
                        print(e.args[0])
                    else:
                        print('TimeoutException')
                finally:
                    gpt.new_chat()
                    print('waiting 5 seconds')
                    time.sleep(5)
                print('---------')
    finally:
        gpt.quit()


if __name__ == '__main__':
    main()
