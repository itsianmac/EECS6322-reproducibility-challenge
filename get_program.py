import argparse
import importlib
import os
import re
import time
import random

import yaml
from selenium.common import TimeoutException

from gpt import GPTClient
from instructions import PromptFactory


CODE_REGEX = re.compile(r'\S+\s*=\s*\S+\s*\([^)]*\)')


def main():
    parser = argparse.ArgumentParser(
        description='Ask ChatGPT to generate programs for each instruction',
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
        'prompts_file',
        type=str,
    )
    parser.add_argument(
        'output_file',
        type=str,
    )
    args = parser.parse_args()

    get_prompt_factory = importlib.import_module(f'instructions.{args.context}').get_prompt_factory
    prompt_factory: PromptFactory = get_prompt_factory(method=args.method, num_prompts=args.num_prompts)

    print('Initializing GPT client')
    gpt = GPTClient()
    time.sleep(1)
    print('GPT client initialized')

    with open(args.prompts_file, 'r') as f:
        prompts = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    seed = args.seed
    rng = random.Random(seed)
    try:
        for prompt_object in prompts:
            print(f'Generating programs for prompt {prompt_object["id"]}')
            prompt_object['programs'] = prompt_object.get('programs', [])
            while len(prompt_object['programs']) < args.num_tries:
                prompt = prompt_factory(seed=rng.randint(0, 2**32-1), **prompt_object['prompt'])
                try:
                    program = gpt.ask(prompt)
                    lines = [line.strip() for line in program.strip().split('\n') if CODE_REGEX.match(line)]
                    program = '\n'.join(lines)
                    if not program:
                        raise ValueError('No program generated')
                    prompt_object['programs'].append(program)
                    print(f'Generated program: {program}')
                    with open(args.output_file, 'w') as f:
                        yaml.dump(prompts, f, default_style='|', sort_keys=False)
                except (TimeoutException, ValueError) as e:
                    if isinstance(e, ValueError):
                        print('ValueError')
                        if not e.args or e.args[0] != 'No program generated':
                            raise
                    else:
                        print('TimeoutException')
                finally:
                    gpt.new_chat()
                    print('waiting 5 seconds')
                    time.sleep(5)
            print(f'Generated {args.num_tries} programs for prompt {prompt_object["id"]}')
            print('---------')
    finally:
        gpt.quit()


if __name__ == '__main__':
    main()
