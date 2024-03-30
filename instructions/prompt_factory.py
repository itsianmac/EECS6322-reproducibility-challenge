import random
from typing import List, Literal, Dict, Any


class PromptFactory:

    def __init__(self, prefix: str, examples: List[str], format_: str, method: Literal['all', 'random'] = 'random',
                 num_prompts: int = 8):
        self.prefix = prefix
        self.examples = examples
        self.format = format_
        self.method = method
        self.num_prompts = num_prompts

    def get_examples(self, seed: int = 42):
        if self.method == 'all':
            prompt_examples = self.examples
        elif self.method == 'random':
            rng = random.Random(seed)
            prompt_examples = rng.sample(self.examples, self.num_prompts)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        prompt_examples = '\n'.join(prompt_examples)
        prompt_examples = f'{self.prefix}\n\n{prompt_examples}'
        return prompt_examples

    def __call__(self, seed: int = 42, **prompt_variables):
        prompt_examples = self.get_examples(seed)
        return prompt_examples + self.format.format(**prompt_variables)


class BulkPromptFactory(PromptFactory):

    def __call__(self, prompts: List[Dict[str, Any]], seed: int = 42):
        prompt_examples = self.get_examples(seed)
        return prompt_examples + '\n\nNow propose programs for the following statements.\n\n' + \
            '\n\n'.join(self.format.format(**prompt) for prompt in prompts)
