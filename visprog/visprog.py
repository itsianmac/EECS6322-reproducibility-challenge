from typing import List, Dict, Any, Tuple

from gpt import GPTClient
from instructions import PromptFactory
from modules import VisProgModule
from visprog import ProgramRunner
from visprog.program_runner import ProgramResult


class VisProg:

    def __init__(self, prompt_factory: PromptFactory, gpt: GPTClient, modules: List[VisProgModule]):
        self.prompt_factory = prompt_factory
        self.gpt = gpt
        self.program_runner = ProgramRunner(modules)

    def run(self, initial_state: Dict[str, Any], seed: int = 42, **prompt: str) -> Tuple[List[str], ProgramResult]:
        prompt = self.prompt_factory(seed=seed, **prompt)
        program = self.gpt.ask(prompt)
        return self.program_runner.execute_program(program, initial_state)
