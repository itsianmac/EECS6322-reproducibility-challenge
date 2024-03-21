import re
import warnings

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from modules import VisProgModule


@dataclass
class ProgramResult:
    state: Dict[str, Any]
    output: Any
    step_details: List[Dict[str, Any]]


class ProgramRunner:

    def __init__(self, modules: List[VisProgModule]):
        self.modules = modules

    def execute_program(self, program: str, initial_state: Dict[str, Any]) -> Tuple[List[str], ProgramResult]:
        steps = [step.strip() for step in program.split('\n') if step.strip()]
        return self.execute_steps(steps, initial_state)

    def execute_steps(self, steps: List[str], initial_state: Dict[str, Any]) -> Tuple[List[str], ProgramResult]:
        state = initial_state.copy()
        step_details = []
        output = None
        executed_steps = []
        for i, step in enumerate(steps):
            matched: Tuple[VisProgModule, Optional[re.Match]] = next(((module, match)
                                                                      for module in self.modules
                                                                      if (match := module.match(step))),
                                                                     (None, None))
            module, match = matched
            if match is None:
                warnings.warn(f"No module matched step {i}: {step}. Skipping."
                              f" This may be a bug in the program generation.")
                continue
            output, details = module.execute(step, state, match=match)
            executed_steps.append(step)
            step_details.append(details)

        return executed_steps, ProgramResult(state, output, step_details)
