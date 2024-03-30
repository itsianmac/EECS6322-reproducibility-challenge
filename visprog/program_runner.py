import re

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from modules import VisProgModule, ExecutionError


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

    def match_step(self, step: str) -> Optional[Tuple[VisProgModule, re.Match]]:
        matched = next(((module, match)
                        for module in self.modules
                        if (match := module.match(step))),
                       None)
        return matched

    def execute_steps(self, steps: List[str], initial_state: Dict[str, Any]) -> Tuple[List[str], ProgramResult]:
        state = initial_state.copy()
        step_details = []
        output = None
        executed_steps = []
        try:
            for i, step in enumerate(steps):
                try:
                    matched: Tuple[VisProgModule, Optional[re.Match]] = self.match_step(step)
                    if matched is None:
                        continue
                    module, match = matched
                    output, details = module.execute(step, state, match=match)
                    executed_steps.append(step)
                    step_details.append(details)
                except ExecutionError as e:
                    raise
                except Exception as e:
                    print(f"Error in executing step {i}: {step}, {e}")
                    raise
        except ExecutionError as e:
            raise ExecutionError(e.step, e.error, previous_step_details=step_details)

        return executed_steps, ProgramResult(state, output, step_details)
