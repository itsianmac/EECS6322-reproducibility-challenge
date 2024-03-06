import re

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

    def execute(self, steps: List[str], initial_state: Dict[str, Any]) -> ProgramResult:
        state = initial_state.copy()
        step_details = []
        output = None
        for i, step in enumerate(steps):
            matched: Tuple[VisProgModule, Optional[re.Match]] = next(((module, match)
                                                                      for module in self.modules
                                                                      if (match := module.match(step))),
                                                                     None)
            module, match = matched
            if match is None:
                raise ValueError(f"No module matched the step {i + 1}: {step}")
            output, details = module.execute(step, state, match=match)
            step_details.append(details)

        return ProgramResult(state, output, step_details)
