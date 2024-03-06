import re
from typing import Any, Dict, Optional, Tuple

from dataclasses import dataclass, field


@dataclass
class ParsedStep:
    output_var_name: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    input_var_names: Dict[str, str] = field(default_factory=dict)


class VisProgModule:
    pattern: re.Pattern[str]
    
    def __init__(self):
        """ Load a trained model, move it to gpu, etc. """
        pass

    def html(self, output: Any, **inputs) -> Dict[str, Any]:
        """ Return an html string visualizing step I/O

        Parameters
        ----------
        inputs : list
            I am guessing inputs to the module?
            
        output : Any
            I am guessing outputs from the module?
            
        """
        pass

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names 
            and output variable name. 

        Parameters
        ----------
        match : re.Match[str]

        step : str

        Returns
        -------
        inputs : ?

        input_var_names : ?

        output_var_name : ?
            
        """
        pass

    def perform_module_function(self, **inputs):
        """ NOTE: I added this for us. The idea is we can implement
            this, parse, and html, and just call execute in a loop on 
            a list of VisProgModules.

        Parameters
        ----------
        inputs : ? 
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """

        pass

    def match(self, step: str) -> Optional[re.Match[str]]:
        """ Match the step to the pattern and return the match object

        Parameters
        ----------
        step : str
            The step to match

        Returns
        -------
        re.Match[str]
            The match object
        """
        return self.pattern.match(step)

    def execute(self, step: str, state: dict, match: Optional[re.Match[str]] = None) -> Tuple[Any, Dict[str, Any]]:
        if match is None:
            match = self.match(step)
            if match is None:
                raise ValueError(f"Step {step} does not match pattern {self.pattern}")
        parsed_step = self.parse(match, step)

        inputs = parsed_step.inputs.copy()
        # Get values of input variables form state
        for input_name, var_name in parsed_step.input_var_names.items():
            inputs[input_name] = state[var_name]

        # Perform computation using the loaded module
        output = self.perform_module_function(**inputs)

        # Update state
        state[parsed_step.output_var_name] = output

        step_html = self.html(output, **inputs)

        return output, step_html


