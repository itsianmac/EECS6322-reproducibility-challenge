import re
from typing import Any, Dict, Tuple

from PIL import Image

from modules.visprog_module import VisProgModule, ParsedStep


class Result(VisProgModule):

    def parse(self, step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        step : str
            with the format OUTPUT=RESULT(var=Var,...)

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        pattern = re.compile(r"(?P<output>.*)\s*=\s*RESULT\s*.*")
        match = pattern.match(step)
        if match is None:
            raise ValueError(f"Could not parse step: {step}")
        variable_pattern = re.compile(r"(?P<dict_key>[a-zA-Z0-9_]+)\s*=\s*(?P<var>[a-zA-Z0-9_]+)")
        # iterate through the matches and add them to the dictionary
        inputs = {}
        for i, match in enumerate(variable_pattern.finditer(step)):
            if i == 0:
                continue  # skip the first match, which is the output
            inputs[match.group('dict_key')] = match.group('var')
        return ParsedStep(match.group('output'),
                          input_var_names=inputs)


    def perform_module_function(self, **kwargs: Any) -> Any:
        return kwargs

    def html(self, output: Any, **kwargs) -> Dict[str, Any]:
        """ Generate HTML to display the output
        """
        return output
