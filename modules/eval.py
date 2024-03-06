import re
from typing import Any, Dict, Tuple

from PIL import Image

from modules.visprog_module import VisProgModule, ParsedStep


class Eval(VisProgModule):
    pattern = re.compile(r"(?P<output>.*)\s*=\s*EVAL\s*"
                         r"\(\s*expr\s*=\s*\"(?P<expr>.*)\"\s*\)")

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format ANSWER=EVAL(expr="'yes' if {ANSWER0} > 0 else 'no'")

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        replace_pattern = re.compile(r"\{(?P<var>[^}]+)}")
        variable_names = []
        expression = match.group('expr')
        for var_match in replace_pattern.finditer(step):
            expression = expression.replace(var_match.group(0), var_match.group('var'))
            variable_names.append(var_match.group('var'))
        return ParsedStep(match.group('output'),
                          inputs={'expr': expression},
                          input_var_names={var: var for var in variable_names})

    def perform_module_function(self, expr: str, **kwargs) -> Any:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        box : Tuple[float,...]
            The box to crop the image to (x1, y1, x2, y2)

        Returns
        -------
        Image.Image
            The color popped image
        """
        return eval(expr, kwargs)

    def html(self, output: Any, expr: str, **kwargs) -> Dict[str, Any]:
        """ Generate HTML to display the output

        Parameters
        ----------
        inputs : Dict[str, Any]
            The input variables and their values

        output : Image.Image
            The output image

        Returns
        -------
        str
            The HTML to display the output
        """
        return {
            'expr': expr,
            'args': kwargs,
            'output': output,
        }
