import re
from typing import Any, Dict, Tuple

from PIL import Image

from modules.visprog_module import VisProgModule, ParsedStep


class Count(VisProgModule):
    pattern = re.compile(r"(?P<output>.*)\s*=\s*COUNT\s*"
                         r"\(\s*box\s*=\s*(?P<box>.*)\s*\)")

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format ANSWER=COUNT(box=BOX)

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        return ParsedStep(match.group('output'),
                          input_var_names={
                              'boxes': match.group('box')
                          })

    def perform_module_function(self, boxes: Tuple[Tuple[float,...],...]) -> int:
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
        return len(boxes)

    def html(self, output: Any, boxes: Tuple[Tuple[float,...],...]) -> Dict[str, Any]:
        """ Generate HTML to display the output

        Parameters
        ----------
        inputs : Dict[str, Any]
        """
        return {
            'text': f"The number of boxes is {output}"
        }
