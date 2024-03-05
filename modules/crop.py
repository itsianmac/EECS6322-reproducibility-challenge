import re
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter
import PIL

from modules.visprog_module import VisProgModule, ParsedStep


class Crop(VisProgModule):

    def parse(self, step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        step : str
            with the format OUTPUT=CROP(image=IMAGE,box=BOX)

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        pattern = re.compile(r"(?P<output>.*)\s*=\s*BGBLUR\s*"
                             r"\(\s*image\s*=\s*(?P<image>.*)\s*"
                             r",\s*box\s*=\s*(?P<box>.*)\s*\)")
        match = pattern.match(step)
        if match is None:
            raise ValueError(f"Could not parse step: {step}")
        return ParsedStep(match.group('output'),
                          input_var_names={
                              'image': match.group('image'),
                              'box': match.group('box')
                          })

    def perform_module_function(self, image: Image.Image, box: Tuple[float,...]) -> Image.Image:
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
        return image.crop(box)

    def html(self, output: Image.Image, image: Image.Image, box: Tuple[float,...]) -> Dict[str, Any]:
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
            'input': image,
            'output': output,
        }
