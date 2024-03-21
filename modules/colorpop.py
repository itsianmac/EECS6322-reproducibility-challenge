import re
from typing import Any, Dict

import numpy as np
from PIL import Image
import PIL

from modules.visprog_module import VisProgModule, ParsedStep


class ColorPop(VisProgModule):
    pattern = re.compile(r"(?P<output>\S*)\s*=\s*COLORPOP\s*"
                         r"\(\s*image\s*=\s*(?P<image>\S*)\s*"
                         r",\s*object\s*=\s*(?P<object>\S*)\s*\)")

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format OUTPUT=COLORPOP(image=IMAGE,object=OBJ)

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        return ParsedStep(match.group('output'),
                          input_var_names={
                              'image': match.group('image'),
                              'object': match.group('object')
                          })

    def perform_module_function(self, image: Image.Image, object: np.ndarray) -> Image.Image:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        object : np.ndarray
            The object binary mask

        Returns
        -------
        Image.Image
            The color popped image
        """
        image_array = np.array(image)
        image_grayscale = np.array(image.convert('L').convert('RGB'))
        mask = object[..., None].repeat(3, axis=-1)
        image_array = np.where(mask, image_grayscale, image_array)
        return PIL.Image.fromarray(image_array)

    def html(self, output: Image.Image, image: Image.Image, object: np.ndarray) -> Dict[str, Any]:
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
        image_array = np.array(image)

        return {
            'background': PIL.Image.fromarray(image_array * (~object[..., None])),
            'foreground': PIL.Image.fromarray(image_array * object[..., None]),
            'output': output
        }
