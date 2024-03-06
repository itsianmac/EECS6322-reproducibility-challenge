import re
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter
import PIL

from modules.visprog_module import VisProgModule, ParsedStep


class BGBlur(VisProgModule):
    pattern = re.compile(r"(?P<output>.*)\s*=\s*BGBLUR\s*"
                         r"\(\s*image\s*=\s*(?P<image>.*)\s*"
                         r",\s*object\s*=\s*(?P<object>.*)\s*\)")

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format OUTPUT=BGBLUR(image=IMAGE,object=OBJ)

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
        blured_image = np.array(image.filter(ImageFilter.GaussianBlur(radius=5)))
        image_array = np.array(image)
        image_array[~object] = blured_image[~object]
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
