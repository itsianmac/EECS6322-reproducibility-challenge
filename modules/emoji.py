import os
import re
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter
import PIL
import augly.image as imaugs
from augly.utils.constants import SMILEY_EMOJI_DIR

from modules.visprog_module import VisProgModule, ParsedStep


class Emoji(VisProgModule):

    def parse(self, step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        step : str
            with the format OUTPUT=EMOJI(image=IMAGE,object=OBJ,emoji='<some-str>')

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        pattern = re.compile(r"(?P<output>.*)\s*=\s*EMOJI\s*"
                             r"\(\s*image\s*=\s*(?P<image>.*)\s*"
                             r",\s*object\s*=\s*(?P<object>.*)\s*"
                             r",\s*emoji\s*=\s*'(?P<emoji>.*)'\s*\)")
        match = pattern.match(step)
        if match is None:
            raise ValueError(f"Could not parse step: {step}")
        return ParsedStep(match.group('output'),
                          inputs={'emoji': match.group('emoji')},
                          input_var_names={
                              'image': match.group('image'),
                              'bbox': match.group('object')
                          })

    @staticmethod
    def get_emoji_path(emoji: str) -> str:
        """ Get the path to the emoji image

        Parameters
        ----------
        emoji : str
            The emoji to overlay on the image

        Returns
        -------
        str
            The path to the emoji image
        """
        return os.path.join(SMILEY_EMOJI_DIR, f"{emoji}.png")

    def perform_module_function(self, image: Image.Image, bbox: Tuple[float,...], emoji: str) -> Image.Image:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        bbox : np.ndarray
            The object bounding box

        emoji : str
            The emoji to overlay on the image

        Returns
        -------
        Image.Image
            The color popped image
        """
        aug_image = imaugs.overlay_emoji(image, emoji_path=self.get_emoji_path(emoji),
                                         x_pos=bbox[0] / image.size[0], y_pos=bbox[1] / image.size[1],
                                         emoji_size=max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / image.size[1])
        return aug_image

    def html(self, output: Image.Image, image: Image.Image, bbox: Tuple[float,...], emoji: str) -> Dict[str, Any]:
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
            'output': output
        }
