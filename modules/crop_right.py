import re
from typing import Tuple

from PIL import Image

from modules import Crop


class CropRight(Crop):
    pattern = re.compile(r"(?P<output>.*)\s*=\s*CROP_RIGHTOF\s*"
                         r"\(\s*image\s*=\s*(?P<image>.*)\s*"
                         r",\s*box\s*=\s*(?P<box>.*)\s*\)")

    def perform_module_function(self, image: Image.Image, box: Tuple[Tuple[float, ...], ...]) -> Image.Image:
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
        # TODO: which box we should use?
        original_box = box[0]
        right_box = (original_box[2], 0, image.width, image.height)
        return image.crop(right_box)
