import re
from typing import Tuple

from PIL import Image

from modules import Crop


class CropBelow(Crop):
    pattern = re.compile(r"(?P<output>\S*)\s*=\s*CROP_BELOW\s*"
                         r"\(\s*image\s*=\s*(?P<image>\S*)\s*"
                         r",\s*box\s*=\s*(?P<box>\S*)\s*\)")

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
        below_box = (0, original_box[3], image.width, image.height)
        return image.crop(below_box)
