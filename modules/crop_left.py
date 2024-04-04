import re
from typing import Tuple

from PIL import Image

from modules import Crop


class CropLeft(Crop):
    pattern = re.compile(
        r"(?P<output>\S*)\s*=\s*CROP_LEFTOF\s*"
        r"\(\s*image\s*=\s*(?P<image>\S*)\s*"
        r",\s*box\s*=\s*(?P<box>\S*)\s*\)"
    )

    def perform_module_function(
        self, image: Image.Image, box: Tuple[Tuple[float, ...], ...]
    ) -> Image.Image:
        """Perform the color pop operation on the image using the object mask

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
        original_box = box[0] if len(box) > 0 else (0, 0, image.width, image.height)
        left_box = (0, 0, original_box[0], image.height)
        # check that bounding box dimensions are valid
        if left_box[0] < left_box[2] and left_box[1] < left_box[3] and len(box) > 0:
            return image.crop(left_box)
        else:
            return image
