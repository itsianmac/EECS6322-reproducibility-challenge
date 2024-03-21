import re
from typing import Any, Dict, Tuple

import face_detection
import numpy as np
from PIL import Image, ImageDraw
import torch

from modules.visprog_module import VisProgModule, ParsedStep


class FaceDet(VisProgModule):
    pattern = re.compile(r"(?P<output>\S*)\s*=\s*FACEDET\s*"
                         r"\(\s*image\s*=\s*(?P<image>\S*)\s*\)")

    def __init__(self, device: str = "cpu", confidence_threshold: float = 0.1, nms_iou_threshold: float = 0.1):
        super().__init__()
        self.detector = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold, device=device)

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format OBJ=FACEDET(image=IMAGE)

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        return ParsedStep(match.group('output'),
                          input_var_names={
                              'image': match.group('image')
                          })

    def perform_module_function(self, image: Image.Image) -> Tuple[Tuple[float,...],...]:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        Returns
        -------
        Tuple[float,...]
            The box of the object in the image (x1, y1, x2, y2)
        """
        image = np.array(image)
        boxes: torch.Tensor = self.detector.detect(image)
        return tuple((xmin, ymin, xmax, ymax) for xmin, ymin, xmax, ymax, detection_confidence in boxes)

    def html(self, output: Tuple[Tuple[float,...],...], image: Image.Image) -> Dict[str, Any]:
        """ Generate HTML to display the output

        Parameters
        ----------
        inputs : Dict[str, Any]
            The input variables and their values

        output : Tuple[Tuple[float,...],...]
            The output boxes

        Returns
        -------
        Dict[str, Any]
            The HTML to display
        """
        bbox_drawn_image = image.copy()
        draw = ImageDraw.Draw(bbox_drawn_image)
        for box in output:
            draw.rectangle(box, outline="red", width=3)

        return {
            'prompt': object,
            'image': image,
            'image_with_bbox': bbox_drawn_image,
        }
