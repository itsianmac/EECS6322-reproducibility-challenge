import re
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from modules.visprog_module import VisProgModule, ParsedStep


class Loc(VisProgModule):
    pattern = re.compile(r"(?P<output>\S*)\s*=\s*LOC\s*"
                         r"\(\s*image\s*=\s*(?P<image>\S*)\s*"
                         r",\s*object\s*=\s*'(?P<object>\S*)'\s*\)")

    def __init__(self, device: str = "cpu", threshold: float = 0.1):
        super().__init__()
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model = self.model.to(device)
        self.device = device
        self.threshold = threshold

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format BOX=LOC(image=IMAGE,object='TOP')

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        return ParsedStep(match.group('output'),
                          inputs={
                              'object': match.group('object')
                          },
                          input_var_names={
                              'image': match.group('image')
                          })

    def perform_module_function(self, image: Image.Image, object: str) -> Tuple[Tuple[float,...],...]:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        object : str
            The text prompt for the object

        Returns
        -------
        Tuple[float,...]
            The box of the object in the image (x1, y1, x2, y2)
        """
        inputs = self.processor(text=[object], images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes,
                                                               threshold=self.threshold)
        boxes = results[0]['boxes'].detach().cpu().numpy()
        return tuple(tuple(box) for box in boxes)

    def html(self, output: Tuple[Tuple[float,...],...], image: Image.Image, object: str) -> Dict[str, Any]:
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
