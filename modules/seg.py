import re
from typing import Any, Dict

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation

from modules.visprog_module import VisProgModule, ParsedStep


class Seg(VisProgModule):

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-ade")
        self.model = self.model.to(device)
        self.device = device

    def parse(self, step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        step : str
            with the format OBJ=SEG(image=IMAGE)

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        pattern = re.compile(r"(?P<output>.*)\s*=\s*SEG\s*"
                             r"\(\s*image\s*=\s*(?P<image>.*)\s*\)")
        match = pattern.match(step)
        if match is None:
            raise ValueError(f"Could not parse step: {step}")
        return ParsedStep(match.group('output'),
                          input_var_names={
                              'image': match.group('image')
                          })

    def perform_module_function(self, image: Image.Image) -> np.ndarray:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        Returns
        -------
        np.ndarray
            The mask of the object in the image
        """
        inputs = self.image_processor(image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        predicted_semantic_map = self.image_processor.post_process_semantic_segmentation(   # TODO: I'm assuming this is a map of class labels
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        return predicted_semantic_map.numpy()

    def html(self, output: np.ndarray, image: Image.Image) -> Dict[str, Any]:
        """ Generate HTML to display the output

        Parameters
        ----------
        inputs : Dict[str, Any]
        """
        image_array = np.array(image)
        unique_classes = np.unique(output)
        segments = []
        for class_label in unique_classes:
            mask = output == class_label
            masked_image = image_array * mask
            masked_image = Image.fromarray(masked_image)
            segments.append(masked_image)

        return {
            'input': image,
            'output': segments
        }
