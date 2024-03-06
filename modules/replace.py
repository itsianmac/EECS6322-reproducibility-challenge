import re
from typing import Dict

import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline

from modules.visprog_module import VisProgModule, ParsedStep


class Replace(VisProgModule):

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            # TODO: fix device
            # revision="fp16",
            # torch_dtype=torch.float16,
        )
        # self.model = self.pipe.to(device)
        self.device = device

    def parse(self, step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        step : str
            with the format OUTPUT=REPLACE(image=IMAGE,object=OBJ,prompt='blue bus')

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        pattern = re.compile(r"(?P<output>.*)\s*=\s*REPLACE\s*"
                             r"\(\s*image\s*=\s*(?P<image>.*)\s*"
                             r",\s*object\s*=\s*(?P<object>.*)\s*"
                             r",\s*prompt\s*=\s*'(?P<prompt>.*)'\s*\)")
        match = pattern.match(step)
        if match is None:
            raise ValueError(f"Could not parse step: {step}")
        return ParsedStep(match.group('output'),
                          inputs={
                              'prompt': match.group('prompt')
                          }, input_var_names={
                'image': match.group('image'),
                'seg_map': match.group('object')
            })

    def perform_module_function(self, image: Image.Image, seg_map: np.ndarray, prompt: str) -> Image.Image:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        seg_map : np.ndarray
            The mask of the object in the image

        prompt : str
            The prompt to use for the replacement

        Returns
        -------
        Image.Image
            The image with the object replaced
        """
        return self.pipe(prompt=prompt, image=image, mask_image=seg_map).images[0]

    def html(self, output: Image.Image, image: Image.Image, seg_map: np.ndarray, prompt: str) -> str:
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
        masked_image = image_array * seg_map
        masked_image = Image.fromarray(masked_image)

        return {
            'input': image,
            'prompt': prompt,
            'masked_image': masked_image,
            'output': output,
        }
