import re
from typing import Dict, Union, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
from diffusers import StableDiffusionInpaintPipeline

from modules.visprog_module import VisProgModule, ParsedStep


class Replace(VisProgModule):
    pattern = re.compile(r"(?P<output>\S*)\s*=\s*REPLACE\s*"
                         r"\(\s*image\s*=\s*(?P<image>\S*)\s*"
                         r",\s*object\s*=\s*(?P<object>\S*)\s*"
                         r",\s*prompt\s*=\s*'(?P<prompt>.*)'\s*\)")

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            **(dict(
                revision="fp16",
                torch_dtype=torch.float16,
            ) if device != 'cpu' else {})
        )
        self.model = self.pipe.to(device)
        self.device = device

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format OUTPUT=REPLACE(image=IMAGE,object=OBJ,prompt='blue bus')

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        return ParsedStep(match.group('output'),
                          inputs={
                              'prompt': match.group('prompt')
                          }, input_var_names={
                'image': match.group('image'),
                'object': match.group('object')
            })

    @staticmethod
    def get_seg_map(image: Image.Image,
                    object: Union[np.ndarray, Tuple[Tuple[float, ...], ...]]) -> np.ndarray:
        if isinstance(object, np.ndarray):  # object is a segmentation map
            seg_map = object
        else:   # object is a list of bounding boxes
            seg_map = np.zeros(image.size[::-1], dtype=np.uint8)
            for obj in object:
                x1, y1, x2, y2 = map(int, obj)
                seg_map[y1:y2, x1:x2] = 1

        return seg_map

    @staticmethod
    def square_image(image: Image.Image) -> Image.Image:
        max_dim = max(image.size)
        new_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        new_image.paste(image, (0, 0))
        return new_image

    @staticmethod
    def square_seg_map(seg_map: np.ndarray) -> np.ndarray:
        max_dim = max(seg_map.shape)
        new_seg_map = np.zeros((max_dim, max_dim), dtype=np.uint8)
        new_seg_map[:seg_map.shape[0], :seg_map.shape[1]] = seg_map
        return new_seg_map

    @staticmethod
    def desquare_image(output: Image.Image, image: Image.Image) -> Image.Image:
        max_dim = max(image.size)
        output = output.resize((max_dim, max_dim))
        return output.crop((0, 0, *image.size))

    def perform_module_function(self, image: Image.Image, object: np.ndarray, prompt: str) -> Image.Image:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        object : np.ndarray
            The mask of the object in the image

        prompt : str
            The prompt to use for the replacement

        Returns
        -------
        Image.Image
            The image with the object replaced
        """
        seg_map = self.get_seg_map(image, object)
        square_image = self.square_image(image)
        square_seg_map = self.square_seg_map(seg_map)
        output: Image.Image = self.pipe(prompt=prompt, image=square_image,
                                        mask_image=square_seg_map.astype('float')).images[0]
        return self.desquare_image(output, image)

    def html(self, output: Image.Image, image: Image.Image, object: np.ndarray, prompt: str) -> str:
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
        if isinstance(object, np.ndarray):
            image_array = np.array(image)
            masked_image = image_array * (~object[..., None])
            masked_image = Image.fromarray(masked_image)

            return {
                'input': image,
                'prompt': prompt,
                'masked_image': masked_image,
                'output': output,
            }

        image_with_bbox = image.copy()
        draw = ImageDraw.Draw(image_with_bbox)
        for obj in object:
            draw.rectangle(obj, outline="red", width=3)

        return {
            'input': image,
            'prompt': prompt,
            'masked_image': image_with_bbox,
            'output': output,
        }
