import re
from typing import Any, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch
from transformers import CLIPProcessor, CLIPModel

from modules.visprog_module import VisProgModule, ParsedStep


class Select(VisProgModule):

    def __init__(self, category_id_to_name: Dict[int, str], category_name_to_id: Dict[str, int], device: str = "cpu"):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.model = self.model.to(device)
        self.device = device
        self.category_id_to_name = category_id_to_name
        self.category_name_to_id = {}
        for k, v in category_name_to_id.items():
            keys = k.split(', ')
            for key in keys:
                self.category_name_to_id[key] = v

    def parse(self, step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        step : str
            with the format OUTPUT=SELECT(image=IMAGE,object=OBJ,query='<prompt>',category=None or '<category>')

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        pattern = re.compile(r"(?P<output>.*)\s*=\s*SELECT\s*"
                             r"\(\s*image\s*=\s*(?P<image>.*)\s*"
                             r",\s*object\s*=\s*(?P<object>.*)\s*"
                             r",\s*query\s*=\s*'(?P<query>.*)'\s*"
                             r",\s*category\s*=\s*(?P<category>.*)\s*\)")
        match = pattern.match(step)
        if match is None:
            raise ValueError(f"Could not parse step: {step}")
        return ParsedStep(match.group('output'),
                          inputs={
                              'query': match.group('query'),
                              'category': None if match.group('category') == 'None'
                              else match.group('category').strip("'")
                          }, input_var_names={
                              'image': match.group('image'),
                              'seg_map': match.group('object')
                          })

    def perform_module_function(self, image: Image.Image, seg_map: np.ndarray,
                                query: str, category: Optional[str] = None) -> np.ndarray:
        """ Select the object in the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        seg_map : np.ndarray
            The mask of the object in the image

        query : str
            The text prompt for the object

        category : Optional[str]
            The category of the object

        Returns
        -------
        np.ndarray
            The mask of the selected object in the image
        """
        queries = query.split(',')

        unique_labels = set(np.unique(seg_map))
        category_ids = []
        if category is not None:
            keywords = re.split(r'[,-]', category)
            category_ids = [self.category_name_to_id[keyword] for keyword in keywords
                             if
                             keyword in self.category_name_to_id and self.category_name_to_id[keyword] in unique_labels]

        if len(category_ids) == 0:
            category_ids = list(np.unique(seg_map))

        masked_images = []
        for category_id in category_ids:
            mask = seg_map == category_id
            masked_image = image * mask[..., None]
            masked_image = Image.fromarray(masked_image)
            masked_images.append(masked_image)

        inputs = self.processor(text=queries, images=masked_images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        best_index_per_query = logits_per_image.argmax(dim=0)
        assert len(best_index_per_query) == len(queries)
        selected_category_ids = [category_ids[i] for i in best_index_per_query]
        return np.isin(seg_map, selected_category_ids).astype(np.uint8)

    def html(self, output: np.ndarray, image: Image.Image, seg_map: np.ndarray, query: str,
             category: Optional[str] = None) -> Dict[str, Any]:
        """ Generate HTML to display the output

        Parameters
        ----------
        inputs : Dict[str, Any]
            The input variables and their values

        output : np.ndarray
            The output mask

        Returns
        -------
        Dict[str, Any]
            The HTML to display
        """
        image_array = np.array(image)
        masked_image = image_array * output[..., None]
        masked_image = Image.fromarray(masked_image)

        return {
            'prompt': query,
            'category': category,
            'input': image,
            'output': masked_image
        }
