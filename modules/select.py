import re
from typing import Any, Dict, Tuple, Optional, Union, List

import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch
from transformers import CLIPProcessor, CLIPModel

from modules.visprog_module import VisProgModule, ParsedStep


class Select(VisProgModule):
    pattern = re.compile(r"(?P<output>\S*)\s*=\s*SELECT\s*"
                         r"\(\s*image\s*=\s*(?P<image>\S*)\s*"
                         r",\s*object\s*=\s*(?P<object>\S*)\s*"
                         r",\s*query\s*=\s*'(?P<query>.*)'\s*"
                         r",\s*category\s*=\s*(?P<category>\S.*\S*)\s*\)")

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

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format OUTPUT=SELECT(image=IMAGE,object=OBJ,query='<prompt>',category=None or '<category>')

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        return ParsedStep(match.group('output'),
                          inputs={
                              'query': match.group('query'),
                              'category': None if match.group('category') == 'None'
                              else match.group('category').strip("'")
                          }, input_var_names={
                              'image': match.group('image'),
                              'object': match.group('object')
                          })

    def get_seg_map_and_category_ids(self, image: Image.Image, object: Union[np.ndarray, Tuple[Tuple[float, ...], ...]],
                                     category: Optional[str] = None) -> Tuple[np.ndarray, List[int]]:
        if isinstance(object, np.ndarray):  # object is a segmentation map
            seg_map = object
            unique_labels = set(np.unique(seg_map))
            category_ids = []
            if category is not None:
                keywords = re.split(r'[,-]', category)
                category_ids = [self.category_name_to_id[keyword] for keyword in keywords
                                if
                                keyword in self.category_name_to_id
                                and self.category_name_to_id[keyword] in unique_labels]

            if len(category_ids) == 0:
                category_ids = list(np.unique(seg_map))

        else:   # object is a list of bounding boxes
            seg_map = np.zeros(image.size[::-1], dtype=np.uint8)
            for i, box in enumerate(object):
                x1, y1, x2, y2 = map(int, box)
                seg_map[y1:y2, x1:x2] = i + 1

            category_ids = list(range(len(object) + 1))

        return seg_map, category_ids

    def perform_module_function(self, image: Image.Image, object: Union[np.ndarray, Tuple[Tuple[float, ...], ...]],
                                query: str,
                                category: Optional[str] = None) -> Union[np.ndarray, Tuple[Tuple[float, ...], ...]]:
        """ Select the object in the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        object : Union[np.ndarray, Tuple[Tuple[float, ...], ...]]
            The segmentation map or bounding boxes

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
        image_array = np.array(image)

        seg_map, category_ids = self.get_seg_map_and_category_ids(image, object, category)

        masked_images = []
        for category_id in category_ids:
            mask = seg_map == category_id
            masked_image = image_array * mask[..., None]
            masked_image = Image.fromarray(masked_image)
            masked_images.append(masked_image)

        inputs = self.processor(text=queries, images=masked_images, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        best_index_per_query = logits_per_image.argmax(dim=0)
        assert len(best_index_per_query) == len(queries)
        selected_category_ids = [category_ids[i] for i in best_index_per_query]
        if isinstance(object, np.ndarray):
            return np.isin(seg_map, selected_category_ids)
        else:
            selected_boxes = [object[i - 1] for i in best_index_per_query]
            return selected_boxes

    def html(self, output: Union[np.ndarray, Tuple[Tuple[float, ...], ...]],
             image: Image.Image, object: Union[np.ndarray, Tuple[Tuple[float, ...], ...]],
             query: str, category: Optional[str] = None) -> Dict[str, Any]:
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
        if isinstance(object, np.ndarray):
            image_array = np.array(image)
            masked_image = image_array * output[..., None]
            masked_image = Image.fromarray(masked_image)
            return {
                'prompt': query,
                'category': category,
                'input': image,
                'output': masked_image
            }

        image_with_bbox = image.copy()
        draw = ImageDraw.Draw(image_with_bbox)
        for box in output:
            draw.rectangle(box, outline="red", width=3)

        return {
            'prompt': query,
            'category': category,
            'input': image,
            'output': image_with_bbox
        }
