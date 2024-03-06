import re
from typing import Any, Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import PIL
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

from modules.visprog_module import VisProgModule, ParsedStep


class VQA(VisProgModule):
    pattern = re.compile(r"(?P<output>.*)\s*=\s*VQA\s*"
                         r"\(\s*image\s*=\s*(?P<image>.*)\s*"
                         r",\s*question\s*=\s*'(?P<question>.*)'\s*\)")

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model = self.model.to(device)
        self.device = device

    def parse(self, match: re.Match[str], step: str) -> ParsedStep:
        """ Parse step and return list of input values/variable names
            and output variable name.

        Parameters
        ----------
        match : re.Match[str]
            The match object from the regex pattern
        step : str
            with the format ANSWER=VQA(image=IMAGE,question='Who is carrying the umbrella?')

        Returns
        -------
        inputs : ?
            Likely just images for our task... this doesn't look to be typed
            in the original Visprog paper... so will need to take some liberties here
            and just use a torch tensor or image... can also leave it untyped
        """
        return ParsedStep(match.group('output'),
                          inputs={
                              'question': match.group('question')
                          }, input_var_names={
                              'image': match.group('image')
                          })

    def perform_module_function(self, image: Image.Image, question: str) -> str:
        """ Perform the color pop operation on the image using the object mask

        Parameters
        ----------
        image : Image.Image
            The original image

        question : str
            The question to ask about the image

        Returns
        -------
        Tuple[str,...]
            The answer to the question
        """
        encoding = self.processor(image, question, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.model.config.id2label[idx]

    def html(self, output: str, image: Image.Image, question: str) -> Dict[str, Any]:
        """ Generate HTML to display the output

        Parameters
        ----------
        inputs : Dict[str, Any]
            The input variables and their values

        output : str
            The output answer

        Returns
        -------
        Dict[str, Any]
            The HTML to display the output
        """
        return {
            'input': image,
            'question': question,
            'output': output
        }
