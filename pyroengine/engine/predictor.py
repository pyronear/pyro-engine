# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch


class PyronearPredictor:
    """This class use the last pyronear model and run our smoke detection model on it
    Examples:
        >>> pyronearPredictor = PyronearPredictor()
        >>> im = Image.open("image.jpg")
        >>> res = pyronearPredictor.predict(im)
    """

    def __init__(self, model_weights: str = None, conf: float = 0.25):
        """Init predictor"""
        # Model definition
        if model_weights is None:
            model_weights = "https://github.com/pyronear/pyro-vision/releases/download/v0.1.2/yolov5s_v001.pt"
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_weights
        )  # local model
        self.model.conf = conf

    def predict(self, im):
        """Run prediction"""
        pred = self.model(im)

        return pred.xyxy[0].numpy()
