# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import torch


class PyronearPredictor:
    """This class use the last pyronear model to predict if a sigle frame contain a fire or not
    Examples:
        >>> pyronearPredictor = PyronearPredictor("path/to/model.pt")
        >>> im = Image.open("image.jpg")
        >>> res = pyronearPredictor.predict(im)
    """

    def __init__(self, path_to_model, conf=0.25):
        """Init predictor"""
        # Model definition
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=path_to_model
        )  # local model
        self.model.conf = conf

    def predict(self, im):
        """Run prediction"""
        pred = self.model(im)

        return pred.xyxy[0].numpy()
