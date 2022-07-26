# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from pyroengine.models.yolov5 import Yolo_v5


class PyronearPredictor:
    """This class use the last pyronear model and run our smoke detection model on it
    Examples:
        >>> pyronearPredictor = PyronearPredictor(model_weights=/path/to/model.onnx)
        >>> im = Image.open("image.jpg")
        >>> res = pyronearPredictor.predict(im)
    """

    def __init__(self, model_weights: str = None, conf: float = 0.25):
        """Init predictor"""
        # Model definition
        self.model = Yolo_v5(model_weights=model_weights, conf_thres=conf)

    def predict(self, im):
        """Run prediction"""
        pred = self.model(im)

        return pred
